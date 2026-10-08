#!/usr/bin/env python3
"""
Take machine paths out of fold books, and find any that hide in binaries.

    python tools/scrub_fold_books.py scan  <file or dir> [...]
    python tools/scrub_fold_books.py scrub <book.npz> [...] [--out-dir DIR]

WHY A TEXT GREP IS NOT ENOUGH. A fold book records the cache it was built
against in its metadata (``cache_dir``), and that is an absolute path on the
machine that built it: a user name, a project and a cluster. The metadata is a
JSON string stored as a NumPy unicode array inside a deflated ``.npz``, so the
path is compressed, and even once decompressed NumPy keeps unicode as UTF-32,
four bytes per character. ``grep`` over the archive, or over the unzipped file,
finds nothing. ``scan`` therefore opens every container it meets (zip, npz,
torch checkpoints, gzip, tar.xz) and searches the decompressed bytes in UTF-8,
UTF-16 and UTF-32 at once.

WHAT ``scrub`` CHANGES, AND WHAT IT DOES NOT. Only metadata string values that
look like a machine path or name a machine, a user or a project are replaced by
``"redacted"``. No reader uses those values: ``FoldBook.load`` parses the
metadata and the split comes from ``assignment`` alone. Every other member of
the archive (``assignment``, ``writer_index``) is copied as the same
decompressed bytes, and every other metadata field keeps its value. The result
is reloaded and compared before the run counts as done: the arrays are equal
byte for byte, the metadata differs only in the redacted keys, and a rescan
finds nothing.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import lzma
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

#: Substrings that identify a machine, a user or a project wherever they occur.
STRONG = ("/mnt/", "lustre", "payman", "auto_tads", "grete", "gwdg",
          "goettingen", "keshtkar", "sadegh", "/home/", "/Users/", "/scratch/",
          "/usr/users/", "glogin", "ggpu")

#: A shorter token that is only identifying inside a path (``uni`` alone
#: would match ``uniform``).
PATH_TOKENS = re.compile(r"(^|[/_])uni([/_.-]|$)", re.IGNORECASE)

REDACTED = "redacted"


def sensitive(value: str) -> bool:
    """Whether a metadata string names a machine, a user or a project."""
    low = value.lower()
    if any(token.lower() in low for token in STRONG):
        return True
    return "/" in value and bool(PATH_TOKENS.search(value))


def redact(node, path=()) -> Tuple[object, List[str]]:
    """The metadata with every sensitive string replaced, and where it was."""
    if isinstance(node, dict):
        out, hits = {}, []
        for key, value in node.items():
            out[key], sub = redact(value, path + (str(key),))
            hits += sub
        return out, hits
    if isinstance(node, list):
        out, hits = [], []
        for i, value in enumerate(node):
            new, sub = redact(value, path + (str(i),))
            out.append(new)
            hits += sub
        return out, hits
    if isinstance(node, str) and sensitive(node):
        return REDACTED, [".".join(path)]
    return node, []


# ------------------------------------------------------------------ scanning
def _needles() -> List[Tuple[str, str, bytes]]:
    out = []
    for token in STRONG:
        for enc in ("utf-8", "utf-16-le", "utf-32-le"):
            out.append((token, enc, token.lower().encode(enc)))
    return out


NEEDLES = _needles()


def _lower(blob: bytes) -> bytes:
    return blob.lower()


def search(blob: bytes) -> List[Tuple[str, str]]:
    """Every (token, encoding) present in a byte string, case-insensitive."""
    low = _lower(blob)
    return sorted({(t, e) for t, e, n in NEEDLES if n in low})


def walk(name: str, blob: bytes, depth: int = 0) -> Iterator[Tuple[str, str, str]]:
    """Search a blob, descending into any container it turns out to be."""
    if depth > 6:
        return
    if blob[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for info in z.infolist():
                if not info.is_dir():
                    yield from walk(f"{name}::{info.filename}", z.read(info), depth + 1)
        return
    if blob[:2] == b"\x1f\x8b":
        yield from walk(f"{name}::gunzip", gzip.decompress(blob), depth + 1)
        return
    if blob[:6] == b"\xfd7zXZ\x00":
        data = lzma.decompress(blob)
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as t:
                for member in t:
                    if member.isfile():
                        yield from walk(f"{name}::{member.name}",
                                        t.extractfile(member).read(), depth + 1)
            return
        except tarfile.TarError:
            yield from walk(f"{name}::unxz", data, depth + 1)
            return
    for token, enc in search(blob):
        yield name, token, enc


def scan(paths: List[Path]) -> List[Tuple[str, str, str]]:
    hits = []
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            hits += list(walk(str(path), path.read_bytes()))
    return hits


# ------------------------------------------------------------------ scrubbing
def scrub_bytes(blob: bytes) -> Tuple[bytes, List[str]]:
    """A scrubbed copy of one npz fold book, and the metadata keys it redacted."""
    src = zipfile.ZipFile(io.BytesIO(blob))
    original = np.load(io.BytesIO(blob), allow_pickle=False)
    text = str(original["metadata"])
    meta = json.loads(text)
    if json.dumps(meta) != text:
        raise SystemExit("metadata does not round-trip through json.dumps; refusing "
                         "to rewrite it, because other fields could change")
    clean, hits = redact(meta)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            if info.filename == "metadata.npy":
                buf = io.BytesIO()
                np.save(buf, np.asarray(json.dumps(clean)), allow_pickle=False)
                data = buf.getvalue()
            else:
                data = src.read(info)
            new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new.compress_type = info.compress_type
            new.external_attr = info.external_attr
            dst.writestr(new, data)
    return out.getvalue(), hits


def verify(before: bytes, after: bytes, hits: List[str]) -> None:
    """Refuse unless the scrubbed book is the original with only ``hits`` changed."""
    a, b = zipfile.ZipFile(io.BytesIO(before)), zipfile.ZipFile(io.BytesIO(after))
    if [i.filename for i in a.infolist()] != [i.filename for i in b.infolist()]:
        raise SystemExit("member list changed")
    for name in a.namelist():
        if name != "metadata.npy" and a.read(name) != b.read(name):
            raise SystemExit(f"member {name} changed")
    na = np.load(io.BytesIO(before), allow_pickle=False)
    nb = np.load(io.BytesIO(after), allow_pickle=False)
    for key in na.files:
        if key == "metadata":
            continue
        if na[key].dtype != nb[key].dtype or not np.array_equal(na[key], nb[key]):
            raise SystemExit(f"array {key} changed")
    ma, mb = json.loads(str(na["metadata"])), json.loads(str(nb["metadata"]))
    if list(ma) != list(mb):
        raise SystemExit("metadata keys changed")
    for key in ma:
        if any(h == key or h.startswith(key + ".") for h in hits):
            continue
        if ma[key] != mb[key]:
            raise SystemExit(f"metadata field {key} changed")
    left = list(walk("scrubbed", after))
    if left:
        raise SystemExit(f"still identifying after the scrub: {left}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("paths", nargs="+", type=Path)
    c = sub.add_parser("scrub")
    c.add_argument("books", nargs="+", type=Path)
    c.add_argument("--out-dir", type=Path, default=None,
                   help="Write here instead of in place.")
    args = ap.parse_args()

    if args.cmd == "scan":
        hits = scan(args.paths)
        for name, token, enc in hits:
            print(f"{name}\t{token}\t{enc}")
        print(f"{len(hits)} hit(s) in {len({h[0] for h in hits})} file(s)", file=sys.stderr)
        return 1 if hits else 0

    for book in args.books:
        before = book.read_bytes()
        after, hits = scrub_bytes(before)
        verify(before, after, hits)
        target = (args.out_dir / book.name) if args.out_dir else book
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        print(f"{book.name}: redacted {hits or 'nothing'}  {len(before)} -> {len(after)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
