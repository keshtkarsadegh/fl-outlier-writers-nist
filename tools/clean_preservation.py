#!/usr/bin/env python3
"""
Preservation measured only on source rows the shipped model never trained on.

    python tools/clean_preservation.py --root "$FOA_STUDY_DIR" \\
        --tex paper/main.tex \\
        --json clean_preservation.json --table clean_preservation_table.tex \\
        --numbers clean_preservation_numbers.tex

    T=study/artifacts/Digits_study01/tables/paper
    python tools/clean_preservation.py --root "$FOA_STUDY_DIR" \\
        --tex <out>/tables/{references,agg_winners,reg_winners,combos,scaling}.tex \\
              <out>/tables/selection_axis{,_cross}.tex $T/schedule_table_as_printed.tex \\
        --json $T/clean_preservation.json --table $T/clean_preservation_table.tex \\
        --numbers $T/clean_preservation_numbers.tex

THE PRINTED TABLES CAN COME FROM THE MANUSCRIPT OR FROM THE CLONE. ``--tex``
takes one file or several, read as one text. The clone holds no manuscript, but
`tools/regenerate_paper.sh` writes seven of the eight tables read here into
``<out>/tables/``, with the cell text the page prints; the eighth, the schedule
table, is set by hand in the manuscript and has no generator, so it ships as
printed in ``schedule_table_as_printed.tex``. Either input gives the same
outputs byte for byte, because every label and number is read from the rows.

WHY THIS EXISTS. Every run on every fold starts from ONE shipped model, the
candidate trained on fold 1 of the source book (``g0_selection.json``,
``selected_fold: 1``). The source book's five folds are independent 60/20/20
re-draws of each writer's rows, not a partition, so the test rows of folds 2-5
are, to roughly 54-56%, rows that model was TRAINED on. Preservation as the
tables print it is the mean, over source folds 1-5, of the final model's
accuracy on each fold's test rows (``final_evaluation.old.mean``), so 43.8% of
the rows it averages over are training rows of the model it measures
forgetting from. Only source fold 1's test rows are clean. Its accuracy is
stored separately in every record (``final_evaluation.old.accuracies["1"]``),
so a clean preservation can be read back without re-running anything.

WHAT IS COMPARED. For every arm a printed table carries:

    printed     P  = five-run mean of old.mean            P0  = g0, mean of folds 1-5
    clean       P1 = five-run mean of old.accuracies[1]   P0c = g0 on fold 1 only
    score       (A - A0) - (P0  - P )
    score_clean (A - A0) - (P0c - P1)

Adaptation A and its baseline A0 are untouched: the cohort side was never seen
by the shipped model. The shipped model's own number moves too (P0 -> P0c),
because the same contamination inflates it; leaving P0 at its contaminated
value would charge every arm for a gap that is only the overlap.

WHAT IS CHECKED BEFORE ANYTHING IS REPORTED. Every row of every table this
script covers is matched to exactly one arm of the records by its printed
adaptation and preservation, and every other printed number on that row (spent,
score, the validation columns of the selection-axis tables, the schedule
table's fold-paired differences) is recomputed and compared at the precision it
is printed to. A table whose rows do not all match is a different table from
the one this script reasons about, and it is reported as a failure, not
skipped.

THE VALIDATION AXIS HAS NO CLEAN FIVE-FOLD READING. The selection read
``source_val_accuracies``, and the runner fills that series from the run's own
fold f of the source book (``source_series.fold``). Its validation rows overlap
fold 1's training rows on folds 2-5 as much as the test rows do. The records
keep no per-source-fold validation accuracy and no per-row prediction, so the
clean part of folds 2-5 cannot be separated out. The only clean validation
reading is the fold-1 run's own series. The clean selection re-run here
therefore keeps the five-fold validation adaptation and replaces the five-fold
validation preservation by the fold-1 run's last value. It is one fold, and is
reported as a sensitivity of the selections, not as a replacement for them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
for entry in (str(Path(__file__).resolve().parent), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import report_tables  # noqa: E402

TAG = "d01"
FOLDS = (1, 2, 3, 4, 5)

#: Run-folder stems of the stages the printed tables read, in table order.
STAGES = ("aggfull", "regfull", "combo", "c10d10", "five", "c20d10", "c20d20")

#: Tolerance for "the record says the same thing twice": two fields written
#: from the same floats.
SAME = 1e-12


# ------------------------------------------------------------------ records
def _last(series) -> Optional[float]:
    values = [v for v in (series or []) if v is not None]
    return values[-1] if values else None


def _fold_of(path: Path) -> int:
    folder = next(p for p in path.parts if "_fold" in p and p.startswith(TAG))
    return int(folder.split("_fold")[1][0])


def read_stage(root: Path, stem: str) -> Dict[tuple, Dict[int, dict]]:
    """
    Every run of one stage, per arm and per fold, both axes, every source fold.

    Keyed exactly as ``report_tables.read_runs`` keys its own output, through
    the same ``cell_and_family`` and ``in_study``, so an arm here is an arm
    there. Per fold rather than pooled, because the clean reading needs source
    fold 1 out of every run and the schedule table needs fold-paired values.
    """
    prefix = f"{TAG}_{stem}_"
    out: Dict[tuple, Dict[int, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        cell, family = report_tables.cell_and_family(Path(path), prefix)
        if cell is None or not report_tables.in_study(str(Path(path))):
            continue
        body = next(iter(json.loads(Path(path).read_text()).values()))
        final = body.get("final_evaluation") or {}
        old = final.get("old") or {}
        accs = {int(k): float(v) for k, v in (old.get("accuracies") or {}).items()}
        fold = _fold_of(Path(path))
        series = body.get("source_series") or {}
        # THE RECORD MUST AGREE WITH ITSELF before it is read two ways: the
        # printed preservation is the plain mean of the five per-fold values,
        # the run is the fold its folder says, and its per-round source series
        # is that same fold.
        if sorted(accs) != list(FOLDS):
            raise SystemExit(f"{path}: old.accuracies covers {sorted(accs)}, not folds 1-5")
        if abs(st.fmean(accs.values()) - float(old["mean"])) > SAME:
            raise SystemExit(f"{path}: old.mean is not the mean of old.accuracies")
        if int(final.get("fold")) != fold or int(series.get("fold")) != fold:
            raise SystemExit(f"{path}: folder fold {fold}, final_evaluation.fold "
                             f"{final.get('fold')}, source_series.fold {series.get('fold')}")
        if fold in out[(cell, family)]:
            raise SystemExit(f"{path}: a second run of {cell}/{family} fold {fold}")
        out[(cell, family)][fold] = {
            "a_test": float(final["clients"]["accuracy"]),
            "p_test": float(old["mean"]),
            "p_old": accs,
            "a_val": _last(body.get("pool_val_accuracies")),
            "p_val": _last(body.get("source_val_accuracies")),
            "p_val_round0": (body.get("source_val_accuracies") or [None])[0],
            "series_fold": int(series.get("fold")),
        }
    return {k: v for k, v in out.items() if sorted(v) == list(FOLDS)}


def references(root: Path, tag: str = "c10") -> Dict[str, Dict[int, list]]:
    """
    The reference ladder of Table 1, per fold, with every source fold.

    Read the way ``report_tables.references`` reads it: the mean over every
    record file of the rung (fifty for isolated, one model per client per fold;
    five for centralized), adaptation from ``pooled.own``.
    """
    rungs = {}
    for label, folder, stem, init in (
            ("isolated (from g-0)", f"isolated_{tag}", "isolated", "global"),
            ("isolated (scratch)", f"isolated_{tag}", "isolated", "scratch"),
            ("centralized (from g-0)", f"centralized_{tag}", "centralized", "global"),
            ("centralized (scratch)", f"centralized_{tag}", "centralized", "scratch")):
        files = []
        for path in sorted(glob.glob(str(root / folder / f"{stem}_{init}_*.json"))):
            d = json.loads(Path(path).read_text())
            own = d["pooled"]["own"]
            own = own["mean"] if isinstance(own, dict) else own
            if stem == "isolated":
                (client,) = d["trained_clients"]
                per = d["per_client"][client]
                accs = {int(k): float(v) for k, v in per["old_folds"].items()}
                printed = d["pooled"]["old"]["mean"]
            else:
                accs = {int(k): float(v) for k, v in d["old_folds_accuracy"].items()}
                printed = d["old"]
            if abs(st.fmean(accs.values()) - printed) > SAME:
                raise SystemExit(f"{path}: old mean is not the mean of its folds")
            files.append({"fold": int(d["fold"]), "a_test": float(own),
                          "p_test": float(printed), "p_old": accs})
        rungs[label] = files
    return rungs


# ---------------------------------------------------------------- baselines
def shipped(root: Path) -> dict:
    """
    The shipped model on the source rows, contaminated and clean.

    P0 is what the tables use: the mean over the five source folds' test rows.
    P0c is the same model on source fold 1's test rows only, the one fold whose
    rows it never trained on. Its fold-1 VALIDATION accuracy is the clean
    validation baseline, and it is checked against the round-0 entry of every
    fold-1 run's own series (round 0 is the shipped model, before training).
    """
    g0 = json.loads((root / "g0_evaluations.json").read_text())
    per_fold = {f: float(g0[f"old_data_fold{f}"]["accuracy"]) for f in FOLDS}
    sel = json.loads((root / "g0_selection.json").read_text())
    if int(sel["selected_fold"]) != 1:
        raise SystemExit("g0_selection.json does not select the fold-1 model; the "
                         "clean fold is then not fold 1 and this script does not apply")
    val1 = next(float(f["val_accuracy"]) for f in sel["folds"] if int(f["fold"]) == 1)
    return {"p0": st.fmean(per_fold.values()), "p0_clean": per_fold[1],
            "p0_per_fold": per_fold, "p0_val_clean": val1,
            "selected_fold": int(sel["selected_fold"]), "sha256": sel["sha256"]}


def overlap(root: Path) -> dict:
    """
    How many of each source fold's held-out rows the shipped model trained on.

    Read from the released book itself, so the percentages are a count, not an
    estimate: a row is contaminated when it is a TRAIN row of fold 1.
    """
    import numpy as np
    book = np.load(root / "fold_books" / "old_data.foldbook.npz", allow_pickle=False)
    parts = book["assignment"]
    train1 = parts[0] == 0
    out = {}
    for f in FOLDS:
        row = {}
        for name, code in (("val", 1), ("test", 2)):
            held = parts[f - 1] == code
            row[name] = {"rows": int(held.sum()), "fold1_train": int((held & train1).sum())}
            row[name]["share"] = row[name]["fold1_train"] / row[name]["rows"]
        out[f] = row
    total = sum(out[f]["test"]["rows"] for f in FOLDS)
    hit = sum(out[f]["test"]["fold1_train"] for f in FOLDS)
    return {"per_fold": out, "test_pooled": {"rows": total, "fold1_train": hit,
                                             "share": hit / total}}


# ------------------------------------------------------------------ an arm
def summarise(runs: List[dict], a0: float, base: dict) -> dict:
    """Both readings of one arm, from its per-fold runs."""
    a = [r["a_test"] for r in runs]
    p = [r["p_test"] for r in runs]
    p1 = [r["p_old"][1] for r in runs]
    by_old = {k: st.fmean(r["p_old"][k] for r in runs) for k in FOLDS}
    A, P, P1 = st.fmean(a), st.fmean(p), st.fmean(p1)
    score = (A - a0) - (base["p0"] - P)
    clean = (A - a0) - (base["p0_clean"] - P1)
    return {
        "adaptation": A, "preservation": P, "preservation_sd": st.pstdev(p),
        "preservation_clean": P1, "preservation_clean_sd": st.pstdev(p1),
        "preservation_clean_sd_sample": st.stdev(p1) if len(p1) > 1 else 0.0,
        "preservation_by_old_fold": by_old,
        "printed_minus_clean": P - P1,
        "F": P - base["p0"], "F_clean": P1 - base["p0_clean"],
        "G": A - a0, "score": score, "score_clean": clean,
        "score_change": clean - score, "a0": a0, "runs": len(runs),
    }


def fold_scores(per_fold: Dict[int, dict], a0f: Dict[int, float], base: dict) -> dict:
    """Per-fold score, printed and clean, for fold-paired differences."""
    out = {"score": {}, "score_clean": {}}
    for f, r in per_fold.items():
        out["score"][f] = (r["a_test"] - a0f[f]) - (base["p0"] - r["p_test"])
        out["score_clean"][f] = (r["a_test"] - a0f[f]) - (base["p0_clean"] - r["p_old"][1])
    return out


def val_axis(per_fold: Dict[int, dict]) -> dict:
    """The validation axis the selections read, and its one clean variant."""
    return {"val_adaptation": st.fmean(r["a_val"] for r in per_fold.values()),
            "val_preservation": st.fmean(r["p_val"] for r in per_fold.values()),
            "val_preservation_fold1": per_fold[1]["p_val"],
            "val_preservation_by_run": {f: r["p_val"] for f, r in per_fold.items()}}


# ------------------------------------------------------------ the manuscript
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

#: A fold spread printed after a mean, ``0.9263{\scriptsize±0.0166}``. It is
#: removed before a cell is read, so the mean is what gets compared.
SPREAD = re.compile(r"\{\\scriptsize\s*(?:±|\$\\pm\$|\\pm)[^}]*\}")


def table_rows(tex: str, label: str) -> List[Tuple[int, str, List[str]]]:
    """
    The data rows of one table environment: (block, label text, cells).

    Blocks are counted on the band rows and \\midrule lines a table uses to
    separate its schedules or settings; only rows with numbers in them are
    returned.
    """
    at = tex.index("\\label{%s}" % label)
    start = tex.rfind("\\begin{table}", 0, at)
    end = tex.index("\\end{table}", at)
    body = tex[start:end]
    body = body[body.index("\\toprule"):] if "\\toprule" in body else body
    block, rows, seen_data = 0, [], False
    for line in body.splitlines():
        s = SPREAD.sub("", line.strip())
        # A band row, a rule, or a caption row spanning the table all open a
        # new block once the current one holds data.
        opens = (s.startswith("\\rowcolor") or s.startswith("\\midrule")
                 or ("multicolumn" in s and "&" not in s))
        if opens:
            if seen_data:
                block += 1
                seen_data = False
            continue
        if "&" not in s or not s.endswith("\\\\") or "multicolumn" in s:
            continue
        cells = [c.strip() for c in s[:-2].split("&")]
        if not any(NUM.fullmatch(c.replace("$", "").strip()) for c in cells[1:]):
            continue
        rows.append((block, cells[0], cells[1:]))
        seen_data = True
    return rows


def rows_or_fail(tex: str, label: str, failures: List[str]):
    """The rows of one table; an empty table is recorded as a failure."""
    rows = table_rows(tex, label)
    if not rows:
        failures.append(f"{label}: no data rows could be read")
    return rows


def number(cell: str) -> float:
    m = NUM.search(cell.replace("{", " ").replace("}", " "))
    if m is None:
        raise ValueError(cell)
    return float(m.group(0))


def same(printed: str, value: float, nd: int) -> bool:
    """A printed number equals a value at the printed precision."""
    return f"{value:.{nd}f}" == f"{number(printed):.{nd}f}"


def decimals(cell: str) -> int:
    m = NUM.search(cell)
    text = m.group(0)
    return len(text.split(".")[1]) if "." in text else 0


# ------------------------------------------------------------------ matching
def match(rows, arms: Dict[tuple, dict], a_col: int, p_col: int, scale: float,
          family_of_block=None) -> List[tuple]:
    """
    Each printed row's arm, by its printed adaptation and preservation.

    A row matching no arm or more than one is a failure: the script reasons
    about the arms it found and must find exactly the ones printed.
    """
    out = []
    for block, label, cells in rows:
        nd_a, nd_p = decimals(cells[a_col]), decimals(cells[p_col])
        hits = [key for key, s in arms.items()
                if (family_of_block is None or key[1] == family_of_block(block, cells))
                and same(cells[a_col], s["adaptation"] * scale, nd_a)
                and same(cells[p_col], s["preservation"] * scale, nd_p)]
        out.append((block, label, cells, hits))
    return out


def block_family(names):
    return lambda block, cells: names[block]


# ------------------------------------------------------------------ orderings
def ordering_change(entries: List[dict]) -> dict:
    """
    Where each arm sits by printed score and by clean score, within one block.

    Unrounded scores, ties to the label, so a tie that the printed two
    decimals show is not reported as a move.
    """
    by_p = sorted(entries, key=lambda e: (-e["score"], e["label"]))
    by_c = sorted(entries, key=lambda e: (-e["score_clean"], e["label"]))
    rank_p = {e["label"]: i for i, e in enumerate(by_p, 1)}
    rank_c = {e["label"]: i for i, e in enumerate(by_c, 1)}
    moved = [{"label": e["label"], "rank_printed": rank_p[e["label"]],
              "rank_clean": rank_c[e["label"]],
              "score": e["score"], "score_clean": e["score_clean"]}
             for e in by_p if rank_p[e["label"]] != rank_c[e["label"]]]
    return {"n": len(entries), "moved": moved,
            "top_printed": by_p[0]["label"], "top_clean": by_c[0]["label"]}


def shortlist(rows: List[dict], key: str) -> List[str]:
    """One cell per method, the first three by ``key`` (ties by cell id)."""
    seen, top = set(), []
    for r in sorted(rows, key=lambda r: (-r[key], r["cell"])):
        if r["method"] in seen:
            continue
        seen.add(r["method"])
        top.append(r["cell"])
        if len(top) == 3:
            break
    return top


# ------------------------------------------------- per-method screen winners
def screen_winners(root: Path, base: dict, a0: float) -> dict:
    """
    Each method's screen winner, as selected and with one fold's preservation.

    The study's own selection code is reused: the catalogue, the extensions,
    the readers and the trade score. The only change is that the preservation
    of every fold but one is withdrawn before the fold means are taken, so the
    fold mean is that fold's value. The unchanged run is checked against the
    winners the selection records store, so a reproduction failure is not
    mistaken for an effect of the overlap.

    WHY ALL FIVE SINGLE FOLDS AND NOT ONLY FOLD 1. Fold 1 is the clean one,
    but it is also one fold where the selection averaged five. A winner that
    moves under fold 1 alone has moved for one of two reasons - the overlap, or
    reading one fold's noise - and the four contaminated single folds are the
    control that tells them apart: if they move it as often, fold 1 has shown
    noise, not contamination.
    """
    import study_emit as SE
    from federated_outlier_adaptation.training import agg_cells, reg_cells
    from federated_outlier_adaptation.training import study_lines as SL

    p0 = base["p0"]
    result = {"aggregation": {}, "regularisation": {}}

    cells = SE.with_extensions(root, agg_cells.screen_cells(), "agg")
    by_method: Dict[tuple, list] = {}
    for cell in cells:
        by_method.setdefault((cell["path"], SL.agg_method_of(cell)), []).append(cell)
    stored = json.loads((root / "tables" / "p11_agg_method_winners.json").read_text())
    for keep in (None,) + FOLDS:
        found = SE.agg_selector.collect(root, cells, f"{TAG}_agg_")
        if keep is not None:
            for entry in found.values():
                for fold, rec in entry["folds"].items():
                    if fold != keep:
                        rec["preservation"] = None
        rows = SE.agg_selector.summarise(found)
        for key in SL.AGG_METHODS:
            ids = {c["id"] for c in by_method[key]}
            scored = [r for r in rows if r["id"] in ids and r["adaptation"]["mean"] is not None
                      and r["preservation"]["mean"] is not None]
            top = max(scored, key=lambda r: SE.trade_score(r, a0, p0))
            name = "/".join(key)
            slot = result["aggregation"].setdefault(name, {"stored": stored[name]["winner"]})
            slot["reproduced" if keep is None else f"fold{keep}"] = top["id"]
            slot["considered"] = len(scored)

    cells = SE.with_extensions(root, reg_cells.screen_cells(), "reg")
    by_rmethod: Dict[str, list] = {name: [] for name in reg_cells.methods()}
    for cell in cells:
        by_rmethod.setdefault(cell["method"], []).append(cell)
    stored = json.loads((root / "tables" / "p13_reg_method_winners.json").read_text())
    for keep in (None,) + FOLDS:
        found = SE.reg_selector.collect(root, cells, f"{TAG}_reg_")
        if keep is not None:
            for entry in found.values():
                for fold, fams in entry["folds"].items():
                    if fold != keep:
                        for rec in fams.values():
                            rec["preservation"] = None
        rows = SE.reg_selector.summarise(found)
        for method in SL.REG_METHODS:
            ids = {c["id"] for c in by_rmethod[method]}
            for family in SL.FAMILIES:
                scored = [r for r in rows if r["id"] in ids and r["family"] == family
                          and r["adaptation"]["mean"] is not None
                          and r["preservation"]["mean"] is not None]
                if not scored:
                    continue
                top = max(scored, key=lambda r: SE.trade_score(r, a0, p0))
                name = f"{method}/{family}"
                slot = result["regularisation"].setdefault(
                    name, {"stored": (stored.get(name) or {}).get("winner")})
                slot["reproduced" if keep is None else f"fold{keep}"] = top["id"]
                slot["considered"] = len(scored)
    return result


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="Assembled study root.")
    ap.add_argument("--tex", required=True, type=Path, nargs="+",
                    help="The manuscript, or the files that print its tables.")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--numbers", type=Path, default=None,
                    help="Also write the LaTeX macros the manuscript quotes.")
    ap.add_argument("--no-screens", action="store_true",
                    help="Skip re-running the per-method screen selections.")
    args = ap.parse_args()
    root, tex = args.root, "\n".join(path.read_text() for path in args.tex)

    base = shipped(root)
    book = overlap(root)
    stage_runs = {stem: read_stage(root, stem) for stem in STAGES}
    a0 = {stem: report_tables.baselines_for(root, stem)[0] for stem in STAGES}
    g0c = {}
    for stem in STAGES:
        book_name = report_tables.COHORT_BOOKS[report_tables.cohort_of(stem)]
        g = json.loads((root / book_name).read_text())
        recs = g if isinstance(g, list) else list(g.values())
        g0c[stem] = {int(r["fold"]): float(r["accuracy"]) for r in recs}
    arms = {stem: {key: summarise(list(runs.values()), a0[stem], base)
                   for key, runs in stage_runs[stem].items()} for stem in STAGES}
    for stem in STAGES:
        for key, runs in stage_runs[stem].items():
            arms[stem][key].update(val_axis(runs))

    # The per-round source series and the shipped model's round-0 value on it.
    series_check = {"series_fold_equals_run_fold": True, "round0_by_fold": defaultdict(set)}
    for stem in STAGES:
        for runs in stage_runs[stem].values():
            for f, r in runs.items():
                series_check["round0_by_fold"][f].add(round(r["p_val_round0"], 12))
    series_check["round0_by_fold"] = {f: sorted(v) for f, v in
                                      sorted(series_check["round0_by_fold"].items())}
    if series_check["round0_by_fold"][1] != [round(base["p0_val_clean"], 12)]:
        raise SystemExit("round 0 of the fold-1 source series is not g0's fold-1 "
                         "validation accuracy")

    # Reference ladder.
    ref_a0, _ = report_tables.baselines(root, "c10")
    ref = {}
    for label, files in references(root).items():
        s = summarise(files, ref_a0, base)
        s["preservation_clean_sd"] = st.pstdev(
            st.fmean(x["p_old"][1] for x in files if x["fold"] == f) for f in FOLDS)
        ref[label] = s
    report = {"baselines": base, "overlap": book, "series": series_check,
              "tables": {}, "failures": []}

    # ------------------------------------------------------ table by table
    def check_row(table, label, cells, s, cols):
        """cols: list of (cell index, value, decimals-or-None)."""
        for idx, value, _ in cols:
            if not same(cells[idx], value, decimals(cells[idx])):
                report["failures"].append(
                    f"{table}: {label!r} column {idx} prints {cells[idx]} "
                    f"but the records give {value:.6f}")

    def entry(table, block, label, stem, key, s):
        return {"table": table, "block": block, "label": label, "stage": stem,
                "cell": key[0], "family": key[1],
                **{k: v for k, v in s.items() if not isinstance(v, dict)},
                "preservation_by_old_fold": s["preservation_by_old_fold"]}

    out_rows: List[dict] = []

    def run_table(table, label, rows, pool, a_col, p_col, scale, fam, extra):
        found = match(rows, pool, a_col, p_col, scale, fam)
        for block, lab, cells, hits in found:
            if len(hits) != 1:
                report["failures"].append(f"{table}: {lab!r} matches {len(hits)} arms")
                continue
            key = hits[0]
            s = pool[key]
            check_row(table, lab, cells, s, extra(s, cells))
            stem = s.get("_stem", "")
            out_rows.append(entry(table, block, lab, stem, key, s))

    def pool_of(stem):
        """One stage's arms, keyed (cell, schedule)."""
        return {key: dict(s, _stem=stem) for key, s in arms[stem].items()}

    def pool2(*stems):
        """(cell, family) keys that may repeat across stages: keep the stage."""
        pool = {}
        for stem in stems:
            for key, s in arms[stem].items():
                pool[(key[0], key[1], stem)] = dict(s, _stem=stem)
        return pool

    # Table 1: the reference ladder.
    rows = rows_or_fail(tex, "tab:references", report["failures"])
    rpool = {(k, "-"): dict(v, _stem="references") for k, v in ref.items()}
    run_table("tab:references", "", rows, rpool, 0, 1, 1.0, None,
              lambda s, c: [(2, s["G"] * 100, 2), (3, -s["F"] * 100, 2),
                            (4, s["score"] * 100, 2)])

    # Table 2.
    par_cyc = block_family({0: "parallel", 1: "cyclic"})
    rows = rows_or_fail(tex, "tab:agg_winners", report["failures"])
    run_table("tab:agg_winners", "", rows, pool_of("aggfull"), 0, 1, 1.0, par_cyc,
              lambda s, c: [(2, -s["F"] * 100, 2), (3, s["score"] * 100, 2)])

    # Table 3: the penalties plus the aggregation control row.
    rows = rows_or_fail(tex, "tab:reg_winners", report["failures"])
    pool = {k: v for k, v in pool2("regfull", "aggfull").items()
            if k[2] == "regfull" or k[0] == "control_fedavg"}
    run_table("tab:reg_winners", "", rows, pool, 0, 1, 1.0, par_cyc,
              lambda s, c: [(2, -s["F"] * 100, 2), (3, s["score"] * 100, 2)])

    # Table 4: the cross.
    rows = rows_or_fail(tex, "tab:combos", report["failures"])
    run_table("tab:combos", "", [(b, l + " + " + c[0], c[1:]) for b, l, c in rows],
              pool_of("combo"), 0, 1, 1.0, par_cyc,
              lambda s, c: [(2, s["score"] * 100, 2)])

    # Table 7: four settings.
    rows = rows_or_fail(tex, "tab:scaling", report["failures"])
    settings = {0: "c10d10", 1: "five", 2: "c20d10", 3: "c20d20"}
    for block, stem in settings.items():
        sub = [(b, l + " / " + c[0], c) for b, l, c in rows if b == block]
        run_table("tab:scaling", "", sub, pool_of(stem), 1, 2, 1.0,
                  lambda b, c: c[0].strip(),
                  lambda s, c: [(3, s["score"] * 100, 2)])
    # The scaling rows carry their schedule in a column; re-check it.
    for r in out_rows:
        if r["table"] == "tab:scaling" and not r["label"].endswith(r["family"]):
            report["failures"].append(f"tab:scaling: {r['label']!r} matched a "
                                      f"{r['family']} arm")

    # Table 5: schedule pairs, percent, fold-paired differences.
    rows = rows_or_fail(tex, "tab:schedule", report["failures"])
    spool = pool2("aggfull", "regfull", "combo")
    sched = []
    for block, lab, cells in rows:
        halves = []
        for fam, ca, cp in (("parallel", 0, 1), ("cyclic", 2, 3)):
            hits = [k for k, s in spool.items() if k[1] == fam
                    and same(cells[ca], s["adaptation"] * 100, 2)
                    and same(cells[cp], s["preservation"] * 100, 2)]
            # The same runs can sit under two stages' readers only if they are
            # the same arm; prefer the stage order the table states.
            if len({(spool[h]["adaptation"], spool[h]["preservation"]) for h in hits}) != 1:
                report["failures"].append(f"tab:schedule: {lab!r} {fam} matches {hits}")
                halves.append(None)
                continue
            halves.append(hits[0])
        if None in halves:
            continue
        hp, hc = halves
        sp, sc = spool[hp], spool[hc]
        rp = stage_runs[hp[2]][(hp[0], hp[1])]
        rc = stage_runs[hc[2]][(hc[0], hc[1])]
        fp = fold_scores(rp, g0c[hp[2]], base)
        fc = fold_scores(rc, g0c[hc[2]], base)
        diff = {f: (fc["score"][f] - fp["score"][f]) * 100 for f in FOLDS}
        diffc = {f: (fc["score_clean"][f] - fp["score_clean"][f]) * 100 for f in FOLDS}
        won = sum(1 for v in diff.values() if v > 0)
        wonc = sum(1 for v in diffc.values() if v > 0)
        printed_won = re.search(r"\((\d)/5\)", cells[6])
        check_row("tab:schedule", lab, cells, sp,
                  [(4, sp["score"] * 100, 2), (5, sc["score"] * 100, 2),
                   (6, st.fmean(diff.values()), 2)])
        if printed_won is None or int(printed_won.group(1)) != won:
            report["failures"].append(f"tab:schedule: {lab!r} folds won {won} vs {cells[6]}")
        sched.append({"label": lab, "parallel": hp, "cyclic": hc,
                      "diff_mean": st.fmean(diff.values()), "folds_won": won,
                      "diff_mean_clean": st.fmean(diffc.values()), "folds_won_clean": wonc,
                      "diff_by_fold": diff, "diff_by_fold_clean": diffc,
                      "sign_changes": (st.fmean(diff.values()) > 0)
                      != (st.fmean(diffc.values()) > 0),
                      "folds_won_changes": won != wonc})
        for key, s in ((hp, sp), (hc, sc)):
            out_rows.append(entry("tab:schedule", block, lab + " / " + key[1],
                                  key[2], key, s))
    report["tables"]["tab:schedule"] = sched

    # B1 and B2: both axes.
    sel_rows = list(csv.DictReader(open(root / "tables" / "paper" / "selection_axis.csv")))
    axis = {"aggregation": "aggfull", "regularisation": "regfull", "combination": "combo"}
    for table in ("tab:selection_axis", "tab:selection_axis_cross"):
        rows = rows_or_fail(tex, table, report["failures"])
        stems = ("aggfull", "regfull") if table == "tab:selection_axis" else ("combo",)
        blocks = ({0: ("aggfull", "parallel"), 1: ("aggfull", "cyclic"),
                   2: ("regfull", "parallel"), 3: ("regfull", "cyclic")}
                  if table == "tab:selection_axis" else
                  {0: ("combo", "parallel"), 1: ("combo", "cyclic")})
        # The band row of B1 opens a block, then each schedule's caption row is
        # a multicolumn that table_rows skips: re-derive blocks from order.
        current, prev = 0, None
        for block, lab, cells in rows:
            stem, fam = blocks[block] if block in blocks else (None, None)
            pool = {k: v for k, v in pool_of(stem).items() if k[1] == fam} if stem else {}
            hits = [k for k, s in pool.items()
                    if same(cells[4], s["adaptation"], 4) and same(cells[5], s["preservation"], 4)]
            if len(hits) != 1:
                report["failures"].append(f"{table}: {lab!r} matches {len(hits)} arms")
                continue
            s = pool[hits[0]]
            a0s = a0[stem]
            vscore = (s["val_adaptation"] - a0s) - (base["p0"] - s["val_preservation"])
            check_row(table, lab, cells, s,
                      [(0, s["val_adaptation"], 4), (1, s["val_preservation"], 4),
                       (2, vscore * 100, 2), (6, s["score"] * 100, 2)])
            out_rows.append(entry(table, block, lab, stem, hits[0], s))

    # ------------------------------------------------------ orderings
    orders = {}
    for table in ("tab:agg_winners", "tab:reg_winners", "tab:combos", "tab:scaling",
                  "tab:references"):
        for block in sorted({r["block"] for r in out_rows if r["table"] == table}):
            ents = [r for r in out_rows if r["table"] == table and r["block"] == block]
            orders[f"{table}/{block}"] = ordering_change(ents)
    report["orderings"] = orders

    # Validation-axis selections: as selected, with the clean fold-1 run's
    # preservation, and - as the control for reading one fold instead of five -
    # with each contaminated single fold in its place. Only the ordering
    # matters to a selection, so the constant baselines are left out.
    sel = {}
    for stage, stem in axis.items():
        for fam in ("parallel", "cyclic"):
            pop = [r for r in sel_rows if r["stage"] == stage and r["schedule"] == fam]
            rows = []
            for r in pop:
                s = arms[stem][(r["cell"], fam)]
                if abs(s["val_preservation"] - float(r["val_preservation"])) > 5e-7:
                    report["failures"].append(f"selection_axis.csv {r['cell']}: val "
                                              "preservation differs from the records")
                row = {"cell": r["cell"], "method": r["method"] or r["cell"],
                       "role": r["role"],
                       "v": s["val_adaptation"] + s["val_preservation"],
                       "t": s["adaptation"] + s["preservation"] - base["p0"],
                       "t_clean": s["adaptation"] + s["preservation_clean"] - base["p0_clean"]}
                for k in FOLDS:
                    row[f"v{k}"] = s["val_adaptation"] + s["val_preservation_by_run"][k]
                rows.append(row)
            sel[f"{stage}/{fam}"] = rows

    def order(rows, key):
        return [r["cell"] for r in sorted(rows, key=lambda r: (-r[key], r["cell"]))]

    selections = {}
    for name in ("aggregation", "regularisation"):
        for fam in ("parallel", "cyclic"):
            rows = sel[f"{name}/{fam}"]
            entry_ = {
                "shortlist_stored": sorted(r["cell"] for r in rows if r["role"] == "shortlist"),
                "shortlist_reproduced": sorted(shortlist(rows, "v")),
                "val_order": order(rows, "v"),
            }
            for k in FOLDS:
                entry_[f"shortlist_val_fold{k}"] = sorted(shortlist(rows, f"v{k}"))
                entry_[f"val_order_fold{k}"] = order(rows, f"v{k}")
            selections[f"{name}/{fam}"] = entry_
    cross = sel["combination/parallel"] + sel["combination/cyclic"]
    crowning = {
        "crowned_stored": [r["cell"] for r in cross if r["role"] == "crowned"],
        "crowned_reproduced": order(cross, "v")[0],
        "val_order": order(cross, "v"),
        "test_top_printed": order(cross, "t")[0],
        "test_top_clean": order(cross, "t_clean")[0],
    }
    for k in FOLDS:
        crowning[f"crowned_val_fold{k}"] = order(cross, f"v{k}")[0]
        crowning[f"val_order_fold{k}"] = order(cross, f"v{k}")
    selections["crowning"] = crowning

    # The same control on the test axis: how many arms move within a block
    # when the five-fold source mean is replaced by any single source fold.
    single = {}
    for name, ent in orders.items():
        table, block = name.rsplit("/", 1)
        ents = [r for r in out_rows if r["table"] == table and r["block"] == int(block)]
        by_p = sorted(ents, key=lambda e: (-e["score"], e["label"]))
        rank_p = {e["label"]: i for i, e in enumerate(by_p, 1)}
        single[name] = {}
        for k in FOLDS:
            def sk(e, k=k):
                return (e["G"] - (base["p0_per_fold"][k]
                                  - e["preservation_by_old_fold"][k]))
            by_k = sorted(ents, key=lambda e: (-sk(e), e["label"]))
            single[name][k] = sum(1 for i, e in enumerate(by_k, 1)
                                  if rank_p[e["label"]] != i)
    report["single_fold_moves"] = single
    if not args.no_screens:
        selections["screen_winners"] = screen_winners(root, base, a0["aggfull"])
    report["selections"] = selections

    # ------------------------------------------------------ summary
    printed_rows = [r for r in out_rows]
    deltas = [r["printed_minus_clean"] for r in printed_rows]
    changes = [r["score_change"] for r in printed_rows]
    report["summary"] = {
        "rows_matched": len(printed_rows),
        "failures": len(report["failures"]),
        "printed_minus_clean_pts": [min(deltas) * 100, max(deltas) * 100],
        "score_change_pts": [min(changes) * 100, max(changes) * 100],
        "p0_minus_p0_clean_pts": (base["p0"] - base["p0_clean"]) * 100,
        "orderings_changed": {k: v["moved"] for k, v in orders.items() if v["moved"]},
        "schedule_sign_changes": [s["label"] for s in sched if s["sign_changes"]],
        "schedule_folds_won_changes": [(s["label"], s["folds_won"], s["folds_won_clean"])
                                       for s in sched if s["folds_won_changes"]],
    }
    report["rows"] = printed_rows
    report["numbers"] = numbers(printed_rows, base, book)
    if args.numbers:
        write_numbers(args.numbers, report["numbers"])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1, default=str) + "\n")

    write_table(args.table, printed_rows, base, sched)
    print(json.dumps(report["summary"], indent=1, default=str))
    for failure in report["failures"]:
        print("FAIL", failure)
    return 1 if report["failures"] else 0


def gap_narrowing(rows: List[dict]) -> Tuple[float, dict]:
    """
    The largest amount by which a protected arm's lead over the control shrinks.

    A protected arm is every non-control row of Tables 3, 4 and 7. Its control
    is "FedAvg (control)" of the same schedule in the same setting: the
    aggregation stage's control for Tables 3 and 4, and the block's own
    control row for the Table 7 settings that print one. The narrowing is the
    control's score change less the arm's, in points. A positive value means
    the clean reading shrinks the arm's lead.
    """
    ctrl = {(r["stage"], r["family"]): r["score_change"] for r in rows
            if "control" in r["label"] and "FedAvg" in r["label"]}
    best, where = float("-inf"), {}
    for r in rows:
        if r["table"] not in ("tab:reg_winners", "tab:combos", "tab:scaling"):
            continue
        if "control" in r["label"]:
            continue
        key = (r["stage"] if r["table"] == "tab:scaling" else "aggfull", r["family"])
        if key not in ctrl:
            continue
        value = (ctrl[key] - r["score_change"]) * 100
        if value > best:
            best, where = value, {"table": r["table"], "label": r["label"],
                                  "stage": r["stage"], "family": r["family"]}
    return best, where


def numbers(rows: List[dict], base: dict, book: dict) -> dict:
    """
    The figures the manuscript quotes from this analysis, unrounded.

    Shifts are over the federated rows only (Table 1's reference ladder is
    left out, as its scratch rungs share no starting point with the rest).
    The fold overlap is the test-row share of source folds 2-5 that are
    fold-1 training rows.
    """
    fed = [r for r in rows if r["table"] != "tab:references"]
    shares = [book["per_fold"][f]["test"]["share"] * 100 for f in FOLDS if f != 1]
    gap, where = gap_narrowing(rows)
    return {
        "nCleanShiftMin": min(r["score_change"] for r in fed) * 100,
        "nCleanShiftMax": max(r["score_change"] for r in fed) * 100,
        "nCleanPresShiftMin": min(r["printed_minus_clean"] for r in fed) * 100,
        "nCleanPresShiftMax": max(r["printed_minus_clean"] for r in fed) * 100,
        "nCleanOverlapPooled": book["test_pooled"]["share"] * 100,
        "nCleanOverlapFoldMin": min(shares),
        "nCleanOverlapFoldMax": max(shares),
        "nCleanGZeroFoldOne": base["p0_clean"],
        "nCleanGapNarrow": gap,
        "_gap_narrow_arm": where,
    }


#: How each macro is printed: points to two decimals, percentages and the gap
#: to one, the shipped model's accuracy to four, as the tables print them.
NUMBER_FORMAT = {
    "nCleanShiftMin": "%.2f", "nCleanShiftMax": "%.2f",
    "nCleanPresShiftMin": "%.2f", "nCleanPresShiftMax": "%.2f",
    "nCleanOverlapPooled": "%.1f", "nCleanOverlapFoldMin": "%.1f",
    "nCleanOverlapFoldMax": "%.1f", "nCleanGZeroFoldOne": "%.4f",
    "nCleanGapNarrow": "%.1f",
}


def write_numbers(path: Path, values: dict) -> None:
    lines = ["% Generated by tools/clean_preservation.py. Do not edit by hand."]
    for name, fmt in NUMBER_FORMAT.items():
        text = fmt % values[name]
        lines.append("\\newcommand{\\%s}{%s}" % (name, text))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


#: Block headers name the table each block comes from by its label, so the
#: page prints the manuscript's own number for it whatever the numbering is.
TABLE_NAMES = {
    "tab:references": "Table~\\ref{tab:references} (reference ladder)",
    "tab:agg_winners": "Table~\\ref{tab:agg_winners} (server rules)",
    "tab:reg_winners": "Table~\\ref{tab:reg_winners} (client penalties)",
    "tab:combos": "Table~\\ref{tab:combos} (cross)",
    "tab:schedule": "Table~\\ref{tab:schedule} (schedule pairs)",
    "tab:scaling": "Table~\\ref{tab:scaling} (settings)",
    "tab:selection_axis": "Table~\\ref{tab:selection_axis} (shortlists, test columns)",
    "tab:selection_axis_cross": "Table~\\ref{tab:selection_axis_cross} (cross, test columns)",
}


#: Two arms the tables renamed after clean_preservation.json was written. The
#: JSON keeps the cell text the tables printed then, which is how a row is found
#: again in its table; the table written here prints the names they print now,
#: so it matches the manuscript. A JSON rebuilt from today's tables already
#: carries the new names, and the replacement is then a no-op.
RENAMED = {
    "eta_0p95": ("FedAvg (", "Server step ("),
    "control_fedavg_earlystop": ("FedAvg (control, stop rule armed)",
                                 "FedAvg (control, replicate)"),
}


def write_table(path: Path, rows: List[dict], base: dict, sched: List[dict]) -> None:
    """
    One longtable of every printed arm, printed against clean, then the
    schedule pairs.

    Labels are the manuscript's own cell text, so a row can be found in the
    table it came from. An arm printed in more than one table is listed once,
    under the first table that prints it: the schedule table and the two
    selection-axis tables reuse the runs of Tables 2-4, and repeating them
    would only repeat their numbers.
    """
    lines = [
        "% Generated by tools/clean_preservation.py. Do not edit by hand.",
        "% P: printed preservation (source splits 1-5). P1: source split 1 only,",
        "% the one split the shipped model never trained on (five-run mean, SD",
        "% over the five runs). Score: (A-A0)-(P0-P); clean: (A-A0)-(P0c-P1),",
        "%% P0 = %.4f, P0c = %.4f." % (base["p0"], base["p0_clean"]),
        # Set small and tight, so the long arm labels fit the text width.
        "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{@{}llrrrrrr@{}}",
        "\\caption{Preservation on the source rows the shipped model never trained on."
        " $P$ is the printed preservation (test rows of source splits 1--5);"
        " $P_1$ is the same runs on source split~1's test rows only, mean (SD) over the"
        " five runs. Scores in points; the clean score uses $P_0^{c}=%.4f$ in place of"
        " $P_0=%.4f$.}\\label{tab:clean_preservation}\\\\" % (base["p0_clean"], base["p0"]),
        "\\toprule",
        "Arm & Sched. & $P$ & $P_1$ (SD) & $P-P_1$ & Score & Clean & $\\Delta$ \\\\",
        "\\midrule\\endfirsthead",
        "\\toprule",
        "Arm & Sched. & $P$ & $P_1$ (SD) & $P-P_1$ & Score & Clean & $\\Delta$ \\\\",
        "\\midrule\\endhead",
    ]
    current, listed = None, set()
    for r in rows:
        key = (r["stage"], r["cell"], r["family"])
        if key in listed and r["stage"] != "references":
            continue
        listed.add(key)
        if r["table"] != current:
            current = r["table"]
            lines.append("\\multicolumn{8}{@{}l}{\\emph{%s}} \\\\" % TABLE_NAMES[current])
        label = r["label"].replace("\\quad", "").strip()
        old, new = RENAMED.get(r["cell"], ("", ""))
        if old and label.startswith(old):
            label = new + label[len(old):]
        lines.append(
            "%s & %s & %.4f & %.4f (%.4f) & $%.2f$ & $%.2f$ & $%.2f$ & $%+.2f$ \\\\" % (
                label, r["family"], r["preservation"], r["preservation_clean"],
                r["preservation_clean_sd"], r["printed_minus_clean"] * 100,
                r["score"] * 100, r["score_clean"] * 100, r["score_change"] * 100))
    lines += ["\\bottomrule", "\\end{longtable}", "\\endgroup", "",
              "\\begin{table}[!htbp]", "\\centering",
              "\\caption{The schedule pairs of Table~\\ref{tab:schedule} with clean"
              " preservation: the"
              " split-paired cyclic $-$ parallel score difference (points) and the splits"
              " the cyclic side wins.}\\label{tab:clean_schedule}", "\\small",
              "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
              "Pair & Printed & Clean & Splits won & Clean \\\\", "\\midrule"]
    for s in sched:
        lines.append("%s & $%+.2f$ & $%+.2f$ & %d/5 & %d/5 \\\\" % (
            s["label"].strip(), s["diff_mean"], s["diff_mean_clean"],
            s["folds_won"], s["folds_won_clean"]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
