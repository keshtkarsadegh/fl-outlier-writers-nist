#!/usr/bin/env python3
"""
What the forgetting-weight sensitivity counts, said in a table rather than a ratio.

Appendix B states that recomputing every method's winning cell at
``w`` in {2, 3, 5} "moves the winner in two, four and seven of the 17
aggregation methods and in four, five and eight of the 14 penalties". Neither
denominator is a row count of a table the paper prints - Table 2 ranks fifteen
server rules, Table 3 prints seven penalty families per schedule beside the
composites and a control - so a reader cannot check the sentence against the
page. This tool says what the two numbers count, and which methods move.

    python tools/sensitivity_counts.py --root "$FOA_STUDY_DIR"
    python tools/sensitivity_counts.py --root "$FOA_STUDY_DIR" --out OUT/

WHAT `weight_sensitivity.py` COUNTS, read from it rather than restated: one row
per SELECTION GROUP of the two quarter-horizon SCREENS (``d01_agg_`` and
``d01_reg_``), scored on VALIDATION (the last round of ``pool_val_accuracies``
against ``source_val_accuracies``), which is what the selection itself read.

  * the aggregation groups are ``(path, method)`` of ``agg_cells.screen_cells``:
    eight parallel rule families, seven cyclic ones, and the two FedAvg control
    arms, which the screen carries as groups of their own - 17;
  * the regularisation groups are ``(schedule, method)``: the seven penalty
    families screened on each schedule - 14.

A group holding one cell has nothing to move to, so it is in the denominator
and can never be in the numerator: the two controls, the coordinate median and
six of the seven cyclic rules are such groups, so only eight of the 17
aggregation groups can move at all. The composites of Table 3 were
built after the screen and are not in it, and Tables 2 and 3 print the
full-horizon finals of each group's winner, not the screen.

Nothing is recomputed differently here: the groups, the scores and the winners
are ``weight_sensitivity.agg_rows``, ``reg_rows`` and ``sensitivity`` over the
same baselines, so the counts are the ones `make_numbers.py` sets from
``weight_sensitivity_*.csv`` and are checked against those views.

WHAT IT WRITES, into ``--out`` (default the metadata core's ``tables``, beside
``weight_sensitivity_*.csv``):

    w_sensitivity_table.tex   per grid and schedule: groups, groups with more
                              than one cell, and at each w the methods whose
                              winner moves, by name
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tools", REPO / "tools" / "paper_figures"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import weight_sensitivity as WS  # noqa: E402
from paper_names import FAMILY, label  # noqa: E402

DEFAULT_OUT = REPO / "study" / "artifacts" / "Digits_study01" / "tables"

#: The weights the appendix quotes, beside the reference the study runs at.
WEIGHTS = WS.DEFAULT_WEIGHTS

#: How a group's first key part is printed: the screen's own path names.
PATH_NAME = {"concurrent": "parallel", "sequential": "cyclic", "control": "control"}


def method_name(row: dict) -> str:
    """A group's published family name; a control group is named as its cell."""
    return FAMILY.get(row["method"]) or label(row["method"])


def rows_for(grid: str, root: Path, a0: float, p0: float) -> List[dict]:
    builder, _title = WS.GRIDS[grid]
    grouped = builder(root)
    out = []
    for row in WS.sensitivity(grouped, a0, p0, WEIGHTS):
        key = tuple(row["group"].split("/"))
        out.append({
            "grid": grid,
            "path": PATH_NAME.get(key[0], key[0]),
            "method": key[1],
            "cells": len(grouped[key]),
            "winners": {w: row["winners"][w] for w in WEIGHTS},
            "moved": {w: row["winners"][w] != row["winners"][WEIGHTS[0]] for w in WEIGHTS[1:]},
        })
    return out


def check_view(rows: List[dict], grid: str, root: Path) -> List[str]:
    """The winners against the shipped ``weight_sensitivity_<grid>.csv``."""
    path = root / "tables" / f"weight_sensitivity_{grid}.csv"
    if not path.is_file():
        path = DEFAULT_OUT / f"weight_sensitivity_{grid}.csv"
    view = {r["method"]: r for r in csv.DictReader(open(path))}
    problems = []
    if len(view) != len(rows):
        problems.append(f"{path.name}: {len(view)} rows, {len(rows)} recomputed")
    for row in rows:
        key = {"parallel": "concurrent", "cyclic": "sequential"}.get(row["path"], row["path"])
        shipped = view.get(f"{key}/{row['method']}")
        if shipped is None:
            problems.append(f"{path.name}: no row {key}/{row['method']}")
            continue
        for w in WEIGHTS:
            if shipped[f"w={w:g}"] != row["winners"][w]:
                problems.append(f"{path.name} {key}/{row['method']} w={w:g}: view "
                                f"{shipped[f'w={w:g}']}, recomputed {row['winners'][w]}")
    return problems


def render(rows: List[dict]) -> str:
    """One line per grid and schedule, the moving methods named at each w."""
    blocks: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        blocks.setdefault((row["grid"], row["path"]), []).append(row)
    grid_name = {"agg": "Server rules", "reg": "Client penalties"}
    lines = [
        "% generated by tools/sensitivity_counts.py --- do not edit by hand",
        "% Selection groups of the two quarter-horizon screens (validation axis),",
        "% as tools/weight_sensitivity.py forms them; a method is listed under a",
        "% weight when its winning cell at that weight differs from the w=1 winner.",
        # The three weight columns list method names, so they wrap rather than
        # widen the table past the text.
        "\\begin{tabular}{@{}llrr" + ">{\\raggedright\\arraybackslash}p{0.2\\linewidth}" * 3
        + "@{}}",
        "\\toprule",
        "Grid & Schedule & Groups & $>1$ cell & $w=2$ & $w=3$ & $w=5$ \\\\",
        "\\midrule",
    ]
    totals: Dict[str, dict] = {}
    for (grid, path) in sorted(blocks, key=lambda k: (k[0], ["parallel", "cyclic",
                                                            "control"].index(k[1]))):
        here = blocks[(grid, path)]
        cells = []
        for w in WEIGHTS[1:]:
            moved = [method_name(r) for r in here if r["moved"][w]]
            cells.append("%d%s" % (len(moved), (": " + ", ".join(moved)) if moved else ""))
        lines.append(" & ".join([grid_name[grid], path, "%d" % len(here),
                                 "%d" % sum(1 for r in here if r["cells"] > 1)] + cells)
                     + " \\\\")
        t = totals.setdefault(grid, {"groups": 0, "movable": 0,
                                     **{w: 0 for w in WEIGHTS[1:]}})
        t["groups"] += len(here)
        t["movable"] += sum(1 for r in here if r["cells"] > 1)
        for w in WEIGHTS[1:]:
            t[w] += sum(1 for r in here if r["moved"][w])
    lines.append("\\midrule")
    for grid in sorted(totals):
        t = totals[grid]
        lines.append(" & ".join(["\\emph{%s, all}" % grid_name[grid], "",
                                 "%d" % t["groups"], "%d" % t["movable"]]
                                + ["%d" % t[w] for w in WEIGHTS[1:]]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def check_page(rendered: str, text: str) -> List[str]:
    """The page's tab:wsens, if it carries one, row for row against this table."""
    at = text.find("\\label{tab:wsens}")
    if at < 0:
        return []
    def rows(chunk):
        return [" ".join(line.split()) for line in chunk.splitlines()
                if "&" in line and line.strip().endswith("\\\\")]
    page = rows(text[at:text.find("\\end{table}", at)])
    mine = rows(rendered)
    if page != mine:
        return ["tab:wsens: the page's rows are not the recomputed ones"] + [
            f"    page {a}\n    mine {b}" for a, b in zip(page, mine) if a != b]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root with the screen records ($FOA_STUDY_DIR).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Directory w_sensitivity_table.tex is written into "
                         "(default: %(default)s).")
    ap.add_argument("--tex", type=Path, default=None,
                    help="A LaTeX file that may carry tab:wsens (the manuscript); its "
                         "rows are held to the recomputed table.")
    args = ap.parse_args()

    a0, p0 = WS.baselines(args.root)
    rows: List[dict] = []
    problems: List[str] = []
    for grid in ("agg", "reg"):
        found = rows_for(grid, args.root, a0, p0)
        problems += check_view(found, grid, args.root)
        rows += found
    if args.tex is not None:
        problems += check_page(render(rows), args.tex.read_text())
    if problems:
        print("THE RECOMPUTED WINNERS DO NOT MATCH THE SHIPPED VIEWS OR THE PAGE:")
        for p in problems:
            print("  " + p)
        return 1

    for grid, name in (("agg", "aggregation"), ("reg", "regularisation")):
        here = [r for r in rows if r["grid"] == grid]
        single = [f"{r['path']}/{r['method']}" for r in here if r["cells"] == 1]
        print(f"{name}: {len(here)} groups, {len(here) - len(single)} with more than one "
              f"cell; one-cell groups (cannot move): {', '.join(single) or 'none'}")
        for w in WEIGHTS[1:]:
            moved = [f"{r['path']}/{r['method']}" for r in here if r["moved"][w]]
            print(f"  w={w:g}: {len(moved)} of {len(here)} move: {', '.join(moved)}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "w_sensitivity_table.tex").write_text(render(rows))
    print(f"  -> {args.out / 'w_sensitivity_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
