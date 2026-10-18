#!/usr/bin/env python3
"""
The quarter-horizon screens' own orderings, printed.

    python tools/screen_ranking.py --root "$FOA_STUDY_DIR"     # views + tables
    python tools/screen_ranking.py --render --out <dir>        # tables only

Both grids were screened at a quarter of the horizon to RANK settings inside a
method, and every method then re-ran its screen winner for the full hundred
rounds. The paper reports only those full-horizon numbers, so two of its claims
rested on orderings no table printed: that the no-penalty control places
mid-table in the regularisation screen, and that logit matching's strength is
one a quarter-horizon screen cannot price. This tool prints those orderings.

WHAT IT WRITES. Three CSV views, read from the records, and five tables
rendered from the views alone:

    screen_ranking_reg.csv     every regularisation cell, both schedules
    screen_ranking_logit.csv   logit matching's row: screen beside full horizon
    screen_ranking_agg.csv     every aggregation cell, both schedules

    screen_ranking_reg_{parallel,cyclic}.tex
    screen_ranking_logit.tex
    screen_ranking_agg_{parallel,cyclic}.tex

THE NUMBERS ARE THE SELECTORS' OWN. Nothing here scores a run. The cells are
read by ``select_reg_screen.collect``/``summarise`` and
``select_agg_screen.collect``/``summarise``, ordered by
``study_emit.ranked_by_trade`` and scored by ``study_emit.trade_score`` against
``study_emit.shipped_baselines`` - the exact calls ``study_emit reg-full`` and
``agg-full`` chose the full-horizon re-runs with. The winner of each method is
then recomputed the way those two stages compute it and checked against the
record they wrote (``tables/p13_reg_method_winners.json`` and
``tables/p11_agg_method_winners.json``). A disagreement there means the
ordering printed is not the ordering that selected, and the tool refuses to
write rather than print it.

VALIDATION, NOT TEST. A screen's ordering is the one the selection saw, so the
screen columns are validation on both axes. ``report_tables.py --what
reg-screen`` prints the same runs on the TEST axis, ordered by the test score;
that is a reporting view and a different ordering, and it is not what chose
anything.

THE FULL-HORIZON HALF OF THE LOGIT VIEW IS WHATEVER RAN, AND NO MORE. Only the
screen winners were re-run at the full horizon, one per schedule. Each such task
computes both schedules, and the folder is named for the one it was selected
to serve; ``report_tables.py`` reads that half only. The other half is a real
hundred-round run of a setting the screen chose for the OTHER schedule, so it
is carried here as well, marked ``other``, because it is the only full-horizon
evidence on a second strength. It was never selected and nothing reads it as a
result.

RENDERING NEEDS NO RECORDS. ``--render`` reads the three views - from
``FOA_PAPER_DATA`` when it is set, as every paper script does, otherwise from
the study's ``tables/paper`` - and writes the tables. That is the mode
``tools/regenerate_paper.sh`` runs, so the paper still regenerates with no
dataset, no GPU and no records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "tools", REPO / "tools" / "paper_figures"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from paper_names import label  # noqa: E402

#: The schedules in the order the paper's tables print them, with the family
#: name the run records use for each.
SCHEDULES: Tuple[Tuple[str, str], ...] = (("parallel", "concurrent"),
                                          ("cyclic", "sequential"))

#: Where the views live, relative to the repository: beside every other view
#: the paper renders from, so ``FOA_PAPER_DATA`` reaches them the same way.
VIEWS = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "paper"

#: Where the tables are written by default. A separate place from the views on
#: purpose: the views are data a reader can check, the tables are typesetting.
OUT = REPO / "study" / "artifacts"

#: Rows per side-by-side panel. A 140-cell screen does not fit one column of a
#: page, and splitting it into two tables would split the one ordering the
#: table exists to show; two panels of one table keep it a single ranking.
PANEL_ROWS = 70

REG_FIELDS = ["schedule", "rank", "of", "cell", "method", "method_rank",
              "method_of", "adaptation", "adaptation_sd", "preservation",
              "preservation_sd", "score", "folds", "rounds", "selected",
              "control", "printed"]
LOGIT_FIELDS = ["schedule", "cell", "lam", "screen_rounds", "screen_rank",
                "screen_of", "method_rank", "method_of", "screen_adaptation",
                "screen_preservation", "screen_score", "selected",
                "full_source", "full_rounds", "full_val_adaptation",
                "full_val_preservation", "full_val_score",
                "full_test_adaptation", "full_test_preservation",
                "full_test_score"]


# --------------------------------------------------------------------------- #
# the records half: reads runs, imports the study's selectors
# --------------------------------------------------------------------------- #
def _selectors():
    """
    The study's own readers and scorers, imported only when records are read.

    Imported late because they pull in the training package, and ``--render``
    must run on the bare interpreter ``regenerate_paper.sh`` promises.
    """
    import select_agg_screen
    import select_reg_screen
    import study_emit
    from federated_outlier_adaptation.training import agg_cells, reg_cells
    from federated_outlier_adaptation.training import study_lines

    return study_emit, select_reg_screen, select_agg_screen, reg_cells, \
        agg_cells, study_lines


def _is_reg_control(row: Dict[str, Any]) -> bool:
    """
    Whether a regularisation row is the no-penalty control.

    The proximal term at mu = 0 is an exact no-op (the trainer returns zero for
    any lam <= 0), which is why it is the control - and the same test
    ``select_reg_screen.emit_full`` uses to carry it into the finals.
    """
    return row["method"] == "param_l2" and row["hypers"].get("lam") == 0.0


def _read_printed(root: Path, name: str) -> set:
    """``(cell, schedule)`` of every row a paper winners view prints."""
    path = root / "tables" / "paper" / name
    if not path.is_file():
        return set()
    with open(path, newline="") as handle:
        return {(r["cell"], r["family"]) for r in csv.DictReader(handle)}


def _recorded(root: Path, name: str) -> Dict[str, str]:
    """``{"<method>/<family>": winner}`` from a stage's selection record."""
    path = root / "tables" / name
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} is missing; the winners recomputed here "
                         "are checked against it and cannot be checked otherwise.")
    return {key: entry["winner"] for key, entry in json.loads(path.read_text()).items()}


def _ranked_view(rows, method_of, selected, control, printed, schedule,
                 rounds, E, a0, p0) -> List[Dict[str, Any]]:
    """One schedule's rows, best first by the selection score, as view rows."""
    ranked = E.ranked_by_trade(rows, "adaptation", a0, p0)
    per_method: Dict[str, List[str]] = {}
    for row in ranked:
        per_method.setdefault(method_of(row), []).append(row["id"])
    out = []
    for rank, row in enumerate(ranked, 1):
        method = method_of(row)
        out.append({
            "schedule": schedule, "rank": rank, "of": len(ranked),
            "cell": row["id"], "method": method,
            "method_rank": per_method[method].index(row["id"]) + 1,
            "method_of": len(per_method[method]),
            "adaptation": row["adaptation"]["mean"],
            "adaptation_sd": row["adaptation"]["sd"],
            "preservation": row["preservation"]["mean"],
            "preservation_sd": row["preservation"]["sd"],
            "score": E.trade_score(row, a0, p0),
            "folds": len(row["folds"]), "rounds": rounds,
            "selected": int(row["id"] in selected),
            "control": int(control(row)),
            "printed": int((row["id"], schedule) in printed),
        })
    return out


def reg_view(root: Path, tag: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Every regularisation screen cell on both schedules, and the check lines.

    The winner of each (method, family) is recomputed exactly as
    ``study_emit.reg_full`` computes it - ``max`` of ``trade_score`` over that
    method's rows, in the order ``summarise`` returns them, so a tie resolves
    the same way - and must equal the cell its record names.
    """
    E, R, _, reg_cells, _, SL = _selectors()
    cells = E.with_extensions(root, reg_cells.screen_cells(), "reg")
    rows = R.summarise(R.collect(root, cells, E.prefixes(_cfg(tag))["reg_screen"]))
    a0, p0 = E.shipped_baselines(root)
    recorded = _recorded(root, "p13_reg_method_winners.json")
    printed = _read_printed(root, "reg-winners.csv")

    view, notes = [], []
    for schedule, family in SCHEDULES:
        scored = [r for r in rows if r["family"] == family
                  and r["adaptation"]["mean"] is not None]
        selected = set()
        for method in SL.REG_METHODS:
            siblings = [r for r in scored if r["method"] == method]
            if not siblings:
                continue
            top = max(siblings, key=lambda r: E.trade_score(r, a0, p0))
            want = recorded.get(f"{method}/{family}")
            if top["id"] != want:
                raise SystemExit(
                    f"FATAL: {method}/{family} recomputes to {top['id']} but "
                    f"p13_reg_method_winners.json records {want}; this is not "
                    "the ordering that selected, and it is not printed.")
            selected.add(top["id"])
            notes.append(f"  reg {schedule:<8} {method:<14} screen winner "
                         f"{top['id']:<22} "
                         + ("printed in reg-winners" if (top["id"], schedule) in printed
                            else "NOT A ROW OF reg-winners"))
        view += _ranked_view(scored, lambda r: r["method"], selected,
                             _is_reg_control, printed, schedule,
                             reg_cells.SCREEN_ROUNDS, E, a0, p0)
    return view, notes


def logit_view(root: Path, tag: str,
               screen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Logit matching's whole swept row: the screen beside the full horizon.

    The full-horizon columns come from the re-run folders, read by the same
    ``collect``/``summarise`` the reg-top3 stage reads them with; the test axis
    is the per-fold ``adaptation_test``/``preservation_test`` those readers
    carry, scored by the same ``trade_score``. A strength with no re-run has
    empty full-horizon columns - it was never measured at a hundred rounds.
    """
    E, R, _, reg_cells, _, _ = _selectors()
    cells = [c for c in reg_cells.screen_cells() if c["method"] == "logit_l2"]
    a0, p0 = E.shipped_baselines(root)
    prefix = E.prefixes(_cfg(tag))["reg_full"]

    finals: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, chosen_for in SCHEDULES:
        found = R.collect(root, cells, f"{prefix}{chosen_for}_")
        for row in R.summarise(found):
            source = "selected" if row["family"] == chosen_for else "other"
            key = (row["id"], row["family"])
            if key in finals and finals[key]["full_source"] == "selected":
                continue
            per_fold = [families[row["family"]]
                        for families in found[row["id"]]["folds"].values()
                        if row["family"] in families]
            test = {"adaptation": R._agg([f["adaptation_test"] for f in per_fold]),
                    "preservation": R._agg([f["preservation_test"] for f in per_fold])}
            finals[key] = {
                "full_source": source, "full_rounds": reg_cells.FULL_ROUNDS,
                "full_val_adaptation": row["adaptation"]["mean"],
                "full_val_preservation": row["preservation"]["mean"],
                "full_val_score": E.trade_score(row, a0, p0),
                "full_test_adaptation": test["adaptation"]["mean"],
                "full_test_preservation": test["preservation"]["mean"],
                "full_test_score": E.trade_score(test, a0, p0),
            }

    by_key = {(r["cell"], r["schedule"]): r for r in screen}
    lams = {c["id"]: c["hypers"]["lam"] for c in cells}
    out = []
    for schedule, family in SCHEDULES:
        for cell in cells:
            s = by_key.get((cell["id"], schedule))
            if s is None:   # never screened on this schedule: nothing to set
                continue
            full = finals.get((cell["id"], family), {})
            out.append({
                "schedule": schedule, "cell": cell["id"],
                "lam": lams[cell["id"]], "screen_rounds": s["rounds"],
                "screen_rank": s["rank"], "screen_of": s["of"],
                "method_rank": s["method_rank"], "method_of": s["method_of"],
                "screen_adaptation": s["adaptation"],
                "screen_preservation": s["preservation"],
                "screen_score": s["score"], "selected": s["selected"],
                **{k: full.get(k, "") for k in LOGIT_FIELDS
                   if k.startswith("full_")},
            })
    return out


def agg_view(root: Path, tag: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Every aggregation screen cell on both schedules, and the check lines.

    A server rule belongs to one schedule, so each schedule ranks its own
    path's cells. The two controls ran both schedules in one folder, and
    ``select_agg_screen.read_run`` reads the first payload only - the parallel
    one - so the controls' cyclic half is read with
    ``select_reg_screen.read_family_runs``, which tells the halves apart by the
    scenario each payload names. The parallel half read that way must equal the
    selector's own row, which is what says the right half was taken.
    """
    E, R, A, _, agg_cells, SL = _selectors()
    cells = E.with_extensions(root, agg_cells.screen_cells(), "agg")
    prefix = E.prefixes(_cfg(tag))["agg_screen"]
    rows = A.summarise(A.collect(root, cells, prefix))
    a0, p0 = E.shipped_baselines(root)
    recorded = _recorded(root, "p11_agg_method_winners.json")
    printed = _read_printed(root, "agg-winners.csv")
    by_id = {cell["id"]: cell for cell in cells}

    halves = _control_halves(root, [c for c in cells if c["path"] == "control"],
                             prefix, R, A)
    for row in rows:
        if row["path"] == "control":
            mine = halves[(row["id"], "concurrent")]
            if abs(mine["adaptation"]["mean"] - row["adaptation"]["mean"]) > 1e-12 \
                    or abs(mine["preservation"]["mean"]
                           - row["preservation"]["mean"]) > 1e-12:
                raise SystemExit(f"FATAL: {row['id']}'s parallel half does not match "
                                 "select_agg_screen's own row.")

    def method_of(row):
        return SL.agg_method_of(by_id[row["id"]])

    view, notes = [], []
    for schedule, family in SCHEDULES:
        own = [r for r in rows if r["path"] == family
               and r["adaptation"]["mean"] is not None]
        selected = set()
        for path, method in SL.AGG_METHODS:
            if path != family:
                continue
            siblings = [r for r in own if method_of(r) == method]
            if not siblings:
                continue
            top = max(siblings, key=lambda r: E.trade_score(r, a0, p0))
            want = recorded.get(f"{path}/{method}")
            if top["id"] != want:
                raise SystemExit(
                    f"FATAL: {path}/{method} recomputes to {top['id']} but "
                    f"p11_agg_method_winners.json records {want}; this is not "
                    "the ordering that selected, and it is not printed.")
            selected.add(top["id"])
            notes.append(f"  agg {schedule:<8} {method:<14} screen winner "
                         f"{top['id']:<22} "
                         + ("printed in agg-winners" if (top["id"], schedule) in printed
                            else "NOT A ROW OF agg-winners"))
        controls = [halves[(c["id"], family)] for c in cells
                    if c["path"] == "control" and (c["id"], family) in halves]
        view += _ranked_view(own + controls, method_of, selected,
                             lambda r: r["path"] == "control", printed, schedule,
                             agg_cells.SCREEN_ROUNDS, E, a0, p0)
    return view, notes


def _control_halves(root: Path, controls, prefix: str, R, A) -> Dict[tuple, dict]:
    """Each control's per-schedule row, shaped like ``select_agg_screen``'s."""
    out = {}
    for cell in controls:
        per_family: Dict[str, Dict[int, dict]] = {}
        for run_dir in sorted(root.glob(f"{prefix}{cell['id']}_fold*")):
            parsed = A.parse_run_dir(run_dir.name, prefix)
            if not run_dir.is_dir() or parsed is None or parsed[0] != cell["id"]:
                continue
            for family, run in R.read_family_runs(run_dir).items():
                per_family.setdefault(family, {})[parsed[1]] = run
        for family, folds in per_family.items():
            out[(cell["id"], family)] = {
                "id": cell["id"], "path": "control", "folds": sorted(folds),
                "adaptation": A._agg([r["adaptation"] for r in folds.values()]),
                "preservation": A._agg([r["preservation"] for r in folds.values()]),
            }
    return out


def _cfg(tag: str):
    """The minimum ``study_emit.prefixes`` reads: the study's folder tag."""
    return argparse.Namespace(tag=tag)


def write_view(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  view -> {path}")


# --------------------------------------------------------------------------- #
# the rendering half: standard library and paper_names only
# --------------------------------------------------------------------------- #
def read_view(name: str, views: Optional[Path]) -> List[Dict[str, str]]:
    """
    One view, from ``--views``, else ``FOA_PAPER_DATA``, else the shipped core.

    The same order of precedence the paper scripts use, so a reader who points
    ``FOA_PAPER_DATA`` at rebuilt views gets these tables from them too.
    """
    if views is not None:
        dirs = [views]
    elif os.environ.get("FOA_PAPER_DATA"):
        dirs = [Path(p) for p in os.environ["FOA_PAPER_DATA"].split(os.pathsep) if p]
    else:
        dirs = [VIEWS]
    for folder in dirs:
        if (folder / name).is_file():
            with open(folder / name, newline="") as handle:
                return list(csv.DictReader(handle))
    raise SystemExit(f"{name}: not in {os.pathsep.join(str(d) for d in dirs)}; "
                     "rebuild it with --root, see docs/REPRODUCE.md section 10")


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _f4(value: str) -> str:
    return "" if value == "" else f"{float(value):.4f}"


def _pts(value: str) -> str:
    return "" if value == "" else f"{float(value) * 100:.2f}"


def _cell_text(row: Dict[str, str]) -> str:
    """The published name, the control in bold, the screen winner starred."""
    text = label(row["cell"])
    if row["control"] == "1":
        text = r"\textbf{" + text + "}"
    if row["selected"] == "1":
        text += r"$^{\star}$"
    return text


def render_ranking(rows: List[Dict[str, str]], schedule: str, grid: str) -> str:
    """One schedule's whole screen ordering as one table, in side-by-side panels."""
    mine = sorted((r for r in rows if r["schedule"] == schedule),
                  key=lambda r: int(r["rank"]))
    if not mine:
        raise SystemExit(f"no {grid} rows for the {schedule} schedule")
    n = len(mine)
    panels = max(1, math.ceil(n / PANEL_ROWS))
    height = math.ceil(n / panels)
    columns = [mine[i * height:(i + 1) * height] for i in range(panels)]
    rounds = mine[0]["rounds"]
    controls = [r for r in mine if r["control"] == "1"]
    winners = sum(1 for r in mine if r["selected"] == "1")

    if grid == "reg":
        what = "regularisation grid"
        ref = r"Table~\ref{tab:reg_winners}"
        control_note = (
            f"The no-penalty control (FedProx at $\\mu{{=}}0$, in bold) ranks "
            f"{ordinal(int(controls[0]['rank']))} of {n}." if controls else "")
        tab = f"tab:screen_reg_{schedule}"
    else:
        what = "aggregation grid"
        ref = r"Table~\ref{tab:agg_winners}"
        control_note = ("The controls, in bold, rank " + " and ".join(
            f"{ordinal(int(r['rank']))} ({label(r['cell'])})" for r in controls)
            + f" of {n}." if controls else "")
        tab = f"tab:screen_agg_{schedule}"

    spec = r"@{\hspace{1.2em}}".join(["rlrrr"] * panels)
    head = " & ".join([r"\# & Cell & Adapt. & Pres. & Score"] * panels)
    lines = [
        f"% GENERATED by tools/screen_ranking.py from screen_ranking_{grid}.csv;"
        " do not edit.",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{The quarter-horizon screen of the " + what + f", {schedule} "
        f"schedule: all {n} cells, ranked by the selection score of "
        r"Eq.~\eqref{eq:score} at round " + rounds + r". Five-split means, "
        r"\textbf{validation} axis (adaptation and preservation; score in "
        r"points). $\star$ marks the cell the screen selected within its "
        f"method ({winners} on this schedule), the setting {ref} re-runs at the "
        f"full horizon. {control_note}" + "}",
        r"\label{" + tab + "}",
        # Two panels need the smaller face to fit a page; one does not.
        r"\tiny" if panels > 1 else r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\providecommand{\fittable}[1]{\resizebox{\linewidth}{!}{#1}}",
        r"\fittable{%",
        r"\begin{tabular}{@{}" + spec + "@{}}",
        r"\toprule",
        head + r" \\",
        r"\midrule",
    ]
    for i in range(height):
        cells = []
        for column in columns:
            if i < len(column):
                r = column[i]
                cells += [r["rank"], _cell_text(r), _f4(r["adaptation"]),
                          _f4(r["preservation"]), _pts(r["score"])]
            else:
                cells += [""] * 5
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def render_logit(rows: List[Dict[str, str]]) -> str:
    """Logit matching's swept strengths: screen score beside full-horizon score."""
    by = {(r["cell"], r["schedule"]): r for r in rows}
    cells = []
    for r in rows:
        if r["cell"] not in cells:
            cells.append(r["cell"])
    cells.sort(key=lambda c: float(by[(c, SCHEDULES[0][0])]["lam"]))
    first = by[(cells[0], SCHEDULES[0][0])]
    measured = sorted({r["cell"] for r in rows if r["full_source"] == "selected"},
                      key=lambda c: float(by[(c, SCHEDULES[0][0])]["lam"]))
    if not measured:
        raise SystemExit("screen_ranking_logit.csv: no strength has a "
                         "full-horizon run, so there is nothing to set beside "
                         "the screen")
    full_rounds = by[(measured[0], SCHEDULES[0][0])]["full_rounds"] or \
        by[(measured[0], SCHEDULES[1][0])]["full_rounds"]
    names = " and ".join("$" + label(c).split("$")[1] + "$" for c in measured)

    def block(r):
        mark = r"$^{\star}$" if r["selected"] == "1" else ""
        other = r"$^{\dagger}$" if r["full_source"] == "other" else ""
        full_val = _pts(r["full_val_score"])
        full_test = _pts(r["full_test_score"])
        return [_pts(r["screen_score"]) + mark,
                f"{r['method_rank']}/{r['method_of']}",
                (full_val + other) if full_val else "--",
                (full_test + other) if full_test else "--"]

    lines = [
        "% GENERATED by tools/screen_ranking.py from screen_ranking_logit.csv;"
        " do not edit.",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Logit matching, every swept strength: the quarter-horizon "
        r"screen's score (round " + first["screen_rounds"]
        + r", \textbf{validation}) and its rank among the "
        f"{first['method_of']} strengths, beside the full-horizon score (round "
        f"{full_rounds}) on validation and on \\textbf{{test}}, in "
        r"points, five-split means, scored by Eq.~\eqref{eq:score}. $\star$ marks "
        r"the strength the screen selected on that schedule. Only the selected "
        f"strengths, {names}, were re-run at the full horizon; a dash is a "
        r"strength never measured there. Each re-run computed both schedules: "
        r"$\dagger$ is the half of a run selected for the \emph{other} "
        r"schedule, a measurement no selection read.}",
        r"\label{tab:screen_logit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{\hspace{1.5em}}rrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{Parallel schedule}} & "
        r"\multicolumn{4}{c}{\textbf{Cyclic schedule}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
        r" & Screen & & Full & Full & Screen & & Full & Full \\",
        r"$\lambda$ & val & Rank & val & test & val & Rank & val & test \\",
        r"\midrule",
    ]
    for cell in cells:
        lam = label(cell).split("$")[1].replace(r"\lambda{=}", "")
        row = ["$" + lam + "$"]
        for schedule, _ in SCHEDULES:
            row += block(by[(cell, schedule)])
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def render(views: Optional[Path], out: Path, grids: Tuple[str, ...]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    emitted = []
    if "reg" in grids:
        reg = read_view("screen_ranking_reg.csv", views)
        for schedule, _ in SCHEDULES:
            emitted.append((f"screen_ranking_reg_{schedule}.tex",
                            render_ranking(reg, schedule, "reg")))
        emitted.append(("screen_ranking_logit.tex",
                        render_logit(read_view("screen_ranking_logit.csv", views))))
    if "agg" in grids:
        agg = read_view("screen_ranking_agg.csv", views)
        for schedule, _ in SCHEDULES:
            emitted.append((f"screen_ranking_agg_{schedule}.tex",
                            render_ranking(agg, schedule, "agg")))
    for name, text in emitted:
        (out / name).write_text(text)
        print(f"  table -> {out / name}")


def summary(reg: List[Dict[str, Any]], agg: List[Dict[str, Any]]) -> None:
    """The lines a reader checks the paper's sentences against."""
    for schedule, _ in SCHEDULES:
        for row in reg:
            if row["schedule"] == schedule and row["control"]:
                print(f"  reg {schedule:<8} no-penalty control ranks "
                      f"{row['rank']} of {row['of']}")
        for row in agg:
            if row["schedule"] == schedule and row["control"]:
                print(f"  agg {schedule:<8} {row['cell']} ranks "
                      f"{row['rank']} of {row['of']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path,
                      help="The assembled study root ($FOA_STUDY_DIR): rebuild "
                           "the views from the records, then render.")
    mode.add_argument("--render", action="store_true",
                      help="Render the tables from the views only; no records.")
    ap.add_argument("--views", type=Path, default=None,
                    help="Directory of the CSV views (default: "
                         f"{VIEWS.relative_to(REPO)}; with "
                         "--render, FOA_PAPER_DATA is honoured when this is unset).")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="Directory the .tex tables are written to "
                         f"(default: {OUT.relative_to(REPO)}).")
    ap.add_argument("--grid", choices=("both", "reg", "agg"), default="both",
                    help="Which screen to print (default: both).")
    ap.add_argument("--study-tag", default="d01",
                    help="Folder tag of the study's runs (default: d01).")
    args = ap.parse_args()

    grids = ("reg", "agg") if args.grid == "both" else (args.grid,)
    if args.root is not None:
        views = args.views or VIEWS
        reg_rows: List[Dict[str, Any]] = []
        agg_rows: List[Dict[str, Any]] = []
        notes: List[str] = []
        print("basis: VALIDATION on both axes at the screen horizon, the axis "
              "that selected; every winner checked against its stage record.")
        if "reg" in grids:
            reg_rows, reg_notes = reg_view(args.root, args.study_tag)
            notes += reg_notes
            write_view(views / "screen_ranking_reg.csv", reg_rows, REG_FIELDS)
            write_view(views / "screen_ranking_logit.csv",
                       logit_view(args.root, args.study_tag, reg_rows), LOGIT_FIELDS)
        if "agg" in grids:
            agg_rows, agg_notes = agg_view(args.root, args.study_tag)
            notes += agg_notes
            write_view(views / "screen_ranking_agg.csv", agg_rows, REG_FIELDS)
        print("\n".join(notes))
        summary(reg_rows, agg_rows)
        render(views, args.out, grids)
    else:
        render(args.views, args.out, grids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
