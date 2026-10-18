#!/usr/bin/env python3
"""
The five folds behind every five-fold mean the results tables print.

Every results table of the manuscript prints a five-fold MEAN, and a mean
printed without its spread cannot be read against the next row: a 0.3-point
lead is a finding when the folds agree to a tenth and noise when they disagree
by a point and a half. `compare_arms.py` documents what reading the combination
stage on means alone once did. This tool puts the spread back beside every
mean, from the run records, and adds nothing else.

    python tools/fold_spread.py --root "$FOA_STUDY_DIR"
    python tools/fold_spread.py --root "$FOA_STUDY_DIR" --tex paper/main.tex
    python tools/fold_spread.py --root "$FOA_STUDY_DIR" \\
        --tex "$OUT"/tables/*.tex --out "$OUT/revision"

WHAT IT REPORTS. For every arm of Tables 2 to 7 and B2 to B4 (the two
selection-axis tables and the per-client table): the five per-fold values of
adaptation, preservation and score, with their mean, SAMPLE standard deviation
(n - 1), minimum and maximum. For every fold-paired column - Table 4's delta
against the better component and the text's beats-both count, Table 5's cyclic
minus parallel, Table 6's tuned minus reference, Table 7's margins over the
controls - the five per-fold differences, their mean, SD, minimum, maximum and
folds won. And two numbers the text needs to qualify its verdicts: how many of
Table 4's eighteen deltas would agree in sign on all five folds by chance, and
the per-fold differences behind the 0.57-point replicate spread.

THE SD IS THE SAMPLE SD, AND THE SHIPPED VIEWS' IS NOT. ``report_tables.py``
writes ``adaptation_sd`` with ``statistics.pstdev`` (divide by n), and so does
the extension view's ``score_sd``. Five splits are a sample of re-draws, not
the population of them, so this tool divides by n - 1; its SDs are
sqrt(5/4) = 1.118 times the views' wherever both exist.

THE INVARIANT, AND WHY IT IS CHECKED TWICE. Nothing here may print a spread
around a mean the paper does not print. So every recomputed mean is held

  1. to the shipped view the table was rendered from, at 1e-9 - which says the
     records read here are the runs the view was computed from; and
  2. to the number the page prints, at the printed precision, for every row of
     every table found in the ``--tex`` files (the manuscript itself, or the
     ``tables/*.tex`` that ``make_paper_tables.py`` writes).

Any mismatch is printed and nothing is written. A table the ``--tex`` files do
not carry is still held to its view, and its LaTeX rows are emitted with run
ids in place of the printed labels.

WHICH ARMS, AND IN WHICH ORDER, IS NOT DECIDED HERE. Every row set and row
order is the generator's own: ``make_paper_tables`` (``by_score``,
``reg_printed``, ``selection_block``, ``SETTING``/``ARM_ORDER``, the fairness
blocks), `export_schedule_views` for Table 5 and `export_extension_views` for
Table 6. Every per-fold number comes through the readers the views were built
by - ``export_schedule_views.per_fold`` for the test axis, the fold-keyed form
of ``report_tables.read_runs``, and ``fairness_cost``'s helpers for the
per-client table - so a row here and the row the page prints cannot be two
readings of two different runs.

THE SCORE IS THE PAPER'S, per fold: ``(a_f - A0) - (P0 - p_f)`` with the
stage's one A0 and P0 from ``report_tables.baselines_for``, so the mean of the
five fold scores is exactly the printed score. Its SD therefore carries the
difference in difficulty between the five cohort test splits as well as the
arm's own instability; the JSON also carries the SD against each fold's OWN
shipped-model accuracy, which removes the first, for a reader who wants it.

WHAT IT WRITES, into ``--out`` (default the paper views of the metadata core):

    fold_spread.json         every arm, every fold, every paired column
    fold_spread_tables.tex   each table's body in the page's own row order and
                             column layout, each mean followed by
                             {\\scriptsize +-SD} at the mean's own precision;
                             Table B4 (fairness) gains a "delta worst, same client" column
    revision_numbers.tex     the macros the prose cites from these numbers, in
                             a block of its own beside `stopping_test_axis.py`'s

A macro this tool emits is held, like a table cell, to every definition of the
same name in the ``--tex`` and ``--macros`` files: a regenerated number that no
longer equals the one the manuscript (or the shipped ``revision_numbers.tex``)
sets is a refusal, not a silent update.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tools", REPO / "tools" / "paper_figures"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import report_tables  # noqa: E402
import fairness_cost  # noqa: E402
import export_extension_views  # noqa: E402
import export_schedule_views  # noqa: E402
from export_selection_axis import RECORDS as SELECTION_RECORDS  # noqa: E402
import make_paper_tables as MPT  # noqa: E402

from federated_outlier_adaptation.training import study_lines as SL  # noqa: E402

#: Where the outputs go when ``--out`` is not given: beside the paper views
#: they qualify, in the metadata core every other view ships in.
DEFAULT_OUT = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "paper"

FOLDS = tuple(SL.FOLDS)

#: Runner family -> the schedule word every shipped view uses.
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: Selection-axis stage -> the run-folder stem its arms were run under, as
#: `export_selection_axis` records it.
SELECTION_STEM = {stage: stem for stage, _record, stem in SELECTION_RECORDS}

#: Under the null of no difference, a five-fold paired difference agrees in
#: sign on every fold with probability 2 * 0.5^5.
SIGN_AGREEMENT_NULL = 2 * 0.5 ** len(FOLDS)


# ------------------------------------------------------------------ reading
@functools.lru_cache(maxsize=None)
def test_folds(root: str, prefix: str) -> Dict[Tuple[str, str], Dict[int, Tuple[float, float]]]:
    """``{(cell, schedule): {fold: (adaptation, preservation)}}`` on TEST.

    `export_schedule_views.per_fold` is `report_tables.read_runs` with the fold
    kept: the same files, the same two ``final_evaluation`` fields, the same
    `cell_and_family` key and the same `in_study` filter.
    """
    return export_schedule_views.per_fold(Path(root), prefix)


@functools.lru_cache(maxsize=None)
def val_folds(root: str, prefix: str) -> Dict[Tuple[str, str], Dict[int, Tuple[float, float]]]:
    """``{(cell, schedule): {fold: (adaptation, preservation)}}`` on VALIDATION.

    The fold-keyed form of `export_selection_axis.read_axes`: the last entry of
    ``pool_val_accuracies`` and of ``source_val_accuracies``, which is what the
    selection records were cut on and what Tables B2 and B3 print on their
    validation side.
    """
    import glob

    found: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] = {}
    for name in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        path = Path(name)
        cell, schedule = report_tables.cell_and_family(path, prefix)
        if cell is None or not report_tables.in_study(name):
            continue
        fold = export_schedule_views.FOLD_IN_PATH.search(name)
        if fold is None:
            continue
        body = next(iter(json.loads(path.read_text()).values()))
        adaptation = _last(body.get("pool_val_accuracies"))
        preservation = _last(body.get("source_val_accuracies"))
        if adaptation is None or preservation is None:
            continue
        found.setdefault((cell, schedule), {})[int(fold.group(1))] = (adaptation, preservation)
    return found


def _last(series) -> Optional[float]:
    values = [v for v in (series or []) if v is not None]
    return float(values[-1]) if values else None


@functools.lru_cache(maxsize=None)
def fold_a0(root: str, stem: str) -> Dict[int, float]:
    """The shipped model's own cohort accuracy on each fold's test split."""
    book = report_tables.COHORT_BOOKS[report_tables.cohort_of(stem)]
    payload = json.loads((Path(root) / book).read_text())
    records = payload if isinstance(payload, list) else list(payload.values())
    return {int(r["fold"]): float(r["accuracy"]) for r in records
            if isinstance(r, dict) and "fold" in r and "accuracy" in r}


@functools.lru_cache(maxsize=None)
def fair_folds(root: str, prefix: str) -> Dict[Tuple[str, str], Dict[int, dict]]:
    """Per-fold order statistics of the cohort's clients, as `fairness_cost`.

    The same payloads, the same ``distribution``, ``pick_reference`` and
    ``share_improved`` the fairness views are folded from; only the fold mean
    is not taken. ``same_client`` is the worst-served client's own accuracy
    minus ITS OWN shipped-model accuracy, beside ``lift_worst``, which is the
    worst client now minus the worst client under the shipped model - the two
    are the same number only on folds where the worst client did not change.
    """
    books = fairness_cost.g0_books(Path(root))
    out: Dict[Tuple[str, str], Dict[int, dict]] = {}
    for run in fairness_cost.payloads(Path(root), (prefix,)):
        if run["cell"] is None:
            continue
        final = run["body"].get("final_evaluation") or {}
        clients = (final.get("clients") or {}).get("per_client") or {}
        fold = final.get("fold")
        if not clients or fold is None:
            continue
        reference = fairness_cost.pick_reference(clients, books.get(int(fold), {}))
        spread = fairness_cost.distribution(clients.values())
        worst = min(clients, key=lambda c: (float(clients[c]), c))
        out.setdefault((run["cell"], run["family"]), {})[int(fold)] = {
            "cohort": float(final["clients"]["accuracy"]),
            "min": spread["min"],
            "reference_min": min(reference.values()) if reference else None,
            "improved": fairness_cost.share_improved(clients, reference),
            "worst_client": worst,
            "same_client": (float(clients[worst]) - reference[worst]
                            if worst in reference else None),
            "worst_client_g0": (min(reference, key=lambda c: (reference[c], c))
                                if reference else None),
        }
    return out


# -------------------------------------------------------------------- stats
def summary(values: Sequence[float]) -> dict:
    """Five values and what the tables need of them: mean, sample SD, range."""
    values = [float(v) for v in values]
    return {"values": values, "mean": st.mean(values),
            "sd": st.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def paired(left: Sequence[float], right: Sequence[float]) -> dict:
    """``left - right`` fold by fold, with the folds the left side won."""
    diffs = [a - b for a, b in zip(left, right)]
    out = summary(diffs)
    out["won"] = sum(1 for d in diffs if d > 0)
    out["lost"] = sum(1 for d in diffs if d < 0)
    out["same_sign_all_folds"] = out["won"] == len(diffs) or out["lost"] == len(diffs)
    return out


class Stat:
    """One printed cell: five fold values, how the page rounds their mean."""

    def __init__(self, name: str, values: Sequence[float], nd: int, signed: bool = False,
                 won: Optional[int] = None, view: Optional[float] = None,
                 tol: float = 1e-9, added: bool = False):
        self.name = name
        self.tol = tol
        #: A column this tool adds to a table that did not print it before.
        #: A page without it is held on the columns it has; a page with it is
        #: held on all of them.
        self.added = added
        self.summary = summary(values)
        self.nd = nd
        self.signed = signed
        self.won = won
        self.view = view

    @property
    def mean(self) -> float:
        return self.summary["mean"]

    def text(self) -> str:
        """The mean as the page sets it, followed by the SD as a small suffix."""
        mean = _fmt(self.mean, self.nd)
        sd = "%.*f" % (self.nd, self.summary["sd"])
        if mean.startswith("-"):
            core = "$%s$" % mean
        elif self.signed:
            core = "$+%s$" % mean
        else:
            core = mean
        out = "%s{\\scriptsize±%s}" % (core, sd)
        if self.won is not None:
            out += " (%d/%d)" % (self.won, len(self.summary["values"]))
        return out


class Fixed:
    """A printed cell this tool does not recompute (a rank, an "n/a")."""

    def __init__(self, name: str, text: str):
        self.name = name
        self.fallback = text


def _fmt(value: float, nd: int) -> str:
    out = "%.*f" % (nd, value)
    return "0.%s" % ("0" * nd) if out.startswith("-") and float(out) == 0 else out


class Row:
    """One printed row: the arm(s) behind it, and its cells in page order."""

    def __init__(self, key: str, fallback: Sequence[str], cells: list, meta: dict):
        self.key = key
        self.fallback = list(fallback)
        self.cells = cells
        self.meta = meta
        #: Words the printed label must contain, where rows are not in the
        #: view's own order and a positional match would be a guess.
        self.expect: Optional[str] = None

    def to_json(self) -> dict:
        out = dict(self.meta)
        out["folds"] = list(FOLDS)
        out["columns"] = {}
        for cell in self.cells:
            if isinstance(cell, Stat):
                entry = dict(cell.summary)
                entry["printed_decimals"] = cell.nd
                if cell.won is not None:
                    entry["won"] = cell.won
                out["columns"][cell.name] = entry
        return out


# --------------------------------------------------------------- the arms
class Study:
    """The study root and the per-fold readers, bound once."""

    def __init__(self, root: Path, tag: str):
        self.root = Path(root)
        self.tag = tag

    def prefix(self, stem: str) -> str:
        return f"{self.tag}_{stem}_"

    def base(self, stem: str) -> Tuple[float, float]:
        return report_tables.baselines_for(self.root, stem)

    def folds(self, stem: str, cell: str, schedule: str, axis: str = "test",
              prefix: Optional[str] = None) -> Dict[int, Tuple[float, float]]:
        reader = test_folds if axis == "test" else val_folds
        found = reader(str(self.root), prefix or self.prefix(stem)).get((cell, schedule)) or {}
        missing = [f for f in FOLDS if f not in found]
        if missing:
            raise SystemExit(
                f"FATAL: {cell!r} ({schedule}) has no {axis} run of fold(s) {missing} "
                f"under {self.root}/{prefix or self.prefix(stem)}*. The view this "
                "row was printed from says it ran; the records and the views have "
                "come apart.")
        return found

    def arm(self, stem: str, cell: str, schedule: str, axis: str = "test",
            prefix: Optional[str] = None) -> dict:
        """Per-fold adaptation, preservation, spent and score of one arm."""
        folds = self.folds(stem, cell, schedule, axis, prefix)
        a0, p0 = self.base(stem)
        own = fold_a0(str(self.root), stem)
        a = [folds[f][0] for f in FOLDS]
        p = [folds[f][1] for f in FOLDS]
        return {
            "stem": stem, "cell": cell, "schedule": schedule, "axis": axis,
            "a0": a0, "p0": p0,
            "adaptation": a, "preservation": p,
            "spent": [100.0 * (p0 - x) for x in p],
            "score": [100.0 * ((x - a0) - (p0 - y)) for x, y in zip(a, p)],
            "score_own_fold_a0": [100.0 * ((x - own[f]) - (p0 - y))
                                  for f, x, y in zip(FOLDS, a, p)],
        }


def _meta(arm: dict, **extra) -> dict:
    out = {"stem": arm["stem"], "cell": arm["cell"], "schedule": arm["schedule"],
           "axis": arm["axis"], "A0": arm["a0"], "P0": arm["p0"],
           "score_sd_against_own_fold_a0": st.stdev(arm["score_own_fold_a0"])}
    out.update(extra)
    return out


def _view(row: dict, column: str, scale: float = 1.0) -> float:
    return float(row[column]) * scale


def _score_row(arm: dict, view: Optional[dict], spent: bool = True) -> list:
    """Adaptation, preservation, (spent,) score - the layout of Tables 2 and 3."""
    cells = [Stat("adaptation", arm["adaptation"], 4,
                  view=_view(view, "adaptation") if view else None),
             Stat("preservation", arm["preservation"], 4,
                  view=_view(view, "preservation") if view else None)]
    if spent:
        cells.append(Stat("spent", arm["spent"], 2,
                          view=_view(view, "spent", 100) if view else None))
    cells.append(Stat("score", arm["score"], 2,
                      view=_view(view, "score", 100) if view else None))
    return cells


def t_agg(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table 2: every server rule, by schedule, ordered by score."""
    rows = []
    for fam, title in MPT.SCHEDULE_BLOCKS:
        for r in MPT.by_score([x for x in views["agg"] if x["family"] == fam]):
            arm = study.arm("aggfull", r["cell"], fam)
            rows.append(Row(f"{r['cell']}/{fam}", [MPT.code(r["cell"])],
                            _score_row(arm, r), _meta(arm, block=title)))
    return rows, {}


def t_reg(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table 3: the printed penalties by score, then the borrowed control.

    Each penalty is also paired by fold against its block's no-penalty
    control (the borrowed FedAvg row), which is what a claim that every
    penalty beats the control has to rest on. The proximal row at mu = 0 is
    itself a control and is not counted as a penalty.
    """
    agg_cells = {r["cell"] for r in views["agg"]}
    printed = MPT.reg_printed(views["regu"], views["combo"], agg_cells)
    rows, against = [], []
    for fam, title in MPT.SCHEDULE_BLOCKS:
        here = [x for x in printed if x[0]["family"] == fam]
        control = MPT.pick(views["agg"], cell=MPT.NO_PENALTY_CONTROL, family=fam)
        control_arm = study.arm("aggfull", control["cell"], fam)
        for r, roles in sorted(here, key=lambda x: -float(x[0]["score"])):
            arm = study.arm("regfull", r["cell"], fam)
            rows.append(Row(f"{r['cell']}/{fam}", [MPT.code(r["cell"])],
                            _score_row(arm, r), _meta(arm, block=title, roles=list(roles))))
            if r["cell"] != NO_PENALTY_CELL:
                against.append({"cell": r["cell"], "schedule": fam,
                                "minus_control": paired(arm["score"], control_arm["score"])})
        rows.append(Row(f"{control['cell']}/{fam}", [MPT.code(control["cell"])],
                        _score_row(control_arm, control),
                        _meta(control_arm, block=title, borrowed=True)))
    low = min(against, key=lambda a: a["minus_control"]["mean"])
    fewest = min(against, key=lambda a: a["minus_control"]["won"])
    return rows, {"penalties_minus_control": against,
                  "penalties_minus_control_summary": {
                      "penalties": len(against),
                      "min_mean_margin": low["minus_control"]["mean"],
                      "min_mean_margin_row": f"{low['cell']}/{low['schedule']}",
                      "min_folds_won": fewest["minus_control"]["won"],
                      "min_folds_won_row": f"{fewest['cell']}/{fewest['schedule']}",
                      "rows_winning_every_fold": sum(
                          1 for a in against if a["minus_control"]["won"] == len(FOLDS)),
                      "smallest_single_fold": min(a["minus_control"]["min"] for a in against)}}


def t_combos(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table 4, and the fold-paired reading of its delta column.

    The delta is taken against the component with the larger MEAN score, as
    the page takes it; the per-fold differences are against that same
    component, so the mean of the five is the printed delta.
    """
    agg_cells = {r["cell"] for r in views["agg"]}
    rows, paired_rows = [], []
    for fam, title in MPT.SCHEDULE_BLOCKS:
        for r in MPT.by_score([x for x in views["combo"] if x["family"] == fam]):
            rule, pen = MPT.split_combo(r["cell"], agg_cells)
            a = MPT.pick(views["agg"], cell=rule, family=fam)
            b = MPT.pick(views["regu"], cell=pen, family=fam)
            better = a if float(a["score"]) >= float(b["score"]) else b
            arm = study.arm("combo", r["cell"], fam)
            arm_rule = study.arm("aggfull", rule, fam)
            arm_pen = study.arm("regfull", pen, fam)
            arm_better = arm_rule if better is a else arm_pen
            delta = paired(arm["score"], arm_better["score"])
            vs_rule = paired(arm["score"], arm_rule["score"])
            vs_pen = paired(arm["score"], arm_pen["score"])
            view_delta = 100.0 * (float(r["score"]) - float(better["score"]))
            cells = _score_row(arm, r, spent=False)
            cells.append(Stat("delta_vs_better_component", delta["values"], 2,
                              signed=True, view=view_delta))
            rows.append(Row(f"{r['cell']}/{fam}",
                            [MPT.code(rule), MPT.code(pen)], cells,
                            _meta(arm, block=title, rule=rule, penalty=pen,
                                  better_component=better["cell"])))
            paired_rows.append({
                "cell": r["cell"], "schedule": fam, "rule": rule, "penalty": pen,
                "better_component": better["cell"],
                "vs_better_component": delta, "vs_rule": vs_rule, "vs_penalty": vs_pen,
                "beats_better_on_means": delta["mean"] > 0,
                "beats_both_on_every_fold": vs_rule["won"] == len(FOLDS)
                and vs_pen["won"] == len(FOLDS),
            })
    n = len(paired_rows)
    agree = sum(1 for p in paired_rows if p["vs_better_component"]["same_sign_all_folds"])
    positive = sum(1 for p in paired_rows if p["vs_better_component"]["won"] == len(FOLDS))
    extra = {
        "paired": paired_rows,
        "sign_agreement": {
            "pairs": n,
            "null_probability_all_folds_same_sign": SIGN_AGREEMENT_NULL,
            "expected_count_under_null": SIGN_AGREEMENT_NULL * n,
            "actual_count": agree,
            "rows": [p["cell"] + "/" + p["schedule"] for p in paired_rows
                     if p["vs_better_component"]["same_sign_all_folds"]],
            "null_probability_all_folds_positive": 0.5 ** len(FOLDS),
            "expected_all_positive_under_null": 0.5 ** len(FOLDS) * n,
            "actual_all_positive": positive,
        },
        "beats_better_component_on_means": sum(1 for p in paired_rows
                                                if p["beats_better_on_means"]),
        "beats_both_components_on_every_fold": sum(
            1 for p in paired_rows if p["beats_both_on_every_fold"]),
    }
    return rows, extra


def t_schedule(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table 5: each pair parallel then cyclic, and cyclic minus parallel."""
    scored, stages = export_schedule_views.read_stages(study.root, study.tag)
    rows = []
    for r in views["schedule_pairs"]:
        sides = {}
        for side, cell, sched in (("par", r["parallel_cell"], "parallel"),
                                  ("cyc", r["cyclic_cell"], "cyclic")):
            stem = stages[(cell, sched)]
            sides[side] = study.arm(stem, cell, sched)
        diff = paired(sides["cyc"]["score"], sides["par"]["score"])
        won = int(r["cyclic_wins"].split("/")[0])
        assert won == diff["won"], (r["pair"], won, diff["won"])
        # schedule_pairs.csv stores accuracies to six places and scores to
        # four, so the view check is held to that rounding.
        acc, pts = 5e-5 + 1e-9, 5e-5 + 1e-9
        cells = [
            Stat("parallel_adaptation_pct", [100 * x for x in sides["par"]["adaptation"]], 2,
                 view=100 * float(r["par_adapt"]), tol=acc),
            Stat("parallel_preservation_pct", [100 * x for x in sides["par"]["preservation"]], 2,
                 view=100 * float(r["par_presv"]), tol=acc),
            Stat("cyclic_adaptation_pct", [100 * x for x in sides["cyc"]["adaptation"]], 2,
                 view=100 * float(r["cyc_adapt"]), tol=acc),
            Stat("cyclic_preservation_pct", [100 * x for x in sides["cyc"]["preservation"]], 2,
                 view=100 * float(r["cyc_presv"]), tol=acc),
            Stat("parallel_score", sides["par"]["score"], 2, view=float(r["par_score"]), tol=pts),
            Stat("cyclic_score", sides["cyc"]["score"], 2, view=float(r["cyc_score"]), tol=pts),
            Stat("cyclic_minus_parallel", diff["values"], 2, signed=True, won=diff["won"],
                 view=float(r["diff_mean"]), tol=pts),
        ]
        rows.append(Row(r["pair"], [MPT.tex(r["pair"])], cells,
                        {"pair": r["pair"], "parallel": _meta(sides["par"]),
                         "cyclic": _meta(sides["cyc"]),
                         "cyclic_minus_parallel": diff}))
    return rows, {}


#: The printed order of Table 6 inside a schedule, by the role each row plays,
#: and the words its label is recognised by on the page.
JOINT_ORDER = (("rule alone", "Rule alone"), ("penalty alone", "Penalty alone"),
               ("untuned pair", "untuned"), ("crowned pair", "Crowned pair"),
               ("tuned pair", "Tuned pair"))


def t_jointtune(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table 6: the jointly tuned pair beside its four references.

    The arms and their run-folder prefixes are `export_extension_views`'s
    ``arm_sources`` for the selected pair's grid, which is where the view that
    tabulates them names them.
    """
    grid = export_extension_views.GRIDS["combo_tune_selected"]
    view = {(r["family"], r["role"]): r for r in views["joint"]}
    # The view stores accuracies to six places and scores to four.
    acc, pts = 5e-5 + 1e-9, 5e-5 + 1e-9
    rows, extra = [], {"tuned_minus_reference": []}
    for family in export_extension_views.FAMILIES:
        sched = SCHEDULE[family]
        arms = {}
        for role, cell, prefix in export_extension_views.arm_sources(study.root, family, grid):
            # The jointly tuned runs sit under a stem `report_tables` keeps no
            # baseline for; they federate the ten-writer cohort the cross ran
            # on, so they are read against the cross's A0 - which is also what
            # the extension view scores them against (compare_arms.baselines).
            stem = prefix[len(f"{study.tag}_"):].split("_")[0]
            stem = "combo" if stem.startswith("ctune") else stem
            arms[role] = study.arm(stem, cell, sched, prefix=prefix)
            arms[role]["stem"] = prefix
        tuned = arms["tuned pair"]
        against = views["joint_tuned"][family]
        for role, words in JOINT_ORDER:
            arm, v = arms[role], view[(family, role)]
            cells = [Stat("cohort_pct", [100 * x for x in arm["adaptation"]], 2,
                          view=100 * float(v["adaptation"]), tol=acc),
                     Stat("source_pct", [100 * x for x in arm["preservation"]], 2,
                          view=100 * float(v["preservation"]), tol=acc),
                     Stat("score", arm["score"], 2, view=float(v["score"]), tol=pts)]
            meta = _meta(arm, family=family, role=role)
            if role == "tuned pair":
                cells.append(Fixed("tuned_minus_reference", "n/a"))
            else:
                stem = {"rule alone": "rule", "penalty alone": "penalty",
                        "untuned pair": "untuned", "crowned pair": "crowned"}[role]
                diff = paired(tuned["score"], arm["score"])
                if int(against[f"folds_won_vs_{stem}"]) != diff["won"]:
                    raise SystemExit(f"FATAL: {family} tuned vs {role}: the view says "
                                     f"{against[f'folds_won_vs_{stem}']} folds won, the "
                                     f"records {diff['won']}.")
                cells.append(Stat("tuned_minus_reference", diff["values"], 2, signed=True,
                                  won=diff["won"], view=float(against[f"mean_vs_{stem}"]),
                                  tol=pts))
                meta["tuned_minus_reference"] = diff
                for quantity in ("adaptation", "preservation"):
                    meta[f"tuned_minus_reference_{quantity}_pts"] = paired(
                        [100 * x for x in tuned[quantity]], [100 * x for x in arm[quantity]])
                extra["tuned_minus_reference"].append(
                    {"schedule": sched, "reference": role, "score": diff,
                     "adaptation_pts": meta["tuned_minus_reference_adaptation_pts"],
                     "preservation_pts": meta["tuned_minus_reference_preservation_pts"]})
            row = Row(f"{family}/{role}", [sched, role], cells, meta)
            row.expect = words
            rows.append(row)
    return rows, extra


def t_scaling(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table 7, and each carried arm's fold-paired margin over the controls."""
    rows, margins = [], []
    for key, _stage, title in MPT.SETTING:
        table = views["sizes"][key]
        arms = {}
        for cell in MPT.ARM_ORDER:
            for r in [x for x in table if x["cell"] == cell]:
                arm = study.arm(key, cell, r["family"])
                arms[(cell, r["family"])] = (arm, r)
                rows.append(Row(f"{key}/{cell}/{r['family']}",
                                [MPT.tex(cell), r["family"]],
                                _score_row(arm, r, spent=False),
                                _meta(arm, block=title)))
        controls = [(k, v) for k, v in arms.items() if k[0] == "control"]
        if not controls:
            continue
        best = max(controls, key=lambda kv: float(kv[1][1]["score"]))
        for (cell, fam), (arm, r) in arms.items():
            if cell == "control":
                continue
            entry = {"setting": key, "arm": cell, "schedule": fam,
                     "better_control": best[0][1]}
            for (_c, cfam), (carm, _cr) in controls:
                entry[f"minus_control_{cfam}"] = paired(arm["score"], carm["score"])
            entry["minus_better_control"] = entry[f"minus_control_{best[0][1]}"]
            margins.append(entry)
    low = min(margins, key=lambda m: m["minus_better_control"]["mean"])
    return rows, {"margins_over_control": margins,
                  "min_margin_over_better_control": {
                      "setting": low["setting"], "arm": low["arm"],
                      "mean": low["minus_better_control"]["mean"],
                      "every_fold_positive": all(
                          m["minus_better_control"]["won"] == len(FOLDS) for m in margins),
                      "smallest_single_fold": min(
                          m["minus_better_control"]["min"] for m in margins)}}


def t_selection(study: Study, views: dict, stages) -> Tuple[List[Row], dict]:
    """Tables B2/B3: each printed arm on both axes, validation then test."""
    reported = {"aggregation": views["agg"], "regularisation": views["regu"],
                "combination": views["combo"]}
    rows = []
    for stage, title, _which in stages:
        stage_rows = [r for r in views["sel"] if r["stage"] == stage]
        stem = SELECTION_STEM[stage]
        for fam, fam_title in MPT.SCHEDULE_BLOCKS:
            _block, within, extra = MPT.selection_block(stage_rows, stage, fam)
            for r in within + extra:
                val = study.arm(stem, r["cell"], fam, axis="val")
                test = study.arm(stem, r["cell"], fam)
                rep = MPT.pick(reported[stage], cell=r["cell"], family=fam)
                # selection_axis.csv stores its columns to six places.
                cells = [Stat("val_adaptation", val["adaptation"], 4,
                              view=float(r["val_adaptation"]), tol=5e-7 + 1e-12),
                         Stat("val_preservation", val["preservation"], 4,
                              view=float(r["val_preservation"]), tol=5e-7 + 1e-12),
                         Stat("val_score", val["score"], 2, view=100 * float(r["val_score"]),
                              tol=5e-5 + 1e-9),
                         Fixed("val_rank", r["val_score_rank"]),
                         Stat("test_adaptation", test["adaptation"], 4,
                              view=float(rep["adaptation"])),
                         Stat("test_preservation", test["preservation"], 4,
                              view=float(rep["preservation"])),
                         Stat("test_score", test["score"], 2, view=100 * float(rep["score"])),
                         Fixed("test_rank", r["test_score_rank"])]
                rows.append(Row(f"{stage}/{r['cell']}/{fam}", [MPT.code(r["cell"])], cells,
                                {"stage": stage, "block": f"{title} / {fam_title}",
                                 "validation": _meta(val), "test": _meta(test)}))
    return rows, {}


def t_fairness(study: Study, views: dict) -> Tuple[List[Row], dict]:
    """Table B4: the per-client spread, per fold, for every printed arm."""
    fair = views["fair"]
    best = max(fair["combo"], key=lambda r: float(r["lift_worst"]))
    rows = []
    changed_worst = 0
    total = 0
    for key, _stage, title in MPT.FAIRNESS_BLOCKS:
        if key == "combo":
            here = [MPT.CROWNED, MPT.BALANCED, MPT.CYCLIC_ARM, (best["cell"], best["family"])]
            picked = [MPT.pick(fair[key], cell=c, family=f) for c, f in here]
        else:
            picked = [r for cell in MPT.ARM_ORDER for r in fair[key] if r["cell"] == cell]
        folds_of = fair_folds(str(study.root), study.prefix(key))
        for r in picked:
            folds = folds_of.get((r["cell"], r["family"])) or {}
            missing = [f for f in FOLDS if f not in folds]
            if missing:
                raise SystemExit(f"FATAL: no fairness payload for {r['cell']} "
                                 f"({r['family']}) fold(s) {missing} under {key}.")
            per = [folds[f] for f in FOLDS]
            lift = [x["min"] - x["reference_min"] for x in per]
            same = [x["same_client"] for x in per]
            if any(x is None for x in same):
                raise SystemExit(f"FATAL: {key}/{r['cell']}: a worst-served client has no "
                                 "shipped-model reference in any g-0 book.")
            moved = sum(1 for x in per if x["worst_client"] != x["worst_client_g0"])
            changed_worst += moved
            total += len(per)
            cells = [Stat("cohort", [x["cohort"] for x in per], 4, view=float(r["cohort"])),
                     Stat("worst_client", [x["min"] for x in per], 4, view=float(r["min"])),
                     Stat("delta_worst_pts", [100 * x for x in lift], 2, signed=True,
                          view=100 * float(r["lift_worst"])),
                     # The worst-served client against ITS OWN shipped-model
                     # accuracy: the reading the fairness note once claimed for the
                     # column beside it, which compares two different writers
                     # on most folds.
                     Stat("delta_worst_same_client_pts", [100 * x for x in same], 2,
                          signed=True, added=True),
                     Stat("lifted", [x["improved"] for x in per], 2,
                          view=float(r["improved"]))]
            rows.append(Row(f"{key}/{r['cell']}/{r['family']}",
                            [MPT.tex(r["cell"]), r["family"]], cells,
                            {"setting": key, "cell": r["cell"], "schedule": r["family"],
                             "block": title,
                             "delta_worst_same_client_pts": summary([100 * x for x in same]),
                             "folds_where_worst_client_changed": moved}))
    return rows, {"delta_worst_definition": {
        "printed": "per fold, worst client now minus worst client under g0 "
                   "(fairness_cost.fairness_rows lift_worst); the two may be "
                   "different writers",
        "arm_folds": total,
        "arm_folds_where_the_worst_writer_changed": changed_worst}}


#: The proximal family's mu = 0 cell: no penalty at all, so it is a control and
#: not one of the penalties a "every penalty beats the control" claim is about.
NO_PENALTY_CELL = "param_l2_mu0"


#: Every table, in manuscript order: label, name, builder, label columns.
TABLES = (
    ("tab:agg_winners", "Table 2", t_agg, 1),
    ("tab:reg_winners", "Table 3", t_reg, 1),
    ("tab:combos", "Table 4", t_combos, 2),
    ("tab:schedule", "Table 5", t_schedule, 1),
    ("tab:jointtune", "Table 6", t_jointtune, 2),
    ("tab:scaling", "Table 7", t_scaling, 2),
    ("tab:selection_axis", "Table B2",
     lambda s, v: t_selection(s, v, MPT.SELECTION_STAGES[:2]), 1),
    ("tab:selection_axis_cross", "Table B3",
     lambda s, v: t_selection(s, v, MPT.SELECTION_STAGES[2:]), 1),
    ("tab:fairness", "Table B4", t_fairness, 2),
)


def load_views() -> dict:
    """The shipped views the generator renders from, by the generator's reader."""
    joint = MPT.rows("extension_combo_tune_selected.csv")
    tuned = {r["family"]: r for r in joint if r["role"] == "tuned pair"}
    return {
        "agg": MPT.rows("agg-winners.csv"),
        "regu": MPT.rows("reg-winners.csv"),
        "combo": MPT.rows("combos.csv"),
        "sel": MPT.rows("selection_axis.csv"),
        "schedule_pairs": MPT.rows("schedule_pairs.csv"),
        "joint": joint,
        "joint_tuned": tuned,
        "sizes": {k: MPT.rows("sizes_%s.csv" % k) for k, _, _ in MPT.SETTING},
        "fair": {k: MPT.rows("fairness_%s.csv" % k) for k, _, _ in MPT.FAIRNESS_BLOCKS},
    }


# ---------------------------------------------------------- the page side
NUMBER = re.compile(r"[-+]?\d+\.\d+")
WON = re.compile(r"\((\d+)/(\d+)\)")


def table_body(text: str, label: str) -> Optional[List[str]]:
    """The tabular body lines of one table, between its header rule and the end."""
    at = text.find("\\label{%s}" % label)
    if at < 0:
        return None
    start = text.find("\\midrule", at)
    end = text.find("\\bottomrule", start)
    if start < 0 or end < 0:
        return None
    lines = text[start:end].splitlines()
    return [line.rstrip() for line in lines[1:] if line.strip()]


def is_data(line: str) -> bool:
    s = line.strip()
    return ("&" in s and s.endswith("\\\\") and not s.startswith("\\rowcolor")
            and not s.startswith("\\multicolumn"))


def split_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip()[:-2].split("&")]


def check_row(row: Row, printed: Sequence[str], where: str) -> List[str]:
    """Every recomputed mean against its printed cell, at the printed precision."""
    problems = []
    stats = [c for c in row.cells]
    if len(printed) != len(stats):
        stats = [c for c in row.cells if not getattr(c, "added", False)]
    if len(printed) != len(stats):
        return [f"{where}: {len(printed)} number cells printed, {len(stats)} expected"]
    for cell, text in zip(stats, printed):
        if isinstance(cell, Fixed):
            cell.fallback = text
            continue
        found = NUMBER.search(text.replace("$", ""))
        if not found:
            problems.append(f"{where} [{cell.name}]: no number in {text!r}")
            continue
        page = found.group(0)
        nd = len(page.split(".")[1])
        mine = _fmt(cell.mean, nd)
        if _fmt(float(page), nd) != mine:
            problems.append(f"{where} [{cell.name}]: page prints {page}, records give "
                            f"{cell.mean:.6f} -> {mine}")
        if nd != cell.nd:
            problems.append(f"{where} [{cell.name}]: page prints {nd} decimals, "
                            f"this tool expected {cell.nd}")
        won = WON.search(text)
        if won and cell.won is not None and int(won.group(1)) != cell.won:
            problems.append(f"{where} [{cell.name}]: page says {won.group(0)} folds won, "
                            f"records give {cell.won}/{len(FOLDS)}")
    return problems


def check_views(rows: Sequence[Row], name: str) -> List[str]:
    """Every recomputed mean against the shipped view's own value.

    At 1e-9 where the view stores full precision; where it stores a rounded
    figure (the schedule and extension views write six or four places) the
    tolerance is that rounding, set on the cell that reads it.
    """
    problems = []
    for row in rows:
        for cell in row.cells:
            if isinstance(cell, Stat) and cell.view is not None:
                if abs(cell.mean - cell.view) > cell.tol:
                    problems.append(f"{name} {row.key} [{cell.name}]: records "
                                    f"{cell.mean:.10f}, view {cell.view:.10f}")
    return problems


def render(rows: Sequence[Row], body: Optional[List[str]], label_cols: int) -> List[str]:
    """The table body with every mean followed by its SD, rows in page order."""
    if body is None:
        out = ["% no manuscript source was given for this table: the label "
               "columns are run ids"]
        for row in rows:
            cells = row.fallback + [c.text() if isinstance(c, Stat) else c.fallback
                                    for c in row.cells]
            out.append(" & ".join(cells) + " \\\\")
        return out
    out, data = [], iter(rows)
    # A column this tool adds widens the table, so a full-width block header of
    # the page's own body has to span it too.
    first = next((line for line in body if is_data(line)), None)
    printed = len(split_cells(first)) - label_cols if first else 0
    wider = len(rows[0].cells) - printed if rows else 0
    for line in body:
        if not is_data(line):
            if wider > 0:
                line = re.sub(r"\\multicolumn\{(\d+)\}",
                              lambda m: "\\multicolumn{%d}" % (int(m.group(1)) + wider), line)
            out.append(line)
            continue
        row = next(data)
        labels = [c for c in line.strip()[:-2].split("&")][:label_cols]
        cells = [c.text() if isinstance(c, Stat) else c.fallback for c in row.cells]
        out.append("&".join(labels).rstrip() + " & " + " & ".join(cells) + " \\\\")
    return out


def sd_range(rows: Sequence[Row]) -> dict:
    """Per column, the smallest and largest fold SD over a table's rows."""
    out: Dict[str, dict] = {}
    for row in rows:
        for cell in row.cells:
            if isinstance(cell, Stat):
                sd = cell.summary["sd"]
                entry = out.setdefault(cell.name, {"min": sd, "max": sd,
                                                   "min_row": row.key, "max_row": row.key})
                if sd < entry["min"]:
                    entry.update(min=sd, min_row=row.key)
                if sd > entry["max"]:
                    entry.update(max=sd, max_row=row.key)
    return out


def control_replicates(study: Study) -> dict:
    """The fold-paired replicate behind nControlSeedSpread (0.57) and its sibling.

    make_numbers.py sets nControlSeedSpread to the agg-winners score of
    ``control_fedavg`` minus that of ``control_fedavg_earlystop`` on the cyclic
    schedule, and nControlSeedSpreadPar to the same on the parallel one: one
    task file run twice, differing in the client-sampling seed block and in a
    stop flag that never fired. Here the same two arms, fold by fold.
    """
    out = {}
    for fam in ("cyclic", "parallel"):
        plain = study.arm("aggfull", "control_fedavg", fam)
        armed = study.arm("aggfull", "control_fedavg_earlystop", fam)
        out[fam] = {
            "macro": "nControlSeedSpread" if fam == "cyclic" else "nControlSeedSpreadPar",
            "plain_minus_armed": paired(plain["score"], armed["score"]),
            "plain_score": summary(plain["score"]),
            "armed_score": summary(armed["score"]),
            "plain_minus_armed_adaptation_pts": paired(
                [100 * x for x in plain["adaptation"]], [100 * x for x in armed["adaptation"]]),
            "plain_minus_armed_preservation_pts": paired(
                [100 * x for x in plain["preservation"]],
                [100 * x for x in armed["preservation"]]),
        }
    return out


# ------------------------------------------------------------ the macros
NUMBERS_FILE = "revision_numbers.tex"

#: The tools that own a block of the numbers file, in the order it lists them.
NUMBER_OWNERS = ("fold_spread.py", "stopping_test_axis.py")

DEFINITION = re.compile(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}")


def defined(texts: Sequence[str]) -> Dict[str, List[str]]:
    """Every ``\\newcommand{\\name}{value}`` in these texts, by name."""
    out: Dict[str, List[str]] = {}
    for text in texts:
        for name, value in DEFINITION.findall(text):
            out.setdefault(name, []).append(value)
    return out


def check_macros(macros: Sequence[Tuple[str, str, str]], texts: Sequence[str]) -> List[str]:
    """Each emitted macro against every definition of its name already written."""
    known = defined(texts)
    return [f"\\{name}: recomputed {value}, defined as {other}"
            for name, value, _source in macros
            for other in known.get(name, []) if other != value]


def write_numbers(path: Path, owner: str, macros: Sequence[Tuple[str, str, str]]) -> None:
    """Rewrite this tool's block of the numbers file and keep the other's.

    Two tools set the revision's macros, and each owns one delimited block, so
    either can be re-run alone without the other's numbers going missing.
    """
    blocks: Dict[str, str] = {}
    if path.is_file():
        found = re.findall(r"% begin (\S+)\n(.*?)% end \1\n", path.read_text(), re.S)
        blocks = {name: body for name, body in found}
    blocks[owner] = "".join("\\newcommand{\\%s}{%s}  %% %s\n" % m for m in macros)
    lines = ["% generated by tools/fold_spread.py and tools/stopping_test_axis.py",
             "% --- do not edit by hand; each tool rewrites its own block.", ""]
    for name in NUMBER_OWNERS:
        if name in blocks:
            lines += [f"% begin {name}", blocks[name].rstrip("\n"), f"% end {name}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _p2(value: float) -> str:
    return "%.2f" % value


def macros_of(payload: dict) -> List[Tuple[str, str, str]]:
    """The fold-spread numbers the prose cites, each with where it is read."""
    tables = payload["tables"]
    rep = payload["control_replicates"]
    out = [
        ("nReplicateFoldSDCyc", _p2(rep["cyclic"]["plain_minus_armed"]["sd"]),
         "SD over folds of control_fedavg minus control_fedavg_earlystop, cyclic"),
        ("nReplicateFoldSDPar", _p2(rep["parallel"]["plain_minus_armed"]["sd"]),
         "the same, parallel"),
        ("nReplicateFoldMaxPar",
         _p2(max(abs(v) for v in rep["parallel"]["plain_minus_armed"]["values"])),
         "largest single-fold replicate difference, parallel, magnitude"),
    ]
    for macro, label, column in (("nScoreSDAgg", "tab:agg_winners", "score"),
                                 ("nScoreSDReg", "tab:reg_winners", "score"),
                                 ("nPairedSDCombo", "tab:combos", "delta_vs_better_component")):
        spread = tables[label]["sd_range"][column]
        out.append((macro + "Min", _p2(spread["min"]), f"{label} {column}: smallest fold SD"))
        out.append((macro + "Max", _p2(spread["max"]), f"{label} {column}: largest fold SD"))
    out.append(("nCarryMinFoldMargin",
                _p2(tables["tab:scaling"]["min_margin_over_better_control"]["smallest_single_fold"]),
                "tab:scaling: smallest single-fold margin of a carried arm over the better control"))
    # FOUR RUNS OF ONE UPDATE RULE. The cyclic block of Table 2 holds plain
    # FedAvg twice (one with a stop flag that never fired), cyclic FedAvg and
    # cyclic FedAvg in a shuffled order: alpha_i = n_k / N_t in every one, and
    # they differ only in the client-draw seed or the order. Their spread is a
    # second, larger replicate of one configuration than the control pair alone.
    same_rule = ("control_fedavg", "control_fedavg_earlystop", "seq_fedavg",
                 "seq_order_shuffle")
    runs = {r["cell"]: r["columns"]["score"] for r in tables["tab:agg_winners"]["rows"]
            if r["schedule"] == "cyclic" and r["cell"] in same_rule}
    if sorted(runs) != sorted(same_rule):
        raise SystemExit(f"FATAL: the cyclic block of Table 2 lacks {sorted(set(same_rule) - set(runs))}")
    means = [runs[c]["mean"] for c in same_rule]
    per_fold = [st.stdev([runs[c]["values"][i] for c in same_rule]) for i in range(len(FOLDS))]
    out.append(("nCyclicReplicateRange", _p2(max(means) - min(means)),
                "tab:agg_winners cyclic: max minus min score of the four same-rule rows"))
    out.append(("nCyclicReplicateSD", _p2(st.stdev(means)),
                "the same four rows: sample SD of their five-fold mean scores"))
    out.append(("nCyclicReplicateFoldSD", _p2(st.mean(per_fold)),
                "the same four rows: per fold the SD of their four scores, mean over folds"))
    versus = tables["tab:reg_winners"]["penalties_minus_control_summary"]
    out.append(("nPenaltyVsControlMinMargin", _p2(versus["min_mean_margin"]),
                "tab:reg_winners: smallest mean fold-paired margin of a penalty over its "
                "block's no-penalty control"))
    out.append(("nPenaltyVsControlMinWon", "%d" % versus["min_folds_won"],
                "tab:reg_winners: fewest folds (of five) a penalty beats that control on"))
    fair = tables["tab:fairness"]["delta_worst_definition"]
    out.append(("nFairWorstChangedArmFolds",
                "%d" % fair["arm_folds_where_the_worst_writer_changed"],
                "tab:fairness: arm-folds whose worst-served writer is not g-0's worst"))
    out.append(("nFairArmFolds", "%d" % fair["arm_folds"],
                "tab:fairness: arm-folds the table is a mean over"))
    out.append(("nComboChanceExpected",
                "%.1f" % tables["tab:combos"]["sign_agreement"]["expected_count_under_null"],
                "tab:combos: rows agreeing in sign on all five folds expected by chance"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root with the run records ($FOA_STUDY_DIR, "
                         "assembled as docs/REPRODUCE.md section 10 says).")
    ap.add_argument("--tex", type=Path, nargs="*", default=[],
                    help="LaTeX files the printed tables are parsed from and held "
                         "to: the manuscript, or the tables/*.tex make_paper_tables "
                         "writes. The first file carrying a table's label is used.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Directory fold_spread.json and fold_spread_tables.tex are "
                         "written into (default: %(default)s).")
    ap.add_argument("--numbers", type=Path, default=None,
                    help="The macros file this tool's block is written into "
                         "(default: revision_numbers.tex in --out).")
    ap.add_argument("--macros", type=Path, nargs="*", default=[],
                    help="LaTeX files whose \\newcommand definitions the emitted "
                         "macros must equal, beside the --tex files (e.g. the "
                         "shipped revision_numbers.tex).")
    ap.add_argument("--study-tag", default="d01")
    args = ap.parse_args()

    study = Study(args.root, args.study_tag)
    views = load_views()
    texts = []
    for path in args.tex:
        if not path.is_file():
            raise SystemExit(f"--tex {path}: no such file")
        texts.append((path.name, path.read_text()))

    problems: List[str] = []
    payload = {"generated_by": "tools/fold_spread.py",
               "root": "$FOA_STUDY_DIR",
               "basis": "TEST (final_evaluation) except the val_* columns of "
                        "Tables B2/B3, which are VALIDATION (last round of "
                        "pool_val_accuracies / source_val_accuracies)",
               "sd": "sample standard deviation over the five folds (n - 1)",
               "score": "points, (a_f - A0) - (P0 - p_f) with the stage's A0/P0",
               "checked_against": {}, "tables": {}}
    rendered: List[str] = [
        "% generated by tools/fold_spread.py --- do not edit by hand",
        "% each mean is the printed five-fold mean; {\\scriptsize±x} is the",
        "% sample SD over the five folds (n-1), at the mean's own precision.",
    ]
    for label, name, build, label_cols in TABLES:
        rows, extra = build(study, views)
        problems += check_views(rows, name)
        body, source = None, None
        for fname, text in texts:
            body = table_body(text, label)
            if body is not None:
                source = fname
                break
        if body is not None:
            data = [line for line in body if is_data(line)]
            if len(data) != len(rows):
                problems.append(f"{name} ({label}) in {source}: {len(data)} data rows "
                                f"printed, {len(rows)} arms recomputed")
            else:
                for i, (line, row) in enumerate(zip(data, rows), start=1):
                    printed = split_cells(line)[label_cols:]
                    problems += check_row(row, printed, f"{name} row {i} ({row.key})")
                    words = " ".join(split_cells(line)[:label_cols])
                    if row.expect and row.expect.lower() not in words.lower():
                        problems.append(f"{name} row {i} ({row.key}): the printed label "
                                        f"{words!r} does not name {row.expect!r}")
            payload["checked_against"][label] = source
        else:
            payload["checked_against"][label] = "views only (no --tex carries it)"
        payload["tables"][label] = {"name": name, "rows": [r.to_json() for r in rows],
                                    "sd_range": sd_range(rows), **extra}
        # Named by label only: a table number written into a generated file
        # goes stale the first time a float is added before it.
        rendered += ["", f"% {label} --- body rows, page order"
                     + (f", labels from {source}" if source else "")]
        rendered += render(rows, body, label_cols)

    payload["control_replicates"] = control_replicates(study)
    macros = macros_of(payload)
    problems += check_macros(macros, [t for _n, t in texts]
                             + [m.read_text() for m in args.macros if m.is_file()])
    if problems:
        print("THE RECOMPUTED MEANS DO NOT MATCH WHAT IS PRINTED:")
        for p in problems:
            print("  " + p)
        print(f"{len(problems)} mismatch(es); nothing written.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fold_spread.json").write_text(json.dumps(payload, indent=1) + "\n")
    (args.out / "fold_spread_tables.tex").write_text("\n".join(rendered) + "\n")
    numbers = args.numbers or args.out / NUMBERS_FILE
    write_numbers(numbers, "fold_spread.py", macros)
    for label, name, _b, _l in TABLES:
        t = payload["tables"][label]
        n = len(t["rows"])
        sc = t["sd_range"].get("score") or t["sd_range"].get("test_score") or {}
        span = (f"score SD {sc['min']:.2f}-{sc['max']:.2f} pts" if sc else "")
        print(f"{name:<9}{label:<27}{n:>3} rows  checked against "
              f"{payload['checked_against'][label]:<20} {span}")
    sign = payload["tables"]["tab:combos"]["sign_agreement"]
    print(f"Table 4: 5/5 sign agreement expected {sign['expected_count_under_null']:.3f} "
          f"of {sign['pairs']} under the null, observed {sign['actual_count']}")
    for fam, rep in payload["control_replicates"].items():
        d = rep["plain_minus_armed"]
        print(f"{rep['macro']} ({fam}): mean {d['mean']:.4f}, folds "
              + " ".join(f"{x:+.2f}" for x in d["values"]) + f", SD {d['sd']:.2f}")
    print(f"  -> {args.out / 'fold_spread.json'}")
    print(f"  -> {args.out / 'fold_spread_tables.tex'}")
    print(f"  -> {numbers} (" + ", ".join(f"{n}={v}" for n, v, _ in macros) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
