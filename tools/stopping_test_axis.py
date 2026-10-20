#!/usr/bin/env python3
"""
The plateau rule read on the test axis, and on a trace a server could compute.

`plateau_rule.py` prices the stopping rule on the VALIDATION trace - the kept
round is chosen on ``pool_val_accuracies`` and the round it keeps is scored on
``pool_val_accuracies`` against ``source_val_accuracies`` - and that is the
basis Table 8 prints. Two things that table cannot say are asked here.

    python tools/stopping_test_axis.py --root "$FOA_STUDY_DIR"
    python tools/stopping_test_axis.py --root "$FOA_STUDY_DIR" --tex paper/main.tex

FIRST, WHAT THE KEPT ROUND IS WORTH ON TEST. Choosing a round and scoring it on
the same rows is a selection scored on itself: the gain can be partly the
noise the choice was made on. So the round the rule keeps is scored again on
the held-out halves - ``pool_test_accuracies`` for the cohort and the per-round
source TEST accuracy - which played no part in choosing it.

THE PER-ROUND SOURCE TEST SERIES EXISTS, and it is not the one the summaries
name. Each runner appends ``(pool in-sample, source test)`` to its
``accuracies`` list every round (`runners/concurrent_runner.run_round`,
``SOURCE_TEST_SET``), and ``accuracies_<i>.json`` stores it as
``accuracies[r][1]``; ``analysis/forgetting_signals`` reads the same element as
``source_test``. It is the run's OWN fold's source test split - its size is
``evaluation.sets.source_test`` - while ``final_evaluation.old.mean`` and P0
average all five folds' splits. P0 is the mean of the five per-fold shipped-model
source accuracies, so the fold mean of round 0 of this series is P0 exactly, as
the fold mean of ``pool_test_accuracies[0]`` is A0: the test score of the
shipped model is zero on this axis as on the others, and that is checked.

SECOND, WHAT A SERVER COULD ACTUALLY HAVE RUN. ``pool_val_accuracies`` pools the
validation rows of EVERY cohort client every round, including the ones the
sampler did not draw - a server only hears from the clients it contacts. So the
rule is run a second time on a DRAWN-ONLY trace: each round, the validation
accuracy pooled over that round's ``participants`` only, from
``client_val_accuracies`` weighted by each client's validation rows in the
run's own fold book - the weighting ``pool_val_accuracies`` itself uses
(`runners/population.evaluate_pool`). Pooled over all clients, that
reconstruction reproduces the stored trace exactly; this is checked on every
run and round, so the drawn-only trace differs from the stored one by the
undrawn clients and by nothing else.

WHAT IS HELD FIXED, BY IMPORT. The 83 arms, their stages, the fold mean, the
per-stage A0 and P0, ``k = 20``, ``eps = 0`` and the checkpoint-kept variant are
`plateau_rule`'s and `stopping_table`'s own, read through ``verdicts``,
``fire_round`` and ``scored_round``. Like the published rule, the primary
reading fires on the FOLD-MEAN trace, so every fold of an arm keeps the same
round. Because one deployment is one run, the JSON also carries the per-run
reading - the rule fired on each fold's own trace and scored on that fold - for
both traces.

WHAT IT WRITES, into ``--out`` (default the stopping views of the metadata core):

    stopping_test_axis.json         every arm: fire and kept rounds on both
                                    traces, scores at each on both axes, and
                                    the per-run reading
    stopping_test_axis_table.tex    Table 8's rows, one column per basis, and
                                    the arms each basis hurts on test
    revision_numbers.tex            this tool's block of the prose macros
                                    (beside fold_spread.py's block), each held
                                    to any definition of it in --tex/--macros

The validation gain is recomputed and held to `plateau_stages.csv` (and, with
``--tex``, to the gain column the page prints); a mismatch stops the tool before
anything is written.
"""

from __future__ import annotations

import argparse
import glob
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
from extreme_stopping import fold_mean  # noqa: E402
from plateau_rule import (  # noqa: E402
    ALL,
    CHECKPOINT_BEST,
    PRIMARY_MARGIN,
    PRIMARY_PATIENCE,
    fire_round,
    outcomes,
    scored_round,
    verdicts,
)
from stopping_table import STAGES  # noqa: E402
from make_paper_tables import STAGE_LABEL  # noqa: E402
import fold_spread  # noqa: E402

from federated_outlier_adaptation.data.fold_book import FoldBook  # noqa: E402
from federated_outlier_adaptation.training import study_lines as SL  # noqa: E402

DEFAULT_OUT = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "stopping"

FOLDS = tuple(SL.FOLDS)
FOLD_IN_PATH = re.compile(r"_fold(\d+)")
PLACEHOLDER = "$FOA_STUDY_DIR"

#: How close the all-client reconstruction must come to the stored trace. The
#: two are the same sums in a different order, so only float rounding is left.
RECONSTRUCTION = 1e-12


# ------------------------------------------------------------------ reading
class Books:
    """The fold books runs name, loaded once; validation rows per client."""

    def __init__(self, root: Path):
        self.root = root
        self.loaded: Dict[str, FoldBook] = {}
        self.counts: Dict[Tuple[str, int], Dict[str, int]] = {}

    def val_rows(self, book: str, fold: int, clients: Sequence[str]) -> Dict[str, int]:
        key = (book, fold)
        if key not in self.counts:
            if book not in self.loaded:
                self.loaded[book] = FoldBook.load(book.replace(PLACEHOLDER, str(self.root)))
            self.counts[key] = {}
        known = self.counts[key]
        for client in clients:
            if client not in known:
                known[client] = len(self.loaded[book].part(fold, client, "val"))
        return {c: known[c] for c in clients}


def pooled(accuracies: Dict[str, float], rows: Dict[str, int], clients) -> float:
    """Accuracy over the union of these clients' validation rows."""
    total = sum(rows[c] for c in clients)
    return sum(accuracies[c] * rows[c] for c in clients) / total


def read_traces(root: Path, prefix: str, books: Books) -> Dict[Tuple[str, str], Dict[int, dict]]:
    """Every per-round series this tool needs, per arm and fold.

    Keyed as `stopping_table.read_arms` keys (``cell_and_family``, ``in_study``),
    so an arm here is an arm there.
    """
    out: Dict[Tuple[str, str], Dict[int, dict]] = {}
    for name in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        path = Path(name)
        cell, family = report_tables.cell_and_family(path, prefix)
        if cell is None or not report_tables.in_study(name):
            continue
        fold = int(FOLD_IN_PATH.search(name).group(1))
        document = json.loads(path.read_text())
        exp = next(iter(document))
        body = document[exp]
        payload_path = path.parent / exp / "accuracies_0.json"
        if not payload_path.is_file():
            raise SystemExit(f"FATAL: {payload_path} is missing; the per-round source "
                             "test series lives there and nowhere else.")
        payload = json.loads(payload_path.read_text())
        pool_val = list(body["pool_val_accuracies"])
        rounds = len(pool_val)
        source_test = [float(pair[1]) for pair in payload["accuracies"]]
        pool_test = list(body["pool_test_accuracies"])
        source_val = list(body["source_val_accuracies"])
        client_val = body["client_val_accuracies"]
        participants = body.get("participants") or [None] * rounds
        if not (len(source_test) == len(pool_test) == len(source_val) == rounds
                == len(client_val)):
            raise SystemExit(f"FATAL: {name}: per-round series of unequal length")
        if any(v is None for series in (pool_val, pool_test, source_val, source_test)
               for v in series):
            raise SystemExit(f"FATAL: {name}: a per-round series has a gap")

        final = body["final_evaluation"]
        rows = books.val_rows(final["fold_book"], fold, sorted(client_val[0]))
        drawn, drawn_plain, sizes, worst = [], [], [], 0.0
        for r in range(rounds):
            everyone = sorted(client_val[r])
            worst = max(worst, abs(pooled(client_val[r], rows, everyone) - pool_val[r]))
            chosen = list(dict.fromkeys(participants[r] or everyone))
            drawn.append(pooled(client_val[r], rows, chosen))
            drawn_plain.append(st.mean(client_val[r][c] for c in chosen))
            sizes.append(len(chosen))
        if worst > RECONSTRUCTION:
            raise SystemExit(
                f"FATAL: {name}: pooling client_val_accuracies over every client by "
                f"the fold book's validation rows misses pool_val_accuracies by "
                f"{worst:.3g}. The drawn-only trace is built the same way and would "
                "differ from the stored one by more than the undrawn clients.")
        out.setdefault((cell, family), {})[fold] = {
            "pool_val": pool_val, "pool_test": pool_test,
            "source_val": source_val, "source_test": source_test,
            "drawn_val": drawn, "drawn_val_unweighted": drawn_plain,
            "drawn_clients": sizes, "cohort_clients": len(client_val[0]),
            "source_test_rows": (body.get("evaluation") or {}).get("sets", {}).get("source_test"),
            "final_evaluation": {"cohort": final["clients"]["accuracy"],
                                 "source": final["old"]["mean"]},
            "reconstruction_error": worst,
        }
    return out


# ----------------------------------------------------------------- the rule
def keep(trace: Sequence[float]) -> Tuple[Optional[int], int]:
    """``(fire index or None, kept index)`` under the published setting."""
    fired = fire_round(trace, PRIMARY_PATIENCE, PRIMARY_MARGIN)
    return fired, scored_round(trace, fired, CHECKPOINT_BEST)


def test_scores(fold: dict, a0: float, p0: float) -> List[float]:
    """The study's score per round, on the held-out halves of both populations."""
    return [(a - a0) - (p0 - s) for a, s in zip(fold["pool_test"], fold["source_test"])]


def val_scores(fold: dict, a0: float, p0: float) -> List[float]:
    return [(a - a0) - (p0 - s) for a, s in zip(fold["pool_val"], fold["source_val"])]


def arm_reading(stage: str, verdict, traces: Dict[int, dict], a0: float, p0: float) -> dict:
    """One arm under the rule, on both traces, scored on both axes."""
    folds = [traces[f] for f in FOLDS]
    n = len(verdict.values)
    stored = fold_mean([f["pool_val"] for f in folds])[:n]
    if stored != list(verdict.arm.adaptation[:n]):
        raise SystemExit(f"FATAL: {stage}/{verdict.arm.cell}/{verdict.arm.family}: the "
                         "fold-mean validation trace read here is not the one "
                         "plateau_rule reads; the two tools are reading different runs.")
    drawn = fold_mean([f["drawn_val"] for f in folds])[:n]
    test = fold_mean([test_scores(f, a0, p0) for f in folds])[:n]
    val = list(verdict.values)

    fired_s, kept_s = keep(stored)
    fired_d, kept_d = keep(drawn)

    def at(series, index):
        return series[index]

    reading = {
        "stage": stage, "cell": verdict.arm.cell, "family": verdict.arm.family,
        "rounds": n,
        "stored": {"fired": fired_s is not None,
                   "fire_round": None if fired_s is None else fired_s + 1,
                   "kept_round": kept_s + 1},
        "drawn_only": {"fired": fired_d is not None,
                       "fire_round": None if fired_d is None else fired_d + 1,
                       "kept_round": kept_d + 1},
        "kept_round_difference": kept_d - kept_s,
        "validation": {"final": val[-1], "at_stored_kept": at(val, kept_s),
                       "at_drawn_kept": at(val, kept_d),
                       "gain_stored": at(val, kept_s) - val[-1],
                       "gain_drawn": at(val, kept_d) - val[-1]},
        "test": {"round_0": test[0], "final": test[-1],
                 "at_stored_kept": at(test, kept_s), "at_drawn_kept": at(test, kept_d),
                 "gain_stored": at(test, kept_s) - test[-1],
                 "gain_drawn": at(test, kept_d) - test[-1],
                 "oracle": max(test), "oracle_round": test.index(max(test)) + 1,
                 "cohort_at_stored_kept": st.mean(f["pool_test"][kept_s] for f in folds),
                 "source_at_stored_kept": st.mean(f["source_test"][kept_s] for f in folds),
                 "cohort_final": st.mean(f["pool_test"][n - 1] for f in folds),
                 "source_final": st.mean(f["source_test"][n - 1] for f in folds),
                 "final_evaluation_score": st.mean(
                     (f["final_evaluation"]["cohort"] - a0)
                     - (p0 - f["final_evaluation"]["source"]) for f in folds)},
        "drawn_clients_per_round": sorted({c for f in folds for c in f["drawn_clients"]}),
        "cohort_clients": folds[0]["cohort_clients"],
    }

    # One deployment is one run: the rule fired on each fold's own trace.
    per_run = {}
    for basis, key in (("stored", "pool_val"), ("drawn_only", "drawn_val")):
        gains_test, gains_val, kept = [], [], []
        for f in folds:
            trace = f[key][:n]
            _fired, index = keep(trace)
            t = test_scores(f, a0, p0)[:n]
            v = val_scores(f, a0, p0)[:n]
            gains_test.append(t[index] - t[-1])
            gains_val.append(v[index] - v[-1])
            kept.append(index + 1)
        per_run[basis] = {"kept_rounds": kept, "test_gain": st.mean(gains_test),
                          "validation_gain": st.mean(gains_val),
                          "test_gain_per_fold": gains_test}
    reading["per_run"] = per_run
    return reading


# ------------------------------------------------------------------- tables
def stage_table(readings: Sequence[dict]) -> List[dict]:
    """Per stage, then over every arm: the means the table prints."""
    order = [stem.strip("_") for stem, _ in STAGES] + [ALL]
    out = []
    for stage in order:
        rows = list(readings) if stage == ALL else [r for r in readings if r["stage"] == stage]
        if not rows:
            continue
        worst = min(rows, key=lambda r: (r["test"]["gain_stored"], r["cell"]))
        worst_d = min(rows, key=lambda r: (r["test"]["gain_drawn"], r["cell"]))
        out.append({
            "stage": stage, "arms": len(rows),
            "validation_fixed": st.mean(r["validation"]["final"] for r in rows),
            "validation_gain": st.mean(r["validation"]["gain_stored"] for r in rows),
            "validation_gain_drawn": st.mean(r["validation"]["gain_drawn"] for r in rows),
            "test_fixed": st.mean(r["test"]["final"] for r in rows),
            "test_gain_stored": st.mean(r["test"]["gain_stored"] for r in rows),
            "test_gain_drawn": st.mean(r["test"]["gain_drawn"] for r in rows),
            "test_oracle_gain": st.mean(r["test"]["oracle"] - r["test"]["final"] for r in rows),
            "mean_abs_kept_difference": st.mean(abs(r["kept_round_difference"]) for r in rows),
            "arms_kept_round_differs": sum(1 for r in rows if r["kept_round_difference"]),
            "fires_stored": sum(1 for r in rows if r["stored"]["fired"]),
            "fires_drawn": sum(1 for r in rows if r["drawn_only"]["fired"]),
            "worst_test_gain_stored": worst["test"]["gain_stored"],
            "worst_test_arm_stored": f"{worst['cell']}/{worst['family']}",
            "worst_test_gain_drawn": worst_d["test"]["gain_drawn"],
            "worst_test_arm_drawn": f"{worst_d['cell']}/{worst_d['family']}",
            "arms_hurt_test_stored": sum(1 for r in rows if r["test"]["gain_stored"] < 0),
            "arms_hurt_test_drawn": sum(1 for r in rows if r["test"]["gain_drawn"] < 0),
            "per_run_test_gain_stored": st.mean(r["per_run"]["stored"]["test_gain"] for r in rows),
            "per_run_test_gain_drawn": st.mean(r["per_run"]["drawn_only"]["test_gain"]
                                               for r in rows),
            "per_run_validation_gain_stored": st.mean(
                r["per_run"]["stored"]["validation_gain"] for r in rows),
        })
    return out


def _pts(value: float, signed: bool = True) -> str:
    out = "%.2f" % (100 * value)
    if out in ("0.00", "-0.00"):
        return "0.00"
    if out.startswith("-"):
        return "$%s$" % out
    return "$+%s$" % out if signed else out


def render(table: Sequence[dict]) -> str:
    """Table 8's rows, with the test-axis gains and the arms each basis hurts."""
    lines = [
        "% generated by tools/stopping_test_axis.py --- do not edit by hand",
        "%% Patience rule (k=%d, eps=%g, best checkpoint kept), fired on the fold-mean"
        % (PRIMARY_PATIENCE, PRIMARY_MARGIN),
        "% trace exactly as tab:plateau. Gains are mean score in points, the rule's",
        "% kept round less the run-out round 100. Validation: pool_val against",
        "% source_val (reproduces tab:plateau). Test: pool_test against the per-round",
        "% source test series accuracies[r][1]. Drawn-only: the rule fired on the",
        "% validation trace pooled over each round's drawn clients only. Arms hurt:",
        "% arms whose test score at the kept round is below their score at round 100.",
        "% Ceiling: the best test-axis score over the rounds of the fold-mean trace,",
        "% less round 100 - tab:plateau's best-round ceiling, read on test.",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Stage & Arms & Validation & \\multicolumn{3}{c}{Test axis (pts)} & Arms hurt \\\\",
        "\\cmidrule(lr){4-6}",
        " & & gain (pts) & gain, stored & gain, drawn-only & ceiling & (stored/drawn) \\\\",
        "\\midrule",
    ]
    lines += rows_of(table)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def rows_of(table: Sequence[dict]) -> List[str]:
    """The body lines of the test-axis table, rules included."""
    label = dict(STAGE_LABEL)
    lines = []
    for row in table:
        if row["stage"] == "extreme" or row["stage"] == ALL:
            lines.append("\\midrule")
        name = "\\emph{All arms}" if row["stage"] == ALL else label[row["stage"]]
        lines.append(" & ".join([
            name, "%d" % row["arms"],
            _pts(row["validation_gain"]) if row["fires_stored"] else "0.00",
            _pts(row["test_gain_stored"]), _pts(row["test_gain_drawn"]),
            _pts(row["test_oracle_gain"]),
            "%d/%d" % (row["arms_hurt_test_stored"], row["arms_hurt_test_drawn"])])
            + " \\\\")
    return lines


# ------------------------------------------------------------------ checks
def check_view(table: Sequence[dict], root: Path, shipped: Path) -> List[str]:
    """The validation gain against `plateau_stages.csv`, row for row."""
    import csv

    own = root / "tables" / "stopping" / "plateau_stages.csv"
    path = own if own.is_file() else shipped
    rows = {r["stage"]: r for r in csv.DictReader(open(path))
            if int(r["patience"]) == PRIMARY_PATIENCE
            and float(r["margin"]) == PRIMARY_MARGIN and r["variant"] == CHECKPOINT_BEST}
    problems = []
    for row in table:
        view = rows.get(row["stage"])
        if view is None:
            problems.append(f"{row['stage']}: no row in {path.name}")
            continue
        for mine, theirs in (("arms", "arms"), ("validation_gain", "gain"),
                             ("validation_fixed", "fixed_mean"), ("fires_stored", "fires")):
            if abs(float(row[mine]) - float(view[theirs])) > 1e-12:
                problems.append(f"{row['stage']} {mine}: recomputed {row[mine]}, "
                                f"{path.name} {view[theirs]}")
    return problems


def check_page(table: Sequence[dict], text: str) -> List[str]:
    """The validation gain and arm count against the page's tab:plateau."""
    at = text.find("\\label{tab:plateau}")
    if at < 0:
        return ["--tex: no tab:plateau"]
    body = text[text.find("\\midrule", at):text.find("\\bottomrule", at)]
    data = [line.strip() for line in body.splitlines()
            if "&" in line and line.strip().endswith("\\\\")]
    if len(data) != len(table):
        return [f"tab:plateau prints {len(data)} rows, {len(table)} recomputed"]
    problems = []
    for line, row in zip(data, table):
        cells = [c.strip() for c in line[:-2].split("&")]
        arms = int(cells[1])
        gain = float(cells[4].replace("$", ""))
        fixed = float(cells[2])
        mine_gain = float("%.2f" % (100 * row["validation_gain"]))
        mine_fixed = float("%.2f" % (100 * row["validation_fixed"]))
        if arms != row["arms"] or gain != mine_gain or fixed != mine_fixed:
            problems.append(f"tab:plateau {cells[0]}: page {arms} arms, fixed {fixed}, "
                            f"gain {gain}; recomputed {row['arms']}, {mine_fixed}, "
                            f"{mine_gain}")
    return problems


def check_page_test(table: Sequence[dict], text: str) -> List[str]:
    """The page's test-axis table, if it carries one, row for row."""
    at = text.find("\\label{tab:plateau_test}")
    if at < 0:
        return []
    body = text[text.find("\\midrule", at):text.find("\\bottomrule", at)]
    page = [" ".join(line.split()) for line in body.splitlines()
            if "&" in line and line.strip().endswith("\\\\")]
    mine = [" ".join(line.split()) for line in rows_of(table) if "&" in line]
    if page != mine:
        return ["tab:plateau_test: the page's rows are not the recomputed ones:"]\
            + [f"    page {a}\n    mine {b}" for a, b in zip(page, mine) if a != b]\
            + ([f"    {len(page)} rows printed, {len(mine)} recomputed"]
               if len(page) != len(mine) else [])
    return []


def macros_of(table: Sequence[dict]) -> List[Tuple[str, str, str]]:
    """The test-axis numbers the prose cites, each with where it is read."""
    every = next(r for r in table if r["stage"] == ALL)
    extreme = next(r for r in table if r["stage"] == "extreme")
    if every["worst_test_gain_stored"] >= 0 or every["worst_test_gain_drawn"] >= 0:
        raise SystemExit("FATAL: no arm loses on the test axis; the worst-loss macros "
                         "state a loss and would be wrong.")
    p = lambda value: "%.2f" % (100 * value)  # noqa: E731
    return [
        ("nStopTestGainAll", p(every["test_gain_stored"]),
         "all arms: test gain at the stored-trace kept round"),
        ("nStopTestGainDrawn", p(every["test_gain_drawn"]),
         "all arms: test gain at the drawn-only kept round"),
        ("nStopTestOracleAll", p(every["test_oracle_gain"]),
         "all arms: best test round less round 100"),
        ("nStopTestGainExtreme", p(extreme["test_gain_stored"]),
         "the extremes: test gain at the stored-trace kept round"),
        ("nStopTestOracleExtreme", p(extreme["test_oracle_gain"]),
         "the extremes: best test round less round 100"),
        ("nStopTestHurtStored", "%d" % every["arms_hurt_test_stored"],
         "arms the rule lowers on test, stored trace"),
        ("nStopTestHurtDrawn", "%d" % every["arms_hurt_test_drawn"],
         "arms the rule lowers on test, drawn-only trace"),
        ("nStopTestWorstStored", p(-every["worst_test_gain_stored"]),
         "largest test loss, stored trace, magnitude"),
        ("nStopTestWorstDrawn", p(-every["worst_test_gain_drawn"]),
         "largest test loss, drawn-only trace, magnitude"),
        ("nStopKeptDiffArms", "%d" % every["arms_kept_round_differs"],
         "arms whose kept round differs between the two traces"),
        ("nStopKeptDiffRounds", "%.1f" % every["mean_abs_kept_difference"],
         "mean absolute kept-round difference, rounds"),
        ("nStopPerRunTestStored", p(every["per_run_test_gain_stored"]),
         "rule fired per fold: test gain, stored trace"),
        ("nStopPerRunTestDrawn", p(every["per_run_test_gain_drawn"]),
         "rule fired per fold: test gain, drawn-only trace"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root with the run records ($FOA_STUDY_DIR).")
    ap.add_argument("--tex", type=Path, default=None,
                    help="A LaTeX file carrying tab:plateau (the manuscript, or "
                         "tables/plateau.tex) to hold the validation gain to.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory the JSON and the table are written into "
                         f"(default: {DEFAULT_OUT}).")
    ap.add_argument("--numbers", type=Path, default=None,
                    help="The macros file this tool's block is written into "
                         "(default: revision_numbers.tex in --out, or beside "
                         "fold_spread.py's outputs when --out is not given).")
    ap.add_argument("--macros", type=Path, nargs="*", default=[],
                    help="LaTeX files whose \\newcommand definitions the emitted "
                         "macros must equal (e.g. the shipped revision_numbers.tex).")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--min-folds", type=int, default=5)
    args = ap.parse_args()

    staged = verdicts(args.root, args.study_tag, args.min_folds)
    if not staged:
        print("no hundred-round arm on disk under this root.")
        return 1
    reference = {(r["stage"], r["cell"], r["family"]): r
                 for r in outcomes(staged, PRIMARY_PATIENCE, PRIMARY_MARGIN, CHECKPOINT_BEST)}
    refs = {stem.strip("_"): report_tables.baselines_for(args.root, stem) for stem, _ in STAGES}
    books = Books(args.root)
    traces: Dict[str, dict] = {}
    readings = []
    problems: List[str] = []
    for stage, verdict in staged:
        if stage not in traces:
            traces[stage] = read_traces(args.root, f"{args.study_tag}_{stage}_", books)
        a0, p0 = refs[stage]
        reading = arm_reading(stage, verdict, traces[stage][(verdict.arm.cell,
                                                             verdict.arm.family)], a0, p0)
        ref = reference[(stage, verdict.arm.cell, verdict.arm.family)]
        if (ref["kept_round"] != reading["stored"]["kept_round"]
                or abs(ref["gain"] - reading["validation"]["gain_stored"]) > 1e-15):
            problems.append(f"{stage}/{verdict.arm.cell}/{verdict.arm.family}: plateau_rule "
                            f"keeps round {ref['kept_round']} for {ref['gain']:.6f}, this "
                            f"tool {reading['stored']['kept_round']} for "
                            f"{reading['validation']['gain_stored']:.6f}")
        if abs(reading["test"]["round_0"]) > 1e-12:
            problems.append(f"{stage}/{verdict.arm.cell}/{verdict.arm.family}: the test "
                            f"score of round 0 is {reading['test']['round_0']:.3g}, not 0")
        readings.append(reading)

    table = stage_table(readings)
    problems += check_view(table, args.root,
                           REPO / "study" / "artifacts" / "Digits_study01" / "tables"
                           / "stopping" / "plateau_stages.csv")
    numbers = args.numbers or ((args.out or fold_spread.DEFAULT_OUT) / fold_spread.NUMBERS_FILE)
    out = args.out or DEFAULT_OUT
    macros = macros_of(table)
    texts = [m.read_text() for m in args.macros if m.is_file()]
    if args.tex is not None:
        page = args.tex.read_text()
        problems += check_page(table, page) + check_page_test(table, page)
        texts.append(page)
    problems += fold_spread.check_macros(macros, texts)
    if problems:
        print("THE VALIDATION READING DOES NOT REPRODUCE TABLE 8:")
        for p in problems:
            print("  " + p)
        print(f"{len(problems)} mismatch(es); nothing written.")
        return 1

    every = [f for stage in traces.values() for arm in stage.values() for f in arm.values()]
    payload = {
        "generated_by": "tools/stopping_test_axis.py",
        "root": PLACEHOLDER,
        "rule": {"patience": PRIMARY_PATIENCE, "margin": PRIMARY_MARGIN,
                 "variant": CHECKPOINT_BEST, "fired_on": "fold-mean trace"},
        "source_test_per_round": {
            "exists": True,
            "field": "accuracies_0.json accuracies[r][1] (SOURCE_TEST_SET, every round)",
            "rows": "the run's own fold's source test split (evaluation.sets.source_test);"
                    " final_evaluation.old.mean and P0 average all five folds' splits",
            "round_0_test_score_max_abs": max(abs(r["test"]["round_0"]) for r in readings),
        },
        "drawn_only_trace": {
            "definition": "per round, client_val_accuracies pooled over that round's "
                          "participants, weighted by validation rows in the run's fold book",
            "reconstruction_max_error_all_clients": max(f["reconstruction_error"]
                                                        for f in every),
            "runs": len(every),
        },
        "baselines": {stage: {"A0": a0, "P0": p0} for stage, (a0, p0) in refs.items()},
        "stages": table,
        "arms": readings,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "stopping_test_axis.json").write_text(json.dumps(payload, indent=1) + "\n")
    (out / "stopping_test_axis_table.tex").write_text(render(table))
    fold_spread.write_numbers(numbers, "stopping_test_axis.py", macros)

    print(f"{'stage':<10}{'arms':>5}{'val gain':>10}{'test/stored':>13}{'test/drawn':>12}"
          f"{'|dr|':>7}{'fires s/d':>11}{'worst test':>12}")
    for row in table:
        print(f"{row['stage']:<10}{row['arms']:>5}{100 * row['validation_gain']:>+10.2f}"
              f"{100 * row['test_gain_stored']:>+13.2f}{100 * row['test_gain_drawn']:>+12.2f}"
              f"{row['mean_abs_kept_difference']:>7.1f}"
              f"{row['fires_stored']:>6}/{row['fires_drawn']:<4}"
              f"{100 * row['worst_test_gain_stored']:>+12.2f}")
    print(f"  -> {out / 'stopping_test_axis.json'}")
    print(f"  -> {out / 'stopping_test_axis_table.tex'}")
    print(f"  -> {numbers} (" + ", ".join(f"{n}={v}" for n, v, _ in macros) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
