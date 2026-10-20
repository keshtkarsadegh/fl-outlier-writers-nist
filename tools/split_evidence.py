#!/usr/bin/env python3
"""
The split-level evidence behind the paper's headline sentences, in one table.

Six sentences of the abstract and the findings rest on a per-split fact that
no other table prints: that logit matching leads the next single penalty on
three of five splits, that every penalty beats the no-penalty control on at
least four of five, that two combinations beat their penalty on every split,
that the patience rule costs at most 0.39 points on test (2.30 when the
server polls only the drawn clients), and that the merged pair keeps about
half a point of forgetting at the round the rule keeps. A count of splits
cannot be checked against a five-split mean, so this tool prints, for each of
those sentences, the two arms compared, the reading, the five within-split
differences, their mean and the number of splits on which the difference is
positive, all computed from the run records.

    python tools/split_evidence.py --root "$FOA_STUDY_DIR" \\
        --stopping-json "$OUT/revision/stopping_test_axis.json" \\
        --macros "$OUT/numbers.tex" \\
                 study/artifacts/Digits_study01/tables/paper/revision_numbers.tex \\
        --json "$OUT/revision/split_evidence.json" \\
        --table "$OUT/revision/split_evidence_table.tex"

NOTHING HERE IS A NEW READING. Every row is computed through the reader that
already prints its mean: the final-test rows through `fold_spread`'s ``t_reg``
and ``t_combos`` (which hold every recomputed mean to the shipped views), the
stopping rows through `stopping_test_axis`'s traces, rule and score, and the
kept rounds of the stopping rows come from the ``stopping_test_axis.json``
written just before this tool runs and are re-derived here from the traces.
So a row of this table and the sentence it supports cannot be two readings of
two different runs.

EVERY MEAN IS HELD TO THE NUMBER THE PAPER PRINTS. Each row names the prose
macro its mean (or its count) is quoted through - ``\\nSingleLeadMeanPar``,
``\\nPenaltyVsControlMinMargin``, ``\\nNtdAnchorGain``, ``\\nStopTestWorstDrawn``,
``\\nDualOraclePres`` and the rest - and the value recomputed here must equal
every definition of that macro in the ``--macros`` files. A mismatch, or a
macro no file defines, is a refusal and nothing is written. The one row no
macro covers is the single-penalty lead re-priced at w = 3, which the paper
quotes from this table and nowhere else.

WHAT "SPLITS > 0" COUNTS. The last column is the number of splits on which
the row's difference is positive, whatever the row measures: for a lead or a
margin those are the splits the first arm wins; for the stopping rows they
are the splits on which the kept round scores above round 100 (so a loss is
a row with few); for the extreme rows, whose difference is the forgetting
P0 - A_src at the kept round, they are the splits on which source accuracy
is below the shipped model's. The table's group headings say which.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tools", REPO / "tools" / "paper_figures"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import fold_spread as FS  # noqa: E402
import paper_names as P  # noqa: E402

FOLDS = FS.FOLDS
PLACEHOLDER = "$FOA_STUDY_DIR"

#: Where the shipped copies live, beside the other revision outputs.
SHIPPED = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "paper"

#: The price on forgetting the paper's first finding is also re-read at.
#: Appendix B's w-sensitivity table moves screen winners at w = 3, and a
#: reader can recompute the finals' lead there from the printed means; this
#: row does it per split, from the records.
REPRICE_W = 3

#: The combinations whose five within-split deltas against the better
#: component are all positive, and the macro the text quotes each gain by.
#: A combination that clears every split and is not here is a refusal: the
#: text would be naming the wrong pairs.
COMBO_GAIN_MACRO = {"anchor_h2_ntd_b0p01_t0p5": "nNtdAnchorGain",
                    "eta_0p95_ntd_b0p01_t0p5": "nNtdEtaGain"}

#: The two extreme arrangements, merged pair first as the abstract reads them,
#: and the macros the text quotes their kept round and preservation by.
EXTREME_MACROS = (("dual", "nDualSatRound", "nDualOraclePres"),
                  ("double", "nDoubleSatRound", "nDoubleOraclePres"))

#: The rule the stopping rows read; a JSON written under another is refused.
RULE = {"patience": 20, "margin": 0.0, "fired_on": "fold-mean trace"}

READING = {"test": "final test", "trace": "test trace", "val": "val.\\ trace"}
SCHEDULE_SHORT = {"parallel": "par.", "cyclic": "cyc."}


# -------------------------------------------------------------------- rows
def row(key: str, group: str, label: str, schedule: str, reading: str,
        values: Sequence[float], checks: Sequence[Tuple[str, str, str]] = (),
        **meta) -> dict:
    """One printed row: five within-split values, their mean, splits above zero.

    ``checks`` lists ``(macro, recomputed text, what)``: the text this row
    gives for a number the prose quotes through that macro.
    """
    values = [float(v) for v in values]
    if len(values) != len(FOLDS):
        raise SystemExit(f"FATAL: {key}: {len(values)} split values, {len(FOLDS)} expected")
    out = {"key": key, "group": group, "label": label, "schedule": schedule,
           "reading": reading, "folds": list(FOLDS), "values": values,
           "mean": st.mean(values), "positive": sum(1 for v in values if v > 0),
           "checks": [list(c) for c in checks]}
    out.update(meta)
    return out


def score_at(arm: dict, w: float) -> List[float]:
    """The per-split score G - w F in points, with the arm's own A0 and P0."""
    return [100.0 * ((a - arm["a0"]) - w * (arm["p0"] - p))
            for a, p in zip(arm["adaptation"], arm["preservation"])]


def finding_one(study, views) -> Tuple[List[dict], dict]:
    """The single-penalty lead, re-priced at w = 3, and every penalty against the control."""
    _rows, extra = FS.t_reg(study, views)
    lead = next(x for x in extra["single_penalty_lead"] if x["schedule"] == "parallel")
    d = lead["leader_minus_next"]
    name = f"{P.label(lead['leader'])} $-$ {P.label(lead['next'])}"
    out = [row("lead", "lead", name, "parallel", "test", d["values"],
               [("nSingleLeadMeanPar", "%.2f" % d["mean"], "mean lead, w = 1"),
                ("nSingleLeadWonPar", "%d" % d["won"], "splits the leader wins")],
               leader=lead["leader"], next=lead["next"], w=1)]
    a = study.arm("regfull", lead["leader"], "parallel")
    b = study.arm("regfull", lead["next"], "parallel")
    if [round(x, 9) for x in FS.paired(score_at(a, 1), score_at(b, 1))["values"]] != \
            [round(x, 9) for x in d["values"]]:
        raise SystemExit("FATAL: the w = 1 lead recomputed from the two arms is not "
                         "fold_spread's; the re-priced row would read other runs.")
    priced = [x - y for x, y in zip(score_at(a, REPRICE_W), score_at(b, REPRICE_W))]
    out.append(row("lead_w%d" % REPRICE_W, "lead", f"{name}, at $w{{=}}{REPRICE_W}$",
                   "parallel", "test", priced, leader=lead["leader"], next=lead["next"],
                   w=REPRICE_W))

    against = extra["penalties_minus_control"]
    summary = extra["penalties_minus_control_summary"]
    weakest = [x for x in against if x["minus_control"]["won"] < len(FOLDS)]
    low = min(against, key=lambda x: x["minus_control"]["mean"])
    if low not in weakest:
        weakest.append(low)
    for x in sorted(weakest, key=lambda x: (x["schedule"] != "parallel",
                                            -x["minus_control"]["mean"])):
        m = x["minus_control"]
        checks = []
        if x is low:
            checks.append(("nPenaltyVsControlMinMargin", "%.2f" % m["mean"],
                           "smallest mean margin of a penalty over the control"))
        if m["won"] == summary["min_folds_won"]:
            checks.append(("nPenaltyVsControlMinWon", "%d" % m["won"],
                           "fewest splits a penalty beats the control on"))
        out.append(row(f"control/{x['cell']}/{x['schedule']}", "control",
                       f"{P.label(x['cell'])} $-$ {P.label(FS_CONTROL)}",
                       x["schedule"], "test", m["values"], checks, cell=x["cell"]))
    counts = {"penalties": summary["penalties"],
              "below_five": sum(1 for x in against if x["minus_control"]["won"] < len(FOLDS)),
              "all_five": summary["rows_winning_every_fold"]}
    if counts["below_five"] + counts["all_five"] != counts["penalties"]:
        raise SystemExit(f"FATAL: penalty counts do not add up: {counts}")
    return out, counts


#: The no-penalty control every penalty is differenced against (fold_spread's
#: ``t_reg`` borrows the aggregation stage's plain FedAvg row on each schedule).
FS_CONTROL = "control_fedavg"


def finding_two(study, views) -> Tuple[List[dict], dict]:
    """The combinations that beat their better component on every split."""
    _rows, extra = FS.t_combos(study, views)
    clear = [p for p in extra["paired"] if p["vs_better_component"]["won"] == len(FOLDS)]
    out = []
    for p in sorted(clear, key=lambda p: -p["vs_better_component"]["mean"]):
        macro = COMBO_GAIN_MACRO.get(p["cell"])
        if macro is None:
            raise SystemExit(f"FATAL: {p['cell']} ({p['schedule']}) beats its better "
                             "component on every split and the text names no gain for it.")
        d = p["vs_better_component"]
        out.append(row(f"combo/{p['cell']}/{p['schedule']}", "combo",
                       f"{P.short(p['cell'])} $-$ {P.label(p['better_component'])}",
                       p["schedule"], "test", d["values"],
                       [(macro, "%.2f" % d["mean"], "mean gain over the better component")],
                       cell=p["cell"], better_component=p["better_component"]))
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    counts = {"pairs": extra["sign_agreement"]["pairs"], "every_split": len(clear),
              "checks": [["nComboBeatFolds", words.get(len(clear), str(len(clear))),
                          "combinations that beat their better component on every split"]]}
    return out, counts


def finding_three(root: Path, payload: dict) -> List[dict]:
    """The patience rule's largest test loss on each trace, and the extremes' forgetting."""
    import stopping_test_axis as STA
    from extreme_stopping import fold_mean

    rule = payload.get("rule") or {}
    if any(rule.get(k) != v for k, v in RULE.items()):
        raise SystemExit(f"FATAL: {rule} is not the published rule {RULE}")
    every = next(s for s in payload["stages"] if s["stage"] == "all")
    arms = {(a["stage"], a["cell"], a["family"]): a for a in payload["arms"]}
    books = STA.Books(root)
    traces: Dict[str, dict] = {}

    def folds_of(stage, cell, family):
        if stage not in traces:
            traces[stage] = STA.read_traces(root, f"d01_{stage}_", books)
        found = traces[stage][(cell, family)]
        return [found[f] for f in FOLDS]

    def kept(stage, cell, family):
        """Both kept rounds re-derived from the traces and held to the JSON."""
        arm = arms[(stage, cell, family)]
        folds = folds_of(stage, cell, family)
        n = arm["rounds"]
        _f, ks = STA.keep(fold_mean([f["pool_val"] for f in folds])[:n])
        _f, kd = STA.keep(fold_mean([f["drawn_val"] for f in folds])[:n])
        if (ks + 1, kd + 1) != (arm["stored"]["kept_round"], arm["drawn_only"]["kept_round"]):
            raise SystemExit(f"FATAL: {stage}/{cell}/{family}: kept rounds {ks + 1}/{kd + 1} "
                             f"from the records, {arm['stored']['kept_round']}/"
                             f"{arm['drawn_only']['kept_round']} in the JSON")
        return arm, folds, n, {"stored": ks, "drawn": kd}

    out = []
    base = payload["baselines"]
    for trace, gain, worst_key, macro in (
            ("stored", "gain_stored", "worst_test_arm_stored", "nStopTestWorstStored"),
            ("drawn", "gain_drawn", "worst_test_arm_drawn", "nStopTestWorstDrawn")):
        cell, family = every[worst_key].split("/")
        # A cell id can recur across stages; the worst arm is the one whose gain
        # is the stage table's worst, which names it uniquely.
        stage = next(s for (s, c, f), a in arms.items() if c == cell and f == family
                     and a["test"][gain] == every[worst_key.replace("arm", "gain")])
        arm, folds, n, k = kept(stage, cell, family)
        a0, p0 = base[stage]["A0"], base[stage]["P0"]
        values = []
        for f in folds:
            t = STA.test_scores(f, a0, p0)[:n]
            values.append(100.0 * (t[k[trace]] - t[-1]))
        if abs(st.mean(values) - 100.0 * arm["test"][gain]) > 1e-9:
            raise SystemExit(f"FATAL: {stage}/{cell}/{family}: the split values average "
                             f"{st.mean(values):.6f}, the JSON's gain is "
                             f"{100 * arm['test'][gain]:.6f}")
        word = "stored" if trace == "stored" else "drawn-only"
        out.append(row(f"stop/{trace}", "stop",
                       f"{P.label(cell)}, kept round {k[trace] + 1} $-$ round {n} ({word} trace)",
                       family, "trace", values,
                       [(macro, "%.2f" % -st.mean(values),
                         f"largest test loss of the rule, {word} trace, magnitude")],
                       cell=cell, stage=stage, kept_round=k[trace] + 1))

    for cell, round_macro, pres_macro in EXTREME_MACROS:
        family = next(f for (s, c, f) in arms if s == "extreme" and c == cell)
        arm, folds, n, k = kept("extreme", cell, family)
        p0 = base["extreme"]["P0"]
        source = [f["source_val"][k["stored"]] for f in folds]
        values = [100.0 * (p0 - s) for s in source]
        own = [100.0 * (f["source_val"][0] - s) for f, s in zip(folds, source)]
        out.append(row(f"extreme/{cell}", "extreme",
                       f"{P.label(cell)}, forgetting at kept round {k['stored'] + 1}",
                       family, "val", values,
                       [(round_macro, "%d" % (k["stored"] + 1), "the round the rule keeps"),
                        (pres_macro, "%.4f" % st.mean(source),
                         "source validation accuracy at that round")],
                       cell=cell, kept_round=k["stored"] + 1,
                       forgetting_against_own_round_0=own))
    return out


# ---------------------------------------------------------------- the page
def _num(value: float) -> str:
    text = "%.2f" % value
    if text in ("0.00", "-0.00"):
        return "0.00"
    return "$%s$" % text if text.startswith("-") else "$+%s$" % text


def headings(counts: dict) -> Dict[str, str]:
    """The group headings, each naming the headline it backs and the reading."""
    c, k = counts["control"], counts["combo"]
    return {
        "lead": "Finding 1: the leading single penalty against the next, final test "
                "evaluation, score; splits $>0$ are splits the leader wins",
        "control": ("Finding 1: each penalty against the FedAvg control, final test "
                    "evaluation, score; the %d of %d penalties that win fewer than five "
                    "splits, and the one with the smallest mean (the other %d win all five)"
                    % (c["below_five"], c["penalties"], c["all_five"])),
        "combo": ("Finding 2: the %d of %d combinations that beat their better component "
                  "on every split, final test evaluation, score" % (k["every_split"], k["pairs"])),
        "stop": "Finding 3: the patience rule's largest test loss, per-round test trace, "
                "score at the kept round less round 100",
        "extreme": "Finding 3: the extreme arrangements at the kept round, per-round "
                   "validation trace, forgetting $P_0-\\Asrc$ in points",
    }


ORDER = ("lead", "control", "combo", "stop", "extreme")


def render(payload: dict) -> str:
    """The tabular the manuscript wraps in its own table float."""
    heads = headings(payload["counts"])
    lines = [
        "% generated by tools/split_evidence.py --- do not edit by hand",
        "% each row: the five within-split differences (points), their mean, and",
        "% the number of splits on which the difference is positive.",
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.38\\linewidth}"
        "llrrrrrrr@{}}",
        "\\toprule",
        "Comparison & Sched. & Reading & \\multicolumn{5}{c}{Split} & Mean & Splits \\\\",
        "\\cmidrule(lr){4-8}",
        " & & & 1 & 2 & 3 & 4 & 5 & (pts) & $>0$ \\\\",
    ]
    for group in ORDER:
        rows = [r for r in payload["rows"] if r["group"] == group]
        if not rows:
            continue
        lines.append("\\midrule")
        lines.append("\\multicolumn{10}{@{}p{\\linewidth}@{}}{\\emph{%s}} \\\\" % heads[group])
        for r in rows:
            lines.append(" & ".join(
                [r["label"], SCHEDULE_SHORT[r["schedule"]], READING[r["reading"]]]
                + [_num(v) for v in r["values"]]
                + [_num(r["mean"]), "%d/%d" % (r["positive"], len(r["values"]))]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ checks
def check_macros(payload: dict, texts: Sequence[str]) -> List[str]:
    """Every number a row backs against every definition of its macro."""
    known = FS.defined(texts)
    wanted = [tuple(c) for r in payload["rows"] for c in r["checks"]]
    wanted += [tuple(c) for c in payload["counts"]["combo"]["checks"]]
    problems = []
    for macro, value, what in wanted:
        found = known.get(macro)
        if not found:
            problems.append(f"\\{macro} ({what}): defined in no --macros file")
            continue
        problems += [f"\\{macro} ({what}): recomputed {value}, defined as {other}"
                     for other in found if other != value]
    return problems


def build(root: Path, stopping: dict) -> dict:
    study = FS.Study(root, "d01")
    views = FS.load_views()
    one, control = finding_one(study, views)
    two, combo = finding_two(study, views)
    three = finding_three(root, stopping)
    return {"generated_by": "tools/split_evidence.py", "root": PLACEHOLDER,
            "basis": "points; final test = final_evaluation (fold_spread); test trace = "
                     "per-round pool_test against the run's own source test split at the "
                     "patience rule's kept round less round 100 (stopping_test_axis); "
                     "val. trace = P0 less source_val_accuracies at the kept round",
            "counts": {"control": control, "combo": combo},
            "rows": one + two + three}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root with the run records ($FOA_STUDY_DIR).")
    ap.add_argument("--stopping-json", required=True, type=Path,
                    help="stopping_test_axis.json, as stopping_test_axis.py wrote it.")
    ap.add_argument("--macros", type=Path, nargs="+", required=True,
                    help="LaTeX files whose \\newcommand definitions each row's mean "
                         "must equal: numbers.tex and the shipped revision_numbers.tex.")
    ap.add_argument("--json", type=Path, default=SHIPPED / "split_evidence.json")
    ap.add_argument("--table", type=Path, default=SHIPPED / "split_evidence_table.tex")
    args = ap.parse_args()

    payload = build(args.root, json.loads(args.stopping_json.read_text()))
    missing = [str(m) for m in args.macros if not m.is_file()]
    problems = [f"--macros {m}: no such file" for m in missing]
    problems += check_macros(payload, [m.read_text() for m in args.macros if m.is_file()])
    if problems:
        print("THE SPLIT EVIDENCE DOES NOT REPRODUCE WHAT THE PAPER PRINTS:")
        for p in problems:
            print("  " + p)
        print(f"{len(problems)} mismatch(es); nothing written.")
        return 1
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1) + "\n")
    args.table.write_text(render(payload))
    for r in payload["rows"]:
        print(f"{r['key']:<45} {r['reading']:<6} mean {r['mean']:+.4f}  splits "
              + " ".join(f"{v:+.2f}" for v in r["values"])
              + f"  >0 on {r['positive']} of {len(r['values'])}")
    print(f"  -> {args.json}")
    print(f"  -> {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
