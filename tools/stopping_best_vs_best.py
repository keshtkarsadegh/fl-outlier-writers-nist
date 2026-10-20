#!/usr/bin/env python3
"""
The best server rule against the best penalty, with and without the stopping rule.

    python tools/stopping_best_vs_best.py
    python tools/stopping_best_vs_best.py --json <out>/revision/stopping_test_axis.json \\
        --out <out>/revision/stopping_best_vs_best.tex \\
        --macros study/artifacts/Digits_study01/tables/stopping/stopping_best_vs_best.tex

WHY THIS EXISTS. The paper's answer to "rule or penalty?" is read at the fixed
budget: on the test axis the best penalty clears the best server rule by more
than two points on either schedule. A reader who has seen Table 8 asks the
obvious next question - the server rules are the arms that peak early and then
drift, so does stopping them at the patience rule's kept round close the gap?
That is a comparison of two maxima, best rule against best penalty, and it has
to be made on both readings with the same pools, or the answer is a comparison
of two different selections.

WHAT IS COMPARED. Per schedule (parallel, cyclic), every arm of the aggregation
stage (``aggfull``, the server rules) and every arm of the regularisation stage
(``regfull``, the penalties), ranked by test score twice:

    fixed budget    ``test.final``            the score at round 100
    patience rule   ``test.at_stored_kept``   the score at the round the patience
                                              rule (k=20, eps=0, best checkpoint)
                                              keeps on the validation trace

The best of each pool, their two scores and the gap are emitted; under the
patience rule also each winner's kept round and its source accuracy at that
round (``test.source_at_stored_kept``), because a server rule that wins back
score by stopping early does so by keeping more of the source, and the prose
says how much. The winner is ranked on the unrounded score, ties to the cell
id, so the pick does not depend on the order the arms were written in.

THE INPUT IS stopping_test_axis.py's JSON, NOT THE RECORDS. Every number here
is already in ``stopping_test_axis.json``, scored there on the per-round source
test series and held there to Table 8, so this only selects and prints. It runs
in the clone against the shipped JSON, and `tools/regenerate_paper.sh` runs it
against the JSON it has just rebuilt from the records.

WHAT IS CHECKED. The JSON must state the published rule; every pool must hold
at least one arm; the printed gap must equal the difference of the two printed
scores, so the prose never shows a subtraction that does not add up; and, with
``--macros``, every emitted macro must equal any definition of it already
written there (the shipped copy), as fold_spread.py and stopping_test_axis.py
hold theirs. Any failure writes nothing.

    stopping_best_vs_best.tex   one \\newcommand per number, each commented
                                with the field it was read from
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "tools" / "paper_figures",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import paper_names  # noqa: E402

STOPPING = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "stopping"
DEFAULT_JSON = STOPPING / "stopping_test_axis.json"
DEFAULT_OUT = STOPPING / "stopping_best_vs_best.tex"

#: The rule the test-axis JSON must have been computed under: the published
#: setting of Table 8. A JSON from a sweep of other settings is not this one.
RULE = {"patience": 20, "margin": 0.0, "variant": "checkpoint_best"}

#: Pool -> the stage its arms come from. Each pool is the whole stage, the
#: controls included: a control that out-scored every method would be the
#: honest best of its pool, and the name macro would say so.
POOLS = (("Rule", "aggfull"), ("Pen", "regfull"))

#: Schedule -> the suffix every macro of it carries.
SCHEDULES = (("parallel", "Par"), ("cyclic", "Cyc"))

#: Reading -> (the field of ``test`` it ranks by, the word in the macro name).
READINGS = (("final", "Fixed"), ("at_stored_kept", "Stop"))

DEFINITION = re.compile(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}")

Macro = Tuple[str, str, str]


def points(value: float) -> str:
    """A score in points to two places, as every score table prints it."""
    return "%.2f" % (100 * value)


def percent(value: float) -> str:
    """An accuracy as a percentage to one place, as the prose quotes it."""
    return "%.1f" % (100 * value)


def prose_name(cell: str) -> str:
    """
    The method's name as a sentence uses it: the paper's label, no setting.

    Read from `paper_names.label` - the one translation every table uses - so
    the prose and the tables cannot name an arm two ways. The setting in its
    trailing parentheses is dropped, since the macro sits in running text, and
    the first letter is lowered unless the first word is an acronym (FedAvg,
    KD+EWC), which keeps its case.
    """
    name = paper_names.label(cell)
    if name.endswith(")") and " (" in name:
        depth = 0
        for at in range(len(name) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(name[at], 0)
            if depth == 0:
                name = name[:at].rstrip()
                break
    first = name.split()[0] if name else ""
    if len(first) > 1 and first[0].isupper() and first[1:].islower():
        name = name[0].lower() + name[1:]
    return name


def best(arms: Sequence[dict], stage: str, schedule: str, field: str) -> dict:
    """The arm of one pool with the highest test score on one reading."""
    pool = [a for a in arms if a["stage"] == stage and a["family"] == schedule]
    if not pool:
        raise SystemExit(f"FATAL: no {stage} arm on the {schedule} schedule in the JSON")
    return sorted(pool, key=lambda a: (-a["test"][field], a["cell"]))[0]


def compare(payload: dict) -> Dict[Tuple[str, str], dict]:
    """
    Per (schedule, reading): the two winners, their scores and the gap.

    The gap is penalty minus rule, unrounded, so its sign says which side
    leads.
    """
    rule = {k: payload.get("rule", {}).get(k) for k in RULE}
    if rule != RULE:
        raise SystemExit(f"FATAL: the JSON was computed under {rule}, not the "
                         f"published rule {RULE}")
    out = {}
    for schedule, _ in SCHEDULES:
        for field, _ in READINGS:
            side = {tag: best(payload["arms"], stage, schedule, field)
                    for tag, stage in POOLS}
            out[(schedule, field)] = {
                "rule": side["Rule"], "pen": side["Pen"],
                "rule_score": side["Rule"]["test"][field],
                "pen_score": side["Pen"]["test"][field],
                "gap": side["Pen"]["test"][field] - side["Rule"]["test"][field],
            }
    return out


def macros_of(found: Dict[Tuple[str, str], dict]) -> List[Macro]:
    """The numbers and names the prose cites, each with the field it is read from."""
    out: List[Macro] = []
    for schedule, s in SCHEDULES:
        for field, word in READINGS:
            row = found[(schedule, field)]
            rule, pen = row["rule"], row["pen"]
            where = f"{schedule}, test.{field}"
            out += [
                (f"nBestRule{word}{s}", points(row["rule_score"]),
                 f"{where}: best aggfull arm ({rule['cell']})"),
                (f"nBestPen{word}{s}", points(row["pen_score"]),
                 f"{where}: best regfull arm ({pen['cell']})"),
                (f"nBestGap{word}{s}", points(row["gap"]),
                 f"{where}: best penalty less best server rule"),
                (f"n{word}BestRule{s}", prose_name(rule["cell"]),
                 f"{where}: {rule['cell']}"),
                (f"n{word}BestPen{s}", prose_name(pen["cell"]),
                 f"{where}: {pen['cell']}"),
            ]
            if field == "at_stored_kept":
                for tag, arm in (("Rule", rule), ("Pen", pen)):
                    out += [
                        (f"nStopBest{tag}Kept{s}", "%d" % arm["stored"]["kept_round"],
                         f"{schedule}: stored.kept_round of {arm['cell']}"),
                        (f"nStopBest{tag}Src{s}",
                         percent(arm["test"]["source_at_stored_kept"]),
                         f"{schedule}: test.source_at_stored_kept of {arm['cell']}, percent"),
                    ]
    return out


def check_gaps(found: Dict[Tuple[str, str], dict]) -> List[str]:
    """Each printed gap against the difference of the two printed scores."""
    problems = []
    for (schedule, field), row in found.items():
        shown = float(points(row["pen_score"])) - float(points(row["rule_score"]))
        if "%.2f" % shown != points(row["gap"]):
            problems.append(f"{schedule} test.{field}: gap {points(row['gap'])} is not "
                            f"{points(row['pen_score'])} - {points(row['rule_score'])}")
    return problems


def check_macros(macros: Sequence[Macro], texts: Sequence[str]) -> List[str]:
    """Each emitted macro against every definition of its name already written."""
    known: Dict[str, List[str]] = {}
    for text in texts:
        for name, value in DEFINITION.findall(text):
            known.setdefault(name, []).append(value)
    return [f"\\{name}: recomputed {value}, defined as {other}"
            for name, value, _source in macros
            for other in known.get(name, []) if other != value]


def render(macros: Sequence[Macro]) -> str:
    lines = ["% generated by tools/stopping_best_vs_best.py --- do not edit by hand",
             "% best server rule (aggfull) against best penalty (regfull) by test score,",
             "% at the fixed budget (Fixed) and at the patience rule's kept round (Stop);",
             "% scores and gaps in points, source accuracies in percent.", ""]
    lines += ["\\newcommand{\\%s}{%s}  %% %s" % m for m in macros]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON,
                    help="stopping_test_axis.py's JSON (default: the shipped "
                         f"{DEFAULT_JSON.relative_to(REPO)}).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"The macros file to write (default: {DEFAULT_OUT.relative_to(REPO)}).")
    ap.add_argument("--macros", type=Path, nargs="*", default=[],
                    help="LaTeX files whose \\newcommand definitions the emitted "
                         "macros must equal (e.g. the shipped stopping_best_vs_best.tex).")
    args = ap.parse_args()

    found = compare(json.loads(args.json.read_text()))
    macros = macros_of(found)
    texts = [m.read_text() for m in args.macros if m.is_file()]
    problems = check_gaps(found) + check_macros(macros, texts)
    if problems:
        print("THE BEST-AGAINST-BEST NUMBERS DO NOT HOLD:")
        for p in problems:
            print("  " + p)
        print(f"{len(problems)} problem(s); nothing written.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(macros))
    print(f"{'schedule':<10}{'reading':<16}{'best rule':>26}{'best penalty':>28}{'gap':>7}")
    for schedule, _ in SCHEDULES:
        for field, _ in READINGS:
            row = found[(schedule, field)]
            print(f"{schedule:<10}{field:<16}"
                  f"{row['rule']['cell']:>19} {points(row['rule_score']):>6}"
                  f"{row['pen']['cell']:>21} {points(row['pen_score']):>6}"
                  f"{points(row['gap']):>7}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
