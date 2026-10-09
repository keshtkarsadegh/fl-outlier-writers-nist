"""
tools/stopping_best_vs_best.py: best server rule against best penalty, stopped or not.

It reads only stopping_test_axis.json, which ships, so the whole tool can be
held here: that the shipped macros are what the shipped JSON gives, that the
winner of each pool is the highest test score whatever order the arms come in,
that a JSON from another rule or a macro that moved is refused, and that the
names it prints are the paper's own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STOPPING = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "stopping"
for entry in (REPO / "src", REPO / "tools", REPO / "tools" / "paper_figures"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import stopping_best_vs_best as BB  # noqa: E402


def _arm(stage, family, cell, final, stopped, kept=50, source=0.99):
    return {"stage": stage, "family": family, "cell": cell, "stored": {"kept_round": kept},
            "test": {"final": final, "at_stored_kept": stopped,
                     "source_at_stored_kept": source}}


def _payload(arms):
    return {"rule": dict(BB.RULE), "arms": arms}


def test_the_shipped_macros_are_what_the_shipped_json_gives():
    payload = json.loads((STOPPING / "stopping_test_axis.json").read_text())
    macros = BB.macros_of(BB.compare(payload))
    assert BB.render(macros) == (STOPPING / "stopping_best_vs_best.tex").read_text()
    values = {name: value for name, value, _ in macros}
    assert (values["nBestGapFixedPar"], values["nBestGapFixedCyc"]) == ("2.44", "3.46")
    assert (values["nBestGapStopPar"], values["nBestGapStopCyc"]) == ("0.60", "1.23")
    assert values["nStopBestRulePar"] == "uniform weighting"
    assert (values["nStopBestRuleSrcPar"], values["nStopBestPenSrcPar"]) == ("98.7", "98.9")


def test_each_pool_is_ranked_on_its_own_reading_and_ties_go_to_the_cell_id():
    arms = []
    for family in ("parallel", "cyclic"):
        arms += [_arm("aggfull", family, "weight_q0", 0.070, 0.095, kept=44),
                 _arm("aggfull", family, "anchor_h2", 0.077, 0.085),
                 _arm("regfull", family, "logit_l2_lam0p003", 0.100, 0.100),
                 _arm("regfull", family, "logit_l2_lam0p001", 0.100, 0.100),
                 _arm("combo", family, "not_a_pool", 0.500, 0.500)]
    found = BB.compare(_payload(list(reversed(arms))))
    fixed, stopped = found[("parallel", "final")], found[("parallel", "at_stored_kept")]
    assert fixed["rule"]["cell"] == "anchor_h2" and stopped["rule"]["cell"] == "weight_q0"
    assert fixed["pen"]["cell"] == stopped["pen"]["cell"] == "logit_l2_lam0p001"
    assert stopped["gap"] == pytest.approx(0.005)
    values = {name: value for name, value, _ in BB.macros_of(found)}
    assert values["nStopBestRuleKeptCyc"] == "44"


def test_a_json_from_another_rule_and_a_macro_that_moved_are_refused():
    payload = _payload([_arm(s, f, "c", 0.1, 0.1) for s in ("aggfull", "regfull")
                        for f in ("parallel", "cyclic")])
    payload["rule"]["patience"] = 10
    with pytest.raises(SystemExit):
        BB.compare(payload)
    page = "\\newcommand{\\nBestGapStopPar}{0.60}  % shipped\n"
    assert BB.check_macros([("nBestGapStopPar", "0.60", ""), ("nNew", "1", "")], [page]) == []
    assert BB.check_macros([("nBestGapStopPar", "0.61", "")], [page])


def test_names_are_the_papers_labels_without_their_setting():
    assert BB.prose_name("weight_q0") == "uniform weighting"
    assert BB.prose_name("logit_l2_lam0p001") == "logit matching"
    assert BB.prose_name("seq_delta_scaled") == "cyclic, scaled $\\delta$"
    assert BB.prose_name("hybrid_seq_mix0p75") == "KD+EWC composite"


def test_regenerate_runs_it_on_the_test_axis_json_it_has_just_written(tmp_path):
    env = dict(os.environ, FOA_STUDY_DIR=str(tmp_path / "records"))
    out = subprocess.run(["bash", str(REPO / "tools" / "regenerate_paper.sh"), "--dry-run",
                          str(tmp_path)], capture_output=True, text=True, env=env,
                         check=True).stdout
    lines = out.splitlines()
    axis = next(i for i, l in enumerate(lines) if "tools/stopping_test_axis.py" in l)
    best = next(i for i, l in enumerate(lines) if "tools/stopping_best_vs_best.py" in l)
    assert best > axis
    assert (f"--json {tmp_path}/revision/stopping_test_axis.json "
            f"--out {tmp_path}/revision/stopping_best_vs_best.tex "
            "--macros study/artifacts/Digits_study01/tables/stopping/"
            "stopping_best_vs_best.tex") in lines[best]
