"""
tools/split_evidence.py: the split-level evidence behind the headline sentences.

The tool reads the run records, which do not ship in the clone, so what is
held here is everything that does not need them: that the shipped table is
what the shipped JSON renders to, that every row the JSON carries backs the
number the paper prints (held to the shipped macro files, as the tool holds
it), that a row count, a mean or a macro that moved is refused, that the
group headings say which reading each block is, and that regenerate_paper.sh
runs the tool after the JSON it reads and compares its table with the
shipped copy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "paper"
for entry in (REPO / "src", REPO / "tools", REPO / "tools" / "paper_figures"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import split_evidence as SE  # noqa: E402


def _shipped() -> dict:
    return json.loads((PAPER / "split_evidence.json").read_text())


def _macro_texts() -> list:
    return [(PAPER / "revision_numbers.tex").read_text(),
            (REPO / "tools" / "paper_figures" / "numbers.template.tex").read_text()]


def test_the_shipped_table_is_what_the_shipped_json_renders_to():
    assert SE.render(_shipped()) == (PAPER / "split_evidence_table.tex").read_text()


def test_every_shipped_row_backs_the_number_the_paper_prints():
    payload = _shipped()
    assert SE.check_macros(payload, _macro_texts()) == []
    rows = {r["key"]: r for r in payload["rows"]}
    lead = rows["lead"]
    assert ("%.2f" % lead["mean"], lead["positive"]) == ("1.09", 3)
    assert "%.2f" % rows["lead_w3"]["mean"] == "0.50"
    combos = [r for r in payload["rows"] if r["group"] == "combo"]
    assert sorted("%.2f" % r["mean"] for r in combos) == ["0.77", "1.21"]
    assert all(r["positive"] == 5 for r in combos)
    control = [r for r in payload["rows"] if r["group"] == "control"]
    assert min(r["positive"] for r in control) == 4
    assert "%.2f" % min(r["mean"] for r in control) == "1.34"
    assert ("%.2f" % rows["stop/stored"]["mean"], "%.2f" % rows["stop/drawn"]["mean"]) \
        == ("-0.39", "-2.30")
    assert "%.2f" % rows["extreme/dual"]["mean"] == "0.55"
    assert payload["counts"]["control"] == {"penalties": 19, "below_five": 4, "all_five": 15}


def test_each_mean_is_the_mean_of_its_five_splits_and_the_count_its_positives():
    for r in _shipped()["rows"]:
        assert len(r["values"]) == 5
        assert r["mean"] == pytest.approx(sum(r["values"]) / 5, abs=1e-12)
        assert r["positive"] == sum(1 for v in r["values"] if v > 0)


def test_a_moved_mean_or_an_undefined_macro_is_refused():
    payload = _shipped()
    page = "\\newcommand{\\nNtdAnchorGain}{1.21}\n"
    found = SE.row("x", "combo", "a", "parallel", "test", [1.0, 1.2, 1.4, 1.2, 1.25],
                   [("nNtdAnchorGain", "1.21", "gain")])
    assert SE.check_macros({"rows": [found], "counts": {"combo": {"checks": []}}}, [page]) == []
    moved = SE.row("x", "combo", "a", "parallel", "test", [1.0, 1.2, 1.4, 1.2, 1.3],
                   [("nNtdAnchorGain", "1.22", "gain")])
    assert SE.check_macros({"rows": [moved], "counts": {"combo": {"checks": []}}}, [page])
    assert SE.check_macros(payload, [page])  # the other macros are defined nowhere
    with pytest.raises(SystemExit):
        SE.row("x", "combo", "a", "parallel", "test", [1.0, 2.0])


def test_the_score_is_re_priced_with_the_arms_own_baselines():
    arm = {"a0": 0.8, "p0": 0.99, "adaptation": [0.9] * 5, "preservation": [0.98] * 5}
    assert SE.score_at(arm, 1) == pytest.approx([9.0] * 5)
    assert SE.score_at(arm, 3) == pytest.approx([7.0] * 5)


def test_the_headings_name_the_finding_and_the_reading_of_every_block():
    text = (PAPER / "split_evidence_table.tex").read_text()
    for words in ("final test evaluation", "per-round test trace",
                  "per-round validation trace", "Finding 1", "Finding 2", "Finding 3"):
        assert words in text
    assert "\\label" not in text  # the manuscript's float carries the label


def test_regenerate_runs_it_after_the_json_it_reads_and_compares_its_table(tmp_path):
    env = dict(os.environ, FOA_STUDY_DIR=str(tmp_path / "records"))
    out = subprocess.run(["bash", str(REPO / "tools" / "regenerate_paper.sh"), "--dry-run",
                          str(tmp_path)], capture_output=True, text=True, env=env,
                         check=True).stdout
    lines = out.splitlines()
    axis = next(i for i, l in enumerate(lines) if "tools/stopping_test_axis.py" in l)
    numbers = next(i for i, l in enumerate(lines) if "make_numbers.py" in l)
    evidence = next(i for i, l in enumerate(lines) if "tools/split_evidence.py" in l)
    assert evidence > axis > numbers
    assert (f"--stopping-json {tmp_path}/revision/stopping_test_axis.json "
            f"--macros {tmp_path}/numbers.tex "
            "study/artifacts/Digits_study01/tables/paper/revision_numbers.tex") in lines[evidence]
    assert (f"+ cmp {tmp_path}/revision/split_evidence_table.tex "
            "study/artifacts/Digits_study01/tables/paper/split_evidence_table.tex") in out
