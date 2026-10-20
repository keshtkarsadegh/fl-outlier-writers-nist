"""
tools/clean_preservation.py: preservation on the source rows g-0 never trained on.

The tool reads the run records, which do not ship in the clone, so what is held
here is what does not need them: the parsing that matches a printed row to an
arm by its printed numbers, the arithmetic of the clean score, and the shipped
outputs against each other - the macros against the JSON they are printed from,
and the hand-set schedule table against the rows the tool must find in it. The
record-reading half is held by the tool itself, which fails when a printed row
matches no arm, and by tools/regenerate_paper.sh, which compares its outputs
with the shipped ones.
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

import clean_preservation as CP  # noqa: E402

PAGE = r"""
\begin{table}
\label{tab:toy}
\begin{tabular}{lrr}
\toprule
Arm & Adapt. & Pres. \\
\midrule
\quad One & 0.9187{\scriptsize±0.0166} & 0.9740 \\
\midrule
\quad Two & 0.9000 & 0.9800 \\
\bottomrule
\end{tabular}
\end{table}
"""


def test_a_printed_row_is_matched_to_exactly_one_arm_by_its_printed_numbers():
    rows = CP.table_rows(PAGE, "tab:toy")
    assert [(block, label) for block, label, _ in rows] == [(0, "\\quad One"), (1, "\\quad Two")]
    # The fold spread after a mean is not read as the mean.
    assert rows[0][2] == ["0.9187", "0.9740"]
    arms = {("one", "parallel"): {"adaptation": 0.91874, "preservation": 0.97401},
            ("two", "parallel"): {"adaptation": 0.95, "preservation": 0.98}}
    found = CP.match(rows, arms, 0, 1, 1.0)
    assert [hits for *_, hits in found] == [[("one", "parallel")], []]


def test_the_clean_score_swaps_both_source_baselines_and_leaves_adaptation_alone():
    base = {"p0": 0.99, "p0_clean": 0.995}
    runs = [{"a_test": 0.92, "p_test": 0.98, "p_old": {k: 0.97 + 0.002 * k for k in CP.FOLDS}}
            for _ in range(5)]
    s = CP.summarise(runs, 0.82, base)
    assert s["G"] == pytest.approx(0.10)
    assert s["score"] == pytest.approx(0.10 - (0.99 - 0.98))
    assert s["score_clean"] == pytest.approx(0.10 - (0.995 - 0.972))
    assert s["score_change"] == pytest.approx(s["score_clean"] - s["score"])


def test_the_shipped_macros_are_the_shipped_json_printed():
    report = json.loads((PAPER / "clean_preservation.json").read_text())
    assert report["failures"] == []
    text = (PAPER / "clean_preservation_numbers.tex").read_text()
    for name, fmt in CP.NUMBER_FORMAT.items():
        assert "\\newcommand{\\%s}{%s}" % (name, fmt % report["numbers"][name]) in text


def test_the_hand_set_schedule_table_carries_the_rows_the_tool_reads():
    """Six pairs, each with both schedules' adaptation and preservation."""
    rows = CP.table_rows((PAPER / "schedule_table_as_printed.tex").read_text(), "tab:schedule")
    assert len(rows) == 6
    assert all(len(cells) == 7 for _, _, cells in rows)
    assert rows[0][1].startswith("No-penalty control")


def test_regenerate_runs_it_on_the_generated_tables_and_compares_its_outputs(tmp_path):
    env = dict(os.environ, FOA_STUDY_DIR=str(tmp_path / "records"))
    out = subprocess.run(["bash", str(REPO / "tools" / "regenerate_paper.sh"), "--dry-run",
                          str(tmp_path)], capture_output=True, text=True, env=env,
                         check=True).stdout
    line = next(l for l in out.splitlines() if "tools/clean_preservation.py" in l)
    assert f"--root {tmp_path / 'records'}" in line
    for name in ("references", "agg_winners", "reg_winners", "combos", "scaling",
                 "selection_axis", "selection_axis_cross"):
        assert f"{tmp_path}/tables/{name}.tex " in line
    assert "tables/paper/schedule_table_as_printed.tex" in line
    for name in ("clean_preservation_table.tex", "clean_preservation_numbers.tex"):
        assert (f"+ cmp {tmp_path}/revision/{name} "
                f"study/artifacts/Digits_study01/tables/paper/{name}") in out


# ------------------------------------------------ labels and marks on the page
def test_the_caption_explains_exactly_the_marks_the_rows_carry():
    """Every mark a row sets is explained in the caption, in the main tables' words.

    The table repeats rows of the main tables with their marks, and a mark
    explained only in the prose above the float is a mark its own caption
    leaves unexplained; one that no caption could explain stops the run.
    """
    labels = [r"Anchor + FedNTD$^{\star}$", r"KD+EWC composite$^{\ddagger}$", "plain"]
    note = CP.marks_note(labels)
    assert r"$\star$ marks the selected pair" in note
    assert r"$\ddagger$ the composite that schedule's validation ordering" in note
    assert r"\dagger$ the composite that leads" not in note
    assert CP.marks_note(["plain"]) == ""
    with pytest.raises(SystemExit):
        CP.marks_note([r"Anchor + FedNTD$^{\star\P}$"])


def test_the_shipped_table_names_every_setting_and_explains_its_marks():
    """Rows of the federation-settings group carry their setting; marks are captioned.

    That group repeats the same carried arms at four settings, so a label
    without its setting reads as four rows of one arm with four numbers.
    """
    text = (PAPER / "clean_preservation_table.tex").read_text()
    body = text.split("\\midrule\\endhead", 1)[1].split("\\bottomrule", 1)[0]
    group, settings = None, []
    for line in body.splitlines():
        if line.startswith("\\multicolumn"):
            group = line
            continue
        if group and "tab:scaling" in group and "&" in line:
            label = line.split("&", 1)[0].strip()
            settings.append(label.rsplit(", ", 1)[-1])
    assert settings and set(settings) == set(CP.SETTING_OF_STAGE.values()), settings
    caption = text.split("\\caption{", 1)[1].split("\\label{", 1)[0]
    for mark, words in CP.MARKS:
        if "$^{%s}$" % mark in body:
            assert words in caption, mark
    for old in ("(cross)", "crowned", "blend", "cyclic-tuned", "parallel-tuned", "sibling"):
        assert old not in text, old
