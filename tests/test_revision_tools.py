"""
The three revision tools: fold spread, the stopping rule on test, the w counts.

All three read the run records, which do not ship in the clone, so what can be
held here is everything that does not need them: the arithmetic the tables are
rendered with, the parsing that holds a recomputed mean to a printed one, the
pooling the drawn-only trace is built from, and the shipped outputs against the
views they must agree with. The record-reading half is held by the tools
themselves, which refuse to write when a recomputed mean misses the page.
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
TABLES = REPO / "study" / "artifacts" / "Digits_study01" / "tables"

for entry in (str(REPO / "src"), str(TOOLS), str(TOOLS / "paper_figures")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fold_spread as FS  # noqa: E402
import stopping_test_axis as STA  # noqa: E402


# ------------------------------------------------------------ the arithmetic
def test_the_spread_is_the_sample_sd_not_the_population_one():
    """Five splits are a sample of re-draws; the views' pstdev is not used."""
    out = FS.summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert out["mean"] == 3.0
    assert out["sd"] == pytest.approx(math.sqrt(2.5))
    assert (out["min"], out["max"]) == (1.0, 5.0)


def test_a_paired_difference_counts_the_folds_each_side_won():
    out = FS.paired([3, 2, 5, 1, 4], [1, 2, 4, 2, 3])
    assert out["values"] == [2, 0, 1, -1, 1]
    assert (out["won"], out["lost"]) == (3, 1)
    assert not out["same_sign_all_folds"]
    assert FS.paired([2, 2, 2, 2, 2], [1, 1, 1, 1, 1])["same_sign_all_folds"]


def test_a_cell_prints_its_mean_as_the_page_does_and_the_sd_beside_it():
    acc = FS.Stat("adaptation", [0.9187, 0.9190, 0.9150, 0.9220, 0.9188], 4)
    assert acc.text().startswith("%.4f{\\scriptsize±" % acc.mean)
    delta = FS.Stat("delta", [0.5, -0.2, 0.3, 0.1, 0.2], 2, signed=True, won=4)
    assert delta.text().startswith("$+0.18${\\scriptsize±")
    assert delta.text().endswith(" (4/5)")
    loss = FS.Stat("delta", [-0.1] * 5, 2, signed=True)
    assert loss.text() == "$-0.10${\\scriptsize±0.00}"


# ------------------------------------------------------ holding to the page
PAGE = r"""
\label{tab:toy}
\begin{tabular}{lrr}
\toprule
Arm & Adapt. & Score \\
\midrule
\rowcolor{blockband}\multicolumn{3}{l}{\textbf{Block}} \\
\quad One & 0.9187 & $+7.75$ \\
\midrule
\quad Two & 0.9000 & 6.00 \\
\bottomrule
\end{tabular}
"""


def test_the_body_is_read_between_the_header_rule_and_the_bottom_rule():
    body = FS.table_body(PAGE, "tab:toy")
    data = [line for line in body if FS.is_data(line)]
    assert [FS.split_cells(line)[0] for line in data] == ["\\quad One", "\\quad Two"]
    assert FS.table_body(PAGE, "tab:missing") is None


def test_a_mean_that_does_not_round_to_the_printed_number_is_refused():
    row = FS.Row("one", ["one"], [FS.Stat("adaptation", [0.91871] * 5, 4),
                                  FS.Stat("score", [7.75] * 5, 2, signed=True)], {})
    assert FS.check_row(row, ["0.9187", "$+7.75$"], "toy") == []
    wrong = FS.check_row(row, ["0.9188", "$+7.75$"], "toy")
    assert len(wrong) == 1 and "0.9188" in wrong[0]


def test_the_rendered_body_keeps_every_rule_and_label_of_the_page():
    rows = [FS.Row("one", ["one"], [FS.Stat("a", [0.91871] * 5, 4),
                                    FS.Stat("s", [7.75] * 5, 2, signed=True)], {}),
            FS.Row("two", ["two"], [FS.Stat("a", [0.9] * 5, 4),
                                    FS.Stat("s", [6.0] * 5, 2)], {})]
    out = FS.render(rows, FS.table_body(PAGE, "tab:toy"), 1)
    assert out[0].startswith("\\rowcolor") and out[2] == "\\midrule"
    assert out[1].startswith("\\quad One & 0.9187{")
    assert out[3].startswith("\\quad Two & 0.9000{")


# ------------------------------------------------------- the drawn-only trace
def test_pooling_weights_each_client_by_its_validation_rows():
    accuracies = {"a": 1.0, "b": 0.5, "c": 0.0}
    rows = {"a": 30, "b": 10, "c": 60}
    assert STA.pooled(accuracies, rows, ["a", "b", "c"]) == pytest.approx(35 / 100)
    assert STA.pooled(accuracies, rows, ["a", "b"]) == pytest.approx(35 / 40)


def test_the_rule_is_the_published_setting():
    import plateau_rule

    assert (STA.PRIMARY_PATIENCE, STA.PRIMARY_MARGIN) == (20, 0.0)
    assert STA.CHECKPOINT_BEST == plateau_rule.CHECKPOINT_BEST
    flat = [0.5] + [0.4] * 30
    fired, kept = STA.keep(flat)
    assert (fired, kept) == (20, 0)


# ------------------------------------------------------ the shipped outputs
def test_the_shipped_validation_gains_are_table_eights():
    """stopping_test_axis.json's validation column is plateau_stages.csv's gain."""
    shipped = json.loads((TABLES / "stopping" / "stopping_test_axis.json").read_text())
    stages = {r["stage"]: r for r in csv.DictReader(open(TABLES / "stopping" / "plateau_stages.csv"))
              if r["patience"] == "20" and float(r["margin"]) == 0.0
              and r["variant"] == "checkpoint_best"}
    assert [r["stage"] for r in shipped["stages"]][-1] == "all"
    for row in shipped["stages"]:
        assert row["arms"] == int(stages[row["stage"]]["arms"])
        assert row["validation_gain"] == pytest.approx(float(stages[row["stage"]]["gain"]),
                                                       abs=1e-12)
    assert shipped["source_test_per_round"]["round_0_test_score_max_abs"] < 1e-12
    assert shipped["drawn_only_trace"]["reconstruction_max_error_all_clients"] < 1e-12


def test_the_shipped_fold_spread_covers_every_table_and_states_the_null():
    shipped = json.loads((TABLES / "paper" / "fold_spread.json").read_text())
    assert [t for t in shipped["tables"]] == [label for label, *_ in FS.TABLES]
    sign = shipped["tables"]["tab:combos"]["sign_agreement"]
    assert sign["pairs"] == 18
    assert sign["expected_count_under_null"] == pytest.approx(18 * 2 * 0.5 ** 5)
    for table in shipped["tables"].values():
        for row in table["rows"]:
            for column in row["columns"].values():
                assert len(column["values"]) == 5
    spread = shipped["control_replicates"]["cyclic"]["plain_minus_armed"]
    assert round(spread["mean"], 2) == 0.57


def test_the_shipped_w_counts_are_the_sensitivity_views_counts():
    text = (TABLES / "w_sensitivity_table.tex").read_text()
    for grid, name in (("agg", "Server rules"), ("reg", "Client penalties")):
        view = list(csv.DictReader(open(TABLES / f"weight_sensitivity_{grid}.csv")))
        moved = [sum(1 for r in view if r[f"w={w}"] != r["w=1"]) for w in (2, 3, 5)]
        line = next(l for l in text.splitlines() if l.startswith(f"\\emph{{{name}, all}}"))
        cells = [c.strip() for c in line.rstrip(" \\").split("&")]
        assert int(cells[2]) == len(view)
        assert [int(c) for c in cells[4:7]] == moved


def test_regenerate_runs_the_three_only_when_the_records_are_there(tmp_path):
    script = TOOLS / "regenerate_paper.sh"
    env = {k: v for k, v in os.environ.items() if k != "FOA_STUDY_DIR"}
    out = subprocess.run(["bash", str(script), "--dry-run", str(tmp_path)],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "skipped fold_spread.py" in out and "+ python tools/fold_spread.py" not in out
    env["FOA_STUDY_DIR"] = str(tmp_path / "records")
    out = subprocess.run(["bash", str(script), "--dry-run", str(tmp_path)],
                         capture_output=True, text=True, env=env, check=True).stdout
    for tool in ("fold_spread.py", "stopping_test_axis.py", "sensitivity_counts.py"):
        assert f"+ python tools/{tool} --root {tmp_path / 'records'}" in out
    assert out.count("--macros study/artifacts/Digits_study01/tables/paper/revision_numbers.tex") == 2


# ------------------------------------------------- added columns and macros
def test_a_page_without_an_added_column_is_held_on_the_columns_it_has():
    row = FS.Row("one", ["one"], [FS.Stat("a", [0.91871] * 5, 4),
                                  FS.Stat("new", [1.0] * 5, 2, signed=True, added=True),
                                  FS.Stat("s", [7.75] * 5, 2)], {})
    assert FS.check_row(row, ["0.9187", "7.75"], "old page") == []
    assert FS.check_row(row, ["0.9187", "$+1.00$", "7.75"], "new page") == []
    assert len(FS.check_row(row, ["0.9187", "$+1.10$", "7.75"], "new page")) == 1


def test_an_added_column_widens_the_block_headers_of_the_page_body():
    body = ["\\multicolumn{2}{l}{\\emph{Block}} \\\\", "\\quad One & 0.9187 \\\\"]
    rows = [FS.Row("one", ["one"], [FS.Stat("a", [0.91871] * 5, 4),
                                    FS.Stat("new", [1.0] * 5, 2, added=True)], {})]
    out = FS.render(rows, body, 1)
    assert out[0].startswith("\\multicolumn{3}")
    assert out[1].count("&") == 2


def test_each_tool_rewrites_only_its_own_block_of_the_numbers_file(tmp_path):
    path = tmp_path / FS.NUMBERS_FILE
    FS.write_numbers(path, "stopping_test_axis.py", [("nB", "2", "b")])
    FS.write_numbers(path, "fold_spread.py", [("nA", "1", "a")])
    FS.write_numbers(path, "fold_spread.py", [("nA", "3", "a")])
    found = FS.defined([path.read_text()])
    assert found == {"nA": ["3"], "nB": ["2"]}
    text = path.read_text()
    assert text.index("% begin fold_spread.py") < text.index("% begin stopping_test_axis.py")


def test_a_macro_that_moved_is_refused_and_a_new_one_is_not():
    page = "\\newcommand{\\nA}{0.85}\n"
    assert FS.check_macros([("nA", "0.85", ""), ("nNew", "1", "")], [page]) == []
    assert FS.check_macros([("nA", "0.86", "")], [page])


def test_the_shipped_numbers_are_the_ones_the_shipped_json_implies():
    numbers = FS.defined([(TABLES / "paper" / FS.NUMBERS_FILE).read_text()])
    fold = json.loads((TABLES / "paper" / "fold_spread.json").read_text())
    stop = json.loads((TABLES / "stopping" / "stopping_test_axis.json").read_text())
    for name, value, _source in FS.macros_of(fold) + STA.macros_of(stop["stages"]):
        assert numbers[name] == [value], name
    assert numbers["nStopTestGainAll"] == ["0.85"]
    assert numbers["nCyclicReplicateRange"] == ["0.80"]


# ------------------------------------------------------------ second round
def test_every_penalty_is_paired_by_fold_against_its_blocks_control():
    shipped = json.loads((TABLES / "paper" / "fold_spread.json").read_text())
    reg = shipped["tables"]["tab:reg_winners"]
    cells = {(a["cell"], a["schedule"]) for a in reg["penalties_minus_control"]}
    assert (FS.NO_PENALTY_CELL, "cyclic") not in cells
    assert len(cells) == reg["penalties_minus_control_summary"]["penalties"]
    for entry in reg["penalties_minus_control"]:
        assert len(entry["minus_control"]["values"]) == 5


def test_the_test_axis_table_carries_its_ceiling_beside_the_gains():
    text = (TABLES / "stopping" / "stopping_test_axis_table.tex").read_text()
    shipped = json.loads((TABLES / "stopping" / "stopping_test_axis.json").read_text())
    every = next(r for r in shipped["stages"] if r["stage"] == "all")
    line = next(l for l in text.splitlines() if l.startswith("\\emph{All arms}"))
    cells = [c.strip() for c in line.rstrip(" \\").split("&")]
    assert len(cells) == 7
    assert cells[5] == "$+%.2f$" % (100 * every["test_oracle_gain"])


def test_no_generated_table_names_another_by_a_literal_number():
    """A number written into a generated file goes stale when a float is added."""
    import re

    for path in (TABLES / "paper" / "fold_spread_tables.tex",
                 TABLES / "stopping" / "stopping_test_axis_table.tex",
                 TABLES / "w_sensitivity_table.tex"):
        assert not re.search(r"Table [A-Z]?\d", path.read_text()), path.name


def test_the_baselines_figure_names_the_shipped_model_as_the_paper_does():
    source = (TOOLS / "paper_figures" / "fig_baselines.py").read_text()
    assert "$g$-$0$" not in source
    assert source.count("$\\\\theta_g$") == 3


# ------------------------------------------- the lead of the best single penalty
def test_the_single_penalty_lead_is_paired_split_by_split():
    """The leader and the runner-up are ranked by mean, then paired per split.

    The paper quotes the best single penalty's lead over the next one as a
    difference of five-split means and states on how many splits it holds, so
    what is pinned is the ranking (composites and the mu = 0 control are not
    single penalties), the five within-split differences, and the count.
    """
    singles = [("fisher_lam8", {"score": [9.0, 8.0, 7.0, 12.0, 9.0]}),
               ("logit_l2_lam0p001", {"score": [8.0, 11.0, 9.0, 11.5, 10.5]}),
               ("param_l2_mu0p0001", {"score": [9.0, 9.0, 9.0, 9.0, 8.0]})]
    lead = FS.single_penalty_lead("parallel", singles)
    assert (lead["leader"], lead["next"]) == ("logit_l2_lam0p001", "fisher_lam8")
    d = lead["leader_minus_next"]
    assert d["values"] == [-1.0, 3.0, 2.0, -0.5, 1.5]
    assert d["won"] == 3 and not d["same_sign_all_folds"]
    assert d["mean"] == pytest.approx(1.0)


def test_composites_and_the_zero_penalty_are_not_single_penalties():
    for cell in ("hybrid_seq_mix0p75", "hybrid_mix0p5", "blend_lam0p1_T0p5_mix0p25",
                 "param_l2_mu0"):
        assert not FS.is_single_penalty(cell), cell
    for cell in ("logit_l2_lam0p001", "fisher_lam8", "param_l2_mu0p0001",
                 "ntd_b0p01_t0p5", "kd_T0p25_a0p9"):
        assert FS.is_single_penalty(cell), cell


def test_the_shipped_lead_macros_are_the_shipped_json_printed():
    """revision_numbers.tex carries the lead and its split count as fold_spread.json has them.

    On the parallel schedule this is the lead the paper quotes: logit matching
    at lambda = 1e-3 over EWC at lambda = 8, 1.09 points on means, ahead on
    three of the five splits, so it is a lead on means and not split-paired.
    """
    payload = json.loads((TABLES / "paper" / "fold_spread.json").read_text())
    numbers = (TABLES / "paper" / "revision_numbers.tex").read_text()
    leads = {x["schedule"]: x for x in
             payload["tables"]["tab:reg_winners"]["single_penalty_lead"]}
    assert set(leads) == {"parallel", "cyclic"}
    for schedule, tag in (("parallel", "Par"), ("cyclic", "Cyc")):
        d = leads[schedule]["leader_minus_next"]
        assert d["won"] == sum(1 for v in d["values"] if v > 0)
        assert d["mean"] == pytest.approx(sum(d["values"]) / len(d["values"]))
        assert "\\newcommand{\\nSingleLeadMean%s}{%.2f}" % (tag, d["mean"]) in numbers
        assert "\\newcommand{\\nSingleLeadWon%s}{%d}" % (tag, d["won"]) in numbers
    par = leads["parallel"]
    assert (par["leader"], par["next"]) == ("logit_l2_lam0p001", "fisher_lam8")
    assert "%.2f" % par["leader_minus_next"]["mean"] == "1.09"
    assert par["leader_minus_next"]["won"] == 3
    # The rows it pairs are the rows the table prints, value for value.
    rows = {(r["cell"], r["schedule"]): r["columns"]["score"]["values"]
            for r in payload["tables"]["tab:reg_winners"]["rows"]}
    lead = rows[("logit_l2_lam0p001", "parallel")]
    nxt = rows[("fisher_lam8", "parallel")]
    assert par["leader_minus_next"]["values"] == pytest.approx(
        [a - b for a, b in zip(lead, nxt)], abs=1e-12)
