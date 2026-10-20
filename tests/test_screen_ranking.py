"""
tools/screen_ranking.py: the quarter-horizon screens' own orderings.

Two halves are tested. The **records half** - that it orders a synthetic screen
by the selection score, marks each method's winner and the control, refuses an
ordering whose winners disagree with the stage record, and carries the
full-horizon re-runs (both halves of each) beside logit matching's screen row.
And the **shipped views** - that the screen winners they mark are the settings
the winners table prints, and that the shipped tables are what the views render
to, so a stale table cannot ship.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for entry in (REPO / "tools", REPO / "tools" / "paper_figures", REPO / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import screen_ranking as SR  # noqa: E402

A0, P0 = 0.80, 1.00

#: (adaptation, preservation) on validation, per family. Score = a + p - 1.8.
SCREEN = {
    "concurrent": {
        "logit_l2_lam0p001": (0.910, 0.995),   # 0.105
        "logit_l2_lam0p003": (0.905, 0.996),   # 0.101
        "param_l2_mu0p0001": (0.910, 0.990),   # 0.100
        "param_l2_mu0": (0.900, 0.990),        # 0.090 - the control
        "logit_l2_lam1": (0.850, 0.999),       # 0.049
    },
    "sequential": {
        "logit_l2_lam0p003": (0.910, 0.995),   # 0.105
        "logit_l2_lam0p001": (0.905, 0.995),   # 0.100
        "param_l2_mu0": (0.905, 0.993),        # 0.098 - control AND winner
        "param_l2_mu0p0001": (0.900, 0.990),   # 0.090
        "logit_l2_lam1": (0.850, 0.999),       # 0.049
    },
}

WINNERS = {
    "param_l2/concurrent": "param_l2_mu0p0001",
    "param_l2/sequential": "param_l2_mu0",
    "logit_l2/concurrent": "logit_l2_lam0p001",
    "logit_l2/sequential": "logit_l2_lam0p003",
}


def _payload(path: Path, family: str, a: float, p: float, test=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ta, tp = test if test is not None else (a - 0.01, p - 0.001)
    (path / "accuracies_0.json").write_text(json.dumps({
        "scenario": family, "agg_method_name": "x",
        "accuracies": [[a, 0.5]] * 25,
        "pool_val_accuracies": [a] * 25,
        "source_val_accuracies": [p] * 25,
        "final_evaluation": {"clients": {"accuracy": ta},
                             "old": {"mean": tp, "sd": 0.0}},
    }))


def _run(root: Path, parent: str, fold: int, families: dict,
         trainer: str = "AnchoredTrainer") -> None:
    for family, (a, p, *test) in families.items():
        _payload(root / f"{parent}_fold{fold}_{trainer}_grid_search"
                 / f"fold{fold}_seed_1" / f"{family}_delta" / f"base_agg_x_{family}",
                 family, a, p, test[0] if test else None)


def _books(root: Path) -> None:
    (root / "g0_perfold_evaluations.json").write_text(json.dumps([{"accuracy": A0}]))
    (root / "g0_evaluations.json").write_text(json.dumps([{"accuracy": P0}]))


@pytest.fixture()
def reg_root(tmp_path):
    _books(tmp_path)
    cells = set(SCREEN["concurrent"])
    for fold in (1, 2):
        for cell in cells:
            _run(tmp_path, f"d01_reg_{cell}", fold,
                 {f: SCREEN[f][cell] for f in SCREEN})
        # The full-horizon re-run of the concurrent logit winner: one task,
        # both schedules, the folder named for the one it was selected for.
        _run(tmp_path, "d01_regfull_concurrent_logit_l2_lam0p001", fold,
             {"concurrent": (0.92, 0.990, (0.93, 0.988)),
              "sequential": (0.93, 0.992, (0.94, 0.990))})
    (tmp_path / "tables" / "paper").mkdir(parents=True)
    (tmp_path / "tables" / "p13_reg_method_winners.json").write_text(
        json.dumps({k: {"winner": v} for k, v in WINNERS.items()}))
    with open(tmp_path / "tables" / "paper" / "reg-winners.csv", "w") as handle:
        handle.write("cell,family\nparam_l2_mu0p0001,parallel\nparam_l2_mu0,cyclic\n"
                     "logit_l2_lam0p001,parallel\nlogit_l2_lam0p003,cyclic\n")
    return tmp_path


def _of(view, schedule):
    return [r for r in view if r["schedule"] == schedule]


def test_every_cell_is_ranked_by_the_selection_score(reg_root):
    view, _ = SR.reg_view(reg_root, "d01")
    parallel = _of(view, "parallel")
    assert [r["cell"] for r in parallel] == [
        "logit_l2_lam0p001", "logit_l2_lam0p003", "param_l2_mu0p0001",
        "param_l2_mu0", "logit_l2_lam1"]
    assert [r["rank"] for r in parallel] == [1, 2, 3, 4, 5]
    assert {r["of"] for r in parallel} == {5}
    assert parallel[0]["score"] == pytest.approx(0.105)


def test_the_winner_and_the_control_are_marked(reg_root):
    view, notes = SR.reg_view(reg_root, "d01")
    parallel, cyclic = _of(view, "parallel"), _of(view, "cyclic")
    assert {r["cell"] for r in parallel if r["selected"]} == {
        "logit_l2_lam0p001", "param_l2_mu0p0001"}
    assert [r["rank"] for r in parallel if r["control"]] == [4]
    # On the cyclic schedule FedProx's own winner IS the control.
    both = [r for r in cyclic if r["control"]]
    assert len(both) == 1 and both[0]["selected"] == 1
    assert all(r["printed"] for r in view if r["selected"])
    assert all("printed in reg-winners" in line for line in notes)


def test_rank_within_the_method_is_carried(reg_root):
    view, _ = SR.reg_view(reg_root, "d01")
    logit = [r for r in _of(view, "parallel") if r["method"] == "logit_l2"]
    assert [(r["method_rank"], r["method_of"]) for r in logit] == [(1, 3), (2, 3), (3, 3)]


def test_an_ordering_that_did_not_select_is_refused(reg_root):
    record = reg_root / "tables" / "p13_reg_method_winners.json"
    wrong = dict(WINNERS, **{"logit_l2/concurrent": "logit_l2_lam0p003"})
    record.write_text(json.dumps({k: {"winner": v} for k, v in wrong.items()}))
    with pytest.raises(SystemExit, match="p13_reg_method_winners"):
        SR.reg_view(reg_root, "d01")


def test_the_logit_view_carries_both_halves_of_the_rerun(reg_root):
    screen, _ = SR.reg_view(reg_root, "d01")
    rows = {(r["cell"], r["schedule"]): r for r in SR.logit_view(reg_root, "d01", screen)}
    chosen = rows[("logit_l2_lam0p001", "parallel")]
    assert chosen["full_source"] == "selected"
    assert chosen["full_val_score"] == pytest.approx(0.92 + 0.990 - 1.8)
    assert chosen["full_test_score"] == pytest.approx(0.93 + 0.988 - 1.8)
    assert rows[("logit_l2_lam0p001", "cyclic")]["full_source"] == "other"
    # Never re-run: the full-horizon columns stay empty, not zero.
    assert rows[("logit_l2_lam0p003", "cyclic")]["full_source"] == ""
    assert rows[("logit_l2_lam1", "parallel")]["full_test_score"] == ""
    assert chosen["screen_rank"] == 1 and chosen["selected"] == 1


def test_the_tables_render_from_the_views_alone(reg_root, tmp_path_factory):
    views = tmp_path_factory.mktemp("views")
    screen, _ = SR.reg_view(reg_root, "d01")
    SR.write_view(views / "screen_ranking_reg.csv", screen, SR.REG_FIELDS)
    SR.write_view(views / "screen_ranking_logit.csv",
                  SR.logit_view(reg_root, "d01", screen), SR.LOGIT_FIELDS)
    reg = SR.read_view("screen_ranking_reg.csv", views)
    text = SR.render_ranking(reg, "parallel", "reg")
    assert "ranks 4th of 5" in text
    assert r"Logit matching ($\lambda{=}10^{-3}$)" + SR.SCREEN_MARK in text
    assert r"\textbf{No penalty (control)}" in text
    assert "logit_l2" not in text.split("\n", 1)[1]   # no run id on the page
    logit = SR.render_logit(SR.read_view("screen_ranking_logit.csv", views))
    assert SR.OTHER_MARK in logit and "--" in logit
    # The screen's marks are its own letters, never a symbol the main tables
    # give another meaning (the selected pair, the selected or test-leading
    # composite, the borrowed control).
    for symbol in (r"\star", r"\dagger", r"\ddagger", r"\S"):
        assert symbol not in text and symbol not in logit, symbol
    assert r"$3{\times}10^{-3}$" in logit


def test_aggregation_controls_are_read_per_schedule(tmp_path):
    _books(tmp_path)
    for fold in (1, 2):
        _run(tmp_path, "d01_agg_eta_1", fold, {"concurrent": (0.91, 0.99)},
             trainer="BaseTrainer")
        _run(tmp_path, "d01_agg_seq_fedavg", fold, {"sequential": (0.90, 0.99)},
             trainer="BaseTrainer")
        _run(tmp_path, "d01_agg_control_fedavg", fold,
             {"concurrent": (0.905, 0.99), "sequential": (0.895, 0.98)},
             trainer="BaseTrainer")
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "p11_agg_method_winners.json").write_text(json.dumps({
        "concurrent/eta": {"winner": "eta_1"},
        "sequential/seq_fedavg": {"winner": "seq_fedavg"}}))
    view, _ = SR.agg_view(tmp_path, "d01")
    cyclic = {r["cell"]: r for r in _of(view, "cyclic")}
    assert cyclic["control_fedavg"]["adaptation"] == pytest.approx(0.895)
    assert cyclic["control_fedavg"]["control"] == 1
    assert cyclic["seq_fedavg"]["selected"] == 1
    assert [r["cell"] for r in _of(view, "parallel")] == ["eta_1", "control_fedavg"]


def test_ordinals():
    assert [SR.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 18, 21, 29, 102)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "18th", "21st",
        "29th", "102nd"]


# --------------------------------------------------------------------------- #
# the shipped views and tables
# --------------------------------------------------------------------------- #
def _shipped(name):
    with open(SR.VIEWS / name, newline="") as handle:
        return list(csv.DictReader(handle))


def test_shipped_screen_winners_are_the_settings_the_winners_table_prints():
    rows = _shipped("screen_ranking_reg.csv")
    printed = {(r["cell"], r["family"]) for r in _shipped("reg-winners.csv")}
    for schedule, _ in SR.SCHEDULES:
        mine = [r for r in rows if r["schedule"] == schedule]
        winners = [r for r in mine if r["selected"] == "1"]
        assert len(mine) == 140 and len(winners) == 7
        assert all((r["cell"], schedule) in printed for r in winners)
        assert sum(r["control"] == "1" for r in mine) == 1


def test_shipped_logit_view_reproduces_the_winners_table():
    winners = {(r["cell"], r["family"]): r for r in _shipped("reg-winners.csv")}
    measured = [r for r in _shipped("screen_ranking_logit.csv")
                if r["full_source"] == "selected"]
    assert len(measured) == 2
    for row in measured:
        assert row["selected"] == "1"
        printed = winners[(row["cell"], row["schedule"])]
        assert float(row["full_test_score"]) == pytest.approx(float(printed["score"]),
                                                              abs=1e-12)


def test_shipped_tables_are_what_the_views_render(tmp_path):
    SR.render(SR.VIEWS, tmp_path, ("reg", "agg"))
    for made in sorted(tmp_path.glob("screen_ranking_*.tex")):
        assert made.read_text() == (SR.OUT / made.name).read_text(), made.name


def test_render_runs_without_the_training_package(tmp_path):
    # -I drops PYTHONPATH, so the training package is reachable only through
    # the imports --render must not make.
    subprocess.run(
        [sys.executable, "-I", "-c",
         "import runpy, sys; sys.argv = sys.argv[1:]; "
         "sys.modules['federated_outlier_adaptation'] = None; "
         "runpy.run_path(sys.argv[0], run_name='__main__')",
         str(REPO / "tools" / "screen_ranking.py"), "--render", "--out", str(tmp_path)],
        capture_output=True, text=True, check=True)
    assert len(list(tmp_path.glob("screen_ranking_*.tex"))) == 5


def test_regenerate_paper_renders_the_screen_tables():
    out = subprocess.run([str(REPO / "tools" / "regenerate_paper.sh"), "--dry-run",
                          "/tmp/paper"], capture_output=True, text=True, check=True)
    assert "python tools/screen_ranking.py --render --out /tmp/paper/tables" in out.stdout
