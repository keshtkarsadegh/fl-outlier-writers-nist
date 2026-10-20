#!/bin/bash
# Regenerate every number, table and figure the manuscript sets, in one go.
#
#     tools/regenerate_paper.sh [--dry-run] <out dir>
#
# Runs, in the order docs/REPRODUCE.md section 6 gives them, the three
# generators and the seven figure scripts under tools/paper_figures/, and the
# screen-ranking tables of tools/screen_ranking.py:
#
#     <out dir>/numbers.tex        make_numbers.py
#     <out dir>/tables/*.tex       make_paper_tables.py, make_sensitivity_tables.py
#     <out dir>/tables/screen_ranking_*.tex
#                                  screen_ranking.py --render
#     <out dir>/figures/*.pdf      fig_problem, fig_aggs, fig_regs, fig_blends,
#                                  fig_combo, fig_extremes, fig_baselines
#
# and, when FOA_STUDY_DIR names an assembled records tree (docs/REPRODUCE.md
# section 10), the four revision tools that read the records themselves, the
# one that reads what the second of them wrote, and the one that reads both:
#
#     <out dir>/revision/          fold_spread.py, stopping_test_axis.py,
#                                  sensitivity_counts.py (tables, JSON and the
#                                  prose macros of revision_numbers.tex),
#                                  clean_preservation.py (the clean-row table
#                                  and its macros), stopping_best_vs_best.py
#                                  (best rule against best penalty, macros),
#                                  split_evidence.py (the split-level evidence
#                                  behind the headline sentences)
#
# It adds nothing to them. Each script is run exactly as REPRODUCE documents
# it, with FOA_PAPER_OUT pointing at <out dir>, so the per-script commands there
# stay the reference and this is only the list typed once. --dry-run prints the
# commands and runs none of them.
#
# THE INPUT IS THE CSV VIEWS, NOT THE RECORDS. Every script here renders from
# views and computes nothing, and the views ship in the clone under
# study/artifacts/Digits_study01/tables/, so this runs with no dataset, no GPU
# and no release asset. The records asset is what rebuilds the views (the
# export and report tools of REPRODUCE section 6, against the tree of section
# 10); point FOA_PAPER_DATA at rebuilt views and this renders from those
# instead, the same variable each script reads.
#
# It never writes into the clone: <out dir> is required, because both
# generators refuse a source checkout without FOA_PAPER_OUT and the figures
# would otherwise land beside their scripts. Run it from the activated venv of
# the README's Quickstart; it calls `python`, as every command in docs/ does.
set -euo pipefail

usage() {
    echo "usage: tools/regenerate_paper.sh [--dry-run] <out dir>" >&2
    exit 2
}

dry=0
if [ "${1:-}" = "--dry-run" ]; then
    dry=1
    shift
fi
[ $# -eq 1 ] || usage
case "$1" in -*) usage ;; esac

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$1" in /*) out=$1 ;; *) out=$PWD/$1 ;; esac  # the scripts run from $repo

# The views every script reads. The scripts find the shipped ones on their own;
# this only refuses early, with somewhere to look, when they are not there.
if [ -n "${FOA_PAPER_DATA:-}" ]; then
    IFS=: read -r -a views <<< "$FOA_PAPER_DATA"
else
    t=$repo/study/artifacts/Digits_study01/tables
    views=("$t" "$t/paper" "$t/paper_figures" "$t/stopping")
fi
for d in "${views[@]}"; do
    [ -z "$d" ] || [ -d "$d" ] && continue
    echo "no views directory $d" >&2
    echo "the views ship in study/artifacts/Digits_study01/tables/; see docs/REPRODUCE.md" >&2
    echo "section 6 for what writes each one and section 10 for the records they come from" >&2
    exit 1
done

run() {  # print the command as REPRODUCE writes it, then run it unless --dry-run
    echo "+ $*"
    [ $dry -eq 1 ] || "$@"
}

cd "$repo"
[ $dry -eq 1 ] || mkdir -p "$out/figures"
run env FOA_PAPER_OUT="$out" python tools/paper_figures/make_numbers.py
run env FOA_PAPER_OUT="$out" python tools/paper_figures/make_paper_tables.py
run env FOA_PAPER_OUT="$out" python tools/paper_figures/make_sensitivity_tables.py
# --render reads the screen-ranking views only, never the records, so this step
# keeps the promise above; rebuilding those views is screen_ranking.py --root.
run python tools/screen_ranking.py --render --out "$out/tables"
for fig in fig_problem fig_aggs fig_regs fig_blends fig_combo fig_extremes fig_baselines; do
    run env FOA_PAPER_OUT="$out/figures" python "tools/paper_figures/$fig.py"
done

# THESE READ THE RECORDS, NOT THE VIEWS. The fold spread, the test-axis
# reading of the stopping rule, the sensitivity counts and the clean-row
# preservation are all quantities the views fold away - per-fold values,
# per-round series, the screen's selection groups, per-source-fold accuracies -
# so they need the tree of REPRODUCE section 10 and the package environment,
# which the renderers above do not. They are held to the tables just written:
# fold_spread.py and stopping_test_axis.py refuse, and write nothing, if a mean
# they recompute does not round to the number those tables print, or if a macro
# they emit differs from the shipped revision_numbers.tex; clean_preservation.py
# fails if any printed row of the eight tables it reads matches no arm or
# misprints a number, and its two outputs are then compared with the shipped
# copies; stopping_best_vs_best.py reads the test-axis JSON just written and
# refuses, writing nothing, if a macro differs from its shipped file. Without
# FOA_STUDY_DIR they are skipped and the line below says so.
if [ -n "${FOA_STUDY_DIR:-}" ]; then
    tabs=$out/tables
    # The prose macros of revision_numbers.tex are held to the shipped copy:
    # a recomputed number that no longer equals the one the paper cites stops
    # the run instead of quietly replacing it.
    shipped=study/artifacts/Digits_study01/tables/paper/revision_numbers.tex
    run python tools/fold_spread.py --root "$FOA_STUDY_DIR" --out "$out/revision" \
        --tex "$tabs/agg_winners.tex" "$tabs/reg_winners.tex" "$tabs/combos.tex" \
              "$tabs/scaling.tex" "$tabs/selection_axis.tex" \
              "$tabs/selection_axis_cross.tex" "$tabs/fairness.tex" \
        --macros "$shipped"
    run python tools/stopping_test_axis.py --root "$FOA_STUDY_DIR" \
        --tex "$tabs/plateau.tex" --out "$out/revision" --macros "$shipped"
    run python tools/sensitivity_counts.py --root "$FOA_STUDY_DIR" --out "$out/revision"
    # The clean-row reading is held to the tables the manuscript prints: the
    # seven written above and the schedule table, which is set by hand and so
    # ships as printed. Its table and macros are then compared byte for byte
    # with the shipped ones, so a number that moved stops the run here.
    paper=study/artifacts/Digits_study01/tables/paper
    run python tools/clean_preservation.py --root "$FOA_STUDY_DIR" \
        --tex "$tabs/references.tex" "$tabs/agg_winners.tex" "$tabs/reg_winners.tex" \
              "$tabs/combos.tex" "$tabs/scaling.tex" "$tabs/selection_axis.tex" \
              "$tabs/selection_axis_cross.tex" "$paper/schedule_table_as_printed.tex" \
        --json "$out/revision/clean_preservation.json" \
        --table "$out/revision/clean_preservation_table.tex" \
        --numbers "$out/revision/clean_preservation_numbers.tex"
    run cmp "$out/revision/clean_preservation_table.tex" "$paper/clean_preservation_table.tex"
    run cmp "$out/revision/clean_preservation_numbers.tex" "$paper/clean_preservation_numbers.tex"
    run python tools/stopping_best_vs_best.py --json "$out/revision/stopping_test_axis.json" \
        --out "$out/revision/stopping_best_vs_best.tex" \
        --macros study/artifacts/Digits_study01/tables/stopping/stopping_best_vs_best.tex
    # The split-level evidence behind the headline sentences reads the records
    # and the test-axis JSON just written; every mean it prints is held to the
    # macro the prose quotes it by, in the numbers.tex written above and the
    # shipped revision_numbers.tex, and its table is then compared byte for
    # byte with the shipped copy the manuscript sets.
    run python tools/split_evidence.py --root "$FOA_STUDY_DIR" \
        --stopping-json "$out/revision/stopping_test_axis.json" \
        --macros "$out/numbers.tex" "$shipped" \
        --json "$out/revision/split_evidence.json" \
        --table "$out/revision/split_evidence_table.tex"
    run cmp "$out/revision/split_evidence_table.tex" "$paper/split_evidence_table.tex"
else
    echo "FOA_STUDY_DIR is not set: skipped fold_spread.py, stopping_test_axis.py,"
    echo "sensitivity_counts.py, clean_preservation.py, stopping_best_vs_best.py and"
    echo "split_evidence.py, which read the run records (docs/REPRODUCE.md section 10)"
fi

[ $dry -eq 1 ] || echo "numbers.tex and tables/ in $out, figures in $out/figures"
