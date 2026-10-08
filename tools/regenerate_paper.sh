#!/bin/bash
# Regenerate every number, table and figure the manuscript sets, in one go.
#
#     tools/regenerate_paper.sh [--dry-run] <out dir>
#
# Runs, in the order docs/REPRODUCE.md section 6 gives them, the three
# generators and the seven figure scripts under tools/paper_figures/:
#
#     <out dir>/numbers.tex        make_numbers.py
#     <out dir>/tables/*.tex       make_paper_tables.py, make_sensitivity_tables.py
#     <out dir>/figures/*.pdf      fig_problem, fig_aggs, fig_regs, fig_blends,
#                                  fig_combo, fig_extremes, fig_baselines
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
for fig in fig_problem fig_aggs fig_regs fig_blends fig_combo fig_extremes fig_baselines; do
    run env FOA_PAPER_OUT="$out/figures" python "tools/paper_figures/$fig.py"
done

[ $dry -eq 1 ] || echo "numbers.tex and tables/ in $out, figures in $out/figures"
