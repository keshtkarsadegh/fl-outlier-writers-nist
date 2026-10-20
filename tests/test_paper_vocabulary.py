"""
The paper's vocabulary on everything the generators put on a page.

The manuscript fixes one word for each thing (its glossary, Table 1): the
"selected pair", the "most-preserving pair", the "cyclic leader", the
"two-client pair" and the "merged pair", the "KD+EWC composite" named by its
coefficients, the "patience rule", a "combination". The run records and the
code keep older nicknames as identifiers - a cell ``balanced``, a role
``crowned``, a file ``blends.csv`` - and those are fine where they are keys.
What must not happen is a nickname reaching a printed label, legend, caption
or table note, because then the page needs a glossary of its own: the captions
once carried sentences such as "the legend's 'blend' is the composite". So the
printed text of every figure and generated table is held here against the old
words, with identifiers (LaTeX labels and references, macro names, comments)
taken out first.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
FIGURES = TOOLS / "paper_figures"
ARTIFACTS = REPO / "study" / "artifacts"
PAPER = ARTIFACTS / "Digits_study01" / "tables" / "paper"
STOPPING = ARTIFACTS / "Digits_study01" / "tables" / "stopping"

for entry in (REPO / "src", TOOLS, FIGURES):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: The retired words. "cross" only as a noun for the combination stage, not in
#: cross-entropy; "-tuned" only as the composites' old parentage tags, not in
#: "fine-tuned" or "jointly tuned".
OLD = re.compile(r"\b(?:blends?|crown(?:ed|ing)?|balanced|plateau|sibling|double|dual"
                 r"|(?:cyclic|parallel)-tuned)\b|\bcross\b(?!-entropy)", re.IGNORECASE)

#: Identifiers that may legitimately spell an old word: LaTeX labels and
#: references (tab:blends, tab:plateau), macro names (\nBalancedWorst...),
#: and TeX comments, which no reader sees.
IDENTIFIERS = re.compile(r"\\(?:label|ref|eqref)\{[^}]*\}|\\n[A-Z][A-Za-z]*\{\}|(?<!\\)%.*$",
                         re.MULTILINE)


def printed(text: str) -> str:
    return IDENTIFIERS.sub("", text)


def old_words(text: str) -> list:
    return sorted({m.group(0) for m in OLD.finditer(printed(text))})


def test_the_identifier_filter_keeps_words_and_drops_keys():
    assert old_words(r"see Table~\ref{tab:blends} and \nBalancedWorst{} % blend") == []
    assert old_words("the blend, $m{=}0.5$") == ["blend"]
    assert old_words("a plain cross-entropy objective, fine-tuned, jointly tuned") == []
    assert old_words("Table 7 (cross)") == ["cross"]


# ------------------------------------------------------------ the name module
CELLS = ("hybrid_mix0p5", "hybrid_seq_mix0p75", "blend_lam0p1_T0p5_mix0p25",
         "eta_0p95_hybrid_seq_mix0p5", "seq_fedavg_hybrid_mix0p5",
         "ctune_ewc1em05_kd0p001_T2_mix0p75_eta0p95",
         "ctune_ewc1em05_kd0p001_T2_mix0p75",
         "ctunesel_ewc1em05_kd0p001_T2_mix0p75_h2",
         "ctunesel_ewc1em05_kd0p001_T2_mix0p75",
         "winner", "balanced", "sequential", "control", "double", "dual")


@pytest.mark.parametrize("cell", CELLS)
def test_every_name_paper_names_prints_uses_the_papers_words(cell):
    import paper_names

    for how in (paper_names.label, paper_names.short, paper_names.short_mix):
        assert old_words(how(cell)) == [], (cell, how.__name__, how(cell))


def test_a_composite_is_named_by_its_coefficients_with_the_mix_inside():
    """The manuscript's own form: KD+EWC composite ($\\lambda{=}0.0101$, $m{=}0.5$)."""
    import paper_names

    assert paper_names.short_mix("hybrid_seq_mix0p5") == \
        r"KD+EWC composite ($\lambda{=}0.0101$, $m{=}0.5$)"
    assert paper_names.label("hybrid_mix0p5") == \
        r"KD+EWC composite ($\lambda{=}0.111,T{=}0.25,m{=}0.5$)"
    assert paper_names.label("winner") == "Anchor + FedNTD (selected)"
    assert paper_names.label("balanced") == \
        "Server step + KD+EWC composite (most-preserving)"
    assert paper_names.label("sequential") == "Cyclic FedAvg + FedNTD"


# ---------------------------------------------------------------- the figures
def test_no_figure_prints_an_old_word(tmp_path, monkeypatch):
    """Every text artist of every figure - legends, titles, labels, notes - and its caption.

    The scripts are run, not read: a legend is often built from paper_names at
    run time, so only the drawn figure says what reaches the page.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    import figstyle

    monkeypatch.setattr(figstyle, "OUT", str(tmp_path))
    drawn = {}

    def capture(figure, path, *args, **kwargs):
        # Every script ends in Figure.savefig, through figstyle.save or not,
        # so the texts are read at the moment the page would be written.
        drawn[Path(str(path)).stem] = [t.get_text() for t in figure.findobj(Text)
                                       if t.get_text()]

    monkeypatch.setattr(Figure, "savefig", capture)
    for path in sorted(FIGURES.glob("fig_*.py")):
        importlib.import_module(path.stem).main()
        plt.close("all")
    assert len(drawn) == 7, sorted(drawn)
    for stem, texts in drawn.items():
        for text in texts:
            assert old_words(text) == [], (stem, text)
    for caption in sorted(tmp_path.glob("*_caption.txt")):
        assert old_words(caption.read_text()) == [], caption.name
    # The two legends that once needed a caption sentence to decode them.
    assert any("composite" in t for t in drawn["fig_blends"])
    assert any(t.endswith("(selected)") for t in drawn["fig_combo"])


# ------------------------------------------------------------ the tables
def test_no_generated_table_prints_an_old_word(tmp_path, monkeypatch):
    import make_paper_tables

    monkeypatch.setenv("FOA_PAPER_OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["make_paper_tables.py"])
    make_paper_tables.main()
    tables = sorted((tmp_path / "tables").glob("*.tex"))
    assert tables
    for path in tables:
        assert old_words(path.read_text()) == [], path.name


@pytest.mark.parametrize("path", sorted(ARTIFACTS.glob("screen_ranking_*.tex"))
                         + [PAPER / "clean_preservation_table.tex",
                            PAPER / "clean_preservation_numbers.tex",
                            STOPPING / "stopping_best_vs_best.tex"],
                         ids=lambda p: p.name)
def test_no_shipped_table_the_manuscript_inputs_prints_an_old_word(path):
    """The files the manuscript inputs as they ship, comments included.

    These are copied into the manuscript's source directory, so here even a
    comment counts: the check is on the raw text, identifiers aside.
    """
    raw = re.sub(r"\\(?:label|ref|eqref)\{[^}]*\}", "", path.read_text())
    raw = re.sub(r"\\n[A-Z][A-Za-z]*\{", "{", raw)
    assert sorted({m.group(0) for m in OLD.finditer(raw)}) == [], path.name
