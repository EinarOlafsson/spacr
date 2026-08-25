"""Gene facts survive the cells and rows that carry no single answer.

Everything on a gene tile comes out of a merged frame, which means the values
are numpy scalars, sometimes arrays, and the rows are whatever the bundled
CSV holds. Three of those cases have no answer to give:

* a cell holding more than one number cannot be reduced to a scalar, so it is
  shown as it is rather than raising inside a tile that is already painting;
* a segment row whose gene id does not name a gene is skipped, because
  attaching its residue coordinates to some other gene would put a signal
  peptide on a protein that has none;
* a gene with nothing bundled prints its reason instead of a blank panel.

The text form is the one a log records and a test reads, so it has to render
the same sections the Qt tile does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import annotation, gene_facts


def test_a_multi_valued_cell_is_shown_rather_than_raising():
    """An array cell has no scalar to unwrap, so the value passes through.

    ``.item()`` on a size-2 array raises; letting that out would take down the
    tile that is painting the gene rather than showing one odd cell.
    """
    cell = np.array([1, 2])

    assert gene_facts._plain(cell) is cell


def test_a_cell_whose_unwrap_is_broken_is_shown_as_it_is():
    """An object with an ``item`` that raises falls back to the object."""

    class _Odd:
        def item(self):
            raise AttributeError("no scalar here")

    cell = _Odd()

    assert gene_facts._plain(cell) is cell


def test_a_numpy_scalar_is_unwrapped():
    """The normal path still turns a numpy scalar into a python one."""
    assert gene_facts._plain(np.float64(2.5)) == 2.5
    assert isinstance(gene_facts._plain(np.bool_(True)), bool)


def test_the_text_form_renders_every_section_with_a_blank_line_between():
    """Two sections are separated, headed, and carry their label/value rows.

    The text form is what a log records; a run of unseparated rows makes the
    grouping the tile shows unrecoverable from the record.
    """
    facts = gene_facts.GeneFacts(
        gene="224750",
        values={"gene_name": "SRS40F", "topology": "SP"},
    )

    text = facts.to_text()

    assert "IDENTITY" in text
    assert "MEMBRANE TOPOLOGY" in text
    assert "  gene name: SRS40F" in text
    assert "  DeepTMHMM class: SP" in text
    assert "\n\nMEMBRANE TOPOLOGY" in text


def test_the_text_form_of_an_empty_record_gives_its_reason():
    """Nothing known prints the reason, never a blank string."""
    facts = gene_facts.GeneFacts(gene="1", reason="no row in the annotation")

    assert facts.to_text() == "no row in the annotation"


def test_the_text_form_of_a_reasonless_empty_record_still_says_something():
    """Even with no reason recorded the panel is never blank."""
    assert gene_facts.GeneFacts().to_text() == \
        "Nothing is known about this gene."


def test_a_segment_row_whose_id_names_no_gene_is_skipped(monkeypatch):
    """A malformed ``gene_nr`` contributes no segments to any gene.

    Falling through would attach the row's residue coordinates to whichever
    gene the loop was building, which is a claim about a protein made from
    another protein's file row.
    """
    frame = pd.DataFrame({
        "gene_nr": ["", "224750"],
        "sp_start": [1, 1],
        "sp_end": [20, 39],
        "sp_length": [20, 39],
    })
    monkeypatch.setattr(annotation, "supplementary", lambda: frame,
                        raising=False)
    gene_facts._segment_index.cache_clear()
    try:
        index = gene_facts._segment_index()

        assert set(index) == {"224750"}
        assert len(index["224750"]) == 1
        assert index["224750"][0].start == 1
        assert index["224750"][0].end == 39
    finally:
        gene_facts._segment_index.cache_clear()


def test_no_supplementary_table_means_no_segments(monkeypatch):
    """A missing DeepTMHMM table yields an empty index, not an error."""
    monkeypatch.setattr(annotation, "supplementary", lambda: None,
                        raising=False)
    gene_facts._segment_index.cache_clear()
    try:
        assert gene_facts._segment_index() == {}
    finally:
        gene_facts._segment_index.cache_clear()
