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

import pandas as pd

from spacr import annotation, gene_facts


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
