"""The extraction preview counts what a drop would produce, without pixels.

The planners are driven over real folders of real (tiny) TIFFs so the row
count and the canonical names are the ones the extraction would actually
write, rather than the ones a stubbed walker was told to return.
"""
from __future__ import annotations

from spacr.qt import ingest_preview as ip


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# folders
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# rows in, mappings out
# --------------------------------------------------------------------------

def test_an_edited_row_round_trips_back_into_a_mapping():
    rows = [{"original": "/data/x.tif", "plate": "plate2",
             "well": "plate2_C03", "field": "2", "channel": "3",
             "time": "4", "canonical": "plate2_C03_T0004F002L01C03.tif"}]
    mapping, = ip.rows_to_mappings(rows)
    assert mapping.plate == "plate2"
    assert (mapping.field, mapping.channel, mapping.time) == (2, 3, 4)
    assert ip.mapping_to_row(mapping) == {
        "original": "/data/x.tif", "plate": "plate2", "well": "plate2_C03",
        "field": 2, "channel": 3, "time": 4,
        "canonical": "plate2_C03_T0004F002L01C03.tif"}


def test_a_blank_index_becomes_one_rather_than_zero():
    """An emptied table cell must not produce `F000`, which no reader of the
    Yokogawa convention expects."""
    mapping, = ip.rows_to_mappings([{"original": "x", "field": "",
                                     "channel": 0, "time": None}])
    assert (mapping.field, mapping.channel, mapping.time) == (1, 1, 1)


# --------------------------------------------------------------------------
# the one-line summary
# --------------------------------------------------------------------------
