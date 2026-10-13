"""The four field columns are an object identity, even without ``prcf``.

``png_list`` carries plateID/rowID/columnID/fieldID and never carries the
paste of them, so a crop table read straight out of the database looked to
the Compare panel like a table with no object identity at all -- and every
morphological measurement in the screen became unreachable from it.
"""
from __future__ import annotations

import pandas as pd

from spacr.gene_measurement_compare import object_identity


def test_the_four_field_columns_are_pasted_into_the_prcfo_key():
    frame = pd.DataFrame({
        "plateID": ["plate1", "plate1"],
        "rowID": ["r5", "r5"],
        "columnID": ["c1", "c1"],
        "fieldID": ["f16", "f17"],
        "object_label": [2, 3],
    })

    identity = object_identity(frame)

    assert list(identity) == ["plate1_r5_c1_f16_2", "plate1_r5_c1_f17_3"]


def test_a_png_style_object_label_loses_its_letter_before_the_paste():
    frame = pd.DataFrame({
        "plateID": ["plate1"],
        "rowID": ["r5"],
        "columnID": ["c1"],
        "fieldID": ["f17"],
        "cell_id": ["o2"],
    })

    identity = object_identity(frame)

    assert list(identity) == ["plate1_r5_c1_f17_2"], (
        "the crop table's `o2` names the same object as the table's `2`")


def test_a_field_key_with_no_object_label_is_not_an_identity():
    frame = pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r5"],
        "columnID": ["c1"], "fieldID": ["f17"],
    })

    assert object_identity(frame) is None


def test_an_object_label_with_no_field_key_is_not_an_identity():
    assert object_identity(pd.DataFrame({"object_label": [1, 2]})) is None
