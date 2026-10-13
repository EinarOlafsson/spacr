"""Which files an illumination estimate is built from, and what it ignores.

A merged folder holds more than fields: sidecars, logs, and whatever the user
left there. Estimating an illumination field from a non-array would fail at
``np.load`` deep inside the estimator, so the filter is what keeps the failure
out of the maths -- and the refusal above it is what keeps a wrong ``src`` from
producing an empty model that silently corrects nothing.
"""
from __future__ import annotations

import numpy as np
import pytest


def _merged(tmp_path, names):
    merged = tmp_path / "merged"
    merged.mkdir()
    for name in names:
        if name.endswith(".npy"):
            np.save(merged / name, np.ones((4, 4), dtype=np.uint16))
        else:
            (merged / name).write_text("not an array")
    return merged


def test_only_npy_fields_are_gathered(tmp_path):
    """Arc 492 -> 491: the loop passes over everything else.

    channel_order.json sits in every merged folder spaCR writes, so this is
    not a hypothetical stray file -- it is guaranteed to be there.
    """
    from spacr.illumination import _merged_files

    merged = _merged(tmp_path, ["plate1_A01_F001.npy", "plate1_A02_F001.npy",
                                "channel_order.json", "notes.txt",
                                "plate1_A03_F001.npy.bak"])

    grouped = _merged_files(str(merged))

    gathered = [p for paths in grouped.values() for p in paths]
    assert len(gathered) == 2
    assert all(p.endswith(".npy") for p in gathered)


def test_fields_are_grouped_by_plate(tmp_path):
    """The grouping the estimate depends on: one field per plate, not per run.

    An illumination field is a property of the optics for THAT plate, so
    pooling two plates into one estimate would correct both by the average of
    their two illuminations.
    """
    from spacr.illumination import _merged_files

    merged = _merged(tmp_path, ["plate1_A01_F001.npy", "plate1_A02_F001.npy",
                                "plate2_A01_F001.npy"])

    grouped = _merged_files(str(merged))

    assert set(grouped) == {"plate1", "plate2"}
    assert len(grouped["plate1"]) == 2
    assert len(grouped["plate2"]) == 1


def test_a_folder_with_no_arrays_gathers_nothing(tmp_path):
    """Every iteration skipping, which is a folder pointed at the wrong place.

    Empty rather than an error: the caller decides what an empty estimate
    means, and it reports it rather than raising here.
    """
    from spacr.illumination import _merged_files

    merged = _merged(tmp_path, ["channel_order.json", "notes.txt"])

    assert _merged_files(str(merged)) == {}


def test_a_src_that_is_not_a_folder_is_refused_with_the_setting_named(tmp_path):
    """The raise above the loop, which names the setting to fix.

    "not a folder" without saying which setting points at it is a message the
    user cannot act on, and the estimate is reached from a settings file.
    """
    from spacr.illumination import IlluminationError, _merged_files

    missing = tmp_path / "no_such_folder"

    with pytest.raises(IlluminationError) as excinfo:
        _merged_files(str(missing))

    assert "settings['src']" in str(excinfo.value)
