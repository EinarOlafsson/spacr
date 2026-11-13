"""Narrow coverage for failure-side branches of :mod:`spacr.io`.

Every test drives a path a user only meets when something has already gone
wrong: a notebook import that is mid-initialisation, a plate folder holding no
images, a preloader thread that will not die, a merge failure that is not a
duplicate key, an unreadable resume sidecar. Those paths rot silently, because
a green pipeline never walks them. All CPU-only and offline.
"""
from __future__ import annotations

import importlib.util
import sys

import pandas as pd
import pytest

import spacr.io as IO


# module import: the IPython.display fallback

def test_importing_io_survives_a_half_initialised_ipython(monkeypatch):
    """spacr.io must import even when ``IPython.display`` cannot be imported.

    ``import IPython.display`` can fail while IPython is being initialised by
    another thread; the GUI imports spacr.io from a worker thread, so failing
    there takes the application down before any window appears. The module
    answers with a no-op ``display``. This loads the same file again with the
    import poisoned and checks the fallback is installed and callable.
    """
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    name = "spacr._io_ipython_fallback_probe"
    spec = importlib.util.spec_from_file_location(name, IO.__file__)
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, probe)
    spec.loader.exec_module(probe)

    # The real IPython.display.display is not what we got.
    assert probe.display is not IO.display
    # ... and the stand-in swallows any call shape without raising.
    assert probe.display("anything", extra=1) is None
    # The rest of the module still loaded.
    assert callable(probe.preprocess_img_data)


# migrate_unescaped_plate_names


# spacrDataLoader.cleanup


# _merge_channels: the modern layout has no per-channel folders


# _concatenate_channel: the timelapse branch reports its own failures


# preprocess_img_data: the "nothing was produced" diagnosis


# _merge_with_cardinality

def test_a_non_cardinality_merge_error_is_not_relabelled():
    """Merge failures that are not cardinality violations keep pandas' message.

    The wrapper turns "MergeError" into a sentence naming the two tables and
    the duplicated keys. Dressing up every MergeError that way would
    misdiagnose a mis-specified join as duplicated data and send the user off
    de-duplicating a table that is perfectly fine.
    """
    from spacr.io import _merge_with_cardinality, MergeCardinalityError

    left = pd.DataFrame({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    right = pd.DataFrame({"j": [1, 2, 3], "w": [1.0, 2.0, 3.0]})

    # Nothing to join on at all: a MergeError with no duplicated key behind it.
    with pytest.raises(pd.errors.MergeError) as excinfo:
        _merge_with_cardinality(left, right, validate="one_to_one",
                                left_name="cell table",
                                right_name="nucleus table")

    assert not isinstance(excinfo.value, MergeCardinalityError)
    assert "No common columns" in str(excinfo.value)

    # The same wrapper WITH a real duplicate is relabelled, so the branch
    # above is a genuine discrimination and not simply a dead wrapper.
    dup = pd.DataFrame({"k": [1, 1], "w": [1.0, 2.0]})
    other = pd.DataFrame({"k": [1, 2], "z": [1.0, 2.0]})
    with pytest.raises(MergeCardinalityError) as dup_info:
        _merge_with_cardinality(dup, other, on="k", validate="one_to_one",
                                left_name="cell table",
                                right_name="nucleus table")
    assert "cell table has duplicated" in str(dup_info.value)


# _load_and_concatenate_arrays


# _read_and_merge_data


