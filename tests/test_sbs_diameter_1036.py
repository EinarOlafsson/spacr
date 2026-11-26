"""Item 1036: the OPS sequencing nuclei get an expected Cellpose diameter.

``ops_sbs_cellpose_diameter`` (pixels of the full-resolution cycle DAPI)
reaches the Cellpose call of the stitched-well windows and the live spot
preview through ``ops_engine._sbs_diameter``. The default is None again
(schema 980: no diameter, native scale); files stamped 973 to 979 replay
11.0 and files stamped below 973 replay None.
"""
from __future__ import annotations

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

from spacr import ops_engine
from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS, ops_defaults
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            _fold_older_defaults)
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.test_ops_objects_resume_958 import (  # noqa: F401
    _CountingModel, _run, _tables, stitched)


class _RecordingModel(_CountingModel):
    """The threshold model, recording the diameter of every eval."""

    def __init__(self):
        super().__init__()
        self.diameters = []

    def eval(self, image, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
        """Record ``diameter`` and label the window as the threshold model.

        ``channel_axis`` is named and checked against the installed
        cellpose (tests/conftest.py), so this double cannot accept an axis
        the real ``CellposeModel.eval`` would reject. The SBS windows are
        2-D DAPI planes, which the OPS call sites leave to Cellpose to
        auto-detect, as the other OPS doubles allow.
        """
        check_cellpose_eval_call(image, channel_axis, require_channel_axis=False)
        if channel_axis is not MISSING_CHANNEL_AXIS:
            kwargs["channel_axis"] = channel_axis
        self.diameters.append(kwargs.get("diameter"))
        return super().eval(image, **kwargs)


def test_the_resolution_rule():
    assert OPS_DEFAULTS["ops_sbs_cellpose_diameter"] is None
    assert ops_engine._sbs_diameter(ops_defaults({})) == {}
    assert ops_engine._sbs_diameter({"ops_sbs_cellpose_diameter": 11.0}) == {"diameter": 11.0}
    assert ops_engine._sbs_diameter({"ops_sbs_cellpose_diameter": None}) == {}
    assert ops_engine._sbs_diameter({"ops_sbs_cellpose_diameter": 0}) == {}
    assert ops_engine._sbs_diameter({"ops_sbs_cellpose_diameter": "13"}) == {"diameter": 13.0}
    assert ops_engine._sbs_diameter({"cellpose_diameter": 15,
                                     "ops_sbs_cellpose_diameter": 11.0}) == {"diameter": 15.0}
    # None is exactly what the base computed for the no-diameter recipe.
    assert (ops_engine._sbs_diameter({"cellpose_diameter": None,
                                      "ops_sbs_cellpose_diameter": None})
            == ops_engine._ops_diameter(None, "cellpose") == {})


def test_the_diameter_reaches_the_cellpose_call(stitched, tmp_path, monkeypatch):
    model = _RecordingModel()
    _run(stitched, tmp_path, monkeypatch, model=model,
         ops_sbs_cellpose_diameter=11.0)
    assert model.diameters and set(model.diameters) == {11.0}


def test_none_passes_no_diameter_and_gives_the_same_tables(
        stitched, tmp_path, monkeypatch):
    none_model = _RecordingModel()
    _run(stitched, tmp_path / "none", monkeypatch, model=none_model,
         ops_sbs_cellpose_diameter=None)
    assert none_model.diameters and set(none_model.diameters) == {None}
    old_model = _RecordingModel()
    _run(stitched, tmp_path / "old", monkeypatch, model=old_model,
         ops_sbs_cellpose_diameter=None, cellpose_diameter=None)
    assert _tables(tmp_path / "none") == _tables(tmp_path / "old")
    assert none_model.diameters == old_model.diameters


def test_files_replay_the_diameter_of_their_era():
    assert SETTINGS_SCHEMA_VERSION >= 980
    assert DEFAULT_CHANGES[973] == {"ops_sbs_cellpose_diameter": None}
    assert DEFAULT_CHANGES[980] == {"ops_sbs_cellpose_diameter": 11.0}
    known = set(OPS_DEFAULTS)
    for version, expected in ((965, None), (972, None), (973, 11.0),
                              (975, 11.0), (979, 11.0)):
        old_file = {"plate": "P1"}
        applied = _fold_older_defaults(old_file, version, known=known)
        assert applied["ops_sbs_cellpose_diameter"][0] == expected, version
        want = {} if expected is None else {"diameter": expected}
        assert ops_engine._sbs_diameter(ops_defaults(old_file)) == want, version
    new_file = {"plate": "P1"}
    assert "ops_sbs_cellpose_diameter" not in _fold_older_defaults(
        new_file, 980, known=known)
    assert ops_engine._sbs_diameter(ops_defaults(new_file)) == {}
    explicit = {"plate": "P1", "ops_sbs_cellpose_diameter": 9.5}
    _fold_older_defaults(explicit, 965, known=known)
    assert ops_engine._sbs_diameter(ops_defaults(explicit)) == {"diameter": 9.5}


def test_the_tooltip():
    text = OPS_TOOLTIPS["ops_sbs_cellpose_diameter"]
    assert len(text) <= 600 and text.endswith("Default None.")
