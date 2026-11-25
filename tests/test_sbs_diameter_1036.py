"""Item 1036: the OPS sequencing nuclei get an expected Cellpose diameter.

``ops_sbs_cellpose_diameter`` (default 11.0, pixels of the full-resolution
cycle DAPI) reaches the Cellpose call of the stitched-well windows and the
live spot preview through ``ops_engine._sbs_diameter``. None is the old
behaviour (no diameter, native scale), and settings files stamped before
schema 973 replay None.
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
from tests.test_ops_objects_resume_958 import (  # noqa: F401
    _CountingModel, _run, _tables, stitched)


class _RecordingModel(_CountingModel):
    """The threshold model, recording the diameter of every eval."""

    def __init__(self):
        super().__init__()
        self.diameters = []

    def eval(self, image, **kwargs):
        """Record ``diameter`` and label the window as the threshold model."""
        self.diameters.append(kwargs.get("diameter"))
        return super().eval(image, **kwargs)


def test_the_resolution_rule():
    assert OPS_DEFAULTS["ops_sbs_cellpose_diameter"] == 11.0
    assert ops_engine._sbs_diameter(ops_defaults({})) == {"diameter": 11.0}
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
    _run(stitched, tmp_path, monkeypatch, model=model)
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


def test_files_before_973_replay_no_diameter():
    assert SETTINGS_SCHEMA_VERSION >= 973
    assert DEFAULT_CHANGES[973] == {"ops_sbs_cellpose_diameter": None}
    known = set(OPS_DEFAULTS)
    old_file = {"plate": "P1"}
    applied = _fold_older_defaults(old_file, 965, known=known)
    assert applied["ops_sbs_cellpose_diameter"] == (None, 973)
    assert ops_engine._sbs_diameter(ops_defaults(old_file)) == {}
    new_file = {"plate": "P1"}
    assert "ops_sbs_cellpose_diameter" not in _fold_older_defaults(new_file, 973, known=known)
    assert ops_engine._sbs_diameter(ops_defaults(new_file)) == {"diameter": 11.0}
    explicit = {"plate": "P1", "ops_sbs_cellpose_diameter": 9.5}
    _fold_older_defaults(explicit, 965, known=known)
    assert ops_engine._sbs_diameter(ops_defaults(explicit)) == {"diameter": 9.5}


def test_the_tooltip():
    text = OPS_TOOLTIPS["ops_sbs_cellpose_diameter"]
    assert len(text) <= 600 and text.endswith("Default 11.0.")
