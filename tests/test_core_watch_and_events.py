"""Table-driven tests of the folder-watch and smart-microscope helpers in ``spacr.core``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the voxel size an ND2 handle declares
(``_watch_nd2_voxel``); the volumetric recipe a Z watch of vendor stacks is
given (``_watch_vendor_recipe``); the measure snapshot a Z watch hands its
analysis when the user set no voxel size (``_watch_folder_and_analyse``); and
the objects a field nominates for re-imaging (``_microscope_events``).  ND2
handles are small stand-ins exposing only what the reader touches; every other
input is a real file under ``tmp_path``.
"""
from __future__ import annotations

import json
import re
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import core
from tests.test_watch_folder_and_analyse import MASK, MEASURE
from tests.test_watch_nested_f548 import _fast


def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    folder = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    folder.mkdir()
    return folder


# ---------------------------------------------------------------------------
# _watch_nd2_voxel
# ---------------------------------------------------------------------------

class _Nd2:
    """An ``nd2.ND2File`` stand-in exposing only what the voxel reader uses."""

    def __init__(self, calibrated, voxel, channels=True):
        volume = SimpleNamespace(axesCalibrated=calibrated)
        self.metadata = SimpleNamespace(
            channels=[SimpleNamespace(volume=volume)] if channels else [])
        self._voxel = voxel

    def voxel_size(self):
        return self._voxel


def test_watch_nd2_voxel_declares_only_calibrated_square_axes(subtests):
    """XY needs calibrated, square pixels; Z needs a calibrated axis and >1 plane."""
    cases = [
        ("calibrated volume", _Nd2((True,) * 3, (0.13, 0.13, 0.5)), {"Z": 4},
         {"xy": 0.13, "z": 0.5}),
        ("single plane has no z step", _Nd2((True,) * 3, (0.13, 0.13, 0.5)),
         {"Z": 1}, {"xy": 0.13, "z": None}),
        ("no Z axis has no z step", _Nd2((True,) * 3, (0.13, 0.13, 0.5)), {},
         {"xy": 0.13, "z": None}),
        ("uncalibrated x drops xy", _Nd2((False, True, True), (0.13, 0.13, 0.5)),
         {"Z": 4}, {"xy": None, "z": 0.5}),
        ("uncalibrated z drops z", _Nd2((True, True, False), (0.13, 0.13, 0.5)),
         {"Z": 4}, {"xy": 0.13, "z": None}),
        ("non-square pixels drop xy", _Nd2((True,) * 3, (0.1, 0.3, 0.5)),
         {"Z": 4}, {"xy": None, "z": 0.5}),
        ("no volume metadata declares nothing",
         _Nd2((True,) * 3, (0.13, 0.13, 0.5), channels=False), {"Z": 4},
         {"xy": None, "z": None}),
    ]
    for case_id, handle, sizes, expected in cases:
        with subtests.test(msg=case_id):
            got = core._watch_nd2_voxel(handle, sizes)
            assert set(got) == {"xy", "z"}
            for axis in ("xy", "z"):
                if expected[axis] is None:
                    assert got[axis] is None
                else:
                    assert got[axis] == pytest.approx(expected[axis])


# ---------------------------------------------------------------------------
# _watch_vendor_recipe
# ---------------------------------------------------------------------------

_RECIPE_BASE = {"z_stack": True, "z_segmentation_mode": None, "z_axis": None}
_FILLED = {"z_segmentation_mode": "volumetric", "z_axis": 0}


def test_watch_vendor_recipe_fills_one_existing_folder_only(subtests, tmp_path):
    """One real folder in any spelling gets the volumetric recipe; a non-path gets none."""
    folder = str(tmp_path)
    cases = [
        ("plain string", folder, _FILLED),
        ("one-folder list", [folder], _FILLED),
        ("repr of a one-folder list", repr([folder]), _FILLED),
        ("an integer", 123, {}),
        ("a float", 4.5, {}),
        ("a mapping", {"folder": folder}, {}),
    ]
    for case_id, src, filled in cases:
        with subtests.test(msg=case_id):
            settings = {**_RECIPE_BASE, "src": src}
            snapshot = dict(settings)
            result, got = core._watch_vendor_recipe(settings)
            assert got == filled
            assert settings == snapshot, "the input is never changed"
            if filled:
                assert result == {**snapshot, **filled}
                assert result["src"] == src
            else:
                assert result is settings


# ---------------------------------------------------------------------------
# _watch_folder_and_analyse
# ---------------------------------------------------------------------------

def test_watch_folder_and_analyse_stages_ome_voxel_sizes_into_the_measure_snapshot(
        subtests, tmp_path):
    """A Z watch with no voxel setting reads it from each file's OME metadata."""
    cases = [("anisotropic", 0.1, 0.4), ("isotropic", 0.2, 0.2)]
    for case_id, xy, z in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            volume = np.full((3, 2, 32, 32), 40, np.uint16)
            volume[:, :, 5:17, 5:17] = 500
            tifffile.imwrite(folder / "plate-1.ome.tif", volume, ome=True,
                             metadata={"axes": "ZCYX", "PhysicalSizeX": xy,
                                       "PhysicalSizeY": xy,
                                       "PhysicalSizeZ": z})
            measure = {k: v for k, v in MEASURE.items() if k != "anisotropy"}
            recipe = folder / "measure.json"
            recipe.write_text(json.dumps(measure), encoding="utf-8")
            settings = {**MASK, "z_stack": True, "save_original_images": False,
                        **_fast(folder), "watch_pipeline": "mask_measure",
                        "watch_measure_settings": str(recipe)}
            received = []

            def analyse(_field_dir, field_settings):
                received.append(field_settings)
                raise RuntimeError("stop after staging")

            result = core._watch_folder_and_analyse(settings, analyse)
            assert result["failed"] == ["plate1"]
            assert len(received) == 1
            snapshot = received[0]["watch_measure_snapshot"]
            assert (snapshot["voxel_size_z_um"],
                    snapshot["voxel_size_xy_um"]) == pytest.approx((z, xy))
            ledger = json.loads((folder / "spacr_watch" / "watch_ledger.json")
                                .read_text(encoding="utf-8"))["fields"]
            assert ledger["plate1"]["voxel_size"]["source"] == {
                "z": "OME-TIFF metadata", "xy": "OME-TIFF metadata"}


# ---------------------------------------------------------------------------
# _microscope_events
# ---------------------------------------------------------------------------

def _field(folder, *, db=True, stack=True, table=None):
    field = folder / "field"
    (field / "merged").mkdir(parents=True)
    if stack:
        np.save(field / "merged" / "plate1_A01_1.npy", np.zeros((8, 10, 3)))
    if db:
        (field / "measurements").mkdir()
        with sqlite3.connect(field / "measurements" / "measurements.db") as con:
            con.execute("CREATE TABLE settings (k TEXT)")
            if table is not None:
                table.to_sql("cell", con, index=False)
    return str(field)


def test_microscope_events_nominate_located_objects_or_explain_why_not(
        subtests, tmp_path):
    """Events need measurements, a stack and centroids; a bad query is named."""
    plain = pd.DataFrame({"object_label": [1, 2], "centroid-0": [2.0, None],
                          "centroid-1": [3.0, 4.0]})
    cases = [
        ("no measurements", dict(db=False), {}, ValueError,
         "microscope_feedback needs measurements: set watch_pipeline to "
         "'mask_measure'."),
        ("no merged stack", dict(stack=False), {}, ValueError,
         "The field has no merged stack to size events by."),
        ("no event table yet", {}, {}, None, ([], (8, 10))),
        ("no centroid columns",
         dict(table=pd.DataFrame({"object_label": [1], "area": [3.0]})), {},
         ValueError, "The cell table has no centroid columns."),
        ("plain centroids, missing ones skipped", dict(table=plain), {}, None,
         ([{"object": "1", "pixel": [2.0, 3.0]}], (8, 10))),
        ("a failing query", dict(table=plain),
         {"microscope_event_query": "no_such_column > 1"}, ValueError,
         "microscope_event_query 'no_such_column > 1' failed on the cell "
         "table: "),
    ]
    for case_id, field_kw, extra, error, expected in cases:
        with subtests.test(msg=case_id):
            field = _field(_case_dir(tmp_path, case_id), **field_kw)
            settings = {"microscope_event_table": "cell", **extra}
            if error is None:
                assert core._microscope_events(field, settings) == expected
            else:
                with pytest.raises(error) as excinfo:
                    core._microscope_events(field, settings)
                assert str(excinfo.value).startswith(expected)
