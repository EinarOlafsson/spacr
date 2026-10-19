"""The alpha ``export_vvvv`` setting of Mask Generation and Measure (N685).

With the setting on, both pipelines write the same vvvv gamma export Make
Masks writes (``spacr.qt.mask_engine._export_vvvv``) for every field they
process, one folder per field and object type; with it off they write
nothing. Measure's objects table carries its own per-object measurements
after the export's columns. Segmentation is stubbed; the merge, the
measurement and the export run for real on small synthetic fields.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from spacr import tabular                                      # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "vvvv"
SAVED = "spacr_field_01_saved_by_vvvv_7_4.vl"
P = "{property}"
FILES = ("{s}_labels.png", "{s}_outlines.png", "{s}_image.png",
         "{s}_objects.csv", "manifest.json", "{s}.vl", "README.txt")
BASE = ["label", "centroid_x", "centroid_y", "area", "bbox_x0", "bbox_y0",
        "bbox_x1", "bbox_y1"]


def _parse(path):
    return ElementTree.fromstring(
        Path(path).read_text(encoding="utf-8-sig").encode("utf-8"))


def _node_name(node):
    ref = node.find(P + "NodeReference")
    return [choice.get("Name") for choice in ref.findall("Choice")][-1]


def _vvvv_nodes():
    root = _parse(DATA / SAVED)
    known = {}
    for node in root.iter("Node"):
        if node.find(P + "NodeReference") is None \
                or node.get("Name") == "Application":
            continue
        known.setdefault(_node_name(node), set()).update(
            pin.get("Name") for pin in node.findall("Pin"))
    return known


def _check_folder(folder, stem, *, has_class=False):
    """The N685 file set, and a project shaped like the one vvvv saved."""
    folder = Path(folder)
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        name.format(s=stem) for name in FILES)
    root = _parse(folder / f"{stem}.vl")
    paths = {pad.get("Value") for pad in root.iter("Pad")
             if pad.find(P + "TypeAnnotation/Choice").get("Name") == "Path"}
    assert paths == {f"{stem}_image.png", f"{stem}_outlines.png",
                     f"{stem}_objects.csv", "manifest.json"}
    header = list(tabular.read_table(folder / f"{stem}_objects.csv",
                                     canonicalise=False).columns)
    region = [node for node in root.iter("Node")
              if node.find(P + "NodeReference") is not None
              and _node_name(node) == "ForEach"][0]
    indices = sorted(int(pad.get("Value")) for pad in region.iter("Pad")
                     if pad.get("Comment") == "Index")
    expected = [0, 1, 2] + ([header.index("class")] if has_class else [])
    assert indices == expected
    assert [header[i] for i in (0, 1, 2)] == BASE[:3]
    known = _vvvv_nodes()
    for node in root.iter("Node"):
        if node.get("Name") == "Application":
            continue
        name = _node_name(node)
        assert name in known, name
        assert {pin.get("Name") for pin in node.findall("Pin")} <= known[name]
    ends = {element.get("Id") for element in root.iter() if element.get("Id")}
    for link in root.iter("Link"):
        source, sink = link.get("Ids").split(",")
        assert source in ends and sink in ends
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert manifest["format"] == "spacr-vvvv-export"
    assert manifest["files"]["project"] == f"{stem}.vl"
    return header, manifest


def _field(shape=(48, 64)):
    cells = np.zeros(shape, np.uint16)
    cells[4:20, 6:30] = 1
    cells[26:44, 36:60] = 2
    nuclei = np.zeros(shape, np.uint16)
    nuclei[8:14, 10:18] = 1
    nuclei[30:38, 40:50] = 2
    rng = np.random.default_rng(3)
    image = rng.integers(50, 400, shape + (2,)).astype(np.uint16)
    image[..., 0][cells == 1] += 1000
    return image, cells, nuclei


# ---------------------------------------------------------------- Measure ---

def _measure_settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "cytoplasm": False, "save_png": False,
        "save_arrays": False, "plot": False, "verbose": False, "n_jobs": 1,
    })
    settings.update(over)
    return settings


def _measure_plate(root):
    merged = root / "merged"
    merged.mkdir(parents=True)
    image, cells, nuclei = _field()
    np.save(merged / "plate1_A01_1.npy",
            np.dstack([image, cells, nuclei]).astype(np.uint16))
    return merged, image, cells


def test_export_vvvv_is_off_by_default_in_both_modules():
    from spacr.settings import (get_measure_crop_settings,
                                set_default_settings_preprocess_generate_masks)

    assert get_measure_crop_settings({})["export_vvvv"] is False
    assert set_default_settings_preprocess_generate_masks(
        {"src": "x"})["export_vvvv"] is False


def test_measure_writes_nothing_with_the_setting_off(tmp_path):
    from spacr.measure import measure_crop

    merged, _image, _cells = _measure_plate(tmp_path)
    measure_crop(_measure_settings(merged))
    assert not (tmp_path / "vvvv").exists()


def test_measure_exports_every_object_with_its_measurements(tmp_path):
    from spacr.measure import measure_crop

    merged, image, cells = _measure_plate(tmp_path)
    measure_crop(_measure_settings(merged, export_vvvv=True,
                                   voxel_size_xy_um=0.5))
    root = tmp_path / "vvvv" / "measure"
    assert sorted(p.name for p in root.iterdir()) == [
        "cell", "cytoplasm", "nucleus"]
    folder = root / "cell" / "plate1_A01_1"
    header, manifest = _check_folder(folder, "plate1_A01_1")
    assert header[:8] == BASE
    assert header[8:10] == ["mean_intensity_channel_0",
                            "mean_intensity_channel_1"]
    assert {"cell_area", "cell_channel_0_mean_intensity"} <= set(header)
    assert len(header) == len(set(header))
    table = tabular.read_table(folder / "plate1_A01_1_objects.csv",
                               canonicalise=False).set_index("label")
    assert list(table.index) == [1, 2]
    assert table.loc[1, "area"] == 16 * 24
    assert table.loc[1, "cell_area"] == table.loc[1, "area"]
    assert table.loc[1, "centroid_x"] == pytest.approx(18.0)
    assert table.loc[1, "centroid_y"] == pytest.approx(12.0)
    assert (table.loc[1, ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]]
            .tolist() == [6, 4, 30, 20])
    expected = image[..., 0][cells == 1].mean()
    assert table.loc[1, "mean_intensity_channel_0"] == pytest.approx(expected)
    assert table.loc[1, "cell_channel_0_mean_intensity"] == pytest.approx(
        expected)
    labels = imageio.imread(folder / "plate1_A01_1_labels.png")
    assert labels.dtype == np.uint16 and np.array_equal(labels, cells)
    assert manifest["pixel_size"] == {"x": 0.5, "y": 0.5, "unit": "µm"}
    assert manifest["objects"] == 2
    assert "spaCR Measure" in (folder / "README.txt").read_text("utf-8")
    nucleus = root / "nucleus" / "plate1_A01_1"
    header, _manifest = _check_folder(nucleus, "plate1_A01_1")
    assert "nucleus_area" in header


def test_measure_without_saved_measurements_writes_the_base_table(tmp_path):
    from spacr.measure import measure_crop

    merged, _image, _cells = _measure_plate(tmp_path)
    measure_crop(_measure_settings(merged, export_vvvv=True,
                                   save_measurements=False,
                                   nucleus_mask_dim=None))
    folder = tmp_path / "vvvv" / "measure" / "cell" / "plate1_A01_1"
    header, manifest = _check_folder(folder, "plate1_A01_1")
    assert header == BASE + ["mean_intensity_channel_0",
                             "mean_intensity_channel_1"]
    assert manifest["pixel_size"] is None


def test_measured_class_and_score_reach_the_table_and_project(tmp_path):
    from spacr.measure import _export_vvvv_measured_field

    image, cells, nuclei = _field()
    morphology = {"cell": pd.DataFrame({
        "label": [1, 2], "cell_area": [384, 432], "class": ["a", "b"],
        "score": [0.9, 0.2]})}
    intensities = {"cell": pd.DataFrame({
        "label": [1, 2], "cell_area": [0, 0], "x": [5.0, 6.0]})}
    written = _export_vvvv_measured_field(
        str(tmp_path), "f.npy", {"src": str(tmp_path / "merged"),
                                 "cell_mask_dim": 2, "nucleus_mask_dim": 3,
                                 "pathogen_mask_dim": None},
        False, image, {"cell": cells, "nucleus": nuclei,
                       "pathogen": np.zeros_like(cells)},
        morphology, intensities)
    assert [Path(p).parent.name for p in written] == ["cell", "nucleus"]
    header, _manifest = _check_folder(written[0], "f", has_class=True)
    table = tabular.read_table(Path(written[0]) / "f_objects.csv",
                               canonicalise=False)
    assert table["class"].tolist() == ["a", "b"]
    assert table["score"].tolist() == [0.9, 0.2]
    assert table["cell_area"].tolist() == [384, 432]
    assert table["x"].tolist() == [5.0, 6.0]


def test_measure_skips_a_z_stack(tmp_path, capsys):
    from spacr.measure import _export_vvvv_measured_field

    assert _export_vvvv_measured_field(
        str(tmp_path), "z.npy", {"src": str(tmp_path)}, True, None, {}) == []
    assert "only 2-D fields" in capsys.readouterr().out
    assert not (tmp_path / "vvvv").exists()


# ------------------------------------------------------- Mask Generation ---

def _mask_settings(src, **over):
    settings = {
        "src": str(src), "metadata_type": "cellvoyager",
        "channels": [0, 1], "cell_channel": 0, "nucleus_channel": 1,
        "pathogen_channel": None, "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "batch_size": 10,
        "save": True, "custom_regex": None, "randomize": False,
    }
    settings.update(over)
    return settings


def _mask_plate(root, fields=("plate1_A01_1", "plate1_B02_1")):
    src = root / "plate1"
    for folder in ("stack", "masks/cell_mask_stack",
                   "masks/nucleus_mask_stack"):
        (src / folder).mkdir(parents=True)
    image, cells, nuclei = _field()
    for name in fields:
        np.save(src / "stack" / f"{name}.npy", image)
        np.save(src / "masks" / "cell_mask_stack" / f"{name}.npy", cells)
        np.save(src / "masks" / "nucleus_mask_stack" / f"{name}.npy", nuclei)
    return src, image, cells


@pytest.fixture
def no_segmentation(monkeypatch):
    import spacr.object as sobj

    def refuse(*_args, **_kwargs):
        raise AssertionError("masks exist; nothing should be segmented")

    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam", refuse)
    monkeypatch.setattr(sobj, "generate_organelle_masks_sam", refuse)


@pytest.mark.parametrize("enabled", [False, True])
def test_mask_generation_exports_exactly_when_on(tmp_path, no_segmentation,
                                                 enabled):
    from spacr.core import preprocess_generate_masks

    src, image, cells = _mask_plate(tmp_path)
    preprocess_generate_masks(_mask_settings(src, export_vvvv=enabled))
    assert (src / "merged" / "plate1_A01_1.npy").is_file()
    root = src / "vvvv" / "mask"
    if not enabled:
        assert not (src / "vvvv").exists()
        return
    assert sorted(p.name for p in root.iterdir()) == ["cell", "nucleus"]
    for role in ("cell", "nucleus"):
        assert sorted(p.name for p in (root / role).iterdir()) == [
            "plate1_A01_1", "plate1_B02_1"]
    folder = root / "cell" / "plate1_A01_1"
    header, manifest = _check_folder(folder, "plate1_A01_1")
    assert header == BASE + ["mean_intensity_channel_0",
                             "mean_intensity_channel_1"]
    table = tabular.read_table(folder / "plate1_A01_1_objects.csv",
                               canonicalise=False).set_index("label")
    assert table.loc[2, "area"] == 18 * 24
    assert table.loc[1, "mean_intensity_channel_0"] == pytest.approx(
        image[..., 0][cells == 1].mean())
    assert np.array_equal(
        imageio.imread(folder / "plate1_A01_1_labels.png"), cells)
    assert manifest["source_path"].endswith("plate1_A01_1.npy")
    assert manifest["width"] == 64 and manifest["height"] == 48
    readme = (folder / "README.txt").read_text("utf-8")
    assert "spaCR Mask Generation" in readme


def test_merged_export_needs_the_plane_layout(tmp_path, capsys):
    from spacr.io import _export_vvvv_merged

    (tmp_path / "merged").mkdir()
    np.save(tmp_path / "merged" / "a.npy", np.zeros((4, 4, 2), np.uint16))
    assert _export_vvvv_merged(str(tmp_path)) == []
    assert "no plane layout" in capsys.readouterr().out


def test_merged_export_reports_fields_it_cannot_write(tmp_path, capsys):
    from spacr.crops import MERGED_LAYOUT_SIDECAR
    from spacr.io import _export_vvvv_merged

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1, "intensity_channels": [0],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 1, "nucleus": 2}}))
    np.save(merged / "volume.npy", np.zeros((2, 4, 4, 3), np.float32))
    wide = np.zeros((6, 6, 3), np.float64)
    wide[1:3, 1:3, 1] = 70000.0
    wide[3:5, 3:5, 2] = 2.0
    np.save(merged / "wide.npy", wide)
    (merged / "notes.txt").write_text("not a field")
    written = _export_vvvv_merged(str(tmp_path), pixel_size=(0.3, 0.3))
    out = capsys.readouterr().out
    assert "volume.npy: only 2-D fields" in out
    assert "skipped cell in wide.npy" in out
    assert [Path(p).relative_to(tmp_path).as_posix() for p in written] == [
        "vvvv/mask/nucleus/wide"]
    manifest = json.loads((Path(written[0]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pixel_size"]["x"] == 0.3
    assert manifest["channel_names"] == ["channel_0"]


def test_field_export_skips_a_volume_mask(tmp_path, capsys):
    from spacr.io import _export_vvvv_field

    assert _export_vvvv_field(str(tmp_path), "v.npy",
                              {"cell": np.zeros((2, 4, 4), np.uint16)},
                              None) == []
    assert "needs a 2-D label image" in capsys.readouterr().out


def test_the_v2_mask_path_exports_too(tmp_path, monkeypatch):
    import spacr._v1_v2_bridge as bridge
    import spacr.core as core
    import spacr.io as sio
    import spacr.pipeline_v2 as v2

    calls = []
    monkeypatch.setattr(v2, "run_v2", lambda *a, **k: {"stacks": []})
    monkeypatch.setattr(bridge, "v2_channels_from_settings",
                        lambda settings: ([0], ["cell"]))
    monkeypatch.setattr(bridge, "report_disk_savings", lambda *a: None)
    monkeypatch.setattr(core, "_score_v2_masks", lambda *a, **k: None)
    monkeypatch.setattr(sio, "_export_vvvv_merged",
                        lambda src, **k: calls.append(src) or [])
    settings = _mask_settings(tmp_path, pipeline_style="v2")
    core.preprocess_generate_masks(dict(settings))
    assert calls == []
    core.preprocess_generate_masks(dict(settings, export_vvvv=True))
    assert calls == [str(tmp_path)]


@pytest.mark.parametrize("enabled", [False, True])
def test_one_measured_field_exports_in_process(tmp_path, enabled):
    """The per-field worker itself, run in this process."""
    from spacr.measure import _measure_crop_core

    merged, _image, cells = _measure_plate(tmp_path)
    settings = _measure_settings(merged, export_vvvv=enabled,
                                 save_measurements=False,
                                 nucleus_mask_dim=None, experiment="exp",
                                 timelapse=False)
    (tmp_path / "measurements").mkdir()
    result = _measure_crop_core(0, [], "plate1_A01_1.npy", settings)
    assert result[-1] == ""
    if not enabled:
        assert not (tmp_path / "vvvv").exists()
        return
    folder = tmp_path / "vvvv" / "measure" / "cell" / "plate1_A01_1"
    _check_folder(folder, "plate1_A01_1")
    assert np.array_equal(
        imageio.imread(folder / "plate1_A01_1_labels.png"), cells)


def test_merged_export_without_intensities_or_with_a_missing_plane(tmp_path):
    from spacr.crops import MERGED_LAYOUT_SIDECAR
    from spacr.io import _export_vvvv_merged

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1, "intensity_channels": [],
        "mask_plane_order": ["cell", "ghost"],
        "mask_dims": {"cell": 0, "ghost": 5}}))
    plane = np.zeros((6, 6, 1), np.uint16)
    plane[1:4, 1:4, 0] = 3
    np.save(merged / "only.npy", plane)
    written = _export_vvvv_merged(str(tmp_path))
    assert [Path(p).relative_to(tmp_path).as_posix() for p in written] == [
        "vvvv/mask/cell/only"]
    names = sorted(p.name for p in Path(written[0]).iterdir())
    assert "only_image.png" not in names and "only_labels.png" in names


def test_a_table_without_labels_is_ignored(tmp_path):
    from spacr.qt.mask_engine import _export_vvvv

    labels = np.zeros((5, 5), np.uint16)
    labels[1:3, 1:3] = 1
    folder = Path(_export_vvvv(tmp_path, "t.png", labels,
                               table=pd.DataFrame({"x": [1]})))
    header = tabular.read_table(folder / "t_objects.csv",
                                canonicalise=False).columns
    assert list(header) == BASE
