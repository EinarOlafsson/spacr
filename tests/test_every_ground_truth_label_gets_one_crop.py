"""Item 470: one crop per ground-truth object, traceable to its field and label.

The tools under test turn the four cross-channel datasets into per-object
crops for real / not-real annotation:

  tools/build_ground_truth_crop_stacks.py      one (H, W, 6) stack per field
  tools/measure_ground_truth_crops.py          Measure, one object type a run
  tools/build_ground_truth_annotation_sample.py a seeded sample Annotate opens

These tests run all three on a tiny synthetic project (two fields, CPU only)
and hold what the brief asks for: every label gets exactly one crop, each
crop names the field and label it came from, a mask a field's datasets did
not draw gives no crops, and reruns neither overwrite nor lose judgements.
"""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stacks = _load("build_ground_truth_crop_stacks")
cropper = _load("measure_ground_truth_crops")
sampler = _load("build_ground_truth_annotation_sample")

BOTH = "CSA_screen__screen_20250124_133156__plate1_A01_8_1"
TOXO_ONLY = "CSA_screen__screen_20250124_133156__plate1_B02_3_1"
SIZE = 360


def _disk(mask, cy, cx, r, label):
    yy, xx = np.ogrid[:SIZE, :SIZE]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label


def _masks():
    """Non-contiguous label ids, so a renumbering would show."""
    cell = np.zeros((SIZE, SIZE), np.uint16)
    nucleus = np.zeros_like(cell)
    pv = np.zeros_like(cell)
    for label, (cy, cx) in {4: (100, 100), 11: (250, 250)}.items():
        _disk(cell, cy, cx, 60, label)
        _disk(nucleus, cy, cx, 18, label + 1)
        _disk(pv, cy + 30, cx + 30, 10, label * 2 + 1)
    _disk(nucleus, 300, 60, 15, 30)
    return cell, nucleus, pv


def _project(tmp_path):
    """Synthetic data/ with the layout the stack builder reads."""
    data = tmp_path / "data"
    rng = np.random.default_rng(470)
    cell, nucleus, pv = _masks()
    holds = {BOTH: stacks.DATASETS,
             TOXO_ONLY: ("toxoplasma_from_hoechst", "toxoplasma_from_cellmask")}
    for stem, datasets in holds.items():
        image = lambda: rng.integers(100, 4000, (SIZE, SIZE)).astype(np.uint16)
        (data / "_dsred_source").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(data / "_dsred_source" / f"{stem}.tif", image())
        for dataset in datasets:
            for folder in ("images", "masks", "masks_pv"):
                (data / dataset / folder).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(data / dataset / "images" / f"{stem}.tif", image())
        if "cell_from_hoechst" in datasets:
            tifffile.imwrite(data / "cell_from_hoechst" / "masks" / f"{stem}.tif", cell)
        if "nuclei_from_cellmask" in datasets:
            tifffile.imwrite(data / "nuclei_from_cellmask" / "masks" / f"{stem}.tif",
                             nucleus)
        tifffile.imwrite(data / "toxoplasma_from_cellmask" / "masks_pv" / f"{stem}.tif",
                         pv)
    return data


@pytest.fixture()
def project(tmp_path, monkeypatch):
    data = _project(tmp_path)
    monkeypatch.setattr(stacks, "DATA", data)
    monkeypatch.setattr(stacks, "nas_plane", lambda *a, **k: None)
    out = tmp_path / "ground_truth_crops"
    assert stacks.main(["--out", str(out)]) == 0
    return out


def test_thp1_plates_from_different_acquisitions_get_different_names():
    a = "THP1_screen__screen_20250218_153752__PLATE1_A01_1_1"
    b = "THP1_screen__screen_20250219_101010__PLATE1_A01_1_1"
    names = stacks.crop_names([a, b])
    assert len(set(names.values())) == 2
    assert all("_" not in n.rsplit("_", 2)[0] for n in names.values())


def test_a_field_without_its_parasite_channel_is_skipped_not_blanked(tmp_path,
                                                                    monkeypatch):
    data = _project(tmp_path)
    (data / "_dsred_source" / f"{BOTH}.tif").unlink()
    monkeypatch.setattr(stacks, "DATA", data)
    assert stacks.build_stack(BOTH, {}) is None
    assert stacks.build_stack(TOXO_ONLY, {}).shape == (SIZE, SIZE, 6)


def test_stacks_hold_three_channels_and_the_masks_their_datasets_drew(project):
    plate = project / "CSAscreen-plate1"
    with open(plate / "fields.csv", encoding="utf-8") as handle:
        rows = {r["stem"]: r for r in csv.DictReader(handle)}
    assert set(rows) == {BOTH, TOXO_ONLY}
    assert rows[BOTH]["datasets"].split(";") == list(stacks.DATASETS)
    toxo = np.load(plate / "merged" / f"{rows[TOXO_ONLY]['field']}.npy")
    assert toxo.shape == (SIZE, SIZE, 6) and toxo.dtype == np.uint16
    assert toxo[..., 3].max() == 0 and toxo[..., 4].max() == 0
    assert set(np.unique(toxo[..., 5])) == {0, 9, 23}


def test_a_rerun_keeps_the_stacks_it_already_wrote(project, capsys):
    target = next((project / "CSAscreen-plate1" / "merged").glob("*.npy"))
    stamp = target.stat().st_mtime_ns
    assert stacks.main(["--out", str(project)]) == 0
    assert "written 0, already there 2" in capsys.readouterr().out
    assert target.stat().st_mtime_ns == stamp


def test_every_label_gets_one_crop_that_names_its_field_and_label(project):
    assert cropper.main(["--root", str(project), "--jobs", "1"]) == 0
    expected = {"pathogen": {("A01", "8", 9), ("A01", "8", 23),
                             ("B02", "3", 9), ("B02", "3", 23)},
                "cell": {("A01", "8", 4), ("A01", "8", 11)},
                "nucleus": {("A01", "8", 5), ("A01", "8", 12), ("A01", "8", 30)}}
    for kind, labels in expected.items():
        folder = project / kind / "CSAscreen-plate1"
        with sqlite3.connect(folder / "measurements" / "measurements.db") as db:
            rows = db.execute(
                f"SELECT png_path, file_name, {kind}_id FROM png_list").fetchall()
        found = set()
        for path, name, object_id in rows:
            assert Path(path).is_file()
            _, well, field = name.split("_")[:3]
            found.add((well, field, int(str(object_id).lstrip("o"))))
        assert len(rows) == len(labels)
        assert found == labels
        assert len(list(folder.rglob(f"{kind}_png/*.png"))) == len(labels)


def test_the_sample_keeps_every_judgement_when_it_grows(project):
    cropper.main(["--root", str(project), "--jobs", "1", "--type", "nucleus"])
    (project / "cell").mkdir(exist_ok=True)
    (project / "pathogen").mkdir(exist_ok=True)
    sampler.main(["--root", str(project), "--per-type", "2"])
    database = project / "annotation_sample" / "nucleus" / "measurements" / "measurements.db"
    with sqlite3.connect(database) as db:
        first = [r[0] for r in db.execute("SELECT png_path FROM png_list")]
        db.execute("UPDATE png_list SET real = 2 WHERE png_path = ?", (first[0],))
    sampler.main(["--root", str(project), "--per-type", "3"])
    with sqlite3.connect(database) as db:
        rows = dict(db.execute("SELECT png_path, real FROM png_list"))
    assert len(rows) == 3 and rows[first[0]] == 2
