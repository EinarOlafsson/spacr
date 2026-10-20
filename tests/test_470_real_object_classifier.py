"""The real / not-real classifier: trained from Annotate calls, scored on
held-out wells, and used after detection to erase objects it rejects."""
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr import object_classifier as oc
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "train_real_object_classifier.py"


def _crop(rng, real, side=40):
    """A real object is a bright disc; a false one is flat noise."""
    image = rng.normal(30, 6, (side, side, 3))
    if real:
        yy, xx = np.mgrid[:side, :side]
        image[(yy - side / 2) ** 2 + (xx - side / 2) ** 2 < (side / 4) ** 2] += 150
    return np.clip(image, 0, 255).astype(np.uint8)


def _annotated_db(folder, rng, per_well=10, wells=("A01", "A02", "B01", "B02", "C01")):
    crops = folder / "crops"
    crops.mkdir(parents=True)
    rows = []
    for well in wells:
        for index in range(per_well):
            real = index % 3 != 0
            path = crops / f"plate1_{well}_1_{index + 1}.png"
            Image.fromarray(_crop(rng, real)).save(path)
            rows.append({"png_path": str(path), "file_name": path.name,
                         "real": 1 if real else 2})
    rows.append({"png_path": str(path), "file_name": path.name, "real": None})
    database = folder / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame(rows).to_sql("png_list", connection, index=False)
    return database


def test_annotations_group_by_plate_and_well(tmp_path):
    database = _annotated_db(tmp_path, np.random.default_rng(0), per_well=3)
    frame = oc._annotated_real_crops([str(database)])
    assert len(frame) == 15
    assert set(frame["group"]) == {f"plate1|{w}" for w in
                                   ("A01", "A02", "B01", "B02", "C01")}
    assert frame["real"].sum() == 10


def test_training_needs_both_calls_and_two_wells(tmp_path):
    database = _annotated_db(tmp_path, np.random.default_rng(1), per_well=3,
                             wells=("A01",))
    frame = oc._annotated_real_crops([str(database)])
    with pytest.raises(ValueError, match="two wells"):
        oc._train_real_classifier(frame, "pathogen")
    with pytest.raises(ValueError, match="real and not real"):
        oc._train_real_classifier(frame[frame["real"]], "pathogen")
    with pytest.raises(ValueError, match="object type"):
        oc._train_real_classifier(frame, "virus")


def test_cli_trains_scores_on_held_out_wells_and_rescores(tmp_path):
    database = _annotated_db(tmp_path / "round1", np.random.default_rng(2))
    model = tmp_path / "models" / "pathogen.joblib"
    subprocess.run([sys.executable, str(TOOL), "train", "--type", "pathogen",
                    "--db", str(database), "--out", str(model)],
                   check=True, capture_output=True, text=True, timeout=300)
    card = json.loads(model.with_suffix(".scorecard.json").read_text(encoding="utf-8"))
    assert card["object_type"] == "pathogen"
    assert card["n_annotated"] == 50
    assert card["wells_held_out"] and all(w.startswith("plate1|")
                                          for w in card["wells_held_out"])
    assert card["n"] + card["n_train"] == 50
    assert card["balanced_accuracy"] >= 0.9

    fresh = _annotated_db(tmp_path / "round2", np.random.default_rng(3),
                          per_well=4, wells=("D01", "D02"))
    out = tmp_path / "round2.json"
    subprocess.run([sys.executable, str(TOOL), "score", "--model", str(model),
                    "--db", str(fresh), "--out", str(out)],
                   check=True, capture_output=True, text=True, timeout=300)
    rescored = json.loads(out.read_text(encoding="utf-8"))
    assert rescored["n"] == 8 and rescored["roc_auc"] >= 0.9


def _bundle(tmp_path, kind="pathogen"):
    import joblib

    database = _annotated_db(tmp_path / "train", np.random.default_rng(4))
    bundle = oc._train_real_classifier(
        oc._annotated_real_crops([str(database)]), kind)
    folder = tmp_path / "classifiers"
    folder.mkdir()
    joblib.dump(bundle, folder / f"{kind}.joblib")
    return folder


def _plate(src):
    """One field: object 5 a bright disc, object 9 a patch of background."""
    rng = np.random.default_rng(5)
    image = rng.normal(300, 60, (120, 120, 4))
    yy, xx = np.mgrid[:120, :120]
    image[(yy - 30) ** 2 + (xx - 30) ** 2 < 100] += 3000
    mask = np.zeros((120, 120), np.uint16)
    mask[(yy - 30) ** 2 + (xx - 30) ** 2 < 100] = 5
    mask[(yy - 90) ** 2 + (xx - 90) ** 2 < 100] = 9
    (src / "stack").mkdir(parents=True)
    (src / "masks" / "pathogen_mask_stack").mkdir(parents=True)
    np.save(src / "stack" / "plate1_A01_1.npy", image.astype(np.uint16))
    np.save(src / "masks" / "pathogen_mask_stack" / "plate1_A01_1.npy", mask)


def test_objects_called_not_real_are_erased_and_ids_kept(tmp_path):
    folder = _bundle(tmp_path)
    src = tmp_path / "plate"
    _plate(src)
    settings = {"real_object_classifier": str(folder),
                "real_object_threshold": 0.5, "cell_channel": 3,
                "pathogen_channel": 2, "nucleus_channel": 0}
    assert oc._drop_unreal_objects(str(src), settings) == {"pathogen": 1}
    mask = np.load(src / "masks" / "pathogen_mask_stack" / "plate1_A01_1.npy")
    assert set(np.unique(mask)) == {0, 5}
    verdicts = pd.read_csv(src / "qc" / "real_object_filter_pathogen.csv")
    assert dict(zip(verdicts["label"], verdicts["class"])) == {5: "real", 9: "not real"}


def test_a_missing_channel_skips_rather_than_guesses(tmp_path, capsys):
    folder = _bundle(tmp_path)
    src = tmp_path / "plate"
    _plate(src)
    before = np.load(src / "masks" / "pathogen_mask_stack" / "plate1_A01_1.npy")
    assert oc._drop_unreal_objects(str(src), {
        "real_object_classifier": str(folder / "pathogen.joblib"),
        "cell_channel": None, "pathogen_channel": 2, "nucleus_channel": 0}) == {}
    assert "skipped" in capsys.readouterr().out
    np.testing.assert_array_equal(
        before, np.load(src / "masks" / "pathogen_mask_stack" / "plate1_A01_1.npy"))
    missing = str(tmp_path / "absent")
    with pytest.raises(ValueError) as refused:
        oc._real_classifier_bundles(missing)
    assert "real_object_classifier" in str(refused.value)
    assert repr(missing) in str(refused.value)
    assert "Available compatible zoo identities: []." in str(refused.value)


def test_the_mask_run_applies_the_classifier_when_set(yokogawa_cellvoyager_dir,
                                                      monkeypatch):
    import torch
    import spacr.object as O
    import spacr.plot as PL
    from spacr.core import preprocess_generate_masks
    from spacr.settings import set_default_settings_preprocess_generate_masks

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(PL, "plot_cellpose4_output", lambda *a, **k: None)

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D, stitch_threshold=stitch_threshold)
            masks = []
            for image in x:
                mask = np.zeros(np.asarray(image).shape[:2], np.uint16)
                mask[20:60, 20:60] = 1
                masks.append(mask)
            return masks, [np.zeros_like(m, np.float32) for m in masks], None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_Model))
    seen = []

    def record(src, settings):
        seen.append((Path(src), settings["real_object_classifier"],
                     settings["real_object_threshold"]))
        assert (Path(src) / "masks" / "cell_mask_stack").is_dir()
        assert not (Path(src) / "merged").exists() or not any(
            (Path(src) / "merged").glob("*.npy"))
        return {}

    monkeypatch.setattr(oc, "_drop_unreal_objects", record)
    src = yokogawa_cellvoyager_dir["src"]
    preprocess_generate_masks(set_default_settings_preprocess_generate_masks({
        "src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
        "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": None, "plot": False, "batch_size": 1,
        "normalize": True, "randomize": False, "verbose": False,
        "preprocess": True, "masks": True, "save": True, "adjust_cells": False,
        "n_jobs": 1, "cell_diameter": 40, "nucleus_diameter": 20,
        "magnification": 20, "real_object_classifier": "/models/real",
        "real_object_threshold": 0.7}))
    assert seen == [(Path(src), "/models/real", 0.7)]


def test_annotation_reader_preserves_source_database_bytes(tmp_path):
    database = _annotated_db(tmp_path, np.random.default_rng(470), per_well=3)
    before = database.read_bytes()
    frame = oc._annotated_real_crops([str(database)])
    assert len(frame) == 15
    assert database.read_bytes() == before


def test_annotation_reader_coalesces_repeated_identical_source_rows(tmp_path):
    database = _annotated_db(tmp_path, np.random.default_rng(471), per_well=3)
    before = database.read_bytes()
    once = oc._annotated_real_crops([str(database)])
    repeated = oc._annotated_real_crops([str(database), str(database)])
    pd.testing.assert_frame_equal(repeated, once)
    assert database.read_bytes() == before


@pytest.mark.parametrize('conflict', ['label', 'group'])
def test_annotation_reader_refuses_conflicts_without_modifying_sources(tmp_path, conflict):
    import shutil

    database = _annotated_db(tmp_path / 'original', np.random.default_rng(472), per_well=3)
    other = tmp_path / 'other.db'
    shutil.copyfile(database, other)
    with sqlite3.connect(other) as db:
        if conflict == 'label':
            db.execute('UPDATE png_list SET real = 1 WHERE rowid = 1')
        else:
            db.execute('ALTER TABLE png_list ADD COLUMN dataset_plate TEXT')
            db.execute("UPDATE png_list SET dataset_plate = 'another_plate'")
    before = {path: path.read_bytes() for path in (database, other)}
    with pytest.raises(ValueError, match='conflicting labels or acquisition identities'):
        oc._annotated_real_crops([str(database), str(other)])
    assert all(path.read_bytes() == raw for path, raw in before.items())


def _sample_fingerprinted_annotation(tmp_path):
    import hashlib

    database = _annotated_db(tmp_path, np.random.default_rng(473), per_well=3)
    with sqlite3.connect(database) as db:
        crop = Path(db.execute('SELECT png_path FROM png_list WHERE rowid = 1').fetchone()[0])
        pixels = crop.read_bytes()
        db.execute('ALTER TABLE png_list ADD COLUMN sample_png_sha256 TEXT')
        db.execute('ALTER TABLE png_list ADD COLUMN sample_png_bytes INTEGER')
        db.execute('UPDATE png_list SET sample_png_sha256 = ?, sample_png_bytes = ? WHERE rowid = 1',
                   (hashlib.sha256(pixels).hexdigest(), len(pixels)))
    return database, crop


def test_annotation_reader_accepts_matching_sample_creation_fingerprint(tmp_path):
    database, crop = _sample_fingerprinted_annotation(tmp_path)
    before = database.read_bytes(), crop.read_bytes()
    assert len(oc._annotated_real_crops([str(database)])) == 15
    assert (database.read_bytes(), crop.read_bytes()) == before


@pytest.mark.parametrize('problem', ['pixels', 'digest', 'size', 'fractional', 'partial'])
def test_annotation_reader_refuses_changed_or_incomplete_sample_fingerprint(tmp_path, problem):
    database, crop = _sample_fingerprinted_annotation(tmp_path)
    if problem == 'pixels':
        crop.write_bytes(crop.read_bytes() + b'changed after sampling')
    else:
        with sqlite3.connect(database) as db:
            statement = {
                'digest': "sample_png_sha256 = 'not-a-digest'",
                'size': 'sample_png_bytes = -1',
                'fractional': 'sample_png_bytes = 2.5',
                'partial': 'sample_png_sha256 = NULL',
            }[problem]
            db.execute('UPDATE png_list SET ' + statement + ' WHERE rowid = 1')
    before = database.read_bytes(), crop.read_bytes()
    with pytest.raises(ValueError, match='sample-creation fingerprint'):
        oc._annotated_real_crops([str(database)])
    assert (database.read_bytes(), crop.read_bytes()) == before


def test_annotation_reader_does_not_create_missing_databases(tmp_path):
    database = tmp_path / 'missing.db'
    with pytest.raises(sqlite3.OperationalError):
        oc._annotated_real_crops([str(database)])
    assert not database.exists()


@pytest.mark.parametrize('language', ['de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN'])
@pytest.mark.parametrize('problem', ['fingerprint', 'conflict'])
def test_annotation_reader_uses_exact_reviewed_runtime_diagnostics(tmp_path, monkeypatch, language, problem):
    import shutil

    from spacr.qt import i18n

    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    database, crop = _sample_fingerprinted_annotation(tmp_path / 'original')
    databases = [str(database)]
    if problem == 'fingerprint':
        crop.write_bytes(crop.read_bytes() + b'changed after sampling')
        source = 'Annotated crop {path} no longer matches its sample-creation fingerprint.'
    else:
        other = tmp_path / 'other.db'
        shutil.copyfile(database, other)
        with sqlite3.connect(other) as db:
            db.execute('UPDATE png_list SET real = 1 WHERE rowid = 1')
        databases.append(str(other))
        source = 'Annotated crop {path} has conflicting labels or acquisition identities.'
    reviews = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-annotated-crop-integrity.json').read_text(encoding="utf-8"))
    target = next(row['translation'] for row in reviews['records'] if row['source'] == source)
    before = {Path(path): Path(path).read_bytes() for path in databases}
    with pytest.raises(ValueError) as error:
        oc._annotated_real_crops(databases)
    assert str(error.value) == target.format(path=str(crop))
    assert all(path.read_bytes() == raw for path, raw in before.items())
