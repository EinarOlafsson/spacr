"""CPU coverage for spacr.io's dataset builders, loaders and format converters.

Everything here runs on synthetic images/DBs — no GPU, no Cellpose model,
no network. Targets the large uncovered blocks of io.py: the Dataset /
DataLoader classes, the train/test dataset builders, the annotation-driven
dataset builders and the Yokogawa filename converters.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest
import tifffile
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _png(path, rng, size=32):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _class_dirs(root, rng, classes=("nc", "pc"), n=6):
    out = []
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        out.append([_png(d / f"{cls}_{i}.png", rng) for i in range(n)])
    return out


# ---------------------------------------------------------------------------
# generate_dataset_from_lists
# ---------------------------------------------------------------------------

def test_generate_dataset_from_lists_splits_train_test(tmp_path, rng):
    from spacr.io import generate_dataset_from_lists
    data = _class_dirs(tmp_path / "src", rng, n=10)
    train, test = generate_dataset_from_lists(
        str(tmp_path / "out"), data, ["nc", "pc"], test_split=0.2,
        group_by="cell")
    assert os.path.isdir(train) and os.path.isdir(test)
    n_train = sum(len(files) for _r, _d, files in os.walk(train))
    n_test = sum(len(files) for _r, _d, files in os.walk(test))
    assert n_train + n_test == 20
    assert n_test == 4          # 20% of 10 per class


def test_generate_dataset_from_lists_length_mismatch(tmp_path, rng):
    from spacr.io import generate_dataset_from_lists
    data = _class_dirs(tmp_path / "src", rng, n=2)
    with pytest.raises(ValueError):
        generate_dataset_from_lists(str(tmp_path / "o"), data, ["only_one"])


# ---------------------------------------------------------------------------
# training_dataset_from_annotation (+ _metadata variant)
# ---------------------------------------------------------------------------

def _png_list_db(tmp_path, rng, n=20, annotate=True, two_classes=True):
    meas = tmp_path / "measurements"
    meas.mkdir(parents=True, exist_ok=True)
    pngs = tmp_path / "pngs"
    pngs.mkdir(exist_ok=True)
    paths = [_png(pngs / f"o{i}.png", rng) for i in range(n)]
    if annotate:
        if two_classes:
            anno = [1 if i % 2 == 0 else 2 for i in range(n)]
        else:
            anno = [1 if i < n // 2 else None for i in range(n)]
    else:
        anno = [None] * n
    db = meas / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (png_path TEXT, test INT, "
                "plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT)")
    con.executemany(
        "INSERT INTO png_list VALUES (?,?,?,?,?,?)",
        [(p, a, "plate1", "r1", "c1" if i % 2 == 0 else "c2", "f1")
         for i, (p, a) in enumerate(zip(paths, anno))])
    con.commit(); con.close()
    return str(db)


def test_training_dataset_from_annotation_two_classes(tmp_path, rng):
    from spacr.io import training_dataset_from_annotation
    db = _png_list_db(tmp_path, rng)
    out = training_dataset_from_annotation(
        db, str(tmp_path / "dst"), annotation_column="test",
        annotated_classes=(1, 2))
    assert len(out) == 2 and all(len(c) > 0 for c in out)


def test_training_dataset_from_annotation_single_class_samples_other(tmp_path, rng):
    """One annotated class → an equal-sized 'other' class is sampled."""
    from spacr.io import training_dataset_from_annotation
    db = _png_list_db(tmp_path, rng, two_classes=False)
    out = training_dataset_from_annotation(
        db, str(tmp_path / "dst"), annotation_column="test",
        annotated_classes=(1,))
    assert len(out) == 2
    assert len(out[0]) == len(out[1])


def test_training_dataset_from_annotation_metadata(tmp_path, rng):
    from spacr.io import training_dataset_from_annotation_metadata
    db = _png_list_db(tmp_path, rng)
    out = training_dataset_from_annotation_metadata(
        db, str(tmp_path / "dst"), annotation_column="test",
        annotated_classes=(1, 2), metadata_type_by="columnID",
        class_metadata=[["c1"], ["c2"]])
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Dataset / DataLoader classes
# ---------------------------------------------------------------------------

def test_no_class_dataset_returns_image_and_path(tmp_path, rng):
    from spacr.io import NoClassDataset
    d = tmp_path / "imgs"; d.mkdir()
    for i in range(5):
        _png(d / f"i{i}.png", rng)
    # NoClassDataset takes the DIRECTORY, not a list of paths.
    ds = NoClassDataset(str(d), transform=None, shuffle=False)
    assert len(ds) == 5
    item = ds[0]
    assert isinstance(item, tuple) and len(item) >= 2


def test_spacr_dataset_reads_class_folders(tmp_path, rng):
    from spacr.io import spacrDataset
    root = tmp_path / "train"
    _class_dirs(root, rng, n=4)
    ds = spacrDataset(str(root), loader_classes=["nc", "pc"], transform=None,
                      shuffle=False)
    assert len(ds) == 8
    img, label, path = ds[0]
    assert label in (0, 1)


def test_tar_image_dataset(tmp_path, rng):
    import tarfile
    from spacr.io import TarImageDataset
    d = tmp_path / "imgs"; d.mkdir()
    for i in range(3):
        _png(d / f"i{i}.png", rng)
    tar_path = tmp_path / "ds.tar"
    with tarfile.open(tar_path, "w") as t:
        for p in d.iterdir():
            t.add(p, arcname=p.name)
    ds = TarImageDataset(str(tar_path), transform=None)
    assert len(ds) == 3
    out = ds[0]
    assert isinstance(out, tuple)


# ---------------------------------------------------------------------------
# generate_loaders
# ---------------------------------------------------------------------------

def test_generate_loaders_train_mode(tmp_path, rng):
    from spacr.io import generate_loaders
    _class_dirs(tmp_path / "train", rng, n=6)
    _class_dirs(tmp_path / "test", rng, n=3)
    train, val, _ = generate_loaders(
        str(tmp_path), mode="train", image_size=32, batch_size=2,
        classes=["nc", "pc"], n_jobs=0, validation_split=0.25,
        pin_memory=False, normalize=True, channels=["r", "g", "b"],
        augment=False, verbose=False)
    assert train is not None and val is not None
    assert train.num_workers == 0
    assert val.num_workers == 0
    batch = next(iter(train))
    assert len(batch) >= 2


def test_generate_loaders_test_mode(tmp_path, rng):
    from spacr.io import generate_loaders
    _class_dirs(tmp_path / "train", rng, n=4)
    _class_dirs(tmp_path / "test", rng, n=4)
    test, _, _ = generate_loaders(
        str(tmp_path), mode="test", image_size=32, batch_size=2,
        classes=["nc", "pc"], n_jobs=0, validation_split=0.0,
        pin_memory=False, normalize=True, channels=["r", "g", "b"],
        augment=False, verbose=True)
    assert test is not None


# ---------------------------------------------------------------------------
# Yokogawa converters
# ---------------------------------------------------------------------------

def test_convert_separate_files_to_yokogawa(tmp_path, rng):
    """Per-slice TIFFs are grouped, MIP'd and renamed to the CV convention."""
    from spacr.io import convert_separate_files_to_yokogawa
    folder = tmp_path / "raw"; folder.mkdir()
    # two z-slices for one well/field/channel → max projected into one file
    for z in (1, 2):
        tifffile.imwrite(
            str(folder / f"WellA1_F1_T1_C1_Z{z}.tif"),
            rng.integers(0, 500, (16, 16)).astype(np.uint16))
    regex = (r"Well(?P<wellID>[A-Z]\d+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
             r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")
    convert_separate_files_to_yokogawa(str(folder), regex)
    out = list(folder.glob("*.tif"))
    assert any("plate1_" in p.name for p in out), [p.name for p in out]
    assert (folder / "rename_log.csv").is_file()


def test_convert_to_yokogawa_tif_stack(tmp_path, rng):
    """A multi-page TIFF is expanded into per-plane Yokogawa TIFFs."""
    from spacr.io import convert_to_yokogawa
    folder = tmp_path / "raw"; folder.mkdir()
    stack = rng.integers(0, 500, (3, 16, 16)).astype(np.uint16)
    tifffile.imwrite(str(folder / "movie.tif"), stack)
    convert_to_yokogawa(str(folder))
    produced = [p.name for p in folder.glob("*.tif") if p.name != "movie.tif"]
    assert produced, "no Yokogawa-named files produced"


# ---------------------------------------------------------------------------
# cellpose train/test split + numpy→tiff
# ---------------------------------------------------------------------------

def test_generate_cellpose_train_test(tmp_path, rng):
    from spacr.io import generate_cellpose_train_test
    src = tmp_path / "cp"
    masks = src / "masks"
    masks.mkdir(parents=True)
    for i in range(10):
        name = f"img_{i}.tif"
        tifffile.imwrite(str(src / name),
                         rng.integers(0, 500, (16, 16)).astype(np.uint16))
        m = np.zeros((16, 16), np.uint16); m[2:6, 2:6] = 1
        tifffile.imwrite(str(masks / name), m)
    generate_cellpose_train_test(str(src), test_split=0.2)
    assert (src.parent / "train").is_dir() or (src / "train").is_dir()


def test_convert_numpy_to_tiff(tmp_path, rng):
    from spacr.io import convert_numpy_to_tiff
    folder = tmp_path / "npys"; folder.mkdir()
    for i in range(3):
        np.save(folder / f"a{i}.npy",
                rng.integers(0, 500, (16, 16)).astype(np.uint16))
    convert_numpy_to_tiff(str(folder))
    tiffs = list(folder.rglob("*.tif"))
    assert tiffs


# ---------------------------------------------------------------------------
# misc small helpers
# ---------------------------------------------------------------------------

def test_parse_gz_files(tmp_path):
    from spacr.io import parse_gz_files
    d = tmp_path / "fq"; d.mkdir()
    for name in ("s1_R1_001.fastq.gz", "s1_R2_001.fastq.gz"):
        (d / name).write_bytes(b"\x1f\x8b")
    out = parse_gz_files(str(d))
    assert isinstance(out, dict) and out


def test_delete_empty_subdirectories(tmp_path):
    from spacr.io import delete_empty_subdirectories
    (tmp_path / "empty").mkdir()
    full = tmp_path / "full"; full.mkdir()
    (full / "f.txt").write_text("x")
    delete_empty_subdirectories(str(tmp_path))
    assert not (tmp_path / "empty").exists()
    assert full.exists()


def test_is_dir_empty(tmp_path):
    from spacr.io import _is_dir_empty
    e = tmp_path / "e"; e.mkdir()
    assert _is_dir_empty(str(e)) is True
    (e / "x").write_text("1")
    assert _is_dir_empty(str(e)) is False


def test_process_non_tif_non_2D_images_converts_png(tmp_path, rng):
    from spacr.io import process_non_tif_non_2D_images
    d = tmp_path / "mixed"; d.mkdir()
    Image.fromarray(rng.integers(0, 255, (16, 16)).astype(np.uint8)).save(d / "g.png")
    process_non_tif_non_2D_images(str(d))
    assert list(d.glob("*.tif")), "grayscale PNG should be converted to TIFF"


def test_save_and_load_object_mask_roundtrip(tmp_path):
    from spacr.io import save_object_mask, _load_array_any, _mask_variant_path
    m = np.zeros((8, 8), np.uint16); m[1:4, 1:4] = 7
    out = save_object_mask(str(tmp_path), "field1.npy", m, compression="lzw")
    assert os.path.isfile(out)
    back = _load_array_any(out)
    assert back.max() == 7
    assert _mask_variant_path(str(tmp_path), "field1.npy") is not None


def test_create_database_and_object_counts(tmp_path):
    from spacr.io import _create_database, _save_object_counts_to_database
    db = tmp_path / "m.db"
    _create_database(str(db))
    arr = np.zeros((8, 8), np.uint16); arr[1:3, 1:3] = 1; arr[5:7, 5:7] = 2
    _save_object_counts_to_database([arr], "cell", ["f1.npy"], str(db), "")
    con = sqlite3.connect(db)
    rows = con.execute("SELECT * FROM object_counts").fetchall()
    con.close()
    assert rows
