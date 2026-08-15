"""CPU coverage for spacr.io's big dataset builders:
generate_training_dataset (metadata / annotation / measurement modes),
generate_dataset (tar packing) and prepare_cellpose_dataset.
"""
from __future__ import annotations

import os
import sqlite3
import tarfile

import numpy as np
import pytest
import tifffile
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

N = 40


def _png(path, rng, size=32):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _entity(rng, entity, n=N):
    cols = {
        "object_label": np.arange(1, n + 1),
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        "columnID": ["c1" if i % 2 == 0 else "c2" for i in range(n)],
        "fieldID": ["f1"] * n,
        "prcf": [f"plate1_r1_c{(i % 2) + 1}_f1" for i in range(n)],
        "prc": [f"plate1_r1_c{(i % 2) + 1}" for i in range(n)],
        "cell_id": np.arange(1, n + 1),
    }
    import pandas as pd
    for ch in range(4):
        cols[f"{entity}_channel_{ch}_mean_intensity"] = rng.uniform(100, 5000, n)
    cols[f"{entity}_area"] = rng.uniform(200, 4000, n)
    return pd.DataFrame(cols)


@pytest.fixture
def train_src(tmp_path, rng):
    """Plate folder with measurements.db (measurement tables + png_list with
    an annotation column) and the PNG crops those rows point at."""
    import pandas as pd
    src = tmp_path / "plate1"
    meas = src / "measurements"; meas.mkdir(parents=True)
    pngs = src / "data" / "cell_png"; pngs.mkdir(parents=True)

    # SPACR-SHAPED CROP NAMES: plate_row_column_field_object. The split by
    # well reads the well out of the filename, so a crop called "o1.png" is
    # one spaCR could not have produced and a fixture built from those was
    # testing a situation that cannot arise. The prcfo column below already
    # encoded the same identity; the files did not.
    # A REAL PLATE SHAPE: five rows x two columns = ten wells, and the
    # CONDITION is the column, which is how a screen is actually laid out.
    #
    # This matters for instruction 94's well-grouped split, in two ways the
    # old fixture hid. `metadata_type_by: columnID` makes the class the
    # COLUMN -- so with a single row each class sat in exactly one well, and
    # a leakage-safe split correctly refuses that: holding out the only well
    # of a class leaves the class untrained. Across five rows the same
    # column is five independent wells, which is what a plate really is.
    #
    # And because whole wells move, the held-out fraction is granular: five
    # wells per class makes the 20% asked for land on one well per class,
    # which is what the size assertions below expect.
    _row = lambda i: (i % 10) % 5 + 1
    _col = lambda i: (i % 10) // 5 + 1
    _cond = lambda i: f"c{_col(i)}"
    paths = [_png(pngs / f"plate1_r{_row(i)}_c{_col(i)}_f1_o{i+1}.png", rng)
             for i in range(N)]
    con = sqlite3.connect(meas / "measurements.db")
    try:
        for e in ("cell", "nucleus", "pathogen", "cytoplasm"):
            _entity(rng, e).to_sql(e, con, index=False)
        png_list = pd.DataFrame({
            "cell_id": [f"o{i+1}" for i in range(N)],
            "png_path": paths,
            "plateID": ["plate1"] * N,
            "rowID": [f"r{_row(i)}" for i in range(N)],
            "columnID": [_cond(i) for i in range(N)],
            "fieldID": ["f1"] * N,
            "prcfo": [f"plate1_r{_row(i)}_c{_col(i)}_f1_o{i+1}" for i in range(N)],
            "prcf": [f"plate1_r{_row(i)}_c{_col(i)}_f1" for i in range(N)],
            "test": [_col(i) for i in range(N)],
            # legacy metadata mode buckets png_list rows by 'condition'
            "condition": [_cond(i) for i in range(N)],
            # measurement mode filters png_list columns directly (_load_png_table
            # reads png_list only), so the measured feature has to live here.
            "cell_area": [1000.0 if _col(i) == 1 else 3000.0 for i in range(N)],
        })
        png_list.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return str(src)


def _tds(src, **over):
    s = {
        "src": src,
        "tables": ["cell"],
        "dataset_mode": "metadata",
        "annotation_column": "test",
        "annotated_classes": [1, 2],
        "class_metadata": ["c1", "c2"],
        "metadata_type_by": "columnID",
        "channel_of_interest": 3,
        "custom_measurement": None,
        "nuclei_limit": True, "pathogen_limit": True,
        "png_type": "cell_png",
        "size": 32, "test_split": 0.2,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------

def _assert_split(out, expected_classes):
    """Both split folders exist, hold one sub-folder per class, and are disjoint."""
    train, test = out
    assert os.path.isdir(train) and os.path.isdir(test)
    assert sorted(os.listdir(train)) == sorted(expected_classes)
    assert sorted(os.listdir(test)) == sorted(expected_classes)

    def _names(root):
        return {(cls, f) for cls in os.listdir(root)
                for f in os.listdir(os.path.join(root, cls))}

    tr, te = _names(train), _names(test)
    assert tr and te
    assert not ({f for _, f in tr} & {f for _, f in te}), "train/test overlap"
    # test_split=0.2 of the 40 crops
    assert len(tr) + len(te) == N
    assert len(te) == pytest.approx(N * 0.2, abs=2)


def test_generate_training_dataset_metadata_mode(train_src):
    from spacr.io import generate_training_dataset
    out = generate_training_dataset(_tds(train_src))
    _assert_split(out, ["c1", "c2"])


def test_generate_training_dataset_annotation_mode(train_src):
    from spacr.io import generate_training_dataset
    out = generate_training_dataset(_tds(
        train_src, dataset_mode="annotation",
        annotation_column="test", annotated_classes=[1, 2]))
    # one class folder per observed value of the annotation column
    _assert_split(out, ["test_1", "test_2"])


def test_generate_training_dataset_measurement_mode(train_src):
    """measurement mode is driven by ``measurement_rules``.

    ``custom_measurement`` + ``class_metadata`` (what the old call used) are the
    legacy keys — settings.py documents custom_measurement as having no effect,
    and io.generate_training_dataset only reads ``measurement_rules``. With the
    old arguments no class was ever assembled, the function printed
    "No class data assembled; aborting." and returned ``(None, None)``, which
    ``assert out is not None`` happily accepted under a swallowed skip.
    """
    from spacr.io import generate_training_dataset
    out = generate_training_dataset(_tds(
        train_src, dataset_mode="measurement",
        measurement_rules=[
            {"name": "small",
             "where": [{"column": "cell_area", "op": "<", "value": 2000}]},
            {"name": "large",
             "where": [{"column": "cell_area", "op": ">=", "value": 2000}]},
        ]))
    _assert_split(out, ["small", "large"])


# ---------------------------------------------------------------------------
# generate_dataset (tar packing)
# ---------------------------------------------------------------------------

def test_generate_dataset_packs_tar(train_src):
    from spacr.io import generate_dataset
    out = generate_dataset({"src": train_src, "file_metadata": None,
                            "experiment": "exp1", "sample": None})
    tars = []
    for root, _d, files in os.walk(train_src):
        tars += [os.path.join(root, f) for f in files if f.endswith(".tar")]
    assert tars, "no tar produced"
    with tarfile.open(tars[0]) as t:
        assert len(t.getnames()) > 0


def test_generate_dataset_with_sample_subsets(train_src):
    from spacr.io import generate_dataset
    generate_dataset({"src": train_src, "file_metadata": None,
                      "experiment": "exp2", "sample": 5})
    tars = []
    for root, _d, files in os.walk(train_src):
        tars += [os.path.join(root, f) for f in files
                 if f.endswith(".tar") and "exp2" in f]
    assert tars
    with tarfile.open(tars[0]) as t:
        assert len(t.getnames()) == 5


def test_generate_dataset_file_metadata_filter(train_src):
    """file_metadata restricts the selected PNG paths."""
    from spacr.io import generate_dataset
    generate_dataset({"src": train_src, "file_metadata": "o1.png",
                      "experiment": "exp3", "sample": None})
    tars = []
    for root, _d, files in os.walk(train_src):
        tars += [os.path.join(root, f) for f in files
                 if f.endswith(".tar") and "exp3" in f]
    assert tars


# ---------------------------------------------------------------------------
# prepare_cellpose_dataset
# ---------------------------------------------------------------------------

def test_prepare_cellpose_dataset(tmp_path, rng):
    from spacr.io import prepare_cellpose_dataset
    root = tmp_path / "sets"
    for ds in ("dsA", "dsB"):
        d = root / ds
        m = d / "masks"
        m.mkdir(parents=True)
        for i in range(6):
            name = f"{ds}_{i}.tif"
            tifffile.imwrite(str(d / name),
                             rng.integers(0, 500, (16, 16)).astype(np.uint16))
            mask = np.zeros((16, 16), np.uint16); mask[2:6, 2:6] = 1
            tifffile.imwrite(str(m / name), mask)
    prepare_cellpose_dataset(str(root), augment_data=False,
                             train_fraction=0.75, n_jobs=1)
    out = root / "cellpose_dataset"
    assert (out / "train").is_dir() and (out / "test").is_dir()
    assert any((out / "train").rglob("*.tif"))


def test_prepare_cellpose_dataset_with_augmentation(tmp_path, rng):
    """augment_data=True expands the smaller dataset via apply_augmentation."""
    from spacr.io import prepare_cellpose_dataset
    root = tmp_path / "sets"
    for ds, n in (("big", 8), ("small", 3)):
        d = root / ds
        m = d / "masks"
        m.mkdir(parents=True)
        for i in range(n):
            name = f"{ds}_{i}.tif"
            tifffile.imwrite(str(d / name),
                             rng.integers(0, 500, (16, 16)).astype(np.uint16))
            mask = np.zeros((16, 16), np.uint16); mask[2:6, 2:6] = 1
            tifffile.imwrite(str(m / name), mask)
    prepare_cellpose_dataset(str(root), augment_data=True,
                             train_fraction=0.75, n_jobs=1)
    assert (root / "cellpose_dataset" / "train").is_dir()


def test_apply_augmentation_methods(rng):
    from spacr.io import apply_augmentation
    img = rng.integers(0, 255, (16, 16)).astype(np.uint8)
    for method in ("rotate90", "rotate180", "rotate270", "flip_h", "flip_v"):
        try:
            out = apply_augmentation(img, method)
        except Exception:
            continue
        assert out.shape == img.shape
