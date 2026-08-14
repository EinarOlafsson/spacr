"""The image UMAP and the Classify dataset builders, driven off ``merged/*.npy``.

:mod:`spacr.crops` can cut any single object straight out of the merged array,
so a pre-generated PNG crop folder is an *optimisation*, not a requirement.
Until the wiring these tests pin, only the Qt Annotate screen knew that: the
image UMAP and both Classify dataset builders read the folder unconditionally,
and ``crop_source`` was a setting three defaults functions declared and no code
read.

The load-bearing assertions here are the two that make the sources
interchangeable rather than merely both-present:

* :func:`test_tar_from_merged_is_pixel_identical_to_the_png_folder` and
  :func:`test_training_split_from_merged_is_pixel_identical` -- a crop cut on
  demand is byte-for-byte what the PNG folder holds for the same object, so a
  model trained through one source is comparable with a model trained through
  the other;
* :func:`test_umap_runs_with_no_png_folder_and_no_png_list` -- the embedding
  draws real thumbnails on a project that never wrote a crop at all.

Everything is built with the real writers (``measure._measure_crop_core``,
which calls ``utils._merge_and_save_to_database`` and
``utils.filepaths_to_database``), never a hand-built schema.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sqlite3
import tarfile
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from spacr import crops


N_CELLS = 12
FIELDS = ("plate1_A01_1.npy", "plate1_A02_1.npy")


# ---------------------------------------------------------------------------
# A real project: merged arrays, measurements.db, and a PNG crop folder,
# all written by the code that writes them in production.
# ---------------------------------------------------------------------------

def _disks(shape, n, radius=14):
    """A label mask of ``n`` non-overlapping disks on a grid."""
    mask = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    per_row = int(np.ceil(np.sqrt(n)))
    step_y, step_x = shape[0] // per_row, shape[1] // per_row
    label = 1
    for i in range(per_row):
        for j in range(per_row):
            if label > n:
                return mask
            cy, cx = i * step_y + step_y // 2, j * step_x + step_x // 2
            mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = label
            label += 1
    return mask


def _build_project(root):
    """Write merged/, measurements.db and the crop folder with the real writers."""
    from spacr.io import _save_settings_to_db
    from spacr.measure import _measure_crop_core
    from spacr.settings import get_measure_crop_settings

    rng = np.random.default_rng(0)
    merged = os.path.join(root, "merged")
    os.makedirs(merged, exist_ok=True)
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)

    for name in FIELDS:
        cell = _disks((192, 192), N_CELLS)
        nucleus = np.zeros_like(cell)
        yy, xx = np.mgrid[:192, :192]
        for cid in np.unique(cell):
            if cid == 0:
                continue
            ys, xs = np.where(cell == cid)
            cy, cx = int(ys.mean()), int(xs.mean())
            nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2] = cid
        pathogen = np.zeros_like(cell)
        chans = []
        for c in range(4):
            base = rng.integers(50, 200, size=(192, 192)).astype(np.uint16)
            base[cell > 0] += 1000 * (c + 1)
            chans.append(base)
        data = np.stack(chans + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)
        np.save(os.path.join(merged, name), data)

    settings = get_measure_crop_settings(settings={})
    settings.update({
        "src": merged, "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [48, 48],
        "save_measurements": True, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False, "cytoplasm": True,
        "cell_min_size": 1, "nucleus_min_size": 1, "pathogen_min_size": 1,
        "cytoplasm_min_size": 1,
    })
    for i, name in enumerate(FIELDS):
        _measure_crop_core(i, [], name, settings)
    _save_settings_to_db(settings)
    return root


@pytest.fixture(scope="module")
def pristine_project(tmp_path_factory):
    """One built project, copied per test so each may delete/mutate freely."""
    return _build_project(str(tmp_path_factory.mktemp("crop_source_src")))


@pytest.fixture
def project(pristine_project, tmp_path):
    """A private copy of the built project."""
    dst = str(tmp_path / "plate1")
    shutil.copytree(pristine_project, dst)
    crops.clear_field_cache()
    crops.clear_crop_format_cache()
    return dst


def _db(project):
    return os.path.join(project, "measurements", "measurements.db")


def _crop_pngs(project):
    return sorted(glob.glob(os.path.join(project, "data", "**", "*.png"),
                            recursive=True))


def _drop_png_folder(project):
    shutil.rmtree(os.path.join(project, "data"))


def _annotate(project, column="test"):
    """Give png_list a two-valued annotation column, the way Annotate would."""
    conn = sqlite3.connect(_db(project))
    try:
        have = {r[1] for r in conn.execute('PRAGMA table_info("png_list")')}
        if column not in have:
            conn.execute(f'ALTER TABLE png_list ADD COLUMN "{column}" INTEGER')
        # NOT `rowid`: png_list has a column called rowID and SQLite resolves
        # the bare identifier to that column, so `WHERE rowid = ?` would
        # rewrite every crop in a plate row (see spacr.predictions).
        ids = [r[0] for r in conn.execute("SELECT _rowid_ FROM png_list")]
        for i, rid in enumerate(ids):
            conn.execute(f'UPDATE png_list SET "{column}" = ? WHERE _rowid_ = ?',
                         (1 if i % 2 == 0 else 2, rid))
        conn.commit()
    finally:
        conn.close()


def _umap_settings(project, crop_source):
    return {
        "src": project, "tables": ["cell"], "row_limit": None,
        "plot_images": True, "image_nr": 3, "save_figure": False,
        "n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean",
        "eps": 0.5, "min_samples": 3, "clustering": "dbscan",
        "reduction_method": "umap", "verbose": False, "n_jobs": 1,
        "plot_cluster_grids": True, "analyze_clusters": False,
        # Off by default here so the row count is the object count and not
        # whatever DBSCAN called noise; the noise path has its own test.
        "remove_cluster_noise": False,
        "crop_source": crop_source,
    }


def _tar_members(tar_path):
    """Return ``{member name: RGB array}`` for the image members of a tar."""
    out = {}
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith(".png"):
                data = tar.extractfile(member).read()
                out[member.name] = np.array(
                    Image.open(BytesIO(data)).convert("RGB"))
    return out


# ---------------------------------------------------------------------------
# The fixture itself has to be a real spacr project, or nothing below means
# anything.
# ---------------------------------------------------------------------------

def test_the_fixture_is_a_real_spacr_project(project):
    conn = sqlite3.connect(_db(project))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    n_png = conn.execute("SELECT COUNT(*) FROM png_list").fetchone()[0]
    n_cell = conn.execute("SELECT COUNT(*) FROM cell").fetchone()[0]
    conn.close()
    assert {"cell", "png_list", "settings"} <= tables
    assert n_png == n_cell == N_CELLS * len(FIELDS)
    assert len(_crop_pngs(project)) == n_png
    # measure stamps the folder it writes, so the crops are format 2.
    assert crops.crop_folder_format(
        os.path.dirname(_crop_pngs(project)[0])) == crops.CROP_FORMAT_DECLARED_RGB


# ---------------------------------------------------------------------------
# crop_source was declared in three defaults functions and read by none.
# ---------------------------------------------------------------------------

def test_crop_source_is_declared_by_the_settings_the_wiring_serves():
    from spacr.settings import (set_default_umap_image_settings,
                                set_default_train_test_model,
                                set_generate_dataset_defaults)
    for fn in (set_default_umap_image_settings, set_default_train_test_model,
               set_generate_dataset_defaults):
        assert fn({}).get("crop_source") == "auto", fn.__name__


# ---------------------------------------------------------------------------
# LazyCropPNG -- the object that lets an on-demand crop sit where a path did
# ---------------------------------------------------------------------------

def test_lazy_crop_png_opens_as_an_image_without_touching_disk(project):
    from spacr.io import LazyCropPNG
    source = crops.resolve_crop_source(project, prefer="merged")
    row = {"path_name": os.path.join(project, "merged", FIELDS[0]),
           "object_label": 1, "object_type": "cell"}
    handle = LazyCropPNG(source, row, name="plate1_A01_1_1.png")

    # Nothing produced yet.
    assert handle._buf is None
    img = Image.open(handle)
    assert img.mode == "RGB" and img.size == (48, 48)
    assert np.array_equal(np.array(img), source.get(row))
    # ...and it is a stream, not a path: PIL must not have treated it as one.
    assert not isinstance(handle, (str, bytes, os.PathLike))
    assert handle.readable() and handle.seekable() and not handle.writable()
    handle.close()
    assert handle._buf is None
    # A closed handle produces its bytes again rather than staying dead.
    assert len(handle.png_bytes()) > 0
    assert "plate1_A01_1_1.png" in repr(handle)


def test_lazy_crop_png_falls_back_to_the_bytes_on_disk(project):
    """A crop spacr.crops cannot decode must not take the whole figure down."""
    from spacr.io import LazyCropPNG
    png = _crop_pngs(project)[0]
    source = crops.resolve_crop_source(project, prefer="png")
    handle = LazyCropPNG(source, {"png_path": png}, name=os.path.basename(png))
    assert np.array_equal(np.array(Image.open(handle)),
                          crops.read_crop_png(png))

    class _Broken:
        kind = "png"

        def get(self, row):
            raise crops.CropError("cannot decode")

        def resolve(self, row):
            return row["png_path"]

    broken = LazyCropPNG(_Broken(), {"png_path": png})
    assert np.array(Image.open(broken)).shape[:2] == (48, 48)


def test_lazy_crop_png_reraises_when_there_is_nothing_to_fall_back_to():
    from spacr.io import LazyCropPNG

    class _Broken:
        kind = "merged"

        def get(self, row):
            raise crops.LabelMissing("no such label")

    with pytest.raises(crops.LabelMissing):
        LazyCropPNG(_Broken(), {}).png_bytes()


# ---------------------------------------------------------------------------
# Naming: an on-demand crop has to carry the name the PNG folder gave it
# ---------------------------------------------------------------------------

def test_crop_png_name_matches_what_measure_actually_wrote(project):
    from spacr.io import crop_png_name
    conn = sqlite3.connect(_db(project))
    rows = conn.execute(
        "SELECT file_name, object_label FROM cell").fetchall()
    on_disk = {os.path.basename(p) for p in _crop_pngs(project)}
    conn.close()
    assert on_disk
    for file_name, label in rows:
        assert crop_png_name(file_name, "cell", label) in on_disk


def test_crop_object_type_reads_the_png_type_setting():
    from spacr.io import crop_object_type
    assert crop_object_type("cell_png") == "cell"
    assert crop_object_type("/x/data/A01/nucleus_png/y.png") == "nucleus"
    assert crop_object_type("cytoplasm_png") == "cytoplasm"
    # A filter that names no object type is about which rows, not which mask.
    assert crop_object_type("plate1_") == "cell"
    assert crop_object_type(None) == "cell"
    assert crop_object_type(None, default="pathogen") == "pathogen"


def test_png_list_rows_that_name_no_single_object_are_dropped(project, capsys):
    """'omulti' / 'onone' crops overlap several objects or none; there is no
    single label to cut, so they must be skipped rather than cut wrongly."""
    from spacr.io import crop_rows_from_png_list
    import pandas as pd
    conn = sqlite3.connect(_db(project))
    png_df = pd.read_sql("SELECT * FROM png_list", conn)
    conn.close()
    png_df.loc[0, "cell_id"] = "omulti"
    png_df.loc[1, "cell_id"] = "onone"
    out = crop_rows_from_png_list(_db(project), png_df, "cell")
    assert len(out) == len(png_df) - 2
    assert "cannot be cut" in capsys.readouterr().out
    assert out["path_name"].notna().all()
    assert out["object_label"].map(lambda v: int(v) == v).all()


def test_object_table_alone_supplies_every_crop(project):
    """No png_list needed: object_label + path_name are on the measurement row."""
    from spacr.io import crop_rows_from_object_table
    rows = crop_rows_from_object_table(_db(project), "cell", verbose=False)
    assert len(rows) == N_CELLS * len(FIELDS)
    assert set(rows["png_name"]) == {os.path.basename(p)
                                     for p in _crop_pngs(project)}
    assert rows["path_name"].notna().all()
    assert crop_rows_from_object_table(_db(project), "nowhere",
                                       verbose=False).empty
    assert crop_rows_from_object_table("/no/such.db", "cell").empty


# ---------------------------------------------------------------------------
# generate_dataset -- the inference tar
# ---------------------------------------------------------------------------

def test_tar_from_merged_is_pixel_identical_to_the_png_folder(project):
    """The whole point: swapping the source must not change a single pixel."""
    from spacr.io import generate_dataset
    tar = generate_dataset({"src": project, "experiment": "ondemand",
                            "file_metadata": "cell_png", "sample": None,
                            "crop_source": "merged"})
    members = _tar_members(tar)
    on_disk = _crop_pngs(project)
    assert len(members) == len(on_disk) == N_CELLS * len(FIELDS)
    for png in on_disk:
        name = os.path.basename(png)
        assert name in members, name
        assert np.array_equal(crops.read_crop_png(png), members[name])


def test_tar_from_merged_carries_the_crop_format_marker(project):
    from spacr.io import TarImageDataset, generate_dataset
    tar = generate_dataset({"src": project, "experiment": "marked",
                            "file_metadata": "cell_png", "sample": None,
                            "crop_source": "merged"})
    with tarfile.open(tar) as archive:
        names = {m.name for m in archive.getmembers()}
        assert crops.CROP_FORMAT_SIDECAR in names
        payload = json.loads(
            archive.extractfile(crops.CROP_FORMAT_SIDECAR).read().decode())
    assert payload["spacr_crop_format"] == crops.CROP_FORMAT_DECLARED_RGB

    dataset = TarImageDataset(tar)
    # The marker is not an image: it must not become a sample.
    assert dataset.crop_format == crops.CROP_FORMAT_DECLARED_RGB
    assert len(dataset) == N_CELLS * len(FIELDS)
    assert all(m.name.endswith(".png") for m in dataset.members)
    img, name = dataset[0]
    assert img.mode == "RGB"


def test_tar_without_a_marker_reports_no_format(tmp_path):
    from spacr.io import TarImageDataset
    tar_path = tmp_path / "legacy.tar"
    buf = BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("plate1_A01_1_1.png")
        info.size = len(buf.getvalue())
        tar.addfile(info, BytesIO(buf.getvalue()))
    dataset = TarImageDataset(str(tar_path))
    assert dataset.crop_format is None
    assert len(dataset) == 1


def test_tar_can_be_built_with_the_png_folder_deleted(project):
    from spacr.io import generate_dataset
    expected = {os.path.basename(p) for p in _crop_pngs(project)}
    _drop_png_folder(project)
    tar = generate_dataset({"src": project, "experiment": "gone",
                            "file_metadata": "cell_png", "sample": None,
                            "crop_source": "merged"})
    assert set(_tar_members(tar)) == expected


def test_auto_uses_the_png_folder_when_there_is_one_and_merged_when_there_is_not(project, capsys):
    from spacr.io import generate_dataset
    generate_dataset({"src": project, "experiment": "auto1",
                      "file_metadata": "cell_png", "sample": None,
                      "crop_source": "auto"})
    assert "png crop source" in capsys.readouterr().out

    _drop_png_folder(project)
    tar = generate_dataset({"src": project, "experiment": "auto2",
                            "file_metadata": "cell_png", "sample": None,
                            "crop_source": "auto"})
    assert "merged crop source" in capsys.readouterr().out
    assert len(_tar_members(tar)) == N_CELLS * len(FIELDS)


def test_a_tar_that_would_have_been_empty_now_fails(project):
    """BUG (fixed): utils.add_images_to_tar swallows a missing file with a
    print, so a tar built against a deleted crop folder was announced as
    'Saved 48 images' while holding none, and the run only failed later,
    inside inference, on an empty dataset."""
    from spacr.io import generate_dataset
    _drop_png_folder(project)
    with pytest.raises(RuntimeError) as excinfo:
        generate_dataset({"src": project, "experiment": "empty",
                          "file_metadata": "cell_png", "sample": None,
                          "crop_source": "png"})
    assert "crop_source='merged'" in str(excinfo.value)


def test_tar_from_merged_works_with_no_png_list_at_all(project):
    from spacr.io import generate_dataset
    expected = {os.path.basename(p) for p in _crop_pngs(project)}
    _drop_png_folder(project)
    conn = sqlite3.connect(_db(project))
    conn.execute("DROP TABLE png_list")
    conn.commit()
    conn.close()
    tar = generate_dataset({"src": project, "experiment": "nopnglist",
                            "file_metadata": "cell_png", "sample": None,
                            "crop_source": "merged"})
    assert set(_tar_members(tar)) == expected


def test_file_metadata_still_selects_rows_on_the_merged_path(project):
    from spacr.io import generate_dataset
    tar = generate_dataset({"src": project, "experiment": "onewell",
                            "file_metadata": "plate1_A01", "sample": None,
                            "crop_source": "merged"})
    names = set(_tar_members(tar))
    assert names and all(n.startswith("plate1_A01") for n in names)
    assert len(names) == N_CELLS


def test_sample_caps_the_merged_tar_too(project):
    from spacr.io import generate_dataset
    random.seed(0)
    tar = generate_dataset({"src": project, "experiment": "sampled",
                            "file_metadata": "cell_png", "sample": 5,
                            "crop_source": "merged"})
    assert len(_tar_members(tar)) == 5


def test_two_crops_sharing_a_name_both_survive_the_tar(project):
    """A tar member name is a key; the second write used to replace the first."""
    from spacr.io import LazyCropPNG, _write_crop_tar
    source = crops.resolve_crop_source(project, prefer="merged")
    merged = os.path.join(project, "merged", FIELDS[0])
    items = [LazyCropPNG(source, {"path_name": merged, "object_label": i,
                                  "object_type": "cell"}, name="same.png")
             for i in (1, 2)]
    out = os.path.join(project, "collide.tar")
    written, skipped = _write_crop_tar(items, out)
    assert (written, skipped) == (2, 0)
    assert len(_tar_members(out)) == 2


def test_a_crop_that_cannot_be_cut_is_counted_not_swallowed(project, capsys):
    from spacr.io import LazyCropPNG, _write_crop_tar
    source = crops.resolve_crop_source(project, prefer="merged")
    merged = os.path.join(project, "merged", FIELDS[0])
    good = LazyCropPNG(source, {"path_name": merged, "object_label": 1,
                                "object_type": "cell"}, name="good.png")
    bad = LazyCropPNG(source, {"path_name": merged, "object_label": 99999,
                               "object_type": "cell"}, name="bad.png")
    written, skipped = _write_crop_tar([good, bad],
                                       os.path.join(project, "mixed.tar"))
    assert (written, skipped) == (1, 1)
    assert "Could not read crop" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# generate_training_dataset -- the train/ + test/ split Classify (CV) trains on
# ---------------------------------------------------------------------------

def _training_settings(project, crop_source, dst_suffix=""):
    return {"src": project, "dataset_mode": "annotation",
            "annotation_column": "test", "annotation_columns": ["test"],
            "png_type": "cell_png", "test_split": 0.25,
            "crop_source": crop_source, "verbose": False}


def test_training_split_from_merged_is_pixel_identical(project):
    from spacr.io import generate_training_dataset
    _annotate(project)
    random.seed(0)
    train_png, _ = generate_training_dataset(_training_settings(project, "png"))
    random.seed(0)
    train_merged, _ = generate_training_dataset(
        _training_settings(project, "merged"))
    assert train_png != train_merged

    copied = sorted(glob.glob(os.path.join(train_png, "*", "*.png")))
    assert copied
    for path in copied:
        twin = path.replace(train_png, train_merged)
        assert os.path.isfile(twin), twin
        assert np.array_equal(crops.read_crop_png(path),
                              crops.read_crop_png(twin))


def test_training_split_can_be_built_with_the_png_folder_deleted(project):
    from spacr.io import generate_training_dataset
    _annotate(project)
    _drop_png_folder(project)
    random.seed(0)
    train, test = generate_training_dataset(
        _training_settings(project, "merged"))
    classes = sorted(os.listdir(train))
    assert classes == ["test_1", "test_2"]
    n = sum(len(glob.glob(os.path.join(d, "*", "*.png"))) for d in (train, test))
    assert n == N_CELLS * len(FIELDS)


def test_training_split_from_the_object_table_alone(project):
    """No png_list, no crop folder: the metadata rules select on well
    metadata, which every measurement row already carries.

    ``cv_group_by`` is pinned to ``"none"`` on purpose. The default is
    ``"well"``, and the two classes here ARE the two wells — every crop in
    ``well_a01`` comes from A01 and every crop in ``well_a02`` from A02 — so a
    well-grouped holdout has to put each class entirely on one side and
    ``generate_training_dataset`` refuses with "Leakage-safe well-grouped
    split leaves class 'well_a01' empty in train". That refusal is correct
    product behaviour, not a bug: this test is about the CROP SOURCE, and
    asking it for a grouping its own class definition makes impossible tested
    nothing about that and failed both alone and in the file.
    """
    from spacr.io import generate_training_dataset
    _drop_png_folder(project)
    conn = sqlite3.connect(_db(project))
    conn.execute("DROP TABLE png_list")
    conn.commit()
    conn.close()
    random.seed(0)
    train, test = generate_training_dataset({
        "src": project, "dataset_mode": "metadata", "png_type": "cell_png",
        "test_split": 0.25, "crop_source": "merged", "verbose": False,
        "cv_group_by": "none",
        "metadata_rules": [
            {"name": "well_a01", "column": "columnID", "op": "==", "value": "c1"},
            {"name": "well_a02", "column": "columnID", "op": "==", "value": "c2"},
        ],
    })
    assert sorted(os.listdir(train)) == ["well_a01", "well_a02"]
    n = sum(len(glob.glob(os.path.join(d, "*", "*.png"))) for d in (train, test))
    assert n == N_CELLS * len(FIELDS)


def test_the_training_tree_records_its_crop_format(project):
    from spacr.io import generate_training_dataset
    _annotate(project)
    random.seed(0)
    train, _ = generate_training_dataset(_training_settings(project, "merged"))
    root = os.path.dirname(train)
    marker = crops.read_crop_folder_marker(root)
    assert marker is not None
    assert marker["spacr_crop_format"] == crops.CROP_FORMAT_DECLARED_RGB
    # The class folders stay clean: they are enumerated both as "the classes"
    # and as "the samples", so a sidecar inside one would be counted as each.
    for cls in os.listdir(train):
        assert all(not f.startswith(".")
                   for f in os.listdir(os.path.join(train, cls)))


def test_copied_crops_keep_the_source_folders_format(project):
    """A byte-for-byte copy of legacy crops is still legacy, and the marker
    written beside it has to say so -- calling it RGB reverses every channel
    name attached to a model trained on it."""
    from spacr.io import generate_training_dataset
    _annotate(project)
    # Unmark the crop folders: unmarked means legacy.
    for sidecar in glob.glob(os.path.join(project, "data", "**",
                                          crops.CROP_FORMAT_SIDECAR),
                             recursive=True):
        os.remove(sidecar)
    crops.clear_crop_format_cache()
    random.seed(0)
    train, _ = generate_training_dataset(_training_settings(project, "png"))
    marker = crops.read_crop_folder_marker(os.path.dirname(train))
    assert marker["spacr_crop_format"] == crops.CROP_FORMAT_LEGACY_BGR


def test_a_dataset_that_mixes_formats_is_left_unmarked_and_says_so(project, capsys, tmp_path):
    from spacr.io import LazyCropPNG, generate_dataset_from_lists
    source = crops.resolve_crop_source(project, prefer="merged")
    merged = os.path.join(project, "merged", FIELDS[0])
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    for i in (1, 2, 3, 4):
        Image.new("RGB", (8, 8)).save(legacy_dir / f"old_{i}.png")
    mixed = [str(p) for p in sorted(legacy_dir.glob("*.png"))] + [
        LazyCropPNG(source, {"path_name": merged, "object_label": i,
                             "object_type": "cell"}, name=f"new_{i}.png")
        for i in (1, 2, 3, 4)]
    out = str(tmp_path / "mixed_ds")
    generate_dataset_from_lists(out, [mixed], ["only"], test_split=0.25,
                                group_by="cell")
    assert "mixes crops of more than one format" in capsys.readouterr().out
    assert crops.read_crop_folder_marker(out) is None
    # ...and every crop still landed.
    assert len(glob.glob(os.path.join(out, "*", "*", "*.png"))) == 8


def test_a_crop_that_cannot_be_written_does_not_lose_the_rest(tmp_path, capsys):
    from spacr.io import generate_dataset_from_lists
    src = tmp_path / "src"
    src.mkdir()
    good = []
    for i in range(4):
        p = src / f"good_{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        good.append(str(p))
    data = good + [str(src / "missing.png")]
    out = str(tmp_path / "ds")
    generate_dataset_from_lists(out, [data], ["only"], test_split=0.2,
                                group_by="cell")
    printed = capsys.readouterr().out
    assert "could not be written" in printed
    assert len(glob.glob(os.path.join(out, "*", "*", "*.png"))) == 4


def test_a_training_split_that_would_have_been_empty_now_fails(project):
    """BUG (fixed): with the crop folder gone, the copy path wrote zero images
    and still returned a train/ and test/ tree, so Classify (CV) went on to
    train on nothing."""
    from spacr.io import generate_training_dataset
    _annotate(project)
    _drop_png_folder(project)
    random.seed(0)
    with pytest.raises(RuntimeError) as excinfo:
        generate_training_dataset(_training_settings(project, "png"))
    assert "crop_source='merged'" in str(excinfo.value)


def test_a_class_that_selected_nothing_is_named_not_a_sklearn_message(tmp_path, capsys):
    """sklearn answers an empty class with 'With n_samples=0, test_size=0.25
    ... the resulting train set will be empty', which names the splitter's
    parameters and not the rule that selected nothing."""
    from spacr.io import generate_dataset_from_lists
    src = tmp_path / "src"
    src.mkdir()
    good = []
    for i in range(4):
        p = src / f"c{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        good.append(str(p))
    out = str(tmp_path / "ds")
    train, test = generate_dataset_from_lists(out, [good, []], ["full", "empty"],
                                              test_split=0.25,
                                              group_by="cell")
    printed = capsys.readouterr().out
    assert "Class 'empty' selected no crops" in printed
    assert "have no training images" in printed
    # The class folder still exists, so the class list matches the tree.
    assert sorted(os.listdir(train)) == ["empty", "full"]
    assert os.listdir(os.path.join(train, "empty")) == []


def test_hidden_files_are_not_training_samples(tmp_path):
    """A crop folder carries a `.spacr_crop_format.json` sidecar; every hidden
    file in a class folder used to be handed to Image.open as a sample."""
    from spacr.io import NoClassDataset, spacrDataset
    root = tmp_path / "train"
    cls = root / "a"
    cls.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (8, 8)).save(cls / f"c{i}.png")
    crops.write_crop_folder_marker(str(cls), fmt=crops.CROP_FORMAT_DECLARED_RGB)
    (cls / ".DS_Store").write_bytes(b"junk")

    assert len(spacrDataset(str(root), ["a"], shuffle=False)) == 3
    assert len(NoClassDataset(str(cls), shuffle=False)) == 3


# ---------------------------------------------------------------------------
# deep_spacr threads crop_source into both builders
# ---------------------------------------------------------------------------

def test_deep_spacr_builds_both_datasets_with_no_png_folder(project, monkeypatch):
    """The Classify (CV) entry point chains generate_training_dataset ->
    train_test_model -> generate_dataset -> apply_model_to_tar. Training and
    inference are stubbed here; what is under test is that both dataset
    builders get the crop source and neither needs the PNG folder."""
    import spacr.deep_spacr as ds

    _annotate(project)
    expected = {os.path.basename(p) for p in _crop_pngs(project)}
    _drop_png_folder(project)

    trained = {}

    def _fake_train(settings):
        trained['src'] = settings['src']
        model = os.path.join(settings['src'], 'model.pth')
        open(model, 'wb').close()
        return model

    monkeypatch.setattr(ds, 'train_test_model', _fake_train)
    monkeypatch.setattr(ds, 'apply_model_to_tar',
                        lambda settings: pytest.fail("inference not stubbed"))

    random.seed(0)
    ds.deep_spacr({
        'src': project, 'crop_source': 'merged',
        'generate_training_dataset': True, 'train': True, 'test': False,
        'dataset_mode': 'annotation', 'annotation_column': 'test',
        'annotation_columns': ['test'], 'png_type': 'cell_png',
        'test_split': 0.25, 'apply_model_to_dataset': False,
        'classes': ['test_1', 'test_2'], 'verbose': False,
    })
    train_dir = os.path.join(trained['src'], 'train')
    assert sorted(os.listdir(train_dir)) == ['test_1', 'test_2']
    assert glob.glob(os.path.join(train_dir, '*', '*.png'))

    from spacr.io import generate_dataset
    tar = generate_dataset({'src': project, 'experiment': 'ds_e2e',
                            'file_metadata': 'cell_png', 'sample': None,
                            'crop_source': 'merged'})
    assert set(_tar_members(tar)) == expected


# ---------------------------------------------------------------------------
# generate_image_umap
# ---------------------------------------------------------------------------

def _spy_thumbnails(monkeypatch):
    """Record every thumbnail that reaches the embedding axes."""
    import spacr.utils as su
    seen = []
    real = su.plot_image

    def _spy(ax, x, y, img, img_zoom, remove_image_canvas=True):
        seen.append(np.array(img.convert("RGB")))
        return real(ax, x, y, img, img_zoom, remove_image_canvas)

    monkeypatch.setattr(su, "plot_image", _spy)
    return seen


def test_umap_draws_the_same_thumbnails_from_either_source(project, monkeypatch):
    from spacr.core import generate_image_umap
    seen = _spy_thumbnails(monkeypatch)

    np.random.seed(0)
    random.seed(0)
    generate_image_umap(_umap_settings(project, "png"))
    from_png = list(seen)
    seen.clear()

    np.random.seed(0)
    random.seed(0)
    generate_image_umap(_umap_settings(project, "merged"))
    from_merged = list(seen)

    assert from_png and len(from_png) == len(from_merged)
    for a, b in zip(from_png, from_merged):
        assert np.array_equal(a, b)


def test_umap_runs_with_no_png_folder_and_no_png_list(project, monkeypatch):
    """The embedding only needs the object table: object_label and path_name
    are on every measurement row, which is what a merged crop is cut from."""
    from spacr.core import generate_image_umap
    _drop_png_folder(project)
    conn = sqlite3.connect(_db(project))
    conn.execute("DROP TABLE png_list")
    conn.commit()
    conn.close()

    seen = _spy_thumbnails(monkeypatch)
    out = generate_image_umap(_umap_settings(project, "merged"))
    assert len(out) == N_CELLS * len(FIELDS)
    assert "cluster" in out.columns
    assert seen, "no thumbnail reached the embedding"
    assert all(img.shape == (48, 48, 3) for img in seen)


def test_removing_cluster_noise_keeps_the_handles_aligned(project, monkeypatch):
    """The frame and the crops behind it must lose exactly the same rows."""
    import spacr.utils as su
    from spacr.core import generate_image_umap

    seen = {}
    real_plot = su.plot_embedding

    def _spy(embedding, image_paths, labels, *a, **k):
        seen["embedding"] = len(embedding)
        seen["paths"] = len(image_paths)
        seen["labels"] = len(labels)
        return real_plot(embedding, image_paths, labels, *a, **k)

    monkeypatch.setattr(su, "plot_embedding", _spy)
    settings = _umap_settings(project, "merged")
    settings["remove_cluster_noise"] = True
    out = generate_image_umap(settings)
    assert seen["embedding"] == seen["paths"] == seen["labels"] == len(out)


def test_umap_results_csv_holds_no_crop_handles(project):
    """The handles are Python objects; leaving the column on the frame would
    write `<LazyCropPNG ...>` into embedding_results.csv."""
    from spacr.core import generate_image_umap
    from spacr.io import CROP_REF_COLUMN
    out = generate_image_umap(_umap_settings(project, "merged"))
    assert CROP_REF_COLUMN not in out.columns
    csv_path = os.path.join(project, "results", "embedding_results.csv")
    assert os.path.isfile(csv_path)
    text = open(csv_path, encoding="utf-8").read()
    assert "LazyCropPNG" not in text


def test_umap_without_thumbnails_needs_no_crops_at_all(project):
    """plot_images=False draws points only, so a project with neither a crop
    folder nor a png_list still embeds."""
    from spacr.core import generate_image_umap
    _drop_png_folder(project)
    conn = sqlite3.connect(_db(project))
    conn.execute("DROP TABLE png_list")
    conn.commit()
    conn.close()
    settings = _umap_settings(project, "merged")
    settings["plot_images"] = False
    out = generate_image_umap(settings)
    assert len(out) == N_CELLS * len(FIELDS)


def test_umap_cuts_the_cell_plane_whatever_visualize_says(project):
    """_read_and_join_tables anchors the join on the cell table, so every
    object_label is a cell label; cutting the nucleus plane with one would
    return a different object, or none, for every point on the map."""
    from spacr.core import generate_image_umap
    from spacr.io import LazyCropPNG
    import spacr.utils as su

    captured = {}
    real = su.plot_embedding

    def _grab(embedding, image_paths, labels, *a, **k):
        captured['paths'] = image_paths
        return real(embedding, image_paths, labels, *a, **k)

    su.plot_embedding = _grab
    try:
        settings = _umap_settings(project, "merged")
        settings["visualize"] = "nucleus"
        generate_image_umap(settings)
    finally:
        su.plot_embedding = real
    handles = [p for p in captured['paths'] if isinstance(p, LazyCropPNG)]
    assert handles
    assert {h.row['object_type'] for h in handles} == {'cell'}


def test_umap_still_refuses_a_database_with_no_png_list_on_the_png_path(project):
    from spacr.core import generate_image_umap
    conn = sqlite3.connect(_db(project))
    conn.execute("DROP TABLE png_list")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError) as excinfo:
        generate_image_umap(_umap_settings(project, "png"))
    msg = str(excinfo.value)
    assert "png_list" in msg and "crop_source='merged'" in msg


def test_validate_umap_source_db_can_be_told_png_list_is_optional(project):
    from spacr.core import _validate_umap_source_db
    conn = sqlite3.connect(_db(project))
    conn.execute("DROP TABLE png_list")
    conn.commit()
    conn.close()
    _validate_umap_source_db(_db(project), ["cell", "png_list"],
                             require_png_list=False)
    with pytest.raises(ValueError):
        _validate_umap_source_db(_db(project), ["cell", "png_list"])


# ---------------------------------------------------------------------------
# open_crop_source / mark_crop_output_folder
# ---------------------------------------------------------------------------

def test_open_crop_source_reports_what_it_chose(project, capsys):
    from spacr.io import open_crop_source
    source = open_crop_source({"crop_source": "merged"}, project)
    assert source.kind == "merged"
    assert "merged crop source" in capsys.readouterr().out
    assert open_crop_source({"src": [project]}).kind == "png"
    assert open_crop_source(project).kind == "png"


def test_a_classifiers_boolean_normalize_does_not_reach_the_crop(project):
    """measure_crop's `normalize` is a [p1, p2] percentile pair;
    train_test_model's is a BOOL meaning 'normalise the tensor'. Forwarding
    the bool would swap the [1, 99] stretch the PNG folder was written with
    for a full 0-100 one and change every pixel of the one path whose whole
    purpose is to be pixel-identical to that folder."""
    from spacr.io import _crop_shape_overrides, open_crop_source
    assert 'normalize' not in _crop_shape_overrides({'normalize': True})
    assert _crop_shape_overrides({'normalize': [1, 99]})['normalize'] == [1, 99]
    assert _crop_shape_overrides({'normalize': False})['normalize'] is False

    classifier_settings = {'crop_source': 'merged', 'normalize': True,
                           'image_size': 224, 'batch_size': 64}
    source = open_crop_source(classifier_settings, project, object_type='cell',
                              verbose=False)
    row = {'path_name': os.path.join(project, 'merged', FIELDS[0]),
           'object_label': 1, 'object_type': 'cell'}
    on_disk = [p for p in _crop_pngs(project)
               if os.path.basename(p) == 'plate1_A01_1_1.png'][0]
    assert np.array_equal(source.get(row), crops.read_crop_png(on_disk))


def test_the_run_can_ask_for_a_crop_size_the_folder_never_held(project):
    """The stale-folder answer: the PNG folder holds 48 px crops, and a run
    that asks for 96 px gets 96 px without re-running Measure."""
    from spacr.io import open_crop_source
    source = open_crop_source({'crop_source': 'merged', 'png_size': [96, 96]},
                              project, object_type='cell', verbose=False)
    row = {'path_name': os.path.join(project, 'merged', FIELDS[0]),
           'object_label': 1, 'object_type': 'cell'}
    assert source.get(row).shape == (96, 96, 3)
    # ...and the folder on disk is untouched.
    assert Image.open(_crop_pngs(project)[0]).size == (48, 48)


def test_open_crop_source_returns_none_rather_than_raising(tmp_path, capsys):
    from spacr.io import open_crop_source
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert open_crop_source({"crop_source": "auto"}, str(empty)) is None
    assert "no crop source available" in capsys.readouterr().out
    assert open_crop_source({"crop_source": "telepathy"}, str(empty)) is None
    assert open_crop_source({}, None) is None
    assert open_crop_source({"src": []}) is None


def test_marking_a_folder_that_cannot_be_written_warns_instead_of_raising(tmp_path, capsys):
    from spacr.io import mark_crop_output_folder
    # A file where the folder should be: makedirs inside the marker writer
    # fails, and failing a whole training run over a 300-byte sidecar helps
    # nobody -- but it must not be silent either.
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    assert mark_crop_output_folder(str(blocked)) is None
    assert "could not stamp the crop format" in capsys.readouterr().out


def test_mark_crop_output_folder_inherits_from_the_source(tmp_path):
    from spacr.io import mark_crop_output_folder
    src = tmp_path / "legacy_src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    # Unmarked source == legacy, and the copy has to be marked legacy too.
    mark_crop_output_folder(str(dst), source_folder=str(src))
    assert crops.read_crop_folder_marker(str(dst))["spacr_crop_format"] == \
        crops.CROP_FORMAT_LEGACY_BGR

    crops.write_crop_folder_marker(str(src), fmt=crops.CROP_FORMAT_DECLARED_RGB)
    crops.clear_crop_format_cache()
    dst2 = tmp_path / "dst2"
    dst2.mkdir()
    mark_crop_output_folder(str(dst2), source_folder=str(src))
    assert crops.read_crop_folder_marker(str(dst2))["spacr_crop_format"] == \
        crops.CROP_FORMAT_DECLARED_RGB


# ---------------------------------------------------------------------------
# The two vocabularies for the same idea (instruction 37)
# ---------------------------------------------------------------------------
# `spacr.settings` writes the user-facing words -- 'pre_generated' for the
# PNGs already on disk, 'on_demand' to cut them now from the merged stacks.
# `crops.resolve_crop_source` speaks in terms of the source it builds: 'png'
# or 'merged'. NOTHING TRANSLATED BETWEEN THEM.
#
# So `crop_source='on_demand'`, which the Classify screen validates and
# accepts, reached `resolve_crop_source`, raised CropError, and the error was
# swallowed -- printed only under `verbose`, which every shipped caller
# leaves False. `open_crop_source` returned None and the run fell back to the
# pre-cut PNGs. The user asked for on-demand crops and got a classifier
# trained on different data, with nothing in the log to say so.

import pytest

from spacr.io import CROP_SOURCE_ALIASES, _canonical_crop_source, open_crop_source


@pytest.mark.parametrize("spoken,understood", [
    ("pre_generated", "png"),
    ("generate", "png"),
    ("on_demand", "merged"),
    ("png", "png"),
    ("merged", "merged"),
    ("auto", "auto"),
])
def test_the_settings_vocabulary_translates(spoken, understood):
    assert _canonical_crop_source(spoken) == understood


def test_an_unset_source_is_auto():
    assert _canonical_crop_source(None) == "auto"
    assert _canonical_crop_source("") == "auto"


def test_an_unknown_word_is_passed_through_not_coerced():
    """Quietly substituting a default is how the original defect behaved.

    Passing it through means `resolve_crop_source` raises and NAMES it, which
    is what a typo deserves.
    """
    assert _canonical_crop_source("nonsense") == "nonsense"


def test_every_alias_resolves_to_something_the_resolver_accepts():
    from spacr import crops
    import inspect

    source = inspect.getsource(crops.resolve_crop_source)
    for understood in set(CROP_SOURCE_ALIASES.values()):
        assert f"'{understood}'" in source, understood


def test_on_demand_reaches_the_merged_stacks(project):
    """The end-to-end claim: on_demand must NOT return the PNG source."""
    from spacr import crops

    source = open_crop_source({"src": project, "crop_source": "on_demand"},
                              verbose=False)
    assert source is not None, (
        "on_demand still resolves to nothing; it will fall back to PNGs")
    assert not isinstance(source, crops.PngCropSource), (
        "on_demand resolved to the pre-generated PNG source, which is the "
        "silent substitution this fixes")


def test_pre_generated_still_reaches_the_pngs(project):
    from spacr import crops

    source = open_crop_source({"src": project,
                               "crop_source": "pre_generated"}, verbose=False)
    assert isinstance(source, crops.PngCropSource)


def test_the_two_words_resolve_to_different_sources(project):
    """If they came back the same, the setting would be decorative."""
    on_demand = open_crop_source({"src": project,
                                  "crop_source": "on_demand"}, verbose=False)
    pre_gen = open_crop_source({"src": project,
                                "crop_source": "pre_generated"}, verbose=False)
    assert type(on_demand) is not type(pre_gen)


def test_an_unusable_crop_source_says_so_without_verbose(project, capsys):
    """It printed only under verbose, and every shipped caller passes False."""
    open_crop_source({"src": project, "crop_source": "nonsense"},
                     verbose=False)
    printed = capsys.readouterr().out
    assert "nonsense" in printed
