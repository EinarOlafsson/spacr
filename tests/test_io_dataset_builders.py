"""Table-driven tests of the training-dataset builders in ``spacr.io``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: crop naming, field selection, the crop tar and its
reader, ``generate_dataset``, the class-list writer, the leakage-safe
validation holdout, the Cellpose dataset levelling, and
``generate_training_dataset`` in metadata and annotation mode.  Every plate
is a tiny synthetic folder (``measurements.db`` + PNG crops) under
``tmp_path``.
"""
from __future__ import annotations

import io as _io
import os
import sqlite3
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

import spacr.io as IO
from spacr.crops import CROP_FORMAT_LEGACY_BGR, CROP_FORMAT_SIDECAR

N = 40


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _png(path, value=200, size=(8, 8)):
    """Write a tiny constant RGB PNG and return its path as a string."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    Image.fromarray(np.full((size[0], size[1], 3), value, np.uint8)).save(
        str(path))
    return str(path)


def _row(i):
    """Plate row (1..5) of crop ``i``."""
    return (i % 10) % 5 + 1


def _col(i):
    """Plate column (1..2) of crop ``i``."""
    return (i % 10) // 5 + 1


def _png_src(root, n=N, prefix="o", extra_cols=None, png_paths=None):
    """``<root>/measurements/measurements.db`` (png_list) + spaCR-named crops.

    Five rows per column, so each column condition spans five wells and a
    well-grouped split can hold one out.  ``test`` is the column number.
    """
    src = Path(root)
    (src / "measurements").mkdir(parents=True, exist_ok=True)
    png_dir = src / "data" / "cell_png"
    real = [_png(png_dir / f"plate1_r{_row(i)}_c{_col(i)}_f1_{prefix}{i + 1}.png",
                 value=10 + i)
            for i in range(n)]
    df = pd.DataFrame({
        "png_path": real if png_paths is None else list(png_paths),
        "cell_id": [f"o{i + 1}" for i in range(n)],
        "plateID": ["plate1"] * n,
        "rowID": [f"r{_row(i)}" for i in range(n)],
        "columnID": [f"c{_col(i)}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "test": [_col(i) for i in range(n)],
    })
    for key, values in (extra_cols or {}).items():
        df[key] = values
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        df.to_sql("png_list", con, index=False, if_exists="replace")
    return str(src)


def _gtd(src, **over):
    """generate_training_dataset settings in metadata mode, 20 % test."""
    settings = {"src": src, "dataset_mode": "metadata",
                "png_type": "cell_png", "test_split": 0.2}
    settings.update(over)
    return settings


def _counts(train_dir, test_dir):
    """``{class: (n train png, n test png)}``."""
    return {cls: (len(list((Path(train_dir) / cls).glob("*.png"))),
                  len(list((Path(test_dir) / cls).glob("*.png"))))
            for cls in sorted(os.listdir(train_dir))}


def _db_of(src):
    return os.path.join(src, "measurements", "measurements.db")


def _well_rows(root, wells, column="c1", annotation=None, extra=None):
    """Two crops (o1, o2) per well in ``plate1_<well>_1``, written to disk."""
    rows = []
    for well in wells:
        for obj in (1, 2):
            field = f"plate1_{well}_1"
            rows.append(dict(
                png_path=os.path.join(root, "data", "plate1_r1", "cell_png",
                                      f"{field}_o{obj}.png"),
                plateID="plate1", rowID="r1", columnID=column, fieldID="f1",
                cell_id=f"o{obj}", test=annotation, **(extra or {})))
    return rows


def _project(root, rows, write_crops=True):
    """A plate folder whose png_list is ``rows`` (crops written unless told)."""
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    if write_crops:
        for row in rows:
            if row.get("png_path") and not os.path.exists(row["png_path"]):
                _png(row["png_path"])
    with sqlite3.connect(_db_of(root)) as con:
        pd.DataFrame(rows).to_sql("png_list", con, index=False,
                                  if_exists="replace")
    return root


def _merged_array():
    """(32, 32, 4) uint16 field: three flat channels + a label plane."""
    array = np.zeros((32, 32, 4), np.uint16)
    array[:, :, 0], array[:, :, 1], array[:, :, 2] = 100, 120, 140
    labels = np.zeros((32, 32), np.uint16)
    labels[4:12, 4:12] = 1
    labels[20:28, 20:28] = 2
    array[:, :, 3] = labels
    return array


_MERGED = {"crop_source": "merged", "png_dims": [0, 1, 2],
           "png_size": [16, 16], "cell_mask_dim": 3, "crop_mode": ["cell"],
           "normalize": False}


def _pngs_in(folder):
    """Sorted PNG names in ``folder`` (format markers excluded)."""
    return sorted(n for n in os.listdir(folder) if n.endswith(".png"))


def _tar_names(path):
    with tarfile.open(path) as tar:
        return sorted(tar.getnames())


# ---------------------------------------------------------------------------
# crop_png_name / select_fields
# ---------------------------------------------------------------------------

def test_crop_png_name_puts_the_parent_cell_between_field_and_label(subtests):
    """Cells are ``<stem>_<label>``; children ``<stem>_<cell or none>_<label>``."""
    cases = [
        ("cell", ("plate1_A01_1", "cell", 5), "plate1_A01_1_5.png"),
        ("nucleus with parent, .npy stem",
         ("plate1_A01_1.npy", "nucleus", 7, 3), "plate1_A01_1_3_7.png"),
        ("pathogen with o-prefixed parent",
         ("plate1_A01_1", "pathogen", 2, "o11"), "plate1_A01_1_11_2.png"),
        ("unresolvable parent", ("plate1_A01_1", "nucleus", 4, "omulti"),
         "plate1_A01_1_none_4.png"),
        ("no parent", ("plate1_A01_1", "nucleus", 4, None),
         "plate1_A01_1_none_4.png"),
    ]
    for case_id, args, expected in cases:
        with subtests.test(msg=case_id):
            assert IO.crop_png_name(*args) == expected


def test_select_fields_matches_any_spelling_of_a_field(subtests, monkeypatch):
    """Field requests normalise ``3``/``f3``/``F003``; junk matches literally."""
    stacks = ["plate1_A01_f1.npy", "plate1_A01_f2.npy", "plate1_A02_f10.npy",
              "not_a_field_stem.npy"]
    f34 = ["plate1_A01_f3.npy", "plate1_A01_f4.npy"]
    cases = [
        ("int", f34, 3, ["plate1_A01_f3.npy"]),
        ("digit string", f34, "3", ["plate1_A01_f3.npy"]),
        ("f-prefixed", f34, "f3", ["plate1_A01_f3.npy"]),
        ("padded upper case", f34, "F003", ["plate1_A01_f3.npy"]),
        ("comma list keeps the order", stacks, "f1, f2",
         ["plate1_A01_f1.npy", "plate1_A01_f2.npy"]),
        ("only separators is no filter", stacks, " , , ", stacks),
        ("empty list is no filter", stacks, [], stacks),
        ("unparseable token matches nothing", stacks, "banana", []),
        ("glob skips names that are not field stems", stacks, "f1*",
         ["plate1_A01_f1.npy", "plate1_A02_f10.npy"]),
    ]
    for case_id, names, fields, expected in cases:
        with subtests.test(msg=case_id):
            assert IO.select_fields(names, fields) == expected

    with subtests.test(msg="a schema that refuses the token falls back"):
        import spacr.schema as schema

        def refuses(_token):
            raise ValueError("unrecognised field token")

        monkeypatch.setattr(schema, "field_index", refuses)
        assert IO.select_fields(stacks, "f1") == ["plate1_A01_f1.npy"]


# ---------------------------------------------------------------------------
# _write_crop_tar / TarImageDataset
# ---------------------------------------------------------------------------

class _ConstantSource:
    """A crop source that cuts a constant 4x4 RGB square."""

    kind = "merged"

    def get(self, _row):
        return np.full((4, 4, 3), 200, dtype=np.uint8)


def test_write_crop_tar_counts_written_and_skipped_crops(
        subtests, tmp_path, capsys):
    """PNG paths and lazy crops share one tar; unreadable ones are counted."""
    d = tmp_path / "cell_png"
    on_disk = _png(d / "plate1_A01_1_1.png", value=7, size=(4, 4))
    lazy = IO.LazyCropPNG(_ConstantSource(), {"object_label": 2},
                          name="plate1_A01_1_2.png")
    same_a = _png(tmp_path / "p1" / "cell_png" / "same.png", value=9)
    same_b = _png(tmp_path / "p2" / "cell_png" / "same.png", value=9)
    missing = [str(tmp_path / "gone" / f"m{i}.png") for i in range(7)]
    cases = [
        # id, items, (written, skipped), png members, "Could not read" lines
        ("path + lazy + deleted",
         [on_disk, lazy, str(d / "deleted.png")], (2, 1),
         ["plate1_A01_1_1.png", "plate1_A01_1_2.png"], 1),
        ("same basename twice", [same_a, same_b], (2, 0), None, 0),
        ("only five missing crops are named", [on_disk] + missing, (1, 7),
         ["plate1_A01_1_1.png"], 5),
    ]
    for i, (case_id, items, counts, members, named) in enumerate(cases):
        with subtests.test(msg=case_id):
            tar = str(tmp_path / f"t{i}.tar")
            capsys.readouterr()
            assert IO._write_crop_tar(items, tar) == counts
            out = capsys.readouterr().out
            assert out.count("Could not read crop") == named
            names = _tar_names(tar)
            assert CROP_FORMAT_SIDECAR in names
            pngs = [n for n in names if n.endswith(".png")]
            if members is None:
                assert len(pngs) == len(set(pngs)) == 2
            else:
                assert pngs == members
    with subtests.test(msg="an existing PNG is copied byte for byte"):
        with tarfile.open(str(tmp_path / "t0.tar")) as t:
            assert t.extractfile("plate1_A01_1_1.png").read() == \
                Path(on_disk).read_bytes()


def _tar_with(path, members, sidecar_bytes=None, add_directory=False):
    """A tar holding grey 4x4 PNG ``members`` plus optional extras."""
    with tarfile.open(path, "w") as tar:
        if add_directory:
            info = tarfile.TarInfo("subdir")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        if sidecar_bytes is not None:
            info = tarfile.TarInfo(CROP_FORMAT_SIDECAR)
            info.size = len(sidecar_bytes)
            tar.addfile(info, _io.BytesIO(sidecar_bytes))
        for name in members:
            buf = _io.BytesIO()
            Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(
                buf, format="PNG")
            info = tarfile.TarInfo(name)
            info.size = len(buf.getvalue())
            tar.addfile(info, _io.BytesIO(buf.getvalue()))
    return str(path)


def test_tar_image_dataset_lists_file_members_and_reads_the_format_marker(
        subtests, tmp_path):
    """Directories are not samples; the sidecar sets or refuses the format."""
    with subtests.test(msg="directory member is skipped"):
        ds = IO.TarImageDataset(_tar_with(tmp_path / "a.tar",
                                          ["plate1_A01_1_1.png"],
                                          add_directory=True))
        assert len(ds) == 1
        assert [m.name for m in ds.members] == ["plate1_A01_1_1.png"]
    with subtests.test(msg="readable marker is surfaced"):
        ds = IO.TarImageDataset(_tar_with(
            tmp_path / "b.tar", ["plate1_A01_1_1.png"],
            sidecar_bytes=b'{"spacr_crop_format": %d}' % CROP_FORMAT_LEGACY_BGR))
        assert ds.crop_format == CROP_FORMAT_LEGACY_BGR
        assert len(ds) == 1
    with subtests.test(msg="corrupt marker refuses to guess"):
        with pytest.raises(ValueError, match="Invalid tar crop marker"):
            IO.TarImageDataset(_tar_with(tmp_path / "c.tar",
                                         ["plate1_A01_1_1.png"],
                                         sidecar_bytes=b"{not json at all"))


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

def _ds_src(root, name="plate1", n=6):
    """A plate whose png_list names ``n`` spaCR-shaped crops on disk."""
    src = root / name
    (src / "measurements").mkdir(parents=True)
    paths = [_png(src / "data" / "cell_png"
                  / f"{name}_r{(i % 10) % 5 + 1}_c{(i % 10) // 5 + 1}_f1_o{i + 1}.png")
             for i in range(n)]
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        pd.DataFrame({"png_path": paths,
                      "cell_id": [f"o{i + 1}" for i in range(n)]}
                     ).to_sql("png_list", con, index=False)
    return str(src)


def test_generate_dataset_tars_the_selected_crops(subtests, tmp_path, capsys):
    """The tar holds exactly the selected crops plus the format marker."""
    a = _ds_src(tmp_path, "plateA", n=3)
    b = _ds_src(tmp_path, "plateB", n=3)
    six = _ds_src(tmp_path, "plate6", n=6)
    four = _ds_src(tmp_path, "plate4", n=4)

    gappy = str(tmp_path / "gappy")
    crop_dir = os.path.join(gappy, "data", "plate1_r1", "cell_png")
    good = _png(os.path.join(crop_dir, "plate1_A01_1_o1.png"))
    not_a_file = os.path.join(crop_dir, "plate1_A03_1_o3.png")
    os.makedirs(not_a_file)
    _project(gappy, [{"png_path": p} for p in
                     (good, os.path.join(crop_dir, "plate1_A02_1_o2.png"),
                      not_a_file)], write_crops=False)

    lazy = str(tmp_path / "lazy")
    os.makedirs(os.path.join(lazy, "merged"))
    np.save(os.path.join(lazy, "merged", "plate1_A01_1.npy"), _merged_array())

    def crop_row(root, field, column):
        return {"png_path": os.path.join(root, "data", "plate1_r1", "cell_png",
                                         f"{field}_o1.png"),
                "cell_id": "o1",
                "path_name": os.path.join(root, "merged", f"{field}.npy"),
                "plateID": "plate1", "rowID": "r1", "columnID": column,
                "fieldID": "f1"}

    _project(lazy, [crop_row(lazy, "plate1_A01_1", "c1"),
                    crop_row(lazy, "plate1_A02_1", "c2")], write_crops=False)

    cases = [
        # id, settings, tar folder, name suffix, png members (None=count only),
        # n png, printed fragments
        ("two sources get a combined name",
         {"src": [a, b], "file_metadata": None, "experiment": "multi",
          "sample": None}, os.path.join(a, "datasets"), "_combined_multi.tar",
         None, 6, []),
        ("sample as a list subsamples",
         {"src": six, "file_metadata": None, "experiment": "listsample",
          "sample": [4]}, os.path.join(six, "datasets"), "_listsample.tar",
         None, 4, []),
        ("missing and non-file crops are counted",
         {"src": gappy, "experiment": "e1", "crop_source": "png"},
         os.path.join(gappy, "datasets"), "_e1.tar",
         ["plate1_A01_1_o1.png"], 1,
         ["2 of 3 selected PNGs were missing", "Saved 1 images"]),
        ("on-demand crops that cannot be cut are counted",
         dict(_MERGED, src=lazy, experiment="e1"),
         os.path.join(lazy, "datasets"), "_e1.tar",
         ["plate1_A01_1_o1.png"], 1,
         ["1 of 2 crops could not be produced"]),
    ]
    for case_id, settings, folder, suffix, members, n_png, said in cases:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            tar = IO.generate_dataset(settings)
            out = capsys.readouterr().out
            assert os.path.dirname(tar) == folder
            assert os.path.basename(tar).endswith(suffix)
            names = _tar_names(tar)
            assert ".spacr_crop_format.json" in names
            pngs = [n for n in names if n.endswith(".png")]
            assert len(pngs) == n_png
            if members is not None:
                assert pngs == members
            for fragment in said:
                assert fragment in out

    with subtests.test(msg="an existing tar is not overwritten"):
        cfg = {"src": four, "file_metadata": None, "experiment": "dup",
               "sample": None}
        first, second = IO.generate_dataset(cfg), IO.generate_dataset(cfg)
        assert first != second
        assert os.path.isfile(first) and os.path.isfile(second)
        assert os.path.basename(second).startswith(
            os.path.basename(first)[:-len(".tar")])

    nothing = str(tmp_path / "nothing")
    _project(nothing, [crop_row(nothing, "plate1_A01_1", "c1")],
             write_crops=False)
    refusals = [
        ("src neither str nor list",
         {"src": Path(six), "file_metadata": None, "experiment": "e",
          "sample": None}, "must be a string or list"),
        ("every on-demand crop failed", dict(_MERGED, src=nothing,
                                             experiment="e1"),
         "No image could be written"),
    ]
    for case_id, settings, match in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(RuntimeError, match=match):
                IO.generate_dataset(settings)

    with subtests.test(msg="settings=None uses the default 'path' source"):
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            meas = tmp_path / "path" / "measurements"
            meas.mkdir(parents=True)
            with sqlite3.connect(meas / "measurements.db") as con:
                con.execute("CREATE TABLE png_list (png_path TEXT)")
            with pytest.raises(RuntimeError, match="No images selected"):
                IO.generate_dataset(None)
        finally:
            os.chdir(cwd)


# ---------------------------------------------------------------------------
# generate_dataset_from_lists
# ---------------------------------------------------------------------------

def test_generate_dataset_from_lists_writes_class_folders_and_split(
        subtests, tmp_path, capsys):
    """Class folders always exist; the marker and split need actual crops."""
    def crops(prefix):
        return [_png(tmp_path / "data" / "cell_png"
                     / f"plate1_{prefix}0{w}_1_o{i}.png")
                for w in (1, 2) for i in (1, 2)]

    nc, pc = crops("A"), crops("B")
    cases = [
        # id, class_data, classes, kwargs, root exists, marker+split written,
        # printed fragment
        ("no classes at all", [], [], {}, False, False, None),
        ("two empty classes", [[], []], ["nc", "pc"], {}, True, False,
         "Class 'nc' selected no crops"),
        ("one populated class", [nc], ["only"], {"test_split": 0.25}, True,
         True, None),
        ("two populated classes", [nc, pc], ["nc", "pc"],
         {"test_split": 0.25}, True, True, None),
    ]
    for i, (case_id, data, classes, kwargs, exists, stamped,
            said) in enumerate(cases):
        with subtests.test(msg=case_id):
            dst = str(tmp_path / f"d{i}")
            capsys.readouterr()
            train, test = IO.generate_dataset_from_lists(dst, data, classes,
                                                         **kwargs)
            assert (train, test) == (os.path.join(dst, "train"),
                                     os.path.join(dst, "test"))
            assert os.path.exists(dst) is exists
            for split in (train, test):
                for k, cls in enumerate(classes):
                    assert os.path.isdir(os.path.join(split, cls))
                    if not data[k]:
                        assert os.listdir(os.path.join(split, cls)) == []
            if exists:
                n_written = sum(len(_pngs_in(os.path.join(s, c)))
                                for s in (train, test) for c in classes)
                assert n_written == sum(len(d) for d in data)
            assert os.path.exists(os.path.join(dst, CROP_FORMAT_SIDECAR)) \
                is stamped
            assert os.path.exists(os.path.join(dst, ".spacr_split.json")) \
                is stamped
            if said:
                assert said in capsys.readouterr().out

    with subtests.test(msg="a class confined to one well is refused"):
        folder = tmp_path / "crops"
        folder.mkdir()

        def named(wells, offset):
            made = []
            for k, well in enumerate(wells):
                p = folder / f"plate1_{well}_1_1_cell_{offset + k}.png"
                p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
                made.append(str(p))
            return made

        with pytest.raises(ValueError) as caught:
            IO.generate_dataset_from_lists(
                str(tmp_path / "out"),
                [named(["A01"] * 6, 1),
                 named(["B01", "B02", "B03", "B04", "B05", "B06"], 100)],
                ["positive", "negative"], test_split=0.5, group_by="well")
        message = str(caught.value)
        for fragment in ("leakage-safe well-grouped split cannot put every "
                         "class in both train and test",
                         "choose a finer level",
                         "a random fallback would report memorisation as "
                         "transfer"):
            assert fragment in message


# ---------------------------------------------------------------------------
# make_validation_holdout
# ---------------------------------------------------------------------------

def test_make_validation_holdout_is_whole_group_stratified_and_seeded(subtests):
    """Groups never straddle the split; the realised fraction is exact or warned."""
    labels = [i % 2 for i in range(80)]
    groups = [f"g{i // 10}" for i in range(80)]
    y, g = np.asarray(labels), np.asarray(groups)
    cases = [
        # id, fraction, seed, realised fraction, warning fragments
        ("honoured 0.25 seed 0", 0.25, 0, 0.25, None),
        ("honoured 0.25 seed 3", 0.25, 3, 0.25, None),
        ("honoured 0.25 seed 12345", 0.25, 12345, 0.25, None),
        ("large fraction clamped", 0.9, 0, 0.5, ["0.9", "0.5"]),
        ("group granularity", 0.05, 0, 0.125,
         ["0.05", "0.125", "8 distinct group"]),
    ]
    for case_id, fraction, seed, realised, warned in cases:
        with subtests.test(msg=case_id):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                train, val = IO.make_validation_holdout(labels, fraction,
                                                        groups, seed=seed)
            user = [str(w.message) for w in record
                    if issubclass(w.category, UserWarning)]
            assert len(val) / len(labels) == realised
            assert set(g[train]).isdisjoint(set(g[val]))
            assert sorted(train.tolist() + val.tolist()) == list(range(80))
            assert set(y[val]) == {0, 1}
            if warned is None:
                assert user == []
            else:
                assert all(fragment in user[0] for fragment in warned)

    with subtests.test(msg="one seed is reproducible, another moves the split"):
        a = IO.make_validation_holdout(labels, 0.25, groups, seed=7)
        b = IO.make_validation_holdout(labels, 0.25, groups, seed=7)
        assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()
        assert IO.make_validation_holdout(labels, 0.25, groups, seed=0)[1].tolist() != \
            IO.make_validation_holdout(labels, 0.25, groups, seed=12345)[1].tolist()

    refusals = [
        ("too few groups", [0, 1, 0, 1], 0.5, ["a", "b"], "one group per sample"),
        ("no groups", [0, 1, 0, 1], 0.5, None, "one group per sample"),
        ("one distinct group", [0, 1, 0, 1], 0.5, ["a"] * 4, "two distinct groups"),
        ("fraction 0", [0, 1, 0, 1], 0.0, list("aabb"), "strictly between 0 and 1"),
        ("fraction 1", [0, 1, 0, 1], 1.0, list("aabb"), "strictly between 0 and 1"),
        ("fraction 1.5", [0, 1, 0, 1], 1.5, list("aabb"), "strictly between 0 and 1"),
        ("negative fraction", [0, 1, 0, 1], -0.2, list("aabb"),
         "strictly between 0 and 1"),
    ]
    for case_id, labs, fraction, grps, match in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=match):
                IO.make_validation_holdout(labs, fraction, grps)


# ---------------------------------------------------------------------------
# prepare_cellpose_dataset
# ---------------------------------------------------------------------------

def _pairs(root, name, n, marker=None):
    """``n`` image/mask pairs; image pixel [0, 0] carries ``marker + i``."""
    d = root / name
    (d / "masks").mkdir(parents=True)
    for i in range(n):
        if marker is None:
            image = np.full((8, 8), i + 1, np.uint16)
            mask = np.zeros((8, 8), np.uint16)
            mask[1:4, 1:4] = 1
        else:
            image = np.zeros((4, 4), np.uint8)
            image[0, 0] = marker + i
            mask = image
        tifffile.imwrite(str(d / f"{name}_{i}.tif"), image)
        tifffile.imwrite(str(d / "masks" / f"{name}_{i}.tif"), mask)


def _cellpose_out(root):
    """``{(subset, kind): n tif}`` under ``root/cellpose_dataset``."""
    out = root / "cellpose_dataset"
    return {(s, k): len(list((out / s / k).glob("*.tif")))
            for s in ("train", "test") for k in ("images", "masks")}


def test_prepare_cellpose_dataset_levels_folders_then_splits(
        subtests, tmp_path, monkeypatch, capsys):
    """Every folder contributes the same number of pairs, then train/test split."""
    cases = [
        # id, folders {name: n}, kwargs, expected counts train/test
        ("augment pads the small folder", {"big": 12, "small": 6},
         {"augment_data": True, "train_fraction": 0.75, "n_jobs": 1}, 18, 6),
        ("no augmentation samples down", {"small": 2, "big": 5},
         {"augment_data": False, "train_fraction": 1.0, "n_jobs": 1}, 4, 0),
        ("n_jobs=None uses cpu_count", {"only": 4},
         {"augment_data": False, "train_fraction": 0.5, "n_jobs": None}, 2, 2),
    ]
    monkeypatch.setattr(IO, "cpu_count", lambda: 2)
    for i, (case_id, folders, kwargs, n_train, n_test) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"c{i}"
            for name, n in folders.items():
                _pairs(root, name, n)
            IO.prepare_cellpose_dataset(str(root), **kwargs)
            assert _cellpose_out(root) == {
                ("train", "images"): n_train, ("train", "masks"): n_train,
                ("test", "images"): n_test, ("test", "masks"): n_test}
            first = sorted((root / "cellpose_dataset" / "train" / "images")
                           .glob("*.tif"))[0]
            assert tifffile.imread(str(first)).shape == (8, 8)

    with subtests.test(msg="augmentation grows a folder by exactly the gap"):
        root = tmp_path / "exact"
        _pairs(root, "small", 3, marker=10)
        _pairs(root, "big", 7, marker=100)
        IO.prepare_cellpose_dataset(str(root), augment_data=True, n_jobs=1)
        out = root / "cellpose_dataset"
        images = [tifffile.imread(str(p)) for s in ("train", "test")
                  for p in sorted((out / s / "images").glob("*.tif"))]
        n_masks = sum(len(list((out / s / "masks").glob("*.tif")))
                      for s in ("train", "test"))
        small = [im for im in images if im.max() < 100]
        originals = [im for im in small if im[0, 0] == im.max()]
        assert (len(images), n_masks, len(small), len(originals)) == \
            (14, 14, 7, 3)

    with subtests.test(msg="unpaired images and pairless folders are dropped"):
        root = tmp_path / "unpaired"
        good, empty = root / "ds_good", root / "ds_empty"
        for d in (good, empty):
            (d / "masks").mkdir(parents=True)
        one = np.ones((4, 4), np.uint16)
        for path in (good / "a.tif", good / "masks" / "a.tif",
                     good / "orphan.tif", empty / "b.tif",
                     empty / "masks" / "other.tif"):
            tifffile.imwrite(str(path), one)
        capsys.readouterr()
        IO.prepare_cellpose_dataset(str(root), augment_data=False,
                                    train_fraction=1.0, n_jobs=1)
        out = capsys.readouterr().out
        assert "Found 1 images" in out and str(good) in out
        assert str(empty) not in out
        assert os.listdir(root / "cellpose_dataset" / "train" / "images") == \
            ["00000.tif"]

    with subtests.test(msg="no dataset folders at all"):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "loose.tif").write_bytes(b"x")
        with pytest.raises(ValueError, match="No valid datasets"):
            IO.prepare_cellpose_dataset(str(empty), n_jobs=1)


# ---------------------------------------------------------------------------
# training_dataset_from_annotation_metadata
# ---------------------------------------------------------------------------

def test_training_dataset_from_annotation_metadata_filters_then_balances(
        subtests, tmp_path):
    """Annotation classes are intersected with one metadata column."""
    db = _db_of(_png_src(tmp_path / "plateM", n=20))
    cases = [
        ("default class_metadata c1/c2", {"annotated_classes": (1, 2),
                                          "metadata_type_by": "columnID",
                                          "class_metadata": None}, [10, 10]),
        ("rowID r1 keeps two of each", {"annotated_classes": (1, 2),
                                        "metadata_type_by": "rowID",
                                        "class_metadata": ["r1"]}, [2, 2]),
        ("one class gets a disjoint random other",
         {"annotated_classes": (1,), "metadata_type_by": "columnID",
          "class_metadata": None}, [10, 10]),
    ]
    for case_id, kwargs, sizes in cases:
        with subtests.test(msg=case_id):
            out = IO.training_dataset_from_annotation_metadata(
                db, str(tmp_path / "dst"), annotation_column="test", **kwargs)
            assert [len(x) for x in out] == sizes
            assert not set(out[0]) & set(out[1])
    with subtests.test(msg="unknown metadata key"):
        with pytest.raises(ValueError, match="Invalid metadata_type_by"):
            IO.training_dataset_from_annotation_metadata(
                db, str(tmp_path / "dst"), metadata_type_by="wellID")


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------

def test_generate_training_dataset_builds_the_requested_classes(
        subtests, tmp_path, monkeypatch, capsys):
    """Each class rule or annotation yields its folder and a 80/20 well split."""
    import spacr.utils as U

    flag = [1 if i < 10 else None for i in range(N)]
    many = [1 if i < 30 else None for i in range(N)]
    grades = ["A" if i % 2 == 0 else "B" for i in range(N)]
    rules = [
        {"name": "colc1", "column": "columnID", "op": "==", "value": "c1"},
        {"name": "colc2", "where": [{"column": "columnID", "op": "in",
                                     "value": ["c2"]}]},
        {"name": "notr2", "where": [{"column": "rowID", "op": "notin",
                                     "value": ["r2"]}]},
        {"name": "lowtest", "where": [{"column": "test", "op": "<=",
                                       "value": 1}]},
        {"name": "hightest", "where": [{"column": "test", "op": ">",
                                        "value": 1}]},
        {"name": "everything"},
    ]
    stored = []
    for i in range(N):
        name = f"plate1_r{_row(i)}_c{_col(i)}_f1_o{i + 1}.png"
        stored.append([None, f"/other/machine/plate9/data/cell_png/{name}",
                       f"data/cell_png/{name}", f"/elsewhere/{name}"][i // 10]
                      or str(tmp_path / "c_repair" / "data" / "cell_png" / name))
    cases = [
        # id, _png_src kwargs, settings overrides, {class: (train, test)} or
        # {class: total} when only the total is fixed, printed fragment
        ("unnamed rules get derived names", {},
         {"metadata_rules": [{"column": "columnID", "op": "==", "value": "c1"},
                             {"column": "columnID", "op": "==", "value": "c2"}]},
         {"columnID==c1": (16, 4), "columnID==c2": (16, 4)}, None),
        ("balance disabled keeps sizes", {},
         {"balance_to_smallest": False, "metadata_rules": [
             {"name": "half", "column": "columnID", "op": "==", "value": "c1"},
             {"name": "all"}]},
         {"all": (32, 8), "half": (16, 4)}, None),
        ("named rules over every operator", {}, {"metadata_rules": rules},
         dict.fromkeys(["colc1", "colc2", "everything", "hightest",
                        "lowtest", "notr2"], 20), None),
        ("png paths are re-rooted, unrepairable ones dropped",
         {"png_paths": stored},
         {"metadata_rules": [{"name": "all"}], "balance_to_smallest": False},
         {"all": 30}, None),
        ("annotation value filter", {},
         {"dataset_mode": "annotation", "annotation_columns": ["test"],
          "annotation_values": {"test": [1]}}, {"test_1": 20},
         None),
        ("retired measurement mode maps to annotation", {},
         {"dataset_mode": "measurement", "annotation_column": "test",
          "annotated_classes": [1, 2]}, {"test_1": 20, "test_2": 20}, None),
        ("non-numeric annotations are used verbatim",
         {"extra_cols": {"grade": grades}},
         {"dataset_mode": "annotation", "annotation_columns": ["grade"]},
         {"grade_A": (16, 4), "grade_B": (16, 4)}, None),
        ("fully annotated column invents no random class",
         {"extra_cols": {"allone": [1] * N}},
         {"dataset_mode": "annotation", "annotation_columns": ["allone"]},
         {"allone_1": 40}, "no unannotated rows available"),
        ("more positives than negatives balances down",
         {"extra_cols": {"many": many}},
         {"dataset_mode": "annotation", "annotation_columns": ["many"]},
         {"many_1": (8, 2), "many_random": (8, 2)}, None),
        ("single value gains a random class",
         {"extra_cols": {"flag": flag}},
         {"dataset_mode": "annotation",
          "annotation_columns": ["flag", "missing_column"]},
         {"flag_1": (8, 2), "flag_random": (8, 2)}, None),
    ]
    for i, (case_id, build, over, expected, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / ("c_repair" if "png_paths" in build else f"c{i}")
            src = _png_src(root, **build)
            capsys.readouterr()
            train_dir, test_dir = IO.generate_training_dataset(_gtd(src, **over))
            out = capsys.readouterr().out
            assert train_dir == os.path.join(src, "datasets", "training",
                                             "train")
            counts = _counts(train_dir, test_dir)
            assert sorted(counts) == sorted(expected)
            for cls, want in expected.items():
                if isinstance(want, tuple):
                    assert counts[cls] == want, cls
                else:
                    assert sum(counts[cls]) == want, cls
                    assert 0.15 <= counts[cls][1] / want <= 0.30, cls
            if said:
                assert said in out

    with subtests.test(msg="existing destinations get a _<n> suffix"):
        src = _png_src(tmp_path / "suffix")
        os.makedirs(os.path.join(src, "datasets", "training"))
        os.makedirs(os.path.join(src, "datasets", "training_1"))
        real_save = U.save_settings

        def flaky_save(settings, name="settings", show=False):
            if not show:
                raise RuntimeError("settings volume went away")
            return real_save(settings, name, show)

        monkeypatch.setattr(U, "save_settings", flaky_save)
        capsys.readouterr()
        train_dir, _ = IO.generate_training_dataset(_gtd(src, metadata_rules=[
            {"name": "all"}]))
        assert train_dir == os.path.join(src, "datasets", "training_2", "train")
        assert "Creating new directory for training" in capsys.readouterr().out

    with subtests.test(msg="two sources align annotation columns by index"):
        src_a = _png_src(tmp_path / "plateA", n=20, prefix="a", extra_cols={
            "alpha": [1] * 10 + [2] * 10, "beta": [1] * 10 + [2] * 10})
        src_b = _png_src(tmp_path / "plateB", n=20, prefix="b", extra_cols={
            "alpha": [1] * 10 + [2] * 10})
        capsys.readouterr()
        train_dir, test_dir = IO.generate_training_dataset(_gtd(
            [src_a, src_b], dataset_mode="annotation",
            annotation_columns=["alpha", "beta"]))
        out = capsys.readouterr().out
        assert "class name/order mismatch" in out
        assert "annotation column 'beta' not in png_list" in out
        assert train_dir == os.path.join(src_a, "datasets", "training_all",
                                         "train")
        assert {k: sum(v) for k, v in _counts(train_dir, test_dir).items()} == \
            dict.fromkeys(["alpha_1", "alpha_2", "beta_1", "beta_2"], 10)

    with subtests.test(msg="class_metadata given as a CSV string"):
        src = str(tmp_path / "csv")
        _project(src, _well_rows(src, ["A01", "A02"], column="c1")
                 + _well_rows(src, ["A03", "A04"], column="c2"))
        train_dir, test_dir = IO.generate_training_dataset({
            "src": src, "dataset_mode": "metadata", "class_metadata": "c1, c2",
            "metadata_type_by": "columnID", "test_split": 0.25,
            "random_seed": 1})
        assert sorted(os.listdir(train_dir)) == ["c1", "c2"]
        c1 = [n for d in (train_dir, test_dir)
              for n in _pngs_in(os.path.join(d, "c1"))]
        assert sorted(c1) == sorted(
            f"plate1_{w}_1_o{o}.png" for w in ("A01", "A02") for o in (1, 2))

    with subtests.test(msg="per-cell split of selected classes"):
        src = str(tmp_path / "cells")
        _project(src, _well_rows(src, ["A01", "A02"]))
        train_dir, _ = IO.generate_training_dataset({
            "src": src, "dataset_mode": "metadata",
            "class_metadata": [["c1"]], "metadata_type_by": "columnID",
            "test_split": 0.5, "cv_group_by": "cell", "random_seed": 1})
        assert {c: len(_pngs_in(os.path.join(train_dir, c)))
                for c in os.listdir(train_dir)} == {"c1": 2}
        train_dir, _ = IO.generate_training_dataset({
            "src": src, "dataset_mode": "metadata",
            "metadata_rules": [
                {"name": "ones", "where": [{"column": "cell_id", "op": "==",
                                            "value": "o1"}]},
                {"name": "twos", "where": [{"column": "cell_id", "op": "==",
                                            "value": "o2"}]}],
            "test_split": 0.5, "cv_group_by": "cell", "random_seed": 1})
        assert sorted(os.listdir(train_dir)) == ["ones", "twos"]


def test_generate_training_dataset_refuses_selections_that_cannot_train(
        subtests, tmp_path):
    """Every impossible request is refused with a message naming the cause."""
    from spacr.training_basis import RETIRED_BASES, TrainingBasisError

    assert RETIRED_BASES["measurement"] == "annotation"
    src = _png_src(tmp_path / "plate1")
    ghost_dir = tmp_path / "ghost"
    ghost_paths = [None if i < 10 else
                   str(ghost_dir / "data" / "cell_png" / f"o{i + 1}.png")
                   for i in range(N)]
    ghost = _png_src(ghost_dir, png_paths=ghost_paths,
                     extra_cols={"ghost": [1 if i < 10 else None
                                           for i in range(N)]})
    small = str(tmp_path / "small")
    _project(small, _well_rows(small, ["A01", "A02"], annotation=1))
    cases = [
        ("invalid mode", _gtd(src, dataset_mode="nonsense"),
         TrainingBasisError, "is not one of"),
        ("annotation without columns",
         _gtd(src, dataset_mode="annotation", annotation_columns=[],
              annotation_column=None), ValueError,
         "requires at least one annotation_columns"),
        ("annotation column selecting nothing",
         {"src": small, "dataset_mode": "annotation",
          "annotation_columns": ["test"], "annotation_values": {"test": [2]}},
         ValueError, "selected no crops for any class"),
        ("retired mode reaches the class builder",
         {"src": small, "dataset_mode": "measurement",
          "annotation_columns": ["test"], "annotation_values": {"test": [2]}},
         ValueError, "selected no crops for any class"),
        ("positives without paths", _gtd(ghost, dataset_mode="annotation",
                                         annotation_columns=["ghost"],
                                         png_type=""),
         ValueError, "selected no crops"),
        ("empty class_metadata", _gtd(src, metadata_rules=None,
                                      class_metadata=[]),
         ValueError, "requires at least one class_metadata"),
        ("a class value that never occurs",
         {"src": small, "dataset_mode": "metadata",
          "class_metadata": [["zz"]], "metadata_type_by": "columnID"},
         ValueError, "selected no crops for any class"),
        ("metadata column missing", _gtd(src, metadata_rules=None,
                                         metadata_type_by="condition",
                                         class_metadata=["c1"]),
         ValueError, r"'condition'(.|\n)*Classes editor"),
        ("rule naming an unknown column", _gtd(src, metadata_rules=[
            {"name": "ghost", "where": [{"column": "not_a_column", "op": "==",
                                         "value": 1}]}]),
         ValueError, "unknown column 'not_a_column'"),
        ("rule with an unsupported operator", _gtd(src, metadata_rules=[
            {"name": "badop", "where": [{"column": "columnID", "op": "~=",
                                         "value": 1}]}]),
         ValueError, "unsupported operator '~='"),
        ("a condition that is not a mapping",
         {"src": small, "dataset_mode": "metadata", "test_split": 0.25,
          "metadata_rules": [{"name": "ones", "where": ["columnID == c1"]}]},
         ValueError, "conditions must be mappings"),
    ]
    for case_id, settings, exc, match in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(exc, match=match):
                IO.generate_training_dataset(settings)


def test_generate_training_dataset_marks_the_random_class_in_png_list(
        subtests, tmp_path):
    """``<col>_random`` marks exactly the sampled unannotated rows, once."""
    with subtests.test(msg="column added, negatives disjoint from positives"):
        src = _png_src(tmp_path / "p1", extra_cols={
            "flag": [1 if i < 10 else None for i in range(N)]})
        IO.generate_training_dataset(_gtd(
            src, dataset_mode="annotation", annotation_columns=["flag"],
            write_random_annotation_column=True))
        with sqlite3.connect(_db_of(src)) as con:
            marked = pd.read_sql("SELECT flag, flag_random FROM png_list", con)
        assert int(marked["flag_random"].fillna(0).sum()) == 10
        assert marked.loc[marked["flag"].notna(), "flag_random"].isna().all()

    with subtests.test(msg="an existing column is filled, not added twice"):
        src = str(tmp_path / "p2")
        _project(src, _well_rows(src, ["A01", "A02"], annotation=1,
                                 extra={"test_random": None})
                 + _well_rows(src, ["B01", "B02"], annotation=None,
                              extra={"test_random": None}))
        train_dir, _ = IO.generate_training_dataset({
            "src": src, "dataset_mode": "annotation",
            "annotation_columns": ["test"],
            "write_random_annotation_column": True, "test_split": 0.25,
            "random_seed": 1})
        assert sorted(os.listdir(train_dir)) == ["test_1", "test_random"]
        with sqlite3.connect(_db_of(src)) as con:
            columns = [r[1] for r in con.execute('PRAGMA table_info("png_list")')]
            marked = pd.read_sql("SELECT test, test_random FROM png_list", con)
        assert columns.count("test_random") == 1
        assert marked.loc[marked["test"].isna(), "test_random"].astype(
            int).tolist() == [1, 1, 1, 1]
        assert marked.loc[marked["test"].notna(), "test_random"].isna().all()

    with subtests.test(msg="an on-demand crop without a png_path is not marked"):
        src = str(tmp_path / "p3")
        merged = os.path.join(src, "merged")
        os.makedirs(merged)
        rows = []
        for well, annotation in (("A01", 1), ("A02", 1), ("B01", None),
                                 ("B02", None)):
            field = f"plate1_{well}_1"
            np.save(os.path.join(merged, field + ".npy"), _merged_array())
            for obj in (1, 2):
                rows.append({
                    "png_path": None if (well, obj) == ("B02", 2) else
                    os.path.join(src, "data", "plate1_r1", "cell_png",
                                 f"{field}_o{obj}.png"),
                    "plateID": "plate1", "rowID": "r1", "columnID": "c1",
                    "fieldID": "f1", "cell_id": f"o{obj}", "test": annotation,
                    "path_name": os.path.join(merged, field + ".npy")})
        _project(src, rows, write_crops=False)
        train_dir, test_dir = IO.generate_training_dataset(dict(
            _MERGED, src=src, dataset_mode="annotation",
            annotation_columns=["test"], write_random_annotation_column=True,
            png_type="", path_string="", cv_group_by="cell", test_split=0.25,
            random_seed=1))
        written = sorted(n for d in (train_dir, test_dir)
                         for n in _pngs_in(os.path.join(d, "test_random")))
        assert len(written) == 4 and "crop.png" in written
        with sqlite3.connect(_db_of(src)) as con:
            marked = pd.read_sql("SELECT png_path, test_random FROM png_list",
                                 con)
        assert int(marked["test_random"].fillna(0).sum()) == 3
        assert marked.loc[marked["png_path"].isna(), "test_random"].isna().all()
