"""Round-5 coverage for the tail of :mod:`spacr.io`.

What this file pins is the *second* answer each of these functions can give:
the read direction that is neither R1 nor R2, the crop that cannot be cut, the
class column that no rule names, the training folder that already exists, the
4-D TIFF that will not say which axis is time. Every one of them is a path a
green pipeline never walks, which is exactly why they rot -- and several of
them decide what a model is trained on rather than merely what is printed.

CPU-only and offline throughout.
"""
from __future__ import annotations

import os
import sqlite3
import tarfile

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

import spacr.io as IO


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _png(path, value=200, size=(8, 8)):
    """Write a tiny RGB PNG and return its path as a string."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    arr = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# parse_gz_files
# ---------------------------------------------------------------------------

def test_a_read_direction_that_is_neither_r1_nor_r2_is_not_filed(tmp_path):
    """Only R1/R2 are recognised; an R3/index file creates no sample.

    The sample dict feeds the read merger. An empty entry used to survive until
    that consumer indexed a missing mate, raising far away from the unreadable
    filename. Invalid files now contribute nothing; valid files in the same
    directory must still be grouped normally.
    """
    folder = tmp_path / "fastq"
    folder.mkdir()
    for name in ("s1_R1_001.fastq.gz", "s1_R2_001.fastq.gz",
                 "s1_R3_001.fastq.gz", "s2_I1_001.fastq.gz"):
        (folder / name).write_bytes(b"")

    samples = IO.parse_gz_files(str(folder))

    # The positive half matters: valid files are still grouped, while neither
    # invalid-only sample nor the invalid third read appears in the result.
    assert set(samples) == {"s1"}
    assert sorted(samples["s1"]) == ["R1", "R2"]
    assert os.path.basename(samples["s1"]["R1"]) == "s1_R1_001.fastq.gz"


# ---------------------------------------------------------------------------
# _class_column
# ---------------------------------------------------------------------------

def test_the_class_column_is_the_first_rule_that_actually_names_one():
    """Entries that name no column are skipped, not treated as the answer.

    The Classes editor stores one dict per class, and a half-filled row (a
    string placeholder, or a rule whose column has been cleared) is normal
    while the user is typing. Reading it as the column selects on ''  -- which
    matches nothing -- so it has to be stepped over.
    """
    # A non-mapping rule and a mapping with a blank column are both stepped
    # over; the third rule is the one that answers.
    assert IO._class_column({'classes': {
        'a': 'not a rule at all',
        'b': {'value': 'c1'},
        'c': {'column': '  rowID  ', 'value': 'r2'},
    }}) == 'rowID'


def test_classes_that_name_no_column_at_all_fall_back_to_columnid():
    """Exhausting the rules without finding a column gives the default.

    A class definition with no column is not an error -- the shipped default
    selects on ``columnID`` -- but it must not raise or return ''.
    """
    assert IO._class_column({'classes': {'a': {'value': 'c1'},
                                         'b': {'column': ''}}}) == 'columnID'
    # The legacy key still wins over everything when it is set, which is what
    # makes the walk above the fallback rather than the rule.
    assert IO._class_column({'metadata_type_by': 'plateID',
                             'classes': {'a': {'column': 'rowID'}}}) == 'plateID'


# ---------------------------------------------------------------------------
# process_instruction
# ---------------------------------------------------------------------------

def test_a_pair_copied_without_augmentation_is_copied_unchanged(tmp_path):
    """``augment`` falsy copies image and mask through untouched.

    The augmented and un-augmented instructions go into the same pool, so the
    originals have to survive the same code path that rotates the copies --
    an augmentation applied to every entry would train Cellpose on a dataset
    holding no un-rotated example at all.
    """
    img = np.arange(12, dtype=np.uint16).reshape(3, 4)
    msk = (img > 5).astype(np.uint16)
    tifffile.imwrite(str(tmp_path / "i.tif"), img)
    tifffile.imwrite(str(tmp_path / "m.tif"), msk)

    entry = {"src_img": str(tmp_path / "i.tif"),
             "src_msk": str(tmp_path / "m.tif"),
             "dst_img": str(tmp_path / "plain_i.tif"),
             "dst_msk": str(tmp_path / "plain_m.tif"),
             "augment": None}
    assert IO.process_instruction(entry) == 1
    np.testing.assert_array_equal(tifffile.imread(entry["dst_img"]), img)
    np.testing.assert_array_equal(tifffile.imread(entry["dst_msk"]), msk)

    # The same call with an augmentation really does change the pixels, so
    # the equality above is the no-op branch and not a test of nothing.
    rotated = dict(entry, dst_img=str(tmp_path / "rot_i.tif"),
                   dst_msk=str(tmp_path / "rot_m.tif"), augment="rotate90")
    IO.process_instruction(rotated)
    np.testing.assert_array_equal(tifffile.imread(rotated["dst_img"]),
                                  np.rot90(img, k=-1))
    assert tifffile.imread(rotated["dst_img"]).shape == (4, 3)


# ---------------------------------------------------------------------------
# _write_crop_tar
# ---------------------------------------------------------------------------

def test_only_the_first_five_unreadable_crops_are_named(tmp_path, capsys):
    """A folder that has gone missing must not print one line per crop.

    ``generate_dataset`` calls this with every selected crop; a screen of
    60,000 objects whose PNG folder has been deleted would otherwise emit
    60,000 identical lines and bury the summary that follows. Five examples,
    then a count from the caller.
    """
    good = _png(tmp_path / "data" / "cell_png" / "good.png")
    missing = [str(tmp_path / "gone" / f"m{i}.png") for i in range(7)]

    tar_name = str(tmp_path / "out.tar")
    written, skipped = IO._write_crop_tar([good] + missing, tar_name)

    assert (written, skipped) == (1, 7)
    out = capsys.readouterr().out
    assert out.count("Could not read crop") == 5
    with tarfile.open(tar_name) as tar:
        names = tar.getnames()
    # The one readable crop is in the archive, beside the format marker.
    assert "good.png" in names
    assert not [n for n in names if n.startswith("m")]


# ---------------------------------------------------------------------------
# prepare_cellpose_dataset: pairing
# ---------------------------------------------------------------------------

def test_an_image_with_no_mask_and_a_folder_with_no_pairs_are_both_dropped(
        tmp_path, capsys):
    """Unpaired images are skipped, and a folder left with none is not a dataset.

    Cellpose is trained on pairs; an image whose mask was never written would
    otherwise be copied into ``images/`` with nothing in ``masks/`` and shift
    every later pair's index by one. A folder that contributes no pair at all
    must not enter the balancing, whose target is the smallest folder -- a
    zero-length entry there balances every other folder down to nothing.
    """
    root = tmp_path / "root"
    # A dataset with one paired and one unpaired image.
    good = root / "ds_good"
    (good / "masks").mkdir(parents=True)
    tifffile.imwrite(str(good / "a.tif"), np.ones((4, 4), np.uint16))
    tifffile.imwrite(str(good / "masks" / "a.tif"), np.ones((4, 4), np.uint16))
    tifffile.imwrite(str(good / "orphan.tif"), np.ones((4, 4), np.uint16))
    # A dataset with a masks/ folder but no image that pairs with anything.
    empty = root / "ds_empty"
    (empty / "masks").mkdir(parents=True)
    tifffile.imwrite(str(empty / "b.tif"), np.ones((4, 4), np.uint16))
    tifffile.imwrite(str(empty / "masks" / "other.tif"),
                     np.ones((4, 4), np.uint16))

    IO.prepare_cellpose_dataset(str(root), augment_data=False,
                                train_fraction=1.0, n_jobs=1)

    out = capsys.readouterr().out
    assert "Found 1 images" in out and str(good) in out
    assert str(empty) not in out

    copied = sorted(os.listdir(os.path.join(str(root), "cellpose_dataset",
                                            "train", "images")))
    assert len(copied) == 1


# ---------------------------------------------------------------------------
# convert_separate_files_to_yokogawa
# ---------------------------------------------------------------------------

_SEP_REGEX = (r"(?P<wellID>[A-Za-z0-9\-]+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
              r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")


def test_two_source_names_for_one_well_address_do_not_collide(tmp_path,
                                                              capsys):
    """``A01`` and ``a-1`` are the same address; only one may keep it.

    Pass 1 hands every well its real address, and two source spellings of one
    address both ask for ``plate1_A01``. Letting the second keep it would make
    two different source wells write the same output file -- the earlier one
    silently overwritten -- so the second is pushed into pass 2 and given a
    synthetic address instead.
    """
    folder = tmp_path / "raw"
    folder.mkdir()
    tifffile.imwrite(str(folder / "A01_F1_T1_C1_Z1.tif"),
                     np.full((2, 2), 5, np.uint16))
    tifffile.imwrite(str(folder / "a-1_F1_T1_C1_Z1.tif"),
                     np.full((2, 2), 9, np.uint16))

    IO.convert_separate_files_to_yokogawa(str(folder), _SEP_REGEX)

    produced = sorted(p.name for p in folder.glob("plate1_*.tif"))
    # Two inputs, two outputs: nothing was overwritten.
    assert len(produced) == 2
    assert "plate1_A01_T0001F001L01C01.tif" in produced
    log = pd.read_csv(folder / "rename_log.csv")
    assert len(log) == 2
    assert sorted(log["Renamed TIFF"]) == produced
    values = {int(tifffile.imread(str(folder / n)).max()) for n in produced}
    assert values == {5, 9}
    # The loser of the address is told what it became.
    assert "is not a plate address" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# convert_to_yokogawa
# ---------------------------------------------------------------------------

def test_an_already_converted_plate_file_is_not_converted_again(tmp_path):
    """A ``plate``-prefixed TIFF, and a non-image file, are both passed over.

    The conversion runs *in place*, so its own output sits in the folder the
    next run scans. Re-converting it would rename a plate file onto a fresh
    synthetic well and, with two runs, walk the whole plate one address at a
    time.
    """
    folder = tmp_path / "raw"
    folder.mkdir()
    tifffile.imwrite(str(folder / "a_raw.tif"), np.full((2, 2), 3, np.uint16))
    already = folder / "plate9_B02_T0001F001L01C01.tif"
    tifffile.imwrite(str(already), np.full((2, 2), 7, np.uint16))
    (folder / "notes.txt").write_text("not an image")

    IO.convert_to_yokogawa(str(folder))

    log = pd.read_csv(folder / "rename_log.csv")
    assert list(log["Original File"]) == ["a_raw.tif"]
    # The raw file really was converted (so the skip above is a skip, not a
    # run that converted nothing) ...
    converted = folder / log["Renamed TIFF"][0]
    assert int(tifffile.imread(str(converted)).max()) == 3
    # ... and the already-converted file kept its name and its pixels.
    assert int(tifffile.imread(str(already)).max()) == 7


def test_a_four_d_tiff_that_will_not_name_its_axes_is_read_as_time_first(
        tmp_path, monkeypatch, capsys):
    """An unreadable axis declaration is a warning, not a crash.

    The 4-D branch asks tifffile which of the two leading axes is time. A file
    whose series metadata cannot be read at all must still convert -- under
    the documented (T, Z, Y, X) assumption -- and must say so, because if the
    guess is wrong every "timepoint" written is really a z-plane.
    """
    images = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    class _NoSeries:
        """A TiffFile whose series index cannot be read."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def asarray(self):
            return images

        @property
        def series(self):
            raise RuntimeError("series index is corrupt")

    folder = tmp_path / "raw"
    folder.mkdir()
    IO.write_tiff(str(folder / "stack.tif"), images)
    monkeypatch.setattr(IO.tifffile, "TiffFile", lambda path: _NoSeries())

    IO.convert_to_yokogawa(str(folder))

    out = capsys.readouterr().out
    assert "declares axes (none)" in out
    assert "reading it as (T, Z, Y, X)" in out.replace("\n", " ").replace(
        "  ", " ")
    produced = sorted(p.name for p in folder.glob("plate1_A01_T*.tif"))
    assert produced == ["plate1_A01_T0001F001L01C01.tif",
                        "plate1_A01_T0002F001L01C01.tif"]
    # Each output is the projection over z of one timepoint, which is what
    # "read as (T, Z, Y, X)" means.
    np.testing.assert_array_equal(
        tifffile.imread(str(folder / produced[0])), images[0].max(axis=0))


def test_a_lif_image_whose_frames_are_all_missing_writes_nothing(
        tmp_path, monkeypatch, capsys):
    """A channel with no readable z-plane is skipped, not written empty.

    ``np.stack([])`` raises, so an image whose frames all fail would take the
    whole conversion down inside the per-file ledger and lose the files that
    follow it. The channel is dropped with a message instead.

    The double mirrors readlif 0.6.5's real surface -- ``LifFile``,
    ``get_iter_image``, ``get_frame``, and a channel count on the image
    rather than on ``dims`` -- because that is what :mod:`spacr.io` now
    calls. It used to call ``readlif.Reader``/``getIterImage``/``getFrame``,
    an older camelCase API that no longer exists, which meant every LIF
    import died on its first line; see
    tests/test_cov_lif_uses_the_real_readlif_api.py.
    """
    import types

    frame = np.full((4, 4), 6, np.uint16)

    class _FakeImage:
        def __init__(self, readable):
            # Dims is namedtuple("Dims", "x y z t m") -- deliberately no `c`,
            # so a reader that looks for channels here finds nothing.
            self.dims = types.SimpleNamespace(x=4, y=4, z=2, t=1, m=1)
            self.channels = 1
            self._readable = readable

        def get_frame(self, z, t, c):
            if not self._readable:
                raise IndexError("frame not in file")
            return frame

    class _FakeReader:
        def __init__(self, path):
            self._readable = "full" in os.path.basename(path)

        def get_iter_image(self):
            return iter([_FakeImage(self._readable)])

    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "a_dark.lif").write_bytes(b"")
    (folder / "b_full.lif").write_bytes(b"")
    monkeypatch.setattr(IO.readlif.reader, "LifFile", _FakeReader)

    IO.convert_to_yokogawa(str(folder))

    out = capsys.readouterr().out
    assert out.count("Missing frame") == 2      # both z of the broken file
    log = pd.read_csv(folder / "rename_log.csv")
    # The readable file converted; the broken one produced no TIFF at all.
    assert list(log["Original File"]) == ["b_full.lif"]
    assert int(tifffile.imread(str(folder / log["Renamed TIFF"][0])).max()) == 6


# ---------------------------------------------------------------------------
# open_crop_source
# ---------------------------------------------------------------------------

def test_an_explicit_src_beats_a_path_passed_as_the_settings(tmp_path,
                                                             capsys):
    """``open_crop_source(path, src)`` resolves ``src``, not ``path``.

    The first argument doubles as "a settings dict" and "a source path", so a
    caller that passes both has to be answered from the one it named second.
    Resolving the wrong one silently builds a source over somebody else's
    plate.
    """
    usable = tmp_path / "has_merged"
    (usable / "merged").mkdir(parents=True)
    np.save(str(usable / "merged" / "plate1_A01_1.npy"),
            np.zeros((8, 8, 5), np.uint16))
    bare = tmp_path / "bare"
    bare.mkdir()

    # The path on its own resolves, so the folder really is usable ...
    assert IO.open_crop_source(str(usable)).kind == "merged"
    capsys.readouterr()

    # ... and handing it as the settings while naming an unusable src gives
    # the unusable one's answer, naming it.
    assert IO.open_crop_source(str(usable), src=str(bare)) is None
    out = capsys.readouterr().out
    assert str(bare) in out
    assert str(usable) not in out


# ---------------------------------------------------------------------------
# _dataset_crop_refs
# ---------------------------------------------------------------------------

def _png_list_db(path, rows):
    with sqlite3.connect(str(path)) as con:
        pd.DataFrame(rows).to_sql('png_list', con, index=False)
    return str(path)


def _crop_rows(n=2):
    return [{'png_path': f'/x/data/plate1_r1/cell_png/plate1_A0{i}_1_o{i}.png',
             'cell_id': f'o{i}',
             'path_name': f'/x/merged/plate1_A0{i}_1.npy',
             'plateID': 'plate1', 'rowID': 'r1', 'columnID': f'c{i}',
             'fieldID': 'f1'}
            for i in range(1, n + 1)]


def test_without_a_file_metadata_filter_every_crop_is_selected(tmp_path):
    """An unset ``file_metadata`` selects the whole table, not nothing.

    The filter is a substring match on ``png_path``; treating an unset filter
    as "match nothing" would build an empty tar out of a full database, and
    treating it as a literal '' would be right by accident. Both spellings are
    checked against the same table.
    """
    db = _png_list_db(tmp_path / "m.db", _crop_rows())
    source = object()          # only stored on the refs, never called here

    every = IO._dataset_crop_refs(db, source, {}, 'cell', verbose=False)
    assert [ref.name for ref in every] == ['plate1_A01_1_o1.png',
                                           'plate1_A02_1_o2.png']

    # The same table with a filter really does drop a row, so "everything"
    # above is the unfiltered branch and not a filter that never bites.
    one = IO._dataset_crop_refs(db, source, {'file_metadata': 'A01'}, 'cell',
                                verbose=False)
    assert [ref.name for ref in one] == ['plate1_A01_1_o1.png']


def test_a_source_with_no_database_contributes_no_crops(tmp_path):
    """A missing ``measurements.db`` yields an empty list, not an exception.

    ``generate_dataset`` walks several sources and one of them may be a folder
    that was never measured. It has to contribute nothing and let the others
    through, rather than taking the whole dataset build down.
    """
    source = object()
    assert IO._dataset_crop_refs(str(tmp_path / "absent.db"), source, {},
                                 'cell', verbose=False) == []

    # A source that does have the database contributes crops, so the empty
    # answer above is the missing-file branch.
    db = _png_list_db(tmp_path / "m.db", _crop_rows(1))
    assert len(IO._dataset_crop_refs(db, source, {}, 'cell',
                                     verbose=False)) == 1


# ---------------------------------------------------------------------------
# generate_dataset_from_lists
# ---------------------------------------------------------------------------

def test_classes_that_selected_nothing_still_get_their_folders(tmp_path,
                                                               capsys):
    """Empty classes produce empty folders and no crop-format marker.

    The class list downstream is read off the directory tree, so a class whose
    rule matched nothing must still appear -- otherwise the model is built
    with fewer outputs than the settings declare. Nothing was cut, so there is
    no format to stamp.
    """
    from spacr import crops

    dst = tmp_path / "empty_ds"
    train, test = IO.generate_dataset_from_lists(str(dst), [[], []],
                                                 ['nc', 'pc'])

    assert (train, test) == (os.path.join(str(dst), 'train'),
                             os.path.join(str(dst), 'test'))
    assert os.listdir(os.path.join(train, 'nc')) == []
    assert os.listdir(os.path.join(test, 'pc')) == []
    assert not os.path.exists(os.path.join(str(dst), crops.CROP_FORMAT_SIDECAR))
    assert not os.path.exists(os.path.join(str(dst), '.spacr_split.json'))
    out = capsys.readouterr().out
    assert "Class 'nc' selected no crops" in out

    # The same call with crops in it does stamp the format and does record
    # the split, so the two absences above are this branch and not a function
    # that never writes them.
    full = tmp_path / "full_ds"
    nc = [_png(tmp_path / "data" / "cell_png" / f"plate1_A0{w}_1_o{i}.png")
          for w in (1, 2) for i in (1, 2)]
    pc = [_png(tmp_path / "data" / "cell_png" / f"plate1_B0{w}_1_o{i}.png")
          for w in (1, 2) for i in (1, 2)]
    IO.generate_dataset_from_lists(str(full), [nc, pc], ['nc', 'pc'],
                                   test_split=0.25)
    assert os.path.exists(os.path.join(str(full), crops.CROP_FORMAT_SIDECAR))
    assert os.path.exists(os.path.join(str(full), '.spacr_split.json'))


def test_a_dataset_with_no_classes_at_all_writes_nothing(tmp_path):
    """No classes means no tree, and certainly no split provenance.

    ``generate_training_dataset`` refuses an empty class list before it gets
    here, but this function is public and is called directly; it must answer
    with the paths it would have used rather than raising from inside the
    splitter.
    """
    dst = tmp_path / "no_classes"
    train, test = IO.generate_dataset_from_lists(str(dst), [], [])

    assert (train, test) == (os.path.join(str(dst), 'train'),
                             os.path.join(str(dst), 'test'))
    # Nothing was created: not the root, not the split sidecar.
    assert not os.path.exists(str(dst))

    # One populated class does create the root, so the absence is the
    # empty-input branch rather than a function that never writes.
    crops_ = [_png(tmp_path / "data" / "cell_png" / f"plate1_A0{w}_1_o{i}.png")
              for w in (1, 2) for i in (1, 2)]
    IO.generate_dataset_from_lists(str(tmp_path / "one_class"), [crops_],
                                   ['only'], test_split=0.25)
    assert os.path.isdir(str(tmp_path / "one_class" / "train" / "only"))


# ---------------------------------------------------------------------------
# _read_and_merge_data
# ---------------------------------------------------------------------------

def _well(obj_key):
    row, col = f"r{(obj_key % 2) + 1}", f"c{(obj_key % 2) + 1}"
    return {"plateID": "plate1", "rowID": row, "columnID": col,
            "fieldID": "f1", "prcf": f"plate1_{row}_{col}_f1"}


def _object_frame(role, pairs, extra=None):
    """One measurement table: ``pairs`` is (object_label, parent cell_id)."""
    rows = []
    for label, cell in pairs:
        row = dict(_well(cell), object_label=label)
        if role != 'cell' and role != 'cytoplasm':
            row['cell_id'] = cell
        row[f'{role}_area'] = 10.0 + label
        row[f'{role}_channel_0_mean_intensity'] = 100.0 + label
        row.update(extra or {})
        rows.append(row)
    return pd.DataFrame(rows)


def _measure_db(path, tables):
    with sqlite3.connect(str(path)) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, index=False, if_exists='replace')
    return str(path)


def test_a_cytoplasm_only_database_becomes_the_anchor_quietly(tmp_path,
                                                              capsys):
    """With no cell table the cytoplasm table anchors the merge, silently.

    Every shipped caller passes ``verbose=False``, so the anchor selection has
    to work with the reporting switched off -- the loud path is the one the
    tests usually take and the silent one is the one users run.
    """
    cytoplasm = _object_frame('cytoplasm', [(1, 1), (2, 2), (3, 3)])
    db = _measure_db(tmp_path / "m.db", {"cytoplasm": cytoplasm})

    merged, obj_dfs = IO._read_and_merge_data([db], ["cytoplasm"])

    assert len(merged) == 3
    assert 'cytoplasm_area' in merged.columns
    assert all(idx.endswith(('_o1', '_o2', '_o3')) for idx in merged.index)
    assert len(obj_dfs) == 1 and len(obj_dfs[0]) == 3
    # Nothing was printed about the grouping: this is the quiet path.
    assert 'cytoplasms grouped' not in capsys.readouterr().out


def test_a_nuclei_limit_that_is_not_a_number_filters_nothing(tmp_path):
    """``nuclei_limit`` must be True or a number; anything else is inert.

    A settings CSV can hand this through as text. Silently reading '2' as
    "keep at most two" would be a guess; silently reading it as True would
    delete every multinucleate cell. Neither: an unusable value leaves the
    nuclei alone, and the paired call with ``True`` shows the filter really
    does bite.
    """
    cells = _object_frame('cell', [(1, 1), (2, 2)])
    # Cell 1 carries two nuclei, cell 2 exactly one.
    nuclei = _object_frame('nucleus', [(1, 1), (2, 1), (3, 2)])
    db = _measure_db(tmp_path / "m.db", {"cell": cells, "nucleus": nuclei})

    unfiltered, _ = IO._read_and_merge_data([db], ["cell", "nucleus"],
                                            nuclei_limit="2")
    assert sorted(int(v) for v in unfiltered['nucleus_prcfo_count']) == [1, 2]

    single, _ = IO._read_and_merge_data([db], ["cell", "nucleus"],
                                        nuclei_limit=True)
    # nucleus joins inner, so the two-nucleus cell is gone entirely.
    assert len(single) == 1
    assert sorted(int(v) for v in single['nucleus_prcfo_count']) == [1]


def test_a_pathogen_only_database_anchors_quietly_and_ignores_a_text_limit(
        tmp_path, capsys):
    """A pathogen table with no host table anchors the merge on its cell.

    Same two rules as the nuclei limit: an unusable ``pathogen_limit`` filters
    nothing, and the anchor selection has to work with reporting off.
    """
    pathogens = _object_frame('pathogen', [(1, 1), (2, 1), (3, 2)])
    db = _measure_db(tmp_path / "m.db", {"pathogen": pathogens})

    merged, _ = IO._read_and_merge_data([db], ["pathogen"],
                                        pathogen_limit="lots")
    # Two parent cells; the doubly-infected one kept both pathogens.
    assert len(merged) == 2
    assert sorted(int(v) for v in merged['pathogen_prcfo_count']) == [1, 2]
    assert 'pathogens grouped' not in capsys.readouterr().out

    strict, _ = IO._read_and_merge_data([db], ["pathogen"],
                                        pathogen_limit=True)
    assert len(strict) == 1


def test_an_organelle_table_merges_onto_a_cell_table_loudly_and_alone_quietly(
        tmp_path, capsys):
    """The organelle role reports the same way whether or not it is the anchor.

    Organelle slots were bolted on after cell/nucleus/pathogen, so both of its
    branches -- anchor, and merged onto an earlier role -- are newer than the
    code around them and neither is exercised by an ordinary run.
    """
    organelles = _object_frame('organelle', [(1, 1), (2, 1), (3, 2), (4, 3)])

    # Anchor, with reporting off.
    alone_db = _measure_db(tmp_path / "alone.db", {"organelle": organelles})
    alone, _ = IO._read_and_merge_data([alone_db], ["organelle"])
    assert len(alone) == 3                      # three parent cells
    assert 'organelle grouped' not in capsys.readouterr().out

    # Merged onto a cell table, with reporting on.
    cells = _object_frame('cell', [(1, 1), (2, 2), (3, 3)])
    both_db = _measure_db(tmp_path / "both.db",
                          {"cell": cells, "organelle": organelles})
    both, _ = IO._read_and_merge_data([both_db], ["cell", "organelle"],
                                      verbose=True)
    assert len(both) == 3
    assert {'cell_area', 'organelle_area'} <= set(both.columns)
    out = capsys.readouterr().out
    assert 'organelle: 4, organelle grouped: 3' in out


# ---------------------------------------------------------------------------
# generate_cv_loaders
# ---------------------------------------------------------------------------

def test_cross_validation_falls_back_to_the_nc_pc_class_names(tmp_path):
    """``classes=None`` means the shipped two-class screen layout.

    The GUI leaves ``classes`` unset for an ordinary negative/positive screen,
    so the default is what most cross-validation runs actually use. It has to
    name the folders on disk -- an empty or wrongly ordered default reads the
    crops of one class as the other and every fold score is a coin flip.
    """
    for label, (cls, value) in enumerate((("nc", 40), ("pc", 200))):
        folder = tmp_path / "train" / cls
        folder.mkdir(parents=True)
        for well in ("A01", "A02", "A03", "A04"):
            # Distinct object ids per class so a crop can be followed from one
            # call to the next; the loader shuffles.
            _png(folder / f"plate1_{well}_1_o{label + 1}.png", value=value,
                 size=(16, 16))

    folds, info = IO.generate_cv_loaders(str(tmp_path), 2, batch_size=2,
                                         image_size=16, n_jobs=0)

    assert info['classes'] == ['nc', 'pc']
    assert len(folds) == 2
    # Folder order decides the label, so naming the classes the other way
    # round really does relabel the crops -- the default is a choice, not a
    # formality.
    _, flipped = IO.generate_cv_loaders(str(tmp_path), 2, classes=['pc', 'nc'],
                                        batch_size=2, image_size=16, n_jobs=0)

    def _by_crop(details):
        names = IO.dataset_filenames(details['dataset'])
        return {os.path.basename(str(name)): int(label)
                for name, label in zip(names, details['labels'])}

    default_labels, flipped_labels = _by_crop(info), _by_crop(flipped)
    assert set(default_labels) == set(flipped_labels) and len(default_labels) == 8
    assert all(flipped_labels[crop] == 1 - label
               for crop, label in default_labels.items())


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

def _png_source(root, rows):
    """A plate folder holding a measurements.db whose png_list is ``rows``."""
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    with sqlite3.connect(os.path.join(root, 'measurements',
                                      'measurements.db')) as con:
        pd.DataFrame(rows).to_sql('png_list', con, index=False)
    return root


def test_a_tar_reports_the_pngs_that_were_not_in_it(tmp_path, capsys):
    """Missing files and non-file entries are counted, not announced as saved.

    ``add_images_to_tar`` swallows a missing crop with a print, so the count
    of members actually written is the only honest number. A tar announced as
    "Saved 3 images" while holding one fails later, inside inference, on a
    dataset that is silently a third of the size.
    """
    src = str(tmp_path / "plate1")
    crop_dir = os.path.join(src, 'data', 'plate1_r1', 'cell_png')
    good = _png(os.path.join(crop_dir, 'plate1_A01_1_o1.png'))
    gone = os.path.join(crop_dir, 'plate1_A02_1_o2.png')
    # A directory where a crop should be: tarred as a directory member, which
    # is not a file and must not be counted as a written crop.
    not_a_file = os.path.join(crop_dir, 'plate1_A03_1_o3.png')
    os.makedirs(not_a_file)
    _png_source(src, {'png_path': [good, gone, not_a_file]})

    tar_name = IO.generate_dataset({'src': src, 'experiment': 'e1',
                                    'crop_source': 'png'})

    # The destination is derived from the first source, which is why the
    # `dst is None` guard below the selection cannot fire (see the proof
    # section at the end of this file).
    assert tar_name.startswith(os.path.join(src, 'datasets') + os.sep)
    with tarfile.open(tar_name) as tar:
        assert tar.getnames() == ['plate1_A01_1_o1.png']
    out = capsys.readouterr().out
    assert "2 of 3 selected PNGs were missing" in out
    assert "Saved 1 images" in out


def _merged_source(root, fields, rows):
    """A plate folder with merged arrays and a png_list naming them."""
    merged = os.path.join(root, 'merged')
    os.makedirs(merged, exist_ok=True)
    array = np.zeros((32, 32, 4), np.uint16)
    array[:, :, 0], array[:, :, 1], array[:, :, 2] = 100, 120, 140
    labels = np.zeros((32, 32), np.uint16)
    labels[4:12, 4:12] = 1
    array[:, :, 3] = labels
    for name in fields:
        np.save(os.path.join(merged, name), array)
    return _png_source(root, rows)


_MERGED_SETTINGS = {'experiment': 'e1', 'crop_source': 'merged',
                    'png_dims': [0, 1, 2], 'png_size': [16, 16],
                    'cell_mask_dim': 3, 'crop_mode': ['cell'],
                    'normalize': False}


def _crop_row(root, field, column):
    return {'png_path': os.path.join(root, 'data', 'plate1_r1', 'cell_png',
                                     f'{field}_o1.png'),
            'cell_id': 'o1',
            'path_name': os.path.join(root, 'merged', f'{field}.npy'),
            'plateID': 'plate1', 'rowID': 'r1', 'columnID': column,
            'fieldID': 'f1'}


def test_on_demand_crops_that_cannot_be_cut_are_counted_not_included(
        tmp_path, capsys):
    """A merged array that has moved costs its crops, not the whole tar.

    Cutting on demand reaches back into ``merged/*.npy``; a field that is no
    longer there must not abort the tar, but the crops it would have provided
    must not be quietly presented as present either.
    """
    src = str(tmp_path / "plate1")
    # Only the first field's array exists; the second row names a missing one.
    _merged_source(src, ['plate1_A01_1.npy'],
                   [_crop_row(src, 'plate1_A01_1', 'c1'),
                    _crop_row(src, 'plate1_A02_1', 'c2')])

    tar_name = IO.generate_dataset(dict(_MERGED_SETTINGS, src=src))

    with tarfile.open(tar_name) as tar:
        assert 'plate1_A01_1_o1.png' in tar.getnames()
        assert 'plate1_A02_1_o1.png' not in tar.getnames()
    out = capsys.readouterr().out
    assert "1 of 2 crops could not be produced" in out


def test_a_tar_in_which_every_on_demand_crop_failed_is_refused(tmp_path):
    """No crop cut at all is a broken project, not an empty dataset.

    An empty tar passes every check downstream until the first training batch,
    where it surfaces as a DataLoader error naming neither the plate nor the
    merged folder. Refuse it here, where ``merged/`` is known.
    """
    src = str(tmp_path / "plate2")
    # No arrays at all; every row names a field that is not on disk.
    _merged_source(src, [], [_crop_row(src, 'plate1_A01_1', 'c1')])

    with pytest.raises(RuntimeError, match="No image could be written"):
        IO.generate_dataset(dict(_MERGED_SETTINGS, src=src))


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------

def _training_project(root, rows, write_crops=True):
    """A plate folder whose ``png_list`` is ``rows``, with the crops on disk."""
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    if write_crops:
        for row in rows:
            if row.get('png_path'):
                _png(row['png_path'])
    with sqlite3.connect(os.path.join(root, 'measurements',
                                      'measurements.db')) as con:
        pd.DataFrame(rows).to_sql('png_list', con, index=False)
    return root


def _crop_rows_for(root, wells, column='c1', annotation=None, extra=None):
    rows = []
    for well in wells:
        for obj in (1, 2):
            field = f'plate1_{well}_1'
            rows.append(dict(
                png_path=os.path.join(root, 'data', 'plate1_r1', 'cell_png',
                                      f'{field}_o{obj}.png'),
                plateID='plate1', rowID='r1', columnID=column, fieldID='f1',
                cell_id=f'o{obj}', test=annotation, **(extra or {})))
    return rows


def test_a_comma_separated_class_metadata_string_is_two_classes_not_seventeen(
        tmp_path, capsys):
    """``class_metadata`` read from a CSV is parsed, not iterated per character.

    A settings CSV stores the repr of the list, and a caller handing the raw
    string through used to have it iterated one character at a time -- so
    ``"c1, c2"`` became a class per bracket, quote and letter. A string that
    is not a literal falls back to splitting on commas.

    The destination is checked at the same time: two earlier training folders
    already exist beside it, and neither may be written into -- a rerun that
    lands in the previous run's folder mixes two splits.
    """
    src = str(tmp_path / "plate1")
    rows = (_crop_rows_for(src, ['A01', 'A02'], column='c1')
            + _crop_rows_for(src, ['A03', 'A04'], column='c2'))
    _training_project(src, rows)
    os.makedirs(os.path.join(src, 'datasets', 'training'))
    os.makedirs(os.path.join(src, 'datasets', 'training_1'))

    train_dir, _test_dir = IO.generate_training_dataset({
        'src': src, 'dataset_mode': 'metadata', 'class_metadata': "c1, c2",
        'metadata_type_by': 'columnID', 'test_split': 0.25, 'random_seed': 1})

    # Neither existing folder was reused.
    assert os.path.dirname(train_dir).endswith('training_2')
    assert "Creating new directory for training" in capsys.readouterr().out
    # Exactly two classes, and each holds the crops of its own column.
    assert sorted(os.listdir(train_dir)) == ['c1', 'c2']
    test_dir = os.path.join(os.path.dirname(train_dir), 'test')
    everything = {
        (os.path.basename(root), cls): sorted(os.listdir(os.path.join(root, cls)))
        for root in (train_dir, test_dir)
        for cls in ('c1', 'c2')
    }
    assert all(names for names in everything.values())
    # Each class holds only its own column's crops: c1 came from A01/A02.
    assert all(name.startswith(('plate1_A01', 'plate1_A02'))
               for key, names in everything.items() if key[1] == 'c1'
               for name in names)


def test_a_selection_rule_that_is_not_a_mapping_is_refused_by_name(tmp_path):
    """A malformed ``where`` clause names itself instead of raising TypeError.

    The rules arrive from a settings CSV or a JSON payload, so a bare string
    where a condition should be is a plausible typo. ``cond.get`` on it raises
    ``AttributeError`` several frames from anything the user wrote.
    """
    src = str(tmp_path / "plate1")
    _training_project(src, _crop_rows_for(src, ['A01', 'A02']))

    with pytest.raises(ValueError, match="conditions must be mappings"):
        IO.generate_training_dataset({
            'src': src, 'dataset_mode': 'metadata',
            'metadata_rules': [{'name': 'ones',
                                'where': ['columnID == c1']}],
            'test_split': 0.25})

    # The same rule spelled properly selects crops, so the refusal above is
    # about the shape of the condition and not about the rules being unread.
    train_dir, _ = IO.generate_training_dataset({
        'src': src, 'dataset_mode': 'metadata',
        'metadata_rules': [
            {'name': 'ones', 'where': [{'column': 'cell_id', 'op': '==',
                                        'value': 'o1'}]},
            {'name': 'twos', 'where': [{'column': 'cell_id', 'op': '==',
                                        'value': 'o2'}]}],
        'test_split': 0.5, 'cv_group_by': 'cell', 'random_seed': 1})
    assert sorted(os.listdir(train_dir)) == ['ones', 'twos']


def test_an_existing_random_column_is_filled_in_not_added_twice(tmp_path):
    """Rerunning an annotation dataset reuses the ``<col>_random`` column.

    ``ALTER TABLE ... ADD COLUMN`` fails on a column that is already there, so
    a second run against the same database would abort with an sqlite error
    after the crops had already been selected.
    """
    src = str(tmp_path / "plate1")
    rows = (_crop_rows_for(src, ['A01', 'A02'], annotation=1,
                           extra={'test_random': None})
            + _crop_rows_for(src, ['B01', 'B02'], annotation=None,
                             extra={'test_random': None}))
    _training_project(src, rows)
    db = os.path.join(src, 'measurements', 'measurements.db')

    settings = {'src': src, 'dataset_mode': 'annotation',
                'annotation_columns': ['test'],
                'write_random_annotation_column': True,
                'test_split': 0.25, 'random_seed': 1}
    train_dir, _ = IO.generate_training_dataset(settings)

    assert sorted(os.listdir(train_dir)) == ['test_1', 'test_random']
    with sqlite3.connect(db) as con:
        columns = [row[1] for row in
                   con.execute('PRAGMA table_info("png_list")')]
        marked = pd.read_sql(
            'SELECT test, test_random FROM png_list', con)
    # One column, not two, and the unannotated rows are the ones marked.
    assert columns.count('test_random') == 1
    assert [int(v) for v in marked.loc[marked['test'].isna(),
                                       'test_random']] == [1] * 4
    assert marked.loc[marked['test'].notna(), 'test_random'].isna().all()


def test_a_crop_with_no_recorded_png_path_is_cut_but_not_marked(tmp_path):
    """An on-demand crop the database has no path for cannot be marked.

    With ``crop_source='merged'`` the pixels come from ``merged/*.npy``, so a
    row whose crop was never written to disk still yields a usable training
    image. The ``<col>_random`` column, though, is keyed on ``png_path`` --
    there is no row to update, and the ``UPDATE ... WHERE png_path = ''`` this
    guard prevents would mark every unwritten crop in the table at once.
    """
    src = str(tmp_path / "plate1")
    merged = os.path.join(src, 'merged')
    os.makedirs(merged)
    array = np.zeros((32, 32, 4), np.uint16)
    array[:, :, 0], array[:, :, 1], array[:, :, 2] = 100, 120, 140
    labels = np.zeros((32, 32), np.uint16)
    labels[4:12, 4:12] = 1
    labels[20:28, 20:28] = 2
    array[:, :, 3] = labels

    rows = []
    for well, annotation in (('A01', 1), ('A02', 1), ('B01', None),
                             ('B02', None)):
        field = f'plate1_{well}_1'
        np.save(os.path.join(merged, field + '.npy'), array)
        for obj in (1, 2):
            # One unannotated row records no crop path at all.
            pathless = well == 'B02' and obj == 2
            rows.append({
                'png_path': None if pathless else os.path.join(
                    src, 'data', 'plate1_r1', 'cell_png',
                    f'{field}_o{obj}.png'),
                'plateID': 'plate1', 'rowID': 'r1', 'columnID': 'c1',
                'fieldID': 'f1', 'cell_id': f'o{obj}', 'test': annotation,
                'path_name': os.path.join(merged, field + '.npy')})
    _training_project(src, rows, write_crops=False)
    db = os.path.join(src, 'measurements', 'measurements.db')

    train_dir, test_dir = IO.generate_training_dataset({
        'src': src, 'dataset_mode': 'annotation',
        'annotation_columns': ['test'],
        'write_random_annotation_column': True,
        'crop_source': 'merged', 'png_type': '', 'path_string': '',
        'cv_group_by': 'cell', 'test_split': 0.25, 'random_seed': 1,
        'png_dims': [0, 1, 2], 'png_size': [16, 16], 'cell_mask_dim': 3,
        'crop_mode': ['cell'], 'normalize': False})

    # The pathless crop was still cut out of merged/ and written into the
    # random class ...
    written = [name
               for root in (train_dir, test_dir)
               for name in os.listdir(os.path.join(root, 'test_random'))]
    assert len(written) == 4
    assert 'crop.png' in written
    # ... but only the three rows that have a png_path were marked.
    with sqlite3.connect(db) as con:
        marked = pd.read_sql('SELECT png_path, test_random FROM png_list', con)
    assert int(marked['test_random'].fillna(0).sum()) == 3
    assert marked.loc[marked['png_path'].isna(), 'test_random'].isna().all()


# ---------------------------------------------------------------------------
# Proved unreachable
#
# Each test below pins the invariant that makes one of these branches
# impossible to reach, rather than contorting an input to reach it. Nothing is
# excluded from coverage; the guard stays in the source and the guarantee that
# makes it dead is asserted here, so that if the guarantee ever stops holding
# a test says so.
#
# Four more have no cheap executable pin and are argued in place:
#
# * ``io.py:2169`` ``if not arrays: continue`` in
#   ``_create_movies_from_npy_per_channel``. ``organized_files[key]`` is
#   created at 2155-2157 by appending an entry, so every ``file_list`` holds at
#   least one file, and the loop at 2163 appends one array per file. ``arrays``
#   is never empty.
# * ``io.py:2597`` ``elif mask.ndim not in [2, 3]`` in
#   ``_get_avg_object_size``. The ``else`` it sits in is entered only when
#   ``mask.ndim not in [2, 3] or not np.any(mask)``; the ``if`` above it
#   consumes the second disjunct, so the first must hold and the ``elif`` is
#   always true.
# * ``io.py:2511`` ``if ch in seen`` in ``preprocess_img_data``. ``seen`` is
#   built at 2360-2371 from ``settings.get(key)`` over the same
#   ``mask_channel_keys``, through the same ``int()`` coercion and the same two
#   ``continue`` guards. Nothing between the two loops writes a ``*_channel``
#   key -- ``set_default_settings_preprocess_img_data`` sets none of them, and
#   neither ``_rename_and_organize_image_files`` nor
#   ``concatenate_and_normalize`` assigns one -- so every ``ch`` that reaches
#   2511 was put into ``seen`` at 2369.
# * ``io.py:6620`` the ``for j in range(1, 100000)`` of ``_ensure_unique_dir``
#   completing without a ``break``. Reaching the line after the loop needs
#   99,999 sibling directories, ``training_1`` through ``training_99999``, to
#   exist at once. It is not unreachable in principle, only at any cost worth
#   paying in a test.
# * ``io.py:7734`` ``if file not in file_to_well`` in ``convert_to_yokogawa``.
#   The dict is keyed by the loop variable of ``for file in
#   sorted(os.listdir(folder))`` and written nowhere else, so the entries are
#   distinct and the test is true on every visit.
# * ``io.py:7466`` ``if grouped_splits is None`` in
#   ``generate_dataset_from_lists``. It is reached only when ``data`` is
#   non-empty (7458 continues otherwise), and a non-empty class means
#   ``class_data`` was truthy and ``flat_items`` collected at least that item,
#   so ``grouped_splits`` was built at 7420.
# * ``io.py:5321`` ``if dst is None``. An empty ``src`` list never reaches it:
#   ``save_settings`` indexes ``settings['src'][0]`` at io.py:5266. A
#   non-empty list assigns ``dst`` on its first iteration, which
#   ``test_a_tar_reports_the_pngs_that_were_not_in_it`` pins.
# ---------------------------------------------------------------------------

def test_escaping_the_plate_of_a_five_part_stem_always_changes_it():
    """Why ``if safe == stem: continue`` in the migration cannot be true.

    ``migrate_unescaped_plate_names`` only considers stems with more than four
    separator-delimited components, and the three trailing components are
    fixed (well, field, time). Everything before them is the plate, so a stem
    that got past the length guard has a plate holding at least one separator
    -- and ``escape_filename_component`` always rewrites that separator as
    ``%5F``. The escaped stem therefore always differs from the original.
    """
    from spacr.schema import KEY_SEPARATOR, escape_field_stem_plate

    for stem in ('exp_1_A01_1_1', 'a_b_c_A01_1_1', 'plate 1_x_A01_1_1'):
        assert len(stem.split(KEY_SEPARATOR)) > 4
        escaped = escape_field_stem_plate(stem, timelapse=True)
        assert escaped != stem
        assert '%5F' in escaped
        # The tail is untouched; only the plate half was rewritten.
        assert escaped.endswith('_A01_1_1')

    # A four-component stem is left alone -- which is why the migration's
    # length guard, not this equality, is what makes it a no-op.
    assert escape_field_stem_plate('plate1_A01_1_1',
                                   timelapse=True) == 'plate1_A01_1_1'


def test_a_numeric_prcf_is_rewritten_as_text_before_the_object_split():
    """Why ``if 'prcf' in metadata.columns`` in ``_read_and_merge_data`` holds.

    ``metadata`` is the non-numeric half of ``utils._split_data``, and that
    function rebuilds ``prcf`` from ``plateID``/``rowID``/``columnID``/
    ``fieldID`` as a string on every call -- overwriting a numeric column of
    that name. A string column cannot land in the numeric half, so ``prcf`` is
    always in ``metadata`` and the ``else`` that rebuilds ``prcfo`` from the
    four keys is dead.
    """
    from spacr.utils import _split_data

    frame = pd.DataFrame({'plateID': ['plate1'] * 2, 'rowID': ['r1'] * 2,
                          'columnID': ['c1'] * 2, 'fieldID': ['f1'] * 2,
                          # numeric on the way in ...
                          'prcf': [1, 2],
                          'object_label': [1, 2], 'area': [1.0, 2.0]})
    numeric, non_numeric = _split_data(frame, 'prcfo', 'object_label')

    # ... text on the way out, and in the non-numeric half.
    assert 'prcf' in non_numeric.columns
    assert 'prcf' not in numeric.columns
    assert list(non_numeric['prcf']) == ['plate1_r1_c1_f1'] * 2


def test_a_class_that_selected_nothing_is_refused_before_balancing(tmp_path):
    """Why ``_balance_lists`` is never handed an empty list of classes.

    ``generate_training_dataset`` raises on an empty selection two guards
    above the call, so the ``if not list_of_lists`` at the top of the balancer
    cannot fire. The refusal is the useful behaviour anyway: balancing to a
    class of zero writes a plausible-looking but empty training tree.
    """
    src = str(tmp_path / "plate1")
    _training_project(src, _crop_rows_for(src, ['A01', 'A02'], column='c1'))

    with pytest.raises(ValueError, match="selected no crops for any class"):
        IO.generate_training_dataset({
            'src': src, 'dataset_mode': 'metadata',
            'class_metadata': [['zz']], 'metadata_type_by': 'columnID'})

    # A value that does occur selects crops, so the refusal is about the
    # class being empty and not about the mode never selecting anything.
    train_dir, _ = IO.generate_training_dataset({
        'src': src, 'dataset_mode': 'metadata',
        'class_metadata': [['c1']], 'metadata_type_by': 'columnID',
        'test_split': 0.5, 'cv_group_by': 'cell', 'random_seed': 1})
    assert sorted(os.listdir(train_dir)) == ['c1']
    assert len(os.listdir(os.path.join(train_dir, 'c1'))) == 2


def test_annotation_mode_refuses_an_empty_column_list_before_it_builds_classes(
        tmp_path):
    """Why ``if not ann_cols`` inside ``_annotation_classes_from_columns`` is dead.

    Its only caller filters the requested columns and raises when nothing is
    left, so the helper is never entered with an empty list. The refusal is
    what a user needs: annotation mode with no column selects every crop into
    no class at all.
    """
    src = str(tmp_path / "plate1")
    _training_project(src, _crop_rows_for(src, ['A01', 'A02']))

    with pytest.raises(ValueError,
                       match="requires at least one annotation_columns"):
        IO.generate_training_dataset({
            'src': src, 'dataset_mode': 'annotation',
            'annotation_columns': [], 'annotation_column': None})

    # A column that is named gets as far as reading it, so the guard above is
    # about the column list being empty.
    with pytest.raises(ValueError, match="selected no crops for any class"):
        IO.generate_training_dataset({
            'src': src, 'dataset_mode': 'annotation',
            'annotation_columns': ['test']})


def test_an_unknown_dataset_mode_is_refused_by_the_basis_resolver(tmp_path):
    """Why ``generate_training_dataset``'s "Invalid dataset_mode" is dead.

    ``training_basis.resolve_basis`` raises on anything outside
    ``TRAINING_BASES`` and migrates the retired ``'measurement'``, so the
    ``else`` branch below its call can only see ``'metadata'`` or
    ``'annotation'``. The retired spelling has to keep running, which is the
    half of this that a settings CSV depends on.
    """
    from spacr.training_basis import TrainingBasisError

    src = str(tmp_path / "plate1")
    _training_project(src, _crop_rows_for(src, ['A01', 'A02'], annotation=1))

    with pytest.raises(TrainingBasisError, match="is not one of"):
        IO.generate_training_dataset({'src': src,
                                      'dataset_mode': 'nonsense'})

    # 'measurement' is migrated to annotation rather than refused: it reaches
    # the class builder and fails there, on the labels, not on the mode.
    with pytest.raises(ValueError, match="selected no crops for any class"):
        IO.generate_training_dataset({
            'src': src, 'dataset_mode': 'measurement',
            'annotation_columns': ['test'],
            'annotation_values': {'test': [2]}})


def test_a_grouped_split_never_leaves_a_class_on_one_side():
    """Why the "leaves class ... empty" guard in the dataset writer is dead.

    ``grouped_split`` keeps only candidate splits whose train and test sides
    both hold every label it was given, and refuses the design outright when
    no such candidate exists. Every non-empty class therefore appears on both
    sides by the time ``generate_dataset_from_lists`` checks.
    """
    from spacr.classifier_evaluation import grouped_split

    # A deliberately lopsided design: twelve crops of class 0 over four wells,
    # four of class 1 over two.
    labels = [0] * 12 + [1] * 4
    groups = [f'w{index // 3}' for index in range(12)] + ['x0', 'x0', 'x1',
                                                          'x1']
    train, test, _report = grouped_split(groups, labels, 0.25, seed=0,
                                         group_by='well')

    labels = np.asarray(labels)
    assert set(labels[train]) == {0, 1}
    assert set(labels[test]) == {0, 1}
    # ... and a design that genuinely cannot do it is refused here rather
    # than handed on as a split with an empty class.
    with pytest.raises(ValueError, match="cannot put every class"):
        grouped_split(['w0'] * 4 + ['w1'] * 4, [0] * 4 + [1] * 4, 0.25,
                      seed=0, group_by='well')


def test_without_augmentation_every_folder_contributes_the_smallest_count(
        tmp_path):
    """Why ``if augment_data`` inside the short-folder branch is always true.

    ``target_size`` is the *smallest* folder when augmentation is off, so no
    folder can be shorter than the target and the branch that pads a short one
    is only entered when augmentation is on. The balancing itself is the point:
    a five-pair folder must not outweigh a two-pair one.
    """
    root = tmp_path / "root"
    for name, count in (("small", 2), ("big", 5)):
        folder = root / name
        (folder / "masks").mkdir(parents=True)
        for index in range(count):
            tifffile.imwrite(str(folder / f"{index}.tif"),
                             np.ones((4, 4), np.uint16))
            tifffile.imwrite(str(folder / "masks" / f"{index}.tif"),
                             np.ones((4, 4), np.uint16))

    IO.prepare_cellpose_dataset(str(root), augment_data=False,
                                train_fraction=1.0, n_jobs=1)

    images = os.path.join(str(root), "cellpose_dataset", "train", "images")
    # Two folders, two pairs each: the big one was sampled down, not the
    # small one padded.
    assert len(os.listdir(images)) == 4
    assert len(os.listdir(os.path.join(
        str(root), "cellpose_dataset", "train", "masks"))) == 4


def test_every_plane_split_out_of_an_image_carries_a_channel_index(tmp_path):
    """Why ``save_grayscale_images`` is never called without a channel.

    All three call sites pass ``channel=c+1`` for ``c`` in
    ``range(image.shape[2])``, so the ``if channel is not None`` in the
    suffix builder cannot be false. What that guarantees is what matters: two
    planes of one source never collide on a file name.
    """
    array = np.zeros((8, 8, 3), np.uint8)
    array[..., 0], array[..., 1], array[..., 2] = 10, 20, 30
    Image.fromarray(array).save(str(tmp_path / "img.png"))

    IO.process_non_tif_non_2D_images(str(tmp_path))

    written = sorted(p.name for p in tmp_path.glob("*.tif"))
    assert written == ["img_C1.tif", "img_C2.tif", "img_C3.tif"]
    assert [int(tifffile.imread(str(tmp_path / name)).max())
            for name in written] == [10, 20, 30]
