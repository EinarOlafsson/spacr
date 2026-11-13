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


# ---------------------------------------------------------------------------
# prepare_cellpose_dataset: pairing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# convert_separate_files_to_yokogawa
# ---------------------------------------------------------------------------

_SEP_REGEX = (r"(?P<wellID>[A-Za-z0-9\-]+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
              r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")


# ---------------------------------------------------------------------------
# convert_to_yokogawa
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _read_and_merge_data
# ---------------------------------------------------------------------------


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


_MERGED_SETTINGS = {'experiment': 'e1', 'crop_source': 'merged',
                    'png_dims': [0, 1, 2], 'png_size': [16, 16],
                    'cell_mask_dim': 3, 'crop_mode': ['cell'],
                    'normalize': False}


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------


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


