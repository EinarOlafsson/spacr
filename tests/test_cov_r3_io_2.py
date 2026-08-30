"""Narrow-condition coverage for ``spacr.io``.

Each test drives one guard a successful run never takes: a plane the
converter cannot shape, a file it must not touch, an already normalised
image, a dataset index past the end, a preloader told to stop mid-batch, a
merge whose keys match nothing.  CPU-only, offline, all output under.
"""
from __future__ import annotations

import os
import queue
import sqlite3
import types

import numpy as np
import pandas as pd
import pytest
import tifffile

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _png(path, value=7, size=8):
    from PIL import Image
    Image.fromarray(np.full((size, size, 3), value, np.uint8)).save(str(path))
    return str(path)


def _float_tif(path, arr):
    tifffile.imwrite(str(path), arr.astype(np.float32))
    return str(path)


def test_six_axis_tiff_and_foreign_files_leave_the_folder_alone(tmp_path):
    """A plane layout the splitter does not understand must not be guessed at.

    ``split_channels`` handles 2-5 axes; a 6-axis acquisition falls off the
    end of the chain and writes nothing, and a non-image file beside it is
    skipped by extension.  Either one producing output would mean inventing a
    channel order for data whose axes were never checked.
    """
    from spacr.io import process_non_tif_non_2D_images

    folder = tmp_path / "raw"
    folder.mkdir()
    tifffile.imwrite(str(folder / "six.tif"),
                     np.arange(2 * 3 * 4 * 5 * 6 * 2, dtype=np.uint16
                               ).reshape(2, 3, 4, 5, 6, 2))
    tifffile.imwrite(str(folder / "three.tif"),
                     np.arange(8 * 8 * 2, dtype=np.uint16).reshape(8, 8, 2))
    (folder / "notes.txt").write_text("not an image")

    process_non_tif_non_2D_images(str(folder))

    written = sorted(p.name for p in folder.iterdir())
    assert "three_C1.tif" in written and "three_C2.tif" in written
    assert not [n for n in written if n.startswith("six_")]
    assert (folder / "notes.txt").read_text() == "not an image"
    assert not (folder / "notes.tif").exists()


def test_images_already_in_zero_to_one_are_not_rescaled(tmp_path):
    """Re-dividing an already normalised image would destroy its scale.

    Cellpose training data is often handed in as float 0-1.  The loader
    divides by ``max()`` only when the maximum exceeds 1; were that guard
    inverted, a 0-1 image would be stretched until its brightest pixel became
    1.0 and every intensity would change meaning between runs.
    """
    from spacr.io import _load_images_and_labels

    low = _float_tif(tmp_path / "low.tif", np.full((4, 4), 0.25))
    high = _float_tif(tmp_path / "high.tif", np.full((4, 4), 8.0))
    lbl_a = _float_tif(tmp_path / "low_mask.tif", np.ones((4, 4)))
    lbl_b = _float_tif(tmp_path / "high_mask.tif", np.ones((4, 4)))

    imgs, labels, names, _ = _load_images_and_labels([low, high],
                                                     [lbl_a, lbl_b])
    assert names == ["low.tif", "high.tif"]
    assert len(labels) == 2
    assert imgs[0].max() == pytest.approx(0.25)   # untouched
    assert imgs[1].max() == pytest.approx(1.0)    # divided by 8.0

    imgs_only, _, names_only, _ = _load_images_and_labels([low, high], [])
    assert names_only == ["low.tif", "high.tif"]
    assert imgs_only[0].max() == pytest.approx(0.25)
    assert imgs_only[1].max() == pytest.approx(1.0)


def test_combined_dataset_index_past_the_end_returns_nothing(tmp_path):
    """An out-of-range index yields ``None`` instead of raising.

    ``__getitem__`` walks its sub-datasets and falls off the end, so a
    sampler built from a stale length hands the training loop a ``None``
    sample rather than an ``IndexError`` that would name the bug.  Pinning it
    keeps that silent contract visible to whoever changes it.
    """
    from spacr.io import CombinedDataset, NoClassDataset

    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        _png(d / f"i{i}.png")
    comb = CombinedDataset([NoClassDataset(str(d), shuffle=False)],
                           shuffle=False)

    assert len(comb) == 3
    assert comb[0] is not None            # in range: a real sample
    assert comb[3] is None                # one past the end
    assert comb[99] is None


def test_unshuffled_dataset_never_touches_the_filename_order(tmp_path,
                                                             monkeypatch):
    """``shuffle=False`` must mean the disk order is the sample order.

    ``shuffle_dataset`` is public and callers invoke it directly; on a
    dataset built with ``shuffle=False`` it has to be a no-op, because the
    caller is pairing those filenames with an external list of predictions.
    A shuffle there silently re-labels every result.
    """
    import random
    import spacr.io as IO

    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(5):
        _png(d / f"i{i}.png")

    calls = []
    monkeypatch.setattr(random, "shuffle", lambda seq: calls.append(list(seq)))

    quiet = IO.NoClassDataset(str(d), shuffle=False)
    before = list(quiet.filenames)
    quiet.shuffle_dataset()
    assert quiet.filenames == before
    assert calls == []                     # random.shuffle never reached

    loud = IO.NoClassDataset(str(d), shuffle=True)
    assert len(calls) == 1                 # the shuffling dataset did reach it
    loud.shuffle_dataset()
    assert len(calls) == 2


class _StopSignal:
    """``is_set`` answers False for the first ``n`` calls, then True."""

    def __init__(self, false_calls):
        self.false_calls = false_calls
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.false_calls


def _bare_loader():
    from spacr.io import spacrDataLoader
    loader = object.__new__(spacrDataLoader)
    loader.pin_memory = False
    loader._sentinel = object()
    loader._error = None
    return loader


def test_a_batch_is_dropped_when_the_stop_arrives_before_the_queue_put():
    """A stopping preloader must not park a batch in a queue nobody drains.

    ``cleanup`` sets the stop signal and then joins; a producer that pushed
    after that point would block forever on the bounded queue and the daemon
    thread would be abandoned with the loader's memory still pinned.  So the
    signal is re-checked immediately before the put, sentinel included.
    """
    q = queue.Queue(maxsize=4)
    loader = _bare_loader()

    loader._preload_next_batches(q, iter(["a", "b"]), _StopSignal(1))
    assert q.empty(), "a stopping producer queues neither batch nor sentinel"

    q2 = queue.Queue(maxsize=4)
    loader._preload_next_batches(q2, iter(["a", "b"]), _StopSignal(10 ** 6))
    assert [q2.get_nowait(), q2.get_nowait()] == ["a", "b"]
    assert q2.get_nowait() is loader._sentinel


def test_a_preloader_that_will_not_die_is_reported_not_hidden(caplog):
    """Abandoning a live daemon thread has to leave a trace in the log.

    ``cleanup`` gives the producer five seconds and then gives up.  Without
    the message the process is left with a thread still holding pinned memory
    and a queue reference while the next ``__iter__`` starts a second
    producer beside it -- a leak that otherwise looks like nothing.
    """
    import logging
    import spacr.io as IO

    class _Undying:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    clock = iter(range(1000))
    monkey = types.SimpleNamespace(monotonic=lambda: float(next(clock)))

    loader = _bare_loader()
    loader._iteration_active = True
    loader._stop_signal = None
    loader.thread = _Undying()
    loader.batch_queue = queue.Queue()
    loader.batch_queue.put("leftover")

    original_time = IO.time
    IO.time = monkey
    try:
        with caplog.at_level(logging.ERROR, logger="spacr.io"):
            loader.cleanup()
    finally:
        IO.time = original_time

    assert loader._iteration_active is False
    assert "did not stop within five seconds" in caplog.text
    assert loader.batch_queue.empty(), "the queue is drained while joining"


def test_a_populated_stack_folder_is_never_rebuilt(tmp_path):
    """Re-running ingest over a finished plate must not redo or delete it.

    The organiser builds ``stack/`` only when it is missing or empty.  Were
    it to run again over a populated one it would re-read source images that
    ``save_original_images=False`` has already removed, and report a channel
    count recomputed from nothing.
    """
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    (src / "stack").mkdir(parents=True)
    arr = np.ones((4, 4, 2), np.uint16)
    np.save(str(src / "stack" / "plate1_A01_1.npy"), arr)
    (src / "plate1_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"junk")

    channels = _rename_and_organize_image_files(
        str(src), r'(?P<plateID>.*)_(?P<wellID>[A-Z]\d{2})_(?P<chanID>C\d{2})')

    assert channels == 0
    assert sorted(p.name for p in (src / "stack").iterdir()) == [
        "plate1_A01_1.npy"]
    assert np.array_equal(np.load(str(src / "stack" / "plate1_A01_1.npy")), arr)


def test_merge_file_leaves_an_existing_stack_untouched(tmp_path):
    """An existing merged stack is a checkpoint, not something to overwrite.

    ``_merge_file`` runs once per field over a folder that may already hold
    results from an interrupted run.  Rewriting them re-reads every channel
    image for work already done, and a crash mid-write replaces a good stack
    with a truncated one.
    """
    from spacr.io import _merge_file

    chan_dir = tmp_path / "01"
    chan_dir.mkdir()
    tifffile.imwrite(str(chan_dir / "f1.tif"), np.full((4, 4), 3, np.uint16))
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    existing = np.full((4, 4, 1), 99, np.uint16)
    np.save(str(stack_dir / "f1.npy"), existing)

    assert _merge_file([str(chan_dir)], str(stack_dir), "f1.tif") is None
    assert np.array_equal(np.load(str(stack_dir / "f1.npy")), existing)


def test_non_numeric_well_field_channel_and_time_tokens_survive(tmp_path):
    """Only zero-padded NUMBERS are un-padded; text ids must pass through.

    A custom regex on a non-Yokogawa scope routinely yields field ``F1`` or
    channel ``GFP``.  ``_int_or_token`` is applied only to tokens starting
    with a digit, because coercing text ids would map two different channels
    onto one folder.  A sidecar with a foreign extension is ignored outright.
    """
    from spacr.io import _move_to_chan_folder

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "plateA-wellB-fieldC-chanD-timeE.tif").write_bytes(b"x")
    (src / "readme.md").write_text("ignore me")

    regex = (r'(?P<plateID>[^-]+)-(?P<wellID>[^-]+)-(?P<fieldID>[^-]+)'
             r'-(?P<chanID>[^-]+)-(?P<timeID>[^.]+)')
    _move_to_chan_folder(str(src), regex, timelapse=False, metadata_type='')

    assert (src / "chanD").is_dir()        # folder named by the raw chanID
    moved = sorted(p.name for p in (src / "chanD").iterdir())
    assert len(moved) == 1
    assert "wellB" in moved[0] and "fieldC" in moved[0]
    assert (src / "readme.md").read_text() == "ignore me"
    assert not (src / "chanD" / "readme.md").exists()


def test_merge_channels_without_channel_subfolders_says_so_and_returns(
        tmp_path, capsys):
    """No per-channel folders is the normal modern layout, not a failure.

    ``_merge_channels`` is the older two-step path.  Indexing ``chan_dirs[0]``
    on a folder that has none raised ``IndexError`` from inside a stage whose
    message named the plate, so an ordinary source folder read as a corrupt
    one.  It now returns 0 and lets the caller build ``stack/`` directly.
    """
    from spacr.io import _merge_channels

    src = tmp_path / "plate1"
    (src / "notachannel").mkdir(parents=True)

    assert _merge_channels(str(src), plot=False) == 0
    assert "No single-channel folders" in capsys.readouterr().out
    assert not (src / "stack").exists()


def _norm_settings(nucleus, cell, pathogen):
    s = {'nucleus_channel': nucleus, 'cell_channel': cell,
         'pathogen_channel': pathogen}
    for role in ('nucleus', 'cell', 'pathogen'):
        s[f'{role}_background'] = 100
        s[f'{role}_Signal_to_noise'] = 5
        s[f'remove_background_{role}'] = True
    return s


def test_a_blank_channel_from_a_settings_csv_produces_no_settings_at_all():
    """A channel read back as NaN matches nothing and is silently dropped.

    Settings loaded from a CSV give a blank cell as ``float('nan')``, not
    ``None``.  ``nan is not None`` is True, so it passes the "channel was
    given" guard, but ``nan == nan`` is False, so it matches none of the
    three roles and contributes no background, signal-to-noise or threshold.
    """
    from spacr.io import _get_lists_for_normalization

    good = _get_lists_for_normalization(_norm_settings(0, 1, 2))
    assert [len(part) for part in good] == [3, 3, 3, 3]
    assert good[0] == [100, 100, 100]

    nan = float('nan')
    blank = _get_lists_for_normalization(_norm_settings(nan, nan, nan))
    assert [len(part) for part in blank] == [0, 0, 0, 0]


def test_an_unreachable_signal_threshold_leaves_the_percentile_at_the_top(
        tmp_path):
    """A threshold no percentile can meet must not abort the normalisation.

    ``signal_thresholds`` is a user setting; on a dim channel, or a badly
    chosen value, no percentile between 98 and 100 reaches it.  The search
    then runs to the end and normalisation continues at the 100th percentile,
    so one over-ambitious setting cannot kill a whole plate.
    """
    from spacr.io import _normalize_stack

    src = tmp_path / "channel_stack"
    src.mkdir()
    stack = np.linspace(1, 100, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 1)
    np.savez(str(src / "plate1.npz"), data=stack,
             filenames=np.array(["a.npy", "b.npy"]))

    _normalize_stack(str(src), backgrounds=[0], remove_backgrounds=[False],
                     signal_to_noise=[5], signal_thresholds=[10 ** 9])

    out = tmp_path / "masks" / "plate1_norm_stack.npz"
    assert out.is_file()
    with np.load(str(out)) as data:
        saved = data['data']
    assert saved.shape == stack.shape
    assert np.isfinite(saved).all()


def test_cardinality_errors_name_the_side_that_is_actually_duplicated():
    """The error must accuse the table that broke the contract, not both.

    ``validate='one_to_many'`` constrains only the LEFT side and
    ``'many_to_one'`` only the RIGHT.  Naming the wrong table sends the user
    to de-duplicate a file that was already fine, and a merge error that is
    not about cardinality at all must keep pandas' own message.
    """
    from spacr.io import MergeCardinalityError, _merge_with_cardinality

    left_dupes = pd.DataFrame({'k': ['a', 'a', 'b'], 'v': [0, 1, 2]})
    right_unique = pd.DataFrame({'k': ['a', 'b'], 'w': [1, 2]})

    with pytest.raises(MergeCardinalityError) as one_to_many:
        _merge_with_cardinality(left_dupes, right_unique, validate='one_to_many',
                                left_name='cell', right_name='png_list', on='k')
    assert 'cell has duplicated' in str(one_to_many.value)
    assert 'png_list has duplicated' not in str(one_to_many.value)

    with pytest.raises(MergeCardinalityError) as many_to_one:
        _merge_with_cardinality(right_unique, left_dupes, validate='many_to_one',
                                left_name='cell', right_name='png_list', on='k')
    assert 'png_list has duplicated' in str(many_to_one.value)
    assert 'cell has duplicated' not in str(many_to_one.value)

    # A MergeError that is not a cardinality violation is re-raised as-is.
    with pytest.raises(pd.errors.MergeError) as plain:
        _merge_with_cardinality(pd.DataFrame({'a': [1]}),
                                pd.DataFrame({'b': [1]}),
                                validate='one_to_one',
                                left_name='cell', right_name='png_list')
    assert not isinstance(plain.value, MergeCardinalityError)
    assert 'common columns' in str(plain.value)


def _join_db(path, **tables):
    con = sqlite3.connect(str(path))
    try:
        for name, frame in tables.items():
            frame.to_sql(name, con, index=False, if_exists='replace')
    finally:
        con.close()
    return str(path)


def test_asking_only_for_a_table_that_is_not_there_yields_no_frame(tmp_path,
                                                                   capsys):
    """A join with no cell table to anchor on cannot return a result.

    Every missing table is announced and skipped, so naming only one that
    does not exist leaves nothing to join and no frame to hand back.
    Recording that keeps "reported, not fatal" from quietly becoming
    "returns something meaningless".
    """
    from spacr.io import _read_and_join_tables

    db = _join_db(tmp_path / 'm.db',
                  cell=pd.DataFrame({'object_label': [1],
                                     'prcf': ['plate1_r01_c01_1'],
                                     'cell_area': [3.0]}))

    with pytest.raises(AttributeError):
        _read_and_join_tables(db, table_names=['does_not_exist'])
    assert 'Table does_not_exist not found' in capsys.readouterr().out

    out = _read_and_join_tables(db, table_names=['cell'])
    assert list(out['object_label']) == [1]


def test_collapse_off_keeps_both_copies_of_a_shared_column(tmp_path):
    """``collapse_duplicate_identity=False`` must be honoured, not ignored.

    ``anndata_export`` documents ``drop_redundant_identity=False`` as keeping
    the suffixed duplicates a join produces.  Collapsing anyway would make
    that option a no-op and the export would silently lose the columns it
    promised.  Both the cytoplasm join and the child-role join are driven.
    """
    from spacr.io import _read_and_join_tables

    prcf = 'plate1_r01_c01_1'
    cell = pd.DataFrame({'object_label': [1, 2], 'prcf': [prcf] * 2,
                         'plateID': ['plate1'] * 2, 'area': [100.0, 200.0]})
    cytoplasm = pd.DataFrame({'object_label': [1, 2], 'prcf': [prcf] * 2,
                              'plateID': ['plate1'] * 2, 'area': [70.0, 80.0]})
    nucleus = pd.DataFrame({'cell_id': [1, 2], 'prcf': [prcf] * 2,
                            'nucleus_area': [10.0, 20.0]})
    db = _join_db(tmp_path / 'm.db', cell=cell, cytoplasm=cytoplasm,
                  nucleus=nucleus)

    kept = _read_and_join_tables(db, table_names=['cell', 'cytoplasm',
                                                  'nucleus'],
                                 collapse_duplicate_identity=False)
    assert 'area_cytoplasm' in kept.columns
    assert 'plateID_cytoplasm' in kept.columns
    assert kept['area'].tolist() == [100.0, 200.0]
    assert kept['area_cytoplasm'].tolist() == [70.0, 80.0]
    assert kept['count_nucleus'].tolist() == [1, 1]


def _merge_src(root, masks):
    """stack/fov.npy plus one mask folder per (name, array)."""
    stack = os.path.join(root, 'stack')
    os.makedirs(stack, exist_ok=True)
    img = np.zeros((6, 6, 2), np.float32)
    img[..., 1] = 1.0
    np.save(os.path.join(stack, 'fov.npy'), img)
    for name, arr in masks.items():
        d = os.path.join(root, 'masks', f'{name}_mask_stack')
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 'fov.npy'), arr)
    return img


def test_a_mask_that_already_carries_a_channel_axis_is_not_padded_again(
        tmp_path):
    """A mask saved as (Y, X, 1) must join the stack as ONE plane, not two.

    The axis is appended only when the mask is one axis short of the image it
    belongs to.  Appending unconditionally would push a (6, 6, 1) mask to
    (6, 6, 1, 1), where ``np.concatenate`` either fails or -- worse -- shifts
    every downstream channel index.
    """
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path)
    mask = np.zeros((6, 6, 1), np.uint16)
    mask[1:4, 1:4, 0] = 5
    img = _merge_src(root, {'cell': mask})

    _load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=0,
                                 nucleus_chann_dim=None,
                                 pathogen_chann_dim=None,
                                 organelle_chann_dim=None)

    merged = np.load(os.path.join(root, 'merged', 'fov.npy'))
    assert merged.shape == (6, 6, 3)      # 2 image channels + 1 mask plane
    assert np.array_equal(merged[..., 0], img[..., 0])
    assert np.array_equal(merged[..., 2], mask[..., 0])


def test_a_merge_with_no_mask_folders_writes_no_stack(tmp_path):
    """Copying the intensity stack into merged/ would be a lie about content.

    ``merged/`` means "image planes plus mask planes"; with no mask folders
    the concatenation is the input itself, and writing it would create a file
    measurement treats as carrying masks it does not have.  Adding one mask
    folder to the same source proves the writer is otherwise live.
    """
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path / "plate")
    os.makedirs(root)
    _merge_src(root, {})

    _load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=None,
                                 nucleus_chann_dim=None,
                                 pathogen_chann_dim=None,
                                 organelle_chann_dim=None)
    assert [f for f in os.listdir(os.path.join(root, 'merged'))
            if f.endswith('.npy')] == []

    _merge_src(root, {'cell': np.zeros((6, 6), np.uint16)})
    _load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=0,
                                 nucleus_chann_dim=None,
                                 pathogen_chann_dim=None,
                                 organelle_chann_dim=None)
    assert np.load(os.path.join(root, 'merged', 'fov.npy')).shape == (6, 6, 3)


_STAMP = {'measurement_ndim': 2, 'measurement_units': 'px', 'n_z': 1,
          'voxel_size_z_um': 1.0, 'voxel_size_xy_um': 0.5}


def _cell_frame(n=3):
    rows = []
    for obj in range(1, n + 1):
        row = {'object_label': obj, 'plateID': 'plate1', 'rowID': 'r01',
               'columnID': 'c01', 'fieldID': '1', 'prcf': 'plate1_r01_c01_1',
               'cell_area': 100.0 + obj}
        row.update(_STAMP)
        rows.append(row)
    return pd.DataFrame(rows)


def test_read_db_reads_a_path_that_is_not_text(tmp_path):
    """A bytes path must reach SQLite instead of being mangled on the way.

    ``expanduser``/``expandvars`` are applied only to text-like paths; a
    bytes path -- what ``os.fsencode`` and several stdlib walkers hand back
    -- skips them and goes straight to ``sqlite3.connect``, which accepts it.
    Forcing it through the string expansion would raise on a valid path.
    """
    from spacr.io import _read_db

    db = _join_db(tmp_path / 'm.db', cell=_cell_frame(2))
    frames = _read_db(os.fsencode(db), ['cell'])

    assert len(frames) == 1
    assert list(frames[0]['object_label']) == [1, 2]
    assert frames[0]['cell_area'].tolist() == [101.0, 102.0]


def test_crops_that_belong_to_no_measured_cell_empty_the_merge(tmp_path,
                                                               capsys):
    """The population must not vanish without the merge saying so.

    ``png_list`` joins INNER: a crop set whose object ids match no measured
    cell -- crops from another plate, or another crop mode -- deletes every
    row.  The loss is printed and named by table, because a silent empty
    result looks exactly like "this plate had no cells".
    """
    from spacr.io import _read_and_merge_data

    png_rows = []
    for obj in range(1, 4):
        row = {'cell_id': f'o{90 + obj}', 'png_path': f'/x/o{90 + obj}.png',
               'plateID': 'plate1', 'rowID': 'r01', 'columnID': 'c01',
               'fieldID': '1', 'prcf': 'plate1_r01_c01_1', 'test': obj}
        row.update(_STAMP)
        png_rows.append(row)
    db = _join_db(tmp_path / 'm.db', cell=_cell_frame(3),
                  png_list=pd.DataFrame(png_rows))

    merged, obj_dfs = _read_and_merge_data([db], ['cell', 'png_list'])

    out = capsys.readouterr().out
    assert 'no row in png_list' in out
    assert '3 of 3 objects' in out
    assert len(merged) == 0
    # the cells were measured -- they are simply unjoinable to any crop.
    assert len(obj_dfs[0]) == 3


def test_merging_no_databases_at_all_is_not_a_silent_empty_result(capsys):
    """An empty list of locations must not be mistaken for an empty screen.

    ``locs=[]`` leaves every table as the empty list it was initialised to,
    which the verbose report prints as zero rows before the cell branch trips
    over the list.  A caller whose plate filter matched nothing therefore
    gets a failure rather than a plausible-looking empty frame.
    """
    from spacr.io import _read_and_merge_data

    with pytest.raises(AttributeError):
        _read_and_merge_data([], ['cell'], verbose=True)
    assert 'cell: 0' in capsys.readouterr().out


def _quiet_pipeline(monkeypatch):
    """Silence the two heavy collaborators and record their calls."""
    import spacr.io as IO
    import spacr.plot as PLOT
    seen = {'concat': [], 'plot': []}
    monkeypatch.setattr(IO, 'concatenate_and_normalize',
                        lambda *a, **k: seen['concat'].append(k) or 'masks')
    monkeypatch.setattr(PLOT, 'plot_arrays',
                        lambda *a, **k: seen['plot'].append(a))
    return seen


def _pre_settings(src, **over):
    s = {'src': str(src), 'metadata_type': 'cellvoyager', 'custom_regex': None,
         'channels': [0, 1], 'nucleus_channel': 0, 'cell_channel': 1,
         'pathogen_channel': None, 'organelle_channel': None, 'plot': False,
         'batch_size': 1, 'test_mode': False, 'timelapse': False,
         'normalize': True}
    s.update(over)
    return s


def test_test_mode_on_a_plate_with_no_leftover_test_folder(
        yokogawa_cellvoyager_dir, monkeypatch, capsys):
    """Test mode on a clean plate must build its subset without a stale folder.

    ``test/`` is deleted only when one is already there; on the ordinary
    first run there is nothing to remove and the subset is copied straight
    away.  Were that guard the other way round, the first test run on every
    plate would fail on a directory that never existed.
    """
    from spacr.io import preprocess_img_data

    src = yokogawa_cellvoyager_dir['src']
    assert not (src / 'test').exists()
    seen = _quiet_pipeline(monkeypatch)

    out_settings, out_src = preprocess_img_data(
        _pre_settings(src, test_mode=True, test_images=1, random_test=True))

    assert 'Running spacr in test mode' in capsys.readouterr().out
    assert out_src == str(src / 'test')
    assert out_settings['plot'] is True
    stacks = sorted((src / 'test' / 'stack').glob('*.npy'))
    assert len(stacks) == 1
    assert np.load(str(stacks[0])).shape == (128, 128, 2)
    assert seen['concat'][0]['src'] == str(src / 'test' / 'stack')


def test_an_empty_stack_says_which_of_two_mistakes_was_made(tmp_path,
                                                            monkeypatch):
    """"Wrong folder" and "wrong filenames" need different advice.

    A source holding sub-folders and no images is almost always a screen
    directory rather than a plate, and saying so here -- where ``src`` is
    known -- replaces an ``os.listdir`` failure on a path the user never
    typed.  When the folder DOES still hold images that hint would be wrong
    advice, so the count of image files present is given instead.
    """
    from spacr.io import preprocess_img_data
    _quiet_pipeline(monkeypatch)

    screen = tmp_path / 'screen'
    for plate in ('plate1', 'plate2'):
        (screen / plate).mkdir(parents=True)
        (screen / plate / 'img.tif').write_bytes(b'x')
    with pytest.raises(FileNotFoundError) as parent:
        preprocess_img_data(_pre_settings(screen))
    message = str(parent.value)
    assert 'No image stacks were produced' in message
    assert 'found 0 image file(s)' in message.lower()
    assert 'plate1' in message and 'plate2' in message
    assert 'point src at one of them' in message

    # An UPPERCASE extension: the organiser matches extensions case
    # sensitively, so the image is still sitting in src when this is raised.
    plate = tmp_path / 'plate1'
    plate.mkdir()
    tifffile.imwrite(str(plate / 'PLATE1_A01_T0001F001L01A01Z01C01.TIF'),
                     np.ones((8, 8), np.uint16))
    with pytest.raises(FileNotFoundError) as unmatched:
        preprocess_img_data(_pre_settings(plate))
    message = str(unmatched.value)
    assert 'found 1 image file(s)' in message.lower()
    assert 'point src at one of them' not in message
