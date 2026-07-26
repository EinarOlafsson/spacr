"""Tests for the versioned object-crop PNG format in :mod:`spacr.crops`.

The defect these pin down: ``spacr.measure.save_and_add_image_to_grid`` wrote
crops with ``cv2.imwrite(img_path, png_channels)``. cv2 reads a 3-channel
array as **BGR**; every consumer in spaCR opens the file with PIL, which reads
**RGB**. So ``png_dims[0]`` -- the channel the user asked to be red -- landed
in the file's blue slot, and ``png_dims[2]`` in its red. The classifier's
accuracy never noticed, because the ordering was self-consistent. Anything
that named a channel did: "the model attends to the DAPI channel" was wrong by
reversal, the Annotate app's r/g/b filters addressed the wrong stains, and
``train_channels=['r','g','b']`` selected them backwards.

The fix reverses the array once, in the writer, so cv2's BGR interpretation
lands ``png_dims[0]`` in red. ``png_dims`` keeps its plain meaning. Because a
PNG carries no field saying which order it was written in, the format is
versioned: a ``.spacr_crop_format.json`` sidecar marks a folder, an *unmarked*
folder is legacy (every crop that exists today is unmarked and legacy), and
the reader corrects legacy content on load.

The second half of the same problem, also covered here: crops are 16-bit PNGs
and PIL narrows them two different ways -- the high byte for an RGB image, a
*clip* at 255 for a single-channel one, which returns solid white for any
crop brighter than 255/65535. spaCR now narrows them itself, one way, always.
"""

import json
import os
import sqlite3

import numpy as np
import pytest

from spacr import crops
from spacr.crops import (
    CROP_FORMAT_CURRENT,
    CROP_FORMAT_LEGACY_BGR,
    CROP_FORMAT_RGB,
    CROP_FORMAT_SIDECAR,
    CropError,
    CropFormatConflict,
    CropSpec,
    MergedCropSource,
    PngCropSource,
    crop_folder_format,
    crop_format_for_png,
    extract_crop,
    migrate_crop_folder,
    narrow_to_uint8,
    png_view,
    read_crop_folder_marker,
    read_crop_png,
    stamp_crop_folder,
    to_cv2_bgr,
)

cv2 = pytest.importorskip("cv2")
from PIL import Image                                    # noqa: E402

CELL_DIM, NUC_DIM, PATH_DIM = 4, 5, 6
MASK_DIMS = {"cell": CELL_DIM, "nucleus": NUC_DIM, "pathogen": PATH_DIM}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_format_cache():
    """The folder-marker cache is module-global; never leak it between tests."""
    crops.clear_crop_format_cache()
    yield
    crops.clear_crop_format_cache()


def _unremovable(_path):
    """An ``os.remove`` that always fails, for the cleanup-of-the-cleanup paths."""
    raise OSError("nope")


def _crop(seed=0, shape=(12, 10, 3), dtype=np.uint16, low=1, high=65535):
    """Return a synthetic ``png_channels`` array with three distinguishable planes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(low, high, size=shape).astype(dtype)
    if shape[-1] == 3 and dtype == np.uint16:
        # Make the planes trivially distinguishable after the // 256 narrowing:
        # plane 0 dark, plane 1 mid, plane 2 bright.
        arr[:, :, 0] = (arr[:, :, 0] % 60) * 256 + 3
        arr[:, :, 1] = (arr[:, :, 1] % 60 + 90) * 256 + 3
        arr[:, :, 2] = (arr[:, :, 2] % 60 + 180) * 256 + 3
    return arr


def _write_legacy(path, arr):
    """Write ``arr`` exactly as spaCR used to: cv2.imwrite of the raw array."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert cv2.imwrite(str(path), arr)
    return str(path)


def _write_current(path, arr):
    """Write ``arr`` the way the corrected writer does."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp_crop_folder(os.path.dirname(str(path)))
    assert cv2.imwrite(str(path), to_cv2_bgr(arr))
    return str(path)


def _legacy_folder(tmp_path, n=4, name="cell_png", seed=0):
    """Build an unmarked folder of legacy crops; return (folder, {name: array})."""
    folder = tmp_path / "data" / "plate1_A01" / name
    arrays = {}
    for i in range(n):
        arr = _crop(seed=seed + i)
        fname = f"plate1_A01_1_{i}.png"
        _write_legacy(folder / fname, arr)
        arrays[fname] = arr
    return str(folder), arrays


def _make_field(h=64, w=72, n_channels=4, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 4000, size=(h, w, n_channels + 3)).astype(np.uint16)
    for d in (CELL_DIM, NUC_DIM, PATH_DIM):
        data[:, :, d] = 0
    for label, y0, y1, x0, x1 in [(1, 8, 26, 10, 30), (2, 36, 56, 40, 66)]:
        data[y0:y1, x0:x1, CELL_DIM] = label
        data[y0 + 2:y1 - 2, x0 + 2:x1 - 2, NUC_DIM] = label
    return data


# ===========================================================================
# 1. The correctness assertion, stated the right way round
# ===========================================================================

def test_new_writer_puts_png_dims_channel_n_in_slot_n(tmp_path):
    """THE test. A crop written by the new writer opens with png_dims[N] in slot N.

    Read with plain ``PIL.Image.open`` -- no spaCR reader, no marker, no
    cleverness. What a naive consumer sees off the file has to already be
    right, because that is what "the format is fixed" means.
    """
    arr = _crop(seed=1)
    path = _write_current(tmp_path / "cell_png" / "a.png", arr)

    with Image.open(path) as img:
        assert img.mode == "RGB"
        seen = np.array(img)

    for n in range(3):
        assert np.array_equal(seen[:, :, n], arr[:, :, n] // 256), (
            f"png_dims[{n}] must be slot {n} of the PNG")
    # And it is genuinely not the old order.
    assert not np.array_equal(seen, (arr // 256)[:, :, ::-1])


def test_legacy_writer_is_the_reverse_and_that_is_the_bug(tmp_path):
    """The old writer's output, shown for what it is: png_dims reversed."""
    arr = _crop(seed=2)
    path = _write_legacy(tmp_path / "cell_png" / "a.png", arr)
    with Image.open(path) as img:
        seen = np.array(img)
    assert np.array_equal(seen[:, :, 0], arr[:, :, 2] // 256)
    assert np.array_equal(seen[:, :, 2], arr[:, :, 0] // 256)


def test_to_cv2_bgr_channel_counts(tmp_path):
    """1 channel and 2-D pass through, 2 is padded, 4+ is refused."""
    two_d = np.arange(6, dtype=np.uint16).reshape(2, 3)
    assert np.array_equal(to_cv2_bgr(two_d), two_d)

    one = two_d[:, :, None]
    assert np.array_equal(to_cv2_bgr(one), one)

    two = np.stack([np.full((2, 3), 5, np.uint16),
                    np.full((2, 3), 7, np.uint16)], axis=-1)
    out = to_cv2_bgr(two)
    assert out.shape == (2, 3, 3)
    # padded plane (index 2 of the crop) is written to cv2's slot 0 = file blue
    assert (out[:, :, 0] == 0).all() and (out[:, :, 2] == 5).all()

    with pytest.raises(CropError, match="at most 3"):
        to_cv2_bgr(np.zeros((2, 3, 4), np.uint16))
    with pytest.raises(CropError, match="2-D or"):
        to_cv2_bgr(np.zeros((2, 3, 4, 5), np.uint16))


# ===========================================================================
# 2. Marker resolution
# ===========================================================================

def test_unmarked_folder_is_legacy_and_is_read_correctly(tmp_path):
    """An unmarked folder is BGR -- the only default that cannot corrupt data."""
    folder, arrays = _legacy_folder(tmp_path)
    assert read_crop_folder_marker(folder) is None
    assert crop_folder_format(folder) == CROP_FORMAT_LEGACY_BGR

    for name, arr in arrays.items():
        got = read_crop_png(os.path.join(folder, name))
        assert np.array_equal(got, png_view(arr)), name
        # i.e. the user's png_dims[0] comes back as red, from a file that
        # physically holds it in blue.
        assert np.array_equal(got[:, :, 0], arr[:, :, 0] // 256)


def test_marked_folder_is_read_as_is(tmp_path):
    """A folder stamped format 2 is taken at face value, no reversal applied."""
    folder = tmp_path / "cell_png"
    arr = _crop(seed=3)
    path = _write_current(folder / "a.png", arr)

    marker = read_crop_folder_marker(str(folder))
    assert marker["spacr_crop_format"] == CROP_FORMAT_RGB
    assert marker["channel_order"] == "rgb"
    assert crop_folder_format(str(folder)) == CROP_FORMAT_RGB

    got = read_crop_png(path)
    assert np.array_equal(got, png_view(arr))
    with Image.open(path) as img:
        assert np.array_equal(got, np.array(img))     # nothing was reordered


def test_marker_survives_a_folder_copy(tmp_path):
    """The sidecar is inside the folder, so copying the folder carries it."""
    import shutil

    src = tmp_path / "cell_png"
    arr = _crop(seed=4)
    _write_current(src / "a.png", arr)

    dst = tmp_path / "elsewhere" / "cell_png"
    shutil.copytree(src, dst)
    crops.clear_crop_format_cache()

    assert os.path.isfile(dst / CROP_FORMAT_SIDECAR)
    assert crop_folder_format(str(dst)) == CROP_FORMAT_RGB
    assert np.array_equal(read_crop_png(str(dst / "a.png")), png_view(arr))


def test_a_corrupt_sidecar_is_treated_as_absent(tmp_path):
    """A marker that will not parse must not outrank "no marker"."""
    folder = tmp_path / "cell_png"
    folder.mkdir()
    (folder / CROP_FORMAT_SIDECAR).write_text("{not json")
    assert read_crop_folder_marker(str(folder)) is None
    assert crop_folder_format(str(folder)) == CROP_FORMAT_LEGACY_BGR

    (folder / CROP_FORMAT_SIDECAR).write_text(json.dumps({"spacr_crop_format": 99}))
    crops.clear_crop_format_cache()
    assert read_crop_folder_marker(str(folder)) is None


def test_stamp_is_idempotent_and_written_atomically(tmp_path):
    """Stamping twice writes one marker; the write leaves no temp files behind."""
    folder = tmp_path / "cell_png"
    folder.mkdir()
    first = stamp_crop_folder(str(folder))
    second = stamp_crop_folder(str(folder))          # remembered in-process
    assert first == second
    leftovers = [n for n in os.listdir(folder) if n.startswith(".spacr_tmp_")]
    assert leftovers == []
    assert sorted(os.listdir(folder)) == [CROP_FORMAT_SIDECAR]

    # A fresh process (a second measure worker, say) finds the marker on disk
    # and leaves it alone rather than rewriting it under the first one.
    before = os.stat(first).st_mtime_ns
    crops.clear_crop_format_cache()
    assert stamp_crop_folder(str(folder)) == first
    assert os.stat(first).st_mtime_ns == before


def test_writing_new_crops_into_an_old_folder_is_called_out(tmp_path, capsys):
    """One marker cannot describe a folder holding both formats -- so say so.

    Re-measuring into a crop folder that still holds pre-fix crops is the one
    case the folder-level marker cannot express. The run marks its own output
    and names the files that were there first, instead of quietly declaring
    them corrected.
    """
    folder, _ = _legacy_folder(tmp_path, n=3, seed=95)
    crops.clear_crop_format_cache()
    assert stamp_crop_folder(folder) is not None
    out = capsys.readouterr().out
    assert "already holds 3 unmarked crop PNG(s)" in out
    assert "python -m spacr.crops" in out
    assert crop_folder_format(folder) == CROP_FORMAT_RGB

    # A folder deliberately marked legacy, then written into by a new run.
    other, _ = _legacy_folder(tmp_path / "b", n=1, seed=96)
    migrate_crop_folder(other, mode="mark")
    crops.clear_crop_format_cache()
    stamp_crop_folder(other)
    out = capsys.readouterr().out
    assert "is marked crop format 1 (bgr)" in out
    assert crop_folder_format(other) == CROP_FORMAT_RGB

    # An empty folder is the normal case and says nothing at all.
    fresh = tmp_path / "fresh_png"
    fresh.mkdir()
    stamp_crop_folder(str(fresh))
    assert capsys.readouterr().out == ""

    # Recording an unmarked folder as legacy is just making the default
    # explicit -- nothing changes, so nothing is announced.
    plain, _ = _legacy_folder(tmp_path / "c", n=1, seed=97)
    crops.clear_crop_format_cache()
    stamp_crop_folder(plain, CROP_FORMAT_LEGACY_BGR)
    assert capsys.readouterr().out == ""
    assert crop_folder_format(plain) == CROP_FORMAT_LEGACY_BGR


def test_write_crop_folder_marker_rejects_an_unknown_format(tmp_path):
    with pytest.raises(CropError, match="unknown crop format"):
        crops.write_crop_folder_marker(str(tmp_path), 7)


def test_stamp_failure_warns_instead_of_killing_the_run(tmp_path, monkeypatch, capsys):
    """A marker that cannot be written must be loud, but must not abort measure."""
    folder = tmp_path / "cell_png"
    folder.mkdir()

    def _boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(crops, "write_crop_folder_marker", _boom)
    assert stamp_crop_folder(str(folder)) is None
    out = capsys.readouterr().out
    assert "crop-format marker" in out and "reversed" in out


# ===========================================================================
# 3. Sidecar vs database column
# ===========================================================================

def _png_list_db(tmp_path, rows, fmt=None):
    """Build a measurements.db with a png_list table, optionally stamped."""
    db_dir = tmp_path / "measurements"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "measurements.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE png_list (png_path TEXT)")
    conn.executemany("INSERT INTO png_list VALUES (?)", [(r,) for r in rows])
    conn.commit()
    conn.close()
    if fmt is not None:
        crops.stamp_crop_format_in_db(db_path, None, fmt)
    return db_path


def test_db_column_is_used_when_there_is_no_sidecar(tmp_path):
    """With no sidecar the database column answers -- better than guessing legacy."""
    folder = tmp_path / "data" / "plate1_A01" / "cell_png"
    arr = _crop(seed=5)
    path = str(folder / "a.png")
    os.makedirs(folder, exist_ok=True)
    assert cv2.imwrite(path, to_cv2_bgr(arr))          # format 2 content...
    assert read_crop_folder_marker(str(folder)) is None  # ...but no sidecar

    db_path = _png_list_db(tmp_path, [path], fmt=CROP_FORMAT_RGB)
    assert crops.read_db_crop_format(db_path) == CROP_FORMAT_RGB
    assert crop_folder_format(str(folder), db_path) == CROP_FORMAT_RGB
    assert np.array_equal(read_crop_png(path, db_path=db_path), png_view(arr))
    # Without the database it falls back to legacy and reads it reversed.
    assert not np.array_equal(read_crop_png(path), png_view(arr))


def test_sidecar_wins_over_a_disagreeing_db_column(tmp_path, capsys):
    """Documented resolution: the sidecar wins, and the conflict is reported.

    The sidecar travels with the crops; a database row does not. A folder
    copied to another machine keeps its sidecar and loses its database, so the
    marker that is still there when they disagree is the one to trust.
    """
    folder = tmp_path / "data" / "plate1_A01" / "cell_png"
    arr = _crop(seed=6)
    path = _write_current(folder / "a.png", arr)          # sidecar says 2
    db_path = _png_list_db(tmp_path, [path], fmt=CROP_FORMAT_LEGACY_BGR)  # db says 1

    assert crop_folder_format(str(folder), db_path) == CROP_FORMAT_RGB
    out = capsys.readouterr().out
    assert "crop format conflict" in out
    assert "Using the sidecar" in out

    # The pixels follow the sidecar, so the crop is read correctly.
    assert np.array_equal(read_crop_png(path, db_path=db_path), png_view(arr))

    # And a caller that wants the disagreement to be fatal can have that.
    with pytest.raises(CropFormatConflict, match="crop format conflict"):
        crop_folder_format(str(folder), db_path, strict=True)


def test_db_helpers_are_quiet_when_there_is_nothing_to_read(tmp_path):
    assert crops.read_db_crop_format(None) is None
    assert crops.read_db_crop_format(str(tmp_path / "nope.db")) is None
    db_path = _png_list_db(tmp_path, ["/a/b.png"])
    assert crops.read_db_crop_format(db_path) is None          # no column yet
    assert crops.stamp_crop_format_in_db(db_path, ["/a/b.png"]) == 1
    assert crops.read_db_crop_format(db_path, "/a/b.png") == CROP_FORMAT_CURRENT
    # An unknown path falls back to the table-wide answer rather than None.
    assert crops.read_db_crop_format(db_path, "/somewhere/else.png") == CROP_FORMAT_CURRENT
    assert crops.stamp_crop_format_in_db(str(tmp_path / "gone.db")) == 0
    with pytest.raises(CropError, match="unknown crop format"):
        crops.stamp_crop_format_in_db(db_path, None, 42)


def test_the_db_answer_is_asked_for_once_not_once_per_crop(tmp_path, monkeypatch):
    """The fallback query is a full-table SELECT DISTINCT; a grid must not repeat it."""
    folder = tmp_path / "data" / "plate1_A01" / "cell_png"
    folder.mkdir(parents=True)
    db_path = _png_list_db(tmp_path, ["/a.png"], fmt=CROP_FORMAT_RGB)

    calls = []
    real = crops.read_db_crop_format
    monkeypatch.setattr(crops, "read_db_crop_format",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    for _ in range(5):
        assert crop_folder_format(str(folder), db_path) == CROP_FORMAT_RGB
    assert len(calls) == 1

    # Changing what the database says invalidates it.
    crops.stamp_crop_format_in_db(db_path, None, CROP_FORMAT_LEGACY_BGR)
    assert crop_folder_format(str(folder), db_path) == CROP_FORMAT_LEGACY_BGR


def test_a_db_holding_both_formats_is_ambiguous_not_a_guess(tmp_path):
    db_path = _png_list_db(tmp_path, ["/a.png", "/b.png"])
    crops.stamp_crop_format_in_db(db_path, ["/a.png"], CROP_FORMAT_RGB)
    crops.stamp_crop_format_in_db(db_path, ["/b.png"], CROP_FORMAT_LEGACY_BGR)
    assert crops.read_db_crop_format(db_path) is None
    assert crops.read_db_crop_format(db_path, "/a.png") == CROP_FORMAT_RGB


# ===========================================================================
# 4. png_view and the PNG path agree, under both formats
# ===========================================================================

@pytest.mark.parametrize("fmt", [CROP_FORMAT_LEGACY_BGR, CROP_FORMAT_RGB])
def test_png_view_and_the_png_path_agree_under_both_formats(tmp_path, fmt):
    """The load-bearing equality: an on-demand crop == the crop read off disk.

    ``png_view`` exists so a crop cut from ``merged/*.npy`` is pixel-identical
    to the one in the PNG folder. That has to hold for a legacy folder *and*
    for a new one, or an annotation made on an old dataset stops being
    comparable with a model trained on a new one.
    """
    data = _make_field(seed=7)
    merged = tmp_path / "merged"
    merged.mkdir()
    npy = str(merged / "plate1_A01_1.npy")
    np.save(npy, data)

    folder = tmp_path / "data" / "plate1_A01" / "cell_png"
    for label in (1, 2):
        crop = extract_crop(npy, "cell", label, channels=(0, 1, 2),
                            size=(32, 32), mask_dims=MASK_DIMS)
        target = str(folder / f"plate1_A01_1_{label}.png")
        if fmt == CROP_FORMAT_LEGACY_BGR:
            _write_legacy(target, crop)
        else:
            _write_current(target, crop)

        assert crop_format_for_png(target) == fmt
        assert np.array_equal(read_crop_png(target), png_view(crop))

        # ...and through the two CropSource implementations, which is how the
        # GUIs and the datasets actually reach a crop.
        png_src = PngCropSource(root=str(tmp_path))
        merged_src = MergedCropSource(
            spec=CropSpec(merged_path="", channels=(0, 1, 2), size=(32, 32),
                          mask_dims=MASK_DIMS))
        assert np.array_equal(
            png_src.get({"png_path": target}),
            merged_src.get({"path_name": npy, "object_label": label}))


# ===========================================================================
# 5. The 16-bit narrowing is one behaviour
# ===========================================================================

def test_16_bit_narrowing_is_one_rule_for_rgb_and_single_channel(tmp_path):
    """One narrowing: the high byte. Not PIL's high-byte-or-clip coin flip.

    A 16-bit RGB PNG comes back from PIL as ``value // 256``; a 16-bit
    single-channel one loads as mode ``I;16`` and ``convert('RGB')`` CLIPS it
    at 255, so any crop brighter than that was solid white. Same file format,
    same bit depth, two incompatible answers depending on the channel count.
    """
    folder = tmp_path / "cell_png"
    rgb = np.full((4, 5, 3), 40000, np.uint16)
    rgb[:, :, 1] = 300
    gray = np.full((4, 5, 1), 40000, np.uint16)

    p_rgb = _write_current(folder / "rgb.png", rgb)
    p_gray = _write_current(folder / "gray.png", gray)

    got_rgb = read_crop_png(p_rgb)
    got_gray = read_crop_png(p_gray)

    assert (got_rgb[:, :, 0] == 40000 // 256).all()
    assert (got_rgb[:, :, 1] == 300 // 256).all()
    assert (got_gray == 40000 // 256).all()
    # The two channel counts agree on what 40000 narrows to. Under PIL's rules
    # they did not: the single-channel one saturated.
    assert got_gray[0, 0, 0] == got_rgb[0, 0, 0]
    with Image.open(p_gray) as img:
        assert img.mode.startswith("I")
        assert np.array(img.convert("RGB")).max() == 255     # what we no longer do

    # png_view narrows the same way, so the on-demand path agrees.
    assert np.array_equal(png_view(rgb), got_rgb)
    assert np.array_equal(png_view(gray), got_gray)


def test_narrow_to_uint8_rules():
    assert narrow_to_uint8(np.array([[7]], np.uint8))[0, 0] == 7
    assert narrow_to_uint8(np.array([[40000]], np.uint16))[0, 0] == 40000 // 256
    assert narrow_to_uint8(np.array([[70000]], np.int32))[0, 0] == 255
    assert narrow_to_uint8(np.array([[-5]], np.int32))[0, 0] == 0
    assert narrow_to_uint8(np.array([[300.0]]))[0, 0] == 255
    assert narrow_to_uint8(np.array([[200]], np.int8))[0, 0] == 0   # int8 max 127


def test_read_crop_png_forced_format_and_missing_file(tmp_path):
    folder = tmp_path / "cell_png"
    arr = _crop(seed=8)
    path = _write_current(folder / "a.png", arr)
    # "the file is legacy" (it is not) -> read as if it were, i.e. reversed.
    forced = read_crop_png(path, fmt=CROP_FORMAT_LEGACY_BGR)
    assert np.array_equal(forced, png_view(arr)[:, :, ::-1])
    # "give me legacy-ordered pixels" out of a format-2 file: same array,
    # asked for the other way round.
    assert np.array_equal(
        read_crop_png(path, as_format=CROP_FORMAT_LEGACY_BGR), forced)
    # Asking for both is asking for no change at all.
    assert np.array_equal(
        read_crop_png(path, fmt=CROP_FORMAT_LEGACY_BGR,
                      as_format=CROP_FORMAT_LEGACY_BGR),
        png_view(arr))
    with pytest.raises(CropError, match="unknown crop format"):
        read_crop_png(path, as_format=9)
    from spacr.crops import MergedFileMissing
    with pytest.raises(MergedFileMissing, match="crop PNG not found"):
        read_crop_png(str(folder / "absent.png"))


def test_read_crop_png_handles_palette_and_grayscale_8bit(tmp_path):
    """Odd PNG modes still come back as (H, W, 3) uint8 rather than exploding."""
    folder = tmp_path / "cell_png"
    folder.mkdir()
    stamp_crop_folder(str(folder))
    gray = (np.arange(20, dtype=np.uint8).reshape(4, 5))
    Image.fromarray(gray, mode="L").save(folder / "l.png")
    Image.fromarray(gray, mode="L").convert("P").save(folder / "p.png")
    for name in ("l.png", "p.png"):
        out = read_crop_png(str(folder / name))
        assert out.shape == (4, 5, 3) and out.dtype == np.uint8


# ===========================================================================
# 6. The migrator
# ===========================================================================

def test_migrator_converts_stamps_and_is_a_no_op_second_time(tmp_path):
    """One shot: convert, stamp, and refuse to do it again."""
    folder, arrays = _legacy_folder(tmp_path, n=5)
    before = {n: read_crop_png(os.path.join(folder, n)) for n in arrays}

    result = migrate_crop_folder(folder)
    assert sorted(result.converted) == sorted(arrays)
    assert result.failed == [] and not result.already

    marker = read_crop_folder_marker(folder)
    assert marker["spacr_crop_format"] == CROP_FORMAT_RGB
    assert marker["migrated_from"] == CROP_FORMAT_LEGACY_BGR
    assert "migration" not in marker            # the journal is gone when done
    assert crop_folder_format(folder) == CROP_FORMAT_RGB

    for name, arr in arrays.items():
        path = os.path.join(folder, name)
        # The file now physically holds png_dims[0] in red...
        with Image.open(path) as img:
            assert np.array_equal(np.array(img)[:, :, 0], arr[:, :, 0] // 256)
        # ...and full bit depth was preserved, not narrowed on the way through.
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert raw.dtype == np.uint16
        # ...and what a reader gets back is unchanged by the migration.
        assert np.array_equal(read_crop_png(path), before[name])
        assert np.array_equal(read_crop_png(path), png_view(arr))

    # Second run: nothing decoded, nothing written.
    again = migrate_crop_folder(folder)
    assert again.already and again.converted == []
    for name in arrays:
        assert np.array_equal(read_crop_png(os.path.join(folder, name)),
                              before[name])

    # No staging or temp files survive.
    assert not [n for n in os.listdir(folder)
                if n.endswith(crops.CROP_MIGRATION_SUFFIX)
                or n.startswith(".spacr_tmp_")]


def test_migrator_leaves_no_stray_files_and_reports_a_dry_run(tmp_path):
    folder, arrays = _legacy_folder(tmp_path, n=3)
    plan = migrate_crop_folder(folder, dry_run=True)
    assert sorted(plan.converted) == sorted(arrays)
    assert plan.dry_run
    assert read_crop_folder_marker(folder) is None      # nothing written
    assert sorted(os.listdir(folder)) == sorted(arrays)


def test_migrator_skips_single_channel_crops_but_still_marks_them(tmp_path):
    """A grayscale crop has no channel order to fix; only the marker changes."""
    folder = tmp_path / "cell_png"
    gray = np.full((4, 5, 1), 40000, np.uint16)
    _write_legacy(folder / "g.png", gray)
    before = read_crop_png(str(folder / "g.png"))

    result = migrate_crop_folder(str(folder))
    assert result.converted == [] and result.skipped == ["g.png"]
    assert crop_folder_format(str(folder)) == CROP_FORMAT_RGB
    assert np.array_equal(read_crop_png(str(folder / "g.png")), before)


def test_mark_mode_records_legacy_without_touching_a_pixel(tmp_path):
    """The escape hatch for a folder something outside spaCR reads bit for bit."""
    folder, arrays = _legacy_folder(tmp_path, n=2)
    digests = {n: open(os.path.join(folder, n), "rb").read() for n in arrays}

    result = migrate_crop_folder(folder, mode="mark")
    assert result.converted == []
    assert crop_folder_format(folder) == CROP_FORMAT_LEGACY_BGR
    for name, blob in digests.items():
        assert open(os.path.join(folder, name), "rb").read() == blob
    # Marked legacy still reads correctly, because the reader converts.
    for name, arr in arrays.items():
        assert np.array_equal(read_crop_png(os.path.join(folder, name)),
                              png_view(arr))

    assert migrate_crop_folder(folder, mode="mark").already

    # A dry run of mark mode reports and writes nothing.
    fresh, _ = _legacy_folder(tmp_path / "b", n=1, seed=91)
    assert migrate_crop_folder(fresh, mode="mark", dry_run=True).mode == "mark"
    assert read_crop_folder_marker(fresh) is None

    # Marking a converted folder legacy would reverse every crop in it: refuse.
    migrate_crop_folder(folder)
    with pytest.raises(CropError, match="would make every crop"):
        migrate_crop_folder(folder, mode="mark")


def test_migrator_argument_validation(tmp_path):
    folder, _ = _legacy_folder(tmp_path, n=1)
    with pytest.raises(CropError, match="mode must be"):
        migrate_crop_folder(folder, mode="sideways")
    with pytest.raises(CropError, match="on_error must be"):
        migrate_crop_folder(folder, on_error="shrug")
    with pytest.raises(CropError, match="not a crop folder"):
        migrate_crop_folder(str(tmp_path / "nowhere"))


def test_migrator_reports_progress_and_stamps_the_db(tmp_path):
    folder, arrays = _legacy_folder(tmp_path, n=3)
    db_path = _png_list_db(
        tmp_path, [os.path.join(folder, n) for n in sorted(arrays)])
    seen = []
    result = migrate_crop_folder(folder, db_path=db_path,
                                 progress=lambda d, t, n: seen.append((d, t, n)))
    assert len(result.converted) == 3
    assert [d for d, _, _ in seen] == [1, 2, 3]
    assert crops.read_db_crop_format(db_path) == CROP_FORMAT_RGB


# ---------------------------------------------------------------------------
# Interruption
# ---------------------------------------------------------------------------

def _kill_after(monkeypatch, target, n, predicate):
    """Make ``target`` raise the (n+1)-th time ``predicate`` holds."""
    calls = {"n": 0}
    real = getattr(crops.os, target)

    def _wrapped(src, dst, *a, **k):
        if predicate(str(dst)):
            if calls["n"] >= n:
                raise KeyboardInterrupt("simulated kill")
            calls["n"] += 1
        return real(src, dst, *a, **k)

    monkeypatch.setattr(crops.os, target, _wrapped)


@pytest.mark.parametrize("kill_after", [0, 1, 2])
def test_an_interrupted_migration_leaves_no_corrupt_file(tmp_path, monkeypatch,
                                                         kill_after):
    """Kill the migration between the watermark and the install, at every point.

    The invariant: every crop at its real name is a complete, decodable PNG at
    all times, every one of them still resolves to the right format, and
    ``read_crop_png`` returns the same pixels it did before the migration
    started. Re-running finishes the job, and nothing gets reversed twice.
    """
    folder, arrays = _legacy_folder(tmp_path, n=4, seed=20)
    before = {n: read_crop_png(os.path.join(folder, n)) for n in arrays}

    # Blow up on the (kill_after+1)-th install of a converted file, i.e. the
    # os.replace whose destination is the crop's real name.
    _kill_after(monkeypatch, "replace", kill_after,
                lambda dst: dst.endswith(".png"))
    with pytest.raises(KeyboardInterrupt):
        migrate_crop_folder(folder)
    monkeypatch.undo()
    crops.clear_crop_format_cache()

    # Nothing is corrupt, and everything reads back exactly as before.
    for name, arr in arrays.items():
        path = os.path.join(folder, name)
        assert os.path.isfile(path)
        assert cv2.imread(path, cv2.IMREAD_UNCHANGED) is not None, f"{name} unreadable"
        assert np.array_equal(read_crop_png(path), before[name]), name
        assert np.array_equal(read_crop_png(path), png_view(arr)), name

    # Mixed folder: some files converted, some not, each resolved individually.
    formats = {n: crop_format_for_png(os.path.join(folder, n))
               for n in sorted(arrays)}
    assert sorted(formats.values(), reverse=True)[:kill_after] == \
        [CROP_FORMAT_RGB] * kill_after
    assert CROP_FORMAT_LEGACY_BGR in formats.values()

    # Resume: completes, and no crop was reversed twice.
    result = migrate_crop_folder(folder)
    assert not result.already
    assert crop_folder_format(folder) == CROP_FORMAT_RGB
    for name, arr in arrays.items():
        assert np.array_equal(read_crop_png(os.path.join(folder, name)),
                              png_view(arr)), name
    assert not [n for n in os.listdir(folder)
                if n.endswith(crops.CROP_MIGRATION_SUFFIX)]


def test_an_interrupted_conversion_leaves_no_partial_staging_file(tmp_path,
                                                                  monkeypatch):
    """A crash mid-encode must not leave a truncated file anywhere."""
    folder, arrays = _legacy_folder(tmp_path, n=3, seed=30)
    before = {n: read_crop_png(os.path.join(folder, n)) for n in arrays}

    real_imwrite = cv2.imwrite
    calls = {"n": 0}

    def _flaky(path, arr, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated kill mid-encode")
        return real_imwrite(path, arr, *a, **k)

    monkeypatch.setattr(cv2, "imwrite", _flaky)
    with pytest.raises(KeyboardInterrupt):
        migrate_crop_folder(folder)
    monkeypatch.undo()
    crops.clear_crop_format_cache()

    assert not [n for n in os.listdir(folder) if n.startswith(".spacr_tmp_")]
    for name in arrays:
        path = os.path.join(folder, name)
        assert cv2.imread(path, cv2.IMREAD_UNCHANGED) is not None
        assert np.array_equal(read_crop_png(path), before[name]), name

    migrate_crop_folder(folder)
    for name in arrays:
        assert np.array_equal(read_crop_png(os.path.join(folder, name)),
                              before[name]), name


def test_a_leftover_staging_file_is_installed_not_re_converted(tmp_path):
    """"A staging file exists" outranks the watermark, in both directions."""
    folder, arrays = _legacy_folder(tmp_path, n=2, seed=40)
    name = sorted(arrays)[0]
    path = os.path.join(folder, name)
    before = read_crop_png(path)

    # Hand-build the state a crash between the watermark and the install
    # leaves: staged content ready, watermark already past it.
    staged = path + crops.CROP_MIGRATION_SUFFIX
    scratch = os.path.join(folder, "staged_tmp.png")
    assert cv2.imwrite(scratch, to_cv2_bgr(cv2.imread(path, cv2.IMREAD_UNCHANGED)))
    os.replace(scratch, staged)
    crops.write_crop_folder_marker(
        folder, CROP_FORMAT_RGB,
        migration={"from": CROP_FORMAT_LEGACY_BGR, "done_through": name,
                   "started_utc": "now"})

    # The watermark says "done", the staging file says "not installed yet" --
    # and the staging file is right.
    assert crop_format_for_png(path) == CROP_FORMAT_LEGACY_BGR
    assert np.array_equal(read_crop_png(path), before)

    migrate_crop_folder(folder)
    assert np.array_equal(read_crop_png(path), before)
    assert not os.path.exists(staged)


def test_a_file_that_cannot_be_converted_is_loud_or_recorded(tmp_path):
    """A 4-channel crop cannot be written safely: raise, or record and move on."""
    folder = tmp_path / "cell_png"
    good = _crop(seed=50)
    _write_legacy(folder / "a_good.png", good)
    four = np.full((4, 5, 4), 30000, np.uint16)
    _write_legacy(folder / "b_bad.png", four)
    _write_legacy(folder / "c_good.png", _crop(seed=51))

    with pytest.raises(CropError, match="could not be converted"):
        migrate_crop_folder(str(folder))
    # The failure stopped everything after it; the folder is still readable.
    crops.clear_crop_format_cache()
    assert np.array_equal(read_crop_png(str(folder / "a_good.png")),
                          png_view(good))

    result = migrate_crop_folder(str(folder), on_error="skip")
    assert [n for n, _ in result.failed] == ["b_bad.png"]
    assert "c_good.png" in result.converted
    marker = read_crop_folder_marker(str(folder))
    assert marker["spacr_crop_format"] == CROP_FORMAT_RGB
    assert marker["unconverted"] == ["b_bad.png"]
    assert np.array_equal(read_crop_png(str(folder / "a_good.png")),
                          png_view(good))

    # The one file that could not be rewritten is STILL legacy, inside a
    # folder the marker calls format 2 -- so it has to be resolved per file,
    # or it would be read reversed from here on.
    assert crop_format_for_png(str(folder / "b_bad.png")) == CROP_FORMAT_LEGACY_BGR
    assert crop_format_for_png(str(folder / "a_good.png")) == CROP_FORMAT_RGB

    # A later run retries only the leftovers, and touches nothing else.
    replay = migrate_crop_folder(str(folder), on_error="skip")
    assert not replay.already
    assert [n for n, _ in replay.failed] == ["b_bad.png"]
    assert sorted(replay.skipped) == ["a_good.png", "c_good.png"]
    assert np.array_equal(read_crop_png(str(folder / "a_good.png")),
                          png_view(good))

    # Fix the file and the retry finishes the job.
    _write_legacy(folder / "b_bad.png", _crop(seed=52))
    fixed = _crop(seed=52)
    done = migrate_crop_folder(str(folder))
    assert done.converted == ["b_bad.png"]
    assert "unconverted" not in read_crop_folder_marker(str(folder))
    assert np.array_equal(read_crop_png(str(folder / "b_bad.png")),
                          png_view(fixed))
    assert np.array_equal(read_crop_png(str(folder / "a_good.png")),
                          png_view(good))
    assert migrate_crop_folder(str(folder)).already


def test_an_undecodable_png_is_reported_by_name(tmp_path):
    folder = tmp_path / "cell_png"
    folder.mkdir()
    (folder / "junk.png").write_bytes(b"not a png at all")
    with pytest.raises(CropError, match="junk.png could not be converted"):
        migrate_crop_folder(str(folder))


# ---------------------------------------------------------------------------
# Whole-tree migration
# ---------------------------------------------------------------------------

def test_find_and_migrate_a_whole_experiment_tree(tmp_path):
    root = tmp_path / "exp"
    arrays = {}
    for well in ("plate1_A01", "plate1_A02"):
        for mode in ("cell_png", "nucleus_png"):
            f = root / "data" / well / mode
            arr = _crop(seed=hash((well, mode)) % 1000)
            _write_legacy(f / "x.png", arr)
            arrays[str(f / "x.png")] = arr

    folders = crops.find_crop_folders(str(root))
    assert len(folders) == 4
    assert crops.find_crop_folders(folders[0]) == [folders[0]]

    results = crops.migrate_crop_tree(str(root))
    assert len(results) == 4
    assert all("converted 1 crop" in r.describe() for r in results)
    for path, arr in arrays.items():
        assert np.array_equal(read_crop_png(path), png_view(arr))

    assert all(r.already for r in crops.migrate_crop_tree(str(root)))
    with pytest.raises(CropError, match="no '\\*_png' crop folders"):
        crops.migrate_crop_tree(str(tmp_path / "empty"))


def test_find_crop_folders_without_a_data_directory(tmp_path):
    """A folder handed straight to the migrator, with no data/ layer above it."""
    root = tmp_path / "loose"
    _write_legacy(root / "plate1_A01" / "cell_png" / "a.png", _crop(seed=76))
    assert crops.find_crop_folders(str(root)) == [
        str(root / "plate1_A01" / "cell_png")]
    assert crops.find_crop_folders(str(tmp_path / "absent")) == []

    # An empty data/ does not shadow crop folders sitting beside it.
    (root / "data").mkdir()
    assert crops.find_crop_folders(str(root)) == [
        str(root / "plate1_A01" / "cell_png")]


def test_migration_result_describe_covers_every_shape(tmp_path):
    r = crops.MigrationResult(folder="f", already=True)
    assert "already format" in r.describe()
    r = crops.MigrationResult(folder="f", mode="mark")
    assert "marked as legacy" in r.describe()
    r = crops.MigrationResult(folder="f", converted=["a"], dry_run=True,
                              failed=[("b", "boom")])
    assert "would convert 1" in r.describe() and "FAILED 1" in r.describe()


# ===========================================================================
# 7. The consumers
# ===========================================================================

def test_annotate_engine_loader_corrects_a_legacy_crop(tmp_path):
    """The Qt annotate screen reads through the format-aware loader."""
    from spacr.qt.annotate_engine import load_crop_image

    folder, arrays = _legacy_folder(tmp_path, n=1, seed=60)
    name, arr = next(iter(arrays.items()))
    img = load_crop_image(os.path.join(folder, name))
    assert img.mode == "RGB"
    assert np.array_equal(np.array(img), png_view(arr))


def test_tk_annotate_app_loader_corrects_a_legacy_crop(tmp_path):
    """The Tk AnnotateApp reads through the same reader (no Tk needed)."""
    from spacr.gui_elements import AnnotateApp

    folder, arrays = _legacy_folder(tmp_path, n=1, seed=61)
    name, arr = next(iter(arrays.items()))

    app = AnnotateApp.__new__(AnnotateApp)     # no Tk root, no window
    app.db_path = None
    app.image_size = (32, 32)
    app.percentiles = (1, 99)
    app.normalize_channels = None
    app.channels = None
    app.outline = None
    img, ann = app.load_single_image((os.path.join(folder, name), 1))
    assert ann == 1
    assert img.size == (32, 32)
    # Resized for display, so compare against the corrected crop resized the
    # same way -- the point is the channel order, not the interpolation.
    assert np.array_equal(
        np.array(img), np.array(Image.fromarray(png_view(arr)).resize((32, 32))))

    # A missing crop is still a blank tile, not an exception.
    blank, _ = app.load_single_image((os.path.join(folder, "absent.png"), None))
    assert blank.size == (32, 32)


def test_legacy_png_view_handles_every_shape_it_used_to(tmp_path):
    """The bug-compatible view keeps working for 2-D, float and 2-channel input."""
    two_d = np.array([[1, 2], [3, 4]], np.uint8)
    out = crops.legacy_png_view(two_d)
    assert out.shape == (2, 2, 3) and (out[:, :, 0] == out[:, :, 2]).all()

    assert crops.legacy_png_view(np.full((2, 2, 3), 300.0)).max() == 255

    two = np.dstack([np.full((2, 2), 40000, np.uint16),
                     np.full((2, 2), 20000, np.uint16)])
    out = crops.legacy_png_view(two)
    # Reversed, so the *empty* padded plane came back as red -- the bug.
    assert (out[:, :, 0] == 0).all() and (out[:, :, 2] == 40000 // 256).all()


# ===========================================================================
# 8. Defensive branches -- the ones that only fire when something is wrong
# ===========================================================================

def test_coerce_format_rejects_everything_that_is_not_a_known_format():
    assert crops._coerce_format(None) is None
    assert crops._coerce_format("nonsense") is None
    assert crops._coerce_format(object()) is None
    assert crops._coerce_format(99) is None
    assert crops._coerce_format("2") == CROP_FORMAT_RGB


def test_cache_stamp_of_a_folder_that_is_not_there(tmp_path):
    assert crops._cache_stamp(str(tmp_path / "gone")) == ("gone",)
    assert read_crop_folder_marker(str(tmp_path / "gone")) is None


def test_a_failed_marker_write_removes_its_temp_file(tmp_path, monkeypatch):
    """A half-written marker would read as "no marker" over corrected crops."""
    folder = tmp_path / "cell_png"
    folder.mkdir()

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(crops.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        crops.write_crop_folder_marker(str(folder))
    monkeypatch.undo()
    assert os.listdir(folder) == []          # no temp file, no partial marker

    # ...and if even the cleanup fails, the original error still surfaces.
    real_remove = os.remove
    monkeypatch.setattr(crops.json, "dump", _boom)
    monkeypatch.setattr(crops.os, "remove", _unremovable)
    with pytest.raises(RuntimeError, match="disk full"):
        crops.write_crop_folder_marker(str(folder))
    monkeypatch.undo()
    for name in os.listdir(folder):
        real_remove(os.path.join(folder, name))


def test_db_reads_survive_a_file_that_is_not_a_database(tmp_path, monkeypatch):
    junk = tmp_path / "not.db"
    junk.write_text("hello")
    assert crops.read_db_crop_format(str(junk)) is None      # query fails

    db_path = _png_list_db(tmp_path, ["/a.png"], fmt=CROP_FORMAT_RGB)

    def _no_connect(*a, **k):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(crops.sqlite3, "connect", _no_connect)
    assert crops.read_db_crop_format(db_path) is None        # connect fails


def test_stamping_a_db_without_the_table_is_a_no_op(tmp_path):
    db_path = str(tmp_path / "empty.db")
    sqlite3.connect(db_path).close()
    assert crops.stamp_crop_format_in_db(db_path) == 0


def test_a_folder_mid_migration_reports_legacy_as_a_whole(tmp_path):
    """The folder-wide answer has to be the conservative one while it is mixed."""
    folder, arrays = _legacy_folder(tmp_path, n=2, seed=70)
    crops.write_crop_folder_marker(
        folder, CROP_FORMAT_RGB,
        migration={"from": CROP_FORMAT_LEGACY_BGR, "done_through": "",
                   "started_utc": "now"})
    assert crop_folder_format(folder) == CROP_FORMAT_LEGACY_BGR
    # ...but a file already past the watermark is resolved individually.
    first = sorted(arrays)[0]
    crops.write_crop_folder_marker(
        folder, CROP_FORMAT_RGB,
        migration={"from": CROP_FORMAT_LEGACY_BGR, "done_through": first,
                   "started_utc": "now"})
    assert crop_format_for_png(os.path.join(folder, first)) == CROP_FORMAT_RGB
    assert crop_format_for_png(
        os.path.join(folder, sorted(arrays)[1])) == CROP_FORMAT_LEGACY_BGR


def test_a_file_the_migration_gave_up_on_is_still_read_as_legacy(tmp_path):
    folder, arrays = _legacy_folder(tmp_path, n=2, seed=71)
    names = sorted(arrays)
    crops.write_crop_folder_marker(
        folder, CROP_FORMAT_RGB,
        migration={"from": CROP_FORMAT_LEGACY_BGR, "done_through": names[-1],
                   "started_utc": "now", "unconverted": [names[0]]})
    assert crop_format_for_png(os.path.join(folder, names[0])) == CROP_FORMAT_LEGACY_BGR
    assert crop_format_for_png(os.path.join(folder, names[1])) == CROP_FORMAT_RGB
    assert np.array_equal(read_crop_png(os.path.join(folder, names[0])),
                          png_view(arrays[names[0]]))


def test_an_unlistable_crop_folder_is_reported(tmp_path, monkeypatch):
    folder = tmp_path / "cell_png"
    folder.mkdir()

    def _boom(_p):
        raise PermissionError("no")

    monkeypatch.setattr(crops.os, "listdir", _boom)
    with pytest.raises(CropError, match="cannot list crop folder"):
        migrate_crop_folder(str(folder))


def test_an_encoder_that_silently_returns_false_is_an_error(tmp_path, monkeypatch):
    """cv2.imwrite returns False rather than raising when it cannot encode."""
    folder, _ = _legacy_folder(tmp_path, n=1, seed=72)
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(CropError, match="could not be converted"):
        migrate_crop_folder(folder)


def test_atomic_convert_reraises_even_if_the_cleanup_fails(tmp_path, monkeypatch):
    folder = tmp_path / "cell_png"
    folder.mkdir()
    junk = folder / "junk.png"
    junk.write_bytes(b"not a png")
    real_remove = os.remove

    monkeypatch.setattr(crops.os, "remove", _unremovable)
    with pytest.raises(CropError, match="could not decode"):
        crops._atomic_convert(str(junk), str(junk) + crops.CROP_MIGRATION_SUFFIX)
    monkeypatch.undo()
    for name in os.listdir(folder):
        if name.startswith(".spacr_tmp_"):
            real_remove(os.path.join(folder, name))


def test_mark_mode_also_stamps_the_database(tmp_path):
    folder, arrays = _legacy_folder(tmp_path, n=1, seed=73)
    db_path = _png_list_db(tmp_path, [os.path.join(folder, n) for n in arrays])
    migrate_crop_folder(folder, mode="mark", db_path=db_path)
    assert crops.read_db_crop_format(db_path) == CROP_FORMAT_LEGACY_BGR
    # The sidecar and the database now agree, so nothing is reported.
    assert crop_folder_format(folder, db_path) == CROP_FORMAT_LEGACY_BGR


def test_progress_and_dry_run_over_a_partially_migrated_folder(tmp_path,
                                                               monkeypatch):
    """Resume paths report progress too, and a dry run counts what is left."""
    folder, arrays = _legacy_folder(tmp_path, n=4, seed=74)
    _kill_after(monkeypatch, "replace", 2, lambda dst: dst.endswith(".png"))
    with pytest.raises(KeyboardInterrupt):
        migrate_crop_folder(folder)
    monkeypatch.undo()
    crops.clear_crop_format_cache()

    plan = migrate_crop_folder(folder, dry_run=True)
    assert len(plan.skipped) == 2 and len(plan.converted) == 2

    seen = []
    result = migrate_crop_folder(folder, progress=lambda d, t, n: seen.append(n))
    assert len(seen) == 4                      # skipped files report too
    assert len(result.skipped) == 2 and len(result.converted) == 2


def test_progress_fires_for_a_file_that_could_not_be_converted(tmp_path):
    folder = tmp_path / "cell_png"
    _write_legacy(folder / "a.png", np.full((4, 5, 4), 30000, np.uint16))
    _write_legacy(folder / "b.png", _crop(seed=75))
    seen = []
    result = migrate_crop_folder(str(folder), on_error="skip",
                                 progress=lambda d, t, n: seen.append(n))
    assert seen == ["a.png", "b.png"]
    assert [n for n, _ in result.failed] == ["a.png"]


def test_command_line_migrator(tmp_path, capsys):
    """``python -m spacr.crops <root>`` is the one-shot the user actually runs."""
    root = tmp_path / "exp"
    folder = root / "data" / "plate1_A01" / "cell_png"
    arr = _crop(seed=90)
    _write_legacy(folder / "a.png", arr)

    assert crops.main([str(root), "--dry-run"]) == 0
    assert "would convert 1" in capsys.readouterr().out
    assert read_crop_folder_marker(str(folder)) is None

    assert crops.main([str(root)]) == 0
    assert "converted 1 crop" in capsys.readouterr().out
    assert crop_folder_format(str(folder)) == CROP_FORMAT_RGB
    assert np.array_equal(read_crop_png(str(folder / "a.png")), png_view(arr))

    assert crops.main([str(tmp_path / "nothing-here")]) == 1
    assert "no '*_png' crop folders" in capsys.readouterr().err


def test_command_line_migrator_mark_and_failures(tmp_path, capsys):
    root = tmp_path / "exp"
    folder = root / "data" / "plate1_A01" / "cell_png"
    _write_legacy(folder / "a.png", _crop(seed=91))
    assert crops.main([str(root), "--mark-legacy"]) == 0
    assert "marked as legacy" in capsys.readouterr().out
    assert crop_folder_format(str(folder)) == CROP_FORMAT_LEGACY_BGR

    bad = tmp_path / "exp2" / "data" / "w" / "cell_png"
    _write_legacy(bad / "b.png", np.full((4, 5, 4), 30000, np.uint16))
    assert crops.main([str(tmp_path / "exp2"), "--skip-errors"]) == 1
    assert "FAILED 1" in capsys.readouterr().out


def test_module_is_runnable_as_a_script():
    """The ``python -m spacr.crops`` entry point really is wired up."""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-m", "spacr.crops", "--help"],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "corrected channel order" in proc.stdout


def test_a_legacy_trained_model_can_still_be_fed_legacy_pixels(tmp_path):
    """The escape hatches for a classifier trained before the fix.

    Its input planes were the file's R/G/B, which in a legacy file are
    ``png_dims`` reversed. Nothing here should silently flip that under it.
    """
    folder, arrays = _legacy_folder(tmp_path, n=1, seed=80)
    name, arr = next(iter(arrays.items()))
    path = os.path.join(folder, name)
    raw = np.array(Image.open(path))              # what the model was fed

    # 1. Mark the folder instead of rewriting it: the bytes never move, so a
    #    raw reader keeps seeing exactly what it trained on...
    migrate_crop_folder(folder, mode="mark")
    assert np.array_equal(np.array(Image.open(path)), raw)
    # ...while spaCR's own reader still shows it the right way round.
    assert np.array_equal(read_crop_png(path), png_view(arr))

    # 2. Ask any folder, in any format, for legacy-ordered pixels by name.
    assert np.array_equal(
        read_crop_png(path, as_format=CROP_FORMAT_LEGACY_BGR), raw)

    # 3. Apply the model to corrected crops with its channels swapped.
    assert crops.legacy_channel_names(["r", "g", "b"]) == ["b", "g", "r"]
    assert crops.legacy_channel_names(["r", "g"]) == ["b", "g"]
    assert crops.legacy_channel_names(["B"]) == ["r"]
    assert crops.legacy_channel_names(["x"]) == ["x"]
    # The swap really does put the model's plane back where it was.
    new_folder = tmp_path / "new" / "cell_png"
    new_path = _write_current(new_folder / "a.png", arr)
    new_seen = np.array(Image.open(new_path))
    order = {"r": 0, "g": 1, "b": 2}
    for legacy_ch in ("r", "g", "b"):
        mapped = crops.legacy_channel_names([legacy_ch])[0]
        assert np.array_equal(raw[:, :, order[legacy_ch]],
                              new_seen[:, :, order[mapped]])


def test_png_crop_source_finds_the_db_next_to_the_root(tmp_path):
    folder = tmp_path / "data" / "plate1_A01" / "cell_png"
    arr = _crop(seed=62)
    path = str(folder / "a.png")
    os.makedirs(folder, exist_ok=True)
    assert cv2.imwrite(path, to_cv2_bgr(arr))       # format 2, unmarked folder
    _png_list_db(tmp_path, [path], fmt=CROP_FORMAT_RGB)

    src = PngCropSource(root=str(tmp_path))
    assert src.db_path is not None
    assert np.array_equal(src.get({"png_path": path}), png_view(arr))

    # No database in sight -> legacy, which is the safe default.
    assert PngCropSource(root=str(tmp_path / "other")).db_path is None
