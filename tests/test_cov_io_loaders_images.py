"""CPU-only coverage for the image/label loading helpers at the top of
``spacr.io``.

Covers the multi-dimensional / non-TIFF splitter
(``process_non_tif_non_2D_images``) including its 4-D/5-D, CZI, ND2 and
error branches, plus the two cellpose-backed loaders
(``_load_images_and_labels`` / ``_load_normalized_images_and_labels``)
including their "file failed to load" and auto-percentile paths.

Everything here is synthetic, offline and sub-second: the CZI/ND2 readers
are stubbed at the ``spacr.io`` module level (no proprietary sample files
needed) and unreadable frames are injected by monkeypatching
``cellpose.io.imread``.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest
import tifffile

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    """Never let figures leak between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ramp(shape, dtype=np.uint16, mod=500, start=0):
    """Deterministic non-constant array of the requested shape."""
    n = int(np.prod(shape))
    return ((np.arange(start, start + n) % mod) + 1).reshape(shape).astype(dtype)


def _write_tif(path, arr):
    tifffile.imwrite(str(path), arr)
    return str(path)


def _label_arr(shape=(16, 16), n=3):
    m = np.zeros(shape, np.uint16)
    for i in range(1, n + 1):
        m[i * 3:i * 3 + 2, 2:5] = i
    return m


# ===========================================================================
# process_non_tif_non_2D_images
# ===========================================================================

def test_split_4d_tif_writes_one_tiff_per_channel_and_z(tmp_path):
    """A (H, W, C, Z) TIFF is split into C*Z grayscale TIFFs named _C{c}_Z{z}."""
    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "four"
    d.mkdir()
    arr = _ramp((8, 9, 2, 3))
    _write_tif(d / "vol.tif", arr)

    process_non_tif_non_2D_images(str(d))

    produced = sorted(p.name for p in d.glob("vol_*.tif"))
    assert produced == [
        "vol_C1_Z1.tif", "vol_C1_Z2.tif", "vol_C1_Z3.tif",
        "vol_C2_Z1.tif", "vol_C2_Z2.tif", "vol_C2_Z3.tif",
    ]
    # Content and bit depth must be preserved slice-for-slice.
    back = tifffile.imread(str(d / "vol_C2_Z3.tif"))
    assert back.dtype == np.uint16
    assert back.shape == (8, 9)
    np.testing.assert_array_equal(back, arr[..., 1, 2])


def test_split_5d_tif_writes_one_tiff_per_channel_z_and_time(tmp_path):
    """A (H, W, C, Z, T) TIFF is split into C*Z*T TIFFs named _C{c}_Z{z}_T{t}."""
    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "five"
    d.mkdir()
    arr = _ramp((6, 7, 2, 2, 2), dtype=np.uint8, mod=200)
    _write_tif(d / "movie.tif", arr)

    process_non_tif_non_2D_images(str(d))

    produced = sorted(p.name for p in d.glob("movie_*.tif"))
    assert len(produced) == 8
    assert produced[0] == "movie_C1_Z1_T1.tif"
    assert "movie_C2_Z2_T2.tif" in produced
    back = tifffile.imread(str(d / "movie_C2_Z2_T2.tif"))
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, arr[..., 1, 1, 1])


def test_czi_file_is_read_through_czifile_and_split(tmp_path, monkeypatch):
    """The .czi branch loads through czifile.CziFile and splits channels."""
    import spacr.io as io_mod
    from spacr.io import process_non_tif_non_2D_images

    arr = _ramp((6, 7, 3))

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def asarray(self):
            return arr

    opened = []

    def _CziFile(path):
        opened.append(path)
        return _Ctx()

    monkeypatch.setattr(io_mod, "czifile",
                        types.SimpleNamespace(CziFile=_CziFile))

    d = tmp_path / "czi"
    d.mkdir()
    (d / "scan.czi").write_bytes(b"stub-czi-payload")

    process_non_tif_non_2D_images(str(d))

    assert opened == [str(d / "scan.czi")], "czifile.CziFile was not used"
    produced = sorted(p.name for p in d.glob("*.tif"))
    assert produced == ["scan_C1.tif", "scan_C2.tif", "scan_C3.tif"]
    np.testing.assert_array_equal(tifffile.imread(str(d / "scan_C1.tif")),
                                  arr[..., 0])


def test_nd2_file_is_read_through_nd2reader_and_split(tmp_path, monkeypatch):
    """The .nd2 branch loads through ND2Reader and splits channels."""
    import spacr.io as io_mod
    from spacr.io import process_non_tif_non_2D_images

    arr = _ramp((5, 6, 2), dtype=np.uint8, mod=200)
    opened = []

    class _FakeND2:
        def __init__(self, path):
            opened.append(path)

        def __enter__(self):
            return arr

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(io_mod, "ND2Reader", _FakeND2)

    d = tmp_path / "nd2"
    d.mkdir()
    (d / "movie.nd2").write_bytes(b"stub-nd2-payload")

    process_non_tif_non_2D_images(str(d))

    assert opened == [str(d / "movie.nd2")], "ND2Reader was not used"
    produced = sorted(p.name for p in d.glob("*.tif"))
    assert produced == ["movie_C1.tif", "movie_C2.tif"]
    back = tifffile.imread(str(d / "movie_C2.tif"))
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, arr[..., 1])


def test_2d_tif_is_left_untouched(tmp_path, capsys):
    """An already-grayscale TIFF is reported as skipped and nothing is written."""
    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "flat"
    d.mkdir()
    arr = _ramp((12, 13))
    _write_tif(d / "plain.tif", arr)
    before = sorted(p.name for p in d.iterdir())

    process_non_tif_non_2D_images(str(d))

    out = capsys.readouterr().out
    assert "already grayscale and in TIFF format" in out
    assert sorted(p.name for p in d.iterdir()) == before
    np.testing.assert_array_equal(tifffile.imread(str(d / "plain.tif")), arr)


def test_unreadable_file_is_reported_not_raised(tmp_path, capsys):
    """A corrupt TIFF is caught and reported; the loop keeps going."""
    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "broken"
    d.mkdir()
    (d / "bad.tif").write_bytes(b"this is definitely not a tiff")
    good = _ramp((4, 5, 2))
    _write_tif(d / "good.tif", good)

    process_non_tif_non_2D_images(str(d))

    out = capsys.readouterr().out
    assert "Error processing bad.tif" in out
    # The healthy multi-channel file after it was still split.
    assert sorted(p.name for p in d.glob("good_*.tif")) == ["good_C1.tif",
                                                           "good_C2.tif"]


def test_unsupported_extension_inside_load_image_is_reported(tmp_path,
                                                             capsys,
                                                             monkeypatch):
    """load_image raises ValueError for an extension it does not know.

    The dispatch table in ``load_image`` is a superset guard: the caller
    already filters on ``supported_formats``, so the only way to execute the
    ``else: raise ValueError`` is to have the extension change between the
    caller's check and load_image's own check. Inject exactly that.
    """
    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "weird"
    d.mkdir()
    target = str(d / "weird.tif")
    _write_tif(target, _ramp((4, 4)))

    real_splitext = os.path.splitext
    seen = {"n": 0}

    def fake_splitext(p):
        if str(p) == target:
            seen["n"] += 1
            if seen["n"] >= 2:  # 1st call = caller's filter, 2nd = load_image
                return (str(p)[:-4], ".xyz")
        return real_splitext(p)

    monkeypatch.setattr(os.path, "splitext", fake_splitext)

    process_non_tif_non_2D_images(str(d))

    out = capsys.readouterr().out
    assert "Unsupported file extension: .xyz" in out
    assert "Error processing weird.tif" in out


def test_split_channels_is_inert_for_2d_input(tmp_path, monkeypatch):
    """split_channels returns immediately when handed a 2-D array.

    In production this early return is dead code: the caller intercepts
    ``ndim == 2`` one line earlier. Feed a malformed array whose reported
    ``ndim`` changes between the two checks to prove the guard is inert
    (nothing written, no exception).
    """
    import spacr.io as io_mod
    from spacr.io import process_non_tif_non_2D_images

    class _FlipNdim(np.ndarray):
        """Reports ndim=3 on the first access, 2 on every later one."""
        _flipped = False

        @property
        def ndim(self):
            if not type(self)._flipped:
                type(self)._flipped = True
                return 3
            return 2

    d = tmp_path / "flip"
    d.mkdir()
    src = _write_tif(d / "odd.tif", _ramp((4, 4)))
    payload = np.zeros((4, 4), np.uint16).view(_FlipNdim)

    monkeypatch.setattr(io_mod.tifffile, "imread", lambda p: payload)

    process_non_tif_non_2D_images(str(d))

    assert _FlipNdim._flipped is True, "ndim was never consulted"
    # Nothing new on disk: the 2-D guard short-circuited the split.
    assert sorted(p.name for p in d.iterdir()) == [os.path.basename(src)]


def test_rgb_png_is_split_into_channels(tmp_path):
    """An RGB PNG should be split into one grayscale TIFF per channel."""
    from PIL import Image

    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "png_rgb"
    d.mkdir()
    rgb = (np.arange(8 * 8 * 3).reshape(8, 8, 3) % 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(str(d / "rgb.png"))

    process_non_tif_non_2D_images(str(d))

    assert sorted(p.name for p in d.glob("*.tif")) == [
        "rgb_C1.tif", "rgb_C2.tif", "rgb_C3.tif"]


def test_grayscale_png_conversion_preserves_bit_depth(tmp_path):
    """An 8-bit grayscale PNG should convert to an 8-bit TIFF."""
    from PIL import Image

    from spacr.io import process_non_tif_non_2D_images

    d = tmp_path / "png_gray"
    d.mkdir()
    gray = (np.arange(8 * 8).reshape(8, 8) % 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(str(d / "g.png"))

    process_non_tif_non_2D_images(str(d))

    out = tifffile.imread(str(d / "g.tif"))
    np.testing.assert_array_equal(out, gray)
    assert out.dtype == np.uint8, f"bit depth inflated to {out.dtype}"


# ===========================================================================
# _load_images_and_labels
# ===========================================================================

def _patch_imread_none_for(monkeypatch, marker):
    """Make cellpose.io.imread return None for paths containing `marker`."""
    import cellpose.io

    real = cellpose.io.imread

    def fake(path, *a, **kw):
        if marker in str(path):
            return None
        return real(path, *a, **kw)

    monkeypatch.setattr(cellpose.io, "imread", fake)


def test_pairs_skip_unloadable_image(tmp_path, capsys, monkeypatch):
    """A None image drops the whole (image, label) pair."""
    from spacr.io import _load_images_and_labels

    idir, ldir = tmp_path / "i", tmp_path / "l"
    idir.mkdir()
    ldir.mkdir()
    ifiles = [_write_tif(idir / "ok.tif", _ramp((16, 16))),
              _write_tif(idir / "bad_img.tif", _ramp((16, 16)))]
    lfiles = [_write_tif(ldir / "ok.tif", _label_arr()),
              _write_tif(ldir / "bad_img.tif", _label_arr())]
    _patch_imread_none_for(monkeypatch, "bad_img")

    images, labels, image_names, label_names = _load_images_and_labels(
        ifiles, lfiles)

    assert len(images) == 1 and len(labels) == 1
    assert images[0].shape == (16, 16)
    assert labels[0].max() == 3
    assert image_names == ["bad_img.tif", "ok.tif"]
    assert label_names == ["bad_img.tif", "ok.tif"]
    assert "Could not load image" in capsys.readouterr().out


def test_pairs_skip_unloadable_label(tmp_path, capsys, monkeypatch):
    """A None label drops the pair even though the image loaded fine."""
    from spacr.io import _load_images_and_labels

    idir, ldir = tmp_path / "i", tmp_path / "l"
    idir.mkdir()
    ldir.mkdir()
    ifiles = [_write_tif(idir / "a.tif", _ramp((16, 16))),
              _write_tif(idir / "b.tif", _ramp((16, 16)))]
    lfiles = [_write_tif(ldir / "a.tif", _label_arr()),
              _write_tif(ldir / "bad_lbl.tif", _label_arr())]
    _patch_imread_none_for(monkeypatch, "bad_lbl")

    images, labels = _load_images_and_labels(ifiles, lfiles)[:2]

    assert len(images) == 1 and len(labels) == 1
    assert "Could not load label" in capsys.readouterr().out


def test_pairs_invert_and_scale_to_unit_range(tmp_path):
    """With invert=True the pair branch inverts, then divides by the new max."""
    from spacr.io import _load_images_and_labels

    idir, ldir = tmp_path / "i", tmp_path / "l"
    idir.mkdir()
    ldir.mkdir()
    raw = _ramp((16, 16))
    ifiles = [_write_tif(idir / "a.tif", raw)]
    lfiles = [_write_tif(ldir / "a.tif", _label_arr())]

    images, labels = _load_images_and_labels(ifiles, lfiles, invert=True)[:2]

    inverted = np.iinfo(np.uint16).max - raw
    np.testing.assert_allclose(images[0], inverted / inverted.max())
    assert images[0].max() == pytest.approx(1.0)
    np.testing.assert_array_equal(labels[0], _label_arr())


def test_images_only_skip_unloadable_and_invert(tmp_path, capsys, monkeypatch):
    """Image-only mode: None images are skipped, invert flips the intensities."""
    from spacr.io import _load_images_and_labels

    idir = tmp_path / "i"
    idir.mkdir()
    raw = _ramp((16, 16))
    ifiles = [_write_tif(idir / "a.tif", raw),
              _write_tif(idir / "bad_img.tif", _ramp((16, 16)))]
    _patch_imread_none_for(monkeypatch, "bad_img")

    images, labels, image_names, label_names = _load_images_and_labels(
        ifiles, [], invert=True)

    assert labels == [] and label_names == []
    assert len(images) == 1
    inverted = np.iinfo(np.uint16).max - raw
    np.testing.assert_allclose(images[0], inverted / inverted.max())
    assert "Could not load image" in capsys.readouterr().out


def test_labels_only_skip_unloadable(tmp_path, capsys, monkeypatch):
    """Label-only mode returns labels untouched and no images."""
    from spacr.io import _load_images_and_labels

    ldir = tmp_path / "l"
    ldir.mkdir()
    lfiles = [_write_tif(ldir / "z.tif", _label_arr()),
              _write_tif(ldir / "bad_lbl.tif", _label_arr()),
              _write_tif(ldir / "a.tif", _label_arr(n=2))]
    _patch_imread_none_for(monkeypatch, "bad_lbl")

    images, labels, image_names, label_names = _load_images_and_labels(
        [], lfiles)

    assert images == [] and image_names == []
    assert label_names == ["a.tif", "bad_lbl.tif", "z.tif"]
    assert len(labels) == 2
    assert labels[0].max() == 3 and labels[1].max() == 2
    assert labels[0].dtype == np.uint16
    assert "Could not load label" in capsys.readouterr().out


# ===========================================================================
# _load_normalized_images_and_labels
# ===========================================================================

def _img_and_label(tmp_path, n=2, shape=(16, 16)):
    idir, ldir = tmp_path / "i", tmp_path / "l"
    idir.mkdir(exist_ok=True)
    ldir.mkdir(exist_ok=True)
    ifiles, lfiles, raws = [], [], []
    for k in range(n):
        raw = _ramp(shape, start=k * 7)
        raws.append(raw)
        ifiles.append(_write_tif(idir / f"f{k}.tif", raw))
        lfiles.append(_write_tif(ldir / f"f{k}.tif", _label_arr(shape)))
    return ifiles, lfiles, raws


def test_non_numeric_percentiles_fall_back_to_auto(tmp_path, capsys):
    """percentiles=['a','b'] is rejected (ValueError) and auto-percentiles run."""
    from spacr.io import _load_normalized_images_and_labels

    ifiles, lfiles, raws = _img_and_label(tmp_path, n=2)

    norm, labels, image_names, label_names, orig_dims = (
        _load_normalized_images_and_labels(
            ifiles, lfiles, channels=None, percentiles=["a", "b"],
            background=0, Signal_to_noise=10))

    out = capsys.readouterr().out
    assert "Average 1st percentiles" in out, "auto-percentile path not taken"
    assert len(norm) == 2 and len(labels) == 2
    assert orig_dims == [(16, 16), (16, 16)]
    assert image_names == ["f0.tif", "f1.tif"]
    assert label_names == ["f0.tif", "f1.tif"]
    # 2-D input gets a trailing channel axis, and is rescaled into [0, 1].
    assert norm[0].shape == (16, 16, 1)
    assert norm[0].dtype == np.float64
    assert norm[0].min() >= 0.0 and norm[0].max() <= 1.0
    assert norm[0].max() > norm[0].min(), "image collapsed to a constant"
    assert labels[0].shape == (16, 16) and labels[0].dtype == np.uint8


def test_percentiles_none_with_invert(tmp_path):
    """percentiles=None takes the else-branch; invert flips before normalising."""
    from spacr.io import _load_normalized_images_and_labels

    ifiles, _lfiles, raws = _img_and_label(tmp_path, n=1)

    norm_plain = _load_normalized_images_and_labels(
        ifiles, None, percentiles=None, background=0, Signal_to_noise=10)[0]
    norm_inv, labels, _names, label_names, _dims = (
        _load_normalized_images_and_labels(
            ifiles, None, percentiles=None, invert=True,
            background=0, Signal_to_noise=10))

    assert labels == [] and label_names == []
    assert norm_inv[0].shape == (16, 16, 1)
    # Inversion reverses the intensity ordering: the brightest pixel of the
    # plain image is the dimmest of the inverted one.
    plain, inv = norm_plain[0][..., 0], norm_inv[0][..., 0]
    assert np.argmax(plain) == np.argmin(inv)
    assert np.argmin(plain) == np.argmax(inv)


def test_percentiles_tuple_is_not_accepted(tmp_path, capsys):
    """A non-list percentiles argument is discarded (auto-percentiles instead)."""
    from spacr.io import _load_normalized_images_and_labels

    ifiles, _l, _raws = _img_and_label(tmp_path, n=1)
    norm = _load_normalized_images_and_labels(
        ifiles, None, percentiles=(1, 99), background=0, Signal_to_noise=10)[0]
    assert "Average 1st percentiles" in capsys.readouterr().out
    assert norm[0].min() >= 0.0 and norm[0].max() <= 1.0


def test_signal_threshold_above_image_falls_back_to_p1(tmp_path, capsys):
    """When no percentile beats background*S/N, avg_p99 falls back to avg_p1."""
    from spacr.io import _load_normalized_images_and_labels

    ifiles, _l, raws = _img_and_label(tmp_path, n=1)

    norm = _load_normalized_images_and_labels(
        ifiles, None, percentiles=None,
        background=10000, Signal_to_noise=10)[0]

    out = capsys.readouterr().out
    # No percentile cleared background*S/N, so avg_p99 falls back to avg_p1
    # and the two printed vectors are identical.
    assert "Average 1st percentiles" in out
    printed_p1 = out.split("Average 1st percentiles: ")[1].split("], Average")[0]
    printed_p99 = out.split("Average 99th percentiles: ")[1].split("]")[0]
    assert printed_p1 == printed_p99 != ""
    assert str(float(np.percentile(raws[0], 2))) in printed_p1
    # in_range=(p1, p1) collapses the rescale to a constant.
    assert np.allclose(norm[0], 1.0)


def test_channel_selection_and_background_removal(tmp_path):
    """channels=[...] subsets a 3-channel image; remove_background zeroes dim px."""
    from spacr.io import _load_normalized_images_and_labels

    idir = tmp_path / "i"
    idir.mkdir()
    raw = _ramp((16, 16, 3), mod=400)
    src = _write_tif(idir / "rgb.tif", raw)

    norm, _labels, _n, _ln, orig_dims = _load_normalized_images_and_labels(
        [src], None, channels=[0, 2], percentiles=[2, 98],
        remove_background=True, background=100, Signal_to_noise=10)

    assert orig_dims == [(16, 16)]
    assert norm[0].shape == (16, 16, 2), "channel subset not applied"
    # Pixels below `background` were zeroed before normalisation, so every
    # such pixel sits at the bottom of the rescaled range.
    below = raw[..., 0] < 100
    assert below.any()
    assert np.all(norm[0][..., 0][below] == 0.0)


def test_visualize_calls_plot_resize(tmp_path, monkeypatch):
    """visualize=True renders the 2x2 original/resized comparison figure."""
    import matplotlib.pyplot as mpl_plt
    from spacr.io import _load_normalized_images_and_labels

    ifiles, lfiles, _raws = _img_and_label(tmp_path, n=1)

    shown = {}

    def fake_show(*a, **kw):
        fig = mpl_plt.gcf()
        shown["axes"] = len(fig.axes)
        shown["titles"] = [ax.get_title() for ax in fig.axes]

    monkeypatch.setattr(mpl_plt, "show", fake_show)

    norm, labels = _load_normalized_images_and_labels(
        ifiles, lfiles, percentiles=[2, 98], visualize=True,
        background=0, Signal_to_noise=10, target_height=8, target_width=8)[:2]

    assert shown, "plot_resize never reached plt.show()"
    assert shown["axes"] == 4
    assert "Original Image" in shown["titles"]
    assert "Resized Label" in shown["titles"]
    assert norm[0].shape == (8, 8, 1)
    assert labels[0].shape == (8, 8)
