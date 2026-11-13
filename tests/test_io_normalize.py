"""Table-driven tests of the percentile normalisation and merge helpers in ``spacr.io``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture.  ``_normalize_img_batch`` is checked plane for plane
against an independent scalar re-implementation of its percentile search;
``concatenate_and_normalize``, ``_normalize_stack``,
``_get_lists_for_normalization`` and ``_load_and_concatenate_arrays`` are
driven over tiny synthetic stacks under ``tmp_path``.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

import spacr.io as IO


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _stack(n=3, h=8, w=8, c=2, seed=0):
    """(N, H, W, C) float32 stack with strictly positive intensities."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, h, w, c)) * 2950.0 + 50.0).astype(np.float32)


def _plane(h=8, w=8, c=2, seed=0):
    """A single (H, W, C) field as written by the stack step."""
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, c)) * 2950.0 + 50.0).astype(np.float32)


def _settings(**over):
    """Settings with every key ``concatenate_and_normalize`` touches."""
    s = {"timelapse": False, "randomize": False, "batch_size": 10,
         "lower_percentile": 2, "nucleus_channel": 0, "cell_channel": 1,
         "pathogen_channel": None, "organelle_channel": None,
         "nucleus_background": 100, "nucleus_signal_to_noise": 5,
         "remove_background_nucleus": False, "cell_background": 100,
         "cell_signal_to_noise": 5, "remove_background_cell": False,
         "background": 100, "Signal_to_noise": 5, "remove_background": False,
         "plot": False, "figuresize": 4, "cmap": "inferno", "nr": 1}
    s.update(over)
    return s


def _reference(stack, channels, profiles, lower, save_dtype):
    """The original scalar percentile search, plane by plane."""
    from skimage import exposure

    result = np.zeros_like(stack, dtype=np.float32)
    for channel in channels:
        background, threshold, remove = profiles[channel]
        source = stack[..., channel]
        if remove:
            source[source < background] = 0
        nonzero = source[source != 0]
        if not nonzero.size:
            continue
        low = np.percentile(nonzero, lower)
        high = None
        for percentile in np.linspace(98, 99.5, num=16):
            value = np.percentile(nonzero, percentile)
            if value >= threshold:
                high = value
                break
        if high is None:
            high = np.percentile(nonzero, 99.5)
        for index in range(source.shape[0]):
            result[index, ..., channel] = exposure.rescale_intensity(
                source[index], in_range=(low, high), out_range=(0, 1))
    return result.astype(save_dtype, copy=False)


def _write_fields(folder, names, shape=(8, 8, 2), seed=0):
    os.makedirs(folder, exist_ok=True)
    for i, name in enumerate(names):
        np.save(os.path.join(folder, name), _plane(*shape, seed=seed + i))


def _npz(path):
    with np.load(path) as z:
        return z["data"], [str(f) for f in z["filenames"]]


# ---------------------------------------------------------------------------
# _normalize_img_batch
# ---------------------------------------------------------------------------

def _none_roles(**over):
    s = {"nucleus_channel": None, "cell_channel": None,
         "pathogen_channel": None, "organelle_channel": None,
         "lower_percentile": 2}
    s.update(over)
    return s


def test_normalize_img_batch_matches_the_scalar_percentile_search(
        subtests, capsys):
    """Every role's background / threshold / removal reaches the exact planes."""
    rng = np.random.default_rng(548)
    mixed = rng.integers(0, 1001, size=(2, 3, 17, 5))
    mixed[0, 0, :4, :] = 0
    mixed[..., 4] = 0
    mixed_settings = dict(
        lower_percentile=None, pathogen_channel=2, organelle_channel=3,
        nucleus_background=1, nucleus_signal_to_noise=1,
        cell_background=1, cell_signal_to_noise=992,
        pathogen_background=1, pathogen_signal_to_noise=1000000,
        remove_background_pathogen=False,
        organelle_background=400, organelle_signal_to_noise=2.4725,
        remove_background_organelle=True)
    mixed_profiles = ((1, 1, False), (1, 992, False), (1, 1000000, False),
                      (400, 989, True), (100, 500, False))
    small = _stack(n=2, h=10, w=10, c=1, seed=8)

    def nonfinite(special):
        s = np.arange(1, 49, dtype=np.float32).reshape(2, 3, 8, 1)
        s[0, 0, 0, 0] = special
        s[1, 2, 7, 0] = 0
        return s

    cases = [
        # id, stack, channels, save_dtype, settings, per-channel profiles,
        # lower percentile, printed "Processing channel" line
        ("pathogen settings", _stack(n=2, c=2, seed=3), [1], np.float32,
         _none_roles(pathogen_channel=1, pathogen_background=250,
                     pathogen_signal_to_noise=2,
                     remove_background_pathogen=True),
         {1: (250, 500, True)}, 2,
         "Processing channel 1: background=250, signal_threshold=500, "
         "remove_background=True"),
        ("organelle settings override the generic ones",
         _stack(n=2, c=3, seed=4), [2], np.float32,
         dict(_none_roles(nucleus_channel=0, cell_channel=1,
                          organelle_channel=2),
              background=100, Signal_to_noise=5, remove_background=False,
              organelle_background=200, organelle_signal_to_noise=3,
              remove_background_organelle=True),
         {2: (200, 600, True)}, 2,
         "Processing channel 2: background=200, signal_threshold=600, "
         "remove_background=True"),
        ("organelle falls back to the generic values",
         _stack(n=2, c=2, seed=5), [0], np.float32,
         _none_roles(organelle_channel=0, background=300, Signal_to_noise=4,
                     remove_background=False),
         {0: (300, 1200, False)}, 2,
         "Processing channel 0: background=300, signal_threshold=1200, "
         "remove_background=False"),
        ("an unassigned channel uses the generic values",
         _stack(n=2, c=3, seed=1), [2], np.float32,
         dict(_settings(), background=100, Signal_to_noise=5),
         {2: (100, 500, False)}, 2, None),
        ("string channels and float16", _stack(n=2, c=2, seed=9), ["0", "1"],
         np.float16, _settings(), {0: (100, 500, False), 1: (100, 500, False)},
         2, None),
        ("nucleus removes background, cell keeps it",
         _stack(n=3, h=16, w=16, c=2, seed=0), [0, 1], np.float32,
         _settings(remove_background_nucleus=True),
         {0: (100, 500, True), 1: (100, 500, False)}, 2, None),
        ("the first percentile over the threshold wins", small, [0],
         np.float32, _none_roles(background=1, Signal_to_noise=1,
                                 remove_background=False),
         {0: (1, 1, False)}, 2, None),
        ("no percentile reaches the threshold: 99.5", small, [0], np.float32,
         _none_roles(background=100000, Signal_to_noise=10,
                     remove_background=False),
         {0: (100000, 1000000, False)}, 2, None),
        ("background removal", _stack(n=2, c=1, seed=6), [0], np.float32,
         _none_roles(background=1500, Signal_to_noise=1,
                     remove_background=True),
         {0: (1500, 1500, True)}, 2, None),
    ]
    for special in (np.nan, np.inf, -np.inf):
        cases.append((f"nonfinite pixel {special}", nonfinite(special), [0],
                      np.float32,
                      _settings(nucleus_channel=None, cell_channel=None,
                                background=1, Signal_to_noise=1),
                      {0: (1, 1, False)}, 2, None))
    for src_dtype in (np.uint16, np.float32):
        for save_dtype in (np.float16, np.float32):
            for lower in (0, 2, 17, 63.5):
                cases.append((
                    f"all roles {np.dtype(src_dtype).name}->"
                    f"{np.dtype(save_dtype).name} lower={lower}",
                    mixed.astype(src_dtype), [3, 0, 4, 2, 1], save_dtype,
                    _settings(**dict(mixed_settings, lower_percentile=lower)),
                    dict(enumerate(mixed_profiles)), lower, None))
    for case_id, stack, channels, save_dtype, settings, profiles, lower, \
            said in cases:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            idx = [int(c) for c in channels]
            full = {c: profiles.get(c, (0, 0, False))
                    for c in range(stack.shape[-1])}
            expected = _reference(stack.copy(), idx, full, lower, save_dtype)
            actual = IO._normalize_img_batch(stack.copy(), channels,
                                             save_dtype, settings)
            assert (actual.shape, actual.dtype) == (expected.shape,
                                                    expected.dtype)
            np.testing.assert_array_equal(actual, expected)
            untouched = [c for c in range(stack.shape[-1]) if c not in idx]
            assert np.all(actual[..., untouched] == 0)
            if said:
                assert said in capsys.readouterr().out

    with subtests.test(msg="the two upper cut-offs really differ here"):
        nz = small[..., 0][small[..., 0] != 0]
        assert not np.isclose(np.percentile(nz, 98), np.percentile(nz, 99.5))


# ---------------------------------------------------------------------------
# _get_lists_for_normalization
# ---------------------------------------------------------------------------

def test_get_lists_for_normalization_follows_nucleus_cell_pathogen_order(
        subtests):
    """Lists hold one entry per configured role, in role order."""
    def roles(n, c, p, **values):
        s = {"nucleus_channel": n, "cell_channel": c, "pathogen_channel": p}
        s.update(values)
        return s

    full = dict(nucleus_background=111, nucleus_signal_to_noise=3,
                remove_background_nucleus=False, cell_background=222,
                cell_signal_to_noise=4, remove_background_cell=True,
                pathogen_background=333, pathogen_signal_to_noise=5,
                remove_background_pathogen=False)
    same = {f"{r}_{k}": v for r in ("nucleus", "cell", "pathogen")
            for k, v in (("background", 100), ("signal_to_noise", 5))}
    same.update({f"remove_background_{r}": True
                 for r in ("nucleus", "cell", "pathogen")})
    nan = float("nan")
    cases = [
        ("role order, not channel order", roles(5, 1, 3, **full),
         ([111, 222, 333], [3, 4, 5], [333, 888, 1665], [False, True, False])),
        ("channel order 0, 1, 2", roles(0, 1, 2, **dict(
            full, nucleus_background=100, nucleus_signal_to_noise=5,
            remove_background_nucleus=True, cell_background=200,
            remove_background_cell=False, pathogen_background=150,
            pathogen_signal_to_noise=6, remove_background_pathogen=True)),
         ([100, 200, 150], [5, 4, 6], [500, 800, 900], [True, False, True])),
        ("pathogen only", roles(None, None, 3, pathogen_background=130,
                                pathogen_signal_to_noise=2,
                                remove_background_pathogen=True),
         ([130], [2], [260], [True])),
        ("cell only", roles(None, 1, None, cell_background=200,
                            cell_signal_to_noise=4,
                            remove_background_cell=False),
         ([200], [4], [800], [False])),
        ("same values everywhere", roles(0, 1, 2, **same),
         ([100] * 3, [5] * 3, [500] * 3, [True] * 3)),
        ("no roles", roles(None, None, None), ([], [], [], [])),
        ("blank CSV cells (NaN) match nothing", roles(nan, nan, nan, **same),
         ([], [], [], [])),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert IO._get_lists_for_normalization(settings) == expected


# ---------------------------------------------------------------------------
# concatenate_and_normalize
# ---------------------------------------------------------------------------

def test_concatenate_and_normalize_writes_normalized_batches(
        subtests, tmp_path, monkeypatch, capsys):
    """Readable stacks become ``masks/stack_<i>_norm.npz`` (or per-field timelapse)."""
    import spacr.plot

    real_listdir = os.listdir

    def broken(name):
        return lambda d: (d / name).write_bytes(b"definitely not a numpy file")

    def odd_shapes(d):
        np.save(d / "a_small.npy", _plane(8, 8, 2, seed=0))
        np.save(d / "b_wide.npy", _plane(10, 6, 2, seed=1))

    def text(name):
        return lambda d: (d / name).write_text("x", encoding="utf-8")

    def tl_mismatch(d):
        np.save(d / "plate1_A01_1_0001.npy", _plane(8, 8, 2, seed=0))
        np.save(d / "plate1_A01_1_0002.npy", _plane(12, 12, 2, seed=1))

    four = ["plate1_A01_1.npy", "plate1_A01_2.npy", "plate1_A02_1.npy",
            "plate1_A02_2.npy"]
    tl = ["plate1_A01_1_0002.npy", "plate1_A01_1_0001.npy",
          "plate1_A01_2_0001.npy", "plate1_A01_2_0002.npy"]
    cases = [
        # id, (names, shape), extra builders, channels, settings overrides,
        # {npz: (shape, sorted filenames)}, plot calls, printed
        ("one batch", (four, (8, 8, 2)), [], [0, 1], {},
         {"stack_0_norm.npz": ((4, 8, 8, 2), sorted(four))}, 0,
         ["All files concatenated and normalized"]),
        ("channel subset", (four[:2], (8, 8, 3)), [], [0, 2], {},
         {"stack_0_norm.npz": ((2, 8, 8, 2), four[:2])}, 0, []),
        ("string channels", (four[:2], (8, 8, 3)), [], ["0", "2"], {},
         {"stack_0_norm.npz": ((2, 8, 8, 2), four[:2])}, 0, []),
        ("None channels are dropped", (four[:2], (8, 8, 2)), [],
         [0, None, 1], {},
         {"stack_0_norm.npz": ((2, 8, 8, 2), four[:2])}, 0, []),
        ("non-npy and broken npy are skipped",
         (["g1_A01_1.npy", "g2_A01_1.npy", "g3_A01_1.npy"], (8, 8, 2)),
         [text("readme.txt"), broken("b_broken.npy")], [0, 1], {},
         {"stack_0_norm.npz": ((3, 8, 8, 2), ["g1_A01_1.npy", "g2_A01_1.npy",
                                              "g3_A01_1.npy"])}, 0,
         ["Error loading file", "b_broken.npy"]),
        ("a broken last file keeps the trailing batch",
         (["a_A01_1.npy", "b_A01_1.npy"], (8, 8, 2)),
         [broken("z_broken.npy")], [0, 1], {},
         {"stack_0_norm.npz": ((2, 8, 8, 2), ["a_A01_1.npy",
                                              "b_A01_1.npy"])}, 0, []),
        ("mismatched shapes are padded", ([], None), [odd_shapes], [0, 1], {},
         {"stack_0_norm.npz": ((2, 10, 8, 2), ["a_small.npy",
                                               "b_wide.npy"])}, 0,
         ["arrays with multiple shapes found in batch"]),
        ("plots only the first batch", (four[:3], (8, 8, 2)), [], [0, 1],
         {"batch_size": 1, "plot": True, "figuresize": 6, "cmap": "viridis",
          "nr": 2},
         {f"stack_{i}_norm.npz": ((1, 8, 8, 2), None) for i in range(3)}, 1,
         ["plotting:"]),
        ("timelapse groups by field, ordered by time",
         (tl, (8, 8, 2)),
         [text("junk.txt"), lambda d: np.save(d / "plate1_A01.npy", _plane())],
         [0, 1], {"timelapse": True, "plot": True},
         {"plate1_A01_1_norm_timelapse.npz": (
             (2, 8, 8, 2), ["plate1_A01_1_0001.npy", "plate1_A01_1_0002.npy"]),
          "plate1_A01_2_norm_timelapse.npz": (
             (2, 8, 8, 2), ["plate1_A01_2_0001.npy", "plate1_A01_2_0002.npy"])},
         1, []),
        ("timelapse frames that disagree are reported", ([], None),
         [tl_mismatch], [0, 1], {"timelapse": True}, {}, 0,
         ["Error processing files", "plate_well_field_time.npy", "Error:"]),
        ("timelapse without parsable names", ([], None), [text("nope.txt")],
         [0, 1], {"timelapse": True}, {}, 0, []),
    ]
    for i, (case_id, (names, shape), extras, channels, over, expected,
            n_plots, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"c{i}" / "stack"
            src.mkdir(parents=True)
            if names:
                _write_fields(str(src), names, shape=shape)
            for build in extras:
                build(src)
            calls = []
            with monkeypatch.context() as m:
                m.setattr(spacr.plot, "plot_arrays",
                          lambda *a, **kw: calls.append((a, kw)))
                m.setattr(os, "listdir",
                          lambda *a, **k: sorted(real_listdir(*a, **k)))
                capsys.readouterr()
                out = IO.concatenate_and_normalize(
                    str(src), channels, save_dtype=np.float32,
                    settings=_settings(**over))
            printed = capsys.readouterr().out
            assert out == os.path.join(str(tmp_path / f"c{i}"), "masks")
            assert sorted(real_listdir(out)) == sorted(expected)
            for npz, (npz_shape, npz_names) in expected.items():
                data, got = _npz(os.path.join(out, npz))
                assert (data.shape, data.dtype) == (npz_shape, np.float32)
                if npz_names is not None:
                    assert sorted(got) == sorted(npz_names)
                    if over.get("timelapse"):
                        assert got == npz_names
                assert 0.0 <= float(data.min())
                assert float(data.max()) <= 1.0 + 1e-6
            assert len(calls) == n_plots
            for fragment in said:
                assert fragment in printed
            if case_id.startswith("plots only"):
                args, kwargs = calls[0]
                assert args == (os.path.join(out, "stack_0_norm.npz"), 6,
                                "viridis")
                assert kwargs == {"nr": 2, "normalize": False}
            if case_id.startswith("mismatched"):
                data, got = _npz(os.path.join(out, "stack_0_norm.npz"))
                s, w = data[got.index("a_small.npy")], \
                    data[got.index("b_wide.npy")]
                assert np.all(s[8:] == 0) and np.all(w[:, 6:] == 0)
                assert s[:8].max() > 0 and w[:, :6].max() > 0
            if case_id == "one batch":
                data, _ = _npz(os.path.join(out, "stack_0_norm.npz"))
                assert float(data.max()) == pytest.approx(1.0, abs=1e-5)

    with subtests.test(msg="randomized batches cover every field once"):
        src = tmp_path / "rand" / "stack"
        names = [f"plate1_A0{k}_1.npy" for k in range(1, 6)]
        _write_fields(str(src), names)
        random.seed(1234)
        out = IO.concatenate_and_normalize(
            str(src), [0, 1], settings=_settings(batch_size=2, randomize=True))
        saved = sorted(os.listdir(out))
        assert saved == ["stack_0_norm.npz", "stack_1_norm.npz",
                         "stack_2_norm.npz"]
        batches = [_npz(os.path.join(out, f)) for f in saved]
        assert [d.shape[0] for d, _ in batches] == [2, 2, 1]
        assert sorted(n for _, ns in batches for n in ns) == names

    with subtests.test(msg="settings are required, before any side effect"):
        src = tmp_path / "nosettings" / "stack"
        _write_fields(str(src), ["p_A01_1.npy"])
        with pytest.raises(ValueError, match="requires a settings dict"):
            IO.concatenate_and_normalize(str(src), [0, 1], settings=None)
        assert not (tmp_path / "nosettings" / "masks").exists()


# ---------------------------------------------------------------------------
# _normalize_stack
# ---------------------------------------------------------------------------

def _ramp(n=64, lo=1.0, hi=1000.0):
    side = int(round(n ** 0.5))
    return np.linspace(lo, hi, n, dtype=np.float32).reshape(side, side)


def test_normalize_stack_rescales_frames_that_clear_the_snr(
        subtests, tmp_path):
    """Frames above the SNR are rescaled to [0, 1]; empty/flat ones copied raw."""
    ramp = _ramp()
    one = dict(backgrounds=[0], remove_backgrounds=[False],
               signal_to_noise=[5], signal_thresholds=[1.0])

    def frames(*planes, c=1):
        arr = np.zeros((len(planes), 8, 8, c), np.float32)
        for k, plane in enumerate(planes):
            arr[k] = plane[..., None] if plane.ndim == 2 else plane
        return arr

    two_chan = np.zeros((2, 8, 8, 2), np.float32)
    two_chan[..., 0] = ramp
    two_chan[..., 1] = 300.0
    wide = np.zeros((2, 8, 8, 3), np.float32)
    wide[...] = _ramp(lo=50.0, hi=5000.0)[None, :, :, None]
    rng = np.random.default_rng(0)
    noisy = (rng.random((3, 16, 16, 2)) * 3000).astype(np.uint16)
    noisy3 = (rng.random((3, 16, 16, 3)) * 3000).astype(np.uint16)
    linear = np.linspace(1, 100, 32, dtype=np.float32).reshape(2, 4, 4, 1)
    zero = np.zeros((8, 8), np.float32)
    flat = np.full((8, 8), 500.0, np.float32)
    cases = [
        # id, {npz stem: data}, kwargs, per-frame expectation over channel 0:
        # "rescaled" (min 0, max 1), "zero", or a raw constant; dtype
        ("rescaled, empty and flat frames", {"plate1": frames(ramp, zero, flat)},
         one, ["rescaled", "zero", 500.0], np.float32),
        ("a blank first frame", {"blankfirst": frames(zero, ramp)}, one,
         ["zero", "rescaled"], np.float32),
        ("an all-zero channel", {"allzero": frames(zero, zero)}, one,
         ["zero", "zero"], np.float32),
        ("save_dtype float64", {"dt": frames(ramp, ramp)},
         dict(one, save_dtype=np.float64), ["rescaled", "rescaled"],
         np.float64),
        ("two channels, two files", {"p1": two_chan, "p2": two_chan},
         dict(backgrounds=[0, 0], remove_backgrounds=[False, False],
              signal_to_noise=[5, 5], signal_thresholds=[1.0, 1.0]),
         ["rescaled", "rescaled"], np.float32),
        ("None arguments use the defaults", {"defaults": wide}, {},
         ["rescaled", "rescaled"], np.float32),
        ("an unreachable threshold stops at the top",
         {"plate1": linear}, dict(one, signal_thresholds=[10 ** 9]),
         [None, None], np.float32),
        ("uint16 batch, two channels", {"batch1": noisy},
         dict(backgrounds=[100, 100], remove_backgrounds=[True, False],
              signal_to_noise=[2, 2], signal_thresholds=[500, 500]),
         [None, None, None], np.float32),
        ("uint16 batch, defaults", {"b": noisy3}, {}, [None, None, None],
         np.float32),
    ]
    for i, (case_id, files, kwargs, per_frame, dtype) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"c{i}" / "stack"
            src.mkdir(parents=True)
            for stem, data in files.items():
                np.savez(str(src / f"{stem}.npz"), data=data, filenames=np.array(
                    [f"img_{k}.npy" for k in range(data.shape[0])]))
            IO._normalize_stack(str(src), **kwargs)
            masks = tmp_path / f"c{i}" / "masks"
            assert sorted(p.name for p in masks.iterdir()) == sorted(
                f"{s}_norm_stack.npz" for s in files)
            for stem, data in files.items():
                res, names = _npz(masks / f"{stem}_norm_stack.npz")
                assert (res.shape, res.dtype) == (data.shape, dtype)
                assert names == [f"img_{k}.npy" for k in range(data.shape[0])]
                assert np.isfinite(res).all()
                for k, want in enumerate(per_frame):
                    plane = res[k, ..., 0]
                    if want == "rescaled":
                        assert (float(plane.min()), float(plane.max())) == \
                            pytest.approx((0.0, 1.0))
                    elif want == "zero":
                        assert np.count_nonzero(res[k]) == 0
                    elif want is not None:
                        assert np.allclose(plane, want)
                if case_id.startswith("two channels"):
                    assert np.allclose(res[..., 1], 300.0)

    with subtests.test(msg="removed background clamps dim pixels to zero"):
        src = tmp_path / "bg" / "stack"
        src.mkdir(parents=True)
        np.savez(str(src / "bg.npz"),
                 data=np.stack([ramp, ramp])[..., None],
                 filenames=np.array(["a.npy", "b.npy"]))
        IO._normalize_stack(str(src), backgrounds=[200],
                            remove_backgrounds=[True], signal_to_noise=[2],
                            signal_thresholds=[1.0])
        res, _ = _npz(tmp_path / "bg" / "masks" / "bg_norm_stack.npz")
        dim = ramp < 200
        assert int(dim.sum()) == 13
        assert np.all(res[0, ..., 0][dim] == 0)
        assert float(res[0, ..., 0][~dim].max()) == pytest.approx(1.0)

    with subtests.test(msg="per-channel settings must cover every channel"):
        src = tmp_path / "short" / "stack"
        src.mkdir(parents=True)
        np.savez(str(src / "two.npz"), data=np.zeros((1, 8, 8, 2), np.float32),
                 filenames=np.array(["a.npy"]))
        with pytest.raises(ValueError, match="2 channels.*incomplete"):
            IO._normalize_stack(str(src), **one)


# ---------------------------------------------------------------------------
# _load_and_concatenate_arrays
# ---------------------------------------------------------------------------

_DIMS = ("cell_chann_dim", "nucleus_chann_dim", "pathogen_chann_dim",
         "organelle_chann_dim")


def _merge_root(root, masks, n_channels=2, shape=(10, 10)):
    """stack/fov.npy with plane c == c + 1, plus one mask folder per entry."""
    h, w = shape
    os.makedirs(os.path.join(root, "stack"), exist_ok=True)
    img = np.stack([np.full((h, w), float(c + 1), np.float32)
                    for c in range(n_channels)], axis=-1)
    np.save(os.path.join(root, "stack", "fov.npy"), img)
    for name, arr in masks.items():
        d = os.path.join(root, "masks", f"{name}_mask_stack")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "fov.npy"), arr)
    return img


def _mask(i, shape=(10, 10), extra_axis=False):
    m = np.zeros(shape, np.uint16)
    m[1 + i:4 + i, 1:4] = 10 * (i + 1)
    return m[..., None] if extra_axis else m


def test_load_and_concatenate_arrays_appends_mask_planes_to_each_field(
        subtests, tmp_path, monkeypatch):
    """``merged/<field>.npy`` = image channels + one plane per mask folder."""
    roles = ("cell", "nucleus", "pathogen", "organelle")

    def lonely(root):
        np.save(os.path.join(root, "stack", "lonely.npy"),
                np.zeros((10, 10, 2), np.float32))

    def notes(root):
        with open(os.path.join(root, "stack", "notes.txt"), "w") as fh:
            fh.write("not an array")

    cases = [
        # id, mask roles, n image channels, channels arg, dims, setup,
        # expected merged shape (None = nothing written), padded shape
        ("all four roles in folder order", roles, 2, [0, 1], (0, 1, 2, 3),
         None, (10, 10, 6)),
        ("pathogen/organelle found with every dim None",
         ("pathogen", "organelle"), 1, None, (None,) * 4, None, (10, 10, 3)),
        ("cell + nucleus over three channels", ("cell", "nucleus"), 3,
         [0, 1, 2], (0, 1, None, None), None, (10, 10, 5)),
        ("a field missing from a mask folder is skipped", ("cell",), 2,
         [0, 1], (0, None, None, None), lonely, (10, 10, 3)),
        ("non-npy entries are ignored", ("cell",), 2, [0, 1],
         (0, None, None, None), notes, (10, 10, 3)),
        ("no mask folders writes no merged array", (), 2, [0, 1],
         (None,) * 4, None, None),
    ]
    for i, (case_id, mask_roles, n_chan, channels, dims, setup,
            shape) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = str(tmp_path / f"c{i}")
            masks = {r: _mask(k) for k, r in enumerate(mask_roles)}
            img = _merge_root(root, masks, n_channels=n_chan)
            if setup:
                setup(root)
            IO._load_and_concatenate_arrays(root, channels=channels,
                                            **dict(zip(_DIMS, dims)))
            merged = os.path.join(root, "merged")
            written = sorted(f for f in os.listdir(merged) if f.endswith(".npy"))
            if shape is None:
                assert written == []
                continue
            assert written == ["fov.npy"]
            arr = np.load(os.path.join(merged, "fov.npy"))
            assert arr.shape == shape
            np.testing.assert_array_equal(arr[..., :n_chan], img)
            for k, role in enumerate(mask_roles):
                np.testing.assert_array_equal(arr[..., n_chan + k], masks[role])

    with subtests.test(msg="a (Y, X, 1) mask joins as one plane"):
        root = str(tmp_path / "axis")
        mask = _mask(0, (6, 6), extra_axis=True)
        _merge_root(root, {"cell": mask}, shape=(6, 6))
        IO._load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=0,
                                        nucleus_chann_dim=None,
                                        pathogen_chann_dim=None,
                                        organelle_chann_dim=None)
        arr = np.load(os.path.join(root, "merged", "fov.npy"))
        assert arr.shape == (6, 6, 3)
        np.testing.assert_array_equal(arr[..., 2], mask[..., 0])

    with subtests.test(msg="smaller images are padded to the largest mask"):
        root = str(tmp_path / "pad")
        _merge_root(root, {"cell": _mask(0, (20, 18)),
                           "nucleus": _mask(1, (20, 18))}, n_channels=3,
                    shape=(16, 16))
        IO._load_and_concatenate_arrays(root, channels=None, cell_chann_dim=0,
                                        nucleus_chann_dim=1,
                                        pathogen_chann_dim=None,
                                        organelle_chann_dim=None)
        assert np.load(os.path.join(root, "merged", "fov.npy")).shape == \
            (20, 18, 5)

    with subtests.test(msg="an unreadable layout sidecar refuses to resume"):
        from spacr.crops import MERGED_LAYOUT_SIDECAR

        root = str(tmp_path / "resume")
        _merge_root(root, {})
        merged = os.path.join(root, "merged")
        os.makedirs(merged)
        sidecar = os.path.join(merged, MERGED_LAYOUT_SIDECAR)
        with open(sidecar, "w", encoding="utf-8") as fh:
            fh.write("{not json at all")
        with pytest.raises(ValueError) as caught:
            IO._load_and_concatenate_arrays(
                root, channels=[0, 1], resume=True,
                **dict(zip(_DIMS, (None,) * 4)))
        assert "Cannot resume" in str(caught.value)
        assert MERGED_LAYOUT_SIDECAR in str(caught.value)
        assert os.listdir(merged) == [MERGED_LAYOUT_SIDECAR]
        with open(sidecar, encoding="utf-8") as fh:
            assert fh.read() == "{not json at all"

    with subtests.test(msg="a failed sidecar write removes its temporary"):
        root = str(tmp_path / "sidecar")
        _merge_root(root, {})
        removed = []

        def bad_dump(*_a, **_k):
            raise RuntimeError("disk full while writing layout")

        def bad_remove(path, *a, **k):
            removed.append(path)
            raise OSError("cannot unlink the temporary either")

        with monkeypatch.context() as m:
            m.setattr(IO.json, "dump", bad_dump)
            m.setattr(IO.os, "remove", bad_remove)
            with pytest.raises(RuntimeError,
                               match="disk full while writing layout"):
                IO._load_and_concatenate_arrays(
                    root, channels=[0, 1], **dict(zip(_DIMS, (None,) * 4)))
        assert len(removed) == 1
        assert os.path.basename(removed[0]).startswith(".spacr_plane_layout_")
        assert os.path.dirname(removed[0]) == os.path.join(root, "merged")
