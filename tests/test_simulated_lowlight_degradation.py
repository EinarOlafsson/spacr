"""The simulated low-light model of tools/validate_n2v_lowlight.py (item 557).

The benchmark degrades clean fields with known masks to a photon budget, so
the numbers it reports are only as good as the noise model: shot noise plus
read noise, the clean image's bit depth, and a seed that repeats exactly.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def V():
    """The validator module, loaded from tools/ without running it."""
    patch = pytest.MonkeyPatch()
    patch.setenv("BENCH_LOG", "")
    patch.setenv("OUT", str(ROOT / ".pytest-557-unused"))
    patch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("validate_n2v_lowlight",
                                                  ROOT / "tools" / "validate_n2v_lowlight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    patch.undo()
    sys.modules.pop("validate_n2v_lowlight", None)


def _field(dtype=np.uint16, signal=1000, background=50, size=256):
    """A clean field: a flat background and one bright square object."""
    clean = np.full((size, size), background, dtype=dtype)
    foreground = np.zeros((size, size), dtype=bool)
    foreground[64:192, 64:192] = True
    clean[foreground] = signal
    return clean, foreground


def test_mean_and_variance_follow_poisson_plus_read_noise(V):
    clean, foreground = _field()
    photons, read, gain, offset = 20.0, 2.0, 1.5, 100.0
    noisy, record = V.simulate_lowlight(clean, foreground, photons, np.random.default_rng(1),
                                        read_noise=read, gain=gain, offset=offset,
                                        background_fraction=0.1)
    adu = noisy.astype(np.float64)
    for mask, lam in ((foreground, photons + 2.0), (~foreground, 2.0)):
        assert adu[mask].mean() == pytest.approx(gain * lam + offset, rel=0.01)
        expected = gain ** 2 * (lam + read ** 2) + 1 / 12
        assert adu[mask].var() == pytest.approx(expected, rel=0.05)
    assert record["snr_expected"] == pytest.approx(photons / np.sqrt(photons + 2.0 + read ** 2))
    assert record["clipped_fraction"] == 0.0


def test_photon_budget_is_the_mean_signal_over_objects(V):
    clean, foreground = _field()
    lam, scaling = V.expected_photons(clean, foreground, 5.0, background_fraction=0.0)
    assert lam[foreground].mean() == pytest.approx(5.0)
    assert lam[~foreground].max() == 0.0
    assert scaling["background_level"] == 50


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_bit_depth_is_kept_and_values_clip_to_it(V, dtype):
    clean, foreground = _field(dtype=dtype, signal=200, background=20)
    noisy, record = V.simulate_lowlight(clean, foreground, 50.0, np.random.default_rng(0))
    assert noisy.dtype == np.dtype(dtype)
    assert record["dtype"] == np.dtype(dtype).name
    bright, _ = V.simulate_lowlight(clean, foreground, 5000.0, np.random.default_rng(0))
    assert bright.max() <= np.iinfo(dtype).max
    if dtype is np.uint8:
        assert V.simulate_lowlight(clean, foreground, 5000.0,
                                   np.random.default_rng(0))[1]["clipped_fraction"] > 0


def test_float_images_are_refused(V):
    clean, foreground = _field()
    with pytest.raises(ValueError):
        V.simulate_lowlight(clean.astype(np.float32), foreground, 10.0, np.random.default_rng(0))


def test_a_seed_repeats_exactly_and_another_seed_differs(V):
    clean, foreground = _field()
    first = V.simulate_lowlight(clean, foreground, 2.0, np.random.default_rng([0, 2000, 3, 0]))[0]
    again = V.simulate_lowlight(clean, foreground, 2.0, np.random.default_rng([0, 2000, 3, 0]))[0]
    other = V.simulate_lowlight(clean, foreground, 2.0, np.random.default_rng([0, 2000, 3, 1]))[0]
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_scaling_inverts_back_to_the_clean_image(V):
    clean, foreground = _field()
    noisy, record = V.simulate_lowlight(clean, foreground, 10.0, np.random.default_rng(0))
    lam, _ = V.expected_photons(clean, foreground, 10.0)
    expected_adu = lam * record["gain"] + record["offset"]
    assert np.allclose(V.photons_to_clean(expected_adu, record), clean)
    back = V.photons_to_clean(noisy, record)
    assert back[foreground].mean() == pytest.approx(clean[foreground].mean(), rel=0.01)
