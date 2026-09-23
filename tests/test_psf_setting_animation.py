"""The Gaussian width illustration preserves light and its fixed input."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from generate_setting_animations import _psf_profiles
from spacr.setting_animations import animation_for_setting


def test_gaussian_width_spreads_signal_without_changing_total_intensity():
    first_input, narrow = _psf_profiles(0)
    last_input, broad = _psf_profiles(1)
    assert first_input == last_input
    for values in (first_input, narrow, broad):
        assert min(values) >= 0
        assert sum(values) == pytest.approx(1, abs=1e-12)
        assert sum(x * value for x, value in enumerate(values)) == pytest.approx(60)
    variance = lambda values: sum((x - 60) ** 2 * value for x, value in enumerate(values))
    assert variance(first_input) < variance(narrow) < variance(broad)
    assert max(first_input) > max(narrow) > max(broad)
    assert broad[60] > narrow[60]


def test_gaussian_width_setting_resolves_to_its_packaged_animation():
    animation = animation_for_setting('psf_fwhm_um')
    assert animation.slug == 'psf_fwhm_um'
    assert animation.path.is_file()
