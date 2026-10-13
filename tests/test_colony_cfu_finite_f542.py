"""Unrepresentable colony titres must remain missing rather than report infinity."""
import math

import pytest

from spacr.plaque import _cfu_per_ml, _dilution_factor


@pytest.mark.parametrize("dilution", [1e-320, float("inf"), float("nan"), 10**400])
def test_nonfinite_or_unrepresentable_dilution_is_missing(dilution):
    assert _dilution_factor(dilution) is None
    assert _cfu_per_ml(12, dilution, 100) is None


@pytest.mark.parametrize("count", [float("inf"), -float("inf"), float("nan"), 10**400])
def test_nonfinite_or_unrepresentable_count_is_missing(count):
    assert _cfu_per_ml(count, 100, 100) is None


@pytest.mark.parametrize("volume", [float("inf"), float("nan"), 10**400, 5e-324])
def test_invalid_or_underflowed_volume_is_missing(volume):
    assert _cfu_per_ml(12, 100, volume) is None


@pytest.mark.parametrize("count,dilution,volume", [(1e308, 100, 1000), (10, 1, 1e-305)])
def test_overflowed_titre_is_missing(count, dilution, volume):
    assert _cfu_per_ml(count, dilution, volume) is None


def test_zero_and_large_finite_titres_remain_valid():
    assert _cfu_per_ml(0, 10000, 100) == 0.0
    assert _cfu_per_ml(150, 1e-4, 100) == pytest.approx(1.5e7)
    titre = _cfu_per_ml(1e200, 1, 1000)
    assert math.isfinite(titre)
    assert titre == 1e200
