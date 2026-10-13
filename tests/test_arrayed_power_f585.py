"""Independent probability checks for the planner's noncentral-t tails."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi, norm, t

from spacr.sp_stats import _arrayed_power


@pytest.mark.parametrize('ncp,replicates,paired', [
    (12.318921904599211, 2, True), (4., 3, True), (.25, 4, False),
])
def test_power_matches_integrated_normal_chi_representation(ncp, replicates, paired):
    # With cell variance 1 and one cell/field/well, variance of the
    # difference is 2. Integrate Z and sqrt(chi-square) independently;
    # this oracle does not use either noncentral-t tail implementation.
    components = {'cell': 1.}
    effect = ncp * np.sqrt(2 / replicates)
    df = replicates - 1 if paired else 2 * replicates - 2
    critical = t.ppf(.975, df)
    expected, error = quad(
        lambda scale: (norm.sf(critical * scale / np.sqrt(df) - ncp)
                       + norm.cdf(-critical * scale / np.sqrt(df) - ncp))
        * chi.pdf(scale, df), 0, np.inf, epsabs=1e-11)
    assert error < 1e-8
    actual = _arrayed_power(components, effect, replicates=replicates,
                           wells=1, fields=1, cells=1, paired=paired)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_null_keeps_both_tiny_tails():
    actual = _arrayed_power({'cell': 1.}, 0., replicates=5, wells=1,
                           fields=1, cells=1, alpha=1e-10)
    assert actual == pytest.approx(1e-10, rel=2e-6, abs=0)


def test_failed_distribution_never_becomes_certain_power(monkeypatch):
    from scipy.stats import nct

    monkeypatch.setattr(nct, 'sf', lambda *_a, **_k: float('nan'))
    with pytest.raises(ValueError, match='could not be computed reliably'):
        _arrayed_power({'cell': 1.}, 1., replicates=2, wells=1,
                       fields=1, cells=1, paired=True)
