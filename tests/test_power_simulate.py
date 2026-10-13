"""Property tests for :mod:`spacr.power_simulate`.

Statistical code is the easiest place in this codebase to be confidently wrong:
a simulator with a transposed parameterisation, a flipped sign or a silently
clamped moment still produces numbers of the right magnitude and the right
shape, and every summary statistic downstream still looks plausible. So nothing
here asserts "the function ran". Each test states a property that would be
*violated* if the distribution underneath were the wrong one, and several are
written specifically against the failure mode of the R original they were ported
from:

* the gamma helper is checked against the ``rate`` vs ``scale`` inversion that
  is the classic R-to-numpy port bug;
* the beta helper is checked against clamping an unattainable variance;
* the Dirichlet helper is checked against numpy's own underflow, which it exists
  to avoid — that test fails if the stable implementation is swapped back for
  ``Generator.dirichlet``;
* the whole screen is checked for *orientation*: a strong classifier signal must
  make hits score **above** non-hits, because a simulator whose signal points the
  wrong way hands the model half an AUROC of ``1 - AUROC`` that looks entirely
  believable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from spacr.power_simulate import (
    AbundanceClippedWarning,
    ImpossibleMomentsError,
    MalformedPlateError,
    PowerSimulationError,
    ScreenDesignError,
    SequencingScaleError,
    rbeta_mean_variance,
    rdirichlet_stable,
    resolve_rng,
    rgamma_mean_variance,
    rnbinom_mean_variance,
    sample_count_mean_variance,
    simulate_imaging_plate,
    simulate_library,
    simulate_screen,
    simulate_sequencing_plate,
    simulate_spot_plate,
)
from spacr.errors import SpacrError


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

#: Parameters near the values the R vignettes fitted to the real T. gondii
#: screen, shrunk so a test finishes in well under a second.
SCREEN_KWARGS = dict(
    n_genes_in_library=60,
    gene_abundance_alpha=20.0,
    gene_hit_rate=0.15,
    n_wells_per_screen=48,
    well_abundance_factor_mu=4.0,
    well_abundance_factor_var=0.5,
    imaging_n_cells_per_well_mu=123.0,
    imaging_n_cells_per_well_var=8000.0,
    class_pos_mu=0.80,
    class_pos_var=0.10,
    class_neg_mu=0.12,
    class_neg_var=0.01,
    sequencing_n_cells_per_well_lambda=1000.0,
    pcr_factor_mu=2.0,
    pcr_factor_var=1.0,
    n_reads_per_well=30000,
)


def _small_spot_plate(seed=101, n_genes=25, n_wells=40, alpha=20.0, hit_rate=0.2,
                      mu=None, var=0.3):
    """Return a spot plate small enough to test on but large enough to average over.

    ``mu`` defaults to ``n_genes / 5``, which keeps the
    ``gene_abundance * well_abundance`` product comfortably below 1 for any
    library size the tests use. A fixed default would clip on the four-gene
    plates (abundance ~ 0.25 each) and bury every test's output under
    :class:`AbundanceClippedWarning` noise that has nothing to do with what is
    being tested — clipping has its own dedicated test.
    """
    if mu is None:
        mu = n_genes / 5.0
    library = simulate_library(n_genes, alpha, hit_rate, seed=seed)
    return simulate_spot_plate(library, n_wells, mu, var, seed=seed + 1)


# ---------------------------------------------------------------------------
# Random-number discipline
# ---------------------------------------------------------------------------

#: Every public sampler, with a set of otherwise-valid arguments. Parameterised
#: so a newly added sampler that forgets the seed contract shows up here.
SAMPLERS = [
    (rgamma_mean_variance, (10, 2.0, 1.0)),
    (rnbinom_mean_variance, (10, 5.0, 20.0)),
    (rbeta_mean_variance, (10, 0.5, 0.01)),
    (rdirichlet_stable, (2.0, 5)),
    (sample_count_mean_variance, (10, 5.0, 20.0)),
    (simulate_library, (5, 2.0, 0.5)),
]


@pytest.mark.parametrize('func,args', SAMPLERS, ids=lambda v: getattr(v, '__name__', ''))
def test_sampler_refuses_to_draw_from_entropy(func, args):
    """No seed and no rng must be an error, not a silent unreproducible draw."""
    with pytest.raises(ScreenDesignError, match='seed'):
        func(*args)


@pytest.mark.parametrize('func,args', SAMPLERS, ids=lambda v: getattr(v, '__name__', ''))
def test_sampler_refuses_both_seed_and_rng(func, args):
    """Supplying both would make the caller believe the seed pinned the run."""
    with pytest.raises(ScreenDesignError, match='exactly one'):
        func(*args, seed=0, rng=np.random.default_rng(0))


def test_resolve_rng_refuses_the_legacy_randomstate():
    """RandomState cannot be spawned into independent per-stage streams."""
    with pytest.raises(ScreenDesignError, match='numpy.random.Generator'):
        resolve_rng(rng=np.random.RandomState(0))


def test_errors_are_catchable_as_spacr_errors():
    """Callers that only care that spaCR refused should not need our class names."""
    assert issubclass(PowerSimulationError, SpacrError)
    for subclass in (ImpossibleMomentsError, ScreenDesignError,
                     MalformedPlateError, SequencingScaleError):
        assert issubclass(subclass, PowerSimulationError)


def test_same_seed_reproduces_the_screen_bit_for_bit():
    first = simulate_screen(**SCREEN_KWARGS, seed=4242)
    second = simulate_screen(**SCREEN_KWARGS, seed=4242)
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_gives_a_different_screen():
    """A seed that does not change the answer is a seed that is being ignored."""
    first = simulate_screen(**SCREEN_KWARGS, seed=1)
    second = simulate_screen(**SCREEN_KWARGS, seed=2)
    assert not first['positive'].equals(second['positive'])
    assert not first['n_reads_per_gene_per_well'].equals(
        second['n_reads_per_gene_per_well']
    )


def test_passing_a_generator_is_equivalent_to_passing_its_seed():
    by_seed = simulate_screen(**SCREEN_KWARGS, seed=77)
    by_rng = simulate_screen(**SCREEN_KWARGS, rng=np.random.default_rng(77))
    pd.testing.assert_frame_equal(by_seed, by_rng)


def test_simulating_never_touches_the_global_numpy_random_state():
    """The global state is shared with every other library in the process.

    Drawing from it would make this simulator's output depend on whatever else
    ran first in the same interpreter, which is the failure mode that makes a
    seeded run irreproducible *between* machines while looking reproducible on
    the one it was developed on.
    """
    np.random.seed(12345)
    before = np.random.get_state()
    simulate_screen(**SCREEN_KWARGS, seed=9)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# ---------------------------------------------------------------------------
# Distribution helpers: mean/variance reparameterisation
# ---------------------------------------------------------------------------

def test_rgamma_mean_variance_delivers_the_requested_moments():
    """Also the rate-vs-scale test.

    R's ``rgamma`` takes a rate and numpy's takes a scale ``= 1/rate``. Passing
    the rate straight through turns ``mean=4, var=2`` into ``mean=16, var=32``,
    so these two assertions are exactly what catches the classic port bug.
    """
    draws = rgamma_mean_variance(400000, mean=4.0, var=2.0, seed=0)
    assert draws.min() > 0.0
    assert abs(draws.mean() - 4.0) < 0.02
    assert abs(draws.var() - 2.0) < 0.03


@pytest.mark.parametrize('mean,var', [(0.0, 1.0), (-1.0, 1.0), (2.0, 0.0), (2.0, -1.0)])
def test_rgamma_rejects_moments_it_cannot_represent(mean, var):
    with pytest.raises(ScreenDesignError):
        rgamma_mean_variance(5, mean=mean, var=var, seed=0)


def test_rnbinom_mean_variance_delivers_the_requested_moments():
    draws = rnbinom_mean_variance(400000, mean=10.0, var=40.0, seed=1)
    assert draws.dtype.kind in 'iu'
    assert draws.min() >= 0
    assert abs(draws.mean() - 10.0) < 0.1
    assert abs(draws.var() - 40.0) < 1.0


@pytest.mark.parametrize('var', [10.0, 9.9, 0.0])
def test_rnbinom_refuses_variance_at_or_below_the_mean(var):
    """At ``var == mean`` the size parameter is ``+inf``.

    R's ``assertthat`` allows the equality and hands ``Inf`` to ``rnbinom``,
    which returns a column of ``NA``. Refusing is the only way the caller finds
    out; :func:`sample_count_mean_variance` is where equi-dispersion is handled.
    """
    with pytest.raises(ImpossibleMomentsError, match='over-dispersed'):
        rnbinom_mean_variance(5, mean=10.0, var=var, seed=1)


def test_rbeta_mean_variance_delivers_the_requested_moments():
    draws = rbeta_mean_variance(400000, mean=0.8, var=0.01, seed=2)
    assert draws.min() >= 0.0 and draws.max() <= 1.0
    assert abs(draws.mean() - 0.8) < 0.002
    assert abs(draws.var() - 0.01) < 0.0005


@pytest.mark.parametrize('mean,var', [
    (0.5, 0.25),      # exactly the Bernoulli bound: no beta attains it
    (0.5, 0.30),      # above the bound
    (0.9, 0.09),      # above 0.9 * 0.1 = 0.09
    (0.0, 0.01),      # a mean on the boundary admits no positive variance
    (1.0, 0.01),
])
def test_rbeta_refuses_an_unattainable_variance_instead_of_clamping(mean, var):
    """``mean * (1 - mean)`` is the supremum over all distributions on [0, 1].

    Clamping to the nearest feasible variance would hand back a classifier whose
    spread is not the one requested, which silently changes the effect size the
    entire power analysis is measuring.
    """
    with pytest.raises(ImpossibleMomentsError):
        rbeta_mean_variance(5, mean=mean, var=var, seed=2)


@pytest.mark.parametrize('mean', [-0.01, 1.01, np.nan])
def test_rbeta_refuses_a_mean_outside_the_unit_interval(mean):
    with pytest.raises(ScreenDesignError, match=r'\[0, 1\]'):
        rbeta_mean_variance(5, mean=mean, var=0.001, seed=2)


def test_rbeta_zero_variance_is_the_exact_point_mass():
    """A perfect classifier (mu=1, var=0) has to be expressible."""
    assert rbeta_mean_variance(4, mean=1.0, var=0.0, seed=2).tolist() == [1.0] * 4
    assert rbeta_mean_variance(4, mean=0.0, var=0.0, seed=2).tolist() == [0.0] * 4
    assert rbeta_mean_variance(4, mean=0.3, var=0.0, seed=2).tolist() == [0.3] * 4


def test_rbeta_refuses_a_negative_variance():
    with pytest.raises(ScreenDesignError, match='var'):
        rbeta_mean_variance(5, mean=0.5, var=-0.01, seed=2)


def test_rdirichlet_stable_sums_to_one_and_matches_its_moments():
    """Dirichlet(alpha 1_n) has mean 1/n and var (1/n)(1-1/n)/(n*alpha+1)."""
    n, alpha = 20, 2.0
    rng = np.random.default_rng(3)
    draws = np.array([rdirichlet_stable(alpha, n, rng=rng) for _ in range(30000)])
    assert np.allclose(draws.sum(axis=1), 1.0, atol=1e-12)
    expected_mean = 1.0 / n
    expected_var = expected_mean * (1 - expected_mean) / (n * alpha + 1)
    assert np.allclose(draws.mean(axis=0), expected_mean, atol=0.002)
    assert np.allclose(draws.var(axis=0), expected_var, rtol=0.08)


def test_rdirichlet_stable_avoids_the_underflow_numpy_dirichlet_hits():
    """The reason this helper exists rather than calling ``Generator.dirichlet``.

    At the concentrations a real skewed library sits at, numpy's gamma-ratio
    Dirichlet underflows individual components to exactly ``0.0``. A gene with
    abundance exactly zero is in no well at all, and its read fraction is a
    ``0 / 0`` downstream. This test fails if the stable implementation is ever
    replaced by the plain numpy call.
    """
    n_categories, alpha, n_draws = 452, 0.05, 200
    numpy_zeros = 0
    stable_zeros = 0
    rng = np.random.default_rng(4)
    for _ in range(n_draws):
        numpy_zeros += int((rng.dirichlet(np.full(n_categories, alpha)) == 0.0).sum())
        stable_zeros += int((rdirichlet_stable(alpha, n_categories, rng=rng) == 0.0).sum())
    assert numpy_zeros > 0, 'numpy no longer underflows here; retune the test'
    assert stable_zeros == 0


@pytest.mark.parametrize('alpha,n_categories', [
    (0.0, 5), (-1.0, 5), (np.inf, 5), (2.0, 0),
])
def test_rdirichlet_stable_rejects_degenerate_concentrations(alpha, n_categories):
    with pytest.raises(ScreenDesignError):
        rdirichlet_stable(alpha, n_categories, seed=4)


def test_rdirichlet_stable_requires_a_category_count_for_a_scalar_alpha():
    with pytest.raises(ScreenDesignError, match='n_categories'):
        rdirichlet_stable(2.0, seed=4)


def test_rdirichlet_stable_accepts_a_per_category_alpha_vector():
    weights = rdirichlet_stable([10.0, 10.0, 0.01], seed=4)
    assert weights.shape == (3,)
    assert abs(weights.sum() - 1.0) < 1e-12


def test_rdirichlet_stable_rejects_a_contradictory_category_count():
    with pytest.raises(ScreenDesignError, match='contradicts'):
        rdirichlet_stable([1.0, 1.0], 3, seed=4)


@pytest.mark.parametrize('mean,var,expect_relation', [
    (20.0, 60.0, 'over'),
    (20.0, 20.0, 'equi'),
    (20.0, None, 'equi'),
    (100.0, 25.0, 'under'),
])
def test_sample_count_covers_over_equi_and_under_dispersion(mean, var, expect_relation):
    """The three-way dispatch that replaces COM-Poisson.

    The under-dispersed row is the one that matters: a lazy implementation that
    always drew from a Poisson would pass the mean assertion and fail the
    variance one by a factor of four.
    """
    draws = sample_count_mean_variance(300000, mean=mean, var=var, seed=5)
    assert draws.dtype.kind in 'iu'
    assert draws.min() >= 0
    target_var = mean if var is None else var
    # The mean is the moment this function preserves exactly, so it is held to
    # sampling error (about 4 standard errors) rather than to a loose tolerance.
    standard_error = np.sqrt(target_var / draws.size)
    assert abs(draws.mean() - mean) < 4 * standard_error
    # The variance is the moment that absorbs the binomial's integer-n rounding,
    # so it gets a relative tolerance instead.
    assert abs(draws.var() - target_var) < 0.05 * target_var
    if expect_relation == 'over':
        assert draws.var() > mean
    elif expect_relation == 'under':
        assert draws.var() < mean


def test_sample_count_refuses_a_negative_variance():
    with pytest.raises(ImpossibleMomentsError, match='variance'):
        sample_count_mean_variance(5, mean=10.0, var=-1.0, seed=5)


def test_sample_count_treats_a_one_ulp_variance_difference_as_poisson():
    """``var`` arriving as ``mean * 1.0`` must not flip the family.

    Without the equidispersion tolerance this lands one ULP either side of the
    mean and dispatches to a negative binomial with ``size = 1e16`` or a
    binomial with ``n = 1e16``, both of which are the same Poisson but only one
    of which numpy will sample in finite time.
    """
    mean = 37.0
    draws = sample_count_mean_variance(50000, mean=mean, var=np.nextafter(mean, 0.0),
                                       seed=5)
    assert abs(draws.mean() - mean) < 0.3
    assert abs(draws.var() - mean) < 1.5


# ---------------------------------------------------------------------------
# Stage 1 -- the library
# ---------------------------------------------------------------------------

def test_library_abundances_sum_to_one_and_are_strictly_positive():
    library = simulate_library(500, 5.0, 0.05, seed=6)
    assert len(library) == 500
    assert abs(library['gene_abundance'].sum() - 1.0) < 1e-12
    assert (library['gene_abundance'] > 0).all()


def test_library_gene_ids_are_one_based_and_contiguous():
    """The R package indexes genes from 1; the model half joins on this column."""
    library = simulate_library(7, 5.0, 0.5, seed=6)
    assert library['gene'].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_library_hit_fraction_matches_the_requested_rate():
    library = simulate_library(40000, 5.0, 0.025, seed=7)
    assert set(library['hit'].unique()) <= {0, 1}
    assert abs(library['hit'].mean() - 0.025) < 0.003


@pytest.mark.parametrize('rate', [0.0, 1.0])
def test_library_honours_the_degenerate_hit_rates_exactly(rate):
    library = simulate_library(200, 5.0, rate, seed=7)
    assert library['hit'].mean() == rate


def test_library_concentration_controls_how_even_the_library_is():
    """Large alpha -> every gene near 1/n; small alpha -> a few genes dominate.

    Stated as a comparison rather than an absolute so it tests the *direction*
    of the parameter, which is the thing a transposed reparameterisation would
    invert.
    """
    even = simulate_library(300, 500.0, 0.0, seed=8)['gene_abundance']
    skewed = simulate_library(300, 0.2, 0.0, seed=8)['gene_abundance']
    assert even.std() < skewed.std() / 10
    assert skewed.max() > 10 * even.max()


@pytest.mark.parametrize('kwargs,match', [
    (dict(n_genes_in_library=0, gene_abundance_alpha=1.0, gene_hit_rate=0.1),
     'n_genes_in_library'),
    (dict(n_genes_in_library=-3, gene_abundance_alpha=1.0, gene_hit_rate=0.1),
     'n_genes_in_library'),
    (dict(n_genes_in_library=10, gene_abundance_alpha=0.0, gene_hit_rate=0.1),
     'gene_abundance_alpha'),
    (dict(n_genes_in_library=10, gene_abundance_alpha=1.0, gene_hit_rate=1.5),
     'gene_hit_rate'),
    (dict(n_genes_in_library=10, gene_abundance_alpha=1.0, gene_hit_rate=-0.1),
     'gene_hit_rate'),
    (dict(n_genes_in_library=2.5, gene_abundance_alpha=1.0, gene_hit_rate=0.1),
     'whole number'),
])
def test_library_refuses_a_degenerate_design_by_name(kwargs, match):
    """An empty library must never come back as an empty frame.

    Downstream, an empty frame reads as "nothing was a hit" — a believable
    scientific conclusion produced by a broken configuration.
    """
    with pytest.raises(ScreenDesignError, match=match):
        simulate_library(**kwargs, seed=8)


def test_library_of_one_gene_works():
    library = simulate_library(1, 3.0, 1.0, seed=8)
    assert len(library) == 1
    assert library['gene_abundance'].iloc[0] == 1.0
    assert library['hit'].iloc[0] == 1


# ---------------------------------------------------------------------------
# Stage 2 -- the spot plate
# ---------------------------------------------------------------------------

def test_spot_plate_is_rectangular_and_in_expand_grid_order():
    """``tidyr::expand_grid`` varies the LAST variable fastest, unlike base R."""
    library = simulate_library(5, 10.0, 0.5, seed=9)
    spot = simulate_spot_plate(library, 3, 2.0, 0.1, seed=10)
    assert len(spot) == 15
    assert spot['gene'].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    assert spot['well'].tolist() == [1, 2, 3] * 5
    assert set(spot['gene_in_well'].unique()) <= {0, 1}


def test_spot_plate_carries_every_library_column_through():
    """The imaging stage reads ``hit`` and ``gene_abundance`` off this frame."""
    library = simulate_library(6, 10.0, 0.5, seed=9)
    spot = simulate_spot_plate(library, 4, 2.0, 0.1, seed=10)
    for column in library.columns:
        assert column in spot.columns
    per_gene = spot.groupby('gene')['hit'].nunique()
    assert (per_gene == 1).all()


def test_spot_plate_occupancy_matches_abundance_times_well_factor():
    """``P(gene in well) = gene_abundance * well_abundance``, and nothing else.

    Averaged over the whole plate the realised occupancy has to equal the mean
    of that product. A version that used only the abundance, only the well
    factor, or their sum would still produce a sensible-looking 0/1 matrix.
    """
    library = simulate_library(60, 30.0, 0.1, seed=11)
    spot = simulate_spot_plate(library, 600, 5.0, 0.3, seed=12)
    expected = (spot['gene_abundance'] * spot['well_abundance']).mean()
    assert abs(spot['gene_in_well'].mean() - expected) < 0.004
    # ... and it is a per-well count of about mu genes, which is the knob the
    # whole R sweep turned.
    assert abs(spot.groupby('well')['gene_in_well'].sum().mean() - 5.0) < 0.4


def test_spot_plate_warns_loudly_and_clips_when_the_probability_exceeds_one():
    """R's ``rbinom(prob > 1)`` returns NA and poisons every downstream count.

    Clipping keeps the run alive, but the realised genes-per-well is then below
    the requested ``well_abundance_factor_mu``, so the warning has to carry the
    count — a silent clip would mislabel the whole power curve.
    """
    library = simulate_library(6, 0.3, 0.0, seed=13)
    with pytest.warns(AbundanceClippedWarning, match=r'\d+ of \d+ gene-in-well'):
        spot = simulate_spot_plate(library, 400, 3.0, 0.05, seed=14)
    assert set(spot['gene_in_well'].unique()) <= {0, 1}
    assert spot['gene_in_well'].notna().all()
    assert spot.attrs['n_prob_clipped'] > 0
    # The gene whose abundance saturates every well is in every well.
    dominant = library['gene_abundance'].idxmax() + 1
    assert spot.loc[spot['gene'] == dominant, 'gene_in_well'].mean() == 1.0


class _no_clip_warning:
    """Context manager asserting no :class:`AbundanceClippedWarning` is raised.

    ``pytest.warns(None)`` was removed in pytest 8, and ``filterwarnings`` as a
    marker cannot assert the *absence* of one warning class while allowing
    others, so the check is spelled out.
    """

    def __enter__(self):
        self._catcher = warnings.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        warnings.simplefilter('always')
        return self

    def __exit__(self, *exc_info):
        clipped = [r for r in self._records
                   if issubclass(r.category, AbundanceClippedWarning)]
        self._catcher.__exit__(*exc_info)
        assert not clipped, (
            f'unexpected clip warnings: {[str(r.message) for r in clipped]}'
        )
        return False


def test_spot_plate_does_not_warn_when_no_probability_is_clipped():
    """The warning has to be specific: a simulator that cries wolf gets muted."""
    library = simulate_library(200, 50.0, 0.1, seed=13)
    with _no_clip_warning():
        spot = simulate_spot_plate(library, 40, 2.0, 0.05, seed=14)
    assert spot.attrs['n_prob_clipped'] == 0


def test_spot_plate_refuses_zero_wells():
    """R's ``1:0`` is ``c(1, 0)``: asking for no wells silently gives two."""
    library = simulate_library(5, 10.0, 0.5, seed=15)
    with pytest.raises(ScreenDesignError, match='n_wells_per_screen'):
        simulate_spot_plate(library, 0, 2.0, 0.1, seed=16)


def test_spot_plate_of_one_well_works():
    library = simulate_library(5, 10.0, 0.5, seed=15)
    spot = simulate_spot_plate(library, 1, 2.0, 0.1, seed=16)
    assert len(spot) == 5
    assert spot['well'].unique().tolist() == [1]


@pytest.mark.parametrize('mu,var', [(0.0, 0.1), (-1.0, 0.1), (2.0, 0.0), (2.0, -0.1)])
def test_spot_plate_refuses_non_positive_abundance_moments(mu, var):
    library = simulate_library(5, 10.0, 0.5, seed=15)
    with pytest.raises(ScreenDesignError, match='well_abundance_factor'):
        simulate_spot_plate(library, 4, mu, var, seed=16)


def test_spot_plate_refuses_a_library_with_duplicate_genes():
    library = simulate_library(5, 10.0, 0.5, seed=15)
    doubled = pd.concat([library, library], ignore_index=True)
    with pytest.raises(MalformedPlateError, match='duplicate gene'):
        simulate_spot_plate(doubled, 4, 2.0, 0.1, seed=16)


def test_spot_plate_refuses_a_library_missing_its_abundance_column():
    library = simulate_library(5, 10.0, 0.5, seed=15).drop(columns=['gene_abundance'])
    with pytest.raises(MalformedPlateError, match='gene_abundance'):
        simulate_spot_plate(library, 4, 2.0, 0.1, seed=16)


# ---------------------------------------------------------------------------
# Stage 3 -- the imaging plate
# ---------------------------------------------------------------------------

def test_imaging_recovers_the_requested_classifier_operating_point():
    """The gap between these two numbers IS the signal the power analysis measures.

    Pooling cells by ground-truth hit status, the positive-call rate must come
    back at ``class_pos_mu`` for hits and ``class_neg_mu`` for non-hits. A port
    that swapped the two, or applied one to everything, would still produce a
    frame full of plausible positive counts.
    """
    spot = _small_spot_plate(seed=201, n_genes=200, n_wells=1200, alpha=30.0,
                             hit_rate=0.3, mu=8.0, var=0.5)
    imaging = simulate_imaging_plate(
        spot, 400.0, None,
        class_pos_mu=0.80, class_pos_var=0.02,
        class_neg_mu=0.12, class_neg_var=0.005,
        seed=202,
    )
    merged = imaging.merge(spot[['gene', 'well', 'hit']], on=['gene', 'well'])
    rates = {}
    for hit_status in (0, 1):
        rows = merged[merged['hit'] == hit_status]
        cells = rows['imaging_n_cells_per_gene_per_well'].sum()
        assert cells > 100000, 'not enough cells to estimate the rate'
        rates[hit_status] = rows['positive'].sum() / cells
    # Tolerances are ~5 standard errors of the *beta* spread across
    # (gene, well) rows, which dominates the binomial noise within a row: the
    # probability is drawn once per row, so ~2900 hit rows at var=0.02 gives
    # sqrt(0.02/2900) ~ 0.0026 per rate. Both are an order of magnitude tighter
    # than the 0.68 gap between the two rates, so swapping them, or applying one
    # rate to every cell, fails loudly.
    assert abs(rates[1] - 0.80) < 0.015
    assert abs(rates[0] - 0.12) < 0.008
    assert rates[1] - rates[0] > 0.6


def test_imaging_positive_calls_never_exceed_the_cells_that_were_imaged():
    spot = _small_spot_plate(seed=203)
    imaging = simulate_imaging_plate(spot, 200.0, 4000.0, 0.8, 0.01, 0.2, 0.01,
                                     seed=204)
    assert (imaging['positive'] <= imaging['imaging_n_cells_per_gene_per_well']).all()
    assert (imaging['positive'] >= 0).all()


def test_imaging_puts_no_cells_in_a_gene_that_was_never_spotted():
    """A genotype absent from a well cannot contribute cells to that well."""
    spot = _small_spot_plate(seed=205)
    imaging = simulate_imaging_plate(spot, 200.0, None, 0.8, 0.01, 0.2, 0.01,
                                     seed=206)
    merged = imaging.merge(spot[['gene', 'well', 'gene_in_well']], on=['gene', 'well'])
    absent = merged[merged['gene_in_well'] == 0]
    assert len(absent) > 0
    assert (absent['imaging_n_cells_per_gene_per_well'] == 0).all()
    assert (absent['positive'] == 0).all()


def test_imaging_emits_the_per_well_offset_column_the_model_half_needs():
    """``imaging_n_cells_per_well`` is absent from the R output — see module docs.

    ``R/fit_model.R`` reads it as the Poisson offset ``log(Ntotal)``; R's ``$``
    partial matching finds three columns starting with that string, returns
    ``NULL``, and the offset silently disappears. Here it exists and equals the
    realised per-well cell total.
    """
    spot = _small_spot_plate(seed=207)
    imaging = simulate_imaging_plate(spot, 200.0, 4000.0, 0.8, 0.01, 0.2, 0.01,
                                     seed=208)
    assert 'imaging_n_cells_per_well' in imaging.columns
    totals = imaging.groupby('well')['imaging_n_cells_per_gene_per_well'].sum()
    stated = imaging.groupby('well')['imaging_n_cells_per_well'].first()
    pd.testing.assert_series_equal(totals, stated, check_names=False)
    # and it is repeated identically on every row of the well
    assert (imaging.groupby('well')['imaging_n_cells_per_well'].nunique() == 1).all()


def test_imaging_well_cell_counts_match_the_requested_moments():
    """Only wells with at least one gene get cells; those wells match mu/var."""
    library = simulate_library(40, 30.0, 0.2, seed=209)
    spot = simulate_spot_plate(library, 4000, 6.0, 0.4, seed=210)
    imaging = simulate_imaging_plate(spot, 150.0, 6000.0, 0.5, 0.01, 0.5, 0.01,
                                     seed=211)
    occupied = spot.groupby('well')['gene_in_well'].sum()
    totals = imaging.groupby('well')['imaging_n_cells_per_well'].first()
    totals = totals[occupied > 0]
    assert abs(totals.mean() - 150.0) < 5.0
    assert abs(totals.var() - 6000.0) < 700.0


def test_imaging_split_uniform_matches_r_and_abundance_does_not():
    """Both modes exist because upstream's choice is a modelling decision, not a bug.

    R passes the 0/1 ``gene_in_well`` vector as the multinomial probability, so
    a gene that is 60% of the library gets exactly as many cells as one that is
    1%. That understates the imbalance between genotypes and therefore
    overstates power, so ``'abundance'`` is the default here.
    """
    # A hand-built library rather than a sampled one: the property under test is
    # about the *split*, so the abundances are pinned to a 20:1 harmonic spread
    # and the well factor is made effectively deterministic (variance 1e-6). That
    # removes both the "did this seed happen to make a gene vanish" lottery and
    # any probability clipping, leaving the split as the only moving part.
    n_genes = 20
    raw = 1.0 / np.arange(1, n_genes + 1)
    library = pd.DataFrame({
        'gene': np.arange(1, n_genes + 1, dtype=np.int64),
        'gene_abundance': raw / raw.sum(),
        'hit': np.zeros(n_genes, dtype=np.int64),
    })
    with _no_clip_warning():
        spot = simulate_spot_plate(library, 3000, 3.0, 1e-6, seed=213)

    def _cells_over_even_share(split):
        """Return each gene's cells divided by what an even split would give it.

        For every well the gene is in, an even split would hand it
        ``well_total / n_genes_present``. Pooling observed over expected across
        wells normalises away both the well-to-well cell count and how many
        genes happened to land in each well, so an even split scores exactly 1.0
        for *every* gene regardless of abundance -- and an abundance-weighted
        split cannot.
        """
        imaging = simulate_imaging_plate(spot, 500.0, None, 0.5, 0.01, 0.5, 0.01,
                                         imaging_split=split, seed=214)
        merged = imaging.merge(spot[['gene', 'well', 'gene_in_well']],
                               on=['gene', 'well'])
        merged['n_present'] = merged.groupby('well')['gene_in_well'].transform('sum')
        present = merged[merged['gene_in_well'] == 1].copy()
        present['even'] = present['imaging_n_cells_per_well'] / present['n_present']
        totals = present.groupby('gene')[
            ['imaging_n_cells_per_gene_per_well', 'even']].sum()
        assert (totals['even'] > 2000).all(), 'a gene is too rare to score'
        return totals['imaging_n_cells_per_gene_per_well'] / totals['even']

    dominant, faintest = 1, n_genes

    uniform = _cells_over_even_share('uniform')
    assert np.allclose(uniform.to_numpy(), 1.0, atol=0.05), uniform.to_dict()

    abundance = _cells_over_even_share('abundance')
    assert abundance[dominant] > 1.5, abundance.to_dict()
    assert abundance[faintest] < 0.4, abundance.to_dict()
    assert abundance[dominant] > 5 * abundance[faintest]


def test_imaging_handles_a_well_with_no_genes_spotted_into_it():
    """An empty well still exists; it just has nothing to attribute cells to."""
    library = simulate_library(3, 100.0, 0.5, seed=215)
    # abundance ~ 1/3 each, well factor mean 0.01 -> essentially every well empty
    spot = simulate_spot_plate(library, 60, 0.01, 1e-6, seed=216)
    assert (spot.groupby('well')['gene_in_well'].sum() == 0).any()
    imaging = simulate_imaging_plate(spot, 200.0, None, 0.8, 0.01, 0.2, 0.01,
                                     seed=217)
    empty_wells = spot.groupby('well')['gene_in_well'].sum()
    empty_wells = set(empty_wells[empty_wells == 0].index)
    rows = imaging[imaging['well'].isin(empty_wells)]
    assert (rows['imaging_n_cells_per_gene_per_well'] == 0).all()
    assert (rows['imaging_n_cells_per_well'] == 0).all()
    assert (rows['positive'] == 0).all()


def test_imaging_with_a_perfect_classifier_calls_exactly_the_hit_cells():
    spot = _small_spot_plate(seed=218, n_genes=20, n_wells=60, hit_rate=0.5)
    imaging = simulate_imaging_plate(
        spot, 100.0, None,
        class_pos_mu=1.0, class_pos_var=0.0,
        class_neg_mu=0.0, class_neg_var=0.0,
        seed=219,
    )
    merged = imaging.merge(spot[['gene', 'well', 'hit']], on=['gene', 'well'])
    hits = merged[merged['hit'] == 1]
    others = merged[merged['hit'] == 0]
    assert hits['imaging_n_cells_per_gene_per_well'].sum() > 0
    assert (hits['positive'] == hits['imaging_n_cells_per_gene_per_well']).all()
    assert (others['positive'] == 0).all()


def test_imaging_refuses_an_unknown_split_mode():
    spot = _small_spot_plate(seed=220)
    with pytest.raises(ScreenDesignError, match='imaging_split'):
        simulate_imaging_plate(spot, 100.0, None, 0.8, 0.01, 0.2, 0.01,
                               imaging_split='proportional', seed=221)


def test_imaging_refuses_a_non_positive_cell_count_mean():
    spot = _small_spot_plate(seed=222)
    with pytest.raises(ScreenDesignError, match='imaging_n_cells_per_well_mu'):
        simulate_imaging_plate(spot, 0.0, None, 0.8, 0.01, 0.2, 0.01, seed=223)


def test_imaging_refuses_a_ragged_plate():
    """A missing (gene, well) pair would leave a NaN hole in the pivot."""
    spot = _small_spot_plate(seed=224)
    ragged = spot.iloc[1:].reset_index(drop=True)
    with pytest.raises(MalformedPlateError, match='one row per'):
        simulate_imaging_plate(ragged, 100.0, None, 0.8, 0.01, 0.2, 0.01, seed=225)


def test_imaging_refuses_a_plate_with_duplicated_pairs():
    spot = _small_spot_plate(seed=226, n_genes=4, n_wells=4)
    duplicated = pd.concat([spot.iloc[:8], spot.iloc[:8]], ignore_index=True)
    with pytest.raises(MalformedPlateError):
        simulate_imaging_plate(duplicated, 100.0, None, 0.8, 0.01, 0.2, 0.01, seed=227)


def test_imaging_result_is_independent_of_the_row_order_it_was_handed():
    """The pivot is by id, not by position — a shuffled plate must not reattribute.

    This is the one that catches a positional reshape: shuffling the rows leaves
    every id intact, so any implementation keyed on ids gives the identical
    answer, and any implementation keyed on position silently assigns each
    gene's cells to a different gene.
    """
    spot = _small_spot_plate(seed=228)
    shuffled = spot.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ordered = simulate_imaging_plate(spot, 150.0, None, 0.9, 0.01, 0.1, 0.01, seed=229)
    scrambled = simulate_imaging_plate(shuffled, 150.0, None, 0.9, 0.01, 0.1, 0.01,
                                       seed=229)
    pd.testing.assert_frame_equal(ordered, scrambled)


# ---------------------------------------------------------------------------
# Stage 4 -- the sequencing plate
# ---------------------------------------------------------------------------

def test_sequencing_reads_never_exceed_the_depth_or_the_barcode_pool():
    """Reads are drawn without replacement, so both ceilings are hard."""
    spot = _small_spot_plate(seed=301)
    seq = simulate_sequencing_plate(spot, 500.0, 1.0, 0.2, 5000, seed=302)
    per_well = seq.groupby('well')['n_reads_per_gene_per_well'].sum()
    assert (per_well <= 5000).all()
    assert (seq['n_reads_per_gene_per_well']
            <= seq['n_barcodes_per_genes_per_well']).all()
    assert (seq['n_reads_per_gene_per_well'] >= 0).all()


def test_sequencing_gives_no_reads_to_a_gene_that_is_not_in_the_well():
    spot = _small_spot_plate(seed=303)
    seq = simulate_sequencing_plate(spot, 500.0, 1.0, 0.2, 5000, seed=304)
    merged = seq.merge(spot[['gene', 'well', 'gene_in_well']], on=['gene', 'well'])
    absent = merged[merged['gene_in_well'] == 0]
    assert len(absent) > 0
    assert (absent['sequencing_n_cells_per_gene_per_well'] == 0).all()
    assert (absent['n_reads_per_gene_per_well'] == 0).all()


def test_sequencing_exhausts_the_pool_when_the_depth_exceeds_it():
    """Sampling *without replacement* to the bottom of the urn returns the urn.

    A multinomial (with-replacement) implementation would not: it would scatter
    the reads and leave some barcodes unread even at unlimited depth. This is the
    cheapest sharp test that the draw really is hypergeometric.
    """
    spot = _small_spot_plate(seed=305, n_genes=10, n_wells=20)
    seq = simulate_sequencing_plate(spot, 50.0, 0.0, 0.0, 10 ** 7, seed=306)
    assert seq['n_barcodes_per_genes_per_well'].sum() > 0
    assert (seq['n_reads_per_gene_per_well']
            == seq['n_barcodes_per_genes_per_well']).all()


def test_sequencing_read_share_tracks_barcode_share():
    """At a depth well below the pool, read fraction estimates barcode fraction."""
    spot = _small_spot_plate(seed=307, n_genes=20, n_wells=200, mu=6.0)
    seq = simulate_sequencing_plate(spot, 2000.0, 2.0, 0.0, 20000, seed=308)
    present = seq[seq['n_barcodes_per_genes_per_well'] > 0].copy()
    well_barcodes = present.groupby('well')['n_barcodes_per_genes_per_well'].transform('sum')
    well_reads = present.groupby('well')['n_reads_per_gene_per_well'].transform('sum')
    barcode_share = present['n_barcodes_per_genes_per_well'] / well_barcodes
    read_share = present['n_reads_per_gene_per_well'] / well_reads.clip(lower=1)
    assert np.abs(read_share - barcode_share).mean() < 0.01


def test_sequencing_depth_cv_zero_pins_every_well_to_the_target():
    spot = _small_spot_plate(seed=309)
    seq = simulate_sequencing_plate(spot, 2000.0, 2.0, 0.5, 7000, seed=310)
    assert seq['n_reads_per_well'].unique().tolist() == [7000]
    assert seq['n_reads_total'].unique().tolist() == [7000 * spot['well'].nunique()]


def test_sequencing_depth_cv_produces_the_requested_well_to_well_variation():
    """``read_depth_cv`` has to move the depth, and by the amount asked for.

    Real screens are far from uniform depth, and shallow wells are where hits go
    to die; a cv parameter that did nothing would make the simulator answer the
    wrong question with total confidence.
    """
    spot = _small_spot_plate(seed=311, n_genes=6, n_wells=3000)
    seq = simulate_sequencing_plate(spot, 100.0, 1.0, 0.1, 20000,
                                    read_depth_cv=0.35, seed=312)
    depths = seq.groupby('well')['n_reads_per_well'].first().to_numpy(dtype=float)
    assert abs(depths.mean() - 20000) < 500
    assert abs(depths.std() / depths.mean() - 0.35) < 0.03


def test_sequencing_refuses_a_barcode_pool_it_cannot_sample_exactly():
    """A linear-scale ``pcr_factor_mu`` blows the pool past numpy's ceiling."""
    spot = _small_spot_plate(seed=313, n_genes=10, n_wells=4)
    with pytest.raises(SequencingScaleError, match='LOG-scale'):
        simulate_sequencing_plate(spot, 100000.0, 25.0, 0.0, 10 ** 6, seed=314)


@pytest.mark.parametrize('kwargs,match', [
    (dict(sequencing_n_cells_per_well_lambda=0.0), 'sequencing_n_cells_per_well_lambda'),
    (dict(pcr_factor_var=-1.0), 'pcr_factor_var'),
    (dict(n_reads_per_well=-1), 'n_reads_per_well'),
    (dict(read_depth_cv=-0.1), 'read_depth_cv'),
])
def test_sequencing_refuses_out_of_range_parameters(kwargs, match):
    spot = _small_spot_plate(seed=315, n_genes=4, n_wells=4)
    call = dict(sequencing_n_cells_per_well_lambda=100.0, pcr_factor_mu=1.0,
                pcr_factor_var=0.1, n_reads_per_well=1000)
    call.update(kwargs)
    with pytest.raises(ScreenDesignError, match=match):
        simulate_sequencing_plate(spot, seed=316, **call)


def test_sequencing_with_zero_depth_returns_zero_reads_not_an_error():
    """A well sequenced to zero depth is a real, recoverable outcome."""
    spot = _small_spot_plate(seed=317, n_genes=5, n_wells=5)
    seq = simulate_sequencing_plate(spot, 100.0, 1.0, 0.1, 0, seed=318)
    assert (seq['n_reads_per_gene_per_well'] == 0).all()
    assert seq['n_barcodes_per_genes_per_well'].sum() > 0


def test_sequencing_result_is_independent_of_the_row_order_it_was_handed():
    spot = _small_spot_plate(seed=319)
    shuffled = spot.sample(frac=1.0, random_state=1).reset_index(drop=True)
    ordered = simulate_sequencing_plate(spot, 500.0, 1.0, 0.2, 5000, seed=320)
    scrambled = simulate_sequencing_plate(shuffled, 500.0, 1.0, 0.2, 5000, seed=320)
    pd.testing.assert_frame_equal(ordered, scrambled)


# ---------------------------------------------------------------------------
# The whole screen
# ---------------------------------------------------------------------------

def test_screen_is_one_row_per_gene_well_pair_with_every_stage_column():
    screen = simulate_screen(**SCREEN_KWARGS, seed=401)
    n_pairs = SCREEN_KWARGS['n_genes_in_library'] * SCREEN_KWARGS['n_wells_per_screen']
    assert len(screen) == n_pairs
    assert not screen.duplicated(subset=['gene', 'well']).any()
    for column in (
        'gene', 'well', 'hit', 'gene_abundance', 'gene_in_well',
        'imaging_n_cells_per_gene_per_well', 'imaging_n_cells_per_well', 'positive',
        'sequencing_n_cells_per_gene_per_well', 'n_barcodes_per_genes_per_well',
        'n_reads_per_gene_per_well', 'pcr_factor',
    ):
        assert column in screen.columns, column
    assert not screen.isna().any().any()


def test_screen_imaging_and_sequencing_observe_the_same_spot_plate():
    """Which genotypes are in a well is one physical fact, observed twice.

    If the two stages had been given independently simulated spot plates, a gene
    could be sequenced out of a well the microscope never saw it in — which
    would decouple the read fraction from the phenotype and quietly destroy the
    very association the model half regresses on.
    """
    screen = simulate_screen(**SCREEN_KWARGS, seed=402)
    absent = screen[screen['gene_in_well'] == 0]
    assert len(absent) > 0
    assert (absent['imaging_n_cells_per_gene_per_well'] == 0).all()
    assert (absent['sequencing_n_cells_per_gene_per_well'] == 0).all()
    assert (absent['n_reads_per_gene_per_well'] == 0).all()


def test_screen_stages_draw_from_independent_streams():
    """Changing an imaging parameter must not move the sequencing draws.

    Sharing one stream would make every point of a parameter sweep differ by an
    unrelated re-randomisation as well as by the parameter, so a sweep curve
    would be measuring noise it attributes to the knob.
    """
    baseline = simulate_screen(**SCREEN_KWARGS, seed=403)
    changed = dict(SCREEN_KWARGS, imaging_n_cells_per_well_mu=600.0)
    perturbed = simulate_screen(**changed, seed=403)
    pd.testing.assert_series_equal(
        baseline['n_reads_per_gene_per_well'], perturbed['n_reads_per_gene_per_well']
    )
    assert not baseline['positive'].equals(perturbed['positive'])

    changed = dict(SCREEN_KWARGS, n_reads_per_well=90000)
    perturbed = simulate_screen(**changed, seed=403)
    pd.testing.assert_series_equal(baseline['positive'], perturbed['positive'])
    assert not baseline['n_reads_per_gene_per_well'].equals(
        perturbed['n_reads_per_gene_per_well']
    )


def test_screen_of_one_gene_and_one_well_works():
    """The smallest non-degenerate screen must not need a special case."""
    # A one-gene library has abundance exactly 1.0, so the default
    # well_abundance_factor_mu of 4 would put the Bernoulli probability at 4 and
    # clip; 0.5 keeps the spot draw an honest coin flip.
    screen = simulate_screen(
        **dict(SCREEN_KWARGS, n_genes_in_library=1, n_wells_per_screen=1,
               gene_hit_rate=1.0, well_abundance_factor_mu=0.5,
               well_abundance_factor_var=0.01),
        seed=404,
    )
    assert len(screen) == 1
    assert screen['gene'].tolist() == [1]
    assert screen['well'].tolist() == [1]
    assert screen['hit'].tolist() == [1]


def test_screen_propagates_a_degenerate_design_as_a_named_error():
    with pytest.raises(ScreenDesignError, match='n_wells_per_screen'):
        simulate_screen(**dict(SCREEN_KWARGS, n_wells_per_screen=0), seed=405)
    with pytest.raises(ScreenDesignError, match='n_genes_in_library'):
        simulate_screen(**dict(SCREEN_KWARGS, n_genes_in_library=0), seed=405)
    with pytest.raises(ImpossibleMomentsError):
        simulate_screen(**dict(SCREEN_KWARGS, class_pos_mu=0.5, class_pos_var=0.4),
                        seed=405)


def test_a_strong_signal_makes_hits_score_above_non_hits():
    """Orientation. The single most dangerous thing to get backwards.

    The model half regresses per-well positive counts on per-gene log10 read
    fraction and calls a *higher* coefficient stronger evidence of a hit. If the
    simulator's signal ran the other way — hits depressing well positivity — the
    downstream AUROC would come back as ``1 - AUROC``, which for a
    strongly-powered screen is a small number that looks like a bug in the model
    rather than a sign error in the data.

    So: score each gene by its read-fraction-weighted mean well positivity, with
    no model in between, and require that hits rank at the top. Near 1 proves the
    orientation; near 0 would prove it inverted, and near 0.5 would prove the
    simulator carries no signal at all.
    """
    screen = simulate_screen(
        n_genes_in_library=120, gene_abundance_alpha=30.0, gene_hit_rate=0.1,
        n_wells_per_screen=400, well_abundance_factor_mu=3.0,
        well_abundance_factor_var=0.2,
        imaging_n_cells_per_well_mu=400.0, imaging_n_cells_per_well_var=None,
        class_pos_mu=0.95, class_pos_var=0.001,
        class_neg_mu=0.02, class_neg_var=0.0005,
        sequencing_n_cells_per_well_lambda=1000.0,
        pcr_factor_mu=1.0, pcr_factor_var=0.1, n_reads_per_well=30000,
        seed=406,
    )
    per_well = screen.groupby('well').agg(
        n_positive=('positive', 'sum'),
        n_cells=('imaging_n_cells_per_well', 'first'),
        well_reads=('n_reads_per_gene_per_well', 'sum'),
    )
    per_well['well_positivity'] = (
        per_well['n_positive'] / per_well['n_cells'].clip(lower=1)
    )
    joined = screen.merge(
        per_well[['well_positivity', 'well_reads']], left_on='well', right_index=True
    )
    # Read fraction is exactly the covariate the model half builds.
    weight = joined['n_reads_per_gene_per_well'] / joined['well_reads'].clip(lower=1)
    joined = joined.assign(_w=weight, _wx=weight * joined['well_positivity'])
    totals = joined.groupby('gene')[['_w', '_wx']].sum()
    score = totals['_wx'] / totals['_w'].clip(lower=1e-12)
    truth = screen.groupby('gene')['hit'].first()

    assert truth.sum() >= 5, 'not enough true hits to score'
    auroc = roc_auc_score(truth.loc[score.index].to_numpy(), score.to_numpy())
    assert auroc > 0.9, f'signal is absent or inverted: AUROC={auroc}'


def test_a_useless_classifier_carries_no_signal():
    """The negative control for the orientation test above.

    With ``class_pos_mu == class_neg_mu`` there is nothing to find, and the same
    naive score must land near chance. Without this, the orientation test could
    be passing on an artefact of the scoring rule rather than on the simulated
    phenotype.
    """
    screen = simulate_screen(
        n_genes_in_library=120, gene_abundance_alpha=30.0, gene_hit_rate=0.25,
        n_wells_per_screen=400, well_abundance_factor_mu=3.0,
        well_abundance_factor_var=0.2,
        imaging_n_cells_per_well_mu=400.0, imaging_n_cells_per_well_var=None,
        class_pos_mu=0.30, class_pos_var=0.001,
        class_neg_mu=0.30, class_neg_var=0.001,
        sequencing_n_cells_per_well_lambda=1000.0,
        pcr_factor_mu=1.0, pcr_factor_var=0.1, n_reads_per_well=30000,
        seed=407,
    )
    per_well = screen.groupby('well').agg(
        n_positive=('positive', 'sum'),
        n_cells=('imaging_n_cells_per_well', 'first'),
        well_reads=('n_reads_per_gene_per_well', 'sum'),
    )
    per_well['well_positivity'] = (
        per_well['n_positive'] / per_well['n_cells'].clip(lower=1)
    )
    joined = screen.merge(
        per_well[['well_positivity', 'well_reads']], left_on='well', right_index=True
    )
    weight = joined['n_reads_per_gene_per_well'] / joined['well_reads'].clip(lower=1)
    joined = joined.assign(_w=weight, _wx=weight * joined['well_positivity'])
    totals = joined.groupby('gene')[['_w', '_wx']].sum()
    score = totals['_wx'] / totals['_w'].clip(lower=1e-12)
    truth = screen.groupby('gene')['hit'].first()
    auroc = roc_auc_score(truth.loc[score.index].to_numpy(), score.to_numpy())
    assert 0.35 < auroc < 0.65, f'a null screen should score at chance, got {auroc}'
