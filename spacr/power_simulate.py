"""Simulator half of the spaCR power analysis — a Python port of ``spaCRPower``.

Ported from
`spaCRPower <https://github.com/maomlab/spaCRPower>`_ (``R/simulate_screen.R``),
Copyright (c) 2025 Matthew O'Meara (maom@umich.edu, ORCID 0000-0002-3128-5331),
released under the MIT licence:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

What this is for
----------------
A pooled Toxoplasma CRISPR screen has three noisy layers stacked on top of each
other — which genotypes landed in which well, how many of a well's cells the
microscope saw and how well the classifier called them, and how many reads each
genotype got out of the amplicon pool. The question a screener has to answer
*before* spending the microscope time is "with this library, this many wells,
this many cells per well and a classifier at 0.80/0.12, would I actually find a
hit?". That is a power analysis, and it is only answerable by simulation
because no layer is analytically tractable once the other two are stacked on it.

This module is the **simulator half**: it produces the tidy ``(well, gene)``
table that the model half (a Poisson regression of per-well positive counts on
per-gene log10 read fraction) consumes by column name. It deliberately does not
fit anything.

Four stages plus an orchestrator, mirroring ``R/simulate_screen.R``:

* :func:`simulate_library` — the genes, which of them are hits, and how
  unevenly the library is represented.
* :func:`simulate_spot_plate` — which genes landed in which wells.
* :func:`simulate_imaging_plate` — how many cells of each genotype were imaged
  in each well, and how many of them the classifier called positive.
* :func:`simulate_sequencing_plate` — the read count for each genotype in each
  well, after PCR amplification and a finite sequencing depth.
* :func:`simulate_screen` — all four, joined on ``(well, gene)``.

Reproducibility is not optional
-------------------------------
Every public sampler takes exactly one of ``seed=`` or ``rng=`` and refuses to
run with neither. The global numpy random state is never touched. A power
analysis is a number somebody defends in a grant review or a methods section;
one that cannot be re-run bit-for-bit is not evidence, so drawing silently from
OS entropy is treated as a configuration error rather than a convenience.

Deviations from the R package, and why
--------------------------------------
The R sources are the specification, but four of their behaviours are defects
that a line-by-line translation would faithfully reproduce. They are listed
here, and each is repeated at the function that departs from upstream. See also
``proposals/SIM_PORT_PLAN.md`` §3 in this repository, which catalogues them.

1. **``imaging_n_cells_per_well`` did not exist.** ``R/fit_model.R`` reads
   ``well_data$imaging_n_cells_per_well[1]`` as the Poisson offset, but
   ``simulate_imaging_plate`` emits only ``..._mu``, ``..._var`` and
   ``..._gene_per_well``; R's ``$`` partial matching is ambiguous across those
   three and yields ``NULL``, so the offset column vanishes and the simulate →
   fit path does not run as committed. :func:`simulate_imaging_plate` emits the
   column for real, as the realised per-well cell total.
2. **Reads per well were divided by the gene count, not the well count.**
   ``round(n_reads_total / nrow(well_data))`` groups by well, so ``nrow`` is the
   library size; with 452 genes and ``n_reads_total = 128318`` that is 284 reads
   per well against a real screen near 3e4. The parameter was also documented as
   "total reads" in one vignette and "geometric mean of well reads" in another.
   :func:`simulate_sequencing_plate` takes an unambiguous ``n_reads_per_well``
   and derives ``n_reads_total``.
3. **The hypergeometric draw size could exceed the urn.** Upstream guards
   ``rmvhyper`` with ``k = min(n_cells_in_well * pcr_factor, n_reads_per_well)``,
   but the urn is ``round(cells * pcr)`` computed **element-wise**, whose sum is
   not ``round(sum(cells) * pcr)``. The guard here is computed from the actual
   colour vector.
4. **The COM-Poisson branch is unusable.** ``COMPoissonReg::rcmp`` is called but
   not declared in the R package's ``DESCRIPTION``, and has no maintained Python
   equivalent inside spaCR's dependency set. :func:`sample_count_mean_variance`
   replaces the ``nu`` dispersion knob with a mean/variance pair dispatched to
   negative-binomial / Poisson / binomial, which spans the same over-, equi- and
   under-dispersed range in closed form with no new dependency. Third and higher
   moments differ from COM-Poisson; nothing upstream used them.

One further departure is a *choice*, not a bug fix, and is exposed as a
parameter: upstream splits a well's imaged cells **uniformly** across the genes
present (``prob = gene_in_well`` is a 0/1 vector), ignoring how abundant each
gene is. :func:`simulate_imaging_plate` defaults to ``imaging_split='abundance'``
and keeps ``'uniform'`` for exact R parity.

Two things neither package modelled
-----------------------------------
The list above is about *fidelity* to the R. These two are about fidelity to a
real screen, and both make the answer worse — which is why they are worth
having and why they are **off by default**. A simulator whose baseline shifted
under a version bump would make every power figure already quoted from it
wrong, so turning either on is an explicit act.

* **Sequencing error** (:func:`misassign_reads`,
  ``sequencing_error_rate=``). Both packages treat the read fraction as an
  exact record of which genotypes were in a well. Substitution errors, index
  hopping, PCR chimeras and mismatch-tolerant demultiplexing all credit reads
  to the wrong gene. Simulating it says the direct dilution is small — but
  that it silently disables the unidentified-gene check, turning genes that
  were correctly reported as untested into confident non-hits. On the
  reference design that costs ten times more than the dilution does. See
  :func:`misassign_reads`.
* **Well dropout from too few imaged cells**
  (:func:`drop_low_cell_wells`, ``min_cells_per_well=``). A well where the
  microscope found three cells enters the fit as one observation next to a
  well with four hundred. Its positive fraction can only be 0, 1/3, 2/3 or 1,
  and its standard error is several times the classifier's whole signal gap.
  The Poisson offset stops it dominating the scale of the fit; it does not
  stop its read-fraction covariate being paired with noise. Dropping such
  wells is what an analyst does by hand, and it costs wells — so which way the
  trade comes out is a thing to simulate rather than assert.

Column names follow the R package exactly so the model half can consume them by
name — including the upstream misspelling ``n_barcodes_per_genes_per_well``,
which is kept rather than quietly corrected, because renaming it would break the
join for anyone reading against the R vignettes.

:seealso: ``spacr.sim`` for the older, unrelated in-house screen simulator.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .errors import SpacrError

__all__ = [
    'PowerSimulationError',
    'ImpossibleMomentsError',
    'ScreenDesignError',
    'MalformedPlateError',
    'SequencingScaleError',
    'AbundanceClippedWarning',
    'resolve_rng',
    'rgamma_mean_variance',
    'rnbinom_mean_variance',
    'rbeta_mean_variance',
    'rdirichlet_stable',
    'sample_count_mean_variance',
    'simulate_library',
    'simulate_spot_plate',
    'simulate_imaging_plate',
    'simulate_sequencing_plate',
    'simulate_screen',
    'misassign_reads',
    'drop_low_cell_wells',
    'MAX_HYPERGEOMETRIC_URN',
    'DEFAULT_SEQUENCING_ERROR_RATE',
    'DEFAULT_MIN_CELLS_PER_WELL',
]

#: numpy's exact multivariate hypergeometric sampler documents a hard ceiling of
#: ``10**9`` on ``sum(colors)`` for the ``'marginals'`` method, above which it
#: loses precision. A barcode pool that large means the PCR factor or the
#: cells-per-well parameter is wrong by orders of magnitude, so we say so rather
#: than sample something subtly wrong.
MAX_HYPERGEOMETRIC_URN = 10 ** 9

#: A sane starting point for :func:`misassign_reads`, and the value the design
#: screen offers. Illumina's own substitution error is ~0.1 % per base, but the
#: quantity that matters here is the rate at which a *whole barcode* ends up
#: credited to the wrong gene, which also collects index hopping (0.1-2 % on a
#: patterned flow cell), PCR chimeras, and demultiplexing mismatch tolerance.
#: 0.5 % is the low end of what a real amplicon screen sees; a screen with
#: single-mismatch barcode rescue and a non-patterned flow cell can be under
#: it, and one with a crowded barcode set will be well over.
DEFAULT_SEQUENCING_ERROR_RATE = 0.005

#: A sane starting point for :func:`drop_low_cell_wells`. Below roughly 25
#: imaged cells the well's positive *fraction* is quantised in steps of 4 %
#: and its binomial standard error is larger than the classifier's whole
#: signal gap, so the well contributes noise the model cannot down-weight
#: enough. Zero means the filter is off, which is the default everywhere.
DEFAULT_MIN_CELLS_PER_WELL = 25

#: Relative tolerance used to decide that a requested variance *equals* the
#: requested mean, i.e. that the count distribution is Poisson. Without a
#: tolerance, ``var = mean`` computed through any arithmetic at all (say
#: ``mean * 1.0``) lands one ULP either side and silently switches between a
#: negative binomial with ``size = 1e16`` and a binomial with ``n = 1e16`` —
#: both of which are Poisson, but only one of which numpy will sample quickly.
_EQUIDISPERSION_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------

class PowerSimulationError(SpacrError):
    """Base class for every error this simulator raises deliberately.

    Subclasses :class:`spacr.errors.SpacrError` so callers can catch spaCR's own
    failures separately from an incidental ``ValueError`` out of numpy.
    """


class ImpossibleMomentsError(PowerSimulationError):
    """The requested (mean, variance) pair is not attainable by the distribution.

    Raised instead of clamping. A beta distribution cannot have variance above
    ``mean * (1 - mean)``, and a negative binomial cannot have variance below its
    mean; quietly moving the request onto the nearest feasible point would hand
    back samples from a distribution the caller did not ask for and never find
    out about — which is precisely how a power analysis ends up defending a
    number it did not compute.
    """


class ScreenDesignError(PowerSimulationError):
    """The screen design is degenerate or out of range.

    Zero genes, zero wells, a hit rate outside ``[0, 1]``, a non-positive
    abundance concentration. R's ``1:n_wells`` idiom turns ``n_wells = 0`` into
    the two-element vector ``c(1, 0)`` and simulates two wells; there is no
    silently-empty frame to return here, so we refuse the design instead.
    """


class MalformedPlateError(PowerSimulationError):
    """A plate frame handed to a downstream stage is not a valid plate.

    Missing a column the stage needs, or not exactly one row per
    ``(well, gene)`` pair. The downstream stages pivot the frame to a
    gene-by-well matrix, and a duplicated or missing pair would either silently
    drop observations or fill them with ``NaN`` that propagates into counts.
    """


class SequencingScaleError(PowerSimulationError):
    """The amplified barcode pool is too large to sample exactly.

    See :data:`MAX_HYPERGEOMETRIC_URN`. Almost always means ``pcr_factor_mu`` was
    given on the linear scale when it is a *log*-scale parameter.
    """


class AbundanceClippedWarning(UserWarning):
    """A ``gene_abundance x well_abundance`` product exceeded 1 and was clipped.

    That product is used as a Bernoulli probability. R's ``rbinom`` returns
    ``NA`` for ``prob > 1`` with a warning nobody reads, which poisons every
    downstream count. Clipping keeps the run alive, but it means the realised
    genes-per-well is *below* the requested ``well_abundance_factor_mu`` for the
    affected genes, so the warning carries the clip count and the worst offender.
    """


# ---------------------------------------------------------------------------
# Random number plumbing
# ---------------------------------------------------------------------------

def resolve_rng(
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.random.Generator:
    """Return the generator to draw from, insisting the caller chose one.

    Exactly one of ``rng`` or ``seed`` must be supplied.

    :param rng: An existing :class:`numpy.random.Generator` to advance. Use this
        to thread one stream through several stages so the whole screen is one
        reproducible draw.
    :param seed: Integer seed for a fresh :func:`numpy.random.default_rng`.
    :returns: The generator to sample from.
    :raises ScreenDesignError: If neither or both were supplied. Neither would
        mean seeding from OS entropy, which makes the result impossible to
        reproduce; both would make it ambiguous which one actually applied, and
        a caller who passed a seed would reasonably but wrongly believe the run
        was pinned by it.

    :example:

    >>> a = resolve_rng(seed=0).normal(size=3)
    >>> b = resolve_rng(seed=0).normal(size=3)
    >>> bool((a == b).all())
    True
    """
    if rng is not None and seed is not None:
        raise ScreenDesignError(
            'pass exactly one of rng= or seed=, not both: with both supplied it '
            'is ambiguous which stream the draw came from, and the seed you '
            'passed would not be the thing that pins the result.'
        )
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise ScreenDesignError(
                f'rng= must be a numpy.random.Generator, got '
                f'{type(rng).__name__}. Legacy numpy.random.RandomState objects '
                f'and the global numpy random state are refused on purpose: '
                f'they cannot be spawned into independent streams, so a '
                f'multi-stage screen drawn from them is not reproducible '
                f'stage-by-stage.'
            )
        return rng
    if seed is None:
        raise ScreenDesignError(
            'a seed is required: pass seed=<int> or rng=<numpy.random.Generator>. '
            'Defaulting to OS entropy would make this power analysis '
            'impossible to reproduce, and a power analysis you cannot re-run '
            'bit-for-bit is not evidence.'
        )
    return np.random.default_rng(seed)


def _spawn(rng: np.random.Generator, count: int) -> list:
    """Return ``count`` independent child generators derived from ``rng``.

    Uses :class:`numpy.random.SeedSequence` spawning rather than re-seeding from
    integers drawn out of ``rng``: spawned streams are guaranteed independent,
    whereas ``default_rng(rng.integers(...))`` correlates in ways that are
    invisible until a sweep of 100 grid points starts sharing structure.

    :param rng: Parent generator.
    :param count: Number of children to produce.
    :returns: List of ``count`` :class:`numpy.random.Generator` objects.
    """
    parent_seq = rng.bit_generator.seed_seq
    return [np.random.default_rng(child) for child in parent_seq.spawn(count)]


# ---------------------------------------------------------------------------
# Distribution reparameterisations (mean/variance instead of shape/rate)
# ---------------------------------------------------------------------------

def _check_positive(name: str, value: float) -> float:
    """Return ``value`` as a float after asserting it is finite and positive.

    :param name: Parameter name, used verbatim in the error message.
    :param value: Value to check.
    :returns: ``float(value)``.
    :raises ScreenDesignError: If the value is not finite or not > 0.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ScreenDesignError(f'{name} must be finite and positive, got {value!r}')
    return value


def _check_count(name: str, value) -> int:
    """Return ``value`` as a non-negative int, refusing anything lossy.

    :param name: Parameter name, used verbatim in the error message.
    :param value: Value to coerce.
    :returns: ``int(value)``.
    :raises ScreenDesignError: If the value is not an integer-valued number, or
        is negative. ``n=2.7`` is refused rather than truncated: a sample size
        that silently loses 0.7 of itself is the kind of thing that shows up as
        an off-by-one in a power curve months later.
    """
    as_float = float(value)
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise ScreenDesignError(f'{name} must be a whole number, got {value!r}')
    as_int = int(as_float)
    if as_int < 0:
        raise ScreenDesignError(f'{name} must be >= 0, got {value!r}')
    return as_int


def rgamma_mean_variance(
    n: int,
    mean: float,
    var: float,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw ``n`` gamma variates parameterised by mean and variance.

    A gamma with shape ``k`` and rate ``r`` has ``mean = k / r`` and
    ``var = k / r**2``; inverting gives ``r = mean / var`` and
    ``k = mean**2 / var``.

    **The R-to-numpy trap this function exists to close:** R's ``rgamma`` takes a
    *rate*, numpy's takes a *scale*, and ``scale = 1 / rate = var / mean``. A
    port that passes the rate to numpy gets a distribution wrong by a factor of
    ``var**2 / mean**2`` in the variance while still looking entirely plausible.

    :param n: Number of variates.
    :param mean: Target mean, must be finite and > 0.
    :param var: Target variance, must be finite and > 0.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: Float array of shape ``(n,)``.
    :raises ScreenDesignError: If ``mean`` or ``var`` is not finite and positive,
        or if ``n`` is negative, or if neither/both of ``rng`` and ``seed`` given.

    :example:

    >>> x = rgamma_mean_variance(200000, mean=4.0, var=2.0, seed=0)
    >>> bool(abs(x.mean() - 4.0) < 0.05 and abs(x.var() - 2.0) < 0.05)
    True
    """
    generator = resolve_rng(rng, seed)
    n = _check_count('n', n)
    mean = _check_positive('mean', mean)
    var = _check_positive('var', var)
    return generator.gamma(shape=mean ** 2 / var, scale=var / mean, size=n)


def rnbinom_mean_variance(
    n: int,
    mean: float,
    var: float,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw ``n`` negative-binomial variates parameterised by mean and variance.

    With ``size = mean**2 / (var - mean)`` and ``prob = mean / var`` the
    distribution has exactly the requested first two moments. numpy's
    ``negative_binomial(n, p)`` and R's ``rnbinom(size, prob)`` agree on the
    convention (both have mean ``n(1-p)/p``), so only the moment inversion
    needed porting.

    :param n: Number of variates.
    :param mean: Target mean, must be finite and > 0.
    :param var: Target variance. Must be **strictly greater** than ``mean``.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: Integer array of shape ``(n,)``.
    :raises ImpossibleMomentsError: If ``var <= mean``. The negative binomial is
        over-dispersed by construction: at ``var == mean`` the size parameter is
        ``+inf`` (the Poisson limit, which numpy cannot sample as an NB), and
        below it the size is negative. R's ``assertthat`` allows ``var == mean``
        and then hands ``Inf`` to ``rnbinom``, which returns ``NA`` — a whole
        column of silently missing counts. Use
        :func:`sample_count_mean_variance` if you want the equi- and
        under-dispersed cases handled for you.
    :raises ScreenDesignError: If ``mean`` is not finite and positive, ``n`` is
        negative, or neither/both of ``rng`` and ``seed`` given.

    :example:

    >>> x = rnbinom_mean_variance(200000, mean=10.0, var=40.0, seed=1)
    >>> bool(abs(x.mean() - 10.0) < 0.2 and abs(x.var() - 40.0) < 2.0)
    True
    """
    generator = resolve_rng(rng, seed)
    n = _check_count('n', n)
    mean = _check_positive('mean', mean)
    var = float(var)
    if not np.isfinite(var) or var <= mean:
        raise ImpossibleMomentsError(
            f'a negative binomial is over-dispersed: its variance is strictly '
            f'greater than its mean. Requested mean={mean!r}, var={var!r}. '
            f'At var == mean the distribution is Poisson and the size parameter '
            f'is infinite; below it the size is negative. Call '
            f'sample_count_mean_variance() to dispatch across the over-, equi- '
            f'and under-dispersed cases.'
        )
    return generator.negative_binomial(
        n=mean ** 2 / (var - mean), p=mean / var, size=n
    )


def rbeta_mean_variance(
    n: int,
    mean: float,
    var: float,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw ``n`` beta variates parameterised by mean and variance.

    Inverting ``mean = a / (a + b)`` and ``var = ab / ((a+b)**2 (a+b+1))`` gives
    ``a = mean * (mean(1-mean)/var - 1)`` and ``b = (1-mean) * (same)``.

    ``var == 0`` is accepted and returns a constant array at ``mean``. That is
    the exact limit of the family as ``a, b -> inf`` with the mean held, and it
    is the only way to express a *perfect* classifier — ``class_pos_mu=1.0,
    class_pos_var=0.0`` — which is the single most useful configuration for
    testing the imaging stage, since every hit cell must then be called positive.

    :param n: Number of variates.
    :param mean: Target mean, must lie in ``[0, 1]``.
    :param var: Target variance, must lie in ``[0, mean * (1 - mean))``.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: Float array of shape ``(n,)`` with values in ``[0, 1]``.
    :raises ImpossibleMomentsError: If ``var`` is negative, or if
        ``var >= mean * (1 - mean)``. That bound is the variance of the Bernoulli
        with the same mean and is the supremum over *all* distributions on
        ``[0, 1]``; no beta reaches it. Note this makes ``mean`` of exactly 0 or
        1 admissible only with ``var == 0``. Clamping instead of raising would
        return a distribution with the wrong spread and, for a classifier
        accuracy, silently change the effect size the whole power analysis is
        measuring.
    :raises ScreenDesignError: If ``mean`` is outside ``[0, 1]`` or not finite,
        ``n`` is negative, or neither/both of ``rng`` and ``seed`` given.

    :example:

    >>> p = rbeta_mean_variance(200000, mean=0.8, var=0.01, seed=2)
    >>> bool(abs(p.mean() - 0.8) < 0.005 and abs(p.var() - 0.01) < 0.001)
    True
    >>> rbeta_mean_variance(3, mean=1.0, var=0.0, seed=2).tolist()
    [1.0, 1.0, 1.0]
    """
    generator = resolve_rng(rng, seed)
    n = _check_count('n', n)
    mean = float(mean)
    var = float(var)
    if not np.isfinite(mean) or mean < 0.0 or mean > 1.0:
        raise ScreenDesignError(f'mean must be in [0, 1], got {mean!r}')
    if not np.isfinite(var) or var < 0.0:
        raise ScreenDesignError(f'var must be finite and >= 0, got {var!r}')

    max_var = mean * (1.0 - mean)
    if var == 0.0:
        # Degenerate limit: a point mass at the mean. Kept out of the beta call
        # because the shape parameters diverge.
        return np.full(n, mean, dtype=float)
    if var >= max_var:
        raise ImpossibleMomentsError(
            f'a distribution supported on [0, 1] with mean {mean!r} has variance '
            f'at most {max_var!r} (attained only by the Bernoulli), and a beta '
            f'is strictly below that. Requested var={var!r}. Lower the variance, '
            f'or move the mean away from the boundary.'
        )

    concentration = max_var / var - 1.0
    return generator.beta(
        a=mean * concentration, b=(1.0 - mean) * concentration, size=n
    )


def rdirichlet_stable(
    alpha: Union[float, Sequence[float], np.ndarray],
    n_categories: Optional[int] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw one Dirichlet vector, in log space so small ``alpha`` does not underflow.

    ``numpy.random.Generator.dirichlet`` draws independent gammas and normalises
    them. At small concentration the individual gamma draws underflow to exactly
    ``0.0``, so a gene gets an abundance of exactly zero — measured here, at
    ``alpha = 0.05`` with 452 categories, numpy produces exact zeros in roughly
    one draw in ten, and this routine produces none. Zero abundance is not merely
    unlucky: it removes the gene from every well, and the downstream read
    fraction for that gene becomes ``0 / 0`` in any well where nothing else
    amplified.

    The fix is Marsaglia and Tsang's boosting identity — for ``a < 1``,
    ``G(a) =d= G(a + 1) * U**(1/a)`` — evaluated in logs, followed by a
    log-sum-exp normalisation. Underflow then requires the *log* weight to
    overflow, which is ~700 times further away.

    :param alpha: Either a scalar concentration (then ``n_categories`` is
        required) or a 1-D array of per-category concentrations.
    :param n_categories: Number of categories when ``alpha`` is a scalar.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: Float array of shape ``(n_categories,)`` summing to 1.
    :raises ScreenDesignError: If any concentration is not finite and positive,
        if ``n_categories`` is missing for a scalar ``alpha`` or is not positive,
        or if neither/both of ``rng`` and ``seed`` given.

    :example:

    >>> w = rdirichlet_stable(0.6, 452, seed=3)
    >>> bool(abs(w.sum() - 1.0) < 1e-12 and (w > 0).all())
    True
    """
    generator = resolve_rng(rng, seed)
    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.ndim == 0:
        if n_categories is None:
            raise ScreenDesignError(
                'n_categories is required when alpha is a scalar concentration'
            )
        n_categories = _check_count('n_categories', n_categories)
        if n_categories == 0:
            raise ScreenDesignError('n_categories must be positive, got 0')
        alpha_array = np.full(n_categories, float(alpha_array))
    elif alpha_array.ndim != 1:
        raise ScreenDesignError(
            f'alpha must be a scalar or a 1-D array, got shape {alpha_array.shape}'
        )
    elif alpha_array.size == 0:
        raise ScreenDesignError('alpha must have at least one category')
    elif n_categories is not None and int(n_categories) != alpha_array.size:
        raise ScreenDesignError(
            f'n_categories={n_categories} contradicts len(alpha)={alpha_array.size}'
        )

    if not np.all(np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
        raise ScreenDesignError(
            f'every Dirichlet concentration must be finite and positive; got '
            f'min={np.nanmin(alpha_array)!r}'
        )

    # Boosted gammas: log G(a) = log G(a + 1) + log(U) / a.
    boosted = generator.gamma(shape=alpha_array + 1.0)
    uniforms = generator.random(alpha_array.shape)
    log_weights = np.log(boosted) + np.log(uniforms) / alpha_array
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)
    # exp() of the normalised logs sums to 1 only to within rounding; divide
    # again so callers can assert `sum == 1` rather than `isclose`.
    return weights / weights.sum()


def sample_count_mean_variance(
    n: int,
    mean: float,
    var: Optional[float] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw ``n`` non-negative integer counts with the requested mean and variance.

    Dispatches on the dispersion, because no single two-parameter count family
    covers the whole range:

    ===================  ==================================================
    dispersion           distribution
    ===================  ==================================================
    ``var > mean``       ``NegativeBinomial(mean**2/(var-mean), mean/var)``
    ``var == mean``      ``Poisson(mean)``
    ``0 <= var < mean``  ``Binomial(n, mean/n)`` with ``n`` the nearest
                         integer to ``mean**2 / (mean - var)``
    ===================  ==================================================

    This replaces upstream's ``COMPoissonReg::rcmp`` dispersion parameter
    ``nu``, which is called but not declared in the R package's ``DESCRIPTION``
    and has no maintained Python equivalent inside spaCR's dependency set. The
    third and higher moments differ from COM-Poisson; nothing upstream used them.

    **Where the under-dispersed case is approximate, and which way.** A binomial
    has an integer number of trials, so it cannot hit an arbitrary
    ``(mean, var)`` pair. Given the rounded ``n``, the success probability is set
    to ``mean / n`` so the **mean is exact** and the realised variance is
    ``mean * (1 - mean/n)``, the nearest value the family attains — within about
    ``1/n`` relative of the request. The mean is the one preserved because it is
    the biologically meaningful quantity (cells per well) and it sets the scale
    of everything downstream, whereas a variance off by a percent changes only
    how wide the well-to-well spread is. Rounding ``n`` while keeping
    ``p = (mean - var) / mean`` — the naive inversion — biases the *mean* instead,
    by up to half a count, which is a systematic error in every well of a sweep.

    :param n: Number of variates.
    :param mean: Target mean, must be finite and > 0.
    :param var: Target variance. ``None`` means "Poisson", i.e. ``var = mean``.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: Integer array of shape ``(n,)``.
    :raises ImpossibleMomentsError: If ``var`` is negative. A count distribution
        cannot have negative variance, and the caller has almost certainly
        passed a standard deviation or a coefficient of variation by mistake.
    :raises ScreenDesignError: If ``mean`` is not finite and positive, ``n`` is
        negative, or neither/both of ``rng`` and ``seed`` given.

    :example:

    >>> under = sample_count_mean_variance(100000, mean=100.0, var=25.0, seed=4)
    >>> bool(abs(under.mean() - 100.0) < 0.5 and abs(under.var() - 25.0) < 1.0)
    True
    """
    generator = resolve_rng(rng, seed)
    n = _check_count('n', n)
    mean = _check_positive('mean', mean)
    if var is None:
        var = mean
    var = float(var)
    if not np.isfinite(var) or var < 0.0:
        raise ImpossibleMomentsError(
            f'a count distribution cannot have variance {var!r}; variance is '
            f'non-negative. If you meant a standard deviation, square it first.'
        )

    if abs(var - mean) <= _EQUIDISPERSION_RTOL * mean:
        return generator.poisson(lam=mean, size=n)
    if var > mean:
        return rnbinom_mean_variance(n, mean=mean, var=var, rng=generator)

    # Under-dispersed: binomial. n_trials = mean^2 / (mean - var) is generally
    # not an integer. Round it, then recover p from the rounded n as `mean/n`
    # rather than using the closed form `(mean - var)/mean`: that keeps the mean
    # exact and pushes the whole rounding error into the variance. With the naive
    # p the mean is short by up to half a count in *every* well, which a sweep
    # over the variance turns into a drifting baseline that looks like signal.
    n_trials = int(round(mean ** 2 / (mean - var)))
    # p <= 1 requires n_trials >= mean; the closed form satisfies that for any
    # var >= 0, but rounding down can just cross it when var is tiny.
    n_trials = max(n_trials, int(np.ceil(mean)), 1)
    return generator.binomial(n=n_trials, p=mean / n_trials, size=n)


# ---------------------------------------------------------------------------
# Stage 1 — the library
# ---------------------------------------------------------------------------

def simulate_library(
    n_genes_in_library: int,
    gene_abundance_alpha: float,
    gene_hit_rate: float,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate the perturbation library: which genes exist, how abundant, which are hits.

    ``gene_abundance ~ Dirichlet(gene_abundance_alpha * 1_n)`` and
    ``hit_i ~ Bernoulli(gene_hit_rate)``, independently.

    ``gene_abundance_alpha`` controls how *even* the library is, and it runs the
    opposite way from intuition: large alpha gives every gene about ``1/n`` of
    the pool, alpha near zero concentrates the whole pool on one gene, and
    ``alpha = 1`` corresponds to a Gini index of 0.5. The real *T. gondii*
    screen this simulator was fitted to came out at ``alpha = 0.6``, i.e. quite
    skewed — which is why some genes in it never reached enough wells to be
    callable at all.

    :param n_genes_in_library: Number of genes, must be > 0.
    :param gene_abundance_alpha: Dirichlet concentration, must be > 0.
    :param gene_hit_rate: Probability each gene is a true hit, in ``[0, 1]``.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: DataFrame with ``n_genes_in_library`` rows and columns
        ``[gene, gene_abundance_alpha, gene_hit_rate, gene_abundance, hit]``.
        ``gene`` is a 1-based integer index (matching the R package), and the
        ``gene_abundance`` column sums to exactly 1.
    :raises ScreenDesignError: If the library is empty, the concentration is not
        positive, the hit rate is outside ``[0, 1]``, or neither/both of ``rng``
        and ``seed`` given. An empty library is refused rather than returned as
        an empty frame, because every downstream stage would then produce an
        empty frame too and the run would report "no hits found" instead of
        "you asked for no genes".

    :example:

    >>> lib = simulate_library(500, gene_abundance_alpha=10.0,
    ...                        gene_hit_rate=0.1, seed=5)
    >>> bool(abs(lib['gene_abundance'].sum() - 1.0) < 1e-12)
    True
    >>> int(lib['gene'].iloc[0]), int(lib['gene'].iloc[-1])
    (1, 500)
    """
    generator = resolve_rng(rng, seed)
    n_genes_in_library = _check_count('n_genes_in_library', n_genes_in_library)
    if n_genes_in_library == 0:
        raise ScreenDesignError(
            'n_genes_in_library must be > 0; a library with no genes produces an '
            'empty screen, which downstream reads as "nothing was a hit" rather '
            'than as a broken design.'
        )
    gene_abundance_alpha = _check_positive('gene_abundance_alpha', gene_abundance_alpha)
    gene_hit_rate = float(gene_hit_rate)
    if not np.isfinite(gene_hit_rate) or not 0.0 <= gene_hit_rate <= 1.0:
        raise ScreenDesignError(
            f'gene_hit_rate must be a probability in [0, 1], got {gene_hit_rate!r}'
        )

    abundance = rdirichlet_stable(
        gene_abundance_alpha, n_genes_in_library, rng=generator
    )
    hit = generator.binomial(n=1, p=gene_hit_rate, size=n_genes_in_library)

    return pd.DataFrame({
        'gene': np.arange(1, n_genes_in_library + 1, dtype=np.int64),
        'gene_abundance_alpha': np.full(n_genes_in_library, gene_abundance_alpha),
        'gene_hit_rate': np.full(n_genes_in_library, gene_hit_rate),
        'gene_abundance': abundance,
        'hit': hit.astype(np.int64),
    })


# ---------------------------------------------------------------------------
# Stage 2 — the spot plate
# ---------------------------------------------------------------------------

def simulate_spot_plate(
    gene_library: pd.DataFrame,
    n_wells_per_screen: int,
    well_abundance_factor_mu: float,
    well_abundance_factor_var: float,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate which genes landed in which wells.

    ``well_abundance_j ~ Gamma(mean=well_abundance_factor_mu,
    var=well_abundance_factor_var)`` and
    ``gene_in_well_ij ~ Bernoulli(gene_abundance_i * well_abundance_j)``.

    ``well_abundance_factor_mu`` is the knob that trades genes-per-well against
    wells-per-gene, and it is the sweep the R package cared most about: with 452
    genes, ``mu = 4.6`` gives roughly 4.6 genes per well.

    **Probabilities above 1 are clipped, loudly.** The Bernoulli probability is a
    product of two independently drawn quantities and nothing constrains it to
    ``[0, 1]``; at ``alpha = 0.6`` the most abundant gene's share times a
    ``mu = 4.6`` well factor exceeds 1 routinely. R's ``rbinom`` returns ``NA``
    there and carries on. This clips to 1 and emits one
    :class:`AbundanceClippedWarning` carrying the number of clipped cells and the
    largest offending probability, because a clipped run has a realised
    genes-per-well *below* the one that was requested — the answer is still
    usable, but not for the parameter value written on it.

    :param gene_library: Result of :func:`simulate_library`; needs at least
        ``gene`` and ``gene_abundance``, and all its columns are carried through.
    :param n_wells_per_screen: Number of wells, must be > 0.
    :param well_abundance_factor_mu: Mean per-well abundance factor, must be > 0.
    :param well_abundance_factor_var: Variance of it, must be > 0.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: DataFrame with one row per ``(gene, well)`` pair — gene-major,
        well varying fastest, matching ``tidyr::expand_grid`` — and columns
        ``[gene, well, <library columns>, well_abundance_factor_mu,
        well_abundance_factor_var, well_abundance, gene_in_well]``.
        ``DataFrame.attrs['n_prob_clipped']`` records the clip count; note pandas
        drops ``attrs`` through most merges, so the warning is the durable record.
    :raises ScreenDesignError: If there are no wells, the abundance moments are
        not positive, or neither/both of ``rng`` and ``seed`` given. Zero wells
        is refused because R's ``1:0`` idiom silently yields *two* wells.
    :raises MalformedPlateError: If ``gene_library`` is empty or is missing
        ``gene`` / ``gene_abundance``.

    :example:

    >>> lib = simulate_library(50, 10.0, 0.1, seed=6)
    >>> spot = simulate_spot_plate(lib, 8, 2.0, 0.1, seed=7)
    >>> len(spot), sorted(spot['well'].unique().tolist())[:3]
    (400, [1, 2, 3])
    """
    generator = resolve_rng(rng, seed)
    n_wells_per_screen = _check_count('n_wells_per_screen', n_wells_per_screen)
    if n_wells_per_screen == 0:
        raise ScreenDesignError(
            'n_wells_per_screen must be > 0; the R original writes `1:n_wells`, '
            'which turns 0 wells into the two-element vector c(1, 0) and '
            'simulates two of them.'
        )
    well_abundance_factor_mu = _check_positive(
        'well_abundance_factor_mu', well_abundance_factor_mu
    )
    well_abundance_factor_var = _check_positive(
        'well_abundance_factor_var', well_abundance_factor_var
    )
    _require_columns(gene_library, ('gene', 'gene_abundance'), 'gene_library')
    if len(gene_library) == 0:
        raise MalformedPlateError(
            'gene_library is empty; simulate_library() refuses to build one, so '
            'an empty frame here means it was filtered away upstream.'
        )
    if gene_library['gene'].duplicated().any():
        raise MalformedPlateError(
            'gene_library has duplicate gene ids; the expand-grid below would '
            'then produce several rows per (gene, well) and every downstream '
            'per-well pivot would silently take only one of them.'
        )

    n_genes = len(gene_library)
    genes = gene_library['gene'].to_numpy()
    wells = np.arange(1, n_wells_per_screen + 1, dtype=np.int64)

    well_abundance = rgamma_mean_variance(
        n_wells_per_screen,
        mean=well_abundance_factor_mu,
        var=well_abundance_factor_var,
        rng=generator,
    )

    # expand_grid(gene, well): gene-major, well varying fastest.
    gene_index = np.repeat(np.arange(n_genes), n_wells_per_screen)
    well_index = np.tile(np.arange(n_wells_per_screen), n_genes)

    abundance = gene_library['gene_abundance'].to_numpy(dtype=float)[gene_index]
    well_factor = well_abundance[well_index]
    prob = abundance * well_factor

    n_clipped = int(np.count_nonzero(prob > 1.0))
    worst = float(prob.max()) if prob.size else 0.0
    if n_clipped:
        prob = np.clip(prob, 0.0, 1.0)
        warnings.warn(
            f'{n_clipped} of {prob.size} gene-in-well probabilities exceeded 1 '
            f'and were clipped (largest was {worst:.3f}). The realised '
            f'genes-per-well is therefore below the requested '
            f'well_abundance_factor_mu={well_abundance_factor_mu!r} for the '
            f'affected genes: their abundance already saturates every well. '
            f'Lower well_abundance_factor_mu, or raise gene_abundance_alpha to '
            f'flatten the library.',
            AbundanceClippedWarning,
            stacklevel=2,
        )

    gene_in_well = generator.binomial(n=1, p=prob).astype(np.int64)

    frame = pd.DataFrame({
        'gene': genes[gene_index],
        'well': wells[well_index],
    })
    for column in gene_library.columns:
        if column == 'gene':
            continue
        frame[column] = gene_library[column].to_numpy()[gene_index]
    frame['well_abundance_factor_mu'] = well_abundance_factor_mu
    frame['well_abundance_factor_var'] = well_abundance_factor_var
    frame['well_abundance'] = well_factor
    frame['gene_in_well'] = gene_in_well
    frame.attrs['n_prob_clipped'] = n_clipped
    return frame


# ---------------------------------------------------------------------------
# Plate-frame plumbing shared by stages 3 and 4
# ---------------------------------------------------------------------------

def _require_columns(frame: pd.DataFrame, columns, label: str) -> None:
    """Raise :class:`MalformedPlateError` unless ``frame`` has every named column.

    :param frame: Frame to check.
    :param columns: Iterable of required column names.
    :param label: Name of the parameter, used verbatim in the error message.
    :raises MalformedPlateError: If ``frame`` is not a DataFrame or lacks a column.
    """
    if not isinstance(frame, pd.DataFrame):
        raise MalformedPlateError(
            f'{label} must be a pandas DataFrame, got {type(frame).__name__}'
        )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MalformedPlateError(
            f'{label} is missing required column(s) {missing}; it has '
            f'{list(frame.columns)}'
        )


def _plate_grid(spot_plate: pd.DataFrame, label: str = 'spot_plate'):
    """Return ``(genes, wells, index_of_pair)`` for a rectangular ``(gene, well)`` frame.

    Downstream stages need a gene-by-well matrix, and the honest way to get one
    from a tidy frame is a pivot on the *sorted* unique ids rather than a
    reshape of whatever row order arrived. A reshape is what R's ``dplyr::do()``
    pipeline effectively assumes, and it is wrong the moment a caller sorts or
    filters the frame — the counts would then be attributed to the wrong genes,
    which is invisible in every summary statistic.

    :param spot_plate: Tidy frame with ``gene`` and ``well`` columns.
    :param label: Parameter name for error messages.
    :returns: Tuple ``(genes, wells, position)`` where ``genes`` and ``wells``
        are sorted unique id arrays and ``position`` is an ``(n_rows,)`` integer
        array giving each row's index into a flattened ``(n_genes, n_wells)``
        matrix.
    :raises MalformedPlateError: If the frame is empty, or does not contain
        exactly one row per ``(gene, well)`` pair.
    """
    _require_columns(spot_plate, ('gene', 'well'), label)
    if len(spot_plate) == 0:
        raise MalformedPlateError(f'{label} has no rows')

    genes = np.unique(spot_plate['gene'].to_numpy())
    wells = np.unique(spot_plate['well'].to_numpy())
    expected = genes.size * wells.size
    if len(spot_plate) != expected:
        raise MalformedPlateError(
            f'{label} must have exactly one row per (gene, well) pair: '
            f'{genes.size} genes x {wells.size} wells = {expected} rows, but the '
            f'frame has {len(spot_plate)}. A ragged plate would leave holes in '
            f'the per-well pivot that fill with NaN and propagate into counts.'
        )

    gene_position = np.searchsorted(genes, spot_plate['gene'].to_numpy())
    well_position = np.searchsorted(wells, spot_plate['well'].to_numpy())
    position = gene_position * wells.size + well_position
    if np.unique(position).size != expected:
        raise MalformedPlateError(
            f'{label} contains duplicate (gene, well) pairs; every pair must '
            f'appear exactly once.'
        )
    return genes, wells, position


def _as_matrix(values: np.ndarray, position: np.ndarray, shape) -> np.ndarray:
    """Scatter ``values`` into a dense ``shape`` matrix using ``position``.

    :param values: 1-D array of per-row values.
    :param position: Flattened destination index for each row, from
        :func:`_plate_grid`.
    :param shape: ``(n_genes, n_wells)``.
    :returns: Matrix of ``values`` laid out gene-by-well.
    """
    flat = np.empty(shape[0] * shape[1], dtype=values.dtype)
    flat[position] = values
    return flat.reshape(shape)


def _expand_grid_frame(genes: np.ndarray, wells: np.ndarray) -> pd.DataFrame:
    """Return the ``(gene, well)`` skeleton in ``tidyr::expand_grid`` order.

    ``tidyr::expand_grid`` varies the *last* variable fastest — the opposite of
    base R's ``expand.grid`` — so ``well`` cycles within each ``gene``.

    :param genes: Sorted unique gene ids.
    :param wells: Sorted unique well ids.
    :returns: Two-column DataFrame with ``len(genes) * len(wells)`` rows.
    """
    return pd.DataFrame({
        'gene': np.repeat(genes, wells.size),
        'well': np.tile(wells, genes.size),
    })


def _multinomial_pvals(weights: np.ndarray) -> np.ndarray:
    """Normalise ``weights`` into pvals numpy's ``multinomial`` will accept.

    numpy rejects ``sum(pvals[:-1]) > 1`` — including when the excess is a single
    ULP left over from dividing by the sum — while R's ``rmultinom``
    renormalises internally. The last entry absorbs the rounding here so the
    caller does not have to catch a ``ValueError`` that means nothing.

    :param weights: Non-negative weights, not all zero.
    :returns: Probability vector of the same length.
    """
    pvals = weights / weights.sum()
    head_sum = pvals[:-1].sum()
    if head_sum > 1.0:
        pvals[:-1] /= head_sum
        head_sum = pvals[:-1].sum()
    pvals[-1] = max(0.0, 1.0 - head_sum)
    return pvals


# ---------------------------------------------------------------------------
# Stage 3 — the imaging plate
# ---------------------------------------------------------------------------

def simulate_imaging_plate(
    spot_plate: pd.DataFrame,
    imaging_n_cells_per_well_mu: float,
    imaging_n_cells_per_well_var: Optional[float],
    class_pos_mu: float,
    class_pos_var: float,
    class_neg_mu: float,
    class_neg_var: float,
    *,
    imaging_split: str = 'abundance',
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate the imaged cells per genotype per well and the classifier's calls.

    Per well: draw the well's total imaged cell count from
    :func:`sample_count_mean_variance`, split it multinomially across the genes
    present, then call each cell positive with a probability drawn per
    ``(gene, well)`` from a beta — ``class_pos_*`` for cells of a hit genotype,
    ``class_neg_*`` otherwise.

    The gap between ``class_pos_mu`` and ``class_neg_mu`` **is** the signal the
    whole power analysis is trying to detect. The real MaxViT classifier this was
    fitted to sat at 0.80 / 0.12, which is a modest gap, and the point of the
    exercise is that a modest gap is survivable given enough wells.

    ``imaging_split`` chooses how a well's cells are divided between the genes
    in it:

    * ``'abundance'`` (default) weights by each gene's library abundance, so a
      gene that is 20% of the well gets 20% of its cells.
    * ``'uniform'`` splits evenly, which is what the R package does — its
      ``prob = gene_in_well`` is a 0/1 vector, so abundance is ignored at this
      step. Kept for parity; it understates the imbalance between genotypes and
      therefore *overstates* power.

    :param spot_plate: Result of :func:`simulate_spot_plate`. Needs ``gene``,
        ``well``, ``gene_in_well`` and ``hit``, plus ``gene_abundance`` when
        ``imaging_split='abundance'``.
    :param imaging_n_cells_per_well_mu: Mean cells imaged per well, must be > 0.
    :param imaging_n_cells_per_well_var: Variance of that count. ``None`` means
        Poisson, and the emitted ``imaging_n_cells_per_well_var`` column then
        echoes the mean, because that is the variance the model used. Unlike the
        R original, variance *below* the mean is allowed and draws from a
        binomial.
    :param class_pos_mu: Mean probability a hit-genotype cell is called positive.
    :param class_pos_var: Variance of it across ``(gene, well)``; 0 is allowed
        and gives a deterministic classifier.
    :param class_neg_mu: Mean probability a non-hit cell is called positive.
    :param class_neg_var: Variance of it across ``(gene, well)``.
    :param imaging_split: ``'abundance'`` or ``'uniform'``, see above.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: DataFrame with one row per ``(gene, well)`` and columns
        ``[gene, well, imaging_n_cells_per_well_mu, imaging_n_cells_per_well_var,
        imaging_n_cells_per_gene_per_well, imaging_n_cells_per_well,
        class_pos_mu, class_pos_var, class_neg_mu, class_neg_var, positive]``.

        ``imaging_n_cells_per_well`` is the well total, repeated on every row of
        the well. It **is not in the R output** — ``R/fit_model.R`` reads it as
        the Poisson offset, R's ``$`` partial matching finds three columns with
        that prefix and returns ``NULL``, and the fit silently loses its offset.
        It is emitted here because the model half needs it.
    :raises ScreenDesignError: If ``imaging_split`` is not one of the two
        supported values, the cell-count mean is not positive, or neither/both of
        ``rng`` and ``seed`` given.
    :raises ImpossibleMomentsError: If a classifier ``(mu, var)`` pair is not a
        realisable beta — see :func:`rbeta_mean_variance`.
    :raises MalformedPlateError: If ``spot_plate`` is not one row per
        ``(gene, well)`` or is missing a required column.

    :example:

    A perfect classifier must call every hit cell positive and no other:

    >>> lib = simulate_library(20, 50.0, 0.5, seed=8)
    >>> spot = simulate_spot_plate(lib, 40, 2.0, 0.05, seed=9)
    >>> img = simulate_imaging_plate(spot, 100.0, None,
    ...                              class_pos_mu=1.0, class_pos_var=0.0,
    ...                              class_neg_mu=0.0, class_neg_var=0.0, seed=10)
    >>> merged = img.merge(spot[['gene', 'well', 'hit']], on=['gene', 'well'])
    >>> bool((merged.loc[merged['hit'] == 0, 'positive'] == 0).all())
    True
    """
    generator = resolve_rng(rng, seed)
    if imaging_split not in ('abundance', 'uniform'):
        raise ScreenDesignError(
            f"imaging_split must be 'abundance' or 'uniform', got "
            f"{imaging_split!r}"
        )
    imaging_n_cells_per_well_mu = _check_positive(
        'imaging_n_cells_per_well_mu', imaging_n_cells_per_well_mu
    )
    needed = ['gene', 'well', 'gene_in_well', 'hit']
    if imaging_split == 'abundance':
        needed.append('gene_abundance')
    _require_columns(spot_plate, needed, 'spot_plate')
    genes, wells, position = _plate_grid(spot_plate)
    shape = (genes.size, wells.size)

    in_well = _as_matrix(
        spot_plate['gene_in_well'].to_numpy(dtype=np.int64), position, shape
    )
    hit = _as_matrix(spot_plate['hit'].to_numpy(dtype=np.int64), position, shape)
    if imaging_split == 'abundance':
        weight_source = _as_matrix(
            spot_plate['gene_abundance'].to_numpy(dtype=float), position, shape
        )
    else:
        weight_source = np.ones(shape, dtype=float)

    # One well total per well, drawn before the split so that the number of
    # imaged cells does not depend on how many genes happened to land there.
    well_totals = sample_count_mean_variance(
        wells.size,
        mean=imaging_n_cells_per_well_mu,
        var=imaging_n_cells_per_well_var,
        rng=generator,
    )

    counts = np.zeros(shape, dtype=np.int64)
    for well_column in range(wells.size):
        weights = in_well[:, well_column] * weight_source[:, well_column]
        total = int(well_totals[well_column])
        if total == 0 or weights.sum() <= 0.0:
            # R has this branch too: a well with no genes spotted into it still
            # exists, it just has nothing to attribute its cells to.
            continue
        counts[:, well_column] = generator.multinomial(
            total, _multinomial_pvals(weights)
        )

    realised_well_totals = counts.sum(axis=0)

    n_rows = genes.size * wells.size
    # Both classifier probabilities are drawn for every row and then selected
    # between, exactly as R's `ifelse(hit, rbeta(...), rbeta(...))` does: the
    # per-row draw is what makes the classifier's operating point vary between
    # observations rather than being one number for the whole screen.
    prob_pos = rbeta_mean_variance(
        n_rows, mean=class_pos_mu, var=class_pos_var, rng=generator
    ).reshape(shape)
    prob_neg = rbeta_mean_variance(
        n_rows, mean=class_neg_mu, var=class_neg_var, rng=generator
    ).reshape(shape)
    prob_positive = np.where(hit.astype(bool), prob_pos, prob_neg)

    positive = generator.binomial(n=counts, p=prob_positive).astype(np.int64)

    frame = _expand_grid_frame(genes, wells)
    frame['imaging_n_cells_per_well_mu'] = imaging_n_cells_per_well_mu
    # Echo the variance the count model actually used, not the literal argument:
    # `None` means Poisson, and a Poisson's variance is its mean. Writing NaN
    # here instead would put a NaN column into the frame the model half consumes,
    # where "this parameter was left at its default" is indistinguishable from
    # "this number failed to compute".
    frame['imaging_n_cells_per_well_var'] = (
        imaging_n_cells_per_well_mu if imaging_n_cells_per_well_var is None
        else float(imaging_n_cells_per_well_var)
    )
    frame['imaging_n_cells_per_gene_per_well'] = counts.reshape(-1)
    frame['imaging_n_cells_per_well'] = np.tile(
        realised_well_totals, genes.size
    ).astype(np.int64)
    frame['class_pos_mu'] = float(class_pos_mu)
    frame['class_pos_var'] = float(class_pos_var)
    frame['class_neg_mu'] = float(class_neg_mu)
    frame['class_neg_var'] = float(class_neg_var)
    frame['positive'] = positive.reshape(-1)
    return frame


# ---------------------------------------------------------------------------
# Stage 4a — sequencing error
# ---------------------------------------------------------------------------

def misassign_reads(
    reads: np.ndarray,
    sequencing_error_rate: float,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Credit a fraction of each well's reads to the wrong gene.

    Neither spaCRPower nor the port had this stage, and its absence is
    optimistic in a specific way: it makes the read fraction an *exact*
    measure of which genotypes were in a well. In a real amplicon screen it is
    not. A barcode arrives at the wrong gene through base-call substitutions,
    index hopping on a patterned flow cell, PCR chimeras across the pooled
    reaction, and mismatch-tolerant demultiplexing — and every one of those
    mechanisms moves reads *toward the middle*, because the destination is
    drawn from the whole library rather than from the wells the source
    genotype was actually in.

    Mis-assignment gives a gene phantom reads in wells it never entered, so
    the covariate the model regresses positive counts on is a
    shrunk-toward-uniform version of the truth. The obvious consequence is
    regression dilution — an attenuated coefficient and a smaller apparent
    effect — and measuring it in this simulator says that at realistic rates
    it is **small**: on the 452-gene reference design, 0.5 % mis-assignment
    moves the hit/non-hit separation from 0.799 to 0.794 among the genes that
    were identifiable to begin with. Deep sequencing averages most of it away.

    The consequence that is *not* obvious, and is much larger, is what it does
    to the genes that were never testable. A gene that landed in every well,
    or in none, has a constant read-fraction column; its coefficient is
    confounded with the intercept and ``power_model.prepare_model_data``
    reports it as unidentified rather than as a non-hit. That check is the
    one honest thing standing between a thin design and a page of confident
    negative results — and mis-assignment defeats it, because phantom reads
    give every gene a covariate that *varies*. On the same reference design,
    0.5 % error takes the scored library from 317 genes to all 452: the 135
    genes that were correctly flagged "untested" become scored non-hits with
    a covariate made entirely of noise, and the screen-wide separation falls
    from 0.799 to 0.723 — fourteen times the direct dilution, and all of it
    from a safeguard being switched off rather than from any loss of signal.

    So the reason to simulate this is not to price the noise. It is that a
    screen with sequencing error and a screen without it disagree about *how
    many genes were tested*, and only one of them is telling the truth.

    The model, per well:

    1. Each gene's reads survive independently with probability
       ``1 - sequencing_error_rate``.
    2. Everything lost is pooled and redistributed **uniformly across the
       whole library**, including genes absent from the well and including the
       gene it came from.

    Uniform, not abundance-weighted, because the destination of a misread
    barcode is set by which barcodes are one edit away from it, not by how
    much of the source was in the tube. And because self-assignment is
    allowed, the *effective* mis-assignment rate is
    ``sequencing_error_rate * (1 - 1/n_genes)``; at a 452-gene library that is
    a 0.2 % correction, and keeping it makes the read total exactly conserved,
    which is a much more useful invariant to be able to assert.

    Cross-*well* hopping — a read landing in the wrong sample entirely — is
    not modelled. It needs a plate-level index layout the simulator does not
    carry, and within-well gene confusion is the mechanism that dilutes the
    effect being measured.

    :param reads: ``(n_genes, n_wells)`` integer read counts.
    :param sequencing_error_rate: probability a read is credited elsewhere,
        in ``[0, 1]``. ``0.0`` returns the input unchanged.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: a new ``(n_genes, n_wells)`` array. Every column sums to
        exactly what it summed to before.
    :raises ScreenDesignError: If the rate is outside ``[0, 1]``, the array is
        not 2-D or holds a negative count, or neither/both of ``rng`` and
        ``seed`` are given.

    :example:

    Total reads are conserved, and at a full error rate nothing of the
    original assignment survives except by chance:

    >>> counts = np.array([[100, 0], [0, 100]])
    >>> out = misassign_reads(counts, 1.0, seed=3)
    >>> [int(column.sum()) for column in out.T]
    [100, 100]
    """
    generator = resolve_rng(rng, seed)
    rate = float(sequencing_error_rate)
    if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ScreenDesignError(
            f'sequencing_error_rate is a probability and must be in [0, 1], '
            f'got {sequencing_error_rate!r}'
        )
    counts = np.asarray(reads)
    if counts.ndim != 2:
        raise ScreenDesignError(
            f'misassign_reads expects a (n_genes, n_wells) matrix, got shape '
            f'{counts.shape}'
        )
    counts = counts.astype(np.int64)
    if (counts < 0).any():
        raise ScreenDesignError('read counts must not be negative')
    if rate == 0.0 or counts.size == 0:
        return counts.copy()

    n_genes = counts.shape[0]
    lost = generator.binomial(counts, rate)
    kept = counts - lost
    uniform = np.full(n_genes, 1.0 / n_genes)
    out = kept
    for well_column in range(counts.shape[1]):
        pool = int(lost[:, well_column].sum())
        if pool == 0:
            continue
        out[:, well_column] += generator.multinomial(pool, uniform)
    return out


def drop_low_cell_wells(
    screen: pd.DataFrame,
    min_cells_per_well: int,
    *,
    drop: bool = True,
) -> pd.DataFrame:
    """Remove wells whose imaged cell count is too low to be informative.

    The second thing neither spaCRPower nor the port did. A well where the
    microscope found three cells produces a positive *fraction* that can only
    take the values 0, 1/3, 2/3 and 1; its binomial standard error is around
    0.27, which is three times the entire gap between the classifier's
    hit-cell and background rates. It carries essentially no information about
    which genotypes were in it — and it enters the fit as one more observation
    alongside a well with four hundred cells.

    The Poisson offset does part of the job: a well with three cells has a
    small expected count, so it does not dominate the *scale* of the fit. What
    the offset does not do is stop the well's read-fraction covariate from
    being paired with a response that is almost pure noise, and a screen with
    a long tail of thin wells is a screen whose covariate-response
    relationship is being averaged against nothing.

    Dropping them is what an analyst does by hand, and it costs something:
    fewer wells is less power. Simulating both sides of that trade is the
    reason this is a parameter rather than a fixed rule.

    Whole wells go, never single ``(gene, well)`` rows. A partially dropped
    well would leave the well's positive total and its cell total describing
    different sets of genes, which is a table the model half is entitled to
    assume cannot exist.

    :param screen: joined screen table; needs ``well`` and one of
        ``imaging_n_cells_per_well`` / ``imaging_n_cells_per_gene_per_well``.
    :param min_cells_per_well: wells with **fewer** imaged cells than this are
        removed. ``0`` disables the filter and returns the input unchanged.
    :param drop: ``False`` annotates with ``well_kept`` and removes nothing,
        for a caller that wants to see what would go.
    :returns: a new frame carrying a boolean ``well_kept`` column, filtered
        when ``drop``. ``frame.attrs['n_wells_dropped']`` and
        ``['n_wells_before']`` record the cost.
    :raises MalformedPlateError: If neither cell-count column is present, or a
        well's ``imaging_n_cells_per_well`` is not constant within the well.
    :raises ScreenDesignError: If ``min_cells_per_well`` is negative.
    """
    threshold = _check_count('min_cells_per_well', min_cells_per_well)
    frame = screen.copy()
    if 'well' not in frame.columns:
        raise MalformedPlateError(
            "drop_low_cell_wells needs a 'well' column; got "
            f"{list(frame.columns)}"
        )
    if 'imaging_n_cells_per_well' in frame.columns:
        per_well = frame.groupby('well')['imaging_n_cells_per_well']
        if int(per_well.nunique().max() or 1) > 1:
            raise MalformedPlateError(
                'imaging_n_cells_per_well varies within a well; it is the '
                'well total repeated on every row and cannot differ between '
                'the genes of one well.'
            )
        totals = per_well.first()
    elif 'imaging_n_cells_per_gene_per_well' in frame.columns:
        totals = frame.groupby('well')['imaging_n_cells_per_gene_per_well'].sum()
    else:
        raise MalformedPlateError(
            'drop_low_cell_wells needs imaging_n_cells_per_well or '
            'imaging_n_cells_per_gene_per_well to know how thin a well is.'
        )
    keep = totals >= threshold
    frame['well_kept'] = frame['well'].map(keep).astype(bool)
    n_before = int(len(keep))
    n_dropped = int((~keep).sum())
    if drop and n_dropped:
        frame = frame.loc[frame['well_kept']].reset_index(drop=True)
    frame.attrs.update(screen.attrs)
    frame.attrs['n_wells_before'] = n_before
    frame.attrs['n_wells_dropped'] = n_dropped
    frame.attrs['min_cells_per_well'] = threshold
    return frame


# ---------------------------------------------------------------------------
# Stage 4 — the sequencing plate
# ---------------------------------------------------------------------------

def simulate_sequencing_plate(
    spot_plate: pd.DataFrame,
    sequencing_n_cells_per_well_lambda: float,
    pcr_factor_mu: float,
    pcr_factor_var: float,
    n_reads_per_well: float,
    *,
    sequencing_n_cells_per_well_var: Optional[float] = None,
    read_depth_cv: float = 0.0,
    sequencing_error_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate the barcode read counts for each genotype in each well.

    Four steps per well, the last of them optional:

    1. Cells contributing DNA: ``gene_in_well * Count(lambda, var)``. This is
       normally far larger than the *imaged* count, because sequencing sees the
       whole well and the microscope sees a few fields.
    2. Amplification: one lognormal PCR factor per well,
       ``pcr_factor ~ LogNormal(meanlog=pcr_factor_mu, sdlog=sqrt(pcr_factor_var))``.
       Note both parameters are on the **log** scale despite their names; a
       ``pcr_factor_mu`` of 2.0 is a median amplification of ``exp(2) ~ 7.4``.
       The factor is per well, not per gene, because a well is one PCR reaction —
       which is exactly why read counts within a well are not independent.
    3. Sequencing: reads are drawn from the amplified barcode pool **without
       replacement** (multivariate hypergeometric), because a finite flow cell
       reading a finite library is sampling without replacement, and modelling it
       as multinomial overstates how much independent information deep wells
       carry.
    4. **Mis-assignment**, when ``sequencing_error_rate > 0``: a fraction of
       the reads is credited to the wrong gene. Off by default, because it is
       not in the R and a silently different baseline would make every number
       already quoted from this module wrong. See :func:`misassign_reads` for
       what it models and why it always costs power.

    **Two departures from the R original, both deliberate.** Upstream computes
    ``n_reads_per_well = round(n_reads_total / nrow(well_data))`` inside a
    ``group_by(well)``, so ``nrow`` is the *library size*: with 452 genes and
    ``n_reads_total = 128318`` that is 284 reads per well, against a real screen
    near 3e4. The parameter is documented as "total reads in the screen" in one
    vignette and "geometric mean of well reads" in another. This function takes
    ``n_reads_per_well`` unambiguously and reports ``n_reads_total`` as the
    derived sum. Second, upstream's guard
    ``k = min(n_cells_in_well * pcr_factor, n_reads_per_well)`` is computed from
    ``round(sum(cells) * pcr)`` while the urn is ``round(cells * pcr)``
    element-wise; the two differ by rounding and the draw can ask for more balls
    than the urn holds. The guard here is computed from the urn itself.

    :param spot_plate: Result of :func:`simulate_spot_plate`; needs ``gene``,
        ``well`` and ``gene_in_well``.
    :param sequencing_n_cells_per_well_lambda: Mean cells per gene per well
        contributing DNA, must be > 0.
    :param pcr_factor_mu: Log-scale mean of the per-well amplification factor.
    :param pcr_factor_var: Log-scale variance of it, must be >= 0.
    :param n_reads_per_well: Target reads per well, must be >= 0.
    :param sequencing_n_cells_per_well_var: Variance of the per-gene cell count;
        ``None`` means Poisson, and the emitted column then echoes the lambda,
        because that is the variance the model used.
    :param read_depth_cv: Coefficient of variation of read depth between wells.
        ``0.0`` gives every well exactly ``n_reads_per_well``; real screens are
        far from uniform, and shallow wells are where hits go to die.
    :param sequencing_error_rate: probability a read is credited to the wrong
        gene, in ``[0, 1]``. ``0.0`` (the default) is the R behaviour;
        :data:`DEFAULT_SEQUENCING_ERROR_RATE` is a realistic figure.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: DataFrame with one row per ``(gene, well)`` and columns
        ``[gene, well, sequencing_n_cells_per_well_lambda,
        sequencing_n_cells_per_well_var, n_reads_per_well, n_reads_total,
        pcr_factor, sequencing_n_cells_per_gene_per_well,
        n_barcodes_per_genes_per_well, sequencing_error_rate,
        n_reads_true_per_gene_per_well, n_reads_per_gene_per_well]``.
        ``n_barcodes_per_genes_per_well`` keeps the R package's misspelling so
        the two are joinable by name.

        ``n_reads_per_gene_per_well`` is what the analyst sees, i.e. **after**
        mis-assignment, because that is what the model half must consume for
        the dilution to be in the answer.
        ``n_reads_true_per_gene_per_well`` is the same quantity before it, so
        a test can plant the truth and measure how far the observation moved.
        With the error rate at zero the two are identical.
    :raises ScreenDesignError: If a parameter is out of range, or neither/both of
        ``rng`` and ``seed`` given.
    :raises SequencingScaleError: If a well's amplified barcode pool exceeds
        :data:`MAX_HYPERGEOMETRIC_URN`, above which numpy's exact sampler loses
        precision. Almost always means ``pcr_factor_mu`` was supplied on the
        linear scale.
    :raises MalformedPlateError: If ``spot_plate`` is not one row per
        ``(gene, well)`` or is missing a required column.

    :example:

    Reads never exceed the requested depth, and a gene absent from a well never
    gets a read:

    >>> lib = simulate_library(30, 20.0, 0.1, seed=11)
    >>> spot = simulate_spot_plate(lib, 12, 3.0, 0.1, seed=12)
    >>> seq = simulate_sequencing_plate(spot, 500.0, 1.0, 0.2, 5000, seed=13)
    >>> per_well = seq.groupby('well')['n_reads_per_gene_per_well'].sum()
    >>> bool((per_well <= 5000).all())
    True
    """
    generator = resolve_rng(rng, seed)
    sequencing_n_cells_per_well_lambda = _check_positive(
        'sequencing_n_cells_per_well_lambda', sequencing_n_cells_per_well_lambda
    )
    pcr_factor_mu = float(pcr_factor_mu)
    pcr_factor_var = float(pcr_factor_var)
    if not np.isfinite(pcr_factor_mu):
        raise ScreenDesignError(f'pcr_factor_mu must be finite, got {pcr_factor_mu!r}')
    if not np.isfinite(pcr_factor_var) or pcr_factor_var < 0.0:
        raise ScreenDesignError(
            f'pcr_factor_var is the variance of log(pcr_factor) and must be '
            f'finite and >= 0, got {pcr_factor_var!r}'
        )
    n_reads_per_well = float(n_reads_per_well)
    if not np.isfinite(n_reads_per_well) or n_reads_per_well < 0.0:
        raise ScreenDesignError(
            f'n_reads_per_well must be finite and >= 0, got {n_reads_per_well!r}'
        )
    read_depth_cv = float(read_depth_cv)
    if not np.isfinite(read_depth_cv) or read_depth_cv < 0.0:
        raise ScreenDesignError(
            f'read_depth_cv must be finite and >= 0, got {read_depth_cv!r}'
        )

    _require_columns(spot_plate, ('gene', 'well', 'gene_in_well'), 'spot_plate')
    genes, wells, position = _plate_grid(spot_plate)
    shape = (genes.size, wells.size)
    in_well = _as_matrix(
        spot_plate['gene_in_well'].to_numpy(dtype=np.int64), position, shape
    )

    cells = in_well * sample_count_mean_variance(
        genes.size * wells.size,
        mean=sequencing_n_cells_per_well_lambda,
        var=sequencing_n_cells_per_well_var,
        rng=generator,
    ).reshape(shape).astype(np.int64)

    pcr_factor = generator.lognormal(
        mean=pcr_factor_mu, sigma=np.sqrt(pcr_factor_var), size=wells.size
    )

    # Per-well sequencing depth. cv == 0 pins every well to the target exactly,
    # which is what makes a "no depth variation" baseline reproducible against
    # an arbitrary generator state.
    if read_depth_cv == 0.0 or n_reads_per_well == 0.0:
        well_depth = np.full(wells.size, n_reads_per_well, dtype=float)
    else:
        well_depth = rgamma_mean_variance(
            wells.size,
            mean=n_reads_per_well,
            var=(read_depth_cv * n_reads_per_well) ** 2,
            rng=generator,
        )
    well_depth = np.rint(well_depth).astype(np.int64)

    barcodes = np.rint(cells * pcr_factor[np.newaxis, :]).astype(np.int64)
    reads = np.zeros(shape, dtype=np.int64)
    for well_column in range(wells.size):
        colors = barcodes[:, well_column]
        urn = int(colors.sum())
        if urn == 0:
            continue
        if urn >= MAX_HYPERGEOMETRIC_URN:
            raise SequencingScaleError(
                f'well {wells[well_column]} has an amplified barcode pool of '
                f'{urn} molecules, at or above numpy\'s exact-sampling ceiling '
                f'of {MAX_HYPERGEOMETRIC_URN}. pcr_factor_mu and pcr_factor_var '
                f'are LOG-scale parameters: pcr_factor_mu={pcr_factor_mu!r} '
                f'means a median amplification of {np.exp(pcr_factor_mu):.3g}x. '
                f'Reduce it, or reduce '
                f'sequencing_n_cells_per_well_lambda='
                f'{sequencing_n_cells_per_well_lambda!r}.'
            )
        # min() against the urn, not against round(sum(cells) * pcr): the two
        # differ because the urn is rounded element-wise, and asking for more
        # balls than the urn holds is a hard numpy error.
        draw = int(min(urn, well_depth[well_column]))
        if draw <= 0:
            continue
        reads[:, well_column] = generator.multivariate_hypergeometric(colors, draw)

    frame = _expand_grid_frame(genes, wells)
    frame['sequencing_n_cells_per_well_lambda'] = sequencing_n_cells_per_well_lambda
    # As in the imaging stage: echo the variance the count model used, so the
    # column is never NaN. `None` means Poisson, whose variance is its mean.
    frame['sequencing_n_cells_per_well_var'] = (
        sequencing_n_cells_per_well_lambda
        if sequencing_n_cells_per_well_var is None
        else float(sequencing_n_cells_per_well_var)
    )
    frame['n_reads_per_well'] = np.tile(well_depth, genes.size)
    frame['n_reads_total'] = np.int64(well_depth.sum())
    frame['pcr_factor'] = np.tile(pcr_factor, genes.size)
    observed = misassign_reads(reads, sequencing_error_rate, rng=generator)

    frame['sequencing_n_cells_per_gene_per_well'] = cells.reshape(-1)
    frame['n_barcodes_per_genes_per_well'] = barcodes.reshape(-1)
    frame['sequencing_error_rate'] = float(sequencing_error_rate)
    frame['n_reads_true_per_gene_per_well'] = reads.reshape(-1)
    frame['n_reads_per_gene_per_well'] = observed.reshape(-1)
    return frame


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def simulate_screen(
    n_genes_in_library: int,
    gene_abundance_alpha: float,
    gene_hit_rate: float,
    n_wells_per_screen: int,
    well_abundance_factor_mu: float,
    well_abundance_factor_var: float,
    imaging_n_cells_per_well_mu: float,
    imaging_n_cells_per_well_var: Optional[float],
    class_pos_mu: float,
    class_pos_var: float,
    class_neg_mu: float,
    class_neg_var: float,
    sequencing_n_cells_per_well_lambda: float,
    pcr_factor_mu: float,
    pcr_factor_var: float,
    n_reads_per_well: float,
    *,
    sequencing_n_cells_per_well_var: Optional[float] = None,
    read_depth_cv: float = 0.0,
    sequencing_error_rate: float = 0.0,
    min_cells_per_well: int = 0,
    imaging_split: str = 'abundance',
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run all four stages and return one tidy ``(well, gene)`` table.

    The imaging and sequencing plates are both simulated from the *same* spot
    plate — which genotypes are in a well is one physical fact, observed twice.
    Each stage draws from its own spawned child stream, so changing the number of
    imaging cells does not shift the sequencing draws; that independence is what
    makes a parameter sweep interpretable, since otherwise every point on the
    sweep would differ by an unrelated re-randomisation as well as by the
    parameter.

    :param n_genes_in_library: See :func:`simulate_library`.
    :param gene_abundance_alpha: See :func:`simulate_library`.
    :param gene_hit_rate: See :func:`simulate_library`.
    :param n_wells_per_screen: See :func:`simulate_spot_plate`.
    :param well_abundance_factor_mu: See :func:`simulate_spot_plate`.
    :param well_abundance_factor_var: See :func:`simulate_spot_plate`.
    :param imaging_n_cells_per_well_mu: See :func:`simulate_imaging_plate`.
    :param imaging_n_cells_per_well_var: See :func:`simulate_imaging_plate`.
    :param class_pos_mu: See :func:`simulate_imaging_plate`.
    :param class_pos_var: See :func:`simulate_imaging_plate`.
    :param class_neg_mu: See :func:`simulate_imaging_plate`.
    :param class_neg_var: See :func:`simulate_imaging_plate`.
    :param sequencing_n_cells_per_well_lambda: See
        :func:`simulate_sequencing_plate`.
    :param pcr_factor_mu: See :func:`simulate_sequencing_plate`.
    :param pcr_factor_var: See :func:`simulate_sequencing_plate`.
    :param n_reads_per_well: See :func:`simulate_sequencing_plate`.
    :param sequencing_n_cells_per_well_var: See :func:`simulate_sequencing_plate`.
    :param read_depth_cv: See :func:`simulate_sequencing_plate`.
    :param sequencing_error_rate: See :func:`misassign_reads`. Off by default;
        turning it on always lowers power, because a mis-assigned read moves
        the covariate toward uniform and attenuates the coefficient.
    :param min_cells_per_well: See :func:`drop_low_cell_wells`. Off by
        default. Turning it on trades wells for well quality, and which way
        that comes out depends on how long the thin tail is -- which is the
        thing worth simulating rather than arguing about.
    :param imaging_split: See :func:`simulate_imaging_plate`.
    :param rng: Generator to draw from; mutually exclusive with ``seed``.
    :param seed: Seed for a fresh generator; mutually exclusive with ``rng``.
    :returns: DataFrame with one row per ``(gene, well)``, carrying every column
        of all four stages: the ground truth ``hit`` and ``gene_abundance``, the
        observed ``positive`` and ``imaging_n_cells_per_well``, and the observed
        ``n_reads_per_gene_per_well``. This is the frame the model half consumes.
    :raises ScreenDesignError: Propagated from any stage; see each.
    :raises ImpossibleMomentsError: Propagated from any stage; see each.

    :example:

    >>> screen = simulate_screen(
    ...     n_genes_in_library=40, gene_abundance_alpha=20.0, gene_hit_rate=0.1,
    ...     n_wells_per_screen=24, well_abundance_factor_mu=4.0,
    ...     well_abundance_factor_var=0.5,
    ...     imaging_n_cells_per_well_mu=120.0, imaging_n_cells_per_well_var=8000.0,
    ...     class_pos_mu=0.8, class_pos_var=0.01,
    ...     class_neg_mu=0.12, class_neg_var=0.005,
    ...     sequencing_n_cells_per_well_lambda=1000.0,
    ...     pcr_factor_mu=2.0, pcr_factor_var=1.0, n_reads_per_well=30000,
    ...     seed=14)
    >>> len(screen)
    960
    >>> bool(screen['positive'].sum() > 0)
    True
    """
    generator = resolve_rng(rng, seed)
    library_rng, spot_rng, imaging_rng, sequencing_rng = _spawn(generator, 4)

    gene_library = simulate_library(
        n_genes_in_library=n_genes_in_library,
        gene_abundance_alpha=gene_abundance_alpha,
        gene_hit_rate=gene_hit_rate,
        rng=library_rng,
    )
    spot_plate = simulate_spot_plate(
        gene_library,
        n_wells_per_screen=n_wells_per_screen,
        well_abundance_factor_mu=well_abundance_factor_mu,
        well_abundance_factor_var=well_abundance_factor_var,
        rng=spot_rng,
    )
    imaging_plate = simulate_imaging_plate(
        spot_plate,
        imaging_n_cells_per_well_mu=imaging_n_cells_per_well_mu,
        imaging_n_cells_per_well_var=imaging_n_cells_per_well_var,
        class_pos_mu=class_pos_mu,
        class_pos_var=class_pos_var,
        class_neg_mu=class_neg_mu,
        class_neg_var=class_neg_var,
        imaging_split=imaging_split,
        rng=imaging_rng,
    )
    sequencing_plate = simulate_sequencing_plate(
        spot_plate,
        sequencing_n_cells_per_well_lambda=sequencing_n_cells_per_well_lambda,
        pcr_factor_mu=pcr_factor_mu,
        pcr_factor_var=pcr_factor_var,
        n_reads_per_well=n_reads_per_well,
        sequencing_n_cells_per_well_var=sequencing_n_cells_per_well_var,
        read_depth_cv=read_depth_cv,
        sequencing_error_rate=sequencing_error_rate,
        rng=sequencing_rng,
    )

    # validate='1:1' rather than a positional concat: dplyr::do() returns groups
    # in sorted key order, not input order, so any positional assumption carried
    # over from the R code would misalign genes without changing a single
    # summary statistic.
    screen = spot_plate.merge(
        imaging_plate, on=['well', 'gene'], how='left', validate='1:1'
    ).merge(
        sequencing_plate, on=['well', 'gene'], how='left', validate='1:1'
    )
    screen.attrs['n_prob_clipped'] = spot_plate.attrs.get('n_prob_clipped', 0)
    # Last, and after the join, because it is a decision about *wells* taken
    # on the realised imaged cell total -- which is only known once the
    # imaging plate exists, and which the sequencing plate knows nothing
    # about. A well removed here takes its sequencing rows with it.
    screen = drop_low_cell_wells(screen, min_cells_per_well)
    return screen
