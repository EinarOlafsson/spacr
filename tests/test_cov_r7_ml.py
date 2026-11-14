"""Round-7 coverage for the last cold corners of :mod:`spacr.ml`.

Round 6 closed most of this tail and PROVED the rest -- the alias table a
canonicaliser has already emptied, the fold count a split has already refused,
the ``if folder:`` on a path that is always rooted. What is left after it is
two things, and they are opposite in kind:

* the absorbing backend's two singularity guards, which are a REAL refusal
  followed by a defensive one. The first is driven here on a design that has
  the same regressor twice; the second is proved dead by the first.

* ``_perform_regression``'s QC block, which round 6 could not reach because
  the manifest died in ``DataFrame.merge`` -- fixed since. It is reachable
  now, but only on a SINGLE-LEVEL run: with both levels fitted the two
  manifests differ and ``pd.concat`` drops ``attrs`` rather than choosing
  between them. Both facts are pinned below, the second because a caller
  asking for `level='both'` still gets no verdict and that is worth a red
  line the day it changes.

Everything here is measured on a real run, not asserted about one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# spacr.ml reaches into spacr.utils lazily; warming it here keeps the one-off
# numba/umap import off whichever test happens to run first.
import spacr.utils  # noqa: E402,F401
from spacr import ml  # noqa: E402

from tests.test_cov_12_ml_backends import absorbable_design  # noqa: E402
from tests.test_cov_ml_perform_regression import (  # noqa: E402
    _score_records, parametric_settings, write_counts, write_metadata)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the absorbed fit: one real singularity guard and one that follows it
# ---------------------------------------------------------------------------

def case_a_design_holding_the_same_regressor_twice_is_refused_by_name():
    """A rank-deficient absorbed design is refused BEFORE the solve.

    ``fraction`` and a copy of it are collinear after the row/column factors
    are projected out, so the demeaned cross-product matrix is exactly
    singular. The refusal has to name the DESIGN rather than the backend:
    statsmodels answers the very same design with a pseudo-inverse, i.e.
    with one arbitrary member of an infinite solution set, and a user who
    reads "pyfixest failed" would switch backends and get a number instead
    of a diagnosis.

    IT NO LONGER WAITS FOR THE SOLVER TO COMPLAIN. This test used to assert
    the ValueError was raised FROM a LinAlgError, which made the refusal
    depend on LAPACK: builds disagree, and on some the solve returns one
    arbitrary member of the solution set rather than raising, so the design
    was accepted and a coefficient table came back that looked fine and was
    not identified. ``matrix_rank`` decides first now, and the rank it found
    is in the message -- so there is no cause to chain, and the refusal is
    the same on every runner.

    Driven beside the identical design without the duplicate, which fits.
    """
    _seen = []
    pytest.importorskip('pyfixest.core.demean')

    healthy = absorbable_design(extra=('fraction',))
    y = np.arange(float(len(healthy)))
    fitted = ml._fit_absorbed_least_squares(healthy, y)
    _seen.append(keep(list(fitted.params.index)))
    _seen.append(bool(
        np.isfinite(fitted.bse['fraction'])))

    singular = absorbable_design(extra=('fraction',))
    singular['fraction_copy'] = singular['fraction']
    _seen.append(bool(
        singular['fraction_copy'].equals(singular['fraction'])))

    with pytest.raises(ValueError) as caught:
        ml._fit_absorbed_least_squares(singular, y)
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))

    message = str(caught.value)
    _seen.append(('normal equations are singular' if 'normal equations are singular' in (message) else ("missing", 'normal equations are singular')))
    _seen.append(('its 2 coefficients are not identified' if 'its 2 coefficients are not identified' in (message) else ("missing", 'its 2 coefficients are not identified')))
    _seen.append(('pseudo-inverse' if 'pseudo-inverse' in (message) else ("missing", 'pseudo-inverse')))
    _seen.append(('rank 1 of 2' if 'rank 1 of 2' in (message) else ("missing", 'rank 1 of 2')))
    _seen.append(bool(
        caught.value.__cause__ is None))
    return _seen


def case_the_covariance_inverse_cannot_fail_where_the_solve_succeeded():
    """``np.linalg.inv(xtx)`` at ml.py:3071 is guarded against nothing.

    ``_fit_absorbed_least_squares`` calls ``np.linalg.solve(xtx, xty)`` two
    statements above it (ml.py:3054) and inverts the SAME ``xtx``. Both go
    through the same LU factorisation and both raise
    :class:`numpy.linalg.LinAlgError` for exactly the matrices whose
    factorisation has an exact zero pivot -- so the second guard can only fire
    for a matrix the first has already refused, and its ``raise`` is dead.

    Driven, not asserted: the singular cross-product matrix that the test
    above stops on is rebuilt here by hand, and BOTH calls are shown to refuse
    it, while a full-rank one is shown to satisfy both.
    """
    _seen = []
    singular = np.array([[135.0, 135.0], [135.0, 135.0]])
    rhs = np.array([270.0, 270.0])
    with pytest.raises(np.linalg.LinAlgError) as _exc1:
        np.linalg.solve(singular, rhs)
    _seen.append(('LinAlgError' if isinstance(_exc1.value, np.linalg.LinAlgError) else type(_exc1.value).__name__))
    with pytest.raises(np.linalg.LinAlgError) as _exc2:
        np.linalg.inv(singular)
    _seen.append(('LinAlgError' if isinstance(_exc2.value, np.linalg.LinAlgError) else type(_exc2.value).__name__))

    full_rank = np.array([[135.0, 12.0], [12.0, 90.0]])
    beta = np.linalg.solve(full_rank, rhs)
    inverse = np.linalg.inv(full_rank)
    _seen.append(bool(
        np.allclose(inverse @ rhs, beta)))
    _seen.append(bool(
        np.allclose(full_rank @ inverse, np.eye(2))))
    return _seen


# ---------------------------------------------------------------------------
# the QC verdict, out of a real run
# ---------------------------------------------------------------------------

def _one_plate_screen(tmp_path):
    """One plate's scores, counts and metadata, as the fixtures write them."""
    scores = tmp_path / "scores"
    counts = tmp_path / "counts"
    scores.mkdir()
    counts.mkdir()
    score_path = scores / "plate1.csv"
    pd.DataFrame(_score_records("plate1", 3)).to_csv(score_path, index=False)
    count_path = write_counts(counts / "plate1.csv", plate="plate1", seed=1)
    meta = write_metadata(tmp_path / "TGME49_Summary.csv")
    return {"root": tmp_path, "score": str(score_path), "count": count_path,
            "meta": meta}


def case_a_single_level_run_hands_its_qc_verdict_back_to_the_caller(tmp_path):
    """Instruction 115: ``output`` says whether the fit it holds is diagnosable.

    ``regression`` puts the QC manifest on ``coef_df.attrs`` and
    ``_perform_regression`` lifts three things off it: the manifest itself,
    the worst panel's verdict, and that verdict's LEVEL -- the level because a
    caller decides what to show from it, and the verdict because it says why.

    Measured on a real guide-level fit, so the verdict is the suite's own
    reading of these residuals and not a value the test put there.
    """
    _seen = []
    settings = parametric_settings(_one_plate_screen(tmp_path), level='grna')
    _seen.append(bool(
        settings.get('regression_qc', True)))

    output = ml.perform_regression(settings)

    _seen.append(('qc' if 'qc' in (output) else ("missing", 'qc')))
    manifest = output['qc']
    _seen.append(bool(
        manifest['verdict_level'] in {'ok', 'check', 'warn', 'fail'}))
    _seen.append(bool(
        output['qc_verdict_level'] == manifest['verdict_level']))
    _seen.append(bool(
        output['qc_verdict'] is manifest['verdict']))
    _seen.append(bool(
        output['qc_verdict'].level == output['qc_verdict_level']))
    _seen.append(bool(
        output['qc_verdict'].headline))
    # the manifest is the one the run wrote, not a summary of it
    _seen.append(bool(
        manifest['panels']))
    import os
    _seen.append(bool(
        os.path.isfile(manifest['report'])))
    return _seen


def case_a_manifest_that_reached_no_verdict_still_reports_a_level(tmp_path,
                                                                  monkeypatch):
    """``verdict`` absent means ``qc_verdict`` absent and the level 'unknown'.

    ``regression_qc_report`` sets ``manifest['verdict']`` to the worst panel
    verdict, and to None when no panel produced one at all (every panel
    skipped or failed). A key holding None is indistinguishable from a suite
    that ran and concluded nothing, so the verdict is left OFF the output
    while ``qc_verdict_level`` still reports 'unknown' -- a caller can always
    ask the level.

    Driven on a real run: the manifest the suite produced is captured first
    and shown to carry a verdict, and the same manifest is then handed on
    without one, so the absence below is this block's doing and not a run that
    produced no QC.
    """
    _seen = []
    captured = {}
    real = ml._write_regression_qc

    def strip_verdict(*args, **kwargs):
        manifest = real(*args, **kwargs)
        captured['verdict'] = manifest.get('verdict')
        captured['level'] = manifest.get('verdict_level')
        manifest.pop('verdict', None)
        manifest.pop('verdict_level', None)
        return manifest

    # patched on the module: `regression` calls it as a module global, and the
    # manifest shape is what is under test, not the panels that build it.
    monkeypatch.setattr(ml, '_write_regression_qc', strip_verdict)

    settings = parametric_settings(_one_plate_screen(tmp_path), level='grna')
    output = ml.perform_regression(settings)

    _seen.append(bool(
        captured['verdict'] is not None))
    _seen.append(bool(
        captured['level'] == captured['verdict'].level))
    _seen.append(('qc' if 'qc' in (output) else ("missing", 'qc')))
    _seen.append((("absent", 'verdict') if 'verdict' not in (output['qc']) else ("present", 'verdict')))
    _seen.append((("absent", 'qc_verdict') if 'qc_verdict' not in (output) else ("present", 'qc_verdict')))
    _seen.append(keep(output['qc_verdict_level']))
    return _seen


def case_both_levels_fitted_means_two_manifests_and_therefore_none(tmp_path):
    """``level='both'`` carries no verdict, and this is the arithmetic of it.

    ``_perform_regression`` stacks the per-level coefficient tables with
    ``pd.concat``, and pandas propagates ``.attrs`` across a concat only when
    every frame carries the SAME attrs. The guide fit and the gene fit each
    write their own QC manifest into their own folder, so the two differ and
    the stack comes back with no attrs at all -- which is why ``output['qc']``
    exists on a one-level run and not on the default one.

    Pinned rather than asserted about: the pandas rule is driven directly on
    three frames (one, two agreeing, two differing) so a pandas that starts
    carrying attrs through a concat of differing frames turns this red instead
    of silently making the one-level restriction disappear.
    """
    _seen = []
    one = pd.DataFrame({'feature': ['a'], 'coefficient': [0.1]})
    one.attrs['qc_manifest'] = {'directory': '/runs/grna'}
    same = pd.DataFrame({'feature': ['b'], 'coefficient': [0.2]})
    same.attrs['qc_manifest'] = {'directory': '/runs/grna'}
    other = pd.DataFrame({'feature': ['c'], 'coefficient': [0.3]})
    other.attrs['qc_manifest'] = {'directory': '/runs/gene'}

    _seen.append(bool(
        pd.concat([one], ignore_index=True).attrs == one.attrs))
    _seen.append(bool(
        pd.concat([one, same], ignore_index=True).attrs == one.attrs))
    _seen.append(keep(pd.concat([one, other], ignore_index=True).attrs))

    # ...and the two levels really do write different directories, which is
    # what makes the third case the one a default run takes.
    settings = parametric_settings(_one_plate_screen(tmp_path), level='both')
    output = ml.perform_regression(settings)

    _seen.append((("absent", 'qc') if 'qc' not in (output) else ("present", 'qc')))
    _seen.append((("absent", 'qc_verdict_level') if 'qc_verdict_level' not in (output) else ("present", 'qc_verdict_level')))
    _seen.append(bool(
        len(output['results']) > 0))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_a_design_holding_the_same_regressor_twice_is_refused_by_name, [['fraction'], True, True, 'ValueError', 'normal equations are singular', 'its 2 coefficients are not identified', 'pseudo-inverse', 'rank 1 of 2', True], id="a_design_holding_the_same_regressor_twice_is_refused_by_name"),
    pytest.param(case_a_single_level_run_hands_its_qc_verdict_back_to_the_caller, [True, 'qc', True, True, True, True, True, True, True], id="a_single_level_run_hands_its_qc_verdict_back_to_the_caller"),
    pytest.param(case_a_manifest_that_reached_no_verdict_still_reports_a_level, [True, True, 'qc', ("absent", 'verdict'), ("absent", 'qc_verdict'), 'unknown'], id="a_manifest_that_reached_no_verdict_still_reports_a_level"),
    pytest.param(case_both_levels_fitted_means_two_manifests_and_therefore_none, [True, True, {}, ("absent", 'qc'), ("absent", 'qc_verdict_level'), True], id="both_levels_fitted_means_two_manifests_and_therefore_none"),
    pytest.param(case_the_covariance_inverse_cannot_fail_where_the_solve_succeeded, ['LinAlgError', 'LinAlgError', True, True], id="the_covariance_inverse_cannot_fail_where_the_solve_succeeded"),
])
def test_r7_ml_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
