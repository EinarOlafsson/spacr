"""Classical machine-learning and regression analysis pipelines."""

import functools
import logging
import os, sys, re
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro
from math import pi

from sklearn.linear_model import (Lasso, Ridge, LassoCV, RidgeCV,
                                  ElasticNet, ElasticNetCV)
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from statsmodels.othermod.betareg import BetaModel
from sklearn.preprocessing import FunctionTransformer
from patsy import dmatrices

from .regression_spec import (DEFAULT_REGRESSION_BACKEND,  # noqa: F401
                              NO_P_VALUE_TYPES,
                              REGRESSION_BACKENDS,
                              REGRESSION_BACKEND_ORDER,
                              REGRESSION_SETTINGS_USED,
                              REGRESSION_TYPES,
                              RUN_LEVEL_SETTINGS,
                              UNSUPPORTED_REGRESSION_TYPES,
                              _MODEL_LEVEL_DEFAULTS,
                              _RUN_LEVEL_DEFAULTS)
from .regression_families import (REGRESSION_FAMILY_ASSUMPTIONS,  # noqa: F401
                                  REGRESSION_FAMILY_GROUPS,
                                  family_group,
                                  family_label,
                                  regression_family_choices)
from .mixed_gpu import MixedBackendUnavailable  # noqa: F401
from .regression_backends import (backend_label,          # noqa: F401
                                  backend_status,
                                  backend_supports,
                                  resolve_backend_name)


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine, euclidean, mahalanobis, cityblock, minkowski, chebyshev, braycurtis
from xgboost import XGBClassifier

from . import frame_handoff, schema, tabular
from .openmp_guard import single_threaded_openmp, guarded_n_jobs  # see spacr/openmp_guard.py — duplicate libomp is fatal
from .plot import save_figure  # every kept figure goes through the format/DPI preference

LOG = logging.getLogger("spacr.ml")

_FLOWVIEW_TRUE_VALUES = frozenset({"1", "on", "true", "yes"})


def _flowview_event(action, *args):
    """Reach optional Classify tracing without importing it when disabled."""

    trace_module = sys.modules.get("spacr.flowview.trace")
    if trace_module is None:
        enabled_by_environment = os.environ.get("SPACR_FLOWVIEW", "")
        if enabled_by_environment.strip().casefold() not in _FLOWVIEW_TRUE_VALUES:
            return False
        try:
            from .flowview import trace as trace_module
        except BaseException:
            return False
    try:
        if not trace_module.is_enabled():
            return False
        from .flowview import _classify_stages

        return bool(getattr(_classify_stages, f"_{action}")(*args))
    except BaseException:
        return False


def _flowview_pipeline(family):
    """Finish or fail the active graph without changing scientific output."""

    def decorate(function):
        @functools.wraps(function)
        def observed(*args, **kwargs):
            settings = args[0] if args else kwargs.get("settings")
            active = _flowview_event("begin", settings, family)
            try:
                result = function(*args, **kwargs)
            except BaseException as scientific_error:
                if active:
                    _flowview_event("fail", scientific_error)
                raise
            if active:
                _flowview_event("finish")
            return result

        return observed

    return decorate


def _flowview_advance(node_id):
    """Record one real operation boundary, or do nothing when disabled."""

    _flowview_event("advance", node_id)


def _flowview_metric(name, value):
    """Record one scalar on the active stage, or do nothing when disabled."""

    _flowview_event("metric", name, value)


from scipy.stats import kstest, normaltest

import matplotlib

# THE HOUSE STYLE (136). `figures.style` imports matplotlib only
# inside its own functions, so naming it here costs nothing at
# import time.
from .figures.style import ROLES, figure_style, theme_target

# Only demote to Agg when there is genuinely nowhere to draw. Doing it
# unconditionally at import time silently killed inline plotting for anyone
# who imported spacr.ml in a notebook, because it overrode a backend the user
# had already selected. spacr.cli and both GUIs set their own backend.
if not (sys.platform.startswith(('win', 'darwin')) or os.environ.get('DISPLAY')):
    matplotlib.use('Agg')

import warnings


def _require_backend(regression_type, regression_backend):
    """Resolve and validate a regression backend for a model family.

    Validate at run time because settings loaded from a file can bypass the
    GUI's disabled backend entries. Never substitute another backend silently.

    :param regression_type: the family being fitted.
    :param regression_backend: name or label, or ``None`` for the default.
    :returns: the canonical backend name.
    :raises ValueError: when the backend cannot fit the family, is not
        installed, or needs a GPU that is not there.
    """
    name = resolve_backend_name(regression_backend)
    if name == DEFAULT_REGRESSION_BACKEND:
        return name
    status = backend_status(name, regression_type)
    if not status['enabled']:
        raise ValueError(
            f"{status['reason']} Set regression_backend='statsmodels' to fit "
            f"it with the default backend, which produced every existing "
            f"result.")
    return name



def _say_what_a_mixed_fit_will_cost(backend, df=None):
    """Describe the expected cost of a statsmodels mixed fit before it starts.

    Print nothing for non-default backends. For statsmodels, include the row
    count when available and state whether the compatible Torch GPU backend
    can be selected instead.
    """
    if backend != DEFAULT_REGRESSION_BACKEND:
        return
    rows = None
    try:
        rows = len(df) if df is not None else None
    except TypeError:                                    # noqa: BLE001
        rows = None
    try:
        status = backend_status('torch', 'mixed')
        available = bool(status.get('enabled'))
        reason = str(status.get('reason') or '')
    except Exception:                                    # noqa: BLE001
        available, reason = False, ''
    size = f" on {rows} wells" if rows else ""
    print(f"Fitting the mixed model{size} with statsmodels. This is the slow "
          f"one: dense linear algebra, measured at 54x OLS on 40 genes and "
          f"rising with screen size, and it prints nothing while it runs.")
    if available:
        print("  The same model, same estimates, is available on the GPU: set "
              "regression_backend='torch'. Measured on this screen: 26 "
              "seconds against >25 minutes.")
    elif reason:
        print(f"  The GPU backend would be faster but is not usable here: "
              f"{reason}")


#: Settings that must lie strictly inside 0 and 1, and what each one is for.
#: Checked BEFORE the run writes anything -- see
#: :func:`_reject_impossible_probabilities`.
_UNIT_INTERVAL_SETTINGS = {
    'fdr_alpha': "the family-level rejection threshold for adjusted P values",
    'p_threshold_alpha': "the cut applied to the P value column",
    'alpha': None,          # 'alpha' is the PENALTY for ridge/lasso, not a
                            # probability, so it is deliberately not checked.
}


def _reject_impossible_probabilities(settings):
    """Validate probability thresholds before the run writes output.

    Check every setting in :data:`_UNIT_INTERVAL_SETTINGS` that represents a
    probability and reject non-numeric values or values outside the open
    interval ``(0, 1)``. The penalty parameter named ``alpha`` is deliberately
    excluded because it is not a probability.
    """
    for key, what in _UNIT_INTERVAL_SETTINGS.items():
        if what is None or key not in settings:
            continue
        value = settings.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{key}={value!r} is not a number. It is {what}, and must "
                f"be strictly between 0 and 1 (usually 0.05).") from None
        if not (0.0 < number < 1.0):
            raise ValueError(
                f"{key}={number!r} is outside 0 and 1. It is {what}, so it "
                f"has no meaning there; the usual value is 0.05. "
                + ("A value one less than what you meant is what a spin box "
                   "does when its arrow or the scroll wheel is nudged -- "
                   f"{number + 1:g} may be the number you set."
                   if -1.0 < number < 0.0 else
                   "Set it in the Significance section, or in "
                   "settings/regression.csv if this run came from a file."))

def _concat_named_csvs(paths):
    """Read every CSV in ``paths`` into one frame.

    A screen's counts and scores are one file per plate, and this question
    is asked of the screen rather than of a plate, so they are read together.
    Each file goes through :func:`spacr.tabular.read_table`, so a header
    spelled ``column_name`` or ``Well`` reaches the fit under the canonical
    key names.

    :param paths: one path, a list of them, or nothing.
    :returns: the concatenated frame.
    :raises ValueError: when there is nothing readable to concatenate.
    """
    import pandas as pd

    from .tabular import read_table

    if not paths:
        raise ValueError("no table was given to read")
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    frames = []
    for one in paths:
        try:
            frames.append(read_table(one))
        except Exception as exc:
            raise ValueError(f"{one} could not be read: {exc}") from exc
    if not frames:
        raise ValueError("no table was given to read")
    return pd.concat(frames, ignore_index=True)


def _well_block_tokens(settings, key):
    """The row/column/well tokens one control-block setting names.

    :param settings: the run's settings mapping.
    :param key: ``'positive_control_wells'`` or its negative twin.
    :returns: the tokens, lower-cased, with the empty ones dropped.

    A list or a bare string, because both spellings reach here: the panel
    writes a list and a settings CSV can carry either.
    """
    raw = (settings or {}).get(key)
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return [str(value).strip().lower() for value in values
            if str(value).strip()]


def _wells_in_block(labels, tokens):
    """Which of ``labels`` the block ``tokens`` name.

    :param labels: well labels as the score table carries them, ``prc`` form
        -- ``plate_row_column``.
    :param tokens: what the plate design calls the block: a column (``c2``),
        a row (``r1``) or a whole well (``plate1_r1_c2``).
    :returns: the matching labels, sorted and de-duplicated.

    THE TOKEN IS MATCHED AGAINST A PART, NOT AS A SUBSTRING. ``'c2' in
    'plate1_r1_c20'`` is true and says nothing, so a plate wider than nine
    columns would fold column 20 into column 2's reference and shift the
    endpoint the whole calibration is anchored on.
    """
    wanted = set(tokens)
    out = set()
    for label in labels:
        text = str(label).strip()
        parts = {part.strip().lower() for part in text.split('_')}
        parts.add(text.lower())
        if parts & wanted:
            out.add(text)
    return sorted(out)


def _calibration_inputs(settings):
    """Gather what the fraction-threshold sweep needs, from the run's own files.

    THE IMAGING SIDE IS THE CLASSIFIER SCORE. `mixed_ratio_calibration` takes
    an ``(n_cells, n_features)`` block and one well label per cell, and asks
    what mixture of positive and negative control each well looks like. The
    per-cell score column is exactly that measurement with one feature, so
    the score table the run already loaded is the imaging side and no second
    source is needed.

    THE PURE WELLS ARE NAMED FROM THE PLATE DESIGN, through
    `positive_control_wells` and `negative_control_wells`. Identifying them by
    their reported fraction would be circular: that fraction is the quantity
    under test, and a bias large enough to matter pushes a pure well the
    wrong side of any cut-off.

    THE WELLS AND THE GUIDE ARE TWO SETTINGS, and reading one for the other is
    what stopped this running at all. `positive_control` is a gene or gRNA ID
    SUBSTRING in a regression -- it defaults to '239740' -- and was being
    matched against well labels, which no well label has ever contained. So
    every screen that ticked the box was refused with "no well matched", and
    the three control-block settings that exist to answer this were never
    read. The guide is what `positive_guide` needs; the wells are what
    `pure_pc_wells` needs.

    :param settings: the regression settings.
    :returns: keyword arguments for
        :func:`spacr.fraction_calibration.sweep_fraction_threshold`.
    :raises ValueError: when the screen cannot answer the question -- no
        control-well block named, no positive-control guide, or no score
        column to read. The caller turns that into a printed reason and the
        threshold the settings gave.
    """
    import numpy as np

    positive_wells = _well_block_tokens(settings, 'positive_control_wells')
    negative_wells = _well_block_tokens(settings, 'negative_control_wells')
    if not positive_wells or not negative_wells:
        raise ValueError(
            "the plate design names no positive_control_wells and "
            "negative_control_wells, and a control-well calibration has "
            "nothing to calibrate against")
    positive_guide = str(settings.get('positive_control') or '').strip()
    if not positive_guide:
        raise ValueError(
            "positive_control names no gRNA, so there is no guide whose "
            "sequenced share can be compared with the imaging")

    counts = _concat_named_csvs(settings.get('count_data'))
    scores = _concat_named_csvs(settings.get('score_data'))
    well_column = str(settings.get('count_well_column') or 'prc')
    score_column = str(settings.get('dependent_variable') or 'pred')
    if score_column not in scores.columns:
        raise ValueError(
            f"the score table has no {score_column!r} column to read the "
            f"imaging side from")
    if well_column not in scores.columns:
        raise ValueError(
            f"the score table has no {well_column!r} column, so a cell "
            f"cannot be placed in a well")

    usable = scores[[well_column, score_column]].dropna()
    features = np.asarray(usable[score_column], dtype=float).reshape(-1, 1)
    wells = [str(w) for w in usable[well_column]]
    pure_pc = _wells_in_block(wells, positive_wells)
    pure_nc = _wells_in_block(wells, negative_wells)
    if not pure_pc or not pure_nc:
        raise ValueError(
            f"no well matched {positive_wells} and {negative_wells}, so "
            f"there is no pure control to anchor the fit")

    return {
        "counts": counts,
        "features": features,
        "wells": wells,
        "positive_guide": positive_guide,
        "pure_pc_wells": pure_pc,
        "pure_nc_wells": pure_nc,
        "normalise": bool(settings.get('normalise_fraction', True)),
        "well_column": well_column,
        "guide_column": str(settings.get('count_grna_column') or 'grna'),
        "count_column": str(settings.get('count_value_column') or 'count'),
    }


def _calibrated_fraction_threshold(settings):
    """The cut-off the control wells imply, or ``None`` if they cannot say.

    Returns ``None`` -- rather than raising -- for every reason the sweep
    might not apply: the plate design names no pure control wells, there
    are too few of them to fit anything, the counts are missing the columns
    it reads, or the optional module is not importable. Each of those is an
    ordinary answer to "can this screen calibrate itself", and none of them
    is a reason to stop a run that already had a usable threshold.

    :param settings: the regression settings, read for the control-well
        names and the count table.
    :returns: the measured threshold, or ``None``.
    """
    try:
        from .fraction_calibration import sweep_fraction_threshold
    except Exception:
        print("fraction-threshold calibration is unavailable; "
              "using the threshold as given")
        return None
    try:
        result = sweep_fraction_threshold(**_calibration_inputs(settings))
    except (KeyError, ValueError, TypeError) as exc:
        # NAMED, NOT SWALLOWED. A user who ticked the box is owed the
        # reason it did nothing, or they will believe it worked.
        print(f"fraction-threshold calibration did not run: {exc}")
        return None
    # `chosen` IS THE KEY THE SWEEP WRITES. It reported `threshold` here,
    # which `sweep_fraction_threshold` has never returned -- so every screen
    # that ticked the box was told the sweep preferred nothing, whatever it
    # had actually measured, and went on using the number the settings gave.
    # `threshold` IS a key, on each row of `candidates`; reading it off the
    # result was reading a per-candidate name at the top level.
    chosen = result.get("chosen") if isinstance(result, dict) else None
    if chosen is None:
        print("fraction-threshold calibration found no cut-off it preferred; "
              "using the threshold as given")
        return None
    try:
        from .fraction_calibration import describe
        print(describe(result))
    except Exception:
        print(f"fraction_threshold calibrated to {chosen}")
    return float(chosen)


def _graph_sequencing_stats(settings):
    """Resolve the sequencing threshold helper through one testable seam."""
    # Keep this lazy to avoid expanding ml.py's already-heavy import graph,
    # while giving callers and tests a stable dependency boundary. Importing
    # the helper directly inside perform_regression made it impossible to
    # substitute reliably after package lazy-loader tests replaced a module
    # object in sys.modules.
    from .sequencing import graph_sequencing_stats
    return graph_sequencing_stats(settings)


#: File types the run treats as a figure when it collects what a helper drew.
_FIGURE_SUFFIXES = ('.pdf', '.png', '.svg', '.jpg', '.jpeg', '.tif', '.tiff',
                    '.eps')


def _screen_figure_folders(settings):
    """Where a sequencing helper drops its figures: beside the COUNT DATA.

    `graph_sequencing_stats` writes ``<count folder>/results/`` for the
    threshold sweep and ``<count folder>/`` for the unique-count plate
    heatmap, both derived from ``settings['count_data'][0]`` inside
    :mod:`spacr.sequencing`. Neither is the run's own folder, which is the
    whole problem this list exists to solve.
    """
    folders = []
    for path in (settings.get('count_data') or []):
        base = os.path.dirname(str(path))
        for candidate in (base, os.path.join(base, 'results')):
            if candidate and candidate not in folders:
                folders.append(candidate)
    return folders


def _figure_stamps(folders):
    """``{path: (mtime, size)}`` for every figure directly inside ``folders``.

    Not recursive, and not a bare listing: a run of the same screen writes
    the same file NAMES, so identity has to include the stamp or a figure
    left by yesterday's run reads as one this run drew.
    """
    stamps = {}
    for folder in folders:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if not entry.name.lower().endswith(_FIGURE_SUFFIXES):
                continue
            try:
                if not entry.is_file():
                    continue
                info = entry.stat()
            except OSError:
                continue
            stamps[entry.path] = (info.st_mtime_ns, info.st_size)
    return stamps


def _keep_figures_with_the_run(before, folders, destination):
    """Copy newly written figures into the run-specific output folder.

    Compare current file stamps with ``before`` and copy only new or changed
    figures. Retain the originals because other workflows may reference the
    screen-level folder.

    :param before: the stamps from :func:`_figure_stamps` taken first.
    :param folders: the same folders it was taken over.
    :param destination: the run folder.
    :returns: the paths written, so the caller can name them.
    """
    import shutil

    kept = []
    for path, stamp in sorted(_figure_stamps(folders).items()):
        if before.get(path) == stamp:
            continue
        target = os.path.join(destination, os.path.basename(path))
        if os.path.abspath(target) == os.path.abspath(path):
            continue
        try:
            os.makedirs(destination, exist_ok=True)
            shutil.copy2(path, target)
        except OSError as error:
            # Advisory. A figure that could not be copied must not cost the
            # run the threshold it just computed.
            print(f"Could not keep {os.path.basename(path)} with the run: "
                  f"{error}")
            continue
        kept.append(target)
    return kept


def _run_random_state(default=None):
    """Return the active run's seed, for an estimator's ``random_state=``.

    Imported inside the call rather than at module scope: :mod:`spacr.runctx`
    reaches :mod:`spacr.settings`, which reaches back here, and a top-level
    import would be a cycle. Outside a run this is whatever ``default`` was,
    which is the literal these call sites used to hard-code.

    :param default: the value to use when no run is open.
    :returns: the run seed, or ``default``.
    """
    from .runctx import random_state
    return random_state(default)


warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")


class _DispersedVariance:
    """Scale a statsmodels variance function by a constant dispersion factor.

    ``Binomial.__init__`` stores a ``varfuncs`` callable in the *instance*
    ``__dict__`` under the name ``variance``, and an instance attribute
    always wins over a subclass method of the same name. Overriding
    ``variance`` in a subclass therefore has no effect on anything
    statsmodels does. Wrapping the stored callable is the only way to make
    the factor reach the fit, and delegating attribute lookups keeps
    ``family.variance.deriv`` — which ``GLM`` calls — working.

    :param varfunc: The variance callable installed by statsmodels.
    :param dispersion: Multiplicative variance scaling.
    """

    def __init__(self, varfunc, dispersion):
        self._varfunc = varfunc
        self.dispersion = dispersion

    def __call__(self, mu):
        """Return ``dispersion * varfunc(mu)``."""
        return self.dispersion * self._varfunc(mu)

    def deriv(self, mu):
        """Return the dispersion-scaled derivative of the variance function."""
        return self.dispersion * self._varfunc.deriv(mu)

    def __getattr__(self, name):
        """Delegate every other attribute to the wrapped variance function.

        Raises ``AttributeError`` - never ``KeyError`` - when ``_varfunc`` is
        not set yet, so ``copy``/``pickle`` can probe for ``__setstate__`` and
        friends on a half-built instance without blowing up.
        """
        try:
            varfunc = self.__dict__['_varfunc']
        except KeyError:
            raise AttributeError(name) from None
        return getattr(varfunc, name)


class QuasiBinomial(Binomial):
    """Binomial GLM family scaled by a dispersion parameter (quasi-binomial).

    :param link: statsmodels link instance. Default ``Logit()``.
    :param dispersion: Multiplicative variance scaling. Default ``1.0``.
    """

    def __init__(self, link=Logit(), dispersion=1.0):
        """Store the dispersion factor after delegating to ``Binomial``."""
        super().__init__(link=link)
        self.dispersion = dispersion
        # See _DispersedVariance: without this the method below is shadowed
        # by the instance attribute statsmodels just installed, so the
        # dispersion was silently ignored by every fit using this family.
        self.variance = _DispersedVariance(self.__dict__['variance'], dispersion)

    def variance(self, mu):
        """Adjust the variance with the dispersion parameter."""
        return self.dispersion * super().variance(mu)

def calculate_p_values(X, y, model):
    """Return OLS-style p-values for a fitted model's coefficients.

    **These are not valid frequentist p-values for a penalised fit**, and the
    two callers that reach them know it in different ways. The standard error
    is the unpenalised ``rse * sqrt(diag((X'X)^-1))`` while the coefficient it
    is divided into has been shrunk, so the test is mis-specified. The
    direction of the error is the one that matters here and it is the safe one:
    the penalty shrinks the numerator and inflates the residual in the
    denominator, so the statistic is too SMALL and the p-value too large. A
    penalised fit under-detects here; it does not manufacture hits.

    ``lasso`` and ``elasticnet`` do not rely on this at all —
    :data:`NO_P_VALUE_TYPES` routes them to a bootstrap selection frequency
    instead. ``ridge`` does, because it never sets a coefficient to exactly
    zero and so has no selection frequency to report (every feature would score
    1.0), and a conservative test is a better answer than no test.
    ``tests/test_regression_orientation.py`` pins the null case, which is where
    an anticonservative version of this would show.

    :param X: Design matrix (``n x p``).
    :param y: Observed responses.
    :param model: Fitted estimator exposing ``predict`` and ``coef_``.
    :returns: 1D array of length ``p``; entries are ``NaN`` when
        ``n <= p + 1``.
    """
    # Coerce y and y_pred to 1D arrays before doing arithmetic so the
    # subtraction does not try to broadcast a length-N array against a
    # single-column DataFrame.
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(model.predict(X)).ravel()

    residuals = y_true - y_pred

    dof = X.shape[0] - X.shape[1] - 1
    if dof <= 0:
        # More features than observations; this happens easily with screen-scale
        # one-hot designs. Standard OLS-style p-values are undefined here.
        return np.full(X.shape[1], np.nan)

    residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)

    # OLS-style standard errors of the coefficients.
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(np.asarray(XtX))
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(np.asarray(XtX))
    se = residual_std_error * np.sqrt(np.diag(XtX_inv))

    coefs = np.asarray(model.coef_).ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = np.where(se > 0, coefs / se, 0.0)
    p_values = 2 * (1 - st.norm.cdf(np.abs(t_stats)))
    return p_values

def perform_mixed_model(y, X, groups, alpha=None,
                        regression_backend=DEFAULT_REGRESSION_BACKEND):
    """Fit a mixed-effects linear model with ``groups`` as the random intercept.

    Collinearity is REPORTED, never silently corrected. The previous revision
    reacted to any VIF above 10 by fitting

    .. code-block:: python

        ridge = Ridge(alpha=alpha).fit(X, y)
        X_ridge = ridge.coef_ * X          # "Adjust X with Ridge coefficients"
        MixedLM(y, X_ridge, groups=groups)

    which is not ridge regression and not a mixed model of anything. It
    multiplies every column by that column's ridge coefficient, so

    * a column whose ridge coefficient is 0 - which is most of them on a
      screen-scale one-hot design - becomes a column of zeros, and the design
      is singular. That is the ``numpy.linalg.LinAlgError: Singular matrix``
      that ``regression_type='mixed'`` died with on real data, thrown from
      inside statsmodels with nothing naming the cause;
    * where it did fit, every fixed effect came back multiplied by an
      arbitrary per-column constant, so the coefficients written to
      ``results.csv`` and ranked on the volcano plot were not effects on the
      response at all. That is the worse of the two outcomes, because it
      completes.

    A one-hot design against an intercept ALWAYS trips VIF > 10, so this path
    was the normal one, not the exception.

    :param y: Response vector.
    :param X: Fixed-effects design matrix (DataFrame).
    :param groups: Cluster identifiers for the random intercept - one entry
        per row of ``X``.
    :param alpha: Must be None. Accepted only so an old call site fails with
        an explanation instead of a TypeError.
    :param regression_backend: WHO fits it. ``'statsmodels'``
        is the default and produced every existing result; ``'torch'`` fits
        the same profiled REML objective on the GPU
        (:mod:`spacr.mixed_gpu`) and returns a result object with the same
        attributes, so nothing downstream can tell which ran except by
        asking. A backend that cannot fit ``'mixed'`` here, is not installed,
        or needs a GPU that is absent is REFUSED with the reason -- see
        :func:`_require_backend`.
    :returns: Fitted ``statsmodels`` ``MixedLMResults``, or the equivalent
        :class:`spacr.mixed_gpu.TorchMixedResults`.
    :raises ValueError: if ``groups`` is None, if ``alpha`` is given, if
        ``groups`` does not align with ``X``, or if the fixed-effects design is
        rank-deficient (which MixedLM would otherwise report as a bare
        LinAlgError from three frames deep).
    """
    # Ensure groups are defined correctly and check for multicollinearity
    if groups is None:
        raise ValueError("Groups must be defined for mixed model regression")

    if alpha is not None:
        raise ValueError(
            "perform_mixed_model takes no penalty: MixedLM has none, and the "
            f"alpha={alpha!r} this used to accept rescaled the design by its "
            "ridge coefficients, which changes what every fixed effect means. "
            "Drop alpha, or fit 'ridge' if you want a penalised model.")

    n_groups = len(np.asarray(groups).reshape(-1))
    if n_groups != X.shape[0]:
        # Silent misalignment here would assign each row to the wrong cluster,
        # which changes every standard error and nothing would look wrong.
        raise ValueError(
            f"groups has {n_groups} entries but the design has {X.shape[0]} "
            f"rows; each row must carry its own cluster id.")

    # Check for multicollinearity by calculating the VIF for each feature.
    # variance_inflation_factor divides by (1 - R^2) and returns inf for a
    # perfectly aliased column, so this doubles as the rank check below.
    X_np = np.asarray(X, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        vif = [variance_inflation_factor(X_np, i) for i in range(X_np.shape[1])]
    print(f"VIF: {vif}")
    if any(v > 10 for v in vif):
        high = [str(c) for c, v in zip(X.columns, vif) if v > 10]
        print(f"Multicollinearity detected with VIF > 10 for: {high}. The "
              f"mixed model is fitted on the design as given - the estimates "
              f"for those terms are unstable, not wrong. Drop or merge the "
              f"aliased terms, or fit 'ridge', if that matters for the "
              f"comparison you are making.")

    rank = np.linalg.matrix_rank(X_np)
    if rank < X_np.shape[1]:
        raise ValueError(
            f"the fixed-effects design is rank {rank} with {X_np.shape[1]} "
            f"columns, so its coefficients are not identified and MixedLM "
            f"cannot solve for them. Some terms are exact linear combinations "
            f"of others - typically a row/column dummy that is constant within "
            f"every group, or a gRNA present in exactly one well. Drop the "
            f"aliased terms, or use random_row_column_effects=True to move "
            f"the plate geometry out of the fixed effects.")

    backend = _require_backend('mixed', regression_backend)
    if backend == 'torch':
        from .mixed_gpu import fit_mixed_reml_torch

        # No variance components: this is the plain random-intercept model
        # `MixedLM(y, X, groups=groups)` fits, which is `re_formula='1'`.
        #
        # THE GPU IS SHARED, AND RUNNING OUT ON IT IS NOT A CRASH. Reported
        # 2026-08-21: `CUDACachingAllocator ... memory allocation failed with
        # OOM on device 0 while trying to allocate 2587885568 bytes (free:
        # 2000093184, total: 25295519744)` -- a 25 GB card with 2 GB free,
        # because something else on the machine had the rest.
        #
        # `mixed_gpu._refuse_if_too_large` already checks free memory before
        # building the DESIGN, and it cannot be enough on a shared device:
        # it covers one allocation, the optimiser makes others, and the free
        # figure it read can be stale by the time any of them run. A
        # co-tenant that allocates between the check and the fit turns a
        # correct check into a wrong one.
        #
        # So the fit falls back to the CPU rather than failing. The same
        # model, the same numbers, slower -- which is the trade the user
        # would have made if asked, and asking is not possible from inside a
        # worker thread twenty minutes into a run.
        try:
            fit = fit_mixed_reml_torch(y, X, groups)
        except Exception as exc:                             # noqa: BLE001
            if not _is_out_of_memory(exc):
                raise
            print(f"■ The GPU ran out of memory during the mixed fit "
                  f"({exc.__class__.__name__}). The card is shared, and what "
                  f"was free when the design was checked was gone by the "
                  f"time the fit asked for it. Falling back to "
                  f"statsmodels on the CPU: same model, same numbers, "
                  f"slower. Re-run when the card is quieter, or set "
                  f"regression_backend='statsmodels (CPU)' to skip the "
                  f"attempt.")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:                                # noqa: BLE001
                pass
            return MixedLM(y, X, groups=groups).fit()
        print(fit.summary_line())
        return fit
    return MixedLM(y, X, groups=groups).fit()

def _is_out_of_memory(exc) -> bool:
    """Is this exception a device or host memory exhaustion?

    Matched by NAME as well as by type, because `torch.cuda.OutOfMemoryError`
    only exists once torch is imported and this must not import it to find
    out. A plain `MemoryError` counts too -- `mixed_gpu` raises one
    deliberately when the design will not fit.
    """
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if "OutOfMemory" in name:
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def create_volcano_filename(csv_path, regression_type, alpha, dst):
    """Build the path this run's volcano plot will be saved to.

    Path construction only: nothing is read, written or created, and the
    ``.pdf`` in the name is not binding - :func:`spacr.plot.save_figure`
    rewrites the extension to whichever format the figure preference selected.

    :param csv_path: Source CSV. Only its basename with the last extension
        stripped becomes the ``<name>_volcano_plot.pdf`` stem, and only its
        directory is used, when ``dst`` is falsy. The file is never opened, so
        a path that does not exist is fine; a bare filename yields a bare
        relative result rather than a path under the working directory.
    :param regression_type: Prefixed to the filename, unless it is exactly
        ``'quantile'`` - then ``alpha`` is prefixed instead. ``None`` is
        stamped literally, giving ``None_...``: :func:`regression` calls this
        before :func:`check_distribution` resolves the auto-selected model, so
        an auto run's plot is never named for the model it actually fitted.
    :param alpha: Read only on the ``'quantile'`` branch; accepted and ignored
        for every other type, whatever its value. :func:`regression` passes
        the ``quantile`` setting here, not the penalty, so two quantiles of
        one screen cannot overwrite each other.
    :param dst: Output directory. Any falsy value, ``None`` and ``''`` alike,
        falls back to the directory of ``csv_path``. It is not created here.
    :returns: The joined path, which :func:`regression` hands to
        :func:`spacr.plot.volcano_plot` as ``save_path``.
    """
    volcano_filename = os.path.splitext(os.path.basename(csv_path))[0] + '_volcano_plot.pdf'
    volcano_filename = f"{regression_type}_{volcano_filename}" if regression_type != 'quantile' else f"{alpha}_{volcano_filename}"

    if dst:
        return os.path.join(dst, volcano_filename)
    return os.path.join(os.path.dirname(csv_path), volcano_filename)

def scale_variables(X, y):
    """Min-max scale the independent (X) and dependent (y) variables to [0, 1].

    Constant columns are passed through UNCHANGED. ``MinMaxScaler`` maps a
    column with zero range to all-zeros, and patsy's intercept is exactly such
    a column, so scaling a design matrix used to silently delete its
    intercept: statsmodels then fitted a model through the origin and still
    printed an ``Intercept`` row, of 0.000, in the summary. Every coefficient
    in that fit absorbs the mean it can no longer estimate.

    :param X: Design matrix (DataFrame).
    :param y: Response, as a 2-D array or single-column frame.
    :returns: ``(X_scaled, y_scaled)`` - a DataFrame with ``X``'s columns and
        a 2-D ``numpy`` array.

    Example:
        .. code-block:: python

            X = pd.DataFrame({'Intercept': 1.0, 'a': [1.0, 2.0, 3.0]})
            scale_variables(X, np.array([[0.0], [1.0], [2.0]]))[0]['Intercept']
            # -> 1.0, 1.0, 1.0   (not 0.0, 0.0, 0.0)
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
    constant = X.nunique(dropna=False) <= 1
    for column in X.columns[constant.values]:
        X_scaled[column] = np.asarray(X[column], dtype=float)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled

def select_glm_family(y):
    """Choose a ``statsmodels`` GLM family from the range and type of the response.

    A coarser rule than :func:`pick_glm_family_and_link`, which also sets
    the link: binary values give ``Binomial``, any other values inside
    ``[0, 1]`` give ``QuasiBinomial``, non-negative integers give
    ``Poisson`` and everything else ``Gaussian``.

    :param y: Response vector.
    :returns: An unfitted ``statsmodels`` family instance on its default link.
    """
    if np.all((y == 0) | (y == 1)):
        print("Using Binomial family (for binary data).")
        return sm.families.Binomial()
    elif (y >= 0).all() and (y <= 1).all():
        print("Using Quasi-Binomial family (for proportion data including 0 and 1).")
        return QuasiBinomial()
    elif np.all(y.astype(int) == y) and (y >= 0).all():
        print("Using Poisson family (for count data).")
        return sm.families.Poisson()
    else:
        print("Using Gaussian family (for continuous data).")
        return sm.families.Gaussian()

#: The two things a fixed-effects screen model can be ABOUT, and the term each
#: one regresses on. One level per fit -- see :func:`prepare_formula`.
LEVEL_TERMS: dict = {
    'grna': 'fraction:grna',
    'gene': 'gene_fraction:gene',
}

#: What ``level`` may be. ``'both'`` is not a design; it is an instruction to
#: fit BOTH of the above SEPARATELY and correct each within itself.
LEVEL_CHOICES: tuple = ('both', 'grna', 'gene')

#: Deprecated formula fragment that combines guide and gene fractions.
#:
#: ``check_and_clean_data`` builds ``gene_fraction`` as the SUM of the gene's
#: gRNA fractions within a well, so every ``gene_fraction:gene[G]`` column is
#: the sum of gene G's ``fraction:grna`` columns whenever G's guides do not
#: share a well. Combining both terms therefore creates exact linear
#: dependencies and a non-identifiable design. The literal remains available
#: so spaCR can detect and refuse that formula explicitly.
COLLINEAR_FORMULA_FRAGMENT = 'fraction:grna + gene_fraction:gene'


def _level_term(level):
    """``LEVEL_TERMS[level]``, with the error that says why ``'both'`` is not one.

    :raises ValueError: for ``'both'`` (two fits, so ask for one at a time) or
        for anything that is not a level at all.
    """
    key = str(level).strip().lower()
    if key in LEVEL_TERMS:
        return LEVEL_TERMS[key]
    if key == 'both':
        raise ValueError(
            "level='both' runs two fits, not one design, so it has no single "
            "formula: call prepare_formula once with level='grna' and once "
            "with level='gene'. Putting both terms in one design is the "
            "collinear model: gene_fraction is the sum of the gene's gRNA "
            "fractions, so the gene block is an exact linear combination of "
            "the gRNA block and the coefficients are not identifiable.")
    raise ValueError(
        f"level={level!r} is not a model level. Choose one of "
        f"{LEVEL_CHOICES!r}.")


#: What the intercept of a screen regression may be asked to be.
#:
#: 'fitted'   the model estimates it, which is what every fit did before
#:            this was a choice;
#: 'zero'     no intercept at all -- the fit passes through the origin, so
#:            a guide's coefficient is its whole predicted score rather
#:            than a departure from a baseline;
#: 'control'  the response is centred on the negative controls before
#:            fitting, so the intercept IS the control level and every
#:            coefficient reads as "above or below the controls";
#: 'value'    the number the user gives. The response is shifted by it and
#:            the term is suppressed, which pins the intercept at exactly
#:            that value rather than estimating one near it.
INTERCEPT_MODES = ("fitted", "zero", "control", "value")


def centre_on_controls(df, dependent_variable, nc):
    """Subtract the negative controls' median response. Returns (df, offset).

    THIS IS WHAT MAKES THE INTERCEPT MEAN SOMETHING. A fitted intercept is
    the response where every predictor is zero, which on a screen design is
    a well with no guide in it -- a point that does not exist. Centred on
    the negative controls, the intercept is the control level, and every
    coefficient reads directly as "this far above or below the controls".

    The offset is returned rather than swallowed so the caller can report
    it: a coefficient table whose response was shifted, with nothing saying
    by how much, is a table nobody can compare with another run.

    :param df: the long frame the fit runs on.
    :param dependent_variable: the response column.
    :param nc: the negative-control guide or gene, as the settings name it.
    :returns: ``(frame, offset)``. The frame is a copy when it was changed
        and the original when it was not; ``offset`` is 0.0 when no control
        row could be identified, and the caller is expected to say so.
    """
    import numpy as _np

    if not nc or dependent_variable not in getattr(df, "columns", ()):
        return df, 0.0
    wanted = str(nc).strip().lower()
    if not wanted:
        return df, 0.0
    # THE GUIDE COLUMN OR THE GENE COLUMN, whichever this frame carries and
    # whichever the control names. `nc` is read as a GENE when it is bare
    # and as a GUIDE when it holds an underscore, which is the rule the
    # rest of the module already applies to it.
    mask = None
    for column in ("grna", "gene", "grna_name", "gene_name"):
        if column not in df.columns:
            continue
        found = df[column].astype(str).str.strip().str.lower() == wanted
        mask = found if mask is None else (mask | found)
    if mask is None or not bool(mask.any()):
        return df, 0.0
    values = _np.asarray(df.loc[mask, dependent_variable], dtype=float)
    values = values[_np.isfinite(values)]
    if not values.size:
        return df, 0.0
    offset = float(_np.median(values))
    if offset == 0.0:
        return df, 0.0
    shifted = df.copy()
    shifted[dependent_variable] = (
        _np.asarray(shifted[dependent_variable], dtype=float) - offset)
    return shifted, offset


def prepare_formula(dependent_variable, random_row_column_effects=False,
                    block_screen=False, level='grna',
                    model_plate_position=True, intercept='fitted'):
    """Build a fixed-effects formula for one screen-analysis level.

    Parameters
    ----------
    dependent_variable : str
        Name of the response column.
    random_row_column_effects : bool, default=False
        Reserve ``rowID`` and ``columnID`` for variance components in
        :func:`fit_mixed_model` instead of adding them as fixed effects.
    block_screen : bool, default=False
        Add ``screenID`` as a fixed effect. Use
        :func:`screen_is_blockable` before enabling this for user data.
    level : {'grna', 'gene'}, default='grna'
        Resolution represented by the formula. ``'grna'`` uses
        ``fraction:grna`` and ``'gene'`` uses ``gene_fraction:gene``.
    intercept : {'fitted', 'zero', 'control', 'value'}, default='fitted'
        What the intercept is. ``'fitted'`` estimates it. ``'zero'`` takes
        it out of the design, so the fit passes through the origin and a
        coefficient is a whole predicted score rather than a departure from
        a baseline. ``'control'`` keeps the term and is completed by the
        caller, which centres the response on the negative controls first --
        the intercept is then the control level by construction.
        ``'value'`` suppresses the term as well, because the caller has
        shifted the response by a number the user gave and the intercept is
        pinned at exactly that number.
    model_plate_position : bool, default=True
        Include plate position in the model. With
        ``random_row_column_effects=False`` it is included as fixed row and
        column terms; with ``random_row_column_effects=True`` it is reserved
        for mixed-model variance components. New application settings default
        to ``False`` even though this helper retains ``True`` for API
        compatibility.

    Returns
    -------
    str
        A patsy-compatible formula for one analysis level.

    Raises
    ------
    ValueError
        If ``level`` is unknown or ``'both'``, or if random plate-position
        effects are requested while plate position is disabled.

    Notes
    -----
    Guide and gene effects are fitted separately because the gene fraction is
    derived from its guide fractions; including both blocks in one design is
    rank deficient. Use :func:`regression_levels` to request both fits.
    """
    from .schema import SCREEN_KEY

    term = _level_term(level)
    screen = f' + {SCREEN_KEY}' if block_screen else ''
    mode = str(intercept or "fitted").strip().lower()
    if mode not in INTERCEPT_MODES:
        raise ValueError(
            f"intercept={intercept!r} is not one of {list(INTERCEPT_MODES)}. "
            f"'fitted' estimates it, 'zero' fits through the origin, "
            f"'control' centres the response on the negative controls so "
            f"the intercept is the control level, and 'value' pins it at a "
            f"number you give.")
    # PATSY'S OWN SUPPRESSION. `- 1` removes the intercept column from the
    # design; there is no other way to say it in a formula, and taking the
    # column out of the design matrix afterwards would leave the formula
    # describing a model that was not the one fitted.
    #
    # 'value' SUPPRESSES IT TOO, and that is what pins it. Fitting
    # `y - c ~ terms - 1` is fitting `y = c + terms`, so the intercept is
    # exactly c; leaving the term in would estimate one NEAR c instead,
    # which is not what asking for a number means.
    origin = ' - 1' if mode in ('zero', 'value') else ''
    if random_row_column_effects and not model_plate_position:
        raise ValueError(
            "model_plate_position=False takes rowID and columnID out of the "
            "model entirely and random_row_column_effects=True asks for them "
            "as variance components, so there is no term left for the mixed "
            "fit to make random: one of the two has to go. Plate position "
            "has three states -- OUT (model_plate_position=False), FIXED "
            "(model_plate_position=True) and RANDOM (both True) -- and this "
            "is a fourth. Set model_plate_position=True to fit row and "
            "column as variance components, or "
            "random_row_column_effects=False to leave plate position out of "
            "the model.")
    if not model_plate_position:
        # OUT. The screen either has no plate-position effect to model -- a
        # randomised layout -- or the caller is spending its 35 parameters
        # somewhere else; see the measurement in this function's docstring for
        # what that costs on a plate that does have one.
        return f'{dependent_variable} ~ {term}{screen}{origin}'
    if random_row_column_effects:
        # Row and column become variance components in fit_mixed_model, so
        # they must not also be fixed terms here.
        return f'{dependent_variable} ~ {term}{screen}{origin}'
    return f'{dependent_variable} ~ {term} + rowID + columnID{screen}{origin}'


def screen_is_blockable(df) -> bool:
    """Whether ``screenID`` can be a term in this frame's design.

    True only when the column exists and carries more than one distinct
    value. A single-screen project is the normal case and must be untouched
    by the design: it has no screenID at all, or one value, and either
    way the term would be a constant column.

    The same rule :func:`spacr.measurement_scan._dummy_block` applies, stated
    once for the formula path so a frame cannot be blocked on by one and not
    the other.
    """
    from .schema import SCREEN_KEY

    if df is None or SCREEN_KEY not in getattr(df, 'columns', ()):
        return False
    return int(df[SCREEN_KEY].astype(str).nunique(dropna=True)) > 1

#: How a guide's BLUP is named in the coefficient table. NOT ``fraction:grna``
#: -- a BLUP is a shrunken prediction of a random effect, not a fixed
#: coefficient, and giving it the fixed term's name is exactly how it would end
#: up in a hit list with a q value beside it.
BLUP_FEATURE_TEMPLATE = 'blup:grna[{}]'

#: What each row of a mixed fit's coefficient table IS. The column exists so
#: nothing downstream has to guess from the name, and so a variance component
#: or a BLUP can never be read as an effect on the response.
TERM_FIXED = 'fixed'
TERM_VARIANCE = 'variance'
TERM_BLUP = 'random_effect_blup'


def _blup_guide_name(key):
    """The guide id inside a statsmodels variance-component BLUP key.

    ``vc_formula={'grna': '0 + C(grna)'}`` labels its columns
    ``grna[C(grna)[244480_3]]``, so the id is the innermost bracket.
    Returns ``None`` for the group's own intercept (``'Group'``) and for
    anything that is not a guide component.
    """
    text = str(key)
    match = re.search(r'C\(grna\)\[(?:T\.)?([^\]]+)\]', text)
    if match:
        return match.group(1)
    return None


def _answering_stop(model):
    """Add a cancellation checkpoint to a statsmodels model instance.

    Wrap the instance's ``loglike`` method because optimizers evaluate it on
    each step, providing finer cancellation granularity than the fit callback.
    The wrapper propagates :class:`spacr.cancellation.PipelineCancelled` and
    returns the same model instance.
    """
    from .cancellation import checkpoint

    original = model.loglike

    def loglike(*args, **kwargs):
        checkpoint()
        return original(*args, **kwargs)

    model.loglike = loglike
    return model


def fit_mixed_model(df, formula, dst, *, random_row_column_effects=False,
                    gene_column='gene', guide_column='grna',
                    regression_backend=DEFAULT_REGRESSION_BACKEND):
    """Fit a mixed model with guides nested within genes.

    The model treats genes as fixed effects and guides as random effects
    nested within genes. In statsmodels notation, ``groups=gene`` supplies the
    outer random intercept and ``vc_formula={'grna': '0 + C(grna)'}`` supplies
    the guide-within-gene variance component.

    A blockable ``screenID`` supplied by :func:`prepare_formula` remains a
    fixed effect. With only two screen levels, a random screen variance would
    be estimated from one degree of freedom. The plate is not nested within
    the screen because plate position is already represented by the row and
    column structure. Single-screen data omit the constant screen term to
    avoid a rank-deficient design.

    Parameters
    ----------
    df : pandas.DataFrame
        Model data containing the formula variables and the gene and guide
        grouping columns.
    formula : str
        Fixed-effects formula, normally returned by :func:`prepare_formula`
        with ``level='gene'``.
    dst : path-like
        Destination for the residual histogram.
    random_row_column_effects : bool, default False
        Add row and column variance components instead of fixed terms.
    gene_column : str, default 'gene'
        Column containing the outer gene groups.
    guide_column : str, default 'grna'
        Column containing guides nested within each gene.
    regression_backend : {'statsmodels', 'torch'}, default 'statsmodels'
        Mixed-model backend. The torch backend fits the same nested model
        with GPU acceleration when available.

    Returns
    -------
    mixed_model
        Fitted backend-specific mixed-model result.
    coef_df : pandas.DataFrame
        Fixed effects, variance components, and guide BLUPs. Variance
        components and BLUPs have ``NaN`` p-values because they are not
        fixed-effect hypothesis tests.

    Raises
    ------
    ValueError
        If required grouping columns are missing, no gene has multiple
        guides, or the backend cannot fit the nested design.
    MixedBackendUnavailable
        If the selected mixed-model backend is unavailable.
    """
    from .plot import plot_histogram

    for column in (gene_column, guide_column):
        if column not in df.columns:
            raise ValueError(
                f"the mixed model nests {guide_column!r} inside "
                f"{gene_column!r}, and this frame has no {column!r} column. "
                f"Columns: {sorted(df.columns)[:20]}")

    response = str(formula).split('~', 1)[0].strip() or 'the response'
    groups = _mixed_model_groups(df, response, df.index,
                                 gene_column=gene_column)

    # THE NESTING HAS TO HAVE SOMETHING TO ESTIMATE. With one guide per gene
    # everywhere, the guide variance component is confounded with the residual
    # and MixedLM returns a boundary variance of zero for it - a number that
    # looks like an answer and is not one.
    guides_per_gene = df.groupby(gene_column, observed=True)[
        guide_column].nunique()
    if int((guides_per_gene > 1).sum()) == 0:
        raise ValueError(
            f"the mixed model nests guides inside genes, and no gene in this "
            f"frame has more than one guide ({len(guides_per_gene)} genes, "
            f"one guide each). The guide variance component would be exactly "
            f"confounded with the residual and would come back as zero. Use a "
            f"fixed-effects regression_type with level='gene' -- with one "
            f"guide per gene the two levels are the same model anyway.")

    vc_formula = {guide_column: f'0 + C({guide_column})'}
    if random_row_column_effects:
        vc_formula['rowID'] = '0 + C(rowID)'
        vc_formula['columnID'] = '0 + C(columnID)'

    backend = _require_backend('mixed', regression_backend)
    _say_what_a_mixed_fit_will_cost(backend, df)
    try:
        if backend == 'torch':
            from .mixed_gpu import mixedlm_torch

            # THE SAME CALL, one line apart. `mixedlm_torch` takes
            # statsmodels' argument shape on purpose so the choice of who
            # fits it cannot become a second code path with its own bugs;
            # everything after this point reads the result the same way.
            mixed_model = mixedlm_torch(formula, df, groups,
                                        vc_formula=vc_formula)
            print(mixed_model.summary_line())
        else:
            model = smf.mixedlm(formula, data=df, groups=groups,
                                re_formula='1', vc_formula=vc_formula)
            mixed_model = _answering_stop(model).fit()
    except MixedBackendUnavailable:
        # THE BACKEND'S OWN REFUSAL SURVIVES. Wrapped in the "MixedLM could
        # not fit this frame" message below it would read as a problem with
        # the screen, and the user would go looking at their data for a
        # missing CUDA device.
        raise
    except Exception as error:
        # SAY WHAT COULD NOT BE EXPRESSED, rather than falling back to a model
        # nobody asked for. The old plate-grouped model is not a substitute:
        # it answers a different question, and substituting it silently is the
        # class of failure this module is most careful about.
        raise ValueError(
            f"MixedLM could not fit y ~ gene_fraction:gene + (1 | "
            f"{gene_column}/{guide_column}) on this frame: "
            f"{type(error).__name__}: {error}. The nesting needs several "
            f"genes, several guides inside at least some of them, and more "
            f"wells than genes. Choose a fixed-effects regression_type with "
            f"level='gene' or level='grna' if this screen cannot support "
            f"it.") from error

    # Plot residuals
    df['residuals'] = mixed_model.resid
    plot_histogram(df, 'residuals', dst=dst)

    # FIXED EFFECTS AND VARIANCE COMPONENTS, kept apart by name.
    # MixedLMResults.params is the fixed effects followed by the variance
    # parameters; fe_params is the fixed half alone, so the difference is what
    # says which is which without parsing ' Var' out of a string.
    fixed_names = set(map(str, mixed_model.fe_params.index))
    coefs = mixed_model.params
    p_values = mixed_model.pvalues
    term_types = [TERM_FIXED if str(name) in fixed_names else TERM_VARIANCE
                  for name in coefs.index]
    parameter_p = np.asarray(p_values.values, dtype=float)
    # A VARIANCE COMPONENT'S WALD P VALUE IS NOT A TEST EITHER, and
    # statsmodels reports one anyway (0.331 and 0.975 for the two components
    # on the synthetic nesting; NaN for others, which is why the NaN alone
    # cannot be relied on to mark them). The null it would test is
    # sigma^2 = 0, which sits on the BOUNDARY of the parameter space, so the
    # normal reference distribution the Wald statistic assumes does not hold
    # and the number is not a probability of anything. Reported as NaN, with
    # the variance itself in `coefficient` where it belongs.
    parameter_p = np.where(
        np.array(term_types) == TERM_VARIANCE, np.nan, parameter_p)
    frames = [pd.DataFrame({
        'feature': [str(name) for name in coefs.index],
        'coefficient': np.asarray(coefs.values, dtype=float),
        'p_value': parameter_p,
        'term_type': term_types,
    })]

    # THE BLUPS, ONE PER GUIDE, WITH NO P VALUE.
    # random_effects is {gene: Series}; each Series carries the group's own
    # intercept under 'Group' and one entry per variance-component column.
    blups = {}
    for group_key, values in (mixed_model.random_effects or {}).items():
        for key, value in dict(values).items():
            guide = _blup_guide_name(key)
            if guide is None:
                continue
            blups[guide] = float(value)
    if blups:
        guides = sorted(blups)
        frames.append(pd.DataFrame({
            'feature': [BLUP_FEATURE_TEMPLATE.format(g) for g in guides],
            'coefficient': [blups[g] for g in guides],
            'p_value': np.full(len(guides), np.nan, dtype=float),
            'term_type': [TERM_BLUP] * len(guides),
        }))

    coef_df = pd.concat(frames, ignore_index=True)
    n_blups = int((coef_df['term_type'] == TERM_BLUP).sum())
    # WHICH BACKEND PRODUCED IT, on every run and not only the fast one
    # (instruction 141: "the run says which backend produced it"). Two runs
    # of the same screen whose numbers differ in the 4th significant figure
    # are explicable only if the log says which fitted them.
    print(f"Mixed model fitted by regression_backend={backend_label(backend)}")
    print(f"Mixed model: gene fixed, guide random nested in gene "
          f"({groups.nunique()} genes, {n_blups} guide BLUPs). "
          f"A BLUP has no p-value, so results_grna.csv from a mixed run is a "
          f"shrunken prediction per guide and carries no q value.")

    # A NON-CONVERGED MLE STILL RETURNS A COEFFICIENT AND A P VALUE, and
    # nothing in statsmodels' return value says it should not be believed.
    # Measured on the maintainer's TSG101 screen (389 genes, 823 guides, 610
    # wells): this fit does not converge inside twenty minutes, and a 50-gene
    # subset converges to a gene-intercept variance on the boundary in 16
    # seconds. Both would have written results.csv in silence.
    #
    # It WARNS rather than raising, because the maintainer chose 'mixed' as
    # the default and a boundary variance component is a normal, informative
    # outcome -- "the genes do not differ in intercept beyond what the gene
    # fixed effect already explains" -- not a broken run. What is not
    # acceptable is not being told.
    if not bool(getattr(mixed_model, 'converged', True)):
        variances = ', '.join(
            f"{name}={value:.3g}" for name, value in
            zip(vc_formula, np.atleast_1d(np.asarray(mixed_model.vcomp,
                                                     dtype=float))))
        print("\n"
              "  ###############################################################\n"
              "  #  WARNING: the mixed model did not converge.                 #\n"
              "  ###############################################################\n"
              f"  Variance components: {variances}; group variance "
              f"{float(np.asarray(mixed_model.cov_re).ravel()[0]):.3g}.\n"
              "  A variance on the boundary at zero is the usual cause and is\n"
              "  itself an answer, but the standard errors and p-values of the\n"
              "  gene fixed effects are not trustworthy while it stands. Fit a\n"
              "  fixed-effects regression_type with level='gene' to get gene\n"
              "  effects whose intervals can be reported.\n")
    return mixed_model, coef_df

def check_and_clean_data(df, dependent_variable):
    """Prepare the merged count / score frame for model fitting.

    Drops rows with a missing ``fraction`` or dependent variable, casts
    the identifier columns to categorical and reports (without dropping)
    collinear columns via VIF. The returned frame keeps only
    ``fraction``, the dependent variable, ``gene``, ``grna``, ``prc``,
    ``plateID``, ``rowID``, ``columnID``, and ``cell_count`` and ``screenID``
    when present, plus a computed ``gene_fraction`` column: the sum of the gene's gRNA
    fractions within each well, which the regression formula regresses on.

    :param df: Merged DataFrame of counts and scores.
    :param dependent_variable: Name of the response column.
    :returns: The cleaned DataFrame used as the model input.
    :raises ValueError: if a ``(prc, grna)`` pair carries more than one
        ``fraction``, which makes ``gene_fraction`` ambiguous.
    """
    
    def handle_missing_values(df, columns):
        """Handle missing values in specified columns."""
        missing_summary = df[columns].isnull().sum()
        print("Missing values summary:")
        print(missing_summary)
        
        # Drop rows with missing values in these fields
        df_cleaned = df.dropna(subset=columns).copy()
        if df_cleaned.shape[0] < df.shape[0]:
            print(f"Dropped {df.shape[0] - df_cleaned.shape[0]} rows with missing values in {columns}.")
        return df_cleaned
    
    def ensure_valid_types(df, columns):
        """Ensure that specified columns are categorical."""
        for col in columns:
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = pd.Categorical(df[col])
                print(f"Converted {col} to categorical type.")
        return df

    def check_collinearity(df, columns):
        """Check for collinearity using VIF (Variance Inflation Factor)."""
        print("Checking for collinearity...")
        
        # Only include fraction and the dependent variable for collinearity check
        df_encoded = df[columns]
        
        # Ensure all data in df_encoded is numeric
        df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
        
        # Check for perfect multicollinearity (i.e., rank deficiency)
        if np.linalg.matrix_rank(df_encoded.values) < df_encoded.shape[1]:
            print("Warning: Perfect multicollinearity detected! Dropping correlated columns.")
            df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df_encoded.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(df_encoded.values, i) for i in range(df_encoded.shape[1])]
        except np.linalg.LinAlgError:
            print("LinAlgError: Unable to compute VIF due to matrix singularity.")
            return df_encoded

        print("Variance Inflation Factor (VIF) for each feature:")
        print(vif_data)

        # Report high VIF (> 10) but do NOT drop. The only columns checked
        # here are 'fraction' and the dependent variable, and both are
        # required downstream: 'gene_fraction' is derived from 'fraction'
        # and the regression formula regresses the dependent variable on
        # it. The previous revision dropped every column above the
        # threshold, so any dependent variable even approximately
        # proportional to 'fraction' (VIF -> inf) dropped both and made the
        # caller die on KeyError: 'Column not found: fraction'.
        high_vif_columns = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
        if high_vif_columns:
            print(f"Warning: high collinearity (VIF > 10) for: {high_vif_columns}. "
                  f"Keeping them - the regression formula requires both - but "
                  f"coefficient estimates may be unstable.")

        return df_encoded
    
    # Step 1: Handle missing values in relevant fields
    df = handle_missing_values(df, ['fraction', dependent_variable])
    
    # Step 2: Ensure grna, gene, plate, row, column, and prc are categorical types
    df = ensure_valid_types(df, ['grna', 'gene', 'plateID', 'rowID', 'columnID', 'prc'])
    
    # Step 3: Check for multicollinearity in fraction and the dependent variable
    df_cleaned = check_collinearity(df, ['fraction', dependent_variable])
    
    # Ensure that the prc, plate, row, and column columns are still included for random effects
    df_cleaned['gene'] = df['gene']
    df_cleaned['grna'] = df['grna']
    df_cleaned['prc'] = df['prc']
    df_cleaned['plateID'] = df['plateID']
    df_cleaned['rowID'] = df['rowID']
    df_cleaned['columnID'] = df['columnID']

    # check_collinearity only returns 'fraction' and the dependent variable,
    # so 'cell_count' used to be stripped unconditionally. regression() then
    # found no 'cell_count' column and passed weights=None, which made the
    # documented GLM-binomial var_weights=cell_count path dead code.
    if 'cell_count' in df.columns:
        df_cleaned['cell_count'] = df['cell_count']

    # 'screenID' is a DESIGN COLUMN when the frame holds more than one screen:
    # regression() asks screen_is_blockable() of the CLEANED frame and patsy
    # then builds the '+ screenID' term from that same frame. Stripping it here
    # made both impossible at once -- the answer was always False, so two
    # screens were pooled with nothing printed and no term in the model, which
    # charges the difference between the experiments to whichever guides are
    # over-represented in one of them.
    from .schema import SCREEN_KEY

    if SCREEN_KEY in df.columns:
        df_cleaned[SCREEN_KEY] = df[SCREEN_KEY]

    # 'gene_fraction' is the share of the well's library that belongs to the
    # gene: the sum of its gRNAs' fractions IN THAT WELL, counted once each.
    #
    # The obvious spelling - groupby(['prc', 'gene'])['fraction'].sum() over
    # the frame - is right only while the frame has exactly one row per
    # (well, gRNA). With agg_type=None (which quantile regression forces, see
    # get_perform_regression_default_settings) perform_regression deliberately
    # joins the well's gRNAs against the well's CELLS, so every (well, gRNA)
    # row appears once per cell and the sum came out multiplied by the well's
    # cell count. Two consequences, both silent: every gene coefficient was
    # divided by roughly that factor, and - because wells do not all hold the
    # same number of cells - the inflation differed per well, so gene_fraction
    # was no longer comparable across the plate.
    grna_key = ['prc', 'gene', 'grna']
    per_grna = df_cleaned[grna_key + ['fraction']].drop_duplicates()
    clash = per_grna.duplicated(subset=grna_key, keep=False)
    if clash.any():
        # One gRNA cannot hold two different shares of the same well's
        # library. Deduplicating past this would pick whichever row sorted
        # first and every gene coefficient downstream would rest on that
        # coin flip.
        offenders = per_grna.loc[clash, grna_key].drop_duplicates()
        raise ValueError(
            f"{len(offenders)} (well, gRNA) pair(s) carry more than one "
            f"'fraction', so the gene's share of the well is ambiguous - e.g. "
            f"{offenders.iloc[0].to_dict()}. This means the count table was "
            f"joined twice, or two count files describe the same plate. "
            f"Aggregate the counts per (prc, grna) before regressing.")
    gene_totals = per_grna.groupby(['prc', 'gene'], observed=False)['fraction'].sum()
    df_cleaned['gene_fraction'] = pd.MultiIndex.from_arrays(
        [df_cleaned['prc'], df_cleaned['gene']]).map(gene_totals)

    print("Data is ready for model fitting.")
    return df_cleaned

def minimum_cell_simulation(settings, num_repeats=10, sample_size=100, tolerance=0.02, smoothing=10, increment=10, dst=None):
    """
    Estimate the minimum number of cells per well needed for a stable well mean.

    For the wells with the most objects, repeatedly subsamples cells at
    increasing sample sizes and records the mean absolute difference from
    the well's full mean. Plots the smoothed curve with a ±1 s.d. band,
    marks the elbow point (or ``settings['min_cell_count']`` when it is
    set) and writes ``cell_min_threshold.pdf`` into ``dst``.

    Pass ``dst`` to keep the figure in a specific run folder. When omitted,
    the function uses the screen-level ``results`` folder derived from
    ``count_data`` for compatibility with direct notebook and script calls.

    :param settings: Requires ``score_data`` (CSV path or list of paths),
        ``dependent_variable``, ``tolerance`` (int percent or float
        fraction) and
        ``min_cell_count``. ``count_data`` is needed only when ``dst`` is
        left unset, and only to locate the figure.
    :param num_repeats: Subsamples drawn per sample size. Default ``10``.
    :param sample_size: Number of wells, taken largest-first by cell
        count, to simulate. Default ``100``.
    :param tolerance: Unused; the tolerance applied is
        ``settings['tolerance']``.
    :param smoothing: Rolling-window width used to smooth the curve.
    :param increment: Step between the simulated sample sizes.
    :param dst: Folder for ``cell_min_threshold.pdf``, created if missing.
        Default ``None``: ``<folder of settings['count_data'][0]>/results``.
    :returns: The elbow point's sample size, i.e. the minimum cell count
        per well, for passing to :func:`process_scores`.
    :raises ValueError: if ``settings['tolerance']`` is neither an int nor
        a float.
    """

    from .utils import correct_metadata_column_names

    # Load and process data
    if isinstance(settings['score_data'], str):
        settings['score_data'] = [settings['score_data']]

    dfs = []
    for i, score_data in enumerate(settings['score_data']):
        # ONE READER: canonical metadata names, one column per key and the
        # `pplate1` repair, all decided in spacr.tabular rather than here.
        df = tabular.read_table(score_data)
        df = correct_metadata_column_names(df)
        df['plateID'] = f'plate{i + 1}'
        
        if 'prc' not in df.columns:
            df['prc'] = _compose_prc_column(df)
            
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Compute the number of cells per well and select the top 100 wells by cell count
    cell_counts = df.groupby('prc').size().reset_index(name='cell_count')
    top_wells = cell_counts.nlargest(sample_size, 'cell_count')['prc']

    # Filter the data to include only the top 100 wells
    df = df[df['prc'].isin(top_wells)]

    # Initialize storage for absolute difference data
    diff_data = []

    # Group by wells and iterate over them
    for i, (prc, group) in enumerate(df.groupby('prc')):
        # `dependent_variable`, NOT `score_column`. The two named the same
        # measurement -- settings.py defaulted one to the other and the
        # tooltip said they must agree -- so instruction 135 A retired the
        # duplicate. This function was its only regression-path reader and
        # kept the old name, which killed every run here with
        # KeyError: 'score_column', AFTER the settings had been
        # canonicalised and before a single well was fitted.
        #
        # `score_column` still exists and still means something ELSE: in
        # interpret_vision_model below it names the CNN score column,
        # default 'cv_predictions'. That is why this is three targeted
        # edits and not a rename.
        original_mean = group[settings['dependent_variable']].mean()
        max_cells = len(group)
        sample_sizes = np.arange(2, max_cells + 1, increment)  # Sample sizes from 2 to max cells

        # Iterate over sample sizes and compute absolute difference
        for sample_size in sample_sizes:
            abs_diffs = []

            # Perform multiple random samples to reduce noise
            for _ in range(num_repeats):
                sample = group.sample(n=sample_size, replace=False)
                sampled_mean = sample[settings['dependent_variable']].mean()
                abs_diff = abs(sampled_mean - original_mean)  # Absolute difference
                abs_diffs.append(abs_diff)

            # Compute the average absolute difference across all repeats
            avg_abs_diff = np.mean(abs_diffs)

            # Store the result for plotting
            diff_data.append((sample_size, avg_abs_diff))

    # Convert absolute difference data to DataFrame for plotting
    diff_df = pd.DataFrame(diff_data, columns=['sample_size', 'avg_abs_diff'])

    # Group by sample size to calculate mean and standard deviation
    summary_df = diff_df.groupby('sample_size').agg(
        mean_abs_diff=('avg_abs_diff', 'mean'),
        std_abs_diff=('avg_abs_diff', 'std')
    ).reset_index()

    # Apply smoothing using a rolling window
    summary_df['smoothed_mean_abs_diff'] = summary_df['mean_abs_diff'].rolling(window=smoothing, min_periods=1).mean()

    # Convert percentage to fraction
    if isinstance(settings['tolerance'], int):
        tolerance_fraction = settings['tolerance'] / 100  # Convert 2% to 0.02
    elif isinstance(settings['tolerance'], float):
        tolerance_fraction = settings['tolerance']
    else:
        raise ValueError("Tolerance must be an integer 0 - 100 or float 0.0 - 1.0.")

    # Compute the relative threshold for each well
    relative_thresholds = {
        prc: tolerance_fraction * group[settings['dependent_variable']].mean()
        for prc, group in df.groupby('prc')
    }

    # Detect the elbow point when mean absolute difference is below the relative threshold
    summary_df['relative_threshold'] = summary_df['sample_size'].map(
        lambda size: np.mean([relative_thresholds[prc] for prc in top_wells])  # Average across selected wells
    )

    elbow_df = summary_df[summary_df['smoothed_mean_abs_diff'] <= summary_df['relative_threshold']]

    # Select the first occurrence if it exists; otherwise, use the last point
    if not elbow_df.empty:
        elbow_point = elbow_df.iloc[0]  # First point where condition is met
    else:
        elbow_point = summary_df.iloc[-1]  # Fallback to last point

    # THE SWEEP, IN PYQTGRAPH. A line through an ordered x with the spread
    # it was summarised from behind it, and the chosen threshold marked --
    # `FastPlot.add_curve` draws exactly that, so the file and the tab are
    # one scene.
    #
    # WHERE THE FIGURE GOES. A `dst` the caller named is used as given; the
    # fallback is the historical screen folder, and it is derived here rather
    # than in the signature because it depends on `settings`.
    if dst is None:
        dst = os.path.join(os.path.dirname(settings['count_data'][0]),
                           'results')
    dst = os.path.abspath(os.path.expanduser(os.fspath(dst)))
    os.makedirs(dst, exist_ok=True)

    mark = (elbow_point['sample_size'] if settings['min_cell_count'] is None
            else settings['min_cell_count'])
    fig_file_path = _draw_the_cell_count_sweep(
        summary_df, mark, os.path.join(dst, 'cell_min_threshold.pdf'))
    if fig_file_path:
        print(f"Saved {fig_file_path}")

        return elbow_point['sample_size']

def _statsmodels_p_values(model, coefs):
    """Return per-coefficient p-values from a statsmodels-shaped results object.

    Every statsmodels results class spaCR fits exposes ``pvalues``. The
    fallback exists for :mod:`spacr.power_model`, whose Laplace approximation
    reports standard errors rather than a test: a two-sided normal p-value
    from ``coef / bse`` is exactly what a Wald test on that approximation is,
    and computing it here keeps the horseshoe fit in the same table as the
    rest instead of giving it a private code path.

    :param model: Fitted results object.
    :param coefs: Its ``params``, already extracted.
    :returns: 1-D float array aligned with ``coefs``.
    :raises ValueError: when the object carries neither ``pvalues`` nor
        ``bse``, so no inference is possible.
    """
    pvalues = getattr(model, 'pvalues', None)
    if pvalues is not None:
        return np.asarray(pvalues, dtype=float).reshape(-1)

    bse = getattr(model, 'bse', None)
    if bse is None:
        raise ValueError(
            f"{type(model).__name__} exposes neither .pvalues nor .bse, so "
            f"spaCR cannot attach a p-value to its coefficients. A results "
            f"object handed to process_model_coefficients must carry one or "
            f"the other.")
    std_err = np.asarray(bse, dtype=float).reshape(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(std_err > 0,
                     np.asarray(coefs, dtype=float).reshape(-1) / std_err,
                     0.0)
    return 2.0 * (1.0 - st.norm.cdf(np.abs(z)))


def _bootstrap_wald_p_values(model, X, y, n_boot=200, random_state=0):
    """Return bootstrap Wald p-values for an estimator with no inference.

    Refits ``model``'s estimator on ``n_boot`` nonparametric resamples of the
    rows, takes the empirical standard deviation of each coefficient across
    the resamples and reports ``2 * (1 - Phi(|coef| / sd))``.

    This is the honest minimum for the hinge backend: an SVM has no
    likelihood, so there is no Wald or likelihood-ratio test to run, and the
    alternative - leaving ``p_value`` NaN - would make
    :func:`perform_regression` select ``p_value <= 0.05`` on an all-NaN column
    and report "0 significant gRNAs" for every hinge run, which reads exactly
    like a screen with no hits.

    A resample that loses a class entirely is skipped rather than fitted; a
    coefficient whose bootstrap standard deviation is zero (never selected, or
    identical in every resample) gets ``p = 1``, never a division by zero.

    :param model: A fitted scikit-learn estimator; cloned, never refitted in
        place, so the caller's model object is untouched.
    :param X: Design matrix.
    :param y: Response the model was fitted on - for hinge, the BINARISED one.
    :param n_boot: Number of resamples. Default 200.
    :param random_state: Seed, so a hit list is reproducible from the settings.
    :returns: 1-D float array of length ``X.shape[1]``.
    :raises RuntimeError: when no resample could be fitted at all.
    """
    rng = np.random.default_rng(random_state)
    X_values = np.asarray(X, dtype=float)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    n = X_values.shape[0]

    draws = []
    one_class = 0
    unfittable = 0
    last_failure = None
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y_boot = y_values[idx]
        if np.unique(y_boot).size < 2:
            # One-class resample: the estimator has no boundary to fit. Common
            # on a screen with few positive wells, and not an error.
            one_class += 1
            continue
        try:
            fitted = clone(model).fit(X_values[idx], y_boot)
        except Exception as exc:
            unfittable += 1
            last_failure = exc
            continue
        draws.append(np.asarray(fitted.coef_, dtype=float).ravel())

    if not draws:
        raise RuntimeError(
            f"none of the {n_boot} bootstrap resamples could be fitted, so no "
            f"standard error is available for the hinge coefficients. This "
            f"usually means one class holds only a handful of wells; check "
            f"hinge_threshold.")

    # Only "none of them" used to be reported, and that is the case where the
    # numbers are least dangerous, because it raises. 199 of 200 failing gave
    # a standard deviation taken over one draw — zero by construction — which
    # makes every p-value exactly 1.0: a hit list that reads like a clean
    # screen with no significant gRNAs in it, with nothing anywhere saying the
    # inference did not happen. So say how many draws the p-values rest on
    # whenever it is not all of them.
    dropped = int(n_boot) - len(draws)
    if dropped:
        LOG.warning(
            "hinge bootstrap: %d of %d resamples produced no coefficients "
            "(%d were one-class, %d would not fit%s). The p-values below are "
            "computed from the remaining %d.",
            dropped, int(n_boot), one_class, unfittable,
            f"; last error: {last_failure}" if last_failure is not None else "",
            len(draws))
    if len(draws) < 2:
        LOG.warning(
            "hinge bootstrap: only %d resample(s) survived, so the coefficient "
            "standard deviation is zero and EVERY p-value below is exactly "
            "1.0. That is an absence of evidence, not evidence of absence — "
            "do not read it as 'no significant gRNAs'.", len(draws))

    coefs = np.asarray(model.coef_, dtype=float).ravel()
    sd = np.std(np.vstack(draws), axis=0, ddof=1) if len(draws) > 1 else \
        np.zeros_like(coefs)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(sd > 0, coefs / sd, 0.0)
    return 2.0 * (1.0 - st.norm.cdf(np.abs(z)))


#: Backends whose fitted results object carries ``params`` and ``pvalues``
#: directly. Most are statsmodels; ``horseshoe`` and ``rra`` are spaCR's own
#: adapters (:class:`_HorseshoeResults`, :class:`_RRAResults`), which exist so
#: that a model with a posterior or a permutation null lands in the same table
#: as the likelihood fits instead of getting a private code path.
#: ``mixed`` is here too, and its variance components are dropped below - they
#: are not effects on the response.
_STATSMODELS_COEF_TYPES = (
    'ols', 'wls', 'rlm', 'huber', 'glm', 'poisson', 'logit', 'probit',
    'quasi_binomial', 'quantile', 'mixed', 'horseshoe', 'rra',
    # `spline` IS an OLS fit -- on a design with a spline basis over the
    # covariates -- so its results object is the same one and its
    # coefficients come out the same way.
    'spline',
)

#: Backends whose fitted object exposes ``coef_`` and ``predict`` and carries
#: no inference of its own, so :func:`calculate_p_values` supplies the
#: (deliberately conservative) p-value. Three are scikit-learn's; ``group_lasso``
#: is :class:`_GroupLassoResults` around :mod:`spacr.group_lasso`, and it is
#: here rather than in a branch of its own precisely so it reports what the
#: other penalised backends report.
_SKLEARN_COEF_TYPES = ('ridge', 'lasso', 'elasticnet', 'group_lasso')


#: The level term patsy writes for one gRNA or one gene:
#: ``fraction:grna[224750_2]`` or ``gene_fraction:gene[T.224750]``. Anchored,
#: so a nuisance column -- ``Intercept``, ``rowID[T.r2]``, ``columnID[T.c7]``,
#: ``screenID[T.b]`` -- does not match and is answered with None. An unanchored
#: search would read ``r2`` out of ``rowID[T.r2]`` and hand the row and column
#: dummies to the grouping as though they were genes.
_LEVEL_TERM_IN_FEATURE = re.compile(
    r'^(?:fraction:grna|gene_fraction:gene)\[(?:T\.)?(.*)\]$')

#: The guide number a gRNA id ends with: ``224750_2`` -> gene ``224750``.
_GUIDE_SUFFIX = re.compile(r'_\d+$')


def _gene_of_design_column(column):
    """The gene a design column belongs to, or ``None`` for a nuisance term.

    THE GROUPING BOTH NEW BACKENDS RUN ON. ``group_lasso`` penalises a gene's
    guide columns as one block and ``rra`` aggregates their ranks, so both need
    to know which columns are the same gene's -- and the design matrix is all
    either of them is given. It is parsed from the column name rather than
    passed in, because the name is what patsy actually built the column from;
    a second, separately supplied grouping could disagree with it and would
    then split a gene silently.

    ``perform_regression`` reduces ``TGGT1_224750_2`` to ``224750_2`` before
    the fit (the three-token org/gene/guide split it makes on the merged
    frame), but a caller that fits ``regression_model`` directly may not have,
    so the trailing guide number is stripped from whatever is there and the
    ORG PREFIX IS LEFT ALONE: it is constant across a screen, so ``TGGT1_224750`` and ``224750``
    are each a consistent key for their own frame, and stripping it would be a
    guess about naming.

    :param column: a design-matrix column name.
    :returns: the gene id, or ``None`` when the column is not a level term.
    """
    text = str(column)
    match = _LEVEL_TERM_IN_FEATURE.match(text)
    if match is None:
        return None
    identifier = match.group(1)
    if text.startswith('gene_fraction:'):
        return identifier
    # A gene whose guides carry no numeric suffix would collapse to the empty
    # string, which is one group for every such guide in the screen; keep the
    # id itself instead, which makes it its own single-guide gene.
    return _GUIDE_SUFFIX.sub('', identifier) or identifier


def _level_term_mask(columns):
    """A boolean mask of the columns that name a gRNA or a gene.

    ``dtype=bool`` is not incidental: an empty design gives ``np.array([])``,
    which is float64, and boolean-indexing a coefficient vector with a float
    array raises ``IndexError`` instead of selecting nothing.
    """
    return np.array([_gene_of_design_column(column) is not None
                     for column in columns], dtype=bool)


def _design_column_groups(columns):
    """One group label per design column, nuisance terms in groups of their own.

    :mod:`spacr.group_lasso` groups by label, so every column needs one. A
    nuisance column gets its OWN name as its label, which makes it a singleton
    group: it is penalised on its own, exactly as ordinary lasso would, and it
    can never be pulled into a gene's block and dragged to zero with it.
    """
    return [_gene_of_design_column(column) or str(column)
            for column in columns]


def _say_when_a_control_matched_nothing(coef_df, nc, pc, controls) -> None:
    """Warn, by name, about a control that selected no coefficient.

    The consequence is named because the number is not obviously missing:
    the run completes, the volcano draws, and the effect-size cut is simply
    measured on nothing.
    """
    counts = coef_df['condition'].value_counts()
    for value, tag, what in ((nc, 'nc', 'negative_control'),
                             (pc, 'pc', 'positive_control')):
        if value in (None, '') or int(counts.get(tag, 0)):
            continue
        print(f"  WARNING: {what}={value!r} matches no coefficient in this "
              f"screen, so the baseline and the effect-size cut that read it "
              f"have nothing to measure. Check it against the guide names in "
              f"the count table -- spaCR reads a bare id as a GENE and one "
              f"with an underscore as a GUIDE.")
    if (controls or []) and not int(counts.get('control', 0)):
        print(f"  WARNING: none of the {len(list(controls))} control(s) named "
              f"matches a coefficient, so there is no control spread to "
              f"measure an effect-size cut on.")


def label_control_condition(features, guides, nc=None, pc=None, controls=None,
                            *, strict: bool = False, verbose: bool = False):
    """Label every coefficient row ``'nc'``, ``'pc'``, ``'control'`` or ``'other'``.

    The ``condition`` column: what the volcano colours by, what the results
    panel offers in "colour by", and -- the reason this is a function rather
    than four lines inside :func:`process_model_coefficients` -- what the
    EFFECT-SIZE CUT measures its spread on. A coefficient table without it is
    a table :meth:`spacr.qt.widgets.regression_results.RegressionResultsPanel.
    set_threshold_method` answers "No control coefficients, so no effect-size
    cut" for, which is what every guide-permutation run used to get.

    Precedence is ``nc``, then ``pc``, then the explicit ``controls`` list, so
    a guide named in two of them is reported once and always the same way.

    :param features: the model term per row, e.g. ``fraction:grna[000000_1]``.
        ``nc`` and ``pc`` are matched as SUBSTRINGS of it, which is how a
        negative control given as a gene id reaches a term named for a guide.
    :param guides: the guide identifier per row, matched whole against
        ``controls``. Both sides are compared as text, so a control list that
        round-tripped through a settings CSV as integers still matches.
    :param nc: negative-control identifier, or ``None`` for no negative
        control.
    :param pc: positive-control identifier, or ``None``.
    :param controls: non-targeting guide identifiers, or ``None``. ``None``
        means "no control list" and labels nothing -- it is the value
        :func:`perform_regression` documents for a control-free screen, and
        the inline version this replaced raised ``TypeError`` on it.
    :param strict: raise :class:`spacr.control_names.ControlNotFound` when a
        NAMED ``nc`` or ``pc`` matches nothing. Off by default so a call on a
        partial frame is not an error; the run turns it on, because there a
        control matching nothing is a number computed against an empty set.
    :param verbose: print what each control resolved to and how much it
        matched.
    :returns: a :class:`pandas.Series` of labels aligned with ``features``.
    """
    features = pd.Series(features).astype(str)
    guides = pd.Series(guides).astype(str)
    guides.index = features.index
    # Control names as TEXT. A gene id like 233460 is a perfectly good
    # negative_control, and a settings file round-trips it back as the INT
    # 233460 -- at which point `nc in row['feature']` raises "'in <string>'
    # requires string as left operand, not int" and the whole regression dies
    # on a value that was legal the moment it was typed into the GUI.
    nc_name = '' if nc is None else str(nc)
    pc_name = '' if pc is None else str(pc)
    control_names = {str(name) for name in (controls or [])}

    # ONE MATCHER (184 C). `nc` and `pc` were matched as SUBSTRINGS of the
    # model term, which is how `nc='23346'` claims `233460` AND `2334600` --
    # and the rows it steals are then reported as controls, which is worse
    # than missing them. `spacr.control_names` reads a typed control as a
    # gene or a guide by the same rule `process_reads` already applies to the
    # data, and matches WHOLE values at that level.
    from .control_names import rows_for

    labels = pd.Series('other', index=features.index, dtype=object)
    # pandas 3 preserves missing values through ``astype(str)``.  When every
    # coefficient is a continuous term (for example Intercept + fraction),
    # every extracted guide is missing and ``str.split`` then produces an
    # all-float intermediate on which a second ``.str`` access raises.  A
    # missing guide means "no guide", so normalize it to empty text before
    # splitting; it cannot match any nonblank control.
    genes = guides.fillna('').str.split('_').str[0]
    library = list(guides.astype(str).unique())
    if control_names:
        for name in sorted(control_names):
            mask, note = rows_for(name, guides, genes, names=library)
            if verbose and note:
                print(f"  {note}")
            labels[mask.to_numpy()] = 'control'
    # PRECEDENCE UNCHANGED: nc over pc over the list, so a guide named twice
    # is reported once and always the same way.
    for name, tag in ((pc_name, 'pc'), (nc_name, 'nc')):
        if not name:
            continue
        mask, note = rows_for(name, guides, genes, names=library,
                              strict=strict,
                              label='positive control' if tag == 'pc'
                              else 'negative control')
        if verbose and note:
            print(f"  {note}")
        labels[mask.to_numpy()] = tag
    return labels


def process_model_coefficients(model, regression_type, X, y, nc, pc, controls,
                               hinge_threshold=None, hinge_n_boot=200):
    """Return a DataFrame of model coefficients and p-values, one row per term.

    Every name in :data:`REGRESSION_TYPES` has a branch here. It is the same
    table for all of them - ``feature``, ``coefficient``, ``p_value``,
    ``-log10(p_value)``, ``grna``, ``condition`` - because everything
    downstream (the volcano plot, the hit table, the metadata merge) reads
    those columns and nothing else.

    :param model: The fitted object from :func:`regression_model`.
    :param regression_type: Which backend produced it.
    :param X: Design matrix, used for the sklearn feature names and for the
        p-value approximations that need the data back.
    :param y: Response, likewise.
    :param nc: Negative-control identifier, matched against the feature name.
    :param pc: Positive-control identifier.
    :param controls: Explicit list of control gRNA identifiers.
    :param hinge_threshold: The binarisation cut used by the hinge fit; the
        bootstrap below must reproduce the SAME two classes the fit saw.
    :param hinge_n_boot: Bootstrap resamples used for the hinge p-values.
    :returns: Coefficient DataFrame with the row/column nuisance terms removed.
    :raises ValueError: on an unsupported ``regression_type``.
    """

    if regression_type == 'beta':
        coefs = model.params
        std_err = model.bse
        wald_stats = coefs / std_err
        p_values = 2 * (1 - st.norm.cdf(np.abs(wald_stats)))

        coef_df = pd.DataFrame({
            'feature': coefs.index,
            'coefficient': coefs.values,
            'std_err': std_err.values,
            'wald_stat': wald_stats.values,
            'p_value': p_values,
        })

    elif regression_type in _STATSMODELS_COEF_TYPES:
        coefs = model.params
        p_values = _statsmodels_p_values(model, coefs)

        coef_df = pd.DataFrame({
            'feature': coefs.index,
            'coefficient': coefs.values,
            'p_value': np.asarray(p_values, dtype=float),
        })
        if regression_type == 'mixed':
            # MixedLMResults.params appends the random-effect variance
            # components ('Group Var', 'Group x ... Cov'). They are variances,
            # not effects on the response, and their p-value is NaN - leaving
            # them in put a row on the volcano plot that no gene owns.
            coef_df = coef_df[coef_df['feature'].isin(
                [str(c) for c in X.columns])].reset_index(drop=True)

    elif regression_type in _SKLEARN_COEF_TYPES:
        coefs = np.asarray(model.coef_).ravel()
        p_values = calculate_p_values(X, y, model)

        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefs,
            'p_value': p_values,
        })

    elif regression_type == 'hinge':
        # LinearSVC has no likelihood, so there is no Wald test to run and
        # calculate_p_values is meaningless here (its residual is the 0/1
        # misclassification, not a Gaussian error). The p-value reported is a
        # BOOTSTRAP Wald: refit the same estimator on hinge_n_boot resamples of
        # the wells, take the empirical standard deviation of each coefficient
        # and compare the point estimate to it. It is a stability statistic,
        # not a likelihood-ratio test, and the tooltip for hinge_n_boot says so.
        coefs = np.asarray(model.coef_).ravel()
        p_values = _bootstrap_wald_p_values(
            model, X, binarise_response(y, hinge_threshold,
                                        name='dependent variable'),
            n_boot=hinge_n_boot)

        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefs,
            'p_value': p_values,
        })

    else:
        raise ValueError(f"Unsupported regression type: {regression_type}")

    coef_df['-log10(p_value)'] = -np.log10(coef_df['p_value'])
    coef_df['grna'] = (
        coef_df['feature']
        .str.extract(r'\[(.*?)\]')[0]
        .str.replace(r'^T\.', '', regex=True)
    )
    # ONE LABELLER, shared with the guide-permutation path. It grew a second
    # copy the moment the permutation table needed a `condition` column too,
    # and two copies of "what counts as a control" is how the run and the
    # panel come to disagree about which coefficients the cut is measured on.
    # LOUD, NOT FATAL (184 D). A control that matches nothing is not "no
    # controls" -- it is every normalisation, every volcano baseline and the
    # whole effect-size cut computed against an empty set, while the run
    # finishes and the figures draw. So it has to be said.
    #
    # IT MUST NOT RAISE, AND THE INSTRUCTION SAID IT SHOULD. spaCR SHIPS
    # nc='233460' and pc='220950' -- Toxoplasma gene ids -- so raising would
    # make every screen that is not this one fail on a value the user never
    # typed. "Error, not a silent zero" is right about the silence and wrong
    # about the exception: the fix for silence is a sentence nobody can miss.
    # `strict=True` remains available for a caller that knows the control was
    # chosen rather than defaulted.
    coef_df['condition'] = label_control_condition(
        coef_df['feature'], coef_df['grna'], nc=nc, pc=pc, controls=controls,
        verbose=True)
    _say_when_a_control_matched_nothing(coef_df, nc, pc, controls)

    return coef_df[~coef_df['feature'].str.contains('row|column')]

def _draw_the_threshold_sweep(settings, res_folder, *,
                              measured: bool = False) -> None:
    """Draw the guide-fraction sweep without replacing the threshold in force.

    :param settings: the regression settings, read for the threshold in force
        and the count tables the sweep reads.
    :param res_folder: this run's folder, kept in the signature because the
        caller collects what the sweep drew into it.
    :param measured: the threshold in force came from the control-well
        calibration rather than from the user.

    THE TWO ANSWERS, SIDE BY SIDE, AND NAMED. The sweep answers "how many
    guides per well do I want" from the counts alone; the calibration answers
    "which cut-off makes imaging and sequencing agree" from the control
    wells. They are different questions, so neither replaces the other and
    the run reports both -- but a run that says "you set 0.0168" about a
    number the calibration measured is telling the user something untrue
    about where their threshold came from, which is the one thing this line
    exists to say.

    Plotting is diagnostic; a rendering failure is reported without
    invalidating the regression run.
    """
    try:
        chosen = settings.get('fraction_threshold')
        derived = _graph_sequencing_stats(settings)
        if derived is not None and chosen is not None:
            whose, mine = (("the control-well calibration measured",
                            "The measured value") if measured
                           else ("you set", "Your value"))
            print(f"gRNA fraction-threshold sweep drawn: {whose} "
                  f"{chosen}; the sweep's own pick on this screen is "
                  f"{derived}. The two answer different questions -- which "
                  f"cut-off the control wells agree at, and how many guides "
                  f"a well should keep -- so neither replaces the other. "
                  f"{mine} is the one in force.")
    except Exception as error:                                # noqa: BLE001
        print(f"the gRNA fraction-threshold sweep could not be drawn "
              f"({type(error).__name__}: {error}); the run is unaffected "
              f"and fraction_threshold={settings.get('fraction_threshold')} "
              f"is still in force")


def _show_response_distribution(before_df, dependent_variable, settings):
    """Display the response distribution before and after transformation.

    The panel is emitted through Matplotlib so the Qt bridge can add it to the
    figure queue. It is also shown when no transformation is selected, making
    the unchanged distribution explicit.
    """
    if before_df is None or not settings.get('plot', True):
        return
    try:
        import matplotlib.pyplot as plt

        from .response_distribution import panel

        # ``process_scores`` can rename the response, while ``before_df``
        # retains its original name. Prefer the requested names and otherwise
        # use the final numeric response column in the aggregated table.
        wanted = [str(dependent_variable),
                  str(settings.get('dependent_variable') or "")]
        column = next((c for c in wanted if c and c in before_df), None)
        if column is None:
            numeric = [c for c in before_df.columns
                       if pd.api.types.is_numeric_dtype(before_df[c])]
            column = numeric[-1] if numeric else None
        if column is None:
            print("the response distribution panel was not drawn: the "
                  "aggregated table carries no numeric response column")
            return
        # PYQTGRAPH, so the panel in the tab and the panel in the run
        # folder are one scene. `fast_panel` overlays the two distributions
        # as outlines on one pair of axes -- two shapes on separate axes
        # with separate scales is the one layout that cannot answer whether
        # the transform moved the shape.
        #
        # The matplotlib version this replaced was wrapped in
        # `figure_style(theme_target())`, which is how a matplotlib artist
        # takes the theme. A pyqtgraph scene takes it from the palette when
        # it is built, so there is nothing left for that context to do.
        _draw_response_panel_in_pyqtgraph(
            before_df[column].to_numpy(dtype=float),
            str(settings.get('transform') or 'none'), str(column),
            settings.get('src'))
    except Exception as error:                                   # noqa: BLE001
        # A diagnostic figure must not invalidate the regression run, but a
        # rendering failure remains visible in the run log.
        print(f"the response distribution panel could not be drawn "
              f"({type(error).__name__}: {error}); the run is unaffected")


def check_distribution(y, epsilon=1e-6):
    """Check the distribution of ``y`` and recommend a regression type.

    :param y: Response vector.
    :param epsilon: How close to 0 or 1 a value may sit before it counts
        as a boundary case. Default ``1e-6``.
    :returns: One of ``'logit'``, ``'quasi_binomial'``, ``'beta'``,
        ``'ols'`` or ``'glm'``, as accepted by :func:`regression`'s
        ``regression_type``.
    """
    
    # Check if the dependent variable is binary (only 0 and 1)
    if np.all((y == 0) | (y == 1)):
        print("Detected binary data.")
        return 'logit'
    
    # Continuous data between 0 and 1 (excluding exact 0 and 1)
    elif (y > 0).all() and (y < 1).all():
        # Check if the data is close to 0 or 1 (boundary issues)
        if np.any((y < epsilon) | (y > 1 - epsilon)):
            print("Detected continuous data near 0 or 1. Using quasi-binomial.")
            return 'quasi_binomial'
        else:
            print("Detected continuous data between 0 and 1 (no boundary issues). Using beta regression.")
            return 'beta'
    
    # Continuous data between 0 and 1 (including exact 0 or 1)
    elif (y >= 0).all() and (y <= 1).all():
        print("Detected continuous data with boundary values (0 or 1). Using quasi-binomial.")
        return 'quasi_binomial'
    
    # Check if the data is normally distributed for OLS suitability
    stat, p_value = stats.normaltest(y)  # D’Agostino and Pearson’s test for normality
    print(f"Normality test p-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("Detected normally distributed data. Using OLS.")
        return 'ols'
    
    # Check if the data fits a Beta distribution
    if stats.kstest(y, 'beta', args=(2, 2)).pvalue > 0.05:
        # Check if the data is close to 0 or 1 (boundary issues)
        if np.any((y < epsilon) | (y > 1 - epsilon)):
            print("Detected continuous data near 0 or 1. Using quasi-binomial.")
            return 'quasi_binomial'
        else:
            print("Detected continuous data between 0 and 1 (no boundary issues). Using beta regression.")
            return 'beta'
    
    print("Detected non-normally distributed data. Using GLM.")
    return 'glm'

MIN_POISSON_SAMPLES = 8


def _validate_poisson_response(y, X=None, minimum_samples=MIN_POISSON_SAMPLES,
                               model="Poisson regression"):
    """Validate a response before fitting a Poisson GLM.

    Poisson endog must contain finite, non-negative integer counts. At least
    eight observations and one residual degree of freedom are required so
    family detection and coefficient inference are not performed on an
    undersized or saturated design.

    :param y: One-dimensional count response.
    :param X: Optional design matrix used to determine the parameter count.
    :param minimum_samples: Absolute observation floor.
    :param model: What to call the model in the refusal. `horseshoe` is a
        sparse Poisson GLM and reaches this validator too, so a user who
        chose it was told "Poisson regression requires integer count data" --
        an error naming a model they did not ask for, followed by advice
        ("use a continuous response model") for a choice they never made.
    :returns: The validated response as a one-dimensional float array.
    :raises ValueError: If the response or sample size is invalid.
    """
    try:
        counts = np.asarray(y, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model} requires numeric count data."
        ) from exc

    if not np.isfinite(counts).all():
        raise ValueError(
            f"{model} requires finite count data; remove or impute "
            "NaN and infinite response values before fitting."
        )
    if np.any(counts < 0):
        raise ValueError(
            f"{model} requires non-negative count data; negative "
            "response values are not valid counts."
        )
    if not np.all(np.isclose(counts, np.rint(counts), rtol=0, atol=1e-8)):
        raise ValueError(
            f"{model} requires integer count data; use a continuous "
            "response model for fractional values."
        )
    if not np.any(counts > 0):
        raise ValueError(
            f"{model} requires at least one positive count; an "
            "all-zero response cannot estimate effects."
        )

    n_parameters = 0
    if X is not None:
        x_shape = np.shape(X)
        if not x_shape or x_shape[0] != counts.size:
            raise ValueError(
                f"{model} requires X and y to contain the same "
                f"number of observations; got {x_shape[0] if x_shape else 0} "
                f"and {counts.size}."
            )
        n_parameters = 1 if len(x_shape) == 1 else int(x_shape[1])

    required = max(int(minimum_samples), n_parameters + 1)
    if counts.size < required:
        raise ValueError(
            f"{model} has too few observations: "
            f"received {counts.size}, but at least {required} are required "
            f"for {n_parameters} model parameters."
        )
    return counts


#: Transforms that are THEMSELVES a link function. Applying one and then
#: handing the result to a family whose link does the same job transforms the
#: response twice, and the model fits a quantity nothing measures.
LINK_LIKE_TRANSFORMS = ('log', 'logit')


def double_transform_warning(name, transform, family) -> str:
    """Describe a response transform that is compounded by the model link.

    For example, a log-transformed response passed to a family with a logit
    link fits ``logit(log(y))``. The function returns an actionable warning
    before fitting; an identity link or a response without a link-like
    transform returns an empty string.

    :param name: Response name shown in the warning.
    :param transform: Transform already applied to the response.
    :param family: Statsmodels family whose link will be inspected.
    :returns: Warning text, or ``""`` when the transforms do not compound.
    """
    kind = str(transform or '').strip().lower()
    if kind not in LINK_LIKE_TRANSFORMS:
        return ""
    link = type(getattr(family, 'link', None)).__name__
    if link in ('', 'NoneType', 'Identity', 'identity'):
        return ""
    # WHAT THIS SENTENCE HAS TO CARRY, and it lost three of them once:
    #
    #   * that the response is transformed TWICE -- the fault, in the word a
    #     reader will remember it by;
    #   * the composed function, so it can be checked;
    #   * THE SYMPTOM. A user does not notice a double transform; they
    #     notice a pseudo-R-squared of -20.3 and have no way to connect the
    #     two. Naming McFadden here is what connects them, and it is the
    #     whole point of the warning existing at all;
    #   * what to do about it. spaCR now resolves this itself -- the
    #     response is fitted as measured and the family's link does the
    #     transforming once -- so the warning explains what was avoided
    #     rather than offering a choice that no longer exists.
    return (
        f"  Warning: {name or 'the response'} would be transformed TWICE. "
        f"transform={kind!r} has already been applied to it, and the "
        f"selected family also carries a {link} link, so the model fits "
        f"{link.lower()}({kind}(y)) -- which is usually why McFadden's "
        f"R-squared comes back negative and meaningless. "
        f"spaCR will drop the transform and let the family's link do the "
        f"work, so it is applied once. To fit the TRANSFORMED response "
        f"instead, use regression_type='ols' on the transformed response."
    )


# A link-like transform combined with a non-identity family link transforms
# the response twice.  The two valid resolutions answer different questions,
# so the caller selects the response scale explicitly.
#
# A log-transformed response handed to a family with a logit link fits
# logit(log(y)), which nothing measures. There are two defensible fixes and
# they are different science, so spaCR offers both rather than choosing:
#
#   'untransformed'  choose the family on the measured response and let the
#                    link do the transforming. `pred` is a proportion, so
#                    Binomial/Logit is right for it and the log is redundant.
#
#   'transformed'    keep the transform and fit an identity link. log(p) is an
#                    ordinary continuous response and a Gaussian model of it is
#                    a standard thing to fit.
#
#   'warn'           what spaCR did before this setting existed: fit the
#                    transformed response, choose the family from it, and print
#                    the warning. Retained for reproducibility of earlier runs;
#                    new analyses should choose one of the two explicit scales.
#: A link-like transform and a GLM's own link are the same operation asked
#: for twice, and there is one right answer: fit the response AS MEASURED
#: and let the family's link do the transforming, once.
#:
#: This used to be a setting with three values. The other two were not
#: choices worth offering: 'transformed' fitted a Gaussian identity model of
#: the transformed response, which is an ordinary linear model and exactly
#: what regression_type='ols' already gives; and 'warn' kept the double
#: transform so an older result could be reproduced. A user who wants the
#: first still has it under its own name, and the second was a bug being
#: preserved.


def resolve_glm_transform_conflict(dependent_variable, transform='',
                                   available=(), regression_type='glm'):
    """Resolve a transform/family-link conflict before fitting a GLM.

    :param dependent_variable: the response column as it stands -- already
        the transformed one, if a transform was asked for.
    :param transform: the transform already applied.
    :param available: the column names the frame actually holds. Used to
        confirm the untransformed column is there before switching to it.
    :param regression_type: only ``'glm'`` chooses its own family, so only
        ``'glm'`` has this conflict to resolve. Everything else is returned
        unchanged.
    :returns: ``(column, transform_in_effect, force_identity, note)``.
        ``note`` explains any scale change for the run log.

    A transform that is not link-like, or a regression type other than
    ``'glm'``, returns the response unchanged. Resolving the column before the
    design matrices are built keeps the fit, coefficients, diagnostics, and
    goodness-of-fit summary on the same scale.
    """
    kind = str(transform or '').strip().lower()
    column = str(dependent_variable)
    if kind not in LINK_LIKE_TRANSFORMS or str(regression_type) != 'glm':
        return column, transform, False, ''

    # Fit the response as measured and let the family's link do the work.
    prefix = f"{kind}_"
    raw = column[len(prefix):] if column.startswith(prefix) else ''
    if not raw or raw not in set(available):
        return column, transform, False, (
            f"  the response before transform={kind!r} is not in the frame "
            f"(looked for {raw or 'a column without the prefix'}), so "
            f"{column} is fitted as it stands and the transform is applied "
            f"twice -- once by hand and once by the family's link.")
    return raw, '', False, (
        f"  fitting the measured response {raw} and ignoring "
        f"transform={kind!r}: the family's own link does the transforming, "
        f"so it is applied once instead of twice.")


def pick_glm_family_and_link(y, name="", transform=""):
    """Select the GLM family and link that suit the response.

    Used by ``regression_type='glm'`` to choose a family from the data rather
    than from the user.

    :param y: Response vector.
    :param name: Response-column name printed with the selected family. The
        name makes clear whether the family was chosen from a derived scale.
    :param transform: the transform already applied, printed with the name and
        checked for the double transform of :func:`double_transform_warning`.
    :returns: A ``statsmodels`` family instance with its link set.
    :raises ValueError: only through :func:`_validate_poisson_response`, when
        the response looks like counts but cannot be one.
    """
    family = _choose_glm_family(y, name=name, transform=transform)
    # AFTER the choice, because the warning depends on which link was picked
    # -- and BEFORE the fit, because by the time this reaches the summary the
    # fit has already run on a doubly transformed response.
    warning = double_transform_warning(name, transform, family)
    if warning:
        print(warning)
    return family


def _choose_glm_family(y, name="", transform=""):
    """The family and link, and the sentence saying which scale was examined."""
    values = np.asarray(y, dtype=float).reshape(-1)
    scale = str(name or 'the response')
    if transform:
        scale = f"{scale} (after transform={str(transform)!r})"

    if np.all((values == 0) | (values == 1)):
        print(f"{scale} is binary. Using Binomial family with Logit link.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    elif (values > 0).all() and (values < 1).all():
        # A proportion strictly inside (0, 1) is a binomial mean, and a
        # binomial GLM with a logit link is the standard model for it. This
        # branch used to raise "Use BetaModel for this data; GLM is not
        # applicable", which was not a principled refusal: the very next
        # branch fits exactly this family as soon as a single well sits at 0.0
        # or 1.0, so one boundary well flipped the same screen from "not
        # applicable" to "fine". Beta regression IS usually the better model
        # here - hence the recommendation - but it is a recommendation, and
        # regression_type='beta' is how you take it.
        print(f"{scale} is strictly between 0 and 1. Using Binomial family "
              f"with Logit link; consider regression_type='beta', which models "
              f"the variance of a bounded response directly, or "
              f"'quasi_binomial' if the wells are overdispersed.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    elif (values >= 0).all() and (values <= 1).all():
        print(f"{scale} is between 0 and 1 including the boundaries. "
              f"Using Quasi-Binomial.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    if (values >= 0).all() and np.all(values.astype(int) == values):
        # Family selection may be used for a short preview without fitting.
        # The actual GLM boundary below enforces the sample/design minimum.
        _validate_poisson_response(values, minimum_samples=1)
        print(f"{scale} looks like counts. Using Poisson with Log link.")
        return sm.families.Poisson(link=sm.families.links.Log())

    stat, p_value = normaltest(values)
    print(f"Normality test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"{scale} is normally distributed. Using Gaussian with "
              f"Identity link.")
        return sm.families.Gaussian(link=sm.families.links.Identity())

    if ((values > 0).all()
            and kstest(values, 'invgauss', args=(1,)).pvalue > 0.05):
        print(f"{scale} looks inverse Gaussian. Using InverseGaussian "
              f"with Log link.")
        return sm.families.InverseGaussian(link=sm.families.links.Log())

    if (values >= 0).all():
        print(f"{scale} looks like overdispersed counts. Using Negative "
              f"Binomial with Log link.")
        return sm.families.NegativeBinomial(link=sm.families.links.Log())

    print(f"{scale}: no family fitted the shape, so Gaussian with an "
          f"Identity link is used.")
    return sm.families.Gaussian(link=sm.families.links.Identity())

# THE BACKEND TABLES LIVE IN A MODULE THAT IMPORTS NOTHING.
#
# `get_setting_dependencies` reads REGRESSION_SETTINGS_USED to decide which
# widgets on a settings panel apply to each other, and importing it from here
# dragged this module's `from .plot import save_figure` -- and so torch, cv2
# and IPython -- onto the GUI thread every time a panel was built: 2.2s and
# 900 MB to look up a dict of strings.
#
# Re-exported rather than moved-and-forgotten, so every existing
# `from spacr.ml import REGRESSION_TYPES` keeps working.
def binarise_response(y, threshold=None, name='response'):
    """Return ``y`` as a 0/1 vector for a classifier backend, refusing to guess.

    The hinge backend fits a decision boundary, so it needs two classes. There
    are exactly two ways to get them and this function will not invent a
    third:

    * ``y`` already holds exactly two distinct finite values (the usual case:
      a per-object class call aggregated to a well, or a 0/1 score). The lower
      value becomes 0 and the higher becomes 1, so the sign of every
      coefficient answers "does this gRNA push wells towards the HIGHER
      class", which is the same direction the continuous models report.
    * ``threshold`` is given explicitly, and ``y > threshold`` becomes 1.

    A continuous response with no threshold is REFUSED. Picking a cut for the
    user — the mean, the median, 0.5 — would silently redefine the hypothesis
    being tested: on a screen whose well scores run 0.2-0.8 a median split
    calls half the plate positive by construction, and the resulting hit list
    is a plausible, unfalsifiable artefact of the split.

    :param y: Response vector (array, Series or single-column frame).
    :param threshold: Explicit cut; values strictly greater become 1.
    :param name: Name used in error messages, for a legible failure.
    :returns: ``numpy`` float array of 0.0/1.0, same length as ``y``.
    :raises ValueError: if ``y`` is continuous and no ``threshold`` is given,
        if a given ``threshold`` puts every observation in one class, or if
        ``y`` holds fewer than two distinct values.

    Example:
        .. code-block:: python

            binarise_response([0, 1, 1, 0])            # -> [0., 1., 1., 0.]
            binarise_response([2, 5, 5], )             # -> [0., 1., 1.]
            binarise_response([0.2, 0.6], threshold=0.4)   # -> [0., 1.]
    """
    values = np.asarray(y, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(
            f"hinge regression requires a finite {name}; remove or impute the "
            f"NaN/infinite values before fitting.")

    # A BLANK BOX IS NO CUT, not a cut spelled ''. Both callers can be
    # reached from the panel, and `float('')` is not an error anybody can act
    # on. Cut here as well as in `regression_model` because the QC path
    # (`_write_regression_qc`) comes in by the other door.
    if _left_blank(threshold):
        threshold = None

    if threshold is not None:
        cut = float(threshold)
        binary = (values > cut).astype(float)
        n_positive = int(binary.sum())
        if n_positive == 0 or n_positive == binary.size:
            raise ValueError(
                f"hinge_threshold={cut!r} puts all {binary.size} observations "
                f"in one class ({name} range "
                f"{values.min():.6g}-{values.max():.6g}); a one-class response "
                f"has no decision boundary to fit.")
        return binary

    unique = np.unique(values)
    if unique.size == 2:
        return (values == unique[1]).astype(float)
    if unique.size < 2:
        raise ValueError(
            f"hinge regression needs two classes but {name} holds the single "
            f"value {unique[0]!r}.")
    raise ValueError(
        f"hinge regression needs a binary {name}, but it holds "
        f"{unique.size} distinct values in "
        f"{values.min():.6g}-{values.max():.6g}. Set hinge_threshold to the "
        f"cut you mean (values strictly above it are the positive class), or "
        f"choose a model for a continuous response ('ols', 'beta', "
        f"'quantile'). spaCR will not pick the cut for you: a split chosen by "
        f"the software decides the hypothesis, not the biology.")


def _left_blank(value) -> bool:
    """Whether a policed setting was left empty rather than answered.

    None is the usual empty; ``''`` is what a Qt line edit and a saved
    settings CSV produce for the same untouched box; whitespace is what a
    hand-edited CSV produces. None of the policed settings takes a string
    value, so a blank one can only mean "not answered".

    AND NaN, which is the FOURTH spelling of empty and the one that
    actually reaches this function. `pandas.read_csv` turns an empty cell
    into `float('nan')`, so a settings CSV with `hinge_threshold,` on a line
    -- which is what a saved file looks like for every box the user did not
    fill -- arrived here as a float. It is not None and not a str, so it
    read as "answered", and an ordinary OLS run was refused with

        regression_type='ols' does not read hinge_threshold=nan

    about a value nobody typed. NaN is never a threshold, a covariance type
    or a quantile, so there is no reading of it that means "answered".
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(value != value)      # NaN is the only value unequal to itself
    except Exception:                                        # noqa: BLE001
        return False


def _reject_unused_settings(regression_type, supplied):
    """Raise when a setting the chosen backend cannot read was set anyway.

    ``supplied`` maps a setting name to ``(value, default)``. A value equal to
    its default is "not asked for" and passes; anything else must appear in
    :data:`REGRESSION_SETTINGS_USED` for this type.

    Comparing against the default is what makes this usable from a GUI, which
    posts every widget on the panel whether or not the user touched it.

    A BLANK IS NOT A REQUEST. An empty box in the panel, and the empty cell
    a saved settings CSV writes for it, both arrive here as ``''`` -- which
    is not equal to a default of ``None`` and was therefore refused. The
    symptom was that the screen's OWN saved settings could not be reloaded
    and refitted under a different regression type: `hinge_threshold` had
    never been typed into, and switching to 'ols' raised on it.

    :param regression_type: The backend about to be fitted.
    :param supplied: ``{name: (value, default)}`` for the policed settings.
    :raises ValueError: naming the setting, the type and the alternative.
    """
    used = REGRESSION_SETTINGS_USED.get(regression_type, ())
    for name, (value, default) in supplied.items():
        if name in used or value == default or _left_blank(value):
            continue
        raise ValueError(
            f"regression_type={regression_type!r} does not read {name}="
            f"{value!r}: {_SETTING_NOT_APPLICABLE[name]} Leave {name} at its "
            f"default ({default!r}), or choose a regression type that uses it "
            f"({', '.join(t for t in REGRESSION_TYPES if name in REGRESSION_SETTINGS_USED[t]) or 'none'}).")


#: Why each policed setting does nothing for the types that do not list it.
#: Split out of :func:`_reject_unused_settings` so the message names the
#: actual reason instead of "not supported".
_SETTING_NOT_APPLICABLE = {
    'alpha': "it is the penalty weight of a penalised fit, and this model is "
             "unpenalised, so the number would change nothing.",
    'l1_ratio': "it splits a penalty between L1 and L2, and only 'elasticnet' "
                "has both.",
    'cov_type': "it selects a sandwich covariance estimator on a likelihood "
                "fit; sklearn's penalised estimators and the robust/quantile "
                "fits do not expose one, so the standard errors would come "
                "from somewhere other than the label suggests.",
    'quantile': "it is the quantile of the conditional distribution being "
                "fitted, which only 'quantile' regression has; every other "
                "model fits the mean (or the median, for 'rlm').",
    'hinge_threshold': "it is the cut that turns a continuous response into "
                       "the two classes a hinge loss separates; no other "
                       "model classifies.",
    'spline_knots': "it sets how many knots each CONTINUOUS covariate's "
                    "basis gets, and only the spline fit builds one; the "
                    "guide columns are untouched either way.",
    'spline_degree': "it sets the polynomial degree of that basis, and only "
                     "the spline fit builds one.",
    'huber_t': "it is the residual, in units of the estimated scale, at which "
               "Huber's loss switches from squared to linear; only the robust "
               "fits have that switch.",
    'lasso_n_boot': "it sizes the bootstrap that ranks features by SELECTION "
                    "frequency, and only a penalty that sets coefficients to "
                    "exactly zero selects anything - ridge keeps every "
                    "feature, so its selection frequency is 1.0 by "
                    "construction, and the likelihood fits are ranked by their "
                    "own p-values.",
    'lasso_selection_threshold': "it is the cut on that same selection "
                                 "frequency, which only the sparse penalties "
                                 "produce.",
    'hinge_n_boot': "it sizes the bootstrap that stands in for the standard "
                    "errors an SVM does not have; every other model reports "
                    "its own inference.",
    'group_lasso_lambda': "it is the block penalty of the group lasso, "
                          "measured against THIS design's own "
                          "group_lasso.max_lambda, so it is not the same "
                          "quantity as 'alpha' and no other model has a "
                          "block to penalise.",
    'rra_alpha': "it is the top fraction of the guide ranking alpha-RRA "
                 "aggregates over, and only 'rra' ranks anything; every other "
                 "model estimates coefficients jointly.",
    'rra_permutations': "it sizes the permutation null RRA's P value is read "
                        "off, and every other model gets its P value from a "
                        "likelihood, a posterior or a bootstrap.",
}



# ---------------------------------------------------------------------------
# THE ABSORBING BACKEND (instruction 141 G.1) -- pyfixest
# ---------------------------------------------------------------------------

#: The design factors :mod:`pyfixest` absorbs instead of carrying as columns.
#:
#: `prepare_formula` puts ``rowID`` and ``columnID`` in the model as FIXED
#: effects, so patsy dummy-codes them. They are nuisance terms --
#: `process_model_coefficients` drops every one of them from the coefficient
#: table before anybody reads it -- and a nuisance term that is never reported
#: does not have to be a column. Absorbing it by alternating projections
#: (Frisch-Waugh-Lovell) leaves the coefficients that ARE reported unchanged
#: to the last digit and takes the solve down with the design.
#:
#: ``screenID`` is deliberately excluded because it blocks combined-screen
#: fits on the experiment and may be useful in the coefficient table.
_ABSORBED_FIXED_EFFECTS = ('rowID', 'columnID')


def _absorbed_factor_codes(X, factors=_ABSORBED_FIXED_EFFECTS):
    """Recover the level of each factor patsy dummy-coded, per observation.

    patsy writes a k-level factor as k-1 indicator columns against a dropped
    reference level, so the reference is the row where every one of them is
    zero. Reading the codes back out of the design is what lets an absorbing
    backend be handed the SAME matrix statsmodels was, rather than the raw
    frame -- there is then no second construction of the design to disagree
    with the first.

    :param X: the design DataFrame patsy built.
    :param factors: term names to look for, each dummy-coded as
        ``name[T.level]``.
    :returns: ``(codes, names, n_absorbed_params)`` -- an ``(n, k)`` uint64
        array of level codes, the factors actually found, and how many
        parameters of the dense design they account for (the intercept plus
        each factor's k-1 indicators, which is what the residual degrees of
        freedom must still be charged for). ``codes`` is ``None`` when no
        factor is present.
    :raises ValueError: when a factor's indicator columns are not 0/1, which
        means the column named ``rowID[...]`` is not a dummy and absorbing it
        would silently fit a different model.
    """
    columns = list(getattr(X, 'columns', []))
    blocks, names = [], []
    n_params = 1 if 'Intercept' in columns else 0
    for factor in factors:
        prefix = f'{factor}['
        block = [c for c in columns if str(c).startswith(prefix)]
        if not block:
            # A factor with ONE level emits no columns at all -- patsy folds
            # it into the intercept. A screen on a single plate row is the
            # documented case (see prepare_formula), and it is not an error.
            continue
        values = np.asarray(X[block], dtype=float)
        if not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError(
                f"the design's {factor!r} columns are not 0/1 indicators, so "
                f"they cannot be a dummy-coded factor and absorbing them "
                f"would fit a different model. Columns: {block[:4]}.")
        if np.any(values.sum(axis=1) > 1):
            raise ValueError(
                f"a row of the design is in more than one {factor!r} level, "
                f"so {factor!r} is not a factor and cannot be absorbed.")
        blocks.append(np.where(values.any(axis=1), values.argmax(axis=1) + 1,
                               0).astype(np.uint64))
        names.append(factor)
        n_params += len(block)
    if not blocks:
        return None, [], n_params
    return np.column_stack(blocks), names, n_params


class _AbsorbedDesign:
    """The design an absorbed fit was run on, in statsmodels' shape.

    :mod:`spacr.regression_qc` recovers a design from ``results.model.exog``
    and decides the scale rule from ``type(results.model).__mro__`` (see
    ``regression_qc._model_kind``), so an absorbed fit that carried neither
    would lose the diagnostics tab. It carries the FULL design -- the one with
    the dummy columns still in it -- because that is the model that was
    fitted; absorption is how it was solved, not what it was.
    """

    def __init__(self, endog, exog, exog_names):
        self.endog = np.asarray(endog, dtype=float).reshape(-1)
        self.exog = np.asarray(exog, dtype=float)
        self.exog_names = list(exog_names)


#: ``kind`` -> the design class :mod:`spacr.regression_qc` resolves BY NAME.
#:
#: ``regression_qc._model_kind`` walks ``type(results.model).__mro__`` looking
#: for a class called ``OLS`` or ``WLS``, because ``sm.OLS(...).fit()`` and
#: ``sm.WLS(...).fit()`` share one results class and only ``results.model``
#: tells them apart. An absorbed least-squares fit obeys exactly the scale
#: rule that name selects -- ``scale`` is RSS / (n - p) over the FULL
#: parameter count, and the weighted version is in the metric of
#: ``sqrt(w) * (y - fitted)`` -- so it answers to it. Built with ``type()``
#: rather than written as two ``class`` statements so ``spacr.ml`` does not
#: grow public names ``OLS`` and ``WLS`` that would read as statsmodels'.
_ABSORBED_DESIGN_CLASSES = {
    name: type(name, (_AbsorbedDesign,),
               {'__doc__': f"The design of an absorbed {name} fit."})
    for name in ('OLS', 'WLS')
}


class _AbsorbedLeastSquaresResults:
    """A least-squares fit solved by absorbing its nuisance factors.

    Reports what :func:`process_model_coefficients` and
    :mod:`spacr.regression_qc` read off a statsmodels results object --
    ``params``, ``bse``, ``pvalues``, ``tvalues``, ``resid``,
    ``fittedvalues``, ``scale``, ``df_resid`` -- for the coefficients that
    SURVIVE absorption. The absorbed ones have no row, which is the one way
    this fit's answer differs from statsmodels' and is why
    ``REGRESSION_BACKENDS['pyfixest']['differs']`` says so.

    The residuals and ``scale`` are the FULL model's, not the demeaned
    regression's: Frisch-Waugh-Lovell makes them the same vector, and the
    degrees of freedom are charged for every absorbed parameter, so the
    standard errors match statsmodels to the last digit rather than to a
    tolerance.
    """

    def __init__(self, params, bse, pvalues, resid, fitted, scale,
                 df_model, df_resid, nobs, model, converged, absorbed,
                 rsquared):
        self.params = params
        self.bse = bse
        self.pvalues = pvalues
        with np.errstate(divide='ignore', invalid='ignore'):
            self.tvalues = params / bse
        self.resid = resid
        self.fittedvalues = fitted
        self.scale = float(scale)
        self.df_model = float(df_model)
        self.df_resid = float(df_resid)
        self.nobs = float(nobs)
        self.model = model
        self.converged = bool(converged)
        self.absorbed = tuple(absorbed)
        self.rsquared = float(rsquared)

    def predict(self, exog=None):
        """Fitted values. ``exog`` is accepted and ignored, as sm's OLS does
        for the in-sample case; an absorbed fit cannot predict a new row
        because it never estimated the absorbed levels."""
        if exog is None:
            return self.fittedvalues
        raise ValueError(
            "an absorbed fit did not estimate the levels of "
            f"{', '.join(self.absorbed) or 'its nuisance factors'}, so it "
            "cannot predict a row it has not seen. Fit with "
            "regression_backend='statsmodels' if you need out-of-sample "
            "predictions.")

    def summary(self):
        """A text summary, so :func:`_write_model_summary` still has one."""
        return _AbsorbedSummary(self)


class _AbsorbedSummary:
    """``.as_text()`` for :class:`_AbsorbedLeastSquaresResults`."""

    def __init__(self, results):
        self._results = results

    def as_text(self):
        r = self._results
        lines = [
            "Absorbed least squares (pyfixest alternating projections)",
            f"  observations          {int(r.nobs)}",
            f"  reported coefficients {len(r.params)}",
            f"  absorbed factors      {', '.join(r.absorbed) or 'none'}",
            f"  residual df           {int(r.df_resid)}",
            f"  error variance        {r.scale:.6g}",
            f"  R-squared             {r.rsquared:.6f}",
            f"  demeaning converged   {r.converged}",
            "",
            "coefficient / std err / t / P>|t|",
        ]
        for name in r.params.index:
            lines.append(f"  {name}  {r.params[name]:.6g}  "
                         f"{r.bse[name]:.6g}  {r.tvalues[name]:.4f}  "
                         f"{r.pvalues[name]:.4g}")
        return "\n".join(lines)

    def __str__(self):
        return self.as_text()


def _fit_absorbed_least_squares(X, y, weights=None, kind='OLS'):
    """Least squares with ``rowID``/``columnID`` absorbed, via pyfixest.

    Use ``pyfixest.core.demean`` to project out row and column factors, then
    solve the remaining normal equations by Cholesky decomposition. This
    avoids including high-cardinality nuisance dummies in the dense solve
    while retaining a coefficient table compatible with the statsmodels path.

    :param X: the design DataFrame, dummy columns included.
    :param y: the response.
    :param weights: per-observation weights for a WLS fit, or ``None``.
    :param kind: ``'OLS'`` or ``'WLS'``, which is what
        :mod:`spacr.regression_qc` reads to pick its scale rule.
    :returns: :class:`_AbsorbedLeastSquaresResults`.
    :raises ValueError: when the design carries no absorbable factor (there
        is then nothing for this backend to do that statsmodels does not do
        better), or when the normal equations are singular.
    """
    columns = list(getattr(X, 'columns', []))
    if not columns:
        raise ValueError(
            "the absorbing backend reads which columns are rowID/columnID "
            "dummies from the design's COLUMN NAMES, so it needs a DataFrame "
            "design; a bare array has no names. Build it with "
            "dmatrices(..., return_type='dataframe'), which is what the "
            "pipeline hands in.")
    codes, absorbed, n_absorbed_params = _absorbed_factor_codes(X)
    if codes is None:
        raise ValueError(
            "regression_backend='pyfixest' absorbs the rowID and columnID "
            "fixed effects, and this design has neither -- either "
            "model_plate_position=False took them out of the model or the "
            "screen sits on one row and one column. There is nothing to "
            "absorb, so the fit would be the statsmodels fit with an extra "
            "projection in front of it. Set "
            "regression_backend='statsmodels'.")

    keep = [c for c in columns
            if str(c) != 'Intercept'
            and not any(str(c).startswith(f'{f}[') for f in absorbed)]
    if not keep:
        raise ValueError(
            "every column of this design is an intercept or an absorbed "
            "fixed effect, so the absorbed fit would report no coefficient "
            "at all.")

    y_flat = np.asarray(y, dtype=float).reshape(-1)
    n = y_flat.size
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n:
            raise ValueError(
                f"the absorbed fit was given {w.size} weights for {n} "
                f"observations.")
        if not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError(
                "WLS weights must be finite and positive (they are per-well "
                f"cell counts); got {np.nanmin(w)}-{np.nanmax(w)}.")

    # Degrees of freedom are a property of the design, not of pyfixest's
    # projection.  Check them before importing the optional backend so an
    # invalid request gets the same useful diagnosis on Python 3.9, where
    # pyfixest itself is unavailable.
    p_full = len(keep) + n_absorbed_params
    df_resid = n - p_full
    if df_resid <= 0:
        raise ValueError(
            f"the design has {p_full} parameters ({len(keep)} reported plus "
            f"{n_absorbed_params} absorbed) for {n} observations, so there "
            f"are no residual degrees of freedom to estimate a standard "
            f"error from.")

    # Import only after backend-independent validation. pyfixest requires
    # Python >=3.10, while spaCR's supported floor is Python 3.9.
    from pyfixest.core.demean import demean

    stacked = np.asfortranarray(
        np.column_stack([y_flat, np.asarray(X[keep], dtype=float)]))
    # tol is on the alternating projections, not on the answer: 1e-10 is
    # tighter than the 1e-6 pyfixest defaults to, because the agreement this
    # backend is held to (instruction 141 D) is against statsmodels' exact
    # solve rather than against another approximation.
    demeaned, converged = demean(stacked, codes, w, tol=1e-10)
    if not converged:
        raise ValueError(
            "the alternating projections that absorb "
            f"{', '.join(absorbed)} did not converge, so the design was "
            "never fully partialled out and the coefficients would not be "
            "the least-squares ones. Fit with "
            "regression_backend='statsmodels'.")
    y_d = demeaned[:, 0]
    X_d = demeaned[:, 1:]

    # WEIGHTED normal equations. `demean` already takes weighted group means,
    # which is the weighted Frisch-Waugh-Lovell projection, so the only thing
    # left is to weight the cross-products.
    Xw = X_d * w[:, None]
    xtx = X_d.T @ Xw
    xty = Xw.T @ y_d
    # RANK BEFORE SOLVE, not the solver's exception. LAPACK builds disagree
    # about a singular system: some raise, and some return one arbitrary
    # member of an infinite solution set. Diagnosing rank first makes the
    # refusal the same everywhere, which matters because the alternative is
    # a coefficient table that looks fine and is not identified.
    _rank = int(np.linalg.matrix_rank(xtx))
    if _rank < xtx.shape[0]:
        raise ValueError(
            f"the absorbed design's normal equations are singular "
            f"(rank {_rank} of {xtx.shape[0]}), so its {len(keep)} "
            f"coefficients are not identified. That is a rank-deficient "
            f"design, not a backend failure: statsmodels answers the same "
            f"design with a pseudo-inverse, which picks one arbitrary "
            f"solution out of infinitely many.")
    beta = np.linalg.solve(xtx, xty)

    resid = y_d - X_d @ beta
    # DEGREES OF FREEDOM ARE CHARGED FOR WHAT WAS ABSORBED. n - p_kept alone
    # would report the standard errors of a model that never had the 36
    # nuisance parameters, which is smaller than the truth and is exactly the
    # way an absorbing fit gets its inference wrong.
    rss = float(resid @ (resid * w))
    scale = rss / df_resid
    cov = scale * np.linalg.inv(xtx)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = np.where(se > 0, beta / se, 0.0)
    p_values = 2.0 * st.t.sf(np.abs(t_stats), df_resid)

    names = [str(c) for c in keep]
    params = pd.Series(beta, index=names)
    # The residual of the DEMEANED regression is the residual of the full
    # one -- that is what Frisch-Waugh-Lovell says -- so the fitted values
    # follow from it and the diagnostics see the same numbers statsmodels
    # would have shown them.
    full_resid = resid
    fitted = y_flat - full_resid
    centred = y_flat - np.average(y_flat, weights=w)
    tss = float(centred @ (centred * w))
    rsquared = 1.0 - rss / tss if tss > 0 else float('nan')

    model = _ABSORBED_DESIGN_CLASSES[kind](
        y_flat, np.asarray(X, dtype=float), [str(c) for c in columns])
    return _AbsorbedLeastSquaresResults(
        params=params,
        bse=pd.Series(se, index=names),
        pvalues=pd.Series(p_values, index=names),
        resid=full_resid, fitted=fitted, scale=scale,
        df_model=p_full - 1, df_resid=df_resid, nobs=n, model=model,
        converged=converged, absorbed=absorbed, rsquared=rsquared)


# ---------------------------------------------------------------------------
# THE FAST-GLM BACKEND (instruction 141 G.3) -- glum
# ---------------------------------------------------------------------------

#: ``regression_type`` -> the glum family, and how the fit is set up.
#:
#: ``probit`` IS NOT HERE, and that is measured rather than an omission: glum
#: 3.4 ships ``IdentityLink``, ``LogLink``, ``LogitLink``, ``CloglogLink`` and
#: ``TweedieLink`` and has no probit link at all, so a probit fitted "by glum"
#: could only be a logit under the wrong label. ``quasi_binomial`` is not here
#: either -- statsmodels spells it as a Binomial mean with the dispersion
#: taken from the Pearson chi-square (``scale='X2'``), and glum has no
#: equivalent knob, so its standard errors would be the fixed-dispersion ones
#: on a model chosen BECAUSE its dispersion is free. `backend_status` greys
#: the pair out for exactly these reasons.
_GLUM_FAMILIES = {
    'poisson': 'poisson',
    'logit': 'binomial',
    'glm': None,        # chosen from the response, like the statsmodels path
}


class _GlumResults:
    """A GLM fitted by glum, reporting what statsmodels' GLM results report.

    Form covariance from the canonical-link information matrix,
    ``(X' W X)^-1`` with
    ``W_ii = v_i (dmu/deta)^2 / V(mu_i)`` and dispersion fixed at one. This
    matches the covariance convention used by the statsmodels GLM path rather
    than glum's optional sandwich or finite-sample corrections.
    """

    def __init__(self, params, bse, pvalues, resid, fitted, scale,
                 df_model, df_resid, nobs, model, family, llf,
                 null_deviance, deviance, n_iter, llnull=None):
        self.params = params
        self.bse = bse
        self.pvalues = pvalues
        with np.errstate(divide='ignore', invalid='ignore'):
            self.tvalues = params / bse
        self.resid = resid
        self.resid_response = resid
        self.fittedvalues = fitted
        self.scale = float(scale)
        self.df_model = float(df_model)
        self.df_resid = float(df_resid)
        self.nobs = float(nobs)
        self.model = model
        self.family = family
        self.llf = float(llf)
        self.null_deviance = float(null_deviance)
        # THE NULL LOG-LIKELIHOOD, because that is what the goodness-of-fit
        # line divides by. `fit_quality_note` takes `llnull` and falls back
        # to `null_deviance / -2` -- so a backend that carried only the
        # deviance would print a DIFFERENT McFadden from statsmodels for the
        # identical fit, which is the one thing this class exists not to do.
        # The null model is fitted here anyway; this only carries its answer.
        self.llnull = None if llnull is None else float(llnull)
        self.deviance = float(deviance)
        self.n_iter = int(n_iter)

    def predict(self, exog=None):
        """In-sample fitted values on the RESPONSE scale, as sm's GLM does."""
        if exog is None:
            return self.fittedvalues
        raise ValueError(
            "this GLM was fitted by glum through spaCR's design matrix and "
            "does not carry the link's inverse for a new row. Fit with "
            "regression_backend='statsmodels' if you need to predict.")

    def summary(self):
        return _GlumSummary(self)


class _GlumSummary:
    """``.as_text()`` for :class:`_GlumResults`."""

    def __init__(self, results):
        self._results = results

    def as_text(self):
        r = self._results
        lines = [
            f"Generalized linear model fitted by glum "
            f"({type(r.family).__name__})",
            f"  observations   {int(r.nobs)}",
            f"  coefficients   {len(r.params)}",
            f"  residual df    {int(r.df_resid)}",
            f"  deviance       {r.deviance:.6g}",
            f"  null deviance  {r.null_deviance:.6g}",
            f"  log-likelihood {r.llf:.6g}",
            f"  IRLS steps     {r.n_iter}",
            "",
            "coefficient / std err / z / P>|z|",
        ]
        for name in r.params.index:
            lines.append(f"  {name}  {r.params[name]:.6g}  "
                         f"{r.bse[name]:.6g}  {r.tvalues[name]:.4f}  "
                         f"{r.pvalues[name]:.4g}")
        return "\n".join(lines)

    def __str__(self):
        return self.as_text()


def _glum_information_weights(family, mu, var_weights):
    """``W_ii`` of the GLM information matrix, for the families glum fits.

    For a canonical link ``dmu/deta`` equals the variance function, so the
    weight collapses to ``v_i V(mu_i)``: ``v * mu`` for a log-link Poisson and
    ``v * mu (1 - mu)`` for a logit Binomial. Gaussian identity is ``v``. They
    are written out per family rather than differenced numerically because a
    finite difference here would put its own error into every standard error
    on the volcano.
    """
    weights = np.asarray(var_weights, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if isinstance(family, sm.families.Poisson):
        return weights * mu
    if isinstance(family, sm.families.Binomial):
        return weights * mu * (1.0 - mu)
    if isinstance(family, sm.families.Gaussian):
        return weights
    raise ValueError(
        f"spaCR does not know the information weight for "
        f"{type(family).__name__}, so it cannot form the standard errors of a "
        f"glum fit of it. Fit with regression_backend='statsmodels'.")


def _fit_glum_glm(X, y, regression_type, weights=None, exposure=None):
    """Fit one of the GLM families through glum instead of statsmodels.

    Use glum's IRLS and active-set solver for supported Poisson, binomial, or
    automatically selected GLM families. Small designs may not amortize the
    backend's setup cost; its advantage is intended for wide model matrices.

    :param X: the design DataFrame.
    :param y: the response.
    :param regression_type: one of :data:`_GLUM_FAMILIES`.
    :param weights: per-well cell counts, used as ``var_weights`` by the
        binomial families exactly as the statsmodels path uses them.
    :param exposure: per-well cell counts for the Poisson ``offset(log(.))``.
    :returns: :class:`_GlumResults`.
    :raises ValueError: for a family glum cannot fit, or a response the family
        refuses.
    """
    columns = list(getattr(X, 'columns', []))
    if not columns:
        raise ValueError(
            "the glum backend reports one coefficient per design column and "
            "reads the names from the design, so it needs a DataFrame; a "
            "bare array has no names.")
    design = np.asarray(X, dtype=float)
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    n = y_flat.size

    offset = None
    var_weights = np.ones(n, dtype=float)
    if regression_type == 'poisson':
        _validate_poisson_response(y, X)
        family = sm.families.Poisson(link=sm.families.links.Log())
    elif regression_type == 'logit':
        family = sm.families.Binomial(link=sm.families.links.Logit())
    else:
        family = pick_glm_family_and_link(y)
        if isinstance(family, sm.families.Poisson):
            _validate_poisson_response(y, X)

    if isinstance(family, sm.families.Poisson):
        # THE SAME OFFSET THE STATSMODELS BRANCH USES. Without it the
        # coefficients are effects on the well's headcount rather than on the
        # per-cell rate -- see `_poisson_offset` for the simulation that
        # measured what that costs -- and a backend that dropped it would be
        # answering a different question, not answering the same one faster.
        n_total = None
        if exposure is not None:
            n_total = np.asarray(exposure, dtype=float).reshape(-1)
            if n_total.size != n:
                raise ValueError(
                    f"the Poisson exposure has {n_total.size} entries but "
                    f"the response has {n}; each well must carry its own "
                    f"cell count.")
            if not np.isfinite(n_total).all() or np.any(n_total <= 0):
                raise ValueError(
                    "the Poisson exposure is the well's cell count, so it "
                    f"must be finite and strictly positive; got "
                    f"{np.nanmin(n_total)}-{np.nanmax(n_total)}.")
            offset = np.log(n_total)
        else:
            print("Warning: no per-well cell count reached the Poisson fit, "
                  "so it models the raw count with no offset(log(cell_count)).")
    elif isinstance(family, sm.families.Binomial) and weights is not None:
        var_weights = np.asarray(weights, dtype=float).reshape(-1)
        if var_weights.size != n:
            raise ValueError(
                f"the binomial fit was given {var_weights.size} weights for "
                f"{n} observations.")

    glum_family = {
        'Poisson': 'poisson', 'Binomial': 'binomial', 'Gaussian': 'normal',
    }.get(type(family).__name__)
    if glum_family is None:
        raise ValueError(
            f"regression_backend='glum' cannot fit a "
            f"{type(family).__name__} family; spaCR routes poisson, binomial "
            f"and gaussian through it. Fit with "
            f"regression_backend='statsmodels'.")

    # Keep validation independent of the optional solver. glum requires
    # Python >=3.10, so Python 3.9 callers must still receive spaCR's precise
    # input error instead of an unrelated ModuleNotFoundError.
    from glum import GeneralizedLinearRegressor

    # alpha=0 is the UNPENALISED fit, which is the only one that can agree
    # with statsmodels. fit_intercept=False because patsy already put an
    # 'Intercept' column in the design and a second one would be collinear
    # with it.
    estimator = GeneralizedLinearRegressor(
        family=glum_family, alpha=0, fit_intercept=False,
        gradient_tol=1e-10, max_iter=500)
    fit_kwargs = {}
    if offset is not None:
        fit_kwargs['offset'] = offset
    if weights is not None and isinstance(family, sm.families.Binomial):
        fit_kwargs['sample_weight'] = var_weights
    estimator.fit(design, y_flat, **fit_kwargs)

    beta = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    eta = design @ beta + (0.0 if offset is None else offset)
    mu = family.link.inverse(eta)

    info_w = _glum_information_weights(family, mu, var_weights)
    xtwx = design.T @ (design * info_w[:, None])
    if isinstance(family, sm.families.Gaussian):
        # A Gaussian GLM has a FREE dispersion, and statsmodels estimates it
        # as the Pearson chi-square over the residual degrees of freedom.
        scale = float(np.sum(info_w * (y_flat - mu) ** 2)) / (n - len(beta))
    else:
        scale = 1.0
    try:
        cov = scale * np.linalg.inv(xtwx)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"the glum fit's information matrix is singular ({exc}), so its "
            f"{len(beta)} coefficients are not identified. statsmodels "
            f"answers the same design with a pseudo-inverse, which picks one "
            f"arbitrary solution out of infinitely many.") from exc
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, beta / se, 0.0)
    p_values = 2.0 * st.norm.sf(np.abs(z))

    # THE NULL MODEL IS FITTED, not approximated, because `regression_model`
    # prints McFadden's R2 for 'glm' and 'poisson' off `null_deviance` and a
    # backend that changed that number would have changed a number a reader
    # compares between runs. It is an intercept-only GLM: one column, so it
    # costs nothing next to the fit above.
    null_kwargs = {'family': family}
    if offset is not None:
        null_kwargs['offset'] = offset
    if weights is not None and isinstance(family, sm.families.Binomial):
        null_kwargs['var_weights'] = var_weights
    null_fit = sm.GLM(y_flat, np.ones((n, 1)), **null_kwargs).fit()

    names = [str(c) for c in columns]
    model = _AbsorbedDesign(y_flat, design, names)
    model.__class__ = _GLUM_DESIGN_CLASS
    llf = family.loglike(y_flat, mu, var_weights=var_weights, scale=scale)
    deviance = family.deviance(y_flat, mu, var_weights=var_weights)
    return _GlumResults(
        params=pd.Series(beta, index=names),
        bse=pd.Series(se, index=names),
        pvalues=pd.Series(p_values, index=names),
        resid=y_flat - mu, fitted=mu, scale=scale,
        df_model=len(beta) - 1, df_resid=n - len(beta), nobs=n, model=model,
        family=family, llf=llf, null_deviance=float(null_fit.null_deviance),
        llnull=float(null_fit.llf),
        deviance=deviance, n_iter=int(getattr(estimator, 'n_iter_', 0)))


#: The design class name :mod:`spacr.regression_qc` resolves a GLM by. Its
#: scale rule reads the dispersion off ``model.scale``, which is what
#: :class:`_GlumResults` reports -- 1 for the fixed-dispersion families and
#: the Pearson estimate for Gaussian, exactly as statsmodels does.
_GLUM_DESIGN_CLASS = type('GLM', (_AbsorbedDesign,),
                          {'__doc__': "The design of a glum-fitted GLM."})

def regression_model(X, y, regression_type='ols', groups=None, alpha=1.0,
                     cov_type=None, weights=None, l1_ratio=0.5, quantile=0.5,
                     hinge_threshold=None, huber_t=1.345, exposure=None,
                     spline_knots=4, spline_degree=3,
                     group_lasso_lambda='auto', rra_alpha=0.25,
                     rra_permutations=10000,
                     regression_backend=DEFAULT_REGRESSION_BACKEND,
                     verbose=False, response_name="", transform="",
                     glm_force_identity=False):
    """Dispatch to the requested regression backend and return the fitted model.

    Every name in :data:`REGRESSION_TYPES` is fittable here, and every one of
    them has a matching branch in :func:`process_model_coefficients`, so a
    model that fits can always be turned into a coefficient table.

    The backends, and what each is for:

    ==================================  ========================================================
    ``ols``                             Ordinary least squares on a continuous well response.
    ``wls``                             Weighted least squares; ``weights`` is the well's cell
                                        count, so a well of 400 cells outweighs one of 30.
    ``rlm``/``huber``                   Robust M-estimation (Huber loss). For outlier-heavy
                                        wells: a handful of runaway wells no longer drag the
                                        fit.
    ``glm``                             GLM with the family auto-selected from the response by
                                        :func:`pick_glm_family_and_link`.
    ``poisson``                         Poisson GLM with a log link and
                                        ``offset(log(exposure))``, for per-well counts - so the
                                        coefficients are effects on the per-cell RATE, not on
                                        the well's headcount.
    ``quasi_binomial``                  Binomial GLM whose dispersion is estimated from the
                                        Pearson chi-square, for overdispersed fractions.
    ``beta``                            Beta regression, for a fraction strictly inside (0, 1).
    ``logit``/``probit``                GLM-binomial on a fraction, weighted by cell count.
    ``quantile``                        Quantile regression at ``quantile``; fits the tail of
                                        the response rather than its mean.
    ``mixed``                           Mixed-effects linear model with ``groups`` as the random
                                        intercept.
    ``lasso``/``ridge``/``elasticnet``  Penalised least squares.
    ``hinge``                           Linear SVM (hinge loss) on a binarised response.
    ``horseshoe``                       Sparse Poisson GLM with a horseshoe prior (spaCRPower's
                                        power-analysis model), via :mod:`spacr.power_model`.
    ``group_lasso``                     Penalised least squares with a gene's guide columns
                                        penalised as ONE block, so a gene is selected or dropped
                                        as a set rather than one guide at a time - the penalised
                                        analogue of the mixed model's nesting, via
                                        :mod:`spacr.group_lasso`.
    ``rra``                             MAGeCK-style robust rank aggregation: guides ranked by
                                        their marginal effect, aggregated to the gene BY RANK
                                        with a permutation P value, via :mod:`spacr.rra`. It
                                        forms no joint fit, so the collinearity and the p >> n
                                        width that constrain every backend above do not reach it.
    ==================================  ========================================================

    Settings a backend cannot read are REFUSED, not ignored — see
    :data:`REGRESSION_SETTINGS_USED`.

    :param X: Design matrix (DataFrame; column names become feature names).
    :param y: Response variable.
    :param regression_type: One of :data:`REGRESSION_TYPES`.
    :param regression_backend: WHO fits it -- one of
        :data:`REGRESSION_BACKEND_ORDER`. Default ``'statsmodels'``, which
        produced every existing result. A backend that cannot fit
        ``regression_type`` is REFUSED here with the reason, not ignored:
        the two controls constrain each other in both directions
       , and a settings CSV reaches this function
        without passing a panel that could have greyed the entry out.
    :param groups: Cluster identifiers for the mixed model.
    :param alpha: Penalty weight for ``lasso``/``ridge``/``elasticnet`` and
        the inverse SVM margin for ``hinge``; ``'auto'`` / ``None`` picks it by
        5-fold cross-validation for all four (mean squared error for the
        penalised least-squares three, balanced accuracy for ``hinge``).
    :param cov_type: Covariance estimator for the likelihood fits
        (``'HC0'``..``'HC3'``); ``None`` for classical standard errors.
    :param weights: Per-observation weights - the well's cell count. Used as
        ``var_weights`` by ``logit``/``probit``/``quasi_binomial`` and as the
        WLS weights by ``wls``.
    :param l1_ratio: ``elasticnet`` mix; 1.0 is lasso, 0.0 is ridge.
    :param quantile: Quantile fitted by ``quantile`` regression, in (0, 1).
    :param hinge_threshold: Cut used to binarise a continuous response for
        ``hinge``; see :func:`binarise_response`.
    :param spline_knots: Knots per continuous covariate for ``spline``.
    :param spline_degree: Polynomial degree of that basis; 3 is cubic.
    :param huber_t: Huber tuning constant for ``rlm``/``huber``, in units of
        the estimated residual scale. 1.345 gives 95% efficiency under
        normality.
    :param exposure: Per-observation exposure (the well's cell count) used as
        ``offset(log(exposure))`` by ``horseshoe`` and by ``poisson`` (and by
        ``glm`` when it auto-selects a Poisson family).
    :param group_lasso_lambda: The block penalty for ``group_lasso``. Its own
        key rather than ``alpha`` because it is compared against
        :func:`spacr.group_lasso.max_lambda`, which is a property of the
        design, so a value carried over from a lasso run would mean something
        else here.
    :param rra_alpha: The top fraction of the guide ranking alpha-RRA
        aggregates over. MAGeCK's 0.25, which is what keeps a gene with one
        strong guide and three that did not cut findable.
    :param rra_permutations: Draws per distinct guide count in RRA's
        permutation null; 10,000 puts the smallest reportable P value at 1e-4.
    :returns: Fitted statsmodels / sklearn estimator.
    :raises ValueError: on an unsupported ``regression_type``, or when a
        setting the chosen backend cannot read was set to a non-default value.

    Example:
        .. code-block:: python

            import pandas as pd
            X = pd.DataFrame({'Intercept': 1.0, 'fraction': [0.1, 0.5, 0.9]})
            model = regression_model(X, pd.Series([0.2, 0.4, 0.7]), 'ols')
            model.params['fraction']   # the recovered slope
    """
    if regression_type in UNSUPPORTED_REGRESSION_TYPES:
        raise ValueError(
            f"Unsupported regression type {regression_type}: "
            f"{UNSUPPORTED_REGRESSION_TYPES[regression_type]}")
    if regression_type not in REGRESSION_TYPES:
        raise ValueError(
            f"Unsupported regression type {regression_type}. "
            f"Supported types: {list(REGRESSION_TYPES)}")

    y_flat = np.asarray(y, dtype=float).reshape(-1)
    use_auto_alpha = alpha is None or (isinstance(alpha, str) and alpha == 'auto')

    # AN EMPTY COVARIANCE BOX IS NO COVARIANCE ESTIMATOR, not an estimator
    # named ''. The panel's line edit and a saved settings CSV both write
    # `''` for a box nobody typed in, and three of the branches below pass
    # it on when it "is not None" -- so a logit, probit or quasi_binomial
    # fit from the screen's own saved settings died inside statsmodels with
    # "cov_type not recognized", naming a value the user never chose.
    if _left_blank(cov_type):
        cov_type = None
    # AND AN EMPTY THRESHOLD BOX IS NO THRESHOLD. `hinge` READS
    # hinge_threshold, so nothing refused the blank: it reached
    # `binarise_response`, which asked float('') for a number and died with
    # "could not convert string to float: ''" -- a message with neither the
    # setting's name nor the model's in it.
    if _left_blank(hinge_threshold):
        hinge_threshold = None

    supplied = {
        # 'auto' and None mean "no penalty chosen, cross-validate it", which
        # is not a value an unpenalised model is being asked to honour, so
        # they count as the default here rather than as a request.
        'alpha': 1.0 if use_auto_alpha else alpha,
        'l1_ratio': l1_ratio,
        'cov_type': cov_type,
        'quantile': quantile,
        'hinge_threshold': hinge_threshold,
        'huber_t': huber_t,
        'spline_knots': spline_knots,
        'spline_degree': spline_degree,
        'group_lasso_lambda': group_lasso_lambda,
        'rra_alpha': rra_alpha,
        'rra_permutations': rra_permutations,
    }
    # WHO fits it, checked before WHAT is fitted is dispatched. A backend
    # that cannot fit this family, is not installed, or wants a GPU that is
    # not here fails now rather than after the design has been built.
    backend = _require_backend(regression_type, regression_backend)
    _reject_unused_settings(regression_type, {
        name: (supplied[name], default)
        for name, default in _MODEL_LEVEL_DEFAULTS.items()})

    def _find_best_alpha(model_cls):
        alphas = np.logspace(-5, 5, 100)
        if model_cls == 'lasso':
            cv = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X, y_flat)
        elif model_cls == 'ridge':
            cv = RidgeCV(alphas=alphas, cv=5).fit(X, y_flat)
        elif model_cls == 'elasticnet':
            cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=5,
                              max_iter=10000).fit(X, y_flat)
        else:
            raise ValueError(f"_find_best_alpha called with unknown model_cls={model_cls!r}")
        print(f"Optimal alpha for {model_cls}: {cv.alpha_:.4g} "
              f"(MSE: {mean_squared_error(y_flat, cv.predict(X)):.4f})")
        return cv

    def _glm_binomial(link=None, scale=None):
        family = sm.families.Binomial(link=link) if link else sm.families.Binomial()
        kwargs = {'family': family}
        if weights is not None:
            kwargs['var_weights'] = np.asarray(weights).ravel()
        fit_kwargs = {}
        if scale is not None:
            fit_kwargs['scale'] = scale
        if cov_type is not None:
            fit_kwargs['cov_type'] = cov_type
        return sm.GLM(y, X, **kwargs).fit(**fit_kwargs)

    def _poisson_offset():
        """``log(exposure)``, or None with a warning when there is no exposure.

        A per-well POSITIVE COUNT is not comparable between wells of different
        size: ``process_scores`` sums the response for the count models, so a
        well of 2000 cells contributes roughly four times the count of a well
        of 500 at the identical underlying rate. Modelling that count without
        ``offset(log(Ntotal))`` asks the covariates to explain well size, and
        any covariate correlated with it comes back as a hit. Measured on a
        400-well simulation with a nuisance covariate that drives well size and
        nothing else, and a true rate coefficient of +1.5: without the offset
        the nuisance term came back at +1.88 with p = 0, ahead of the real
        effect; with it, +0.002 with p = 0.90.

        :returns: ``log(exposure)`` aligned with ``y``, or None.
        :raises ValueError: when the exposure is not positive and finite —
            ``log`` of it would be NaN/-inf and every downstream number would
            silently follow.
        """
        if exposure is None:
            print("Warning: no per-well cell count reached the Poisson fit, so "
                  "it models the raw count with no offset(log(cell_count)). "
                  "Wells of different size are then not comparable and any "
                  "covariate correlated with well size will look like a hit. "
                  "Run the scores through process_scores so each well carries "
                  "its cell count.")
            return None
        n_total = np.asarray(exposure, dtype=float).ravel()
        if n_total.size != np.asarray(y, dtype=float).reshape(-1).size:
            raise ValueError(
                f"the Poisson exposure has {n_total.size} entries but the "
                f"response has {np.asarray(y).reshape(-1).size}; each well "
                f"must carry its own cell count.")
        if not np.isfinite(n_total).all() or np.any(n_total <= 0):
            raise ValueError(
                "the Poisson exposure is the well's cell count, so it must be "
                f"finite and strictly positive; got "
                f"{np.nanmin(n_total)}-{np.nanmax(n_total)}. A well with no "
                f"cells has no rate to estimate and must be filtered out "
                f"(min_cell_count) rather than offset by log(0).")
        return np.log(n_total)

    def _glm_auto():
        # WHICH SCALE THE FAMILY IS CHOSEN ON (instruction 182). The response
        # itself is swapped by the CALLER -- see `regression` -- because
        # everything downstream of the fit (the coefficient table, McFadden,
        # the residual panels) reads the same `y` and a model fitted on a
        # different one would silently disagree with all of it. All that is
        # left here is the link.
        fit_y = y
        if glm_force_identity:
            # The transform IS the link, so the family must not add another.
            family = sm.families.Gaussian(link=sm.families.links.Identity())
            print(f"  Using Gaussian family with Identity link for "
                  f"{response_name or 'the response'}.")
            return sm.GLM(fit_y, X, family=family).fit(
                **({'cov_type': cov_type} if cov_type else {}))
        family = pick_glm_family_and_link(fit_y, name=response_name,
                                          transform=transform)
        if isinstance(family, sm.families.Poisson):
            _validate_poisson_response(fit_y, X)
            # Same exposure the explicit 'poisson' branch uses. A family chosen
            # BY the data must be fitted the same way as one chosen by name, or
            # 'glm' and 'poisson' silently disagree on the same response.
            return sm.GLM(fit_y, X, family=family, offset=_poisson_offset()).fit(
                **({'cov_type': cov_type} if cov_type else {}))
        kwargs = {'family': family}
        if weights is not None and isinstance(family, sm.families.Binomial):
            # A per-well fraction estimated from 30 cells and one estimated
            # from 400 carry very different amounts of information, and the
            # binomial variance function only knows that if it is told. Weight
            # them exactly as the explicit 'logit'/'probit' branches do, and
            # ONLY for the binomial families: var_weights on the Poisson or
            # Gaussian branch would be re-weighting a response that already
            # has the right variance.
            kwargs['var_weights'] = np.asarray(weights).ravel()
        return sm.GLM(fit_y, X, **kwargs).fit(
            **({'cov_type': cov_type} if cov_type else {}))

    def _glm_poisson():
        _validate_poisson_response(y, X)
        family = sm.families.Poisson(link=sm.families.links.Log())
        # offset(log(cell_count)) turns the fit from "how many positive objects
        # are in this well" into "what fraction of this well's cells are
        # positive", which is the quantity the screen is about. See
        # _poisson_offset for the measurement that made this non-optional.
        return sm.GLM(y, X, family=family, offset=_poisson_offset()).fit(
            **({'cov_type': cov_type} if cov_type else {}))

    def _wls():
        # WLS with unit weights IS OLS. Saying so is the point: a user who
        # picks 'wls' on a table with no cell_count column would otherwise get
        # an OLS fit labelled 'wls' in the results folder name, the volcano
        # filename and the settings CSV, and nothing anywhere would disagree.
        if weights is None:
            raise ValueError(
                "regression_type='wls' needs per-well weights, and no "
                "'cell_count' column reached the model. Weighted least "
                "squares with unit weights is exactly OLS, so spaCR will not "
                "fit it under the 'wls' label. Use 'ols', or run the scores "
                "through process_scores so each well carries its cell count.")
        w = np.asarray(weights, dtype=float).ravel()
        if not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError(
                "WLS weights must be finite and positive (they are per-well "
                f"cell counts); got {np.nanmin(w)}-{np.nanmax(w)}.")
        return sm.WLS(y, X, weights=w).fit(
            **({'cov_type': cov_type} if cov_type else {}))

    def _rlm():
        # HuberT's t is in units of the ESTIMATED residual scale (MAD), not of
        # y, so the same t means the same thing whatever the response units.
        return sm.RLM(y, X, M=sm.robust.norms.HuberT(t=huber_t)).fit()

    def _quantile():
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(
                f"quantile must lie strictly inside (0, 1); got {quantile!r}. "
                f"0.5 is the median fit.")
        return sm.QuantReg(y, X).fit(q=float(quantile))

    def _hinge():
        y_binary = binarise_response(y, hinge_threshold,
                                     name='dependent variable')
        # LinearSVC minimises C * sum(hinge) + 0.5 * ||w||^2, so its C is the
        # INVERSE of a regularisation strength. Mapping alpha -> 1/alpha keeps
        # "larger alpha shrinks harder" true across every penalised backend;
        # without it, alpha would mean the opposite here than it does for
        # lasso and ridge, on the same settings key.
        if use_auto_alpha:
            # alpha='auto' means "choose the penalty by cross-validation" for
            # lasso, ridge and elasticnet, and it has to mean the same thing
            # here. It used to mean C = 1: 'auto' and alpha=1.0 produced
            # byte-identical coefficients, so a user who asked for a
            # cross-validated margin got an arbitrary fixed one under a label
            # that says otherwise.
            return _find_best_hinge_alpha(y_binary)
        strength = float(alpha)
        if strength <= 0:
            raise ValueError(
                f"alpha must be positive for hinge regression; got {alpha!r}.")
        model = _hinge_estimator(strength)
        model.fit(X, y_binary)
        return model

    def _hinge_estimator(strength):
        """A LinearSVC at regularisation ``strength`` (``C = 1 / strength``).

        ``class_weight='balanced'`` because a screen's positive class is
        routinely a small minority of wells: an unweighted hinge on a 95/5
        split minimises its loss by calling every well negative, which returns
        a coefficient vector of ~0 for every gRNA and reads downstream as "no
        hits". Balancing reweights each class by its inverse frequency, so the
        decision boundary is fitted to separate the classes rather than to
        count them.
        """
        return LinearSVC(C=1.0 / strength, loss='hinge', dual=True,
                         max_iter=20000, random_state=0,
                         class_weight='balanced')

    def _find_best_hinge_alpha(y_binary):
        """Pick the hinge penalty by stratified CV on balanced accuracy.

        The same 5-fold shape ``_find_best_alpha`` uses for the penalised
        least-squares backends. Balanced accuracy rather than accuracy: on an
        imbalanced screen plain accuracy is maximised by the degenerate
        all-negative fit, so scoring on it would cross-validate its way to the
        very failure ``class_weight='balanced'`` exists to prevent.

        Falls back to the unpenalised-scale default ``C = 1`` when the response
        has too few wells in a class to split five ways — a two-fold CV on
        three positive wells is noise, and choosing a penalty from noise is
        worse than not choosing one.
        """
        from sklearn.model_selection import cross_val_score

        strengths = np.logspace(-3, 3, 13)
        minority = int(min(np.sum(y_binary == 0), np.sum(y_binary == 1)))
        n_splits = min(5, minority)
        if n_splits < 2:
            print(f"hinge: alpha='auto' needs at least two wells in each "
                  f"class to cross-validate and the smaller class has "
                  f"{minority}; falling back to alpha=1.")
            model = _hinge_estimator(1.0)
            model.fit(X, y_binary)
            return model

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=0)
        scores = []
        for strength in strengths:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fold_scores = cross_val_score(
                    _hinge_estimator(strength), X, y_binary, cv=folds,
                    scoring='balanced_accuracy')
            scores.append(float(np.mean(fold_scores)))
        # Ties go to the STRONGER penalty (the larger alpha): among margins
        # that separate the held-out wells equally well, the one that shrinks
        # hardest is the one that generalises, and argmax on a raw list would
        # instead take the weakest.
        best = float(strengths[len(strengths) - 1
                               - int(np.argmax(scores[::-1]))])
        print(f"Optimal alpha for hinge: {best:.4g} "
              f"(balanced accuracy {max(scores):.4f}, {n_splits}-fold)")
        model = _hinge_estimator(best)
        model.fit(X, y_binary)
        return model

    def _named_design(name):
        """The design's column names, or a refusal that says why they matter.

        Gene-aware backends recover the gene behind each predictor from the
        column name produced by patsy. Refuse an unnamed array because group
        lasso would otherwise treat every column as a separate gene and
        reduce to ordinary lasso under a different label.
        """
        columns = getattr(X, 'columns', None)
        if columns is None:
            raise ValueError(
                f"regression_type={name!r} groups the design's columns by "
                f"gene and reads that grouping from the COLUMN NAMES, so it "
                f"needs a DataFrame design; a bare array has no names to "
                f"group by. Build the design with "
                f"dmatrices(..., return_type='dataframe'), which is what the "
                f"pipeline hands in.")
        return columns

    def _group_lasso():
        from . import group_lasso as group_lasso_module

        columns = _named_design('group_lasso')
        design = np.asarray(X, dtype=float)
        blocks = _design_column_groups(columns)
        # WHICH COLUMNS THE ANSWER IS ABOUT, computed before the fit because
        # the cross-validation below needs it too: a penalty that leaves two
        # row dummies standing and not one gene has selected nothing this
        # module can report.
        gene_terms = _level_term_mask(columns)
        # 'auto' CROSS-VALIDATES THE PENALTY, and it is what the panel posts
        # for this backend. A penalty is only large or small relative to the
        # design it is applied to: the shipped 0.05 is nearly half of the
        # tsg101 screen's own ceiling of 0.1285, so every one of its 297 gene
        # blocks came back exactly zero and the run was refused -- from
        # settings in which nobody had touched the penalty (236 C7).
        #
        # ANNOUNCED, never quiet. A penalty chosen for the user and not named
        # is one they cannot put in a methods section.
        if _left_blank(group_lasso_lambda) or (
                isinstance(group_lasso_lambda, str)
                and group_lasso_lambda.strip().lower() == 'auto'):
            lam = group_lasso_module.choose_lambda(
                design, y_flat, blocks,
                required=gene_terms if gene_terms.any() else None)
            print(f"group_lasso_lambda='auto': cross-validated over "
                  f"{group_lasso_module.PATH_POINTS} penalties down from "
                  f"this design's ceiling of "
                  f"{group_lasso_module.max_lambda(design, y_flat, blocks):.4g}"
                  f", chose {lam:.4g}.")
        else:
            lam = float(group_lasso_lambda)
        beta, intercept, converged = group_lasso_module.fit(
            design, y_flat, blocks, lam=lam)

        # THE REFUSAL IS ABOUT THE GENE BLOCKS, not about the design as a
        # whole. `np.any(beta)` is not the test: the row and column dummies
        # are singleton groups with far larger correlations than any guide
        # block, so they survive a penalty that has already emptied every
        # gene -- measured on the 384-well synthetic screen, lambda=0.02
        # leaves two row terms standing and not one gene. The user is then
        # handed a fit whose every gRNA coefficient is zero, which reads
        # downstream as "0 significant gRNAs" and is indistinguishable from a
        # screen with no hits. That is the same failure the lasso branch
        # below refuses, and it has to be refused on the same grounds.
        if not gene_terms.any():
            raise ValueError(
                "regression_type='group_lasso' penalises a GENE's guide "
                f"columns as one block, and none of this design's "
                f"{len(gene_terms)} columns is a gRNA or gene term "
                f"(columns: {[str(c) for c in columns][:6]}). Every column "
                f"would be its own block, which is ordinary lasso under "
                f"another name. It is fitted on the design prepare_formula "
                f"builds, whose terms are 'fraction:grna[...]' or "
                f"'gene_fraction:gene[...]'.")
        if not np.any(beta[gene_terms]):
            # max_lambda is the smallest penalty that zeroes EVERY group, so
            # it is an upper bound rather than the working value; on the same
            # fixture it is 0.384 and the planted gene is recovered at 0.001.
            # The message names it because a scale is what the user is
            # missing, and "lower it" alone gives them none.
            ceiling = group_lasso_module.max_lambda(design, y_flat, blocks)
            raise ValueError(
                f"group_lasso shrank every one of the "
                f"{int(gene_terms.sum())} gRNA/gene coefficients to exactly "
                f"zero at group_lasso_lambda={lam!r}, so the fit carries no "
                f"information about any gene. This design's "
                f"group_lasso.max_lambda -- the penalty above which nothing "
                f"at all survives -- is {ceiling:.4g}, and the gene blocks "
                f"empty well below it. Set group_lasso_lambda='auto' to "
                f"cross-validate it, or a small fraction of that ceiling to "
                f"choose it yourself. Or fit an unpenalised model ('ols') to "
                f"see the effect sizes the penalty is shrinking away.")

        if not converged:
            print(f"Warning: the group lasso did not reach its tolerance in "
                  f"{group_lasso_module.MAX_ITERATIONS} sweeps. The "
                  f"coefficients are the last iterate, not the solution; "
                  f"treat the selection as provisional.")

        genes_in_design = {label for label, is_gene in zip(blocks, gene_terms)
                           if is_gene}
        selected = {label for label, coefficient, is_gene
                    in zip(blocks, beta, gene_terms)
                    if is_gene and coefficient != 0}
        model = _GroupLassoResults(beta, intercept, blocks, lam, converged)
        mse = mean_squared_error(y_flat, model.predict(X))
        print(f"Group lasso MSE: {mse:.4f}, lambda={lam:g}, "
              f"{len(selected)} of {len(genes_in_design)} gene blocks "
              f"selected ({int(np.sum(beta[gene_terms] != 0))} of "
              f"{int(gene_terms.sum())} gRNA columns).")
        return model

    def _rra():
        from . import rra as rra_module

        columns = _named_design('rra')
        design = np.asarray(X, dtype=float)
        genes = [_gene_of_design_column(column) for column in columns]
        gene_terms = _level_term_mask(columns)
        if not gene_terms.any():
            raise ValueError(
                "regression_type='rra' aggregates a GENE's guides by rank, "
                f"and none of this design's {len(genes)} columns is a "
                f"gRNA or gene term (columns: {[str(c) for c in columns][:6]}"
                f"). It is fitted on the design prepare_formula builds, "
                f"whose terms are 'fraction:grna[...]' or "
                f"'gene_fraction:gene[...]'.")

        # THE PER-GUIDE SCORE IS THE MARGINAL SLOPE -- the least-squares slope
        # of the response on that guide's column ALONE, one parameter at a
        # time. It is not the joint fit's coefficient, and that is the whole
        # point of offering RRA: with 823 guides and 610 wells the joint fit
        # is undefined, and every backend that forms one is answering a
        # question the data cannot support (instruction 133). A marginal
        # slope exists at any width and is the direct analogue of MAGeCK's
        # per-guide log fold change, which is what alpha-RRA ranks.
        centred = design - design.mean(axis=0)
        response = y_flat - float(y_flat.mean())
        spread = (centred ** 2).sum(axis=0)
        moving = spread > 0
        slopes = np.zeros(design.shape[1], dtype=float)
        slopes[moving] = (centred[:, moving].T @ response) / spread[moving]

        # A CONSTANT COLUMN IS NOT RANKED. The intercept explains no variation
        # in the response, so it has no slope to rank; NaN is what
        # rank_aggregate drops, and dropping it is right because ranking it
        # would give it a rank it did not earn and shift every real guide's.
        ranked = np.where(moving, slopes, np.nan)
        table = rra_module.rank_aggregate(
            ranked, genes, alpha=float(rra_alpha), direction='both',
            n_permutations=int(rra_permutations))
        if not len(table) or 'p_neg' not in table.columns:
            raise ValueError(
                "regression_type='rra' ranked no guide: every gRNA column of "
                "this design is constant, so no guide has a marginal effect "
                "to rank. Check the fraction threshold - a design whose guide "
                "columns do not vary carries no information about any guide.")

        # BOTH TAILS, COMBINED THE STANDARD WAY. rank_aggregate reports
        # depletion and enrichment separately because they are two questions;
        # the coefficient table has one p_value column, so the two one-sided
        # permutation P values become the two-sided
        # min(1, 2 * min(p_neg, p_pos)). Taking the smaller tail WITHOUT
        # doubling would be a one-sided test chosen after seeing which way the
        # gene went, which is the classic way to halve a P value for free.
        two_sided = np.minimum(1.0, 2.0 * np.minimum(
            table['p_neg'].to_numpy(dtype=float),
            table['p_pos'].to_numpy(dtype=float)))
        by_gene = dict(zip(table['gene'].astype(str), two_sided))
        # ONE ROW PER DESIGN COLUMN, exactly as every other backend produces,
        # and the mapping is: the column keeps its OWN marginal slope as the
        # coefficient and carries its GENE's aggregated P value. RRA tests
        # genes, not guides, so a gene's guides share its P value; at
        # level='grna' that makes the BH family the guide count rather than
        # the gene count, which is conservative, and the level='gene' fit is
        # the one whose family matches what was tested.
        p_values = np.array(
            [by_gene.get(str(gene), np.nan) if gene is not None else np.nan
             for gene in genes], dtype=float)

        called = int(np.sum(two_sided <= 0.05))
        print(f"RRA: {len(table)} genes aggregated from "
              f"{int(np.sum(moving & gene_terms))} ranked guides, "
              f"alpha={float(rra_alpha):g}, {int(rra_permutations)} "
              f"permutations per guide count; {called} genes at an "
              f"uncorrected two-sided p <= 0.05.")
        return _RRAResults(slopes, p_values, columns, table)

    def _horseshoe():
        return _fit_horseshoe_poisson(X, y, exposure)

    def _spline():
        """OLS on a design whose COVARIATES carry a spline basis.

        The guide columns are untouched, so one coefficient and one P value
        per guide survive. The volcano, hit list and attribution can therefore
        read the result with no special case. What becomes free to bend is the
        nuisance trend that the straight line was assuming away.

        A column is treated as a covariate when it is CONTINUOUS and is not
        a guide or gene term -- an indicator has nothing to bend through,
        and expanding one would spend degrees of freedom on nothing.
        """
        from .nonparametric_fits import spline_design

        covariates = []
        for name in getattr(X, "columns", []):
            label = str(name)
            if "grna[" in label or "gene[" in label or label == "Intercept":
                continue
            column = np.asarray(X[name], dtype=float)
            if np.unique(column).size > 4:
                covariates.append(name)
        design = (spline_design(X, covariates,
                                knots=int(spline_knots),
                                degree=int(spline_degree))
                  if covariates else X)
        fitted = (sm.OLS(y, design).fit(cov_type=cov_type) if cov_type
                  else sm.OLS(y, design).fit())
        return fitted

    model_map = {
        'ols':    lambda: sm.OLS(y, X).fit(cov_type=cov_type) if cov_type else sm.OLS(y, X).fit(),
        'spline': _spline,
        'wls':    _wls,
        'rlm':    _rlm,
        'huber':  _rlm,
        'glm':    _glm_auto,
        'poisson': _glm_poisson,
        # Quasi-binomial is a binomial mean with a free dispersion. statsmodels
        # spells that as scale='X2' (dispersion from the Pearson chi-square) on
        # a Binomial family, which is what widens the standard errors; the
        # QuasiBinomial family above takes a dispersion the caller already
        # knows and is not what an overdispersed screen needs.
        'quasi_binomial': lambda: _glm_binomial(link=sm.families.links.Logit(),
                                                scale='X2'),
        'beta':   lambda: BetaModel(endog=y, exog=X).fit(),
        # logit and probit on a CONTINUOUS fraction y are routed through GLM-Binomial
        # with var_weights = cell_count. sm.Logit / sm.Probit require binary y.
        'logit':  lambda: _glm_binomial(link=sm.families.links.Logit()),
        'probit': lambda: _glm_binomial(link=sm.families.links.probit()),
        'quantile': _quantile,
        'mixed':  lambda: perform_mixed_model(
            y, X, groups, regression_backend=regression_backend),
        'lasso':  lambda: _find_best_alpha('lasso') if use_auto_alpha
                          else Lasso(alpha=alpha, max_iter=10000).fit(X, y_flat),
        'ridge':  lambda: _find_best_alpha('ridge') if use_auto_alpha
                          else Ridge(alpha=alpha).fit(X, y_flat),
        'elasticnet': lambda: _find_best_alpha('elasticnet') if use_auto_alpha
                          else ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                          max_iter=10000).fit(X, y_flat),
        'hinge':  _hinge,
        'horseshoe': _horseshoe,
        'group_lasso': _group_lasso,
        'rra': _rra,
    }

    # THE ALTERNATIVE FITTERS COME BEFORE THE DEFAULT MAP, and each one is
    # held to instruction 141 D: it fits the SAME model and reports the same
    # numbers, or it is not offered. What each one may be chosen for is
    # policed by `backend_status` above, so an unroutable pairing has already
    # been refused by name and this is only the dispatch.
    if backend == 'pyfixest':
        if cov_type is not None:
            # A SANDWICH IS NOT ABSORBED FOR FREE. HC0's meat is
            # X~' diag(e^2) X~ and does survive Frisch-Waugh-Lovell, but
            # HC1/HC2/HC3 correct by the FULL model's leverage, which an
            # absorbed fit never forms -- so three of the four spaCR offers
            # would come out different from the statsmodels number under the
            # same label. Instruction 141 D calls that a bug, so it is
            # refused instead of approximated.
            raise ValueError(
                f"regression_backend='pyfixest' absorbs rowID and columnID, "
                f"and the HC1/HC2/HC3 corrections are computed from the full "
                f"model's leverage, which an absorbed fit never forms. It "
                f"reports classical standard errors only, so "
                f"cov_type={cov_type!r} would be a label on numbers that did "
                f"not come from it. Fit with "
                f"regression_backend='statsmodels' to use cov_type, or clear "
                f"cov_type to absorb.")
        if regression_type == 'wls' and weights is None:
            # The same refusal `_wls` makes, made here too: WLS with unit
            # weights IS OLS, and a run labelled 'wls' that fitted OLS is the
            # silent mislabelling that branch exists to prevent.
            raise ValueError(
                "regression_type='wls' needs per-well weights, and no "
                "'cell_count' column reached the model. Weighted least "
                "squares with unit weights is exactly OLS, so spaCR will not "
                "fit it under the 'wls' label. Use 'ols', or run the scores "
                "through process_scores so each well carries its cell count.")
        try:
            model = _fit_absorbed_least_squares(
                X, y, weights=weights if regression_type == 'wls' else None,
                kind='WLS' if regression_type == 'wls' else 'OLS')
        except ValueError as nothing_to_absorb:
            # NOTHING TO ABSORB IS NOT A FAILURE, and this used to end the
            # run 20 seconds in. Reported from a live fit on 2026-08-20:
            # `model_plate_position=False` takes rowID and columnID out of
            # the design, and the absorbing backend then refuses because it
            # has no fixed effects to project out.
            #
            # ITS OWN REFUSAL SAYS WHY THAT IS THE WRONG ANSWER: "the fit
            # would be the statsmodels fit with an extra projection in front
            # of it". With nothing to absorb the two backends compute the
            # SAME numbers, so falling back is not substituting a different
            # method -- it is the identical fit by the only route left. That
            # is what makes this fallback safe where the montage's
            # multivariate one was not: there, the alternative answered a
            # different question.
            if 'nothing to absorb' not in str(nothing_to_absorb):
                raise
            print("  regression_backend='pyfixest' has nothing to absorb "
                  "on this design -- there are no rowID or columnID terms in "
                  "it, so either model_plate_position=False removed them or "
                  "the screen sits on one row and one column. Fitting with "
                  "statsmodels instead: with no factors to project out the "
                  "two backends compute the same numbers, so this is the "
                  "same fit by the only route left, not a different model.")
            model = model_map[regression_type]()
    elif backend == 'glum':
        if cov_type is not None:
            raise ValueError(
                f"regression_backend='glum' reports the classical GLM "
                f"standard errors -- the inverse information matrix at a "
                f"fixed dispersion -- and has no HC0..HC3 estimator that "
                f"matches statsmodels', so cov_type={cov_type!r} would be a "
                f"label on numbers that did not come from it. Fit with "
                f"regression_backend='statsmodels' to use cov_type.")
        model = _fit_glum_glm(X, y, regression_type, weights=weights,
                              exposure=exposure)
    else:
        model = model_map[regression_type]()

    if regression_type in ['glm', 'poisson']:
        print(fit_quality_note(model))
        print(summary_for_console(model, verbose=verbose))

    if regression_type in ['lasso', 'ridge', 'elasticnet']:
        mse = mean_squared_error(y_flat, model.predict(X))
        coefs = np.asarray(model.coef_).ravel()
        n_nonzero = int(np.sum(coefs != 0))
        print(f"{regression_type.capitalize()} regression MSE: {mse:.4f}, "
              f"non-zero coefficients: {n_nonzero} of {X.shape[1]}")
        if n_nonzero == 0:
            # Every coefficient shrunk to exactly zero is not a finding, it is
            # a penalty set too high for the scale of this design - and it
            # reaches the user as "0 significant gRNAs", which is
            # indistinguishable from a screen with no hits. The default
            # alpha=1 does this to a fraction-scale design every time.
            if use_auto_alpha:
                # The penalty was not mis-set, it was CHOSEN: cross-validation
                # preferred the empty model to every non-empty one it tried, so
                # no gRNA predicted the held-out wells better than the mean did.
                # Telling this user to "set alpha to 'auto'" - which the old
                # message did, unconditionally - is telling them to do the
                # thing they just did.
                raise ValueError(
                    f"{regression_type} with alpha='auto' cross-validated its "
                    f"way to the empty model: every one of the "
                    f"{X.shape[1]} coefficients is exactly zero, because no "
                    f"gRNA predicted the held-out wells better than their mean "
                    f"did. That is a null screen, not a misconfiguration - the "
                    f"fit is refused rather than written out as '0 significant "
                    f"gRNAs', which is what it would look like. Check the "
                    f"dependent variable and the aggregation, or fit an "
                    f"unpenalised model ('ols') to see the effect sizes the "
                    f"penalty is shrinking away.")
            raise ValueError(
                f"{regression_type} shrank all {X.shape[1]} coefficients to "
                f"exactly zero at alpha={alpha!r}: the penalty is far larger "
                f"than the scale of this design, so the fit carries no "
                f"information about any gRNA. Lower alpha, or set it to "
                f"'auto' to choose it by cross-validation.")

    return model


def _fit_horseshoe_poisson(X, y, exposure):
    """Fit spaCRPower's sparse Poisson model through :mod:`spacr.power_model`.

    The model is the one ``spaCRPower/R/fit_model.R`` fits::

        Npositive_w ~ Poisson(Ntotal_w * exp(b0 + sum_g b_g * log10expression_wg))
        b_g ~ horseshoe(df = 10)

    i.e. a Poisson GLM with a log link, an ``offset(log(Ntotal))`` exposure and
    a horseshoe sparsity prior doing the variable selection. In spaCR's terms
    ``y`` is the per-well positive-object count (``process_scores`` sums the
    response for this type, as it does for ``'poisson'``), ``exposure`` is the
    well's cell count and ``X`` is the ordinary spaCR design.

    The import is deliberately lazy and inside the branch: the horseshoe
    fitter is a separate module, and neither the ordinary regressions nor
    anything else that imports :mod:`spacr.ml` should pay for it or fail
    without it.

    :param X: Design matrix.
    :param y: Per-well positive counts.
    :param exposure: Per-well total cell counts (the Poisson exposure).
    :returns: The fitted object returned by
        ``spacr.power_model.fit_horseshoe_poisson``, which must expose
        ``params`` and either ``pvalues`` or ``bse`` indexed like
        ``X.columns``.
    :raises ImportError: when :mod:`spacr.power_model` is not installed yet,
        naming the entry point this branch calls.
    :raises ValueError: when no exposure is available, or the returned object
        does not carry the coefficients this pipeline needs.
    """
    if exposure is None:
        raise ValueError(
            "regression_type='horseshoe' fits Npositive ~ ... + "
            "offset(log(Ntotal)), so it needs the per-well cell count as the "
            "exposure, and no 'cell_count' column reached the model. Without "
            "it the counts of a 400-cell well and a 30-cell well would be "
            "compared as if the wells were the same size.")
    try:
        from .power_model import ModelData, fit_model, gather_model_estimate
    except ImportError as exc:
        raise ImportError(
            "regression_type='horseshoe' needs spacr.power_model, which is "
            "not present in this install. The branch calls "
            "spacr.power_model.prepare/ModelData + fit_model + "
            "gather_model_estimate; install or restore that module to use it."
        ) from exc

    counts = _validate_poisson_response(
        y, X, model="horseshoe (a sparse Poisson GLM over well counts)")
    n_total = np.asarray(exposure, dtype=float).ravel()
    if n_total.size != counts.size:
        raise ValueError(
            f"horseshoe exposure has {n_total.size} entries but the response "
            f"has {counts.size}; they are the same wells and must align.")
    if not np.isfinite(n_total).all() or np.any(n_total <= 0):
        raise ValueError(
            "horseshoe exposure (the well cell count) must be finite and "
            f"positive; got {np.nanmin(n_total)}-{np.nanmax(n_total)}. "
            "log(Ntotal) is undefined otherwise.")
    if np.any(counts > n_total):
        raise ValueError(
            "horseshoe needs Npositive <= Ntotal per well: the response is a "
            "count of positive objects and the exposure is how many objects "
            "were imaged, so a well cannot have more positives than cells. "
            f"{int(np.sum(counts > n_total))} well(s) break that.")

    design = np.asarray(X, dtype=float)
    columns = [str(c) for c in X.columns]
    # A constant column - patsy's Intercept - is confounded with the model's
    # own intercept term, which power_model fits separately. Naming it here is
    # what makes power_model return NaN for it rather than a shrunk-to-zero
    # coefficient that reads as "this term was tested and found null".
    constant = [name for name, column in zip(columns, design.T)
                if np.ptp(column) == 0]

    model_data = ModelData(
        wells=np.asarray(X.index),
        genes=np.asarray(columns, dtype=object),
        Npositive=counts,
        Ntotal=n_total,
        log10expression=design,
        unidentified_genes=tuple(constant),
    )
    # standardize=True, unlike power_model's own default. The horseshoe's
    # global scale is calibrated for spaCRPower's log10 read fraction, which
    # has a spread of about 1; spaCR's design is gRNA FRACTIONS, whose columns
    # have standard deviations around 0.05, and on that scale the prior
    # shrinks every coefficient to ~1e-4 and separates nothing. Scaling each
    # column to unit SD is what makes the shrinkage comparable across terms -
    # which is the entire point of the model - at the cost that beta is then
    # "per standard deviation of that gRNA's fraction", not per unit.
    fit = fit_model(model_data, seed=0, standardize=True)
    return _HorseshoeResults(fit, gather_model_estimate(fit))


class _HorseshoeResults:
    """Adapt a :class:`spacr.power_model.PowerFit` to the results API spaCR reads.

    :func:`process_model_coefficients` wants ``params`` and ``pvalues``
    indexed by design column; the horseshoe model reports posterior draws.
    The translation is stated rather than implied:

    * ``params`` is the posterior MEAN of each coefficient - a point estimate
      under shrinkage, not a maximum-likelihood one, so it is already pulled
      towards zero for terms the prior judges null. That is the whole purpose
      of the model and the reason its coefficients are not comparable in
      magnitude with the OLS ones.
    * ``pvalues`` is the two-sided posterior TAIL MASS,
      ``2 * min(P(beta > 0), P(beta < 0))``. It is not a frequentist p-value
      and no null hypothesis was tested to get it; it is reported under that
      name because every consumer downstream - the volcano plot, the hit
      table, ``-log10(p_value)`` - reads that column and would otherwise be
      given nothing. A term whose posterior sits entirely on one side of zero
      gets 0.
    * Unidentified terms (a constant column, or a gRNA present in every well
      at the same fraction) are DROPPED rather than reported as zero: the
      model could not estimate them, and a zero with a p-value would read as
      a tested null.

    :param fit: the ``PowerFit`` returned by ``power_model.fit_model``.
    :param estimates: the frame ``power_model.gather_model_estimate`` builds.
    """

    def __init__(self, fit, estimates):
        required = ('gene', 'mean', 'sd', 'prob_positive', 'identified')
        missing = [c for c in required if c not in estimates.columns]
        if missing:
            # power_model is a separate module with its own release cadence;
            # a renamed column must stop the run here, where it can be named,
            # rather than surface as a KeyError from inside pandas.
            raise ValueError(
                f"spacr.power_model.gather_model_estimate returned columns "
                f"{list(estimates.columns)}; spaCR's coefficient table needs "
                f"{list(required)} and {missing} are absent.")
        self.fit = fit
        self.estimates = estimates
        identified = estimates[estimates['identified'].astype(bool)]
        index = pd.Index(identified['gene'].astype(str), name=None)
        self.params = pd.Series(identified['mean'].to_numpy(), index=index)
        self.bse = pd.Series(identified['sd'].to_numpy(), index=index)
        prob_positive = identified['prob_positive'].to_numpy(dtype=float)
        tail = 2.0 * np.minimum(prob_positive, 1.0 - prob_positive)
        self.pvalues = pd.Series(np.clip(tail, 0.0, 1.0), index=index)
        self.converged = bool(getattr(fit, 'converged', True))
        if not self.converged:
            print("Warning: the horseshoe fit did not meet its own "
                  "convergence criterion; treat the coefficients as "
                  "provisional and re-run with more steps or a NUTS backend.")

    def summary(self):
        """Return the per-term posterior summary, for save_summary_to_file."""
        return self.estimates


class _GroupLassoResults:
    """Adapt :mod:`spacr.group_lasso` to the estimator API spaCR reads.

    ``coef_`` and ``predict`` are all :data:`_SKLEARN_COEF_TYPES`' branch of
    :func:`process_model_coefficients` and :func:`calculate_p_values` ask of a
    penalised fit, so the group lasso reports EXACTLY what ``lasso`` and
    ``elasticnet`` report -- one signed coefficient per design column, and a
    selection frequency attached by the run -- rather than a second convention
    of its own.

    WHY THE COEFFICIENT AND NOT ``gene_effects``' NORM. ``gene_effects``
    answers with ``||b_g||_2``, one non-negative number per gene, which is the
    natural summary of a block but is not what the pipeline downstream of the
    fit is built on: the volcano's x axis, ``coefficient_threshold``'s
    control spread and the hit table's sign all read a SIGNED per-column
    effect. The block's own coefficients carry that sign, and because the
    block is zero or none of it is, ``||b_g||_2 > 0`` and "this gene has a
    non-zero coefficient" are the same statement -- so nothing is lost by
    tabling the coefficients, and a caller who wants the norm is one
    ``np.linalg.norm`` away from it: ``coef_`` and ``groups`` are both on this
    object, which is why ``groups`` is kept rather than discarded after the
    fit.

    :param coefficients: one coefficient per design column, in column order.
    :param intercept: the unpenalised intercept.
    :param groups: the group label of each column, from
        :func:`_design_column_groups`.
    :param lam: the penalty that was applied.
    :param converged: whether block coordinate descent met its tolerance.
    """

    def __init__(self, coefficients, intercept, groups, lam, converged):
        self.coef_ = np.asarray(coefficients, dtype=float).ravel()
        self.intercept_ = float(intercept)
        self.groups = list(groups)
        self.lam = float(lam)
        self.converged = bool(converged)

    def predict(self, X):
        """``X @ coef_ + intercept_``, the fit's prediction for a design.

        :func:`calculate_p_values` needs the residual, and the residual needs
        this. Taking ``np.asarray`` rather than relying on pandas' matmul keeps
        it working for a plain array as well as the DataFrame the pipeline
        hands in.
        """
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class _RRAResults:
    """Adapt :mod:`spacr.rra` to the results API :func:`process_model_coefficients` reads.

    ``params`` and ``pvalues``, the same two attributes every statsmodels fit
    and :class:`_HorseshoeResults` expose, so ``rra`` needs no branch of its
    own. What each one IS, stated rather than implied, because neither comes
    from a joint fit:

    * ``params`` is the guide's MARGINAL least-squares slope -- the slope of
      the response on that guide's column alone, one parameter estimated at a
      time. RRA's whole claim is that it never forms the joint fit (see
      :mod:`spacr.rra`), and this screen is 823 guides against 610 wells,
      where the joint fit is undefined. A marginal slope is defined for every
      column at any width and is the analogue of MAGeCK's per-guide log fold
      change, which is what alpha-RRA ranks.
    * ``pvalues`` is the guide's GENE's permutation P value, two-sided as
      ``min(1, 2 * min(p_neg, p_pos))`` -- the standard combination of the two
      one-sided permutation tests :func:`spacr.rra.rank_aggregate` reports.
      Taking whichever tail is smaller and NOT doubling would be a one-sided
      test chosen after seeing the data.

    A row that names no gene -- the intercept, the row/column dummies -- was
    never ranked, so its P value is NaN rather than 1.0: it was not tested, and
    a 1.0 would read as "tested and found null".

    :param scores: the marginal slope of each design column, in column order.
    :param p_values: the two-sided permutation P value of each column's gene.
    :param index: the design column names.
    :param genes: :func:`spacr.rra.rank_aggregate`'s per-gene table, kept whole
        so ``rho_neg``/``rho_pos`` and the direction split are not lost.
    """

    def __init__(self, scores, p_values, index, genes):
        feature_index = pd.Index([str(name) for name in index])
        self.params = pd.Series(np.asarray(scores, dtype=float),
                                index=feature_index)
        self.pvalues = pd.Series(np.asarray(p_values, dtype=float),
                                 index=feature_index)
        self.genes = genes

    def summary(self):
        """The per-gene RRA table, for :func:`save_summary_to_file`."""
        return self.genes


#: Regression types ``random_row_column_effects=True`` may be combined with.
#: ``'ols'`` is here because it is the DEFAULT value of ``regression_type``, so
#: it cannot be told apart from "the user never touched the model dropdown";
#: ``None`` means "choose from the response", which the mixed branch answers.
#: Every other name is a deliberate choice that the mixed override would throw
#: away.
_RANDOM_EFFECTS_COMPATIBLE = (None, 'ols', 'mixed')




def _reject_unused_run_settings(settings):
    """Refuse a post-fit setting the chosen model will never read.

    :func:`regression_model` polices the six knobs that reach the estimator,
    and raises before a wrong number can become a result. These three do not
    reach the estimator at all — they configure how :func:`perform_regression`
    turns coefficients into a hit list — so nothing was checking them, and
    ``lasso_selection_threshold=0.9`` on an OLS run passed through fifteen of
    the seventeen types in silence.

    ``regression_type=None`` is policed as strictly as a named one:
    :func:`check_distribution` only ever auto-selects ``logit``, ``beta``,
    ``quasi_binomial``, ``ols`` or ``glm``, none of which reads any of these,
    so "it might pick lasso" is not a reason to let them through.

    :param settings: The finished settings dict.
    :raises ValueError: naming the setting, the type and the alternative.
    """
    reg_type = settings.get('regression_type', 'ols')
    _reject_unused_settings(reg_type, {
        name: (settings.get(name, default), default)
        for name, default in _RUN_LEVEL_DEFAULTS.items()})
    return settings


def _reconcile_random_row_column_effects(settings):
    """Make ``random_row_column_effects=True`` and ``regression_type`` agree.

    :func:`regression` reacts to the flag by fitting a MixedLM with row and
    column variance components, whatever ``regression_type`` says — and
    ``_perform_regression_set_paths`` had already named the results folder
    after ``settings['regression_type']``. A run configured as ``'lasso'`` with
    the flag on therefore fitted a mixed model and wrote it to
    ``results/<screen>/lasso/``, where nothing in the folder, the volcano
    filename or the settings CSV disagreed. Every penalty setting that run
    carried was ignored too, silently, because the mixed branch never reaches
    :func:`regression_model` and so never reaches
    :func:`_reject_unused_settings`.

    Two things happen here, both before any file is written:

    * an incompatible model choice is REFUSED, naming both settings;
    * a compatible one is rewritten to ``'mixed'`` in ``settings``, so the
      folder, the volcano filename and the saved settings all name the model
      that was actually fitted.

    :param settings: The finished settings dict; mutated in place.
    :raises ValueError: when the flag is combined with a named model that is
        not a mixed model, with ``model_plate_position=False``, or with a
        setting the mixed model cannot read.
    """
    if not settings.get('random_row_column_effects', False):
        return settings

    # OUT PLUS RANDOM IS NOT A STATE (instruction 143 A). Plate position has
    # three: out of the model (model_plate_position=False), in as fixed
    # effects (True), in as variance components (True plus this flag). Asking
    # for variance components on terms that are not in the model is a fourth,
    # and it is refused here -- before a folder is named or a file is written
    # -- rather than resolved to whichever of the two the reader guesses,
    # because both guesses fit a DIFFERENT model from the one asked for and
    # neither would say so. Same seam, same voice, as the model conflict
    # below; prepare_formula refuses the same pair for a caller that never
    # goes through a settings dict.
    if not settings.get('model_plate_position', True):
        raise ValueError(
            "random_row_column_effects=True fits rowID and columnID as "
            "variance components, and model_plate_position=False takes them "
            "out of the model entirely: there is nothing left for the mixed "
            "fit to make random. Set model_plate_position=True to fit plate "
            "position as variance components, or "
            "random_row_column_effects=False to leave it out.")

    reg_type = settings.get('regression_type', 'ols')
    if reg_type not in _RANDOM_EFFECTS_COMPATIBLE:
        raise ValueError(
            f"random_row_column_effects=True fits a mixed model with row and "
            f"column variance components, so it cannot also fit "
            f"regression_type={reg_type!r}: one of the two has to go. It used "
            f"to win silently, and the {reg_type!r} settings went with it — "
            f"the run wrote a MixedLM fit into results/<screen>/{reg_type}/ "
            f"and said nothing. Set random_row_column_effects=False to fit "
            f"{reg_type!r}, or regression_type='mixed' to fit the mixed model.")

    # The mixed branch reads none of the per-model knobs, so any of them set
    # away from its default is a request nothing will honour. Same seam, same
    # message shape, as the fixed-effects path.
    _reject_unused_settings('mixed', {
        'alpha': (1.0 if settings.get('alpha') in (None, 'auto')
                  else settings.get('alpha', 1.0), 1.0),
        'l1_ratio': (settings.get('l1_ratio', 0.5), 0.5),
        'cov_type': (settings.get('cov_type'), None),
        'quantile': (settings.get('quantile', 0.5), 0.5),
        'hinge_threshold': (settings.get('hinge_threshold'), None),
        'huber_t': (settings.get('huber_t', 1.345), 1.345),
        'spline_knots': (settings.get('spline_knots', 4), 4),
        'spline_degree': (settings.get('spline_degree', 3), 3),
    })

    if reg_type != 'mixed':
        print(f"random_row_column_effects=True: fitting 'mixed' rather than "
              f"{reg_type!r}, and naming the results folder for it.")
    settings['regression_type'] = 'mixed'
    return settings


def _mixed_model_groups(df, dependent_variable, model_index, *,
                        gene_column='gene'):
    """Return the outer random-intercept grouping for ``regression_type='mixed'``.

    The gene is the outer cluster and the guide is nested inside it. The model
    fits

        y ~ gene_fraction:gene + (1 | gene/grna) + rowID + columnID

    where ``(1 | gene/grna)`` is represented as ``groups=gene`` plus a guide
    variance component inside each gene. Row and column structure is carried
    by fixed terms, or by variance components when requested. See
    :func:`fit_mixed_model`.

    A single gene provides only one outer cluster and is refused.

    :param df: The cleaned long-format frame.
    :param dependent_variable: Response column name, named in the refusal so
        the message points at the run the user actually configured.
    :param model_index: Row index patsy kept, so the returned vector aligns
        with the design matrix row for row.
    :param gene_column: The outer grouping column. Default ``'gene'``.
    :returns: Series of gene ids, one per design row.
    :raises ValueError: when the screen has a single gene, naming the way out.
    """
    genes = df.loc[model_index, gene_column]
    n_genes = genes.nunique()
    if n_genes > 1:
        print(f"Mixed model: grouping on {gene_column} ({n_genes} genes), "
              f"with guides nested inside. The gene sits above the guide, "
              f"which is the level the random effect describes.")
        return genes

    raise ValueError(
        f"a mixed model needs at least two clusters and this screen has one "
        f"{gene_column}. The random intercept has to sit above the guide, so "
        f"with a single gene there is nothing left for it to describe and "
        f"every guide BLUP against {dependent_variable!r} would be shrunk to "
        f"the same number. Fit a fixed-effects regression_type with "
        f"level='grna', which tests the guides directly and is the model a "
        f"one-gene screen supports.")


def _write_regression_qc(model, X, y, df, dst, *, coef_df=None,
                         regression_type=None, volcano_path=None):
    """Write the full QC suite for a fit into ``<dst>/regression_qc/``.

    Generate residual, scale-location, Q-Q, influence, collinearity,
    calibration, and coefficient diagnostics while the fitted design matrix
    and response are still available.

    Weights are deliberately not forwarded. ``regression`` passes cell
    counts to ``regression_model`` as ``var_weights`` / WLS weights / Poisson
    exposure for the types that take them, and for those types
    :func:`spacr.regression_qc.build_context` recovers the weights from the
    fitted model itself, so the hat diagonal, the residual and the scale agree.
    The unweighted types (ols, lasso, ridge, elasticnet) never saw the counts,
    and handing them in as ``weights`` would compute a weighted leverage for a
    fit nobody ran. The counts still reach the cell-count panel through
    ``metadata['cell_count']``, which is where that panel looks first.

    :param model: the fitted model.
    :param X: the design matrix that was fitted.
    :param y: the response that was fitted.
    :param df: the cleaned long-format frame, for the per-well metadata.
    :param dst: the run's results folder.
    :param coef_df: the coefficient table, so the p-value histogram shows the
        screen's p-values rather than the design's.
    :param regression_type: the spaCR regression type string.
    :param volcano_path: the volcano plot for this run, named on the report.
    :returns: the manifest dict, or ``None`` if the report could not be written.
    """
    from .regression_qc import regression_qc_report

    # Per-well labels: what turns "well 41 is an outlier" into a plate, a row
    # and a column somebody can go back to the microscope with.
    metadata = None
    try:
        columns = [column for column in (schema.PLATE_KEY, schema.ROW_KEY,
                                         schema.COLUMN_KEY, schema.PRC_KEY,
                                         'cell_count')
                   if column in df.columns]
        if columns:
            metadata = df.loc[X.index, columns]
            # `.loc` with a duplicated index does not raise: it returns the
            # cross product, so a frame whose labels repeat comes back with
            # n*n rows. regression_qc_report would then refuse the whole
            # report -- losing every panel, including the variance-homogeneity
            # one -- over metadata that is only ever used for LABELS. Losing
            # the labels is the proportionate answer.
            if len(metadata) != len(X):
                raise ValueError(
                    f"{len(metadata)} metadata rows for {len(X)} fitted rows; "
                    f"the frame's index does not identify wells uniquely")
    except Exception as error:                      # noqa: BLE001 - advisory
        print(f"Regression QC: could not align per-well metadata to the fitted "
              f"rows ({type(error).__name__}: {error}); the plate/row/column "
              f"panels will skip rather than label the wrong well.")
        metadata = None

    # NO `fmt`. The panels are figures the user keeps, so they follow the
    # format preference -- and `regression_qc_report` reads it itself now, so
    # resolving it here as well would be two places deciding one thing. An
    # explicit `fmt` is a caller FORCING a format, which this caller is not
    # doing; passing the preference under that name made a preference
    # indistinguishable from an override.
    try:
        return regression_qc_report(
            model, X, y, dst, metadata=metadata, coef_df=coef_df,
            regression_type=regression_type, volcano_path=volcano_path,
            verbose=True)
    except Exception as error:                      # noqa: BLE001 - advisory
        # A diagnostic that fails must never destroy a fit that already
        # succeeded and cost an hour. The report itself already downgrades a
        # failing panel to FAILED; this catches the rarer case where the
        # report as a whole cannot be built.
        print(f"Regression QC report could not be written: "
              f"{type(error).__name__}: {error}")
        return None


def resolve_levels(regression_type, level='both'):
    """Which level(s) a run fits, given the backend and the ``level`` setting.

    ``mixed`` fits ONE model that already contains both levels -- the gene as a
    fixed effect and the guide as a random effect nested inside it -- so it
    ignores ``level`` entirely and answers ``('gene',)``. That is why the GUI
    greys the dropdown out rather than hiding it: the
    setting exists, but this model does not read it.

    Every other backend is fixed effects only and cannot nest, so it fits one
    level at a time and ``level`` chooses which. ``'both'`` is TWO FITS.

    :param regression_type: the backend name, or ``None`` (not yet chosen).
    :param level: ``'both'`` (default), ``'grna'`` or ``'gene'``.
    :returns: a tuple of levels to fit, in the order they are fitted.
    :raises ValueError: for a level that is not one of :data:`LEVEL_CHOICES`.
    """
    key = str(level).strip().lower()
    if key not in LEVEL_CHOICES:
        raise ValueError(
            f"level={level!r} is not a model level. Choose one of "
            f"{LEVEL_CHOICES!r}: 'grna' fits y ~ fraction:grna + rowID + "
            f"columnID, 'gene' fits y ~ gene_fraction:gene + rowID + "
            f"columnID, and 'both' fits each of them SEPARATELY.")
    if regression_type == 'mixed':
        return ('gene',)
    if key == 'both':
        return ('grna', 'gene')
    return (key,)


def regression(df, csv_path, dependent_variable='predictions', regression_type=None, alpha=1.0,
               random_row_column_effects=False, nc='233460', pc='220950', controls=None,
               dst=None, cov_type=None, plot=False, l1_ratio=0.5, quantile=0.5,
               hinge_threshold=None, hinge_n_boot=200, huber_t=1.345, qc=True,
               spline_knots=4, spline_degree=3,
               legacy_volcano=False, level='grna', level_dst=None,
               draw_shared_panels=True, group_lasso_lambda='auto',
               rra_alpha=0.25, rra_permutations=10000,
               model_plate_position=True,
               regression_backend=DEFAULT_REGRESSION_BACKEND,
               verbose=False, transform="",
               intercept='fitted', intercept_value=0.0):
    """Run the full regression pipeline: clean, fit, extract coefficients, optional volcano plot.

    :param df: Long-format DataFrame with gRNA/gene fractions and the
        dependent variable.
    :param csv_path: Path used to derive the volcano-plot filename.
    :param dependent_variable: Response column name. Default
        ``'predictions'``.
    :param regression_type: Model type; auto-selected via
        :func:`check_distribution` when ``None``.
    :param regression_backend: WHO fits it, one of
        :data:`REGRESSION_BACKEND_ORDER`. Default ``'statsmodels'``. It is
        threaded to whichever fitter this run reaches -- the mixed branch and
        :func:`regression_model` alike -- so one setting answers for the
        whole run, and a backend that cannot fit the chosen family is refused
        by name before any design is built.
    :param alpha: Regularisation strength for penalised models.
    :param random_row_column_effects: If True, fit a mixed model with
        random row/column effects.
    :param model_plate_position: Whether ``rowID`` and ``columnID`` are terms
        in the model at all. Direct calls default to ``True`` for API
        compatibility; new application settings default to ``False`` so the
        terms are opt-in. See :func:`prepare_formula` for the measured costs
        of including or omitting them. ``False`` with
        ``random_row_column_effects=True`` is refused: there is nothing left
        to make random.
    :param nc: Negative-control gene identifier. Default ``'233460'``.
    :param pc: Positive-control gene identifier. Default ``'220950'``.
    :param controls: Explicit list of control identifiers.
    :param dst: Output directory for plots and summaries.
    :param cov_type: Optional covariance estimator for the likelihood fits.
    :param plot: If True, render the volcano plot after fitting.
    :param l1_ratio: ``elasticnet`` L1/L2 mix.
    :param quantile: Quantile fitted by ``quantile`` regression.
    :param hinge_threshold: Response cut used to binarise for ``hinge``.
    :param hinge_n_boot: Bootstrap resamples behind the hinge p-values.
    :param huber_t: Huber tuning constant for ``rlm``/``huber``.
    :param group_lasso_lambda: Block penalty for ``group_lasso``.
    :param rra_alpha: Top fraction of the guide ranking ``rra`` aggregates.
    :param rra_permutations: Draws per guide count in ``rra``'s null.
    :param qc: Write the regression QC suite into ``<dst>/regression_qc/``.
    :param legacy_volcano: also draw the ORIGINAL matplotlib volcano.
        Default ``False``. The interactive one is far faster and the
        house-style panel is what a run now produces; drawing both gives two
        volcanoes in two idioms on the same grid.
        Requires ``dst`` and a design matrix, so it is skipped for the mixed
        branch and when no destination was given.
    :param level: WHICH MODEL TO FIT -- ``'grna'`` (default) or ``'gene'``.
        One level, one design; ``'both'`` is refused here because it is two
        fits. :func:`regression_levels` is the entry point that does both.
        Ignored by ``regression_type='mixed'``, which fits the gene fixed and
        the guide random inside it and so is already both levels.
    :param level_dst: Where THIS LEVEL's figures go -- the QC suite, the
        volcano, the publication sheet. Defaults to ``dst``, which is what a
        single-level run wants. :func:`regression_levels` gives each level its
        own subfolder so two fits cannot overwrite each other's
        ``regression_figure.pdf``.
    :param draw_shared_panels: Draw the guide-fraction and response
        distributions, which describe the DATA and not the fit. False on the
        second of two fits, so the figure grid gets one copy rather than two
        identical ones.
    :returns: ``(model, coef_df, regression_type)``.
    """

    if controls is None:
        controls = ['']
    from .plot import volcano_plot, plot_histogram

    # create_volcano_filename names a quantile run by the quantile it fitted
    # rather than by the model name, because two quantiles of the same screen
    # are two different results that must not overwrite each other. That used
    # to be alpha, which is no longer the quantile.
    # Per-level figures go where the caller says. A single-level run keeps
    # writing straight into dst, which is every existing path; a two-fit run
    # gets one subfolder per level so the second fit's regression_figure.pdf
    # does not land on top of the first fit's.
    level_dst = dst if level_dst is None else level_dst

    volcano_path = create_volcano_filename(
        csv_path, regression_type,
        quantile if regression_type == 'quantile' else alpha, level_dst)

    if regression_type is None:
        regression_type = check_distribution(df[dependent_variable])

    # ONE LEVEL PER FIT, and `mixed` decides its own. 'both' is refused rather
    # than quietly resolved to one of them: a caller that asked for two fits
    # and silently got one would read the guide table as if it were both.
    wanted = resolve_levels(regression_type, level)
    if len(wanted) != 1:
        raise ValueError(
            f"regression() fits ONE level; level={level!r} asks for "
            f"{list(wanted)}. Call regression_levels(), which fits each of "
            f"them separately and corrects each within itself.")
    level = wanted[0]

    print(f"Using regression type: {regression_type}")

    # WHICH SCALE THE GLM FITS (instruction 182), decided BEFORE the design
    # matrices are built so the fit, the coefficient table, McFadden and the
    # residual panels all read the same response. Both ways out of the double
    # transform are offered because both are defensible and they are
    # different science; spaCR does not choose between them.
    dependent_variable, transform, glm_force_identity, conflict_note = (
        resolve_glm_transform_conflict(
            dependent_variable, transform=transform,
            available=getattr(df, 'columns', ()),
            regression_type=regression_type))
    if conflict_note:
        print(conflict_note)

    df = check_and_clean_data(df, dependent_variable)

    # WHAT THE INTERCEPT IS, decided before the design is built so the fit,
    # the coefficient table and every panel read one response. 'control'
    # shifts the response; 'zero' takes the term out of the formula below;
    # 'fitted' does neither and is what every run did before this existed.
    intercept_offset = 0.0
    intercept_mode = str(intercept or 'fitted').strip().lower()
    if intercept_mode == 'control':
        df, intercept_offset = centre_on_controls(df, dependent_variable, nc)
        if intercept_offset:
            print(f"Intercept set to the negative controls: {dependent_variable} "
                  f"centred by {intercept_offset:.6g}, so a coefficient reads "
                  f"as its distance from {nc!r}.")
        else:
            print(f"Intercept left as fitted: no rows match "
                  f"negative_control={nc!r}, so there is no control level to "
                  f"centre on.")
    elif intercept_mode == 'value':
        # PINNED, NOT NUDGED. Shifting the response by the number and
        # suppressing the term fits `y = c + terms`, so the intercept is
        # exactly what was asked for -- an estimated one would land near it
        # and read as though the number had been a suggestion.
        intercept_offset = float(intercept_value or 0.0)
        if intercept_offset:
            df = df.copy()
            df[dependent_variable] = (
                np.asarray(df[dependent_variable], dtype=float)
                - intercept_offset)
        print(f"Intercept pinned at {intercept_offset:.6g}: every "
              f"coefficient reads as its distance from that value.")

    # The QC report needs the design that was fitted. The mixed branch below
    # never builds one -- fit_mixed_model takes the formula and the frame and
    # keeps its design to itself -- so X and y simply do not exist there, and
    # this stays None to say so rather than letting a NameError find out.
    qc_design = None

    # INSTRUCTION 122: BLOCK ON THE SCREEN WHEN THERE IS MORE THAN ONE.
    #
    # Two screens sharing a guide library are stacked into one frame and fitted
    # together -- twice the wells -- but only if the screen is in the model. A
    # systematic difference between two experiments that is not a term gets
    # charged to whichever guides are over-represented in one of them, which
    # is a false hit that looks exactly like a real one.
    #
    # Decided from the DATA, not from a setting, because the wrong answer is
    # silent in both directions: a constant screenID term makes the design
    # rank-deficient and statsmodels answers with a pseudo-inverse instead of
    # refusing, so a single-screen run would come back with standard errors
    # that mean nothing and no error anywhere.
    block_screen = screen_is_blockable(df)
    if block_screen:
        print(f"Blocking on {df['screenID'].nunique()} screens: "
              f"{sorted(df['screenID'].astype(str).unique())}")

    # THE MIXED BRANCH IS THE NESTED MODEL, and it is reached by NAME as well
    # as by the random row/column flag. `regression_type='mixed'` used to fall
    # through to the fixed-effects branch and be fitted by regression_model
    # with groups=plateID, which is a different model from the one
    # fit_mixed_model built -- two things called 'mixed' in one function. There
    # is one now: gene fixed, guide random nested inside it.
    if regression_type == 'mixed' or random_row_column_effects:
        regression_type = 'mixed'
        level = 'gene'
        formula = prepare_formula(
            dependent_variable,
            random_row_column_effects=random_row_column_effects,
            block_screen=block_screen, level='gene',
            model_plate_position=model_plate_position,
            intercept=intercept)
        mixed_model, coef_df = fit_mixed_model(
            df, formula, level_dst,
            random_row_column_effects=random_row_column_effects,
            regression_backend=regression_backend)
        model = mixed_model
    else:
        formula = prepare_formula(dependent_variable,
                                  random_row_column_effects=False,
                                  block_screen=block_screen, level=level,
                                  model_plate_position=model_plate_position,
                                  intercept=intercept)
        y, X = dmatrices(formula, data=df, return_type='dataframe')
        # Rows patsy actually kept. Every per-row vector handed to the model
        # below - weights, groups, exposure - is taken through this index, so
        # a row patsy dropped (a NaN predictor) cannot shift the rest by one.
        model_index = y.index

        # THE HOUSE-STYLE DISTRIBUTIONS, not the old ones. spacr.figures.
        # distributions draws the same two panels -- the guide fractions and
        # the response -- in the one visual system, and writes the same file
        # names, so the grid, the queue and the tests still find them.
        #
        # Falls back to the old plot_histogram if the new module cannot draw
        # them, because a figure is not worth losing a fit over.
        #
        # DRAWN ONCE PER RUN, NOT ONCE PER FIT. They describe the data, which
        # is the same data both levels are fitted to, so a two-fit run that
        # drew them twice would put two identical panels on the figure grid.
        if draw_shared_panels and not _show_well_distributions(
                df, dependent_variable, dst, plot=plot):
            plot_histogram(y, dependent_variable, dst=dst)
            plot_histogram(df, 'fraction', dst=dst)

        # No scaling, for any type. The design this pipeline builds is
        # one level's fraction terms plus row and column dummies: dummies and
        # fractions, already on one common [0, 1] scale, so there is nothing
        # for a scaler to put on a common footing. What MinMax scaling DID do
        # was divide each gRNA's column by that gRNA's own maximum fraction,
        # which rescales its coefficient by a different constant per feature -
        # and the volcano plot then ranks gRNAs against each other on those
        # coefficients. It also zeroed the intercept column outright (see
        # scale_variables), fitting every unscaled-exempt model through the
        # origin. The exemption list this replaces named lasso and ridge as
        # "already 0/1 from one-hot categorical predictors", which is the
        # right reason - it is just as true of every other type here.
        #
        # scale_variables stays public and correct for callers that scale
        # their own designs; this pipeline no longer needs it.
        print('Data will not be scaled: the design is fractions and dummies '
              'on one common scale, and scaling it per column would rescale '
              'each gRNA coefficient by a different constant.')

        # Per-well cell counts: var_weights for the binomial links, the WLS
        # weights, and the Poisson exposure for the horseshoe model.
        weights = df['cell_count'].loc[model_index] if 'cell_count' in df.columns else None
        # `mixed` never reaches here any more -- it is caught by name at the
        # top and fitted by fit_mixed_model with the gene/guide nesting -- so
        # there is no grouping vector to build on this branch.
        groups = None

        print(f'Performing {regression_type} {level}-level regression')
        model = regression_model(
            X, y,
            regression_type=regression_type,
            groups=groups,
            alpha=alpha,
            cov_type=cov_type,
            weights=weights,
            l1_ratio=l1_ratio,
            quantile=quantile,
            hinge_threshold=hinge_threshold,
            huber_t=huber_t,
            spline_knots=spline_knots,
            spline_degree=spline_degree,
            exposure=weights,
            group_lasso_lambda=group_lasso_lambda,
            rra_alpha=rra_alpha,
            rra_permutations=rra_permutations,
            regression_backend=regression_backend,
            verbose=verbose,
            # WHAT THE RESPONSE IS CALLED AND WHAT WAS DONE TO IT (182 A/C).
            # The family sniffer sees values, not a column, so without these
            # it said "Data strictly between 0 and 1" about a logged
            # proportion and a reader could not tell which scale it had
            # looked at.
            response_name=str(y.name) if hasattr(y, 'name') else '',
            transform=transform,
            glm_force_identity=glm_force_identity,
        )

        coef_df = process_model_coefficients(
            model, regression_type, X, y, nc, pc, controls,
            hinge_threshold=hinge_threshold, hinge_n_boot=hinge_n_boot)
        display(coef_df)
        qc_design = (X, y)

    # THE OLD VOLCANO IS OFF UNLESS ASKED FOR. "your new volcano plot is much
    # much faster than my old one so hide my old version behid a boolean that
    # defaults to off". It is not deleted -- it is what published figures were
    # made with -- but a run does not draw it now, and a run that does draw it
    # produces two volcanoes in two visual idioms, which is the thing that
    # made the grid look wrong.
    if plot and legacy_volcano:
        # plot.volcano_plot is keyword-only past its first argument and has no
        # defaults for the two column names, so the old positional
        # volcano_plot(coef_df, volcano_path) raised TypeError on every
        # plot=True call. coef_df is the frame built by
        # process_model_coefficients / fit_mixed_model, whose columns are
        # feature / coefficient / p_value; the coefficients are already on a
        # signed log-odds-style scale, so no x transform is applied.
        volcano_plot(
            coef_df,
            fold_change_col='coefficient',
            p_value_col='p_value',
            name_col='feature',
            x_transform='none',
            save_path=volcano_path,
            show=False,
        )

    # After the volcano, so the report can name a file that is already on disk.
    # Skipped without a destination: regression_qc_report raises on a falsy dst
    # on purpose, and a fit run with dst=None has nowhere to put diagnostics.
    qc_manifest = None
    if qc and qc_design is not None and level_dst:
        # KEPT, not just written. Instruction 115: the manifest holds the
        # per-panel VERDICT and the renderer that drew each panel, which is
        # the thing a caller most wants out of a run -- and until now it went
        # to disk and nowhere else, so `perform_regression`'s own return value
        # could not say whether the fit it just handed back was diagnosable.
        qc_manifest = _write_regression_qc(
            model, qc_design[0], qc_design[1], df, level_dst,
            coef_df=coef_df, regression_type=regression_type,
            volcano_path=volcano_path if plot else None)

    # THE HOUSE-STYLE PANELS. Asked for on 2026-08-16: "the all figures
    # section should look like a publication ready figure ... with each panel
    # having an uppercase letter ... and be on a grid", and "there are no
    # additional plots that i asked for and all the old plotts look exactly
    # the same".
    #
    # SHOWN, not merely written. A PDF on disk changed nothing about what the
    # application displays, which is exactly the complaint -- the grid still
    # held the same old pictures. These go through plt.show(), which the Qt
    # bridge intercepts, so each panel arrives in the figure queue and lands
    # on the grid as its own lettered cell.
    if level_dst:
        _show_house_style_panels(coef_df, plot=plot)
        _write_regression_sheet(coef_df, level_dst)

    # THE PARAGRAPH. "id also like a little written summary at the end in the
    # console saying what is significant and so on". Printed last so it is the
    # thing left on screen when a run finishes, and built from the same
    # numbers the panels are -- a summary that recomputed them could disagree
    # with the pictures beside it and a reader could not tell which was wrong.
    try:
        from .figures.summary import summarise

        text = summarise(coef_df)
        if text:
            import textwrap
            print()
            print("SUMMARY")
            print(textwrap.fill(text, 88))
    except Exception as error:  # noqa: BLE001 - never lose a run over prose
        print(f"Could not summarise the run: {error}")

    # WHICH MODEL PRODUCED THIS ROW, carried on the table itself. Two fits
    # write two tables and the volcano chooses between them; a row that cannot
    # say which family it belongs to is a row whose q value cannot be read.
    coef_df = coef_df.copy()
    coef_df['level'] = level
    # THE MANIFEST RIDES ON THE FRAME (115). `regression` returns a 3-tuple
    # that `regression_levels` and every caller unpack positionally, so
    # growing it would be a change to all of them for one optional fact.
    # `.attrs` is pandas' own place for exactly this and survives the frame
    # being passed around; a caller that does not know about it is unaffected.
    if qc_manifest is not None and coef_df is not None:
        coef_df.attrs["qc_manifest"] = qc_manifest
    return model, coef_df, regression_type


def regression_levels(df, csv_path, dependent_variable='predictions',
                      regression_type=None, level='both', dst=None, **kwargs):
    """Fit every level the run asked for, SEPARATELY, and return one per level.

    THIS IS THE TWO-FIT ENTRY POINT, and the reason it exists is that the one
    design spaCR used to fit cannot be fitted at all. ``gene_fraction`` is the
    SUM of the gene's gRNA fractions, so

        ``y ~ fraction:grna + gene_fraction:gene + rowID + columnID``

    puts a block of columns and their own sums into one design. Measured on
    the reference TSG101 screen: 1248 parameters at rank 862 -- a
    386-dimensional EXACT null space -- and the fit statsmodels returned had a
    residual sum of squares bit-identical to the one you get by adding seven
    times a null vector to it. See :data:`COLLINEAR_FORMULA_FRAGMENT`.

    Two fits, two tables, TWO CORRECTIONS. Each fit is its own
    multiple-testing family and is corrected within itself. Pooling them would
    be wrong twice over: they are not independent -- same wells, and the gene
    regressor IS the sum of the guide regressors -- and doubling the family
    size costs power for no protection. :func:`perform_regression` applies the
    correction per level and writes ``results_grna.csv`` and
    ``results_gene.csv``.

    ``regression_type='mixed'`` fits ONCE and returns one entry, ``'gene'``:
    that model has both levels inside it already, the gene as a fixed effect
    and the guide as a random effect nested in the gene. Its guide output is
    BLUPs, which is why it cannot be split into two testing families.

    :param level: ``'both'`` (default), ``'grna'`` or ``'gene'``.
    :param dst: the run folder. With more than one fit each level's FIGURES go
        into ``<dst>/<level>/`` so they cannot overwrite each other; the
        tables stay in ``<dst>``, where every consumer looks for them.
    :param kwargs: passed straight through to :func:`regression`.
    :returns: ``dict`` mapping level to ``(model, coef_df, regression_type)``,
        in fit order.
    :raises ValueError: for a level that is not one of :data:`LEVEL_CHOICES`.
    """
    import os

    levels = resolve_levels(regression_type, level)
    if regression_type == 'mixed' and str(level).strip().lower() != 'both':
        # SAID, not silently overridden. The GUI greys `level` out for mixed
        # (instruction 106), but a script can still set it, and a run that
        # quietly ignored it would hand back a gene table to a caller who
        # asked for a guide table.
        print(f"regression_type='mixed' fits the gene fixed with guides "
              f"random nested inside, so it is already both levels and "
              f"level={level!r} is not read. Its guide output is BLUPs, not "
              f"coefficients with p-values.")

    fits = {}
    for index, one in enumerate(levels):
        level_dst = dst
        if dst and len(levels) > 1:
            level_dst = os.path.join(str(dst), one)
            os.makedirs(level_dst, exist_ok=True)
        print(f"Fitting level {index + 1} of {len(levels)}: {one}")
        fits[one] = regression(
            df, csv_path, dependent_variable=dependent_variable,
            regression_type=regression_type, dst=dst, level=one,
            level_dst=level_dst, draw_shared_panels=(index == 0), **kwargs)
        # check_distribution may have chosen the type on the first fit; the
        # second must be the SAME model, not a second auto-selection that
        # could pick differently and give two tables from two backends.
        regression_type = fits[one][2]
    return fits


def _show_well_distributions(frame, response_name, dst, plot=True):
    """Draw the guide-fraction and response distributions in the house style.

    :returns: True when they were drawn. False sends the caller back to the
        original ``plot_histogram``, because a figure is not worth losing a
        fit over.
    """
    try:
        import matplotlib.pyplot as plt

        from .figures import distributions
    except Exception as error:  # noqa: BLE001
        print(f"Could not load the distribution panels: {error}")
        return False

    drawn = 0
    # The response panel takes the COLUMN NAME; the fraction panel takes
    # nothing. Passing the response series would be handing a panel the
    # values when it wants to know which column to read and label.
    per_panel = {"response": {"column": response_name}, "guide_fraction": {}}
    for key in distributions.ORDER:
        try:
            figure, panel = distributions.build_panel(
                key, frame, **per_panel.get(key, {}))
        except Exception as error:  # noqa: BLE001
            print(f"Distribution panel {key} did not draw: {error}")
            continue
        if not getattr(panel, "drawn", False):
            plt.close(figure)
            continue
        figure.set_label(panel.title)
        figure._spacr_title = panel.title
        if dst:
            try:
                from .plot import save_figure

                name = distributions.FILENAMES[key].format(
                    response=response_name)
                # THROUGH THE PREFERENCE, not a literal .pdf. A user who set
                # "PNG" in Preferences and got PDFs anyway is the exact
                # complaint `save_figure` was written to end, and the new
                # panels quietly re-introduced it -- three times.
                save_figure(figure, os.path.join(str(dst), f"{name}.pdf"),
                            bbox_inches="tight")
            except Exception:
                pass
        if plot:
            plt.show()
        # RELEASE THE MANAGER, KEEP THE FIGURE. `plt.show` is the bridge
        # hand-off and the figure has to still be registered during it;
        # after that, pyplot's registry is the only thing holding it, and a
        # fit that draws two panels per run and never releases them is how
        # a long session ends up with hundreds of live canvases. The Figure
        # object survives `close` -- whatever the bridge kept still draws.
        plt.close(figure)
        drawn += 1
    return drawn > 0


def _show_plates(frame, variable, dst):
    """Draw every plate as one small multiple. True when it was drawn."""
    try:
        import matplotlib.pyplot as plt

        from .figures.plates import build_plates
    except Exception as error:  # noqa: BLE001
        print(f"Could not load the plate panel: {error}")
        return False
    try:
        figure, panel = build_plates(frame, variable, grouping="mean",
                                     min_max="allq", min_count=0)
    except Exception as error:  # noqa: BLE001
        print(f"The plate panel did not draw: {error}")
        return False
    if not getattr(panel, "drawn", False):
        plt.close(figure)
        return False
    figure.set_label(panel.title)
    figure._spacr_title = panel.title
    if dst:
        try:
            from .plot import save_figure

            save_figure(
                figure,
                os.path.join(str(dst), f"plate_heatmap_{variable}.pdf"),
                bbox_inches="tight")
        except Exception:
            pass
    plt.show()
    plt.close(figure)               # same hand-off, same release
    return True


def _show_house_style_panels(coef_df, plot=True):
    """Draw each house-style panel and hand it to whatever is watching.

    One figure per panel rather than one sheet, because the grid puts each on
    its own lettered cell and a single composite would be one unreadable
    tile. The sheet is written to disk as well, for the version that goes in
    a paper.

    Never fatal: the fit is already done and losing a run over a figure would
    be the worst possible trade.
    """
    if coef_df is None or not len(coef_df):
        return 0
    try:
        import matplotlib.pyplot as plt

        from .figures import SHEET_ORDER, build_panel
    except Exception as error:  # noqa: BLE001
        print(f"Could not load the figure style: {error}")
        return 0
    shown = 0
    for key in SHEET_ORDER:
        try:
            figure, panel = build_panel(key, coef_df)
        except Exception as error:  # noqa: BLE001
            print(f"Panel {key} did not draw: {error}")
            continue
        if not panel.drawn:
            plt.close(figure)
            continue
        # The figure carries its own name, so the grid captions it "volcano"
        # rather than "fig_00003" -- a temp file's stem is an implementation
        # detail of how the picture reached the screen, not a caption.
        figure.set_label(panel.title)
        figure._spacr_title = panel.title
        if plot:
            plt.show()
        plt.close(figure)           # the hand-off is done; release the manager
        shown += 1
    if shown:
        print(f"Drew {shown} regression panels in the house style.")
    return shown


def _write_regression_sheet(coef_df, dst):
    """Write ``<dst>/regression_figure.pdf`` and its legend.

    Never fatal: a fit that produced a coefficient table has already done the
    work, and losing the run because a panel could not be drawn would be the
    worst possible trade.
    """
    import os

    if coef_df is None or not len(coef_df):
        return None
    try:
        from .figures import build_sheet

        sheet = build_sheet(coef_df, width='double', target='print')
        folder = str(dst)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'regression_figure.pdf')
        from .figure_sink import publish

        # PUBLISHED, not merely saved. Instruction 139 C: saving a figure and
        # showing it are the SAME event. This is THE publication figure of a
        # regression run and it was the one figure of the run nobody could
        # look at -- written and then closed in the next breath, so no
        # `plt.show()` ever walked past it and the gallery never held it.
        #
        # `publish` still writes through `spacr.plot.save_figure`, so the
        # sheet remains the last figure that should ignore the format and
        # resolution the user chose.
        path = publish(sheet.figure, path, bbox_inches='tight') or path
        with open(os.path.join(folder, 'regression_figure_legend.txt'),
                  'w') as handle:
            handle.write(sheet.legend() + '\n')
        try:
            import matplotlib.pyplot as plt
            plt.close(sheet.figure)
        except Exception:
            pass
        print(f"Wrote the regression figure to {path} "
              f"({len(sheet.panels)} panels"
              + (f", {len(sheet.skipped)} not applicable"
                 if sheet.skipped else '') + ').')
        return path
    except Exception as error:  # noqa: BLE001 - never lose a run over a figure
        print(f"Could not draw the regression figure: {error}")
        return None

#: What a run's statsmodels summary is written as, and every older name the
#: reader still accepts. NEWEST FIRST -- the first one found wins.
#:
#: The name used to be ``mode_summary.csv``, which was wrong twice over:
#: "mode" is a typo for "model", and the content is the statsmodels TEXT
#: summary, never CSV. A name that does not follow the file is a path nobody
#: can open, so the format is corrected going forward and the old names are
#: still READ -- a run finished last month keeps its summary.
#: How many coefficient rows the console will print before it stops and
#: points at the file instead. The header of a statsmodels summary is about
#: twenty lines; a screen's coefficient table is hundreds.
CONSOLE_COEFFICIENT_LIMIT = 12


def fit_quality_note(model) -> str:
    """Return a one-line goodness-of-fit summary for a fitted GLM.

    McFadden's pseudo-R-squared compares log-likelihoods, and it is the
    appropriate summary for a GLM with a discrete response. A Gaussian
    identity-link fit instead reports ordinary R-squared because its
    likelihood is a density and the McFadden ratio is not interpretable on
    the usual zero-to-one scale.

    :param model: a fitted statsmodels GLM result.
    :returns: A labelled goodness-of-fit line for the console.
    """
    family = getattr(model, 'family', None)
    if isinstance(family, sm.families.Gaussian):
        try:
            resid = np.asarray(model.resid_response, dtype=float).reshape(-1)
            observed = resid + np.asarray(
                model.fittedvalues, dtype=float).reshape(-1)
            centred = observed - observed.mean()
            total = float(np.dot(centred, centred))
            residual = float(np.dot(resid, resid))
        except (AttributeError, TypeError, ValueError):
            return "R²: not available for this fit"
        if not np.isfinite(total) or total <= 0:
            return "R²: not available for this fit (the response is constant)"
        return (f"R²: {1.0 - residual / total:.4f}  (ordinary R², not "
                f"McFadden -- this is a Gaussian identity-link fit)")
    # THE NULL LOG-LIKELIHOOD, NOT THE NULL DEVIANCE. This read
    # `model.null_deviance / -2`, which equals the null log-likelihood only
    # when the saturated log-likelihood is zero -- true for 0/1 binomial data
    # and false for the per-well PROPORTIONS this pipeline actually fits, so
    # the ratio mixed two conventions. statsmodels fits the null model itself
    # and exposes it as `llnull`.
    try:
        null_value = model.llnull
        if null_value is None:
            raise AttributeError("no null log-likelihood on this result")
        llf, null = float(model.llf), float(null_value)
    except (AttributeError, TypeError, ValueError):
        try:
            llf = float(model.llf)
            null = float(model.null_deviance) / -2.0
        except (AttributeError, TypeError, ValueError):
            return "McFadden's R²: not available for this fit"
    if not np.isfinite(null) or null == 0:
        return "McFadden's R²: not available for this fit"
    return mcfadden_note(1.0 - (llf / null))


def mcfadden_note(r2) -> str:
    """Format McFadden's pseudo-R² and flag a negative value.

    A negative value means the fitted model predicts the response worse than
    an intercept-only model. The returned note explains that the coefficients
    should not be interpreted and points to a common cause: applying a response
    transform that duplicates the fitted family's link.

    :param r2: Pseudo-R² value, or a value convertible to ``float``.
    :returns: One-line diagnostic text suitable for a console or report.
    """
    try:
        value = float(r2)
    except (TypeError, ValueError):
        return "McFadden's R²: not available for this fit"
    if value < 0:
        return (
            f"McFadden's R²: {value:.4f}  <-- NEGATIVE. This fit predicts the "
            f"response WORSE than its own intercept, so its coefficients and "
            f"P values do not describe the data. The usual cause is a "
            f"response transformed twice: check that `transform` is not "
            f"applying a log or logit that the family's link already applies."
        )
    return f"McFadden's R²: {value:.4f}"


def summary_for_console(model, *, verbose=False,
                        limit=CONSOLE_COEFFICIENT_LIMIT) -> str:
    """Return a statsmodels summary sized for terminal output.

    When the coefficient table exceeds ``limit``, the diagnostic header and
    notes are retained while the table is replaced by a pointer to the saved
    summary and the sortable Coefficients view. Set ``verbose=True`` to return
    the complete statsmodels rendering.

    :param model: Fitted model result with a ``summary()`` method.
    :param verbose: Return the complete summary regardless of its size.
    :param limit: Maximum coefficient rows printed in compact mode.
    :returns: Complete or compact plain-text model summary.
    """
    try:
        text = str(model.summary())
    except Exception as error:                          # noqa: BLE001
        return (f"statsmodels could not render a summary for this fit "
                f"({type(error).__name__}: {error}).")
    if verbose:
        return text
    # THE COEFFICIENT ROWS ONLY. Located from the column header -- the line
    # carrying "coef" and "std err" -- and ended at the next '=' rule, rather
    # than by taking everything after the last separator. That simpler cut
    # swallowed the notes table with the rows and MISCOUNTED the coefficients
    # by however many notes the family happens to print.
    #
    # The notes are KEPT. Durbin-Watson and, above all, the condition number
    # are how a reader sees the collinearity that a screen's design has, and
    # they are six lines.
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines)
                   if "coef" in line and "std err" in line), None)
    if header is None:
        return text
    rule = next((i for i in range(header + 1, len(lines))
                 if set(lines[i].strip()) == {"-"}), None)
    if rule is None:
        return text
    end = next((i for i in range(rule + 1, len(lines))
                if set(lines[i].strip()) == {"="}), len(lines))
    rows = [line for line in lines[rule + 1:end] if line.strip()]
    if len(rows) <= limit:
        return text
    return "\n".join(
        lines[:rule + 1]
        + [f"  {len(rows)} coefficients — not printed here. They are in the "
           f"run's model_summary.txt and in the Coefficients tab, which sorts "
           f"and filters them. Set verbose=True to print them."]
        + lines[end:])


SUMMARY_FILENAME = 'model_summary.txt'
SUMMARY_FILENAMES = (SUMMARY_FILENAME, 'mode_summary.csv', 'summary.csv')


def save_summary_to_file(model, file_path=SUMMARY_FILENAME):
    """
    Write ``model.summary().as_text()`` to ``file_path`` as plain text.

    The content is the statsmodels text summary, never CSV -- which is why
    the default name is :data:`SUMMARY_FILENAME` and no longer
    ``summary.csv``. Older runs on disk wrote ``mode_summary.csv``; every
    reader in this repository accepts both, see :data:`SUMMARY_FILENAMES`.

    :param model: Fitted statsmodels results object.
    :param file_path: Destination path. Default :data:`SUMMARY_FILENAME`.
    :returns: the path written, or ``None`` if there was nothing to write.

    NEVER RAISES INTO A FINISHED RUN. This is called after every table has
    been written; a backend whose ``summary()`` throws must not take the run
    down with it, and the caller is told by the ``None`` rather than by a
    traceback.
    """
    summary = getattr(model, 'summary', None)
    if not callable(summary):
        return None
    try:
        summary_str = summary().as_text()
    except Exception as error:  # noqa: BLE001 - a summary is not worth a run
        print(f"Could not render the model summary: "
              f"{type(error).__name__}: {error}")
        return None
    folder = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(summary_str)
    return file_path


def _split_prc(text):
    """Return ``(plateID, rowID, columnID)`` for one ``prc`` well key.

    Parse from right to left because only the leading plate ID may contain the
    key separator. This preserves plate names such as ``'exp1_plate1'``.

    The row and column are returned exactly as they appear — nothing is
    canonicalised, because the caller rebuilds ``prc`` from these columns and
    a rewritten token would change the identity rows are joined on.

    Unescape the plate component to match :func:`spacr.schema.compose_prc` and
    :func:`spacr.schema.parse_prcf`; return row and column tokens unchanged.

    For keys with more than three components, require the final tokens to be
    a recognizable row/column pair. This accepts a plate containing the
    separator while rejecting a deeper ``prcf`` or ``prcfo`` key. Exactly
    three components remain accepted without positional-token validation.

    :param text: a ``prc`` key, e.g. ``'plate1_r1_c1'``.
    :returns: ``(plateID, rowID, columnID)``.
    :raises spacr.schema.KeyParseError: when ``text`` has fewer than three
        components, i.e. it is not a well key at all, or when it has more
        than three and the trailing pair is not a row and a column.
    """
    key = str(text).strip()
    parts = key.split(schema.KEY_SEPARATOR)
    if len(parts) < 3:
        raise schema.KeyParseError(
            f'{text!r} is not a prc: expected plate_row_column, got '
            f'{len(parts)} component(s).')
    plate = schema.KEY_SEPARATOR.join(parts[:-2])
    row, column = parts[-2], parts[-1]
    if not plate.strip():
        raise schema.KeyParseError(
            f'{text!r} is not a prc: it has no plate.')
    if not row.strip() or not column.strip():
        # An empty row or column is not a missing token, it is a key every
        # well of the plate shares: group on it and the wells merge.
        raise schema.KeyParseError(
            f'{text!r} is not a prc: its row is {row!r} and its column is '
            f'{column!r}, and an empty one identifies no well — every well of '
            f'{plate!r} would be grouped together.')
    if len(parts) > 3 and not _is_row_column_pair(row, column):
        raise schema.KeyParseError(
            f'{text!r} is not a prc: it has {len(parts)} components and its '
            f'last two, {row!r} and {column!r}, are not a row and a column. '
            f'{_name_deeper_key(parts)}'
            f'If this really is a plate id containing '
            f'{schema.KEY_SEPARATOR!r}, its row and column must be written '
            f'the way spaCR writes them (r<N>/letters and c<N>/digits) for '
            f'the plate to be separable from them.')
    return schema.unescape_filename_component(plate), row, column


#: ``prc`` for a whole frame. :mod:`spacr.schema` owns it -- one place
#: composes a key -- and this name is kept because seven call sites in this
#: module use it.
_compose_prc_column = schema.compose_prc_column


def _is_row_column_pair(row, column):
    """True when ``(row, column)`` is recognisably a well's row and column.

    Deliberately narrow: it is the guard that stops :func:`_split_prc` from
    absorbing a ``prcf`` into an underscored plate id, so it must reject a
    ``(columnID, fieldID)`` pair and a ``(fieldID, objectID)`` pair.

    :param row: candidate ``rowID`` token.
    :param column: candidate ``columnID`` token.
    :returns: whether the pair can be a row and a column.
    """
    row_text, column_text = str(row).strip(), str(column).strip()
    if not row_text or not column_text:
        return False
    if schema.is_positional_pair(row_text, column_text):
        # parse_well puts an unrecognisable well into both slots verbatim, so
        # an equal unprefixed pair is that passthrough and not a prcf tail
        # (a field never equals the column it sits in).
        return True
    if row_text[:1].lower() == schema.KEY_PREFIXES[schema.ROW_KEY]:
        row_ok = schema.row_index(row_text) is not None
    else:
        row_ok = schema.row_index_from_letters(row_text) is not None
    if not row_ok:
        return False
    if column_text[:1].lower() == schema.KEY_PREFIXES[schema.COLUMN_KEY]:
        return schema.column_index(column_text) is not None
    return column_text.isdigit()


def _name_deeper_key(parts):
    """Return a sentence naming the deeper key ``parts`` looks like, or ''.

    Split out of :func:`_split_prc` only so the error it raises can say
    *which* mistake was made instead of describing the shape and leaving the
    caller to work it out.

    :param parts: the separator-split components of the rejected key.
    :returns: a sentence ending in a space, or ``''`` when the key does not
        look like a ``prcf`` / ``prcfo`` / timepoint key.
    """
    tail = parts[-1]
    if schema.object_index(tail) is not None and len(parts) >= 5:
        return ('That is a prcfo (plate_row_column_field_object); '
                '_split_prc takes a prc. Use schema.parse_prcfo. ')
    if schema.field_index(tail) is not None:
        return ('That is a prcf (plate_row_column_field); _split_prc takes a '
                'prc. Use schema.parse_prcf, or drop the field first. ')
    if schema.time_index(tail) is not None:
        return ('That ends in a timepoint; a prc has none. Aggregate the '
                'timepoints away before keying on the well. ')
    return ''


def _assign_prc_parts(df, column=schema.PRC_KEY,
                      columns=schema.WELL_KEY_COLUMNS):
    """Split ``df[column]`` into plate / row / column and assign them onto ``df``.

    The frame-level counterpart of :func:`_split_prc`, and the ``prc`` sibling
    of :func:`_assign_prcfo_parts`.

    :param df: frame carrying ``column``.
    :param column: name of the ``prc`` column. Default ``'prc'``.
    :param columns: names to assign, in plate / row / column order.
    :returns: ``df``, mutated in place and returned for chaining.
    :raises spacr.schema.KeyParseError: when any value is not a ``prc``.
    """
    parsed = [_split_prc(value) for value in df[column]]
    for position, name in enumerate(columns):
        df[name] = [part[position] for part in parsed]
    return df


def resolve_auto_inference(data, settings, *, well_column='prc',
                           guide_column='grna'):
    """Choose ``analysis_mode`` for ``inference='auto'`` from the design.

    The simultaneous model estimates one coefficient per guide from the wells,
    so it needs more wells than guides -- with an intercept and any plate fixed
    effects on top -- before those coefficients are identifiable at all. Below
    that the design matrix is rank deficient: statsmodels still returns a
    number for every guide, but the numbers are one arbitrary solution out of
    infinitely many, and their P values describe nothing.

    That is not a hypothetical. The screen this was written for has 824 guides
    in 587 analysed wells; the published fit had 825 parameters, rank 579 and
    8 residual degrees of freedom, and refitting it did not reproduce its own
    coefficients.

    ``auto`` therefore picks the permutation test whenever the simultaneous fit
    would be unidentifiable, and says so. It is deliberately conservative: it
    needs a real margin (``_IDENTIFIABILITY_MARGIN`` wells per guide) rather
    than a bare majority, because a design that only just fits is one dropped
    well away from not fitting.

    Anything other than ``inference='auto'`` is returned untouched, so an
    explicit choice is never overridden.

    :returns: ``(analysis_mode, reason)``. ``reason`` is a sentence naming the
        counts, suitable for the log and for the Methods section.
    """
    inference = str(settings.get('inference', 'auto')).strip().lower()
    if inference != 'auto':
        return settings.get('analysis_mode', 'regression'), (
            f"inference={inference!r} was set explicitly.")

    # AUTO MUST NOT CHOOSE A MODE THAT CANNOT RUN ON THIS DATA.
    #
    # The permutation test needs one row per WELL. With per-object rows it
    # refuses -- correctly, and with a clear message -- but "auto" means
    # "choose for me", and choosing something that raises the moment it is
    # used is not a choice. `agg_type is None` is the reliable signal, not
    # `analysis_unit`: some regression types force it to None themselves
    # (quantile fits objects by construction), so a user who set
    # analysis_unit='well' still ends up with object rows.
    # ABSENT IS THE DEFAULT, NOT THE OPPOSITE OF IT. `settings.get('agg_type')`
    # returns None for a key that was never set as readily as for one set to
    # None, and those are opposite answers: the shipped default is
    # agg_type='mean' with analysis_unit='well', so a dict that has not been
    # through `set_default_analysis_settings` -- a sweep trial, a refit, a
    # caller that assembled its own -- was read as PER OBJECT and auto then
    # chose the simultaneous model for a design it could not identify. That is
    # the one outcome this function exists to prevent.
    per_object = str(settings.get('analysis_unit') or 'well').lower() != 'well'
    aggregated = settings['agg_type'] if 'agg_type' in settings else 'mean'
    if per_object or aggregated is None:
        return 'regression', (
            "auto chose the simultaneous model: the rows are one per OBJECT "
            "(agg_type is None or analysis_unit is not 'well'), and the "
            "permutation test needs one row per well. Set an agg_type such "
            "as 'mean' if the permutation test is wanted.")

    try:
        n_wells = int(data[well_column].nunique())
        n_guides = int(data[guide_column].nunique())
    except (KeyError, TypeError):
        # Cannot measure the design; the safe default is the test that stays
        # valid at any width.
        return 'guide_permutation', (
            "The design could not be measured, so the permutation test was "
            "used because it is valid regardless of the number of guides.")

    blocks = 0
    block_column = str(settings.get('guide_permutation_block', 'plateID'))
    if block_column in getattr(data, 'columns', ()):
        blocks = max(int(data[block_column].nunique()) - 1, 0)
    # intercept + block fixed effects + one coefficient per guide
    parameters = 1 + blocks + n_guides
    required = parameters * _IDENTIFIABILITY_MARGIN

    if n_wells >= required:
        return 'regression', (
            f"auto chose the simultaneous model: {n_wells} analysed wells for "
            f"{parameters} parameters ({n_guides} guides + intercept + "
            f"{blocks} block terms), at least the {_IDENTIFIABILITY_MARGIN}x "
            f"margin required.")
    return 'guide_permutation', (
        f"auto chose the permutation test: {n_wells} analysed "
        f"wells cannot identify {parameters} simultaneous parameters "
        f"({n_guides} guides + intercept + {blocks} block terms). Each guide "
        f"is tested as a marginal association instead. Set "
        f"inference='parametric' to force the simultaneous fit.")


#: How many wells per estimated parameter ``auto`` insists on before it will
#: choose the simultaneous model. 1.0 would accept a design with zero residual
#: degrees of freedom, which fits perfectly and tests nothing.
_IDENTIFIABILITY_MARGIN = 2.0


#: Fraction of count wells that must survive the score join before the run is
#: allowed to continue. Below this the two inputs are describing different
#: plates, and every number downstream is computed on whatever happened to
#: overlap.
_MINIMUM_PAIRED_WELL_FRACTION = 0.5


def normalize_regression_input_pairs(settings):
    """Return explicit ``score``/``count`` rows, migrating legacy lists.

    New settings store ``paired_data``. Older files remain valid: their flat
    lists are zipped positionally, exactly matching the former behaviour, and
    the migration is reported so the invisible legacy assumption is visible.
    """
    from itertools import zip_longest

    rows = settings.get('paired_data') or []
    migrated = False
    if rows:
        if not isinstance(rows, (list, tuple)):
            raise ValueError("paired_data must be a list of score/count rows")
        pairs = []
        for index, raw in enumerate(rows):
            if not isinstance(raw, dict):
                raise ValueError(f"paired_data[{index}] must be a mapping")
            pairs.append({
                'score': raw.get('score') or raw.get('score_data'),
                'count': raw.get('count') or raw.get('count_data'),
                'plate': raw.get('plate') or raw.get('plateID'),
                # THE MEASUREMENTS DATABASE SURVIVES THE ROUND TRIP.
                #
                # This dict is written back over `settings['paired_data']`
                # below, so any key it does not name is ERASED -- and the
                # settings CSV a run saves is what a user reloads. Without
                # this line a reloaded run comes back with an empty database
                # column and no sign that it ever had one. Nothing in the fit
                # reads it (the regression runs on scores and counts), which
                # is exactly why it would have gone unnoticed.
                'database': raw.get('database') or raw.get('measurements'),
            })
    else:
        def paths(value):
            if value is None:
                return []
            return list(value) if isinstance(value, (list, tuple)) else [value]

        scores = paths(settings.get('score_data'))
        counts = paths(settings.get('count_data'))
        pairs = [
            {'score': score, 'count': count, 'plate': None}
            for score, count in zip_longest(scores, counts)
        ]
        migrated = bool(pairs)
        if migrated:
            print("Legacy score_data/count_data lists were paired by position. "
                  "Review and save the new paired_data table to make that "
                  "relationship explicit.")

    if not pairs or not any(row['score'] for row in pairs) or not any(
            row['count'] for row in pairs):
        raise ValueError(
            "Regression needs at least one score CSV and one count CSV in "
            "paired_data.")
    settings['paired_data'] = pairs

    def unique(key):
        return list(dict.fromkeys(
            os.fspath(row[key]) for row in pairs if row.get(key)))

    # Existing downstream threshold/path helpers still consume these flat
    # views. They are projections of the explicit pairs, not a second pairing
    # mechanism, and repeated shared files are read only once there.
    settings['score_data'] = unique('score')
    settings['count_data'] = unique('count')
    return pairs, migrated


def load_regression_input_pairs(pairs):
    """Read paired inputs and resolve plate identity without filename guesses.

    Resolution order is own column, partner column, then pair-row order.
    Conflicting declarations are refused. Returns ``(count_frame,
    score_frame, audit_rows)``.
    """
    from .utils import correct_metadata

    score_frames = []
    count_frames = []
    seen_score_parts = set()
    seen_count_parts = set()
    audit = []

    # ONE PARSE PER FILE, NOT ONE PER PAIR ROW, AND NO PARSE AT ALL WHEN THE
    # FRAME IS ALREADY IN THIS PROCESS.
    #
    # The Measurements tab points EVERY pair row's score at the single merged
    # frame, so a four-plate screen handed the same file four times and this
    # parsed it four times. That file is 2.75 GB on a four-plate screen: the
    # process sat at 82% CPU with zero disk I/O -- reading it back out of the
    # page cache -- for minutes, having already written it.
    #
    # The merge that produced it runs in this same process, so `frame_handoff`
    # lets it offer the frame under the path it wrote; then there is no parse
    # to pay for and no 2.75 GB round trip through the filesystem. A caller
    # that offered nothing reads the file exactly as before.
    #
    # NO BLANKET COPY. Four copies of a 2.75 GB frame is eleven gigabytes of
    # allocation for a mutation that happens on ONE branch below -- stamping
    # `plateID` onto a file that names no plate. That branch copies; the
    # filtering branches build new frames of their own and cannot reach the
    # cached one.
    _parsed: dict = {}

    def read(path):
        import time

        if not path:
            return None
        key = frame_handoff.key_for(path)
        if key not in _parsed:
            offered = frame_handoff.held(path)
            if offered is not None:
                # SAY SO. Between the merge finishing and the fit starting the
                # run used to print nothing at all for minutes, which is what
                # made a working run look dead.
                note = frame_handoff.describe(path)
                print(f"Input {note}." if note else
                      f"Input {os.path.basename(key)} handed over in memory.",
                      flush=True)
                _parsed[key] = correct_metadata(offered)
            else:
                size = os.path.getsize(key) if os.path.exists(key) else 0
                print(f"Reading {os.path.basename(key)} "
                      f"({size / 1e6:.1f} MB)...", flush=True)
                started = time.time()
                frame = correct_metadata(tabular.read_table(os.fspath(path)))
                print(f"  {len(frame):,} rows in "
                      f"{time.time() - started:.1f} s.", flush=True)
                _parsed[key] = frame
        return _parsed[key]

    def plates(frame):
        if frame is None or 'plateID' not in frame.columns:
            return set()
        return {str(value) for value in frame['plateID'].dropna().unique()}

    for index, pair in enumerate(pairs):
        score = read(pair.get('score'))
        count = read(pair.get('count'))
        score_plates = plates(score)
        count_plates = plates(count)
        fallback = f'plate{index + 1}'

        if score_plates and count_plates:
            if score_plates == count_plates:
                resolved = score_plates
                rule = 'both files agree'
            elif count_plates < score_plates:
                # A single consolidated score file may intentionally be
                # reused in several rows, one per plate-specific count file.
                score = score[score['plateID'].astype(str).isin(count_plates)]
                resolved = count_plates
                rule = 'matched score rows to count-file plate subset'
            elif score_plates < count_plates:
                count = count[count['plateID'].astype(str).isin(score_plates)]
                resolved = score_plates
                rule = 'matched count rows to score-file plate subset'
            else:
                raise ValueError(
                    f"paired_data row {index + 1} conflicts: score file "
                    f"declares {sorted(score_plates)}, count file declares "
                    f"{sorted(count_plates)}. Pair files from the same "
                    "plate.")
        elif score_plates:
            # ONE SIDE HOLDS EVERY PLATE AND THE OTHER NAMES NONE. This is
            # the Measurements tab's own shape: `column_run_settings` points
            # every pair row's score at the single merged frame, which carries
            # all four plates, while a real count CSV carries
            # `row_name, column_name, grna_name, count` and no plate column at
            # all. Copying four plates onto a partner that names none is not
            # possible, and refusing was wrong: the pair ROW already says
            # which plate this row is, and that is the third resolution rule
            # this function documents. So use it -- and only when the plate it
            # names is one the partner actually holds, so a screen whose
            # plates are named anything else still refuses rather than
            # inventing a match.
            if (len(score_plates) > 1 and count is not None
                    and fallback in score_plates):
                score = score[score['plateID'].astype(str) == fallback]
                resolved = {fallback}
                rule = ('assigned from pair row order; score file holds '
                        f'{len(score_plates)} plates')
            else:
                resolved = score_plates
                rule = 'copied from score file'
        elif count_plates:
            if (len(count_plates) > 1 and score is not None
                    and fallback in count_plates):
                count = count[count['plateID'].astype(str) == fallback]
                resolved = {fallback}
                rule = ('assigned from pair row order; count file holds '
                        f'{len(count_plates)} plates')
            else:
                resolved = count_plates
                rule = 'copied from count file'
        else:
            resolved = {fallback}
            rule = 'assigned from pair row order'

        if (score is not None and count is not None and len(resolved) != 1
                and (not score_plates or not count_plates)):
            raise ValueError(
                f"paired_data row {index + 1} cannot copy {sorted(resolved)} "
                "onto a partner with no plateID: one file contains several "
                "plates. Split that partner or give it an explicit plateID.")
        # THE ONLY MUTATION, so the only place a copy is owed: `read` hands
        # back the cached frame itself, and stamping a plate onto it would
        # write the first pair row's plate into every later row's score.
        if score is not None and not score_plates:
            score = score.copy()
            score['plateID'] = next(iter(resolved))
        if count is not None and not count_plates:
            count = count.copy()
            count['plateID'] = next(iter(resolved))
        label = ', '.join(sorted(resolved))
        pair['plate'] = label
        audit.append({'row': index + 1, 'plate': label, 'rule': rule,
                      'score': pair.get('score'), 'count': pair.get('count')})
        print(f"Input pair {index + 1} ({label}): {rule}.")
        score_part = (os.fspath(pair.get('score')), tuple(sorted(resolved))) \
            if score is not None else None
        count_part = (os.fspath(pair.get('count')), tuple(sorted(resolved))) \
            if count is not None else None
        if score is not None and score_part not in seen_score_parts:
            score_frames.append(score)
            seen_score_parts.add(score_part)
        if count is not None and count_part not in seen_count_parts:
            count_frames.append(count)
            seen_count_parts.add(count_part)

    return (pd.concat(count_frames, ignore_index=True),
            pd.concat(score_frames, ignore_index=True), audit)


def _check_score_count_pairing(independent_df, dependent_df, merged_df, *,
                               well_column='prc', record=None):
    """Fail loudly when the score and count tables describe different wells.

    The two inputs are never paired file-to-file: each list is concatenated
    and the two are joined on ``prc`` (``plateID_rowID_columnID``). So the
    plate ID is the pairing key, and a plate ID that differs by one character
    between the two sides silently produces an empty join.

    That is not hypothetical. A legacy score CSV carries its plate in a
    ``plate`` column stamped ``pplate1``, while the sequencing counts carry
    ``plate1``. Before :func:`spacr.utils.correct_metadata` was fixed to
    normalise that after the legacy promotion, the join returned zero rows and
    the run continued for another two hundred lines before dying inside a plot
    with ``KeyError: 0`` -- an error naming neither the plates, the files, nor
    the join.

    :param independent_df: Count-table rows before the score/count join.
    :param dependent_df: Score-table rows before the score/count join.
    :param merged_df: Rows retained by the score/count join.
    :param well_column: Column containing the unique well identifier.
    :param record: Optional mutable mapping that receives the matched and
        unmatched well counts for the persisted run summary.
    :raises ValueError: when the join is empty, or retains less than
        :data:`_MINIMUM_PAIRED_WELL_FRACTION` of the smaller input's wells.
    """
    def _plates(frame):
        if well_column not in frame.columns:
            return []
        return sorted(frame[well_column].astype(str).str.split('_').str[0]
                      .dropna().unique())

    count_wells = independent_df[well_column].nunique() if \
        well_column in independent_df.columns else 0
    score_wells = dependent_df[well_column].nunique() if \
        well_column in dependent_df.columns else 0
    score_plates = _plates(dependent_df)
    count_plates = _plates(independent_df)
    matched = merged_df[well_column].nunique() if \
        well_column in merged_df.columns else 0

    # THE DENOMINATOR IS THE SMALLER SIDE, and getting that wrong made this
    # guard reject a correct run.
    #
    # The two sides are not expected to be the same size. Sequencing covers
    # every well on the plate; imaging keeps only the wells that survive
    # segmentation and the minimum-cell filter. On the TSG101 screen that is
    # 463 score wells against 1,344 count wells -- and all 463 found a
    # partner, which is a perfect join. Measured against the count side it
    # reads as 34%, and the guard refused to run a screen that was completely
    # paired.
    #
    # What actually matters is whether the wells that CAN be fitted found
    # their partner, so the denominator is the smaller side. An unusually
    # large unused remainder on either side is worth saying out loud, but it
    # is not an error: those wells simply contribute nothing.
    comparable = min(score_wells, count_wells)
    if comparable and matched / comparable >= _MINIMUM_PAIRED_WELL_FRACTION:
        unused_counts = count_wells - matched
        unused_scores = score_wells - matched
        # Persist the join counts so the run summary can report the data that
        # entered the analysis without relying on the transient console log.
        if record is not None:
            record["wells_paired"] = int(matched)
            record["wells_unpaired_counts"] = int(unused_counts)
            record["wells_unpaired_scores"] = int(unused_scores)
        if unused_counts or unused_scores:
            paired_label = "well" if matched == 1 else "wells"
            count_label = "well" if unused_counts == 1 else "wells"
            score_label = "well" if unused_scores == 1 else "wells"
            print(
                f"Paired {matched} {paired_label}. {unused_counts} "
                f"count-table {count_label} and {unused_scores} score-table "
                f"{score_label} had no matching identifier and were "
                f"excluded from the "
                f"regression.")
        return

    shared = sorted(set(score_plates) & set(count_plates))
    detail = (
        f"score wells:   {score_wells} on plates {score_plates}\n"
        f"  count wells:   {count_wells} on plates {count_plates}\n"
        f"  shared plates: {shared or 'NONE'}\n"
        f"  paired wells:  {matched}"
    )
    if matched == 0:
        raise ValueError(
            f"The score and count tables have no well in common, so the "
            f"regression has nothing to fit.\n\n"
            f"  {detail}\n\n"
            f"They are joined on prc = plateID_rowID_columnID, so the plate "
            f"ID is what pairs them -- the ORDER you listed the files in does "
            f"not matter, and the two lists need not be the same length. Make "
            f"the plate IDs agree: give every input a plateID column with "
            f"matching values, or state the pairing explicitly.")
    raise ValueError(
        f"Only {matched} of {comparable} pairable wells "
        f"({matched / comparable:.1%}) found a partner, which is below the "
        f"{_MINIMUM_PAIRED_WELL_FRACTION:.0%} required. The two inputs are "
        f"probably describing different plates or different well layouts.\n\n"
        f"  {detail}\n\n"
        f"Continuing would fit the model on whichever wells happened to "
        f"overlap and report it as the whole screen.")


def _identifiability_warning(data, settings, *, well_column='prc',
                             guide_column='grna', level='grna'):
    """Warn when a fit is about to be run on too few wells.

    Returns the warning text, or ``None`` when the design is fine. Kept
    separate from :func:`resolve_auto_inference` because this one never
    changes what runs -- it only makes sure the user cannot miss what they
    are about to get.

    Count terms at the requested fit level because guide- and gene-level
    designs can have different widths. Return a warning only when the number
    of estimated intercept, block, and identifier terms is at least the
    number of analyzed wells.

    :param level: ``'grna'`` (default) or ``'gene'`` -- which fit is about to
        run, and therefore which identifiers are the parameters.
    """
    identifier = 'gene' if str(level).strip().lower() == 'gene' else guide_column
    try:
        n_wells = int(data[well_column].nunique())
        n_terms = int(data[identifier].nunique())
    except (KeyError, TypeError):
        return None
    blocks = 0
    block_column = str(settings.get('guide_permutation_block', 'plateID'))
    if block_column in getattr(data, 'columns', ()):
        blocks = max(int(data[block_column].nunique()) - 1, 0)
    parameters = 1 + blocks + n_terms
    if n_wells > parameters:
        return None
    return (
        "\n"
        "  ###############################################################\n"
        "  #  WARNING: this fit is saturated or not identifiable.        #\n"
        "  ###############################################################\n"
        f"  {n_wells} analysed wells are being used to estimate "
        f"{parameters} parameters\n"
        f"  ({n_terms} {identifier}s + intercept + {blocks} block terms).\n"
        "\n"
        "  With at least as many parameters as wells, the model has no\n"
        "  residual degrees of freedom and may also be rank deficient.\n"
        "  Individual guide coefficients, standard errors and P values\n"
        "  cannot be interpreted reliably.\n"
        "\n"
        "  Set inference='nonparametric' to test each guide as a\n"
        "  marginal association, wells reshuffled within each plate,\n"
        "  coefficients simultaneously, or inference='auto' to let spaCR\n"
        "  choose. The design\n"
        "  diagnostics written beside the results show the rank, the\n"
        "  residual degrees of freedom and the collinear guide pairs.\n")


def _usable_nuisance_columns(data, settings) -> list:
    """The nuisance columns that are actually in the frame, said out loud.

    `guide_nuisance_columns` defaults to row and column, which every spaCR
    screen has and an imported table might not. `_nuisance_design` raises on
    an absent column -- correct for one the user typed, wrong for one that
    arrived as a default -- so the filtering happens here.

    SAID, NOT SILENT. A user who believes position was removed and reads a
    p-value computed without removing it has been told something false by
    omission, and the exchangeability the permutation rests on is exactly
    what those columns were there to protect.
    """
    wanted = [str(c) for c in (settings.get('guide_nuisance_columns') or [])]
    if not wanted:
        return []
    have = set(map(str, getattr(data, 'columns', ())))
    usable = [c for c in wanted if c in have]
    missing = [c for c in wanted if c not in have]
    # AND THEY MUST NOT MAKE THE DESIGN SINGULAR. `rowID` and `columnID` are
    # a DEFAULT now, and on a layout where the plates align with plate
    # position -- every plate its own block of columns, say -- the position
    # dummies are a linear combination of the block dummies and
    # `_nuisance_design` refuses the whole design.
    #
    # A DEFAULT MUST NOT BE ABLE TO KILL A RUN. Dropped one at a time, worst
    # last, so a screen where only one of the two is collinear keeps the
    # other.
    if usable:
        from .guide_permutation import _nuisance_design

        block = str(settings.get('guide_permutation_block', 'plateID'))
        while usable:
            try:
                _nuisance_design(data, block, usable)
                break
            except ValueError as exc:
                # ONLY RANK DEFICIENCY DROPS A COLUMN. `_nuisance_design`
                # raises the same exception type when the BLOCK column is
                # absent, and treating that as collinearity threw away a
                # perfectly good nuisance column -- caught by the test that
                # passes a frame with no plate column at all.
                if "rank deficient" not in str(exc):
                    break
                dropped = usable.pop()
                print(f"■ guide_nuisance_columns: {dropped!r} is collinear "
                      f"with {block!r} on this layout -- every level of one "
                      f"determines a level of the other -- so it cannot be "
                      f"removed separately. Dropped; {block!r} already "
                      f"absorbs it.")
            except Exception:                                # noqa: BLE001
                break
    if missing:
        print(f"■ guide_nuisance_columns named {len(missing)} column(s) this "
              f"table does not have: {', '.join(missing)}. They are not "
              f"removed before the permutation, so any structure they carry "
              f"stays in the residual the shuffle treats as noise.")
    return usable


def _report_exchangeability(data, outcome_column, settings, destination):
    """Measure and report whether the within-block shuffle is defensible.

    A COURTESY, NOT A PRECONDITION -- the same rule the montage pre-flight
    follows. It must never be the reason a run that produced results fails
    to report them, so every step is inside the guard.
    """
    try:
        from .guide_permutation import (_nuisance_design, _residualize,
                                        prepare_long_guide_data)
        from .permutation_qc import (block_residual_report,
                                     exchangeability_verdict)

        block = str(settings.get('guide_permutation_block', 'plateID'))
        nuisance = _usable_nuisance_columns(data, settings)
        wanted = list(dict.fromkeys([*nuisance, 'rowID', 'columnID']))
        present = [c for c in wanted
                   if c in getattr(data, 'columns', ())]
        _f, outcomes, _m = prepare_long_guide_data(
            data, outcome_column, block_column=block,
            nuisance_columns=present)
        y = pd.to_numeric(outcomes[outcome_column],
                          errors='coerce').to_numpy(dtype=float)
        basis, _r = np.linalg.qr(
            _nuisance_design(outcomes, block, nuisance), mode='reduced')
        residuals = _residualize(y, basis)

        # POSITION IS MEASURED EVEN WHEN IT WAS REMOVED, which is the point
        # of measuring it: a column already in `nuisance` should come back
        # explaining nothing, and if it does not, the removal did not work.
        positions = {c: outcomes[c] for c in present
                     if c in outcomes.columns and c != block}
        report = block_residual_report(
            residuals, outcomes[block], positions)
        verdict = exchangeability_verdict(report)

        if verdict['ok']:
            print(f"Exchangeability: nothing found. Durbin-Watson "
                  f"{report['durbin_watson']:.2f} over {report['n']:,} well(s) "
                  f"in {report['blocks']} block(s), and no position column "
                  f"explains the residual.")
            return report
        print("■ Exchangeability: the within-block shuffle is questionable.")
        for finding in verdict['findings'][:4]:
            print(f"    {finding}")
        if verdict['remedy']:
            print(f"    -> {verdict['remedy']}")
        return report
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not report exchangeability", exc_info=True)
        return None


def resolve_regression_src(requested, automatic):
    """Resolve the root directory used for regression output.

    A blank ``requested`` value selects ``automatic``. An existing requested
    directory is used directly. If only the final path component is missing,
    that directory is created; missing parent directories are never created.
    A requested file, an unavailable parent, or a directory-creation error
    returns the automatic location with an explanatory message.

    :param requested: Requested output directory, or ``None``/blank to use
        the automatic location.
    :param automatic: Existing fallback directory, normally the directory
        containing the first count table.
    :returns: A ``(path, message)`` tuple. ``message`` is ``'automatic'``
        when no override was requested; otherwise it describes the selected
        directory or the reason for falling back.
    """
    if not isinstance(requested, str) or not requested.strip():
        return automatic, 'automatic'

    # Resolve user-home and relative components before checking the parent.
    wanted = os.path.abspath(os.path.expanduser(requested.strip()))

    if os.path.isdir(wanted):
        return wanted, f"Regression output directory: {wanted}."
    if os.path.exists(wanted):
        return automatic, (
            f"The configured regression output path {wanted} is not a "
            f"directory. Results will be written to the automatic location "
            f"{automatic}.")

    parent = os.path.dirname(wanted)
    if os.path.isdir(parent):
        try:
            os.mkdir(wanted)
        except OSError as error:
            return automatic, (
                f"The regression output directory {wanted} could not be "
                f"created ({error.strerror or type(error).__name__}). "
                f"Results will be written to the automatic location "
                f"{automatic}.")
        return wanted, f"Created regression output directory: {wanted}."

    return automatic, (
        f"The regression output directory {wanted} was not created because "
        f"its parent directory {parent} does not exist. Results will be "
        f"written to the automatic location {automatic}.")


def _run_guide_permutation_analysis(data, outcome, destination, settings):
    """Run and persist the marginal guide analysis.

    This is the ``perform_regression`` branch used when
    ``analysis_mode='guide_permutation'``. Keeping it as a top-level function
    makes the correction and output contract testable without replaying score
    aggregation and sequencing QC.

    :returns: The long results, selected support family, significant rows, and
        a mapping of every artifact written by the analysis.
    """
    from .guide_permutation import (
        analyse_long_guide_table,
        plot_guide_permutation_volcano,
        save_guide_permutation_results,
    )

    thresholds = settings.get('guide_min_wells', [1, 2, 3, 4])
    if isinstance(thresholds, (int, np.integer)):
        thresholds = [int(thresholds)]
    thresholds = sorted({int(value) for value in thresholds})
    if not thresholds or any(value < 1 for value in thresholds):
        raise ValueError('guide_min_wells must contain positive integers')
    # A BLANK BOX MEANS "the first threshold", the same as an absent key.
    # `guide_primary_min_wells` is an optional field, so the panel leaves it
    # empty and the settings CSV writes an empty cell -- which read back as
    # '' and reached int(''), taking the whole nonparametric path down with
    # "invalid literal for int() with base 10: ''". The permutation test was
    # unreachable from the screen's own saved settings (236 C7).
    primary = settings.get('guide_primary_min_wells')
    primary = thresholds[0] if _left_blank(primary) else int(primary)
    if primary not in thresholds:
        raise ValueError(
            f'guide_primary_min_wells={primary} is not in '
            f'guide_min_wells={thresholds}')

    destination = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    os.makedirs(destination, exist_ok=True)
    # One or several responses. Naming more than one fits each independently
    # and corrects each as its OWN multiple-testing family -- pooling them
    # would make two correlated readouts of the same wells look like twice as
    # many tests. Concordance between independently trained classifiers is
    # evidence precisely because the families are separate.
    outcomes = [outcome] if isinstance(outcome, str) else list(outcome)
    missing = [column for column in outcomes if column not in data.columns]
    if missing:
        raise ValueError(
            f"dependent_variable names {missing} which are not columns of the "
            f"merged table. Available: {sorted(data.columns)[:20]}")
    # THE PERMUTATION TEST IS A TEST ABOUT WELLS, so it needs one row per
    # well. `analysis_unit='cell'` (agg_type=None) hands it one row per CELL,
    # and the phenotype then varies within a well -- which the permutation
    # code catches, but nine frames deep and phrased as a data-integrity
    # failure:
    #
    #     ValueError: Phenotype/block/nuisance values are not constant
    #     within well 'plate1_r1_c12'.
    #
    # Reported 2026-08-17 after a 20-second run that had already written its
    # regression data, three summary plots and their statistics. The
    # combination is not a corrupt table; it is two settings that cannot both
    # be honoured, and saying so costs nothing and is checkable HERE, before
    # any of that work.
    #
    # It refuses rather than aggregating silently: rolling cells up to wells
    # changes what was analysed, and a run that quietly analysed something
    # other than what was asked for is the failure this module is most
    # careful about elsewhere.
    # AND `agg_type is None` SAYS THE SAME THING. The check above reads
    # `analysis_unit`, which is what a user SETS; `agg_type` is what decides
    # whether the rows actually got rolled up, and some types force it to
    # None themselves -- quantile fits objects by construction. A run with
    # analysis_unit='well' and agg_type=None therefore reached the
    # permutation test with per-object rows and died on
    #
    #     ValueError: Phenotype/block/nuisance values are not constant
    #     within well 'plate1_r1_c4'
    #
    # which names a well and a pandas invariant rather than the two settings
    # that cannot both be honoured. The message this branch already carries
    # is the right one; it just was not reachable that way.
    per_object = str(settings.get('analysis_unit', 'well')).lower() != 'well'
    unaggregated = settings.get('agg_type') is None
    if per_object or unaggregated:
        why = (f"analysis_unit={settings.get('analysis_unit')!r}"
               if per_object else
               f"agg_type is None (regression_type="
               f"{settings.get('regression_type')!r} fits objects)")
        raise ValueError(
            f"analysis_mode='guide_permutation' tests each guide across "
            f"WELLS, so it needs one row per well -- but "
            f"{why} gives one row "
            f"per object, and a well's phenotype then has many values. Set "
            f"analysis_unit='well' (with an agg_type such as 'mean'), or "
            f"choose analysis_mode='regression', which can model objects.")

    results = analyse_long_guide_table(
        data,
        outcomes,
        min_wells=thresholds,
        block_column=str(settings.get('guide_permutation_block', 'plateID')),
        nuisance_columns=_usable_nuisance_columns(data, settings),
        n_permutations=int(settings.get('guide_permutations', 200000)),
        random_state=int(settings.get('guide_permutation_seed', 0)),
        multiple_testing=str(settings.get('multiple_testing_method', 'fdr_bh')),
        alpha=float(settings.get('fdr_alpha', 0.05)),
        presence_threshold=float(settings.get('guide_presence_threshold', 0.0)),
        batch_size=int(settings.get('guide_permutation_batch_size', 500)),
        statistic=str(settings.get('grna_statistic', 'pearson')),
    )
    # WHETHER THE SHUFFLE WAS ALLOWED (224). The test permutes phenotype
    # residuals within each block, which is valid only if those residuals are
    # exchangeable there -- and nothing said so until now. A parametric fit
    # writes a QC folder; this path returned before it, so the analysis that
    # residualises was the one that showed no residuals.
    _report_exchangeability(data, outcomes, settings, destination)
    # THE SAME ALIASES ON THE FULL TABLE, and on the primary slice taken from
    # it further down -- one block now, rather than two lists that had to stay
    # in step. results.csv on disk holds the primary slice while the returned
    # output['results'] holds every minimum-wells family, and when only one of
    # them was aliased everything that consumes a coefficient table (the
    # results panel, guide concordance, the volcano, the sweep's hit counts)
    # raised KeyError('feature') on the nonparametric path while working fine
    # on the parametric one.
    #
    # Built HERE, before anything is saved or drawn, because the effect-size
    # cut below is measured on `coefficient` and the volcano has to draw the
    # cut it produces.
    #
    # ADDED rather than swapped: a caller that wants the permutation
    # quantities themselves still has every one of them, and the names say
    # that the inferential quantities are marginal effects, empirical P
    # values and already-adjusted values.
    results = results.copy()
    results['grna'] = results['guide']
    results['feature'] = (
        'fraction:grna[' + results['guide'].astype(str) + ']')
    results['coefficient'] = results['standardized_marginal_effect']
    results['p_value'] = results['permutation_p_value']
    results['q_value'] = results['adjusted_p_value']

    # AN EFFECT-SIZE CUT IS NOT A PARAMETRIC IDEA, and saying it was is the
    # answer the maintainer was given: "why cant i see the coefficient
    # threshold if im running nonparametric regression?"
    #
    # A P value says an effect is distinguishable from zero. The effect-size
    # cut says it is big enough to be worth an experiment. That is a question
    # about the COEFFICIENT, and this table has a real one for every guide --
    # `standardized_marginal_effect`, aliased to `coefficient` two lines up,
    # 1,726 of them on the screen this was reported from. How the P value was
    # obtained does not change how wide a control's effect is.
    #
    # `condition` is what the cut is measured on, and this table did not carry
    # it. Measured before this was written, on a permutation-shaped frame:
    # `RegressionResultsPanel._threshold_sentence()` answered "No control
    # coefficients, so no effect-size cut." -- and the run itself returned
    # from `perform_regression` before the parametric branch that computes
    # one, so a permutation run drew no cut and reported no cut either.
    results['condition'] = label_control_condition(
        results['feature'], results['grna'],
        nc=settings.get('negative_control'),
        pc=settings.get('positive_control'),
        controls=settings.get('controls'))

    from .thresholds import coefficient_threshold

    # MEASURED ON THE PRIMARY FAMILY. The same guide appears once per
    # minimum-wells threshold with an identical coefficient, so pooling the
    # families would count each control up to four times and shrink the
    # spread the cut is built from.
    control_effects = results.loc[
        (results['minimum_wells_threshold'] == primary)
        & results['condition'].isin(('nc', 'control')), 'coefficient']
    effect_threshold, effect_rule = coefficient_threshold(
        control_effects,
        method=settings.get('threshold_method', 'std'),
        multiplier=settings.get('threshold_multiplier', 3.0),
        # The MEDIAN of the controls, computed inside, rather than the mean:
        # `000000_22` is a non-targeting control and the strongest effect in
        # the screen at +4.37, and a mean centre moves the cut for every
        # guide because of it.
        centre=None)
    print(f"Effect-size cut: {effect_rule}")

    # RECORDED PER ROW, not only printed. A cut a reader cannot recompute
    # from the results CSV is a cut they cannot report.
    results['effect_size_threshold'] = (
        np.nan if effect_threshold is None else float(effect_threshold))
    results['passes_effect_size'] = (
        True if effect_threshold is None
        else results['coefficient'].abs() >= float(effect_threshold))

    paths = dict(save_guide_permutation_results(
        results, destination, prefix='guide_permutation'))
    if settings.get('guide_permutation_plot', True):
        single = len(outcomes) == 1
        for response in outcomes:
            for threshold in thresholds:
                # A THRESHOLD NOTHING REACHES IS AN ANSWER, NOT A FAILURE.
                # `guide_min_wells` is a SWEEP -- [1, 2, 3, 4] asks the same
                # question four times at four strictnesses -- and on a
                # one-plate screen no guide appears in four wells. The
                # analysis is finished by the time this loop runs, so
                # raising here threw away the results for 1, 2 and 3 as
                # well, at the drawing stage, with a message about a plot.
                # Reported by driving the tsg101 screen (236 C7).
                have = results.loc[
                    (results['outcome'] == response)
                    & (results['minimum_wells_threshold'] == int(threshold))]
                if have.empty:
                    print(f"No guide reached {threshold} well(s) for "
                          f"{response!r}, so that panel of the "
                          f"guide_min_wells sweep is not drawn. The "
                          f"thresholds that did have guides are unaffected.")
                    continue
                for suffix in ('pdf', 'png'):
                    # One response keeps the historical filenames and keys, so
                    # scripts that look for guide_permutation_min_1_wells.pdf
                    # still find it.
                    stem = (f'guide_permutation_min_{threshold}_wells'
                            if single else
                            f'guide_permutation_{response}_min_'
                            f'{threshold}_wells')
                    key = (f'plot_min_{threshold}_{suffix}' if single else
                           f'plot_{response}_min_{threshold}_{suffix}')
                    paths[key] = plot_guide_permutation_volcano(
                        results,
                        outcome=response,
                        minimum_wells=threshold,
                        save_path=os.path.join(
                            destination, f'{stem}.{suffix}'),
                        # DRAWN, not only computed. The cut is the same number
                        # on the plot, in the CSV and in the log line above.
                        effect_threshold=effect_threshold,
                        effect_threshold_label=effect_rule,
                    )

    # Diagnostics are written for every run, not on request. The failure this
    # analysis mode exists to prevent -- a confident coefficient from a
    # rank-deficient design -- is invisible on the volcano and obvious on the
    # design panel, so the design panel has to be produced by default.
    try:
        from .guide_permutation import prepare_long_guide_data
        from .regression_diagnostics import write_diagnostic_suite

        fractions, well_outcomes, _metadata = prepare_long_guide_data(
            data, outcomes,
            block_column=str(settings.get('guide_permutation_block', 'plateID')),
            nuisance_columns=list(settings.get('guide_nuisance_columns') or []))
        for response in outcomes:
            family = results.loc[
                (results['outcome'] == response)
                & (results['minimum_wells_threshold'] == primary)]
            written = write_diagnostic_suite(
                os.path.join(destination, 'diagnostics'),
                fractions=fractions,
                block=well_outcomes[
                    str(settings.get('guide_permutation_block', 'plateID'))],
                p_values=family['permutation_p_value'].to_numpy(),
                adjusted=family['adjusted_p_value'].to_numpy(),
                alpha=float(settings.get('fdr_alpha', 0.05)),
                label=response if len(outcomes) > 1 else '',
                presence_threshold=float(
                    settings.get('guide_presence_threshold', 0.0)),
            )
            # Namespace the KEYS per response as well as the filenames. Both
            # responses write distinct files, but they returned the same keys,
            # so a two-classifier run reported only the second one's paths and
            # the first classifier's diagnostics looked as if they were never
            # produced.
            prefix = f'{response}_' if len(outcomes) > 1 else ''
            paths.update({f'{prefix}{key}': value
                          for key, value in written.items()})
    except Exception as error:  # noqa: BLE001 - diagnostics are advisory
        print(f"Regression diagnostics were skipped: "
              f"{type(error).__name__}: {error}")

    # ONE SET OF ALIASES, built once above and inherited by this slice.
    #
    # results.csv on disk holds primary_table while the returned
    # output['results'] holds the full multi-threshold frame, and the two used
    # to be aliased by two separate blocks of code -- one name, two shapes.
    # Everything that consumes a coefficient table (the results panel, guide
    # concordance, the volcano, the sweep's hit counts) raised
    # KeyError('feature') on the nonparametric path while working fine on the
    # parametric one. Slicing the aliased frame is what makes them the same
    # table by construction rather than by two lists staying in step.
    primary_table = results.loc[
        results['minimum_wells_threshold'] == primary
    ].copy()

    # A HIT CLEARS BOTH BARS, the way the parametric path's hit list does:
    # corrected P below alpha AND an effect at least as wide as the cut.
    # `passes_effect_size` is all-True when there is no cut to apply -- a
    # control-free screen, or a `threshold_method='none'` -- so a run without
    # controls calls exactly the hits it called before.
    #
    # Nothing is dropped silently: every guide keeps its row, its
    # `effect_size_threshold` and its `passes_effect_size` in results.csv, and
    # the line below says how many the cut removed.
    called = primary_table['significant'].astype(bool)
    wide_enough = primary_table['passes_effect_size'].astype(bool)
    significant = primary_table.loc[called & wide_enough].copy()
    if effect_threshold is not None:
        print(f"Effect-size cut removed {int((called & ~wide_enough).sum())} "
              f"of {int(called.sum())} guides that passed correction but "
              f"whose effect is narrower than {float(effect_threshold):.3g}.")

    # THE GENE PASS. Instruction 132 gives the parametric path two fits and two
    # tables; the permutation path answered only the guide question, so
    # choosing inference='nonparametric' silently lost the gene level
    # altogether -- results_gene.csv was never written by this branch at all.
    #
    # Each gene is tested as a SET: its regressor is the SUM of its guides'
    # fractions, which is the same `gene_fraction` the parametric gene fit
    # uses, residualized against the same block design and permuted with the
    # same Freedman--Lane scheme and the same seed. It is NOT a combination of
    # the guides' P values -- Fisher and Stouffer both assume independence, and
    # guides scored in the same wells share that well's phenotype, plate and
    # cells, so combining them would claim a confidence the design cannot
    # support.
    #
    # ITS OWN BH FAMILY, never pooled with the guides: same wells, and the gene
    # regressor is literally the sum of the guide regressors.
    # WHICH LEVELS THIS RUN REPORTS -- the same `level` key the fitted path
    # reads, so one control answers the question on both sides. It used to be
    # `guide_permutation_gene_level` alone, which is in no category and so had
    # no control at all: on the permutation side the level was unchoosable,
    # and `level` itself was greyed out because a mixed regression_type does
    # not read it. Between them a reader who asked for genes had no way to ask.
    #
    # `guide_permutation_gene_level` still WINS when it is set explicitly, so
    # a saved settings file that names it keeps meaning what it said.
    wanted_level = str(settings.get('level') or 'both').strip().lower()
    wants_gene = wanted_level in ('gene', 'both')
    if 'guide_permutation_gene_level' in settings:
        wants_gene = bool(settings.get('guide_permutation_gene_level'))
    gene_primary = None
    if wants_gene:
        try:
            from .guide_permutation import analyse_long_gene_table

            gene_results = analyse_long_gene_table(
                data, outcomes,
                min_wells=thresholds,
                block_column=str(settings.get('guide_permutation_block',
                                              'plateID')),
                nuisance_columns=list(settings.get('guide_nuisance_columns')
                                      or []),
                n_permutations=int(settings.get('guide_permutations', 200000)),
                random_state=int(settings.get('guide_permutation_seed', 0)),
                multiple_testing=str(settings.get('multiple_testing_method',
                                                  'fdr_bh')),
                alpha=float(settings.get('fdr_alpha', 0.05)),
                presence_threshold=float(
                    settings.get('guide_presence_threshold', 0.0)),
                batch_size=int(settings.get('guide_permutation_batch_size',
                                            500)),
            )
            gene_results['feature'] = (
                'gene_fraction:gene[' + gene_results['gene'].astype(str) + ']')
            gene_results['grna'] = None
            gene_results['coefficient'] = gene_results[
                'standardized_marginal_effect']
            gene_results['p_value'] = gene_results['permutation_p_value']
            gene_results['q_value'] = gene_results['adjusted_p_value']
            gene_results['condition'] = label_control_condition(
                gene_results['feature'], gene_results['gene'],
                nc=settings.get('negative_control'),
                pc=settings.get('positive_control'),
                controls=settings.get('controls'))
            gene_primary = gene_results.loc[
                gene_results['minimum_wells_threshold'] == primary].copy()
            print(f"Gene pass: {len(gene_primary)} genes tested as sets in "
                  f"the primary >={primary}-well family, corrected as their "
                  f"OWN BH family beside the {len(primary_table)} guides.")
        except Exception as error:  # noqa: BLE001 - the guide pass still stands
            print(f"The gene-level permutation pass could not run: "
                  f"{type(error).__name__}: {error}. results_gene.csv will be "
                  f"empty; the guide results are unaffected.")
            gene_primary = None

    compatibility = {
        'results': os.path.join(destination, 'results.csv'),
        'results_grna': os.path.join(destination, 'results_grna.csv'),
        'results_gene': os.path.join(destination, 'results_gene.csv'),
        'significant': os.path.join(destination, 'results_significant.csv'),
    }
    # `results.csv` CARRIES EVERY LEVEL THE RUN PRODUCED, which is the
    # convention the fitted path already follows: a level='both' regression
    # writes its guide and gene rows into one table and the results panel
    # filters them apart by the `level` column.
    #
    # This branch used to write the guide table alone, so a permutation run
    # that HAD tested genes -- and written them to results_gene.csv -- showed
    # a reader nothing when they asked for genes. The rows existed, in a file
    # the panel never opens, because it loads results.csv and stops.
    #
    # The two tables do not share a schema (a gene has `wells_with_gene` and
    # `guides_in_gene`; a guide has `wells_with_guide`), so the union carries
    # blanks where a column belongs to the other level. That is correct: the
    # question "how many wells hold this guide" has no answer for a gene.
    #
    # `level` decides which of them results.csv CARRIES. The guide pass runs
    # either way -- a gene's regressor is the sum of its guides' fractions, so
    # there is no gene answer without it -- and results_grna.csv always holds
    # those rows. What level='gene' means is that the reader asked for genes,
    # so genes are what the primary table reports.
    levelled = primary_table.copy()
    levelled['level'] = 'grna'
    gene_rows = None
    if gene_primary is not None and len(gene_primary):
        gene_rows = gene_primary.copy()
    if wanted_level == 'gene' and gene_rows is not None:
        combined = gene_rows
    elif gene_rows is not None and wanted_level != 'grna':
        combined = pd.concat([levelled, gene_rows], ignore_index=True,
                             sort=False)
    else:
        combined = levelled
    combined.to_csv(compatibility['results'], index=False)
    primary_table.to_csv(compatibility['results_grna'], index=False)
    # Written every run, empty when the pass could not be made, because a file
    # that is absent is indistinguishable from a run that crashed.
    (gene_primary if gene_primary is not None
     else primary_table.iloc[0:0]).to_csv(
        compatibility['results_gene'], index=False)
    significant.to_csv(compatibility['significant'], index=False)
    paths.update(compatibility)
    return {
        'analysis_mode': 'guide_permutation',
        # ONE ROW PER GUIDE, NOT ONE PER GUIDE PER FAMILY.
        #
        # `guide_min_wells` defaults to [1, 2, 3, 4], so this analysis runs
        # FOUR times at four inclusion thresholds -- four separate analyses of
        # the same guides. `results` is all four stacked: 1,612 rows for 789
        # guides on the real screen, with `225160_2` appearing four times at
        # the identical effect 0.25406.
        #
        # Handing that to the results panel drew every guide FOUR TIMES on one
        # volcano. Reported as "GRA14 and 225160 occur in the top right side
        # of the graph 4 times each which is obviously wrong", and it was --
        # twice I explained it away as a q-value tie artefact before checking
        # the row counts, which the maintainer had already told me: "my data
        # say 1612 gRNAs".
        #
        # The panel gets the PRIMARY family, which is exactly what
        # `results.csv` on disk already holds, so the file and the screen
        # finally agree. Every family stays reachable: `families` carries the
        # full frame and each one is still written to its own
        # `guide_permutation_min_<n>_wells.csv`.
        #
        # `combined`, NOT `primary_table`: the file gained the gene rows, and
        # a caller reading the dict must not get a different table from a
        # caller reading the file of the same name. `primary` below is the
        # guide rows alone for anyone who wants exactly those.
        'results': combined,
        'families': results,
        # The gene pass, corrected within itself. None when it was declined
        # with guide_permutation_gene_level=False or could not be made.
        'gene_results': gene_primary,
        'primary': primary_table,
        'significant': significant,
        'primary_min_wells': primary,
        # The cut, and the sentence that attributes it. A threshold a reader
        # cannot attribute is a threshold they cannot put in a methods
        # section, which is why the rule travels with the number.
        'effect_size_threshold': effect_threshold,
        'effect_size_rule': effect_rule,
        'paths': {key: str(path) for key, path in paths.items()},
    }


#: Settings a run chose for itself because the user left them unset. Filled
#: by :func:`perform_regression` as each is derived, and printed once both
#: are known -- the settings table is rendered before either exists.
_AUTOMATIC_SETTINGS: dict = {}


def _perform_regression_set_paths(settings):
    # _perform_regression_read_data has already normalised both keys to
    # lists by the time this runs, so the old scalar fallbacks here were
    # unreachable.
    csv_path = settings['count_data'][0]

    # A configured output root takes precedence. Blank values retain the
    # established behavior of writing beside the first count table.
    automatic = os.path.dirname(settings['count_data'][0])
    src, how = resolve_regression_src(settings.get('src'), automatic)
    settings['src'] = src
    # Report any explicit override, including a documented fallback.
    if how != 'automatic':
        print(how)

    # WHERE A RUN'S OUTPUT GOES: <count data folder>/results/<type>,
    # and never on top of an earlier run.
    #
    # Asked for on 2026-08-16: "just store everything in the same location
    # as the first count data ... then the type so for me
    # .../claude/results/ols. if there is already an ols folder then ols_1
    # then ols_2 and so on".
    #
    # The old path was <src>/results/<score_source>/<type>/list -- two
    # levels nobody asked for, one of them named after a CSV, and a fixed
    # leaf that meant a second run of the same type silently replaced the
    # first. That is also why the results panel could not find anything:
    # the path it had to guess at was four levels deep and named after a
    # file rather than the run.
    kind = results_folder_kind(settings)
    res_folder = _next_results_folder(os.path.join(src, 'results'), kind)
    _stage(settings, "placing the results folder")
    # WHERE A FAILURE REPORT GOES, recorded as soon as the folder exists.
    settings["_regression_folder"] = res_folder

    os.makedirs(res_folder, exist_ok=True)
    results_filename = 'results.csv'
    results_filename_gene = 'results_gene.csv'
    results_filename_grna = 'results_grna.csv'
    hits_filename = 'results_significant.csv'
    results_path=os.path.join(res_folder, results_filename)
    results_path_gene=os.path.join(res_folder, results_filename_gene)
    results_path_grna=os.path.join(res_folder, results_filename_grna)
    hits_path=os.path.join(res_folder, hits_filename)

    return results_path, results_path_gene, results_path_grna, hits_path, res_folder, csv_path


def results_folder_kind(settings) -> str:
    """What a run's results folder is NAMED after.

    The inference method when it decides the answer, and the regression type
    otherwise. Under `analysis_mode='guide_permutation'` the regression type
    is never read -- ols and mixed produce byte-identical results -- so a
    folder called `ridge` would name something the run did not do.

    PUBLIC, AND THE ONLY COPY. A test that re-derived this rule went stale
    when the rule changed and reported 39 missing CSVs while every run that
    wrote them was fine, which is the failure the `results_dir` helper in
    tests/test_cov_ml_perform_regression.py was already written to prevent
    once. A suite pointing at the wrong file is worse than a silent one.
    """
    settings = settings or {}
    if settings.get('analysis_mode') == 'guide_permutation':
        return 'guide_permutation'
    if settings.get('regression_type') is None:
        return 'auto'
    return str(settings['regression_type'])


def _next_results_folder(root, kind, limit=1000):
    """``<root>/<kind>``, or ``<kind>_1``, ``<kind>_2`` ... if taken.

    A run never writes on top of an earlier one. The old fixed path meant
    comparing two corrections, or re-running with one setting changed, left
    only the last on disk with nothing said about it -- and the results the
    user was looking at were not the results they thought.

    A folder counts as taken when it EXISTS AND HAS ANYTHING IN IT. An empty
    one is a directory somebody made and did not fill, and stepping past it
    would strand it forever.

    :param limit: stop after this many, rather than spinning if a filesystem
        keeps answering "yes, that exists too".
    """
    import os

    base = os.path.join(root, str(kind))
    for index in range(limit):
        candidate = base if index == 0 else f"{base}_{index}"
        try:
            if not os.path.isdir(candidate) or not os.listdir(candidate):
                return candidate
        except OSError:            # unreadable: treat as taken and move on
            continue
    return f"{base}_{limit}"


def _bracketed_identifier(pattern, text):
    """The id inside ``pattern``'s bracket, or ``None`` when there is none.

    The inline version of this was ``re.search(...).group(1) if 'grna' in x
    else None``, which assumes a term containing the word also contains the
    bracket. It does not: a mixed fit's variance component is named
    ``'grna Var'``, so the search returns None and ``.group(1)`` raises
    ``AttributeError`` on a run that had already fitted its model.
    """
    match = re.search(pattern, str(text))
    return match.group(1) if match else None


def _annotate_level_coefficients(coef_df, n_grna, n_gene):
    """Attach the guide / gene id and the per-id row counts to ONE fit's table.

    :param coef_df: one level's coefficient table, straight out of
        :func:`regression`.
    :param n_grna: value_counts frame, one row per guide.
    :param n_gene: value_counts frame, one row per gene.
    :returns: a new frame with ``grna``, ``gene``, ``n_grna`` and ``n_gene``.
    """
    coef_df = coef_df.copy()
    coef_df['grna'] = coef_df['feature'].map(
        lambda value: _bracketed_identifier(r'grna\[(.*?)\]', value))
    coef_df['gene'] = coef_df['feature'].map(
        lambda value: _bracketed_identifier(r'gene\[(.*?)\]', value))

    # n_grna / n_gene are value_counts frames, so one row per gRNA and one per
    # gene. coef_df is many rows against either of them — every gene[...] term
    # carries grna=None and vice versa — so many-to-one is the contract, and it
    # is the right side (the counts) that must stay unique: a duplicate there
    # would fan the coefficient table out and every hit would be written to
    # results_significant.csv more than once.
    #
    # CARRY `.attrs` ACROSS. `DataFrame.merge` does not propagate it --
    # `copy` and `concat` do, which is what makes the loss easy to miss --
    # and `regression` puts the QC manifest there for `_perform_regression`
    # to read back. Without this the manifest died between the two, so a
    # run's `output` carried no 'qc' key at all and instruction 115's
    # verdict never reached the caller.
    carried = dict(getattr(coef_df, "attrs", {}) or {})
    coef_df = coef_df.merge(n_grna, how='left', on='grna',
                            validate='many_to_one')
    coef_df = coef_df.merge(n_gene, how='left', on='gene',
                            validate='many_to_one')
    if carried:
        coef_df.attrs.update(carried)
    return coef_df


def _level_control_rows(frame, level, controls):
    """The control rows of ONE fit's table, matched at that fit's own level.

    ``settings['controls']`` names GUIDES. The guide fit matches them whole,
    exactly as it always has. The gene fit has no guide column at all -- every
    ``gene_fraction:gene[...]`` row carries ``grna=None`` -- so matching the
    same list there selects nothing and the gene table silently gets no
    effect-size cut. A control guide identifies its gene by spaCR's own rule
    (:func:`spacr.hits.gene_of`: truncate at the first underscore), so the
    gene fit matches on that.
    """
    if not (controls or []):
        return frame.iloc[0:0]
    # THE SAME MATCHER THE VOLCANO USES (184 C). This took
    # `name.split('_')[0]` as the gene, which reads `TGGT1` as the gene of
    # `TGGT1_000000_1` -- so a control pasted from a library file selected
    # nothing at gene level and the gene table silently got no effect-size
    # cut, which is the exact failure this function was written to fix, one
    # spelling further along. `spacr.control_names` measures the organism
    # prefix instead of assuming there is not one.
    from .control_names import matches, resolve_controls

    guides = frame['grna'] if 'grna' in frame.columns else frame.index
    library = [str(g) for g in pd.Series(guides).astype(str).unique()]
    genes = frame['gene'] if 'gene' in frame.columns else None
    specs = resolve_controls(controls, names=library)
    if not specs:
        return frame.iloc[0:0]
    keep = None
    for spec in specs:
        # AT THIS FIT'S OWN LEVEL. A gene fit has no guide column -- every
        # `gene_fraction:gene[...]` row carries grna=None -- so a guide-level
        # control is matched by the gene it belongs to there.
        if level == 'gene':
            from .control_names import GENE, ControlSpec

            spec = ControlSpec(spec.typed, GENE,
                               spec.value.split('_')[0] if not spec.is_gene
                               else spec.value, spec.prefix)
            mask = matches(spec, frame['gene'].astype(str),
                           frame['gene'].astype(str))
        else:
            mask = matches(spec, pd.Series(guides).astype(str), genes)
        keep = mask if keep is None else (keep | mask)
    return frame.loc[keep.to_numpy()]


#: Current and legacy keys for enabling bundled *Toxoplasma* annotation.
#:
#: Both spellings remain accepted so existing settings files continue to
#: enable annotation instead of silently ignoring the legacy key.
TOXOPLASMA_KEYS = ('Toxoplasma', 'toxo')


#: What `annotation_source` calls the bundled, offline Toxoplasma path.
BUNDLED_ANNOTATION = "toxoplasma"


def _toxoplasma_is_on(settings) -> bool:
    """Whether the bundled *Toxoplasma* annotation was asked for.

    The NEW key wins when both are present, because a user who set the new
    one meant it; an old CSV that carries only `toxo` still works.
    """
    for key in TOXOPLASMA_KEYS:
        if key in settings:
            return bool(settings[key])
    return False


def _annotation_source(settings) -> str:
    """Which organism's annotation this run asked for, or "" for none.

    `Toxoplasma=True` and `annotation_source='toxoplasma'` are the same
    request and the field wins, because a user who typed a name meant it.
    `Toxoplasma=False` with no field is the one case that means NO
    annotation, and it has to keep meaning that.
    """
    named = str(settings.get('annotation_source', '') or '').strip()
    if named:
        return named
    return BUNDLED_ANNOTATION if _toxoplasma_is_on(settings) else ""


def _annotation_cache(settings):
    """Where a UniProt answer is kept, so a rerun needs no network."""
    src = settings.get('src')
    if isinstance(src, (list, tuple)):
        src = src[0] if src else None
    if not src:
        return None
    return os.path.join(str(src), 'annotation_cache')


def _call_level_hits(coef_df, level, settings, regression_type,
                     merged_df, dependent_variable, bootstrap=None):
    """Correct one fit within itself and call that fit's hits.

    Treat guide and gene fits as separate multiple-testing families. They use
    the same wells and gene regressors are sums of guide regressors, so pooling
    both levels would count correlated hypotheses as independent tests.

    :param coef_df: one fit's annotated coefficient table.
    :param level: ``'grna'`` or ``'gene'`` -- which fit this is.
    :param regression_type: the backend that produced it.
    :param merged_df: the pre-clean long frame, for the lasso bootstrap.
    :param bootstrap: ``perform_regression``'s ``bootstrap_selection_frequencies``
        closure. It is defined inside that function, so it cannot be looked up
        from here and the penalised backends need it passed in.
    :returns: ``(coef_df, significant, reg_threshold, effect_rule)``.
    """
    from .thresholds import coefficient_threshold

    # OWNED, not borrowed. Every caller so far handed in a slice of a bigger
    # frame -- the mixed path slices the BLUP rows off -- and assigning
    # `q_value` onto a slice raises SettingWithCopyWarning, which this suite
    # promotes to an error and pandas 3 will make a hard failure.
    coef_df = coef_df.copy()

    # reg_threshold used to be bound only inside the branch below, so a
    # control-free screen (settings['controls'] is None) hit UnboundLocalError
    # as soon as the toxo volcano block read it. 0 is custom_volcano_plot's own
    # default and means "no coefficient cut-off, select on p <= 0.05 alone",
    # which is the only sensible threshold when there are no controls to
    # calibrate against.
    reg_threshold = 0
    effect_rule = 'no effect-size cut'

    if settings['controls'] is not None:
        control_coef_df = _level_control_rows(
            coef_df, level, settings['controls'])

        # SEVEN METHODS, in one place. It was two -- std and var -- and the
        # maintainer asked for "at least 4 more" reachable from the plot, so
        # the arithmetic moved to `spacr.thresholds` where the GUI can reach
        # the same list rather than keeping a second copy of it.
        measured_threshold, threshold_rule = coefficient_threshold(
            control_coef_df['coefficient'],
            method=settings['threshold_method'],
            multiplier=settings['threshold_multiplier'],
            # The MEDIAN of the controls, computed inside, rather than the
            # mean this used to add: `000000_22` is a non-targeting control
            # and the strongest effect in this whole screen at +4.37, and a
            # mean centre moves the cut for every guide because of it.
            centre=None)
        effect_rule = threshold_rule
        print(f"Effect-size cut ({level}): {threshold_rule}")

        # `coefficient_threshold` answers None when NO CUT CAN BE MADE --
        # `threshold_method='none'`, fewer than two control coefficients, or a
        # set of controls with no spread at all. It is deliberately not a
        # silent 0, so that the caller has to decide what to do about it, and
        # every one of this function's three readers wanted a number:
        #
        #   * the hit list below compared a Series against None. pandas 2.x
        #     evaluates `series >= None` as all-False rather than raising, so
        #     BOTH masks were empty and the run wrote an EMPTY
        #     results_significant.csv -- measured on the synthetic screen,
        #     16 of 16 corrected hits lost, with nothing said.
        #   * `custom_volcano_plot` does `abs(threshold)` and died with
        #     `TypeError: bad operand type for abs(): 'NoneType'` after the
        #     whole fit, every results CSV and every QC panel had been
        #     written.
        #   * `plot.volcano_plot` takes it as `fold_change_threshold`.
        #
        # 0 is what "no coefficient cut" already means to all three -- the
        # value a control-free screen has carried for as long as this line has
        # existed -- and `threshold_rule`, printed above, is what says WHY
        # there is none. The reason is on the record; only the sentinel is
        # normalised.
        reg_threshold = (0 if measured_threshold is None
                         else float(measured_threshold))
    else:
        # SAID, not left silent. A run WITH controls prints its cut and the
        # rule behind it; a run without them printed nothing, so "there is no
        # effect-size cut" looked exactly like "that line scrolled past". It
        # is the more surprising of the two, because a hit list called on the
        # corrected P value alone is a different claim from one that also had
        # to clear a width.
        print(f"Effect-size cut ({level}): no control gRNAs were named, so "
              f"there is none; a hit is the corrected P value alone.")

    if regression_type in NO_P_VALUE_TYPES and bootstrap is None:
        raise ValueError(
            f"regression_type={regression_type!r} ranks features by bootstrap "
            f"selection frequency and has no p-value to correct, so "
            f"_call_level_hits needs perform_regression's "
            f"bootstrap_selection_frequencies passed as `bootstrap`.")

    if regression_type in NO_P_VALUE_TYPES:
        # Lasso and elastic net have no valid frequentist p-values (the ones
        # process_model_coefficients attaches are OLS-style and ignore the
        # penalty). Use bootstrap selection frequency as the feature-importance
        # ranking. Treat as a selection method, not a hypothesis test.
        n_boot = settings.get('lasso_n_boot', 200)
        sel_threshold = settings.get('lasso_selection_threshold', 0.6)
        formula = prepare_formula(
            dependent_variable, random_row_column_effects=False,
            block_screen=screen_is_blockable(merged_df), level=level,
            # The bootstrap must resample the design the fit used, so the
            # selection frequencies are frequencies for THAT model. Reading
            # the setting rather than passing True was worth a comment: with
            # plate position out of the fit and in the bootstrap, a guide
            # would be selected against a different set of competitors.
            model_plate_position=settings.get('model_plate_position', True))
        # Apply the same preprocessing the OLS path uses, so derived columns
        # referenced by the formula (e.g. gene_fraction) exist in the bootstrap.
        cleaned_df = check_and_clean_data(merged_df.copy(), dependent_variable)
        sel_df = bootstrap(
            X=cleaned_df,
            y=cleaned_df[dependent_variable],
            formula=formula,
            alpha=settings.get('alpha', 'auto'),
            n_boot=n_boot,
            random_state=0,
            regression_type=regression_type,
            l1_ratio=settings['l1_ratio'],
            group_lasso_lambda=settings.get('group_lasso_lambda', 'auto'),
        )
        # One row per model term on both sides: coef_df['feature'] is the
        # design-matrix column index (X.columns for lasso), sel_df['feature']
        # is the same index taken off the reference design built once at the
        # top of bootstrap_selection_frequencies. Both are pandas Index
        # objects from patsy, so the join is one-to-one by construction, and a
        # duplicate on either side means the two designs have gone out of step
        # — which is exactly when a selection frequency must not be silently
        # attached to the wrong coefficient.
        coef_df = coef_df.merge(sel_df, on='feature', how='left',
                                validate='one_to_one')

        significant = coef_df[
            (coef_df['coefficient'] != 0)
            & (coef_df['selection_frequency'] >= sel_threshold)
        ].copy()
        significant = significant.sort_values(
            by='coefficient', key=lambda c: c.abs(), ascending=False,
        )
        significant = significant[~significant['feature'].str.contains('row|column')]
        return coef_df, significant, reg_threshold, effect_rule

    # THE CORRECTION IS APPLIED HERE, and until instruction 128 it never was.
    #
    # `multiple_testing_method` has existed as a setting, been offered in
    # the panel and been named in Methods sections, while this branch
    # called a hit on the RAW OLS p-value. With 1,208 coefficients an
    # uncorrected 0.05 expects about sixty false positives from noise
    # alone, and that is the defect behind a published volcano whose
    # figure showed a P = 0.05 line while its Methods claimed BH q < 0.05.
    #
    # The family is the guide/gene coefficients OF THIS FIT -- not the
    # intercept and not the row/column nuisance terms, which are covariates
    # rather than hypotheses and would only inflate the family, and not the
    # other fit's coefficients either.
    #
    # 'none' reproduces the historical rule exactly, so a run that wants
    # the old behaviour can still ask for it and is on record as having
    # asked.
    from .multiple_testing import adjust_p_values, canonical_method

    method = canonical_method(settings.get('multiple_testing_method',
                                           'fdr_bh'))
    alpha = float(settings.get('fdr_alpha', 0.05))
    # WHERE THE LINE IS DRAWN, AND ON WHICH P. Instruction 135, asked for on
    # 2026-08-17: "add a setting that setts what alpha the p threshold is set
    # at and if adjusted p or raw p is used".
    #
    # Until these two, the CORRECTION's alpha was also the hit cut, and the
    # cut was always on the adjusted P -- while the volcano's own right-click
    # menu could switch the axis to the raw P. So the exported hit list and
    # the picture printed beside it could mean two different things by
    # "significant", with nothing saying which.
    #
    # They are separate from `fdr_alpha` on purpose. `fdr_alpha` is the level
    # the CORRECTION targets, an input to the procedure; this is the level a
    # coefficient is CALLED at. Same number by default, and a reader is
    # entitled to move one without the other -- correcting at 0.05 and
    # reporting at 0.01 is an ordinary thing to want.
    cut_alpha = float(settings.get('p_threshold_alpha', alpha) or alpha)
    cut_kind = str(settings.get('p_threshold_kind', 'adjusted')).strip().lower()
    cut_column = 'p_value' if cut_kind == 'raw' else 'q_value'
    # ONE STATEMENT OF WHAT IS BEING TESTED, shared with the volcano.
    # A plot drawn from a different family than the one corrected here is
    # a plot of a different experiment; see spacr.hits.tested_family.
    from .hits import tested_family

    tested = pd.Series(tested_family(coef_df['feature']),
                       index=coef_df.index)
    # A ROW WITHOUT A P VALUE IS NOT A TEST. The mixed fit returns variance
    # components and guide BLUPs alongside its fixed effects, and both carry
    # a NaN p-value BY CONSTRUCTION -- a BLUP is a shrunken prediction of a
    # random effect, not an estimate of a parameter that could be zero. Left
    # in the family they would enlarge it (weakening every real q value) and
    # come back with a q value of their own, which is a p-value manufactured
    # for a quantity that has none.
    tested &= coef_df['p_value'].notna()
    # A VARIANCE IS NOT A HYPOTHESIS ABOUT A GENE. The mixed fit's 'grna Var'
    # row carries a real Wald p-value (0.109 on the synthetic nesting), so the
    # NaN guard above does not catch it -- and left in the family it both
    # enlarges the correction and comes back with a q value, which would put a
    # 'hit' on the volcano that no gene owns. Only fixed effects are tests.
    if 'term_type' in coef_df.columns:
        tested &= coef_df['term_type'].eq(TERM_FIXED)
    coef_df['q_value'] = np.nan
    coef_df['multiple_testing_method'] = method
    if tested.any():
        adjusted, _rejected = adjust_p_values(
            coef_df.loc[tested, 'p_value'].to_numpy(dtype=float),
            method=method, alpha=alpha)
        coef_df.loc[tested, 'q_value'] = adjusted
    raw_hits = int((coef_df.loc[tested, 'p_value'] <= alpha).sum())
    corrected_hits = int((coef_df.loc[tested, 'q_value'] < alpha).sum())
    print(f"Multiple testing ({level}): {method} across {int(tested.sum())} "
          f"tested coefficients at alpha={alpha:g} — {raw_hits} pass the raw "
          f"P value, {corrected_hits} pass correction.")
    if cut_kind == 'raw' or cut_alpha != alpha:
        # SAID OUT LOUD, because it is the one line that decides what the
        # exported table means. A cut on the raw P over hundreds of guides is
        # a defensible choice and an indefensible accident, and the only way
        # to tell them apart is whether the run announced it.
        print(f"  Calling hits on the {cut_kind} P at {cut_alpha:g}"
              + (", NOT corrected for multiple testing."
                 if cut_kind == 'raw' else "."))

    significant = coef_df.loc[coef_df[cut_column] < cut_alpha].copy()
    # THE EFFECT-SIZE CUT IS A WIDTH, SO IT IS SYMMETRIC. Until instruction
    # 128 this was a pair of one-sided masks whose UNION was every row:
    #
    #     high = coefficient >= reg_threshold
    #     low  = coefficient <= reg_threshold
    #
    # `reg_threshold` is `|median| + k x spread`, so it is never negative
    # and every coefficient satisfies one side or the other. Measured on
    # the synthetic screen with `threshold_method='std'`: the cut was
    # 0.57, and all 16 corrected hits survived it -- the narrowest at
    # |coefficient| = 0.0026, more than two hundred times inside the cut.
    # `custom_volcano_plot`, handed the SAME number, marks hits with
    # `abs(coefficient) >= abs(threshold)` and would have called none of
    # them, so the figure and results_significant.csv described different
    # experiments and only the figure was right.
    #
    # A cut that admits +0.9 but not -0.9 would also call half a screen:
    # a guide that moves the phenotype DOWN by more than the controls ever
    # move is exactly as much a hit as one that moves it up, and which
    # direction is 'good' is the biology's business, not the filter's.
    #
    # This is the rule `_run_guide_permutation_analysis` already applies
    # (`passes_effect_size`), so the parametric and nonparametric paths
    # now call a hit the same way.
    #
    # NOTHING IS DROPPED SILENTLY: every coefficient keeps its row in
    # results.csv with its q value, and the line below says how many the
    # cut removed and how wide it was.
    coef_df['effect_size_threshold'] = (
        np.nan if not reg_threshold else abs(float(reg_threshold)))
    coef_df['effect_size_rule'] = effect_rule
    significant = significant.assign(
        effect_size_threshold=(np.nan if not reg_threshold
                               else abs(float(reg_threshold))),
        effect_size_rule=effect_rule)
    if reg_threshold:
        wide_enough = (significant['coefficient'].abs()
                       >= abs(reg_threshold))
        called = len(significant)
        significant = significant.loc[wide_enough].copy()
        print(f"Effect-size cut ({level}) removed {called - len(significant)} "
              f"of {called} coefficients that passed correction but whose "
              f"effect is narrower than {abs(reg_threshold):.3g}.")
    significant = significant.sort_values(
        by='coefficient', ascending=False)
    significant = significant[~significant['feature'].str.contains('row|column')]
    return coef_df, significant, reg_threshold, effect_rule


def _stage(settings, name):
    """Record the current fit stage, announce it, and never raise.

    Fall back to storing ``_regression_stage`` in the settings mapping when
    resource measurement is unavailable.

    THE ANNOUNCEMENT IS THE POINT AS MUCH AS THE RECORD. Reading the counts
    and fitting the model each take minutes on a four-plate screen, and a step
    that prints nothing while it runs cannot be told apart from a step that
    has hung -- which is how a working run comes to be reported as a dead one.
    The recorded resident size goes on the same line, because the other thing
    a long silent step invites is a guess about memory.
    """
    reading = {}
    try:
        from .fit_resources import record_stage

        reading = record_stage(settings, name)
    except Exception:                                            # noqa: BLE001
        try:
            settings["_regression_stage"] = str(name)
        except Exception:                                        # noqa: BLE001
            pass
    try:
        rss = reading.get("rss") if isinstance(reading, dict) else None
        note = f" (resident {rss / 1e9:.1f} GB)" if rss else ""
        print(f"Regression: {name}{note}.", flush=True)
    except Exception:                                            # noqa: BLE001
        pass
    return reading


#: Panels that need a fitted object exposing residuals, and the models that
#: cannot supply one. RRA is a rank statistic -- it never fits a linear
#: predictor, so "residual" has no meaning for it rather than being
#: unavailable. Naming them here rather than catching AttributeError keeps the
#: REASON, which is the whole point of instruction 322: a missing QQ plot and
#: an inapplicable one look identical to a reader, and only one is fine.
RESIDUAL_FREE_MODELS: dict = {
    "rra": ("Robust Rank Aggregation is a rank statistic: it ranks guides "
            "within each well and aggregates those ranks, so it never forms a "
            "linear predictor and there is no residual to plot."),
    "horseshoe": ("The horseshoe fit is sampled rather than solved, so it has "
                  "a posterior rather than one set of fitted values."),
}


def _diagnostic_inputs(model):
    """``(observed, fitted, design)`` from a fitted model, or ``(None,)*3``.

    Duck-typed on purpose. statsmodels results expose ``fittedvalues``,
    ``resid`` and ``model.exog``; the backends spaCR wraps do not share a base
    class, so asking what an object HAS is the only question that works across
    all of them.
    """
    fitted = getattr(model, "fittedvalues", None)
    resid = getattr(model, "resid", None)
    if fitted is None or resid is None:
        return None, None, None
    try:
        observed = np.asarray(fitted, dtype=float) + np.asarray(resid,
                                                                dtype=float)
    except Exception:                                        # noqa: BLE001
        return None, None, None
    design = getattr(getattr(model, "model", None), "exog", None)
    return observed, np.asarray(fitted, dtype=float), design


def _write_regression_diagnostics(res_folder, fractions, fits, settings):
    """Write the diagnostic suite for a completed fit.

    THE DESIGN REPORT IS UNCONDITIONAL. It needs no fit at all -- only the
    well-by-guide matrix -- so it is available for every model including RRA,
    and it is the one that would have caught the failure
    :mod:`spacr.regression_diagnostics` was written for: 824 guides in 587
    wells returning a confident P value for every guide out of a rank-deficient
    matrix.

    RESIDUAL PANELS SAY WHY WHEN THEY CANNOT BE DRAWN. `write_diagnostic_suite`
    skips a block whose inputs are absent, silently, which is right for a
    library and wrong here: the user asked for these plots "whenever possible",
    and the interesting case is precisely when it is not possible. So a model
    that cannot support residuals writes a note naming the reason beside the
    panels that did run.

    Never raises. A diagnostic that took the analysis down with it would be
    worse than no diagnostic -- the numbers the user came for are already
    computed by the time this runs.
    """
    from . import regression_diagnostics as rd

    if not res_folder:
        return {}
    destination = os.path.join(res_folder, "diagnostics")
    written: dict = {}
    try:
        model, _coef, model_type = next(iter(fits.values()))
    except Exception:                                        # noqa: BLE001
        model, model_type = None, str(settings.get("regression_type") or "")

    observed, fitted, design = _diagnostic_inputs(model)
    reason = RESIDUAL_FREE_MODELS.get(str(model_type).lower())
    if observed is None and reason is None:
        reason = (f"The {model_type or 'selected'} backend did not expose "
                  "fitted values and residuals, so the residual panels could "
                  "not be computed for this run.")

    try:
        written = dict(rd.write_diagnostic_suite(
            destination, fractions=fractions,
            observed=observed, fitted=fitted, design=design,
            label=str(model_type or "")))
    except Exception as error:                               # noqa: BLE001
        print(f"Diagnostics could not be written: "
              f"{type(error).__name__}: {error}")
        return {}

    if observed is None and reason:
        # THE NOTE IS A FILE, not a print. A run is read from its folder
        # afterwards, usually by someone who did not watch it run, and a
        # console line is gone by then -- which is exactly how an inapplicable
        # panel becomes indistinguishable from a missing one.
        note_path = os.path.join(destination, "residual_panels_not_available.txt")
        try:
            with open(note_path, "w", encoding="utf-8") as handle:
                handle.write(reason + "\n")
            written["residuals_unavailable"] = note_path
        except OSError:
            pass
        print(f"Residual diagnostics were not computed: {reason}")
    return written


def perform_regression(settings):
    """Run the regression and report actionable details if it fails.

    On failure, the original exception is re-raised unchanged after a report
    is printed and written to the run folder. The report includes the most
    recent stage stored in ``settings['_regression_stage']``, available design
    dimensions, and a remedy for recognized failures.

    :param settings: Regression settings consumed by the fitting pipeline.
    :returns: Result returned by the regression implementation.
    :raises Exception: Re-raises the original regression failure.
    """
    from .regression_failure import describe_failure, write_failure_report

    # DUCK-TYPED, NOT `isinstance(settings, dict)`. The contract test in
    # tests/test_regression_entry_points.py scans every call this function
    # hands `settings` to, so that the keys each one reads can be checked for a
    # default -- which is how six missing defaults were once found. A bare
    # `isinstance(settings, ...)` registers as such a call and the scan then
    # asks which spacr module `isinstance` lives in. Asking forgiveness keeps
    # the settings dict out of a call the scan has to reason about.
    try:
        settings.setdefault("_regression_stage", "starting")
    except AttributeError:                       # not a mapping; nothing to do
        pass
    try:
        outcome = _perform_regression(settings)
    except Exception as error:                                   # noqa: BLE001
        stage = ""
        folder = ""
        frame = None
        try:
            stage = str(settings.get("_regression_stage", "") or "")
            folder = str(settings.get("_regression_folder", "") or "")
            frame = settings.get("_regression_frame")
        except AttributeError:
            pass
        print(describe_failure(error, stage=stage, settings=settings,
                               frame=frame, include_traceback=False))
        written = write_failure_report(folder, error, stage=stage,
                                       settings=settings, frame=frame)
        if written:
            print(f"The full report, with the traceback, is in {written}")
        # RE-RAISED UNCHANGED. The reporter adds to a failure; it must never
        # replace one, or a caller that handles a specific exception type
        # stops seeing it.
        raise
    _write_fit_resources(outcome, settings)
    return outcome


#: What a completed run's resource record is called, beside its results.
FIT_RESOURCES_FILENAME = "fit_resources.txt"


def _write_fit_resources(outcome, settings):
    """Write per-stage and peak resource use for a successful fit.

    Store the report beside the run results when a destination and
    measurements are available. Return an empty string on missing data or any
    measurement/write failure so resource reporting cannot fail the fit.
    """
    try:
        from .fit_resources import describe_resources, peak

        folder = ""
        if isinstance(outcome, dict):
            folder = str(outcome.get("res_folder") or "")
        if not folder:
            folder = str(settings.get("_regression_folder", "") or "")
        table = describe_resources(settings)
        if not folder or not table or not os.path.isdir(folder):
            return ""
        high = peak(settings)
        lines = ["WHAT THIS FIT COST", "==================", "",
                 "Recorded per stage as the run went. 'not measured' is not "
                 "zero:", "psutil absent, or no CUDA tensor allocated yet.",
                 "", table, ""]
        if not high:
            lines.append("No reading could be taken on this machine.")
        path = os.path.join(folder, FIT_RESOURCES_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return path
    except Exception:                                            # noqa: BLE001
        return ""


def _warn_if_penalised_no_hits(settings, coef_df):
    """Explain why a penalised fit with no small P values is inconclusive."""
    penalised = str(settings.get('regression_type', '')).lower() in (
        'ridge', 'lasso', 'elasticnet')
    if penalised and len(coef_df):
        p_values = pd.to_numeric(coef_df.get('p_value'), errors='coerce')
        if not (p_values < 0.05).any():
            print(
                f"\nNOTE: {settings['regression_type']} returned no "
                f"coefficient below p=0.05. Its p-values are conservative "
                f"by construction -- the standard error is unpenalised "
                f"while the coefficient it is divided into has been shrunk "
                f"-- so this is NOT evidence of no effect. Refit with "
                f"regression_type='ols' (or 'rlm' for a robust check) "
                f"before concluding anything from it.")
            return True
    return False


def _perform_regression(settings):
    """Regress per-well phenotype scores against gRNA / gene counts to identify hits from a pooled CRISPR screen.

    Reads one or more score CSVs (from :func:`generate_ml_scores` or a
    deep-learning classifier) and one or more sgRNA count CSVs (from
    :func:`spacr.sequencing.generate_barecode_mapping`), aligns them on
    plate / well, fits the requested regression model, merges metadata,
    and emits volcano plots, plate heatmaps, gene phenotype plots and
    GO enrichment reports.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.get_perform_regression_default_settings`.
        Key entries:

        - ``paired_data`` — ordered rows that explicitly pair one score CSV
          with one sgRNA-count CSV. Plate identity comes from the files when
          they agree, from the partner when only one declares it, or from the
          pair-row order when neither does. Legacy ``score_data`` and
          ``count_data`` lists are migrated positionally with a visible log.
        - ``dependent_variable`` — column of ``score_data`` to regress
          (e.g. ``'pred'``, ``'recruitment'``,
          ``'pathogen_nucleus_shortest_distance'``).
        - ``regression_type`` — any name in :data:`REGRESSION_TYPES`, or
          ``None`` to choose one from the response distribution. See
          :func:`regression_model` for what each backend is for.
        - ``analysis_mode='guide_permutation'`` — instead test plate-adjusted
          marginal guide associations with empirical P values and apply
          ``multiple_testing_method`` (Benjamini--Hochberg by default) within
          each requested ``guide_min_wells`` family.
        - the per-model settings each backend reads — ``alpha``,
          ``l1_ratio``, ``cov_type``, ``quantile``, ``hinge_threshold``,
          ``hinge_n_boot``, ``huber_t``, ``random_row_column_effects``.
          A setting the chosen type cannot read is refused rather than
          ignored; see :data:`REGRESSION_SETTINGS_USED`.
        - ``batch_correction`` — optional ``combat``, ``center``, ``zscore``,
          ``robust_zscore`` or reference-control ``control_center``
          normalization of the dependent variable before well aggregation.
        - ``fraction_threshold``, ``min_n``, ``metadata_files``,
          ``volcano``, ``heatmap_feature``.

    :returns: Path to the merged, metadata-annotated results DataFrame
        (also written to ``results/<score_source>/<regression_type>/
        results.csv``). Related gene/gRNA CSVs and significance calls
        are saved alongside.
    :raises ValueError: if paired files declare incompatible plate IDs,
        ``dependent_variable`` is not a score column, ``regression_type`` is
        unsupported, or a guide-permutation support family or correction
        setting is invalid.

    Example:
        .. code-block:: python

            from spacr.ml import perform_regression
            settings = {
                'paired_data': [{
                    'score': '/data/plate01/results/xgb_scores.csv',
                    'count': '/data/plate01/sequencing/counts.csv',
                }],
                'dependent_variable': 'pred',
                'regression_type': 'mixed',
            }
            perform_regression(settings)

    See Also:
        :func:`generate_ml_scores` — produce the ``score_data`` input.
        :func:`spacr.sequencing.generate_barecode_mapping` — produce the
        ``count_data`` input.
    """
    from .plot import plot_plates, plot_data_from_csv
    from .utils import merge_regression_res_with_metadata, save_settings, correct_metadata
    from .settings import get_perform_regression_default_settings
    from .toxo import custom_volcano_plot, plot_gene_phenotypes, plot_gene_heatmaps

    def _perform_regression_read_data(settings):
            _stage(settings, "reading the input tables")
            pairs, _migrated = normalize_regression_input_pairs(settings)
            count_data_df, score_data_df, audit = \
                load_regression_input_pairs(pairs)
            settings['paired_data'] = pairs
            settings['input_pair_audit'] = audit

            print(f"Score data: {len(score_data_df)} rows from "
                  f"{len(settings['score_data'])} file(s)")
            print(f"Count data: {len(count_data_df)} rows from "
                  f"{len(settings['count_data'])} file(s)")

            print(f"Dependent variable: {len(score_data_df)}")
            print(f"Independent variable: {len(count_data_df)}")

            if settings['dependent_variable'] not in score_data_df.columns:
                if not settings['dependent_variable'] == 'pathogen_nucleus_shortest_distance':
                    # Name the likeliest cause, not only the symptom. A count
                    # table has grna/count columns and no score column, so a
                    # score slot holding one is a swapped input -- the
                    # commonest way to reach this error, and invisible in a
                    # bare "not found in the DataFrame" followed by a column
                    # dump the user has to interpret themselves.
                    looks_like_counts = sorted(
                        {'grna', 'grna_name', 'count'}.intersection(
                            score_data_df.columns))
                    if looks_like_counts:
                        hint = (
                            f"\n\nThe score table has {looks_like_counts} and "
                            f"no score column, which is the shape of a COUNT "
                            f"file. The score and count inputs look swapped.")
                    else:
                        numeric = [
                            column for column in score_data_df.columns
                            if pd.api.types.is_numeric_dtype(
                                score_data_df[column])
                            and column not in {'plateID', 'rowID', 'columnID',
                                               'fieldID', 'objectID', 'count'}]
                        hint = (f"\n\nColumns that could be the response: "
                                f"{numeric[:12]}") if numeric else ""
                    raise ValueError(
                        f"dependent_variable="
                        f"{settings['dependent_variable']!r} is not a column "
                        f"of the score table, which has "
                        f"{list(score_data_df.columns)[:15]}.{hint}")

            # The whitelist is REGRESSION_TYPES itself, not a copy of it. The
            # copy that used to live here disagreed with the dispatcher in
            # both directions: it refused 'beta' and 'quasi_binomial', which
            # regression_model fits and check_distribution auto-selects, and
            # it accepted 'gls', 'wls', 'rlm' and 'quantile', which had no
            # backend - 'quantile' failing only at the last statement, after
            # every CSV and QC plot had been written.
            _reject_impossible_probabilities(settings)
            mode = str(settings.get('analysis_mode', 'regression')).strip().lower()
            if mode not in {'regression', 'guide_permutation'}:
                raise ValueError(
                    f"Unsupported analysis_mode {mode!r}; choose 'regression' "
                    "or 'guide_permutation'.")
            settings['analysis_mode'] = mode
            if mode == 'regression':
                reg_type = settings['regression_type']
                if reg_type is not None and reg_type not in REGRESSION_TYPES:
                    if reg_type in UNSUPPORTED_REGRESSION_TYPES:
                        raise ValueError(
                            f"Unsupported regression type {reg_type}: "
                            f"{UNSUPPORTED_REGRESSION_TYPES[reg_type]}")
                    print(f'Possible regression types: '
                          f'{list(REGRESSION_TYPES) + [None]}')
                    raise ValueError(f"Unsupported regression type {reg_type}")

                # Order matters: the reconcile can rewrite regression_type to
                # 'mixed', and the run-level knobs have to be policed against
                # the model that will actually be fitted.
                _reconcile_random_row_column_effects(settings)
                _reject_unused_run_settings(settings)

            return count_data_df, score_data_df
    
    
    
    def _count_variable_instances(df, column_1, column_2):
        # The single call site always passes both column names, so the
        # variable-arity returns this used to carry (two-tuple / bare df) were
        # unreachable; it now always returns the three-tuple its caller
        # unpacks.
        for col in (column_1, column_2):
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in independent_df. "
                    f"Available columns: {list(df.columns)}"
                )

        n_grna = df[column_1].value_counts().reset_index()
        n_grna.columns = [column_1, f"n_{column_1}"]

        n_gene = df[column_2].value_counts().reset_index()
        n_gene.columns = [column_2, f"n_{column_2}"]

        return df, n_grna, n_gene
    # WHAT THESE COUNT, because the names invite a wrong reading and the
    # maintainer asked outright whether they work.
    #
    # `df` is one row per (well, guide). So:
    #
    #   n_grna  for a guide = the number of WELLS that guide appears in.
    #   n_gene  for a gene  = the number of (well x guide) ROWS it has,
    #                         i.e. wells MULTIPLIED BY guides.
    #
    # n_gene is therefore NOT "how many guides target this gene" and NOT
    # "how many wells this gene is in". On the real screen gene 244480 has
    # ONE guide and n_gene = 5 (that guide in five wells), while 239740 has
    # TWO guides and n_gene = 15. A reader comparing n_gene across genes is
    # comparing a product, not a count of anything.
    #
    # Left as the product rather than quietly redefined: `min_n` filters on
    # it and the results CSVs of every past run carry it, so changing what
    # the number MEANS is a separate decision from fixing WHICH ROWS it is
    # taken over. The guide-support table beside it already reports guides
    # per gene, which is the number a reader usually wants.


    def _qc_plot(plot_settings):
        """Render one QC plot, reporting - not raising - on failure.

        The QC tables written between these calls are data outputs, so a
        plotting failure must not cost them. spacrGraph runs a group
        comparison, which scipy rejects with "Must enter at least two input
        sample vectors" on the very common single-plate run.
        """
        try:
            return plot_data_from_csv(settings=plot_settings)
        except Exception as e:
            print(f"Skipping QC plot {plot_settings['graph_name']!r}: {e}")
            return None, None

    def grna_metricks(df):
        """Return per-gRNA and per-well coverage counts derived from a long ``prc`` DataFrame.

        :param df: DataFrame with ``prc``, ``grna`` and ``gene`` columns.
        :returns: ``(final_grna_df, prc_gene_count_df)`` — per-gRNA
            well counts and per-well distinct-gene counts.
        """
        _assign_prc_parts(df)

        # --- 2) Compute GRNA-level Well Counts ---
        # For each (grna, plate), count the number of unique prc (wells)
        grna_well_counts = (df.groupby(['grna', 'plateID'])['prc'].nunique().reset_index(name='grna_well_count'))

        # --- 3) Compute Gene-level Well Counts ---
        # For each (gene, plate), count the number of unique prc
        gene_well_counts = (df.groupby(['gene', 'plateID'])['prc'].nunique().reset_index(name='gene_well_count'))

        # --- 4) Merge These Counts into a Single DataFrame ---
        # Because each grna is typically associated with one gene, we bring them together.
        # First, create a unique (grna, gene, plate) reference from the original df
        unique_triplets = df[['grna', 'gene', 'plateID']].drop_duplicates()

        # Merge the grna_well_count.
        #
        # Both count frames come straight off a groupby on their own join key,
        # so each holds exactly one row per key: the joins are many-to-one and
        # must not change the row count of unique_triplets. Stating that is not
        # decoration — a gRNA mapped to two genes puts the same (grna, plateID)
        # on the left twice, and if the right side ever gained a duplicate too
        # (two count CSVs for one plate concatenated, say) the result would
        # quietly gain rows and every well count written to grna_well.csv would
        # be counted more than once, with no error anywhere.
        merged_df = pd.merge(unique_triplets, grna_well_counts,
                             on=['grna', 'plateID'], how='left',
                             validate='many_to_one')

        # Merge the gene_well_count. Many gRNAs share a gene, so the left side
        # is legitimately many; gene_well_counts is one row per (gene, plate).
        merged_df = pd.merge(merged_df, gene_well_counts,
                             on=['gene', 'plateID'], how='left',
                             validate='many_to_one')

        # Keep only the columns needed (if you want to keep 'gene', remove the drop below)
        final_grna_df = merged_df[['grna', 'plateID', 'grna_well_count', 'gene_well_count']]

        # --- 5) Compute gene_count per prc ---
        # For each prc (well), how many distinct genes are there?
        prc_gene_count_df = (df.groupby('prc')['gene'].nunique().reset_index(name='gene_count'))
        _assign_prc_parts(prc_gene_count_df)

        return final_grna_df, prc_gene_count_df
    
    def get_outlier_reference_values(df, outlier_col, return_col):
        """Return unique ``return_col`` values whose ``outlier_col`` falls outside 1.5*IQR.

        :param df: Input DataFrame.
        :param outlier_col: Numeric column screened for outliers.
        :param return_col: Column whose distinct values are returned.
        :returns: List of unique reference values for outlier rows.
        """
        # Calculate Q1, Q3, and IQR for the outlier_col
        Q1 = df[outlier_col].quantile(0.05)
        Q3 = df[outlier_col].quantile(0.95)
        IQR = Q3 - Q1
        
        # Determine the outlier cutoffs
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create a mask for outliers
        outlier_mask = (df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)
        
        outliers = df.loc[outlier_mask, return_col]
        
        outliers_ls = outliers.unique().tolist()
        
        return outliers_ls
    
    def bootstrap_selection_frequencies(X, y, formula, alpha='auto', n_boot=200,
                                        random_state=None,
                                        regression_type='lasso', l1_ratio=0.5,
                                        group_lasso_lambda='auto'):
        """Return per-feature selection frequencies from a nonparametric bootstrap.

        Output ranks features by how often their coefficient is non-zero
        across resamples; this is a stability score, not a hypothesis test.

        :param X: Long-form DataFrame; design matrix is built per resample
            from ``formula`` for stable factor levels.
        :param y: Response array aligned with ``X`` by index.
        :param formula: Patsy formula for ``dmatrices``.
        :param alpha: Regularisation strength; ``'auto'``/``None`` runs the
            cross-validated estimator per resample.
        :param n_boot: Number of bootstrap resamples. Default ``200``.
        :param random_state: Seed for the resampling RNG.
        :param regression_type: ``'lasso'``, ``'elasticnet'`` or
            ``'group_lasso'`` - the same penalty the reported coefficients
            were fitted with, or the frequencies would describe a different
            model from the one in ``results.csv``.
        :param l1_ratio: ``elasticnet`` mix; ignored for ``'lasso'``.
        :param group_lasso_lambda: the block penalty, for ``'group_lasso'``.
            It is a separate argument from ``alpha`` because it is a separate
            setting: ``alpha`` is not read by that backend at all, so a
            resample fitted at ``alpha`` would be a different model from the
            one whose coefficients this is ranking.
        :returns: DataFrame with columns ``feature``,
            ``selection_frequency`` and ``mean_coefficient``.
        :raises RuntimeError: if every resample fails to fit.
        """
        rng = np.random.default_rng(random_state)
        n = len(X)
        use_cv = alpha is None or (isinstance(alpha, str) and alpha == 'auto')

        def _estimator():
            if regression_type == 'elasticnet':
                return (ElasticNetCV(l1_ratio=l1_ratio, cv=5, max_iter=10000)
                        if use_cv else
                        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
            return (LassoCV(cv=5, max_iter=10000) if use_cv
                    else Lasso(alpha=alpha, max_iter=10000))

        # Build the reference design once so the feature index is stable.
        # The response comes back with it, because choosing a group-lasso
        # penalty below needs the pair rather than the design alone.
        y0, X0 = dmatrices(formula, data=X, return_type='dataframe')
        feature_index = pd.Index(X0.columns)

        # THE GROUP LASSO IS RESAMPLED THROUGH ITS OWN SOLVER, not through
        # sklearn's. Falling through to `_estimator()` would have fitted an
        # ORDINARY lasso on every resample and reported its stability under
        # the group lasso's name -- selecting one guide out of a gene's
        # correlated set, which is the exact behaviour the group penalty
        # exists to remove.
        #
        # PER FEATURE, not per gene, even though spacr.group_lasso.
        # stability_selection answers per gene: `_call_level_hits` merges this
        # frame onto the coefficient table on `feature`, one to one. Nothing
        # is lost by it -- a block is entirely zero or entirely non-zero, so
        # every column of a gene carries that gene's frequency -- and the
        # resampling scheme stays the one the other two penalties use, so the
        # number in the column means the same thing whichever penalty wrote it.
        blocks = (_design_column_groups(feature_index)
                  if regression_type == 'group_lasso' else None)
        # THE SAME PENALTY THE FIT USED, chosen once rather than per
        # resample. 'auto' reaches here too -- it is the panel's default for
        # this backend -- and `float('auto')` is not a number. Cross-
        # validating inside the bootstrap would also mean 200 different
        # penalties, so the frequency would be "how often a gene survives
        # SOME penalty", which is a different and weaker claim.
        block_penalty = None
        if blocks is not None:
            from . import group_lasso as group_lasso_module

            if _left_blank(group_lasso_lambda) or (
                    isinstance(group_lasso_lambda, str)
                    and group_lasso_lambda.strip().lower() == 'auto'):
                block_penalty = group_lasso_module.choose_lambda(
                    np.asarray(X0, dtype=float),
                    np.asarray(y0, dtype=float).ravel(), blocks)
            else:
                block_penalty = float(group_lasso_lambda)

        def _resample_coefficients(design, response):
            if blocks is not None:
                from . import group_lasso as group_lasso_module

                beta, _intercept, _converged = group_lasso_module.fit(
                    np.asarray(design, dtype=float),
                    np.asarray(response, dtype=float).ravel(),
                    blocks, lam=block_penalty)
                return np.asarray(beta, dtype=float).ravel()
            return np.asarray(_estimator().fit(design, response).coef_).ravel()

        nonzero_counts = pd.Series(0.0, index=feature_index)
        coef_sums = pd.Series(0.0, index=feature_index)

        successful = 0
        dropped = 0
        last_failure = None
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot = X.iloc[idx].reset_index(drop=True)
            try:
                yb, Xb = dmatrices(formula, data=boot, return_type='dataframe')
            except Exception as exc:
                # A resample can occasionally drop a factor level entirely.
                dropped += 1
                last_failure = exc
                continue
            Xb = Xb.reindex(columns=feature_index, fill_value=0.0)
            yb = np.asarray(yb).ravel()
            coefs = pd.Series(_resample_coefficients(Xb, yb),
                              index=feature_index)
            nonzero_counts += (coefs != 0).astype(float)
            coef_sums += coefs
            successful += 1

        if successful == 0:
            raise RuntimeError("All bootstrap resamples failed to fit. "
                            "Check the formula and ensure factor levels are not too sparse.")

        # Same trap as _bootstrap_wald_p_values: only "none of them" raised,
        # and that is the harmless case. `selection_frequency` is divided by
        # `successful`, so 199 of 200 resamples dropping gives a stability
        # frequency computed from a single draw — every selected feature at
        # 1.00 and every other at 0.00 — reported in the same column, with the
        # same name, as a frequency over 200.
        if dropped:
            LOG.warning(
                "stability selection: %d of %d resamples produced no design "
                "matrix (last error: %s). selection_frequency and "
                "mean_coefficient below are over the remaining %d, not over "
                "%d.", dropped, n_boot, last_failure, successful, n_boot)

        return pd.DataFrame({
            'feature': feature_index,
            'selection_frequency': (nonzero_counts / successful).values,
            'mean_coefficient': (coef_sums / successful).values,
        })

    settings = get_perform_regression_default_settings(settings)
    count_data_df, score_data_df = _perform_regression_read_data(settings)
    
    if "rowID" in count_data_df.columns:
        # A count CSV can carry rowID as the composite '<plate>_<row>' that
        # 'plate_row' columns are written with (process_reads splits the same
        # shape). Reduce it to the row by taking the token after the LAST
        # separator, per row: the plate is the component that may itself
        # contain one, the row never is.
        #
        # This used to read count_data_df['rowID'].iloc[0] alone, count its
        # parts, and apply split[1] to the whole frame, which is wrong in
        # three ways that all end in a silently mis-keyed regression:
        #   * 'exp1_plate1_r2' has three parts, not two, so it was left
        #     untouched and prc became 'plate1_exp1_plate1_r2_c1';
        #   * on a frame where only some rows carry the plate prefix, split[1]
        #     is NaN for every row that does not, so their rowID was erased;
        #   * an empty count table raised IndexError on the .iloc[0].
        count_data_df['rowID'] = (
            count_data_df['rowID'].astype(str)
            .str.rsplit(schema.KEY_SEPARATOR, n=1).str[-1]
        )

    # Pair resolution is authoritative. Filenames are suggestions in the UI,
    # never a second silent source of plate identity here.
    if {'plateID', 'rowID', 'columnID'}.issubset(score_data_df.columns):
        score_data_df['prc'] = (
            _compose_prc_column(score_data_df)
        )
    #test 1
    if settings.get('verbose'):
        print("score_data_df plateID counts:")
        print(score_data_df['plateID'].value_counts())
        print("count_data_df plateID counts:")
        print(count_data_df['plateID'].value_counts())
        
    results_path, results_path_gene, results_path_grna, hits_path, res_folder, csv_path = _perform_regression_set_paths(settings)

    batch_method = str(
        settings.get('batch_correction', 'none') or 'none'
    ).strip().lower()
    if batch_method not in {'none', 'off', 'false'}:
        dependent_variable = settings['dependent_variable']
        if dependent_variable not in score_data_df.columns:
            raise ValueError(
                f"Batch correction cannot run because dependent_variable="
                f"{dependent_variable!r} is not present in the score table. "
                "Choose an existing score column or set batch_correction=none."
            )
        from .batch_correction import (
            correct_from_metadata,
            correction_kwargs,
            write_report,
        )
        # Beside `correction_kwargs`, not inside it: that helper's output
        # is `**`-splatted into several different signatures, and adding a
        # key to it turns every caller that has not grown the parameter
        # into a TypeError. combat's two keys are named here instead.
        corrected, correction_report = correct_from_metadata(
            score_data_df[[dependent_variable]],
            score_data_df,
            batch_covariate_column=settings.get('batch_covariate_column'),
            batch_combat_mean_only=bool(
                settings.get('batch_combat_mean_only', False)),
            **correction_kwargs(settings),
        )
        report_path = write_report(
            correction_report,
            os.path.join(res_folder, 'batch_correction.json'),
        )
        # IT SAYS HOW FAR IT MOVED THE DATA, and says so when the answer is
        # "not at all".
        #
        # Instruction 135 D, reported on 2026-08-17: "plate and batch
        # correction is good but im not sure i see a diference when i use
        # it". MEASURED rather than explained. On three plates with a real
        # offset, `center` and `zscore` collapse the centroid spread from
        # 0.527 to exactly 0.000 and move each value by ~0.46 on average --
        # the correction works. On ONE plate they are an exact no-op,
        # mean|delta| = 0.000000, because there is no between-plate variance
        # to remove; on plates that genuinely agree they move values by
        # ~0.01. Both are correct, and both used to print a centroid-spread
        # line that a reader could not tell apart from a correction that had
        # done something.
        #
        # The centroid spread alone does not answer the question either: it
        # goes to 0.000 in every case that ran, including the ones that
        # changed nothing. The mean absolute shift is the number a user
        # comparing two runs is actually looking for.
        shift = float(np.abs(
            np.asarray(corrected[dependent_variable], dtype=float)
            - np.asarray(score_data_df[dependent_variable], dtype=float)
        ).mean()) if len(score_data_df) else 0.0
        n_batches = len(getattr(correction_report, 'batches', ()) or ())
        print(
            f"Batch correction {correction_report.method}: "
            f"{correction_report.centroid_spread_before} -> "
            f"{correction_report.centroid_spread_after} centroid spread, "
            f"across {n_batches} batch(es); "
            f"{dependent_variable} moved by {shift:.6g} on average. "
            f"Report: {report_path}"
        )
        if n_batches < 2:
            print(
                f"  It changed nothing, and could not: batch correction "
                f"removes variance BETWEEN batches and this run has "
                f"{n_batches}. Set batch_column to a column that varies, or "
                f"batch_correction='none' -- the result is identical either "
                f"way."
            )
        elif shift == 0.0:
            print(
                "  It changed nothing: the batches already agree on "
                f"{dependent_variable}. That is a finding about the screen, "
                "not a failure of the correction."
            )
        # ASSIGNED LAST, so the shift above compares the corrected values
        # with the originals rather than with themselves.
        score_data_df.loc[:, dependent_variable] = corrected[
            dependent_variable
        ]
        for note in correction_report.warnings:
            print(f"Warning: batch correction: {note}")

    save_settings(settings, name='regression', show=True)

    # The volcano goes with the rest of the run's output, not beside the INPUT
    # data. Writing it to the count CSV's folder put the module's headline
    # figure two directories away from every table and plot it belongs with --
    # so a run that had produced it perfectly well looked like it had produced
    # no graph at all, which is exactly how it was reported.
    count_source = os.path.dirname(settings['count_data'][0])
    volcano_path = os.path.join(res_folder, 'volcano_plot.pdf')

    if isinstance(settings['filter_value'], list):
        filter_value = list(settings['filter_value'])
    else:
        filter_value = []
    # THE CONTROL BLOCKS COME OUT TOO, WITHOUT BEING TYPED TWICE (221). A
    # well of pure control is not a screen well -- it holds one guide by
    # construction, so its phenotype says what that guide does and nothing
    # about any gene under test -- and left in it is modelled as a random
    # draw from the library, at high leverage when the control is strong.
    #
    # ADDED TO `filter_value` RATHER THAN FILTERED SEPARATELY, so there is
    # one removal, one printed line per well and one place a reader has to
    # look to know what left the run.
    try:
        from .well_spec import control_block_wells

        for well in control_block_wells(settings):
            if well not in filter_value:
                filter_value.append(well)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not resolve the control blocks", exc_info=True)
    # filter_column used to be bound only in the `isinstance(..., str)` branch,
    # so both None (the natural "do not filter" value) and the list form that
    # process_reads documents left it unbound and the process_reads call below
    # raised UnboundLocalError. clean_controls handles str / list / None.
    filter_column = settings['filter_column']

    score_data_df = clean_controls(score_data_df, settings['filter_value'], filter_column)

    # OUTLIERS GO NOW, BEFORE ANYTHING COUNTS THEM (instruction 210). Every
    # normalising step below -- the cell-count threshold, the guide
    # fractions, the aggregation -- has its denominator set by which objects
    # are present, so a segmentation artefact removed AFTER the fractions
    # are formed leaves its reads redistributed across the guides in its
    # well. Removed here, it never contributed.
    #
    # OFF UNLESS ASKED FOR, and always reported: a filter that silently
    # drops objects is a filter that will be forgotten and then blamed on
    # the annotation.
    try:
        from .outlier_filter import apply as _drop_outliers, describe

        score_data_df, _outlier_report = _drop_outliers(score_data_df,
                                                        settings)
        _said = describe(_outlier_report)
        if _said:
            print(_said)
    except Exception as _error:                                  # noqa: BLE001
        # SAID OUT LOUD. A filter the user switched on that did not run is
        # the one thing worse than one that ran silently: the numbers below
        # would be the unfiltered ones and nothing would say so.
        print(f"[outliers] the pre-annotation filter did not run "
              f"({type(_error).__name__}: {_error}); the counts below are "
              f"unfiltered")

    if settings['verbose']:
        print(f"Dependent variable after clean_controls: {len(score_data_df)}")

    # Which settings this run DERIVED rather than being given. Reset here, at
    # the top of the run, so a second run in one process cannot inherit the
    # first one's -- a GUI session runs many.
    _AUTOMATIC_SETTINGS.clear()

    # WRITTEN INTO THE RUN FOLDER, not copied into it afterwards.
    #
    # `cell_min_threshold.pdf` used to go to <count folder>/results/ -- the
    # SCREEN folder, one path shared by every run of the screen -- and this
    # call site snapshotted that folder and copied back whatever appeared, the
    # way it still does for the sequencing sweep below.
    #
    # That worked for one run at a time and NOT for the sweep.
    # `parameter_sweep.run_sweep_parallel` fits n_jobs trials of the same
    # screen in a ProcessPoolExecutor. `_trial_settings` gives each trial its
    # own `src`, so the RUN folders are already separate -- but the default
    # destination here comes from `count_data`, which every trial shares, and
    # this figure is drawn on EVERY trial (the call is unconditional; only
    # whether its ANSWER is used depends on min_cell_count). So n_jobs
    # workers wrote one path at once, and "every figure whose stamp changed
    # since I started" cannot tell one worker's curve from another's: a trial
    # could file the neighbouring trial's picture as its own, or copy one
    # mid-write.
    #
    # `res_folder` is this run's own folder, so naming it removes the shared
    # path rather than working around it.
    screen_folders = _screen_figure_folders(settings)
    sim_min_count = minimum_cell_simulation(
        settings, tolerance=settings['tolerance'], dst=res_folder)

    if settings['min_cell_count'] is None:
        settings['min_cell_count'] = sim_min_count
        _AUTOMATIC_SETTINGS['min_cell_count'] = sim_min_count
        
    if settings['verbose']:
        print(f"Minimum cell count: {settings['min_cell_count']}")
        print(f"Dependent variable after minimum cell count filter: {len(score_data_df)}")
        display(score_data_df)

    orig_dv = settings['dependent_variable']

    # THE RESPONSE BEFORE THE TRANSFORM, kept for the panel below. Taken
    # here because `process_scores` applies the transform and hands back only
    # the result -- and the whole point of instruction 218's panel is the
    # comparison, which is unrecoverable once the untransformed values are
    # gone.
    _before_transform = None
    try:
        _before_transform, _ = process_scores(
            score_data_df, settings['dependent_variable'], None,
            settings['min_cell_count'], settings['agg_type'],
            None, settings['regression_type'],
            settings['invert_dependent_variable'])
    except Exception:                                            # noqa: BLE001
        # A panel is not worth losing a run to. The comparison is dropped,
        # the fit is not.
        _before_transform = None

    dependent_df, dependent_variable = process_scores(
        score_data_df, settings['dependent_variable'], None,
        settings['min_cell_count'], settings['agg_type'],
        settings['transform'], settings['regression_type'],
        settings['invert_dependent_variable'])

    _show_response_distribution(_before_transform, dependent_variable,
                                settings)
    
    if settings['verbose']:
        print(f"Dependent variable after process_scores: {len(dependent_df)}")
        display(dependent_df)
    
    if settings.get('calibrate_fraction_threshold'):
        # MEASURED FROM THE CONTROL WELLS, when the user asked for that.
        #
        # `target_unique_count` answers a different question -- how many
        # gRNAs a well should end up with -- and answers it from the counts
        # alone. This one asks which cut-off makes the imaging and the
        # sequencing agree, which is the question a screen is actually
        # asking, and it can only be asked where the plate design names
        # pure control wells.
        #
        # A sweep that cannot run says so and falls through to whatever the
        # settings already chose. It must not take the run down: the
        # calibration is an improvement on a number that already has a
        # value, not a prerequisite for having one.
        measured = _calibrated_fraction_threshold(settings)
        if measured is not None:
            settings['fraction_threshold'] = measured
            _AUTOMATIC_SETTINGS['fraction_threshold'] = measured

    if settings['fraction_threshold'] is None:
        # THE gRNA THRESHOLD GRAPH BELONGS TO THE RUN.
        #
        # `graph_sequencing_stats` derives its own destination from
        # `count_data[0]` inside spacr.sequencing, so the sweep curve and the
        # unique-count plate heatmap land in the SCREEN folder. Both are
        # streamed through `plt.show()` and so still reach the live figure
        # queue -- measured, not assumed -- but neither was ever in the run
        # folder, which is what the all-figures grid walks for a saved run and
        # what a reader opens by hand. Reported as "for some reason now i
        # dont see the grna threshold graph".
        before_sweep = _figure_stamps(screen_folders)
        settings['fraction_threshold'] = _graph_sequencing_stats(settings)
        _AUTOMATIC_SETTINGS['fraction_threshold'] = settings['fraction_threshold']
        for kept in _keep_figures_with_the_run(before_sweep, screen_folders,
                                               res_folder):
            print(f"Kept with the run: {kept}")
    else:
        # AND IT IS DRAWN ANYWAY. Reported twice -- "for some reason now i
        # dont see the grna threshold graph", and again 2026-08-21: "in the
        # figure view i nevers ee the frna threhsold graph".
        #
        # THE ANSWER LAST TIME WAS A SENTENCE SAYING WHY, which is not what
        # was asked for. The sweep is a fact about the SCREEN -- how many
        # guides survive at each threshold -- and it is worth the same
        # whether spaCR chose the threshold or the user did. It is arguably
        # worth MORE when the user chose one, because then it is the only
        # thing that says where their number sits on the curve.
        #
        # The default is 0.02, so the old gate meant the graph was never
        # drawn on an ordinary run: the one case it fired in was the one
        # nobody was in.
        before_sweep = _figure_stamps(screen_folders)
        _draw_the_threshold_sweep(
            settings, res_folder,
            measured='fraction_threshold' in _AUTOMATIC_SETTINGS)
        for kept in _keep_figures_with_the_run(before_sweep, screen_folders,
                                               res_folder):
            print(f"Kept with the run: {kept}")

    # WHAT THE RUN ACTUALLY USED, said where the settings are read.
    #
    # Asked for 2026-08-17: "if no fraction threshold and min cell cound is
    # set these are set automatically, these automatic values should be shown
    # in the runs values rows".
    #
    # The settings table is printed -- and `save_settings` writes the CSV --
    # BEFORE either of these is derived, so both showed `None` there and the
    # numbers only ever appeared in passing prose ("Closest Fraction
    # Threshold: 0.0168"). A settings record that says None for a value the
    # run chose is a record you cannot reproduce the run from.
    if _AUTOMATIC_SETTINGS:
        print("\nChosen automatically (not set by the user):")
        for key, value in _AUTOMATIC_SETTINGS.items():
            print(f"  {key:<28}{value}")
        # Re-saved so the CSV carries the resolved values rather than the
        # Nones it was written with. Same path, so it is one file and the
        # complete version wins.
        try:
            save_settings(settings, name='regression', show=False)
        except Exception as error:                               # noqa: BLE001
            print(f"Could not re-save the resolved settings: {error}")

    # WHERE THE EXCLUSION COUNTS ARE COLLECTED (instruction 156). One dict on
    # the settings, filled by whichever step drops rows, read by the run
    # summary at the end. It lives on `settings` rather than on the frame
    # because a count has to survive the joins and re-indexes the frame does
    # not carry `.attrs` through.
    _exclusions = settings.setdefault("_regression_exclusions", {})
    _stage(settings, "reading the counts")
    independent_df = process_reads(
        count_data_df, settings['fraction_threshold'], None,
        filter_column=filter_column, filter_value=filter_value,
        record=_exclusions)
        
    if settings['verbose']:
        print("independent_df columns:", list(independent_df.columns))
        print("independent_df head:")
        print(independent_df.head())
        print(independent_df)
        
    # COUNTED AFTER THE MERGE, NOT HERE. See below -- the counts are taken
    # from `merged_df`, which is what actually reached the fit. Counting
    # `independent_df` here counted rows the inner merge was about to drop.
    
    if settings['verbose']:
        print(f"Independent variable after process_reads: {len(independent_df)}")
    
    # The regression's own join, and the one whose cardinality decides every
    # number this function goes on to report. independent_df is one row per
    # (well, gRNA); what dependent_df is depends on agg_type:
    #
    #   agg_type in {'mean', 'median', 'quantile'} or poisson
    #       process_scores groups on prc, so it is exactly one row per well
    #       and this is many-to-one. A duplicated prc on that side would
    #       multiply every gRNA row of that well, inflating cell_count, the
    #       per-well gRNA counts and the regression's effective n, with no
    #       error and no visible symptom.
    #   agg_type is None (forced for quantile regression, see settings.py)
    #       process_scores returns one row per OBJECT, so the join is a
    #       deliberate cross product of the well's gRNAs with the well's
    #       cells. That is many-to-many and saying so explicitly is what
    #       stops a blanket 'many_to_one' here from crashing quantile
    #       regression on perfectly good data.
    merge_validate = (
        'many_to_many' if settings['agg_type'] is None else 'many_to_one')
    merged_df = pd.merge(independent_df, dependent_df, on='prc',
                         validate=merge_validate)

    _check_score_count_pairing(independent_df, dependent_df, merged_df,
                               record=settings.get('_regression_exclusions'))

    # n_grna / n_gene DESCRIBE THE ROWS THAT REACHED THE FIT.
    #
    # They were counted on `independent_df`, BEFORE this merge -- and the
    # merge is an INNER join (no `how=`), so every sequencing well without an
    # imaging partner was counted and then dropped. On the real screen that
    # is 724 of 1,344 wells: "Paired 620 wells. 724 sequencing well(s) ...
    # take no part in the regression." Measured on a synthetic case with half
    # the wells unpaired, every count came out EXACTLY 2x too high.
    #
    # It matters beyond the display. `min_n` filters the hit list on these
    #     significant[significant['n_grna'] > settings['min_n']]
    # so an inflated count lets a guide through a filter it should fail --
    # which is a hit reported on evidence that is not there.
    _merged_for_counts, n_grna, n_gene = _count_variable_instances(
        merged_df, column_1='grna', column_2='gene')

    if settings['verbose']:
        display(independent_df)
        display(dependent_df)
        display(merged_df)

    _assign_prc_parts(merged_df)

    try:
        os.makedirs(res_folder, exist_ok=True)
        data_path = os.path.join(res_folder, 'regression_data.csv')
        merged_df.to_csv(data_path, index=False)
        print(f"Saved regression data to {data_path}")
        
        # plot_data_from_csv reads settings['remove_outliers'] directly and
        # never applies its own defaults, so omitting the key raised KeyError
        # on the very first QC plot; combined with the swallow-everything
        # try/except around this block, grna_well.csv and well_grna.csv were
        # then silently never written.
        cell_settings = {'src':data_path,
                        'graph_name':'cell_count',
                        'data_column':['cell_count'],
                        'grouping_column':'plateID',
                        'graph_type':'jitter_bar',
                        'theme':'bright',
                        'save':True,
                        'y_lim':[None,None],
                        'log_y':False,
                        'log_x':False,
                        'representation':'well',
                        'remove_outliers':False,
                        'verbose':False}
        
        _, _ = _qc_plot(cell_settings)
        
        final_grna_df, prc_gene_count_df = grna_metricks(merged_df)
        
        if settings['outlier_detection']:
            outliers_grna = get_outlier_reference_values(final_grna_df,outlier_col='grna_well_count',return_col='grna')
            if len (outliers_grna) > 0:
                # .copy() IS LOAD-BEARING, not tidiness. Without it this is a
                # slice of `merged_df`, and `grna_metricks` calls
                # `_assign_prc_parts`, which ASSIGNS plateID/rowID/columnID
                # onto the frame it is handed. Writing to a slice raises
                # SettingWithCopyWarning -- which this suite promotes to an
                # error (pytest.ini) and which pandas may in any case decline
                # to write through -- and the exception was caught by the
                # blanket `except` at the bottom of this QC block, so
                # `grna_well.csv` and `well_grna.csv` were never written and
                # the only trace was the warning text printed on its own line.
                # Reproduced 2026-08-17: with outlier_detection=True the run
                # completed and produced every regression output, and the two
                # gRNA-coverage tables were simply absent.
                merged_df = merged_df[
                    ~merged_df['grna'].isin(outliers_grna)].copy()
                final_grna_df, prc_gene_count_df = grna_metricks(merged_df)
                merged_df.to_csv(data_path, index=False)
                print(f"Saved regression data to {data_path}")

        grna_data_path = os.path.join(res_folder, 'grna_well.csv')
        final_grna_df.to_csv(grna_data_path, index=False)
        print(f"Saved grna per well data to {grna_data_path}")
        
        wells_per_gene_settings = {'src':grna_data_path,
                                'graph_name':'wells_per_gene',
                                'data_column':['grna_well_count'],
                                'grouping_column':'plateID',
                                'graph_type':'jitter_bar',
                                'theme':'bright',
                                'save':True,
                                'y_lim':[None,None],
                                'log_y':False,
                                'log_x':False,
                                'representation':'object',
                                'remove_outliers':False,
                                'verbose':True}
        
        _, _ = _qc_plot(wells_per_gene_settings)
        
        grna_well_data_path = os.path.join(res_folder, 'well_grna.csv')
        prc_gene_count_df.to_csv(grna_well_data_path, index=False)
        print(f"Saved well per grna data to {grna_well_data_path}")
        
        grna_per_well_settings = {'src':grna_well_data_path,
                                'graph_name':'gene_per_well',
                                'data_column':['gene_count'],
                                'grouping_column':'plateID',
                                'graph_type':'jitter_bar',
                                'theme':'bright',
                                'save':True,
                                'y_lim':[None,None],
                                'log_y':False,
                                'log_x':False,
                                'representation':'well',
                                'remove_outliers':False,
                                'verbose':False}
        
        _, _ = _qc_plot(grna_per_well_settings)
        
    except Exception as e:
        print(e)

    # inference='auto' is decided here and not in settings.py, because it is
    # the first point at which the guides and analysed wells can be counted.
    if str(settings.get('inference', 'parametric')).lower() == 'auto':
        resolved_mode, reason = resolve_auto_inference(merged_df, settings)
        settings['analysis_mode'] = resolved_mode
        print(f"inference='auto': {reason}")
    elif settings.get('analysis_mode') == 'regression':
        # The user chose the simultaneous fit. It is theirs to choose, and it
        # runs -- but a fit with more parameters than wells returns one
        # arbitrary solution out of infinitely many, and saying nothing is how
        # a published figure came to carry coefficients that could not be
        # reproduced from their own inputs. So: run it, and say so loudly.
        # ONE CHECK PER FIT. `level='both'` runs two models of very
        # different widths and only one of them may be too wide, so a single
        # verdict for the run would either cry wolf about the gene fit or
        # stay silent about the guide fit.
        for _one in resolve_levels(settings.get('regression_type'),
                                   settings.get('level', 'both')):
            warning = _identifiability_warning(merged_df, settings,
                                               level=_one)
            if warning:
                print(f"  level={_one!r}:")
                print(warning)

    if settings.get('analysis_mode') == 'guide_permutation':
        # SAID BEFORE THE FIT, not only in the summary afterwards. With
        # inference='nonparametric' -- the default since 2026-08-18 -- the
        # permutation path fits no model, so regression_type is never read.
        # Verified on the maintainer's four-plate screen: 'ols' and 'mixed'
        # produced byte-identical results, 1612 rows across all 24 columns.
        # That is why "i ran a mixed model and an ols model and even if the
        # ols model is marked as loaded i think i still see the mixed
        # results" was a correct observation: they ARE the same numbers. A
        # user who is told this before the run does not queue the second one.
        _chosen = settings.get('regression_type')
        if _chosen:
            print(f"inference='nonparametric': this is a permutation test, so "
                  f"it fits no model and regression_type={_chosen!r} is not "
                  f"read. Choosing a different regression_type with this "
                  f"inference gives the same numbers; set "
                  f"inference='parametric' to fit {_chosen!r} itself.")
        _stage(settings, "permuting the guides")
        output = _run_guide_permutation_analysis(
            merged_df, dependent_variable, res_folder, settings)
        _stage(settings, "the permutation has returned")
        if settings.get('verbose'):
            print(
                f"Guide permutation analysis tested "
                f"{len(output['primary'])} guides in the primary "
                f">={output['primary_min_wells']}-well family and called "
                f"{len(output['significant'])} at "
                f"{settings['multiple_testing_method']} "
                f"alpha={settings['fdr_alpha']}."
            )
        # THE SUMMARY, BEFORE THE EARLY RETURN. Instruction 156 placed its
        # call at the end of the parametric path, which this branch never
        # reaches -- so the ONE mode that has no statsmodels summary to fall
        # back on was also the one mode that wrote no spaCR summary either,
        # which is exactly the run the maintainer reported: "No summary: this
        # run came back without a fitted model, so there is none to
        # summarise", from a nonparametric mixed fit.
        #
        # There is no `model` here and there never will be: a permutation test
        # has no design matrix and no coefficient covariance. That is what the
        # summary says, rather than being the reason it is absent.
        try:
            from .regression_summary import write_run_summary
        except ImportError:
            pass
        else:
            try:
                # No `inference=` argument: `_is_nonparametric` reads it off
                # the settings, which is the more robust answer -- an
                # `inference='auto'` resolved into `analysis_mode`, and a
                # settings CSV predating the `inference` key, both still come
                # out right, where a keyword passed from here would only be
                # right at this one call site.
                write_run_summary(
                    res_folder, model=None, settings=settings,
                    coef_df=output.get('primary'),
                    regression_type=settings.get('regression_type'))
            except Exception as error:  # noqa: BLE001 - never lose a run
                print(f"Could not write the run summary: "
                      f"{type(error).__name__}: {error}")
        # THE KEYS EVERY CONSUMER OF A RUN READS, on this branch too.
        # `app_screen._on_regression_done` and the Measurements queue both
        # take the run's folder from `res_folder`, and this early return was
        # the one path that did not carry it -- so the DEFAULT inference
        # produced a complete results folder that the GUI then registered
        # with no folder at all, which is the "No summary: this panel was
        # opened from a results table on disk" the maintainer reported. A
        # copy of `settings`, for the same reason the parametric path hands
        # one back: the shared settings/ file is overwritten by the next run
        # of the same screen, so it describes the wrong one.
        output.setdefault('res_folder', res_folder)
        output.setdefault('settings', dict(settings))
        output.setdefault('regression_type', settings.get('regression_type'))
        return output
        
    # EVERY PLATE AS ONE FIGURE, on one colour scale, with square wells.
    # The old call wrote one wide, short PDF per measurement into a fixed
    # name -- so repeat runs overwrote each other, four plates took eight
    # grid slots, and each plate got its OWN colour scale, which makes two
    # plates incomparable at a glance. See spacr.figures.plates.
    if not _show_plates(merged_df, orig_dv, res_folder):
        _ = plot_plates(merged_df, variable=orig_dv, grouping='mean',
                        min_max='allq', cmap='viridis', min_count=None,
                        dst=res_folder)

    # TWO FITS, NOT ONE DESIGN WITH BOTH LEVELS IN IT.
    #
    # `gene_fraction` is the SUM of the gene's gRNA fractions
    # (check_and_clean_data), so the design this pipeline fitted until
    # instruction 132 -- `fraction:grna + gene_fraction:gene + rowID +
    # columnID` -- contained a block of columns and its own sums. Measured on
    # the maintainer's TSG101 screen: 1945 rows, 1248 parameters, RANK 862, an
    # exact 386-dimensional null space, condition number 2.3e18. statsmodels
    # pseudo-inverted it and reported a coefficient and a P value for every
    # term; the residual sum of squares is bit-identical at the answer it gave
    # and at that answer plus seven times a null vector. 102 single-guide
    # genes came back as exact duplicates of their one guide -- 244480 and
    # 244480_3 both 3.389291 at 2.873149e-13.
    #
    # Split in two, each level is full rank: 859 parameters at rank 859 for
    # the guide fit, 425 at 425 for the gene fit.
    _stage(settings, "fitting the model")
    fits = regression_levels(
        merged_df, csv_path, dependent_variable=dependent_variable,
        regression_type=settings['regression_type'],
        # WHO fits it (instruction 141). `.get`, not indexed, for the same
        # reason as `model_plate_position` below: no settings CSV written
        # before 2026-08-18 carries this key, and what every one of those
        # files meant is the backend that produced them.
        regression_backend=settings.get('regression_backend',
                                        DEFAULT_REGRESSION_BACKEND),
        level=settings.get('level', 'both'),
        alpha=settings['alpha'],
        random_row_column_effects=settings['random_row_column_effects'],
        # IS PLATE POSITION IN THE MODEL AT ALL (instruction 143 A).
        # `.get`, not indexed, for the same reason as the three below: no
        # settings CSV written before 2026-08-18 carries this key, and the
        # value an absent one meant is True -- every run before today fitted
        # rowID and columnID unconditionally.
        model_plate_position=settings.get('model_plate_position', True),
        nc=settings['negative_control'], pc=settings['positive_control'],
        controls=settings['controls'], dst=res_folder,
        # 183: a quiet run gets the summary HEADER and a pointer at the file;
        # verbose gets every coefficient, which is what verbose is for.
        verbose=bool(settings.get('verbose')),
        # 182 A/C: what was already done to the response, so the family
        # sniffer can name the scale it examined and refuse to be quiet about
        # a link stacked on a transform.
        transform=str(settings.get('transform') or ''),
        cov_type=settings['cov_type'],
        l1_ratio=settings['l1_ratio'],
        quantile=settings['quantile'],
        hinge_threshold=settings['hinge_threshold'],
        hinge_n_boot=settings['hinge_n_boot'],
        huber_t=settings['huber_t'],
        spline_knots=settings.get('spline_knots', 4),
        spline_degree=settings.get('spline_degree', 3),
        # DEFAULTED HERE, not indexed. `group_lasso_lambda`, `rra_alpha` and
        # `rra_permutations` are declared in spacr.settings, but a settings
        # CSV written before instruction 133 has none of them and must still
        # run -- and every one of these three is only read by the backend that
        # names it, so its default is never the difference between two
        # answers for any other type.
        group_lasso_lambda=settings.get('group_lasso_lambda', 'auto'),
        rra_alpha=settings.get('rra_alpha', 0.25),
        rra_permutations=settings.get('rra_permutations', 10000),
        # THE QC SUITE HAS TO BE DECLINABLE, and until this line it was not.
        # `regression()` grew a `qc` parameter precisely so a parameter sweep
        # could turn it off, and then nothing passed one -- so every trial of
        # every sweep paid the full diagnostic suite: ~5.8 s and ~19 figures
        # plus a combined PDF, i.e. roughly ten minutes and two thousand files
        # per hundred trials, with no way to say no. On a single analysis it
        # is exactly what you want, which is why it stays on by default.
        qc=bool(settings.get('regression_qc', True)),
        legacy_volcano=bool(settings.get('legacy_volcano', False)),
        # WHAT THE INTERCEPT IS. `.get`, not indexed, for the reason the
        # three above give: no settings CSV written before this key existed
        # carries it, and the value an absent one meant is a fitted
        # intercept -- which is what every run before it did.
        intercept=str(settings.get('intercept') or 'fitted'),
        intercept_value=float(settings.get('intercept_value') or 0.0),
    )
    regression_type = next(iter(fits.values()))[2]

    # THE DIAGNOSTICS, WRITTEN HERE BECAUSE THIS IS WHERE THE INPUTS ARE.
    # `spacr.regression_diagnostics` has computed all of these since it was
    # written -- after a fit that returned a confident P value for every one of
    # 824 guides in 587 wells out of a rank-deficient matrix -- and until now
    # nothing called it, so the checks that would have caught that failure were
    # unreachable by a user. Instruction 322.
    settings['_regression_diagnostics'] = _write_regression_diagnostics(
        res_folder, merged_df, fits, settings)

    level_tables = {
        one: _annotate_level_coefficients(one_coef, n_grna, n_gene)
        for one, (_model, one_coef, _type) in fits.items()
    }

    # THE MIXED FIT IS ALREADY BOTH LEVELS, so it is split by TERM TYPE rather
    # than fitted twice. Its gene rows are fixed effects with standard errors
    # and p-values; its guide rows are BLUPs -- shrunken predictions of a
    # random effect. A BLUP has no null hypothesis to reject, so it gets no
    # q value here and no line in the hit list, and results_grna.csv from a
    # mixed run says so in its `term_type` column.
    if regression_type == 'mixed' and 'gene' in level_tables:
        whole = level_tables.pop('gene')
        blups = whole['term_type'] == TERM_BLUP
        level_tables['gene'] = whole.loc[~blups].copy()
        guide_table = whole.loc[blups].copy()
        guide_table['level'] = 'grna'
        guide_table['q_value'] = np.nan
        guide_table['multiple_testing_method'] = 'none'
        level_tables['grna'] = guide_table
        print(f"Mixed fit: {len(level_tables['gene'])} gene rows corrected as "
              f"one family, {len(guide_table)} guide BLUPs written without a "
              f"q value. Choose a fixed-effects model with level='grna' for a "
              f"guide-level hit list.")

    # EACH FIT CORRECTED WITHIN ITSELF. Two families, never one: same wells,
    # and the gene regressor IS the sum of the guide regressors, so pooling
    # would both break the independence the correction assumes and double the
    # family for no protection.
    corrected = {}
    hits_by_level = {}
    thresholds_by_level = {}
    for one, table in level_tables.items():
        if regression_type == 'mixed' and one == 'grna':
            # BLUPs: no test, so nothing to correct and nothing to call.
            corrected[one] = table
            hits_by_level[one] = table.iloc[0:0]
            thresholds_by_level[one] = 0
            continue
        table, level_hits, level_threshold, _rule = _call_level_hits(
            table, one, settings, regression_type, merged_df,
            dependent_variable, bootstrap=bootstrap_selection_frequencies)
        corrected[one] = table
        hits_by_level[one] = level_hits
        thresholds_by_level[one] = level_threshold

    # THE PRIMARY LEVEL is the guide when there is one: the guide is the unit
    # the screen measures, and it is what results.csv, the volcano's default
    # and the model summary have always been about.
    primary = 'grna' if 'grna' in fits else next(iter(fits))
    model = fits[primary][0]
    reg_threshold = thresholds_by_level.get(primary, 0)

    # ONE ROW PER GUIDE / PER GENE in the per-level files. The intercept and
    # the mixed fit's variance components are terms of the fit, not units of
    # the screen, and results_gene.csv has never carried them -- the results
    # panel, `hits.load_results` and the volcano all read these files as a list
    # of things that were tested. They stay in results.csv, which is the whole
    # fit. (`n_grna` / `n_gene` are NaN exactly for the rows that name no unit,
    # which is the same rule this used before the split.)
    grna_coef_df = corrected.get('grna')
    gene_coef_df = corrected.get('gene')
    if grna_coef_df is not None:
        grna_coef_df = grna_coef_df.dropna(subset=['n_grna'])
    if gene_coef_df is not None:
        gene_coef_df = gene_coef_df.dropna(subset=['n_gene'])
    # A LEVEL THAT WAS NOT FITTED GETS AN EMPTY TABLE WITH THE RIGHT COLUMNS,
    # not a missing file. `hits.load_results`, the results panel and
    # `run_compare` all read results_gene.csv / results_grna.csv by name, and
    # a file that is absent is indistinguishable from a run that crashed.
    template = corrected[primary].iloc[0:0]
    if grna_coef_df is None:
        print("level='gene': no guide fit was run, so results_grna.csv is "
              "written empty. Set level='both' or level='grna' for one.")
        grna_coef_df = template
    if gene_coef_df is None:
        print("level='grna': no gene fit was run, so results_gene.csv is "
              "written empty. Set level='both' or level='gene' for one.")
        gene_coef_df = template

    # results.csv is BOTH tables stacked, each row carrying the `level` it was
    # fitted and corrected at. One row per guide and one per gene, never a
    # gene once per guide -- which is what the collinear single design
    # produced and what put every gene on the volcano several times.
    def _stack(frames):
        # pd.concat([]) raises "No objects to concatenate", and a run where
        # neither level called a hit is an ordinary outcome, not an error --
        # it is what a screen with nothing in it looks like. The empty table
        # keeps its columns so results_significant.csv still has a header.
        kept = [frame for frame in frames if len(frame)]
        return pd.concat(kept, ignore_index=True) if kept else template

    coef_df = _stack(corrected.values())
    significant = _stack(hits_by_level.values())

    # EVERY EXPORTED TABLE CARRIES THE ANNOTATION, not just the volcano's
    # colours. Instruction 133, asked for on 2026-08-17: "if it is on all the
    # exported tables should be merged with the relevant Toxoplasma
    # information".
    #
    # Until this block `toxo=True` reached two places -- the volcano and two
    # heatmaps -- and the CSV a reader actually opens came out as bare gene
    # numbers and coefficients. The annotation was then joined by hand in a
    # spreadsheet, which is where wrong-key mistakes live.
    #
    # `spacr.annotation` declares every merge many_to_one and collapses each
    # source to one row per gene first, so this cannot change a row count.
    # It is checked anyway: this table's contract is one row per coefficient
    # and it is worth being the kind of code that says so.
    if _annotation_source(settings):
        from .annotation import annotate_with, supplementary

        source = _annotation_source(settings)
        cache = _annotation_cache(settings)
        annotated, notes = {}, []
        for name, frame in (('results', coef_df), ('gene', gene_coef_df),
                            ('grna', grna_coef_df),
                            ('significant', significant)):
            before = len(frame)
            annotated[name], note = annotate_with(
                frame, source, cache_dir=cache, quiet=(name != 'results'))
            if note and name == 'results':
                notes.append(note)
            if len(annotated[name]) != before:
                raise ValueError(
                    f"the {source} annotation changed {name} from {before} "
                    f"to {len(annotated[name])} row(s).")
        for note in notes:
            print(f"Annotation: {note}")
        coef_df = annotated['results']
        gene_coef_df = annotated['gene']
        grna_coef_df = annotated['grna']
        significant = annotated['significant']

        # The DeepTMHMM topology, as its own supplementary table: 72 columns
        # of segment coordinates beside a coefficient is a table nobody opens
        # twice, and "where does its third helix start" is a different
        # question from "does this protein have a signal peptide".
        supplementary(
            coef_df['feature'] if 'feature' in coef_df.columns else None,
            path=os.path.join(res_folder, 'supplementary_topology.csv'))

    coef_df.to_csv(results_path, index=False)
    gene_coef_df.to_csv(results_path_gene, index=False)
    grna_coef_df.to_csv(results_path_grna, index=False)
        
    if regression_type in ['ols', 'beta']:
        # WRITTEN WHETHER OR NOT ANYBODY IS WATCHING. The save used to sit
        # inside the `verbose` branch beside the print, so a quiet run -- the
        # normal case -- left no summary on disk at all, and the results panel
        # re-opened from that folder had nothing to read back. Printing is a
        # console preference; the summary is part of the run's output.
        if settings['verbose']:
            print(model.summary())
        save_summary_to_file(
            model, file_path=os.path.join(res_folder, SUMMARY_FILENAME))

    # THE spaCR SUMMARY, for EVERY mode -- instruction 156. The block above
    # writes the statsmodels summary and only two of the supported regression
    # types reach it; a nonparametric run has no fitted model at all, so it got
    # nothing. This writes what spaCR itself knows about the fit -- the design,
    # the assumptions with their tests, the call, and what was excluded -- so a
    # mode statsmodels cannot summarise still has a summary.
    #
    # GUARDED, and deliberately so: a run must not die for a summary. The
    # module is optional at this point in its life, and a failure here is
    # reported rather than raised, because losing an hour's fit to a reporting
    # bug is the trade nobody would make.
    try:
        from .regression_summary import write_run_summary
    except ImportError:
        pass
    else:
        try:
            _stage(settings, "the fit has returned")
            write_run_summary(res_folder, model=model, settings=settings,
                              coef_df=coef_df, regression_type=regression_type)
        except Exception as error:  # noqa: BLE001 - never lose a run
            print(f"Could not write the run summary: "
                  f"{type(error).__name__}: {error}")

    significant.to_csv(hits_path, index=False)
    significant_grna_filtered = significant[significant['n_grna'] > settings['min_n']]
    significant_gene_filtered = significant[significant['n_gene'] > settings['min_n']]
    significant_filtered = pd.concat([significant_grna_filtered, significant_gene_filtered])
    filtered_hit_path = os.path.join(os.path.dirname(hits_path), 'results_significant_filtered.csv')
    significant_filtered.to_csv(filtered_hit_path, index=False)

    if isinstance(settings['metadata_files'], str):
        settings['metadata_files'] = [settings['metadata_files']]

    # THE VOLCANO MUST NOT DEPEND ON HAVING A METADATA FILE.
    #
    # These three names were bound ONLY inside the loop below. With no
    # metadata file the loop never ran, and the toxo block -- which reads all
    # three unconditionally -- raised NameError before drawing anything. The
    # run otherwise completed, wrote its histograms, heatmaps and every
    # results CSV, and simply produced no volcano: the one figure the module
    # exists to make, missing with no error the user could see.
    #
    # The results tables are the correct default. Metadata is an annotation
    # join that adds columns; it is not what makes a volcano plottable.
    merged_df = tabular.read_table(results_path, report=None)
    gene_merged_df = tabular.read_table(results_path_gene, report=None)
    grna_merged_df = tabular.read_table(results_path_grna, report=None)

    for metadata_file in settings['metadata_files']:
        file = os.path.basename(metadata_file)
        filename, _ = os.path.splitext(file)
        # AN UNREADABLE ANNOTATION FILE MUST NOT DESTROY A FINISHED FIT.
        #
        # The regression is complete and written by this point; this loop only
        # decorates the results with gene metadata. An empty or missing file
        # here raised EmptyDataError straight out of perform_regression, so a
        # perfectly good run was reported as a failure and its coefficients
        # went unused -- which is what it looks like from a sweep, where one
        # bad metadata path fails every trial that touches it.
        try:
            if not os.path.isfile(metadata_file) \
                    or os.path.getsize(metadata_file) == 0:
                print(f"Skipping empty or missing metadata file: "
                      f"{metadata_file}")
                continue
        except OSError:
            continue
        try:
            _ = merge_regression_res_with_metadata(hits_path, metadata_file, name=filename)
            merged_df = merge_regression_res_with_metadata(results_path, metadata_file, name=filename)
            gene_merged_df = merge_regression_res_with_metadata(results_path_gene, metadata_file, name=filename)
            grna_merged_df = merge_regression_res_with_metadata(results_path_grna, metadata_file, name=filename)
        except Exception as metadata_error:
            print(f"Could not merge metadata from {metadata_file}: "
                  f"{metadata_error}")
            continue

    # ONE BOOLEAN FOR EVERY OLD VOLCANO. "hide my old version behind a
    # boolean that defaults to off" -- and the first attempt gated exactly one
    # of the three call sites, the one a Toxoplasma screen never reaches.
    # `toxo` defaults to TRUE, so `custom_volcano_plot` below is the picture
    # the maintainer was still being shown after being told it was hidden.
    #
    # Resolved ONCE, here, so the three branches cannot disagree, and read
    # through the same key the fit already reads at the `regression()` call.
    draw_legacy_volcano = bool(settings.get('legacy_volcano', False))
    if not draw_legacy_volcano:
        print("Legacy volcano: off (the interactive volcano and the house-"
              "style figure are drawn instead). Set legacy_volcano=True to "
              "draw the original matplotlib one as well.")

    if _toxoplasma_is_on(settings):
        data_path = merged_df
        data_path_gene = gene_merged_df
        data_path_grna = grna_merged_df
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_dir, 'resources', 'data', 'lopit.csv')
        
        # THE GENE TABLE, ALWAYS. The `volcano` setting used to choose
        # between the merged, gene and gRNA tables here, and it is GONE --
        # "remove the Volcano setting in regression, it is now redundant".
        #
        # It is redundant because 129 A moved that choice onto the plot: the
        # interactive volcano filters to genes or guides by right-click, on
        # the SAME fit, with no re-run. A setting chosen before the run could
        # only ever answer it once.
        #
        # THIS CALL IS NOT ONLY A PICTURE, which is why the branch collapses
        # to `gene` rather than disappearing. `custom_volcano_plot` also
        # RETURNS the hit list that the GT1 phenotype plot and the ME49
        # transcription heatmap are built from, and those are gene-level
        # reports -- so the gene table is the one they need, and it was
        # already this setting's default.
        gene_list = custom_volcano_plot(
            gene_merged_df, metadata_path, metadata_column='tagm_location',
            point_size=600, figsize=20, threshold=reg_threshold,
            # `.get`, NOT `[...]`. Both keys are optional axis limits with a
            # documented None meaning ("auto-scale", and [-0.5, 0.5] for
            # x_lim), and `get_perform_regression_default_settings` does not
            # put either of them in the dict -- so this raised
            # `KeyError: 'x_lim'` from inside the Toxoplasma block, AFTER the
            # fit, every results CSV and every QC panel had been written.
            # A key that is absent and a key that is None mean the same thing
            # to `custom_volcano_plot`, and neither is an error.
            save_path=volcano_path, x_lim=settings.get('x_lim'),
            y_lims=settings.get('y_lims'),
            draw=draw_legacy_volcano,
        )

        # SAY WHERE IT WENT. Every other artifact this module writes announces
        # itself ("Saved regression data to ...", "Plot -> ..."), and the
        # volcano -- the figure the module exists to produce -- was written
        # silently. With nothing naming it, a run that had drawn one perfectly
        # well was indistinguishable from a run that had drawn none, and was
        # reported as "I can't see the regression plot".
        if not draw_legacy_volcano:
            # Nothing was drawn, so nothing is claimed. A stale file left by
            # an EARLIER run sits at this exact path, and reporting it would
            # announce a figure this run did not make.
            pass
        elif os.path.exists(volcano_path):
            print(f"Saved volcano plot to {volcano_path}")
        else:
            print(f"WARNING: the legacy volcano was requested but no file was "
                  f"written to {volcano_path}")

        display(gene_list) if gene_list is not None else None

        phenotype_plot = os.path.join(res_folder, 'phenotype_plot.pdf')
        transcription_heatmap = os.path.join(res_folder, 'transcription_heatmap.pdf')
        # These two OPTIONAL reports need two specific curated tables -- a GT1
        # phenotype table and an ME49 expression table, in that positional
        # order. Indexing [1] and [0] unconditionally meant a run with no
        # metadata files died with `IndexError: list index out of range`
        # AFTER the volcano had been drawn, so the run was reported as failed
        # and the figure it had just produced looked like it was never made.
        # They are extras; missing them is not a failure.
        metadata_files = list(settings.get('metadata_files') or [])
        have_curated_tables = len(metadata_files) >= 2
        if not have_curated_tables:
            print(f"Skipping the phenotype and transcription reports: they "
                  f"need two curated metadata tables (GT1 phenotypes and "
                  f"ME49 expression) and {len(metadata_files)} were given. "
                  f"The volcano and every results table are unaffected.")
        # canonicalise=False: these are curated third-party annotation
        # tables whose headers are the vendor's ('Gene ID', 'sense - EES1'),
        # not spaCR metadata, and the columns below are selected by those
        # exact names.
        data_GT1 = (tabular.read_table(metadata_files[1], low_memory=False,
                                       canonicalise=False, report=None)
                    if have_curated_tables else None)
        data_ME49 = (tabular.read_table(metadata_files[0], low_memory=False,
                                        canonicalise=False, report=None)
                     if have_curated_tables else None)
        columns = ['sense - Tachyzoites', 'sense - Tissue cysts',
                'sense - EES1', 'sense - EES2', 'sense - EES3',
                'sense - EES4', 'sense - EES5']

        # The whole block was duplicated verbatim below this point: the same
        # two reports were built twice, the second copy unguarded, so a run
        # that survived the first died in the second. One copy, guarded.
        if gene_list and have_curated_tables:
            print('Plotting gene phenotypes and heatmaps')
            print(gene_list)
            plot_gene_phenotypes(data=data_GT1, gene_list=gene_list,
                                 save_path=phenotype_plot)
            plot_gene_heatmaps(
                data=data_ME49, gene_list=gene_list, columns=columns,
                x_column='Gene ID', normalize=True,
                save_path=transcription_heatmap,
            )
        elif not gene_list:
            print("No gene_list produced; skipping phenotype and heatmap plots.")

        #if len(significant) > 2:
        #    metadata_path = os.path.join(base_dir, 'resources', 'data', 'toxoplasma_metadata.csv')
        #    go_term_enrichment_by_column(significant, metadata_path)
    
    # A VOLCANO IS NOT A TOXOPLASMA FEATURE.
    #
    # Everything above sits under `if _toxoplasma_is_on(settings)`, because the
    # compartment colouring needs the LOPIT table. But the volcano itself is
    # the figure this module exists to produce, and gating it on an
    # organism-specific flag meant a run with toxo=False wrote sixteen
    # diagnostic figures and NOT the one the user came for -- silently, with
    # nothing saying why. Drawn here without the compartment colouring, which
    # is the only part that ever needed the metadata.
    if not _toxoplasma_is_on(settings) and draw_legacy_volcano:
        try:
            from .plot import volcano_plot as _plain_volcano
            # The gene table, for the same reason as the toxo branch above.
            _source = results_path_gene
            _plain_volcano(
                _source,
                fold_change_col='coefficient',
                p_value_col='p_value',
                name_col='feature',
                x_transform='none', y_transform='-log10',
                fold_change_threshold=reg_threshold,
                p_value_threshold=float(settings.get('fdr_alpha', 0.05) or 0.05),
                point_size=20.0, figsize=(10.0, 8.0),
                title=f"{settings.get('regression_type', 'ols')} - gene",
                save_path=volcano_path, show=False)
        except Exception as _volcano_error:
            print(f"Could not draw the volcano plot: "
                  f"{type(_volcano_error).__name__}: {_volcano_error}")
        if os.path.exists(volcano_path):
            print(f"Saved volcano plot to {volcano_path}")

    print('Significant Genes')
    grnas = significant['grna'].unique().tolist()
    genes = significant['gene'].unique().tolist()
    print(f"Found p<0.05 coedfficients for {len(grnas)} gRNAs and {len(genes)} genes")
    display(significant)

    # A PENALISED FIT THAT FINDS NOTHING IS NOT THE SAME AS NO SIGNAL.
    #
    # ridge's p-values come from calculate_p_values, which divides an
    # unpenalised standard error into a shrunken coefficient: conservative by
    # construction, and deliberately so. On this screen every one of them
    # comes back at q=1.0 -- including the two genes that OLS puts at
    # q=2e-05. A user reading that sees "no hits" and cannot tell it apart
    # from "no effect", which is the one conclusion the number does not
    # support.
    _warn_if_penalised_no_hits(settings, coef_df)

    # WHAT THE VOLCANO CANNOT SHOW.
    #
    # A gene backed by ONE surviving guide and a gene whose guides all agree
    # are the same single dot, and they rank by the same p-value -- but only
    # one of them is independent evidence. On this screen the top of the list
    # is a single-guide gene sitting above two genes with full guide support,
    # so the ordering alone misleads about which hits to follow up.
    try:
        from .guide_concordance import concordance_report
        controls = {}
        for _key, _role in (('positive_control', 'positive'),
                            ('negative_control', 'negative')):
            _value = settings.get(_key)
            if _value not in (None, ''):
                controls[str(_value)] = _role
        print()
        print(concordance_report(
            coef_df, alpha=float(settings.get('fdr_alpha', 0.05) or 0.05),
            controls=controls))
    except Exception as concordance_error:
        print(f"Could not summarise guide support: {concordance_error}")

    # THE MODEL AND THE DESIGN COME BACK TOO.
    #
    # Returning only the coefficient table meant every downstream consumer
    # could report WHAT was significant and nothing about whether the fit
    # deserved to be believed: no R-squared, no residuals to test for
    # heteroscedasticity, no way to count the wells and guides that actually
    # reached the design. A sweep row could say '10 hits' and not whether the
    # run that produced them was well specified.
    #
    # Both are already in scope here; they were simply dropped on the way out.
    output = {'results':coef_df,
              'significant':significant,
              'model': model,
              'model_data': merged_df,
              'regression_type': regression_type,
              'res_folder': res_folder,
              # THE SETTINGS THAT PRODUCED IT, so a caller offering to re-fit
              # the same screen through a different model has the run's own
              # dict rather than a file. The saved copy under settings/ is
              # overwritten by every later run of the same screen, so on a
              # second run it describes the wrong one. Copied, because the
              # caller is a GUI and this dict is still being read here.
              'settings': dict(settings)}

    # THE QC VERDICT, CARRIED OUT OF THE RUN (instruction 115). It was written
    # to disk and nowhere else, so the dict a caller gets back could not say
    # whether the fit it was holding is diagnosable -- and the manifest is the
    # only thing in the run that knows: it carries the per-panel verdict, the
    # WORST of them, and the renderer that drew each one.
    #
    # `.attrs` off the coefficient frame, which is where `regression` put it,
    # and absent rather than None when QC did not run: a key holding None is
    # indistinguishable from a suite that ran and concluded nothing.
    manifest = getattr(coef_df, "attrs", {}).get("qc_manifest")
    if manifest:
        output['qc'] = manifest
        # The key the report writes is `verdict`, with `verdict_level` beside
        # it. Both are lifted, because a caller wants the LEVEL to decide what
        # to show and the verdict itself to say why.
        worst = manifest.get('verdict')
        if worst is not None:
            output['qc_verdict'] = worst
        output['qc_verdict_level'] = manifest.get('verdict_level', 'unknown')

    return output


#: The fixed head of a ``prcfo`` key, in order. The object id is always the
#: LAST token and anything between the two is the timepoint, which is how a
#: five-token and a six-token key are told apart without guessing.
_PRCFO_HEAD = schema.FIELD_KEY_COLUMNS


def _assign_prcfo_parts(df, object_column='objectID'):
    """Split ``prcfo`` into its named components and assign them onto ``df``.

    ``prcfo`` is written by :func:`spacr.utils._map_wells_png` and rebuilt by
    :func:`spacr.utils._split_data`. It has **five** tokens on a plain screen
    (``plate_row_column_field_object``) and **six** on a timelapse
    (``plate_row_column_field_TIME_object``).

    Three places in this module used to spell that as

    .. code-block:: python

        df[['plateID', 'rowID', 'columnID', 'fieldID', 'objectID']] = \\
            df['prcfo'].str.split('_', expand=True)

    which is not a mis-assignment on a timelapse — it is a hard stop. Six
    split columns against five keys makes pandas raise ``ValueError: Columns
    must be same length as key``, so :func:`ml_analysis` threw away a
    completed model at its very last statement (measured on a real 2-well x
    2-field x 3-frame x 3-object database: 36 rows in, fit and permutation
    importance done, then ``ValueError`` at ``ml.py:2517``). The five names
    would *also* have been wrong had it not raised — the fifth token of a
    timelapse key is the timepoint, so ``objectID`` would have held ``'t1'``
    and the object id would have been dropped entirely.

    Splitting the head from the left and the object from the right recovers
    both forms, and the timepoint is kept rather than discarded: it is written
    under whichever spelling ``df`` already uses (``timeID`` canonical,
    ``time_id`` legacy — resolved through :func:`spacr.utils._time_column`),
    defaulting to ``timeID``.

    This doubles as repair-on-read for a scores CSV whose ``objectID`` was
    filled in by a positional guess over a timelapse crop name — the same
    guess :func:`spacr.ml.interperate_vision_model` already refuses to trust —
    because the components are recomputed from ``prcfo`` and overwrite what is
    there.

    :param df: Frame carrying a ``prcfo`` column.
    :param object_column: Name to give the object id. ``'objectID'`` for the
        read/score paths, ``'object'`` in :func:`ml_analysis`, which is what
        each of them already wrote.
    :returns: ``df``, with the component columns assigned.
    :raises TimelapseKeyMismatch: when the frame mixes five- and six-token
        keys — two runs that disagreed about ``timelapse`` were concatenated,
        and there is no single answer to what the fifth token means.
    :raises ValueError: when a key has neither five nor six tokens.
    """
    from .io import TimelapseKeyMismatch
    from .utils import _time_column

    values = df['prcfo']
    tokens = values.astype(str).str.split(schema.KEY_SEPARATOR)
    widths = tokens.map(len)
    seen = set(widths.unique().tolist())

    unexpected = sorted(seen - {5, 6})
    if unexpected:
        example = values[widths.isin(unexpected)].iloc[0]
        raise ValueError(
            f"prcfo must be plate_row_column_field_object (5 tokens) or "
            f"plate_row_column_field_time_object (6, timelapse); found "
            f"{unexpected} token(s), e.g. {example!r}."
        )
    if seen == {5, 6}:
        raise TimelapseKeyMismatch(
            f"prcfo mixes {int((widths == 5).sum())} key(s) without a "
            f"timepoint and {int((widths == 6).sum())} with one, so the fifth "
            f"token is an object id in some rows and a timepoint in others. "
            f"Two runs that disagreed about 'timelapse' have been combined; "
            f"re-run the non-timelapse half rather than splitting this."
        )

    # The width check above answers "do these rows agree with each other?",
    # which schema deliberately does not; schema.parse_prcfo answers "what is
    # this one key?", which this used to do positionally. Parsing right to
    # left is what makes the six-token form safe: the timepoint is optional
    # and in the middle, so counting from the left puts the object id in
    # 'timeID' and drops it.
    parsed = [schema.parse_prcfo(value) for value in values.astype(str)]
    for name in _PRCFO_HEAD:
        df[name] = [getattr(obj, name) for obj in parsed]
    df[object_column] = [obj.objectID for obj in parsed]
    if seen == {6}:
        df[_time_column(df.columns) or schema.TIME_KEY] = [
            obj.timeID for obj in parsed]
    return df


def process_reads(csv_path, fraction_threshold, plate, filter_column=None,
                  filter_value=None, record=None):
    """Load a per-gRNA read-count CSV and return per-well normalised fractions.

    Splits derived ``plate_row`` or ``prcfo`` identifiers, computes each
    gRNA's fraction of the well total, applies an optional
    fraction-cutoff filter and returns a compact ``(prc, grna, fraction)``
    frame (with ``gene`` derived from the gRNA when possible).

    :param csv_path: Path to the counts CSV, or an already-loaded DataFrame.
    :param fraction_threshold: Drop rows below this fraction; must be in
        ``[0, 1]`` or ``None``.
    :param plate: Plate identifier used when no ``plateID`` column is
        present.
    :param filter_column: Column (or list of columns) to filter rows on.
    :param filter_value: Values (or list of values) to drop from
        ``filter_column``.
    :returns: DataFrame with columns ``prc``, ``grna``, ``fraction``.
    :raises ValueError: on missing required columns, invalid
        ``fraction_threshold``, or when the threshold removes all rows.
    """
    from .utils import correct_metadata

    if isinstance(csv_path, pd.DataFrame):
        csv_df = csv_path
    else:
        # Read the CSV file into a DataFrame
        csv_df = tabular.read_table(csv_path)

    csv_df = correct_metadata(csv_df)    
    
    if 'grna_name' in csv_df.columns:
        csv_df = csv_df.rename(columns={'grna_name': 'grna'})
    if 'plate_row' in csv_df.columns:
        # 'plate_row' is '<plate>_<row>'. Split on the LAST separator rather
        # than on every one: the plate is the component that may itself
        # contain a separator ('exp1_plate1_r2'), the row is not, so counting
        # from the right is the only reading that survives it. The two-column
        # positional split this replaces raised the opaque "Columns must be
        # same length as key" on such a plate.
        pieces = csv_df['plate_row'].astype(str).str.rsplit(
            schema.KEY_SEPARATOR, n=1)
        malformed = pieces.map(len) < 2
        if malformed.any():
            example = csv_df.loc[malformed, 'plate_row'].iloc[0]
            raise ValueError(
                f"'plate_row' must be '<plate>{schema.KEY_SEPARATOR}<row>', "
                f"but {int(malformed.sum())} of {len(csv_df)} value(s) hold no "
                f"{schema.KEY_SEPARATOR!r}, e.g. {example!r}. Supply separate "
                f"'plateID' and 'rowID' columns instead, or repair the count "
                f"table — guessing which half is the plate would key the whole "
                f"screen on the wrong well.")
        csv_df['plateID'] = pieces.str[0]
        csv_df['rowID'] = pieces.str[-1]

    if not 'plateID' in csv_df.columns:
        if not plate is None:
            csv_df['plateID'] = plate
        else:
            csv_df['plateID'] = 'plate1'
            
    if 'prcfo' in csv_df.columns:
        #csv_df = csv_df.loc[:, ~csv_df.columns.duplicated()].copy()
        csv_df = _assign_prcfo_parts(csv_df, object_column='objectID')
        csv_df['prc'] = _compose_prc_column(csv_df)

    if isinstance(filter_column, str):
        filter_column = [filter_column]

    if isinstance(filter_value, str):
        filter_value = [filter_value]
            
    if isinstance(filter_column, list):            
        for filter_col in filter_column:
            for value in filter_value:
                csv_df = csv_df.loc[csv_df[filter_col] != value].copy()

    # Ensure the necessary columns are present
    if not all(col in csv_df.columns for col in ['rowID','columnID','grna','count']):
        raise ValueError("The CSV file must contain 'grna', 'count', 'rowID', and 'columnID' columns.")

    # Create the prc column
    csv_df['prc'] = _compose_prc_column(csv_df)

    # Group by prc and calculate the sum of counts
    grouped_df = csv_df.groupby('prc')['count'].sum().reset_index()
    grouped_df = grouped_df.rename(columns={'count': 'total_counts'})
    # grouped_df is one row per well by construction (groupby('prc')), csv_df is
    # one row per (well, gRNA): many-to-one. The contract matters because the
    # very next line divides by total_counts — a duplicated well total would
    # duplicate every gRNA row of that well and the fractions would still sum to
    # 1 per copy, so the corruption would be invisible in every downstream QC.
    merged_df = pd.merge(csv_df, grouped_df, on='prc', validate='many_to_one')
    merged_df['fraction'] = merged_df['count'] / merged_df['total_counts']

    # Filter rows with fraction under the threshold
    #if fraction_threshold is not None:
    #    observations_before = len(merged_df)
    #    merged_df = merged_df[merged_df['fraction'] >= fraction_threshold]
    #    observations_after = len(merged_df)
    #    removed = observations_before - observations_after
    #    print(f'Removed {removed} observation below fraction threshold: {fraction_threshold}')
        
    if fraction_threshold is not None:
        if not 0 <= fraction_threshold <= 1:
            raise ValueError(
                f"fraction_threshold={fraction_threshold} is outside the valid range [0, 1]. "
                f"The 'fraction' column is a relative abundance bounded between 0 and 1."
            )

        observations_before = len(merged_df)
        frac_min = merged_df['fraction'].min()
        frac_max = merged_df['fraction'].max()
        frac_median = merged_df['fraction'].median()

        merged_df = merged_df[merged_df['fraction'] >= fraction_threshold]
        observations_after = len(merged_df)
        removed = observations_before - observations_after
        # RECORDED, NOT ONLY PRINTED (instruction 156). The summary used to
        # say "the run printed how many it removed and did not record it, so
        # the count is in the console log and not in any file this summary can
        # read" -- which was honest and is a gap rather than an answer. A
        # console scrolls; a run somebody asks about tomorrow needs the number.
        # Accumulated because this runs once per plate.
        if record is not None:
            record["fraction_threshold"] = (
                record.get("fraction_threshold", 0) + int(removed))
            record["fraction_threshold_of"] = (
                record.get("fraction_threshold_of", 0) + int(observations_before))

        pct_retained = 100 * observations_after / observations_before if observations_before else 0
        print(
            f"Removed {removed} of {observations_before} observations "
            f"below fraction threshold {fraction_threshold} "
            f"({pct_retained:.1f}% retained). "
            f"Fraction range in input: [{frac_min:.4g}, {frac_max:.4g}], median {frac_median:.4g}."
        )

        if observations_after == 0:
            raise ValueError(
                f"All {observations_before} rows were removed by fraction_threshold={fraction_threshold}. "
                f"Observed fraction range was [{frac_min:.4g}, {frac_max:.4g}], median {frac_median:.4g}. "
                f"Choose a threshold below the median, or pass None to auto-compute."
            )

    merged_df = merged_df[['prc', 'grna', 'fraction']]

    # This split IS positional, legitimately: the pooled-library naming
    # convention is '<org>_<gene>_<guide>' ('TGGT1_GENEA_g1') and there is
    # nothing in the name itself that says which token is which. So the
    # assumption is stated and checked rather than removed.
    #
    # What is removed is the bare `except Exception`. It made two very
    # different inputs look identical from the outside:
    #   * every name a single token ('g0', 'g1') — a library that simply
    #     has no org/gene structure. Three keys against one split column
    #     raised, and skipping is the right answer.
    #   * names of mixed width ('TGGT1_GENEA_g1' next to 'GENEA_g1').
    #     str.split(expand=True) pads with None instead of raising, so a
    #     short name got its GUIDE token as its gene and then grna=None
    #     out of the gene + '_' + guide concatenation — its reads were
    #     silently deleted from the screen while every long name sailed
    #     through.
    # Requiring every name to have the same three components refuses the
    # second case outright instead of half-applying to it.
    tokens = merged_df['grna'].astype(str).str.split(schema.KEY_SEPARATOR)
    widths = sorted(set(tokens.map(len).tolist()))
    if widths == [3]:
        merged_df['gene'] = tokens.str[1]
        merged_df['grna'] = (tokens.str[1] + schema.KEY_SEPARATOR
                             + tokens.str[2])
    else:
        example = merged_df['grna'].iloc[0] if len(merged_df) else None
        print(f"Not splitting 'grna' into org/gene/grna: that split is "
              f"positional and needs every name to be "
              f"'<org>{schema.KEY_SEPARATOR}<gene>{schema.KEY_SEPARATOR}"
              f"<guide>' (3 components), but this table holds names with "
              f"{widths} component(s), e.g. {example!r}. No 'gene' column "
              f"is produced; a step that needs one will name it.")

    return merged_df

#: The squeeze applied before a logit, and the reason it exists.
#:
#: A classification score is a PROPORTION, and a screen produces exact 0 and
#: exact 1 -- neither of which has a logit. Smithson and Verkuilen's transform
#: pulls the whole scale off the endpoints by (n-1)/n plus a half, which is
#: the standard treatment and is reported in the run summary rather than
#: applied quietly: a transform that silently moved a user's 0 to 0.001
#: changed their data.
BETA_SQUEEZE_NOTE = (
    "beta: the response was mapped to the logit scale. A proportion of "
    "exactly 0 or 1 has no logit, so the scale was squeezed off its "
    "endpoints by the Smithson-Verkuilen rule ((y*(n-1)+0.5)/n) first")


def beta_logit(values):
    """A proportion on the logit scale, with the endpoints squeezed in.

    ``transform='beta'`` is intended for proportional responses such as
    classification scores and their well aggregates, where a logarithm is
    not appropriate.

    This is distinct from ``regression_type='beta'``, which selects a beta
    GLM. One transforms the response; the other selects the model family.
    """
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    n = int(finite.sum())
    if n < 1:
        return array
    squeezed = array.copy()
    # Only squeeze when an endpoint is actually present: a response already
    # inside (0, 1) is left exactly as the user measured it.
    inside = array[finite]
    if inside.min() <= 0.0 or inside.max() >= 1.0:
        squeezed[finite] = (inside * (n - 1) + 0.5) / n
    squeezed[finite] = np.clip(squeezed[finite], 1e-9, 1.0 - 1e-9)
    out = np.array(array, dtype=float, copy=True)
    out[finite] = np.log(squeezed[finite] / (1.0 - squeezed[finite]))
    return out


def apply_transformation(X, transform):
    """Return an sklearn ``FunctionTransformer`` for the named transform.

    :param X: Ignored (kept for compatibility with sklearn pipeline flow).
    :param transform: One of ``'log'``, ``'sqrt'``, ``'square'``, ``'beta'``.
        Any other value returns ``None``.
    :returns: A ``FunctionTransformer`` or ``None``.
    """
    if transform == 'log':
        transformer = FunctionTransformer(np.log1p, validate=True)
    elif transform == 'sqrt':
        transformer = FunctionTransformer(np.sqrt, validate=True)
    elif transform == 'square':
        transformer = FunctionTransformer(np.square, validate=True)
    elif transform == 'beta':
        transformer = FunctionTransformer(beta_logit, validate=True)
    else:
        transformer = None
    return transformer

def check_normality(data, variable_name, verbose=False):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        if verbose:
            print(f"Shapiro-Wilk Test for {variable_name}: at least 3 finite "
                  f"values are required; received {values.size}.")
        return False
    stat, p_value = shapiro(values)
    if verbose:
        print(f"Shapiro-Wilk Test for {variable_name}:\nStatistic: {stat}, P-value: {p_value}")
    if p_value > 0.05:
        if verbose:
            print(f"Normal distribution: The data for {variable_name} is normally distributed.")
        return True
    else:
        if verbose:
            print(f"Normal distribution: The data for {variable_name} is not normally distributed.")
        return False

def clean_controls(df,values, column):
    """Drop rows whose ``column`` holds one of the listed ``values``.

    :param df: Source DataFrame.
    :param values: List of values to remove. Anything that is not a list
        (a bare value included) is a no-op.
    :param column: Column, or list of columns, to check. ``None`` is a
        no-op.
    :returns: Filtered DataFrame (unchanged if ``column`` is missing or
        ``values`` is not a list).
    """
    if column is None:
        return df
    # A bare `column in df.columns` raised "TypeError: unhashable type: 'list'"
    # for the list form that process_reads accepts and documents. Anything
    # that is not a sequence of names stays a single name, as before.
    columns = list(column) if isinstance(column, (list, tuple, set)) else [column]
    if isinstance(values, list):
        for col in columns:
            if col in df.columns:
                for value in values:
                    df = df[~df[col].isin([value])]
                    print(f'Removed data from {value}')
    return df

def process_scores(df, dependent_variable, plate, min_cell_count=25, agg_type='mean', transform=None, regression_type='ols', invert_dependent_variable=False):
    """Aggregate per-object model scores to per-well summaries, ready for regression.

    Ensures ``plateID/rowID/columnID/prc`` columns exist, applies an
    optional inversion of the raw response, aggregates by well according
    to ``agg_type`` (or with ``sum`` for the count models
    ``'poisson'`` and ``'horseshoe'``), enforces
    ``min_cell_count`` and optionally transforms the aggregated response.

    :param df: Per-object score DataFrame.
    :param dependent_variable: Column being aggregated.
    :param plate: Plate identifier to stamp when the frame is
        single-plate; ignored (with warning) when multiple plates exist.
    :param min_cell_count: Wells with fewer objects are dropped.
        Default ``25``.
    :param agg_type: ``'mean'``, ``'median'``, ``'quantile'`` or None.
    :param transform: Optional post-aggregation transform name
        (see :func:`apply_transformation`).
    :param regression_type: If ``'poisson'`` or ``'horseshoe'``, aggregation
        uses ``sum`` - both model a per-well count, not a per-well average.
    :param invert_dependent_variable: ``False``/``0`` = no inversion;
        ``True``/``1`` = ``1 - x``; ``-1`` = ``1 / x``.
    :returns: ``(dependent_df, dependent_variable)`` — the per-well
        DataFrame and the (possibly transformed) response column name.
    :raises ValueError: on missing identifiers, unsupported ``agg_type``
        or unrecognised ``invert_dependent_variable``.
    """
    from .utils import correct_metadata
    df = df.reset_index(drop=True)
    if 'prcfo' in df.columns:
        df = df.loc[:, ~df.columns.duplicated()].copy()
        if not all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df = _assign_prcfo_parts(df, object_column='objectID')
        df['prc'] = _compose_prc_column(df)
    else:
        df = correct_metadata(df)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Only stamp a single plateID on every row when the caller asked for it AND
        # the frame is single-plate (or has no plateID at all). For a multi-plate
        # frame, ignore 'plate' so wells from different plates do not get collapsed
        # to the same prc and silently averaged together by the groupby below.
        n_plates_in_df = df['plateID'].nunique(dropna=True) if 'plateID' in df.columns else 0

        if plate is not None:
            if n_plates_in_df > 1:
                print(f"Warning: process_scores received plate={plate!r} but the input "
                      f"DataFrame already contains {n_plates_in_df} distinct plateIDs. "
                      f"Ignoring the 'plate' argument and using the per-row plateID "
                      f"column to avoid collapsing plates.")
            else:
                df['plateID'] = plate

        if 'plateID' not in df.columns or df['plateID'].isna().all():
            raise ValueError(
                "process_scores: DataFrame has no usable 'plateID' column "
                "and no 'plate' argument was provided."
            )

        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df['prc'] = _compose_prc_column(df)
        else:
            raise ValueError("The DataFrame must contain 'plateID', 'rowID', and 'columnID' columns.")

    df = df[['prc', dependent_variable]]
    
    df = df[['prc', dependent_variable]].copy()

    # Optional inversion of the raw dependent variable, applied before
    # aggregation and before any transform.
    #   False / 0 : no inversion
    #   True  / 1 : x -> 1 - x   (complement; for probability / score in [0, 1])
    #   -1        : x -> 1 / x   (reciprocal; for rate- or time-like quantities)
    if invert_dependent_variable in (True, 1):
        df[dependent_variable] = 1.0 - df[dependent_variable]
        print(f"Inverted '{dependent_variable}' as 1 - x on raw values.")
    elif invert_dependent_variable == -1:
        raw = df[dependent_variable]
        n_zero = int((raw == 0).sum())
        if n_zero > 0:
            print(f"Warning: '{dependent_variable}' contains {n_zero} zero "
                  f"values; 1/x is undefined for those rows. They will be set "
                  f"to NaN and dropped from this analysis.")
        df[dependent_variable] = 1.0 / raw.where(raw != 0)
        df = df.dropna(subset=[dependent_variable])
        print(f"Inverted '{dependent_variable}' as 1/x on raw values.")
    elif invert_dependent_variable in (False, 0):
        pass
    else:
        raise ValueError(
            f"invert_dependent_variable must be one of False, True, 1, -1; "
            f"got {invert_dependent_variable!r}."
        )

    # Group by prc and calculate the mean and count of the dependent_variable
    grouped = df.groupby('prc')[dependent_variable]

    # Both count models take the well's SUM. 'horseshoe' is spaCRPower's
    # Npositive ~ ... + offset(log(Ntotal)): its response is the number of
    # positive objects in the well, not their mean, and the exposure it is
    # offset by is the cell_count computed just below. Aggregating it like a
    # continuous score would hand a Poisson model a fraction, which
    # _validate_poisson_response refuses - loudly, but at the very end.
    count_models = ('poisson', 'horseshoe')

    if regression_type not in count_models:

        print(f'Using agg_type: {agg_type}')

        if agg_type == 'median':
            dependent_df = grouped.median().reset_index()
        elif agg_type == 'mean':
            dependent_df = grouped.mean().reset_index()
        elif agg_type == 'quantile':
            dependent_df = grouped.quantile(0.75).reset_index()
        elif agg_type is None:
            dependent_df = df.reset_index()
            if 'prcfo' in dependent_df.columns:
                dependent_df = dependent_df.drop(columns=['prcfo'])
        else:
            raise ValueError(f"Unsupported aggregation type {agg_type}")

    if regression_type in count_models:
        agg_type = 'count'
        print(f'Using agg_type: {agg_type} for {regression_type} regression')
        dependent_df = grouped.sum().reset_index()

        # REFUSED HERE, NOT AT THE END OF THE FIT. The comment above already
        # says a continuous score hands a Poisson model a fraction and that
        # `_validate_poisson_response` refuses it "loudly, but at the very
        # end" -- and the end is after both input CSVs are read, the QC
        # tables written and the diagnostic plots drawn. Measured on the
        # maintainer's own run 2026-08-19: 19.2 seconds to be told that two
        # settings could not both be honoured, which was checkable the
        # moment the response was summed.
        #
        # The well SUM is what these models fit, so that is the number to
        # judge: a per-well sum of counts is an integer, and a per-well sum
        # of a classification score is not.
        summed = pd.to_numeric(dependent_df.get(dependent_variable),
                               errors='coerce')
        if summed is not None and len(summed):
            finite = summed[np.isfinite(summed)]
            if len(finite) and not np.all(
                    np.isclose(finite, np.rint(finite), rtol=0, atol=1e-8)):
                # THE MESSAGE NAMES THE CAUSE, not the symptom. The one
                # this replaces said "requires integer count data; use a
                # continuous response model for fractional values" -- true,
                # and it left the reader to work out WHY their counts were
                # fractional. They are fractional because these models take
                # the well's POSITIVE COUNT as the sum of a per-cell 0/1
                # label, and a classification SCORE is a probability: summing
                # 152 cells at ~0.14 gives 21.68, which is not a count of
                # anything.
                example = float(finite.iloc[0])
                raise ValueError(
                    f"regression_type={regression_type!r} models the well's "
                    f"positive COUNT -- the number of cells called positive "
                    f"-- and gets it by summing {dependent_variable!r} per "
                    f"well. That column holds continuous scores, so the sum "
                    f"is {example:.4g} rather than a whole number of cells. "
                    f"Either fit a continuous model ('ols', 'mixed', or "
                    f"'beta', which is built for a proportion), or give "
                    f"dependent_variable a per-cell 0/1 label so its "
                    f"per-well sum is a real count.")

    # Calculate cell_count for all cases
    cell_count = grouped.size().reset_index(name='cell_count')

    if agg_type is None:
        # No aggregation, so dependent_df is still one row per object and
        # cell_count is one row per well: many-to-one. Stating it pins the
        # thing that makes the unaggregated path safe — the well's cell count
        # is broadcast onto its objects, never the other way round.
        dependent_df = pd.merge(dependent_df, cell_count, on='prc',
                                validate='many_to_one')
    else:
        dependent_df['cell_count'] = cell_count['cell_count']

    print("1 test")
    display(dependent_df)

    dependent_df = dependent_df[dependent_df['cell_count'] >= min_cell_count]

    print("2 test")
    display(dependent_df)

    is_normal = check_normality(dependent_df[dependent_variable], dependent_variable)

    # A COUNT MODEL'S RESPONSE MUST STAY A COUNT.
    #
    # The sum above is deliberate -- Poisson and horseshoe model the number of
    # positive objects in a well, not their average -- and then a transform
    # was applied to it anyway. The default transform is 'log', so the integer
    # count left here as a float and _validate_poisson_response refused it at
    # the very END of a run that had already read both CSVs and fitted
    # nothing. Neither count family could be started at all.
    #
    # settings.py now clears `transform` for these families before the run, so
    # this is the second line of defence -- and it is the one that covers a
    # direct regression() or process_scores() call, which does not pass
    # through the settings layer at all.
    if transform is not None and regression_type in count_models:
        print(f"Ignoring transform={transform!r}: {regression_type} models a "
              f"per-well count, and a transformed count is not a count.")
        transform = None

    if transform == 'beta':
        # A logit is only defined on a proportion. Saying so BEFORE the fit
        # is the difference between a wrong number and a stopped run: a
        # response in raw intensity units transformed this way produces
        # coefficients that look ordinary and mean nothing.
        column = pd.to_numeric(dependent_df[dependent_variable],
                               errors='coerce')
        inside = column[np.isfinite(column)]
        if len(inside) and (inside.min() < 0.0 or inside.max() > 1.0):
            raise ValueError(
                f"transform='beta' puts the response on the logit scale, "
                f"which is only defined for a proportion, but "
                f"{dependent_variable!r} runs from {inside.min():.4g} to "
                f"{inside.max():.4g}. Use a score or a fraction here, or "
                f"pick transform='log' for a response in measured units.")
        print(BETA_SQUEEZE_NOTE)

    if transform is not None:
        transformer = apply_transformation(dependent_df[dependent_variable], transform=transform)
        transformed_var = f'{transform}_{dependent_variable}'
        dependent_df[transformed_var] = transformer.fit_transform(dependent_df[[dependent_variable]])
        dependent_variable = transformed_var
        is_normal = check_normality(dependent_df[transformed_var], transformed_var)

    if not is_normal:
        print(f'{dependent_variable} is not normally distributed')
    else:
        print(f'{dependent_variable} is normally distributed')

    return dependent_df, dependent_variable


@single_threaded_openmp('classical ML training')
@_flowview_pipeline("ml")
def generate_ml_scores(settings):
    """Train a classical ML classifier (XGBoost / logistic / RF) on per-object features and score every well of a screen.

    Reads the ``measurements.db`` produced by
    :func:`spacr.measure.measure_crop`, merges cell/nucleus/pathogen/
    cytoplasm feature tables, uses the wells marked as
    ``positive_control`` / ``negative_control`` (or an annotation column)
    as training labels, delegates fitting to :func:`ml_analysis`, and
    writes per-object predictions, permutation and feature-importance
    tables plus a plate heatmap into ``results/`` next to the source DB.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.set_default_analyze_screen`. Key entries:

        - ``src`` (str or list) — folder(s) containing
          ``measurements/measurements.db``.
        - ``channel_of_interest`` — 0-based channel for the recruitment
          ratio feature; also drives table selection.
        - ``model_type_ml`` — ``'xgboost'``, ``'logistic_regression'``,
          ``'random_forest'``.
        - ``positive_control`` / ``negative_control`` — well IDs (e.g.
          ``'c2'`` / ``'c1'``) used as training labels.
        - ``annotation_column`` — override controls with a PNG-level
          annotation column.
        - ``location_column`` — ``'columnID'`` or ``'rowID'``.
        - ``heatmap_feature`` — feature plotted on the plate heatmap.
        - ``exclude``, ``n_repeats``, ``top_features``, ``test_size``,
          ``reg_alpha``, ``reg_lambda``, ``learning_rate``,
          ``n_estimators``, ``n_jobs``.
        - ``remove_low_variance_features``,
          ``remove_highly_correlated_features``, ``prune_features``,
          ``cross_validation``, ``verbose``.

    :returns: The two-element list ``[output, plate_heatmap]``, where
        ``output`` is the 10-element result list of :func:`ml_analysis`
        and ``plate_heatmap`` is the plate-heatmap ``matplotlib``
        figure. The CSVs and figures are written to ``results/`` as a
        side effect; their paths are not returned.
    :raises ValueError: if ``annotation_column`` is set but the
        ``png_list`` table lacks ``prcfo`` / that column, its object IDs do
        not join to the measurements, it contains fewer than two observed
        classes, or if ``heatmap_feature`` is not among the trained features.

    Example:
        .. code-block:: python

            from spacr.ml import generate_ml_scores
            settings = {
                'src': '/data/plate01',
                'channel_of_interest': 3,
                'positive_control': 'c2', 'negative_control': 'c1',
                'model_type_ml': 'xgboost', 'heatmap_feature': 'recruitment',
            }
            generate_ml_scores(settings)

    See Also:
        :func:`ml_analysis` — the underlying fit/evaluate routine.
        :func:`perform_regression` — mixed-effects regression on
        per-well ML scores.
    """
    from .io import _read_and_merge_data, _read_db
    from .plot import plot_plates
    from .utils import get_ml_results_paths, calculate_shortest_distance, save_settings
    from .settings import set_default_analyze_screen
    from .predictions import (ML_CLASS_COLUMN, merge_ml_predictions,
                              migrate_prediction_columns)

    settings = set_default_analyze_screen(settings)
    save_settings(settings, name='generate_ml_scores', show=True)
    _flowview_advance("tables")

    srcs = settings['src']
    
    if isinstance(srcs, str):
        srcs = [srcs]
    
    df = pd.DataFrame()
    for idx, src in enumerate(srcs):
        
        if idx == 0:
            src1 = src

        db_loc = [src+'/measurements/measurements.db']
        tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
        
        dft, _ = _read_and_merge_data(db_loc, 
                                    tables,
                                    settings['verbose'],
                                    nuclei_limit=settings['nuclei_limit'],
                                    pathogen_limit=settings['pathogen_limit'])
        df = pd.concat([df, dft])

    _flowview_metric("objects", len(df))
    _flowview_metric("databases", len(srcs))
    _flowview_metric("tables", len(tables) * len(srcs))
    
    try:
        df = calculate_shortest_distance(df, 'pathogen', 'nucleus')
    except Exception as e:
        print(e)
    
    # The basis is now EXPLICIT. This used to read "if annotation_column is
    # not None", which meant filling in an annotation column silently stopped
    # the module training on plate controls, with nothing in the settings
    # panel saying so. `resolve_basis` keeps that old rule as the fallback
    # for a settings CSV with no `dataset_mode`, so an existing project runs
    # exactly as it did -- see spacr.training_basis.
    from .training_basis import resolve_basis
    _basis = resolve_basis(settings)

    #: The column the annotation path trains against. None on the metadata
    #: path, where the caller's own `location_column` is the answer. Declared
    #: here so every branch below has it defined.
    _label_column = None

    if _basis == 'annotation':
        if not settings.get('annotation_column'):
            raise ValueError(
                "dataset_mode='annotation' needs annotation_column set to a "
                "column of png_list. Nothing else in these settings says "
                "which labels to train on.")

        # DERIVED, NOT WRITTEN BACK. This used to be
        #     settings['location_column'] = settings['annotation_column']
        # and that assignment mutated the CALLER'S settings dict -- a
        # user-facing value, shown in the panel and saved with the project.
        #
        # The mutation outlived the run. A user who tried annotation mode
        # once and then switched dataset_mode back to 'metadata' still had
        # `location_column` naming their annotation column, which is not in
        # the measurement frame, so the next run died at `df[[location_
        # column]]` with a pandas KeyError that pointed nowhere near the
        # cause. They could not get out by changing the mode; they had to
        # know an invisible write had happened and undo it by hand.
        # (Issues #91, #92, #93 -- one defect, walked through in sequence.)
        _label_column = settings['annotation_column']

        # Repair-on-read, the same contract utils.rename_columns_in_db has:
        # a database written before the prediction columns were namespaced
        # still carries the ML stage's scores under 'predictions', and is
        # migrated here so the caller never has to do anything by hand. Skipped
        # when the current name already exists.
        migrate_prediction_columns(db_loc[0])
        png_list_df = _read_db(db_loc[0], tables=['png_list'])[0]
        if not {'prcfo', settings['annotation_column']}.issubset(png_list_df.columns):
            raise ValueError("The 'png_list_df' DataFrame must contain 'prcfo' and 'test' columns.")
        annotated_df = png_list_df[['prcfo', settings['annotation_column']]].set_index('prcfo')
        # png_list can legitimately hold more than one crop per object — a
        # database measured twice (cell crops, then pathogen crops) appends to
        # the same table — so the annotation side is 'many'. The measurement
        # side must not be: _read_and_merge_data groups on prcfo, so a repeat
        # there means two source directories were concatenated under the same
        # plate id and the same object identity now describes two different
        # objects. That has to stop here rather than double every measurement
        # row and quietly double the training set.
        measurement_rows = len(df)
        annotation_rows = len(annotated_df)
        df = annotated_df.merge(df, left_index=True, right_index=True,
                                validate='many_to_one')
        if df.empty:
            raise ValueError(
                f"annotation_column={settings['annotation_column']!r} joined "
                f"to 0 measured objects by 'prcfo' ({annotation_rows} "
                f"annotation rows; {measurement_rows} measurement rows), so "
                f"there is no training data. Verify that png_list and the "
                f"measurement tables come from the same source and use the "
                f"same object identities.")
        unique_values = df[settings['annotation_column']].dropna().unique()
        print(f"Unique values in annotation column: {unique_values}")

        # A BINARY CLASSIFIER NEEDS TWO OBSERVED CLASSES. The former one-class
        # fallback randomly labelled unannotated objects as a made-up second
        # class. That made the split run, but it changed unknown samples into
        # ground truth and made every downstream metric scientifically false.
        # Unannotated rows remain available for scoring after a real two-class
        # model is trained; they are never promoted into training examples.
        if len(unique_values) < 2:
            labelled_rows = int(
                df[settings['annotation_column']].notna().sum())
            if not len(unique_values):
                state = (f"has 0 non-empty labels across {len(df)} joined "
                         f"object rows")
            else:
                state = (f"has only one observed class across "
                         f"{labelled_rows} labelled object rows")
            raise ValueError(
                f"annotation_column={settings['annotation_column']!r} "
                f"{state}; binary ML training requires two real annotated "
                f"classes. Annotate objects in a second class, or choose the "
                f"annotation column that already contains both classes. "
                f"Unannotated objects will be scored after training; spaCR "
                f"will not assign them a training label.")
            
        if settings['positive_control'] is None and settings['negative_control'] is None:
            settings['positive_control'] = str(unique_values[0])
            settings['negative_control'] = str(unique_values[1])
            print(f"Automatically set positive control to {settings['positive_control']} and negative control to {settings['negative_control']} based on unique values in annotation column.")
    
    _flowview_advance("dataset")

    # RECRUITMENT NEEDS EXACTLY ONE CHANNEL, and the setting can now name
    # several, or a shape group, or nothing. `feature_selection` returns a
    # bare int only for the one-channel case -- which is the only case in
    # which "the pathogen's intensity over the cytoplasm's" names a number.
    #
    # It used to read `settings['channel_of_interest'] in [0,1,2,3]`, so the
    # panel's multi-select answer `[3]` -- the same feature space as the old
    # `3` -- would have skipped recruitment silently.
    from .utils import feature_selection

    recruitment_channel = feature_selection(settings['channel_of_interest'])
    if isinstance(recruitment_channel, int):
        # `if "a" and "b" in df.columns` only membership-tests "b": the first
        # operand is a non-empty literal and therefore always truthy. A
        # measurements DB whose pathogen table lacks the channel mean
        # intensity died with KeyError instead of skipping recruitment.
        pathogen_col = f"pathogen_channel_{recruitment_channel}_mean_intensity"
        cytoplasm_col = f"cytoplasm_channel_{recruitment_channel}_mean_intensity"
        if pathogen_col in df.columns and cytoplasm_col in df.columns:
            df['recruitment'] = df[pathogen_col]/df[cytoplasm_col]
    
    from .batch_correction import correction_kwargs
    batch_kwargs = correction_kwargs(
        settings,
        default_control_column=(_label_column
                                or settings.get('location_column')),
        default_control_values=settings.get('negative_control'),
    )
    # Added here rather than in `correction_kwargs` — see the note at its
    # other call site. `ml_analysis` grew both parameters; the helper's
    # other consumers did not.
    batch_kwargs['batch_covariate_column'] = settings.get(
        'batch_covariate_column')
    batch_kwargs['batch_combat_mean_only'] = bool(
        settings.get('batch_combat_mean_only', False))
    # `_label_column` is set only on the annotation path and is what that
    # path trains against; metadata runs use the caller's own setting. Either
    # way `settings['location_column']` is left exactly as the user wrote it.
    _training_column = _label_column or settings['location_column']
    output, figs = ml_analysis(df,
                               settings['channel_of_interest'],
                               _training_column,
                               settings['positive_control'],
                               settings['negative_control'],
                               settings['exclude'],
                               settings['n_repeats'],
                               settings['top_features'],
                               settings['reg_alpha'],
                               settings['reg_lambda'],
                               settings['learning_rate'],                               
                               settings['n_estimators'],
                               settings['test_size'],
                               settings['model_type_ml'],
                               settings['n_jobs'],
                               settings['remove_low_variance_features'],
                               settings['remove_highly_correlated_features'],
                               settings['prune_features'],
                               settings['cross_validation'],
                               settings['verbose'],
                               split_by=settings.get('cv_group_by', 'well'),
                               holdout_plate=settings.get('holdout_plate'),
                               **batch_kwargs)
    
    shap_fig = shap_analysis(output[3], output[4], output[5])

    features = output[0].select_dtypes(include=[np.number]).columns.tolist()
    train_features_df = pd.DataFrame(output[9], columns=['feature'])
    
    if not settings['heatmap_feature'] in features:
        raise ValueError(f"Variable {settings['heatmap_feature']} not found in the dataframe. Please choose one of the following: {features}")
    
    plate_heatmap = plot_plates(df=output[0],
                                variable=settings['heatmap_feature'],
                                grouping=settings['grouping'],
                                min_max=settings['min_max'],
                                cmap=settings['cmap'],
                                min_count=settings['min_cell_count'],
                                verbose=settings['verbose'])

    data_path, permutation_path, feature_importance_path, model_metricks_path, permutation_fig_path, feature_importance_fig_path, shap_fig_path, plate_heatmap_path, settings_csv, ml_features = get_ml_results_paths(src1, settings['model_type_ml'], settings['channel_of_interest'])
    df, permutation_df, feature_importance_df, _, _, _, _, _, metrics_df, _ = output

    #settings_df.to_csv(settings_csv, index=False)
    _flowview_metric("objects", len(output[0]))
    _flowview_metric("test_objects", len(output[5]))
    _flowview_advance("scores")
    df.to_csv(data_path, mode='w', encoding='utf-8')
    permutation_df.to_csv(permutation_path, mode='w', encoding='utf-8')
    feature_importance_df.to_csv(feature_importance_path, mode='w', encoding='utf-8')
    train_features_df.to_csv(ml_features, mode='w', encoding='utf-8')
    metrics_df.to_csv(model_metricks_path, mode='w', encoding='utf-8')

    # PUBLISHED, not merely saved -- instruction 139 C. `plot_permutation`,
    # `plot_feature_importance` and `shap_analysis` all RETURN a figure and
    # none of them shows it, and `shap_analysis` closes its own, so these four
    # were written to the results folder and then never seen again by anybody
    # running the app. `publish` writes through `spacr.plot.save_figure`
    # exactly as before -- same file, same format preference -- and announces
    # the figure as part of the same event.
    #
    # The plate heatmap is the one that can arrive twice: `plot_plates` shows
    # it itself when `verbose` is on. The bridge de-duplicates by figure, so
    # it is one tile either way.
    #
    # A FIGURE THAT WAS NEVER DRAWN IS NOT PUBLISHED, and `publish` is where
    # that is decided. `ml_analysis` returns ``feature_importance_fig = None``
    # for every model without ``feature_importances_`` -- logistic regression
    # and HistGradientBoostingClassifier, two of the offered `model_type_ml`
    # values -- and the old `save_figure(figs[1], ...)` went straight into
    # ``None.savefig`` and took the whole scoring run down AFTER the model had
    # been fitted and every object scored.
    from .figure_sink import publish

    plate_heatmap_path = publish(plate_heatmap, plate_heatmap_path)
    permutation_fig_path = publish(figs[0], permutation_fig_path)
    feature_importance_fig_path = publish(
        figs[1], feature_importance_fig_path)
    shap_fig_path = write_plot(shap_fig, shap_fig_path, "SHAP summary")

    # The model scored every object in every source database, so the scores
    # belong back on every one of those databases -- not only in a CSV, and not
    # only when a flag is set. The Annotate app, the active-learning queue and
    # every GUI table read png_list, so a score that stops at results.csv is a
    # score nothing downstream can see.
    #
    # This replaces utils.add_column_to_database, which had three problems for
    # this use: it re-read the CSV that was just written, it appended
    # 'predictions_1', 'predictions_2', ... on every re-run instead of updating
    # in place, and it replaced every 0 with a 2 (the Annotate app's class
    # encoding) so the database disagreed with the CSV from the same run.
    # merge_ml_predictions writes 'predictions' (the class, same column name as
    # before) plus the new 'ml_pred' (the positive-class probability, which the
    # ML stage never stored at all). Neither collides with the CV stage's
    # 'cv_predictions' / 'pred', so running Classify (CV) and Classify (ML)
    # over one database leaves four readable columns rather than two
    # overwritten ones.
    settings['csv_path'] = data_path
    settings['db_path'] = os.path.join(src1, 'measurements', 'measurements.db')
    settings['table_name'] = 'png_list'
    settings['update_column'] = ML_CLASS_COLUMN
    settings['match_column'] = 'prcfo'
    matched_objects = 0
    unmatched_objects = 0
    for src in srcs:
        report = merge_ml_predictions(
            df,
            os.path.join(src, 'measurements', 'measurements.db'),
            table=settings['table_name'],
        )
        if report is not None:
            matched_objects += report.matched_rows
            unmatched_objects += report.unmatched_db_rows
    _flowview_metric("objects", len(df))
    _flowview_metric("matched_objects", matched_objects)
    _flowview_metric("unmatched_objects", unmatched_objects)
    _flowview_metric("databases", len(srcs))

    return [output, plate_heatmap]

def _resolve_controls(df, location_column, negative_control,
                      positive_control, matches):
    """The control values to match, and whether they had to be derived.

    :param matches: ``(series, control) -> boolean mask``. Passed in rather
        than imported because the matcher is defined inside `ml_analysis`;
        taking it as an argument keeps this function module-level and
        testable on its own.
    :returns: ``(negative, positive, derived)``. ``derived`` is True when the
        named controls matched nothing and the column's own two classes were
        used instead.

    THE CASE THIS EXISTS FOR: annotation mode points `location_column` at the
    annotation column, whose values are class labels, while the control
    settings still hold plate column names from the metadata path. Neither
    matches, and the user is told to "set positive_control and
    negative_control to values that appear there" -- for a column that
    already says, unambiguously, what its two classes are.

    Nothing is derived when the named controls DO match: an explicit choice
    is always honoured, including a deliberate two-of-five subset.
    """
    if location_column not in df.columns:
        return negative_control, positive_control, False
    column = df[location_column]
    if isinstance(column, pd.DataFrame):
        return negative_control, positive_control, False

    # NEITHER may match before anything is derived. If ONE does, the user
    # has a real partial match -- 'c1' present and 'c2' mistyped, say -- and
    # deriving would silently replace the control they got RIGHT along with
    # the one they got wrong. The refusal downstream names only the missing
    # one, which is the useful message; overriding both would hide it.
    any_found = (matches(column, negative_control).any()
                 or matches(column, positive_control).any())
    if any_found:
        return negative_control, positive_control, False

    present = sorted(v for v in column.dropna().unique())
    if len(present) != 2:
        # Three or more classes, or one: the user has to say which two, and
        # the refusal below will list what is there.
        return negative_control, positive_control, False

    low, high = present
    print(f"{location_column!r} holds exactly two classes, {low!r} and "
          f"{high!r}, and neither {negative_control!r} nor "
          f"{positive_control!r} appears in it. Training on the column's own "
          f"classes: negative={low!r}, positive={high!r}.")
    return low, high, True


@single_threaded_openmp('classical ML training')
def ml_analysis(
    df,
    channel_of_interest=3,
    location_column='columnID',
    positive_control='c2',
    negative_control='c1',
    exclude=None,
    n_repeats=10,
    top_features=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    learning_rate=0.00001,
    n_estimators=1000,
    test_size=0.2,
    model_type='xgboost',
    n_jobs=-1,
    remove_low_variance_features=True,
    remove_highly_correlated_features=True,
    prune_features=False,
    cross_validation=False,
    verbose=False,
    *,
    split_by='well',
    holdout_plate=None,
    batch_correction='none',
    batch_column='plateID',
    batch_control_column=None,
    batch_control_values=None,
    batch_covariate_column=None,
    batch_combat_mean_only=False,
    batch_min_samples=3,
    batch_missing_control='error',
):
    """Train a per-object classifier on positive/negative control wells and score every row of the input DataFrame.

    Called directly for one-off ML work, and internally by
    :func:`generate_ml_scores`. Filters features by channel, drops
    low-variance and highly correlated columns, splits (or CVs) train /
    test, fits the requested model, computes permutation and native
    feature importances, tunes an optimal decision threshold and writes
    predictions + probabilities back onto the returned DataFrame.

    :param df: Per-object feature DataFrame as produced by merging the
        cell/nucleus/pathogen/cytoplasm tables of a
        :func:`spacr.measure.measure_crop` database.
    :param channel_of_interest: Channel index used to select features.
    :param location_column: Column identifying wells / plate columns.
        Default ``'columnID'``.
    :param positive_control: Value(s) in ``location_column`` treated as
        the positive class. Default ``'c2'``.
    :param negative_control: Value(s) treated as the negative class.
        Default ``'c1'``.
    :param exclude: Columns to remove from feature space.
    :param n_repeats: Repeats for permutation importance. Default ``10``.
    :param top_features: Feature cap when ``prune_features=True``.
    :param reg_alpha: XGBoost L1 penalty.
    :param reg_lambda: XGBoost L2 penalty.
    :param learning_rate: XGBoost learning rate.
    :param n_estimators: Tree count for tree-based models.
    :param test_size: Test-split fraction. Default ``0.2``.
    :param model_type: ``'random_forest'``, ``'logistic_regression'``,
        ``'gradient_boosting'`` or ``'xgboost'``.
    :param n_jobs: Parallel job count where applicable. Default ``-1``.
    :param remove_low_variance_features: Drop low-variance features.
    :param remove_highly_correlated_features: Drop highly correlated features.
    :param prune_features: If True, apply ``SelectKBest`` before training.
    :param cross_validation: If True, run 5-fold stratified CV.
    :param verbose: Log progress details.
    :param split_by: Independent acquisition unit for train/test splitting:
        ``'cell'``, ``'field'``, ``'well'`` (default), or ``'plate'``.
        Legacy ``'none'`` is an alias for ``'cell'``.
    :param batch_correction: plate correction method from
        :mod:`spacr.batch_correction`.
    :param batch_column: metadata column identifying plates/batches.
    :param batch_control_column: metadata column holding reference-control
        labels for ``control_center``.
    :param batch_control_values: negative/reference control value(s).
    :param batch_min_samples: minimum rows or controls per plate.
    :param batch_covariate_column: Metadata column containing a biological
        covariate that ComBat must preserve, such as treatment, cell line, or
        time point. Required when ``batch_correction="combat"``; its
        coefficients remain in the corrected data while estimated batch
        effects are removed.
    :param batch_combat_mean_only: If ``True``, ComBat adjusts batch means
        without scaling batch variances. This can be appropriate when batches
        differ primarily by location or contain too few observations for
        stable variance estimates. Default ``False`` adjusts both means and
        variances.
    :param batch_missing_control: ``error`` or ``skip`` for missing controls.
    :returns: Tuple ``(output, figs)`` where ``output`` is a positional
        tuple of ``(scored_df, permutation_df, feature_importance_df,
        model, X_train, X_test, y_train, y_test, metrics_df,
        train_features)`` and ``figs`` is
        ``(permutation_fig, feature_importance_fig)``.
    :raises ValueError: on unsupported ``model_type`` or when positive /
        negative control rows cannot be located in ``location_column``.

    Example:
        .. code-block:: python

            from spacr.ml import ml_analysis
            output, figs = ml_analysis(
                df, channel_of_interest=3,
                positive_control='c2', negative_control='c1',
                model_type='xgboost',
            )
            scored_df = output[0]

    See Also:
        :func:`generate_ml_scores` — wraps this call with DB I/O.
    """
    
    _flowview_advance("dataset")

    def _match_control_values(series, control):
        """
        Return a boolean mask selecting rows in `series` that match `control`.

        Matching is attempted in this order:
        1. exact value match
        2. numeric coercion match
        3. stripped string match

        `control` can be a scalar or a list/tuple/set of values.
        """

        if isinstance(control, (list, tuple, set, np.ndarray, pd.Series)):
            controls = list(control)
        else:
            controls = [control]

        mask = pd.Series(False, index=series.index)

        for c in controls:
            current_mask = pd.Series(False, index=series.index)

            # 1. exact match
            try:
                current_mask |= (series == c)
            except Exception:
                pass

            # 2. numeric match
            try:
                s_num = pd.to_numeric(series, errors='coerce')
                c_num = pd.to_numeric(pd.Series([c]), errors='coerce').iloc[0]
                if pd.notna(c_num):
                    current_mask |= (s_num == c_num)
            except Exception:
                pass

            # 3. stripped string match
            try:
                s_str = series.astype(str).str.strip()
                c_str = str(c).strip()
                current_mask |= (s_str == c_str)
            except Exception:
                pass

            mask |= current_mask

        return mask
    
    from .utils import filter_dataframe_features
    from .plot import plot_permutation, plot_feature_importance

    # The run's seed, not a literal. Every estimator below takes this as
    # random_state, and an estimator given an explicit random_state ignores
    # the NumPy global stream -- so hard-coding 42 here silently overrode
    # whatever the user set as random_seed. Falls back to 42 outside a run,
    # which is what it always was.
    random_state = _run_random_state(42)

    if 'cells_per_well' in df.columns:
        df = df.drop(columns=['cells_per_well'])

    correction_metadata = df.copy()
    # THE POISONED-SETTINGS SIGNATURE, NAMED RATHER THAN RAISED THROUGH.
    # `df[[name]]` on a missing column raises a pandas KeyError from three
    # frames down that says only "None of [Index([...])] are in the
    # [columns]" -- issue #93. It points at the column and not at the reason
    # the column is being asked for, which was an annotation-mode run that
    # overwrote `location_column` and left it overwritten.
    if location_column not in df.columns:
        available = ", ".join(repr(c) for c in list(df.columns)[:12])
        if len(df.columns) > 12:
            available += f", ... ({len(df.columns)} columns)"
        # The hint is unconditional because the cause is: any missing
        # location_column reaching here is either a typo or the overwrite,
        # and naming the overwrite costs a sentence while a user who cannot
        # find it loses an afternoon. Phrased as a possibility, not a
        # diagnosis, because a typo deserves the column list either way.
        raise ValueError(
            f"location_column={location_column!r} is not a column of the "
            f"measurement table, so there is nothing to group the controls "
            f"by.\n  The table has: {available}"
            f"\n  If you have run this module in annotation mode, that is "
            f"the likely cause: versions before 1.5.0.5 wrote "
            f"annotation_column into location_column and never put it back, "
            f"so a later metadata run looked for an annotation column in the "
            f"measurement table. Set location_column back to your well "
            f"column ('columnID' or 'rowID').")

    # Name an empty measurement source before feature filtering turns it into
    # an empty training set and the control guard misleadingly blames the
    # configured control values.  Keep the missing-column diagnosis above:
    # that remains the more actionable error when the requested column does
    # not exist at all.
    if df.empty:
        raise ValueError(
            "the measurement table contains 0 object rows, so there is "
            "nothing to train on. Check that the selected source contains "
            "measured objects before running the analysis.")

    # A populated table can still have no usable labels.  This is a data-
    # population problem, not a typo in positive_control/negative_control.
    # Do not handle duplicate columns here: ``df[name]`` is then a DataFrame,
    # and the dedicated duplicate-column diagnosis below remains authoritative.
    location_values = df[location_column]
    if isinstance(location_values, pd.Series):
        non_empty_values = location_values.dropna().astype(str).str.strip()
        if not non_empty_values.ne("").any():
            raise ValueError(
                f"location_column={location_column!r} has 0 non-empty values "
                f"across {len(df)} object rows. Populate it with two real "
                f"class labels before running the analysis.")

    df_metadata = df[[location_column]].copy()

    df, features = filter_dataframe_features(df, channel_of_interest, exclude, remove_low_variance_features, remove_highly_correlated_features, verbose)
    print('After filtration:', len(df))

    if str(batch_correction or 'none').strip().lower() not in {
        'none', 'off', 'false',
    }:
        from .batch_correction import correct_from_metadata
        corrected, correction_report = correct_from_metadata(
            df[features],
            correction_metadata.loc[df.index],
            batch_correction=batch_correction,
            batch_column=batch_column,
            batch_control_column=batch_control_column,
            batch_control_values=batch_control_values,
            batch_covariate_column=batch_covariate_column,
            batch_combat_mean_only=batch_combat_mean_only,
            batch_min_samples=batch_min_samples,
            batch_missing_control=batch_missing_control,
        )
        df.loc[:, features] = corrected
        print(
            f"Batch correction {correction_report.method}: "
            f"{correction_report.centroid_spread_before} -> "
            f"{correction_report.centroid_spread_after} centroid spread.")
        for note in correction_report.warnings:
            print(f"Warning: batch correction: {note}")
    
    if verbose:
        print(f'Found {len(features)} numerical features in the dataframe')
        print(f'Features used in training: {features}')
        print(f'Features: {features}')
        
    df = pd.concat([df, df_metadata[location_column]], axis=1)
    # The merged measurement index is the canonical object identity. Keep it
    # beside the filtered features now so duplicate indexes in annotation
    # mode remain positionally aligned instead of being multiplied by .loc.
    df['prcfo'] = df.index.astype(str)
    
    #if verbose:
    #    print(df[location_column].dtype)
    #    print(type(negative_control), negative_control)
    #    print(type(positive_control), positive_control)
    #    print(df[location_column].dropna().unique()[:20])

    # Subset the dataframe based on specified column values
    #if isinstance(negative_control, str):
    #    df1 = df[df[location_column] == negative_control].copy()

    #elif isinstance(negative_control, list):
    #    df1 = df[df[location_column].isin(negative_control)].copy()

    #elif isinstance(negative_control, (int, float)):
    #    df1 = df[df[location_column] == negative_control].copy()
    #if verbose:
    #    print(f'Negative control: {negative_control}, samples: {len(df1)}')
    
    #if isinstance(positive_control, str):
    #    df2 = df[df[location_column] == positive_control].copy()

    #elif isinstance(positive_control, list):
    #    df2 = df[df[location_column].isin(positive_control)].copy()
        
    #elif isinstance(positive_control, (int, float)):
    #    df2 = df[df[location_column] == positive_control].copy()
        
    #if verbose:
    #    print(f'Positive control: {positive_control}, samples: {len(df2)}')
        
    # THE CONTROLS MUST BE VALUES OF THE COLUMN BEING MATCHED. In annotation
    # mode `location_column` is the ANNOTATION column, whose values are the
    # class labels -- 1.0 and 2.0, say -- while positive_control and
    # negative_control default to plate column names like 'c1' and 'c2'.
    # Applying one to the other finds nothing, which is issues #91 and #92.
    #
    # When the named controls appear nowhere in the column but it holds
    # exactly TWO classes, those two ARE the classes: the lower value is the
    # negative and the higher the positive. That is the ordinary annotation
    # case and it should not require the user to restate what the column
    # already says.
    negative_control, positive_control, _derived_classes = _resolve_controls(
        df, location_column, negative_control, positive_control,
        _match_control_values)

    df1 = df[_match_control_values(df[location_column], negative_control)].copy()
    if verbose:
        print(f'Negative control: {negative_control}, samples: {len(df1)}')

    df2 = df[_match_control_values(df[location_column], positive_control)].copy()
    if verbose:
        print(f'Positive control: {positive_control}, samples: {len(df2)}')
        
    # Create target variable
    df1['target'] = 0 # Negative control
    df2['target'] = 1 # Positive control

    # Combine the subsets for analysis
    combined_df = pd.concat([df1, df2])
    combined_df = combined_df.drop(columns=[location_column])
    
    if verbose:
        print(f'Found {len(df1)} samples for {negative_control} and {len(df2)} samples for {positive_control}. Total: {len(combined_df)}')

    # A CLASS NOBODY NAMED IS STILL SCORED, AND THAT HAS TO BE SAID.
    #
    # This fit is binary by construction: one arm is the negative control,
    # the other the positive, and every remaining row of the input is scored
    # afterwards by the model. That is the point of a screen -- the unknown
    # population is what the scores are for -- but with THREE or more
    # classes in the column it is easy to believe all of them were trained
    # on. They were not, and nothing said so (instruction 236 D13).
    #
    # Both controls take a LIST, so classes can be pooled into the two arms
    # deliberately: positive_control=['c3', 'c4'] trains one arm on both.
    # A SENTENCE MUST NOT BE ABLE TO BREAK A RUN. This is cosmetic, and it
    # sits ABOVE the guard that refuses a table with two columns of one
    # name -- where `df[location_column]` is a DataFrame and `.unique()`
    # does not exist. It raised AttributeError there and masked the guard's
    # own message, which is the one the user needed.
    untrained = []
    try:
        column = df[location_column] if location_column in df.columns else None
        if isinstance(column, pd.Series):
            trained_on = set(df1[location_column].unique()) | set(
                df2[location_column].unique())
            present = set(column.dropna().unique())
            untrained = sorted(str(value) for value in present - trained_on)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not list the classes outside the training set",
                  exc_info=True)
    if untrained:
        print(f"{len(untrained)} class(es) of {location_column!r} are not in "
              f"the training set and are SCORED by a model that never saw "
              f"them: {untrained[:10]}"
              f"{'...' if len(untrained) > 10 else ''}. This fit is binary: "
              f"one arm is negative_control={negative_control!r} and the "
              f"other positive_control={positive_control!r}. Both take a "
              f"list, so name several values to pool them into one arm.")
    
    # REFUSE HERE, NAMING WHAT IS ACTUALLY IN THE COLUMN.
    #
    # When neither control matches, df1 and df2 are both empty, combined_df is
    # empty, and the failure surfaces as
    #
    #     ValueError: With n_samples=0, test_size=0.2 and train_size=None,
    #     the resulting train set will be empty
    #
    # from inside sklearn's train_test_split, three frames below anything a
    # user recognises. That traceback was auto-filed to the spaCR tracker TEN
    # TIMES in one day (issues #79-#90) and names neither the setting that is
    # wrong nor the value it should have had.
    #
    # The verbose branch above would have said "samples: 0", but verbose is
    # False on every shipped path.
    if df1.empty or df2.empty:
        column = df[location_column]
        if isinstance(column, pd.DataFrame):
            # TWO COLUMNS OF THAT NAME. `df[name]` is then a DataFrame, every
            # matching strategy in `_match_control_values` fails against it,
            # and no control is ever found. Worth its own sentence: the fix
            # is to the TABLE, not to the control values, and no amount of
            # correcting positive_control will help.
            raise ValueError(
                f"the measurement table has {column.shape[1]} columns named "
                f"{location_column!r}, so the controls cannot be matched "
                f"against it. Drop or rename the duplicate before running "
                f"the analysis.")
        present = column.astype(str).str.strip().unique().tolist()
        shown = ", ".join(repr(v) for v in sorted(present)[:15])
        if len(present) > 15:
            shown += f", ... ({len(present)} distinct values)"
        missing = []
        if df1.empty:
            missing.append(f"negative_control={negative_control!r}")
        if df2.empty:
            missing.append(f"positive_control={positive_control!r}")
        raise ValueError(
            f"no rows matched {' and '.join(missing)} in column "
            f"{location_column!r}, so there is nothing to train on.\n"
            f"  {location_column!r} contains: {shown}\n"
            f"  Set positive_control and negative_control to values that "
            f"appear there, or set location_column to the column that holds "
            f"your controls.")

    X = combined_df[features]
    y = combined_df['target']
    
    if prune_features:
        before_pruning = len(X.columns)
        selector = SelectKBest(score_func=f_classif, k=top_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get the selected feature names
        selected_features = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        features = selected_features.tolist()
        
        after_pruning = len(X.columns)
        print(f"Removed {before_pruning - after_pruning} features using SelectKBest")

    _flowview_metric("objects", len(df))
    _flowview_metric("training_objects", len(combined_df))
    _flowview_metric("features", len(features))
    _flowview_advance("split")

    # Split on an actual experimental unit. The index is the canonical prcfo
    # in merged measurement frames, even when filtering removed its component
    # metadata columns from X.
    from .classifier_evaluation import grouped_split, split_group_values
    split_frame = combined_df[['prcfo']].reset_index(drop=True)
    split_level, split_groups = split_group_values(
        group_by=split_by, frame=split_frame, table='ML control measurements')
    # A NAMED HOLDOUT BEATS A RANDOM ONE. Cross-validation splits within the
    # data it is given, so a model can learn the PLATE rather than the
    # phenotype and every number it reports still looks fine. Naming a plate
    # trains without it and scores on it, which is the one number that says
    # whether the classifier generalises.
    held = holdout_plate
    if held is not None and not isinstance(held, (list, tuple, set)):
        held = [held]
    if held:
        _plate_level, plate_groups = split_group_values(
            group_by='plate', frame=split_frame,
            table='ML control measurements')
        train_index, test_index, split_report = grouped_split(
            plate_groups, y.to_numpy(), test_size, seed=random_state,
            group_by='plate', hold_out_groups=held)
    else:
        train_index, test_index, split_report = grouped_split(
            split_groups, y.to_numpy(), test_size, seed=random_state,
            group_by=split_level)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(split_report.summary())

    # Add data usage labels
    combined_df['data_usage'] = 'train'
    combined_df.loc[X_test.index, 'data_usage'] = 'test'
    df['data_usage'] = 'not_used'
    df.loc[combined_df.index, 'data_usage'] = combined_df['data_usage']
    df['data_usage_group_by'] = split_report.group_by
    df['split_requested_fraction'] = split_report.requested_fraction
    df['split_cell_fraction'] = split_report.cell_fraction
    df['split_group_fraction'] = split_report.group_fraction
    
    _flowview_metric("objects", len(X))
    _flowview_metric("train_objects", len(X_train))
    _flowview_metric("test_objects", len(X_test))
    _flowview_advance("model")

    # Initialize the model based on model_type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'extra_trees':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == 'gradient_boosting':
        model = HistGradientBoostingClassifier(max_iter=n_estimators, random_state=random_state)  # Supports n_jobs internally
    elif model_type == 'xgboost':
        model = XGBClassifier(
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            nthread=n_jobs,
            eval_metric='logloss',
        )
    elif model_type == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("model_type='lightgbm' requires the 'lightgbm' package. Install it with: pip install lightgbm")
        model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'catboost':
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("model_type='catboost' requires the 'catboost' package. Install it with: pip install catboost")
        model = CatBoostClassifier(iterations=n_estimators, learning_rate=learning_rate, l2_leaf_reg=reg_lambda, random_state=random_state, thread_count=n_jobs, verbose=False)
    elif model_type == 'svm':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC
        # scikit-learn 1.9 deprecated SVC(probability=True). A calibrated
        # decision-function SVC provides the same predict_proba contract
        # without relying on the mode removed in 1.11.
        model = CalibratedClassifierCV(
            estimator=SVC(random_state=random_state),
            method='sigmoid',
            cv=3,
            n_jobs=n_jobs,
            ensemble=False,
        )
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(max_iter=max(200, n_estimators), random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Estimators returned here can be persisted with joblib/pickle. Keeping the
    # report on the object makes grouping provenance travel with such a model
    # rather than existing only in stdout or the scored CSV.
    model.spacr_split_report_ = split_report.to_dict()

    _flowview_metric("features", len(X.columns))
    _flowview_advance("training")

    # Perform k-fold cross-validation
    if cross_validation:
        from .io import make_cv_folds

        distinct_groups = len(np.unique(split_groups))
        n_folds = min(5, distinct_groups)
        folds = make_cv_folds(
            y.to_numpy(), n_folds, groups=split_groups,
            seed=random_state)
        expected_classes = set(np.unique(y))
        fold_metrics = []

        for fold_idx, (train_index, test_index) in enumerate(folds, start=1):
            if (set(np.unique(y.iloc[train_index])) != expected_classes or
                    set(np.unique(y.iloc[test_index])) != expected_classes):
                raise ValueError(
                    f"{split_level}-grouped CV fold {fold_idx} cannot put "
                    "every class in both train and test. Add independent "
                    f"class-bearing {split_level}s or choose a finer split.")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Predict for the current test set
            predictions_test = model.predict(X_test)
            combined_df.loc[X_test.index, 'predictions'] = predictions_test

            # Get prediction probabilities for the test set
            prediction_probabilities_test = model.predict_proba(X_test)

            # Find the optimal threshold
            optimal_threshold = find_optimal_threshold(y_test, prediction_probabilities_test[:, 1])
            if verbose:
                print(f'Fold {fold_idx} - Optimal threshold: {optimal_threshold}')

            # Assign predictions and probabilities to the test set in the DataFrame
            df.loc[X_test.index, 'predictions'] = predictions_test
            for i in range(prediction_probabilities_test.shape[1]):
                df.loc[X_test.index, f'prediction_probability_class_{i}'] = prediction_probabilities_test[:, i]

            # Evaluate performance for the current fold
            fold_report = classification_report(
                y_test, predictions_test, output_dict=True, zero_division=0)
            fold_metrics.append(pd.DataFrame(fold_report).transpose())

            if verbose:
                print(f"Fold {fold_idx} Classification Report:")
                print(classification_report(
                    y_test, predictions_test, zero_division=0))

        # Aggregate metrics across all folds
        metrics_df = pd.concat(fold_metrics).groupby(level=0).mean()

        # Re-train on full data (X, y) and then apply to entire df
        model.fit(X, y)  
        all_predictions = model.predict(df[features])  # Predict on entire df
        df['predictions'] = all_predictions

        # Get prediction probabilities for all rows in df
        prediction_probabilities = model.predict_proba(df[features])
        for i in range(prediction_probabilities.shape[1]):
            df[f'prediction_probability_class_{i}'] = prediction_probabilities[:, i]

        #if verbose:
        #    print("\nFinal Classification Report on Full Dataset:")
        #    print(classification_report(y, all_predictions))

        # Generate metrics DataFrame
        #final_report_dict = classification_report(y, all_predictions, output_dict=True)
        #metrics_df = pd.DataFrame(final_report_dict).transpose()
    
    else:
        model.fit(X_train, y_train)
        # Predicting the target variable for the test set
        predictions_test = model.predict(X_test)
        combined_df.loc[X_test.index, 'predictions'] = predictions_test

        # Get prediction probabilities for the test set
        prediction_probabilities_test = model.predict_proba(X_test)

        # Find the optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, prediction_probabilities_test[:, 1])
        if verbose:
            print(f'Optimal threshold: {optimal_threshold}')

        # Predicting the target variable for all other rows in the dataframe
        X_all = df[features]
        all_predictions = model.predict(X_all)
        df['predictions'] = all_predictions

        # Get prediction probabilities for all rows in the dataframe
        prediction_probabilities = model.predict_proba(X_all)
        for i in range(prediction_probabilities.shape[1]):
            df[f'prediction_probability_class_{i}'] = prediction_probabilities[:, i]
            
        if verbose:
            print("\nClassification Report:")
            print(classification_report(
                y_test, predictions_test, zero_division=0))
            
        report_dict = classification_report(
            y_test, predictions_test, output_dict=True, zero_division=0)
        metrics_df = pd.DataFrame(report_dict).transpose()

    _flowview_metric("objects", len(X))
    _flowview_metric("features", len(features))
    _flowview_advance("evaluation")

    # ``model_metrics.csv`` is the classical model's durable card. Repeat the
    # scalar provenance on its rows so it survives CSV and remains filterable.
    metrics_df['split_group_by'] = split_report.group_by
    metrics_df['split_requested_fraction'] = split_report.requested_fraction
    metrics_df['split_group_fraction'] = split_report.group_fraction
    metrics_df['split_cell_fraction'] = split_report.cell_fraction
        
    # joblib workers are fresh threads, so they do not inherit the region's
    # single-thread OpenMP clamp and re-enter the model with a full team.
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=n_repeats, random_state=random_state, n_jobs=guarded_n_jobs(n_jobs, 'permutation importance'))

    # Create a DataFrame for permutation importances
    permutation_df = pd.DataFrame({
        'feature': [features[i] for i in perm_importance.importances_mean.argsort()],
        'importance_mean': perm_importance.importances_mean[perm_importance.importances_mean.argsort()],
        'importance_std': perm_importance.importances_std[perm_importance.importances_mean.argsort()]
    }).tail(top_features)

    permutation_fig = plot_permutation(permutation_df)
    if verbose:
        permutation_fig.show()

    # Feature importance for models that support it. Use hasattr rather than a
    # hardcoded model list: HistGradientBoostingClassifier (model_type=
    # 'gradient_boosting') does NOT expose feature_importances_, so the old
    # list-based check raised AttributeError. Models without the attribute
    # (e.g. logistic_regression) fall through to the else branch, which must
    # also define feature_importance_fig or the return raises UnboundLocalError.
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False).head(top_features)

        feature_importance_fig = plot_feature_importance(feature_importance_df)
        if verbose:
            feature_importance_fig.show()

    else:
        # NO NATIVE IMPORTANCES IS NOT NO IMPORTANCES. Four of the nine
        # models this module offers -- gradient_boosting, logistic_
        # regression, svm and mlp -- do not expose `feature_importances_`,
        # and this branch used to hand back an empty frame and no figure.
        # A user who picks logistic_regression, which the setting's own
        # tooltip recommends as "a good linear sanity check", lost the
        # feature-importance QC panel entirely and was told nothing.
        #
        # THE PERMUTATION IMPORTANCE IS ALREADY COMPUTED, a few lines up,
        # for every model, because it is model-agnostic by construction.
        # It is a DIFFERENT QUANTITY from a tree's split-gain importance --
        # it measures what the fitted model loses when a column is shuffled
        # -- so the panel says which one it is drawing rather than passing
        # one off as the other.
        feature_importance_df = permutation_df.rename(
            columns={"importance_mean": "importance"}
        )[["feature", "importance"]].sort_values(
            by="importance", ascending=False).head(top_features)
        feature_importance_fig = plot_feature_importance(
            feature_importance_df,
            title=f"Top {len(feature_importance_df)} features "
                  f"(permutation importance)")
        if verbose:
            feature_importance_fig.show()

    df = _calculate_similarity(df, features, location_column, positive_control, negative_control)

    df['prcfo'] = df.index.astype(str)
    # Six tokens on a timelapse, five otherwise; see _assign_prcfo_parts. The
    # five-name split raised ValueError here on every timelapse database,
    # discarding a model that had already been fitted and scored.
    df = _assign_prcfo_parts(df, object_column='object')
    df['prc'] = _compose_prc_column(df)
    
    return [df, permutation_df, feature_importance_df, model, X_train, X_test, y_train, y_test, metrics_df, features], [permutation_fig, feature_importance_fig]

#: How many background rows a model-agnostic explainer is given. The
#: permutation explainer is O(background x features) per explained row, so
#: the whole training set turns a 0.3 s panel into minutes. Summarising the
#: background is what the SHAP authors recommend for exactly this.
SHAP_BACKGROUND = 100


def _shap_values(model, X_train, X_test):
    """(values, note). SHAP contributions for every model the panel offers.

    THE FAILURE IS NOT ALWAYS AT CONSTRUCTION. `shap.Explainer(model, X)`
    accepts an xgboost booster happily and raises "Categorical split is not
    yet supported" only when it is CALLED -- so a fallback chosen at
    construction time never ran, and the panel's DEFAULT model produced no
    SHAP at all. Each candidate is therefore tried all the way through.
    """
    import shap

    attempts = list(_shap_explainers(model, X_train))
    trouble = ""
    for explainer, note in attempts:
        try:
            return explainer(X_test), note
        except Exception as error:                           # noqa: BLE001
            trouble = f"{type(error).__name__}: {error}"
            LOG.debug("a SHAP explainer would not run", exc_info=True)
    raise RuntimeError(
        f"No SHAP explainer could explain {type(model).__name__}. The last "
        f"failure was: {trouble}")


def _shap_explainers(model, X_train):
    """Every explainer worth trying for this estimator, best first.


    THREE OF THE NINE MODELS THE PANEL OFFERS COULD NOT BE EXPLAINED AT
    ALL, including the default:

    * `xgboost` raised "Categorical split is not yet supported. You can
      still use TreeExplainer with feature_perturbation=tree_path_dependent"
      -- an error carrying its own fix, which nothing acted on.
    * `svm` and `mlp` raised "The passed model is not callable and cannot be
      analyzed directly with the given masker". A support vector machine and
      a neural net are not trees and not linear; the model-agnostic
      explainer takes a FUNCTION, not an estimator.

    The note is returned rather than printed here so the caller decides
    where it goes, and it is said out loud because the three explainers do
    not compute the same quantity: `tree_path_dependent` conditions on the
    tree's own splits rather than on an independent background, and the
    permutation explainer estimates rather than solves.
    """
    import shap

    try:
        # Older supported SHAP releases accept XGBoost here and silently
        # choose their automatic tree explainer; newer releases reject the
        # same categorical model and reach the explicit TreeExplainer below.
        # In both cases say which semantics the panel used.  The note cannot
        # be tied only to the fallback or identical runs become silent on the
        # minimum dependency stack.
        automatic_note = ""
        if type(model).__module__.split(".", 1)[0] == "xgboost":
            automatic_note = (
                "SHAP: XGBoost was accepted by the automatic tree "
                "explainer, so the panel shows tree contributions rather "
                "than a model-agnostic estimate."
            )
        yield shap.Explainer(model, X_train), automatic_note
    except Exception:                                        # noqa: BLE001
        LOG.debug("the default SHAP explainer would not build",
                  exc_info=True)

    # A tree whose splits are categorical. The library's own message names
    # this remedy, and it is what xgboost needs.
    try:
        yield (shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"),
            "SHAP: this model has categorical splits, so it is explained "
            "with feature_perturbation='tree_path_dependent' -- which "
            "conditions on the tree's own splits rather than on an "
            "independent background.")
    except Exception:                                        # noqa: BLE001
        LOG.debug("this model is not a tree", exc_info=True)

    # Not a tree and not linear -- a support vector machine, a neural net.
    # The model-agnostic explainer takes a FUNCTION, not an estimator, and
    # the background is summarised because it costs O(background) per
    # explained row.
    predict = (getattr(model, "predict_proba", None)
               or getattr(model, "decision_function", None)
               or getattr(model, "predict", None))
    if predict is None:
        return
    background = X_train
    if hasattr(X_train, "shape") and X_train.shape[0] > SHAP_BACKGROUND:
        background = shap.utils.sample(X_train, SHAP_BACKGROUND,
                                       random_state=0)
    try:
        yield (shap.Explainer(predict, background),
               f"SHAP: {type(model).__name__} is neither a tree nor a "
               f"linear model, so it is explained through its predictions "
               f"over {len(background)} background row(s). That is an "
               f"ESTIMATE of each contribution rather than an exact "
               f"decomposition.")
    except Exception:                                        # noqa: BLE001
        LOG.debug("the model-agnostic SHAP explainer would not build",
                  exc_info=True)


def shap_analysis(model, X_train, X_test):
    """Build a SHAP summary beeswarm for ``X_test``.

    The beeswarm is rendered with pyqtgraph so it can be embedded in the same
    scene-based figure workflow as other model-explanation plots.

    The function returns a live
    :class:`~spacr.qt.widgets.fast_plots.FastPlot`; it neither writes a file
    nor returns a matplotlib figure. Pass the result to :func:`write_plot` to
    export it in the configured figure format.

    :param model: Fitted estimator compatible with ``shap.Explainer``.
    :param X_train: Training features used to seed the explainer.
    :param X_test: Test features to explain.
    :returns: A ``FastPlot`` holding the beeswarm, or ``None`` when Qt is
        unavailable or the attribution matrix cannot be plotted.
    """
    import shap

    shap_values, note = _shap_values(model, X_train, X_test)
    if note:
        print(note)
    # TreeExplainer returns one output axis for every classifier class in
    # recent SHAP releases: (samples, features, classes).  A 3-D input is
    # interaction values to anything downstream, which both misrepresents
    # the data and crashes when feature_names is a plain list.  The
    # classifiers used by this pipeline are binary, so explain the positive
    # class.  Keep the only output for estimators with a singleton axis.
    if len(shap_values.shape) == 3:
        output_index = 1 if shap_values.shape[-1] > 1 else 0
        shap_values = shap_values[..., output_index]

    from .figures.headless import application

    application_object, refusal = application()
    if application_object is None:
        print(refusal)
        return None

    from .qt.widgets.fast_plots import FastPlot

    matrix = np.asarray(shap_values.values, dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        return None
    columns = list(X_test.columns)[:matrix.shape[1]]
    # RANKED BY MEAN ABSOLUTE CONTRIBUTION, which is the order the library
    # uses and the only one that answers "which of these matters": a feature
    # that pushes hard in both directions has a mean near zero and belongs
    # at the top, not the bottom.
    order = np.argsort(np.nanmean(np.abs(matrix), axis=0))[::-1]
    names = [str(columns[int(i)]) for i in order]
    plot = FastPlot(title="SHAP summary", x_label="SHAP value", y_label="")
    plot.resize(1200, max(420, 34 * len(names) + 140))
    if not plot.add_beeswarm(names, matrix[:, order],
                             X_test[names].to_numpy(dtype=float)):
        plot.deleteLater()
        return None
    application_object.processEvents()
    return plot


def write_plot(plot, path, title=""):
    """Write a pyqtgraph plot out and announce it, like ``publish`` does.

    The counterpart of :func:`spacr.figure_sink.publish` for a scene rather
    than a matplotlib figure: the format follows the user's preference, the
    file NAME follows the format, and the written file reaches the gallery,
    because saved and visible are the same event.

    ``None`` writes nothing, announces nothing and returns None -- a plot
    that could not be built must not take the run down after the model has
    been fitted and every object scored.

    :param plot: a ``FastPlot``, or None.
    :param path: where to write it; the extension may be rewritten.
    :param title: the name the gallery tile carries.
    :returns: the path written, or None.
    """
    if plot is None:
        return None
    from .figure_sink import publish_file
    from .plot import figure_output_preferences

    chosen = str(figure_output_preferences()[0]).lower().lstrip('.')
    stem, _ = os.path.splitext(str(path))
    target = f"{stem}.{chosen}"
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    try:
        written = plot.export(target)
    finally:
        plot.deleteLater()
    if written:
        # BY NAME. `publish_file(path, title=None)` names its second
        # parameter, and passing it positionally makes every caller that
        # stands in for the sink -- a GUI bridge, a test double -- have to
        # guess that the second positional is the tile's title.
        publish_file(written, title=title or None)
    return written

def find_optimal_threshold(y_true, y_pred_proba):
    """Return the probability threshold maximising F1 on the precision-recall curve.

    :param y_true: Ground-truth binary labels.
    :param y_pred_proba: Predicted probabilities for the positive class.
    :returns: Optimal probability threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # A precision-recall sweep can contain points where precision and recall
    # are both 0 (every predicted positive is a true negative). The plain
    # 2*(p*r)/(p+r) produced NaN there, and np.argmax returns the index of the
    # first NaN rather than the true F1 maximum, so the returned threshold
    # could be one whose F1 is 0. F1 is 0 by definition when p + r == 0.
    denominator = precision + recall
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.where(denominator > 0,
                             2 * (precision * recall) / denominator,
                             0.0)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def _calculate_similarity(df, features, col_to_compare, val1, val2):
    """
    Calculate similarity scores of each well to the positive and negative controls using various metrics.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    features (list): List of feature columns to use for similarity calculation.
    col_to_compare (str): Column name to use for comparing groups.
    val1, val2 (str): Values in col_to_compare to create subsets for comparison.

    Returns:
    pandas.DataFrame: DataFrame with similarity scores.
    """
    # Separate positive and negative control wells
    if isinstance(val1, str):
        pos_control = df[df[col_to_compare] == val1][features].mean()
    elif isinstance(val1, list):
        pos_control = df[df[col_to_compare].isin(val1)][features].mean()
    if isinstance(val2, str):
        neg_control = df[df[col_to_compare] == val2][features].mean()
    elif isinstance(val2, list):
        neg_control = df[df[col_to_compare].isin(val2)][features].mean()
    
    # Standardize features for Mahalanobis distance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Regularize the covariance matrix to avoid singularity
    cov_matrix = np.cov(scaled_features, rowvar=False)
    inv_cov_matrix = None
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add a small value to the diagonal elements for regularization
        epsilon = 1e-5
        inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon)
        
    # Calculate similarity scores
    def safe_similarity(func, row, control, *args, **kwargs):
        """Call ``func(row, control, ...)`` and swallow errors (return ``NaN``)."""
        try:
            return func(row, control, *args, **kwargs)
        except Exception:
            return np.nan
        
    # Calculate similarity scores
    try:
        df['similarity_to_pos_euclidean'] = df[features].apply(lambda row: safe_similarity(euclidean, row, pos_control), axis=1)
        df['similarity_to_neg_euclidean'] = df[features].apply(lambda row: safe_similarity(euclidean, row, neg_control), axis=1)
        df['similarity_to_pos_cosine'] = df[features].apply(lambda row: safe_similarity(cosine, row, pos_control), axis=1)
        df['similarity_to_neg_cosine'] = df[features].apply(lambda row: safe_similarity(cosine, row, neg_control), axis=1)
        df['similarity_to_pos_mahalanobis'] = df[features].apply(lambda row: safe_similarity(mahalanobis, row, pos_control, inv_cov_matrix), axis=1)
        df['similarity_to_neg_mahalanobis'] = df[features].apply(lambda row: safe_similarity(mahalanobis, row, neg_control, inv_cov_matrix), axis=1)
        df['similarity_to_pos_manhattan'] = df[features].apply(lambda row: safe_similarity(cityblock, row, pos_control), axis=1)
        df['similarity_to_neg_manhattan'] = df[features].apply(lambda row: safe_similarity(cityblock, row, neg_control), axis=1)
        df['similarity_to_pos_minkowski'] = df[features].apply(lambda row: safe_similarity(minkowski, row, pos_control, p=3), axis=1)
        df['similarity_to_neg_minkowski'] = df[features].apply(lambda row: safe_similarity(minkowski, row, neg_control, p=3), axis=1)
        df['similarity_to_pos_chebyshev'] = df[features].apply(lambda row: safe_similarity(chebyshev, row, pos_control), axis=1)
        df['similarity_to_neg_chebyshev'] = df[features].apply(lambda row: safe_similarity(chebyshev, row, neg_control), axis=1)
        df['similarity_to_pos_braycurtis'] = df[features].apply(lambda row: safe_similarity(braycurtis, row, pos_control), axis=1)
        df['similarity_to_neg_braycurtis'] = df[features].apply(lambda row: safe_similarity(braycurtis, row, neg_control), axis=1)
    except Exception as e:
        print(f"Error calculating similarity scores: {e}")    
    return df

def _announce_the_bundle(folder, title):
    """Put ONE tile in the gallery for a bundle's figure.

    ONE PICTURE, ONE TILE. A bundle holds the same figure twice, as a PDF
    and as a PNG, because a folder somebody opens should carry both -- but
    announcing both puts two tiles in the gallery for one picture, and a
    reader clicking each of them to find out they are the same is exactly
    the confusion the gallery exists to remove. The one announced is the
    one in the format the user chose.

    :param folder: the bundle directory.
    :param title: the name the tile carries.
    :returns: the path announced, or None when the folder holds no figure.
    """
    from .figure_sink import publish_file
    from .plot import figure_output_preferences

    if not folder or not os.path.isdir(folder):
        return None
    wanted = str(figure_output_preferences()[0]).lower().lstrip('.')
    written = sorted(os.listdir(folder))
    chosen = next((f for f in written if f.lower().endswith(f".{wanted}")),
                  None)
    if chosen is None:
        # The preference names a format this bundle does not hold -- a
        # bundle writes pdf and png whatever the preference says. Announce
        # the vector one rather than nothing.
        chosen = next((f for f in written if f.lower().endswith(".pdf")), None)
    if chosen is None:
        return None
    return publish_file(os.path.join(folder, chosen), title)


_EPHEMERAL_FIGURES = None


def _figure_folder(src, save):
    """Where a drawn figure goes: the run folder, or a temporary one.

    `save` GATES THE RUN FOLDER, NOT THE PICTURE. Before these charts moved
    to pyqtgraph they were `plt.show()`n and never written, so a `save=False`
    run still SAW them -- and writing them into the user's results folder
    now would be a behaviour change nobody asked for. A temporary directory
    is what an ephemeral figure has always been; the gallery gets its tile
    either way, because saved and visible are one event.

    :param src: plate folder.
    :param save: whether this run is writing its results.
    :returns: a directory that exists.
    """
    global _EPHEMERAL_FIGURES

    if save:
        folder = os.path.join(str(src), 'results')
        os.makedirs(folder, exist_ok=True)
        return folder
    if _EPHEMERAL_FIGURES is None:
        import tempfile

        _EPHEMERAL_FIGURES = tempfile.mkdtemp(prefix="spacr-figures-")
    return _EPHEMERAL_FIGURES


def _draw_response_panel_in_pyqtgraph(values, transform, column, src):
    """Draw the response distribution before and after, and publish it.

    :param values: the untransformed response.
    :param transform: the transformation named in the settings.
    :param column: the response's own column name.
    :param src: plate folder, or None to draw without writing a file.
    :returns: the path written, or None.
    """
    from .figures.headless import application

    application_object, refusal = application()
    if application_object is None:
        print(refusal)
        return None

    from .response_distribution import fast_panel

    plot = fast_panel(values, transform, dependent_variable=column)
    if plot is None:
        print("the response distribution panel was not drawn: the "
              "response holds no finite values")
        return None
    plot.resize(1100, 660)
    application_object.processEvents()
    if not src:
        plot.deleteLater()
        return None
    return write_plot(
        plot, os.path.join(_figure_folder(src, True),
                           'response_distribution.pdf'),
        "Response distribution")


def _draw_shap_summary_in_pyqtgraph(shap_values, sample, src, name, top,
                                    save=True):
    """Draw a SHAP beeswarm in pyqtgraph and write its bundle.

    The features are ranked by MEAN ABSOLUTE contribution, which is the
    order `shap.summary_plot` uses and the only one that answers "which of
    these matters": a feature that pushes hard in both directions has a mean
    near zero and belongs at the top, not the bottom.

    :param shap_values: a shap Explanation, or anything with ``.values``.
    :param sample: the frame the values were computed over.
    :param src: plate folder; the bundle goes under ``<src>/results``.
    :param name: bundle name.
    :param top: how many features to show.
    :returns: the folder written, or None when there is no Qt.
    """
    from .figures.headless import application
    from .figure_sink import publish_file

    application_object, refusal = application()
    if application_object is None:
        print(refusal)
        return None

    from .qt.widgets.fast_plots import FastPlot
    from .figures.bundle import save as write_bundle

    matrix = np.asarray(getattr(shap_values, 'values', shap_values),
                        dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        return None
    columns = list(sample.columns)[:matrix.shape[1]]
    strength = np.nanmean(np.abs(matrix), axis=0)
    order = np.argsort(strength)[::-1][:int(top)]
    names = [str(columns[int(i)]) for i in order]
    picked = matrix[:, order]
    values = sample[names].to_numpy(dtype=float)

    title = f"SHAP summary - top {len(names)} features"
    plot = FastPlot(title=title, x_label="SHAP value", y_label="")
    try:
        plot.resize(1200, max(420, 34 * len(names) + 140))
        if not plot.add_beeswarm(names, picked, values):
            return None
        application_object.processEvents()
        folder = write_bundle(_figure_folder(src, save), name,
                              render=plot.export,
                              data=plot.beeswarm_frame(), groups=None,
                              unit="observation",
                              settings={"top_features": int(top),
                                        "figure": name})
    finally:
        plot.deleteLater()
    _announce_the_bundle(folder, title)
    return folder


def _draw_the_cell_count_sweep(summary, mark, path):
    """Draw the sample-size sweep in pyqtgraph and write it out.

    PUBLISHED, NOT SHOWN. `plt.show()` here blocked forever anywhere there
    was no GUI event loop to hand it to: with the Qt backend it calls
    `start_main_loop`, and a script, a notebook or `spacr-run regression`
    then sat in `qt_compat._exec` until it was killed. Saved and visible are
    ONE event, through the figure sink, and a figure reaching the gallery
    must not depend on somebody calling `show`.

    :param summary: frame with ``sample_size``, ``smoothed_mean_abs_diff``
        and ``std_abs_diff``.
    :param mark: the sample size the threshold line is drawn at.
    :param path: destination; the extension follows the format preference.
    :returns: the path written, or None when there is no Qt to draw under.
    """
    from .figures.headless import application

    application_object, refusal = application()
    if application_object is None:
        print(refusal)
        return None

    from .qt.widgets.fast_plots import FastPlot
    from .figures.style import ROLES

    sizes = summary['sample_size'].to_numpy(dtype=float)
    middle = summary['smoothed_mean_abs_diff'].to_numpy(dtype=float)
    spread = summary['std_abs_diff'].to_numpy(dtype=float)
    plot = FastPlot(title="Mean absolute difference against sample size",
                    x_label="Sample size",
                    y_label="Mean absolute difference")
    try:
        plot.resize(1100, 760)
        if not plot.add_curve(sizes, middle, low=middle - spread,
                              high=middle + spread):
            return None
        # THE REFERENCE ROLE, NOT BLACK. A black guide line is invisible on
        # spaCR's dark theme.
        plot.add_line(x=float(mark), colour=ROLES["reference"],
                      label="minimum cell count")
        application_object.processEvents()
        return write_plot(plot, path, "Minimum cell count")
    except Exception:                                            # noqa: BLE001
        plot.deleteLater()
        raise


def _figure_name_for(title):
    """A filename from a figure title: lower case, words joined by _."""
    keep = [ch.lower() if ch.isalnum() else " " for ch in str(title)]
    return "_".join("".join(keep).split()) or "figure"


def _draw_radar_in_pyqtgraph(labels, values, title, src, name,
                             save=True):
    """Draw a radar in pyqtgraph and write its bundle under ``<src>/results``.

    Returns the folder, or None when there is no Qt to render under -- which
    `render`'s own refusal explains rather than leaving the run silent.
    """
    from .figures.headless import application
    from .figure_sink import publish_file

    application_object, refusal = application()
    if application_object is None:
        print(refusal)
        return None

    from .qt.widgets.fast_plots import FastPlot
    from .figures.bundle import save as write_bundle

    plot = FastPlot(title=title, x_label="", y_label="")
    try:
        plot.resize(820, 780)
        if not plot.add_radar(labels, values):
            return None
        application_object.processEvents()
        folder = write_bundle(_figure_folder(src, save), name,
                              render=plot.export,
                              data=plot.radar_frame(), groups=None,
                              unit="feature", settings={"figure": name})
    finally:
        plot.deleteLater()
    _announce_the_bundle(folder, title)
    return folder


def _draw_importance_in_pyqtgraph(frame, title, src, name, top,
                                  save=True):
    """Draw a ranked importance chart in pyqtgraph and write it out.

    The regression and explanation figures were drawn twice -- once in
    pyqtgraph for the tab and once in matplotlib for the file -- so one
    screen produced two pictures of one number from two code paths that can
    disagree. These two were the last that could not move, because twenty
    feature names need HORIZONTAL bars and the plot could not draw them.

    Returns the bundle folder, or None when there is no Qt to render under.
    A None is not a failure: the caller has already written the CSV, and
    `render_bundle` says out loud why it could not draw.

    :param frame: importance table with ``feature`` and ``importance``.
    :param title: the figure's title.
    :param src: plate folder; the bundle goes under ``<src>/results``.
    :param name: bundle name.
    :param top: how many features to show.
    :returns: the folder written, or None.
    """
    from .figures.headless import application
    from .figure_sink import publish_file

    application_object, refusal = application()
    if application_object is None:
        print(refusal)
        return None

    from .qt.widgets.fast_plots import FastPlot
    from .figures.bundle import save as write_bundle

    shown = frame.head(int(top))
    plot = FastPlot(title=title, x_label="Importance", y_label="")
    try:
        plot.resize(1200, max(420, 34 * len(shown) + 140))
        # THE HOUSE RULE: everything grey except what the sentence is about.
        # The sentence here is "these are the features that matter", so the
        # leading three carry the accent and the rest are the context they
        # are being compared against.
        if not plot.add_ranked_bars(list(shown['feature']),
                                    list(shown['importance']),
                                    highlight=3, descending=False):
            return None
        application_object.processEvents()
        folder = write_bundle(_figure_folder(src, save), name,
                              render=plot.export,
                              data=plot.ranked_frame(), groups=None,
                              unit="feature",
                              settings={"top_features": int(top),
                                        "figure": name})
    finally:
        plot.deleteLater()
    _announce_the_bundle(folder, title)
    return folder


def _save_importance_csv(df, src, filename):
    """Write an importance table to ``<src>/results/<filename>``.

    :param df: Importance DataFrame with ``feature`` / ``importance``.
    :param src: Plate folder the explained model was scored from.
    :param filename: Basename of the CSV to write.
    :returns: The full path written.
    """
    results_loc = os.path.join(src, 'results')
    os.makedirs(results_loc, exist_ok=True)
    out_path = os.path.join(results_loc, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path

def interpret_vision_model(settings=None):
    """Explain a spacr vision-model score using RF, permutation and SHAP importance, with per-compartment / per-channel radar plots.

    Merges per-object measurements from ``measurements.db`` with a CSV of
    predicted scores, runs any combination of RF feature importance,
    permutation importance and SHAP over the top features, then
    aggregates SHAP contributions into compartment and channel radar
    plots so you can see which region (cell / nucleus / pathogen /
    cytoplasm) and which fluorescence channel drives the model.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.set_interpret_vision_model_defaults`.
        Key entries:

        - ``src`` — folder containing ``measurements/measurements.db``.
        - ``scores`` — CSV of per-object predictions to explain.
        - ``score_column`` — column of ``scores`` holding the score.
        - ``tables`` — DB tables to merge (default
          ``['cell','nucleus','pathogen','cytoplasm']``).
        - ``feature_importance`` / ``permutation_importance`` / ``shap``
          — enable each explainer.
        - ``top_features`` — cap on features shown.
        - ``nuclei_limit`` / ``pathogen_limit`` — object-count caps.
        - ``n_jobs``, ``save``.

    :returns: The merged per-object DataFrame — the measurement tables
        joined to the scores CSV — that the explainers were fitted on.
        Radar and importance plots are rendered, and with ``save=True``
        importance CSVs are written alongside the DB, as side effects.

    Example:
        .. code-block:: python

            from spacr.ml import interpret_vision_model
            interpret_vision_model({
                'src': '/data/plate01',
                'scores': '/data/plate01/results/pred.csv',
                'score_column': 'pred',
                'shap': True, 'top_features': 30,
            })

    See Also:
        :func:`spacr.submodules.interpret_vision_model` — legacy /
        alternative entry point returning a dict of importance
        DataFrames instead of the merged measurements.
    """
    if settings is None:
        settings = {}
    # io._results_to_csv has the signature (src, df, df_well) and writes
    # cells.csv / wells.csv; it was being called as (df, filename=...), which
    # raised TypeError on every save=True run. The importance tables get their
    # own writer, _save_importance_csv, which follows the same <src>/results
    # convention.
    from .io import (_read_and_merge_data, _report_fan_out, JoinFanOut,
                     TimelapseKeyMismatch)
    from .predictions import crop_name_metadata
    from .settings import set_interpret_vision_model_defaults
    from .utils import save_settings, _time_column

    settings = set_interpret_vision_model_defaults(settings)
    save_settings(settings, name='interperate_vision_model', show=True)

    # Radar plot for individual and combined values, in pyqtgraph.
    def create_extended_radar_plot(values, labels, title):
        """Draw a filled radar for ``values`` labelled by ``labels``.

        A RADAR IS A POLYGON, NOT AN AXIS. This was the last figure on the
        explanation path that could not move to the screen's renderer, on
        the grounds that pyqtgraph has no polar view -- which is true and
        was never the obstacle: each label takes an angle, each value a
        radius, and `FastPlot.add_radar` draws its own rings because a
        radar read against a square grid is unreadable.
        """
        return _draw_radar_in_pyqtgraph(
            list(labels), list(values), title,
            settings['src'], _figure_name_for(title), settings['save'])

    def extract_compartment_channel(feature_name):
        """Return ``(compartment, channel)`` parsed from a feature column name."""
        # Identify compartment as the first part before an underscore
        compartment = feature_name.split('_')[0]
        
        if compartment == 'cells':
            compartment = 'cell'

        # Identify channels based on substring presence
        channels = []
        if 'channel_0' in feature_name:
            channels.append('channel_0')
        if 'channel_1' in feature_name:
            channels.append('channel_1')
        if 'channel_2' in feature_name:
            channels.append('channel_2')
        if 'channel_3' in feature_name:
            channels.append('channel_3')

        # If multiple channels are found, join them with a '+'
        if channels:
            channel = ' + '.join(channels)
        else:
            channel = 'morphology'  # Use 'morphology' if no channel identifier is found

        return (compartment, channel)

    def read_and_preprocess_data(settings):
        """Merge measurement DB tables with a scores CSV and split into ``(X, y, merged_df)``."""
        df, _ = _read_and_merge_data(
            locs=[settings['src']+'/measurements/measurements.db'], 
            tables=settings['tables'], 
            verbose=True, 
            nuclei_limit=settings['nuclei_limit'], 
            pathogen_limit=settings['pathogen_limit']
        )

        scores_df = tabular.read_table(settings['scores'])

        # Clean and align columns for merging
        df['object_label'] = df['object_label'].str.replace('o', '')

        # The join key is prcfo, spelled out as the columns it is made of --
        # the same key spacr.predictions uses to merge scores onto png_list,
        # because this is the same question: which object is this crop?
        #
        # The timepoint is part of that key. _read_and_merge_data returns one
        # row per object PER FRAME, so joining a timelapse database without it
        # matches every frame's object to every frame's score and multiplies
        # the frame by the number of frames. (That used to be masked by
        # _split_data dropping the timepoint from prcf on the way in, which
        # collapsed the frames before they got here; it no longer does.)
        join_cols = ['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']
        df_time = _time_column(df.columns)

        # A scores CSV written by apply_model_to_tar carries the crop file
        # name, and the crop file name carries all of this -- so re-derive it
        # with the writer's own parser rather than trusting the positional
        # guess process_vision_results makes. On a timelapse crop
        # (plate_well_field_time_object) that guess reads the TIMEPOINT as the
        # object id, so its 'object' column is simply wrong there.
        name_col = next((c for c in ('path', 'png_path', 'file_name')
                         if c in scores_df.columns), None)
        if name_col is not None:
            parsed = crop_name_metadata(scores_df[name_col],
                                        timelapse=df_time is not None)
            for col in parsed.columns:
                if col != 'prcfo':
                    scores_df[col] = parsed[col]

        if 'object_label' not in scores_df.columns:
            scores_df['object_label'] = scores_df['object']

        # Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type
        df['object_label'] = df['object_label'].str.replace('o', '').astype(str)

        scores_time = _time_column(scores_df.columns)
        if df_time is not None and scores_time is not None:
            if df_time != scores_time:
                scores_df = scores_df.rename(columns={scores_time: df_time})
            join_cols = join_cols + [df_time]
        elif df_time is not None or scores_time is not None:
            raise TimelapseKeyMismatch(
                f"{settings['scores']} and the measurements database disagree "
                f"about the timepoint: the scores have {scores_time!r} and the "
                f"objects have {df_time!r}. One of the two was produced by a "
                f"non-timelapse run, so there is no timepoint to join on, and "
                f"joining without it would match every frame's object to every "
                f"frame's score. Re-score the dataset, or supply a scores file "
                f"that carries the crop file name so the timepoint can be read "
                f"off it.")

        # Ensure all join columns have the same data type in both DataFrames
        df[join_cols] = df[join_cols].astype(str)
        scores_df[join_cols] = scores_df[join_cols].astype(str)

        # Select only the necessary columns from scores_df for merging
        scores_df = scores_df[join_cols + [settings['score_column']]]

        # Now merge DataFrames.
        #
        # The key contract is many-to-one — one score per object — and it is
        # spelled out, because _report_fan_out does NOT enforce it here. That
        # was the claim this comment used to make and it is false: the check
        # is `len(merged) <= len(left)`, which is only equivalent to the
        # cardinality contract for a LEFT join. This join is INNER, so scored
        # objects fanning out and unscored objects dropping out cancel in the
        # row count. Four objects, a scores file holding o1 twice and o2 once:
        # the merge returns three rows, three <= four, nothing is raised, and
        # o1's measurements are in the training set twice — the exact silent
        # duplication the check was added to stop.
        #
        # pandas is the thing that can actually see the duplicate key, so it
        # does the checking; the message is translated back into the one
        # _report_fan_out would have given, which names the cause (a scores
        # file written twice) and the fix (de-duplicate it) instead of saying
        # only "Merge keys are not unique in right dataset".
        try:
            merged_df = pd.merge(df, scores_df, on=join_cols, how='inner',
                                 validate='many_to_one')
        except pd.errors.MergeError as error:
            duplicated = scores_df[scores_df.duplicated(subset=join_cols,
                                                        keep=False)]
            examples = (duplicated[join_cols].drop_duplicates()
                        .head(3).to_dict('records'))
            raise JoinFanOut(
                f"{settings['scores']} holds more than one score for the same "
                f"object: {list(join_cols)} repeats "
                f"{len(duplicated[join_cols].drop_duplicates())} time(s), e.g. "
                f"{examples}. Joining it to the measurements would put those "
                f"objects into the training set once per duplicate row, so "
                f"every measurement in the result is duplicated. This usually "
                f"means the scoring step ran twice and appended a second set "
                f"of rows; de-duplicate the scores file before reading it."
            ) from error
        # Belt and braces on the row count as well: many_to_one covers a
        # duplicated scores key, this covers anything that would grow df for
        # some other reason. The scores are per object, so the join can only
        # ever shrink df (an object with no score drops out).
        _report_fan_out(df, merged_df, join_cols,
                        left_name='object', right_name='scores')

        # Model inputs come from the measurement schema. Numeric identity and
        # provenance columns (object_label, measurement_ndim, voxel sizes,
        # etc.) are not biological features.
        X = schema.model_feature_frame(
            merged_df,
            exclude=[settings['score_column']],
        )
        y = merged_df[settings['score_column']]

        return X, y, merged_df
    
    X, y, merged_df = read_and_preprocess_data(settings)
    
    # Step 1: Feature Importance using Random Forest
    # The outer guard used to read `feature_importance or feature_importance`
    # — the same key OR'd with itself — so the forest was never fitted unless
    # feature importance was explicitly requested. Permutation importance then
    # hit UnboundLocalError on `model`, and SHAP on `feature_importance_df`,
    # even though the docstring documents the three explainers as independent
    # toggles. The forest and the importance frame are shared by all three;
    # only the reporting and the CSV write belong to feature_importance itself.
    if settings['feature_importance'] or settings['permutation_importance'] or settings['shap']:
        model = RandomForestClassifier(random_state=_run_random_state(42), n_jobs=settings['n_jobs'])
        model.fit(X, y)

        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        if settings['feature_importance']:
            print(f"Feature Importance ...")
            top_feature_importance_df = feature_importance_df.head(settings['top_features'])

            # DRAWN IN PYQTGRAPH, not matplotlib. The tab and the file are
            # one scene now, so the picture in a paper is the picture on
            # screen. `add_ranked_bars` is what made this possible: twenty
            # feature names need horizontal bars, and until it existed the
            # only thing that could draw them was `plt.barh`.
            _draw_importance_in_pyqtgraph(
                feature_importance_df,
                f"Top {settings['top_features']} Features - Feature "
                f"Importance",
                settings['src'], 'feature_importance',
                settings['top_features'], settings['save'])

            if settings['save']:
                _save_importance_csv(feature_importance_df, settings['src'], 'feature_importance.csv')

    # Step 2: Permutation Importance
    if settings['permutation_importance']:
        print(f"Permutation Importance ...")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=_run_random_state(42), n_jobs=settings['n_jobs'])
        perm_importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean})
        perm_importance_df = perm_importance_df.sort_values(by='importance', ascending=False)
        top_perm_importance_df = perm_importance_df.head(settings['top_features'])

        # PYQTGRAPH, for the reason given at the feature-importance chart.
        _draw_importance_in_pyqtgraph(
            perm_importance_df,
            f"Top {settings['top_features']} Features - Permutation "
            f"Importance",
            settings['src'], 'permutation_importance',
            settings['top_features'], settings['save'])

        if settings['save']:
            _save_importance_csv(perm_importance_df, settings['src'], 'permutation_importance.csv')

    # Step 3: SHAP Analysis
    if settings['shap']:
        import shap

        print(f"SHAP Analysis ...")

        # Select top N features based on Random Forest importance and fit the model on these features only
        top_features = feature_importance_df.head(settings['top_features'])['feature']
        X_top = X[top_features]

        # Refit the model on this subset of features
        model = RandomForestClassifier(random_state=_run_random_state(42), n_jobs=settings['n_jobs'])
        model.fit(X_top, y)

        # Sample a smaller subset of rows to speed up SHAP
        if settings['shap_sample']:
            # int(len/100) floors to 0 for any experiment with fewer than
            # 100 objects, which handed shap an empty background AND an
            # empty matrix to explain -> IndexError. Clamp to at least one
            # row; for >=100 objects the clamp is a no-op.
            sample = max(1, min(int(len(X_top) / 100), len(X_top)))
            X_sample = X_top.sample(sample, random_state=_run_random_state(42))
        else:
            X_sample = X_top

        # Initialize SHAP explainer with the same subset of features
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample, max_evals=1500)

        # THE SUMMARY, IN PYQTGRAPH. `shap.summary_plot` draws into a
        # matplotlib figure it makes itself, so it cannot be handed a
        # pyqtgraph scene -- it was the last thing on this path keeping the
        # second renderer alive. The chart is a beeswarm: one row per
        # feature, every sample's contribution as a point, coloured by that
        # sample's own value for the feature. `FastPlot.add_beeswarm` draws
        # exactly that, so the saved file and the tab are one picture.
        _draw_shap_summary_in_pyqtgraph(
            shap_values, X_sample, settings['src'], 'shap_summary',
            settings['top_features'], settings['save'])

        # Convert SHAP values to a DataFrame for easier manipulation
        shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        
        # Apply the function to create MultiIndex columns with compartment and channel
        shap_df.columns = pd.MultiIndex.from_tuples(
            [extract_compartment_channel(feat) for feat in shap_df.columns], 
            names=['compartment', 'channel']
        )
        
        # Aggregate SHAP values by compartment and channel
        shap_features = shap_df.abs().T
        compartment_mean = (
            shap_features.groupby(level='compartment').mean().mean(axis=1))
        channel_mean = (
            shap_features.groupby(level='channel').mean().mean(axis=1))

        # Calculate combined importance for each pair of compartments and channels
        combined_compartment = {}
        for i, comp1 in enumerate(compartment_mean.index):
            for comp2 in compartment_mean.index[i+1:]:
                combined_compartment[f"{comp1} + {comp2}"] = shap_df.loc[:, (comp1, slice(None))].abs().mean().mean() + \
                                                              shap_df.loc[:, (comp2, slice(None))].abs().mean().mean()
        
        combined_channel = {}
        for i, chan1 in enumerate(channel_mean.index):
            for chan2 in channel_mean.index[i+1:]:
                combined_channel[f"{chan1} + {chan2}"] = shap_df.loc[:, (slice(None), chan1)].abs().mean().mean() + \
                                                          shap_df.loc[:, (slice(None), chan2)].abs().mean().mean()

        # Prepare values and labels for radar charts
        all_compartment_importance = list(compartment_mean.values) + list(combined_compartment.values())
        all_compartment_labels = list(compartment_mean.index) + list(combined_compartment.keys())

        all_channel_importance = list(channel_mean.values) + list(combined_channel.values())
        all_channel_labels = list(channel_mean.index) + list(combined_channel.keys())

        # Create radar plots for compartments and channels
        create_extended_radar_plot(all_compartment_importance, all_compartment_labels, "SHAP Importance by Compartment (Individual and Combined)")
        create_extended_radar_plot(all_channel_importance, all_channel_labels, "SHAP Importance by Channel (Individual and Combined)")
    
    return merged_df


# Backward compatibility for the misspelling published in earlier releases.
interperate_vision_model = interpret_vision_model
