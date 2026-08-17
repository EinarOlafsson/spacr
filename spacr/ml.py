"""Classical machine-learning and regression analysis pipelines."""

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

from . import schema
from .openmp_guard import single_threaded_openmp, guarded_n_jobs  # see spacr/openmp_guard.py — duplicate libomp is fatal
from .plot import save_figure  # every kept figure goes through the format/DPI preference

LOG = logging.getLogger("spacr.ml")

from scipy.stats import kstest, normaltest

import matplotlib

# Only demote to Agg when there is genuinely nowhere to draw. Doing it
# unconditionally at import time silently killed inline plotting for anyone
# who imported spacr.ml in a notebook, because it overrode a backend the user
# had already selected. spacr.cli and both GUIs set their own backend.
if not (sys.platform.startswith(('win', 'darwin')) or os.environ.get('DISPLAY')):
    matplotlib.use('Agg')

import warnings


def _graph_sequencing_stats(settings):
    """Resolve the sequencing threshold helper through one testable seam."""
    # Keep this lazy to avoid expanding ml.py's already-heavy import graph,
    # while giving callers and tests a stable dependency boundary. Importing
    # the helper directly inside perform_regression made it impossible to
    # substitute reliably after package lazy-loader tests replaced a module
    # object in sys.modules.
    from .sequencing import graph_sequencing_stats
    return graph_sequencing_stats(settings)


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

def perform_mixed_model(y, X, groups, alpha=None):
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
    :returns: Fitted ``statsmodels`` ``MixedLMResults``.
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

    return MixedLM(y, X, groups=groups).fit()

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

def prepare_formula(dependent_variable, random_row_column_effects=False,
                    block_screen=False):
    """Build the fixed-effects formula for the gRNA / gene regression.

    Both branches regress the response on ``fraction:grna`` and
    ``gene_fraction:gene``. By default ``rowID`` and ``columnID`` are
    added as *fixed* effects; with ``random_row_column_effects=True``
    they are left out of the formula because :func:`fit_mixed_model`
    puts plate, row and column into its random structure instead.

    ``block_screen`` adds ``screenID`` (instruction 122). Two screens sharing
    a guide library can be stacked into one frame and fitted together, which
    is worth twice the wells -- but only if the screen itself is in the model.
    Without the term, a systematic difference between the two experiments is
    charged to whichever guides happen to be over-represented in one of them.

    IT DEFAULTS TO OFF, and the caller decides, because a single-screen
    project's ``screenID`` column has ONE value. A constant term makes the
    design rank-deficient: statsmodels answers with a pseudo-inverse rather
    than refusing, so the run would appear to succeed and hand back standard
    errors that mean nothing. :func:`screen_is_blockable` is the check.

    :param dependent_variable: Name of the response column.
    :param random_row_column_effects: Drop the fixed ``rowID`` /
        ``columnID`` terms. Default ``False``.
    :param block_screen: Add ``screenID`` as a fixed effect. Default
        ``False``.
    :returns: The formula string.
    """
    from .schema import SCREEN_KEY

    screen = f' + {SCREEN_KEY}' if block_screen else ''
    if random_row_column_effects:
        # Random effects for row and column + gene weighted by gene_fraction + grna weighted by fraction
        return (f'{dependent_variable} ~ fraction:grna + '
                f'gene_fraction:gene{screen}')
    return (f'{dependent_variable} ~ fraction:grna + gene_fraction:gene + '
            f'rowID + columnID{screen}')


def screen_is_blockable(df) -> bool:
    """Whether ``screenID`` can be a term in this frame's design.

    True only when the column exists and carries more than one distinct
    value. A single-screen project is the normal case and must be untouched
    by instruction 122: it has no screenID at all, or one value, and either
    way the term would be a constant column.

    The same rule :func:`spacr.measurement_scan._dummy_block` applies, stated
    once for the formula path so a frame cannot be blocked on by one and not
    the other.
    """
    from .schema import SCREEN_KEY

    if df is None or SCREEN_KEY not in getattr(df, 'columns', ()):
        return False
    return int(df[SCREEN_KEY].astype(str).nunique(dropna=True)) > 1

def fit_mixed_model(df, formula, dst):
    """Fit a mixed-effects model with plate/row/column random structure and return coefficients.

    :param df: DataFrame containing the model variables plus
        ``plateID``, ``rowID`` and ``columnID``.
    :param formula: Formula string for fixed effects.
    :param dst: Destination for the residual histogram PDF.
    :returns: ``(mixed_model, coef_df)`` — the fitted results object
        and a DataFrame with columns ``feature``, ``coefficient``,
        ``p_value``.
    """
    from .plot import plot_histogram

    """Fit the mixed model with plate, row_name, and columnID as random effects and return results."""
    # Specify random effects for plate, row, and column
    model = smf.mixedlm(formula, 
                        data=df, 
                        groups=df['plateID'], 
                        re_formula="1 + rowID + columnID", 
                        vc_formula={"rowID": "0 + rowID", "columnID": "0 + columnID"})
    
    mixed_model = model.fit()

    # Plot residuals
    df['residuals'] = mixed_model.resid
    plot_histogram(df, 'residuals', dst=dst)

    # Return coefficients and p-values
    coefs = mixed_model.params
    p_values = mixed_model.pvalues

    coef_df = pd.DataFrame({
        'feature': coefs.index,
        'coefficient': coefs.values,
        'p_value': p_values.values
    })
    
    return mixed_model, coef_df

def check_and_clean_data(df, dependent_variable):
    """Prepare the merged count / score frame for model fitting.

    Drops rows with a missing ``fraction`` or dependent variable, casts
    the identifier columns to categorical and reports (without dropping)
    collinear columns via VIF. The returned frame keeps only
    ``fraction``, the dependent variable, ``gene``, ``grna``, ``prc``,
    ``plateID``, ``rowID``, ``columnID`` and ``cell_count`` when present,
    plus a computed ``gene_fraction`` column: the sum of the gene's gRNA
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

def minimum_cell_simulation(settings, num_repeats=10, sample_size=100, tolerance=0.02, smoothing=10, increment=10):
    """
    Estimate the minimum number of cells per well needed for a stable well mean.

    For the wells with the most objects, repeatedly subsamples cells at
    increasing sample sizes and records the mean absolute difference from
    the well's full mean. Plots the smoothed curve with a ±1 s.d. band,
    marks the elbow point (or ``settings['min_cell_count']`` when it is
    set) and writes ``results/cell_min_threshold.pdf`` next to
    ``settings['count_data'][0]``.

    :param settings: Requires ``score_data`` (CSV path or list of paths),
        ``score_column``, ``tolerance`` (int percent or float fraction),
        ``min_cell_count`` and ``count_data``.
    :param num_repeats: Subsamples drawn per sample size. Default ``10``.
    :param sample_size: Number of wells, taken largest-first by cell
        count, to simulate. Default ``100``.
    :param tolerance: Unused; the tolerance applied is
        ``settings['tolerance']``.
    :param smoothing: Rolling-window width used to smooth the curve.
    :param increment: Step between the simulated sample sizes.
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
        df = pd.read_csv(score_data)
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
        original_mean = group[settings['score_column']].mean()  # Original full-well mean
        max_cells = len(group)
        sample_sizes = np.arange(2, max_cells + 1, increment)  # Sample sizes from 2 to max cells

        # Iterate over sample sizes and compute absolute difference
        for sample_size in sample_sizes:
            abs_diffs = []

            # Perform multiple random samples to reduce noise
            for _ in range(num_repeats):
                sample = group.sample(n=sample_size, replace=False)
                sampled_mean = sample[settings['score_column']].mean()
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
        prc: tolerance_fraction * group[settings['score_column']].mean()  # Compute % of original mean
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

    # Plot the mean absolute difference with standard deviation as shaded area
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        summary_df['sample_size'], summary_df['smoothed_mean_abs_diff'], color='teal', label='Smoothed Mean Absolute Difference'
    )
    ax.fill_between(
        summary_df['sample_size'],
        summary_df['smoothed_mean_abs_diff'] - summary_df['std_abs_diff'],
        summary_df['smoothed_mean_abs_diff'] + summary_df['std_abs_diff'],
        color='teal', alpha=0.3, label='±1 Std. Dev.'
    )

    if settings['min_cell_count'] is None:
        # Mark the elbow point (inflection) on the plot
        ax.axvline(elbow_point['sample_size'], color='black', linestyle='--', label='Elbow Point')
    else:
        ax.axvline(settings['min_cell_count'], color='black', linestyle='--', label='Elbow Point')

    # Formatting the plot
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Mean Absolute Difference vs. Sample Size with Standard Deviation')
    ax.legend().remove()

    # Save the plot if a destination is provided
    dst = os.path.dirname(settings['count_data'][0])
    if dst is not None:
        fig_path = os.path.join(dst, 'results')
        os.makedirs(fig_path, exist_ok=True)
        fig_file_path = os.path.join(fig_path, 'cell_min_threshold.pdf')
        fig_file_path = save_figure(fig, fig_file_path,
                                    bbox_inches='tight')
        print(f"Saved {fig_file_path}")

    plt.show()
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
#: directly (statsmodels). ``mixed`` is here too, and its variance components
#: are dropped below - they are not effects on the response.
_STATSMODELS_COEF_TYPES = (
    'ols', 'wls', 'rlm', 'huber', 'glm', 'poisson', 'logit', 'probit',
    'quasi_binomial', 'quantile', 'mixed', 'horseshoe',
)

#: Backends that expose ``coef_`` (scikit-learn) and no inference of their own.
_SKLEARN_COEF_TYPES = ('ridge', 'lasso', 'elasticnet')


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
    # Control names as TEXT. A gene id like 233460 is a perfectly good
    # negative_control, and a settings file round-trips it back as the INT
    # 233460 -- at which point `nc in row['feature']` raises "'in <string>'
    # requires string as left operand, not int" and the whole regression dies
    # on a value that was legal the moment it was typed into the GUI.
    nc_name = '' if nc is None else str(nc)
    pc_name = '' if pc is None else str(pc)
    coef_df['condition'] = coef_df.apply(
        lambda row: 'nc' if nc_name and nc_name in str(row['feature']) else
                    'pc' if pc_name and pc_name in str(row['feature']) else
                    ('control' if row['grna'] in controls else 'other'),
        axis=1,
    )

    return coef_df[~coef_df['feature'].str.contains('row|column')]

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


def _validate_poisson_response(y, X=None, minimum_samples=MIN_POISSON_SAMPLES):
    """Validate a response before fitting a Poisson GLM.

    Poisson endog must contain finite, non-negative integer counts. At least
    eight observations and one residual degree of freedom are required so
    family detection and coefficient inference are not performed on an
    undersized or saturated design.

    :param y: One-dimensional count response.
    :param X: Optional design matrix used to determine the parameter count.
    :param minimum_samples: Absolute observation floor.
    :returns: The validated response as a one-dimensional float array.
    :raises ValueError: If the response or sample size is invalid.
    """
    try:
        counts = np.asarray(y, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Poisson regression requires numeric count data."
        ) from exc

    if not np.isfinite(counts).all():
        raise ValueError(
            "Poisson regression requires finite count data; remove or impute "
            "NaN and infinite response values before fitting."
        )
    if np.any(counts < 0):
        raise ValueError(
            "Poisson regression requires non-negative count data; negative "
            "response values are not valid counts."
        )
    if not np.all(np.isclose(counts, np.rint(counts), rtol=0, atol=1e-8)):
        raise ValueError(
            "Poisson regression requires integer count data; use a continuous "
            "response model for fractional values."
        )
    if not np.any(counts > 0):
        raise ValueError(
            "Poisson regression requires at least one positive count; an "
            "all-zero response cannot estimate effects."
        )

    n_parameters = 0
    if X is not None:
        x_shape = np.shape(X)
        if not x_shape or x_shape[0] != counts.size:
            raise ValueError(
                "Poisson regression requires X and y to contain the same "
                f"number of observations; got {x_shape[0] if x_shape else 0} "
                f"and {counts.size}."
            )
        n_parameters = 1 if len(x_shape) == 1 else int(x_shape[1])

    required = max(int(minimum_samples), n_parameters + 1)
    if counts.size < required:
        raise ValueError(
            "Poisson regression has too few observations: "
            f"received {counts.size}, but at least {required} are required "
            f"for {n_parameters} model parameters."
        )
    return counts


def pick_glm_family_and_link(y):
    """Select the GLM family and link that suit the response.

    Used by ``regression_type='glm'`` to choose a family from the data rather
    than from the user.

    :param y: Response vector.
    :returns: A ``statsmodels`` family instance with its link set.
    :raises ValueError: only through :func:`_validate_poisson_response`, when
        the response looks like counts but cannot be one.
    """
    values = np.asarray(y, dtype=float).reshape(-1)

    if np.all((values == 0) | (values == 1)):
        print("Binary data detected. Using Binomial family with Logit link.")
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
        print("Data strictly between 0 and 1. Using Binomial family with "
              "Logit link; consider regression_type='beta', which models the "
              "variance of a bounded response directly, or 'quasi_binomial' "
              "if the wells are overdispersed.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    elif (values >= 0).all() and (values <= 1).all():
        print("Data between 0 and 1 (including boundaries). Using Quasi-Binomial.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    if (values >= 0).all() and np.all(values.astype(int) == values):
        # Family selection may be used for a short preview without fitting.
        # The actual GLM boundary below enforces the sample/design minimum.
        _validate_poisson_response(values, minimum_samples=1)
        print("Count data detected. Using Poisson with Log link.")
        return sm.families.Poisson(link=sm.families.links.Log())

    stat, p_value = normaltest(values)
    print(f"Normality test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Normally distributed data detected. Using Gaussian with Identity link.")
        return sm.families.Gaussian(link=sm.families.links.Identity())

    if ((values > 0).all()
            and kstest(values, 'invgauss', args=(1,)).pvalue > 0.05):
        print("Inverse Gaussian distribution detected. Using InverseGaussian with Log link.")
        return sm.families.InverseGaussian(link=sm.families.links.Log())

    if (values >= 0).all():
        print("Overdispersed count data detected. Using Negative Binomial with Log link.")
        return sm.families.NegativeBinomial(link=sm.families.links.Log())

    print("Using default Gaussian family with Identity link.")
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
from .regression_spec import (NO_P_VALUE_TYPES,        # noqa: F401
                              REGRESSION_SETTINGS_USED,
                              REGRESSION_TYPES,
                              RUN_LEVEL_SETTINGS,
                              UNSUPPORTED_REGRESSION_TYPES,
                              _MODEL_LEVEL_DEFAULTS,
                              _RUN_LEVEL_DEFAULTS)


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


def _reject_unused_settings(regression_type, supplied):
    """Raise when a setting the chosen backend cannot read was set anyway.

    ``supplied`` maps a setting name to ``(value, default)``. A value equal to
    its default is "not asked for" and passes; anything else must appear in
    :data:`REGRESSION_SETTINGS_USED` for this type.

    Comparing against the default is what makes this usable from a GUI, which
    posts every widget on the panel whether or not the user touched it.

    :param regression_type: The backend about to be fitted.
    :param supplied: ``{name: (value, default)}`` for the policed settings.
    :raises ValueError: naming the setting, the type and the alternative.
    """
    used = REGRESSION_SETTINGS_USED.get(regression_type, ())
    for name, (value, default) in supplied.items():
        if name in used or value == default:
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
}


def regression_model(X, y, regression_type='ols', groups=None, alpha=1.0,
                     cov_type=None, weights=None, l1_ratio=0.5, quantile=0.5,
                     hinge_threshold=None, huber_t=1.345, exposure=None):
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
    ==================================  ========================================================

    Settings a backend cannot read are REFUSED, not ignored — see
    :data:`REGRESSION_SETTINGS_USED`.

    :param X: Design matrix (DataFrame; column names become feature names).
    :param y: Response variable.
    :param regression_type: One of :data:`REGRESSION_TYPES`.
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
    :param huber_t: Huber tuning constant for ``rlm``/``huber``, in units of
        the estimated residual scale. 1.345 gives 95% efficiency under
        normality.
    :param exposure: Per-observation exposure (the well's cell count) used as
        ``offset(log(exposure))`` by ``horseshoe`` and by ``poisson`` (and by
        ``glm`` when it auto-selects a Poisson family).
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
    }
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
        family = pick_glm_family_and_link(y)
        if isinstance(family, sm.families.Poisson):
            _validate_poisson_response(y, X)
            # Same exposure the explicit 'poisson' branch uses. A family chosen
            # BY the data must be fitted the same way as one chosen by name, or
            # 'glm' and 'poisson' silently disagree on the same response.
            return sm.GLM(y, X, family=family, offset=_poisson_offset()).fit(
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
        return sm.GLM(y, X, **kwargs).fit(
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

    def _horseshoe():
        return _fit_horseshoe_poisson(X, y, exposure)

    model_map = {
        'ols':    lambda: sm.OLS(y, X).fit(cov_type=cov_type) if cov_type else sm.OLS(y, X).fit(),
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
        'mixed':  lambda: perform_mixed_model(y, X, groups),
        'lasso':  lambda: _find_best_alpha('lasso') if use_auto_alpha
                          else Lasso(alpha=alpha, max_iter=10000).fit(X, y_flat),
        'ridge':  lambda: _find_best_alpha('ridge') if use_auto_alpha
                          else Ridge(alpha=alpha).fit(X, y_flat),
        'elasticnet': lambda: _find_best_alpha('elasticnet') if use_auto_alpha
                          else ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                          max_iter=10000).fit(X, y_flat),
        'hinge':  _hinge,
        'horseshoe': _horseshoe,
    }

    model = model_map[regression_type]()

    if regression_type in ['glm', 'poisson']:
        llf_model = model.llf
        llf_null = model.null_deviance / -2
        print(f"McFadden's R²: {1 - (llf_model / llf_null):.4f}")
        print(model.summary())

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

    counts = _validate_poisson_response(y, X)
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
        not a mixed model, or with a setting the mixed model cannot read.
    """
    if not settings.get('random_row_column_effects', False):
        return settings

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
    })

    if reg_type != 'mixed':
        print(f"random_row_column_effects=True: fitting 'mixed' rather than "
              f"{reg_type!r}, and naming the results folder for it.")
    settings['regression_type'] = 'mixed'
    return settings


def _mixed_model_groups(df, dependent_variable, model_index):
    """Return the random-intercept grouping for ``regression_type='mixed'``.

    The cluster is the PLATE, always. A random intercept has to sit at a level
    ABOVE the unit the covariates vary at, and in this design the covariates
    (``fraction``, ``gene_fraction``) are properties of the WELL. Grouping on
    the well therefore asks the model to explain a well-level covariate with a
    well-level random intercept, and the answer it gives is zero - which is
    what ``groups = df['prc']`` did, silently, on every mixed run:

    * with an aggregated response there is one value per well, so the random
      intercept is exactly confounded with the residual and every fixed effect
      came back at ~1e-11 with p ~ 1;
    * with ``agg_type=None`` the frame is the CROSS PRODUCT of the well's
      gRNAs with the well's cells, so a well does have several response
      values - but its within-well variation in ``fraction:grna`` is an
      artefact of that cross product and carries no signal about the
      response. A well-level random intercept makes GLS weight exactly that
      artefact (1/sigma_e^2) far above the between-well contrasts that hold
      the biology (1/(sigma_u^2 + sigma_e^2/n)), and the fixed effects are
      dragged to zero again. Measured on a 96-well synthetic screen with a
      planted +0.45 gene effect: OLS recovered 0.31, the well-grouped mixed
      model returned 0.003 with p = 0.79.

    Both failures WROTE results.csv and neither said anything.

    A single plate leaves one cluster, which is not a random effect at all, so
    that case is refused rather than fitted.

    :param df: The cleaned long-format frame.
    :param dependent_variable: Response column name, named in the refusal so
        the message points at the run the user actually configured.
    :param model_index: Row index patsy kept, so the returned vector aligns
        with the design matrix row for row.
    :returns: Series of plate ids, one per design row.
    :raises ValueError: when the screen has a single plate, naming the three
        ways out.
    """
    plates = df.loc[model_index, 'plateID']
    n_plates = plates.nunique()
    if n_plates > 1:
        print(f"Mixed model: grouping on plateID ({n_plates} plates). Wells "
              f"are the unit the gRNA fractions vary at, so the plate is the "
              f"level a random intercept can describe.")
        return plates

    raise ValueError(
        f"a mixed model needs at least two clusters and this screen has one "
        f"plate. The random intercept has to sit above the well - the well is "
        f"where 'fraction' and 'gene_fraction' vary - so with a single plate "
        f"there is nothing left for it to describe, and it would return a "
        f"coefficient of zero for every gRNA against "
        f"{dependent_variable!r}. Either set random_row_column_effects=True, "
        f"which fits row and column variance components and is the mixed "
        f"model a one-plate screen supports; or pass every plate of the "
        f"experiment in score_data/count_data so the plate effect has "
        f"something to vary against; or use 'ols' with cov_type='HC3'.")


def _write_regression_qc(model, X, y, df, dst, *, coef_df=None,
                         regression_type=None, volcano_path=None):
    """Write the full QC suite for a fit into ``<dst>/regression_qc/``.

    :func:`spacr.regression_qc.regression_qc_report` had no production caller:
    twenty-three tested diagnostic panels -- scale-location (the variance
    homogeneity panel the maintainer asked for by name), residuals-vs-fitted,
    Q-Q, leverage, Cook's distance, DFFITS, VIF, condition number, the p-value
    histogram, calibration -- existed, were tested, and were never produced by
    an actual run. This is the hook.

    It lives here rather than in :func:`perform_regression` because this is the
    only scope where the fitted design exists: ``regression`` returns
    ``(model, coef_df, regression_type)`` and drops ``X`` and ``y`` on the way
    out, so a caller further up has no design matrix to hand the report.

    **Weights are deliberately not forwarded.** ``regression`` passes cell
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
    from .plot import figure_output_preferences
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

    # The panels are figures the user keeps, so they follow the same format
    # preference as every other figure the pipeline writes.
    fmt, _dpi = figure_output_preferences()
    try:
        return regression_qc_report(
            model, X, y, dst, metadata=metadata, coef_df=coef_df,
            regression_type=regression_type, volcano_path=volcano_path,
            fmt=fmt, verbose=True)
    except Exception as error:                      # noqa: BLE001 - advisory
        # A diagnostic that fails must never destroy a fit that already
        # succeeded and cost an hour. The report itself already downgrades a
        # failing panel to FAILED; this catches the rarer case where the
        # report as a whole cannot be built.
        print(f"Regression QC report could not be written: "
              f"{type(error).__name__}: {error}")
        return None


def regression(df, csv_path, dependent_variable='predictions', regression_type=None, alpha=1.0,
               random_row_column_effects=False, nc='233460', pc='220950', controls=None,
               dst=None, cov_type=None, plot=False, l1_ratio=0.5, quantile=0.5,
               hinge_threshold=None, hinge_n_boot=200, huber_t=1.345, qc=True,
               legacy_volcano=False):
    """Run the full regression pipeline: clean, fit, extract coefficients, optional volcano plot.

    :param df: Long-format DataFrame with gRNA/gene fractions and the
        dependent variable.
    :param csv_path: Path used to derive the volcano-plot filename.
    :param dependent_variable: Response column name. Default
        ``'predictions'``.
    :param regression_type: Model type; auto-selected via
        :func:`check_distribution` when ``None``.
    :param alpha: Regularisation strength for penalised models.
    :param random_row_column_effects: If True, fit a mixed model with
        random row/column effects.
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
    :param qc: Write the regression QC suite into ``<dst>/regression_qc/``.
    :param legacy_volcano: also draw the ORIGINAL matplotlib volcano.
        Default ``False``. The interactive one is far faster and the
        house-style panel is what a run now produces; drawing both gives two
        volcanoes in two idioms on the same grid.
        Requires ``dst`` and a design matrix, so it is skipped for the mixed
        branch and when no destination was given.
    :returns: ``(model, coef_df, regression_type)``.
    """

    if controls is None:
        controls = ['']
    from .plot import volcano_plot, plot_histogram

    # create_volcano_filename names a quantile run by the quantile it fitted
    # rather than by the model name, because two quantiles of the same screen
    # are two different results that must not overwrite each other. That used
    # to be alpha, which is no longer the quantile.
    volcano_path = create_volcano_filename(
        csv_path, regression_type,
        quantile if regression_type == 'quantile' else alpha, dst)

    if regression_type is None:
        regression_type = check_distribution(df[dependent_variable])

    print(f"Using regression type: {regression_type}")

    df = check_and_clean_data(df, dependent_variable)

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

    if random_row_column_effects:
        regression_type = 'mixed'
        formula = prepare_formula(dependent_variable,
                                  random_row_column_effects=True,
                                  block_screen=block_screen)
        mixed_model, coef_df = fit_mixed_model(df, formula, dst)
        model = mixed_model
    else:
        formula = prepare_formula(dependent_variable,
                                  random_row_column_effects=False,
                                  block_screen=block_screen)
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
        if not _show_well_distributions(df, dependent_variable, dst,
                                        plot=plot):
            plot_histogram(y, dependent_variable, dst=dst)
            plot_histogram(df, 'fraction', dst=dst)

        # No scaling, for any type. The design this pipeline builds is
        # `fraction:grna + gene_fraction:gene + rowID + columnID`: dummies and
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
        groups = (_mixed_model_groups(df, dependent_variable, model_index)
                  if regression_type == 'mixed' else None)

        print(f'Performing {regression_type} regression')
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
            exposure=weights,
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
    if qc and qc_design is not None and dst:
        _write_regression_qc(
            model, qc_design[0], qc_design[1], df, dst,
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
    if dst:
        _show_house_style_panels(coef_df, plot=plot)
        _write_regression_sheet(coef_df, dst)

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

    return model, coef_df, regression_type


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
        from .plot import save_figure

        # The sheet is THE publication figure of the run, so it is the last
        # one that should ignore the format and resolution the user chose.
        path = save_figure(sheet.figure, path, bbox_inches='tight')
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

def save_summary_to_file(model, file_path='summary.csv'):
    """
    Write ``model.summary().as_text()`` to ``file_path`` as plain text.

    Despite the default ``'summary.csv'`` name, the content is the
    statsmodels text summary, never CSV.

    :param model: Fitted statsmodels results object.
    :param file_path: Destination path. Default ``'summary.csv'``.
    """
    # Get the summary as a string
    summary_str = model.summary().as_text()

    # Save it as a plain text file or CSV
    with open(file_path, 'w') as f:
        f.write(summary_str)


def _split_prc(text):
    """Return ``(plateID, rowID, columnID)`` for one ``prc`` well key.

    Parsed **right to left**, for exactly the reason
    :func:`spacr.schema.parse_prcf` is: the plate id is the only component
    that may itself contain the key separator, and it is the leftmost one.

    Three sites in this module used to spell this as

    .. code-block:: python

        df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)

    which has two failure modes on a plate called ``'exp1_plate1'``, and the
    second one is silent:

    * when every row carries the extra underscore the split returns four
      columns against three keys and pandas raises ``ValueError: Columns must
      be same length as key`` — in :func:`perform_regression` that lands
      inside the ``try`` that writes the QC CSVs, so ``grna_well.csv`` and
      ``well_grna.csv`` were simply never written and the only trace was a
      bare ``print(e)``;
    * when only *some* plates carry it the split still returns four columns,
      so ``columnID`` is filled with the **row** token for every well of every
      other plate and the per-well QC counts are grouped on nonsense.

    The row and column are returned exactly as they appear — nothing is
    canonicalised, because the caller rebuilds ``prc`` from these columns and
    a rewritten token would change the identity rows are joined on.

    THE PLATE IS UNESCAPED, which is the one exception and is not a
    canonicalisation: :func:`spacr.schema.compose_prc` percent-escapes the
    plate on the way in, so returning it raw would hand back ``'a%5Fb'`` for a
    plate named ``'a_b'``. :func:`spacr.schema.parse_prcf` already unescapes,
    so leaving it here made the two parsers disagree on exactly the keys this
    differential pair exists to protect.

    Callers in this module rebuild ``prc`` by hand and are NOT escaped, so a
    plate id holding the separator still round-trips through the
    four-component path below rather than through an escape. That asymmetry
    is real and is recorded in instruction 100.

    **Four components are not automatically an underscored plate.** A key with
    more than three components is one of two things, and they mean opposite
    things:

    * ``'exp1_plate1_r2_c12'`` — a plate id that contains the separator. The
      right-to-left rule handles it, and that is the case this function
      exists for.
    * ``'plate1_r1_c1_f1'`` — a ``prcf`` (or ``prcfo``) handed to the
      function that takes a ``prc``. That is a *caller* bug, and the old
      positional ``str.split`` at least failed loudly on it (``ValueError:
      Columns must be same length as key``). Absorbing it right to left would
      return ``('plate1_r1', 'c1', 'f1')`` — a field id in the ``columnID``
      slot and half the well in the plate — and every per-well count grouped
      on that is a plausible wrong number with nothing anywhere saying so.

    The two are told apart by the trailing pair: a ``prc``'s last two tokens
    are a row and a column, and the tokens spaCR writes for those are
    ``r<N>``/row letters and ``c<N>``/digits (or the equal-valued positional
    passthrough :func:`spacr.schema.is_positional_pair` describes). A
    ``prcf``'s trailing pair is ``(column, field)`` and a ``prcfo``'s is
    ``(field, object)``; neither can pass that test, because a ``columnID``
    is never ``'f1'`` and a ``rowID`` is never ``'c1'``. Anything else with
    more than three components is refused rather than guessed at — the
    ambiguous case fails loudly, exactly as it did before.

    A three-component key is accepted whatever its tokens look like, which is
    what the ``str.split`` this replaces did, so no key that used to parse
    stops parsing.

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


def _compose_prc_column(df):
    """``prc`` from ``plateID``/``rowID``/``columnID``, escaped as the key is.

    The vectorised counterpart of :func:`spacr.schema.compose_prc`, and the
    reason it exists: seven sites in this module built ``prc`` with a bare
    ``df['plateID'] + '_' + ...``, which is correct only while no plate id
    contains the separator or a ``%``. ``compose_prc`` escapes both, so a
    hand-joined key and a composed key were two different strings for the same
    well, and anything joining one to the other silently matched nothing.

    ESCAPING IS NOW THE ONE SPELLING (maintainer's decision, 2026-08-16).
    A plate called ``exp1_plate2`` composes to ``exp1%5Fplate2_rB_c3`` -- three
    components, always -- rather than to a four-component key separated by the
    row/column guard. Databases written before this hold the old four-component
    form, and :func:`_split_prc` still reads it: unescaping is a no-op on a key
    that carries no escape, and the guard that separates an underscored plate
    from a ``prcf`` is untouched. So old data reads and new data is
    unambiguous, which is what "accept both, write one" means here.

    :param df: frame carrying ``plateID``, ``rowID`` and ``columnID``.
    :returns: the ``prc`` series.
    """
    return (df['plateID'].astype(str).map(schema.escape_filename_component)
            + schema.KEY_SEPARATOR + df['rowID'].astype(str)
            + schema.KEY_SEPARATOR + df['columnID'].astype(str))


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
        f"auto chose the plate-blocked permutation test: {n_wells} analysed "
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

    def read(path):
        if not path:
            return None
        return correct_metadata(pd.read_csv(os.fspath(path)))

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
            resolved = score_plates
            rule = 'copied from score file'
        elif count_plates:
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
        if score is not None and not score_plates:
            score['plateID'] = next(iter(resolved))
        if count is not None and not count_plates:
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
                               well_column='prc'):
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

    :raises ValueError: when the join is empty, or keeps less than
        :data:`_MINIMUM_PAIRED_WELL_FRACTION` of the count wells.
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
        if unused_counts or unused_scores:
            print(
                f"Paired {matched} wells. "
                f"{unused_counts} sequencing well(s) and {unused_scores} "
                f"imaging well(s) have no partner and take no part in the "
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
                             guide_column='grna'):
    """Warn when a simultaneous fit is about to be run on too few wells.

    Returns the warning text, or ``None`` when the design is fine. Kept
    separate from :func:`resolve_auto_inference` because this one never
    changes what runs -- it only makes sure the user cannot miss what they
    are about to get.
    """
    try:
        n_wells = int(data[well_column].nunique())
        n_guides = int(data[guide_column].nunique())
    except (KeyError, TypeError):
        return None
    blocks = 0
    block_column = str(settings.get('guide_permutation_block', 'plateID'))
    if block_column in getattr(data, 'columns', ()):
        blocks = max(int(data[block_column].nunique()) - 1, 0)
    parameters = 1 + blocks + n_guides
    if n_wells > parameters:
        return None
    return (
        "\n"
        "  ###############################################################\n"
        "  #  WARNING: this regression is not identifiable.              #\n"
        "  ###############################################################\n"
        f"  {n_wells} analysed wells are being used to estimate "
        f"{parameters} parameters\n"
        f"  ({n_guides} guides + intercept + {blocks} block terms).\n"
        "\n"
        "  With fewer wells than parameters the fit still returns a\n"
        "  coefficient and a P value for every guide, but they are one\n"
        "  arbitrary solution out of infinitely many: refitting the same\n"
        "  data can give different numbers, and neither set is wrong.\n"
        "\n"
        "  Set inference='nonparametric' to test each guide as a\n"
        "  plate-blocked marginal association, which stays valid at any\n"
        "  width, or inference='auto' to let spaCR choose. The design\n"
        "  diagnostics written beside the results show the rank, the\n"
        "  residual degrees of freedom and the collinear guide pairs.\n")


def _run_guide_permutation_analysis(data, outcome, destination, settings):
    """Run and persist the plate-blocked marginal guide analysis.

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
    primary = settings.get('guide_primary_min_wells')
    primary = thresholds[0] if primary is None else int(primary)
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
    results = analyse_long_guide_table(
        data,
        outcomes,
        min_wells=thresholds,
        block_column=str(settings.get('guide_permutation_block', 'plateID')),
        nuisance_columns=list(settings.get('guide_nuisance_columns') or []),
        n_permutations=int(settings.get('guide_permutations', 200000)),
        random_state=int(settings.get('guide_permutation_seed', 0)),
        multiple_testing=str(settings.get('multiple_testing_method', 'fdr_bh')),
        alpha=float(settings.get('fdr_alpha', 0.05)),
        presence_threshold=float(settings.get('guide_presence_threshold', 0.0)),
        batch_size=int(settings.get('guide_permutation_batch_size', 500)),
    )
    paths = dict(save_guide_permutation_results(
        results, destination, prefix='guide_permutation'))
    if settings.get('guide_permutation_plot', True):
        single = len(outcomes) == 1
        for response in outcomes:
            for threshold in thresholds:
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

    primary_table = results.loc[
        results['minimum_wells_threshold'] == primary
    ].copy()
    # Compatibility aliases let existing table consumers display these rows
    # without hiding that the inferential quantities are marginal effects,
    # empirical P values, and already-adjusted values.
    primary_table['grna'] = primary_table['guide']
    primary_table['feature'] = (
        'fraction:grna[' + primary_table['guide'].astype(str) + ']')
    primary_table['coefficient'] = primary_table[
        'standardized_marginal_effect']
    primary_table['p_value'] = primary_table['permutation_p_value']
    primary_table['q_value'] = primary_table['adjusted_p_value']
    # THE SAME ALIASES ON THE FULL TABLE.
    #
    # results.csv on disk holds primary_table, which carries these columns,
    # while the returned output['results'] held the full multi-threshold frame,
    # which did not. One name, two shapes: a caller reading the file and a
    # caller reading the dict got different columns, and everything that
    # consumes a coefficient table -- the results panel, guide concordance,
    # the volcano, the sweep's hit counts -- raised KeyError('feature') on the
    # nonparametric path while working fine on the parametric one.
    #
    # Added rather than swapped, so a caller that genuinely wants every
    # minimum-wells threshold still has all of them.
    results = results.copy()
    results['grna'] = results['guide']
    results['feature'] = (
        'fraction:grna[' + results['guide'].astype(str) + ']')
    results['coefficient'] = results['standardized_marginal_effect']
    results['p_value'] = results['permutation_p_value']
    results['q_value'] = results['adjusted_p_value']

    significant = primary_table.loc[primary_table['significant']].copy()
    compatibility = {
        'results': os.path.join(destination, 'results.csv'),
        'results_grna': os.path.join(destination, 'results_grna.csv'),
        'significant': os.path.join(destination, 'results_significant.csv'),
    }
    primary_table.to_csv(compatibility['results'], index=False)
    primary_table.to_csv(compatibility['results_grna'], index=False)
    significant.to_csv(compatibility['significant'], index=False)
    paths.update(compatibility)
    return {
        'analysis_mode': 'guide_permutation',
        'results': results,
        'primary': primary_table,
        'significant': significant,
        'primary_min_wells': primary,
        'paths': {key: str(path) for key, path in paths.items()},
    }


def _perform_regression_set_paths(settings):
    # _perform_regression_read_data has already normalised both keys to
    # lists by the time this runs, so the old scalar fallbacks here were
    # unreachable.
    score_data = settings['score_data'][0]
    score_source = os.path.splitext(os.path.basename(score_data))[0]

    csv_path = settings['count_data'][0]

    # THE CALLER'S OUTPUT FOLDER IS HONOURED WHEN THERE IS ONE.
    #
    # This used to be `settings['src'] = os.path.dirname(count_data[0])`
    # unconditionally, which threw away whatever the caller asked for and
    # sent every run to the same place beside the input data. Two runs of
    # the same family then wrote to an identical path -- so comparing
    # thirteen corrections, or any two conditions, silently left only the
    # last one on disk. Nothing warned; the earlier results were simply
    # gone.
    #
    # Falling back to the data directory keeps the old behaviour for
    # callers that never set src, which is what the GUI does.
    requested = settings.get('src')
    if isinstance(requested, str) and requested.strip():
        src = os.path.abspath(os.path.expanduser(requested.strip()))
    else:
        src = os.path.dirname(settings['count_data'][0])
    settings['src'] = src

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
    if settings.get('analysis_mode') == 'guide_permutation':
        kind = 'guide_permutation'
    elif settings['regression_type'] is None:
        kind = 'auto'
    else:
        kind = str(settings['regression_type'])
    res_folder = _next_results_folder(os.path.join(src, 'results'), kind)

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


def perform_regression(settings):
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
                                        regression_type='lasso', l1_ratio=0.5):
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
        :param regression_type: ``'lasso'`` or ``'elasticnet'`` - the same
            penalty the reported coefficients were fitted with, or the
            frequencies would describe a different model from the one in
            ``results.csv``.
        :param l1_ratio: ``elasticnet`` mix; ignored for ``'lasso'``.
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
        _, X0 = dmatrices(formula, data=X, return_type='dataframe')
        feature_index = pd.Index(X0.columns)
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
            m = _estimator().fit(Xb, yb)
            coefs = pd.Series(np.asarray(m.coef_).ravel(), index=feature_index)
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
        score_data_df.loc[:, dependent_variable] = corrected[
            dependent_variable
        ]
        report_path = write_report(
            correction_report,
            os.path.join(res_folder, 'batch_correction.json'),
        )
        print(
            f"Batch correction {correction_report.method}: "
            f"{correction_report.centroid_spread_before} -> "
            f"{correction_report.centroid_spread_after} centroid spread. "
            f"Report: {report_path}"
        )
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
        filter_value = settings['filter_value']
    else:
        filter_value = []
    # filter_column used to be bound only in the `isinstance(..., str)` branch,
    # so both None (the natural "do not filter" value) and the list form that
    # process_reads documents left it unbound and the process_reads call below
    # raised UnboundLocalError. clean_controls handles str / list / None.
    filter_column = settings['filter_column']

    score_data_df = clean_controls(score_data_df, settings['filter_value'], filter_column)
    
    if settings['verbose']:
        print(f"Dependent variable after clean_controls: {len(score_data_df)}")

    sim_min_count = minimum_cell_simulation(settings, tolerance=settings['tolerance'])
    
    if settings['min_cell_count'] is None:
        settings['min_cell_count'] = sim_min_count
        
    if settings['verbose']:
        print(f"Minimum cell count: {settings['min_cell_count']}")
        print(f"Dependent variable after minimum cell count filter: {len(score_data_df)}")
        display(score_data_df)

    orig_dv = settings['dependent_variable']

    dependent_df, dependent_variable = process_scores(
        score_data_df, settings['dependent_variable'], None,
        settings['min_cell_count'], settings['agg_type'],
        settings['transform'], settings['regression_type'],
        settings['invert_dependent_variable'])
    
    if settings['verbose']:
        print(f"Dependent variable after process_scores: {len(dependent_df)}")
        display(dependent_df)
    
    if settings['fraction_threshold'] is None:
        settings['fraction_threshold'] = _graph_sequencing_stats(settings)

    independent_df = process_reads(
        count_data_df, settings['fraction_threshold'], None,
        filter_column=filter_column, filter_value=filter_value)
        
    if settings['verbose']:
        print("independent_df columns:", list(independent_df.columns))
        print("independent_df head:")
        print(independent_df.head())
        print(independent_df)
        
    independent_df, n_grna, n_gene = _count_variable_instances(independent_df, column_1='grna', column_2='gene')
    
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

    _check_score_count_pairing(independent_df, dependent_df, merged_df)

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
                merged_df = merged_df[~merged_df['grna'].isin(outliers_grna)]
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
        warning = _identifiability_warning(merged_df, settings)
        if warning:
            print(warning)

    if settings.get('analysis_mode') == 'guide_permutation':
        output = _run_guide_permutation_analysis(
            merged_df, dependent_variable, res_folder, settings)
        if settings.get('verbose'):
            print(
                f"Guide permutation analysis tested "
                f"{len(output['primary'])} guides in the primary "
                f">={output['primary_min_wells']}-well family and called "
                f"{len(output['significant'])} at "
                f"{settings['multiple_testing_method']} "
                f"alpha={settings['fdr_alpha']}."
            )
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

    model, coef_df, regression_type = regression(
        merged_df, csv_path, dependent_variable, settings['regression_type'],
        settings['alpha'], settings['random_row_column_effects'],
        nc=settings['negative_control'], pc=settings['positive_control'],
        controls=settings['controls'], dst=res_folder,
        cov_type=settings['cov_type'],
        l1_ratio=settings['l1_ratio'],
        quantile=settings['quantile'],
        hinge_threshold=settings['hinge_threshold'],
        hinge_n_boot=settings['hinge_n_boot'],
        huber_t=settings['huber_t'],
        # THE QC SUITE HAS TO BE DECLINABLE, and until this line it was not.
        # `regression()` grew a `qc` parameter precisely so a parameter sweep
        # could turn it off, and then nothing passed one -- so every trial of
        # every sweep paid the full diagnostic suite: ~5.8 s and ~19 figures
        # plus a combined PDF, i.e. roughly ten minutes and two thousand files
        # per hundred trials, with no way to say no. On a single analysis it
        # is exactly what you want, which is why it stays on by default.
        qc=bool(settings.get('regression_qc', True)),
        legacy_volcano=bool(settings.get('legacy_volcano', False)),
    )
    
    coef_df['grna'] = coef_df['feature'].apply(lambda x: re.search(r'grna\[(.*?)\]', x).group(1) if 'grna' in x else None)
    coef_df['gene'] = coef_df['feature'].apply(lambda x: re.search(r'gene\[(.*?)\]', x).group(1) if 'gene' in x else None)
    
    # n_grna / n_gene are value_counts frames, so one row per gRNA and one per
    # gene. coef_df is many rows against either of them — every gene[...] term
    # carries grna=None and vice versa — so many-to-one is the contract, and it
    # is the right side (the counts) that must stay unique: a duplicate there
    # would fan the coefficient table out and every hit would be written to
    # results_significant.csv more than once.
    coef_df = coef_df.merge(n_grna, how='left', on='grna',
                            validate='many_to_one')
    coef_df = coef_df.merge(n_gene, how='left', on='gene',
                            validate='many_to_one')

    gene_coef_df = coef_df[coef_df['n_gene'] != None]
    grna_coef_df = coef_df[coef_df['n_grna'] != None]
    gene_coef_df = gene_coef_df.dropna(subset=['n_gene'])
    grna_coef_df = grna_coef_df.dropna(subset=['n_grna'])
    
    # reg_threshold used to be bound only inside the branch below, so a
    # control-free screen (settings['controls'] is None) hit UnboundLocalError
    # as soon as the toxo volcano block read it. 0 is custom_volcano_plot's own
    # default and means "no coefficient cut-off, select on p <= 0.05 alone",
    # which is the only sensible threshold when there are no controls to
    # calibrate against.
    reg_threshold = 0

    if settings['controls'] is not None:

        control_coef_df = grna_coef_df[grna_coef_df['grna'].isin(settings['controls'])]
        mean_coef = control_coef_df['coefficient'].mean()
        significant_c = control_coef_df[control_coef_df['p_value']<= 0.05]
        mean_coef_c = significant_c['coefficient'].mean()
        
        if settings['verbose']:
            print(mean_coef, mean_coef_c)
        
        if settings['threshold_method'] in ['var','variance']:
            coef_mes = control_coef_df['coefficient'].var()
        elif settings['threshold_method'] in ['std', 'standard_deveation']:
            coef_mes = control_coef_df['coefficient'].std()
        else:
            raise ValueError(f"Unsupported threshold method {settings['threshold_method']}. Supported methods: ['var','variance','std','standard_deveation']")
        
        reg_threshold = mean_coef + (settings['threshold_multiplier'] * coef_mes)
    
    coef_df.to_csv(results_path, index=False)
    gene_coef_df.to_csv(results_path_gene, index=False)
    grna_coef_df.to_csv(results_path_grna, index=False)
    
    #v2
    #if regression_type == 'lasso':
    #    significant = coef_df[coef_df['coefficient'] > 0]
    
    #v1
    #if regression_type == 'lasso':
    #    significant = coef_df[coef_df['coefficient'] != 0].copy()
    #    significant = significant.sort_values(by='coefficient', key=lambda c: c.abs(), ascending=False)
    #    significant = significant[~significant['feature'].str.contains('row|column')]
    
    #v3
    if regression_type in NO_P_VALUE_TYPES:
        # Lasso and elastic net have no valid frequentist p-values (the ones
        # process_model_coefficients attaches are OLS-style and ignore the
        # penalty). Use bootstrap selection frequency as the feature-importance
        # ranking. Treat as a selection method, not a hypothesis test.
        n_boot = settings.get('lasso_n_boot', 200)
        sel_threshold = settings.get('lasso_selection_threshold', 0.6)
        formula = prepare_formula(
            dependent_variable, random_row_column_effects=False,
            block_screen=screen_is_blockable(merged_df))
        # Apply the same preprocessing the OLS path uses, so derived columns
        # referenced by the formula (e.g. gene_fraction) exist in the bootstrap.
        cleaned_df = check_and_clean_data(merged_df.copy(), dependent_variable)
        sel_df = bootstrap_selection_frequencies(
            X=cleaned_df,
            y=cleaned_df[dependent_variable],
            formula=formula,
            alpha=settings.get('alpha', 'auto'),
            n_boot=n_boot,
            random_state=0,
            regression_type=regression_type,
            l1_ratio=settings['l1_ratio'],
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
    else:
        # THE CORRECTION IS APPLIED HERE, and until now it never was.
        #
        # `multiple_testing_method` has existed as a setting, been offered in
        # the panel and been named in Methods sections, while this branch
        # called a hit on the RAW OLS p-value. With 1,208 coefficients an
        # uncorrected 0.05 expects about sixty false positives from noise
        # alone, and that is the defect behind a published volcano whose
        # figure showed a P = 0.05 line while its Methods claimed BH q < 0.05.
        #
        # The family is the guide/gene coefficients actually being tested --
        # not the intercept and not the row/column nuisance terms, which are
        # covariates rather than hypotheses and would only inflate the family.
        #
        # 'none' reproduces the historical rule exactly, so a run that wants
        # the old behaviour can still ask for it and is on record as having
        # asked.
        from .multiple_testing import adjust_p_values, canonical_method

        method = canonical_method(settings.get('multiple_testing_method',
                                               'fdr_bh'))
        alpha = float(settings.get('fdr_alpha', 0.05))
        # ONE STATEMENT OF WHAT IS BEING TESTED, shared with the volcano.
        # A plot drawn from a different family than the one corrected here is
        # a plot of a different experiment; see spacr.hits.tested_family.
        from .hits import tested_family

        tested = pd.Series(tested_family(coef_df['feature']),
                           index=coef_df.index)
        coef_df['q_value'] = np.nan
        coef_df['multiple_testing_method'] = method
        if tested.any():
            adjusted, _rejected = adjust_p_values(
                coef_df.loc[tested, 'p_value'].to_numpy(dtype=float),
                method=method, alpha=alpha)
            coef_df.loc[tested, 'q_value'] = adjusted
        raw_hits = int((coef_df.loc[tested, 'p_value'] <= alpha).sum())
        corrected_hits = int((coef_df.loc[tested, 'q_value'] < alpha).sum())
        print(f"Multiple testing: {method} across {int(tested.sum())} tested "
              f"coefficients at alpha={alpha:g} — {raw_hits} pass the raw P "
              f"value, {corrected_hits} pass correction.")
        # Rewrite the tables so the corrected value is in the file, not only
        # in the hit list: a volcano drawn from results.csv must be able to
        # plot the quantity the hits were called on.
        for frame, path in ((coef_df, results_path),
                            (gene_coef_df, results_path_gene),
                            (grna_coef_df, results_path_grna)):
            if frame is not coef_df and 'feature' in frame.columns:
                frame['q_value'] = frame['feature'].map(
                    coef_df.set_index('feature')['q_value'])
                frame['multiple_testing_method'] = method
            frame.to_csv(path, index=False)

        significant = coef_df.loc[coef_df['q_value'] < alpha].copy()
        if settings['controls'] is not None:
            significant_high = significant.loc[
                significant['coefficient'] >= reg_threshold]
            significant_low = significant.loc[
                significant['coefficient'] <= reg_threshold]
            significant = pd.concat([significant_high, significant_low])
        significant = significant.sort_values(
            by='coefficient', ascending=False)
        significant = significant[~significant['feature'].str.contains('row|column')]
        
    if regression_type in ['ols', 'beta']:
        if settings['verbose']:
            print(model.summary())
            save_summary_to_file(model, file_path=f'{res_folder}/mode_summary.csv')
    
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
    merged_df = pd.read_csv(results_path)
    gene_merged_df = pd.read_csv(results_path_gene)
    grna_merged_df = pd.read_csv(results_path_grna)

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

    if settings['toxo']:
        data_path = merged_df
        data_path_gene = gene_merged_df
        data_path_grna = grna_merged_df
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_dir, 'resources', 'data', 'lopit.csv')
        
        gene_list = None

        if settings['volcano'] == 'all':
            print('all')
            gene_list = custom_volcano_plot(
                data_path, metadata_path, metadata_column='tagm_location',
                point_size=600, figsize=20, threshold=reg_threshold,
                save_path=volcano_path, x_lim=settings['x_lim'], y_lims=settings['y_lims'],
            )
        elif settings['volcano'] == 'gene':
            print('gene')
            gene_list = custom_volcano_plot(
                data_path_gene, metadata_path, metadata_column='tagm_location',
                point_size=600, figsize=20, threshold=reg_threshold,
                save_path=volcano_path, x_lim=settings['x_lim'], y_lims=settings['y_lims'],
            )
        elif settings['volcano'] == 'grna':
            print('grna')
            gene_list = custom_volcano_plot(
                data_path_grna, metadata_path, metadata_column='tagm_location',
                point_size=600, figsize=20, threshold=reg_threshold,
                save_path=volcano_path, x_lim=settings['x_lim'], y_lims=settings['y_lims'],
            )
        else:
            print(f"Skipping volcano plot: settings['volcano']={settings['volcano']!r} "
                f"is not one of 'all', 'gene', 'grna'.")

        # SAY WHERE IT WENT. Every other artifact this module writes announces
        # itself ("Saved regression data to ...", "Plot -> ..."), and the
        # volcano -- the figure the module exists to produce -- was written
        # silently. With nothing naming it, a run that had drawn one perfectly
        # well was indistinguishable from a run that had drawn none, and was
        # reported as "I can't see the regression plot".
        if os.path.exists(volcano_path):
            print(f"Saved volcano plot to {volcano_path}")
        elif settings['volcano'] in ('all', 'gene', 'grna'):
            print(f"WARNING: the volcano plot was requested "
                  f"(volcano={settings['volcano']!r}) but no file was written "
                  f"to {volcano_path}")

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
        data_GT1 = (pd.read_csv(metadata_files[1], low_memory=False)
                    if have_curated_tables else None)
        data_ME49 = (pd.read_csv(metadata_files[0], low_memory=False)
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
    # Everything above sits under `if settings['toxo']`, because the
    # compartment colouring needs the LOPIT table. But the volcano itself is
    # the figure this module exists to produce, and gating it on an
    # organism-specific flag meant a run with toxo=False wrote sixteen
    # diagnostic figures and NOT the one the user came for -- silently, with
    # nothing saying why. Drawn here without the compartment colouring, which
    # is the only part that ever needed the metadata.
    if not settings.get('toxo') and settings.get('volcano') in ('all', 'gene', 'grna'):
        try:
            from .plot import volcano_plot as _plain_volcano
            _source = {'gene': results_path_gene,
                       'grna': results_path_grna}.get(settings['volcano'],
                                                      results_path)
            _plain_volcano(
                _source,
                fold_change_col='coefficient',
                p_value_col='p_value',
                name_col='feature',
                x_transform='none', y_transform='-log10',
                fold_change_threshold=reg_threshold,
                p_value_threshold=float(settings.get('fdr_alpha', 0.05) or 0.05),
                point_size=20.0, figsize=(10.0, 8.0),
                title=f"{settings.get('regression_type', 'ols')} - {settings['volcano']}",
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
    try:
        _penalised = str(settings.get('regression_type', '')).lower() in (
            'ridge', 'lasso', 'elasticnet')
        if _penalised and len(coef_df):
            _p = pd.to_numeric(coef_df.get('p_value'), errors='coerce')
            if not (_p < 0.05).any():
                print(
                    f"\nNOTE: {settings['regression_type']} returned no "
                    f"coefficient below p=0.05. Its p-values are conservative "
                    f"by construction -- the standard error is unpenalised "
                    f"while the coefficient it is divided into has been shrunk "
                    f"-- so this is NOT evidence of no effect. Refit with "
                    f"regression_type='ols' (or 'rlm' for a robust check) "
                    f"before concluding anything from it.")
    except Exception:
        pass

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


def process_reads(csv_path, fraction_threshold, plate, filter_column=None, filter_value=None):
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
        csv_df = pd.read_csv(csv_path)

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

    if not all(col in merged_df.columns for col in ['grna', 'gene']):
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

def apply_transformation(X, transform):
    """Return an sklearn ``FunctionTransformer`` for the named transform.

    :param X: Ignored (kept for compatibility with sklearn pipeline flow).
    :param transform: One of ``'log'``, ``'sqrt'``, ``'square'``. Any
        other value returns ``None``.
    :returns: A ``FunctionTransformer`` or ``None``.
    """
    if transform == 'log':
        transformer = FunctionTransformer(np.log1p, validate=True)
    elif transform == 'sqrt':
        transformer = FunctionTransformer(np.sqrt, validate=True)
    elif transform == 'square':
        transformer = FunctionTransformer(np.square, validate=True)
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
        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
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


def _labels_from_measurements(df, settings):
    """Write a class column from ``measurement_rules``, and point at it.

    The measurement basis Classify (ML) never had. Each rule is
    ``{'name': ..., 'where': [{'column':..., 'op':..., 'value':...}, ...]}``
    -- the same shape :mod:`spacr.io` already accepts for Classify (CV), so
    one settings CSV describes the same classes to both modules.

    **More than one measurement is the point.** A single threshold is a gate,
    not a class definition, and it was asked for specifically: a rule may
    carry several clauses and they are ANDed.

    Rows matching no rule are left unlabelled and are dropped downstream by
    the same path that drops unannotated rows. They are not quietly assigned
    to a class, which would invent training data.

    :param df: the merged measurement table. The class column is written into
        it, because the caller reads labels off this frame.
    :param settings: run settings; not modified.
    :returns: a new settings dict whose ``annotation_column`` names the
        column just written.
    :raises ValueError: no rules, a rule naming a column the table lacks, an
        unknown operator, or a rule matching nothing -- each of which would
        otherwise train a classifier on a class with no members.
    """
    import numpy as np

    rules = settings.get('measurement_rules') or []
    if not rules:
        raise ValueError(
            "dataset_mode='measurement' needs measurement_rules, e.g. "
            "[{'name':'big','where':[{'column':'cell_area','op':'>',"
            "'value':500}]}]")

    ops = {
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }
    column = str(settings.get('measurement_class_column')
                 or '_spacr_measurement_class')
    labels = pd.Series(np.nan, index=df.index, dtype=object)

    for rule in rules:
        name = rule.get('name')
        if not name:
            raise ValueError(f"every measurement rule needs a name: {rule!r}")
        clauses = rule.get('where') or []
        if not clauses:
            raise ValueError(
                f"measurement rule {name!r} has no 'where' clauses, so it "
                f"would select every row")
        mask = pd.Series(True, index=df.index)
        for clause in clauses:
            col = clause.get('column')
            op = clause.get('op')
            value = clause.get('value')
            if col not in df.columns:
                raise ValueError(
                    f"measurement rule {name!r} names column {col!r}, which "
                    f"is not in the measurement table")
            if op not in ops:
                raise ValueError(
                    f"measurement rule {name!r} uses operator {op!r}; "
                    f"expected one of {sorted(ops)}")
            mask &= ops[op](df[col], value)
        if not mask.any():
            raise ValueError(
                f"measurement rule {name!r} matches no rows, so its class "
                f"would have no training data")
        # Last rule wins on overlap, deliberately and visibly: overlapping
        # thresholds show up in the class counts, which is where a user can
        # see and fix them.
        labels[mask] = str(name)

    out = dict(settings)
    df[column] = labels
    out['annotation_column'] = column
    return out


@single_threaded_openmp('classical ML training')
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
        ``png_list`` table lacks ``prcfo`` / that column, or if
        ``heatmap_feature`` is not among the trained features.

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

    if _basis == 'measurement':
        settings = _labels_from_measurements(df, settings)
        _basis = 'annotation'      # the rules wrote a column; read it back

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
        df = annotated_df.merge(df, left_index=True, right_index=True,
                                validate='many_to_one')
        unique_values = df[settings['annotation_column']].dropna().unique()
        print(f"Unique values in annotation column: {unique_values}")
        
        if len(unique_values) == 1:
            unannotated_rows = df[df[settings['annotation_column']].isna()].index
            existing_value = unique_values[0]
            next_value = existing_value + 1 

            settings['positive_control'] = str(existing_value)
            settings['negative_control'] = str(next_value)

            existing_count = df[df[settings['annotation_column']] == existing_value].shape[0]
            num_to_select = min(existing_count, len(unannotated_rows))
            selected_rows = np.random.choice(unannotated_rows, size=num_to_select, replace=False)
            df.loc[selected_rows, settings['annotation_column']] = next_value

            # Print the counts for existing_value and next_value
            existing_count_final = df[df[settings['annotation_column']] == existing_value].shape[0]
            next_count_final = df[df[settings['annotation_column']] == next_value].shape[0]

            print(f"Number of rows with value {existing_value}: {existing_count_final}")
            print(f"Number of rows with value {next_value}: {next_count_final}")
            df[settings['annotation_column']] = df[settings['annotation_column']].apply(str)
            
        if settings['positive_control'] is None and settings['negative_control'] is None:
            settings['positive_control'] = str(unique_values[0])
            settings['negative_control'] = str(unique_values[1]) if len(unique_values) > 1 else str(int(unique_values[0]) + 1)
            print(f"Automatically set positive control to {settings['positive_control']} and negative control to {settings['negative_control']} based on unique values in annotation column.")
    
    if settings['channel_of_interest'] in [0,1,2,3]:
        # `if "a" and "b" in df.columns` only membership-tests "b": the first
        # operand is a non-empty literal and therefore always truthy. A
        # measurements DB whose pathogen table lacks the channel mean
        # intensity died with KeyError instead of skipping recruitment.
        pathogen_col = f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"
        cytoplasm_col = f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"
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
                                min_count=settings['minimum_cell_count'],
                                verbose=settings['verbose'])

    data_path, permutation_path, feature_importance_path, model_metricks_path, permutation_fig_path, feature_importance_fig_path, shap_fig_path, plate_heatmap_path, settings_csv, ml_features = get_ml_results_paths(src1, settings['model_type_ml'], settings['channel_of_interest'])
    df, permutation_df, feature_importance_df, _, _, _, _, _, metrics_df, _ = output

    #settings_df.to_csv(settings_csv, index=False)
    df.to_csv(data_path, mode='w', encoding='utf-8')
    permutation_df.to_csv(permutation_path, mode='w', encoding='utf-8')
    feature_importance_df.to_csv(feature_importance_path, mode='w', encoding='utf-8')
    train_features_df.to_csv(ml_features, mode='w', encoding='utf-8')
    metrics_df.to_csv(model_metricks_path, mode='w', encoding='utf-8')

    plate_heatmap_path = save_figure(plate_heatmap, plate_heatmap_path)
    permutation_fig_path = save_figure(figs[0], permutation_fig_path)
    feature_importance_fig_path = save_figure(
        figs[1], feature_importance_fig_path)
    shap_fig_path = save_figure(shap_fig, shap_fig_path)

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
    for src in srcs:
        merge_ml_predictions(df, os.path.join(src, 'measurements', 'measurements.db'),
                             table=settings['table_name'])

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
    :param batch_covariate_column: metadata column naming the BIOLOGY the
        correction must protect -- treatment, cell line, timepoint. Only
        ``combat`` uses it, and for combat it is not optional: the covariate
        coefficients are kept while the batch ones are subtracted, so a
        contrast left out of the design lands in the batch term and is
        removed along with it. Omitting it is how a real effect gets
        "corrected" away.
    :param batch_combat_mean_only: adjust each batch's MEAN and leave its
        variance alone. Use it when a plate is shifted but not differently
        scaled, or when a batch has too few rows for a stable variance
        estimate -- the shrunken scale term is the part that goes wrong on
        small batches. Default False, which adjusts both.
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

    # Split on an actual experimental unit. The index is the canonical prcfo
    # in merged measurement frames, even when filtering removed its component
    # metadata columns from X.
    from .classifier_evaluation import grouped_split, split_group_values
    split_frame = combined_df[['prcfo']].reset_index(drop=True)
    split_level, split_groups = split_group_values(
        group_by=split_by, frame=split_frame, table='ML control measurements')
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

    # Perform k-fold cross-validation
    if cross_validation:
        from .io import make_cv_folds

        distinct_groups = len(np.unique(split_groups))
        n_folds = min(5, distinct_groups)
        if n_folds < 2:
            raise ValueError(
                f"cross-validation by {split_level} needs at least two "
                f"independent groups; found {distinct_groups}")
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
        feature_importance_df = pd.DataFrame()
        feature_importance_fig = None

    df = _calculate_similarity(df, features, location_column, positive_control, negative_control)

    df['prcfo'] = df.index.astype(str)
    # Six tokens on a timelapse, five otherwise; see _assign_prcfo_parts. The
    # five-name split raised ValueError here on every timelapse database,
    # discarding a model that had already been fitted and scored.
    df = _assign_prcfo_parts(df, object_column='object')
    df['prc'] = _compose_prc_column(df)
    
    return [df, permutation_df, feature_importance_df, model, X_train, X_test, y_train, y_test, metrics_df, features], [permutation_fig, feature_importance_fig]

def shap_analysis(model, X_train, X_test):
    """Return a SHAP summary-plot figure for ``model`` explaining ``X_test``.

    :param model: Fitted estimator compatible with ``shap.Explainer``.
    :param X_train: Training features used to seed the explainer.
    :param X_test: Test features to explain.
    :returns: Matplotlib ``Figure`` holding the summary plot.
    """
    import shap

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    # TreeExplainer returns one output axis for every classifier class in
    # recent SHAP releases: (samples, features, classes).  summary_plot treats
    # any 3-D input as interaction values, which both misrepresents the data
    # and crashes when feature_names is a plain list.  The classifiers used by
    # this pipeline are binary, so explain the positive class.  Keep the only
    # output for estimators that expose a singleton output axis.
    if len(shap_values.shape) == 3:
        output_index = 1 if shap_values.shape[-1] > 1 else 0
        shap_values = shap_values[..., output_index]
    # Create a new figure
    fig, ax = plt.subplots()
    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    # Save the current figure (the one that SHAP just created)
    fig = plt.gcf()
    plt.close(fig)  # Close the figure to prevent it from displaying immediately
    return fig

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

    # Function to create radar plot for individual and combined values
    def create_extended_radar_plot(values, labels, title):
        """Draw a filled polar radar plot for ``values`` labelled by ``labels``."""
        values = list(values) + [values[0]]  # Close the loop for radar chart
        angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
        plt.title(title, pad=20)
        plt.show()

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

        scores_df = pd.read_csv(settings['scores'])

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

        if 'rowID' not in scores_df.columns:
            if 'row' in scores_df.columns:
                scores_df['rowID'] = scores_df['row']
            if 'row_name' in scores_df.columns:
                scores_df['rowID'] = scores_df['row_name']

        if 'columnID' not in scores_df.columns:
            if 'col' in scores_df.columns:
                scores_df['columnID'] = scores_df['col']
            if 'column' in scores_df.columns:
                scores_df['columnID'] = scores_df['column']

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

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.barh(top_feature_importance_df['feature'], top_feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f"Top {settings['top_features']} Features - Feature Importance")
            plt.gca().invert_yaxis()
            plt.show()

            if settings['save']:
                _save_importance_csv(feature_importance_df, settings['src'], 'feature_importance.csv')

    # Step 2: Permutation Importance
    if settings['permutation_importance']:
        print(f"Permutation Importance ...")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=_run_random_state(42), n_jobs=settings['n_jobs'])
        perm_importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean})
        perm_importance_df = perm_importance_df.sort_values(by='importance', ascending=False)
        top_perm_importance_df = perm_importance_df.head(settings['top_features'])

        # Plot Permutation Importance
        plt.figure(figsize=(10, 6))
        plt.barh(top_perm_importance_df['feature'], top_perm_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f"Top {settings['top_features']} Features - Permutation Importance")
        plt.gca().invert_yaxis()
        plt.show()
        
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

        # Plot SHAP summary for the selected sample and top features
        shap.summary_plot(shap_values, X_sample, max_display=settings['top_features'])

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
