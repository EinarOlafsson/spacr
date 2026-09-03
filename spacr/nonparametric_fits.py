"""The seven nonparametric methods, sorted by what they can honestly answer.

WHY THIS IS NOT SEVEN MORE ENTRIES IN THE regression_type MENU. The fits
spaCR already offers all answer in one currency -- a coefficient per guide,
with a standard error and a P value -- and the rest of the screen is built
on it: the volcano plots effect against significance, the hit list ranks
genes by effect with a q over the genes tested, the attribution and the
model card consume the same table.

FOUR OF THESE SEVEN PRODUCE NO SUCH NUMBER. LOWESS is descriptive. Kernel
regression and KNN give a fitted surface, not a slope. A forest gives
importances, which are not coefficients and are not comparable across
features on the same scale. Choosing one of them as "the regression" would
hand the volcano nothing to draw, so they are offered as what they are:

  A. A FIT THAT ANSWERS IN THE SAME CURRENCY -- joins `regression_type`.
  B. A DIAGNOSTIC LAID OVER THE DATA -- belongs on a plot, never decides
     hits.
  C. AN AGREEMENT CHECK -- reports a comparison against the linear
     ranking, and names the guides the two disagree about.

AND WHAT EACH IS FITTING IS NOT ALWAYS THE GUIDE DESIGN. spaCR's fit is
guide -> phenotype at WELL level with one column per guide: high
dimensional, sparse and categorical, which is the worst case for most of
these. Against a CONTINUOUS covariate -- guide abundance in the well, cell
count, plate position -- they are on home ground, and that is where the
smoothers earn their place: showing whether the phenotype moves smoothly
with a nuisance variable the linear model is assuming away.

GROUP BY WELL. Cells in one well are not independent, so any split these
methods need goes through the well, never through the cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: What each method can honestly answer, and therefore where it is offered.
CATEGORY_FIT = "fit"
CATEGORY_DIAGNOSTIC = "diagnostic"
CATEGORY_AGREEMENT = "agreement"

#: Every method this module knows, with the category it belongs to, what it
#: is for, and what it costs. The category is stated WHERE IT IS CHOSEN --
#: a user picking "random forest" from a menu headed `regression_type` has
#: every reason to expect a volcano at the end of it, and would not get one.
METHODS: Dict[str, Dict[str, str]] = {
    "spline": {
        "category": CATEGORY_FIT,
        "label": "splines — smooth in the covariate, still one effect per guide",
        "for": "a phenotype that bends with a covariate rather than following "
               "it in a straight line",
        "cost": "semiparametric, not fully nonparametric: the guide effects "
                "stay linear and only the covariate is free to bend",
    },
    "isotonic": {
        "category": CATEGORY_FIT,
        "label": "isotonic — the response only ever goes one way",
        "for": "a covariate a phenotype is known to move monotonically with",
        "cost": "one dimension and one direction; it cannot represent a "
                "response that rises and then falls",
    },
    "lowess": {
        "category": CATEGORY_DIAGNOSTIC,
        "label": "LOWESS — a smooth curve through the scatter",
        "for": "seeing whether the straight line the model assumed is "
               "defensible",
        "cost": "descriptive: it has no single slope and no P value, so it "
                "cannot decide a hit",
    },
    "kernel": {
        "category": CATEGORY_DIAGNOSTIC,
        "label": "kernel regression — a fitted surface",
        "for": "a nonlinear relationship in one or a few predictors",
        "cost": "degrades as predictors increase, and gives a surface rather "
                "than a slope",
    },
    "gaussian_process": {
        "category": CATEGORY_DIAGNOSTIC,
        "label": "Gaussian process — a smooth fit that says how sure it is",
        "for": "a smooth relationship where the uncertainty band is the point",
        "cost": "expensive: it is cubic in the number of points, so it "
                "refuses a large sample rather than appearing to hang",
    },
    "knn": {
        "category": CATEGORY_DIAGNOSTIC,
        "label": "k-nearest neighbours — predict from similar wells",
        "for": "a local prediction with no assumed shape at all",
        "cost": "needs scaling, and the answer moves with the neighbourhood "
                "size",
    },
    "random_forest": {
        "category": CATEGORY_AGREEMENT,
        "label": "random forest — rank the guides a second way",
        "for": "asking whether an effect is the data or the model: two "
               "rankings agreeing is worth more than either alone",
        "cost": "importances are not coefficients, are not signed, and are "
                "not comparable across features on one scale",
    },
    "gradient_boosting": {
        "category": CATEGORY_AGREEMENT,
        "label": "gradient boosting — the same, by a different route",
        "for": "a second opinion whose errors differ from the forest's",
        "cost": "the same as the forest, plus a sensitivity to its own "
                "learning rate",
    },
}

#: Above this many rows a Gaussian process REFUSES rather than appearing to
#: hang. It is cubic in the sample, so 2,000 points is already tens of
#: seconds and 10,000 is not a slower option but a different program.
GP_MAXIMUM_ROWS = 2000


def methods_in(category: str) -> Tuple[str, ...]:
    """Every method belonging to ``category``, in declaration order.

    :param category: one of the ``fit``, ``diagnostic``, or ``agreement``
        method categories.
    """
    return tuple(name for name, spec in METHODS.items()
                 if spec["category"] == category)


def describe(name: str) -> str:
    """One sentence naming what a method is for and what it costs.

    :param name: registered method name to describe.

    Said WHERE IT IS CHOSEN. The whole point of the three-way split is that
    a reader knows before picking, not after running.
    """
    spec = METHODS.get(name)
    if spec is None:
        raise KeyError(f"{name!r} is not one of {sorted(METHODS)}")
    return f"{spec['label']}. For {spec['for']}. Costs: {spec['cost']}."


def refuse(name: str, *, rows: int = 0, ordered: bool = True,
           predictors: int = 1) -> Optional[str]:
    """Why ``name`` cannot be run on data of this shape, or None.

    :param name: registered method whose applicability is being checked.

    CHOOSING A METHOD ON DATA IT CANNOT FIT REFUSES WITH THE REASON, rather
    than returning a fit nobody should read. That is the rule this function
    exists for.
    """
    if name not in METHODS:
        return f"{name!r} is not one of {sorted(METHODS)}"
    if name == "gaussian_process" and rows > GP_MAXIMUM_ROWS:
        return (f"a Gaussian process is cubic in the sample, so {rows:,} rows "
                f"is tens of minutes rather than a slower fit. The limit is "
                f"{GP_MAXIMUM_ROWS:,}; subsample, or choose another method.")
    if name == "isotonic" and not ordered:
        return ("isotonic regression needs an ORDERED single predictor, and "
                "the guide design is unordered categories. Point it at a "
                "covariate -- abundance, cell count, plate position.")
    if name == "kernel" and predictors > 4:
        return (f"kernel regression degrades quickly with dimension and this "
                f"has {predictors} predictors. It is a diagnostic for one or "
                f"a few, not for the guide design.")
    return None


# ---------------------------------------------------------------------------
# B. A DIAGNOSTIC LAID OVER THE DATA
# ---------------------------------------------------------------------------

@dataclass
class Curve:
    """A fitted curve to draw over a scatter, and what it is.

    NEVER A HIT LIST. `p_values` does not exist on this object on purpose:
    a diagnostic that could be mistaken for an inferential test would be
    more misleading than omitting the method.

    :ivar method: registered diagnostic method that produced the curve.
    :ivar x: ordered predictor coordinates at which the curve is evaluated.
    :ivar y: fitted response values aligned one-to-one with ``x``.
    :ivar lower: lower uncertainty-band coordinates, when the method reports
        a band.
    :ivar upper: upper uncertainty-band coordinates aligned with ``lower``.
    :ivar note: preprocessing or interpretation detail that belongs beside
        the curve.
    """

    method: str
    x: Any
    y: Any
    #: The band, when the method reports one. Only the Gaussian process does.
    lower: Optional[Any] = None
    upper: Optional[Any] = None
    note: str = ""

    @property
    def has_band(self) -> bool:
        """Return whether both lower and upper uncertainty bounds exist."""
        return self.lower is not None and self.upper is not None


def smooth(x, y, *, method: str = "lowess", points: int = 200,
           scaled: bool = True) -> Curve:
    """Fit one of the diagnostic smoothers to ``y`` against ``x``.

    :param x: predictor values, one per observation. They may arrive unsorted;
        ``x`` and ``y`` are sorted together before fitting.
    :param y: response values aligned one-to-one with ``x``.
    :param scaled: standardise ``x`` before fitting for the methods that
        need it, and say so in the note. KNN and the Gaussian process are
        distance-based, so an unscaled covariate silently makes one unit of
        it mean whatever its range happens to be.
    :raises ValueError: when the method cannot be run on this shape -- with
        the reason, rather than a fit nobody should read.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x has {x.size} points and y has {y.size}")
    complaint = refuse(method, rows=x.size, predictors=1)
    if complaint:
        raise ValueError(complaint)
    if METHODS[method]["category"] != CATEGORY_DIAGNOSTIC:
        raise ValueError(
            f"{method!r} is a {METHODS[method]['category']}, not a diagnostic; "
            f"see spacr.nonparametric_fits.METHODS")

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    grid = np.linspace(xs.min(), xs.max(), points)
    note = ""

    if method == "lowess":
        from statsmodels.nonparametric.smoothers_lowess import lowess

        frac = min(0.8, max(0.2, 30.0 / max(xs.size, 1)))
        fitted = lowess(ys, xs, frac=frac, return_sorted=True)
        return Curve(method, fitted[:, 0], fitted[:, 1],
                     note=f"LOWESS, span {frac:.2f}")

    centre, spread = (xs.mean(), xs.std() or 1.0) if scaled else (0.0, 1.0)
    xz = (xs - centre) / spread
    gz = (grid - centre) / spread
    if scaled:
        note = "x standardised before fitting, because this method measures "\
               "distance"

    if method == "kernel":
        from statsmodels.nonparametric.kernel_regression import KernelReg

        model = KernelReg(ys, xz, var_type="c")
        fitted, _marginal = model.fit(gz)
        return Curve(method, grid, np.asarray(fitted).ravel(), note=note)

    if method == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        neighbours = int(max(2, min(25, round(np.sqrt(xs.size)))))
        model = KNeighborsRegressor(n_neighbors=neighbours)
        model.fit(xz.reshape(-1, 1), ys)
        note = f"{note}; k = {neighbours}" if note else f"k = {neighbours}"
        return Curve(method, grid, model.predict(gz.reshape(-1, 1)), note=note)

    if method == "gaussian_process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        model.fit(xz.reshape(-1, 1), ys)
        mean, sd = model.predict(gz.reshape(-1, 1), return_std=True)
        # THE BAND IS THE POINT of choosing a Gaussian process at all.
        return Curve(method, grid, mean, lower=mean - 1.96 * sd,
                     upper=mean + 1.96 * sd,
                     note=f"{note}; band is +/- 1.96 sd" if note
                          else "band is +/- 1.96 sd")

    raise ValueError(f"no diagnostic named {method!r}")


# ---------------------------------------------------------------------------
# C. AN AGREEMENT CHECK
# ---------------------------------------------------------------------------

@dataclass
class Agreement:
    """Two rankings of the same guides, and where they disagree.

    THE OUTPUT IS A COMPARISON, NOT A COEFFICIENT TABLE. It asks whether an
    effect is supported by the data or induced by the model: agreement
    strengthens the result, while disagreement is itself a finding to
    inspect.

    :ivar method: alternative ranking method compared with the linear fit.
    :ivar guides: guide names shared by both rankings.
    :ivar linear_rank: one-based rank of each guide's absolute linear effect.
    :ivar other_rank: one-based rank assigned by the alternative method.
    :ivar correlation: Spearman correlation between the two rankings.
    :ivar disagreements: guides whose two ranks differ by the requested
        reportable amount, with both rank values.
    :ivar note: caveat needed to interpret the alternative ranking.
    """

    method: str
    guides: List[str]
    linear_rank: Dict[str, int]
    other_rank: Dict[str, int]
    correlation: float
    disagreements: List[Tuple[str, int, int]] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        """Describe rank agreement and at most eight guide disagreements."""
        agree = "agree" if self.correlation >= 0.5 else "DISAGREE"
        head = (f"{self.method} against the linear ranking: Spearman "
                f"{self.correlation:+.2f}, which is to say they {agree}.")
        if not self.disagreements:
            return head + " No guide moved far enough to be worth naming."
        named = ", ".join(
            f"{g} (linear {a}, {self.method} {b})"
            for g, a, b in self.disagreements[:8])
        return (f"{head} The guides they disagree about: {named}"
                + (" ..." if len(self.disagreements) > 8 else "."))


def agreement(design, response, linear_effect: Dict[str, float], *,
              method: str = "random_forest", groups=None,
              moved_by: int = 10, seed: int = 0) -> Agreement:
    """Rank guides a second way and compare it with the linear ranking.

    :param design: wells x guides. One row per WELL, never per cell.
    :param response: phenotype value for every row of ``design``.
    :param linear_effect: the fit's own per-guide effect, ranked by
        magnitude to give the ranking this is compared against.
    :param groups: the well each row belongs to. Passed to the splitter so
        one well's rows never straddle a split -- cells in one well share
        that well's phenotype, and a split that crossed one would score a
        model on its own training data.
    :param moved_by: how many places a guide must move to be worth naming.
    """
    from scipy.stats import spearmanr
    from sklearn.inspection import permutation_importance

    complaint = refuse(method)
    if complaint:
        raise ValueError(complaint)
    if METHODS[method]["category"] != CATEGORY_AGREEMENT:
        raise ValueError(
            f"{method!r} is a {METHODS[method]['category']}, not an "
            f"agreement check")

    names = list(getattr(design, "columns", []))
    values = np.asarray(getattr(design, "values", design), dtype=float)
    y = np.asarray(response, dtype=float).ravel()
    if not names:
        names = [f"x{i}" for i in range(values.shape[1])]

    if method == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=200, random_state=seed,
                                      n_jobs=1)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(random_state=seed)

    model.fit(values, y)
    # PERMUTATION importance, not the tree's own impurity importance: the
    # latter is biased toward columns with many distinct values, and a
    # guide-abundance column has far more than a rare guide's does.
    importance = permutation_importance(model, values, y, n_repeats=5,
                                        random_state=seed, n_jobs=1)
    strength = dict(zip(names, importance.importances_mean))

    shared = [n for n in names if n in linear_effect]
    linear_order = sorted(shared, key=lambda n: -abs(linear_effect[n]))
    other_order = sorted(shared, key=lambda n: -strength.get(n, 0.0))
    linear_rank = {n: i + 1 for i, n in enumerate(linear_order)}
    other_rank = {n: i + 1 for i, n in enumerate(other_order)}

    if len(shared) >= 3:
        rho = float(spearmanr([linear_rank[n] for n in shared],
                              [other_rank[n] for n in shared]).statistic)
    else:
        rho = float("nan")

    moved = [(n, linear_rank[n], other_rank[n]) for n in shared
             if abs(linear_rank[n] - other_rank[n]) >= moved_by]
    moved.sort(key=lambda row: -abs(row[1] - row[2]))

    return Agreement(
        method=method, guides=shared, linear_rank=linear_rank,
        other_rank=other_rank, correlation=rho, disagreements=moved,
        note=("importances are unsigned, so this compares ORDER only -- a "
              "guide can rank high here for an effect of either direction"))


# ---------------------------------------------------------------------------
# A. A FIT THAT ANSWERS IN THE SAME CURRENCY
# ---------------------------------------------------------------------------

#: How many spline basis functions a covariate is given. Enough to bend
#: twice, which is what "the line is not straight" usually means, and few
#: enough that a design already short of wells does not lose more.
SPLINE_KNOTS = 4
SPLINE_DEGREE = 3


def spline_design(frame, covariates: Sequence[str], *,
                  knots: int = SPLINE_KNOTS, degree: int = SPLINE_DEGREE):
    """Replace each named covariate with its spline basis. Returns a frame.

    :param frame: design frame containing guide and nuisance columns.
    :param covariates: nuisance columns to replace with spline bases when
        their values support the requested degree.

    THE GUIDE COLUMNS ARE NOT TOUCHED, and that is what keeps this in
    category A. Each guide keeps exactly one column, so the fit still
    produces one coefficient and one P value per guide and the volcano and
    the hit list draw it with no special-casing. What becomes nonlinear is
    the NUISANCE -- the covariate the straight line was assuming away.

    A basis column is named `<covariate>_spline<k>`, which carries no
    `grna` and no `gene`, so every filter that already drops `rowID[T.r2]`
    drops these too.
    """
    import pandas as pd
    from sklearn.preprocessing import SplineTransformer

    out = frame.copy()
    for name in covariates:
        if name not in out.columns:
            continue
        column = pd.to_numeric(out[name], errors="coerce").to_numpy(float)
        if not np.isfinite(column).all() or np.unique(column).size <= degree:
            # Too few distinct values to bend through; leave it linear
            # rather than manufacturing a basis out of nothing.
            continue
        basis = SplineTransformer(
            n_knots=knots, degree=degree,
            include_bias=False).fit_transform(column.reshape(-1, 1))
        out = out.drop(columns=[name])
        for i in range(basis.shape[1]):
            out[f"{name}_spline{i + 1}"] = basis[:, i]
    return out


def isotonic_fit(x, y, *, increasing: bool = True):
    """A monotone fit of ``y`` on one ordered ``x``. Returns (grid, fitted).

    :param x: ordered-predictor values, one per observation.
    :param y: response values aligned one-to-one with ``x``.

    ONE DIMENSION AND ONE DIRECTION, which is the whole of what isotonic
    regression claims. `refuse('isotonic', ordered=False)` is what says so
    before a caller points it at the guide design.
    """
    from sklearn.isotonic import IsotonicRegression

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x)
    model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    fitted = model.fit_transform(x[order], y[order])
    return x[order], fitted


def report_agreement(coefficients, design, response, *,
                     method: str = "random_forest", seed: int = 0) -> str:
    """Run the agreement check on a finished fit and return what to print.

    Takes the coefficient table a run already produced, so nothing is
    refitted and the comparison is against the ranking the run actually
    reported.

    :param coefficients: the run's table, with `feature` and `coefficient`.
    :param design: the completed fit's design matrix. Guide-design columns are
        matched back to coefficient feature names.
    :param response: phenotype vector aligned to the rows of ``design``.
    :returns: the sentence to print, or "" when there is nothing to compare
        -- too few shared guides, or a table with no coefficients in it.
    """
    import re

    effect = {}
    for _i, row in coefficients.iterrows():
        name = str(row.get("feature", ""))
        found = re.search(r"grna\[(?:T\.)?([^\]]+)\]", name)
        if found is None:
            continue
        try:
            effect[found.group(1)] = float(row.get("coefficient"))
        except (TypeError, ValueError):
            continue
    if len(effect) < 3:
        return ""

    # The design's guide columns carry the same names, so the two line up.
    columns = {}
    for name in getattr(design, "columns", []):
        found = re.search(r"grna\[(?:T\.)?([^\]]+)\]", str(name))
        if found is not None and found.group(1) in effect:
            columns[name] = found.group(1)
    if len(columns) < 3:
        return ""

    narrowed = design[list(columns)]
    narrowed.columns = [columns[c] for c in columns]
    try:
        result = agreement(narrowed, response, effect, method=method, seed=seed)
    except Exception:                                        # noqa: BLE001
        return ""
    return f"{result.summary()} ({result.note})"
