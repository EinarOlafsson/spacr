"""Interrogate a fitted model: move one input, watch the prediction move.

A coefficient table answers "which terms matter". It does not answer the
question anyone actually has in front of a fitted screen model, which is
*what would this model predict for a well like mine* — and then, having
seen that, *what happens if this one gRNA's fraction doubles and everything
else stays where it is*. That is a profile: one input swept across its
range, every other input pinned at a value the user chose, and the
prediction plotted against the input that moved.

Three things make this harder than it sounds, and this module exists to
settle all three in one place:

**Seventeen backends predict differently.** :data:`spacr.ml.REGRESSION_TYPES`
spans statsmodels results (``OLS``, ``WLS``, ``RLM``, four GLM families,
``QuantReg``, ``MixedLM``, ``BetaModel``), scikit-learn estimators
(``Lasso``, ``Ridge``, ``ElasticNet``, ``LinearSVC``) and the horseshoe
fitter. :func:`predict` is the one call that works on all of them, and
:func:`response_scale` names what came back — because a GLM-binomial returns
a probability, a Poisson GLM returns a rate, and ``LinearSVC.predict``
returns a CLASS LABEL, which would draw a step function and look like a
finding. The hinge backend is profiled through ``decision_function`` for
exactly that reason, and the scale says so.

**Nothing here re-fits.** The model handed in is the model profiled. A
profiler that quietly re-fits is showing the user a second model and
labelling it the first, and on a penalised backend with ``alpha='auto'``
the second one is not even the same model. Where a caller has only the
written-out coefficients and no live object, :func:`from_coefficients`
builds a :class:`FittedLinear` around them — that is *reading* the fit, not
repeating it, and the class says which link it is applying.

**A held value is a choice, not a default.** :func:`reference_row` picks
the median of each column and is explicit that it did; the ``at`` argument
overrides any of them; and every :class:`Profile` carries the full held
vector, so a curve can always be traced back to the assumptions that
produced it.

Public API::

    from spacr.profiler import profile, reference_row, sensitivity

    curve = profile(model, design, "fraction:grna[233460_1]", n=41)
    curve.predictions          # what the model says as that input sweeps
    curve.held                 # where everything else was pinned
    sensitivity(model, design) # which input moves the prediction most
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

import numpy as np
import pandas as pd

__all__ = [
    "FittedLinear",
    "LINKS",
    "Profile",
    "Sensitivity",
    "coefficient_frame",
    "from_coefficients",
    "predict",
    "profile",
    "profile_by",
    "reference_row",
    "response_scale",
    "sensitivity",
]

#: Inverse link functions, by the name a user would recognise. The keys are
#: what :func:`from_coefficients` accepts and what :attr:`FittedLinear.link`
#: reports, so a curve is never drawn on a scale nobody named.
LINKS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda eta: eta,
    "log": np.exp,
    "logit": lambda eta: 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700))),
    "probit": lambda eta: 0.5 * (1.0 + _erf(np.asarray(eta) / math.sqrt(2.0))),
}

#: What each link's output means, for the axis label.
LINK_SCALES: Dict[str, str] = {
    "identity": "response",
    "log": "rate (log link)",
    "logit": "probability (logit link)",
    "probit": "probability (probit link)",
}


def _erf(values: np.ndarray) -> np.ndarray:
    """Vectorised error function, without pulling scipy in for one call."""
    return np.vectorize(math.erf)(np.asarray(values, dtype=float))


# ---------------------------------------------------------------------------
# One prediction call for seventeen backends
# ---------------------------------------------------------------------------

def predict(model: Any, exog: pd.DataFrame, *,
            offset: Optional[Sequence[float]] = None) -> np.ndarray:
    """Predict from any of the fitted objects :mod:`spacr.ml` produces.

    The order the branches are tried in is load-bearing:

    1. ``decision_function`` — ``LinearSVC`` (the ``hinge`` backend) has both
       that and ``predict``, and its ``predict`` returns a 0/1 CLASS. A
       profile of a class label is a step function, which reads as a finding
       and is an artefact of asking the wrong method.
    2. a statsmodels results object — ``predict(exog)`` applies the inverse
       link, so a GLM comes back on the response scale. ``offset`` is passed
       when the object accepts one.
    3. anything else with ``predict`` — the scikit-learn regressors.
    4. the linear predictor from ``params`` / ``coef_``, for a fitted object
       that carries coefficients and nothing else (the horseshoe fitter).

    :param model: a fitted object.
    :param exog: design rows to predict for; columns must match the fit's.
    :param offset: per-row offset for a model fitted with one (Poisson and
        horseshoe use ``log(cell count)``). Omitted means zero offset, i.e.
        the prediction for a well of unit exposure.
    :returns: one float per row of ``exog``.
    :raises TypeError: when the object carries neither a predict method nor
        coefficients — there is nothing to profile, and guessing would draw
        a curve that means nothing.
    """
    frame = _as_frame(exog)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(frame), dtype=float).ravel()

    if hasattr(model, "predict"):
        if offset is not None:
            try:
                return np.asarray(model.predict(frame, offset=np.asarray(
                    offset, dtype=float)), dtype=float).ravel()
            except (TypeError, ValueError):
                pass
        try:
            return np.asarray(model.predict(frame), dtype=float).ravel()
        except Exception:
            # A statsmodels results object whose design does not line up
            # raises from deep inside patsy; fall through to the linear
            # predictor, which aligns by column name and can say what is
            # missing.
            pass

    linear = _linear_predictor(model, frame)
    if linear is None:
        raise TypeError(
            f"{type(model).__name__} carries neither a usable predict() nor "
            f"coefficients, so there is nothing to profile.")
    if offset is not None:
        linear = linear + np.asarray(offset, dtype=float).ravel()
    return linear


def _as_frame(exog: Any) -> pd.DataFrame:
    """Coerce a row, dict or array into a DataFrame of design rows."""
    if isinstance(exog, pd.DataFrame):
        return exog
    if isinstance(exog, pd.Series):
        return exog.to_frame().T
    if isinstance(exog, Mapping):
        return pd.DataFrame([dict(exog)])
    return pd.DataFrame(np.atleast_2d(np.asarray(exog, dtype=float)))


def _coefficients(model: Any) -> Optional[pd.Series]:
    """The fitted coefficients as a named Series, or ``None``."""
    params = getattr(model, "params", None)
    if isinstance(params, pd.Series) and len(params):
        return params.astype(float)
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    values = np.asarray(coef, dtype=float).ravel()
    names = getattr(model, "feature_names_in_", None)
    index = ([str(n) for n in names] if names is not None
             else [str(i) for i in range(values.size)])
    series = pd.Series(values, index=index, dtype=float)
    intercept = getattr(model, "intercept_", None)
    if intercept is not None:
        flat = np.asarray(intercept, dtype=float).ravel()
        if flat.size:
            series["Intercept"] = float(flat[0])
    return series


def _linear_predictor(model: Any, frame: pd.DataFrame) -> Optional[np.ndarray]:
    """``X @ beta`` aligned by column name, or ``None`` with no coefficients."""
    params = _coefficients(model)
    if params is None:
        return None
    total = np.zeros(len(frame), dtype=float)
    for name, value in params.items():
        if name in frame.columns:
            total = total + np.asarray(frame[name], dtype=float) * float(value)
        elif str(name) == "Intercept":
            total = total + float(value)
    return total


def response_scale(model: Any) -> str:
    """Name what :func:`predict` returns for this model, for the axis label.

    Not cosmetic. The same curve means "probability that a well is
    positive", "positive objects per cell" or "distance from a decision
    boundary" depending on the backend, and a plot that does not say which
    invites the wrong reading of all three.
    """
    if isinstance(model, FittedLinear):
        return LINK_SCALES.get(model.link, model.link)
    if hasattr(model, "decision_function"):
        return "decision function (hinge margin)"
    inner = getattr(model, "model", None)
    family = getattr(inner, "family", None)
    if family is not None:
        link = getattr(family, "link", None)
        return (f"{type(family).__name__.lower()} mean "
                f"({type(link).__name__.lower()} link)" if link is not None
                else f"{type(family).__name__.lower()} mean")
    name = type(inner).__name__ if inner is not None else type(model).__name__
    if "QuantReg" in name:
        return "conditional quantile"
    if "MixedLM" in name:
        return "response (fixed effects)"
    if "Beta" in name:
        return "mean proportion (beta)"
    return "response"


# ---------------------------------------------------------------------------
# Reading a written-out fit
# ---------------------------------------------------------------------------

@dataclass
class FittedLinear:
    """A fitted linear predictor rebuilt from coefficients already written.

    Not a re-fit. A regression run writes ``results.csv``, and those numbers
    ARE the fit; this wraps them so the profiler can be pointed at a run that
    finished last week without the original object being alive. It quacks
    like a statsmodels result — ``params`` and ``predict`` — which is exactly
    the surface :func:`predict` and :func:`profile` need.

    :param params: coefficients indexed by design-matrix column name; an
        ``Intercept`` entry is applied to every row.
    :param link: which inverse link to apply; a key of :data:`LINKS`.
    :param label: what to call this model in a plot.
    """

    params: pd.Series
    link: str = "identity"
    label: str = "fitted coefficients"

    def __post_init__(self) -> None:
        if self.link not in LINKS:
            raise ValueError(
                f"unknown link {self.link!r}; choose from {sorted(LINKS)}")
        self.params = pd.Series(self.params, dtype=float)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Every coefficient name except the intercept, in fitted order."""
        return tuple(str(name) for name in self.params.index
                     if str(name) != "Intercept")

    def predict(self, exog: Any) -> np.ndarray:
        """Predict on the response scale, applying :attr:`link`."""
        frame = _as_frame(exog)
        eta = _linear_predictor(self, frame)
        if eta is None:                                # pragma: no cover
            raise TypeError("no coefficients to predict from")
        return np.asarray(LINKS[self.link](eta), dtype=float).ravel()


def coefficient_frame(source: Any) -> pd.DataFrame:
    """Read a coefficient table from a path, DataFrame or fitted object.

    :param source: a CSV path, a DataFrame with ``feature`` and
        ``coefficient`` columns, or a fitted object carrying ``params``.
    :returns: a frame with ``feature`` and ``coefficient``.
    :raises ValueError: when the columns are not there.
    """
    if isinstance(source, pd.DataFrame):
        frame = source
    elif hasattr(source, "params") or hasattr(source, "coef_"):
        params = _coefficients(source)
        if params is None:
            raise ValueError("that object carries no coefficients")
        return pd.DataFrame({"feature": [str(i) for i in params.index],
                             "coefficient": params.to_numpy(dtype=float)})
    else:
        frame = pd.read_csv(source)
    missing = {"feature", "coefficient"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"a coefficient table needs {sorted(missing)}; got "
            f"{list(frame.columns)}")
    return frame[["feature", "coefficient"]]


def from_coefficients(source: Any, *, link: str = "identity",
                      label: str = "", drop_zero: bool = False
                      ) -> FittedLinear:
    """Build a :class:`FittedLinear` from a written-out coefficient table.

    :param source: anything :func:`coefficient_frame` accepts.
    :param link: the inverse link the original fit used; ``"identity"`` for
        the least-squares and robust backends, ``"logit"`` / ``"probit"`` for
        the binomial ones, ``"log"`` for Poisson and horseshoe.
    :param label: what to call the model in a plot.
    :param drop_zero: leave out coefficients that are exactly zero. A
        penalised fit sets most of them there, and a profiler with three
        thousand flat inputs is unusable — but this is OFF by default,
        because "this gRNA does nothing" is a real answer a user may want to
        see the flat line for.
    :returns: a fitted linear predictor over those coefficients.
    """
    frame = coefficient_frame(source)
    frame = frame.dropna(subset=["feature", "coefficient"])
    if drop_zero:
        frame = frame[frame["coefficient"] != 0.0]
    params = pd.Series(
        frame["coefficient"].to_numpy(dtype=float),
        index=[str(name) for name in frame["feature"]], dtype=float)
    params = params[~params.index.duplicated(keep="first")]
    return FittedLinear(params=params, link=link,
                        label=label or "fitted coefficients")


# ---------------------------------------------------------------------------
# Where the other inputs are held
# ---------------------------------------------------------------------------

def reference_row(design: pd.DataFrame, *, method: str = "median",
                  at: Optional[Mapping[str, float]] = None) -> pd.Series:
    """The row every profile holds the non-moving inputs at.

    :param design: the design matrix (or any frame with the fit's columns).
    :param method: ``"median"`` (the default — robust to the long right tail
        a per-gRNA fraction column has), ``"mean"``, ``"zero"`` or ``"min"``.
    :param at: explicit values that override the chosen method, column by
        column. This is the "held at chosen values" half of the profiler.
    :returns: one row, indexed by the design's columns.
    :raises ValueError: on an unknown method, or a column named in ``at``
        that the design does not have — a typo there would silently hold
        nothing and produce a curve with no explanation.
    """
    if method not in ("median", "mean", "zero", "min"):
        raise ValueError(
            f"unknown method {method!r}; use median, mean, zero or min")
    numeric = design.select_dtypes(include=[np.number])
    if method == "zero":
        row = pd.Series(0.0, index=numeric.columns, dtype=float)
    elif method == "mean":
        row = numeric.mean(numeric_only=True).astype(float)
    elif method == "min":
        row = numeric.min(numeric_only=True).astype(float)
    else:
        row = numeric.median(numeric_only=True).astype(float)
    row = row.fillna(0.0)
    # An intercept column is 1 by construction; holding it at its median is
    # right by accident and at zero is wrong on purpose, so it is pinned.
    for name in row.index:
        if str(name).lower() in ("intercept", "const"):
            row[name] = 1.0
    for name, value in (at or {}).items():
        if name not in row.index:
            raise ValueError(
                f"cannot hold {name!r}: it is not a column of the design "
                f"({len(row.index)} columns).")
        row[name] = float(value)
    return row


def _sweep(design: pd.DataFrame, variable: str, n: int,
           values: Optional[Sequence[float]]) -> np.ndarray:
    """The values the moving input takes."""
    if values is not None:
        swept = np.asarray(list(values), dtype=float)
        if swept.size == 0:
            raise ValueError("a profile needs at least one value to sweep")
        return swept
    column = pd.to_numeric(design[variable], errors="coerce").dropna()
    if column.empty:
        low, high = 0.0, 1.0
    else:
        low, high = float(column.min()), float(column.max())
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        # A column with a single observed value has no range to sweep. Widen
        # it symmetrically rather than returning one point: "what if this
        # were different" is the question, and a constant column is exactly
        # when nobody knows the answer.
        centre = low if math.isfinite(low) else 0.0
        spread = abs(centre) if centre else 1.0
        low, high = centre - spread, centre + spread
    return np.linspace(low, high, max(2, int(n)))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """One input swept, everything else pinned, and what the model said.

    :param variable: the input that moved.
    :param values: the values it took.
    :param predictions: the model's output at each of them.
    :param held: every other input and the value it was held at.
    :param baseline: the prediction with EVERY input at its held value —
        the point the curve is a departure from.
    :param scale: what the predictions mean; see :func:`response_scale`.
    :param model_label: what to call the model.
    :param reference_method: how the held values were chosen.
    """

    variable: str
    values: Tuple[float, ...]
    predictions: Tuple[float, ...]
    held: Dict[str, float] = field(default_factory=dict)
    baseline: float = float("nan")
    scale: str = "response"
    model_label: str = ""
    reference_method: str = "median"

    def __len__(self) -> int:
        """How many points the curve has."""
        return len(self.values)

    @property
    def span(self) -> float:
        """How far the prediction moved across the whole sweep."""
        finite = [p for p in self.predictions if math.isfinite(p)]
        return (max(finite) - min(finite)) if finite else float("nan")

    @property
    def slope(self) -> float:
        """Change in prediction per unit of the input, end to end.

        End to end rather than fitted: for a linear model they are the same
        number, and for a GLM the end-to-end value is the honest summary of a
        curve whose local slope changes.
        """
        if len(self.values) < 2:
            return float("nan")
        run = self.values[-1] - self.values[0]
        if run == 0:
            return float("nan")
        return (self.predictions[-1] - self.predictions[0]) / run

    def at(self, value: float) -> float:
        """The prediction at the swept point nearest ``value``."""
        if not self.values:
            return float("nan")
        index = int(np.argmin(np.abs(np.asarray(self.values) - float(value))))
        return self.predictions[index]

    def to_frame(self) -> pd.DataFrame:
        """The curve as a two-column frame, for a plot or a CSV."""
        return pd.DataFrame({self.variable: list(self.values),
                             "prediction": list(self.predictions)})

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy of the profile."""
        return {"variable": self.variable, "values": list(self.values),
                "predictions": list(self.predictions),
                "held": dict(self.held), "baseline": self.baseline,
                "scale": self.scale, "model_label": self.model_label,
                "reference_method": self.reference_method,
                "span": self.span, "slope": self.slope}


@dataclass(frozen=True)
class Sensitivity:
    """How much one input moves the prediction, with the others pinned.

    :param variable: the input.
    :param low: the value swept from.
    :param high: the value swept to.
    :param prediction_low: what the model said at ``low``.
    :param prediction_high: what it said at ``high``.
    :param span: ``prediction_high - prediction_low``; the ranking key is its
        magnitude.
    :param coefficient: the fitted coefficient, when the model exposes one.
    """

    variable: str
    low: float
    high: float
    prediction_low: float
    prediction_high: float
    span: float
    coefficient: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy."""
        return {"variable": self.variable, "low": self.low, "high": self.high,
                "prediction_low": self.prediction_low,
                "prediction_high": self.prediction_high, "span": self.span,
                "coefficient": self.coefficient}


def profile(model: Any, design: pd.DataFrame, variable: str, *,
            values: Optional[Sequence[float]] = None,
            at: Optional[Mapping[str, float]] = None,
            n: int = 25, method: str = "median",
            offset: Optional[float] = None,
            label: str = "") -> Profile:
    """Sweep one input; hold the rest; return what the model predicts.

    :param model: a fitted object — anything :func:`predict` handles.
    :param design: the design matrix, used for the sweep range and for the
        held values. It is READ, never fitted on.
    :param variable: the column to move.
    :param values: sweep these exact values instead of the column's range.
    :param at: hold named inputs at these values instead of the reference.
    :param n: how many points to sweep when ``values`` is not given.
    :param method: how unspecified inputs are held; see :func:`reference_row`.
    :param offset: per-row offset for a model fitted with one.
    :param label: what to call the model in a plot; defaults to its class.
    :returns: a :class:`Profile`.
    :raises KeyError: when ``variable`` is not a column of the design.
    """
    if variable not in design.columns:
        raise KeyError(
            f"{variable!r} is not a column of the design; the model's inputs "
            f"are {list(design.columns)[:8]}"
            f"{'…' if len(design.columns) > 8 else ''}")

    row = reference_row(design, method=method, at=at)
    swept = _sweep(design, variable, n, values)

    grid = pd.DataFrame([row.to_dict()] * len(swept))
    grid[variable] = swept
    offsets = None if offset is None else np.full(len(swept), float(offset))
    predictions = predict(model, grid, offset=offsets)

    base_frame = pd.DataFrame([row.to_dict()])
    base_offset = None if offset is None else np.full(1, float(offset))
    baseline = float(predict(model, base_frame, offset=base_offset)[0])

    held = {str(name): float(value) for name, value in row.items()
            if str(name) != variable}
    return Profile(
        variable=variable, values=tuple(float(v) for v in swept),
        predictions=tuple(float(p) for p in predictions), held=held,
        baseline=baseline, scale=response_scale(model),
        model_label=label or getattr(model, "label", None)
        or type(model).__name__,
        reference_method=method)


def profile_by(model: Any, design: pd.DataFrame, variable: str, *,
               by: str, levels: Sequence[float],
               **kwargs: Any) -> List[Profile]:
    """One profile of ``variable`` per level of a second input.

    The "with the other inputs held at chosen values" question asked several
    times at once: does this gRNA's effect look different in a well with a
    high control fraction? Each returned profile carries the level it was
    drawn at in :attr:`Profile.held`.

    :param model: a fitted object.
    :param design: the design matrix.
    :param variable: the input that moves.
    :param by: the input held at each of ``levels`` in turn.
    :param levels: the values to hold ``by`` at.
    :param kwargs: passed through to :func:`profile`.
    :returns: one :class:`Profile` per level, in the order given.
    :raises ValueError: when ``levels`` is empty.
    """
    if not list(levels):
        raise ValueError("profile_by needs at least one level to hold at")
    profiles: List[Profile] = []
    for level in levels:
        overrides = dict(kwargs)
        held = dict(overrides.get("at") or {})
        held[by] = float(level)
        overrides["at"] = held
        profiles.append(profile(model, design, variable, **overrides))
    return profiles


def sensitivity(model: Any, design: pd.DataFrame, *,
                variables: Optional[Iterable[str]] = None,
                at: Optional[Mapping[str, float]] = None,
                method: str = "median",
                quantiles: Tuple[float, float] = (0.05, 0.95),
                offset: Optional[float] = None,
                limit: Optional[int] = None) -> List[Sensitivity]:
    """Rank the inputs by how far each one moves the prediction.

    Each input is swept from its low quantile to its high quantile with
    everything else held at the reference, and the inputs are ranked by the
    magnitude of the resulting change. Quantiles rather than min/max because
    one outlier well would otherwise decide the ranking.

    This is what turns a three-thousand-column design into something a user
    can open a profiler on: the list is the answer to "which input should I
    move first".

    :param model: a fitted object.
    :param design: the design matrix.
    :param variables: restrict to these columns; default is every numeric
        column that is not constant and not the intercept.
    :param at: hold named inputs at these values.
    :param method: how the rest are held; see :func:`reference_row`.
    :param quantiles: the low and high sweep points.
    :param offset: per-row offset for a model fitted with one.
    :param limit: keep only the top this many.
    :returns: :class:`Sensitivity` records, largest absolute span first.
    """
    low_q, high_q = float(quantiles[0]), float(quantiles[1])
    row = reference_row(design, method=method, at=at)
    params = _coefficients(model)

    if variables is None:
        candidates = [str(name) for name in row.index
                      if str(name).lower() not in ("intercept", "const")]
    else:
        candidates = [str(name) for name in variables]

    rows: List[Dict[str, float]] = []
    kept: List[Tuple[str, float, float]] = []
    for name in candidates:
        if name not in design.columns:
            continue
        column = pd.to_numeric(design[name], errors="coerce").dropna()
        if column.empty:
            continue
        low = float(column.quantile(low_q))
        high = float(column.quantile(high_q))
        if not math.isfinite(low) or not math.isfinite(high) or low == high:
            continue
        for value in (low, high):
            point = row.to_dict()
            point[name] = value
            rows.append(point)
        kept.append((name, low, high))

    if not kept:
        return []

    grid = pd.DataFrame(rows)
    offsets = None if offset is None else np.full(len(grid), float(offset))
    predictions = predict(model, grid, offset=offsets)

    found: List[Sensitivity] = []
    for index, (name, low, high) in enumerate(kept):
        p_low = float(predictions[2 * index])
        p_high = float(predictions[2 * index + 1])
        coefficient = float("nan")
        if params is not None and name in params.index:
            coefficient = float(params[name])
        found.append(Sensitivity(
            variable=name, low=low, high=high, prediction_low=p_low,
            prediction_high=p_high, span=p_high - p_low,
            coefficient=coefficient))

    found.sort(key=lambda s: (-abs(s.span) if math.isfinite(s.span) else 0.0,
                              s.variable))
    return found[:limit] if limit else found
