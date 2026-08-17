"""The two well-level distributions a run writes, in the house style.

`fraction_histogram.pdf` and `log_pred_histogram.pdf` are the last two figures
the regression pipeline still drew in the old idiom -- a saturated teal
`sns.histplot` at 60% alpha on a 10x10 inch canvas, titled "Histogram of
fraction", y-axis "Frequency", followed by a `plt.show()` that pops a window
out of a GUI run. Beside a panel from :mod:`spacr.figures.panels` they read as
a mistake, which is the whole of the "the thumbnail ... looks off" complaint.

WHY THESE TWO ARE NOT IN ``panels.REGISTRY``. Every panel in that module is a
function of the COEFFICIENT table -- one row per fitted term. These two are
functions of the WELL-LEVEL table, one row per guide-in-well, which is the
input to the fit rather than its output. They cannot share a sheet built from
`results.csv` because that table has neither a `fraction` column nor a
response, so this module keeps its own registry and its own saver.

A DISTRIBUTION IN THIS STYLE, from the skill: a pale solid fill
(``ROLES['fill']``), never a translucent saturated colour; a thin grey dashed
reference where one means something; the n stated in-panel; no gridlines, no
frame around the note, no sentence title.

SAYING SOMETHING. "Distribution of fraction" is what the axis label already
says, so a panel that says only that has wasted itself. Each of these exists
to answer one question:

* the fraction histogram -- **is the library evenly represented?** A guide's
  raw fraction cannot answer that on its own, because a well holding 2 guides
  splits into shares near 1/2 and a well holding 15 into shares near 1/15;
  pooling them measures how many guides landed per well, not how evenly. On
  this screen the equal share ranges from 0.033 to 0.28 across the middle 90%
  of wells, an eight-fold spread, so a single "even" line on the raw axis
  would be a line drawn through eight different meanings. Each guide is
  therefore shown against ITS OWN well's equal share, where 1 is exact
  equality and the reference is exact.

* the response histogram -- **is the fitted family's distributional
  assumption plausible?** Not "what does the response look like": what a
  reader does next depends on the skew and the tails, so those are the
  numbers, and a normal with the same mean and SD is drawn over the bars so
  the gap between assumption and data is visible rather than inferred.

CONVENTIONS, stated once. Skewness and excess kurtosis are the moment
estimators. |skew| < 0.5 is read as near-symmetric, 0.5-1 as moderate and
above 1 as strong -- the usual rule of thumb, named in the caption so a reader
is never handed a bare adjective. The Gini coefficient is the standard
evenness statistic and runs 0 (every guide equal) to 1 (one guide holds
everything).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .panels import Panel
from .style import (ROLES, WEIGHTS, annotate, figure_style, reference_line,
                    theme_target)

#: The guide-share column, whatever the caller named it.
FRACTION_COLUMNS = ("fraction", "grna_fraction", "guide_fraction")

#: The well identifier. spaCR's is `prc` -- plate, row, column.
WELL_COLUMNS = ("prc", "prcfo", "well_id", "wellID", "well")

#: Response columns, in the order the pipeline would mean them. Only consulted
#: when the caller did not name one and the frame has more than one numeric
#: column; a panel that had to guess says which column it drew.
RESPONSE_COLUMNS = ("log_pred", "pred", "recruitment", "score", "response",
                    "value")

#: A well holding one guide has no within-well evenness to measure: its single
#: share is 1.0 of the equal share by construction. Left in, those rows pile a
#: spike of pure arithmetic onto the exact x where the reference line is, which
#: is the worst place in the panel for an artefact. On this screen that is 93
#: of 610 wells.
MIN_GUIDES_PER_WELL = 2

#: Below this many values a histogram is a rug and its moments are noise.
MIN_VALUES = 8

#: Read on |skew| and on |excess kurtosis| alike.
NEAR_SYMMETRIC = 0.5
MODERATELY_SKEWED = 1.0


# --------------------------------------------------------------------------- #
#  Finding the columns
# --------------------------------------------------------------------------- #

def _column(frame, names) -> Optional[str]:
    """The first of ``names`` this frame has."""
    for name in names:
        if name in frame.columns:
            return name
    return None


def fraction_column(frame) -> Optional[str]:
    """The guide-share column."""
    return _column(frame, FRACTION_COLUMNS)


def well_column(frame) -> Optional[str]:
    """The well identifier, or None when the frame does not carry one."""
    return _column(frame, WELL_COLUMNS)


def response_column(frame, column: Optional[str] = None) -> Optional[str]:
    """The response column, and it is worth saying how it is decided.

    An explicit name always wins -- the pipeline knows its own dependent
    variable and should pass it. Otherwise a frame with exactly one numeric
    column is unambiguous, which is the shape `dmatrices` hands back. Only
    past that does this guess from :data:`RESPONSE_COLUMNS`, and a panel that
    got here by guessing names the column it drew on the axis, so the guess is
    never invisible.
    """
    if column is not None:
        return column if column in frame.columns else None
    numeric = [name for name in frame.columns
               if str(frame[name].dtype).startswith(("float", "int"))]
    if len(numeric) == 1:
        return numeric[0]
    return _column(frame, RESPONSE_COLUMNS)


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype="float64").ravel()
    return array[np.isfinite(array)]


def _bins(n: int) -> int:
    """Bin count from the sample size, the same rule the QC report uses.

    Root-n, floored at 10 so a small screen still shows a shape and capped at
    60 so a large one does not turn into a comb.
    """
    return int(np.clip(np.sqrt(max(n, 1)), 10, 60))


# --------------------------------------------------------------------------- #
#  The statistics, separately so they can be checked without drawing
# --------------------------------------------------------------------------- #

def gini(values) -> float:
    """The Gini coefficient of a non-negative sample.

    0 when every value is equal, approaching 1 when one value holds
    everything. The standard evenness statistic for a pooled library, and the
    same one :func:`spacr.plot.plot_lorenz_curves` reports.

    THE SAME STATISTIC, NOT THE SAME NUMBER, and the difference has to be
    stated because both are labelled "Gini". The Lorenz curves take raw gRNA
    counts pooled over a plate; this panel takes each guide's share of its own
    well divided by that well's equal split. On the tsg101 screen those are
    0.20 and 0.32 -- the panel's is larger because normalising by the well
    removes the between-well spread that flattens the pooled curve. A reader
    who takes one for the other will read a change in the question as a change
    in the library.

    NaN for an empty sample or one that sums to zero, rather than a
    ZeroDivisionError or a silent 0.0 -- an evenness of zero would be read as
    "perfectly even", which is the opposite of "there was nothing to measure".
    """
    array = np.sort(_finite(values))
    if array.size == 0 or array.min() < 0:
        return float("nan")
    total = array.sum()
    if total <= 0:
        return float("nan")
    index = np.arange(1, array.size + 1)
    return float(((2 * index - array.size - 1) * array).sum()
                 / (array.size * total))


def relative_representation(frame, fraction: str, well: str):
    """Each guide's share of its well, divided by an equal split of that well.

    ``(values, dropped_wells, dropped_rows)``. 1.0 means the guide holds
    exactly its share; 2.0 means twice what an equal split would give it.

    THE POINT OF DIVIDING. The raw fraction confounds evenness with how many
    guides were retained in the well: a two-guide well and a fifteen-guide
    well produce shares an order of magnitude apart with no unevenness
    whatsoever. Dividing by the well's own equal share removes exactly that
    and nothing else, and it is what makes a single reference line legitimate.
    """
    shares = np.asarray(frame[fraction], dtype="float64")

    # THE WELL IS COUNTED OVER ITS USABLE SHARES, NOT OVER ITS ROWS. `size`
    # counts a row whose share is NaN or inf; `sum` does not. A well holding
    # one unknown share therefore divided its total by one guide too many, so
    # every OTHER guide in that well came out over-represented -- silently,
    # because the unknown row is dropped and the dropped-row count still looks
    # right. Worse, a well holding one real guide and one unknown passed the
    # two-guide guard and landed at exactly 2.0x: an artefact of pure
    # arithmetic sitting on the "at least twice equal" cut the panel reports,
    # which is the same failure single-guide wells are excluded to avoid.
    usable_share = np.where(np.isfinite(shares), shares, np.nan)
    per_well = frame[[well]].copy()
    per_well["_share"] = usable_share
    grouped = per_well.groupby(well, observed=True)["_share"]
    counts = grouped.transform("count").to_numpy()
    totals = grouped.transform("sum").to_numpy(dtype="float64")

    usable = (np.isfinite(shares) & np.isfinite(totals) & (totals > 0)
              & (counts >= MIN_GUIDES_PER_WELL))
    equal = np.divide(totals, counts, out=np.full(totals.shape, np.nan),
                      where=usable)
    values = np.divide(shares, equal, out=np.full(shares.shape, np.nan),
                       where=usable)
    values = values[usable]

    kept = frame.loc[usable, well] if usable.any() else frame.loc[[], well]
    dropped_wells = int(frame[well].nunique() - kept.nunique())
    return values, dropped_wells, int((~usable).sum())


def shape_of(values) -> dict:
    """Skewness, excess kurtosis and the word that goes with them.

    Returned rather than printed so a test can assert the number a reader is
    shown, and so the console summary (instruction 124 D) can quote the same
    one the panel does instead of computing its own.
    """
    from scipy import stats

    array = _finite(values)
    if array.size < 3 or array.std() == 0:
        return {"n": int(array.size), "skew": float("nan"),
                "excess_kurtosis": float("nan"), "verdict": "not measurable"}
    skew = float(stats.skew(array))
    kurtosis = float(stats.kurtosis(array))       # already excess
    if abs(skew) < NEAR_SYMMETRIC:
        verdict = "near-symmetric"
    elif abs(skew) < MODERATELY_SKEWED:
        verdict = "moderately skewed"
    else:
        verdict = "strongly skewed"
    if abs(skew) < NEAR_SYMMETRIC and abs(kurtosis) >= MODERATELY_SKEWED:
        # Symmetric and heavy-tailed is a real and different failure: a normal
        # fitted to it has the right centre and the wrong tail probabilities,
        # which is precisely what a p-value is computed from.
        verdict = "symmetric, heavy-tailed"
    if skew >= NEAR_SYMMETRIC:
        verdict += " right"
    elif skew <= -NEAR_SYMMETRIC:
        verdict += " left"
    return {"n": int(array.size), "skew": skew,
            "excess_kurtosis": kurtosis, "verdict": verdict}


def one_value_per_well(frame, column: str, well: Optional[str]):
    """The response once per well, when it is a per-well quantity.

    ``(values, deduplicated)``.

    THE BUG THIS EXISTS FOR. The pipeline hands the response as one row per
    guide-in-well, and the response is a property of the WELL -- on this
    screen `log_pred` has exactly one distinct value in each of the 610 wells,
    repeated once per guide the well retained. The old histogram therefore
    counted a 15-guide well fifteen times and stated n = 1,945 for 610
    independent observations, overstating the evidence three-fold and
    reshaping the distribution towards whatever the crowded wells did.

    Checked rather than assumed: a response that genuinely varies within a
    well is left alone, because collapsing it would then be the error.
    """
    if well is None or well not in frame.columns:
        return _finite(frame[column]), False
    varies = frame.groupby(well, observed=True)[column].nunique(dropna=True)
    if (varies > 1).any():
        return _finite(frame[column]), False
    return _finite(frame.groupby(well, observed=True)[column].first()), True


# --------------------------------------------------------------------------- #
#  The panels
# --------------------------------------------------------------------------- #

def _ratio_ticks(ax, values) -> None:
    """Powers of two labelled as MULTIPLIERS, not as exponents.

    matplotlib's log-base-2 default writes ``2^-3``, and an axis captioned
    "× equal share" then asks the reader to exponentiate before they can say
    whether a guide is under-represented. The tick positions stay powers of
    two -- that is what makes half and twice equidistant -- and only the text
    changes: 0.25, 0.5, 1, 2, 4.
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    low = int(np.floor(np.log2(values.min())))
    high = int(np.ceil(np.log2(values.max())))
    exponents = list(range(low, high + 1))
    step = max(1, int(np.ceil(len(exponents) / 6)))
    # THINNED ON e ≡ 0, never by slicing from the low end. Plain `[::step]`
    # starts at whatever the smallest guide happened to be, and on this screen
    # that dropped the tick at 1 -- leaving the reference line standing over
    # an unlabelled position, on the one axis where 1 is the entire point.
    exponents = [e for e in exponents if e % step == 0]
    powers = np.array([2.0 ** e for e in exponents])
    # SIGNIFICANT DIGITS, NOT FOUR DECIMAL PLACES. A fixed `.4f` is only
    # honest down to 2^-6: 2^-9 printed as "0.002" (2.4% out), 2^-12 as
    # "0.0002" (18% out), and 2^-15 rounded to "0.0000", which `.rstrip("0")`
    # then turned into the label "0." -- twice over on a wide enough axis, so
    # two different ticks read the same. Reachable from the raw view of any
    # deeply sequenced library, where a guide's share of a well is 1e-4.
    labels = [f"{p:g}" if p >= 1 else f"{p:.3g}" for p in powers]
    ax.xaxis.set_major_locator(FixedLocator(powers))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    # The minor ticks of a log axis are the between-decade marks; on a base-2
    # axis they land on top of the majors and thicken every tick.
    ax.xaxis.set_minor_locator(NullLocator())


def _normal_reference(ax, values, bins_edges) -> None:
    """A normal with the sample's own mean and SD, over the counts.

    Scaled by ``n * bin width`` rather than drawing the histogram as a
    density, so the y-axis stays a count of wells -- a reader can still read
    "how many" off it, which is what the axis label promises.

    Grey, thin and dashed: it is a reference, not a result, and the skill is
    explicit that a reference must never out-weigh the data it is drawn over.
    """
    from scipy import stats

    width = float(np.diff(bins_edges).mean())
    grid = np.linspace(bins_edges[0], bins_edges[-1], 200)
    curve = stats.norm.pdf(grid, values.mean(), values.std(ddof=1))
    ax.plot(grid, curve * values.size * width, color=ROLES["reference"],
            lw=WEIGHTS["reference"], ls=(0, (4, 3)), zorder=3)


def guide_fraction(ax, frame, *, well: Optional[str] = None,
                   bins: Optional[int] = None, relative: bool = True) -> Panel:
    """Is the library evenly represented within a well?

    Each guide against its own well's equal share, on a log2 axis because the
    quantity is a RATIO: half and twice equal representation are the same
    distance from 1, which they are not on a linear axis, and a library's
    abundances are log-normal to begin with.
    """
    fraction = fraction_column(frame)
    if fraction is None:
        return Panel("guide_fraction", "guide representation", drawn=False,
                     reason="no fraction column", needs=("fraction",))
    well = well or well_column(frame)
    if relative and well is None:
        return Panel("guide_fraction", "guide representation", drawn=False,
                     reason=("no well column, so a guide's share cannot be "
                             "compared with its own well's equal share"),
                     needs=(fraction, "prc"))

    if relative:
        values, dropped_wells, _dropped_rows = relative_representation(
            frame, fraction, well)
        unit, floor = "× equal share", 1.0
    else:
        values, dropped_wells = _finite(frame[fraction]), 0
        unit, floor = "guide fraction of well", None

    values = values[values > 0]
    if values.size < MIN_VALUES:
        return Panel("guide_fraction", "guide representation", drawn=False,
                     reason=(f"only {values.size} usable guide shares; a "
                             f"histogram needs at least {MIN_VALUES}"),
                     needs=(fraction,))

    # A library where every retained guide holds exactly the same share is
    # degenerate but not impossible (a run that kept one guide per well and a
    # caller who asked for the raw view). geomspace over a zero-width range
    # returns identical edges, which matplotlib bins into nothing.
    low, high = float(values.min()), float(values.max())
    if high <= low:
        low, high = low / 2.0, high * 2.0
    edges = np.geomspace(low, high, (bins or _bins(values.size)) + 1)
    ax.hist(values, bins=edges, color=ROLES["fill"], edgecolor="none")
    ax.set_xscale("log", base=2)
    _ratio_ticks(ax, values)
    if floor is not None:
        reference_line(ax, x=floor, label="equal share")
    ax.set_xlabel(unit)
    ax.set_ylabel("guides")

    spread = np.quantile(values, [0.1, 0.9])
    evenness = gini(values)
    over = float(np.mean(values >= 2.0))
    # UPPER LEFT, because the reference line stands at 1 with its own rotated
    # label and the distribution's right shoulder is under the upper right.
    # The left is the thin end of a log-ratio histogram and is always free.
    #
    # The last two lines are only true of the relative view: on the raw axis
    # there is no "equal" for a guide to be twice of, so saying "≥ 2× equal"
    # there would attach a meaning the panel did not measure.
    note = (f"n = {values.size:,} guides\nGini = {evenness:.2f}\n"
            + (f"80% within {spread[0]:.2f}–{spread[1]:.2f}×\n"
               f"{over:.0%} at ≥ 2× equal" if relative else
               f"80% within {spread[0]:.3f}–{spread[1]:.3f}\n"
               f"median {np.median(values):.3f}"))
    annotate(ax, note, x=0.02, ha="left")

    dropped = (f" Wells holding a single guide ({dropped_wells}) are excluded: "
               f"one guide is 1× its own equal share by construction, and "
               f"there is no within-well evenness to measure."
               if dropped_wells else "")
    if relative:
        built = (f"each as its share of its well divided by an equal split of "
                 f"that well, so 1 is exact equality; the dashed line marks "
                 f"it. The axis is log2 because the quantity is a ratio. The "
                 f"middle 80% span {spread[0]:.2f}–{spread[1]:.2f}× equal "
                 f"representation and {over:.0%} of guides hold at least "
                 f"twice it")
    else:
        # No well column: the shares of a 2-guide and a 15-guide well are
        # pooled, so the spread below is evenness AND how many guides landed
        # per well together. Said plainly rather than left for the reader to
        # discover, because the same picture means much less here.
        built = (f"each as its raw share of its well, on a log2 axis. Wells "
                 f"retaining different numbers of guides are pooled, so this "
                 f"spread ({spread[0]:.3f}–{spread[1]:.3f} over the middle "
                 f"80%) mixes uneven representation with how many guides a "
                 f"well kept; no well column was available to separate them")
    return Panel(
        "guide_fraction", "guide representation",
        caption=(f"Representation of {values.size:,} gRNAs, {built} "
                 f"(Gini = {evenness:.2f}, where 0 is a perfectly even "
                 f"library and 1 is one guide holding everything).{dropped}"),
        needs=(fraction,) + ((well,) if relative else ()))


def response(ax, frame, *, column: Optional[str] = None,
             well: Optional[str] = None, bins: Optional[int] = None,
             family: str = "gaussian") -> Panel:
    """Is the fitted family's distributional assumption plausible?

    The response with a normal of the same mean and SD over it, and the two
    numbers that decide what a reader does next: skewness and excess
    kurtosis. A bare "distribution of the response" leaves them to judge
    symmetry by eye, which is exactly what nobody can do.
    """
    name = response_column(frame, column)
    if name is None:
        return Panel("response", "response distribution", drawn=False,
                     reason="no response column could be identified",
                     needs=("log_pred",))
    well = well or well_column(frame)
    values, deduplicated = one_value_per_well(frame, name, well)
    if values.size < MIN_VALUES:
        return Panel("response", "response distribution", drawn=False,
                     reason=(f"only {values.size} finite response values; a "
                             f"histogram needs at least {MIN_VALUES}"),
                     needs=(name,))

    counts, edges, _patches = ax.hist(
        values, bins=bins or _bins(values.size), color=ROLES["fill"],
        edgecolor="none")
    stats_ = shape_of(values)
    # The reference is the family that was FITTED. Drawing a normal over a
    # Poisson or a beta fit would put a curve on the panel that no part of the
    # model ever assumed, and a reader would take the mismatch for a finding.
    normal = bool(values.std(ddof=1) > 0 and family == "gaussian")
    if normal:
        _normal_reference(ax, values, edges)

    unit = "wells" if deduplicated else "observations"
    ax.set_xlabel(name.replace("_", " "))
    ax.set_ylabel(unit)

    # A transform is only worth having if it did something, and the pipeline
    # names its transformed response after the raw one. When both are here,
    # say what the transform bought -- it is the one number that tells the
    # maintainer whether to keep it.
    before = ""
    raw = name[4:] if name.startswith("log_") else None
    if raw and raw in frame.columns:
        raw_values, _ = one_value_per_well(frame, raw, well)
        if raw_values.size >= 3:
            raw_skew = shape_of(raw_values)["skew"]
            before = (f"\nskew before log: {raw_skew:+.2f}")

    annotate(ax,
             f"n = {values.size:,} {unit}\n"
             f"skew = {stats_['skew']:+.2f}{before}\n"
             f"excess kurtosis = {stats_['excess_kurtosis']:+.2f}\n"
             f"{stats_['verdict']}",
             x=0.98, ha="right")

    collapsed = (f" One value per well: the response is constant within a "
                 f"well, so the {len(frame):,} guide-level rows collapse to "
                 f"{values.size:,} independent observations."
                 if deduplicated else "")
    reference = (f", with a normal of the same mean ({values.mean():.3g}) and "
                 f"SD ({values.std(ddof=1):.3g}) dashed over it" if normal
                 else f" (fitted family: {family}, so no normal is drawn)")
    return Panel(
        "response", "response distribution",
        caption=(f"Distribution of {name.replace('_', ' ')} over "
                 f"{values.size:,} {unit}{reference}. "
                 f"Skewness {stats_['skew']:+.2f} and excess "
                 f"kurtosis {stats_['excess_kurtosis']:+.2f} "
                 f"({stats_['verdict']}; |skew| below 0.5 is read as "
                 f"near-symmetric, above 1 as strong). The fit assumes normal "
                 f"RESIDUALS rather than a normal response, so this is a "
                 f"flag to read the residual and q-q panels with, not a "
                 f"verdict on the model.{collapsed}"),
        needs=(name,))


#: This module's catalog. Deliberately NOT merged into
#: :data:`spacr.figures.panels.REGISTRY`: those panels take the coefficient
#: table and these take the well-level one, and a registry whose entries want
#: different frames is a registry that cannot be iterated.
REGISTRY: Dict[str, Callable] = {
    "guide_fraction": guide_fraction,
    "response": response,
}

#: Reading order, same principle as the sheet's: the input to the fit before
#: the thing that was fitted.
ORDER: Tuple[str, ...] = ("guide_fraction", "response")

#: The file names a run has always written. Kept exactly, because the grid
#: view, the queue and `tests/test_cov_ml_regression_core.py` all find these
#: figures by name.
FILENAMES = {"guide_fraction": "fraction_histogram",
             "response": "{response}_histogram"}


def build_panel(key: str, frame, *, target: Optional[str] = None,
                figsize=(3.4, 2.6), **kwargs):
    """One distribution panel on its own figure. ``(figure, Panel)``.

    The same shape and the same margins as :func:`spacr.figures.sheet.build_panel`
    so a saved distribution sits beside a saved volcano at the same size on
    the grid. The style is a CONTEXT MANAGER, as everywhere in this package:
    spaCR draws from a long-lived GUI and a global rcParams write would
    restyle every later figure in the session.
    """
    import matplotlib.pyplot as plt

    with figure_style(target or theme_target()):
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111)
        panel = REGISTRY[key](ax, frame, **kwargs)
        figure.subplots_adjust(left=.16, right=.97, top=.92, bottom=.16)
        return figure, panel


def save_distributions(frame, dst, *, response_variable: Optional[str] = None,
                       target: Optional[str] = None,
                       order: Sequence[str] = ORDER) -> Dict[str, str]:
    """Write both distributions into a run's results folder.

    ``{key: path}`` for what was written, and a key is simply absent when its
    panel could not be drawn -- an empty figure in a results folder is worse
    than a missing one, because the grid view will show it.

    Replaces the two `plot_histogram` calls in
    :func:`spacr.ml.regression_model`. It does NOT call `plt.show`: the old
    one did, which pops a window out of a headless or GUI-driven run.

    THE DEFAULT TARGET IS ``'print'``, NOT :func:`theme_target`. This function
    writes a FILE, and a file is read on a page. ``theme_target()`` answers a
    different question -- what is the GUI theme doing -- and returns
    ``'screen'`` for every user who has not explicitly set a white figure
    background, which resolves to ``INK_SCREEN`` (#E8EDEE) on a transparent
    PDF: near-white axes, ticks and labels on a white page.
    :data:`spacr.regression_qc._REPORT_TARGET` states the same rule for the QC
    report, and :func:`spacr.ml._save_regression_figure` passes ``'print'``
    for the sheet these two figures land beside on the grid. Passing
    ``target`` explicitly still wins, for a caller drawing into the GUI.
    """
    import os

    import matplotlib.pyplot as plt

    from ..plot import save_figure

    written: Dict[str, str] = {}
    for key in order:
        kwargs = {"column": response_variable} if key == "response" else {}
        figure, panel = build_panel(key, frame, target=target or "print",
                                    **kwargs)
        if not panel.drawn:
            plt.close(figure)
            print(f"Skipped {key}: {panel.reason}")
            continue
        stem = FILENAMES[key].format(
            response=response_variable
            or response_column(frame, response_variable) or "response")
        written[key] = save_figure(figure, os.path.join(dst, f"{stem}.pdf"),
                                   close=True)
    return written


__all__ = ["FILENAMES", "MIN_GUIDES_PER_WELL", "MIN_VALUES", "ORDER",
           "REGISTRY", "RESPONSE_COLUMNS", "build_panel", "fraction_column",
           "gini", "guide_fraction", "one_value_per_well",
           "relative_representation", "response", "response_column",
           "save_distributions", "shape_of", "well_column"]
