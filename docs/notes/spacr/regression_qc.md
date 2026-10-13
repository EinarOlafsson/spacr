# Notes from `spacr/regression_qc.py`

Prose lifted out of `spacr/regression_qc.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [leverage_from_design](#leverage_from_design) (2 entries)
- [variance_inflation_factors](#variance_inflation_factors) (2 entries)
- [condition_number](#condition_number) (1 entry)
- [condition_number._ratio](#condition_number_ratio) (1 entry)
- [calibration_curve](#calibration_curve) (1 entry)
- [residual_normality](#residual_normality) (1 entry)
- [diagnose_p_value_histogram](#diagnose_p_value_histogram) (1 entry)
- [ResidualStandardisation](#residualstandardisation) (1 entry)
- [_decision_score](#_decision_score) (4 entries)
- [_well_labels](#_well_labels) (1 entry)
- [_align_metadata](#_align_metadata) (1 entry)
- [build_context](#build_context) (4 entries)
- [_note](#_note) (1 entry)
- [_trend](#_trend) (1 entry)
- [_trend_off_the_ties](#_trend_off_the_ties) (1 entry)
- [_skip_box](#_skip_box) (1 entry)
- [_panel_residuals_vs_fitted](#_panel_residuals_vs_fitted) (1 entry)
- [_panel_residual_distribution](#_panel_residual_distribution) (4 entries)
- [_panel_scale_location](#_panel_scale_location) (3 entries)
- [_panel_qq_residuals](#_panel_qq_residuals) (3 entries)
- [_panel_observed_vs_predicted](#_panel_observed_vs_predicted) (1 entry)
- [_panel_cooks_distance](#_panel_cooks_distance) (2 entries)
- [_panel_influence](#_panel_influence) (1 entry)
- [_panel_dffits](#_panel_dffits) (3 entries)
- [_house_axes](#_house_axes) (3 entries)
- [_panel_vif](#_panel_vif) (4 entries)
- [_panel_condition_number](#_panel_condition_number) (4 entries)
- [_panel_predictor_correlation](#_panel_predictor_correlation) (7 entries)
- [_panel_coefficient_forest](#_panel_coefficient_forest) (3 entries)
- [_panel_p_value_histogram](#_panel_p_value_histogram) (3 entries)
- [_panel_response_distribution](#_panel_response_distribution) (1 entry)
- [_panel_count_fit](#_panel_count_fit) (1 entry)
- [_grouped_residuals](#_grouped_residuals) (1 entry)
- [_positional_effect_panel](#_positional_effect_panel) (6 entries)
- [_panel_cell_count_vs_effect](#_panel_cell_count_vs_effect) (2 entries)
- [_panel_volcano_reference](#_panel_volcano_reference) (1 entry)
- [Module level](#module-level) (1 entry)
- [_score_observed_vs_predicted](#_score_observed_vs_predicted) (1 entry)
- [_score_precision_recall](#_score_precision_recall) (1 entry)
- [draw_verdict](#draw_verdict) (1 entry)
- [_save](#_save) (1 entry)
- [format_qc_report](#format_qc_report) (3 entries)
- [regression_qc_report](#regression_qc_report) (9 entries)
- [_write_qc_numbers._plain](#_write_qc_numbers_plain) (1 entry)
- [_write_qc_numbers](#_write_qc_numbers) (2 entries)
- [_write_combined_page](#_write_combined_page) (2 entries)

## leverage_from_design

### lines 403-406

```python
def leverage_from_design(X, weights=None):
```

Statistics — public because they are what the tests assert on, and because each is useful on its own from a notebook.

### lines 446-448

```python
if np.any(hat < -1e-6) or np.any(hat > 1 + 1e-6):
```

Numerically h can come out a hair outside [0, 1]; clipping keeps sqrt(1 - h) real. Anything materially outside the interval is a bug, so only the float noise is absorbed.

## variance_inflation_factors

### line 559

```python
for col in varying:
```

One varying predictor cannot be collinear with anything.

### lines 572-574

```python
loading = np.abs(eigvecs[:, null_mask]).max(axis=1)
```

A direction with (essentially) zero variance is an exact linear dependency; every column with weight in it is unidentified, and its VIF is infinite, not the finite number pinv would hand back.

## condition_number

### lines 607-609

```python
norms = np.where(norms > 0, norms, 1.0)
```

A genuinely all-zero column has no direction; scaling it by 1 leaves it zero, which is exactly what makes the matrix singular, and the singular value of 0 that follows is the honest answer.

## condition_number._ratio

### lines 622-626

```python
tolerance = np.finfo(sv.dtype).eps * max(Xm.shape) * float(sv[0])
```

LAPACK implementations do not all return an exact zero for the same rank-deficient matrix. Use the numerical-rank threshold behind ``numpy.linalg.matrix_rank`` so a duplicated predictor is singular on every supported runner, including when roundoff leaves a tiny positive final singular value.

## calibration_curve

### lines 697-700

```python
edges = np.linspace(yp.min(), yp.max() + 1e-12, n_bins + 1)
```

Predictions are (nearly) constant: quantile edges collapse and every point lands in one bin, which is not a curve. Fall back to equal width over the observed range so the panel still says something true, and let the caller see it in `n_bins`.

## residual_normality

### lines 831-833

```python
return {"skew": float("nan"), "excess_kurtosis": float("nan"),
```

skew and kurtosis of two points are not undefined so much as meaningless, and scipy returns them without complaint. Naming the count is the only honest answer.

## diagnose_p_value_histogram

### lines 913-915

```python
middle = counts[1:-1]
```

The middle is the reference: it is where the null lives whatever the tails are doing, so comparing the ends to it (rather than to each other) keeps the verdict stable as the number of real hits changes.

## ResidualStandardisation

### lines 950-952  _(unsure)_

```python
@dataclass
```

Residual standardisation — one registry, keyed on the fitted model's class

## _decision_score

### lines 1434-1435

```python
return None
```

An estimator that advertises decision_function and cannot run it on its own design is not a reason to lose the other twenty panels.

### lines 1438-1439  _(unsure)_

```python
return None
```

Multi-class one-vs-rest returns (n, k): there is no single ranking to draw one ROC from, so the panel falls back and says so.

### line 1447, trailing  _(unsure)_

```python
return score
```

non-numeric labels

### lines 1449-1451

```python
return -score
```

classes_[1] is the SMALLER label, so a larger decision value means a more NEGATIVE well. sklearn sorts classes_ and so never lands here; a hand-rolled or wrapped estimator can.

## _well_labels

### lines 1513-1516

```python
values = metadata[list(parts)].to_numpy(dtype=object)
```

Pandas 3's extension dtypes can retain numeric scalars through ``astype(str)``; joining the rows then raises because ``join`` receives floats.  Convert each scalar at the Python boundary so mixed numeric/string plate metadata always yields text labels.

## _align_metadata

### lines 1543-1546

```python
found = pd.Index(index).isin(metadata.index)
```

`index.isin(metadata.index)`, not the reverse: the question is whether every FITTED row can be found, and a duplicated metadata index would make .loc fan the frame out, so uniqueness is required before the lookup is allowed.

## build_context

### lines 1611-1615

```python
kind, model_class = _model_kind(model)
```

A weighted fit's hat matrix carries its weights. Take them from the model rather than from the caller: they are the weights `model.scale` was formed with, so they are the only ones that make the scale, the residual and the hat diagonal agree. A caller who passes cell counts that are not the fitted weights would otherwise get a leverage for a fit nobody ran.

### lines 1619-1621

```python
leverage, source = None, ""
```

Leverage: prefer whatever the model itself computed, because for a GLM the IRLS weights belong in the hat matrix and the design matrix alone does not know them.

### lines 1638-1642

```python
leverage = None
```

statsmodels raises a different exception per model class for

"influence is not defined here" (MixedLM, regularised fits); the fallback below is exact for the unweighted case and explicitly labelled, so a broad catch costs nothing but the attempt.

### lines 1664-1667

```python
scale = float("nan")
```

No correct scale exists for this model class. An all-NaN array is not a fallback: it is what forces the panels built on a standardised residual to skip, with `standardisation.reason` printed, instead of naming outlier wells off a number that is not a z-score.

## _note

### lines 1784-1785

```python
bbox=dict(boxstyle="round,pad=0.3",
```

178 A: the same fault, the same fix -- this box carries the panel's own note in `color`, which follows the theme.

## _trend

### lines 1820-1837

```python
return float(_trend_off_the_ties(sx, sy))
```

A TIE IS NOT A TREND.

A well-level fit routinely has most of its fitted values identical on the tsg101 screen 451 of 610 wells share one value to seven decimal places, because most wells carry the same guide mixture. LOWESS fits a local regression in a neighbourhood; where the neighbourhood is a single repeated x it is fitting a line through a vertical stack, and the smoothed value there is unconstrained. Two adjacent points 2e-6 apart came back 0.12 apart.

Taking the maximum over that included the artefact, so the panel reported |trend| max = 0.109 where the real curve away from the tie block spans 0.030 -- a 3.6x inflation, on the number the panel exists to report, in the one artist the panel draws in colour.

So the trend is measured where x actually varies. The curve is still drawn in full: hiding the spike would hide the tie, and the tie is worth seeing.

## _trend_off_the_ties

### lines 1863-1864  _(unsure)_

```python
gap_before = np.diff(sx, prepend=sx[0] - span)
```

A point is trustworthy when at least one of its neighbours is a real distance away in x.

## _skip_box

### lines 1905-1909

```python
ax.patch.set_visible(True)
```

set_axis_off() hides the patch as well, so the grey tile has to be turned back on explicitly — otherwise a skipped panel is an invisible gap, which is precisely the failure mode this box exists to prevent. The dashed border does the same job at a glance: a reader scanning the page must be able to tell "not computed" from "computed and unremarkable".

## _panel_residuals_vs_fitted

### lines 1945-1947

```python
ax.scatter(ctx.fitted, ctx.resid, s=18, color=ROLES["data"],
```

The wells are the default ink. This panel asks exactly one question — is there structure left in the residuals — and the smoother is the answer, so the smoother is the only artist entitled to colour.

## _panel_residual_distribution

### lines 1985-1988

```python
ax.plot(grid, sps.gaussian_kde(resid)(grid),
```

A KDE on constant data raises inside scipy (singular covariance); the range check keeps that failure from reaching the report. The empirical density is what the panel is ABOUT — whether the residuals are normal — so it is the one thing that gets colour.

### lines 1991-1993

```python
ax.plot(grid, sps.norm.pdf(grid, resid.mean(), resid.std(ddof=1) or 1e-12),
```

The normal curve is a reference, not a result: grey, thin and dashed, exactly like style.reference_line, which cannot draw it because it is a curve rather than a horizontal or vertical rule.

### lines 2002-2005

```python
shape = residual_normality(resid)
```

ONE STATEMENT OF THE VERDICT, shared with the summary -- see

`residual_normality`. The panel and `spacr.regression_summary` print the same three numbers about the same residuals, and they only stay the same three numbers because there is one function.

### lines 2013-2015

```python
annotate(ax, f"{_wells(resid.size)}\nskew = {skew:+.2f}\n"
```

The n moved off the title and into the note: the descriptor says which panel this is, the note carries everything a reader needs to judge the number.

## _panel_scale_location

### lines 2053-2054  _(unsure)_

```python
levene_p, sd_ratio = float("nan"), float("nan")
```

Brown-Forsythe: Levene centred on the median, which is the version that survives the heavy-tailed residuals screen data actually produces.

### lines 2070-2071

```python
levene_p = float("nan")
```

scipy refuses when a group is constant; that is a real answer about the data, not a panel failure.

### lines 2098-2100

```python
flat = verdict == "no detectable trend in spread"
```

The verdict is the one line a reader acts on, so it is the one line that changes colour: rust when the panel is saying the variance is not constant, plain ink when it is saying nothing is wrong.

## _panel_qq_residuals

### lines 2119-2120  _(unsure)_

```python
quantiles = sps.norm.ppf((np.arange(1, sample.size + 1) - 0.375)
```

Blom plotting positions: the standard choice, and unbiased for the normal order statistics that the reference line assumes.

### lines 2129-2132

```python
shape = residual_normality(ctx.resid)
```

The numerical normality test belongs on the Q-Q panel as well as in the residual-distribution panel.  A Q-Q plot is otherwise an invitation to make an unrecorded visual judgement, and the manuscript diagnostic asks for the exact K-squared, skew and tail-weight values beside the points.

### lines 2142-2145

```python
ax.plot(xs, intercept + slope * xs, color=ROLES["reference"],
```

The quartile line is a REFERENCE, not a result: the claim of this panel is whether the points follow it, so it is drawn the way every other threshold in the report is — thin, dashed and grey. It used to be the boldest artist on the axes.

## _panel_observed_vs_predicted

### lines 2186-2189

```python
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
```

The skill's own words for this plot: "scatter, grey, with a highlighted subset, dotted 1:1 diagonal". There is no subset to highlight here — the whole cloud is the answer — so the diagonal is dotted grey and nothing else is coloured.

## _panel_cooks_distance

### lines 2227-2230

```python
ax.vlines(below, 0, heights[below], color=ROLES["data"],
```

The sentence is "these wells are influential", so the wells above the rule are the only thing on the panel that carries colour, and they are drawn from the same `above` set the stats report. Every other well is the grey it was always entitled to.

### lines 2243-2245

```python
side = "left" if worst >= ctx.n / 2 else "right"
```

The note goes to whichever top corner the worst well is NOT in: the tallest stem carries a label, and a fixed corner puts the two on top of each other whenever the worst well lands on that side.

## _panel_influence

### lines 2275-2277

```python
ax.scatter(ctx.leverage[~named], ctx.std_resid[~named],
```

The five wells the panel NAMES are the five it colours. Labelling a well in one colour and drawing its bubble in another was the panel saying two different things about the same well.

## _panel_dffits

### lines 2321-2335

```python
finite = magnitude[np.isfinite(magnitude)]
```

A SATURATED ROW HAS leverage == 1 AND THEREFORE DFFITS == +inf, and the real tsg101 fit has 186 of them. `nan_to_num(magnitude, nan=0.0)` leaves the infinity alone except to swap it for 1.797e308, which overflows the y-autoscale; matplotlib then falls back to a DEGENERATE view, ylim == (-1e-12, 1e-12). Everything positioned in DATA coordinates `reference_line`'s label at y = threshold, the well labels -- lands about 1e12 axes-heights off the page, so the tight bounding box is two trillion inches tall and `savefig(bbox_inches="tight")` raises out of the Agg renderer. That took down the whole combined report, not just this panel.

So the DRAWING is clamped to the largest finite |DFFITS| while the STATISTICS below are left exactly as they were: `max_abs_dffits` is still `inf` and an unbounded well is still in `flagged`. Clamping the number instead of the mark would be a re-analysis.

### lines 2353-2354

```python
ax.set_ylim(0.0, ceiling * 1.08)
```

Explicit, because a single unbounded well is enough to make the autoscale useless even after the heights are clamped.

### lines 2365-2369

```python
return {"threshold": float(threshold), "n_above": int(above.size),
```

`n_points` so the verdict can score the FRACTION above the line rather than the maximum. 2*sqrt(p/n) is a screening threshold that a correct model is EXPECTED to exceed for a few percent of observations, so on 400 wells the largest |DFFITS| is routinely twice it and a rule read off the maximum flags a clean fit -- measured, and it did.

## _house_axes

### lines 2428-2434

```python
ax.tick_params(which="minor", colors=ink, labelsize=TYPE_SCALE["tick"],
```

MINOR ticks on the same terms. ``tick_params`` defaults to

``which='major'``, and a log axis grows a second, minor set: without this line the cell-count panel labelled its minor decades at matplotlib's default 10 pt, larger than the axis label at 7 and half again the major ticks at 6.2, which made "2 x 10^2" the loudest type in the figure. Set unconditionally — an axis with no minor ticks is unaffected, and a panel that adds a log scale keeps the setting.

### line 2438  _(unsure)_

```python
spine.set_visible(side in ("left", "bottom"))
```

Cell-style L framing: left and bottom only.

### lines 2443-2450

```python
ax.grid(False, which="both")
```

NO GRIDLINES. EVER — the style module's own words. ``rc()`` sets ``axes.grid`` False, but rcParams decide an artist's fate when it is CREATED and the report driver creates this axes outside the style context. That is exactly why the ink, the type and the spines are pushed on here instead of being left to the context manager, and the grid belongs on that list: any caller holding a global grid-on style (``spacr.figure_style.apply`` defaults ``grid`` to True) otherwise rules a spreadsheet through every panel in this report.

## _panel_vif

### lines 2485-2490

```python
colors = [ROLES["down"] if v > 10 else ROLES["data"] for v in shown]
```

Grey and RUST, and nothing else. The traffic light this replaces spent GREEN on a healthy VIF, and GREEN is `up` in the shared vocabulary: a reader who learned that from the coefficient forest would read "no collinearity problem" as "called, upregulated". It also coloured every bar, so on a healthy design the panel shouted in three hues while claiming nothing.

### lines 2498-2505

```python
reference_line(ax, x=guide).set_zorder(2)
```

style.reference_line parks a rule at zorder 0, which is right for a scatter and wrong for bars: the bars would cover the very thresholds they are meant to be read against. Lifted just above them; still thin, still dashed, still grey. Unlabelled, because the x axis already has a tick at 5 and at 10 and the note below counts how many predictors are past each -- the old rotated "VIF=5" tag was ink laid over the top bar to say what two ticks were already saying.

### lines 2512-2514

```python
ax.set_xlim(0.0, max(float(np.nanmax(plot_values)) * 1.45, 11.5))
```

Room for the "inf (aliased)" tags to the right of the longest bar, and never less than the VIF = 10 guide, which on a healthy design sits far beyond every bar and still has to be on the panel.

### lines 2516-2519

```python
ax.set_ylim(len(shown) + 4.5, -0.5)
```

Room UNDER the last bar for the note. A ranked-bar panel has no empty corner -- the old note was a white box laid over the bottom five bars and their aliasing tags, which is what a box is for and exactly why the style has none.

## _panel_condition_number

### lines 2547-2549

```python
ax.bar(positions, np.where(singular > 0, singular, np.nan),
```

The spectrum has no minority to highlight — its SHAPE is the claim — so the bars stay grey and the colour is spent on the verdict, which is the only thing here that can be a warning.

### lines 2556-2563

```python
span = max(top / bottom, 1.0)
```

Headroom for the verdict, measured in decades rather than pixels. The old panel wrote the number at 0.88 of the axes and the verdict at 0.75, both straight over the bars, which on any design with a full-height spectrum made both unreadable. A fixed multiplier cannot work either: a healthy spectrum spans a fraction of one decade and a singular one spans sixteen. The floor of 3 is for the orthonormal case, where the spectrum is flat and proportional headroom would be no headroom at all.

### lines 2569-2575

```python
warning = ROLES["down"] if severe else ink
```

All three blocks are one left-aligned column in the headroom, and not, as before, a centred headline over the bars with the numbers in the bottom-left corner. A singular-value spectrum starts at the axes floor, so on a design whose bars span the panel there IS no bottom-left corner; and anything centred collides with anything right-aligned once the panel is a 4.6-inch cell on the combined page rather than a figure of its own.

### lines 2577-2580

```python
ax.text(0.02, 0.98,
```

One tier up from an annotation, because this number IS the panel — the bars are only the evidence for it. RUST when the design is badly conditioned, plain ink when it is not; never GREEN, which in the shared vocabulary means "called up" and would read as a result.

## _panel_predictor_correlation

### lines 2608-2610

```python
keep = varying.std(ddof=1).sort_values(ascending=False).head(limit).index
```

Beyond ~40 rows the cells are smaller than the axis labels, so the heatmap stops being readable. Keep the predictors with the largest spread, which are the ones carrying the design.

### lines 2619-2625

```python
image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
```

THE DIVERGING MAP STAYS. The skill permits one exactly here:

"diverging colormaps appear only for genuinely signed quantities", and Pearson r on [-1, 1] is the example. Mapping it to the categorical palette or to Palette.SEQUENTIAL would put r = -0.9 and r = +0.9 at the same lightness and destroy the panel's only encoding. The grey-plus-highlight rule does not apply either: there is no minority to pick out, because the whole matrix is the claim.

### lines 2628-2630

```python
named = varying.shape[1] <= 25
```

Named on the ticks, so the axes are not labelled "predictor" twice over the names themselves; named on the axes when there are too many predictors to tick, so the reader still knows what the grid is.

### lines 2643-2645

```python
ax.set_xticklabels([str(c)[:20] for c in varying.columns],
```

20 characters, not 14. Every predictor in a spaCR screen design is named `fraction:grna[<id>]`, and 14 is the length of the prefix -- every tick came out reading `fraction:grna[`.

### lines 2650-2651

```python
rotate_ticks(ax, 45)
```

45 degrees right-aligned, not the 90 this drew before: the skill pins the angle, and a vertical label reads a word at a time.

### lines 2656-2659

```python
caption = (f"largest |r| = {flat.max():.2f} between "
```

28 characters, not 18. Every predictor in a spaCR screen design is named `fraction:grna[<id>]`, and 18 cut every one of them inside the bracket -- "between fraction:grna[2130 and fraction:grna[2276" names no pair at all.

### lines 2665-2666  _(unsure)_

```python
annotate(ax, caption, x=0.5, y=-0.34 if named else -0.16,
```

Below the rotated tick labels when there are any, below the axis label when there are not.

## _panel_coefficient_forest

### lines 2730-2733

```python
colours = np.where(
```

Grey unless the interval excludes zero, and then the sign picks the colour -- the same rule, the same two hues, as spacr.figures.panels.effect_rank, so a reader who has learned green/rust from the volcano sheet reads this panel for free.

### lines 2738-2742

```python
ax.hlines(positions, shown["lower"].to_numpy(),
```

hlines from lower to upper is exactly what errorbar drew from (coefficient - lower, upper - coefficient); the caps are gone because a cap is ink that carries no number, and the interval can now take one colour per term, which errorbar's single `ecolor` could not.

### lines 2748-2750

```python
colours = np.full(len(shown), ROLES["data"])
```

No interval means nothing has been called, so nothing is coloured. A penalised fit whose every term came out green or rust would be stating a claim the fit never made.

## _panel_p_value_histogram

### lines 2800-2801

```python
ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
```

No bar edge. The old white 0.4pt edge drew white gaps between the bars, which on a transparent or dark ground is a comb of light.

### lines 2809-2812

```python
ax.text(0.5, -0.22, textwrap.fill(diag["message"], 64),
```

The verdict is the only thing on this panel that can be a warning, so it is the only thing entitled to a colour. RUST when the shape says the fit is wrong, plain ink when it is not — a verdict drawn red on a healthy histogram teaches the reader to ignore red.

### lines 2822-2824

```python
"limitation": (diag["message"] if diag["verdict"] == "too-few"
```

Too few coefficients to read a shape is a real limitation of the panel, not a clean pass — it is reported as PARTIAL so nobody quotes the histogram of a five-term model.

## _panel_response_distribution

### lines 2837-2838

```python
ax.hist(finite, bins=bins, color=ROLES["fill"], edgecolor="none")
```

The whole distribution IS the claim here — there is no minority to highlight — so every bar is the house fill and nothing is coloured.

## _panel_count_fit

### lines 2983-2984  _(unsure)_

```python
grid = np.linspace(0, hi, 128)
```

Poisson's own +/- 2 SD envelope: outside it the mean-variance assumption is visibly failing, which is the whole point of the panel.

## _grouped_residuals

### lines 3017-3021

```python
keys = np.asarray([
```

Pandas 3 preserves missing values when ``astype(str)`` is applied to its native string dtype.  The resulting object array contains ``nan``; since ``nan != nan``, grouping on it creates a named but empty group and silently drops every well whose position is missing.  Normalise scalars at the Python boundary and give missing positions a stable display name.

## _positional_effect_panel

### lines 3064-3065

```python
kruskal_p = float("nan")
```

scipy refuses when every value is identical; that is a legitimate answer ("no difference"), not a failure of the panel.

### lines 3076-3078

```python
marks = [ROLES["data"]] * len(groups)
```

Who carries colour. Grey by default; the edge claim overrides the largest-median one where the two name the same group, so a fired edge statistic is never quietly recoloured into something else.

### lines 3094-3095  _(unsure)_

```python
box["medians"][index].set(color=colour, lw=WEIGHTS["data"])
```

The median bar is the number a reader actually takes off this panel, so it is the one artist drawn at data weight.

### lines 3103-3104

```python
ax.scatter(i + jitter, v, s=4.5, alpha=0.5, color=marks[i],
```

The skill's superplot exception: the small raw points are the only marks allowed alpha, so the summary reads on top.

### lines 3114-3116

```python
if max(len(name) for name in names) > 3:
```

45 degrees is the rule for labels that would not fit flat. `r16` and `c24` do fit, and rotating them costs a third of the panel's height for nothing; `plate1` does not.

### lines 3120-3125

```python
text = (f"{len(groups)} {label}s, n = {ctx.n:,} observations\n"
```

"observations", not "wells": `ctx.n` counts DESIGN ROWS, which is one per well only when the design has one row per well. The guide-level screen designs put several rows in a well -- the real tsg101 fit has 1,945 of them across 610 wells -- and every one is a mark here. Saying "wells" would misstate the unit of replication by a factor of three on the screen this panel was verified against.

## _panel_cell_count_vs_effect

### lines 3195-3197

```python
driving = extreme & (x <= low_cut)
```

The panel's whole sentence is "the tails are the small wells", and until now it was stated only in the text block while every point on the axes wore the same colour. These are the wells the sentence is about.

### lines 3210-3216

```python
annotate(ax, f"{_wells(int(good.sum()))}\n"
```

The log axis grows minor ticks; `_house_axes` inks AND sizes both sets, so there is nothing left to do here. It used to be done in this panel and only for the colour, which left the minor decade labels at matplotlib's 10 pt — the largest type in the figure. Right-hand corner, not left: the decile guide is the 10th percentile of the counts, so it is always near the left edge and its label runs up the top of the axes -- exactly where a left-hand note sits.

## _panel_volcano_reference

### lines 3257-3258

```python
+ textwrap.fill(f"in {os.path.dirname(ctx.volcano_path) or '.'}",
```

Wrapped: the real screen's results folder is 79 characters and ran off both sides of the panel.

## Module level

### lines 3346-3380

```python
VERDICT_LEVELS = ("unknown", "pass", "check", "fail")
```

THE VERDICT: what the panel CONCLUDED, on the panel (instruction 115).

Requested 2026-08-16 -- "i want the module to test everything relevant to regression like coliniarity, homogenicity, residual analasys and so on, all of these graphs should be saved".

THE PANELS AND THE NUMBERS WERE ALREADY THERE. Twenty-three panels, each returning its own statistics dict, and a text report listing every one of them. What was missing is the step a reader had to do by hand: deciding, per panel, whether the number is fine. A Q-Q plot with a `quantile_correlation` of 0.982 printed beside it tells a statistician something and tells everyone else nothing, and a suite of twenty of those is twenty judgements a user is quietly expected to make -- which is how a run with a rank-deficient design and a plate effect gets reported as "the QC looked fine".

SO EACH PANEL SCORES ITSELF, AND THE VERDICT IS DRAWN ON THE PANEL. Not in a summary table somewhere else: instruction 139 C made saved and visible one event for the same reason, and a verdict a reader has to go and find is a verdict that gets found after the figure has already been believed.

THREE RULES THE THRESHOLDS FOLLOW. They are the CONVENTIONAL ones, cited where there is a convention (VIF of 5 and 10, a scaled condition number of 30 and 100, Cook's D of 0.5 and 1, p >= 0.05 for a diagnostic test). A threshold invented here would be a number a reviewer cannot check. A DIAGNOSTIC TEST'S p IS BACKWARDS, and this is the commonest way to read one wrong: a LARGE p is the good outcome, because the null is "the assumption holds". Every rule below that reads a p is written in that direction, and the sentence beside it says so. "check" IS NOT "fail". A screen with real hits has a p-value spike at zero, a real design has some leverage, and a plate effect may be biology. The middle level says a human should look; only the level above it says the fit is not entitled to its inference.

## _score_observed_vs_predicted

### lines 3546-3548

```python
level = "pass" if r2 >= 0.1 else "check"
```

NOT A PASS/FAIL OF THE MODEL. A screen's guide-level fit explains a small fraction of well-to-well variation and is still the correct model; a near-zero R² is worth SEEING rather than worth failing.

## _score_precision_recall

### lines 3800-3802

```python
level = _band(lift, 1.5, 1.1, above_is_bad=False)
```

A BIGGER LIFT IS THE BETTER OUTCOME, the same way a bigger AUC is, so the thresholds are read downwards: below 1.1x the model is no better than the base rate.

## draw_verdict

### lines 3974-3978

```python
bbox=dict(boxstyle="round,pad=0.28",
```

THE BOX FOLLOWS THE THEME TOO (178 A). It was hard-coded white while its text is `ink` -- which on the dark theme IS white -- so the verdict was drawn, was there, and could not be read. A label that is invisible at one theme setting is the exact fault this instruction names.

## _save

### lines 4065-4069

```python
fig.clf()
```

Figures built via matplotlib.figure.Figure are not registered with pyplot, so there is nothing for plt.close() to close; dropping the last reference is the whole clean-up. clf() is belt-and-braces for the case where a panel parked a callback on the figure. It happens AFTER the publish, because a cleared figure has nothing left to render.

## format_qc_report

### lines 4093-4095

```python
f"standardised by  : {manifest.get('residual_scale')}",
```

The residual scale is on the header because it sets every |z| on the influence panels, and getting it wrong is silent: the wells are still named, they are just the wrong wells.

### lines 4117-4120

```python
if panel.verdict is not None and panel.verdict.level != "unknown":
```

THE VERDICT FIRST, because it is the line a reader acts on and the statistics under it are the evidence for it. A report that lists eight numbers and then concludes is a report whose conclusion is read last or not at all.

### lines 4161-4162

```python
counts = manifest.get("verdict_counts") or {}
```

THE WORST VERDICT, NAMED, AND EVERY PANEL THAT REACHED IT. A count of passes is the summary that hides the one panel the suite was run for.

## regression_qc_report

### lines 4268-4271

```python
renderer, renderer_reason = scene_renderer(renderer)
```

DECIDED ONCE, FOR THE WHOLE SUITE. Asking per panel is not the same question asked twenty times: an earlier attempt elsewhere in this project drew one run's first figure in matplotlib and its other six in pyqtgraph, because the first figure itself changed the answer.

### lines 4292-4294

```python
message = f"{type(exc).__name__}: {exc}"
```

A diagnostic that crashes must be loud (it is printed and it is on the report as FAILED) but must not destroy a fit that already succeeded and cost an hour.

### lines 4302-4306

```python
verdict = score_panel(name, stats)
```

SCORED AFTER IT DREW, STAMPED BEFORE IT IS WRITTEN. The verdict is read off the statistics the panel just returned, so it cannot disagree with the numbers printed beside it, and the axes is still open -- which is what lets the judgement go ON the panel rather than into a table the reader has to go and find.

### lines 4309-4310  _(unsure)_

```python
path = os.path.join(out_dir, name if not fmt else f"{name}.{fmt}")
```

No extension unless the caller forced one: `save_figure` appends the one that matches the format it actually writes.

### lines 4313-4320

```python
path, drew, why = _save(fig, path, fmt=fmt, renderer=renderer,
```

THE PATH THAT WAS WRITTEN, not the one that was asked for. `_save` goes through `spacr.plot.save_figure`, which rewrites the extension to the user's figure-format preference -- so with the preference on PNG this recorded `residuals_vs_fitted.pdf` for a file that is `residuals_vs_fitted.png` on disk. Every consumer of the manifest (the text report, `written`, the gallery link) then named a file that does not exist, which is "saved but I cannot see it" wearing a different hat.

### lines 4325-4327

```python
fell_back.append((name, why))
```

Only when the SUITE was going to be drawn by pyqtgraph. Naming twenty panels as having "fallen back" on a machine that never had Qt is twenty lines saying the one thing the header already said.

### lines 4344-4349

```python
if (str(regression_type or "").strip().lower() == "ols"
```

A compact, stable OLS-only sheet for a supplement.  The complete report above remains the audit trail; this second page contains just the eight assumption/influence/batch panels a reader needs together. It is emitted only when the caller requested the full set, so a deliberately narrow ``panels=[...]`` call still writes exactly what it asked for.

### lines 4386-4389

```python
"renderer": renderer,
```

WHICH LIBRARY DREW THEM, recorded rather than assumed. A user comparing a figure in a run folder against a tab on screen has to be able to find out which one they are holding, and the answer varies per machine.

### lines 4400-4402

```python
worst = worst_verdict(verdicts)
```

THE SUITE'S OWN VERDICT IS ITS WORST PANEL. Nineteen passes and one rank-deficient design is a run whose coefficients are one of infinitely many solutions; "95% passed" is the sentence that loses that.

## _write_qc_numbers._plain

### lines 4456-4459

```python
return value if np.isfinite(value) else None
```

NaN and inf are real answers here -- a test with no finite p-value is not the same as a test that was not run -- and `json.dump` writes them as bare NaN, which is not JSON and which `json.load` in another process may refuse.

## _write_qc_numbers

### lines 4479-4481

```python
"panels": {r.name: _plain(dict(r.stats or {})) for r in results},
```

FLAT, and per panel. Flat is what a reader wants -- one lookup for "the normality p-value" -- and the per-panel copy is what keeps it honest when two panels measure something with the same name.

### lines 4500-4503

```python
print(f"[regression_qc] could not write {QC_NUMBERS_FILE}: "
```

A run is not worth losing to a failure in its own bookkeeping but it is said out loud, because the advisor reading this file is the only thing that notices it is missing, and it notices by going quiet.

## _write_combined_page

### lines 4534-4537

```python
if show_verdicts:
```

THE VERDICT THE PANEL ALREADY REACHED, not a second opinion. Re-scoring the redraw would let the combined page and the individual file disagree about the same panel, which is exactly the failure this suite exists to catch elsewhere.

### lines 4541-4543

```python
ax.clear()
```

The panel drew a moment ago on its own figure, so this can only be an axes-specific problem; state it rather than leaving a blank tile.
