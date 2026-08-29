"""Toxoplasma-specific visualisation helpers.

Every figure here is built inside :func:`_house`, which is
:mod:`spacr.figures.style` applied as a context manager. Read that module
before adding a panel; the rule it exists to enforce is that **everything is
grey except what the sentence is about**, and this file is where breaking it
was measured -- see :func:`custom_volcano_plot`.
"""

import contextlib
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
import pandas as pd
from scipy.stats import fisher_exact

from .figures.style import (ROLES, TYPE_SCALE, Palette, reference_line,
                            resolve_ink, rotate_ticks, text_legend,
                            theme_target)
from .figures.style import rc as style_rc
from . import tabular  # one reader: spacr.tabular is the funnel
# The module-scope routing contract: every figure this module keeps reaches
# the format and DPI preferences through `spacr.plot.save_figure`. NOTHING
# HERE CALLS IT DIRECTLY -- a save goes through `_write_the_figure` ->
# `figures.scene.write_figure`, which draws the scene the screen would show
# and falls back to this helper -- so spying on THIS name sees no call while
# the file is written exactly as asked.
from .plot import save_figure  # noqa: F401

#: The page width the published type scale was measured at: 180 mm, the double
#: column of Cell and Nature Microbiology, which is what
#: :data:`spacr.figures.style.TYPE_SCALE` pins its absolute points to.
REFERENCE_WIDTH_IN = 7.09


@contextlib.contextmanager
def _house(width, *, frame='L'):
    """The house style, with the type scaled to the canvas this module asks for.

    Apply styling through a context so a figure does not modify process-wide
    ``rcParams``. Scale every typography tier by the same factor relative to
    the 180 mm reference width; this preserves the house-style ratios on the
    larger canvases used for volcano plots and related figures.

    :param width: the figure's width in inches.
    :param frame: ``'L'`` (left and bottom spines) or ``'box'``.
    :yields: ``(ink, scale)`` -- the resolved text/axis colour, and the factor
        every explicit ``fontsize`` in the block has to be multiplied by.
    """
    target = theme_target()
    params = style_rc(target, frame=frame)
    scale = max(1.0, float(width) / REFERENCE_WIDTH_IN)
    for key in ('font.size', 'axes.labelsize', 'axes.titlesize',
                'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize'):
        params[key] = params[key] * scale
    with plt.rc_context(params):
        yield resolve_ink(target), scale


#: Leader lines from a label to its point. Grey and hairline, because a
#: leader is furniture: it says which point a name belongs to and nothing
#: else. They were solid black at the default width.
_LEADER = dict(arrowstyle='-', color=ROLES['reference'], linewidth=0.6)

#: Marker shapes for a genuinely categorical second variable, in the order
#: the published figures reach for them. Shape, not hue -- the colour budget
#: belongs to the claim.
_MARKERS = ('o', 's', '^', 'D', 'v', 'P')


def _scaled_sizes(values, low=50.0, high=200.0):
    """Marker areas spanning ``low`` to ``high`` across ``values``.

    What seaborn's ``sizes=(50, 200)`` did for the enrichment panels, kept so
    the points are the same sizes after the hue came out. A constant column
    maps to the small end rather than dividing by a zero range.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    span = float(np.nanmax(values) - np.nanmin(values))
    if not np.isfinite(span) or span <= 0:
        return np.full(values.shape, low)
    return low + (values - np.nanmin(values)) / span * (high - low)


def _sized_text_legend(ax, entries, scale, **kwargs):
    """:func:`spacr.figures.style.text_legend`, at this canvas's type size.

    ``text_legend`` writes at the pinned 6 pt annotation size, which on the
    20-inch volcano is invisible. The entries are re-sized after the fact
    rather than re-implemented, so the placement rule stays in one place.
    """
    first = len(ax.texts)
    text_legend(ax, entries, **kwargs)
    for text in ax.texts[first:]:
        text.set_fontsize(TYPE_SCALE['annotation'] * scale)


def _compartment_mask(values, wanted):
    """Boolean mask of the rows whose localisation is one of ``wanted``.

    :param values: the merged metadata column, one localisation per row.
    :param wanted: ``None``, one compartment name, or a sequence of them.
    :returns: ``(mask, label)``, or ``(None, '')`` when nothing was asked for.
    """
    if wanted is None:
        return None, ''
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    names = [str(name) for name in names if str(name)]
    if not names:
        return None, ''
    mask = values.astype(str).isin(names)
    return mask, ', '.join(names)


def custom_volcano_plot(
    data_path,
    metadata_path,
    metadata_column='tagm_location',
    point_size=50,
    figsize=20,
    threshold=0,
    save_path=None,
    x_lim=None,
    y_lims=None,
    draw=True,
    highlight_location=None,
):
    """Render a volcano plot and return the significant feature names.

    Plot each feature at ``(coefficient, -log10(p_value))``. Features that do
    not pass the call rule are grey; called positive effects are green and
    called negative or zero effects are rust. This direction-based palette
    makes significance and effect direction the primary visual encoding.
    Localization is optional: ``highlight_location`` overlays selected
    compartments in blue instead of assigning simultaneous colours to every
    category.

    Parameters
    ----------
    data_path : pandas.DataFrame or path-like
        Regression table containing ``feature``, ``coefficient``, and
        ``p_value`` columns. DataFrame input is copied.
    metadata_path : pandas.DataFrame or path-like
        Gene metadata containing one row per ``gene_nr`` and the selected
        metadata column. DataFrame input is copied.
    metadata_column : str, optional
        Localization or annotation column used by ``highlight_location``.
    point_size : float, optional
        Marker area passed to ``Axes.scatter``.
    figsize : float, optional
        Width and height of the square figure in inches. Typography scales
        with this value.
    threshold : float, optional
        Absolute coefficient threshold for calls. A row is returned when
        ``p_value <= 0.05`` and
        ``abs(coefficient) >= abs(threshold)``.
    save_path : path-like, optional
        Destination passed to :func:`spacr.figures.scene.write_figure`, which
        draws the scene the screen would show and falls back to
        :func:`spacr.plot.save_figure`. The written extension follows the
        configured figure format either way.
    x_lim : sequence of float, optional
        Two x-axis limits. The default is ``[-0.5, 0.5]``.
    y_lims : sequence, optional
        Use ``[low, high]`` for one axis or
        ``[[lower_low, lower_high], [upper_low, upper_high]]`` for a broken
        y-axis. By default, fit one axis to the finite values.
    draw : bool, optional
        Build, optionally save, and show the figure. If false, return the hit
        list before constructing a figure.
    highlight_location : str or sequence of str, optional
        Values from ``metadata_column`` to overlay in the highlight colour and
        name in the in-panel legend.

    Returns
    -------
    list of str
        Feature-derived ``variable`` names that satisfy the call rule, in
        table order.

    Raises
    ------
    pandas.errors.MergeError
        If the metadata contains duplicate ``gene_nr`` values and therefore
        cannot be joined many-to-one.
    ValueError
        If ``y_lims`` does not match a supported form.
    """
    if x_lim is None:
        x_lim = [-0.5, 0.5]
    from matplotlib.gridspec import GridSpec

    # --- Load data ---
    if isinstance(data_path, pd.DataFrame):
        data = data_path.copy()
    else:
        data = tabular.read_table(data_path, report=None)

    data['variable'] = data['feature'].str.extract(r'\[(.*?)\]')
    data['variable'] = data['variable'].fillna(data['feature'])
    data['gene_nr'] = data['variable'].str.split('_').str[0]
    data = data[data['variable'] != 'Intercept']

    # --- Load metadata ---
    if isinstance(metadata_path, pd.DataFrame):
        # .copy() for the same reason `data` above takes one: the next line
        # rewrites 'gene_nr' to str in place, and without the copy that edit
        # landed in the CALLER's frame. A caller that plots two volcanoes from
        # one metadata table got its integer gene numbers silently retyped by
        # the first call.
        metadata = metadata_path.copy()
    else:
        metadata = tabular.read_table(metadata_path, report=None)

    metadata['gene_nr'] = metadata['gene_nr'].astype(str)
    data['gene_nr'] = data['gene_nr'].astype(str)

    # many_to_one: `data` holds one row per regression *feature*, and several
    # features share a gene -- a gRNA-level fit contributes one row per guide,
    # so gene_nr repeats on the left by design. `metadata` is a lookup table:
    # one localisation per gene, which is what the shipped
    # resources/data/lopit.csv is (3832 rows, 3832 distinct gene_nr). A
    # duplicated gene_nr on the right is therefore not a legitimate shape here,
    # it is a fan-out: every affected gene gets plotted twice and appended to
    # the returned hit list twice, which then propagates into
    # plot_gene_phenotypes and plot_gene_heatmaps as duplicate genes. Declaring
    # the relationship turns that into a stop rather than a wrong figure.
    try:
        merged_data = pd.merge(
            data,
            metadata[['gene_nr', metadata_column]],
            on='gene_nr',
            how='left',
            validate='many_to_one',
        )
    except pd.errors.MergeError as exc:
        duplicated = metadata.loc[
            metadata['gene_nr'].duplicated(keep=False), 'gene_nr']
        if duplicated.empty:
            # MergeError also covers things this message would misdescribe --
            # a colliding suffix, for one. Only claim the cardinality story
            # when the duplicates that would justify it are actually there.
            raise
        examples = duplicated.unique()[:5].tolist()
        raise pd.errors.MergeError(
            f"The gene metadata lists {duplicated.nunique()} gene_nr value(s) "
            f"more than once (e.g. {examples}), so it cannot say which "
            f"{metadata_column!r} belongs to a gene. Joining it anyway would "
            f"plot those genes once per duplicate row and return each of them "
            f"more than once in the hit list. De-duplicate the metadata on "
            f"gene_nr before plotting. (pandas: {exc})"
        ) from exc
    merged_data[metadata_column] = merged_data[metadata_column].fillna('unknown')
    merged_data['neg_log_p'] = -np.log10(merged_data['p_value'])

    # ONE RULE, ONE PLACE. The hit list and the colouring below read the same
    # mask, so a gene the volcano marks and a gene the phenotype plot reports
    # cannot disagree -- they used to be a vectorised expression and a
    # row-by-row `if` written separately.
    called = ((merged_data['p_value'] <= 0.05)
              & (merged_data['coefficient'].abs() >= abs(threshold)))
    hit_list = list(merged_data.loc[called, 'variable'])

    if not draw:
        return hit_list

    # --- Normalise y_lims into (is_broken, lower_lim, upper_lim) ---
    is_broken, lower_lim, upper_lim = _normalize_y_lims(
        y_lims, merged_data['neg_log_p'])

    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams colour an
    # artist when it is CREATED, so a context opened after plt.subplots would
    # leave the spines, ticks and text at the caller's global style.
    with _house(figsize) as (ink, scale):
        # --- Axes ---
        if is_broken:
            fig = plt.figure(figsize=(figsize, figsize))
            gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
            ax_upper = fig.add_subplot(gs[0])
            ax_lower = fig.add_subplot(gs[1], sharex=ax_upper)
            ax_upper.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            all_axes = [ax_lower, ax_upper]
        else:
            fig, ax_lower = plt.subplots(figsize=(figsize, figsize))
            ax_upper = None
            all_axes = [ax_lower]

        coefficient = merged_data['coefficient'].to_numpy(dtype=float)
        neg_log_p = merged_data['neg_log_p'].to_numpy(dtype=float)
        called_mask = called.to_numpy(dtype=bool)
        # Which panel each point belongs to, decided once for the whole column
        # rather than by a function call per row.
        on_upper = (neg_log_p > upper_lim[0]) if is_broken \
            else np.zeros(len(neg_log_p), dtype=bool)

        up = called_mask & (coefficient > 0)
        down = called_mask & (coefficient <= 0)
        wanted, wanted_label = _compartment_mask(
            merged_data[metadata_column], highlight_location)

        # ONE SCATTER CALL PER ARGUMENT, not one per point. The old loop ran
        # `ax.scatter` once for each of the ~3,800 rows and then a second full
        # pass to build the hit list; both are gone. zorder puts the claim on
        # top of the grey rather than leaving it to row order.
        layers = [(~called_mask, ROLES['data'], 1, None),
                  (up, ROLES['up'], 2, f"called, positive ({int(up.sum())})"),
                  (down, ROLES['down'], 2, f"called, negative ({int(down.sum())})")]
        if wanted is not None:
            layers.append((wanted.to_numpy(dtype=bool), ROLES['highlight'], 3,
                           wanted_label))

        entries = []
        for selected, colour, zorder, label in layers:
            for axis, side in ((ax_lower, ~on_upper), (ax_upper, on_upper)):
                if axis is None:
                    continue
                take = selected & side
                if not take.any():
                    continue
                # Opaque, no edge: the published figures handle overplotting
                # with size and with grey, not with alpha.
                axis.scatter(coefficient[take], neg_log_p[take], color=colour,
                             marker='o', s=point_size, linewidths=0,
                             zorder=zorder)
            if label and selected.any():
                entries.append((label, colour))

        # --- Limits and spines ---
        ax_lower.set_ylim(lower_lim)
        ax_lower.set_xlim(x_lim)
        ax_lower.set_xlabel('Coefficient')
        ax_lower.set_ylabel('-log10(p-value)')

        if is_broken:
            ax_upper.set_ylim(upper_lim)
            ax_upper.set_ylabel('-log10(p-value)')
            # The break itself: the two panels face each other across a gap,
            # so the spines that would draw a line through it come off.
            ax_upper.spines['bottom'].set_visible(False)

        # --- Threshold lines ---
        # Grey, thin, dashed, behind the data. A reference is not a result.
        for axis in all_axes:
            if threshold:
                reference_line(axis, x=-abs(threshold))
                reference_line(axis, x=abs(threshold))
            else:
                # threshold=0 means "no coefficient cut": the two lines would
                # land on top of each other, so draw the zero once.
                reference_line(axis, x=0.0)
        reference_line(ax_lower, y=-np.log10(0.05))

        # --- Annotate significant points ---
        texts_upper, texts_lower = [], []
        label_size = TYPE_SCALE['annotation'] * scale
        for index in np.flatnonzero(called_mask):
            axis = ax_upper if on_upper[index] else ax_lower
            text = axis.text(
                coefficient[index],
                neg_log_p[index],
                merged_data['variable'].iat[index],
                fontsize=label_size,
                color=ink,
                ha='center',
                va='bottom',
            )
            if axis is ax_upper:
                texts_upper.append(text)
            else:
                texts_lower.append(text)

        leader = dict(arrowstyle='-', color=ROLES['reference'],
                      linewidth=0.6)
        if texts_lower:
            adjust_text(texts_lower, ax=ax_lower, arrowprops=leader)
        if is_broken and texts_upper:
            adjust_text(texts_upper, ax=ax_upper, arrowprops=leader)

        # --- Legend ---
        # THREE LINES OF TEXT, INSIDE THE PANEL. It was 27 framed swatches
        # anchored outside the axes, which is why `_fit_outside_legend` had to
        # exist and why the data was squeezed into a strip beside it. A legend
        # is an index; with grey as the default ink there are only the called
        # directions to index.
        if entries:
            _sized_text_legend(ax_lower, entries, scale)

        if save_path:
            # Saved INSIDE the context: savefig.transparent and
            # savefig.facecolor are read at write time, so a save outside it
            # would put the default white ground back under the figure.
            save_path = _write_the_figure(fig, save_path,
                                          bbox_inches='tight')
        plt.show()

        return hit_list




def _write_the_figure(fig, save_path, title=None, **kwargs):
    """Write ``fig`` through the SCENE renderer, not straight to matplotlib.

    Every generated figure on this path is asked for in pyqtgraph, so the
    file a run writes is the same scene the screen would show. The
    translation is a whitelist: an artist nobody thought about leaves the
    figure incomplete and the matplotlib page is written instead, with the
    reason recorded rather than a piece of the picture quietly missing.

    ``save_figure`` is still what runs in that fallback, so the format
    preference and the DPI reach the file either way.

    :param fig: the finished matplotlib figure.
    :param save_path: destination; the extension follows the preference.
    :param title: the name the gallery tile carries.
    :returns: the path actually written.
    """
    from .figures.scene import write_figure

    written, _drew, _why = write_figure(fig, os.fspath(save_path),
                                        title=title, **kwargs)
    return written or save_path


def _fit_outside_legend(fig, legend, pad=0.02, min_axes_width=0.45):
    """Make room inside ``fig`` for a legend anchored outside the axes.

    A legend at ``bbox_to_anchor=(1.02, 1)`` sits beyond the axes and therefore
    beyond the figure. ``bbox_inches='tight'`` grows the saved file to cover it,
    so the PDF on disk looks right -- but nothing rescues the figure shown in
    the application, which is drawn at the figure's own extent. The legend is
    simply clipped, reported as "the volcano plot is always cut off on the
    right side".

    Measuring the legend and shrinking the axes to fit it means the figure is
    correct as drawn, on screen and on disk alike, instead of being correct
    only after a save-time rescue.

    NO FIGURE IN THIS MODULE ANCHORS A LEGEND OUTSIDE ITS AXES ANY MORE: the
    house style puts a two-line text legend inside the panel, which is what
    made the 27-swatch column that this helper was written to survive
    unnecessary in the first place. It is kept for a panel that has to.

    :param fig: Figure holding the legend.
    :param legend: The legend to make room for.
    :param pad: Extra figure-width fraction left beside the legend.
    :param min_axes_width: Never shrink the axes below this fraction, so a
        runaway legend cannot squeeze the data down to nothing.
    """
    if legend is None:
        return
    try:
        fig.canvas.draw()
        extent = legend.get_window_extent()
        fig_width = fig.get_figwidth() * fig.dpi
        if fig_width <= 0:
            return
        right = 1.0 - (extent.width / fig_width) - pad
        # A legend wider than the figure would invert the axes; clamp instead.
        fig.subplots_adjust(right=max(min(right, 0.98), min_axes_width))
    except Exception:  # layout is never worth an exception
        pass


def _normalize_y_lims(y_lims, neg_log_p):
    """Coerce y_lims into ``(is_broken, lower_lim, upper_lim)`` for volcano plotting.

    - ``None``: auto-fit a single panel from the data.
    - ``[low, high]``: single panel with explicit limits.
    - ``[[low1, high1], [low2, high2]]``: broken axis (lower, upper).

    :raises ValueError: When ``y_lims`` does not match one of the supported forms.
    """
    if y_lims is None:
        finite = neg_log_p[np.isfinite(neg_log_p)]
        if len(finite) == 0:
            return False, [0.0, 1.0], None
        ymax = float(finite.max()) * 1.05
        return False, [0.0, max(ymax, 1.0)], None

    if not (isinstance(y_lims, (list, tuple)) and len(y_lims) == 2):
        raise ValueError(
            "y_lims must be None, [low, high], or [[low1, high1], [low2, high2]]; "
            f"got {y_lims!r}"
        )

    a, b = y_lims
    if all(isinstance(v, (int, float)) or v is None for v in (a, b)):
        return False, [a, b], None
    if all(isinstance(v, (list, tuple)) and len(v) == 2 for v in (a, b)):
        return True, list(a), list(b)

    raise ValueError(
        "y_lims must be None, [low, high], or [[low1, high1], [low2, high2]]; "
        f"got {y_lims!r}"
    )


def go_term_enrichment_by_column(significant_df, metadata_path, go_term_columns=None):
    """Compute and plot GO-term enrichment for each requested metadata column.

    For every ``go_term_column`` counts occurrences among hit vs background
    genes, runs Fisher's exact test per term, and produces scatter plots of
    enrichment vs ``-log10(p)`` both per column and combined.

    :param significant_df: DataFrame of screen hits with a ``n_gene`` column.
    :param metadata_path: CSV path holding ``Gene ID`` plus GO-term columns.
    :param go_term_columns: Columns to test. Defaults to the four standard
        Computed/Curated GO categories.
    :returns: None. Results are displayed as Matplotlib figures.
    """
    
    #significant_df['variable'].fillna(significant_df['feature'], inplace=True)
    #split_columns = significant_df['variable'].str.split('_', expand=True)
    #significant_df['gene_nr'] = split_columns[0]
    #gene_list = significant_df['gene_nr'].to_list()

    if go_term_columns is None:
        go_term_columns = ['Computed GO Processes', 'Curated GO Components', 'Curated GO Functions', 'Curated GO Processes']
    significant_df = significant_df.loc[
        significant_df['n_gene'].notna()].copy()

    gene_list = significant_df['n_gene'].to_list()

    # Load metadata
    metadata = tabular.read_table(metadata_path, report=None)
    split_columns = metadata['Gene ID'].str.split('_', expand=True)
    metadata['gene_nr'] = split_columns[1]

    # Create a subset of metadata with only the rows that contain genes in gene_list (hits)
    hits_metadata = metadata.loc[
        metadata['gene_nr'].isin(gene_list)].copy()

    # Create a list to hold results from all columns
    combined_results = []

    for go_term_column in go_term_columns:
        # Initialize lists to store results
        go_terms = []
        enrichment_scores = []
        p_values = []

        # Split the GO terms in the entire metadata and hits
        metadata[go_term_column] = metadata[go_term_column].fillna('')
        hits_metadata[go_term_column] = hits_metadata[go_term_column].fillna('')

        all_go_terms = metadata[go_term_column].str.split(';').explode()
        hit_go_terms = hits_metadata[go_term_column].str.split(';').explode()

        # Count occurrences of each GO term in hits and total metadata
        all_go_term_counts = all_go_terms.value_counts()
        hit_go_term_counts = hit_go_terms.value_counts()

        # Perform enrichment analysis for each GO term
        for go_term in all_go_term_counts.index:
            total_with_go_term = all_go_term_counts.get(go_term, 0)
            hits_with_go_term = hit_go_term_counts.get(go_term, 0)

            # Calculate the total number of genes and hits
            total_genes = len(metadata)
            total_hits = len(hits_metadata)

            # Perform Fisher's exact test
            contingency_table = [[hits_with_go_term, total_hits - hits_with_go_term],
                                 [total_with_go_term - hits_with_go_term, total_genes - total_hits - (total_with_go_term - hits_with_go_term)]]
            
            _, p_value = fisher_exact(contingency_table)
            
            # Calculate enrichment score (hits with GO term / total hits with GO term)
            if total_with_go_term > 0 and total_hits > 0:
                enrichment_score = (hits_with_go_term / total_hits) / (total_with_go_term / total_genes)
            else:
                enrichment_score = 0.0

            # Store the results only if enrichment score is non-zero
            if enrichment_score > 0.0:
                go_terms.append(go_term)
                enrichment_scores.append(enrichment_score)
                p_values.append(p_value)

        # Create a results DataFrame for this GO term column
        results_df = pd.DataFrame({
            'GO Term': go_terms,
            'Enrichment Score': enrichment_scores,
            'P-value': p_values,
            'GO Column': go_term_column  # Track the GO term column for final combined plot
        })

        # Sort by enrichment score
        results_df = results_df.sort_values(by='Enrichment Score', ascending=False)

        # Append this DataFrame to the combined list
        combined_results.append(results_df)

        # Plot the enrichment results for each individual column
        # GREY, WITH THE CALLED TERMS COLOURED. `hue='GO Term'` gave every
        # term of the ontology its own hue and its own legend row -- the
        # 27-colour failure again, and worse here, because a GO column carries
        # hundreds of terms and the legend was anchored outside the axes. The
        # sentence is "these terms are enriched among the hits and the
        # enrichment is significant", so significance is what carries colour,
        # and the terms that carry it are named on the points instead.
        # ONE scatter call, in the frame's own row order, so the points are
        # the same points in the same order as before.
        with _house(10) as (ink, scale):
            fig = plt.figure(figsize=(10, 6))
            ax = fig.gca()
            enrichment = results_df['Enrichment Score'].to_numpy(dtype=float)
            significance = -np.log10(results_df['P-value'].to_numpy(dtype=float))
            called = results_df['P-value'].to_numpy(dtype=float) <= 0.05
            # Size still reads the effect, as `sizes=(50, 200)` did; only the
            # hue moved.
            ax.scatter(enrichment, significance, s=_scaled_sizes(enrichment),
                       color=[ROLES['highlight'] if hit else ROLES['data']
                              for hit in called],
                       linewidths=0, zorder=2)
            reference_line(ax, y=-np.log10(0.05))
            # Enrichment of 1 is "as common among the hits as in the
            # background", which is the null this panel is read against.
            reference_line(ax, x=1.0)

            # Set plot labels and title
            ax.set_title(f'GO Term Enrichment Analysis for {go_term_column}')
            ax.set_xlabel('Enrichment Score')
            ax.set_ylabel('-log10(P-value)')

            # The terms that cleared p <= 0.05 are named on the panel, which is
            # what the every-term legend was there to do and could not.
            texts = [ax.text(enrichment[i], significance[i],
                             results_df['GO Term'].iat[i],
                             fontsize=TYPE_SCALE['annotation'] * scale,
                             color=ink)
                     for i in np.flatnonzero(called)]
            if texts:
                adjust_text(texts, ax=ax, arrowprops=_LEADER)
            _sized_text_legend(
                ax,
                [(f'p <= 0.05 ({int(called.sum())})', ROLES['highlight']),
                 (f'not called ({int((~called).sum())})', ROLES['data'])],
                scale)
            fig.tight_layout()
            plt.show()

        # Optionally return or save the results for each column
        print(f'Results for {go_term_column}')

    # Combine results from all columns into a single DataFrame
    combined_df = pd.concat(combined_results)

    # Plot the combined results with text labels
    with _house(12) as (ink, scale):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        enrichment = combined_df['Enrichment Score'].to_numpy(dtype=float)
        significance = -np.log10(combined_df['P-value'].to_numpy(dtype=float))
        called = combined_df['P-value'].to_numpy(dtype=float) <= 0.05
        sizes = _scaled_sizes(enrichment)
        colours = [ROLES['highlight'] if hit else ROLES['data']
                   for hit in called]
        # WHICH ONTOLOGY A TERM CAME FROM IS A REAL SECOND VARIABLE, so it
        # keeps its encoding -- as marker shape, which is what `style='GO
        # Column'` already used and what the style spends no colour on. One
        # collection per shape, in frame order within each.
        for index, column in enumerate(dict.fromkeys(combined_df['GO Column'])):
            rows = np.flatnonzero(
                (combined_df['GO Column'] == column).to_numpy(dtype=bool))
            ax.scatter(enrichment[rows], significance[rows], s=sizes[rows],
                       color=[colours[i] for i in rows],
                       marker=_MARKERS[index % len(_MARKERS)],
                       linewidths=0, zorder=2)
        reference_line(ax, y=-np.log10(0.05))
        reference_line(ax, x=1.0)

        # Set plot labels and title for the combined graph
        ax.set_title('Combined GO Term Enrichment Analysis')
        ax.set_xlabel('Enrichment Score')
        ax.set_ylabel('-log10(P-value)')

        # Annotate the points with labels and connecting lines
        texts = [ax.text(enrichment[i], significance[i],
                         combined_df['GO Term'].iat[i],
                         fontsize=TYPE_SCALE['annotation'] * scale, color=ink)
                 for i in range(len(combined_df))]

        # Adjust text to avoid overlap
        adjust_text(texts, ax=ax, arrowprops=_LEADER)
        fig.tight_layout()
        plt.show()


def plot_gene_phenotypes(data, gene_list, x_column='Gene ID', data_column='T.gondii GT1 CRISPR Phenotype - Mean Phenotype',error_column='T.gondii GT1 CRISPR Phenotype - Standard Error', save_path=None):
    """Plot ranked mean phenotype with SE shading and highlight selected genes.

    Parameters
    ----------
    data : pandas.DataFrame
        Gene identifiers and phenotype/error columns. The frame is copied
        before numeric conversion.
    gene_list : iterable of str
        Gene names or ``TGGT1_<id>`` identifiers to highlight.
    x_column : str, default='Gene ID'
        Column used to match gene identifiers.
    data_column : str
        Mean phenotype column plotted on the y-axis.
    error_column : str
        Standard-error column used for the uncertainty band.
    save_path : path-like, optional
        Figure destination. The configured figure format controls the final
        extension.

    Notes
    -----
    The complete ranked phenotype curve is drawn in grey and selected genes
    use the spaCR accent colour. The figure is displayed after optional save.
    """
    # Ensure x_column is properly processed
    def extract_gene_id(gene):
        """Return the numeric portion of a ``TGGT1_<id>`` tag, or ``gene`` itself."""
        if isinstance(gene, str) and '_' in gene:
            return gene.split('_')[1]
        return str(gene)

    # The caller's table is not ours to retype. `data.loc[:, col] = ...` below
    # writes through to whatever frame was passed, and `ml.perform_regression`
    # passes the GT1 metadata table it read once and uses again.
    data = data.copy()

    data.loc[:, data_column] = pd.to_numeric(data[data_column], errors='coerce')
    data = data.dropna(subset=[data_column])
    data.loc[:, error_column] = pd.to_numeric(data[error_column], errors='coerce')
    data = data.dropna(subset=[error_column])

    data['x'] = data[x_column].apply(extract_gene_id)

    # Sort by the data_column and assign ranks
    data = data.sort_values(by=data_column).reset_index(drop=True)
    data['rank'] = range(1, len(data) + 1)

    # Prepare the x, y, and error values for plotting
    x = data['rank']
    y = data[data_column]
    yerr = data[error_column]

    # Create the plot
    with _house(10) as (ink, scale):
        fig = plt.figure(figsize=(10, 10))

        # Plot the mean phenotype with standard error shading. The band takes
        # the line's own hue at 0.25, which is the only alpha the published
        # figures use on a curve.
        plt.plot(x, y, label='Mean Phenotype', color=Palette.GREY_DARK,
                 linewidth=1.2)
        plt.fill_between(
            x, y - yerr, y + yerr,
            color=Palette.GREY_DARK, alpha=0.25, label='Standard Error',
            linewidth=0,
        )

        # Prepare for adjustText
        texts = []  # Store text objects for adjustment

        # Highlight the genes in the gene_list
        for gene in gene_list:
            gene_id = extract_gene_id(gene)
            gene_data = data[data['x'] == gene_id]
            if not gene_data.empty:
                plt.scatter(
                    gene_data['rank'],
                    gene_data[data_column],
                    color=ROLES['highlight'],
                    s=200,
                    linewidths=0,
                    label=f'Highlighted Gene: {gene}',
                    zorder=3  # Ensure the points are on top
                )
                # Add the text label next to the highlighted gene
                texts.append(
                    plt.text(
                        gene_data['rank'].values[0],
                        gene_data[data_column].values[0],
                        gene,
                        fontsize=TYPE_SCALE['annotation'] * scale,
                        color=ink,
                        ha='right',
                    )
                )

        # Adjust text to avoid overlap with lines drawn from points to text
        adjust_text(texts, arrowprops=_LEADER)

        # Label the plot
        plt.xlabel('Rank')
        plt.ylabel('Mean Phenotype')
        plt.legend().remove()  # Remove the legend if not needed
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            save_path = _write_the_figure(fig, save_path,
                                          bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

def plot_gene_heatmaps(data, gene_list, columns, x_column='Gene ID', normalize=False, save_path=None):
    """Render a heatmap for selected genes across selected metadata columns.

    THE RAMP IS SINGLE-HUE. It was viridis, a rainbow that reads as five
    categories where the quantity is one ordered score; the house style's
    ``Blues`` runs light to dark, so "more" is one direction rather than a
    tour of the spectrum. A diverging map would be right only if the values
    were signed, and after ``normalize`` they run 0 to 1.

    :param data: DataFrame containing per-gene rows. Copied before use -- the
        row-key column this adds used to appear on the caller's frame.
    :param gene_list: Genes to include as heatmap rows.
    :param columns: Column names to include as heatmap columns.
    :param x_column: Column holding gene identifiers for row matching.
    :param normalize: When True, min-max scale each gene's row to [0, 1].
    :param save_path: Optional destination for the figure. Saving goes through
        :func:`spacr.figures.scene.write_figure` and, on its fallback,
        :func:`spacr.plot.save_figure`, so the format and the file extension
        follow the figure preference rather than always being PDF.
    :returns: None. Displays the Matplotlib figure.
    """
    # Ensure x_column is properly processed
    def extract_gene_id(gene):
        """Return the numeric portion of a ``TGGT1_<id>`` tag, or ``gene`` itself."""
        if isinstance(gene, str) and '_' in gene:
            return gene.split('_')[1]
        return str(gene)

    # `data['x'] = ...` is a new column on the caller's table otherwise, and
    # `ml.perform_regression` reuses the ME49 frame it passes here.
    data = data.copy()
    data['x'] = data[x_column].apply(extract_gene_id)

    # Filter the data to only include the specified genes
    filtered_data = data[data['x'].isin(gene_list)].set_index('x')[columns]

    # Normalize each gene's values between 0 and 1 if normalize=True
    if normalize:
        filtered_data = filtered_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

    # Define the figure size dynamically based on the number of genes and columns
    width = len(columns) * 4
    height = len(gene_list) * 1

    # Create the heatmap
    with _house(width, frame='box') as (ink, scale):
        fig = plt.figure(figsize=(width, height))
        cmap = sns.color_palette(Palette.SEQUENTIAL, as_cmap=True)

        # Plot the heatmap with genes on the y-axis and columns on the x-axis
        # linewidths=0: the white rules between cells were a grid, and the
        # rule the style states is no gridlines ever.
        ax = sns.heatmap(
            filtered_data,
            cmap=cmap,
            cbar=True,
            annot=False,
            linewidths=0,
            square=True
        )

        # Long column names rotate 45 and anchor right, as every categorical
        # axis in the style does.
        rotate_ticks(ax, 45)
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal
        plt.xlabel('')
        plt.ylabel('')
        for bar in fig.axes[len(fig.axes) - 1:]:
            bar.tick_params(colors=ink, labelsize=TYPE_SCALE['tick'] * scale)
            for spine in bar.spines.values():
                spine.set_visible(False)

        # Adjust layout to ensure the plot fits well
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            save_path = _write_the_figure(fig, save_path,
                                          bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

def generate_score_heatmap(settings):
    """Build combined classification-score and control-fraction heatmaps for a plate.

    Thin wrapper around :func:`spacr.submodules.generate_score_heatmap`, kept
    only so the historic ``spacr.toxo`` import path keeps working. This module
    used to carry a second copy of that function, identical to it line for line
    apart from the key names and the colormap, and left behind by the
    ``column_name`` -> ``columnID`` rename: it filtered, grouped and merged on
    ``column_name``, a key no spaCR CSV carries any more, and one helper even
    created ``columnID`` and then immediately indexed ``column_name``. Every
    call raised ``KeyError('column_name')`` on a canonical input. Rather than
    repair a second copy, delegate to the one that was migrated.

    The only behavioural difference between the two copies was the colormap:
    this one hard-coded ``'viridis'`` and ignored ``settings['cmap']``, while
    the ``submodules`` version requires it. The default below preserves what
    ``toxo`` callers used to get while now honouring ``cmap`` when they pass it.

    Imported inside the function on purpose: ``spacr.submodules`` pulls in
    cellpose, torch and shap at import time, and ``spacr.ml`` imports this
    module.

    :param settings: Config dict with keys ``folders``, ``csv_name``,
        ``data_column``, ``csv``, ``cv_csv``, ``data_column_cv``, ``plateID``,
        ``columnID``, ``control_sgrnas``, ``fraction_grna``, ``dst`` and,
        optionally, ``cmap``.
    :returns: merged DataFrame joining reads, classifier scores and CV scores per well.
    """
    from .submodules import generate_score_heatmap as _generate_score_heatmap

    settings = dict(settings)
    settings.setdefault('cmap', 'viridis')
    return _generate_score_heatmap(settings)
