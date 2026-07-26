import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
import pandas as pd
from scipy.stats import fisher_exact
from matplotlib.gridspec import GridSpec

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
):
    """Render a volcano plot coloured by T. gondii subcellular localisation.

    Points are placed at ``(coefficient, -log10(p_value))`` and coloured by the
    ``metadata_column`` value on the merged gene metadata. Supports a broken
    y-axis for high ``-log10(p)`` outliers.

    :param data_path: DataFrame or CSV path with ``feature``, ``coefficient``,
        ``p_value`` columns.
    :param metadata_path: DataFrame or CSV path with ``gene_nr`` and the
        ``metadata_column`` values to merge on gene number.
    :param metadata_column: Metadata column that drives point colouring.
    :param point_size: Marker size passed to ``ax.scatter``.
    :param figsize: Side length in inches of the (square) figure.
    :param threshold: Absolute coefficient threshold used to select hits.
    :param save_path: Optional path to save the figure as a PDF.
    :param x_lim: X-axis limits ``[low, high]``. Defaults to ``[-0.5, 0.5]``.
    :param y_lims: None, ``[low, high]``, or ``[[low1, high1], [low2, high2]]``
        for a broken axis.
    :returns: List of ``variable`` names that are significant hits.
    """
    if x_lim is None:
        x_lim = [-0.5, 0.5]
    from matplotlib.gridspec import GridSpec

    colors = {
        'micronemes': 'black',
        'rhoptries 1': 'darkviolet',
        'rhoptries 2': 'darkviolet',
        'nucleus - chromatin': 'blue',
        'nucleus - non-chromatin': 'blue',
        'dense granules': 'teal',
        'ER 1': 'pink',
        'ER 2': 'pink',
        'unknown': 'black',
        'tubulin cytoskeleton': 'slategray',
        'IMC': 'slategray',
        'PM - peripheral 1': 'slategray',
        'PM - peripheral 2': 'slategray',
        'cytosol': 'turquoise',
        'mitochondrion - soluble': 'red',
        'mitochondrion - membranes': 'red',
        'apicoplast': 'slategray',
        'Golgi': 'green',
        'PM - integral': 'slategray',
        'apical 1': 'orange',
        'apical 2': 'orange',
        '19S proteasome': 'slategray',
        '20S proteasome': 'slategray',
        '60S ribosome': 'slategray',
        '40S ribosome': 'slategray',
    }

    fontsize = 18
    plt.rcParams.update({'font.size': fontsize})

    # --- Load data ---
    if isinstance(data_path, pd.DataFrame):
        data = data_path.copy()
    else:
        data = pd.read_csv(data_path)

    data['variable'] = data['feature'].str.extract(r'\[(.*?)\]')
    data['variable'] = data['variable'].fillna(data['feature'])
    data['gene_nr'] = data['variable'].str.split('_').str[0]
    data = data[data['variable'] != 'Intercept']

    # --- Load metadata ---
    if isinstance(metadata_path, pd.DataFrame):
        metadata = metadata_path
    else:
        metadata = pd.read_csv(metadata_path)

    metadata['gene_nr'] = metadata['gene_nr'].astype(str)
    data['gene_nr'] = data['gene_nr'].astype(str)

    merged_data = pd.merge(
        data,
        metadata[['gene_nr', metadata_column]],
        on='gene_nr',
        how='left',
    )
    merged_data[metadata_column] = merged_data[metadata_column].fillna('unknown')
    merged_data['neg_log_p'] = -np.log10(merged_data['p_value'])

    # --- Normalise y_lims into (is_broken, lower_lim, upper_lim) ---
    is_broken, lower_lim, upper_lim = _normalize_y_lims(y_lims, merged_data['neg_log_p'])

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

    def _pick_axis(y_val):
        """Return the upper broken-axis panel when ``y_val`` clears it, else the lower."""
        if is_broken and y_val > upper_lim[0]:
            return ax_upper
        return ax_lower

    hit_list = []

    # --- Scatter ---
    for _, row in merged_data.iterrows():
        y_val = row['neg_log_p']
        ax = _pick_axis(y_val)
        ax.scatter(
            row['coefficient'],
            y_val,
            color=colors.get(row[metadata_column], 'gray'),
            marker='o',
            s=point_size,
            edgecolor='black',
            alpha=0.6,
        )
        if (row['p_value'] <= 0.05) and (abs(row['coefficient']) >= abs(threshold)):
            hit_list.append(row['variable'])

    # --- Limits and spines ---
    ax_lower.set_ylim(lower_lim)
    ax_lower.set_xlim(x_lim)
    ax_lower.set_xlabel('Coefficient')
    ax_lower.set_ylabel('-log10(p-value)')
    ax_lower.spines['right'].set_visible(False)

    if is_broken:
        ax_upper.set_ylim(upper_lim)
        ax_upper.set_ylabel('-log10(p-value)')
        ax_upper.spines['right'].set_visible(False)
        ax_upper.spines['top'].set_visible(False)
        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)
    else:
        ax_lower.spines['top'].set_visible(False)

    # --- Threshold lines ---
    for ax in all_axes:
        ax.axvline(x=-abs(threshold), linestyle='--', color='black')
        ax.axvline(x=abs(threshold), linestyle='--', color='black')
    ax_lower.axhline(y=-np.log10(0.05), linestyle='--', color='black')

    # --- Annotate significant points ---
    texts_upper, texts_lower = [], []
    for _, row in merged_data.iterrows():
        if row['p_value'] > 0.05 or abs(row['coefficient']) < abs(threshold):
            continue
        y_val = row['neg_log_p']
        ax = _pick_axis(y_val)
        text = ax.text(
            row['coefficient'],
            y_val,
            row['variable'],
            fontsize=fontsize,
            ha='center',
            va='bottom',
        )
        if ax is ax_upper:
            texts_upper.append(text)
        else:
            texts_lower.append(text)

    if texts_lower:
        adjust_text(texts_lower, ax=ax_lower, arrowprops=dict(arrowstyle='-', color='black'))
    if is_broken and texts_upper:
        adjust_text(texts_upper, ax=ax_upper, arrowprops=dict(arrowstyle='-', color='black'))

    # --- Legend ---
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color=c, label=name, linewidth=0, markersize=8)
        for name, c in colors.items()
    ]
    ax_lower.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.25,
        labelspacing=2,
        handletextpad=0.25,
        markerscale=1.5,
        prop={'size': fontsize},
    )

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    return hit_list


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
    significant_df = significant_df.dropna(subset=['n_gene'])
    significant_df = significant_df[significant_df['n_gene'] != None]

    gene_list = significant_df['n_gene'].to_list()

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    split_columns = metadata['Gene ID'].str.split('_', expand=True)
    metadata['gene_nr'] = split_columns[1]

    # Create a subset of metadata with only the rows that contain genes in gene_list (hits)
    hits_metadata = metadata[metadata['gene_nr'].isin(gene_list)]

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
        plt.figure(figsize=(10, 6))
        
        # Create a scatter plot of Enrichment Score vs -log10(p-value)
        sns.scatterplot(data=results_df, x='Enrichment Score', y=-np.log10(results_df['P-value']), hue='GO Term', size='Enrichment Score', sizes=(50, 200))
        
        # Set plot labels and title
        plt.title(f'GO Term Enrichment Analysis for {go_term_column}')
        plt.xlabel('Enrichment Score')
        plt.ylabel('-log10(P-value)')
        
        # Move the legend to the right of the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        # Show the plot
        plt.tight_layout()  # Ensure everything fits in the figure area
        plt.show()

        # Optionally return or save the results for each column
        print(f'Results for {go_term_column}')

    # Combine results from all columns into a single DataFrame
    combined_df = pd.concat(combined_results)

    # Plot the combined results with text labels
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=combined_df, x='Enrichment Score', y=-np.log10(combined_df['P-value']),
                    style='GO Column', size='Enrichment Score', sizes=(50, 200))

    # Set plot labels and title for the combined graph
    plt.title('Combined GO Term Enrichment Analysis')
    plt.xlabel('Enrichment Score')
    plt.ylabel('-log10(P-value)')
    
    # Annotate the points with labels and connecting lines
    texts = []
    for i, row in combined_df.iterrows():
        texts.append(plt.text(row['Enrichment Score'], -np.log10(row['P-value']), row['GO Term'], fontsize=9))
    
    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
    
    # Show the combined plot
    plt.tight_layout()
    plt.show()

def plot_gene_phenotypes(data, gene_list, x_column='Gene ID', data_column='T.gondii GT1 CRISPR Phenotype - Mean Phenotype',error_column='T.gondii GT1 CRISPR Phenotype - Standard Error', save_path=None):
    """Plot ranked mean phenotype with SE shading and highlight selected genes.

    :param data: DataFrame with gene identifiers and phenotype/error columns.
    :param gene_list: Gene names (or ``TGGT1_<id>`` tags) to highlight.
    :param x_column: Column holding gene identifiers used for matching.
    :param data_column: Numeric column plotted on the y-axis.
    :param error_column: Numeric column used for the SE shading band.
    :param save_path: Optional PDF path to save the figure.
    :returns: None. Displays the Matplotlib figure.
    """
    # Ensure x_column is properly processed
    def extract_gene_id(gene):
        """Return the numeric portion of a ``TGGT1_<id>`` tag, or ``gene`` itself."""
        if isinstance(gene, str) and '_' in gene:
            return gene.split('_')[1]
        return str(gene)

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
    plt.figure(figsize=(10, 10))

    # Plot the mean phenotype with standard error shading
    plt.plot(x, y, label='Mean Phenotype', color=(0/255, 155/255, 155/255), linewidth=2)
    plt.fill_between(
        x, y - yerr, y + yerr, 
        color=(0/255, 155/255, 155/255), alpha=0.1, label='Standard Error'
    )

    # Prepare for adjustText
    texts = []  # Store text objects for adjustment

    # Highlight the genes in the gene_list
    for gene in gene_list:
        gene_id = extract_gene_id(gene)
        gene_data = data[data['x'] == gene_id]
        if not gene_data.empty:
            # Scatter the highlighted points in purple and add labels for adjustment
            plt.scatter(
                gene_data['rank'], 
                gene_data[data_column], 
                color=(155/255, 55/255, 155/255), 
                s=200,
                alpha=0.6,
                label=f'Highlighted Gene: {gene}',
                zorder=3  # Ensure the points are on top
            )
            # Add the text label next to the highlighted gene
            texts.append(
                plt.text(
                    gene_data['rank'].values[0], 
                    gene_data[data_column].values[0], 
                    gene, 
                    fontsize=18, 
                    ha='right'
                )
            )

    # Adjust text to avoid overlap with lines drawn from points to text
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

    # Label the plot
    plt.xlabel('Rank')
    plt.ylabel('Mean Phenotype')
    #plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.legend().remove()  # Remove the legend if not needed
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_gene_heatmaps(data, gene_list, columns, x_column='Gene ID', normalize=False, save_path=None):
    """Render a viridis heatmap for selected genes across selected metadata columns.

    :param data: DataFrame containing per-gene rows.
    :param gene_list: Genes to include as heatmap rows.
    :param columns: Column names to include as heatmap columns.
    :param x_column: Column holding gene identifiers for row matching.
    :param normalize: When True, min-max scale each gene's row to [0, 1].
    :param save_path: Optional PDF path to save the figure.
    :returns: None. Displays the Matplotlib figure.
    """
    # Ensure x_column is properly processed
    def extract_gene_id(gene):
        """Return the numeric portion of a ``TGGT1_<id>`` tag, or ``gene`` itself."""
        if isinstance(gene, str) and '_' in gene:
            return gene.split('_')[1]
        return str(gene)

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
    plt.figure(figsize=(width, height))
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Plot the heatmap with genes on the y-axis and columns on the x-axis
    sns.heatmap(
        filtered_data, 
        cmap=cmap, 
        cbar=True, 
        annot=False, 
        linewidths=0.5, 
        square=True
    )

    # Set the labels
    plt.xticks(rotation=90, ha='center')  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.xlabel('')
    plt.ylabel('')

    # Adjust layout to ensure the plot fits well
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
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
