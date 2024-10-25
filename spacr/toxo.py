import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
import pandas as pd
from scipy.stats import fisher_exact
from IPython.display import display
from matplotlib.legend import Legend

def custom_volcano_plot(data_path, metadata_path, metadata_column='tagm_location', point_size=50, figsize=20, threshold=0, split_axis_lims = [10, None, None, 10], save_path=None, x_lim=[-0.5, 0.5]):
    """
    Create a volcano plot with the ability to control the shape of points based on a categorical column,
    color points based on a condition, annotate specific points based on p-value and coefficient thresholds,
    and control the size of points.
    """
    volcano_path = save_path
    padd = 30
    fontsize = 18
    # Load the data
    if isinstance(data_path, pd.DataFrame):
        data = data_path
    else:
        data = pd.read_csv(data_path)
        
    data['variable'] = data['feature'].str.extract(r'\[(.*?)\]')
    data['variable'].fillna(data['feature'], inplace=True)
    split_columns = data['variable'].str.split('_', expand=True)
    data['gene_nr'] = split_columns[0]
    data = data[data['variable'] != 'Intercept']

    # Load metadata
    if isinstance(metadata_path, pd.DataFrame):
        metadata = metadata_path
    else:
        metadata = pd.read_csv(metadata_path)

    metadata['gene_nr'] = metadata['gene_nr'].astype(str)
    data['gene_nr'] = data['gene_nr'].astype(str)

    # Merge data and metadata on 'gene_nr'
    merged_data = pd.merge(data, metadata[['gene_nr', 'tagm_location']], on='gene_nr', how='left')

    merged_data.loc[merged_data['gene_nr'].str.startswith('4'), metadata_column] = 'GT1_gene'
    merged_data.loc[merged_data['gene_nr'] == 'Intercept', metadata_column] = 'Intercept'
    merged_data.loc[merged_data['condition'] == 'control', metadata_column] = 'control'
    merged_data[metadata_column].fillna('unknown', inplace=True)

    # Categorize condition for coloring
    merged_data['condition'] = pd.Categorical(
        merged_data['condition'],
        categories=['other','pc', 'nc', 'control'],
        ordered=True)

    # Create subplots with a broken y-axis
    figsize_2 = figsize / 2
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(figsize, figsize), 
        sharex=True, gridspec_kw={'height_ratios': [1, 3]}
    )

    # Define color palette
    palette = {
        'pc': 'red',
        'nc': 'green',
        'control': 'white',
        'other': 'gray'}
    
    # Scatter plot on both axes with legend completely disabled
    sns.scatterplot(
        data=merged_data,
        x='coefficient',
        y='-log10(p_value)',
        hue='condition',
        style=metadata_column if metadata_column else None,
        s=point_size,
        edgecolor='black',
        palette=palette,
        legend=False,  # Disable automatic legend
        alpha=0.6,
        ax=ax2  # Lower plot
    )

    sns.scatterplot(
        data=merged_data[merged_data['-log10(p_value)'] > 10],
        x='coefficient',
        y='-log10(p_value)',
        hue='condition',
        style=metadata_column if metadata_column else None,
        s=point_size,
        edgecolor='black',
        palette=palette,
        legend=False,  # No legend on the upper plot
        alpha=0.6,
        ax=ax1  # Upper plot
    )

    # Ensure no previous legends on ax1 or ax2
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    # Manually gather handles and labels from ax2 after plotting
    handles, labels = ax2.get_legend_handles_labels()

    # Debug: Print the captured handles and labels for verification
    print(f"Handles: {handles}")
    print(f"Labels: {labels}")

    # Identify shape-based legend entries (skip color-based entries)
    n_color_entries = len(set(merged_data['condition']))
    shape_handles = handles[n_color_entries:]
    shape_labels = labels[n_color_entries:]

    # Create and add the legend with shape-based entries
    legend = Legend(
        ax2, shape_handles, shape_labels,
        bbox_to_anchor=(1.05, 1), loc='upper left',
        handletextpad=2.0, labelspacing=1.5, borderaxespad=1.0,
        markerscale=2.0, prop={'size': 14}
    )
    ax2.add_artist(legend)

    if isinstance(split_axis_lims, list):
        if len(split_axis_lims) == 4:
            ylim_min_ax1 = split_axis_lims[0]
            if split_axis_lims[1] is None:
                ylim_max_ax1 = merged_data['-log10(p_value)'].max() + 5
            else:
                ylim_max_ax1 = split_axis_lims[1]
            ylim_min_ax2 = split_axis_lims[2]
            ylim_max_ax2 = split_axis_lims[3]
        else:
            ylim_min_ax1 = None
            ylim_max_ax1 = merged_data['-log10(p_value)'].max() + 5
            ylim_min_ax2 = 0
            ylim_max_ax2 = None

    # Set axis limits and hide unnecessary parts
    ax1.set_ylim(ylim_min_ax1, ylim_max_ax1)
    ax2.set_ylim(0, ylim_max_ax2)

    if x_lim != None:
        ax1.set_xlim(x_lim)
        ax2.set_xlim(x_lim)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False)

    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Add vertical threshold lines to both plots
    if threshold > 0:
        for ax in (ax1, ax2):
            ax.axvline(x=-abs(threshold), linestyle='--', color='black')
            ax.axvline(x=abs(threshold), linestyle='--', color='black')

    # Add a horizontal line at p-value threshold (0.05)
    ax2.axhline(y=-np.log10(0.05), color='black', linestyle='--')

    # Annotate significant points on both axes
    texts_ax1 = []
    texts_ax2 = []

    for i, row in merged_data.iterrows():
        if row['p_value'] <= 0.05 and abs(row['coefficient']) >= abs(threshold):
            ax = ax1 if row['-log10(p_value)'] >= ax1.get_ylim()[0] else ax2
            # Create the annotation on the selected axis
            text = ax.text(
                row['coefficient'],
                -np.log10(row['p_value']),
                row['variable'],
                fontsize=fontsize,
                ha='center',
                va='bottom',
            )

            # Store the text annotation in the correct list
            if ax == ax1:
                texts_ax1.append(text)
            else:
                texts_ax2.append(text)

    # Adjust text positions to avoid overlap for both axes
    adjust_text(texts_ax1, arrowprops=dict(arrowstyle='-', color='black'), ax=ax1, expand_points=(padd, padd), fontsize=fontsize)
    adjust_text(texts_ax2, arrowprops=dict(arrowstyle='-', color='black'), ax=ax2, expand_points=(padd, padd), fontsize=fontsize)

    # Move the legend outside the lower plot
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust the spacing between subplots and move the title
    plt.subplots_adjust(hspace=0.05)
    fig.suptitle('Custom Volcano Plot of Coefficients', y=1.02, fontsize=16)  # Title above the top plot

    # Save the plot as PDF
    plt.savefig(volcano_path, format='pdf', bbox_inches='tight')
    print(f'Saved Volcano plot: {volcano_path}')

    # Show the plot
    plt.show()

def go_term_enrichment_by_column(significant_df, metadata_path, go_term_columns=['Computed GO Processes', 'Curated GO Components', 'Curated GO Functions', 'Curated GO Processes']):
    """
    Perform GO term enrichment analysis for each GO term column and generate plots.

    Parameters:
    - significant_df: DataFrame containing the significant genes from the screen.
    - metadata_path: Path to the metadata file containing GO terms.
    - go_term_columns: List of columns in the metadata corresponding to GO terms.

    For each GO term column, this function will:
    - Split the GO terms by semicolons.
    - Count the occurrences of GO terms in the hits and in the background.
    - Perform Fisher's exact test for enrichment.
    - Plot the enrichment score vs -log10(p-value).
    """
    
    #significant_df['variable'].fillna(significant_df['feature'], inplace=True)
    #split_columns = significant_df['variable'].str.split('_', expand=True)
    #significant_df['gene_nr'] = split_columns[0]
    #gene_list = significant_df['gene_nr'].to_list()

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