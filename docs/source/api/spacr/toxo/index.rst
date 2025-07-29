spacr.toxo
==========

.. py:module:: spacr.toxo




Module Contents
---------------

.. py:function:: custom_volcano_plot(data_path, metadata_path, metadata_column='tagm_location', point_size=50, figsize=20, threshold=0, save_path=None, x_lim=[-0.5, 0.5], y_lims=[[0, 6], [9, 20]])

.. py:function:: go_term_enrichment_by_column(significant_df, metadata_path, go_term_columns=['Computed GO Processes', 'Curated GO Components', 'Curated GO Functions', 'Curated GO Processes'])

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


.. py:function:: plot_gene_phenotypes(data, gene_list, x_column='Gene ID', data_column='T.gondii GT1 CRISPR Phenotype - Mean Phenotype', error_column='T.gondii GT1 CRISPR Phenotype - Standard Error', save_path=None)

   Plot gene phenotype means with error bars, highlighting selected genes.

   :param data: DataFrame with phenotype data.
   :type data: pd.DataFrame
   :param gene_list: List of gene identifiers to highlight.
   :type gene_list: list of str
   :param x_column: Column with gene identifiers. Default: 'Gene ID'.
   :type x_column: str, optional
   :param data_column: Column with mean phenotype values.
   :type data_column: str, optional
   :param error_column: Column with standard error values.
   :type error_column: str, optional
   :param save_path: Path to save PDF. If None, plot is not saved.
   :type save_path: str or None, optional

   :returns: None


.. py:function:: plot_gene_heatmaps(data, gene_list, columns, x_column='Gene ID', normalize=False, save_path=None)

   Generate a teal-to-white heatmap with the specified columns and genes.

   :param data: The input DataFrame containing gene data.
   :type data: pd.DataFrame
   :param gene_list: A list of genes to include in the heatmap.
   :type gene_list: list
   :param columns: A list of column names to visualize as heatmaps.
   :type columns: list
   :param normalize: If True, normalize the values for each gene between 0 and 1.
   :type normalize: bool
   :param save_path: Optional. If provided, the plot will be saved to this path.
   :type save_path: str


.. py:function:: generate_score_heatmap(settings)

   Generates a score comparison heatmap and calculates MAE between classification scores and a reference fraction.

   This function:
   - Combines classification scores from multiple folders.
   - Calculates the control gRNA fractions for a specific gRNA.
   - Merges the fraction data with classification and cross-validation scores.
   - Plots a heatmap of scores.
   - Calculates Mean Absolute Error (MAE) between the prediction scores and the true fraction.
   - Optionally saves the resulting data and figure to disk.

   :param settings: A dictionary containing:
                    - 'folders': List of folders to search for classification CSVs.
                    - 'csv_name': Name of the classification CSV file in each folder.
                    - 'data_column': Column with predicted values to be compared.
                    - 'plateID': Plate identifier (e.g., 1).
                    - 'columnID': Column identifier (e.g., 'c3').
                    - 'csv': Path to the control gRNA CSV file.
                    - 'control_sgrnas': List of two control gRNA names.
                    - 'fraction_grna': gRNA name for which the true fraction will be extracted.
                    - 'cv_csv': Path to the CSV file with cross-validation predictions.
                    - 'data_column_cv': Column name for cross-validation predictions.
                    - 'dst': Output directory path (or None to disable saving).
   :type settings: dict

   :returns: The merged DataFrame with 'fraction', predicted scores, and cross-validation scores.
   :rtype: pandas.DataFrame


