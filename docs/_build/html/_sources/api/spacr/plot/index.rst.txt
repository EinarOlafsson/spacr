spacr.plot
==========

.. py:module:: spacr.plot






Module Contents
---------------

.. py:function:: plot_image_mask_overlay(file, channels, cell_channel, nucleus_channel, pathogen_channel, figuresize=10, percentiles=(2, 98), thickness=3, save_pdf=True, mode='outlines', export_tiffs=False, all_on_all=False, all_outlines=False, filter_dict=None)

   Plot multi-channel microscopy image overlays with cell, nucleus, and pathogen masks.

   This function visualizes microscopy images with optional mask overlays for different object types.
   It supports contour and filled mask modes, object filtering based on size and intensity, and saving output.

   :param file: Path to the `.npy` image stack (H x W x C).
   :type file: str
   :param channels: Indices of the image channels to display.
   :type channels: list
   :param cell_channel: Channel index for the cell mask intensity.
   :type cell_channel: int or None
   :param nucleus_channel: Channel index for the nucleus mask intensity.
   :type nucleus_channel: int or None
   :param pathogen_channel: Channel index for the pathogen mask intensity.
   :type pathogen_channel: int or None
   :param figuresize: Base figure size for each subplot. Default is 10.
   :type figuresize: int
   :param percentiles: Percentile range for image normalization. Default is (2, 98).
   :type percentiles: tuple
   :param thickness: Thickness of mask contour outlines. Default is 3.
   :type thickness: int
   :param save_pdf: If True, saves a PDF of the overlaid image. Default is True.
   :type save_pdf: bool
   :param mode: 'outlines' or 'filled'. Determines how masks are rendered. Default is 'outlines'.
   :type mode: str
   :param export_tiffs: If True, exports grayscale TIFFs for each image channel. Default is False.
   :type export_tiffs: bool
   :param all_on_all: If True, overlays all outlines on all channels. Default is False.
   :type all_on_all: bool
   :param all_outlines: If True, overlays all outlines on non-matching channels. Default is False.
   :type all_outlines: bool
   :param filter_dict: Dictionary of filtering thresholds for each object type.
                       For example: {"cell": [(min_area, max_area), (min_intensity, max_intensity)]}
   :type filter_dict: dict or None

   :returns: The generated overlay figure.
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_cellpose4_output(batch, masks, flows, cmap='inferno', figuresize=10, nr=1, print_object_number=True)

   Plot the masks and flows for a given batch of images.

   :param batch: The batch of images.
   :type batch: numpy.ndarray
   :param masks: The masks corresponding to the images.
   :type masks: list or numpy.ndarray
   :param flows: The flows corresponding to the images.
   :type flows: list or numpy.ndarray
   :param cmap: The colormap to use for displaying the images. Defaults to 'inferno'.
   :type cmap: str, optional
   :param figuresize: The size of the figure. Defaults to 20.
   :type figuresize: int, optional
   :param nr: The maximum number of images to plot. Defaults to 1.
   :type nr: int, optional
   :param file_type: The file type of the flows. Defaults to '.npz'.
   :type file_type: str, optional
   :param print_object_number: Whether to print the object number on the mask. Defaults to True.
   :type print_object_number: bool, optional

   :returns: None


.. py:function:: plot_masks(batch, masks, flows, cmap='inferno', figuresize=10, nr=1, file_type='.npz', print_object_number=True)

   Plot the masks and flows for a given batch of images.

   :param batch: The batch of images.
   :type batch: numpy.ndarray
   :param masks: The masks corresponding to the images.
   :type masks: list or numpy.ndarray
   :param flows: The flows corresponding to the images.
   :type flows: list or numpy.ndarray
   :param cmap: The colormap to use for displaying the images. Defaults to 'inferno'.
   :type cmap: str, optional
   :param figuresize: The size of the figure. Defaults to 20.
   :type figuresize: int, optional
   :param nr: The maximum number of images to plot. Defaults to 1.
   :type nr: int, optional
   :param file_type: The file type of the flows. Defaults to '.npz'.
   :type file_type: str, optional
   :param print_object_number: Whether to print the object number on the mask. Defaults to True.
   :type print_object_number: bool, optional

   :returns: None


.. py:function:: generate_mask_random_cmap(mask)

   Generate a random colormap based on the unique labels in the given mask.

   Args:
   mask (numpy.ndarray): The input mask array.

   Returns:
   matplotlib.colors.ListedColormap: The random colormap.


.. py:function:: random_cmap(num_objects=100)

   Generate a random colormap.

   Args:
   num_objects (int): The number of objects to generate colors for. Default is 100.

   Returns:
   random_cmap (matplotlib.colors.ListedColormap): A random colormap.


.. py:function:: plot_images_and_arrays(folders, lower_percentile=1, upper_percentile=99, threshold=1000, extensions=['.npy', '.tif', '.tiff', '.png'], overlay=False, max_nr=None, randomize=True)

   Plot images and arrays from the given folders.

   :param folders: A list of folder paths containing the images and arrays.
   :type folders: list
   :param lower_percentile: The lower percentile for image normalization. Defaults to 1.
   :type lower_percentile: int, optional
   :param upper_percentile: The upper percentile for image normalization. Defaults to 99.
   :type upper_percentile: int, optional
   :param threshold: The threshold for determining whether to display an image as a mask or normalize it. Defaults to 1000.
   :type threshold: int, optional
   :param extensions: A list of file extensions to consider. Defaults to ['.npy', '.tif', '.tiff', '.png'].
   :type extensions: list, optional
   :param overlay: If True, overlay the outlines of the objects on the image. Defaults to False.
   :type overlay: bool, optional


.. py:function:: plot_arrays(src, figuresize=10, cmap='inferno', nr=1, normalize=True, q1=1, q2=99)

   Plot randomly selected arrays from a given directory or a single .npz/.npy file.

   Args:
   - src (str): The directory path or file path containing the arrays.
   - figuresize (int): The size of the figure (default: 10).
   - cmap (str): The colormap to use for displaying the arrays (default: 'inferno').
   - nr (int): The number of arrays to plot (default: 1).
   - normalize (bool): Whether to normalize the arrays (default: True).
   - q1 (int): The lower percentile for normalization (default: 1).
   - q2 (int): The upper percentile for normalization (default: 99).


.. py:function:: plot_arrays_v1(src, figuresize=10, cmap='inferno', nr=1, normalize=True, q1=1, q2=99)

   Plot randomly selected arrays from a given directory.

   Args:
   - src (str): The directory path containing the arrays.
   - figuresize (int): The size of the figure (default: 50).
   - cmap (str): The colormap to use for displaying the arrays (default: 'inferno').
   - nr (int): The number of arrays to plot (default: 1).
   - normalize (bool): Whether to normalize the arrays (default: True).
   - q1 (int): The lower percentile for normalization (default: 1).
   - q2 (int): The upper percentile for normalization (default: 99).

   Returns:
   None


.. py:function:: plot_merged(src, settings)

   Plot the merged images after applying various filters and modifications.

   :param src: Path to folder with images.
   :type src: path
   :param settings: The settings for the plot.
   :type settings: dict

   :returns: None


.. py:function:: generate_plate_heatmap(df, plate_number, variable, grouping, min_max, min_count)

   Generate a heatmap matrix for a specific plate based on the specified variable.

   :param df: Input DataFrame containing 'prc' and variable to plot.
   :type df: pd.DataFrame
   :param plate_number: Identifier of the plate to generate the heatmap for.
   :type plate_number: str
   :param variable: Column name to be visualized.
   :type variable: str
   :param grouping: Aggregation method: 'mean', 'sum', or 'count'.
   :type grouping: str
   :param min_max: Controls the colormap scaling:
                   - 'all': use full data min/max
                   - 'allq': use 2nd to 98th percentile
                   - (low, high): percentiles (float) or fixed values (int)
   :type min_max: str or tuple
   :param min_count: Minimum number of entries per well to be included.
   :type min_count: int or float

   :returns: Pivoted heatmap matrix indexed by row and column.
             min_max (list): The computed min/max values for the heatmap color scale.
   :rtype: plate_map (pd.DataFrame)

   :raises ValueError: If `grouping` is not one of ['mean', 'sum', 'count'].


.. py:function:: plot_plates(df, variable, grouping, min_max, cmap, min_count=0, verbose=True, dst=None)

   Generate and optionally save heatmaps for one or more plates, showing the spatial distribution
   of a specified variable (e.g., unique_counts or signal intensity).

   :param df: Input DataFrame containing 'prc' in the format 'plate_row_col' and the variable to plot.
   :type df: pd.DataFrame
   :param variable: Name of the column to visualize.
   :type variable: str
   :param grouping: Aggregation method for replicates (e.g., 'mean', 'median').
   :type grouping: str
   :param min_max: Method to determine color scaling. One of {'global', 'local', 'allq'}.
   :type min_max: str
   :param cmap: Colormap to use for heatmap visualization.
   :type cmap: str or Colormap
   :param min_count: Minimum count threshold for inclusion in the heatmap. Default is 0.
   :type min_count: int, optional
   :param verbose: If True, displays the heatmap(s) in an interactive window. Default is True.
   :type verbose: bool, optional
   :param dst: Directory to save the heatmaps. If None, the heatmaps are not saved.
   :type dst: str, optional

   :returns: The figure containing the plotted heatmaps.
   :rtype: matplotlib.figure.Figure

   Side Effects:
       - Displays plate heatmaps using matplotlib/seaborn.
       - Saves the heatmap figure as 'plate_heatmap_#.pdf' in `dst` if provided and writable.


.. py:function:: print_mask_and_flows(stack, mask, flows, overlay=True, max_size=1000, thickness=2)

   Display the original image, mask with outlines, and flow images.

   :param stack: Original image or stack.
   :type stack: np.array
   :param mask: Mask image.
   :type mask: np.array
   :param flows: List of flow images.
   :type flows: list
   :param overlay: Whether to overlay the mask outlines on the original image.
   :type overlay: bool
   :param max_size: Maximum allowed size for any dimension of the images.
   :type max_size: int
   :param thickness: Thickness of the contour outlines.
   :type thickness: int


.. py:function:: plot_resize(images, resized_images, labels, resized_labels)

   Plot original and resized images along with their corresponding labels for visual comparison.

   :param images: List or array of original input images. Each image can be 2D (grayscale)
                  or 3D (with 1, 3, or 4 channels).
   :type images: list or np.ndarray
   :param resized_images: Corresponding resized input images.
   :type resized_images: list or np.ndarray
   :param labels: List or array of original label masks.
   :type labels: list or np.ndarray
   :param resized_labels: Corresponding resized label masks.
   :type resized_labels: list or np.ndarray

   :returns:     - Top-left: Original image
                 - Top-right: Resized image
                 - Bottom-left: Original label
                 - Bottom-right: Resized label
   :rtype: None. Displays a 2x2 grid of plots

   .. rubric:: Notes

   - Images with 1, 3, or 4 channels are supported.
   - Other channel numbers will be visualized by averaging across channels (grayscale fallback).


.. py:function:: normalize_and_visualize(image, normalized_image, title='')

   Display a side-by-side comparison of an original image and its normalized version.

   :param image: The original image. Can be 2D (grayscale) or 3D (multi-channel).
   :type image: np.ndarray
   :param normalized_image: The normalized image. Should have the same dimensions as `image`.
   :type normalized_image: np.ndarray
   :param title: Optional string to append to the plot titles. Defaults to "".
   :type title: str, optional

   :returns:     - Left: Original image
                 - Right: Normalized image
   :rtype: None. Displays a matplotlib figure with two subplots

   .. rubric:: Notes

   - For multi-channel images, the mean across channels is visualized.
   - Axes are hidden for clarity.


.. py:function:: visualize_masks(mask1, mask2, mask3, title='Masks Comparison')

   Display three segmentation masks side-by-side for visual comparison.

   :param mask1: First mask array (2D).
   :type mask1: np.ndarray
   :param mask2: Second mask array (2D).
   :type mask2: np.ndarray
   :param mask3: Third mask array (2D).
   :type mask3: np.ndarray
   :param title: Title for the entire figure. Defaults to "Masks Comparison".
   :type title: str, optional

   :returns:     - Each subplot corresponds to one of the input masks.
                 - Masks are visualized using a randomly generated colormap.
                 - If a mask is binary (0 and 1 only), no normalization is applied.
                 - Otherwise, intensity values are normalized to [0, max(mask)].
   :rtype: None. Displays a matplotlib figure with three subplots

   .. rubric:: Notes

   Requires the function `generate_mask_random_cmap(mask)` to generate a colormap.


.. py:function:: visualize_cellpose_masks(masks, titles=None, filename=None, save=False, src=None)

   Visualize multiple masks with optional titles.

   :param masks: A list of masks to visualize.
   :type masks: list of np.ndarray
   :param titles: A list of titles for the masks. If None, default titles will be used.
   :type titles: list of str, optional
   :param comparison_title: Title for the entire figure.
   :type comparison_title: str


.. py:function:: plot_comparison_results(comparison_results)

   Visualize segmentation comparison metrics using boxplots with overlaid strip plots.

   :param comparison_results: Each dictionary represents one sample, with keys:
                              - 'filename': Name of the sample.
                              - Metric keys such as 'jaccard_*', 'dice_*', 'boundary_f1_*', 'average_precision_*'.
   :type comparison_results: list of dict

   :returns:

             The resulting figure with 4 subplots:
                 - Jaccard Index
                 - Dice Coefficient
                 - Boundary F1 Score
                 - Average Precision
   :rtype: matplotlib.figure.Figure

   .. rubric:: Notes

   - Metrics are grouped by type using substring matching in column names.
   - Outliers and individual sample values are shown with strip plots.
   - Assumes that all metric columns contain numeric values.


.. py:function:: plot_object_outlines(src, objects=['nucleus', 'cell', 'pathogen'], channels=[0, 1, 2], max_nr=10)

   Overlay mask outlines on image channels for visual inspection of segmentation quality.

   :param src: Source directory containing the 'masks' folder and image channels.
   :type src: str
   :param objects: List of object names (e.g., 'nucleus', 'cell', 'pathogen') whose masks to visualize.
   :type objects: list of str
   :param channels: Corresponding channel indices (0-based) for each object.
   :type channels: list of int
   :param max_nr: Maximum number of overlays to display.
   :type max_nr: int

   :returns: None. Displays overlays using `plot_images_and_arrays()`.

   .. rubric:: Notes

   - For each object/channel pair, attempts to load:
       - The mask from 'masks/{object}_mask_stack'
       - The image from '{channel+1}' subfolder (1-based indexing)
   - Assumes `plot_images_and_arrays()` can handle paired overlay inputs.
   - Random selection and percentile contrast stretching applied for visualization.


.. py:function:: volcano_plot(coef_df, filename='volcano_plot.pdf')

   Generate and save a volcano plot based on coefficient and p-value data.

   :param coef_df: DataFrame containing columns:
                   - 'coefficient': effect size for each term.
                   - 'p_value': p-value for each coefficient.
                   - 'condition': category for coloring (e.g., 'pc', 'nc', 'control', 'other').
   :type coef_df: pd.DataFrame
   :param filename: File path for saving the plot as a PDF.
   :type filename: str

   :returns: None. Displays the volcano plot and saves it to the specified file.

   .. rubric:: Notes

   - Highlights p-value threshold at 0.05 with a horizontal dashed red line.
   - Uses pre-defined color palette for conditions.
   - Legend is removed for clarity.


.. py:function:: plot_histogram(df, column, dst=None)

   Plot and optionally save a histogram for a specified column in a DataFrame.

   :param df: DataFrame containing the data.
   :type df: pd.DataFrame
   :param column: Name of the column to plot.
   :type column: str
   :param dst: Directory to save the figure as a PDF. If None, the figure is not saved.
   :type dst: str or None

   :returns: None. Displays the histogram and optionally saves it to disk.

   .. rubric:: Notes

   - The histogram uses a fixed turquoise bar color (RGB: 0,155,155).
   - Saved file is named '{column}_histogram.pdf' and placed in the specified `dst` directory.


.. py:function:: plot_lorenz_curves(csv_files, name_column='grna_name', value_column='count', remove_keys=None, x_lim=[0.0, 1], y_lim=[0, 1], remove_outliers=False, save=True)

   Plot Lorenz curves and compute Gini coefficients for a set of CSV files.

   :param csv_files: Paths to CSV files containing count data.
   :type csv_files: list of str
   :param name_column: Column name containing the gRNA or entity names.
   :type name_column: str
   :param value_column: Column name with the numerical values (e.g., counts).
   :type value_column: str
   :param remove_keys: List of names to exclude from analysis.
   :type remove_keys: list or None
   :param x_lim: X-axis limits for the plot.
   :type x_lim: list
   :param y_lim: Y-axis limits for the plot.
   :type y_lim: list
   :param remove_outliers: Whether to remove outlier entities by well counts.
   :type remove_outliers: bool
   :param save: If True, saves the plot to a PDF.
   :type save: bool

   :returns: None. Displays the plot and prints Gini coefficients.

   .. rubric:: Notes

   - Gini coefficient is a measure of inequality (0 = perfect equality, 1 = maximal inequality).
   - Outlier removal is based on IQR of well counts per entity name.
   - Saves to 'lorenz_curve_with_gini.pdf' under a 'results' folder next to the first input CSV.


.. py:function:: plot_permutation(permutation_df)

   Plot permutation feature importance as a horizontal bar chart.

   :param permutation_df: DataFrame with columns:
                          - 'feature': Feature names.
                          - 'importance_mean': Mean permutation importance.
                          - 'importance_std': Standard deviation of permutation importance.
   :type permutation_df: pd.DataFrame

   :returns: The resulting plot figure object.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Notes

   - Dynamically adjusts figure size and font size based on number of features.
   - Error bars represent the standard deviation across permutation runs.


.. py:function:: plot_feature_importance(feature_importance_df)

   Plot feature importance as a horizontal bar chart.

   :param feature_importance_df: DataFrame with columns:
                                 - 'feature': Feature names.
                                 - 'importance': Importance scores for each feature.
   :type feature_importance_df: pd.DataFrame

   :returns: The resulting plot figure object.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Notes

   - Dynamically adjusts figure size and font size based on number of features.
   - Use for visualizing static (e.g., model-based) feature importance.


.. py:function:: read_and_plot__vision_results(base_dir, y_axis='accuracy', name_split='_time', y_lim=[0.8, 0.9])

   Read multiple vision model test result CSV files and generate a bar plot of average performance.

   :param base_dir: Base directory to recursively search for files ending in '_test_result.csv'.
   :type base_dir: str
   :param y_axis: The performance metric to plot (must be a column in each CSV). Default is 'accuracy'.
   :type y_axis: str
   :param name_split: String delimiter used to extract the model name from the filename. Default is '_time'.
   :type name_split: str
   :param y_lim: Optional y-axis limits for the plot. Default is [0.8, 0.9].
   :type y_lim: list or tuple

   :returns: None. Displays a matplotlib bar plot of average `y_axis` for each model.

   .. rubric:: Notes

   - Assumes each file is named with format: '<model>_time<timestamp>_test_result.csv'.
   - Extracts 'model' and 'epoch' metadata from the path and filename.
   - Saves results to a 'result' folder inside `base_dir` (created if not present).


.. py:function:: jitterplot_by_annotation(src, x_column, y_column, plot_title='Jitter Plot', output_path=None, filter_column=None, filter_values=None)

   Reads a CSV file and creates a jitter plot of one column grouped by another column.

   Args:
   src (str): Path to the source data.
   x_column (str): Name of the column to be used for the x-axis.
   y_column (str): Name of the column to be used for the y-axis.
   plot_title (str): Title of the plot. Default is 'Jitter Plot'.
   output_path (str): Path to save the plot image. If None, the plot will be displayed. Default is None.

   Returns:
   pd.DataFrame: The filtered and balanced DataFrame.


.. py:function:: create_grouped_plot(df, grouping_column, data_column, graph_type='bar', summary_func='mean', order=None, colors=None, output_dir='./output', save=False, y_lim=None, error_bar_type='std')

   Create a grouped plot, perform statistical tests, and optionally export the results along with the plot.

   Args:
   - df: DataFrame containing the data.
   - grouping_column: Column name for the categorical grouping.
   - data_column: Column name for the data to be grouped and plotted.
   - graph_type: Type of plot ('bar', 'violin', 'jitter', 'box', 'jitter_box').
   - summary_func: Summary function to apply to each group ('mean', 'median', etc.).
   - order: List specifying the order of the groups. If None, groups will be ordered alphabetically.
   - colors: List of colors for each group.
   - output_dir: Directory where the figure and test results will be saved if `save=True`.
   - save: Boolean flag indicating whether to save the plot and results to files.
   - y_lim: Optional y-axis min and max.
   - error_bar_type: Type of error bars to plot, either 'std' for standard deviation or 'sem' for standard error of the mean.

   Outputs:
   - Figure of the plot.
   - DataFrame with full statistical test results, including normality tests.


.. py:class:: spacrGraph(df, grouping_column, data_column, graph_type='bar', summary_func='mean', order=None, colors=None, output_dir='./output', save=False, y_lim=None, log_y=False, log_x=False, error_bar_type='std', remove_outliers=False, theme='pastel', representation='object', paired=False, all_to_all=True, compare_group=None, graph_name=None)

   Class for generating grouped plots with optional data preprocessing,
   statistical comparisons, theming, and output control.

   This class is designed to support common grouped plotting tasks for CRISPR screens,
   enabling flexible summarization across different experimental representations
   (e.g., object-level, well-level, or plate-level).

   :param df: Input DataFrame containing data to plot.
   :type df: pd.DataFrame
   :param grouping_column: Column to group by on the x-axis.
   :type grouping_column: str
   :param data_column: One or more numeric columns to summarize and plot.
   :type data_column: str or list
   :param graph_type: Type of plot to create (e.g., 'bar', 'box').
   :type graph_type: str
   :param summary_func: Function used to summarize data (e.g., 'mean', 'median').
   :type summary_func: str or callable
   :param order: Optional list to define the order of categories on the x-axis.
   :type order: list
   :param colors: Optional custom list of colors for plotting.
   :type colors: list
   :param output_dir: Path to directory where output will be saved.
   :type output_dir: str
   :param save: Whether to save the plot as a PDF.
   :type save: bool
   :param y_lim: Tuple defining y-axis limits.
   :type y_lim: tuple
   :param log_y: If True, applies log scaling to y-axis.
   :type log_y: bool
   :param log_x: If True, applies log scaling to x-axis.
   :type log_x: bool
   :param error_bar_type: Error bar type to display ('std' or 'sem').
   :type error_bar_type: str
   :param remove_outliers: Whether to exclude statistical outliers (not implemented here).
   :type remove_outliers: bool
   :param theme: Seaborn theme to use for the plot.
   :type theme: str
   :param representation: Level of summarization; one of {'object', 'well', 'plate'}.
   :type representation: str
   :param paired: If True, assumes samples are paired for statistical tests.
   :type paired: bool
   :param all_to_all: If True, performs all-to-all comparisons.
   :type all_to_all: bool
   :param compare_group: Optional group to compare all others against.
   :type compare_group: str
   :param graph_name: Optional name used in output file naming.
   :type graph_name: str

   Initialize a spacrGraph instance for grouped data visualization.

   :param df: Input dataframe containing data to be plotted.
   :type df: pd.DataFrame
   :param grouping_column: Column name to group data along the x-axis.
   :type grouping_column: str
   :param data_column: Column(s) containing values to summarize and plot.
   :type data_column: str or list
   :param graph_type: Type of plot to generate ('bar', 'box', etc.). Default is 'bar'.
   :type graph_type: str
   :param summary_func: Summary statistic to apply per group (e.g., 'mean', 'median').
   :type summary_func: str or callable
   :param order: Optional order of groups on the x-axis. If None, will be inferred from the data.
   :type order: list or None
   :param colors: List of colors to use for plotting. Defaults to seaborn palette.
   :type colors: list or None
   :param output_dir: Directory where plot will be saved if `save=True`.
   :type output_dir: str
   :param save: Whether to save the plot as a PDF.
   :type save: bool
   :param y_lim: Y-axis limits. If None, determined automatically.
   :type y_lim: tuple or None
   :param log_y: If True, use logarithmic scale on the y-axis.
   :type log_y: bool
   :param log_x: If True, use logarithmic scale on the x-axis.
   :type log_x: bool
   :param error_bar_type: Type of error bars to include ('std' or 'sem').
   :type error_bar_type: str
   :param remove_outliers: Whether to remove outliers from the data before plotting.
   :type remove_outliers: bool
   :param theme: Seaborn color theme for plotting.
   :type theme: str
   :param representation: Level of summarization: 'object', 'well', or 'plate'.
   :type representation: str
   :param paired: Whether samples are paired for statistical testing.
   :type paired: bool
   :param all_to_all: If True, perform all pairwise comparisons.
   :type all_to_all: bool
   :param compare_group: If provided, compares all groups against this one.
   :type compare_group: str or None
   :param graph_name: Optional string to include in saved file names.
   :type graph_name: str or None


   .. py:attribute:: df


   .. py:attribute:: grouping_column


   .. py:attribute:: order


   .. py:attribute:: data_column


   .. py:attribute:: graph_type
      :value: 'bar'



   .. py:attribute:: summary_func
      :value: 'mean'



   .. py:attribute:: colors
      :value: None



   .. py:attribute:: output_dir
      :value: './output'



   .. py:attribute:: save
      :value: False



   .. py:attribute:: error_bar_type
      :value: 'std'



   .. py:attribute:: remove_outliers
      :value: False



   .. py:attribute:: theme
      :value: 'pastel'



   .. py:attribute:: representation
      :value: 'object'



   .. py:attribute:: paired
      :value: False



   .. py:attribute:: all_to_all
      :value: True



   .. py:attribute:: compare_group
      :value: None



   .. py:attribute:: y_lim
      :value: None



   .. py:attribute:: graph_name
      :value: None



   .. py:attribute:: log_x
      :value: False



   .. py:attribute:: log_y
      :value: False



   .. py:attribute:: results_df


   .. py:attribute:: sns_palette
      :value: None



   .. py:attribute:: fig
      :value: None



   .. py:attribute:: results_name
      :value: '___'



   .. py:attribute:: raw_df


   .. py:method:: preprocess_data()

      Preprocess the input DataFrame by removing NaNs and optionally aggregating data.

      Aggregation is controlled by the `representation` parameter:
          - 'object': No aggregation; raw data is retained.
          - 'well': Data is grouped by ['prc', grouping_column].
          - 'plate': Data is grouped by ['plateID', grouping_column],
                  extracting plateID from the 'prc' column if needed.

      Ordering of the x-axis categories is set using the `order` parameter
      or automatically sorted if not provided.

      :returns: Cleaned and optionally aggregated DataFrame.
      :rtype: pd.DataFrame



   .. py:method:: remove_outliers_from_plot()

      Remove outliers from each group in the dataset for plotting purposes.
      This method applies the IQR method to filter extreme values per group.

      :returns: Filtered dataframe with outliers removed for plotting.
      :rtype: pd.DataFrame



   .. py:method:: perform_normality_tests()

      Perform normality tests on each group and data column in the dataframe.

      Uses:
          - Shapiro-Wilk test for sample sizes < 8
          - D'Agostino-Pearson test for sample sizes ≥ 8

      :returns:     - is_normal (bool): True if all groups pass normality test (p > 0.05)
                    - normality_results (list): List of dictionaries with test results
      :rtype: tuple



   .. py:method:: perform_levene_test(unique_groups)

      Perform Levene’s test for homogeneity of variance across groups.

      :param unique_groups: List of group identifiers to compare.
      :type unique_groups: list

      :returns:     - stat (float): Levene test statistic
                    - p_value (float): p-value for the test
      :rtype: tuple



   .. py:method:: perform_statistical_tests(unique_groups, is_normal)

      Perform appropriate statistical tests based on normality and group count.

      If 2 groups:
          - Paired or unpaired t-test (if normal)
          - Wilcoxon or Mann-Whitney U test (if non-normal)

      If >2 groups:
          - ANOVA (if normal)
          - Kruskal-Wallis test (if non-normal)

      :param unique_groups: List of unique group names.
      :type unique_groups: list
      :param is_normal: Whether the data passes normality tests.
      :type is_normal: bool

      :returns: List of dictionaries with test results for each data column.
      :rtype: list



   .. py:method:: perform_posthoc_tests(is_normal, unique_groups)

      Perform post-hoc tests for multiple groups, depending on normality and group count.

      If data are normally distributed and `all_to_all` is True:
          - Tukey HSD test is performed for pairwise comparisons.

      If not normal and `all_to_all` is True:
          - Dunn's test is performed using appropriate p-value adjustment.

      :param is_normal: Whether the data passed normality tests.
      :type is_normal: bool
      :param unique_groups: List of group identifiers.
      :type unique_groups: list

      :returns: List of dictionaries with post-hoc test results per pairwise comparison.
      :rtype: list



   .. py:method:: create_plot(ax=None)

      Create the plot based on the selected `graph_type`.

      If `graph_type` supports it and multiple `data_column` values are used,
      symbols can be placed below the plot to indicate groupings and correspondence.

      :param ax: An existing axis to draw the plot on.
                 If None, a new figure and axis are created.
      :type ax: matplotlib.axes.Axes, optional

      :returns: The resulting figure.
      :rtype: matplotlib.figure.Figure



   .. py:method:: get_results()

      Return the internal results DataFrame with computed statistics or comparisons.

      :returns: Results table.
      :rtype: pd.DataFrame



   .. py:method:: get_figure()

      Return the Matplotlib figure associated with the most recent plot.

      :returns: The plot figure.
      :rtype: matplotlib.figure.Figure



.. py:function:: plot_data_from_db(settings)

   Load measurement data from SQL databases, annotate with experimental conditions,
   and generate grouped plots using spacrGraph.

   :param settings: A dictionary containing the following keys:
                    - 'src' (str or list of str): Source directories containing measurement databases.
                    - 'database' (str or list of str): Corresponding database filenames.
                    - 'table_names' (str): Name of the SQL table to extract data from.
                    - 'graph_name' (str): Name for saving the output plot and settings.
                    - 'grouping_column' (str): Column to group by on the x-axis.
                    - 'data_column' (str): Column to plot on the y-axis.
                    - 'channel_of_interest' (int): Channel number for recruitment calculation (if applicable).
                    - 'cell_types' (list): List of expected host cell types.
                    - 'pathogen_types' (list): List of expected pathogen types.
                    - 'treatments' (list): List of expected treatments.
                    - 'cell_plate_metadata' (str or None): Path to metadata for host cells.
                    - 'pathogen_plate_metadata' (str or None): Path to metadata for pathogens.
                    - 'treatment_plate_metadata' (str or None): Path to metadata for treatments.
                    - 'nuclei_limit' (int): Max number of nuclei to load per plate (optional).
                    - 'pathogen_limit' (int): Max number of pathogens to load per plate (optional).
                    - 'verbose' (bool): Whether to print loading information.
                    - 'graph_type' (str): Type of plot to create (e.g., 'bar', 'box', 'jitter', etc.).
                    - 'representation' (str): Data representation style for plotting (optional).
                    - 'theme' (str): Seaborn color palette theme.
                    - 'y_lim' (tuple or None): Limits for y-axis scaling.
                    - 'save' (bool): Whether to save the plot and results to disk.
   :type settings: dict

   :returns: The plotted figure.
             results_df (pd.DataFrame): Statistical results or summary data used in the plot.
   :rtype: fig (matplotlib.figure.Figure)

   .. rubric:: Notes

   - Automatically handles multi-source input.
   - Computes recruitment ratio if specified.
   - Drops rows with missing grouping or data values.
   - Creates and saves plot and settings to disk if 'save' is True.


.. py:function:: plot_data_from_csv(settings)

   Load measurement data from one or more CSV files, optionally filter and clean the data,
   and generate grouped plots using spacrGraph.

   :param settings: A dictionary containing the following keys:
                    - 'src' (str or list of str): Path(s) to the CSV file(s) containing measurement data.
                    - 'grouping_column' (str): Column to group by on the x-axis.
                    - 'data_column' (str): Column to plot on the y-axis.
                    - 'graph_name' (str): Name for saving the output plot and settings.
                    - 'graph_type' (str): Type of plot to create (e.g., 'bar', 'box', 'violin', 'jitter').
                    - 'theme' (str): Seaborn color palette theme.
                    - 'log_y' (bool): Whether to log-transform the y-axis.
                    - 'log_x' (bool): Whether to log-transform the x-axis.
                    - 'y_lim' (tuple or None): Limits for y-axis scaling (optional).
                    - 'save' (bool): Whether to save the plot and results to disk.
                    - 'verbose' (bool): Whether to print and display the DataFrame before plotting.
                    - 'representation' (str): Plot style to use (e.g., 'jitter_box', 'violin', etc.).
                    - 'keep_groups' (list or str, optional): Restrict plot to a subset of group labels.
                    - 'remove_outliers' (bool): Whether to remove outliers using IQR filtering.
   :type settings: dict

   :returns: The plotted figure.
             results_df (pd.DataFrame): DataFrame with statistical results or plot summary.
   :rtype: fig (matplotlib.figure.Figure)

   .. rubric:: Notes

   - Merges multiple CSVs and tags them with a default 'plateID' if not provided.
   - Attempts to split 'prc' column into 'plateID', 'rowID', and 'columnID' if applicable.
   - Handles missing values in the grouping and data columns by dropping those rows.
   - Automatically creates the output directory and saves results if `save` is True.


.. py:function:: plot_region(settings)

   Generate and save region-specific plots including: mask overlay, raw PNG grid, and activation map grid.

   This function loads image paths and metadata from database tables, filters for the specified region (field of view),
   and plots the following:

   - A mask overlay of the full field image with Cellpose masks.
   - A grid of raw cropped PNGs corresponding to the region.
   - A grid of activation maps (e.g., saliency, Grad-CAM) for the same region.

   :param settings: A dictionary containing the following keys:

                    - 'src' (str): Source folder containing subfolders like 'merged', 'measurements', and 'datasets'.
                    - 'name' (str): Filename (e.g., 'plate1_A01_01.tif') identifying the region/FOV of interest.
                    - 'activation_db' (str): Filename of the activation measurement database (e.g., 'activations.db').
                    - 'activation_mode' (str): Mode of activation ('saliency', 'gradcam', etc.); used to find the correct table.
                    - 'channels' (list of int): Indices of input channels to use for the mask overlay.
                    - 'cell_channel' (int): Channel index used to generate the cell mask.
                    - 'nucleus_channel' (int): Channel index used to generate the nucleus mask.
                    - 'pathogen_channel' (int): Channel index used to generate the pathogen mask.
                    - 'percentiles' (list): Two-element list (e.g., [2, 99]) specifying intensity percentiles for contrast scaling.
                    - 'mode' (str): Image display mode for overlay ('rgb', 'stack', etc.).
                    - 'export_tiffs' (bool): Whether to export TIFF images alongside overlays.
   :type settings: dict

   :returns:

             A 3-element tuple of matplotlib Figure objects or None

                 - fig_1 (matplotlib.figure.Figure or None): Mask overlay figure.
                 - fig_2 (matplotlib.figure.Figure or None): Grid of raw cropped PNGs.
                 - fig_3 (matplotlib.figure.Figure or None): Grid of activation maps.
   :rtype: tuple

   .. rubric:: Notes

   - Figures are saved as PDFs under ``<src>/results/<name>/``.
   - If no relevant PNGs or activations are found, the corresponding figure will be ``None``.
   - Paths are automatically corrected using ``correct_paths``.
   - The figure layout uses ``plot_image_grid`` and ``plot_image_mask_overlay``.


.. py:function:: plot_image_grid(image_paths, percentiles)

   Plots a square grid of images from a list of image paths.
   Unused subplots are filled with black, and padding is minimized.

   Args:
   - image_paths: List of paths to images to be displayed.

   Returns:
   - fig: The generated matplotlib figure.


.. py:function:: overlay_masks_on_images(img_folder, normalize=True, resize=True, save=False, plot=False, thickness=2)

   Load images and masks from folders, overlay mask contours on images, and optionally normalize, resize, and save.

   :param img_folder: Path to the folder containing images.
   :type img_folder: str
   :param mask_folder: Path to the folder containing masks.
   :type mask_folder: str
   :param normalize: If True, normalize images to the 1st and 99th percentiles.
   :type normalize: bool
   :param resize: If True, resize the final overlay to 500x500.
   :type resize: bool
   :param save: If True, save the final overlay in an 'overlay' folder within the image folder.
   :type save: bool
   :param thickness: Thickness of the contour lines.
   :type thickness: int


.. py:function:: graph_importance(settings)

   Generate and display a bar, box, or violin plot of importance values across grouped categories.

   This function reads one or more CSV files containing importance scores (e.g., feature importances,
   saliency values, or other metrics) and visualizes them grouped by a specified column. It wraps
   the `spacrGraph` plotting class to create the plot and saves the settings used.

   :param settings: A dictionary containing the following keys:
                    - 'csvs' (str or list of str): Path(s) to CSV file(s) containing importance data.
                    - 'grouping_column' (str): Column name used for grouping on the x-axis.
                    - 'data_column' (str): Column name containing the data values to be plotted (e.g., importance scores).
                    - 'graph_type' (str): Type of plot to generate ('bar', 'box', or 'violin').
                    - 'save' (bool): Whether to save the plot as a PDF in the same directory as the input CSVs.
   :type settings: dict

   :returns: None

   .. rubric:: Notes

   - If the required columns are missing from the input data, the function will print a warning and exit.
   - The plot is created using the `spacrGraph` class and is shown with `matplotlib.pyplot.show()`.
   - All input CSVs are concatenated before plotting.
   - Settings are saved to disk using `save_settings`.


.. py:function:: plot_proportion_stacked_bars(settings, df, group_column, bin_column, prc_column='prc', level='object', cmap='viridis')

   Generate a stacked bar plot for proportions and perform chi-squared and pairwise tests.

   Args:
   - settings (dict): Analysis settings.
   - df (DataFrame): Input data.
   - group_column (str): Column indicating the groups.
   - bin_column (str): Column indicating the categories.
   - prc_column (str): Optional; column for additional stratification.
   - level (str): Level of aggregation ('well' or 'object').

   Returns:
   - chi2 (float): Chi-squared statistic for the overall test.
   - p (float): p-value for the overall chi-squared test.
   - dof (int): Degrees of freedom for the overall chi-squared test.
   - expected (ndarray): Expected frequencies for the overall chi-squared test.
   - raw_counts (DataFrame): Contingency table of observed counts.
   - fig (Figure): The generated plot.
   - pairwise_results (list): Pairwise test results from `chi_pairwise`.


.. py:function:: create_venn_diagram(file1, file2, gene_column='gene', filter_coeff=0.1, save=True, save_path=None)

   Reads two CSV files, extracts the `gene` column, and creates a Venn diagram
   to show overlapping and non-overlapping genes.

   :param file1: Path to the first CSV file.
   :type file1: str
   :param file2: Path to the second CSV file.
   :type file2: str
   :param gene_column: Name of the column containing gene data (default: "gene").
   :type gene_column: str
   :param filter_coeff: Coefficient threshold for filtering genes.
   :type filter_coeff: float
   :param save: Whether to save the plot.
   :type save: bool
   :param save_path: Path to save the Venn diagram figure.
   :type save_path: str

   :returns: Overlapping and non-overlapping genes.
   :rtype: dict


