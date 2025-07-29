spacr.submodules
================

.. py:module:: spacr.submodules






Module Contents
---------------

.. py:class:: CellposeLazyDataset(image_files, label_files, settings, randomize=True, augment=False)

   Bases: :py:obj:`torch.utils.data.Dataset`


   A PyTorch Dataset for lazy loading and optional augmentation of images and segmentation masks
   for training Cellpose-based models.

   Images and labels are loaded from file paths on-the-fly, optionally normalized and resized,
   and can be augmented with basic geometric transformations.

   :param image_files: List of file paths to image files.
   :type image_files: list of str
   :param label_files: List of file paths to corresponding label files.
   :type label_files: list of str
   :param settings: Dictionary containing dataset settings:
                    - 'normalize' (bool): Whether to apply percentile-based intensity normalization.
                    - 'percentiles' (list of int): Two-element list specifying lower and upper percentiles (default: [2, 99]).
                    - 'target_size' (int): Desired output image size (height and width).
   :type settings: dict
   :param randomize: Whether to shuffle the dataset order. Defaults to True.
   :type randomize: bool, optional
   :param augment: Whether to apply 8-way geometric data augmentation. Defaults to False.
   :type augment: bool, optional

   Initialize the CellposeLazyDataset.

   :param image_files: Paths to input image files.
   :type image_files: list of str
   :param label_files: Paths to corresponding label files.
   :type label_files: list of str
   :param settings: Configuration dictionary with keys:
                    - 'normalize' (bool)
                    - 'percentiles' (list of int)
                    - 'target_size' (int)
   :type settings: dict
   :param randomize: Shuffle file order. Defaults to True.
   :type randomize: bool, optional
   :param augment: Enable 8-fold augmentation. Defaults to False.
   :type augment: bool, optional


   .. py:attribute:: normalize


   .. py:attribute:: percentiles


   .. py:attribute:: target_size


   .. py:attribute:: augment
      :value: False



   .. py:method:: __len__()

      Return the number of samples in the dataset.

      If augmentation is enabled, each sample contributes 8 variants.

      :returns: Total number of samples (augmented if applicable).
      :rtype: int



   .. py:method:: apply_augmentation(image, label, aug_idx)

      Apply one of 8 geometric augmentations to an image-label pair.

      Augmentations include rotations (90°, 180°, 270°) and horizontal/vertical flips.

      :param image: Input image array.
      :type image: ndarray
      :param label: Corresponding label array.
      :type label: ndarray
      :param aug_idx: Index from 0 to 7 specifying the augmentation to apply.
      :type aug_idx: int

      :returns: Augmented (image, label) pair.
      :rtype: tuple



   .. py:method:: __getitem__(idx)

      Retrieve a sample by index, optionally applying augmentation and preprocessing.

      Loads the image and label, normalizes intensity if specified, applies augmentation,
      and resizes to the target shape.

      :param idx: Index of the sample to retrieve.
      :type idx: int

      :returns:     - image (np.ndarray): Preprocessed image, shape (target_size, target_size), dtype float32.
                    - label (np.ndarray): Resized label mask, shape (target_size, target_size), dtype uint8.
      :rtype: tuple



.. py:function:: train_cellpose(settings)

   Train a Cellpose model on custom images and masks using specified settings.

   This function prepares training data from `train/images` and `train/masks` subfolders within
   the provided `settings['src']` directory. It constructs a model name based on training parameters,
   initializes the Cellpose model, and trains it using the specified number of epochs and hyperparameters.

   The dataset can be augmented up to 8-fold, and training images and masks are matched by filename.
   Training progress is visualized (if possible), and the model is saved in `models/cellpose_model`.

   :param settings: Dictionary with the following required keys:
                    - 'src' (str): Root directory containing `train/images`, `train/masks`, and `models`.
                    - 'target_size' (int): Side length to which images/masks are resized.
                    - 'model_name' (str): Base name of the model.
                    - 'n_epochs' (int): Number of training epochs.
                    - 'batch_size' (int): Number of images to train per batch.
                    - 'learning_rate' (float): Learning rate for training.
                    - 'weight_decay' (float): Weight decay (L2 regularization).
                    - 'augment' (bool): Whether to use 8-fold data augmentation.
   :type settings: dict

   Side Effects:
       - Trains a model and saves it to `models/cellpose_model/`.
       - Writes training settings using `save_settings()`.
       - Optionally visualizes a training batch using `plot_cellpose_batch()`.


.. py:function:: test_cellpose_model(settings)

   Evaluate a pretrained Cellpose model on a test dataset and report segmentation performance.

   This function loads test images and ground-truth masks from a specified directory structure,
   applies the Cellpose model to predict masks, and compares them to the ground truth using
   object-level metrics and Jaccard index. Results are saved and visualized if specified.

   :param settings: Dictionary with the following keys:
                    - 'src' (str): Root directory containing `test/images` and `test/masks`.
                    - 'model_path' (str): Path to the pretrained Cellpose model.
                    - 'batch_size' (int): Number of images per inference batch.
                    - 'FT' (float): Flow threshold for Cellpose segmentation.
                    - 'CP_probability' (float): Cell probability threshold for segmentation.
                    - 'target_size' (int): Size to which input images are resized.
                    - 'save' (bool): Whether to save segmentation visualizations and results.
   :type settings: dict

   Side Effects:
       - Loads and processes image and mask files from `test/images` and `test/masks`.
       - Evaluates segmentation and computes the following metrics per image:
           * Jaccard index
           * Number of predicted and ground-truth objects
           * Mean object area (true and predicted)
           * True positives, false positives, false negatives
           * Precision, recall, F1-score, accuracy
       - Saves figures showing input, true mask, predicted mask, probability map, and flow.
       - Optionally saves results to `results/test_results.csv`.
       - Prints performance summary to stdout.


.. py:function:: apply_cellpose_model(settings)

   Apply a pretrained Cellpose model to a folder of images and extract object-level measurements.

   This function processes all `.tif` images in the specified source directory using a pretrained
   Cellpose model, generates segmentation masks, and computes region properties (e.g., area) for each
   detected object. It optionally applies circular masking to limit the field of view and saves results
   and figures if enabled.

   :param settings: Dictionary with the following required keys:
                    - 'src' (str): Directory containing the input `.tif` images.
                    - 'model_path' (str): Path to the pretrained Cellpose model file.
                    - 'batch_size' (int): Number of images processed per batch.
                    - 'FT' (float): Flow threshold for Cellpose segmentation.
                    - 'CP_probability' (float): Cell probability threshold for segmentation.
                    - 'target_size' (int): Resize target for input images.
                    - 'save' (bool): If True, saves visualizations and CSV results.
                    - 'circularize' (bool, optional): If True, applies a circular mask to each prediction.
   :type settings: dict

   Side Effects:
       - Generates and saves segmented mask visualizations (if `save=True`) to `results/`.
       - Extracts per-object measurements (area) and saves:
           * `results/measurements.csv`: one row per object.
           * `results/summary.csv`: average object area and count per image.
       - Displays progress and timing for each batch.


.. py:function:: plot_cellpose_batch(images, labels)

   Display a batch of input images and their corresponding label masks.

   This function plots two rows of subplots:
       - Top row: grayscale input images.
       - Bottom row: corresponding label masks with randomly colored regions.

   :param images: List of 2D grayscale input images.
   :type images: list of np.ndarray
   :param labels: List of 2D label masks corresponding to the images.
   :type labels: list of np.ndarray


.. py:function:: analyze_percent_positive(settings)

   Analyze the fraction of objects above a threshold per well and annotate them accordingly.

   This function loads merged object-level measurements from a SQLite database, applies optional
   filtering, and annotates each object as 'above' or 'below' a specified threshold on a value column.
   It then summarizes the number and fraction of above-threshold objects per condition and well.
   Results are merged with well metadata and saved as a CSV.

   :param settings: Dictionary with the following required keys:
                    - 'src' (str): Root directory containing `measurements.db` and `rename_log.csv`.
                    - 'tables' (list of str): Table names to extract from the SQLite database.
                    - 'value_col' (str): Column to apply the threshold to.
                    - 'threshold' (float): Threshold value to classify objects.
                    - 'filter_1' (tuple or None): Optional filter in the form (column, min_value), or None.
   :type settings: dict

   :returns: Merged DataFrame with annotation summary and metadata, also saved to `result.csv`.
   :rtype: pd.DataFrame

   Side Effects:
       - Reads measurement data from `measurements/measurements.db`.
       - Reads well metadata from `rename_log.csv`.
       - Writes annotated results to `result.csv` in the same directory.
       - Displays the final DataFrame in notebook or console.


.. py:function:: analyze_recruitment(settings)

   Analyze host protein recruitment to the pathogen vacuole across experimental conditions.

   This function loads object-level measurements, applies size and intensity-based filtering, computes
   recruitment scores as the ratio of pathogen-to-cytoplasm intensities, and aggregates results at the
   cell and well level. Recruitment is annotated and visualized, and filtered results are saved.

   :param settings: Dictionary with the following required keys:
                    - 'src' (str): Path to the root folder or `measurements.db` file.
                    - 'cell_types', 'pathogen_types', 'treatments' (list of str): Experimental conditions to annotate.
                    - 'cell_plate_metadata', 'pathogen_plate_metadata', 'treatment_plate_metadata' (str): Metadata sources.
                    - 'channel_of_interest' (int): Channel index for recruitment quantification.
                    - 'channel_dims' (list of int): Channels to calculate recruitment for.
                    - 'cell_chann_dim', 'nucleus_chann_dim', 'pathogen_chann_dim' (int or None): Mask channel assignments.
                    - 'cell_size_range', 'nucleus_size_range', 'pathogen_size_range' (list): Min/max object size filters.
                    - 'cell_intensity_range', etc. (list): Intensity-based object filters.
                    - 'target_intensity_min' (float): Minimum required target channel intensity in cell.
                    - 'nuclei_limit', 'pathogen_limit' (int or None): Max number of nuclei/pathogens to load.
                    - 'cells_per_well' (int): Minimum number of cells per well to retain.
                    - 'plot' (bool): Whether to plot cell outlines on merged images.
                    - 'plot_control' (bool): Whether to plot control comparisons.
                    - 'plot_nr' (int): Maximum number of outlines to plot.
                    - 'figuresize' (float): Size of recruitment plots in inches.
   :type settings: dict

   :returns: [cell-level DataFrame, well-level DataFrame] after filtering and recruitment analysis.
   :rtype: list

   Side Effects:
       - Reads from and writes to files in the source directory.
       - Plots recruitment distributions and controls.
       - Saves `cell_level_results.csv` and `well_level_results.csv` via `_results_to_csv`.
       - Displays status messages and summary statistics.


.. py:function:: analyze_plaques(settings)

   Analyze plaque-like structures from microscopy images using a pretrained Cellpose model.

   This function applies a custom-trained Cellpose model to identify and segment plaques in `.tif` images.
   It calculates object-level statistics including object count and area, and saves the results into a
   SQLite database with three tables: `summary`, `details`, and `stats`.

   :param settings: Dictionary with the following keys:
                    - 'src' (str): Source folder containing input `.tif` images.
                    - 'masks' (bool): If True, (re)generate segmentation masks using Cellpose.
                    - 'custom_model' (str, optional): Will be set internally to the plaque model path.
                    - Any keys required by `get_analyze_plaque_settings()` or `identify_masks_finetune()`.
   :type settings: dict

   Side Effects:
       - Downloads pretrained plaque segmentation model if not available.
       - Saves generated masks (if `masks=True`) to `src/masks/`.
       - Analyzes all `.tif` masks in the folder.
       - Creates and saves the following tables to `plaques_analysis.db`:
           * `summary`: file name, number of plaques, average size
           * `details`: file name, individual plaque sizes
           * `stats`: file name, plaque count, average size, and standard deviation
       - Prints progress and completion messages.


.. py:function:: count_phenotypes(settings)

   Count and summarize annotated phenotypes in a measurement database.

   This function reads annotation data from the `measurements.db` database and performs the following:
     - Counts the number of unique phenotype labels in the specified annotation column.
     - Computes how many unique labels occur per well (defined by plateID, rowID, columnID).
     - Computes the count of each phenotype value per well.
     - Outputs a CSV file with the per-well phenotype counts.

   :param settings: Dictionary with the following keys:
                    - 'src' (str): Path to a measurement database or directory containing it.
                    - 'annotation_column' (str): Column name containing the phenotype annotations.
   :type settings: dict

   :returns: None

   Side Effects:
       - Displays number of unique phenotype values.
       - Displays a table of unique value counts per well.
       - Saves a CSV file `phenotype_counts.csv` summarizing value counts per well
         (one row per well, one column per phenotype).


.. py:function:: compare_reads_to_scores(reads_csv, scores_csv, empirical_dict={'r1': (90, 10), 'r2': (90, 10), 'r3': (80, 20), 'r4': (80, 20), 'r5': (70, 30), 'r6': (70, 30), 'r7': (60, 40), 'r8': (60, 40), 'r9': (50, 50), 'r10': (50, 50), 'r11': (40, 60), 'r12': (40, 60), 'r13': (30, 70), 'r14': (30, 70), 'r15': (20, 80), 'r16': (20, 80)}, pc_grna='TGGT1_220950_1', nc_grna='TGGT1_233460_4', y_columns=['class_1_fraction', 'TGGT1_220950_1_fraction', 'nc_fraction'], column='columnID', value='c3', plate=None, save_paths=None)

   Compare Cellpose-based phenotypic classification scores with sequencing read distributions
   from positive and negative control gRNAs across plate wells.

   This function merges phenotype classification scores and read count data, calculates gRNA
   fractions and class predictions per well, and overlays empirical expectations for controls
   to assess model calibration. It generates and saves line plots for selected y-axis columns.

   :param reads_csv: Path(s) to CSV file(s) containing read counts and gRNA names.
   :type reads_csv: str or list of str
   :param scores_csv: Path(s) to CSV file(s) with classification results per object.
   :type scores_csv: str or list of str
   :param empirical_dict: Mapping of rowID to (positive, negative) control counts.
   :type empirical_dict: dict
   :param pc_grna: Name of the positive control gRNA.
   :type pc_grna: str
   :param nc_grna: Name of the negative control gRNA.
   :type nc_grna: str
   :param y_columns: Columns in the merged dataframe to plot on y-axis.
   :type y_columns: list of str
   :param column: Column to subset on (e.g., 'columnID').
   :type column: str
   :param value: Value to filter the column on (e.g., 'c3').
   :type value: str
   :param plate: Plate name to assign if not present in the CSVs.
   :type plate: str, optional
   :param save_paths: List of two paths to save the plots as PDF.
   :type save_paths: list of str, optional

   :returns: Two matplotlib Figure objects, one for pc_fraction and one for nc_fraction x-axis plots.
   :rtype: list

   Side Effects:
       - Reads and merges input CSVs.
       - Computes gRNA read fractions and classification fractions per well.
       - Merges empirical expectations based on rowID.
       - Displays merged DataFrame.
       - Saves two line plots (if `save_paths` is specified).


.. py:function:: interperate_vision_model(settings={})

   Interpret a vision-based machine learning model using multiple feature attribution strategies.

   This function reads merged object-level data and model scores, engineers relative features, and applies:
     - Random Forest feature importance
     - Permutation importance
     - SHAP (SHapley Additive exPlanations) values

   The model interpretation is performed across cellular compartments and imaging channels, and
   results can be visualized or saved for downstream analysis.

   :param settings: Dictionary of configuration parameters including:
                    - 'src' (str): Path to the data directory.
                    - 'scores' (str): Path to the score CSV file.
                    - 'tables' (list of str): Feature group keywords for compartmental grouping.
                    - 'channels' (list of str): Feature group keywords for channel grouping.
                    - 'score_column' (str): Column name for predicted class scores.
                    - 'nuclei_limit' (int): Optional filter on number of nuclei to load.
                    - 'pathogen_limit' (int): Optional filter on number of pathogens to load.
                    - 'feature_importance' (bool): Whether to compute Random Forest feature importances.
                    - 'permutation_importance' (bool): Whether to compute permutation-based importances.
                    - 'shap' (bool): Whether to compute SHAP values.
                    - 'shap_sample' (bool): Whether to subsample SHAP input for speed.
                    - 'top_features' (int): Number of top features to plot.
                    - 'include_all' (bool): Whether to include a row summing all feature groups.
                    - 'n_jobs' (int): Number of parallel jobs to use for model training or SHAP.
                    - 'save' (bool): Whether to save output CSV files.
   :type settings: dict

   :returns:

             A dictionary of interpretation outputs with the following possible keys:
                 - 'feature_importance': Raw feature importances (Random Forest)
                 - 'feature_importance_compartment': Grouped by compartment
                 - 'feature_importance_channel': Grouped by imaging channel
                 - 'permutation_importance': Permutation importance values
                 - 'shap': SHAP values for each object-feature pair
   :rtype: dict

   Side Effects:
       - Displays importance plots.
       - Optionally saves CSVs to `<src>/results/`.
       - Merges and preprocesses input data, including relative feature construction.


.. py:function:: analyze_endodyogeny(settings)

   Analyze intracellular pathogen replication (endodyogeny) based on compartment volume measurements.

   This function bins single-cell or single-object volume estimates (e.g., pathogen area^1.5) into
   discrete categories, annotates experimental conditions, and visualizes the volume distribution
   across groups as a stacked bar plot. It also performs a chi-squared test to assess whether the
   volume distribution is significantly different between groups.

   :param settings: Configuration dictionary with required keys:
                    - 'src' (str or list): Path(s) to directories containing measurements.db.
                    - 'tables' (list of str): Tables to extract from the SQLite database.
                    - 'compartment' (str): Compartment to analyze (e.g., 'pathogen').
                    - 'min_area_bin' (float): Minimum 2D area threshold to consider (in px² or µm²).
                    - 'max_bins' (int or None): Maximum number of volume bins to display.
                    - 'group_column' (str): Column defining experimental groups (e.g., genotype or treatment).
                    - 'cell_types', 'pathogen_types', 'treatments' (list): Experimental groups for annotation.
                    - 'cell_plate_metadata', 'pathogen_plate_metadata', 'treatment_plate_metadata' (str): Metadata source paths.
                    - 'nuclei_limit', 'pathogen_limit' (int or None): Max number of objects to load per type.
                    - 'change_plate' (bool): Whether to modify plate names for uniqueness.
                    - 'verbose' (bool): Whether to print debug output.
                    - 'um_per_px' (float or None): Pixel-to-micrometer conversion (if None, volume remains in px³).
                    - 'level' (str): Either 'object' or 'well' for aggregation level.
                    - 'cmap' (str): Colormap to use in the plot.
                    - 'save' (bool): Whether to save results and figures.
   :type settings: dict

   :returns:

             A dictionary with:
                 - 'data': Annotated and binned DataFrame.
                 - 'chi_squared': Chi-squared test summary DataFrame.
   :rtype: dict

   Side Effects:
       - Displays a stacked bar plot showing volume bin proportions by group.
       - Performs chi-squared test and prints p-value.
       - Saves data and plots to `results/analyze_endodyogeny/` if `save=True`.


.. py:function:: analyze_class_proportion(settings)

   Analyze class frequency distributions across experimental groups and perform statistical tests.

   This function compares the proportion of classification outcomes (e.g., phenotypic classes)
   across specified experimental groups. It generates stacked bar plots, performs chi-squared
   and parametric/non-parametric statistical tests, and saves annotated data and results.

   :param settings: Dictionary of parameters, including:
                    - 'src' (str or list): Path(s) to measurement directories.
                    - 'tables' (list): Tables to load from measurements.db.
                    - 'class_column' (str): Column indicating class or category.
                    - 'group_column' (str): Column indicating grouping variable (e.g., genotype or treatment).
                    - 'cell_types', 'pathogen_types', 'treatments' (list): Experimental group values.
                    - 'cell_plate_metadata', 'pathogen_plate_metadata', 'treatment_plate_metadata' (str): Metadata paths.
                    - 'nuclei_limit', 'pathogen_limit' (int or None): Optional load limits.
                    - 'verbose' (bool): Whether to print debug information.
                    - 'level' (str): Aggregation level for chi-squared test ('object' or 'well').
                    - 'save' (bool): Whether to save results and figures.
   :type settings: dict

   :returns:

             A dictionary containing:
                 - 'data': Annotated DataFrame.
                 - 'chi_squared': Chi-squared summary results.
   :rtype: dict

   Side Effects:
       - Annotates DataFrame with experimental conditions.
       - Performs chi-squared test on class distribution and plots stacked bars.
       - Plots class heatmap across plates.
       - Performs and prints:
           * Shapiro-Wilk normality test
           * Levene’s test for equal variances
           * ANOVA or Kruskal-Wallis test
           * Post hoc pairwise comparisons
       - Optionally saves all results and figures in a `results/analyze_class_proportion/` subfolder.


.. py:function:: generate_score_heatmap(settings)

   Generate a heatmap comparing predicted classification scores from multiple models
   with empirical gRNA-based fraction measurements across wells.

   This function:
     - Aggregates prediction scores from subfolders.
     - Calculates the gRNA-based empirical fraction from sequencing data.
     - Merges predictions with empirical scores and computes MAE (mean absolute error).
     - Displays and optionally saves a multi-channel heatmap and the corresponding MAE results.

   :param settings: Configuration dictionary with the following keys:
                    - 'folders' (list or str): Folder(s) containing model output subfolders with CSVs.
                    - 'csv_name' (str): Name of the CSV file to search for in each model output subfolder.
                    - 'data_column' (str): Column in CSVs containing the prediction scores.
                    - 'cv_csv' (str): CSV with cross-validated model scores.
                    - 'data_column_cv' (str): Column in `cv_csv` with scores to merge.
                    - 'csv' (str): CSV with gRNA counts used for computing empirical fractions.
                    - 'control_sgrnas' (list): List of two control gRNA names used to compute the fraction.
                    - 'fraction_grna' (str): One of the control gRNAs to extract the empirical fraction from.
                    - 'plateID' (int or str): Plate ID used for labeling and filtering.
                    - 'columnID' (str): Column ID to filter on (e.g., 'c3').
                    - 'cmap' (str): Colormap to use for the heatmap.
                    - 'dst' (str or None): If provided, save outputs to this directory.
   :type settings: dict

   :returns: Merged DataFrame containing all prediction and empirical score columns.
   :rtype: pd.DataFrame

   Side Effects:
       - Displays a multi-channel heatmap of prediction scores.
       - Computes and prints MAE per well per channel.
       - Saves merged data, MAE scores, and the heatmap as PDF/CSV if `dst` is set.


.. py:function:: post_regression_analysis(csv_file, grna_dict, grna_list, save=False)

   Perform post-regression analysis to assess gRNA correlation and infer relative effect sizes.

   This function loads a CSV file containing gRNA abundance data (with columns including
   'grna', 'fraction', and 'prc'), filters it to the specified gRNAs of interest, computes
   a correlation matrix, visualizes it, and estimates inferred effect sizes for all gRNAs
   based on a subset of known fixed effect sizes.

   :param csv_file: Path to the CSV file containing gRNA data with columns ['grna', 'fraction', 'prc'].
   :type csv_file: str
   :param grna_dict: Dictionary mapping selected gRNAs to their known effect sizes.
   :type grna_dict: dict
   :param grna_list: List of gRNAs to include in correlation and effect size inference.
   :type grna_list: list
   :param save: Whether to save outputs including figures and CSVs. Defaults to False.
   :type save: bool, optional

   :returns: None. Saves plots and results if `save=True`.


