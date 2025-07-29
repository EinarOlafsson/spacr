spacr.settings
==============

.. py:module:: spacr.settings






Module Contents
---------------

.. py:function:: set_default_plot_merge_settings()

   Set default configuration values for plotting merged image masks and overlays.

   :returns: Dictionary of default settings for plotting merged masks and overlays.
   :rtype: dict

   Default Parameters:
       - pathogen_limit (int): Maximum number of pathogen objects to display (default: 10).
       - nuclei_limit (int): Maximum number of nuclei objects to display (default: 1).
       - remove_background (bool): Whether to subtract background intensity (default: False).
       - filter_min_max (tuple|None): Optional min/max filter for object values (default: None).
       - channel_dims (list): Indices for raw image channels (default: [0,1,2,3]).
       - backgrounds (list): Background values for each channel (default: [100,100,100,100]).
       - cell_mask_dim (int): Index of cell mask channel (default: 4).
       - nucleus_mask_dim (int): Index of nucleus mask channel (default: 5).
       - pathogen_mask_dim (int): Index of pathogen mask channel (default: 6).
       - outline_thickness (int): Thickness of outline for each object (default: 3).
       - outline_color (str): Outline color in RGB channel string (default: 'gbr').
       - overlay_chans (list): Channels to overlay on visualization (default: [1,2,3]).
       - overlay (bool): Whether to show overlays (default: True).
       - normalization_percentiles (list): Percentiles for image normalization (default: [2,98]).
       - normalize (bool): Whether to normalize the image (default: True).
       - print_object_number (bool): Print number of detected objects (default: True).
       - nr (int): Number of examples to show (default: 1).
       - figuresize (int): Size of output figure (default: 10).
       - cmap (str): Color map to use (default: 'inferno').
       - verbose (bool): Print debug info (default: True).


.. py:function:: set_default_settings_preprocess_generate_masks(settings={})

   Set default settings for the preprocessing and mask generation pipeline using Cellpose or similar tools.

   This function populates a settings dictionary with default values used during image preprocessing,
   segmentation mask generation, background removal, plotting, timelapse handling, and more.

   :param settings: Optional dictionary with user-specified values. Defaults will only fill in missing keys.
   :type settings: dict

   :returns: A dictionary containing default configuration parameters for mask generation.
   :rtype: dict

   Default Parameters:
       # Core pipeline toggles
       - denoise (bool): Apply denoising to input images.
       - src (str): Path to image source directory.
       - delete_intermediate (bool): Delete intermediate files after processing.
       - preprocess (bool): Whether to run preprocessing steps.
       - masks (bool): Whether to generate segmentation masks.
       - save (bool): Save outputs to disk.
       - consolidate (bool): Combine mask outputs from multiple sources.
       - batch_size (int): Number of images to process in parallel.
       - test_mode (bool): Run in test mode for quick inspection.
       - test_images (int): Number of images to use in test mode.
       - magnification (int): Objective magnification used for imaging.
       - custom_regex (str|None): Custom regex for file parsing if needed.
       - metadata_type (str): Format of metadata ('cellvoyager', etc.).
       - n_jobs (int): Number of parallel jobs to use (default: CPU count - 4).
       - randomize (bool): Shuffle processing order.
       - verbose (bool): Print processing details.
       - remove_background_{cell,nucleus,pathogen} (bool): Background subtraction toggles per channel.

       # Object diameter estimates
       - {cell,nucleus,pathogen}_diamiter (float|None): Expected diameter for each object type.

       # Channel and background settings
       - channels (list): List of image channel indices.
       - {cell,nucleus,pathogen}_channel (int|None): Channel indices for each object type.
       - {cell,nucleus,pathogen}_background (int): Background value for each channel.
       - {cell,nucleus,pathogen}_Signal_to_noise (float): Threshold for SNR filtering.
       - {cell,nucleus,pathogen}_CP_prob (float): Minimum Cellpose probability threshold.
       - {cell,nucleus,pathogen}_FT (float): Feature threshold score.

       # Plotting
       - plot (bool): Enable visual output.
       - figuresize (int): Plot size in inches.
       - cmap (str): Color map for plots.
       - normalize (bool): Normalize intensities for plotting.
       - normalize_plots (bool): Normalize before visualizing masks.
       - examples_to_plot (int): Number of plots to generate.

       # Analysis settings
       - pathogen_model (str|None): Optional path to pathogen model.
       - merge_pathogens (bool): Merge all pathogen masks into one.
       - filter (bool): Enable object filtering.
       - lower_percentile (float): Intensity clipping lower bound.

       # Timelapse tracking
       - timelapse (bool): Enable timelapse mode.
       - fps (int): Frames per second for exported timelapse.
       - timelapse_displacement (float|None): Displacement threshold for linking.
       - timelapse_memory (int): Maximum number of frames to retain object identity.
       - timelapse_frame_limits (list): List of frames to keep.
       - timelapse_remove_transient (bool): Remove transient objects.
       - timelapse_mode (str): Tracking algorithm ('trackpy', etc.).
       - timelapse_objects (str|None): Object types to track.

       # Miscellaneous
       - all_to_mip (bool): Convert all frames to maximum intensity projection.
       - upscale (bool): Upsample image resolution.
       - upscale_factor (float): Upsampling multiplier.
       - adjust_cells (bool): Morphologically adjust cell boundaries.
       - use_sam_{cell,nucleus,pathogen} (bool): Use Segment Anything Model (SAM) for segmentation.


.. py:function:: set_default_plot_data_from_db(settings)

   Set default plotting settings for visualizing data from an SQL database.

   :param settings: Settings dictionary to populate.
   :type settings: dict

   :returns: Dictionary with default keys and values for graph plotting from database data.
   :rtype: dict

   Default Parameters:
       - src (str): Path to the database directory (default: 'path').
       - database (str): Filename of SQLite database (default: 'measurements.db').
       - graph_name (str): Output name of the plot (default: 'Figure_1').
       - table_names (list): Tables to include from the database (default: ['cell', 'cytoplasm', 'nucleus', 'pathogen']).
       - data_column (str): Column to plot (default: 'recruitment').
       - grouping_column (str): Column used to group data (default: 'condition').
       - cell_types (list): Cell types to include (default: ['Hela']).
       - cell_plate_metadata, pathogen_plate_metadata, treatment_plate_metadata (None): Optional metadata dictionaries.
       - pathogen_types, treatments (None): Optional lists of types.
       - graph_type (str): Type of plot (default: 'jitter').
       - theme (str): Seaborn theme (default: 'deep').
       - save (bool): Save plot to disk (default: True).
       - y_lim (list): y-axis limits (default: [1, 1.5]).
       - verbose (bool): Verbose output (default: False).
       - channel_of_interest (int): Image channel to analyze (default: 1).
       - nuclei_limit (int): Max nuclei per field (default: 2).
       - pathogen_limit (int): Max pathogens per field (default: 3).
       - representation (str): Plot representation ('well' or 'cell', default: 'well').
       - uninfected (bool): Include uninfected controls (default: False).


.. py:function:: set_default_settings_preprocess_img_data(settings)

   Set default values for preprocessing image data before analysis.

   :param settings: Settings dictionary to populate.
   :type settings: dict

   :returns: Updated dictionary with image preprocessing settings.
   :rtype: dict

   Default Parameters:
       - metadata_type (str): Metadata parsing type (default: 'cellvoyager').
       - custom_regex (str|None): Regex for file parsing (default: None).
       - nr (int): Number of examples to plot (default: 1).
       - plot (bool): Whether to show plots (default: True).
       - batch_size (int): Number of images to process per batch (default: 50).
       - timelapse (bool): Whether this is a time series dataset (default: False).
       - lower_percentile (int): Lower percentile for intensity clipping (default: 2).
       - randomize (bool): Shuffle input file order (default: True).
       - all_to_mip (bool): Convert all frames to max-intensity projection (default: False).
       - cmap (str): Color map for images (default: 'inferno').
       - figuresize (int): Figure size for plots (default: 10).
       - normalize (bool): Normalize image intensities (default: True).
       - save_dtype (str): Data type for saving processed files (default: 'uint16').
       - test_mode (bool): Whether to run in test mode (default: False).
       - test_images (int): Number of test images to run in test mode (default: 10).
       - random_test (bool): Randomly select test images (default: True).
       - fps (int): Frames per second for timelapse visualization (default: 2).


.. py:function:: set_default_umap_image_settings(settings={})

   Set default configuration values for UMAP-based image embedding and clustering.

   :param settings: Optional dictionary of user-provided settings. Keys that are not present
                    will be set to their default values.
   :type settings: dict

   :returns: Updated settings dictionary containing all necessary UMAP image analysis parameters.
   :rtype: dict

   Default Parameters Set:
       - src (str): Path to input directory (default: 'path').
       - row_limit (int): Maximum number of rows to use (default: 1000).
       - tables (list): List of object types to include (default: ['cell', 'cytoplasm', 'nucleus', 'pathogen']).
       - visualize (str): Object type to visualize (default: 'cell').
       - image_nr (int): Number of example images to display (default: 16).
       - dot_size (int): Dot size in the scatter plot (default: 50).
       - n_neighbors (int): UMAP parameter for local neighborhood size (default: 1000).
       - min_dist (float): UMAP parameter controlling embedding compactness (default: 0.1).
       - metric (str): Distance metric used in UMAP (default: 'euclidean').
       - eps (float): DBSCAN epsilon parameter (default: 0.9).
       - min_samples (int): Minimum number of samples per cluster in DBSCAN (default: 100).
       - filter_by (str): Column used to filter features (default: 'channel_0').
       - img_zoom (float): Zoom level for image thumbnails (default: 0.5).
       - plot_by_cluster (bool): Whether to color plot by cluster ID (default: True).
       - plot_cluster_grids (bool): Whether to plot grid of cluster example images (default: True).
       - remove_cluster_noise (bool): Remove outliers/noise clusters (default: True).
       - remove_highly_correlated (bool): Remove highly correlated features (default: True).
       - log_data (bool): Log-transform input features (default: False).
       - figuresize (int): Size of output figure (default: 10).
       - black_background (bool): Whether to use a black background in plots (default: True).
       - remove_image_canvas (bool): Crop out canvas margins in image plots (default: False).
       - plot_outlines (bool): Overlay object outlines on image thumbnails (default: True).
       - plot_points (bool): Plot UMAP/TSNE scatter points (default: True).
       - smooth_lines (bool): Use smoothed lines in plots (default: True).
       - clustering (str): Clustering method, e.g. 'dbscan' (default: 'dbscan').
       - exclude (list|None): List of object classes or conditions to exclude (default: None).
       - col_to_compare (str): Column used to compare conditions (default: 'columnID').
       - pos (str): Label for positive control column (default: 'c1').
       - neg (str): Label for negative control column (default: 'c2').
       - mix (str): Label for mixed/experimental column (default: 'c3').
       - embedding_by_controls (bool): Fit UMAP only on control samples (default: False).
       - plot_images (bool): Whether to include image overlays in plot (default: True).
       - reduction_method (str): Dimensionality reduction method (default: 'umap').
       - save_figure (bool): Save output figure to disk (default: False).
       - n_jobs (int): Number of parallel jobs to use (default: -1, i.e., all cores).
       - color_by (str|None): Column name to use for color-coding scatter points (default: None).
       - exclude_conditions (list|None): List of conditions to exclude from embedding (default: None).
       - analyze_clusters (bool): Perform further statistical analysis on clusters (default: False).
       - resnet_features (bool): Use pretrained ResNet features (default: False).
       - verbose (bool): Print status messages (default: True).


.. py:function:: get_measure_crop_settings(settings={})

   Set default configuration for object measurement and cropping.

   This function initializes and returns a dictionary of settings used for
   measuring and cropping segmented objects such as cells, nuclei, or pathogens.
   It covers measurement parameters, image cropping/export options,
   multiprocessing behavior, and test mode overrides.

   :param settings: Existing settings dictionary to be updated with defaults.
   :type settings: dict, optional

   :returns: Fully populated settings dictionary with default values applied.
   :rtype: dict

   Key Settings:
       - Measurement:
           'save_measurements' (bool): Whether to save measurement results.
           'radial_dist' (bool): Compute radial distance profiles.
           'calculate_correlation' (bool): Compute intensity correlations.
           'manders_thresholds' (list): Thresholds (percentiles) for Manders overlap coefficient.
           'homogeneity' (bool): Compute local homogeneity.
           'homogeneity_distances' (list): Distances (in pixels) for homogeneity calculations.

       - Cropping:
           'save_png' (bool): Export cropped objects as PNGs.
           'save_arrays' (bool): Save raw data arrays for cropped objects.
           'png_size' (list): Output size of cropped PNGs [width, height].
           'png_dims' (list): Channel indices to include in the output PNG.
           'normalize' (bool): Apply intensity normalization.
           'normalize_by' (str): Normalize based on object-level or global stats.
           'crop_mode' (list): Objects to crop, e.g., ['cell'].
           'use_bounding_box' (bool): Use tight bounding boxes instead of masks.
           'dialate_pngs' (bool): Apply dilation to the object mask.
           'dialate_png_ratios' (list): Dilation factors relative to object size.

       - Timelapse:
           'timelapse' (bool): Process timelapse series.
           'timelapse_objects' (list): Objects to track over time.

       - Miscellaneous:
           'src' (str): Input directory path.
           'experiment' (str): Name of the experiment.
           'test_mode' (bool): Run in debug mode with fewer images and visual output.
           'test_nr' (int): Number of test images to process.
           'plot' (bool): Show debug plots.
           'n_jobs' (int): Number of CPU threads to use.
           'verbose' (bool): Enable verbose output.

       - Object masks:
           'cell_mask_dim', 'nucleus_mask_dim', 'pathogen_mask_dim' (int): Channels for respective masks.
           'cytoplasm' (bool): Include cytoplasmic measurements.
           'merge_edge_pathogen_cells' (bool): Option to merge pathogens at borders.
           'min_size' (int): Minimum size for filtering objects by type.

       - Advanced:
           'distance_gaussian_sigma' (float): Smoothing factor for distance transforms.

   .. rubric:: Notes

   - When 'test_mode' is True, verbose and plot modes are automatically enabled.
   - 'os.cpu_count()' is used to allocate available cores for parallel processing.

   .. rubric:: Example

   settings = get_measure_crop_settings()


.. py:function:: set_default_analyze_screen(settings)

   Set default configuration for analyzing a CRISPR or compound screen.

   This function populates a provided settings dictionary with defaults related to
   feature extraction, model training, screen scoring, and heatmap visualization.

   :param settings: Dictionary of user-provided settings to be updated.
   :type settings: dict

   :returns: Updated settings dictionary with all necessary keys and default values.
   :rtype: dict

   Key Settings:
       - Input/Output:
           'src' (str): Path to the screen results folder.
           'save_to_db' (bool): If True, results will be saved to a database.
           'annotation_column' (str or None): Column used to group conditions or annotate classes.
           'location_column' (str): Column identifying spatial layout (e.g., 'columnID').

       - Modeling:
           'model_type_ml' (str): Machine learning model to use ('xgboost' by default).
           'learning_rate' (float): Learning rate for boosting models.
           'n_estimators' (int): Number of trees for boosting.
           'reg_alpha' (float): L1 regularization coefficient.
           'reg_lambda' (float): L2 regularization coefficient.
           'test_size' (float): Proportion of data used for testing.
           'cross_validation' (bool): Whether to perform cross-validation.
           'n_repeats' (int): Number of repetitions for performance evaluation.
           'prune_features' (bool): Whether to apply feature pruning.

       - Feature selection:
           'remove_low_variance_features' (bool): Exclude features with low variance.
           'remove_highly_correlated_features' (bool): Exclude highly collinear features.
           'top_features' (int): Number of top features to retain after training.

       - Screen summarization:
           'heatmap_feature' (str): Feature used for heatmap visualization (e.g., 'predictions').
           'grouping' (str): How to summarize replicate data ('mean', 'median', etc.).
           'min_max' (str): Scaling mode for heatmap normalization ('allq', 'robust', etc.).
           'cmap' (str): Colormap used for heatmap plotting.

       - Controls:
           'positive_control' (str): Label for the positive control condition.
           'negative_control' (str): Label for the negative control condition.
           'exclude' (list or None): List of condition labels to exclude from analysis.

       - Filtering:
           'minimum_cell_count' (int): Minimum number of cells per well required for inclusion.
           'nuclei_limit' (bool): Whether to apply a nuclei count filter.
           'pathogen_limit' (int): Maximum number of pathogens per object.

       - Miscellaneous:
           'channel_of_interest' (int): Imaging channel used for downstream focus.
           'n_jobs' (int): Number of parallel jobs to run (-1 uses all CPUs).
           'verbose' (bool): Enable verbose logging.

   .. rubric:: Example

   settings = set_default_analyze_screen({})


.. py:function:: set_default_train_test_model(settings)

   Set default configuration for training and testing a deep learning classification model.

   This function populates a provided dictionary with default settings used for model training,
   including architecture, optimizer, augmentation, and hardware preferences.

   :param settings: Dictionary of user-provided settings to be updated.
   :type settings: dict

   :returns: Updated settings dictionary with all necessary keys and default values.
   :rtype: dict

   Key Settings:
       - Input/Output:
           'src' (str): Path to dataset directory.
           'train' (bool): Whether to perform training.
           'test' (bool): Whether to run inference on test data.
           'classes' (list): List of class labels (e.g., ['nc', 'pc']).

       - Model architecture:
           'model_type' (str): Model architecture to use (e.g., 'maxvit_t').
           'init_weights' (bool): Whether to initialize model with pretrained weights.
           'dropout_rate' (float): Dropout rate applied before final layers.

       - Optimizer and scheduler:
           'optimizer_type' (str): Optimizer to use (e.g., 'adamw').
           'schedule' (str): Learning rate scheduler ('reduce_lr_on_plateau' or 'step_lr').
           'amsgrad' (bool): Use AMSGrad variant of Adam.
           'weight_decay' (float): Weight decay regularization.
           'learning_rate' (float): Initial learning rate.

       - Loss function:
           'loss_type' (str): Loss function to use (e.g., 'focal_loss', 'binary_cross_entropy_with_logits').

       - Training hyperparameters:
           'batch_size' (int): Batch size used during training.
           'epochs' (int): Number of training epochs.
           'val_split' (float): Proportion of data used for validation.
           'gradient_accumulation' (bool): Enable gradient accumulation to simulate larger batch sizes.
           'gradient_accumulation_steps' (int): Number of steps for accumulation.
           'pin_memory' (bool): Pin memory in DataLoader.

       - Image preprocessing:
           'image_size' (int): Size to which images are resized (assumes square).
           'normalize' (bool): Whether to apply normalization.
           'train_channels' (list): List of channels to use for training (e.g., ['r','g','b']).
           'augment' (bool): Whether to apply data augmentation.

       - Checkpointing:
           'use_checkpoint' (bool): Save model checkpoints.
           'intermedeate_save' (bool): Save intermediate models during training.

       - Parallelization:
           'n_jobs' (int): Number of parallel processes to use (default: available cores - 2).

       - Miscellaneous:
           'verbose' (bool): Enable detailed logging.

   .. rubric:: Example

   settings = set_default_train_test_model({})


.. py:function:: set_generate_training_dataset_defaults(settings)

   Set default configuration for generating a training dataset from measurements and metadata.

   This function populates the given dictionary with default values required for generating
   a structured dataset for supervised learning based on annotated metadata or measurements.

   :param settings: Dictionary to populate with defaults if not already present.
   :type settings: dict

   :returns: Updated settings dictionary with default keys and values for dataset generation.
   :rtype: dict

   Key Settings:
       - Input/Output:
           'src' (str): Path to source data.
           'png_type' (str): Type of image to extract (e.g., 'cell_png').

       - Metadata and labels:
           'dataset_mode' (str): Mode to derive labels ('metadata' or 'measurement').
           'annotation_column' (str): Name of column in metadata used for annotation.
           'annotated_classes' (list): List of class indices (e.g., [1, 2]).
           'class_metadata' (list): List of metadata values corresponding to each class.
           'metadata_item_1_name' (list): Primary metadata variable (e.g., conditions).
           'metadata_item_1_value' (list of list): Metadata values for each class.
           'metadata_item_2_name' (list): Secondary metadata variable.
           'metadata_item_2_value' (list of list): Values of secondary variable by class.
           'metadata_type_by' (str): Column to use for metadata grouping (e.g., 'columnID').

       - Table and image options:
           'tables' (list or None): Table(s) to source data from (e.g., 'cell', 'nucleus').
           'channel_of_interest' (int): Channel to use for class derivation or filtering.
           'custom_measurement' (str or None): Measurement key to use for class label if in measurement mode.

       - Limits and filters:
           'nuclei_limit' (bool): Whether to filter based on number of nuclei.
           'pathogen_limit' (bool): Whether to filter based on number of pathogens.

       - Image settings:
           'size' (int): Target image size (square width/height in pixels).
           'test_split' (float): Fraction of data to reserve for testing.

   .. rubric:: Example

   settings = set_generate_training_dataset_defaults({})


.. py:function:: deep_spacr_defaults(settings)

   Set default arguments for deep learning analysis in spaCR.

   This function fills in default arguments for training, testing, and applying deep learning
   models on spaCR datasets, supporting both metadata and measurement-based annotations.

   :param settings: Dictionary to populate with default arguments.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to dataset.
       dataset_mode (str): 'metadata' or 'measurement' mode for annotation.
       annotation_column (str): Metadata column containing annotations.
       annotated_classes (list): Class indices (e.g., [1, 2]).
       classes (list): Class labels (e.g., ['nc', 'pc']).
       size (int): Image size for training.
       test_split (float): Fraction of data reserved for testing.
       class_metadata (list): Metadata values for each class.
       metadata_type_by (str): Key used for metadata classification (e.g., 'columnID').
       channel_of_interest (int): Channel used for classification or filtering.
       custom_measurement (str or None): Measurement used for label derivation.
       tables (list or None): Tables to extract measurements from (e.g., 'cell').
       png_type (str): Image type used for training, e.g., 'cell_png'.
       custom_model (bool): Whether to load a custom model.
       custom_model_path (str): Path to the custom model file.
       train (bool): Whether to perform training.
       test (bool): Whether to perform testing.
       model_type (str): Model architecture (e.g., 'maxvit_t').
       optimizer_type (str): Optimizer to use (e.g., 'adamw').
       schedule (str): Learning rate schedule ('reduce_lr_on_plateau' or 'step_lr').
       loss_type (str): Loss function ('focal_loss', etc.).
       normalize (bool): Whether to normalize input images.
       image_size (int): Image dimensions (assumes square input).
       batch_size (int): Batch size for training.
       epochs (int): Number of training epochs.
       val_split (float): Validation fraction from training data.
       learning_rate (float): Initial learning rate.
       weight_decay (float): Weight decay for optimizer.
       dropout_rate (float): Dropout rate in model.
       init_weights (bool): Whether to initialize weights.
       amsgrad (bool): Use AMSGrad with Adam.
       use_checkpoint (bool): Save and load checkpoints.
       gradient_accumulation (bool): Use gradient accumulation.
       gradient_accumulation_steps (int): Steps to accumulate gradients.
       intermedeate_save (bool): Save intermediate weights.
       pin_memory (bool): Use pin memory in DataLoader.
       n_jobs (int): Number of CPU cores to use.
       train_channels (list): Color channels for training (e.g., ['r', 'g', 'b']).
       augment (bool): Use data augmentation.
       verbose (bool): Verbose output.
       apply_model_to_dataset (bool): Whether to apply model after training.
       file_metadata (str or None): Path to metadata for inference.
       sample (str or None): Sample identifier.
       experiment (str): Experiment identifier.
       score_threshold (float): Classification threshold.
       dataset (str): Path to dataset for inference.
       model_path (str): Path to saved model file.
       file_type (str): File type of images (e.g., 'cell_png').
       generate_training_dataset (bool): Whether to generate training dataset.
       train_DL_model (bool): Whether to train the deep learning model.


.. py:function:: get_train_test_model_settings(settings)

   Set default args for training and testing a deep learning model.

   This function populates the `settings` dictionary with default args used for
   training and evaluating deep learning models in spaCR.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to input dataset.
       train (bool): Whether to train the model.
       test (bool): Whether to test the model.
       custom_model (bool): Whether to load a custom model.
       classes (list): List of class labels, e.g., ['nc', 'pc'].
       train_channels (list): Channels to use for training (e.g., ['r', 'g', 'b']).
       model_type (str): Type of model architecture, e.g., 'maxvit_t'.
       optimizer_type (str): Optimizer to use, e.g., 'adamw'.
       schedule (str): Learning rate schedule ('reduce_lr_on_plateau' or 'step_lr').
       loss_type (str): Loss function, e.g., 'focal_loss'.
       normalize (bool): Normalize images before training.
       image_size (int): Input image size (square).
       batch_size (int): Batch size for training.
       epochs (int): Number of epochs.
       val_split (float): Fraction of training data used for validation.
       learning_rate (float): Initial learning rate.
       weight_decay (float): Weight decay for regularization.
       dropout_rate (float): Dropout rate for model.
       init_weights (bool): Whether to initialize model weights.
       amsgrad (bool): Use AMSGrad variant of Adam.
       use_checkpoint (bool): Whether to use model checkpointing.
       gradient_accumulation (bool): Use gradient accumulation across batches.
       gradient_accumulation_steps (int): Steps to accumulate gradients before update.
       intermedeate_save (bool): Save intermediate models during training.
       pin_memory (bool): Enable pinned memory in DataLoader.
       n_jobs (int): Number of CPU cores to use.
       augment (bool): Whether to apply data augmentation.
       verbose (bool): Enable verbose output.


.. py:function:: get_analyze_recruitment_default_settings(settings)

   Set default args for recruitment analysis of host/pathogen protein localization.

   This function populates the `settings` dictionary with default values for analyzing
   recruitment of host proteins (e.g., ESCRT) to pathogens under various treatment conditions.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to input dataset.
       target (str): Protein target being analyzed.
       cell_types (list): List of host cell types (e.g., ['HeLa']).
       cell_plate_metadata (list or None): Metadata for host cells.
       pathogen_types (list): List of pathogen types.
       pathogen_plate_metadata (list): Plate layout metadata for pathogens.
       treatments (list): List of treatment conditions.
       treatment_plate_metadata (list): Plate layout metadata for treatments.
       channel_dims (list): List of image channel indices.
       cell_chann_dim (int): Index of cell signal channel.
       cell_mask_dim (int): Index of cell mask channel.
       nucleus_chann_dim (int): Index of nucleus signal channel.
       nucleus_mask_dim (int): Index of nucleus mask channel.
       pathogen_chann_dim (int): Index of pathogen signal channel.
       pathogen_mask_dim (int): Index of pathogen mask channel.
       channel_of_interest (int): Channel to analyze for recruitment.
       plot (bool): Whether to generate plots.
       plot_nr (int): Number of plots to generate.
       plot_control (bool): Include controls in plots.
       figuresize (int): Size of generated figures.
       pathogen_limit (int): Maximum pathogens per field.
       nuclei_limit (int): Minimum nuclei required per field.
       cells_per_well (int): Expected number of cells per well.
       pathogen_size_range (list): Min/max area for pathogen inclusion.
       nucleus_size_range (list): Min/max area for nucleus inclusion.
       cell_size_range (list): Min/max area for cell inclusion.
       pathogen_intensity_range (list): Min/max intensity for pathogen inclusion.
       nucleus_intensity_range (list): Min/max intensity for nucleus inclusion.
       cell_intensity_range (list): Min/max intensity for cell inclusion.
       target_intensity_min (int): Minimum intensity for target signal.


.. py:function:: get_default_test_cellpose_model_settings(settings)

   Set default args for testing a Cellpose model.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to input image directory.
       model_path (str): Path to the Cellpose model.
       save (bool): Whether to save outputs.
       normalize (bool): Apply percentile normalization.
       percentiles (tuple): Lower and upper percentiles for normalization.
       batch_size (int): Number of images to process per batch.
       CP_probability (float): Minimum probability for Cellpose mask inclusion.
       FT (float): Flow threshold for Cellpose model.
       target_size (int): Resize input images to this size for model inference.


.. py:function:: get_default_apply_cellpose_model_settings(settings)

   Set default args for applying a trained Cellpose model.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to input image directory.
       model_path (str): Path to the Cellpose model.
       save (bool): Whether to save outputs.
       normalize (bool): Apply percentile normalization.
       percentiles (tuple): Lower and upper percentiles for normalization.
       batch_size (int): Number of images to process per batch.
       CP_probability (float): Minimum probability for Cellpose mask inclusion.
       FT (float): Flow threshold for Cellpose model.
       circularize (bool): Convert masks to circular contours.
       target_size (int): Resize input images to this size for model inference.


.. py:function:: default_settings_analyze_percent_positive(settings)

   Set default args for analyzing percent-positive cells based on intensity thresholding.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to data table.
       tables (list): List of object types to analyze (e.g., ['cell']).
       filter_1 (list): Filtering rule, e.g., ['cell_area', 1000].
       value_col (str): Column name containing intensity values.
       threshold (float): Threshold above which objects are considered positive.


.. py:function:: get_analyze_reads_default_settings(settings)

   Set default args for analyzing NGS reads to extract barcodes.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to FASTQ file or folder.
       upstream (str): Sequence upstream of the barcode.
       downstream (str): Sequence downstream of the barcode.
       barecode_length_1 (int): Length of the first barcode.
       barecode_length_2 (int): Length of the second barcode.
       chunk_size (int): Number of reads to process per chunk.
       test (bool): Enable test mode with limited read parsing.


.. py:function:: get_map_barcodes_default_settings(settings)

   Set default args for mapping extracted barcodes to plate, grna, and control metadata.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to barcode read count data.
       grna (str): Path to gRNA-barcode mapping CSV.
       barcodes (str): Path to screen barcode-to-well metadata CSV.
       plate_dict (str): Stringified dictionary mapping plate IDs to logical names.
       test (bool): Enable test mode with limited entries.
       verbose (bool): Print detailed progress information.
       pc (str): Positive control gRNA identifier.
       pc_loc (str): Well location of the positive control.
       nc (str): Negative control gRNA identifier.
       nc_loc (str): Well location of the negative control.


.. py:function:: get_train_cellpose_default_settings(settings)

   Set default args for training a Cellpose model.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       model_name (str): Name to assign to the trained model.
       model_type (str): Type of model, e.g., 'cyto', 'nuclei'.
       Signal_to_noise (int): Signal-to-noise ratio in training images.
       background (int): Background intensity value.
       remove_background (bool): Whether to subtract background.
       learning_rate (float): Learning rate for optimization.
       weight_decay (float): Weight decay (L2 penalty).
       batch_size (int): Number of samples per batch.
       n_epochs (int): Maximum number of training epochs.
       from_scratch (bool): Whether to initialize weights randomly.
       diameter (float): Estimated object diameter.
       resize (bool): Whether to resize input images.
       width_height (list): Target width and height for resizing.
       verbose (bool): Print training progress.


.. py:function:: set_generate_dataset_defaults(settings)

   Set default args for generating a dataset from raw data and metadata.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Root directory for the experiment.
       file_metadata (str or None): Path to metadata CSV file.
       experiment (str): Experiment identifier.
       sample (str or None): Sample name or ID.


.. py:function:: get_perform_regression_default_settings(settings)

   Set default args for performing regression on phenotype data.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       count_data (list): List of paths to count tables.
       score_data (list): List of paths to score tables.
       positive_control (str): Gene ID of positive control.
       negative_control (str): Gene ID of negative control.
       min_n (int): Minimum number of data points per group.
       controls (list): List of guide RNA IDs to exclude.
       fraction_threshold (float or None): Minimum fraction threshold.
       dependent_variable (str): Name of response variable.
       threshold_method (str): Method for outlier thresholding (e.g., 'std').
       threshold_multiplier (float): Multiplier for thresholding.
       target_unique_count (int): Minimum unique guides per gene.
       transform (str or None): Apply transformation to input data (e.g., 'log').
       log_x (bool): Apply log10 to x-values.
       log_y (bool): Apply log10 to y-values.
       x_lim (tuple or None): x-axis limits.
       outlier_detection (bool): Enable outlier removal.
       agg_type (str or None): Aggregation type ('mean', 'median').
       min_cell_count (int or None): Minimum cells per sample to include.
       regression_type (str): Regression type ('ols', 'quantile').
       random_row_column_effects (bool): Include row/column as random effects.
       split_axis_lims (str): Delimiter-separated axis ranges for plotting.
       cov_type (str or None): Covariance estimator for robust standard errors.
       alpha (float): Alpha level or quantile (for quantile regression).
       filter_value (list): Values to keep in filter_column.
       filter_column (str): Column name for filtering.
       plateID (str): Plate ID used in analysis.
       metadata_files (list): List of CSVs with gene metadata.
       volcano (str): Label column for volcano plot.
       toxo (bool): Whether to use Toxoplasma-specific annotations.


.. py:function:: get_check_cellpose_models_default_settings(settings)

   Set default args for checking Cellpose model predictions.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       batch_size (int): Number of images per batch.
       CP_prob (float): Cellpose probability threshold.
       flow_threshold (float): Threshold for mask flow consistency.
       save (bool): Save output masks to disk.
       normalize (bool): Normalize image intensities.
       channels (list): Channel indices for [nucleus, cytoplasm].
       percentiles (tuple or None): Intensity percentiles for normalization.
       invert (bool): Invert image intensities.
       plot (bool): Display Cellpose output.
       diameter (float): Estimated object diameter.
       grayscale (bool): Convert image to grayscale.
       remove_background (bool): Subtract background intensity.
       background (int): Background value to subtract.
       Signal_to_noise (int): Signal-to-noise ratio assumption.
       verbose (bool): Print debug info.
       resize (bool): Resize image to target dimensions.
       target_height (int or None): Resize target height.
       target_width (int or None): Resize target width.


.. py:function:: get_identify_masks_finetune_default_settings(settings)

   Set default args for identifying masks using fine-tuned Cellpose models.

   :param settings: Dictionary to populate with default args.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Args:
       src (str): Path to input image directory.
       model_name (str): Pretrained Cellpose model name (e.g., 'cyto').
       custom_model (str or None): Path to a fine-tuned model (.pth).
       channels (list): Channel indices [nucleus, cytoplasm].
       background (int): Background value to subtract.
       remove_background (bool): Whether to subtract background.
       Signal_to_noise (int): Assumed SNR of input.
       CP_prob (float): Cellpose probability threshold.
       diameter (float): Estimated diameter of objects.
       batch_size (int): Number of images per batch.
       flow_threshold (float): Flow consistency threshold.
       save (bool): Save output masks to disk.
       verbose (bool): Print debug information.
       normalize (bool): Normalize image intensities.
       percentiles (tuple or None): Percentile normalization bounds.
       invert (bool): Invert intensities.
       resize (bool): Resize image to target dimensions.
       target_height (int or None): Resize target height.
       target_width (int or None): Resize target width.
       rescale (bool): Rescale image based on object size.
       resample (bool): Resample image for consistent shape.
       grayscale (bool): Convert to grayscale.
       fill_in (bool): Fill in holes in masks.


.. py:data:: q
   :value: None


.. py:data:: expected_types

.. py:data:: categories

.. py:data:: category_keys
   :value: ['Paths', 'General', 'Cellpose', 'Cell', 'Nucleus', 'Pathogen', 'Measurements', 'Object Image',...


.. py:function:: check_settings(vars_dict, expected_types, q=None)

   Validate and parse GUI-derived variable inputs according to expected types.

   :param vars_dict: Dictionary mapping setting keys to a tuple of (label, widget, variable, category).
   :type vars_dict: dict
   :param expected_types: Dictionary of expected types for each setting key.
   :type expected_types: dict
   :param q: Queue to collect error messages. If None, a new queue is created.
   :type q: multiprocessing.Queue, optional

   :returns:     - settings (dict): Parsed settings with values cast to the expected types.
                 - errors (list): List of error messages describing issues with format or type mismatches.
   :rtype: tuple

   .. rubric:: Notes

   - Supports nested structures such as list of lists and dicts.
   - Handles conversions to str, int, float, bool, list, dict, and None.
   - Custom parsing is performed for complex input types like Cellpose metadata or PNG config lists.
   - Errors are collected and returned without interrupting the full parsing process.


.. py:function:: generate_fields(variables, scrollable_frame)

   Dynamically generate labeled input fields with tooltips in a scrollable Tkinter frame.

   :param variables: Dictionary mapping setting keys to (var_type, options, default_value).
   :type variables: dict
   :param scrollable_frame: A scrollable frame widget to place the input fields into.
   :type scrollable_frame: ScrollableFrame

   :returns: Dictionary mapping each setting key to a tuple of (label, widget, variable, parent frame).
   :rtype: vars_dict (dict)

   .. rubric:: Notes

   - Uses `create_input_field` to generate the widgets.
   - Attaches tooltips using `spacrToolTip` where available.
   - Tooltips describe the expected format, data type, and function of each setting.


.. py:data:: descriptions

.. py:function:: set_annotate_default_settings(settings)

   Set default arguments for image annotation and visualization in the annotation tool.

   :param settings: Dictionary to update with default annotation settings.
   :type settings: dict

   :returns:

             Updated dictionary containing defaults for:
                 - src: input path
                 - image_type: type of image files
                 - channels: channels to display
                 - img_size: image display size
                 - annotation_column: metadata column to use for annotation
                 - normalize: apply intensity normalization
                 - normalize_channels: which channels to normalize
                 - outline: None or comma-separated channel(s) used to generate outline
                 - outline_threshold_factor: scale factor for thresholding outlines
                 - outline_sigma: sigma for Gaussian blur during outline generation
                 - percentiles: intensity normalization range
                 - measurement: comma-separated measurement columns to display
                 - threshold: thresholds corresponding to measurement columns
   :rtype: dict


.. py:function:: set_default_generate_barecode_mapping(settings={})

   Sets default settings for barcode mapping from sequencing data.

   :param settings: A dictionary to populate with default parameters if keys are missing.
   :type settings: dict

   :returns: The updated settings dictionary with default values applied.
   :rtype: dict

   Default Settings:
       - src (str): Path to input FASTQ files.
       - regex (str): Regular expression to extract column, grna, and row barcodes.
       - target_sequence (str): Sequence to locate for alignment.
       - offset_start (int): Offset to start looking for target sequence.
       - expected_end (int): Expected ending position of the barcode pattern.
       - column_csv (str): Path to CSV with column barcodes.
       - grna_csv (str): Path to CSV with grna barcodes.
       - row_csv (str): Path to CSV with row barcodes.
       - save_h5 (bool): Whether to save output in HDF5 format.
       - comp_type (str): Compression type for HDF5 ('zlib', 'lzf', etc.).
       - comp_level (int): Compression level if using zlib (0–9).
       - chunk_size (int): Number of reads to process per chunk.
       - n_jobs (int or None): Number of parallel jobs to use.
       - mode (str): 'paired' for paired-end, 'single' for single-end.
       - single_direction (str): 'R1' or 'R2' for direction if mode is 'single'.
       - test (bool): Whether to run in test mode.
       - fill_na (bool): If True, fill missing barcodes with 'NA'.


.. py:function:: get_default_generate_activation_map_settings(settings)

   Sets default settings for generating activation maps (e.g., Grad-CAM) from a trained deep learning model.

   :param settings: A dictionary to populate with default parameters if keys are missing.
   :type settings: dict

   :returns: The updated settings dictionary with default values applied.
   :rtype: dict

   Default Settings:
       - dataset (str): Path to dataset directory containing images.
       - model_type (str): Type of model architecture used ('maxvit', 'resnet', etc.).
       - model_path (str): Path to the trained model checkpoint.
       - image_size (int): Size to which input images will be resized.
       - batch_size (int): Number of images processed per batch.
       - normalize (bool): Whether to normalize input images.
       - cam_type (str): Class activation map type ('gradcam', 'gradcam++', etc.).
       - target_layer (str or None): Specific layer to target for activation visualization.
       - plot (bool): If True, display the activation maps during processing.
       - save (bool): If True, save activation map outputs to disk.
       - normalize_input (bool): Whether to normalize input images before model inference.
       - channels (list): List of channel indices to include in model input.
       - overlay (bool): Whether to overlay activation maps on the original images.
       - shuffle (bool): If True, shuffle the dataset before processing.
       - correlation (bool): If True, compute correlation between channels and activation maps.
       - manders_thresholds (list): List of thresholds for Manders’ overlap coefficient.
       - n_jobs (int or None): Number of parallel workers to use for processing.


.. py:function:: get_analyze_plaque_settings(settings)

   Sets default settings for analyzing plaque formation using Cellpose segmentation and related image processing tools.

   :param settings: A dictionary to populate with default parameters if keys are missing.
   :type settings: dict

   :returns: The updated settings dictionary with default values applied.
   :rtype: dict

   Default Settings:
       - src (str): Path to the directory containing input images.
       - masks (bool): Whether to use precomputed masks or generate new ones.
       - background (int): Background intensity value to subtract during preprocessing.
       - Signal_to_noise (int): Minimum signal-to-noise ratio required for valid segmentation.
       - CP_prob (float): Minimum Cellpose probability threshold for mask acceptance.
       - diameter (int): Expected object diameter (in pixels) for Cellpose segmentation.
       - batch_size (int): Number of images to process per batch.
       - flow_threshold (float): Flow error threshold for accepting Cellpose masks.
       - save (bool): If True, save the segmentation and analysis outputs to disk.
       - verbose (bool): If True, print detailed processing logs.
       - resize (bool): Whether to resize images before processing.
       - target_height (int): Height to which images should be resized (if `resize` is True).
       - target_width (int): Width to which images should be resized (if `resize` is True).
       - rescale (bool): Whether to apply rescaling of image intensity values.
       - resample (bool): If True, resample images using interpolation methods.
       - fill_in (bool): Whether to fill in holes or small gaps in detected masks.


.. py:function:: set_graph_importance_defaults(settings)

   Sets default parameters for plotting graph-based feature importance across groups.

   :param settings: Dictionary to be populated with default values if keys are missing.
   :type settings: dict

   :returns: Updated settings dictionary.
   :rtype: dict

   Default Settings:
       - csvs (str or list of str): Path(s) to CSV files containing importance scores.
       - grouping_column (str): Column name used to group features for plotting (e.g., 'compartment').
       - data_column (str): Column name containing the importance values to be plotted.
       - graph_type (str): Type of graph to plot. Options may include 'jitter_bar', 'violin', etc.
       - save (bool): Whether to save the plot to disk.


.. py:function:: set_interperate_vision_model_defaults(settings)

   Sets default parameters for interpreting vision model outputs using feature importance methods.

   :param settings: Dictionary to populate with default values for any missing keys.
   :type settings: dict

   :returns: Updated settings dictionary with default interpretation parameters.
   :rtype: dict

   Default Settings:
       - src (str): Path to input data directory.
       - scores (str): Path to model output scores (e.g., predictions or classification results).
       - tables (list): List of tables to include in interpretation (e.g., ['cell', 'nucleus', ...]).
       - feature_importance (bool): Whether to compute standard feature importance.
       - permutation_importance (bool): Whether to compute permutation importance.
       - shap (bool): Whether to compute SHAP values for model interpretability.
       - save (bool): Whether to save the results to disk.
       - nuclei_limit (int): Maximum number of nuclei objects to use for interpretation.
       - pathogen_limit (int): Maximum number of pathogen objects to use.
       - top_features (int): Number of top features to include in interpretation output.
       - shap_sample (bool): Whether to subsample data for SHAP computation.
       - n_jobs (int): Number of parallel jobs to use (-1 means all cores).
       - shap_approximate (bool): Use approximate SHAP computations for speed.
       - score_column (str): Column name from the scores file to use for interpretation.


.. py:function:: set_analyze_endodyogeny_defaults(settings)

   Sets default parameters for analyzing endodyogeny (intracellular replication) from segmented object tables.

   :param settings: Dictionary to populate with default values for any missing keys.
   :type settings: dict

   :returns: Updated settings dictionary with defaults for endodyogeny analysis.
   :rtype: dict

   Default Settings:
       - src (str): Path to the input directory containing segmentation tables.
       - tables (list): List of tables to analyze, typically ['cell', 'nucleus', 'pathogen', 'cytoplasm'].
       - cell_types (list): Cell types to include in the analysis, e.g., ['Hela'].
       - cell_plate_metadata (list or None): Metadata mapping for cell plates.
       - pathogen_types (list): Pathogen class labels, e.g., ['nc', 'pc'].
       - pathogen_plate_metadata (list): Metadata defining pathogen class plate layout.
       - treatments (list or None): List of treatments or conditions to group by.
       - treatment_plate_metadata (list or None): Metadata for treatment layout per plate.
       - min_area_bin (int): Minimum object area (in pixels) to consider in binning.
       - group_column (str): Column to use for grouping, e.g., 'pathogen'.
       - compartment (str): Compartment to analyze, typically 'pathogen'.
       - pathogen_limit (int): Max number of pathogens per group for subsampling.
       - nuclei_limit (int): Max number of nuclei per group for subsampling.
       - level (str): Aggregation level, either 'object' or 'well'.
       - um_per_px (float): Microns per pixel for conversion to physical units.
       - max_bins (int or None): Optional cap on number of histogram bins.
       - save (bool): Whether to save analysis results to file.
       - change_plate (bool): Whether to apply plate renaming or normalization.
       - cmap (str): Colormap to use for visualization.
       - verbose (bool): Whether to print additional processing info.


.. py:function:: set_analyze_class_proportion_defaults(settings)

   Sets default parameters for analyzing the proportion of classification labels
   across experimental conditions or cell types.

   :param settings: Dictionary to populate with default values for any missing keys.
   :type settings: dict

   :returns: Updated settings dictionary with defaults for class proportion analysis.
   :rtype: dict

   Default Settings:
       - src (str): Path to the input directory containing measurement tables.
       - tables (list): List of object tables to analyze (e.g., ['cell', 'nucleus', 'pathogen', 'cytoplasm']).
       - cell_types (list): List of cell types to include, e.g., ['Hela'].
       - cell_plate_metadata (list or None): Metadata describing layout of cell types on plates.
       - pathogen_types (list): List of pathogen labels, typically ['nc', 'pc'].
       - pathogen_plate_metadata (list): Metadata describing layout of pathogen classes on plates.
       - treatments (list or None): Experimental treatments to group by.
       - treatment_plate_metadata (list or None): Metadata describing treatment layouts on plates.
       - group_column (str): Metadata column to group by, typically 'condition'.
       - class_column (str): Column containing predicted or manual class labels.
       - pathogen_limit (int): Max number of pathogen objects to include per group.
       - nuclei_limit (int): Max number of nuclei objects to include per group.
       - level (str): Aggregation level, e.g., 'well' or 'object'.
       - save (bool): Whether to save results to disk.
       - verbose (bool): Whether to print processing steps and warnings.


.. py:function:: get_plot_data_from_csv_default_settings(settings)

   Sets default parameters for plotting data from a CSV file.

   :param settings: Dictionary to populate with default values for any missing keys.
   :type settings: dict

   :returns: Updated settings dictionary with default plotting parameters.
   :rtype: dict

   Default Settings:
       - src (str): Path to the CSV file to be plotted.
       - data_column (str): Name of the column containing the data to plot.
       - grouping_column (str): Column to group the data by (e.g., experimental condition).
       - graph_type (str): Type of plot to generate, e.g., 'violin', 'box', 'bar', or 'strip'.
       - save (bool): Whether to save the plot to disk.
       - y_lim (tuple or None): Y-axis limits as (min, max), or None for automatic scaling.
       - log_y (bool): Whether to use a logarithmic scale on the Y-axis.
       - log_x (bool): Whether to use a logarithmic scale on the X-axis.
       - keep_groups (list or None): Specific groups to retain in the plot; others will be excluded.
       - representation (str): Level of data aggregation, typically 'well' or 'object'.
       - theme (str): Visual theme for the plot, e.g., 'dark' or 'light'.
       - remove_outliers (bool): Whether to exclude statistical outliers from the plot.
       - verbose (bool): Whether to print status messages during plotting.


