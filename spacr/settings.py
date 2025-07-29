import os, ast

def set_default_plot_merge_settings():
    """
    Set default configuration values for plotting merged image masks and overlays.

    Returns:
        dict: Dictionary of default settings for plotting merged masks and overlays.

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
    """
    settings = {}
    settings.setdefault('pathogen_limit', 10)
    settings.setdefault('nuclei_limit', 1)
    settings.setdefault('remove_background', False)
    settings.setdefault('filter_min_max', None)
    settings.setdefault('channel_dims', [0,1,2,3])
    settings.setdefault('backgrounds', [100,100,100,100])
    settings.setdefault('cell_mask_dim', 4)
    settings.setdefault('nucleus_mask_dim', 5)
    settings.setdefault('pathogen_mask_dim', 6)
    settings.setdefault('outline_thickness', 3)
    settings.setdefault('outline_color', 'gbr')
    settings.setdefault('overlay_chans', [1,2,3])
    settings.setdefault('overlay', True)
    settings.setdefault('normalization_percentiles', [2,98])
    settings.setdefault('normalize', True)
    settings.setdefault('print_object_number', True)
    settings.setdefault('nr', 1)
    settings.setdefault('figuresize', 10)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('verbose', True)
    return settings

def set_default_settings_preprocess_generate_masks(settings={}):
    """
    Set default settings for the preprocessing and mask generation pipeline using Cellpose or similar tools.

    This function populates a settings dictionary with default values used during image preprocessing,
    segmentation mask generation, background removal, plotting, timelapse handling, and more.

    Parameters:
        settings (dict): Optional dictionary with user-specified values. Defaults will only fill in missing keys.

    Returns:
        dict: A dictionary containing default configuration parameters for mask generation.

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
    """
    settings.setdefault('denoise', False)
    settings.setdefault('src', 'path')
    settings.setdefault('delete_intermediate', False)
    settings.setdefault('preprocess', True)
    settings.setdefault('masks', True)
    settings.setdefault('save', True)
    settings.setdefault('consolidate', False)
    settings.setdefault('batch_size', 50)
    settings.setdefault('test_mode', False)
    settings.setdefault('test_images', 10)
    settings.setdefault('magnification', 20)
    settings.setdefault('custom_regex', None)
    settings.setdefault('metadata_type', 'cellvoyager')
    settings.setdefault('n_jobs', os.cpu_count()-4)
    settings.setdefault('randomize', True)
    settings.setdefault('verbose', True)
    settings.setdefault('remove_background_cell', False)
    settings.setdefault('remove_background_nucleus', False)
    settings.setdefault('remove_background_pathogen', False)
    
    settings.setdefault('cell_diamiter', None)
    settings.setdefault('nucleus_diamiter', None)
    settings.setdefault('pathogen_diamiter', None)

    # Channel settings
    settings.setdefault('cell_channel', None)
    settings.setdefault('nucleus_channel', None)
    settings.setdefault('pathogen_channel', None)
    settings.setdefault('channels', [0,1,2,3])
    settings.setdefault('pathogen_background', 100)
    settings.setdefault('pathogen_Signal_to_noise', 10)
    settings.setdefault('pathogen_CP_prob', 0)
    settings.setdefault('cell_background', 100)
    settings.setdefault('cell_Signal_to_noise', 10)
    settings.setdefault('cell_CP_prob', 0)
    settings.setdefault('nucleus_background', 100)
    settings.setdefault('nucleus_Signal_to_noise', 10)
    settings.setdefault('nucleus_CP_prob', 0)
    settings.setdefault('nucleus_FT', 1.0)
    settings.setdefault('cell_FT', 1.0)
    settings.setdefault('pathogen_FT', 1.0)
    
    # Plot settings
    settings.setdefault('plot', False)
    settings.setdefault('figuresize', 10)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('normalize', True)
    settings.setdefault('normalize_plots', True)
    settings.setdefault('examples_to_plot', 1)

    # Analasys settings
    settings.setdefault('pathogen_model', None)
    settings.setdefault('merge_pathogens', False)
    settings.setdefault('filter', False)
    settings.setdefault('lower_percentile', 2)

    # Timelapse settings
    settings.setdefault('timelapse', False)
    settings.setdefault('fps', 2)
    settings.setdefault('timelapse_displacement', None)
    settings.setdefault('timelapse_memory', 3)
    settings.setdefault('timelapse_frame_limits', [5,])
    settings.setdefault('timelapse_remove_transient', False)
    settings.setdefault('timelapse_mode', 'trackpy')
    settings.setdefault('timelapse_objects', None)

    # Misc settings
    settings.setdefault('all_to_mip', False)
    settings.setdefault('upscale', False)
    settings.setdefault('upscale_factor', 2.0)
    settings.setdefault('adjust_cells', False)
    settings.setdefault('use_sam_cell', False)
    settings.setdefault('use_sam_nucleus', False)
    settings.setdefault('use_sam_pathogen', False)
    
    return settings

def set_default_plot_data_from_db(settings):
    """
    Set default plotting settings for visualizing data from an SQL database.

    Parameters:
        settings (dict): Settings dictionary to populate.

    Returns:
        dict: Dictionary with default keys and values for graph plotting from database data.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('database', 'measurements.db')
    settings.setdefault('graph_name', 'Figure_1')
    settings.setdefault('table_names', ['cell', 'cytoplasm', 'nucleus', 'pathogen'])
    settings.setdefault('data_column', 'recruitment')
    settings.setdefault('grouping_column', 'condition')
    settings.setdefault('cell_types', ['Hela'])
    settings.setdefault('cell_plate_metadata', None)
    settings.setdefault('pathogen_types', None)
    settings.setdefault('pathogen_plate_metadata', None)
    settings.setdefault('treatments', None)
    settings.setdefault('treatment_plate_metadata', None)
    settings.setdefault('graph_type', 'jitter')
    settings.setdefault('theme', 'deep')
    settings.setdefault('save', True)
    settings.setdefault('y_lim', [1,1.5])
    settings.setdefault('verbose', False)
    settings.setdefault('channel_of_interest', 1)
    settings.setdefault('nuclei_limit', 2)
    settings.setdefault('pathogen_limit', 3)
    settings.setdefault('representation', 'well')
    settings.setdefault('uninfected', False)
    return settings

def set_default_settings_preprocess_img_data(settings):
    """
    Set default values for preprocessing image data before analysis.

    Parameters:
        settings (dict): Settings dictionary to populate.

    Returns:
        dict: Updated dictionary with image preprocessing settings.

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
    """
    settings.setdefault('metadata_type', 'cellvoyager')
    settings.setdefault('custom_regex', None)
    settings.setdefault('nr', 1)
    settings.setdefault('plot', True)
    settings.setdefault('batch_size', 50)
    settings.setdefault('timelapse', False)
    settings.setdefault('lower_percentile', 2)
    settings.setdefault('randomize', True)
    settings.setdefault('all_to_mip', False)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('figuresize', 10)
    settings.setdefault('normalize', True)
    settings.setdefault('save_dtype', 'uint16')
    settings.setdefault('test_mode', False)
    settings.setdefault('test_images', 10)
    settings.setdefault('random_test', True)
    settings.setdefault('fps', 2)
    return settings

def _get_object_settings(object_type, settings):
    """
    Generate object-specific segmentation parameters for Cellpose or similar models.

    This function constructs a dictionary of settings tailored to the object type
    (e.g., 'cell', 'nucleus', or 'pathogen') based on user input and default rules.
    It dynamically adjusts diameter, model selection, and filtering options.

    Parameters:
        object_type (str): Type of object to configure. Must be one of 'cell', 'nucleus', or 'pathogen'.
        settings (dict): Configuration dictionary containing global pipeline parameters such as
                         'magnification', diameter overrides, SAM usage flags, and verbosity.

    Returns:
        dict: Object-specific segmentation settings.

    Raises:
        ValueError: If object_type is not one of the supported types.

    Notes:
        - The estimated object diameter is computed from magnification using `_get_diam`.
        - If explicit diameters (e.g., `cell_diamiter`) are provided in settings, they override the computed value.
        - The model name is chosen based on object type and SAM usage.
        - Verbose output is printed if `settings['verbose']` is True.

    Example output structure:
        {
            'diameter': 30.0,
            'minimum_size': 225.0,
            'maximum_size': 9000.0,
            'merge': False,
            'resample': True,
            'remove_border_objects': False,
            'model_name': 'cyto2',
            'filter_size': False,
            'filter_intensity': False,
            'restore_type': None
        }
    """
    from .utils import _get_diam
    object_settings = {}

    object_settings['diameter'] = _get_diam(settings['magnification'], obj=object_type)
    object_settings['minimum_size'] = (object_settings['diameter']**2)/4
    object_settings['maximum_size'] = (object_settings['diameter']**2)*10
    object_settings['merge'] = False
    object_settings['resample'] = True
    object_settings['remove_border_objects'] = False
    object_settings['model_name'] = 'cyto'
    
    if object_type == 'cell':
        if settings['nucleus_channel'] is None:
            object_settings['model_name'] = 'cyto'
        else:
            object_settings['model_name'] = 'cyto2'
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['restore_type'] = settings.get('cell_restore_type', None)
        if settings['cell_diamiter'] is not None:
            if isinstance(settings['cell_diamiter'], (int, float)):
                object_settings['diameter'] = settings['cell_diamiter']
                object_settings['minimum_size'] = (object_settings['diameter']**2)/4
                object_settings['maximum_size'] = (object_settings['diameter']**2)*10
            else:
                print(f'Cell diameter must be an integer or float, got {settings["cell_diamiter"]}')
        if settings['use_sam_cell']:
            object_settings['model_name'] = 'sam'

    elif object_type == 'nucleus':
        object_settings['model_name'] = 'nuclei'
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['restore_type'] = settings.get('nucleus_restore_type', None)
        
        if settings['nucleus_diamiter'] is not None:
            if isinstance(settings['nucleus_diamiter'], (int, float)):
                object_settings['diameter'] = settings['nucleus_diamiter']
                object_settings['minimum_size'] = (object_settings['diameter']**2)/4
                object_settings['maximum_size'] = (object_settings['diameter']**2)*10
            else:
                print(f'Nucleus diameter must be an integer or float, got {settings["nucleus_diamiter"]}')
        if settings['use_sam_nucleus']:
            object_settings['model_name'] = 'sam'

    elif object_type == 'pathogen':
        object_settings['model_name'] = 'cyto'
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['resample'] = False
        object_settings['restore_type'] = settings.get('pathogen_restore_type', None)
        object_settings['merge'] = settings['merge_pathogens']
        
        if settings['pathogen_diamiter'] is not None:
            if isinstance(settings['pathogen_diamiter'], (int, float)):
                object_settings['diameter'] = settings['pathogen_diamiter']
                object_settings['minimum_size'] = (object_settings['diameter']**2)/4
                object_settings['maximum_size'] = (object_settings['diameter']**2)*10
            else:
                print(f'Pathogen diameter must be an integer or float, got {settings["pathogen_diamiter"]}')
                
        if settings['use_sam_pathogen']:
            object_settings['model_name'] = 'sam'
        
    else:
        print(f'Object type: {object_type} not supported. Supported object types are : cell, nucleus and pathogen')
        
    if settings['verbose']:
        print(object_settings)
        
    return object_settings 

def set_default_umap_image_settings(settings={}):
    """
    Set default configuration values for UMAP-based image embedding and clustering.

    Parameters:
        settings (dict): Optional dictionary of user-provided settings. Keys that are not present
                         will be set to their default values.

    Returns:
        dict: Updated settings dictionary containing all necessary UMAP image analysis parameters.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('row_limit', 1000)
    settings.setdefault('tables', ['cell', 'cytoplasm', 'nucleus', 'pathogen'])
    settings.setdefault('visualize', 'cell')
    settings.setdefault('image_nr', 16)
    settings.setdefault('dot_size', 50)
    settings.setdefault('n_neighbors', 1000)
    settings.setdefault('min_dist', 0.1)
    settings.setdefault('metric', 'euclidean')
    settings.setdefault('eps', 0.9)
    settings.setdefault('min_samples', 100)
    settings.setdefault('filter_by', 'channel_0')
    settings.setdefault('img_zoom', 0.5)
    settings.setdefault('plot_by_cluster', True)
    settings.setdefault('plot_cluster_grids', True)
    settings.setdefault('remove_cluster_noise', True)
    settings.setdefault('remove_highly_correlated', True)
    settings.setdefault('log_data', False)
    settings.setdefault('figuresize', 10)
    settings.setdefault('black_background', True)
    settings.setdefault('remove_image_canvas', False)
    settings.setdefault('plot_outlines', True)
    settings.setdefault('plot_points', True)
    settings.setdefault('smooth_lines', True)
    settings.setdefault('clustering', 'dbscan')
    settings.setdefault('exclude', None)
    settings.setdefault('col_to_compare', 'columnID')
    settings.setdefault('pos', 'c1')
    settings.setdefault('neg', 'c2')
    settings.setdefault('mix', 'c3')
    settings.setdefault('embedding_by_controls', False)
    settings.setdefault('plot_images', True)
    settings.setdefault('reduction_method','umap')
    settings.setdefault('save_figure', False)
    settings.setdefault('n_jobs', -1)
    settings.setdefault('color_by', None)
    settings.setdefault('exclude_conditions', None)
    settings.setdefault('analyze_clusters', False)
    settings.setdefault('resnet_features', False)
    settings.setdefault('verbose',True)
    return settings

def get_measure_crop_settings(settings={}):
    """
    Set default configuration for object measurement and cropping.

    This function initializes and returns a dictionary of settings used for 
    measuring and cropping segmented objects such as cells, nuclei, or pathogens.
    It covers measurement parameters, image cropping/export options, 
    multiprocessing behavior, and test mode overrides.

    Parameters:
        settings (dict, optional): Existing settings dictionary to be updated with defaults.

    Returns:
        dict: Fully populated settings dictionary with default values applied.

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

    Notes:
        - When 'test_mode' is True, verbose and plot modes are automatically enabled.
        - 'os.cpu_count()' is used to allocate available cores for parallel processing.

    Example:
        settings = get_measure_crop_settings()
    """
    settings.setdefault('src', 'path')
    settings.setdefault('delete_intermediate', False)
    
    settings.setdefault('verbose', False)
    settings.setdefault('experiment', 'exp')
    
    # Test mode
    settings.setdefault('test_mode', False)
    settings.setdefault('test_nr', 10)
    settings.setdefault('channels', [0,1,2,3])

    #measurement settings
    settings.setdefault('save_measurements',True)
    settings.setdefault('radial_dist', True)
    settings.setdefault('calculate_correlation', True)
    settings.setdefault('manders_thresholds', [15,85,95])
    settings.setdefault('homogeneity', True)
    settings.setdefault('homogeneity_distances', [8,16,32])

    # Cropping settings    # Cropping settings
    settings.setdefault('save_arrays', False)
    settings.setdefault('save_png',True)
    settings.setdefault('use_bounding_box',False)
    settings.setdefault('png_size',[224,224])
    settings.setdefault('png_dims',[0,1,2])
    settings.setdefault('normalize',False)    # Cropping settings
    settings.setdefault('save_arrays', False)
    settings.setdefault('save_png',True)
    settings.setdefault('use_bounding_box',False)
    settings.setdefault('png_size',[224,224])
    settings.setdefault('png_dims',[0,1,2])
    settings.setdefault('normalize',False)
    settings.setdefault('normalize_by','png')
    settings.setdefault('crop_mode',['cell'])
    settings.setdefault('dialate_pngs', False)
    settings.setdefault('dialate_png_ratios', [0.2])

    # Timelapsed settings
    settings.setdefault('timelapse', False)
    settings.setdefault('timelapse_objects', ['cell'])

    # Operational settings
    settings.setdefault('plot',False)
    settings.setdefault('n_jobs', os.cpu_count()-2)

    # Object settings
    settings.setdefault('cell_mask_dim',4)
    settings.setdefault('nucleus_mask_dim',5)
    settings.setdefault('pathogen_mask_dim',6)
    settings.setdefault('cytoplasm',False)
    settings.setdefault('uninfected',True)
    settings.setdefault('cell_min_size',0)
    settings.setdefault('nucleus_min_size',0)
    settings.setdefault('pathogen_min_size',0)
    settings.setdefault('cytoplasm_min_size',0)
    settings.setdefault('merge_edge_pathogen_cells', True)
    
    settings.setdefault('distance_gaussian_sigma', 10)
    
    if settings['test_mode']:
        settings['verbose'] = True
        settings['plot'] = True
        test_imgs = settings['test_nr']
        print(f'Test mode enabled with {test_imgs} images, plotting set to True')

    return settings

def set_default_analyze_screen(settings):
    """
    Set default configuration for analyzing a CRISPR or compound screen.

    This function populates a provided settings dictionary with defaults related to 
    feature extraction, model training, screen scoring, and heatmap visualization.

    Parameters:
        settings (dict): Dictionary of user-provided settings to be updated.

    Returns:
        dict: Updated settings dictionary with all necessary keys and default values.

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

    Example:
        settings = set_default_analyze_screen({})
    """
    settings.setdefault('src', 'path')
    settings.setdefault('annotation_column', None)
    settings.setdefault('save_to_db', False)
    settings.setdefault('model_type_ml','xgboost')
    settings.setdefault('heatmap_feature','predictions')
    settings.setdefault('grouping','mean')
    settings.setdefault('min_max','allq')
    settings.setdefault('cmap','viridis')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('minimum_cell_count',25)
    settings.setdefault('reg_alpha',0.1)
    settings.setdefault('reg_lambda',1.0)
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('n_estimators',1000)
    settings.setdefault('test_size',0.2)
    settings.setdefault('location_column','columnID')
    settings.setdefault('positive_control','c2')
    settings.setdefault('negative_control','c1')
    settings.setdefault('exclude',None)
    settings.setdefault('nuclei_limit',True)
    settings.setdefault('pathogen_limit',3)
    settings.setdefault('n_repeats',10)
    settings.setdefault('top_features',30)
    settings.setdefault('remove_low_variance_features',True)
    settings.setdefault('remove_highly_correlated_features',True)
    settings.setdefault('n_jobs',-1)
    settings.setdefault('prune_features',False)
    settings.setdefault('cross_validation',True)
    settings.setdefault('verbose',True)
    return settings

def set_default_train_test_model(settings):
    """
    Set default configuration for training and testing a deep learning classification model.

    This function populates a provided dictionary with default settings used for model training, 
    including architecture, optimizer, augmentation, and hardware preferences.

    Parameters:
        settings (dict): Dictionary of user-provided settings to be updated.

    Returns:
        dict: Updated settings dictionary with all necessary keys and default values.

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

    Example:
        settings = set_default_train_test_model({})
    """
    cores = os.cpu_count()-2

    settings.setdefault('src','path')
    settings.setdefault('train',True)
    settings.setdefault('test',False)
    settings.setdefault('classes',['nc','pc'])
    settings.setdefault('model_type','maxvit_t')
    settings.setdefault('optimizer_type','adamw')
    settings.setdefault('schedule','reduce_lr_on_plateau') #reduce_lr_on_plateau, step_lr
    settings.setdefault('loss_type','focal_loss') # binary_cross_entropy_with_logits
    settings.setdefault('normalize',True)
    settings.setdefault('image_size',224)
    settings.setdefault('batch_size',64)
    settings.setdefault('epochs',100)
    settings.setdefault('val_split',0.1)
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('weight_decay',0.00001)
    settings.setdefault('dropout_rate',0.1)
    settings.setdefault('init_weights',True)
    settings.setdefault('amsgrad',True)
    settings.setdefault('use_checkpoint',True)
    settings.setdefault('gradient_accumulation',True)
    settings.setdefault('gradient_accumulation_steps',4)
    settings.setdefault('intermedeate_save',True)
    settings.setdefault('pin_memory',False)
    settings.setdefault('n_jobs',cores)
    settings.setdefault('train_channels',['r','g','b'])
    settings.setdefault('augment',False)
    settings.setdefault('verbose',False)
    return settings

def set_generate_training_dataset_defaults(settings):
    """
    Set default configuration for generating a training dataset from measurements and metadata.

    This function populates the given dictionary with default values required for generating
    a structured dataset for supervised learning based on annotated metadata or measurements.

    Args:
        settings (dict): Dictionary to populate with defaults if not already present.

    Returns:
        dict: Updated settings dictionary with default keys and values for dataset generation.

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

    Example:
        settings = set_generate_training_dataset_defaults({})
    """
    settings.setdefault('src','path')
    settings.setdefault('tables', ['cell', 'nucleus', 'pathogen', 'cytoplasm'])
    settings.setdefault('dataset_mode','metadata')
    settings.setdefault('annotation_column','test')
    settings.setdefault('annotated_classes',[1,2])
    settings.setdefault('class_metadata',['nc','pc'])
    settings.setdefault('metadata_item_1_name',['nc','pc'])
    settings.setdefault('metadata_item_1_value',[['c19','c2'],['c3','c4']])
    settings.setdefault('metadata_item_2_name',['sample1','sample2'])
    settings.setdefault('metadata_item_2_value',[['r1','r2'],['r3','r4']])
    settings.setdefault('size',224)
    settings.setdefault('test_split',0.1)
    settings.setdefault('class_metadata',[['c1'],['c2']])
    settings.setdefault('metadata_type_by','columnID')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('custom_measurement',None)
    settings.setdefault('tables',None)
    settings.setdefault('nuclei_limit',True)
    settings.setdefault('pathogen_limit',True)
    settings.setdefault('png_type','cell_png')
    
    return settings

def deep_spacr_defaults(settings):
    """
    Set default arguments for deep learning analysis in spaCR.

    This function fills in default arguments for training, testing, and applying deep learning
    models on spaCR datasets, supporting both metadata and measurement-based annotations.

    Args:
        settings (dict): Dictionary to populate with default arguments.

    Returns:
        dict: Updated settings dictionary.

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
    """
    cores = os.cpu_count()-4
    
    settings.setdefault('src','path')
    settings.setdefault('dataset_mode','metadata')
    settings.setdefault('annotation_column','test')
    settings.setdefault('annotated_classes',[1,2])
    settings.setdefault('classes',['nc','pc'])
    settings.setdefault('size',224)
    settings.setdefault('test_split',0.1)
    settings.setdefault('class_metadata',[['c1'],['c2']])
    settings.setdefault('metadata_type_by','columnID')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('custom_measurement',None)
    settings.setdefault('tables',None)
    settings.setdefault('png_type','cell_png')
    settings.setdefault('custom_model',False)
    settings.setdefault('custom_model_path','path')
    settings.setdefault('train',True)
    settings.setdefault('test',False)
    settings.setdefault('model_type','maxvit_t')
    settings.setdefault('optimizer_type','adamw')
    settings.setdefault('schedule','reduce_lr_on_plateau') #reduce_lr_on_plateau, step_lr
    settings.setdefault('loss_type','focal_loss') # binary_cross_entropy_with_logits
    settings.setdefault('normalize',True)
    settings.setdefault('image_size',224)
    settings.setdefault('batch_size',64)
    settings.setdefault('epochs',100)
    settings.setdefault('val_split',0.1)
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('weight_decay',0.00001)
    settings.setdefault('dropout_rate',0.1)
    settings.setdefault('init_weights',True)
    settings.setdefault('amsgrad',True)
    settings.setdefault('use_checkpoint',True)
    settings.setdefault('gradient_accumulation',True)
    settings.setdefault('gradient_accumulation_steps',4)
    settings.setdefault('intermedeate_save',True)
    settings.setdefault('pin_memory',False)
    settings.setdefault('n_jobs',cores)
    settings.setdefault('train_channels',['r','g','b'])
    settings.setdefault('augment',False)
    settings.setdefault('verbose',True)
    settings.setdefault('apply_model_to_dataset',True)
    settings.setdefault('file_metadata',None)
    settings.setdefault('sample',None)
    settings.setdefault('experiment','exp.')
    settings.setdefault('score_threshold',0.5)
    settings.setdefault('dataset','path')
    settings.setdefault('model_path','path')
    settings.setdefault('file_type','cell_png')
    settings.setdefault('generate_training_dataset', True)
    settings.setdefault('train_DL_model', True)
    return settings

def get_train_test_model_settings(settings):
    """
    Set default args for training and testing a deep learning model.

    This function populates the `settings` dictionary with default args used for
    training and evaluating deep learning models in spaCR.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('train', True)
    settings.setdefault('test', False)
    settings.setdefault('custom_model', False)
    settings.setdefault('classes', ['nc','pc'])
    settings.setdefault('train_channels', ['r','g','b'])
    settings.setdefault('model_type', 'maxvit_t')
    settings.setdefault('optimizer_type', 'adamw')
    settings.setdefault('schedule', 'reduce_lr_on_plateau')
    settings.setdefault('loss_type', 'focal_loss')
    settings.setdefault('normalize', True)
    settings.setdefault('image_size', 224)
    settings.setdefault('batch_size', 64)
    settings.setdefault('epochs', 100)
    settings.setdefault('val_split', 0.1)
    settings.setdefault('learning_rate', 0.0001)
    settings.setdefault('weight_decay', 0.00001)
    settings.setdefault('dropout_rate', 0.1)
    settings.setdefault('init_weights', True)
    settings.setdefault('amsgrad', True)
    settings.setdefault('use_checkpoint', True)
    settings.setdefault('gradient_accumulation', True)
    settings.setdefault('gradient_accumulation_steps', 4)
    settings.setdefault('intermedeate_save',True)
    settings.setdefault('pin_memory', True)
    settings.setdefault('n_jobs', 30)
    settings.setdefault('augment', True)
    settings.setdefault('verbose', True)
    return settings

def get_analyze_recruitment_default_settings(settings):
    """
    Set default args for recruitment analysis of host/pathogen protein localization.

    This function populates the `settings` dictionary with default values for analyzing
    recruitment of host proteins (e.g., ESCRT) to pathogens under various treatment conditions.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('target','protein')
    settings.setdefault('cell_types',['HeLa'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['pathogen_1', 'pathogen_2'])
    settings.setdefault('pathogen_plate_metadata',[['c1', 'c2', 'c3'],['c4','c5', 'c6']])
    settings.setdefault('treatments',['cm', 'lovastatin'])
    settings.setdefault('treatment_plate_metadata',[['r1', 'r2','r3'], ['r4', 'r5','r6']])
    #settings.setdefault('metadata_types',['columnID', 'columnID', 'rowID'])
    settings.setdefault('channel_dims',[0,1,2,3])
    settings.setdefault('cell_chann_dim',3)
    settings.setdefault('cell_mask_dim',4)
    settings.setdefault('nucleus_chann_dim',0)
    settings.setdefault('nucleus_mask_dim',5)
    settings.setdefault('pathogen_chann_dim',2)
    settings.setdefault('pathogen_mask_dim',6)
    settings.setdefault('channel_of_interest',2)
    settings.setdefault('plot',True)
    settings.setdefault('plot_nr',3)
    settings.setdefault('plot_control',True)
    settings.setdefault('figuresize',10)
    settings.setdefault('pathogen_limit',10)
    settings.setdefault('nuclei_limit',1)
    settings.setdefault('cells_per_well',0)
    settings.setdefault('pathogen_size_range',[0,100000])
    settings.setdefault('nucleus_size_range',[0,100000])
    settings.setdefault('cell_size_range',[0,100000])
    settings.setdefault('pathogen_intensity_range',[0,100000])
    settings.setdefault('nucleus_intensity_range',[0,100000])
    settings.setdefault('cell_intensity_range',[0,100000])
    settings.setdefault('target_intensity_min',1)
    return settings

def get_default_test_cellpose_model_settings(settings):
    """
    Set default args for testing a Cellpose model.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('src','path')
    settings.setdefault('model_path','path')
    settings.setdefault('save',True)
    settings.setdefault('normalize',True)
    settings.setdefault('percentiles',(2,98))
    settings.setdefault('batch_size',50)
    settings.setdefault('CP_probability',0)
    settings.setdefault('FT',100)
    settings.setdefault('target_size',1000)
    return settings

def get_default_apply_cellpose_model_settings(settings):
    """
    Set default args for applying a trained Cellpose model.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('src','path')
    settings.setdefault('model_path','path')
    settings.setdefault('save',True)
    settings.setdefault('normalize',True)
    settings.setdefault('percentiles',(2,98))
    settings.setdefault('batch_size',50)
    settings.setdefault('CP_probability',0)
    settings.setdefault('FT',100)
    settings.setdefault('circularize',False)
    settings.setdefault('target_size',1000)
    return settings

def default_settings_analyze_percent_positive(settings):
    """
    Set default args for analyzing percent-positive cells based on intensity thresholding.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

    Default Args:
        src (str): Path to data table.
        tables (list): List of object types to analyze (e.g., ['cell']).
        filter_1 (list): Filtering rule, e.g., ['cell_area', 1000].
        value_col (str): Column name containing intensity values.
        threshold (float): Threshold above which objects are considered positive.
    """
    settings.setdefault('src','path')
    settings.setdefault('tables',['cell'])
    settings.setdefault('filter_1',['cell_area',1000])
    settings.setdefault('value_col','cell_channel_2_mean_intensity')
    settings.setdefault('threshold',2000)
    return settings

def get_analyze_reads_default_settings(settings):
    """
    Set default args for analyzing NGS reads to extract barcodes.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

    Default Args:
        src (str): Path to FASTQ file or folder.
        upstream (str): Sequence upstream of the barcode.
        downstream (str): Sequence downstream of the barcode.
        barecode_length_1 (int): Length of the first barcode.
        barecode_length_2 (int): Length of the second barcode.
        chunk_size (int): Number of reads to process per chunk.
        test (bool): Enable test mode with limited read parsing.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('upstream', 'CTTCTGGTAAATGGGGATGTCAAGTT') 
    settings.setdefault('downstream', 'GTTTAAGAGCTATGCTGGAAACAGCAG') #This is the reverce compliment of the column primer starting from the end #TGCTGTTTAAGAGCTATGCTGGAAACAGCA
    settings.setdefault('barecode_length_1', 8)
    settings.setdefault('barecode_length_2', 7)
    settings.setdefault('chunk_size', 1000000)
    settings.setdefault('test', False)
    return settings

def get_map_barcodes_default_settings(settings):
    """
    Set default args for mapping extracted barcodes to plate, grna, and control metadata.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('grna', '/home/carruthers/Documents/grna_barcodes.csv')
    settings.setdefault('barcodes', '/home/carruthers/Documents/SCREEN_BARCODES.csv')
    settings.setdefault('plate_dict', "{'EO1': 'plate1', 'EO2': 'plate2', 'EO3': 'plate3', 'EO4': 'plate4', 'EO5': 'plate5', 'EO6': 'plate6', 'EO7': 'plate7', 'EO8': 'plate8'}")
    settings.setdefault('test', False)
    settings.setdefault('verbose', True)
    settings.setdefault('pc', 'TGGT1_220950_1')
    settings.setdefault('pc_loc', 'c2')
    settings.setdefault('nc', 'TGGT1_233460_4')
    settings.setdefault('nc_loc', 'c1')
    return settings

def get_train_cellpose_default_settings(settings):
    """
    Set default args for training a Cellpose model.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('model_name','new_model')
    settings.setdefault('model_type','cyto')
    settings.setdefault('Signal_to_noise',10)
    settings.setdefault('background',200)
    settings.setdefault('remove_background',False)
    settings.setdefault('learning_rate',0.2)
    settings.setdefault('weight_decay',1e-05)
    settings.setdefault('batch_size',8)
    settings.setdefault('n_epochs',10000)
    settings.setdefault('from_scratch',False)
    settings.setdefault('diameter',30)
    settings.setdefault('resize',False)
    settings.setdefault('width_height',[1000,1000])
    settings.setdefault('verbose',True)
    return settings

def set_generate_dataset_defaults(settings):
    """
    Set default args for generating a dataset from raw data and metadata.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

    Default Args:
        src (str): Root directory for the experiment.
        file_metadata (str or None): Path to metadata CSV file.
        experiment (str): Experiment identifier.
        sample (str or None): Sample name or ID.
    """
    settings.setdefault('src','path')
    settings.setdefault('file_metadata',None)
    settings.setdefault('experiment','experiment_1')
    settings.setdefault('sample',None)
    return settings

def get_perform_regression_default_settings(settings):
    """
    Set default args for performing regression on phenotype data.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('count_data','list of paths')
    settings.setdefault('score_data','list of paths')
    settings.setdefault('positive_control','239740')
    settings.setdefault('negative_control','233460')
    settings.setdefault('min_n',0)
    settings.setdefault('controls',['000000_1','000000_10','000000_11','000000_12','000000_13','000000_14','000000_15','000000_16','000000_17','000000_18','000000_19','000000_20','000000_21','000000_22','000000_23','000000_24','000000_25','000000_26','000000_27','000000_28','000000_29','000000_3','000000_30','000000_31','000000_32','000000_4','000000_5','000000_6','000000_8','000000_9'])
    settings.setdefault('fraction_threshold',None)
    settings.setdefault('dependent_variable','pred')
    settings.setdefault('threshold_method','std')
    settings.setdefault('threshold_multiplier',3)
    settings.setdefault('target_unique_count',5)
    settings.setdefault('transform',None)
    settings.setdefault('log_x',False)
    settings.setdefault('log_y',False)
    settings.setdefault('x_lim',None)
    settings.setdefault('outlier_detection',True)
    settings.setdefault('agg_type','mean')
    settings.setdefault('min_cell_count',None)
    settings.setdefault('regression_type','ols')
    settings.setdefault('random_row_column_effects',False)
    settings.setdefault('split_axis_lims','')
    settings.setdefault('cov_type',None)
    settings.setdefault('alpha',1)
    settings.setdefault('filter_value',['c1', 'c2', 'c3'])
    settings.setdefault('filter_column','columnID')
    settings.setdefault('plateID','plate1')
    settings.setdefault('metadata_files',['/home/carruthers/Documents/TGGT1_Summary.csv','/home/carruthers/Documents/TGME49_Summary.csv'])
    settings.setdefault('volcano','gene')
    settings.setdefault('toxo', True)

    if settings['regression_type'] == 'quantile':
        print(f"Using alpha as quantile for quantile regression, alpha: {settings['alpha']}")
        settings['agg_type'] = None
        print(f'agg_type set to None for quantile regression')
        
    return settings

def get_check_cellpose_models_default_settings(settings):
    """
    Set default args for checking Cellpose model predictions.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('batch_size', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', True)
    settings.setdefault('normalize', True)
    settings.setdefault('channels', [0,0])
    settings.setdefault('percentiles', None)
    settings.setdefault('invert', False)
    settings.setdefault('plot', True)
    settings.setdefault('diameter', 40)
    settings.setdefault('grayscale', True)
    settings.setdefault('remove_background', False)
    settings.setdefault('background', 100)
    settings.setdefault('Signal_to_noise', 5)
    settings.setdefault('verbose', False)
    settings.setdefault('resize', False)
    settings.setdefault('target_height', None)
    settings.setdefault('target_width', None)
    return settings

def get_identify_masks_finetune_default_settings(settings):
    """
    Set default args for identifying masks using fine-tuned Cellpose models.

    Args:
        settings (dict): Dictionary to populate with default args.

    Returns:
        dict: Updated settings dictionary.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('model_name', 'cyto')
    settings.setdefault('custom_model', None)
    settings.setdefault('channels', [0,0])
    settings.setdefault('background', 100)
    settings.setdefault('remove_background', False)
    settings.setdefault('Signal_to_noise', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('diameter', 30)
    settings.setdefault('batch_size', 50)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', False)
    settings.setdefault('verbose', False)
    settings.setdefault('normalize', True)
    settings.setdefault('percentiles', None)
    settings.setdefault('invert', False)
    settings.setdefault('resize', False)
    settings.setdefault('target_height', None)
    settings.setdefault('target_width', None)
    settings.setdefault('rescale', False)
    settings.setdefault('resample', False)
    settings.setdefault('grayscale', True)
    settings.setdefault('fill_in', True)
    return settings

q = None
expected_types = {
    "src": (str, list),
    "metadata_type": str,
    "custom_regex": (str, type(None)),
    "cov_type": (str, type(None)),
    "experiment": str,
    "channels": list,
    "magnification": int,
    "nucleus_channel": (int, type(None)),
    "nucleus_background": int,
    "nucleus_Signal_to_noise": float,
    "nucleus_CP_prob": float,
    "nucleus_FT": (int, float),
    "cell_channel": (int, type(None)),
    "cell_background": (int, float),
    "cell_Signal_to_noise": (int, float),
    "cell_CP_prob": (int, float),
    "cell_FT": (int, float),
    "pathogen_channel": (int, type(None)),
    "pathogen_background": (int, float),
    "pathogen_Signal_to_noise": (int, float),
    "pathogen_CP_prob": (int, float),
    "pathogen_FT": (int, float),
    "preprocess": bool,
    "masks": bool,
    "examples_to_plot": int,
    "randomize": bool,
    "batch_size": int,
    "timelapse": bool,
    "timelapse_displacement": int,
    "timelapse_memory": int,
    "timelapse_frame_limits": (list, type(None)),  # This can be a list of lists
    #"timelapse_frame_limits": (list, type(None)),  # This can be a list of lists
    "timelapse_remove_transient": bool,
    "timelapse_mode": str,
    "timelapse_objects": (list, type(None)),
    "fps": int,
    "remove_background": bool,
    "lower_percentile": (int, float),
    "merge_pathogens": bool,
    "normalize_plots": bool,
    "all_to_mip": bool,
    "pick_slice": bool,
    "skip_mode": str,
    "save": bool,
    "plot": bool,
    "n_jobs": int,
    "verbose": bool,
    "src": str,
    "cell_mask_dim": int,
    "cell_min_size": int,
    "cytoplasm_min_size": int,
    "nucleus_mask_dim": int,
    "nucleus_min_size": int,
    "pathogen_mask_dim": int,
    "pathogen_min_size": int,
    "save_png": bool,
    "crop_mode": list,
    "use_bounding_box": bool,
    "png_size": list,  # This can be a list of lists 
    "normalize": bool,
    "png_dims": list,
    "normalize_by": str,
    "save_measurements": bool,
    "uninfected": bool,
    "dialate_pngs": bool,
    "dialate_png_ratios": list,
    "n_jobs": int,
    "cells": list,
    "cell_loc": list,
    "pathogens": list,
    "pathogen_loc": (list, list),  # This can be a list of lists 
    "treatments": list,
    "treatment_loc": (list, list),  # This can be a list of lists
    "channel_of_interest": int,
    "compartments": list,
    "measurement": str,
    "nr_imgs": int,
    "um_per_pixel": (int, float),
    "pathogen_limit": int,
    "nuclei_limit": int,
    "filter_min_max": (list, type(None)),
    "channel_dims": list,
    "backgrounds": list,
    "background": str,
    "outline_thickness": int,
    "outline_color": str,
    "overlay_chans": list,
    "overlay": bool,
    "normalization_percentiles": list,
    "filter": bool,
    "fill_in":bool,
    "upscale": bool,
    "upscale_factor": float,
    "adjust_cells": bool,
    "row_limit": int,
    "tables": list,
    "visualize": str,
    "image_nr": int,
    "dot_size": int,
    "n_neighbors": int,
    "min_dist": float,
    "metric": str,
    "eps": float,
    "min_samples": int,
    "filter_by": str,
    "img_zoom": float,
    "plot_by_cluster": bool,
    "plot_cluster_grids": bool,
    "remove_cluster_noise": bool,
    "remove_highly_correlated": bool,
    "log_data": bool,
    "black_background": bool,
    "remove_image_canvas": bool,
    "plot_outlines": bool,
    "plot_points": bool,
    "smooth_lines": bool,
    "clustering": str,
    "exclude": (str, type(None)),
    "col_to_compare": str,
    "pos": str,
    "neg": str,
    "embedding_by_controls": bool,
    "plot_images": bool,
    "reduction_method": str,
    "save_figure": bool,
    "color_by": (str, type(None)),
    "analyze_clusters": bool,
    "resnet_features": bool,
    "test_nr": int,
    "radial_dist": bool,
    "calculate_correlation": bool,
    "manders_thresholds": list,
    "homogeneity": bool,
    "homogeneity_distances": list,
    "save_arrays": bool,
    "cytoplasm": bool,
    "merge_edge_pathogen_cells": bool,
    "cells_per_well": int,
    "pathogen_size_range": list,
    "nucleus_size_range": list,
    "cell_size_range": list,
    "pathogen_intensity_range": list,
    "nucleus_intensity_range": list,
    "cell_intensity_range": list,
    "target_intensity_min": int,
    "model_type": str,
    "heatmap_feature": str,
    "grouping": str,
    "min_max": str,
    "minimum_cell_count": int,
    "n_estimators": int,
    "test_size": float,
    "location_column": str,
    "positive_control": str,
    "negative_control": str,
    "n_repeats": int,
    "top_features": int,
    "remove_low_variance_features": bool,
    "n_jobs": int,
    "classes": list,
    "schedule": str,
    "loss_type": str,
    "image_size": int,
    "epochs": int,
    "val_split": float,
    "learning_rate": float,
    "weight_decay": float,
    "dropout_rate": float,
    "init_weights": bool,
    "amsgrad": bool,
    "use_checkpoint": bool,
    "gradient_accumulation": bool,
    "gradient_accumulation_steps": int,
    "intermedeate_save": bool,
    "pin_memory": bool,
    "n_jobs": int,
    "augment": bool,
    "target": str,
    "cell_types": list,
    "cell_plate_metadata": (list, list),
    "pathogen_types": list,
    "pathogen_plate_metadata": (list, list),  # This can be a list of lists 
    "treatment_plate_metadata": (list, list),  # This can be a list of lists
    "metadata_types": list,
    "cell_chann_dim": int,
    "nucleus_chann_dim": int,
    "pathogen_chann_dim": int,
    "plot_nr": int,
    "plot_control": bool,
    "remove_background": bool,
    "target": str,
    "upstream": str,
    "downstream": str,
    "barecode_length_1": int,
    "barecode_length_2": int,
    "chunk_size": int,
    "grna": str,
    "barcodes": str,
    "plate_dict": dict,
    "pc": str,
    "pc_loc": str,
    "nc": str,
    "nc_loc": str,
    "dependent_variable": str,
    "transform": (str, type(None)),
    "agg_type": str,
    "min_cell_count": int,
    "resize": bool,
    "denoise":bool,
    "target_height": (int, type(None)),
    "target_width": (int, type(None)),
    "rescale": bool,
    "resample": bool,
    "model_name": str,
    "Signal_to_noise": int,
    "learning_rate": float,
    "weight_decay": float,
    "batch_size": int,
    "n_epochs": int,
    "from_scratch": bool,
    "width_height": list,
    "resize": bool,
    "compression": str,
    "complevel": int,
    "gene_weights_csv": str,
    "fraction_threshold": float,
    "barcode_mapping":dict,
    "redunction_method":str,
    "mix":str,
    "model_type_ml":str,
    "exclude_conditions":list,
    "remove_highly_correlated_features":bool,
    'barcode_coordinates':list,  # This is a list of lists 
    'reverse_complement':bool,
    'file_type':str,
    'model_path':str,
    'dataset':str,
    'score_threshold':float,
    'sample':None,
    'file_metadata':(str, type(None), list),
    'apply_model_to_dataset':False,
    "train":bool,
    "test":bool,
    'train_channels':list,
    "optimizer_type":str,
    "dataset_mode":str,
    "annotated_classes":list,
    "annotation_column":str,
    "apply_model_to_dataset":bool,
    "metadata_type_by":str,
    "custom_measurement":str,
    "custom_model":bool,
    "png_type":str,
    "custom_model_path":str,
    "generate_training_dataset":bool,
    "train_DL_model":bool,
    "normalize":bool,
    "overlay":bool,
    "correlate":bool,
    "target_layer":str,
    "save_to_db":bool,
    "test_mode":bool,
    "test_images":int,
    "remove_background_cell":bool,
    "remove_background_nucleus":bool,
    "remove_background_pathogen":bool,
    "figuresize":int,
    "cmap":str,
    "pathogen_model":str,
    "normalize_input":bool,
    "filter_column":str,
    "target_unique_count":int,
    "threshold_multiplier":int,
    "threshold_method":str,
    "count_data":list,
    "score_data":list,
    "min_n":int,
    "controls":list,
    "toxo":bool,
    "volcano":str,
    "metadata_files":list,
    "filter_value":list,
    "split_axis_lims":str,
    "x_lim":(list,None),
    "log_x":bool,
    "log_y":bool,
    "reg_alpha":(int,float),
    "reg_lambda":(int,float),
    "prune_features":bool,
    "cross_validation":bool,
    "offset_start":int,
    "chunk_size":int,
    "single_direction":str,
    "delete_intermediate":bool,
    "outlier_detection":bool,
    "CP_prob":int,
    "diameter":int,
    "flow_threshold":float,
    "cell_diamiter":int,
    "nucleus_diamiter":int,
    "pathogen_diamiter":int,
    "consolidate":bool,
    'use_sam_cell':bool,
    'use_sam_nucleus':bool,
    'use_sam_pathogen':bool,
    "distance_gaussian_sigma": (int, type(None))
}

categories = {"Paths":[ "src", "grna", "barcodes", "custom_model_path", "dataset","model_path","grna_csv","row_csv","column_csv", "metadata_files", "score_data","count_data"],
             "General": ["cell_mask_dim", "cytoplasm", "cell_chann_dim", "cell_channel", "nucleus_chann_dim", "nucleus_channel", "nucleus_mask_dim", "pathogen_mask_dim", "pathogen_chann_dim", "pathogen_channel",  "test_mode", "plot", "metadata_type", "custom_regex", "experiment", "channels", "magnification", "channel_dims", "apply_model_to_dataset", "generate_training_dataset", "train_DL_model", "delete_intermediate", "uninfected", ],
             "Cellpose":["fill_in","from_scratch", "n_epochs", "width_height", "model_name", "custom_model", "resample", "rescale", "CP_prob", "flow_threshold", "percentiles", "invert", "diameter", "grayscale", "Signal_to_noise", "resize", "target_height", "target_width"],
             "Cell": ["cell_diamiter","cell_intensity_range", "cell_size_range", "cell_background", "cell_Signal_to_noise", "cell_CP_prob", "cell_FT", "remove_background_cell", "cell_min_size", "cytoplasm_min_size", "adjust_cells", "cells", "cell_loc"],
             "Nucleus": ["nucleus_diamiter","nucleus_intensity_range", "nucleus_size_range", "nucleus_background", "nucleus_Signal_to_noise", "nucleus_CP_prob", "nucleus_FT", "remove_background_nucleus", "nucleus_min_size", "nucleus_loc"],
             "Pathogen": ["pathogen_diamiter","pathogen_intensity_range", "pathogen_size_range", "pathogen_background", "pathogen_Signal_to_noise", "pathogen_CP_prob", "pathogen_FT", "pathogen_model", "remove_background_pathogen", "pathogen_min_size", "pathogens", "pathogen_loc", "pathogen_types", "pathogen_plate_metadata", ],
             "Measurements": ["remove_image_canvas", "remove_highly_correlated", "homogeneity", "homogeneity_distances", "radial_dist", "calculate_correlation", "manders_thresholds", "save_measurements", "tables", "image_nr", "dot_size", "filter_by", "remove_highly_correlated_features", "remove_low_variance_features", "channel_of_interest"],
             "Object Image": ["save_png", "dialate_pngs", "dialate_png_ratios", "png_size", "png_dims", "save_arrays", "normalize_by", "crop_mode", "use_bounding_box"],
             "Sequencing": ["outlier_detection","offset_start","chunk_size","single_direction", "signal_direction","mode","comp_level","comp_type","save_h5","expected_end","offset","target_sequence","regex", "highlight"],
             "Generate Dataset":["save_to_db","file_metadata","class_metadata", "annotation_column","annotated_classes", "dataset_mode", "metadata_type_by","custom_measurement", "sample", "size"],
             "Hyperparamiters (Training)": ["png_type", "score_threshold","file_type", "train_channels", "epochs", "loss_type", "optimizer_type","image_size","val_split","learning_rate","weight_decay","dropout_rate", "init_weights", "train", "classes", "augment", "amsgrad","use_checkpoint","gradient_accumulation","gradient_accumulation_steps","intermedeate_save","pin_memory"],
             "Hyperparamiters (Embedding)": ["visualize","n_neighbors","min_dist","metric","resnet_features","reduction_method","embedding_by_controls","col_to_compare","log_data"],
             "Hyperparamiters (Clustering)": ["eps","min_samples","analyze_clusters","clustering","remove_cluster_noise"],
             "Hyperparamiters (Regression)":["cross_validation","prune_features","reg_lambda","reg_alpha","cov_type", "plate", "other", "fraction_threshold", "alpha", "random_row_column_effects", "regression_type", "min_cell_count", "agg_type", "transform", "dependent_variable"],
             "Hyperparamiters (Activation)":["cam_type", "overlay", "correlation", "target_layer", "normalize_input"],
             "Annotation": ["filter_column", "filter_value","volcano", "toxo", "controls", "nc_loc", "pc_loc", "nc", "pc", "cell_plate_metadata","treatment_plate_metadata", "metadata_types", "cell_types", "target","positive_control","negative_control", "location_column", "treatment_loc", "channel_of_interest", "measurement", "treatments", "um_per_pixel", "nr_imgs", "exclude", "exclude_conditions", "mix", "pos", "neg"],
             "Plot": ["split_axis_lims", "x_lim","log_x","log_y", "plot_control", "plot_nr", "examples_to_plot", "normalize_plots", "cmap", "figuresize", "plot_cluster_grids", "img_zoom", "row_limit", "color_by", "plot_images", "smooth_lines", "plot_points", "plot_outlines", "black_background", "plot_by_cluster", "heatmap_feature","grouping","min_max","cmap","save_figure"],
             "Timelapse": ["timelapse", "fps", "timelapse_displacement", "timelapse_memory", "timelapse_frame_limits", "timelapse_remove_transient", "timelapse_mode", "timelapse_objects", "compartments"],
             "Advanced": ["merge_edge_pathogen_cells", "test_images", "random_test", "test_nr", "test", "test_split", "normalize", "target_unique_count","threshold_multiplier", "threshold_method", "min_n","shuffle", "target_intensity_min", "cells_per_well", "nuclei_limit", "pathogen_limit", "background", "backgrounds", "schedule", "test_size","exclude","n_repeats","top_features", "model_type_ml", "model_type","minimum_cell_count","n_estimators","preprocess", "remove_background", "normalize", "lower_percentile", "merge_pathogens", "batch_size", "filter", "save", "masks", "verbose", "randomize", "n_jobs"],
             "Beta": ["all_to_mip", "upscale", "upscale_factor", "consolidate", "distance_gaussian_sigma","use_sam_pathogen","use_sam_nucleus", "use_sam_cell", "denoise"]
             }


category_keys = list(categories.keys())

def check_settings(vars_dict, expected_types, q=None):
    """
    Validate and parse GUI-derived variable inputs according to expected types.

    Args:
        vars_dict (dict): Dictionary mapping setting keys to a tuple of (label, widget, variable, category).
        expected_types (dict): Dictionary of expected types for each setting key.
        q (multiprocessing.Queue, optional): Queue to collect error messages. If None, a new queue is created.

    Returns:
        tuple:
            - settings (dict): Parsed settings with values cast to the expected types.
            - errors (list): List of error messages describing issues with format or type mismatches.

    Notes:
        - Supports nested structures such as list of lists and dicts.
        - Handles conversions to str, int, float, bool, list, dict, and None.
        - Custom parsing is performed for complex input types like Cellpose metadata or PNG config lists.
        - Errors are collected and returned without interrupting the full parsing process.
    """
    from .gui_utils import parse_list

    if q is None:
        from multiprocessing import Queue
        q = Queue()

    settings = {}
    errors = []  # Collect errors instead of stopping at the first one

    for key, (label, widget, var, _) in vars_dict.items():
        if key not in expected_types and key not in category_keys:
            errors.append(f"Warning: Key '{key}' not found in expected types.")
            continue

        value = var.get()
        if value in ['None', '']:
            value = None

        expected_type = expected_types.get(key, str)

        try:
            if key in ["cell_plate_metadata", "timelapse_frame_limits", "png_size", "png_dims", "pathogen_plate_metadata", "treatment_plate_metadata", "timelapse_objects", "class_metadata", "crop_mode", "dialate_png_ratios"]:
                if value is None:
                    parsed_value = None
                else:
                    try:
                        parsed_value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Expected a list or list of lists but got an invalid format: {value}")

                if isinstance(parsed_value, list):
                    if all(isinstance(i, list) for i in parsed_value) or all(not isinstance(i, list) for i in parsed_value):
                        settings[key] = parsed_value
                    else:
                        raise ValueError(f"Invalid format: '{key}' contains mixed types (single values and lists).")

                else:
                    raise ValueError(f"Expected a list for '{key}', but got {type(parsed_value).__name__}.")
            
            elif expected_type == list:
                settings[key] = parse_list(value) if value else None

                if isinstance(settings[key], list) and len(settings[key]) == 1:
                    settings[key] = settings[key][0]

            elif expected_type == bool:
                settings[key] = value.lower() in ['true', '1', 't', 'y', 'yes'] if isinstance(value, str) else bool(value)
            
            elif expected_type == (int, type(None)):
                if value is None or str(value).isdigit():
                    settings[key] = int(value) if value is not None else None
                else:
                    raise ValueError(f"Expected an integer or None for '{key}', but got '{value}'.")

            elif expected_type == (float, type(None)):
                if value is None or (isinstance(value, str) and value.replace(".", "", 1).isdigit()):
                    settings[key] = float(value) if value is not None else None
                else:
                    raise ValueError(f"Expected a float or None for '{key}', but got '{value}'.")

            elif expected_type == (int, float):
                try:
                    settings[key] = float(value) if '.' in str(value) else int(value)
                except ValueError:
                    raise ValueError(f"Expected an integer or float for '{key}', but got '{value}'.")

            elif expected_type == (str, type(None)):
                settings[key] = str(value) if value is not None else None

            elif expected_type == (str, type(None), list):
                if isinstance(value, list):
                    settings[key] = parse_list(value) if value else None
                elif isinstance(value, str):
                    settings[key] = str(value)
                else:
                    settings[key] = None
            
            elif expected_type == dict:
                try:
                    if isinstance(value, str):
                        parsed_dict = ast.literal_eval(value)
                    else:
                        raise ValueError("Expected a string representation of a dictionary.")

                    if not isinstance(parsed_dict, dict):
                        raise ValueError(f"Expected a dictionary for '{key}', but got {type(parsed_dict).__name__}.")

                    settings[key] = parsed_dict
                except (ValueError, SyntaxError) as e:
                    settings[key] = {}
                    errors.append(f"Error: Invalid dictionary format for '{key}'. Expected type: dict. Error: {e}")

            elif isinstance(expected_type, tuple):
                for typ in expected_type:
                    try:
                        settings[key] = typ(value) if value else None
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError(f"Value '{value}' for '{key}' does not match any expected types: {expected_type}.")

            else:
                try:
                    settings[key] = expected_type(value) if value else None
                except (ValueError, TypeError):
                    raise ValueError(f"Expected type {expected_type.__name__} for '{key}', but got '{value}'.")

        except (ValueError, SyntaxError) as e:
            expected_type_name = ' or '.join([t.__name__ for t in expected_type]) if isinstance(expected_type, tuple) else expected_type.__name__
            errors.append(f"Error: '{key}' has invalid format. Expected type: {expected_type_name}. Got value: '{value}'. Error: {e}")

    # Send all collected errors to the queue
    for error in errors:
        q.put(error)
    return settings, errors

def generate_fields(variables, scrollable_frame):
    """
    Dynamically generate labeled input fields with tooltips in a scrollable Tkinter frame.

    Args:
        variables (dict): Dictionary mapping setting keys to (var_type, options, default_value).
        scrollable_frame (ScrollableFrame): A scrollable frame widget to place the input fields into.

    Returns:
        vars_dict (dict): Dictionary mapping each setting key to a tuple of (label, widget, variable, parent frame).

    Notes:
        - Uses `create_input_field` to generate the widgets.
        - Attaches tooltips using `spacrToolTip` where available.
        - Tooltips describe the expected format, data type, and function of each setting.
    """
    from .gui_utils import create_input_field
    from .gui_elements import spacrToolTip
    row = 1
    vars_dict = {}
    tooltips = {
        "cell_diamiter": "(int) - Diameter for cellpose objects to segment.",
        "nucleus_diamiter": "(int) - Diameter for cellpose objects to segment.",
        "pathogen_diamiter": "(int) - Diameter for cellpose objects to segment.",
        "adjust_cells": "(bool) - Adjust cell parameters for better segmentation.",
        "agg_type": "(str) - Type of aggregation to use for the data.",
        "alpha": "(float) - Alpha parameter for the regression model.",
        "all_to_mip": "(bool) - Whether to convert all images to maximum intensity projections before processing.",
        "amsgrad": "(bool) - Whether to use AMSGrad optimizer.",
        "analyze_clusters": "(bool) - Whether to analyze the resulting clusters.",
        "augment": "(dict) - Data augmentation settings.",
        "background": "(float) - Background intensity for the images.",
        "backgrounds": "(str) - Background settings for the analysis.",
        "barcodes": "(str) - Path to the file containing barcodes.",
        "batch_size": "(int) - The batch size to use for processing the images. This will determine how many images are processed at once. Images are normalized and segmented in batches. Lower if application runs out of RAM or VRAM.",
        "black_background": "(bool) - Whether to use a black background for plots.",
        "calculate_correlation": "(bool) - Whether to calculate correlations between features.",
        "cell_CP_prob": "(float) - The cellpose probability threshold for the cell channel. This will be used in cell segmentation.",
        "nucleus_CP_prob": "(float) - The cellpose probability threshold for the nucleus channel. This will be used in cell segmentation.",
        "pathogen_CP_prob": "(float) - The cellpose probability threshold for the pathogen channel. This will be used in cell segmentation.",
        "cell_FT": "(float) - The flow threshold for cell objects. This will be used to segment the cells.",
        "nucleus_FT": "(float) - The flow threshold for nucleus objects. This will be used to segment the cells.",
        "pathogen_FT": "(float) - The flow threshold for pathogen objects. This will be used to segment the cells.",
        "cell_background": "(int) - The background intensity for the cell channel. This will be used to remove background noise.",
        "nucleus_background": "(int) - The background intensity for the nucleus channel. This will be used to remove background noise.",
        "pathogen_background": "(int) - The background intensity for the pathogen channel. This will be used to remove background noise.",
        "cell_chann_dim": "(int) - Dimension of the channel to use for cell segmentation.",
        "cell_channel": "(int) - The channel to use for generatin cell masks. If None, cell masks will not be generated.",
        "nucleus_channel": "(int) - The channel to use for generatin nucleus masks. If None, nucleus masks will not be generated.",
        "pathogen_channel": "(int) - The channel to use for generatin pathogen masks. If None, pathogen masks will not be generated.",
        "cell_intensity_range": "(list) - Intensity range for cell segmentation.",
        "cell_loc": "(list) - The locations of the cell types in the images.",
        "cell_mask_dim": "(int) - The dimension of the array the cell mask is saved in (array order:channels,cell, nucleus, pathogen, cytoplasm) array starts at dimension 0.",
        "nucleus_mask_dim": "(int) - The dimension of the array the nucleus mask is saved in (array order:channels,cell, nucleus, pathogen, cytoplasm) array starts at dimension 0.",
        "cell_min_size": "(int) - The minimum size of cell objects in pixels^2.",
        "cell_plate_metadata": "(str) - Metadata for the cell plate.",
        "cell_Signal_to_noise": "(int) - The signal-to-noise ratio for the cell channel. This will be used to determine the range of intensities to normalize images to for cell segmentation.",
        "cell_size_range": "(list) - Size range for cell segmentation.",
        "cell_types": "(list) - Types of cells to include in the analysis.",
        "cells": "(list of lists) - The cell types to include in the analysis.",
        "cells_per_well": "(int) - Number of cells per well.",
        "channel_dims": "(list) - The dimensions of the image channels.",
        "channel_of_interest": "(int) - The channel of interest to use for the analysis.",
        "channels": "(list) - List of channels to use for the analysis. The first channel is 0, the second is 1, and so on. For example, [0,1,2] will use channels 0, 1, and 2.",
        "chunk_size": "(int) - Chunk size for processing the sequencing data.",
        "classes": "(list) - Classes to include in the training.",
        "class_1_threshold": "(float) - Threshold for class 1 classification.",
        "clustering": "(str) - Clustering algorithm to use.",
        "col_to_compare": "(str) - Column to compare in the embeddings.",
        "color_by": "(str) - Coloring scheme for the plots.",
        "compartments": "(list) - The compartments to measure in the images.",
        "consolidate": "(bool) - Consolidate image files from subfolders into one folder named consolidated.",
        "CP_prob": "(float) - Cellpose probability threshold for segmentation.",
        "crop_mode": "(str) - Mode to use for cropping images (cell, nucleus, pathogen, cytoplasm).",
        "custom_model": "(str) - Path to a custom Cellpose model.",
        "custom_regex": "(str) - Custom regex pattern to extract metadata from the image names. This will only be used if 'custom' or 'auto' is selected for 'metadata_type'.",
        "cytoplasm": "(bool) - Whether to segment the cytoplasm (Cell - Nucleus + Pathogen).",
        "cytoplasm_min_size": "(int) - The minimum size of cytoplasm objects in pixels^2.",
        "nucleus_min_size": "(int) - The minimum size of nucleus objects in pixels^2.",
        "normalize_by": "(str) - Normalize cropped png images by png or by field of view.",
        "dependent_variable": "(str) - The dependent variable for the regression analysis.",
        "delete_intermediate": "(bool) - Delete intermediate folders (stack, channel, masks).",
        "diameter": "(float) - Diameter of the objects to segment.",
        "dialate_png_ratios": "(list) - The ratios to use for dilating the PNG images. This will determine the amount of dilation applied to the images before cropping.",
        "dialate_pngs": "(bool) - Whether to dilate the PNG images before saving.",
        "dot_size": "(int) - Size of dots in scatter plots.",
        "downstream": "(str) - Downstream region for sequencing analysis.",
        "dropout_rate": "(float) - Dropout rate for training.",
        "eps": "(float) - Epsilon parameter for clustering.",
        "epochs": "(int) - Number of epochs for training the deep learning model.",
        "examples_to_plot": "(int) - The number of images to plot for each segmented object. This will be used to visually inspect the segmentation results and normalization.",
        "exclude": "(list) - Conditions to exclude from the analysis.",
        "exclude_conditions": "(list) - Specific conditions to exclude from the analysis.",
        "experiment": "(str) - Name of the experiment. This will be used to name the output files.",
        "figuresize": "(tuple) - Size of the figures to plot.",
        "filter": "(dict) - Filter settings for the analysis.",
        "filter_by": "(str) - Feature to filter the data by.",
        "fill_in": "(bool) - Whether to fill in the segmented objects.",
        "flow_threshold": "(float) - Flow threshold for segmentation.",
        "fps": "(int) - Frames per second of the automatically generated timelapse movies.",
        "fraction_threshold": "(float) - Threshold for the fraction of cells to consider in the analysis.",
        "from_scratch": "(bool) - Whether to train the Cellpose model from scratch.",
        "gene_weights_csv": "(str) - Path to the CSV file containing gene weights.",
        "gradient_accumulation": "(bool) - Whether to use gradient accumulation.",
        "gradient_accumulation_steps": "(int) - Number of steps for gradient accumulation.",
        "grayscale": "(bool) - Whether to process the images in grayscale.",
        "grna": "(str) - Path to the file containing gRNA sequences.",
        "grouping": "(str) - Grouping variable for plotting.",
        "heatmap_feature": "(str) - Feature to use for generating heatmaps.",
        "homogeneity": "(float) - Measure of homogeneity for the objects.",
        "homogeneity_distances": "(list) - Distances to use for measuring homogeneity.",
        "image_nr": "(int) - Number of images to process.",
        "image_size": "(int) - Size of the images for training.",
        "img_zoom": "(float) - Zoom factor for the images in plots.",
        "nuclei_limit": "(int) - Whether to include multinucleated cells in the analysis.",
        "pathogen_limit": "(int) - Whether to include multi-infected cells in the analysis.",
        "uninfected": "(bool) - Whether to include uninfected cells in the analysis.",
        "init_weights": "(bool) - Whether to initialize weights for the model.",
        "src": "(str) - Path to the folder containing the images.",
        "intermedeate_save": "(bool) - Whether to save intermediate results.",
        "invert": "(bool) - Whether to invert the image intensities.",
        "learning_rate": "(float) - Learning rate for training.",
        "location_column": "(str) - Column name for the location information.",
        "log_data": "(bool) - Whether to log-transform the data.",
        "lower_percentile": "(float) - The lower quantile to use for normalizing the images. This will be used to determine the range of intensities to normalize images to.",
        "magnification": "(int) - At what magnification the images were taken. This will be used to determine the size of the objects in the images.",
        "manders_thresholds": "(list) - Thresholds for Manders' coefficients.",
        "mask": "(bool) - Whether to generate masks for the segmented objects. If True, masks will be generated for the nucleus, cell, and pathogen.",
        "measurement": "(str) - The measurement to use for the analysis.",
        "metadata_type": "(str) - Type of metadata to expect in the images. If 'custom' is selected, you can provide a custom regex pattern to extract metadata from the image names. auto will attempt to automatically extract metadata from the image names. cellvoyager and cq1 will use the default metadata extraction for CellVoyager and CQ1 images.",
        "metadata_types": "(list) - Types of metadata to include in the analysis.",
        "merge_edge_pathogen_cells": "(bool) - Whether to merge cells that share pathogen objects.",
        "merge_pathogens": "(bool) - Whether to merge pathogen objects that share more than 75 percent of their perimeter.",
        "metric": "(str) - Metric to use for UMAP.",
        "min_cell_count": "(int) - Minimum number of cells required for analysis.",
        "min_dist": "(float) - Minimum distance for UMAP.",
        "min_max": "(tuple) - Minimum and maximum values for normalizing plots.",
        "min_samples": "(int) - Minimum number of samples for clustering.",
        "mix": "(dict) - Mixing settings for the samples.",
        "model_name": "(str) - Name of the Cellpose model.",
        "model_type": "(str) - Type of model to use for the analysis.",
        "model_type_ml": "(str) - Type of model to use for machine learning.",
        "nc": "(str) - Negative control identifier.",
        "nc_loc": "(str) - Location of the negative control in the images.",
        "negative_control": "(str) - Identifier for the negative control.",
        "n_estimators": "(int) - Number of estimators for the model.",
        "n_epochs": "(int) - Number of epochs for training the Cellpose model.",
        "n_jobs": "(int) - The number of n_jobs to use for processing the images. This will determine how many images are processed in parallel. Increase to speed up processing.",
        "n_neighbors": "(int) - Number of neighbors for UMAP.",
        "n_repeats": "(int) - Number of repeats for the pathogen plate.",
        "pathogen_Signal_to_noise": "(int) - The signal-to-noise ratio for the pathogen channel. This will be used to determine the range of intensities to normalize images to for pathogen segmentation.",
        "nucleus_Signal_to_noise": "(int) - The signal-to-noise ratio for the nucleus channel. This will be used to determine the range of intensities to normalize images to for nucleus segmentation.",
        "pathogen_size_range": "(list) - Size range for pathogen segmentation.",
        "pathogen_types": "(list) - Types of pathogens to include in the analysis.",
        "pc": "(str) - Positive control identifier.",
        "pc_loc": "(str) - Location of the positive control in the images.",
        "percentiles": "(list) - Percentiles to use for normalizing the images.",
        "pin_memory": "(bool) - Whether to pin memory for the data loader.",
        "plate": "(str) - Plate identifier for the experiment.",
        "plate_dict": "(dict) - Dictionary of plate metadata.",
        "plot": "(bool) - Whether to plot the results.",
        "plot_by_cluster": "(bool) - Whether to plot images by clusters.",
        "plot_cluster_grids": "(bool) - Whether to plot grids of clustered images.",
        "plot_control": "(dict) - Control settings for plotting.",
        "plot_images": "(bool) - Whether to plot images.",
        "plot_nr": "(int) - Number of plots to generate.",
        "plot_outlines": "(bool) - Whether to plot outlines of segmented objects.",
        "png_dims": "(list) - The dimensions of the PNG images to save. This will determine the dimensions of the saved images. Maximum of 3 dimensions e.g. [1,2,3].",
        "png_size": "(list) - The size of the PNG images to save. This will determine the size of the saved images.",
        "positive_control": "(str) - Identifier for the positive control.",
        "preprocess": "(bool) - Whether to preprocess the images before segmentation. This includes background removal and normalization. Set to False only if this step has already been done.",
        "radial_dist": "(list) - Radial distances for measuring features.",
        "random_test": "(bool) - Whether to randomly select images for testing.",
        "randomize": "(bool) - Whether to randomize the order of the images before processing. Recommended to avoid bias in the segmentation.",
        "regression_type": "(str) - Type of regression to perform.",
        "remove_background": "(bool) - Whether to remove background noise from the images. This will help improve the quality of the segmentation.",
        "remove_background_cell": "(bool) - Whether to remove background noise from the cell channel.",
        "remove_background_nucleus": "(bool) - Whether to remove background noise from the nucleus channel.",
        "remove_background_pathogen": "(bool) - Whether to remove background noise from the pathogen channel.",
        "remove_cluster_noise": "(bool) - Whether to remove noise from the clusters.",
        "remove_highly_correlated": "(bool) - Whether to remove highly correlated features.",
        "remove_highly_correlated_features": "(bool) - Whether to remove highly correlated features from the analysis.",
        "remove_image_canvas": "(bool) - Whether to remove the image canvas after plotting.",
        "remove_low_variance_features": "(bool) - Whether to remove low variance features from the analysis.",
        "random_row_column_effects": "(bool) - Whether to remove row and column effects from the data.",
        "resize": "(bool) - Resize factor for the images.",
        "resample": "(bool) - Whether to resample the images during processing.",
        "rescale": "(float) - Rescaling factor for the images.",
        "reduction_method": "(str) - Dimensionality reduction method to use ().",
        "resnet_features": "(bool) - Whether to use ResNet features for embedding.",
        "row_limit": "(int) - Limit on the number of rows to plot.",
        "save": "(bool) - Whether to save the results to disk.",
        "save_arrays": "(bool) - Whether to save arrays of segmented objects.",
        "save_figure": "(bool) - Whether to save the generated figures.",
        "save_measurements": "(bool) - Whether to save the measurements to disk.",
        "save_png": "(bool) - Whether to save the segmented objects as PNG images.",
        "schedule": "(str) - Schedule for processing the data.",
        "Signal_to_noise": "(int) - Signal-to-noise ratio for the images.",
        "skip_mode": "(str) - The mode to use for skipping images. This will determine how to handle images that cannot be processed.",
        "smooth_lines": "(bool) - Whether to smooth lines in the plots.",
        "src": "(str, path) - Path to source directory.",
        "target": "(str) - Target variable for the analysis.",
        "target_height": "(int) - Target height for resizing the images.",
        "target_intensity_min": "(float) - Minimum intensity for the target objects.",
        "target_width": "(int) - Target width for resizing the images.",
        "tables": "(list) - Tables to include in the analysis.",
        "test": "(bool) - Whether to run the pipeline in test mode.",
        "test_images": "(list) - List of images to use for testing.",
        "test_mode": "(bool) - Mode to use for testing the analysis pipeline.",
        "test_nr": "(int) - Number of test images.",
        "test_size": "(float) - Size of the test set.",
        "treatment_loc": "(list) - The locations of the treatments in the images.",
        "treatments": "(list) - The treatments to include in the analysis.",
        "top_features": "(int) - Top features to include in the analysis.",
        "train": "(bool) - Whether to train the model.",
        "transform": "(dict) - Transformation to apply to the data.",
        "upscale": "(bool) - Whether to upscale the images.",
        "upscale_factor": "(float) - Factor by which to upscale the images.",
        "upstream": "(str) - Upstream region for sequencing analysis.",
        "val_split": "(float) - Validation split ratio.",
        "visualize": "(bool) - Whether to visualize the embeddings.",
        "verbose": "(bool) - Whether to print verbose output during processing.",
        "weight_decay": "(float) - Weight decay for regularization.",
        "width_height": "(tuple) - Width and height of the input images.",
        "barcode_coordinates": "(list of lists) - Coordinates of the barcodes in the sequence.",
        "barcode_mapping": "dict - names and barecode csv files",
        "compression": "str - type of compression (e.g. zlib)",
        "complevel": "int - level of compression (0-9). Higher is slower and yealds smaller files",
        "file_type": "str - type of file to process",
        "model_path": "str - path to the model",
        "dataset": "str - file name of the tar file with image dataset",
        "score_threshold": "float - threshold for classification",
        "sample": "str - number of images to sample for tar dataset (including both classes). Default: None",
        "file_metadata": "str or list of strings - string(s) that must be present in image path to be included in the dataset",
        "apply_model_to_dataset": "bool - whether to apply model to the dataset",
        "train_channels": "list - channels to use for training",
        "dataset_mode": "str - How to generate train/test dataset.",
        "annotated_classes": "list - list of numbers in annotation column.",
        "um_per_pixel": "(float) - The micrometers per pixel for the images.",
        "pathogen_model": "(str) - use a custom cellpose model to detect pathogen objects.",
        "timelapse_displacement": "(int) - Displacement for timelapse tracking.",
        "timelapse_memory": "(int) - Memory for timelapse tracking.",
        "timelapse_mode": "(str) - Mode for timelapse tracking, trackpy or btrack.",
        "timelapse_frame_limits": "(list) - Frame limits for timelapse tracking [start,end].",
        "timelapse_objects": "(list) - Objects to track in the timelapse, cells, nuclei, or pathogens.",
        "timelapse_remove_transient": "(bool) - Whether to remove transient objects in the timelapse.",
        "masks": "(bool) - Whether to generate masks for the segmented objects.",
        "timelapse": "(bool) - Whether to analyze images as a timelapse.",
        "pathogen_min_size": "(int) - The minimum size of pathogen objects in pixels^2.",
        "pathogen_mask_dim": "(int) - The dimension of the array the pathogen mask is saved in (array order:channels,cell, nucleus, pathogen, cytoplasm) array starts at dimension 0.",
        "use_bounding_box": "(bool) - Whether to use the bounding box for cropping the images.",
        "plot_points": "(bool) - Whether to plot scatterplot points.",
        "embedding_by_controls": "(bool) - Use the controlls to greate the embedding, then apply this embedding to all of the data.",
        "pos": "(str) - Positive control identifier.",
        "neg": "(str) - Negative control identifier.",
        "minimum_cell_count": "(int) - Minimum number of cells/well. if number of cells < minimum_cell_count, the well is excluded from the analysis.",
        "highlight": "(str) - highlight genes/grnas containing this string.",
        "pathogen_plate_metadata": "(str) - Metadata for the pathogen plate.",
        "treatment_plate_metadata": "(str) - Metadata for the treatment plate.",
        "regex": "(str) - Regular expression to use.",
        "target_sequence": "(str) - The DNA sequence to look for that the consensus sequence will start with directly downstream of the first barcode.",
        "offset": "(int) - The offset to use for the consensus sequence, e.g. -8 if the barecode is 8 bases before target_sequence.",
        "expected_end": "(int) - The expected length of the sequence from the start of the first barcode to the end of the last.",
        "column_csv": "(path) - path to the csv file containing column barcodes.",
        "row_csv": "(path) - path to the csv file containing row barcodes.",
        "grna_csv": "(path) - path to the csv file containing gRNA sequences.",
        "save_h5": "(bool) - Whether to save the results to an HDF5 file. (this generates a large file, if compression is used this can be very time consuming)",
        "comp_type": "(str) - Compression type for the HDF5 file (e.g. zlib).",
        "comp_level": "(int) - Compression level for the HDF5 file (0-9). Higher is slower and yields smaller files.",
        "mode": "(str) - Mode to use for sequence analysis (either single for R1 or R2 fastq files or paired for the combination of R1 and R2).",
        "signal_direction": "(str) - Direction of fastq file (R1 or R2). only relevent when mode is single.",
        "custom_model_path": "(str) - Path to the custom model to finetune.",
        "cam_type": "(str) - Choose between: gradcam, gradcam_pp, saliency_image, saliency_channel to generate activateion maps of DL models",
        "target_layer": "(str) - Only used for gradcam and gradcam_pp. The layer to use for the activation map.",
        "normalize": "(bool) - Normalize images before overlayng the activation maps.",
        "overlay": "(bool) - Overlay activation maps on the images.",
        "shuffle": "(bool) - Shuffle the dataset bufore generating the activation maps",
        "correlation": "(bool) - Calculate correlation between image channels and activation maps. Data is saved to .db.",
        "use_sam_cell": "(bool) - Whether to use SAM for cell segmentation.",
        "use_sam_nucleus": "(bool) - Whether to use SAM for nucleus segmentation.",
        "use_sam_pathogen": "(bool) - Whether to use SAM for pathogen segmentation.",
        "normalize_input": "(bool) - Normalize the input images before passing them to the model.",
        "normalize_plots": "(bool) - Normalize images before plotting.",
    }
    
    for key, (var_type, options, default_value) in variables.items():
        label, widget, var, frame = create_input_field(scrollable_frame.scrollable_frame, key, row, var_type, options, default_value)
        vars_dict[key] = (label, widget, var, frame)  # Store the label, widget, and variable
        
        # Add tooltip to the label if it exists in the tooltips dictionary
        if key in tooltips:
            spacrToolTip(label, tooltips[key])

        row += 1
        
    return vars_dict

descriptions = {
    'mask': "\n\nHelp:\n- Generate Cells, Nuclei, Pathogens, and Cytoplasm masks from intensity images in src.\n- To ensure that spacr is installed correctly:\n- 1. Downloade the training set (click Download).\n- 2. Import settings (click settings navigate to downloaded dataset settings folder and import preprocess_generate_masks_settings.csv).\n- 3. Run the module.\n- 4. Proceed to the Measure module (click Measure in the menue bar).\n- For further help, click the Help button in the menue bar.",
    
    'measure': "Capture Measurements from Cells, Nuclei, Pathogens, and Cytoplasm objects. Generate single object PNG images for one or several objects. (Requires masks from the Mask module). Function: measure_crop from spacr.measure.\n\nKey Features:\n- Comprehensive Measurement Capture: Obtain detailed measurements for various cellular components, including area, perimeter, intensity, and more.\n- Image Generation: Create high-resolution PNG images of individual objects, facilitating further analysis and visualization.\n- Mask Dependency: Requires accurate masks generated by the Mask module to ensure precise measurements.",
    
    'classify': "Train and Test any Torch Computer vision model. (Requires PNG images from the Measure module). Function: train_test_model from spacr.deep_spacr.\n\nKey Features:\n- Deep Learning Integration: Train and evaluate state-of-the-art Torch models for various classification tasks.\n- Flexible Training: Supports a wide range of Torch models, allowing customization based on specific research needs.\n- Data Requirement: Requires PNG images generated by the Measure module for training and testing.",
    
    'umap': "Generate UMAP or tSNE embeddings and represent points as single cell images. (Requires measurements.db and PNG images from the Measure module). Function: generate_image_umap from spacr.core.\n\nKey Features:\n- Dimensionality Reduction: Employ UMAP or tSNE algorithms to reduce high-dimensional data into two dimensions for visualization.\n- Single Cell Representation: Visualize embedding points as single cell images, providing an intuitive understanding of data clusters.\n- Data Integration: Requires measurements and images generated by the Measure module, ensuring comprehensive data representation.",
    
    'train_cellpose': "Train custom Cellpose models for your specific dataset. Function: train_cellpose_model from spacr.core.\n\nKey Features:\n- Custom Model Training: Train Cellpose models on your dataset to improve segmentation accuracy.\n- Data Adaptation: Tailor the model to handle specific types of biological samples more effectively.\n- Advanced Training Options: Supports various training parameters and configurations for optimized performance.",
    
    'ml_analyze': "Perform machine learning analysis on your data. Function: ml_analysis_tools from spacr.ml.\n\nKey Features:\n- Comprehensive Analysis: Utilize a suite of machine learning tools for data analysis.\n- Customizable Workflows: Configure and run different ML algorithms based on your research requirements.\n- Integration: Works seamlessly with other modules to analyze data produced from various steps.",
    
    'cellpose_masks': "Generate masks using Cellpose for all images in your dataset. Function: generate_masks from spacr.cellpose.\n\nKey Features:\n- Batch Processing: Generate masks for large sets of images efficiently.\n- Robust Segmentation: Leverage Cellpose's capabilities for accurate segmentation across diverse samples.\n- Automation: Automate the mask generation process for streamlined workflows.",
    
    'cellpose_all': "Run Cellpose on all images in your dataset and obtain masks and measurements. Function: cellpose_analysis from spacr.cellpose.\n\nKey Features:\n- End-to-End Analysis: Perform both segmentation and measurement extraction in a single step.\n- Efficiency: Process entire datasets with minimal manual intervention.\n- Comprehensive Output: Obtain detailed masks and corresponding measurements for further analysis.",
    
    'map_barcodes': "\n\nHelp:\n- 1 .Generate consensus read fastq files from R1 and R2 files.\n- 2. Map barcodes from sequencing data for identification and tracking of samples.\n- 3. Run the module to extract and map barcodes from your FASTQ files in chunks.\n- Prepare your barcode CSV files with the appropriate 'name' and 'sequence' columns.\n- Configure the barcode settings (coordinates and reverse complement flags) according to your experimental setup.\n- For further help, click the Help button in the menu bar.",

    'regression': "Perform regression analysis on your data. Function: regression_tools from spacr.analysis.\n\nKey Features:\n- Statistical Analysis: Conduct various types of regression analysis to identify relationships within your data.\n- Flexible Options: Supports multiple regression models and configurations.\n- Data Insight: Gain deeper insights into your dataset through advanced regression techniques.",
    
    'activation': "",

    'analyze_plaques': "Analyze plaque images to quantify plaque properties. Function: analyze_plaques from spacr.analysis.\n\nKey Features:\n- Plaque Analysis: Quantify plaque properties such as size, intensity, and shape.\n- Batch Processing: Analyze multiple plaque images efficiently.\n- Visualization: Generate visualizations to represent plaque data and patterns.",

    'recruitment': "Analyze recruitment data to understand sample recruitment dynamics. Function: recruitment_analysis_tools from spacr.analysis.\n\nKey Features:\n- Recruitment Analysis: Investigate and analyze the recruitment of samples over time or conditions.\n- Visualization: Generate visualizations to represent recruitment trends and patterns.\n- Integration: Utilize data from various sources for a comprehensive recruitment analysis."
}

def set_annotate_default_settings(settings):
    """
    Set default arguments for image annotation and visualization in the annotation tool.

    Args:
        settings (dict): Dictionary to update with default annotation settings.

    Returns:
        dict: Updated dictionary containing defaults for:
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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('image_type', 'cell_png')
    settings.setdefault('channels', "r,g,b")
    settings.setdefault('img_size', 200)
    settings.setdefault('annotation_column', 'test')
    settings.setdefault('normalize', 'False')
    settings.setdefault('normalize_channels', "r,g,b")
    settings.setdefault('outline', None)
    settings.setdefault('outline_threshold_factor', 1)
    settings.setdefault('outline_sigma', 1)
    settings.setdefault('percentiles', [2, 98])
    settings.setdefault('measurement', '') #'cytoplasm_channel_3_mean_intensity,pathogen_channel_3_mean_intensity')
    settings.setdefault('threshold', '') #'2')
    return settings

def set_default_generate_barecode_mapping(settings={}):
    """
    Sets default settings for barcode mapping from sequencing data.

    Args:
        settings (dict): A dictionary to populate with default parameters if keys are missing.

    Returns:
        dict: The updated settings dictionary with default values applied.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('regex', '^(?P<column>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<row>.{8}).*'),
    settings.setdefault('target_sequence', 'TGCTGTTTCCAGCATAGCTCTTAAAC')
    settings.setdefault('offset_start', -8)
    settings.setdefault('expected_end', 89)
    settings.setdefault('column_csv', '/home/carruthers/Documents/column_barcodes.csv')
    settings.setdefault('grna_csv', '/home/carruthers/Documents/grna_barcodes.csv')
    settings.setdefault('row_csv', '/home/carruthers/Documents/row_barcodes.csv')
    settings.setdefault('save_h5', True)
    settings.setdefault('comp_type', 'zlib')
    settings.setdefault('comp_level', 5)
    settings.setdefault('chunk_size', 100000)
    settings.setdefault('n_jobs', None)
    settings.setdefault('mode', 'paired')
    settings.setdefault('single_direction', 'R1')
    settings.setdefault('test', False)
    settings.setdefault('fill_na', False)
    return settings

def get_default_generate_activation_map_settings(settings):
    """
    Sets default settings for generating activation maps (e.g., Grad-CAM) from a trained deep learning model.

    Args:
        settings (dict): A dictionary to populate with default parameters if keys are missing.

    Returns:
        dict: The updated settings dictionary with default values applied.

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
    """
    settings.setdefault('dataset', 'path')
    settings.setdefault('model_type', 'maxvit')
    settings.setdefault('model_path', 'path')
    settings.setdefault('image_size', 224)
    settings.setdefault('batch_size', 64)
    settings.setdefault('normalize', True)
    settings.setdefault('cam_type', 'gradcam')
    settings.setdefault('target_layer', None)
    settings.setdefault('plot', False)
    settings.setdefault('save', True)
    settings.setdefault('normalize_input', True)
    settings.setdefault('channels', [1,2,3])
    settings.setdefault('overlay', True)
    settings.setdefault('shuffle', True)
    settings.setdefault('correlation', True)
    settings.setdefault('manders_thresholds', [15,50, 75])
    settings.setdefault('n_jobs', None)
    return settings

def get_analyze_plaque_settings(settings):
    """
    Sets default settings for analyzing plaque formation using Cellpose segmentation and related image processing tools.

    Args:
        settings (dict): A dictionary to populate with default parameters if keys are missing.

    Returns:
        dict: The updated settings dictionary with default values applied.

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
    """
    settings.setdefault('src', 'path')
    settings.setdefault('masks', True)
    settings.setdefault('background', 200)
    settings.setdefault('Signal_to_noise', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('diameter', 30)
    settings.setdefault('batch_size', 50)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', True)
    settings.setdefault('verbose', True)
    settings.setdefault('resize', True)
    settings.setdefault('target_height', 1120)
    settings.setdefault('target_width', 1120)
    settings.setdefault('rescale', False)
    settings.setdefault('resample', False)
    settings.setdefault('fill_in', True)
    return settings

def set_graph_importance_defaults(settings):
    """
    Sets default parameters for plotting graph-based feature importance across groups.

    Args:
        settings (dict): Dictionary to be populated with default values if keys are missing.

    Returns:
        dict: Updated settings dictionary.

    Default Settings:
        - csvs (str or list of str): Path(s) to CSV files containing importance scores.
        - grouping_column (str): Column name used to group features for plotting (e.g., 'compartment').
        - data_column (str): Column name containing the importance values to be plotted.
        - graph_type (str): Type of graph to plot. Options may include 'jitter_bar', 'violin', etc.
        - save (bool): Whether to save the plot to disk.
    """
    settings.setdefault('csvs','list of paths')
    settings.setdefault('grouping_column','compartment')
    settings.setdefault('data_column','compartment_importance_sum')
    settings.setdefault('graph_type','jitter_bar')
    settings.setdefault('save',False)
    return settings

def set_interperate_vision_model_defaults(settings):
    """
    Sets default parameters for interpreting vision model outputs using feature importance methods.

    Args:
        settings (dict): Dictionary to populate with default values for any missing keys.

    Returns:
        dict: Updated settings dictionary with default interpretation parameters.

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
    """
    settings.setdefault('src','path')
    settings.setdefault('scores','path')
    settings.setdefault('tables',['cell', 'nucleus', 'pathogen','cytoplasm'])
    settings.setdefault('feature_importance',True)
    settings.setdefault('permutation_importance',False)
    settings.setdefault('shap',True)
    settings.setdefault('save',False)
    settings.setdefault('nuclei_limit',1000)
    settings.setdefault('pathogen_limit',1000)
    settings.setdefault('top_features',30)
    settings.setdefault('shap_sample',True)
    settings.setdefault('n_jobs',-1)
    settings.setdefault('shap_approximate',True)
    settings.setdefault('score_column','cv_predictions')
    return settings

def set_analyze_endodyogeny_defaults(settings):
    """
    Sets default parameters for analyzing endodyogeny (intracellular replication) from segmented object tables.

    Args:
        settings (dict): Dictionary to populate with default values for any missing keys.

    Returns:
        dict: Updated settings dictionary with defaults for endodyogeny analysis.

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
    """
    settings.setdefault('src','path')
    settings.setdefault('tables',['cell', 'nucleus', 'pathogen', 'cytoplasm'])
    settings.setdefault('cell_types',['Hela'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['nc', 'pc'])
    settings.setdefault('pathogen_plate_metadata',[['c1'], ['c2']])
    settings.setdefault('treatments',None)
    settings.setdefault('treatment_plate_metadata',None)
    settings.setdefault('min_area_bin',500)
    settings.setdefault('group_column','pathogen')
    settings.setdefault('compartment','pathogen')
    settings.setdefault('pathogen_limit',1)
    settings.setdefault('nuclei_limit',10)
    settings.setdefault('level','object')
    settings.setdefault('um_per_px',0.1)
    settings.setdefault('max_bins',None)
    settings.setdefault('save',False)
    settings.setdefault('change_plate',False)
    settings.setdefault('cmap','viridis')
    settings.setdefault('verbose',False)
    return settings

def set_analyze_class_proportion_defaults(settings):
    """
    Sets default parameters for analyzing the proportion of classification labels
    across experimental conditions or cell types.

    Args:
        settings (dict): Dictionary to populate with default values for any missing keys.

    Returns:
        dict: Updated settings dictionary with defaults for class proportion analysis.

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
    """
    settings.setdefault('src','path')
    settings.setdefault('tables',['cell', 'nucleus', 'pathogen', 'cytoplasm'])
    settings.setdefault('cell_types',['Hela'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['nc','pc'])
    settings.setdefault('pathogen_plate_metadata',[['c1'],['c2']])
    settings.setdefault('treatments',None)
    settings.setdefault('treatment_plate_metadata',None)
    settings.setdefault('group_column','condition')
    settings.setdefault('class_column','test')
    settings.setdefault('pathogen_limit',1000)
    settings.setdefault('nuclei_limit',1000)
    settings.setdefault('level','well')
    settings.setdefault('save',False)
    settings.setdefault('verbose', False)
    return settings

def get_plot_data_from_csv_default_settings(settings):
    """
    Sets default parameters for plotting data from a CSV file.

    Args:
        settings (dict): Dictionary to populate with default values for any missing keys.

    Returns:
        dict: Updated settings dictionary with default plotting parameters.

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
    """
    settings.setdefault('src','path')
    settings.setdefault('data_column','choose column')
    settings.setdefault('grouping_column','choose column')
    settings.setdefault('graph_type','violin')
    settings.setdefault('save',False)
    settings.setdefault('y_lim',None)
    settings.setdefault('log_y',False)
    settings.setdefault('log_x',False)
    settings.setdefault('keep_groups',None)
    settings.setdefault('representation','well')
    settings.setdefault('theme','dark')
    settings.setdefault('remove_outliers',False)
    settings.setdefault('verbose',False)
    return settings