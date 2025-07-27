spacr.core
==========

.. py:module:: spacr.core




Module Contents
---------------

.. py:function:: preprocess_generate_masks(settings)

   Preprocess image data and generate Cellpose segmentation masks for cells, nuclei, and pathogens.

   This function supports preprocessing, metadata conversion, Cellpose-based mask generation, optional
   mask adjustment, result plotting, and intermediate file cleanup. It handles batch operations and
   supports advanced timelapse and channel-specific configurations.

   :param settings: Dictionary containing the following keys:

                    General settings:
                        - src (str or list): Path(s) to input folders. Required.
                        - denoise (bool): Apply denoising during preprocessing. Default is False.
                        - delete_intermediate (bool): Delete intermediate files after processing. Default is False.
                        - preprocess (bool): Perform preprocessing. Default is True.
                        - masks (bool): Generate masks using Cellpose. Default is True.
                        - save (bool or list of bool): Whether to save outputs per object type. Default is True.
                        - consolidate (bool): Consolidate input folder structure. Default is False.
                        - batch_size (int): Number of files processed per batch. Default is 50.
                        - test_mode (bool): Enable test mode with limited data. Default is False.
                        - test_images (int): Number of test images to use. Default is 10.
                        - magnification (int): Magnification of input data. Default is 20.
                        - custom_regex (str or None): Regex for filename parsing in auto metadata mode.
                        - metadata_type (str): Metadata type; "cellvoyager" or "auto". Default is "cellvoyager".
                        - n_jobs (int): Number of parallel processes. Default is os.cpu_count() - 4.
                        - randomize (bool): Randomize processing order. Default is True.
                        - verbose (bool): Print full settings table. Default is True.

                    Channel background correction:
                        - remove_background_cell (bool): Remove background from cell channel. Default is False.
                        - remove_background_nucleus (bool): Remove background from nucleus channel. Default is False.
                        - remove_background_pathogen (bool): Remove background from pathogen channel. Default is False.

                    Channel diameter and index settings:
                        - cell_diamiter (float or None): Cell diameter estimate for Cellpose.
                        - nucleus_diamiter (float or None): Nucleus diameter estimate for Cellpose.
                        - pathogen_diamiter (float or None): Pathogen diameter estimate for Cellpose.
                        - cell_channel (int or None): Channel index for cell. Default is None.
                        - nucleus_channel (int or None): Channel index for nucleus. Default is None.
                        - pathogen_channel (int or None): Channel index for pathogen. Default is None.
                        - channels (list): List of channel indices to include. Default is [0, 1, 2, 3].

                    Cellpose parameters:
                        - pathogen_background (float): Background intensity for pathogen. Default is 100.
                        - pathogen_Signal_to_noise (float): SNR threshold for pathogen. Default is 10.
                        - pathogen_CP_prob (float): Cellpose probability threshold for pathogen. Default is 0.
                        - cell_background (float): Background intensity for cell. Default is 100.
                        - cell_Signal_to_noise (float): SNR threshold for cell. Default is 10.
                        - cell_CP_prob (float): Cellpose probability threshold for cell. Default is 0.
                        - nucleus_background (float): Background intensity for nucleus. Default is 100.
                        - nucleus_Signal_to_noise (float): SNR threshold for nucleus. Default is 10.
                        - nucleus_CP_prob (float): Cellpose probability threshold for nucleus. Default is 0.
                        - nucleus_FT (float): Intensity scaling factor for nucleus. Default is 1.0.
                        - cell_FT (float): Intensity scaling factor for cell. Default is 1.0.
                        - pathogen_FT (float): Intensity scaling factor for pathogen. Default is 1.0.

                    Plotting settings:
                        - plot (bool): Enable plotting. Default is False.
                        - figuresize (int or float): Figure size for plots. Default is 10.
                        - cmap (str): Colormap used for plotting. Default is "inferno".
                        - normalize (bool): Normalize image intensities before processing. Default is True.
                        - normalize_plots (bool): Normalize intensity for plotting. Default is True.
                        - examples_to_plot (int): Number of examples to plot. Default is 1.

                    Analysis settings:
                        - pathogen_model (str or None): Custom model for pathogen ("toxo_pv_lumen" or "toxo_cyto").
                        - merge_pathogens (bool): Whether to merge multiple pathogen types. Default is False.
                        - filter (bool): Apply percentile filter. Default is False.
                        - lower_percentile (float): Lower percentile for intensity filtering. Default is 2.

                    Timelapse settings:
                        - timelapse (bool): Enable timelapse mode. Default is False.
                        - fps (int): Frames per second for timelapse export. Default is 2.
                        - timelapse_displacement (float or None): Max displacement for object tracking.
                        - timelapse_memory (int): Memory for tracking algorithm. Default is 3.
                        - timelapse_frame_limits (list): Frame limits for tracking. Default is [5].
                        - timelapse_remove_transient (bool): Remove short-lived objects. Default is False.
                        - timelapse_mode (str): Tracking algorithm. Default is "trackpy".
                        - timelapse_objects (str or None): Object type for tracking.

                    Miscellaneous:
                        - all_to_mip (bool): Convert all input to MIP. Default is False.
                        - upscale (bool): Upscale images prior to processing. Default is False.
                        - upscale_factor (float): Upscaling factor. Default is 2.0.
                        - adjust_cells (bool): Adjust cell masks based on nuclei and pathogen. Default is False.
                        - use_sam_cell (bool): Use SAM model for cell segmentation. Default is False.
                        - use_sam_nucleus (bool): Use SAM model for nucleus segmentation. Default is False.
                        - use_sam_pathogen (bool): Use SAM model for pathogen segmentation. Default is False.
   :type settings: dict

   :returns: All outputs (masks, merged arrays, plots, databases) are saved to disk under the source folder(s).
   :rtype: None


.. py:function:: generate_cellpose_masks(src, settings, object_type)

   Generate segmentation masks for a specific object type using Cellpose.

   This function applies a Cellpose-based segmentation pipeline to images in `.npz` format, using settings
   for batch size, object type (cell, nucleus, pathogen), filtering, plotting, and timelapse options.
   Masks are optionally filtered, saved, tracked (for timelapse), and summarized into a SQLite database.

   :param src: Path to the source folder containing `.npz` files with image stacks.
   :type src: str
   :param settings: Dictionary of settings used to control preprocessing and segmentation. Includes:

                    General settings:
                        - src (str): Source directory.
                        - denoise (bool): Apply denoising before processing.
                        - delete_intermediate (bool): Remove intermediate files after processing.
                        - preprocess (bool): Enable preprocessing.
                        - masks (bool): Enable mask generation.
                        - save (bool): Save mask outputs.
                        - consolidate (bool): Consolidate image folders.
                        - batch_size (int): Batch size for processing.
                        - test_mode (bool): Enable test mode with limited image count.
                        - test_images (int): Number of test images to process.
                        - magnification (int): Image magnification level.
                        - custom_regex (str or None): Regex pattern for file parsing (metadata_type = 'auto').
                        - metadata_type (str): One of "cellvoyager" or "auto".
                        - n_jobs (int): Number of parallel workers.
                        - randomize (bool): Shuffle file order before processing.
                        - verbose (bool): Print full settings to console.

                    Channel/background/cellpose settings:
                        - remove_background_cell/nucleus/pathogen (bool): Whether to subtract background from channel.
                        - cell_diamiter / nucleus_diamiter / pathogen_diamiter (float or None): Estimated diameter.
                        - cell_channel / nucleus_channel / pathogen_channel (int or None): Channel index.
                        - channels (list): List of channels to include in stack.
                        - cell/background/SNR/CP_prob/FT (float): Intensity/cellpose thresholds and scaling.
                        - pathogen_model (str or None): Custom model for pathogen segmentation (e.g. "toxo_pv_lumen").

                    Plotting:
                        - plot (bool): Plot masks or overlay visualizations.
                        - figuresize (int): Matplotlib figure size.
                        - cmap (str): Colormap to use (e.g. "inferno").
                        - normalize (bool): Normalize input intensities.
                        - normalize_plots (bool): Normalize for plots.
                        - examples_to_plot (int): How many examples to plot.

                    Filtering and merging:
                        - merge_pathogens (bool): Whether to merge pathogen objects.
                        - filter (bool): Apply filtering on masks.
                        - lower_percentile (float): Intensity filter threshold.
                        - merge (bool): Merge adjacent objects if needed.

                    Timelapse:
                        - timelapse (bool): Enable object tracking across timepoints.
                        - timelapse_displacement (float or None): Max tracking displacement.
                        - timelapse_memory (int): Trackpy memory.
                        - timelapse_frame_limits (list): Frames to include in timelapse batch.
                        - timelapse_remove_transient (bool): Remove transient objects.
                        - timelapse_mode (str): One of "trackpy", "btrack", or "iou".
                        - timelapse_objects (list or None): Subset of ['cell', 'nucleus', 'pathogen'] to track.

                    Miscellaneous:
                        - all_to_mip (bool): Convert Z-stacks to max projections.
                        - upscale (bool): Apply upscaling.
                        - upscale_factor (float): Upscaling factor.
                        - adjust_cells (bool): Refine cell masks with nucleus/pathogen.
                        - use_sam_cell/nucleus/pathogen (bool): Use SAM for mask generation.
   :type settings: dict
   :param object_type: One of 'cell', 'nucleus', or 'pathogen'. Determines which mask to generate.
   :type object_type: str

   :returns:     - Generated masks are stored in a `*_mask_stack/` folder.
                 - Object counts are written to `measurements/measurements.db`.
                 - Optional overlay plots are saved if enabled.
                 - Optional timelapse movies are saved in `movies/`.
   :rtype: None. Outputs are saved to disk

   :raises ValueError: If the object_type is missing from the computed channel map, or if invalid tracking settings are provided.


.. py:function:: generate_screen_graphs(settings)

   Generate screen graphs for different measurements in a given source directory.

   :param src: Path(s) to the source directory or directories.
   :type src: str or list
   :param tables: List of tables to include in the analysis (default: ['cell', 'nucleus', 'pathogen', 'cytoplasm']).
   :type tables: list
   :param graph_type: Type of graph to generate (default: 'bar').
   :type graph_type: str
   :param summary_func: Function to summarize data (default: 'mean').
   :type summary_func: str or function
   :param y_axis_start: Starting value for the y-axis (default: 0).
   :type y_axis_start: float
   :param error_bar_type: Type of error bar to use ('std' or 'sem') (default: 'std').
   :type error_bar_type: str
   :param theme: Theme for the graph (default: 'pastel').
   :type theme: str
   :param representation: Representation for grouping (default: 'well').
   :type representation: str

   :returns: List of generated figures.
             results (list): List of corresponding result DataFrames.
   :rtype: figs (list)


