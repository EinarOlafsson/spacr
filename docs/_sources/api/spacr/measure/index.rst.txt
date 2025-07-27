spacr.measure
=============

.. py:module:: spacr.measure




Module Contents
---------------

.. py:function:: get_components(cell_mask, nucleus_mask, pathogen_mask)

   Get the components (nucleus and pathogens) for each cell in the given masks.

   :param cell_mask: Binary mask of cell labels.
   :type cell_mask: ndarray
   :param nucleus_mask: Binary mask of nucleus labels.
   :type nucleus_mask: ndarray
   :param pathogen_mask: Binary mask of pathogen labels.
   :type pathogen_mask: ndarray

   :returns:

             A tuple containing two dataframes - nucleus_df and pathogen_df.
                 nucleus_df (DataFrame): Dataframe with columns 'cell_id' and 'nucleus',
                     representing the mapping of each cell to its nucleus.
                 pathogen_df (DataFrame): Dataframe with columns 'cell_id' and 'pathogen',
                     representing the mapping of each cell to its pathogens.
   :rtype: tuple


.. py:function:: save_and_add_image_to_grid(png_channels, img_path, grid, plot=False)

   Add an image to a grid and save it as PNG.

   :param png_channels: The array representing the image channels.
   :type png_channels: ndarray
   :param img_path: The path to save the image as PNG.
   :type img_path: str
   :param grid: The grid of images to be plotted later.
   :type grid: list

   :returns: Updated grid with the new image added.
   :rtype: grid (list)


.. py:function:: img_list_to_grid(grid, titles=None)

   Plot a grid of images with optional titles.

   :param grid: List of images to be plotted.
   :type grid: list
   :param titles: List of titles for the images.
   :type titles: list

   :returns: The matplotlib figure object containing the image grid.
   :rtype: fig (Figure)


.. py:function:: measure_crop(settings)

   Main driver function to process `.npy` image stacks in a given folder or list of folders.
   Applies morphological measurements, generates cropped images (PNGs), and saves object-level
   metrics and optional visuals using multiprocessing for speed.

   :param settings: Dictionary of processing parameters. Use `get_measure_crop_settings()` to populate defaults.
   :type settings: dict
   :param Key settings include:
                                - src (str or list[str]): Path(s) to folder(s) containing `.npy` stacks.
                                - experiment (str): Experiment name to store alongside outputs in the database.
                                - test_mode (bool): If True, limit number of images and enable plotting.
                                - verbose (bool): Print processing info.
                                - channels (list[int]): Channels to use for intensity measurements and PNG generation.
                                - plot (bool): If True, save figures to memory and optionally to disk.
                                - n_jobs (int): Number of parallel processes (defaults to CPU count minus 2).
                                - cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim (int or None): Indices of object masks in the stack.
                                - cytoplasm (bool): If True, derive cytoplasmic mask.
                                - _min_size (int): Minimum pixel size thresholds for object filtering.
                                - merge_edge_pathogen_cells (bool): If True, merge pathogen/cell masks at edges.
                                - timelapse (bool): If True, enable temporal relabeling and GIF generation.
                                - timelapse_objects (str): Which object type to track temporally ("nucleus" or "cell").
                                - save_measurements (bool): Save morphology and intensity features to SQLite DB.
                                - save_png, save_arrays (bool): Save per-object cropped PNGs and/or subarrays.
                                - png_dims (list[int]): Channel indices to render in PNG.
                                - png_size (list[int] or list[list[int]]): PNG crop size in pixels (width, height).
                                - crop_mode (list[str]): Which objects to crop (e.g., ['cell', 'nucleus']).
                                - normalize (list[int] or False): Percentiles for intensity normalization or False to skip.
                                - normalize_by (str): 'png' or 'fov'—reference frame for normalization.
                                - dialate_pngs (bool or list[bool]): Whether to dilate PNG masks before cropping.
                                - dialate_png_ratios (float or list[float]): Dilation factor relative to object size.
                                - use_bounding_box (bool): Use bounding box rather than minimal crop.
                                - delete_intermediate (bool): If True, delete original input arrays after processing.

   Workflow:
       - Validates and normalizes input settings.
       - Applies multiprocessing to process each `.npy` file using `_measure_crop_core()`.
       - Saves measurement outputs (morphology, intensity) to database.
       - Generates per-object crops as PNGs or arrays, optionally normalized and resized.
       - If `timelapse=True`, generates summary mask GIFs across timepoints.
       - Reports progress and CPU usage throughout execution.

   :returns: None. Results are written to disk and/or SQLite DBs. Completion is reported via print statements.

   :raises ValueError: For invalid or missing keys in `settings`.
   :raises Warnings are printed to console for most incorrect parameter combinations.:

   .. rubric:: Notes

   - The `settings['src']` directory is expected to contain `.npy` files and typically ends with `/merged`.
   - Processing uses up to `settings['n_jobs']` CPU cores but reserves 6 cores by default.
   - Errors during file processing are handled per file; execution continues for remaining files.


.. py:function:: process_meassure_crop_results(partial_results, settings)

   Process the results, display, and optionally save the figures.

   :param partial_results: List of partial results.
   :type partial_results: list
   :param settings: Settings dictionary.
   :type settings: dict
   :param save_figures: Flag to save figures or not.
   :type save_figures: bool


.. py:function:: generate_cellpose_train_set(folders, dst, min_objects=5)

   Prepares a Cellpose training dataset by extracting images and corresponding masks
   from one or more processed spaCR folders. Filters objects by minimum count per mask.

   :param folders: List of source directories. Each must contain:
                   - a `masks/` folder with segmentation masks
                   - the corresponding raw images at the top level.
   :type folders: list[str]
   :param dst: Destination folder where the training dataset will be saved.
               Two subfolders will be created: `dst/masks` and `dst/imgs`.
   :type dst: str
   :param min_objects: Minimum number of objects (excluding background) required
                       in a mask to be included in the training set.
   :type min_objects: int

   Workflow:
       - Iterates through each folder and its `masks/` subfolder.
       - For each `.tif` or `.png` mask, counts the number of unique objects.
       - If `nr_of_objects >= min_objects`, the mask and corresponding image are copied
         to `dst/masks` and `dst/imgs`, respectively.
       - Output files are renamed using `experiment_id + '_' + original_filename` to avoid collisions.

   :returns: None. Selected images and masks are copied to the target location.

   .. rubric:: Notes

   - Skips any unreadable or malformed mask files.
   - Assumes mask files use 0 for background and positive integers for labeled objects.
   - This function does not validate image-mask alignment—ensure file naming is consistent.


.. py:function:: get_object_counts(src)

   Reads the object count summary from the SQLite database and returns aggregated statistics.

   :param src: Source directory containing a `measurements/measurements.db` file
               generated by the `measure_crop()` pipeline.
   :type src: str

   :returns:

             A summary table with one row per `count_type`, including:
                 - total_object_count: Sum of object counts across all files for that type.
                 - avg_object_count_per_file_name: Mean object count per file.
   :rtype: pandas.DataFrame

   Example Output:
       count_type      total_object_count   avg_object_count_per_file_name
       -----------     -------------------  -------------------------------
       nucleus         10500                87.5
       cell            10892                90.77

   .. rubric:: Notes

   - Requires the presence of an `object_counts` table in the database.
   - Fails with an exception if the table does not exist or database is missing.


