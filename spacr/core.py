import os, gc, torch, time, random
import numpy as np
import pandas as pd
from IPython.display import display

import warnings
warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

def preprocess_generate_masks(settings):
    
    """
    Preprocess image data and generate Cellpose segmentation masks for cells, nuclei, and pathogens.

    This function supports preprocessing, metadata conversion, Cellpose-based mask generation, optional
    mask adjustment, result plotting, and intermediate file cleanup. It handles batch operations and 
    supports advanced timelapse and channel-specific configurations.

    Args:
        settings (dict): Dictionary containing the following keys:

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

    Returns:
        None: All outputs (masks, merged arrays, plots, databases) are saved to disk under the source folder(s).
    """

    from .io import preprocess_img_data, _load_and_concatenate_arrays, convert_to_yokogawa, convert_separate_files_to_yokogawa
    from .plot import plot_image_mask_overlay, plot_arrays
    from .utils import _pivot_counts_table, check_mask_folder, adjust_cell_masks, print_progress, save_settings, delete_intermedeate_files, format_path_for_system, normalize_src_path, generate_image_path_map, copy_images_to_consolidated
    from .settings import set_default_settings_preprocess_generate_masks
        
    if 'src' in settings:
        if not isinstance(settings['src'], (str, list)):
            ValueError(f'src must be a string or a list of strings')
            return
    else:
        ValueError(f'src is a required parameter')
        return
    
    settings['src'] = normalize_src_path(settings['src'])
    
    if settings['consolidate']:
        image_map = generate_image_path_map(settings['src'])
        copy_images_to_consolidated(image_map, settings['src'])
        settings['src'] = os.path.join(settings['src'], 'consolidated')

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    if isinstance(settings['src'], list):
        source_folders = settings['src']
        for source_folder in source_folders:
            
            print(f'Processing folder: {source_folder}')
            
            source_folder = format_path_for_system(source_folder)   
            settings['src'] = source_folder
            src = source_folder
            settings = set_default_settings_preprocess_generate_masks(settings)
            
            if settings['metadata_type'] == 'auto':
                if settings['custom_regex'] != None:
                    try:
                        print(f"using regex: {settings['custom_regex']}")
                        convert_separate_files_to_yokogawa(folder=source_folder, regex=settings['custom_regex'])
                    except:
                        try:
                            convert_to_yokogawa(folder=source_folder)
                        except Exception as e:
                            print(f"Error: Tried to convert image files and image file name metadata with regex {settings['custom_regex']} then without regex but failed both.")
                            print(f'Error: {e}')
                            return
                else:
                    try:
                        convert_to_yokogawa(folder=source_folder)
                    except Exception as e:
                        print(f"Error: Tried to convert image files and image file name metadata without regex but failed.")
                        print(f'Error: {e}')
                        return
            
            if settings['cell_channel'] is None and settings['nucleus_channel'] is None and settings['pathogen_channel'] is None:
                print(f'Error: At least one of cell_channel, nucleus_channel or pathogen_channel must be defined')
                return
            
            save_settings(settings, name='gen_mask_settings')
            
            if not settings['pathogen_channel'] is None:
                custom_model_ls = ['toxo_pv_lumen','toxo_cyto']
                if settings['pathogen_model'] not in custom_model_ls:
                    ValueError(f'Pathogen model must be {custom_model_ls} or None')
            
            if settings['timelapse']:
                settings['randomize'] = False
            
            if settings['preprocess']:
                if not settings['masks']:
                    print(f'WARNING: channels for mask generation are defined when preprocess = True')
            
            if isinstance(settings['save'], bool):
                settings['save'] = [settings['save']]*3

            if settings['verbose']:
                settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
                settings_df['setting_value'] = settings_df['setting_value'].apply(str)
                display(settings_df)
                
            if settings['test_mode']:
                print(f'Starting Test mode ...')

            if settings['preprocess']:
                settings, src = preprocess_img_data(settings)

            files_to_process = 3
            files_processed = 0
            if settings['masks']:
                mask_src = os.path.join(src, 'masks')
                if settings['cell_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'cell_mask_stack'):
                        start = time.time()
                        generate_cellpose_masks(mask_src, settings, 'cell')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'cell_mask_gen')
                    
                if settings['nucleus_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'nucleus_mask_stack'):
                        start = time.time()
                        generate_cellpose_masks(mask_src, settings, 'nucleus')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'nucleus_mask_gen')
                    
                if settings['pathogen_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'pathogen_mask_stack'):
                        start = time.time()
                        generate_cellpose_masks(mask_src, settings, 'pathogen')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'pathogen_mask_gen')

                #if settings['organelle'] != None:
                #    if check_mask_folder(src, 'organelle_mask_stack'):
                #        generate_cellpose_masks(mask_src, settings, 'organelle')

                if settings['adjust_cells']:
                    if settings['pathogen_channel'] != None and settings['cell_channel'] != None and settings['nucleus_channel'] != None:

                        start = time.time()
                        cell_folder = os.path.join(mask_src, 'cell_mask_stack')
                        nuclei_folder = os.path.join(mask_src, 'nucleus_mask_stack')
                        parasite_folder = os.path.join(mask_src, 'pathogen_mask_stack')
                        #organelle_folder = os.path.join(mask_src, 'organelle_mask_stack')
                        print(f'Adjusting cell masks with nuclei and pathogen masks')
                        adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, overlap_threshold=5, perimeter_threshold=30, n_jobs=settings['n_jobs'])
                        stop = time.time()
                        adjust_time = (stop-start)/60
                        print(f'Cell mask adjustment: {adjust_time} min.')
                    
                if os.path.exists(os.path.join(src,'measurements')):
                    _pivot_counts_table(db_path=os.path.join(src,'measurements', 'measurements.db'))

                #Concatenate stack with masks
                _load_and_concatenate_arrays(src, settings['channels'], settings['cell_channel'], settings['nucleus_channel'], settings['pathogen_channel'])
                
                if settings['plot']:
                    if not settings['timelapse']:
                        if settings['test_mode'] == True:
                            settings['examples_to_plot'] = len(os.path.join(src,'merged'))

                        try:
                            merged_src = os.path.join(src,'merged')
                            files = os.listdir(merged_src)
                            random.shuffle(files)
                            time_ls = []
                            
                            for i, file in enumerate(files):
                                start = time.time()
                                if i+1 <= settings['examples_to_plot']:
                                    file_path = os.path.join(merged_src, file)
                                    plot_image_mask_overlay(file_path, settings['channels'], settings['cell_channel'], settings['nucleus_channel'], settings['pathogen_channel'], figuresize=10, percentiles=(1,99), thickness=3, save_pdf=True)
                                    stop = time.time()
                                    duration = stop-start
                                    time_ls.append(duration)
                                    files_processed = i+1
                                    files_to_process = settings['examples_to_plot']
                                    print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Plot mask outlines")
                                    
                        except Exception as e:
                            print(f'Failed to plot image mask overly. Error: {e}')
                    else:
                        plot_arrays(src=os.path.join(src,'merged'), figuresize=settings['figuresize'], cmap=settings['cmap'], nr=settings['examples_to_plot'], normalize=settings['normalize'], q1=1, q2=99)
                    
            torch.cuda.empty_cache()
            gc.collect()
            
            if settings['delete_intermediate']:
                print(f"deleting intermediate files")
                delete_intermedeate_files(settings)

            print("Successfully completed run")
    return

def generate_cellpose_masks(src, settings, object_type):
    """
    Generate segmentation masks for a specific object type using Cellpose.

    This function applies a Cellpose-based segmentation pipeline to images in `.npz` format, using settings 
    for batch size, object type (cell, nucleus, pathogen), filtering, plotting, and timelapse options. 
    Masks are optionally filtered, saved, tracked (for timelapse), and summarized into a SQLite database.

    Args:
        src (str): Path to the source folder containing `.npz` files with image stacks.
        settings (dict): Dictionary of settings used to control preprocessing and segmentation. Includes:

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

        object_type (str): One of 'cell', 'nucleus', or 'pathogen'. Determines which mask to generate.

    Returns:
        None. Outputs are saved to disk:
            - Generated masks are stored in a `*_mask_stack/` folder.
            - Object counts are written to `measurements/measurements.db`.
            - Optional overlay plots are saved if enabled.
            - Optional timelapse movies are saved in `movies/`.

    Raises:
        ValueError: If the object_type is missing from the computed channel map, or if invalid tracking settings are provided.
    """
    from .utils import _masks_to_masks_stack, _filter_cp_masks, _get_cellpose_batch_size, _get_cellpose_channels, _choose_model, all_elements_match, prepare_batch_for_segmentation
    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .timelapse import _npz_to_movie, _btrack_track_cells, _trackpy_track_cells
    from .plot import plot_cellpose4_output
    from .settings import set_default_settings_preprocess_generate_masks, _get_object_settings
    from .spacr_cellpose import parse_cellpose4_output
    
    gc.collect()
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
        
    settings['src'] = src
    
    settings = set_default_settings_preprocess_generate_masks(settings)

    if settings['verbose']:
        settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
        settings_df['setting_value'] = settings_df['setting_value'].apply(str)
        display(settings_df)
        
    figuresize=10
    timelapse = settings['timelapse']
    
    if timelapse:
        timelapse_displacement = settings['timelapse_displacement']
        timelapse_frame_limits = settings['timelapse_frame_limits']
        timelapse_memory = settings['timelapse_memory']
        timelapse_remove_transient = settings['timelapse_remove_transient']
        timelapse_mode = settings['timelapse_mode']
        timelapse_objects = settings['timelapse_objects']
    
    batch_size = settings['batch_size']
    
    cellprob_threshold = settings[f'{object_type}_CP_prob']

    flow_threshold = settings[f'{object_type}_FT']

    object_settings = _get_object_settings(object_type, settings)
    
    model_name = object_settings['model_name']
    
    cellpose_channels = _get_cellpose_channels(src, settings['nucleus_channel'], settings['pathogen_channel'], settings['cell_channel'])
    
    if settings['verbose']:
        print(cellpose_channels)
        
    if object_type not in cellpose_channels:
        raise ValueError(f"Error: No channels were specified for object_type '{object_type}'. Check your settings.")
    
    channels = cellpose_channels[object_type]

    #cellpose_batch_size = _get_cellpose_batch_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if object_type == 'pathogen' and not settings['pathogen_model'] is None:
        model_name = settings['pathogen_model']
    
    model = _choose_model(model_name, device, object_type=object_type, restore_type=None, object_settings=object_settings)

    #chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [2,0] if model_name == 'cyto' else [2, 0] if model_name == 'cyto3' else [2, 0]
    
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]    
    
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    average_count = []
    time_ls = []
    
    for file_index, path in enumerate(paths):
        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
        os.makedirs(output_folder, exist_ok=True)
        overall_average_size = 0
        
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
            
            for i, filename in enumerate(filenames):
                output_path = os.path.join(output_folder, filename)
                
                if os.path.exists(output_path):
                    print(f"File {filename} already exists in the output folder. Skipping...")
                    continue
        
        if settings['timelapse']:

            trackable_objects = ['cell','nucleus','pathogen']
            if not all_elements_match(settings['timelapse_objects'], trackable_objects):
                print(f'timelapse_objects {settings["timelapse_objects"]} must be a subset of {trackable_objects}')
                return

            if len(stack) != batch_size:
                print(f'Changed batch_size:{batch_size} to {len(stack)}, data length:{len(stack)}')
                settings['timelapse_batch_size'] = len(stack)
                batch_size = len(stack)
                if isinstance(timelapse_frame_limits, list):
                    if len(timelapse_frame_limits) >= 2:
                        stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1], :, :, :].astype(stack.dtype)
                        filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                        batch_size = len(stack)
                        print(f'Cut batch at indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')
        
        for i in range(0, stack.shape[0], batch_size):
            mask_stack = []
            if stack.shape[3] == 1:
                batch = stack[i: i+batch_size, :, :, [0,0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)

            batch_filenames = filenames[i: i+batch_size].tolist()

            if not settings['plot']:
                batch, batch_filenames = _check_masks(batch, batch_filenames, output_folder)
            if batch.size == 0:
                continue
            
            batch = prepare_batch_for_segmentation(batch)
            batch_list = [batch[i] for i in range(batch.shape[0])]

            if timelapse:
                movie_path = os.path.join(os.path.dirname(src), 'movies')
                os.makedirs(movie_path, exist_ok=True)
                save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                _npz_to_movie(batch, batch_filenames, save_path, fps=2)
                        
            output = model.eval(x=batch_list,
                                batch_size=batch_size,
                                normalize=False,
                                channel_axis=-1,
                                channels=channels,
                                diameter=object_settings['diameter'],
                                flow_threshold=flow_threshold,
                                cellprob_threshold=cellprob_threshold,
                                rescale=None,
                                resample=object_settings['resample'])
                        
            masks, flows, _, _, _ = parse_cellpose4_output(output)

            if timelapse:
                if settings['plot']:
                    plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=1, print_object_number=True)

                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
                if object_type in timelapse_objects:
                    if timelapse_mode == 'btrack':
                        if not timelapse_displacement is None:
                            radius = timelapse_displacement
                        else:
                            radius = 100

                        n_jobs = os.cpu_count()-2
                        if n_jobs < 1:
                            n_jobs = 1

                        mask_stack = _btrack_track_cells(src=src,
                                                         name=name,
                                                         batch_filenames=batch_filenames,
                                                         object_type=object_type,
                                                         plot=settings['plot'],
                                                         save=settings['save'],
                                                         masks_3D=masks,
                                                         mode=timelapse_mode,
                                                         timelapse_remove_transient=timelapse_remove_transient,
                                                         radius=radius,
                                                         n_jobs=n_jobs)
                    
                    if timelapse_mode == 'trackpy' or timelapse_mode == 'iou':
                        if timelapse_mode == 'iou':
                            track_by_iou = True
                        else:
                            track_by_iou = False
                        
                        mask_stack = _trackpy_track_cells(src=src,
                                                          name=name,
                                                          batch_filenames=batch_filenames,
                                                          object_type=object_type,
                                                          masks=masks,
                                                          timelapse_displacement=timelapse_displacement,
                                                          timelapse_memory=timelapse_memory,
                                                          timelapse_remove_transient=timelapse_remove_transient,
                                                          plot=settings['plot'],
                                                          save=settings['save'],
                                                          mode=timelapse_mode,
                                                          track_by_iou=track_by_iou)
                else:
                    mask_stack = _masks_to_masks_stack(masks)
            else:
                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                if object_settings['merge'] and not settings['filter']:
                    mask_stack = _filter_cp_masks(masks=masks,
                                                flows=flows,
                                                filter_size=False,
                                                filter_intensity=False,
                                                minimum_size=object_settings['minimum_size'],
                                                maximum_size=object_settings['maximum_size'],
                                                remove_border_objects=False,
                                                merge=object_settings['merge'],
                                                batch=batch,
                                                plot=settings['plot'],
                                                figuresize=figuresize)

                if settings['filter']:
                    mask_stack = _filter_cp_masks(masks=masks,
                                                flows=flows,
                                                filter_size=object_settings['filter_size'],
                                                filter_intensity=object_settings['filter_intensity'],
                                                minimum_size=object_settings['minimum_size'],
                                                maximum_size=object_settings['maximum_size'],
                                                remove_border_objects=object_settings['remove_border_objects'],
                                                merge=object_settings['merge'],
                                                batch=batch,
                                                plot=settings['plot'],
                                                figuresize=figuresize)
                    
                    _save_object_counts_to_database(mask_stack, object_type, batch_filenames, count_loc, added_string='_after_filtration')
                else:
                    mask_stack = _masks_to_masks_stack(masks)

                    #if settings['plot']:
                    #    plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=1, print_object_number=True)
        
            if not np.any(mask_stack):
                avg_num_objects_per_image, average_obj_size = 0, 0
            else:
                avg_num_objects_per_image, average_obj_size = _get_avg_object_size(mask_stack)
            
            average_count.append(avg_num_objects_per_image)
            average_sizes.append(average_obj_size) 
            overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
            overall_average_count = np.mean(average_count) if len(average_count) > 0 else 0
            print(f'Found {overall_average_count} {object_type}/FOV. average size: {overall_average_size:.3f} px2')

        if not timelapse:
            if settings['plot']:
                plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=batch_size)
                
        if settings['save']:
            for mask_index, mask in enumerate(mask_stack):
                output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                mask = mask.astype(np.uint16)
                np.save(output_filename, mask)
            mask_stack = []
            batch_filenames = []

        gc.collect()
    torch.cuda.empty_cache()
    return

def generate_screen_graphs(settings):
    """
    Generate screen graphs for different measurements in a given source directory.

    Args:
        src (str or list): Path(s) to the source directory or directories.
        tables (list): List of tables to include in the analysis (default: ['cell', 'nucleus', 'pathogen', 'cytoplasm']).
        graph_type (str): Type of graph to generate (default: 'bar').
        summary_func (str or function): Function to summarize data (default: 'mean').
        y_axis_start (float): Starting value for the y-axis (default: 0).
        error_bar_type (str): Type of error bar to use ('std' or 'sem') (default: 'std').
        theme (str): Theme for the graph (default: 'pastel').
        representation (str): Representation for grouping (default: 'well').
        
    Returns:
        figs (list): List of generated figures.
        results (list): List of corresponding result DataFrames.
    """
    
    from .plot import spacrGraph
    from .io import _read_and_merge_data
    from.utils import annotate_conditions

    if isinstance(settings['src'], str):
        srcs = [settings['src']]
    else:
        srcs = settings['src']

    all_df = pd.DataFrame()
    figs = []
    results = []

    for src in srcs:
        db_loc = [os.path.join(src, 'measurements', 'measurements.db')]
        
        # Read and merge data from the database
        df, _ = _read_and_merge_data(db_loc, settings['tables'], verbose=True, nuclei_limit=settings['nuclei_limit'], pathogen_limit=settings['pathogen_limit'])
        
        # Annotate the data
        df = annotate_conditions(df, cells=settings['cells'], cell_loc=None, pathogens=settings['controls'], pathogen_loc=settings['controls_loc'], treatments=None, treatment_loc=None)
        
        # Calculate recruitment metric
        df['recruitment'] = df['pathogen_channel_1_mean_intensity'] / df['cytoplasm_channel_1_mean_intensity']
                
        # Combine with the overall DataFrame
        all_df = pd.concat([all_df, df], ignore_index=True)
    
        # Generate individual plot
        plotter = spacrGraph(df,
                             grouping_column='pathogen',
                             data_column='recruitment',
                             graph_type=settings['graph_type'],
                             summary_func=settings['summary_func'],
                             y_axis_start=settings['y_axis_start'],
                             error_bar_type=settings['error_bar_type'],
                             theme=settings['theme'],
                             representation=settings['representation'])

        plotter.create_plot()
        fig = plotter.get_figure()
        results_df = plotter.get_results()
        
        # Append to the lists
        figs.append(fig)
        results.append(results_df)
    
    # Generate plot for the combined data (all_df)
    plotter = spacrGraph(all_df,
                         grouping_column='pathogen',
                         data_column='recruitment',
                         graph_type=settings['graph_type'],
                         summary_func=settings['summary_func'],
                         y_axis_start=settings['y_axis_start'],
                         error_bar_type=settings['error_bar_type'],
                         theme=settings['theme'],
                         representation=settings['representation'])

    plotter.create_plot()
    fig = plotter.get_figure()
    results_df = plotter.get_results()
    
    figs.append(fig)
    results.append(results_df)
    
    # Save figures and results
    for i, fig in enumerate(figs):
        res = results[i]
        
        if i < len(srcs):
            source = srcs[i]
        else:
            source = srcs[0]

        # Ensure the destination folder exists
        dst = os.path.join(source, 'results')
        print(f"Savings results to {dst}")
        os.makedirs(dst, exist_ok=True)
        
        # Save the figure and results DataFrame
        fig.savefig(os.path.join(dst, f"figure_controls_{i}_{settings['representation']}_{settings['summary_func']}_{settings['graph_type']}.pdf"), format='pdf')
        res.to_csv(os.path.join(dst, f"results_controls_{i}_{settings['representation']}_{settings['summary_func']}_{settings['graph_type']}.csv"), index=False)

    return