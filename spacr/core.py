import os, gc, torch, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import warnings
warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

def preprocess_generate_masks(settings):

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
                mask_src = os.path.join(src, 'norm_channel_stack')
                if settings['cell_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'cell_mask_stack'):
                        start = time.time()
                        if settings['segmentation_mode'] == 'cellpose':
                            generate_cellpose_masks(mask_src, settings, 'cell')
                        elif settings['segmentation_mode'] == 'mediar':
                            generate_mediar_masks(mask_src, settings, 'cell')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'cell_mask_gen')
                    
                if settings['nucleus_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'nucleus_mask_stack'):
                        start = time.time()
                        if settings['segmentation_mode'] == 'cellpose':
                            generate_cellpose_masks(mask_src, settings, 'nucleus')
                        elif settings['segmentation_mode'] == 'mediar':
                            generate_mediar_masks(mask_src, settings, 'nucleus')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'nucleus_mask_gen')
                    
                if settings['pathogen_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'pathogen_mask_stack'):
                        start = time.time()
                        if settings['segmentation_mode'] == 'cellpose':
                            generate_cellpose_masks(mask_src, settings, 'pathogen')
                        elif settings['segmentation_mode'] == 'mediar':
                            generate_mediar_masks(mask_src, settings, 'pathogen')
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
                        adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, overlap_threshold=5, perimeter_threshold=30)
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
    
    from .utils import _masks_to_masks_stack, _filter_cp_masks, _get_cellpose_batch_size, _get_cellpose_channels, _choose_model, mask_object_count, all_elements_match, prepare_batch_for_segmentation
    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .timelapse import _npz_to_movie, _btrack_track_cells, _trackpy_track_cells
    from .plot import plot_masks
    from .settings import set_default_settings_preprocess_generate_masks, _get_object_settings
    
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

    channels = cellpose_channels[object_type]
    cellpose_batch_size = _get_cellpose_batch_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if object_type == 'pathogen' and not settings['pathogen_model'] is None:
        model_name = settings['pathogen_model']
    
    model = _choose_model(model_name, device, object_type=object_type, restore_type=None, object_settings=object_settings)

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [2,0] if model_name == 'cyto' else [2, 0] if model_name == 'cyto3' else [2, 0]
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]    
    
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
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
            
            
            #if settings['denoise']:
            #    if object_type == 'cell':
            #        model_type = "denoise_cyto3"
            #    elif object_type == 'nucleus':
            #        model_type = "denoise_nucleus"
            #    else:
            #        raise ValueError(f"No denoise model for object_type: {object_type}")
            #    dn = denoise.DenoiseModel(model_type=model_type, gpu=device)
            #    batch = dn.eval(imgs=batch, channels=chans, diameter=object_settings['diameter'])

            if timelapse:
                movie_path = os.path.join(os.path.dirname(src), 'movies')
                os.makedirs(movie_path, exist_ok=True)
                save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                _npz_to_movie(batch, batch_filenames, save_path, fps=2)
            
            output = model.eval(x=batch,
                                batch_size=cellpose_batch_size,
                                normalize=False,
                                channels=chans,
                                channel_axis=3,
                                diameter=object_settings['diameter'],
                                flow_threshold=flow_threshold,
                                cellprob_threshold=cellprob_threshold,
                                rescale=None,
                                resample=object_settings['resample'])
            
            if len(output) == 4:
                masks, flows, _, _ = output
            elif len(output) == 3:
                masks, flows, _ = output
            else:
                raise ValueError(f"Unexpected number of return values from model.eval(). Expected 3 or 4, got {len(output)}")

            if timelapse:
                if settings['plot']:
                    for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
                        if idx == 0:
                            num_objects = mask_object_count(mask)
                            print(f'Number of objects: {num_objects}')
                            plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)

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
                    if timelapse_mode == 'trackpy':
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
                                                          mode=timelapse_mode)
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

                    if settings['plot']:
                        for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
                            if idx == 0:
                                num_objects = mask_object_count(mask)
                                print(f'Number of objects, : {num_objects}')
                                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        
            if not np.any(mask_stack):
                average_obj_size = 0
            else:
                average_obj_size = _get_avg_object_size(mask_stack)

            average_sizes.append(average_obj_size) 
            overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
            print(f'object_size:{object_type}: {overall_average_size:.3f} px2')

        if not timelapse:
            if settings['plot']:
                plot_masks(batch, mask_stack, flows, figuresize=figuresize, cmap='inferno', nr=batch_size)
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

def generate_image_umap(settings={}):
    """
    Generate UMAP or tSNE embedding and visualize the data with clustering.
    
    Parameters:
    settings (dict): Dictionary containing the following keys:
    src (str): Source directory containing the data.
    row_limit (int): Limit the number of rows to process.
    tables (list): List of table names to read from the database.
    visualize (str): Visualization type.
    image_nr (int): Number of images to display.
    dot_size (int): Size of dots in the scatter plot.
    n_neighbors (int): Number of neighbors for UMAP.
    figuresize (int): Size of the figure.
    black_background (bool): Whether to use a black background.
    remove_image_canvas (bool): Whether to remove the image canvas.
    plot_outlines (bool): Whether to plot outlines.
    plot_points (bool): Whether to plot points.
    smooth_lines (bool): Whether to smooth lines.
    verbose (bool): Whether to print verbose output.
    embedding_by_controls (bool): Whether to use embedding from controls.
    col_to_compare (str): Column to compare for control-based embedding.
    pos (str): Positive control value.
    neg (str): Negative control value.
    clustering (str): Clustering method ('DBSCAN' or 'KMeans').
    exclude (list): List of columns to exclude from the analysis.
    plot_images (bool): Whether to plot images.
    reduction_method (str): Dimensionality reduction method ('UMAP' or 'tSNE').
    save_figure (bool): Whether to save the figure as a PDF.
    
    Returns:
    pd.DataFrame: DataFrame with the original data and an additional column 'cluster' containing the cluster identity.
    """
 
    from .io import _read_and_join_tables
    from .utils import get_db_paths, preprocess_data, reduction_and_clustering, remove_noise, generate_colors, correct_paths, plot_embedding, plot_clusters_grid, cluster_feature_analysis, map_condition
    from .settings import set_default_umap_image_settings
    settings = set_default_umap_image_settings(settings)

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    if settings['plot_images'] is False:
        settings['black_background'] = False

    if settings['color_by']:
        settings['remove_cluster_noise'] = False
        settings['plot_outlines'] = False
        settings['smooth_lines'] = False

    print(f'Generating Image UMAP ...')
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_dir = os.path.join(settings['src'][0],'settings')
    settings_csv = os.path.join(settings_dir,'embedding_settings.csv')
    os.makedirs(settings_dir, exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)
    display(settings_df)

    db_paths = get_db_paths(settings['src'])
    tables = settings['tables'] + ['png_list']
    all_df = pd.DataFrame()

    for i,db_path in enumerate(db_paths):
        df = _read_and_join_tables(db_path, table_names=tables)
        df, image_paths_tmp = correct_paths(df, settings['src'][i])
        all_df = pd.concat([all_df, df], axis=0)
        #image_paths.extend(image_paths_tmp)
        
    all_df['cond'] = all_df['columnID'].apply(map_condition, neg=settings['neg'], pos=settings['pos'], mix=settings['mix'])

    if settings['exclude_conditions']:
        if isinstance(settings['exclude_conditions'], str):
            settings['exclude_conditions'] = [settings['exclude_conditions']]
        row_count_before = len(all_df)
        all_df = all_df[~all_df['cond'].isin(settings['exclude_conditions'])]
        if settings['verbose']:
            print(f'Excluded {row_count_before - len(all_df)} rows after excluding: {settings["exclude_conditions"]}, rows left: {len(all_df)}')

    if settings['row_limit'] is not None:
        all_df = all_df.sample(n=settings['row_limit'], random_state=42)

    image_paths = all_df['png_path'].to_list()

    if settings['embedding_by_controls']:
        
        # Extract and reset the index for the column to compare
        col_to_compare = all_df[settings['col_to_compare']].reset_index(drop=True)
        print(col_to_compare)
        #if settings['only_top_features']:
        #    column_list = None
            
        # Preprocess the data to obtain numeric data
        numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])

        # Convert numeric_data back to a DataFrame to align with col_to_compare
        numeric_data_df = pd.DataFrame(numeric_data)

        # Ensure numeric_data_df and col_to_compare are properly aligned
        numeric_data_df = numeric_data_df.reset_index(drop=True)

        # Assign the column back to numeric_data_df
        numeric_data_df[settings['col_to_compare']] = col_to_compare

        # Subset the dataframe based on specified column values for controls
        positive_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['pos']].copy()
        negative_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['neg']].copy()
        control_numeric_data_df = pd.concat([positive_control_df, negative_control_df])

        # Drop the comparison column from numeric_data_df and control_numeric_data_df
        numeric_data_df = numeric_data_df.drop(columns=[settings['col_to_compare']])
        control_numeric_data_df = control_numeric_data_df.drop(columns=[settings['col_to_compare']])

        # Convert numeric_data_df and control_numeric_data_df back to numpy arrays
        numeric_data = numeric_data_df.values
        control_numeric_data = control_numeric_data_df.values

        # Train the reducer on control data
        _, _, reducer = reduction_and_clustering(control_numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], mode='fit', model=False)
        
        # Apply the trained reducer to the entire dataset
        numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])
        embedding, labels, _ = reduction_and_clustering(numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], mode=None, model=reducer)

    else:
        if settings['resnet_features']:
            # placeholder for resnet features, not implemented yet
            pass
            #numeric_data, embedding, labels = generate_umap_from_images(image_paths, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['clustering'], settings['eps'], settings['min_samples'], settings['n_jobs'], settings['verbose'])
        else:
            # Apply the trained reducer to the entire dataset
            numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])
            embedding, labels, _ = reduction_and_clustering(numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'])
    
    if settings['remove_cluster_noise']:
        # Remove noise from the clusters (removes -1 labels from DBSCAN)
        embedding, labels = remove_noise(embedding, labels)

    # Plot the results
    if settings['color_by']:
        if settings['embedding_by_controls']:
            labels = all_df[settings['color_by']]
        else:
            labels = all_df[settings['color_by']]
    
    # Generate colors for the clusters
    colors = generate_colors(len(np.unique(labels)), settings['black_background'])

    # Plot the embedding
    umap_plt = plot_embedding(embedding, image_paths, labels, settings['image_nr'], settings['img_zoom'], colors, settings['plot_by_cluster'], settings['plot_outlines'], settings['plot_points'], settings['plot_images'], settings['smooth_lines'], settings['black_background'], settings['figuresize'], settings['dot_size'], settings['remove_image_canvas'], settings['verbose'])
    if settings['plot_cluster_grids'] and settings['plot_images']:
        grid_plt = plot_clusters_grid(embedding, labels, settings['image_nr'], image_paths, colors, settings['figuresize'], settings['black_background'], settings['verbose'])
    
    # Save figure as PDF if required
    if settings['save_figure']:
        results_dir = os.path.join(settings['src'][0], 'results')
        os.makedirs(results_dir, exist_ok=True)
        reduction_method = settings['reduction_method'].upper()
        embedding_path = os.path.join(results_dir, f'{reduction_method}_embedding.pdf')
        umap_plt.savefig(embedding_path, format='pdf')
        print(f'Saved {reduction_method} embedding to {embedding_path} and grid to {embedding_path}')
        if settings['plot_cluster_grids'] and settings['plot_images']:
            grid_path = os.path.join(results_dir, f'{reduction_method}_grid.pdf')
            grid_plt.savefig(grid_path, format='pdf')
            print(f'Saved {reduction_method} embedding to {embedding_path} and grid to {grid_path}')

    # Add cluster labels to the dataframe
    if len(labels) > 0:
        all_df['cluster'] = labels
    else:
        all_df['cluster'] = 1  # Assign a default cluster label
        print("No clusters found. Consider reducing 'min_samples' or increasing 'eps' for DBSCAN.")

    # Save the results to a CSV file
    results_dir = os.path.join(settings['src'][0], 'results')
    results_csv = os.path.join(results_dir,'embedding_results.csv')
    os.makedirs(results_dir, exist_ok=True)
    all_df.to_csv(results_csv, index=False)
    print(f'Results saved to {results_csv}')

    if settings['analyze_clusters']:
        combined_results = cluster_feature_analysis(all_df)
        results_dir = os.path.join(settings['src'][0], 'results')
        cluster_results_csv = os.path.join(results_dir,'cluster_results.csv')
        os.makedirs(results_dir, exist_ok=True)
        combined_results.to_csv(cluster_results_csv, index=False)
        print(f'Cluster results saved to {cluster_results_csv}')

    return all_df

def reducer_hyperparameter_search(settings={}, reduction_params=None, dbscan_params=None, kmeans_params=None, save=False):
    """
    Perform a hyperparameter search for UMAP or tSNE on the given data.
    
    Parameters:
    settings (dict): Dictionary containing the following keys:
    src (str): Source directory containing the data.
    row_limit (int): Limit the number of rows to process.
    tables (list): List of table names to read from the database.
    filter_by (str): Column to filter the data.
    sample_size (int): Number of samples to use for the hyperparameter search.
    remove_highly_correlated (bool): Whether to remove highly correlated columns.
    log_data (bool): Whether to log transform the data.
    verbose (bool): Whether to print verbose output.
    reduction_method (str): Dimensionality reduction method ('UMAP' or 'tSNE').
    reduction_params (list): List of dictionaries containing hyperparameters to test for the reduction method.
    dbscan_params (list): List of dictionaries containing DBSCAN hyperparameters to test.
    kmeans_params (list): List of dictionaries containing KMeans hyperparameters to test.
    pointsize (int): Size of the points in the scatter plot.
    save (bool): Whether to save the resulting plot as a file.
    
    Returns:
    None
    """
    
    from .io import _read_and_join_tables
    from .utils import get_db_paths, preprocess_data, search_reduction_and_clustering, generate_colors, map_condition
    from .settings import set_default_umap_image_settings

    settings = set_default_umap_image_settings(settings)
    pointsize = settings['dot_size']
    if isinstance(dbscan_params, dict):
        dbscan_params = [dbscan_params]

    if isinstance(kmeans_params, dict):
        kmeans_params = [kmeans_params]

    if isinstance(reduction_params, dict):
        reduction_params = [reduction_params]

    # Determine reduction method based on the keys in reduction_param
    if any('n_neighbors' in param for param in reduction_params):
        reduction_method = 'umap'
    elif any('perplexity' in param for param in reduction_params):
        reduction_method = 'tsne'
    elif any('perplexity' in param for param in reduction_params) and any('n_neighbors' in param for param in reduction_params):
        raise ValueError("Reduction parameters must include 'n_neighbors' for UMAP or 'perplexity' for tSNE, not both.")
    
    if settings['reduction_method'].lower() != reduction_method:
        settings['reduction_method'] = reduction_method
        print(f'Changed reduction method to {reduction_method} based on the provided parameters.')
    
    if settings['verbose']:
        display(pd.DataFrame(list(settings.items()), columns=['Key', 'Value']))

    db_paths = get_db_paths(settings['src'])
    
    tables = settings['tables']
    all_df = pd.DataFrame()
    for db_path in db_paths:
        df = _read_and_join_tables(db_path, table_names=tables)
        all_df = pd.concat([all_df, df], axis=0)

    all_df['cond'] = all_df['columnID'].apply(map_condition, neg=settings['neg'], pos=settings['pos'], mix=settings['mix'])

    if settings['exclude_conditions']:
        if isinstance(settings['exclude_conditions'], str):
            settings['exclude_conditions'] = [settings['exclude_conditions']]
        row_count_before = len(all_df)
        all_df = all_df[~all_df['cond'].isin(settings['exclude_conditions'])]
        if settings['verbose']:
            print(f'Excluded {row_count_before - len(all_df)} rows after excluding: {settings["exclude_conditions"]}, rows left: {len(all_df)}')

    if settings['row_limit'] is not None:
        all_df = all_df.sample(n=settings['row_limit'], random_state=42)

    numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])

    # Combine DBSCAN and KMeans parameters
    clustering_params = []
    if dbscan_params:
        for param in dbscan_params:
            param['method'] = 'dbscan'
            clustering_params.append(param)
    if kmeans_params:
        for param in kmeans_params:
            param['method'] = 'kmeans'
            clustering_params.append(param)

    print('Testing paramiters:', reduction_params)
    print('Testing clustering paramiters:', clustering_params)

    # Calculate the grid size
    grid_rows = len(reduction_params)
    grid_cols = len(clustering_params)

    fig_width = grid_cols*10
    fig_height = grid_rows*10

    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))

    # Make sure axs is always an array of axes
    axs = np.atleast_1d(axs)
    
    # Iterate through the Cartesian product of reduction and clustering hyperparameters
    for i, reduction_param in enumerate(reduction_params):
        for j, clustering_param in enumerate(clustering_params):
            if len(clustering_params) <= 1:
                axs[i].axis('off')
                ax = axs[i]
            elif len(reduction_params) <= 1:
                axs[j].axis('off')
                ax = axs[j]
            else:
                ax = axs[i, j]

            # Perform dimensionality reduction and clustering
            if settings['reduction_method'].lower() == 'umap':
                n_neighbors = reduction_param.get('n_neighbors', 15)

                if isinstance(n_neighbors, float):
                    n_neighbors = int(n_neighbors * len(numeric_data))

                min_dist = reduction_param.get('min_dist', 0.1)
                embedding, labels = search_reduction_and_clustering(numeric_data, n_neighbors, min_dist, settings['metric'], 
                                                                    clustering_param.get('eps', 0.5), clustering_param.get('min_samples', 5), 
                                                                    clustering_param['method'], settings['reduction_method'], settings['verbose'], reduction_param, n_jobs=settings['n_jobs'])
                
            elif settings['reduction_method'].lower() == 'tsne':
                perplexity = reduction_param.get('perplexity', 30)

                if isinstance(perplexity, float):
                    perplexity = int(perplexity * len(numeric_data))

                embedding, labels = search_reduction_and_clustering(numeric_data, perplexity, 0.1, settings['metric'], 
                                                                    clustering_param.get('eps', 0.5), clustering_param.get('min_samples', 5), 
                                                                    clustering_param['method'], settings['reduction_method'], settings['verbose'], reduction_param, n_jobs=settings['n_jobs'])
                
            else:
                raise ValueError(f"Unsupported reduction method: {settings['reduction_method']}. Supported methods are 'UMAP' and 'tSNE'")

            # Plot the results
            if settings['color_by']:
                unique_groups = all_df[settings['color_by']].unique()
                colors = generate_colors(len(unique_groups), False)
                for group, color in zip(unique_groups, colors):
                    indices = all_df[settings['color_by']] == group
                    ax.scatter(embedding[indices, 0], embedding[indices, 1], s=pointsize, label=f"{group}", color=color)
            else:
                unique_labels = np.unique(labels)
                colors = generate_colors(len(unique_labels), False)
                for label, color in zip(unique_labels, colors):
                    ax.scatter(embedding[labels == label, 0], embedding[labels == label, 1], s=pointsize, label=f"Cluster {label}", color=color)

            ax.set_title(f"{settings['reduction_method']} {reduction_param}\n{clustering_param['method']} {clustering_param}")
            ax.legend()

    plt.tight_layout()
    if save:
        results_dir = os.path.join(settings['src'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'hyperparameter_search.pdf'))
    else:
        plt.show()

    return

def generate_mediar_masks(src, settings, object_type):
    """
    Generates masks using the MEDIARPredictor.
    
    :param src: Source folder containing images or npz files.
    :param settings: Dictionary of settings for generating masks.
    :param object_type: Type of object to detect (e.g., 'cell', 'nucleus', etc.).
    """
    from .mediar import MEDIARPredictor
    from .io import _create_database, _save_object_counts_to_database
    from .plot import plot_masks
    from .settings import set_default_settings_preprocess_generate_masks
    from .utils import prepare_batch_for_segmentation

    # Clear CUDA cache and check if CUDA is available
    gc.collect()
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')

    settings['src'] = src

    # Preprocess settings
    settings = set_default_settings_preprocess_generate_masks(settings)

    if settings['verbose']:
        settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
        settings_df['setting_value'] = settings_df['setting_value'].apply(str)
        display(settings_df)

    figuresize = 10
    timelapse = settings['timelapse']
    batch_size = settings['batch_size']

    # Get object settings and initialize MEDIARPredictor
    mediar_predictor = MEDIARPredictor(input_path=None, output_path=None, normalize=settings['normalize'], use_tta=False)

    # Paths to input npz files
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]

    # Initialize a database for saving measurements
    count_loc = os.path.join(os.path.dirname(src), 'measurements', 'measurements.db')
    os.makedirs(os.path.dirname(src) + '/measurements', exist_ok=True)
    _create_database(count_loc)

    for file_index, path in enumerate(paths):
        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        output_folder = os.path.join(os.path.dirname(path), f'{object_type}_mask_stack')
        os.makedirs(output_folder, exist_ok=True)

        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

            for i, filename in enumerate(filenames):
                output_path = os.path.join(output_folder, filename)
                if os.path.exists(output_path):
                    print(f"File {filename} already exists. Skipping...")
                    continue

        # Process each batch of images in the stack
        for i in range(0, stack.shape[0], batch_size):
            batch = stack[i: i + batch_size]
            batch_filenames = filenames[i: i + batch_size]

            # Prepare batch for MEDIARPredictor (optional)
            batch = prepare_batch_for_segmentation(batch)

            # Predict masks using MEDIARPredictor
            predicted_masks = mediar_predictor.predict_batch(batch)

            # Save predicted masks
            for j, mask in enumerate(predicted_masks):
                output_filename = os.path.join(output_folder, batch_filenames[j])
                mask = mask.astype(np.uint16)
                np.save(output_filename, mask)

            # Optional: Plot the masks
            if settings['plot']:
                for idx, mask in enumerate(predicted_masks):
                    plot_masks(batch[idx], mask, cmap='inferno', figuresize=figuresize)

            # Save object counts to database
            _save_object_counts_to_database(predicted_masks, object_type, batch_filenames, count_loc)

        # Clear CUDA cache after each file
        gc.collect()
        torch.cuda.empty_cache()

    print("Mask generation completed.")

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