
import os, gc, torch, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from IPython.display import display
import warnings
from cellpose import models as cp_models

from functools import partial
from skimage.segmentation import watershed
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.filters import (threshold_otsu,threshold_local,frangi,sato,meijering,gaussian,difference_of_gaussians,apply_hysteresis_threshold)
from skimage.feature import blob_log, blob_dog, peak_local_max
from skimage.morphology import (remove_small_objects,remove_small_holes,binary_opening,binary_closing,binary_dilation,binary_erosion,disk,skeletonize,white_tophat)
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.restoration import rolling_ball

warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

def merge_split_filter_masks(masks, intensity_images, settings, object_type, batch_filenames=None):
    """Apply merge/split/filter operations directly to in-memory masks."""
    import numpy as np
    from joblib import Parallel, delayed
    from .utils import print_progress, _process_single_fov_in_memory

    pf = settings.get(f'{object_type}_perimeter_fraction', settings.get(f'{object_type}_perimiter_fraction', 0))
    im = settings.get(f'{object_type}_intensity_merge', False)
    isp = settings.get(f'{object_type}_intensity_split', False)
    moa = settings.get(f'{object_type}_min_object_area', 0)
    mna = settings.get(f'{object_type}_min_area', 0)
    mxa = settings.get(f'{object_type}_max_area', 0)
    rb = settings.get(f'{object_type}_remove_border_objects', False)
    mni = settings.get(f'{object_type}_min_intensity_percentile', 0)
    mxi = settings.get(f'{object_type}_max_intensity_percentile', 100)

    needs_work = (
        pf > 0 or im or isp or moa > 0 or mna > 0 or
        (mxa and mxa > 0) or rb or mni > 0 or mxi < 100
    )

    if not needs_work:
        print(f"merge_split_filter_masks({object_type}): no operations needed, skipping")
        return masks

    if masks is None:
        return None

    print(f"merge_split_filter_masks({object_type}): "
          f"perimeter_merge={pf > 0}(frac={pf}), intensity_merge={im}, "
          f"split={isp}, min_area={mna}, max_area={mxa}, "
          f"remove_border={rb}, intensity_pct=[{mni}, {mxi}]")

    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            raise ValueError(f"Unsupported masks ndim: {masks.ndim}")
    else:
        mask_list = list(masks)

    if isinstance(intensity_images, np.ndarray):
        if intensity_images.ndim == 2:
            intensity_list = [intensity_images]
        elif intensity_images.ndim == 3:
            intensity_list = [intensity_images[i] for i in range(intensity_images.shape[0])]
        elif intensity_images.ndim == 4:
            intensity_list = [intensity_images[i] for i in range(intensity_images.shape[0])]
        else:
            raise ValueError(f"Unsupported intensity_images ndim: {intensity_images.ndim}")
    else:
        intensity_list = list(intensity_images)

    if len(mask_list) != len(intensity_list):
        raise ValueError(
            f"Number of masks ({len(mask_list)}) does not match number of intensity images ({len(intensity_list)})."
        )

    if batch_filenames is None:
        batch_filenames = [f'image_{i:06d}' for i in range(len(mask_list))]

    total = len(mask_list)
    time_ls = []

    def _progress(fov_idx, total_fovs, duration, op):
        time_ls.append(duration)
        print_progress(
            fov_idx + 1,
            total_fovs,
            n_jobs=1,
            time_ls=time_ls,
            batch_size=None,
            operation_type=op
        )

    def _run_one(idx, mask, intensity_img):
        out_mask = _process_single_fov_in_memory(
            mask=mask,
            intensity_img=intensity_img,
            intensity_channel=0,
            do_split=isp,
            do_perimeter_merge=(pf > 0),
            do_intensity_merge=(im and intensity_images is not None),
            perimeter_fraction=pf,
            area_multiplier=settings.get(f'{object_type}_area_multiplier', 2.0),
            min_distance=settings.get(f'{object_type}_min_distance', 10),
            min_object_area=moa,
            intensity_threshold_method=settings.get(f'{object_type}_intensity_threshold_method', 'mean'),
            intensity_percentile=settings.get(f'{object_type}_intensity_percentile', 75),
            min_area=mna,
            max_area=mxa if mxa else 0,
            remove_border_objects=rb,
            min_intensity_percentile=mni,
            max_intensity_percentile=mxi,
            progress_callback=_progress,
            fov_index=idx,
            total_fovs=total,
            op_name=f'merge_{object_type}',
        )
        return out_mask

    n_jobs = settings.get('n_jobs', 1)
    
    # Always run serial so progress prints work
    filtered_masks = [
        _run_one(idx, mask, img)
        for idx, (mask, img) in enumerate(zip(mask_list, intensity_list))
    ]

    return filtered_masks

def generate_cellpose_masks_sam(src, settings, object_type):
    
    from .utils import _masks_to_masks_stack, all_elements_match, prepare_batch_for_segmentation, _get_cellpose_channels
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
        
    if settings.get('cellpose_nucleus_channel') is None and settings.get('nucleus_channel') is not None:
        settings['cellpose_nucleus_channel'] = settings['nucleus_channel']
    
    if settings.get('cellpose_cell_channel') is None and settings.get('cell_channel') is not None:
        settings['cellpose_cell_channel'] = settings['cell_channel']
    
    if settings.get('cellpose_pathogen_channel') is None and settings.get('pathogen_channel') is not None:
        settings['cellpose_pathogen_channel'] = settings['pathogen_channel']
        
    channels_to_extract, cellpose_channels = _get_cellpose_channels(settings)
    channels = cellpose_channels.get(object_type, [])
    
    if len(channels) == 0:
        raise ValueError(f"No valid channels defined for object_type '{object_type}'.")
        
    if settings['verbose']:
        print(channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model='cpsam', device=device)
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
                batch = stack[i: i+batch_size, :, :, [0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)
            
            # In the future drop the npz save file step, just keep it in memory and pass the batch directly to the model. This will save time and disk space. For now, keep it for backwards compatibility and to avoid issues with large batches that might not fit in memory.                
            #if stack.shape[3] == 1:
            #    batch = stack[i: i+batch_size, :, :, [0]].astype(stack.dtype)
            #else:
            #    subset = stack[i: i+batch_size, :, :, channels_to_extract].astype(stack.dtype)
            #    batch = subset[:, :, :, channels]

            batch_filenames = filenames[i: i+batch_size].tolist()

            if not settings['plot']:
                batch, batch_filenames = _check_masks(batch, batch_filenames, output_folder)
            if batch.size == 0:
                continue
            
            cp_batch = prepare_batch_for_segmentation(batch)
            batch_list = [cp_batch[i] for i in range(cp_batch.shape[0])]

            if timelapse:
                movie_path = os.path.join(os.path.dirname(src), 'movies')
                os.makedirs(movie_path, exist_ok=True)
                save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                _npz_to_movie(cp_batch, batch_filenames, save_path, fps=2)
                
            
            output = model.eval(
                x=batch_list,
                batch_size=len(batch_list),
                normalize=False,
                channel_axis=-1,
                min_size=object_settings['min_size'],
                progress=True,
                diameter=None,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                resample=object_settings['resample']
                )
                 
            masks, flows, _, _, _ = parse_cellpose4_output(output)
            
            masks = merge_split_filter_masks(
                masks=masks,
                intensity_images=batch,
                settings=settings,
                object_type=object_type,
                batch_filenames=batch_filenames,
            )
            
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
                                                         n_jobs=n_jobs,
                                                         batch_list=None,
                                                         optimizer_time_limit_s=120,
                                                         optimizer_mip_gap=0.01,
                                                         run_optimization=True,
                                                         max_objects_for_optimization=20000)
                    
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
                print("saving to DB")
                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                mask_stack = _masks_to_masks_stack(masks)
        
            if timelapse and settings.get("motility_analysis", False):
                from .timelapse import automated_motility_assay
                _ = automated_motility_assay(settings)
            
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
                plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=len(batch_list))
                
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

def generate_cellpose_masks(src, settings, object_type):
    
    from .utils import _masks_to_masks_stack, _filter_cp_masks, _get_cellpose_channels, _choose_model, all_elements_match, prepare_batch_for_segmentation
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
    
    if settings.get('cellpose_nucleus_channel') is None and settings.get('nucleus_channel') is not None:
        settings['cellpose_nucleus_channel'] = settings['nucleus_channel']

    if settings.get('cellpose_cell_channel') is None and settings.get('cell_channel') is not None:
        settings['cellpose_cell_channel'] = settings['cell_channel']

    if settings.get('cellpose_pathogen_channel') is None and settings.get('pathogen_channel') is not None:
        settings['cellpose_pathogen_channel'] = settings['pathogen_channel']

    cellpose_channels = _get_cellpose_channels(
        src,
        settings.get('cellpose_nucleus_channel'),
        settings.get('cellpose_pathogen_channel'),
        settings.get('cellpose_cell_channel')
    )
        
    if settings['verbose']:
        print(cellpose_channels)
        
    if object_type not in cellpose_channels:
        raise ValueError(f"Error: No channels were specified for object_type '{object_type}'. Check your settings.")
    
    channels = cellpose_channels[object_type]

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
                                                         n_jobs=n_jobs,
                                                         batch_list=None,
                                                         optimizer_time_limit_s=120,
                                                         optimizer_mip_gap=0.01,
                                                         run_optimization=True,
                                                         max_objects_for_optimization=20000)
                    
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
        
            if timelapse and settings.get("motility_analysis", False):
                from .timelapse import automated_motility_assay
                _ = automated_motility_assay(settings)
            
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
                print(f"plotting")
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

def _segment_single_image(img, settings):
    """
    Segment a single 2-D image. This is the unit of work for multiprocessing.
    Must be a top-level function (picklable).
 
    Parameters
    ----------
    img : np.ndarray
        2-D float32 image.
    settings : dict
        Classical segmentation settings (pickle-safe).
 
    Returns
    -------
    np.ndarray
        2-D int32 label array.
    """
    morphology = settings['organelle_morphology']
    method = settings['organelle_method']
 
    if morphology == 'spots':
        return _segment_spots(img, method, settings)
    elif morphology == 'network':
        return _segment_network(img, method, settings)
    elif morphology == 'irregular':
        return _segment_irregular(img, method, settings)
    else:
        raise ValueError(f"Unknown morphology: {morphology}")

def _segment_spots(img, method, settings):
    """
    Segment punctate / spot-like organelles.
 
    Strategies
    ----------
    'otsu'     : top-hat -> Otsu -> watershed
    'adaptive' : top-hat -> adaptive threshold -> watershed
    'log'      : Laplacian-of-Gaussian blob detection -> marker-based watershed
    """
    tophat_radius = settings['organelle_tophat_radius']
    use_watershed = settings['organelle_watershed_spots']
 
    if method == 'log':
        return _spots_log(img, settings, use_watershed)
 
    # --- Pre-filter: white top-hat enhances bright spots on dark bg ---
    filtered = white_tophat(img, disk(tophat_radius))
 
    # --- Threshold ---
    if method == 'otsu':
        thresh_val = threshold_otsu(filtered)
        binary = filtered > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(filtered, block_size=block, offset=offset)
        binary = filtered > local_thresh
    else:
        raise ValueError(f"Unsupported spot method: {method}")
 
    # --- Morphological cleanup ---
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])
 
    # --- Watershed to split touching spots ---
    if use_watershed:
        labeled = _watershed_split(binary, filtered)
    else:
        labeled = sk_label(binary)
 
    return labeled

def _segment_network(img, method, settings):
    """
    Segment filamentous / reticular organelles.
 
    Strategies
    ----------
    'otsu'     : Gaussian smooth -> Otsu -> morphological cleanup
    'adaptive' : Gaussian smooth -> adaptive threshold -> cleanup
    'ridge'    : Ridge filter (Frangi / Sato / Meijering) -> threshold -> label
    """
    if method == 'ridge':
        return _network_ridge(img, settings)
 
    smooth = gaussian(img, sigma=1)
 
    if method == 'otsu':
        thresh_val = threshold_otsu(smooth)
        binary = smooth > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(smooth, block_size=block, offset=offset)
        binary = smooth > local_thresh
    else:
        raise ValueError(f"Unsupported network method: {method}")
 
    morph_r = max(settings['organelle_morph_radius'] // 2, 1)
    binary = binary_closing(binary, disk(morph_r))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])
 
    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)
 
    return sk_label(binary)

def generate_organelle_masks_sam(src, settings, object_type):
    """
    Generate organelle masks using multiple segmentation strategies.

    Supports four morphology modes:
        - 'spots': punctate structures (lipid droplets, vesicles, peroxisomes)
        - 'network': filamentous/reticular structures (mitochondria, microtubules, ER tubules)
        - 'irregular': irregular-shaped organelles (Golgi, ER cisternae, lysosomes)
        - 'ring': hollow / ring-shaped structures (endosomes, autophagosomes, late lysosomes)

    Each mode can use different backends:
        - 'cellpose': deep-learning segmentation via Cellpose
        - 'stardist': star-convex polygon instance segmentation (spots only)
        - 'otsu': global Otsu thresholding with morphological cleanup
        - 'adaptive': local adaptive thresholding
        - 'log': Laplacian of Gaussian blob detection (spots, ring)
        - 'dog': Difference of Gaussians blob detection (spots, ring)
        - 'ridge': ridge/tubeness filter (network only)
        - 'hysteresis': dual-threshold hysteresis (network only)
        - 'unet': user-provided U-Net semantic segmentation (network only)

    Parameters
    ----------
    src : str
        Path to the mask source directory containing .npz stacks.
    settings : dict
        Configuration dictionary. Organelle-specific keys (all prefixed with
        'organelle_') are documented in _set_organelle_defaults.
    object_type : str
        Should be 'organelle' (or a custom name used for folder naming).

    Returns
    -------
    None
        Masks are saved as .npy files in ``{src}/{object_type}_mask_stack/``.
    """

    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .utils import _masks_to_masks_stack, _filter_cp_masks, prepare_batch_for_segmentation
    from .settings import _set_organelle_defaults
    from.plot import plot_organelle_output

    gc.collect()

    settings = _set_organelle_defaults(settings)

    morphology = settings['organelle_morphology']
    method = settings['organelle_method']
    organelle_channel = settings['organelle_channel']

    _validate_organelle_settings(morphology, method)

    n_jobs = settings.get('n_jobs', 1)
    if n_jobs < 1:
        n_jobs = 1

    if settings['verbose']:
        import pandas as pd
        from IPython.display import display
        organ_keys = {k: v for k, v in settings.items() if k.startswith('organelle_')}
        df = pd.DataFrame(list(organ_keys.items()), columns=['setting_key', 'setting_value'])
        df['setting_value'] = df['setting_value'].apply(str)
        display(df)

    paths = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.npz')]
    if not paths:
        print(f'No .npz files found in {src}')
        return

    count_loc = os.path.join(os.path.dirname(src), 'measurements', 'measurements.db')
    os.makedirs(os.path.dirname(count_loc), exist_ok=True)
    _create_database(count_loc)

    batch_size = settings['batch_size']
    average_sizes = []
    average_counts = []
    time_ls = []

    # ------------------------------------------------------------------ #
    #  Load deep-learning model once (if needed)
    # ------------------------------------------------------------------ #
    dl_model = None
    is_dl_method = method in ('cellpose', 'stardist', 'unet')

    if method == 'cellpose':
        from .utils import _choose_model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dl_model = _choose_model(
            settings['organelle_model_name'],
            device,
            object_type=object_type,
            restore_type=None,
            object_settings=_build_object_settings(settings),
        )
    elif method == 'stardist':
        #dl_model = _load_stardist_model(settings)
        print(f"stardist is disabled in this verssion of spacr")
    elif method == 'unet':
        dl_model = _load_unet_model(settings)

    # ------------------------------------------------------------------ #
    #  Build a serialisable settings subset for worker processes
    # ------------------------------------------------------------------ #
    classical_settings = _extract_classical_settings(settings)

    # ------------------------------------------------------------------ #
    #  Optionally load cell masks for per-cell masking
    # ------------------------------------------------------------------ #
    cell_mask_folder = None
    if settings.get('organelle_mask_within_cells', False):
        candidate = os.path.join(os.path.dirname(src), 'cell_mask_stack')
        if os.path.exists(candidate):
            cell_mask_folder = candidate
            print(f'Per-cell masking enabled, using cell masks from {candidate}')
        else:
            print(f'Warning: organelle_mask_within_cells=True but no cell_mask_stack found at {candidate}')

    # ------------------------------------------------------------------ #
    #  Main loop over .npz stacks
    # ------------------------------------------------------------------ #
    for file_index, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        output_folder = os.path.join(os.path.dirname(path), f'{object_type}_mask_stack')
        os.makedirs(output_folder, exist_ok=True)

        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

        # Skip already-processed files
        existing = set(os.listdir(output_folder))
        todo_indices = [i for i, fn in enumerate(filenames) if fn not in existing]
        if not todo_indices:
            print(f'All files in {name} already processed. Skipping.')
            continue

        for i in range(0, stack.shape[0], batch_size):
            start = time.time()
            batch = stack[i: i + batch_size]
            batch_filenames = filenames[i: i + batch_size].tolist()

            # ---------------------------------------------------------- #
            #  Extract the organelle channel
            # ---------------------------------------------------------- #
            if organelle_channel is not None:
                if batch.ndim == 4:
                    img_batch = batch[:, :, :, organelle_channel].astype(np.float32)
                else:
                    img_batch = batch.astype(np.float32)
            else:
                if batch.ndim == 4:
                    img_batch = batch[:, :, :, 0].astype(np.float32)
                else:
                    img_batch = batch.astype(np.float32)

            # ---------------------------------------------------------- #
            #  Per-cell masking: zero out pixels outside cells
            # ---------------------------------------------------------- #
            if cell_mask_folder is not None:
                img_batch = _apply_cell_mask(img_batch, batch_filenames, cell_mask_folder)

            # ---------------------------------------------------------- #
            #  Preprocessing: rolling ball and/or CLAHE
            # ---------------------------------------------------------- #
            img_batch = _preprocess_batch(img_batch, settings)

            # ---------------------------------------------------------- #
            #  Segment
            # ---------------------------------------------------------- #
            if method == 'cellpose':
                masks = _segment_cellpose_sam(
                    img_batch, batch_filenames, dl_model, settings, object_type, output_folder)
            elif method == 'stardist':
                masks = _segment_stardist(img_batch, dl_model, settings)
            elif method == 'unet':
                masks = _segment_unet(img_batch, dl_model, settings)
            else:
                # CPU-bound classical methods — parallelise
                masks = _segment_classical_parallel(
                    img_batch, classical_settings, n_jobs=n_jobs,
                )

            if masks is None or len(masks) == 0:
                continue

            # ---------------------------------------------------------- #
            #  Post-process: size filter, border removal
            # ---------------------------------------------------------- #
            mask_stack = _postprocess_masks(
                masks,
                min_size=settings['organelle_min_size'],
                max_size=settings['organelle_max_size'],
                remove_border=settings['organelle_remove_border'],
            )

            _save_object_counts_to_database(
                mask_stack, object_type, batch_filenames, count_loc, added_string='',
            )

            # Stats
            if not np.any(mask_stack):
                avg_count, avg_size = 0, 0
            else:
                avg_count, avg_size = _get_avg_object_size(mask_stack)

            average_counts.append(avg_count)
            average_sizes.append(avg_size)
            overall_avg_count = np.mean(average_counts)
            overall_avg_size = np.mean(average_sizes)

            stop = time.time()
            duration = stop - start
            time_ls.append(duration)

            print(
                f'Found {overall_avg_count:.1f} {object_type}/FOV, '
                f'average size: {overall_avg_size:.1f} px2 '
                f'[batch {file_index+1}/{len(paths)}, {duration:.1f}s, '
                f'n_jobs={n_jobs if not is_dl_method else "GPU"}]'
            )
            
            # ---------------------------------------------------------- #
            #  Plot (if enabled)
            # ---------------------------------------------------------- #
            if settings.get('plot', False):
                plot_organelle_output(
                    img_batch[: len(mask_stack)],
                    mask_stack,
                    settings,
                    cmap='inferno',
                    figuresize=10,
                    nr=min(settings.get('examples_to_plot', 1), len(mask_stack)),
                    print_object_number=True,
                )

            # ---------------------------------------------------------- #
            #  Save
            # ---------------------------------------------------------- #
            if settings['save']:
                for mask_idx, mask in enumerate(mask_stack):
                    out_path = os.path.join(output_folder, batch_filenames[mask_idx])
                    np.save(out_path, mask.astype(np.uint16))
                mask_stack = []
                batch_filenames = []

            gc.collect()

    torch.cuda.empty_cache()
    return

def _validate_organelle_settings(morphology, method):
    """Raise early on invalid morphology / method combinations."""
    valid_morphologies = ('spots', 'network', 'irregular', 'ring')
    if morphology not in valid_morphologies:
        raise ValueError(
            f"organelle_morphology must be one of {valid_morphologies}, got '{morphology}'"
        )

    method_map = {
        'spots': ('otsu', 'adaptive', 'log', 'dog', 'cellpose', 'stardist'),
        'network': ('otsu', 'adaptive', 'ridge', 'hysteresis', 'cellpose', 'unet'),
        'irregular': ('otsu', 'adaptive', 'cellpose'),
        'ring': ('otsu', 'adaptive', 'dog', 'log', 'cellpose'),
    }
    valid_methods = method_map[morphology]
    if method not in valid_methods:
        raise ValueError(
            f"For morphology='{morphology}', method must be one of {valid_methods}, got '{method}'"
        )


def _build_object_settings(settings):
    """Build an object_settings dict expected by _choose_model / cellpose eval."""
    return {
        'model_name': settings['organelle_model_name'],
        'diameter': settings['organelle_diameter'],
        'minimum_size': settings['organelle_min_size'],
        'maximum_size': settings['organelle_max_size'],
        'resample': settings['organelle_resample'],
        'filter_size': False,
        'filter_intensity': False,
        'remove_border_objects': settings['organelle_remove_border'],
        'merge': False,
    }


def _extract_classical_settings(settings):
    """
    Extract only the plain-data keys needed by classical segmentation workers.
    This dict is safe to pickle for multiprocessing.
    """
    keys = [
        'organelle_morphology', 'organelle_method',
        'organelle_min_size', 'organelle_max_size',
        # Spots
        'organelle_tophat_radius', 'organelle_watershed_spots',
        'organelle_log_min_sigma', 'organelle_log_max_sigma',
        'organelle_log_num_sigma', 'organelle_log_threshold',
        'organelle_dog_sigma_low', 'organelle_dog_sigma_high',
        # Network
        'organelle_ridge_sigmas', 'organelle_ridge_filter',
        'organelle_skeletonize', 'organelle_network_threshold',
        'organelle_hysteresis_low', 'organelle_hysteresis_high',
        # Irregular
        'organelle_adaptive_block_size', 'organelle_adaptive_offset',
        'organelle_morph_radius', 'organelle_fill_holes',
        # Ring
        'organelle_ring_sigma_inner', 'organelle_ring_sigma_outer',
        'organelle_ring_min_prominence', 'organelle_ring_fill_method',
    ]
    return {k: settings[k] for k in keys if k in settings}


# ====================================================================== #
#  Preprocessing
# ====================================================================== #

def _preprocess_batch(img_batch, settings):
    """
    Apply optional preprocessing to the entire image batch.

    Operations (applied in order if enabled):
        1. Rolling ball background subtraction
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization)

    Parameters
    ----------
    img_batch : np.ndarray
        Shape (N, H, W) float32.
    settings : dict

    Returns
    -------
    np.ndarray
        Preprocessed batch, same shape.
    """
    do_rolling_ball = settings.get('organelle_rolling_ball', False)
    do_clahe = settings.get('organelle_clahe', False)

    if not do_rolling_ball and not do_clahe:
        return img_batch

    out = img_batch.copy()

    for idx in range(out.shape[0]):
        img = out[idx]

        if do_rolling_ball:
            radius = settings.get('organelle_rolling_ball_radius', 50)
            bg = rolling_ball(img, radius=radius)
            img = img - bg
            img = np.clip(img, 0, None)

        if do_clahe:
            clip_limit = settings.get('organelle_clahe_clip_limit', 0.01)
            pmin, pmax = np.percentile(img, (0.5, 99.5))
            if pmax - pmin > 0:
                img_norm = np.clip((img - pmin) / (pmax - pmin), 0, 1)
            else:
                img_norm = np.zeros_like(img)
            img = equalize_adapthist(img_norm, clip_limit=clip_limit).astype(np.float32)

        out[idx] = img

    return out


def _apply_cell_mask(img_batch, batch_filenames, cell_mask_folder):
    """
    Zero out pixels outside cell boundaries for per-cell organelle detection.

    Parameters
    ----------
    img_batch : np.ndarray
        Shape (N, H, W) float32.
    batch_filenames : list of str
    cell_mask_folder : str
        Path to cell_mask_stack directory.

    Returns
    -------
    np.ndarray
        Masked batch.
    """
    out = img_batch.copy()
    for idx, fn in enumerate(batch_filenames):
        cell_mask_path = os.path.join(cell_mask_folder, fn)
        if os.path.exists(cell_mask_path):
            cell_mask = np.load(cell_mask_path)
            out[idx][cell_mask == 0] = 0
        else:
            cell_mask_path_npy = cell_mask_path if cell_mask_path.endswith('.npy') else cell_mask_path + '.npy'
            if os.path.exists(cell_mask_path_npy):
                cell_mask = np.load(cell_mask_path_npy)
                out[idx][cell_mask == 0] = 0
    return out


# ====================================================================== #
#  Deep-learning model loaders
# ====================================================================== #

#def _load_stardist_model(settings):
#    """Load a Stardist model (pretrained or custom)."""
#    import os
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
#    # Free any PyTorch GPU state
#    torch.cuda.empty_cache()
#    torch.cuda.synchronize()
#
#    try:
#        from stardist.models import StarDist2D  # type: ignore
#    except (ImportError, RuntimeError) as e:
#        raise ImportError(
#            f"Stardist requires TensorFlow which is not installed. "
#            f"Either install TensorFlow ('pip install tensorflow') or use a "
#            f"different method for spot detection: 'cellpose', 'otsu', 'log', or 'dog'. "
#            f"Original error: {e}"
#        )
#
#    model_name = settings.get('organelle_stardist_model', '2D_versatile_fluo')
#
#    # Try GPU first, fall back to CPU if CUDA context is corrupted
#    for attempt, use_gpu in enumerate([True, False]):
#        try:
#            import tensorflow as tf
#            if use_gpu:
#                gpus = tf.config.list_physical_devices('GPU')
#                if gpus:
#                    for gpu in gpus:
#                        tf.config.experimental.set_memory_growth(gpu, True)
#                    print(f'Loading Stardist model on GPU: {model_name}...')
#                else:
#                    print(f'No GPU found, loading Stardist model on CPU: {model_name}...')
#            else:
#                tf.config.set_visible_devices([], 'GPU')
#                print(f'GPU unavailable for TensorFlow, loading Stardist model on CPU: {model_name}...')
#
#            if os.path.isdir(model_name):
#                model = StarDist2D(None, name=os.path.basename(model_name),
#                                   basedir=os.path.dirname(model_name))
#            else:
#                model = StarDist2D.from_pretrained(model_name)
#
#            print(f'Stardist model loaded successfully ({"GPU" if use_gpu and gpus else "CPU"}).')
#            return model
#
#        except Exception as e:
#            if attempt == 0:
#                print(f'Warning: Failed to load Stardist on GPU ({e}). Falling back to CPU...')
#                # Reset TF state for CPU retry
#                try:
#                    import tensorflow as tf
#                    tf.config.set_visible_devices([], 'GPU')
#                except Exception:
#                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#            else:
#                raise RuntimeError(
#                    f"Failed to load Stardist model on both GPU and CPU. "
#                    f"Error: {e}"
#                )

def _load_unet_model(settings):
    """Load a user-provided U-Net model from a .pt / .pth file."""
    model_path = settings.get('organelle_unet_model_path')
    if model_path is None or not os.path.exists(model_path):
        raise ValueError(
            f"organelle_unet_model_path must point to a valid .pt/.pth file, "
            f"got '{model_path}'"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# ====================================================================== #
#  Cellpose segmentation
# ====================================================================== #

def _segment_cellpose(batch, batch_filenames, model, settings, object_type, output_folder):
    """Run Cellpose on a batch and return a list of 2-D label arrays."""
    from .utils import prepare_batch_for_segmentation
    from .io import _check_masks
    from .spacr_cellpose import parse_cellpose4_output

    organelle_ch = settings['organelle_channel']
    if organelle_ch is None:
        organelle_ch = 0

    if batch.ndim == 4:
        ch0 = batch[:, :, :, organelle_ch: organelle_ch + 1]
        nuc_ch = settings.get('nucleus_channel')
        if nuc_ch is not None and nuc_ch < batch.shape[3]:
            ch1 = batch[:, :, :, nuc_ch: nuc_ch + 1]
        else:
            ch1 = ch0
        cp_batch = np.concatenate([ch0, ch1], axis=-1).astype(batch.dtype)
    else:
        cp_batch = np.stack([batch, batch], axis=-1).astype(batch.dtype)

    if not settings.get('plot', False):
        cp_batch, batch_filenames = _check_masks(cp_batch, batch_filenames, output_folder)
    if cp_batch.size == 0:
        return None

    cp_batch = prepare_batch_for_segmentation(cp_batch)
    batch_list = [cp_batch[j] for j in range(cp_batch.shape[0])]

    output = model.eval(
        x=batch_list,
        batch_size=settings['batch_size'],
        normalize=False,
        channel_axis=-1,
        channels=[0, 1],
        diameter=settings['organelle_diameter'],
        flow_threshold=settings['organelle_FT'],
        cellprob_threshold=settings['organelle_CP_prob'],
        rescale=None,
        resample=settings['organelle_resample'],
    )

    masks, flows, _, _, _ = parse_cellpose4_output(output)
    return masks

def _segment_cellpose_sam(batch, batch_filenames, model, settings, object_type, output_folder):
    """Run Cellpose-SAM on a batch and return a list of 2-D label arrays."""
    from .utils import prepare_batch_for_segmentation
    from .io import _check_masks
    from .spacr_cellpose import parse_cellpose4_output

    if object_type == 'nucleus':
        selected_channels = [settings.get('nucleus_channel')]
    elif object_type == 'cell':
        selected_channels = [settings.get('cell_channel'), settings.get('nucleus_channel')]
    elif object_type == 'pathogen':
        selected_channels = [settings.get('pathogen_channel')]
    elif object_type == 'organelle':
        selected_channels = [settings.get('organelle_channel')]
    else:
        raise ValueError(f"Unsupported object_type: {object_type}")

    selected_channels = [ch for ch in selected_channels if ch is not None]

    if len(selected_channels) == 0:
        raise ValueError(f"No valid channels defined for object_type '{object_type}'.")

    if batch.ndim == 4:
        max_ch = batch.shape[3]
        selected_channels = [ch for ch in selected_channels if ch < max_ch]

        if len(selected_channels) == 0:
            raise ValueError(
                f"Selected channels for object_type '{object_type}' are out of bounds for batch with {max_ch} channels."
            )

        cp_batch = batch[:, :, :, selected_channels].astype(batch.dtype)

    elif batch.ndim == 3:
        cp_batch = batch[:, :, :, np.newaxis].astype(batch.dtype)

    else:
        raise ValueError(f"Expected batch with ndim 3 or 4, got ndim={batch.ndim}")

    if not settings.get('plot', False):
        cp_batch, batch_filenames = _check_masks(cp_batch, batch_filenames, output_folder)
    if cp_batch.size == 0:
        return None

    cp_batch = prepare_batch_for_segmentation(cp_batch)
    batch_list = [cp_batch[j] for j in range(cp_batch.shape[0])]

    output = model.eval(
        x=batch_list,
        batch_size=len(batch_list),
        normalize=False,
        channel_axis=-1,
        diameter=None,
        flow_threshold=settings[f'{object_type}_FT'],
        cellprob_threshold=settings[f'{object_type}_CP_prob'],
        resample=settings.get(f'{object_type}_resample', True)
    )

    masks, flows, _, _, _ = parse_cellpose4_output(output)
    return masks


# ====================================================================== #
#  Stardist segmentation (GPU — not parallelised)
# ====================================================================== #

def _segment_stardist(img_batch, model, settings):
    """
    Run Stardist on a batch of single-channel images.
    Best for dense, convex, spot-like organelles (lipid droplets, peroxisomes).
    """
    prob_thresh = settings.get('organelle_stardist_prob', 0.5)
    nms_thresh = settings.get('organelle_stardist_nms', 0.3)

    masks = []
    for idx in range(img_batch.shape[0]):
        img = img_batch[idx]
        pmin, pmax = np.percentile(img, (1, 99.8))
        if pmax - pmin > 0:
            img_norm = np.clip((img - pmin) / (pmax - pmin), 0, 1).astype(np.float32)
        else:
            img_norm = np.zeros_like(img, dtype=np.float32)

        labels, _ = model.predict_instances(
            img_norm,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )
        masks.append(labels)
    return masks


# ====================================================================== #
#  U-Net semantic segmentation (GPU — not parallelised)
# ====================================================================== #

def _segment_unet(img_batch, model, settings):
    """
    Run a user-provided U-Net for semantic segmentation of networks.
    Expects a model that takes (B, 1, H, W) and outputs (B, 1, H, W) logits.
    """
    device = next(model.parameters()).device
    threshold = settings.get('organelle_unet_threshold', 0.5)
    do_skeleton = settings.get('organelle_skeletonize', False)

    masks = []
    with torch.no_grad():
        for idx in range(img_batch.shape[0]):
            img = img_batch[idx]
            mean, std = img.mean(), img.std()
            if std > 0:
                img_norm = (img - mean) / std
            else:
                img_norm = np.zeros_like(img)

            tensor = torch.from_numpy(img_norm[None, None]).float().to(device)
            pred = model(tensor)

            if pred.shape[1] > 1:
                pred = pred[:, 0:1, :, :]

            pred = pred.sigmoid().cpu().numpy()[0, 0]
            binary = pred > threshold

            binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

            if do_skeleton:
                skeleton = skeletonize(binary)
                skeleton = binary_dilation(skeleton, disk(1))
                masks.append(sk_label(skeleton))
            else:
                masks.append(sk_label(binary))

    return masks


# ====================================================================== #
#  Classical segmentation — parallel dispatcher
# ====================================================================== #

def _segment_classical_parallel(img_batch, classical_settings, n_jobs=1):
    """
    Segment a batch of images using classical methods, optionally in parallel.

    Parameters
    ----------
    img_batch : np.ndarray
        Shape (N, H, W) float32 single-channel images.
    classical_settings : dict
        Pickle-safe subset of settings for classical segmentation.
    n_jobs : int
        Number of worker processes. 1 = sequential (no multiprocessing overhead).

    Returns
    -------
    list of np.ndarray
        List of 2-D integer label arrays, one per image.
    """
    n_images = img_batch.shape[0]

    if n_jobs == 1 or n_images == 1:
        return [_segment_single_image(img_batch[idx], classical_settings)
                for idx in range(n_images)]

    effective_jobs = min(n_jobs, n_images, cpu_count())

    worker_fn = partial(_segment_single_image, settings=classical_settings)
    image_list = [img_batch[idx] for idx in range(n_images)]

    with Pool(processes=effective_jobs) as pool:
        masks = pool.map(worker_fn, image_list)

    return masks


def _segment_single_image(img, settings):
    """
    Segment a single 2-D image. Top-level function for multiprocessing.
    """
    morphology = settings['organelle_morphology']
    method = settings['organelle_method']

    if morphology == 'spots':
        return _segment_spots(img, method, settings)
    elif morphology == 'network':
        return _segment_network(img, method, settings)
    elif morphology == 'irregular':
        return _segment_irregular(img, method, settings)
    elif morphology == 'ring':
        return _segment_ring(img, method, settings)
    else:
        raise ValueError(f"Unknown morphology: {morphology}")


# ====================================================================== #
#  SPOTS segmentation
# ====================================================================== #

def _segment_spots(img, method, settings):
    """
    Segment punctate / spot-like organelles.

    Strategies
    ----------
    'otsu'     : top-hat -> Otsu -> watershed
    'adaptive' : top-hat -> adaptive threshold -> watershed
    'log'      : Laplacian-of-Gaussian blob detection -> marker-based watershed
    'dog'      : Difference-of-Gaussians blob detection -> marker-based watershed
    """
    tophat_radius = settings['organelle_tophat_radius']
    use_watershed = settings['organelle_watershed_spots']

    if method == 'log':
        return _spots_log(img, settings, use_watershed)
    elif method == 'dog':
        return _spots_dog(img, settings, use_watershed)

    # --- Pre-filter: white top-hat enhances bright spots on dark bg ---
    filtered = white_tophat(img, disk(tophat_radius))

    # --- Threshold ---
    if method == 'otsu':
        thresh_val = threshold_otsu(filtered)
        binary = filtered > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(filtered, block_size=block, offset=offset)
        binary = filtered > local_thresh
    else:
        raise ValueError(f"Unsupported spot method: {method}")

    # --- Morphological cleanup ---
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    # --- Watershed to split touching spots ---
    if use_watershed:
        labeled = _watershed_split(binary, filtered)
    else:
        labeled = sk_label(binary)

    return labeled


def _spots_log(img, settings, use_watershed):
    """LoG blob detection -> marker-seeded watershed."""
    min_s = settings['organelle_log_min_sigma']
    max_s = settings['organelle_log_max_sigma']
    num_s = settings['organelle_log_num_sigma']
    thresh = settings['organelle_log_threshold']

    img_norm = _normalize_01(img)

    blobs = blob_log(img_norm, min_sigma=min_s, max_sigma=max_s,
                     num_sigma=num_s, threshold=thresh)

    if len(blobs) == 0:
        return np.zeros(img.shape, dtype=np.int32)

    return _blobs_to_labels(blobs, img_norm, use_watershed)


def _spots_dog(img, settings, use_watershed):
    """
    Difference-of-Gaussians blob detection -> marker-seeded watershed.
    Faster than LoG, nearly equivalent accuracy for fluorescence microscopy.
    """
    sigma_low = settings.get('organelle_dog_sigma_low', 1.0)
    sigma_high = settings.get('organelle_dog_sigma_high', 3.0)
    thresh = settings['organelle_log_threshold']

    img_norm = _normalize_01(img)

    blobs = blob_dog(img_norm, min_sigma=sigma_low, max_sigma=sigma_high,
                     threshold=thresh)

    if len(blobs) == 0:
        return np.zeros(img.shape, dtype=np.int32)

    return _blobs_to_labels(blobs, img_norm, use_watershed)


def _blobs_to_labels(blobs, img_norm, use_watershed):
    """
    Convert blob coordinates (y, x, sigma) to a label image.
    Shared by LoG and DoG methods.
    """
    shape = img_norm.shape
    markers = np.zeros(shape, dtype=np.int32)
    for i, (y, x, sigma) in enumerate(blobs, start=1):
        y, x = int(round(y)), int(round(x))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            markers[y, x] = i

    if not use_watershed:
        labeled = np.zeros(shape, dtype=np.int32)
        for i, (y, x, sigma) in enumerate(blobs, start=1):
            rr, cc = _circle_coords(int(round(y)), int(round(x)),
                                    max(int(round(sigma * np.sqrt(2))), 1),
                                    shape)
            labeled[rr, cc] = i
        return labeled

    smooth = gaussian(img_norm, sigma=1)
    labeled = watershed(-smooth, markers, mask=(smooth > np.percentile(smooth, 20)))
    return labeled


def _circle_coords(cy, cx, radius, shape):
    """Return (row, col) arrays for a filled circle clipped to shape."""
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    circle = yy ** 2 + xx ** 2 <= radius ** 2
    rows = np.clip(cy + np.where(circle)[0] - radius, 0, shape[0] - 1)
    cols = np.clip(cx + np.where(circle)[1] - radius, 0, shape[1] - 1)
    return rows, cols


# ====================================================================== #
#  NETWORK segmentation
# ====================================================================== #

def _segment_network(img, method, settings):
    """
    Segment filamentous / reticular organelles.

    Strategies
    ----------
    'otsu'        : Gaussian smooth -> Otsu -> morphological cleanup
    'adaptive'    : Gaussian smooth -> adaptive threshold -> cleanup
    'ridge'       : Ridge filter (Frangi / Sato / Meijering) -> threshold -> label
    'hysteresis'  : Gaussian smooth -> dual-threshold hysteresis -> cleanup
    """
    if method == 'ridge':
        return _network_ridge(img, settings)
    elif method == 'hysteresis':
        return _network_hysteresis(img, settings)

    smooth = gaussian(img, sigma=1)

    if method == 'otsu':
        thresh_val = threshold_otsu(smooth)
        binary = smooth > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(smooth, block_size=block, offset=offset)
        binary = smooth > local_thresh
    else:
        raise ValueError(f"Unsupported network method: {method}")

    morph_r = max(settings['organelle_morph_radius'] // 2, 1)
    binary = binary_closing(binary, disk(morph_r))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)


def _network_ridge(img, settings):
    """Apply a ridge (tubeness) filter then threshold."""
    sigmas = settings['organelle_ridge_sigmas']
    filter_name = settings['organelle_ridge_filter']
    thresh_method = settings['organelle_network_threshold']

    img_norm = _normalize_01(img)

    ridge_filters = {
        'frangi': frangi,
        'sato': sato,
        'meijering': meijering,
    }
    if filter_name not in ridge_filters:
        raise ValueError(
            f"organelle_ridge_filter must be one of {list(ridge_filters.keys())}, "
            f"got '{filter_name}'"
        )

    enhanced = ridge_filters[filter_name](img_norm, sigmas=sigmas, black_ridges=False)

    if thresh_method == 'otsu':
        t = threshold_otsu(enhanced)
        binary = enhanced > t
    elif thresh_method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_t = threshold_local(enhanced, block_size=block, offset=offset)
        binary = enhanced > local_t
    else:
        t = threshold_otsu(enhanced)
        binary = enhanced > t

    binary = binary_closing(binary, disk(1))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)


def _network_hysteresis(img, settings):
    """
    Dual-threshold hysteresis segmentation for networks.

    Uses a high threshold to seed confident regions and a low threshold
    to extend into dimmer connected filaments. Much better than single-
    threshold approaches for variable-contrast mitochondria and microtubules.

    Settings
    --------
    organelle_hysteresis_low : float
        Low threshold. If <1.0, interpreted as a percentile of the image.
    organelle_hysteresis_high : float
        High threshold. If <1.0, interpreted as a percentile of the image.
    """
    low = settings['organelle_hysteresis_low']
    high = settings['organelle_hysteresis_high']

    smooth = gaussian(img, sigma=1)

    # Interpret values <1.0 as percentiles
    if low < 1.0:
        low = np.percentile(smooth, low * 100)
    if high < 1.0:
        high = np.percentile(smooth, high * 100)

    binary = apply_hysteresis_threshold(smooth, low, high)

    morph_r = max(settings['organelle_morph_radius'] // 2, 1)
    binary = binary_closing(binary, disk(morph_r))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)


# ====================================================================== #
#  IRREGULAR segmentation
# ====================================================================== #

def _segment_irregular(img, method, settings):
    """
    Segment irregularly shaped organelles (Golgi, ER cisternae, lysosomes).

    Strategies
    ----------
    'otsu'     : Gaussian smooth -> Otsu -> morph open/close -> fill holes -> label
    'adaptive' : Gaussian smooth -> adaptive threshold -> morph cleanup -> label
    """
    morph_r = settings['organelle_morph_radius']
    fill_area = settings['organelle_fill_holes']

    smooth = gaussian(img, sigma=max(morph_r / 2, 1))

    if method == 'otsu':
        thresh_val = threshold_otsu(smooth)
        binary = smooth > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(smooth, block_size=block, offset=offset)
        binary = smooth > local_thresh
    else:
        raise ValueError(f"Unsupported irregular method: {method}")

    selem = disk(morph_r)
    binary = binary_closing(binary, selem)
    binary = binary_opening(binary, selem)

    if fill_area > 0:
        binary = remove_small_holes(binary, area_threshold=fill_area)

    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    labeled = _watershed_split(binary, smooth)
    return labeled


# ====================================================================== #
#  RING segmentation
# ====================================================================== #

def _segment_ring(img, method, settings):
    """
    Segment hollow / ring-shaped organelles (endosomes, autophagosomes,
    late endosomes/lysosomes).

    Strategy
    --------
    1. Enhance ring edges using Difference of Gaussians.
    2. Threshold edges using the specified method.
    3. Fill enclosed rings to produce solid instance masks.
    4. Filter out objects that lack ring morphology (no bright-dim
       boundary-interior contrast).

    Settings
    --------
    organelle_ring_sigma_inner : float
        Inner sigma for DoG edge enhancement. Default: 1.0.
    organelle_ring_sigma_outer : float
        Outer sigma for DoG edge enhancement. Default: 3.0.
    organelle_ring_min_prominence : float
        Minimum edge-vs-interior intensity contrast (normalised).
        Objects below this are removed. Default: 0.1.
    organelle_ring_fill_method : str
        'flood' (fill enclosed holes) or 'convex' (convex hull per component).
        Default: 'flood'.
    """
    sigma_inner = settings.get('organelle_ring_sigma_inner', 1.0)
    sigma_outer = settings.get('organelle_ring_sigma_outer', 3.0)
    min_prominence = settings.get('organelle_ring_min_prominence', 0.1)
    fill_method = settings.get('organelle_ring_fill_method', 'flood')

    # Step 1: Enhance ring structures using DoG (edge enhancement)
    img_norm = _normalize_01(img)
    enhanced = np.abs(difference_of_gaussians(img_norm, sigma_inner, sigma_outer))

    # Step 2: Threshold the enhanced image
    if method == 'otsu':
        thresh_val = threshold_otsu(enhanced)
        binary_edges = enhanced > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(enhanced, block_size=block, offset=offset)
        binary_edges = enhanced > local_thresh
    elif method == 'log':
        blobs = blob_log(img_norm,
                         min_sigma=settings['organelle_log_min_sigma'],
                         max_sigma=settings['organelle_log_max_sigma'],
                         num_sigma=settings['organelle_log_num_sigma'],
                         threshold=settings['organelle_log_threshold'])
        if len(blobs) == 0:
            return np.zeros(img.shape, dtype=np.int32)
        thresh_val = threshold_otsu(enhanced)
        binary_edges = enhanced > thresh_val
    elif method == 'dog':
        thresh_val = threshold_otsu(enhanced)
        binary_edges = enhanced > thresh_val
    else:
        raise ValueError(f"Unsupported ring method: {method}")

    # Cleanup edges
    binary_edges = binary_closing(binary_edges, disk(1))
    binary_edges = remove_small_objects(binary_edges, min_size=max(settings['organelle_min_size'] // 4, 3))

    # Step 3: Fill rings to get solid objects
    if fill_method == 'flood':
        filled = _fill_rings_flood(binary_edges)
    elif fill_method == 'convex':
        filled = _fill_rings_convex(binary_edges)
    else:
        filled = _fill_rings_flood(binary_edges)

    # Step 4: Remove objects that lack ring morphology
    labeled = sk_label(filled)
    labeled = _filter_non_rings(labeled, binary_edges, img_norm, min_prominence)

    return labeled


def _fill_rings_flood(binary_edges):
    """
    Fill ring interiors using flood-fill logic.
    Inverts the edge mask, labels connected components, and identifies
    interior regions (not touching the image border) as filled objects.
    """
    inverted = ~binary_edges
    labeled_bg = sk_label(inverted)

    border_labels = set()
    border_labels.update(labeled_bg[0, :].ravel())
    border_labels.update(labeled_bg[-1, :].ravel())
    border_labels.update(labeled_bg[:, 0].ravel())
    border_labels.update(labeled_bg[:, -1].ravel())

    filled = binary_edges.copy()
    for region in regionprops(labeled_bg):
        if region.label not in border_labels:
            filled[labeled_bg == region.label] = True

    return filled


def _fill_rings_convex(binary_edges):
    """
    Fill rings using the convex hull of each connected edge component.
    """
    from skimage.morphology import convex_hull_image

    labeled_edges = sk_label(binary_edges)
    filled = np.zeros_like(binary_edges)

    for region in regionprops(labeled_edges):
        minr, minc, maxr, maxc = region.bbox
        component = labeled_edges[minr:maxr, minc:maxc] == region.label
        hull = convex_hull_image(component)
        filled[minr:maxr, minc:maxc] |= hull

    return filled


def _filter_non_rings(labeled, binary_edges, img_norm, min_prominence):
    """
    Remove objects that lack ring morphology.

    A ring has a bright boundary with a dimmer interior. We measure this
    as the absolute difference between the mean edge intensity and the mean
    interior intensity, normalised by the object's mean intensity.
    """
    props = regionprops(labeled, intensity_image=img_norm)
    output = labeled.copy()

    for prop in props:
        mask = labeled == prop.label
        edge_mask = mask & binary_edges
        interior_mask = mask & ~binary_edges

        if np.sum(edge_mask) == 0 or np.sum(interior_mask) == 0:
            edge_ratio = np.sum(edge_mask) / max(np.sum(mask), 1)
            if edge_ratio < 0.3:
                output[mask] = 0
            continue

        mean_edge = img_norm[edge_mask].mean()
        mean_interior = img_norm[interior_mask].mean()
        object_mean = img_norm[mask].mean()

        if object_mean > 0:
            prominence = abs(mean_edge - mean_interior) / object_mean
        else:
            prominence = 0

        if prominence < min_prominence:
            output[mask] = 0

    return sk_label(output > 0)


# ====================================================================== #
#  Shared helpers
# ====================================================================== #

def _normalize_01(img):
    """Percentile-based normalisation to [0, 1]."""
    img_norm = img.astype(np.float64)
    pmin, pmax = np.percentile(img_norm, (1, 99))
    if pmax - pmin > 0:
        img_norm = np.clip((img_norm - pmin) / (pmax - pmin), 0, 1)
    else:
        img_norm = np.zeros_like(img_norm)
    return img_norm


def _watershed_split(binary, intensity):
    """
    Marker-controlled watershed to separate touching objects in a binary mask.
    Uses distance transform peaks as markers.
    """
    distance = distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=5, labels=binary)
    if len(coords) == 0:
        return sk_label(binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labeled = watershed(-distance, markers, mask=binary)
    return labeled


def _postprocess_masks(masks, min_size=10, max_size=None, remove_border=False):
    """
    Apply size filtering and optional border removal to a list of label masks.

    Returns
    -------
    list of np.ndarray
        Cleaned label arrays.
    """
    processed = []
    for mask in masks:
        mask = mask.copy()

        if remove_border:
            border_labels = set()
            border_labels.update(mask[0, :].ravel())
            border_labels.update(mask[-1, :].ravel())
            border_labels.update(mask[:, 0].ravel())
            border_labels.update(mask[:, -1].ravel())
            border_labels.discard(0)
            for lbl in border_labels:
                mask[mask == lbl] = 0

        if min_size > 0 or max_size is not None:
            props = regionprops(mask)
            for prop in props:
                if prop.area < min_size:
                    mask[mask == prop.label] = 0
                elif max_size is not None and prop.area > max_size:
                    mask[mask == prop.label] = 0

        mask = sk_label(mask > 0)
        processed.append(mask)

    return processed

