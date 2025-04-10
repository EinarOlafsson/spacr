import os, gc, torch, time, random, cv2
import numpy as np
import pandas as pd
from cellpose import models as cp_models
from IPython.display import display
from multiprocessing import Pool
from skimage.transform import resize as resizescikit
from scipy.ndimage import binary_fill_holes

def identify_masks_finetune(settings):
    
    from .plot import print_mask_and_flows
    from .utils import resize_images_and_labels, print_progress, save_settings, fill_holes_in_mask
    from .io import _load_normalized_images_and_labels, _load_images_and_labels
    from .settings import get_identify_masks_finetune_default_settings

    settings = get_identify_masks_finetune_default_settings(settings)
    save_settings(settings, name='generate_cellpose_masks', show=True)
    dst = os.path.join(settings['src'], 'masks')
    os.makedirs(dst, exist_ok=True)

    if not settings['custom_model'] is None:
        if not os.path.exists(settings['custom_model']):
            print(f"Custom model not found: {settings['custom_model']}")
            return 

    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if settings['custom_model'] == None:
        model = cp_models.CellposeModel(gpu=True, model_type=settings['model_name'], device=device)
        print(f"Loaded model: {settings['model_name']}")
    else:
        model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type=None, pretrained_model=settings['custom_model'], diam_mean=settings['diameter'], device=device)
        print("Pretrained Model Loaded:", model.pretrained_model)

    chans = [2, 1] if settings['model_name'] == 'cyto2' else [0,0] if settings['model_name'] == 'nucleus' else [1,0] if settings['model_name'] == 'cyto' else [2, 0]
    
    if settings['grayscale']:
        chans=[0, 0]
    
    print(f"Using channels: {chans} for model of type {settings['model_name']}")
    
    if settings['verbose'] == True:
        print(f"Cellpose settings: Model: {settings['model_name']}, channels: {settings['channels']}, cellpose_chans: {chans}, diameter:{settings['diameter']}, flow_threshold:{settings['flow_threshold']}, cellprob_threshold:{settings['CP_prob']}")
        
    image_files = [os.path.join(settings['src'], f) for f in os.listdir(settings['src']) if f.endswith('.tif')]
    mask_files = set(os.listdir(os.path.join(settings['src'], 'masks')))
    all_image_files = [f for f in image_files if os.path.basename(f) not in mask_files]
    random.shuffle(all_image_files)

    print(f"Found {len(image_files)} Images with {len(mask_files)} masks. Generating masks for {len(all_image_files)} images")

    if len(all_image_files) == 0:
        print(f"Either no images were found in {settings['src']} or all images have masks in {settings['dst']}")
        return

    
    time_ls = []
    for i in range(0, len(all_image_files), settings['batch_size']):
        gc.collect()
        image_files = all_image_files[i:i+settings['batch_size']]
        
        if settings['normalize']:
            images, _, image_names, _, orig_dims = _load_normalized_images_and_labels(image_files=image_files,
                                                                                      label_files=None,
                                                                                      channels=settings['channels'],
                                                                                      percentiles=settings['percentiles'],
                                                                                      invert=settings['invert'],
                                                                                      visualize=settings['verbose'],
                                                                                      remove_background=settings['remove_background'],
                                                                                      background=settings['background'],
                                                                                      Signal_to_noise=settings['Signal_to_noise'],
                                                                                      target_height=settings['target_height'],
                                                                                      target_width=settings['target_width'])
            
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        else:
            images, _, image_names, _ = _load_images_and_labels(image_files=image_files, label_files=None, invert=settings['invert']) 
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[0], image.shape[1]) for image in images]
            if settings['resize']:
                images, _ = resize_images_and_labels(images, None, settings['target_height'], settings['target_width'], True)

        for file_index, stack in enumerate(images):
            start = time.time()
            output = model.eval(x=stack,
                         normalize=False,
                         channels=chans,
                         channel_axis=3,
                         diameter=settings['diameter'],
                         flow_threshold=settings['flow_threshold'],
                         cellprob_threshold=settings['CP_prob'],
                         rescale=settings['rescale'],
                         resample=settings['resample'],
                         progress=True)

            if len(output) == 4:
                mask, flows, _, _ = output
            elif len(output) == 3:
                mask, flows, _ = output
            else:
                raise ValueError("Unexpected number of return values from model.eval()")
            
            if settings['fill_in']:
                mask = fill_holes_in_mask(mask).astype(mask.dtype)

            if settings['resize']:
                dims = orig_dims[file_index]
                mask = resizescikit(mask, dims, order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

            stop = time.time()
            duration = (stop - start)
            time_ls.append(duration)
            files_processed = len(images)
            files_to_process = file_index+1            
            print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="generate cellpose masks")
            
            if settings['verbose']:
                if settings['resize']:
                    stack = resizescikit(stack, dims, preserve_range=True, anti_aliasing=False).astype(stack.dtype)
                print_mask_and_flows(stack, mask, flows)
            if settings['save']:
                os.makedirs(dst, exist_ok=True)
                output_filename = os.path.join(dst, image_names[file_index])
                cv2.imwrite(output_filename, mask)
        del images, output, mask, flows
        gc.collect()
    return

def generate_masks_from_imgs(src, model, model_name, batch_size, diameter, cellprob_threshold, flow_threshold, grayscale, save, normalize, channels, percentiles, invert, plot, resize, target_height, target_width, remove_background, background, Signal_to_noise, verbose):
    
    from .io import _load_images_and_labels, _load_normalized_images_and_labels
    from .utils import resize_images_and_labels, resizescikit, print_progress
    from .plot import print_mask_and_flows

    dst = os.path.join(src, model_name)
    os.makedirs(dst, exist_ok=True)

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [1,0] if model_name == 'cyto' else [2, 0]

    if grayscale:
        chans=[0, 0]
    
    all_image_files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.tif')]
    random.shuffle(all_image_files)
        
    if verbose == True:
        print(f'Cellpose settings: Model: {model_name}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
    
    time_ls = []
    for i in range(0, len(all_image_files), batch_size):
        image_files = all_image_files[i:i+batch_size]

        if normalize:
            images, _, image_names, _, orig_dims = _load_normalized_images_and_labels(image_files, None, channels, percentiles, invert, plot, remove_background, background, Signal_to_noise, target_height, target_width)
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[0], image.shape[1]) for image in images]
        else:
            images, _, image_names, _ = _load_images_and_labels(image_files, None, invert) 
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[0], image.shape[1]) for image in images]
        if resize:
            images, _ = resize_images_and_labels(images, None, target_height, target_width, True)

        for file_index, stack in enumerate(images):
            start = time.time()
            output = model.eval(x=stack,
                         normalize=False,
                         channels=chans,
                         channel_axis=3,
                         diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold,
                         rescale=False,
                         resample=False,
                         progress=False)

            if len(output) == 4:
                mask, flows, _, _ = output
            elif len(output) == 3:
                mask, flows, _ = output
            else:
                raise ValueError("Unexpected number of return values from model.eval()")

            if resize:
                dims = orig_dims[file_index]
                mask = resizescikit(mask, dims, order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

            stop = time.time()
            duration = (stop - start)
            time_ls.append(duration)
            files_processed = file_index+1
            files_to_process = len(images)

            print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Generating masks")

            if plot:
                if resize:
                    stack = resizescikit(stack, dims, preserve_range=True, anti_aliasing=False).astype(stack.dtype)
                print_mask_and_flows(stack, mask, flows)
            if save:
                output_filename = os.path.join(dst, image_names[file_index])
                cv2.imwrite(output_filename, mask)

def check_cellpose_models(settings):

    from .settings import get_check_cellpose_models_default_settings
    
    settings = get_check_cellpose_models_default_settings(settings)
    src = settings['src']

    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
    settings_df['setting_value'] = settings_df['setting_value'].apply(str)
    display(settings_df)

    cellpose_models = ['cyto', 'nuclei', 'cyto2', 'cyto3']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for model_name in cellpose_models:

        model = cp_models.CellposeModel(gpu=True, model_type=model_name, device=device)
        print(f'Using {model_name}')
        generate_masks_from_imgs(src, model, model_name, settings['batch_size'], settings['diameter'], settings['CP_prob'], settings['flow_threshold'], settings['grayscale'], settings['save'], settings['normalize'], settings['channels'], settings['percentiles'], settings['invert'], settings['plot'], settings['resize'], settings['target_height'], settings['target_width'], settings['remove_background'], settings['background'], settings['Signal_to_noise'], settings['verbose'])

    return

def save_results_and_figure(src, fig, results):

    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    results_dir = os.path.join(src, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir,f'results.csv')
    fig_path = os.path.join(results_dir, f'model_comparison_plot.pdf')
    results.to_csv(results_path, index=False)
    fig.savefig(fig_path, format='pdf')
    print(f'Saved figure to {fig_path} and results to {results_path}')

def compare_mask(args):
    src, filename, dirs, conditions = args
    paths = [os.path.join(d, filename) for d in dirs]

    if not all(os.path.exists(path) for path in paths):
        return None

    from .io import _read_mask
    from .utils import boundary_f1_score, compute_segmentation_ap, jaccard_index

    masks = [_read_mask(path) for path in paths]
    file_results = {'filename': filename}

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            mask_i, mask_j = masks[i], masks[j]
            f1_score = boundary_f1_score(mask_i, mask_j)
            jac_index = jaccard_index(mask_i, mask_j)
            ap_score = compute_segmentation_ap(mask_i, mask_j)

            file_results.update({
                f'jaccard_{conditions[i]}_{conditions[j]}': jac_index,
                f'boundary_f1_{conditions[i]}_{conditions[j]}': f1_score,
                f'ap_{conditions[i]}_{conditions[j]}': ap_score
            })
    
    return file_results

def compare_cellpose_masks(src, verbose=False, processes=None, save=True):
    from .plot import visualize_cellpose_masks, plot_comparison_results
    from .io import _read_mask

    dirs = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d != 'results']
    dirs.sort()
    conditions = [os.path.basename(d) for d in dirs]

    # Get common files in all directories
    common_files = set(os.listdir(dirs[0]))
    for d in dirs[1:]:
        common_files.intersection_update(os.listdir(d))
    common_files = list(common_files)

    # Create a pool of n_jobs
    with Pool(processes=processes) as pool:
        args = [(src, filename, dirs, conditions) for filename in common_files]
        results = pool.map(compare_mask, args)

    # Filter out None results (from skipped files)
    results = [res for res in results if res is not None]
    print(results)
    if verbose:
        for result in results:
            filename = result['filename']
            masks = [_read_mask(os.path.join(d, filename)) for d in dirs]
            visualize_cellpose_masks(masks, titles=conditions, filename=filename, save=save, src=src)

    fig = plot_comparison_results(results)
    save_results_and_figure(src, fig, results)
    return
