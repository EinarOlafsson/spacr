"""Cellpose model evaluation and mask-generation workflows."""

import os, torch, time, random

from . import _gc as gc
import numpy as np
import pandas as pd
from cellpose import models as cp_models
try:
    from IPython.display import display
except Exception:
    def display(*args, **kwargs):
        """Discard display payloads when IPython's helper is unavailable."""
        pass
from skimage.transform import resize as resizescikit

from .tiff_io import write_tiff

def cellpose_rescale(value):
    """The ``rescale=`` Cellpose should actually receive. Falsy becomes None.

    ``rescale`` was DEPRECATED-AND-IGNORED in Cellpose 4.0, so spaCR passing
    ``False`` cost nothing and nobody noticed the type was wrong. Cellpose
    4.2 reads it again::

        niter_scale = 1 if rescale is None or not resample else rescale
        niter = int(200/niter_scale) if niter is None or niter == 0 else niter

    With ``rescale=False`` and ``resample=True`` -- spaCR's shipped rescale
    default, and a resample a user is entirely likely to turn on --
    ``niter_scale`` becomes ``False`` and the second line reads
    ``int(200/False)``: ZeroDivisionError, raised from inside Cellpose, on a
    settings combination both GUIs offer.

    ``None`` is Cellpose's own spelling of "not set" and takes the
    ``niter_scale = 1`` branch, which is what ``False`` was always meant to
    mean here. ``0`` goes the same way, for the same reason.

    :param value: whatever the settings carry for ``rescale``.
    :returns: ``None`` for a falsy value, otherwise the value unchanged.
    """
    return None if not value else value


def cellpose_channel_axis(stack):
    """Return the ``channel_axis`` Cellpose 4 accepts for one loaded image.

    ``cellpose.transforms.convert_image`` — which ``CellposeModel.eval``
    calls with whatever ``channel_axis`` it was handed — indexes
    ``x.shape[channel_axis]`` and rejects a non-``None`` axis outright for a
    2-D input. The two shapes spaCR's loaders produce are therefore both
    illegal under the old hard-coded ``channel_axis=3``:

    * ``(H, W, C)`` -> ``IndexError: tuple index out of range`` (there is no
      axis 3 on a 3-D array; the channel axis is 2, i.e. ``-1``).
    * ``(H, W)``    -> ``ValueError: 2D image provided, but channel_axis is
      not None``.

    ``object.py`` already passes ``channel_axis=-1`` because it only ever
    hands Cellpose channels-last stacks. The functions here also serve
    greyscale images (``_load_*_images_and_labels`` squeezes a single-channel
    load down to 2-D), so the axis has to be chosen per image.

    :param stack: One image as loaded by :mod:`spacr.io`, either ``(H, W)``
        or channels-last ``(H, W, C)``.
    :returns: ``-1`` for a channels-last stack, ``None`` for a 2-D image.
    """
    return -1 if np.asarray(stack).ndim >= 3 else None

def parse_cellpose4_output(output):
    """Normalize the return value of ``CellposeModel.eval`` into per-image flow lists.

    Accepts both the batched format (4 stacked arrays) and the per-image list
    format so downstream code can iterate uniformly.

    :param output: Raw ``(masks, flows, ...)`` tuple returned by Cellpose.
    :returns: Tuple ``(masks, flows0, flows1, flows2, flows3)`` with per-image entries.
    :raises ValueError: When the flows structure does not match a known layout.
    """

    masks = output[0]
    flows = output[1]

    if not isinstance(flows, (list, tuple)):
        raise ValueError(f"Unrecognized Cellpose flows type: {type(flows)}")

    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        items = list(flows)
        first, second, third, fourth = (
            items[i] if i < len(items) else None for i in range(4))
        return masks, [first], [second], [third], [fourth]

    try:
        num_images = len(masks)
    except TypeError:
        raise ValueError(f"Cannot determine number of images in masks (type={type(masks)})")

    if len(flows) == 4 and all(isinstance(f, np.ndarray) for f in flows):
        flow0_array, flow1_array, flow2_array, flow3_array = flows

        flows0 = [flow0_array[i] for i in range(num_images)]
        flows1 = [flow1_array[:, i] for i in range(num_images)]
        flows2 = [flow2_array[i] for i in range(num_images)]
        flows3 = [flow3_array[i] for i in range(num_images)]

        return masks, flows0, flows1, flows2, flows3

    elif len(flows) == num_images:
        flows0, flows1, flows2, flows3 = [], [], [], []

        for item in flows:
            if isinstance(item, (list, tuple)):
                n = len(item)
                f0 = item[0] if n > 0 else None
                f1 = item[1] if n > 1 else None
                f2 = item[2] if n > 2 else None
                f3 = item[3] if n > 3 else None
            elif isinstance(item, np.ndarray):
                f0, f1, f2, f3 = item, None, None, None
            else:
                f0 = f1 = f2 = f3 = None

            flows0.append(f0)
            flows1.append(f1)
            flows2.append(f2)
            flows3.append(f3)

        return masks, flows0, flows1, flows2, flows3

    raise ValueError(f"Unrecognized Cellpose flows format: type={type(flows)}, len={len(flows) if hasattr(flows,'__len__') else 'unknown'}")

def identify_masks_finetune(settings):
    """Generate Cellpose masks for a directory of images using a stock or custom model.

    Iterates in batches, optionally normalizing and resizing the inputs, writes
    the resulting masks under ``<src>/masks``, and prints per-image progress.

    :param settings: Settings dict; canonicalized via
        :func:`spacr.settings.get_identify_masks_finetune_default_settings`.
        Must contain ``src``, ``model_name`` (or ``custom_model``), and standard
        Cellpose parameters (``diameter``, ``flow_threshold``, ``CP_prob``, ...).
    :returns: None.
    """
    from .plot import print_mask_and_flows
    from .utils import (resize_images_and_labels, print_progress, save_settings,
                        fill_holes_in_mask, _resolve_cellpose_pretrained)
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

    from .accelerator import cellpose_gpu, cellpose_kwargs, describe

    if not cellpose_gpu():
        print('No GPU available to spaCR, using CPU')
    else:
        print(f'Segmenting on {describe()}')

    if settings['custom_model'] is None:
        pretrained = _resolve_cellpose_pretrained(settings['model_name'])
    else:
        pretrained = settings['custom_model']

    model = cp_models.CellposeModel(pretrained_model=pretrained,
                                    **cellpose_kwargs())
    print(f"Loaded model: {getattr(model, 'pretrained_model', pretrained)}")

    if settings['grayscale']:
        print("grayscale=True has no effect under Cellpose 4: the channel "
              "pair (eval channels=) is deprecated and ignored.")

    if settings['verbose'] == True:
        print(f"Cellpose settings: Model: {pretrained}, channels: {settings['channels']}, diameter:{settings['diameter']}, flow_threshold:{settings['flow_threshold']}, cellprob_threshold:{settings['CP_prob']}")

    image_files = [os.path.join(settings['src'], f) for f in os.listdir(settings['src']) if f.endswith('.tif')]
    mask_files = set(os.listdir(os.path.join(settings['src'], 'masks')))
    all_image_files = [f for f in image_files if os.path.basename(f) not in mask_files]
    random.shuffle(all_image_files)

    print(f"Found {len(image_files)} Images with {len(mask_files)} masks. Generating masks for {len(all_image_files)} images")

    if len(all_image_files) == 0:
        print(f"Either no images were found in {settings['src']} or all images have masks in {dst}")
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
                         channel_axis=cellpose_channel_axis(stack),
                         diameter=settings['diameter'],
                         flow_threshold=settings['flow_threshold'],
                         cellprob_threshold=settings['CP_prob'],
                         rescale=cellpose_rescale(settings['rescale']),
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
                write_tiff(output_filename, mask)
        del images, output, mask, flows
        gc.collect()
    return

def generate_masks_from_imgs(src, model, model_name, batch_size, diameter, cellprob_threshold, flow_threshold, grayscale, save, normalize, channels, percentiles, invert, plot, resize, target_height, target_width, remove_background, background, Signal_to_noise, verbose):
    """Run a Cellpose model over every ``.tif`` in ``src`` and optionally save masks.

    Batches the workload and writes results to ``<src>/<model_name>``.

    :param src: Directory containing input ``.tif`` images.
    :param model: Instantiated ``cellpose.models.CellposeModel``.
    :param model_name: Model identifier; names the output subdirectory. It no
        longer selects a channel pair — Cellpose 4 deprecated
        ``eval(channels=...)``, so the pre-SAM ``cyto``/``cyto2``/``nucleus``
        channel conventions had no effect on the network's input.
    :param batch_size: Number of images loaded per iteration.
    :param diameter: Estimated object diameter in pixels.
    :param cellprob_threshold: Cell probability threshold passed to Cellpose.
    :param flow_threshold: Flow error threshold passed to Cellpose.
    :param grayscale: When True, force single-channel input.
    :param save: When True, write masks under ``<src>/<model_name>``.
    :param normalize: When True, load images with normalization/background pipeline.
    :param channels: Channel indices used when loading images.
    :param percentiles: Percentile clipping range applied during normalization.
    :param invert: When True, invert intensities during load.
    :param plot: When True, display mask/flow diagnostics per image.
    :param resize: When True, resize inputs to ``(target_height, target_width)``.
    :param target_height: Target height for resized inputs.
    :param target_width: Target width for resized inputs.
    :param remove_background: When True, subtract background during normalization.
    :param background: Background value used when ``remove_background`` is set.
    :param Signal_to_noise: Minimum SNR threshold for retained signal.
    :param verbose: When True, print Cellpose settings to the console.
    :returns: None.
    """
    from .io import _load_images_and_labels, _load_normalized_images_and_labels
    from .utils import resize_images_and_labels, resizescikit, print_progress
    from .plot import print_mask_and_flows

    dst = os.path.join(src, model_name)
    os.makedirs(dst, exist_ok=True)

    if grayscale:
        print("grayscale=True has no effect under Cellpose 4: the channel "
              "pair (eval channels=) is deprecated and ignored.")

    all_image_files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.tif')]
    random.shuffle(all_image_files)
        
    if verbose == True:
        print(f'Cellpose settings: Model: {model_name}, channels: {channels}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
    
    time_ls = []
    for i in range(0, len(all_image_files), batch_size):
        image_files = all_image_files[i:i+batch_size]

        if normalize:
            images, _, image_names, _, orig_dims = _load_normalized_images_and_labels(image_files, None, channels, percentiles, invert, plot, remove_background, background, Signal_to_noise, target_height, target_width)
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
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
                         channel_axis=cellpose_channel_axis(stack),
                         diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold,
                         rescale=None,
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
                write_tiff(output_filename, mask)

def check_cellpose_models(settings):
    """Run each stock Cellpose model over ``settings['src']`` for side-by-side comparison.

    :param settings: Settings dict; canonicalized via
        :func:`spacr.settings.get_check_cellpose_models_default_settings`.
    :returns: None.
    """
    from .settings import get_check_cellpose_models_default_settings
    
    settings = get_check_cellpose_models_default_settings(settings)
    src = settings['src']

    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
    settings_df['setting_value'] = settings_df['setting_value'].apply(str)
    display(settings_df)

    cellpose_models = ['cpsam']
    from .accelerator import cellpose_kwargs

    for model_name in cellpose_models:

        model = cp_models.CellposeModel(pretrained_model=model_name,
                                        **cellpose_kwargs())
        print(f'Using {model_name}')
        generate_masks_from_imgs(src, model, model_name, settings['batch_size'], settings['diameter'], settings['CP_prob'], settings['flow_threshold'], settings['grayscale'], settings['save'], settings['normalize'], settings['channels'], settings['percentiles'], settings['invert'], settings['plot'], settings['resize'], settings['target_height'], settings['target_width'], settings['remove_background'], settings['background'], settings['Signal_to_noise'], settings['verbose'])

    return
