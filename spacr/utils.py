
import os, re, sqlite3, torch, torchvision, random, string, shutil, cv2, tarfile, glob, psutil, platform, gzip, subprocess, time, requests, ast, traceback

import numpy as np
import pandas as pd
from cellpose import models as cp_models
from cellpose import denoise
from functools import partial

from skimage import morphology
from skimage.measure import label, regionprops_table, regionprops
import skimage.measure as measure
from skimage.transform import resize as resizescikit
from skimage.morphology import dilation, square
from skimage.measure import find_contours
from skimage.segmentation import clear_border, find_boundaries
from scipy.stats import pearsonr

from skimage.filters import (threshold_otsu, threshold_local, gaussian,frangi, sato, meijering, difference_of_gaussians,apply_hysteresis_threshold,)
from skimage.morphology import white_tophat, disk
from skimage.feature import blob_log, blob_dog

from collections import defaultdict, OrderedDict
from PIL import Image
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from functools import reduce
try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass
from typing import Optional, Any

from multiprocessing import Pool, cpu_count, set_start_method, get_start_method
from concurrent.futures import ThreadPoolExecutor

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Subset
from torch.autograd import grad

from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision import models as tv_models

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl

from scipy import stats
import scipy.ndimage as ndi
from scipy.spatial import distance
from scipy.stats import fisher_exact, f_oneway, kruskal
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_fill_holes

from skimage.exposure import rescale_intensity
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from huggingface_hub import list_repo_files

#from spacr import __file__ as spacr_path
spacr_path = os.path.join(os.path.dirname(__file__), '__init__.py')

import umap.umap_ as umap

import logging
from functools import wraps

import os
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries, watershed
from skimage.feature import peak_local_max
from joblib import Parallel, delayed
import tifffile


def _load_image(filepath):
    """Load a .tif or .npy image."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.npy':
        return np.load(filepath)
    elif ext in ('.tif', '.tiff'):
        return tifffile.imread(filepath)
    return None


def _save_image(filepath, img):
    """Save image as .tif or .npy matching original format."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.npy':
        np.save(filepath, img)
    else:
        tifffile.imwrite(filepath, img)


def _union_find_root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union_find_merge(parent, a, b):
    ra = _union_find_root(parent, a)
    rb = _union_find_root(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def _compute_label_perimeters(label_img):
    """Return dict {label: perimeter_pixel_count}."""
    boundaries = find_boundaries(label_img, mode='inner')
    boundary_labels = label_img[boundaries]
    unique, counts = np.unique(boundary_labels[boundary_labels > 0], return_counts=True)
    return dict(zip(unique.astype(int), counts.astype(int)))


def _compute_shared_boundaries(label_img):
    """Return dict {(min_label, max_label): shared_pixel_count}."""
    shared = {}
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(label_img, dy, axis=0), dx, axis=1)
        mask = (label_img > 0) & (shifted > 0) & (label_img != shifted)
        if not np.any(mask):
            continue
        a = label_img[mask].astype(int)
        b = shifted[mask].astype(int)
        for la, lb in zip(a, b):
            pair = (min(la, lb), max(la, lb))
            shared[pair] = shared.get(pair, 0) + 1
    return shared


def _get_boundary_coords(label_img, la, lb):
    """Get pixel coordinates along the shared boundary between two labels."""
    coords = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(label_img, dy, axis=0), dx, axis=1)
        mask = ((label_img == la) & (shifted == lb)) | ((label_img == lb) & (shifted == la))
        ys, xs = np.where(mask)
        coords.extend(zip(ys, xs))
    return coords


def _merge_by_perimeter(label_img, perimeter_fraction, parent):
    """Mark label pairs for merging based on shared perimeter fraction."""
    perimeters = _compute_label_perimeters(label_img)
    shared = _compute_shared_boundaries(label_img)

    for (la, lb), shared_px in shared.items():
        perim_a = perimeters.get(la, 1)
        perim_b = perimeters.get(lb, 1)
        smaller_perim = min(perim_a, perim_b)
        if shared_px / smaller_perim >= perimeter_fraction:
            _union_find_merge(parent, la, lb)


def _merge_by_intensity(label_img, intensity_img, parent,
                        intensity_threshold_method='mean',
                        intensity_percentile=75):
    """Mark label pairs for merging if boundary intensity is high
    (no real edge between them).

    For each touching pair, compare the mean intensity along the shared
    boundary to the interior intensity of the dimmer object.  If the
    boundary intensity >= threshold, merge.

    Parameters
    ----------
    intensity_threshold_method : str
        'mean'  – boundary mean >= mean of dimmer label interior
        'percentile' – boundary mean >= given percentile of dimmer label
    intensity_percentile : int
        Percentile used when method='percentile'.
    """
    shared = _compute_shared_boundaries(label_img)

    # Pre-compute per-label intensity stats
    labels_present = np.unique(label_img)
    labels_present = labels_present[labels_present > 0]
    label_stats = {}
    for l in labels_present:
        vals = intensity_img[label_img == l]
        label_stats[int(l)] = {
            'mean': np.mean(vals),
            'percentile': np.percentile(vals, intensity_percentile),
        }

    for (la, lb), _ in shared.items():
        coords = _get_boundary_coords(label_img, la, lb)
        if not coords:
            continue
        ys, xs = zip(*coords)
        boundary_intensity = np.mean(intensity_img[ys, xs])

        stats_a = label_stats.get(la)
        stats_b = label_stats.get(lb)
        if stats_a is None or stats_b is None:
            continue

        # Compare to the dimmer of the two objects
        if intensity_threshold_method == 'mean':
            ref = min(stats_a['mean'], stats_b['mean'])
        else:
            ref = min(stats_a['percentile'], stats_b['percentile'])

        if boundary_intensity >= ref:
            _union_find_merge(parent, la, lb)


def _split_by_watershed(label_img, area_multiplier=2.0, min_distance=10,
                        min_object_area=100):
    """Split labels whose area exceeds area_multiplier × median object area.

    Uses distance-transform watershed seeded by local maxima.

    Parameters
    ----------
    area_multiplier : float
        Only split objects with area > multiplier * median area.
    min_distance : int
        Minimum pixel distance between watershed seeds.
    min_object_area : int
        Absolute minimum area (px) below which objects are never split,
        regardless of the median multiplier.
    """
    labels_present = np.unique(label_img)
    labels_present = labels_present[labels_present > 0]
    if len(labels_present) == 0:
        return label_img

    areas = ndimage.sum(np.ones_like(label_img), label_img, labels_present)
    area_map = dict(zip(labels_present.astype(int), areas.astype(int)))
    median_area = np.median(list(area_map.values()))
    threshold = max(area_multiplier * median_area, min_object_area)

    output = label_img.copy()
    next_label = int(label_img.max()) + 1

    for lbl, area in area_map.items():
        if area <= threshold:
            continue

        obj_mask = (label_img == lbl)
        dist = ndimage.distance_transform_edt(obj_mask)

        local_max_coords = peak_local_max(dist, min_distance=min_distance,
                                          labels=obj_mask.astype(int))
        if len(local_max_coords) <= 1:
            continue

        seeds = np.zeros_like(label_img, dtype=np.int32)
        for i, (y, x) in enumerate(local_max_coords, start=1):
            seeds[y, x] = i

        ws = watershed(-dist, markers=seeds, mask=obj_mask)

        ws_labels = np.unique(ws)
        ws_labels = ws_labels[ws_labels > 0]
        for wl in ws_labels:
            output[ws == wl] = next_label
            next_label += 1

    return output


def _relabel_sequential(label_img):
    """Relabel to sequential uint16 IDs starting at 1."""
    present = np.unique(label_img)
    present = present[present > 0]
    mapping = np.zeros(int(label_img.max()) + 1, dtype=np.uint16)
    for new_id, old_id in enumerate(present, start=1):
        mapping[int(old_id)] = new_id
    return mapping[label_img].astype(np.uint16)


def _apply_union_find(label_img, parent):
    """Apply union-find mapping and relabel sequentially."""
    mapping = np.zeros(int(label_img.max()) + 1, dtype=np.int64)
    for l in range(1, len(mapping)):
        if l in parent:
            mapping[l] = _union_find_root(parent, l)
        else:
            mapping[l] = l
    merged = mapping[label_img]
    return _relabel_sequential(merged.astype(np.uint16))
    
def _filter_objects(label_img, intensity_img=None, min_area=0, max_area=0,
                    remove_border=False, min_intensity_percentile=0, 
                    max_intensity_percentile=100):
    """Remove objects by area, border contact, and intensity percentile.

    Parameters
    ----------
    label_img : ndarray (uint16)
        Label image.
    intensity_img : ndarray (float32) or None
        Corresponding intensity image for intensity filtering.
    min_area : int
        Remove objects with area < min_area. 0 = disabled.
    max_area : int
        Remove objects with area > max_area. 0 = disabled.
    remove_border : bool
        Remove objects touching any image edge.
    min_intensity_percentile : float
        Remove objects whose mean intensity is below this percentile
        of all object mean intensities. 0 = disabled.
    max_intensity_percentile : float
        Remove objects whose mean intensity is above this percentile
        of all object mean intensities. 100 = disabled.

    Returns
    -------
    ndarray (uint16)
        Filtered and relabelled image.
    """
    labels_present = np.unique(label_img)
    labels_present = labels_present[labels_present > 0]

    if len(labels_present) == 0:
        return label_img

    remove = set()
    
    # Pre-compute areas
    areas = {}
    for lbl in labels_present:
        areas[int(lbl)] = int(np.sum(label_img == lbl))

    # Area filter
    removed_by_area = 0
    if min_area > 0:
        for lbl, area in areas.items():
            if area < min_area:
                remove.add(lbl)
                removed_by_area += 1
    if max_area > 0:
        for lbl, area in areas.items():
            if area > max_area:
                remove.add(lbl)
                removed_by_area += 1
    if removed_by_area > 0:
        print(f"  Area filter: removed {removed_by_area}/{len(labels_present)} objects "
              f"(min_area={min_area}, max_area={max_area})")

    # Border filter
    if remove_border:
        h, w = label_img.shape
        border_labels = set()
        border_labels.update(np.unique(label_img[0, :]).tolist())
        border_labels.update(np.unique(label_img[-1, :]).tolist())
        border_labels.update(np.unique(label_img[:, 0]).tolist())
        border_labels.update(np.unique(label_img[:, -1]).tolist())
        border_labels.discard(0)
        new_border = border_labels - remove
        remove.update(border_labels)
        if len(new_border) > 0:
            print(f"  Border filter: removed {len(new_border)} additional objects")

    # Intensity percentile filter
    do_intensity_filter = (min_intensity_percentile > 0 or max_intensity_percentile < 100)
    if do_intensity_filter and intensity_img is not None:
        # Compute mean intensity per object
        remaining = [lbl for lbl in labels_present if int(lbl) not in remove]
        if len(remaining) > 1:
            mean_intensities = {}
            for lbl in remaining:
                mean_intensities[int(lbl)] = float(np.mean(intensity_img[label_img == lbl]))
            
            values = list(mean_intensities.values())
            low_thresh = np.percentile(values, min_intensity_percentile) if min_intensity_percentile > 0 else -np.inf
            high_thresh = np.percentile(values, max_intensity_percentile) if max_intensity_percentile < 100 else np.inf
            
            removed_by_intensity = 0
            for lbl, mean_val in mean_intensities.items():
                if mean_val < low_thresh or mean_val > high_thresh:
                    remove.add(lbl)
                    removed_by_intensity += 1
            
            if removed_by_intensity > 0:
                print(f"  Intensity filter: removed {removed_by_intensity}/{len(remaining)} objects "
                      f"(percentile range [{min_intensity_percentile}, {max_intensity_percentile}], "
                      f"thresholds [{low_thresh:.1f}, {high_thresh:.1f}])")

    # Apply removal
    total_removed = len(remove)
    total_original = len(labels_present)
    if remove:
        mask = np.isin(label_img, list(remove))
        label_img[mask] = 0
    
    result = _relabel_sequential(label_img)
    remaining_count = len(np.unique(result)) - 1  # exclude 0
    print(f"  Filter summary: {total_original} objects → {remaining_count} objects ({total_removed} removed)")
    
    return result

def _process_single_fov_in_memory(mask, intensity_img, intensity_channel,
                                  do_split, do_perimeter_merge, do_intensity_merge,
                                  perimeter_fraction, area_multiplier, min_distance,
                                  min_object_area, intensity_threshold_method,
                                  intensity_percentile, min_area, max_area,
                                  remove_border_objects, min_intensity_percentile, 
                                  max_intensity_percentile,
                                  progress_callback=None, fov_index=0, total_fovs=0, op_name=''):
    """Process one field of view in memory: split → merge → filter."""

    start = time.time()

    if mask is None:
        return None

    label_img = np.asarray(mask).astype(np.uint16).copy()
    
    n_before = len(np.unique(label_img)) - 1
    if n_before == 0:
        print(f"  FOV {fov_index}: empty mask, skipping")
        return label_img

    intensity_img_use = None
    if (do_intensity_merge or min_intensity_percentile > 0 or max_intensity_percentile < 100) and intensity_img is not None:
        raw = np.asarray(intensity_img)

        if raw.ndim == 2:
            intensity_img_use = raw.astype(np.float32)
        elif raw.ndim == 3 and intensity_channel is not None:
            if raw.shape[-1] <= 4:
                if intensity_channel >= raw.shape[-1]:
                    raise ValueError(
                        f"intensity_channel={intensity_channel} out of bounds for channel-last image with shape {raw.shape}"
                    )
                intensity_img_use = raw[..., intensity_channel].astype(np.float32)
            elif raw.shape[0] <= 4:
                if intensity_channel >= raw.shape[0]:
                    raise ValueError(
                        f"intensity_channel={intensity_channel} out of bounds for channel-first image with shape {raw.shape}"
                    )
                intensity_img_use = raw[intensity_channel].astype(np.float32)
            else:
                if intensity_channel >= raw.shape[-1]:
                    raise ValueError(
                        f"intensity_channel={intensity_channel} out of bounds for image with shape {raw.shape}"
                    )
                intensity_img_use = raw[..., intensity_channel].astype(np.float32)
        else:
            intensity_img_use = raw.astype(np.float32)

    # --- Split phase ---
    if do_split:
        label_img = _split_by_watershed(
            label_img,
            area_multiplier=area_multiplier,
            min_distance=min_distance,
            min_object_area=min_object_area,
        )
        label_img = _relabel_sequential(label_img)
        n_after_split = len(np.unique(label_img)) - 1
        if n_after_split != n_before:
            print(f"  FOV {fov_index} split: {n_before} → {n_after_split} objects")

    # --- Merge phase ---
    all_labels = np.unique(label_img)
    all_labels = all_labels[all_labels > 0]

    if len(all_labels) > 0:
        parent = {int(l): int(l) for l in all_labels}
        n_before_merge = len(all_labels)

        if do_perimeter_merge:
            _merge_by_perimeter(label_img, perimeter_fraction, parent)

        if do_intensity_merge and intensity_img_use is not None:
            _merge_by_intensity(
                label_img,
                intensity_img_use,
                parent,
                intensity_threshold_method=intensity_threshold_method,
                intensity_percentile=intensity_percentile
            )

        label_img = _apply_union_find(label_img, parent)
        n_after_merge = len(np.unique(label_img)) - 1
        if n_after_merge != n_before_merge:
            print(f"  FOV {fov_index} merge: {n_before_merge} → {n_after_merge} objects")

    # --- Filter phase ---
    label_img = _filter_objects(
        label_img,
        intensity_img_use,
        min_area=min_area,
        max_area=max_area,
        remove_border=remove_border_objects,
        min_intensity_percentile=min_intensity_percentile,
        max_intensity_percentile=max_intensity_percentile,
    )

    duration = time.time() - start
    if progress_callback:
        progress_callback(fov_index, total_fovs, duration, op_name)

    return label_img
    
def merge_split_objects(mask_src, intensity_img_src=None, intensity_channel=None,
                        perimeter_fraction=0.5, intensity_merge=False, intensity_split=False,
                        area_multiplier=2.0, min_distance=10, min_object_area=100,
                        intensity_threshold_method='mean', intensity_percentile=75,
                        min_area=0, max_area=0, remove_border_objects=False,
                        min_intensity_percentile=0, max_intensity_percentile=100,
                        n_jobs=1, progress_callback=None, op_name=''):
    """Split, merge, and filter labeled objects across a directory of masks.

    Runs the split -> merge -> filter pipeline on each mask file in
    ``mask_src`` in parallel, overwriting each mask in place.

    :param mask_src: directory containing mask .tif/.tiff/.npy files.
    :param intensity_img_src: directory of matched intensity images, or ``None``.
    :param intensity_channel: channel index to pull from multi-channel intensity images.
    :param perimeter_fraction: minimum shared-boundary fraction for perimeter-based merging.
    :param intensity_merge: enable boundary-intensity-based merging.
    :param intensity_split: enable watershed splitting of oversized objects.
    :param area_multiplier: split objects with area > multiplier * median.
    :param min_distance: minimum pixel distance between watershed seeds.
    :param min_object_area: absolute minimum area below which objects are never split.
    :param intensity_threshold_method: ``'mean'`` or ``'percentile'`` boundary comparison.
    :param intensity_percentile: percentile used when method is ``'percentile'``.
    :param min_area: remove objects smaller than this (px); 0 disables.
    :param max_area: remove objects larger than this (px); 0 disables.
    :param remove_border_objects: drop objects touching the image border.
    :param min_intensity_percentile: drop objects below this intensity percentile; 0 disables.
    :param max_intensity_percentile: drop objects above this intensity percentile; 100 disables.
    :param n_jobs: parallel worker count.
    :param progress_callback: optional callback(fov_index, total, duration, op_name).
    :param op_name: label passed to the progress callback.
    :returns: None.
    """
    valid_ext = ('.tif', '.tiff', '.npy')
    mask_files = sorted([f for f in os.listdir(mask_src)
                         if os.path.splitext(f)[1].lower() in valid_ext])
    if not mask_files:
        return

    do_perimeter_merge = perimeter_fraction > 0
    do_intensity_merge = intensity_merge and intensity_img_src is not None
    do_split = intensity_split

    mask_paths = [os.path.join(mask_src, f) for f in mask_files]
    if intensity_img_src is not None:
        intensity_paths = [os.path.join(intensity_img_src, f) for f in mask_files]
    else:
        intensity_paths = [None] * len(mask_files)

    total = len(mask_paths)

    Parallel(n_jobs=n_jobs)(
        delayed(_process_single_fov)(
            mp, ip, intensity_channel,
            do_split, do_perimeter_merge, do_intensity_merge,
            perimeter_fraction, area_multiplier, min_distance,
            min_object_area, intensity_threshold_method,
            intensity_percentile, min_area, max_area,
            remove_border_objects, min_intensity_percentile, 
            max_intensity_percentile,
            progress_callback, idx, total, op_name,
        )
        for idx, (mp, ip) in enumerate(zip(mask_paths, intensity_paths))
    )

def _process_single_fov(mask_path, intensity_path, intensity_channel,
                        do_split, do_perimeter_merge, do_intensity_merge,
                        perimeter_fraction, area_multiplier, min_distance,
                        min_object_area, intensity_threshold_method,
                        intensity_percentile, min_area, max_area,
                        remove_border_objects, min_intensity_percentile, 
                        max_intensity_percentile,
                        progress_callback=None, fov_index=0, total_fovs=0, op_name=''):
    """Process one field of view: split → merge → filter."""
    import time
    start = time.time()
    
    label_img = _load_image(mask_path)
    if label_img is None:
        return
    label_img = label_img.astype(np.uint16)

    intensity_img = None
    if (do_intensity_merge or min_intensity_percentile > 0 or max_intensity_percentile < 100) and intensity_path is not None:
        raw = _load_image(intensity_path)
        if raw is not None:
            if raw.ndim >= 2 and intensity_channel is not None:
                intensity_img = raw[intensity_channel].astype(np.float32)
            else:
                intensity_img = raw.astype(np.float32)

    if do_split:
        label_img = _split_by_watershed(
            label_img,
            area_multiplier=area_multiplier,
            min_distance=min_distance,
            min_object_area=min_object_area,
        )
        label_img = _relabel_sequential(label_img)

    all_labels = np.unique(label_img)
    all_labels = all_labels[all_labels > 0]
    if len(all_labels) > 0:
        parent = {int(l): int(l) for l in all_labels}

        if do_perimeter_merge:
            _merge_by_perimeter(label_img, perimeter_fraction, parent)

        if do_intensity_merge and intensity_img is not None:
            _merge_by_intensity(label_img, intensity_img, parent,
                                intensity_threshold_method=intensity_threshold_method,
                                intensity_percentile=intensity_percentile)

        label_img = _apply_union_find(label_img, parent)

    label_img = _filter_objects(label_img, intensity_img,
                                min_area=min_area, max_area=max_area,
                                remove_border=remove_border_objects,
                                min_intensity_percentile=min_intensity_percentile,
                                max_intensity_percentile=max_intensity_percentile)

    _save_image(mask_path, label_img)
    
    duration = time.time() - start
    if progress_callback:
        progress_callback(fov_index, total_fovs, duration, op_name)

def _organelle_diagnostic(img, morphology, method, settings):
    """
    Generate a diagnostic image for organelle segmentation QC.

    Returns the processed intermediate image and a descriptive title,
    depending on the morphology mode and method used.

    Parameters
    ----------
    img : ndarray
        2-D float32 single-channel image.
    morphology : str
        One of 'spots', 'network', 'irregular', 'ring'.
    method : str
        Segmentation method used.
    settings : dict
        Organelle settings.

    Returns
    -------
    diag_img : ndarray
        2-D image showing the intermediate processing step.
    diag_title : str
        Description for the plot title.
    """

    img_norm = img.astype(np.float64)
    pmin, pmax = np.percentile(img_norm, (1, 99))
    if pmax - pmin > 0:
        img_norm = np.clip((img_norm - pmin) / (pmax - pmin), 0, 1)

    if morphology == 'spots':
        if method == 'log':
            blobs = blob_log(img_norm,
                             min_sigma=settings.get('organelle_log_min_sigma', 1),
                             max_sigma=settings.get('organelle_log_max_sigma', 10),
                             num_sigma=settings.get('organelle_log_num_sigma', 10),
                             threshold=settings.get('organelle_log_threshold', 0.01))
            # Draw blob circles on the normalised image
            diag_img = img_norm.copy()
            for y, x, sigma in blobs:
                rr, cc = np.ogrid[-int(sigma*2):int(sigma*2)+1, -int(sigma*2):int(sigma*2)+1]
                circle = rr**2 + cc**2 <= (sigma * np.sqrt(2))**2
                yy = np.clip(int(y) + np.where(circle)[0] - int(sigma*2), 0, img.shape[0]-1)
                xx = np.clip(int(x) + np.where(circle)[1] - int(sigma*2), 0, img.shape[1]-1)
                diag_img[yy, xx] = 1.0
            return diag_img, f'LoG detections ({len(blobs)} blobs)'

        elif method == 'dog':
            blobs = blob_dog(img_norm,
                             min_sigma=settings.get('organelle_dog_sigma_low', 1.0),
                             max_sigma=settings.get('organelle_dog_sigma_high', 3.0),
                             threshold=settings.get('organelle_log_threshold', 0.01))
            diag_img = img_norm.copy()
            for y, x, sigma in blobs:
                rr, cc = np.ogrid[-int(sigma*2):int(sigma*2)+1, -int(sigma*2):int(sigma*2)+1]
                circle = rr**2 + cc**2 <= (sigma * np.sqrt(2))**2
                yy = np.clip(int(y) + np.where(circle)[0] - int(sigma*2), 0, img.shape[0]-1)
                xx = np.clip(int(x) + np.where(circle)[1] - int(sigma*2), 0, img.shape[1]-1)
                diag_img[yy, xx] = 1.0
            return diag_img, f'DoG detections ({len(blobs)} blobs)'

        else:
            # otsu / adaptive: show top-hat filtered image
            radius = settings.get('organelle_tophat_radius', 5)
            filtered = white_tophat(img, disk(radius))
            return filtered, f'Top-hat filtered (r={radius})'

    elif morphology == 'network':
        if method == 'ridge':
            sigmas = settings.get('organelle_ridge_sigmas', [1, 2, 3])
            filter_name = settings.get('organelle_ridge_filter', 'frangi')
            ridge_filters = {'frangi': frangi, 'sato': sato, 'meijering': meijering}
            enhanced = ridge_filters[filter_name](img_norm, sigmas=sigmas, black_ridges=False)
            return enhanced, f'{filter_name} ridge (sigmas={sigmas})'

        elif method == 'hysteresis':
            low = settings.get('organelle_hysteresis_low', 0.2)
            high = settings.get('organelle_hysteresis_high', 0.6)
            smooth = gaussian(img, sigma=1)
            if low < 1.0:
                low_abs = np.percentile(smooth, low * 100)
            else:
                low_abs = low
            if high < 1.0:
                high_abs = np.percentile(smooth, high * 100)
            else:
                high_abs = high
            binary = apply_hysteresis_threshold(smooth, low_abs, high_abs)
            return binary.astype(np.float64), f'Hysteresis (low={low}, high={high})'

        else:
            # otsu / adaptive: show Gaussian smoothed
            smooth = gaussian(img, sigma=1)
            return smooth, 'Gaussian smoothed (σ=1)'

    elif morphology == 'irregular':
        morph_r = settings.get('organelle_morph_radius', 3)
        smooth = gaussian(img, sigma=max(morph_r / 2, 1))
        return smooth, f'Gaussian smoothed (σ={max(morph_r/2, 1):.1f})'

    elif morphology == 'ring':
        sigma_inner = settings.get('organelle_ring_sigma_inner', 1.0)
        sigma_outer = settings.get('organelle_ring_sigma_outer', 3.0)
        enhanced = np.abs(difference_of_gaussians(img_norm, sigma_inner, sigma_outer))
        return enhanced, f'DoG ring enhancement (σ={sigma_inner}/{sigma_outer})'

    else:
        return img_norm, 'Normalised image'

def debug(enabled=True, logger_name = None):
    """Decorator that temporarily sets the given logger to DEBUG for the wrapped call.

    :param enabled: no-op when ``False``.
    :param logger_name: logger name to tweak; defaults to the function's module logger.
    :returns: decorator function.
    """
    def decorator(func):
        """Inner decorator that binds the logger for ``func`` and returns the wrapper."""
        log = logging.getLogger(logger_name or func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Temporarily bump the logger to DEBUG while ``func`` runs, then restore its level."""
            if not enabled:
                return func(*args, **kwargs)

            old_level = log.level  # may be logging.NOTSET
            try:
                log.setLevel(logging.DEBUG)
                log.debug(">>> Entering %s", func.__name__)
                result = func(*args, **kwargs)
                log.debug("<<< Exiting %s", func.__name__)
                return result
            finally:
                log.setLevel(old_level)

        return wrapper

    return decorator

def _generate_mask_random_cmap(mask):
    """Return a ``ListedColormap`` with a random color per label in ``mask``."""
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def filepaths_to_database(img_paths, settings, source_folder, crop_mode):
    """Insert cropped PNG filepaths and parsed well/object IDs into the measurements DB.

    :param img_paths: iterable of PNG paths for cropped objects.
    :param settings: settings dict; ``timelapse`` toggles time_id parsing.
    :param source_folder: experiment root; DB is written to ``measurements/measurements.db``.
    :param crop_mode: one of ``'cell'``, ``'nucleus'``, ``'pathogen'``, ``'cytoplasm'``.
    :returns: None.
    """
    png_df = pd.DataFrame(img_paths, columns=['png_path'])

    png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))

    parts = png_df['file_name'].apply(lambda x: pd.Series(_map_wells_png(x, timelapse=settings['timelapse'])))

    columns = ['plateID', 'rowID', 'columnID', 'fieldID']

    if settings['timelapse']:
        columns = columns + ['time_id']

    columns = columns + ['prcfo']

    if crop_mode == 'cell':
        columns = columns + ['cell_id']

    if crop_mode == 'nucleus':
        columns = columns + ['nucleus_id']

    if crop_mode == 'pathogen':
        columns = columns + ['pathogen_id']

    if crop_mode == 'cytoplasm':
        columns = columns + ['cytoplasm_id']

    png_df[columns] = parts

    try:
        conn = sqlite3.connect(f'{source_folder}/measurements/measurements.db', timeout=5)
        png_df.to_sql('png_list', conn, if_exists='append', index=False)
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}", flush=True)
        traceback.print_exc()

def activation_maps_to_database(img_paths, source_folder, settings):
    """Insert activation-map PNG paths and parsed well IDs into the dataset DB.

    :param img_paths: iterable of PNG paths for activation-map images.
    :param source_folder: experiment root; DB written to ``measurements/<dataset>.db``.
    :param settings: settings dict; must contain ``dataset`` and ``cam_type``.
    :returns: None.
    """
    from .io import _create_database

    png_df = pd.DataFrame(img_paths, columns=['png_path'])
    png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))
    parts = png_df['file_name'].apply(lambda x: pd.Series(_map_wells_png(x, timelapse=False)))
    columns = ['plateID', 'rowID', 'columnID', 'fieldID', 'prcfo', 'object']
    png_df[columns] = parts

    dataset_name = os.path.splitext(os.path.basename(settings['dataset']))[0]
    database_name = f"{source_folder}/measurements/{dataset_name}.db"

    if not os.path.exists(database_name):
        _create_database(database_name)

    try:
        conn = sqlite3.connect(database_name, timeout=5)
        png_df.to_sql(f"{settings['cam_type']}_list", conn, if_exists='append', index=False)
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}", flush=True)
        traceback.print_exc()

def activation_correlations_to_database(df, img_paths, source_folder, settings):
    """Merge per-image correlation stats with parsed well IDs and insert into the dataset DB.

    :param df: DataFrame of correlation stats indexed by ``file_name``.
    :param img_paths: iterable of PNG paths matching rows of ``df``.
    :param source_folder: experiment root; DB written to ``measurements/<dataset>.db``.
    :param settings: settings dict; must contain ``dataset`` and ``cam_type``.
    :returns: None.
    """
    from .io import _create_database

    png_df = pd.DataFrame(img_paths, columns=['png_path'])
    png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))
    parts = png_df['file_name'].apply(lambda x: pd.Series(_map_wells_png(x, timelapse=False)))
    columns = ['plateID', 'rowID', 'columnID', 'fieldID', 'prcfo', 'object']
    png_df[columns] = parts

    # Align both DataFrames by file_name
    png_df.set_index('file_name', inplace=True)
    df.set_index('file_name', inplace=True)

    merged_df = pd.concat([png_df, df], axis=1)
    merged_df.reset_index(inplace=True)

    dataset_name = os.path.splitext(os.path.basename(settings['dataset']))[0]
    database_name = f"{source_folder}/measurements/{dataset_name}.db"

    if not os.path.exists(database_name):
        _create_database(database_name)

    try:
        conn = sqlite3.connect(database_name, timeout=5)
        merged_df.to_sql(f"{settings['cam_type']}_correlations", conn, if_exists='append', index=False)
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}", flush=True)
        traceback.print_exc()

def calculate_activation_correlations(inputs, activation_maps, file_names, manders_thresholds=None):
    """Compute per-image Pearson and Manders correlations between input and activation channels.

    :param inputs: input image batch, tensor of shape ``(B, C, H, W)``.
    :param activation_maps: activation-map batch, tensor of shape ``(B, C, H, W)`` or ``(B, H, W)``.
    :param file_names: file names corresponding to each image in the batch.
    :param manders_thresholds: intensity percentiles used for Manders coefficients. Default ``[15, 50, 75]``.
    :returns: DataFrame with one row per image and one column per channel-pair statistic.
    """
    
    # Ensure tensors are detached and moved to CPU before converting to numpy
    if manders_thresholds is None:
        manders_thresholds = [15, 50, 75]
    inputs = inputs.detach().cpu()
    activation_maps = activation_maps.detach().cpu()

    batch_size, in_channels, height, width = inputs.shape
    
    if activation_maps.dim() == 3:
        # If activation maps have no channels, add a dummy channel dimension
        activation_maps = activation_maps.unsqueeze(1)  # Now shape is (batch_size, 1, height, width)
    
    _, act_channels, act_height, act_width = activation_maps.shape

    # Ensure that the inputs and activation maps are the same size
    if (height != act_height) or (width != act_width):
        activation_maps = torch.nn.functional.interpolate(activation_maps, size=(height, width), mode='bilinear')

    # Dictionary to collect correlation results
    correlations_dict = {'file_name': []}

    # Initialize correlation columns based on input channels and activation map channels
    for in_c in range(in_channels):
        for act_c in range(act_channels):
            correlations_dict[f'channel_{in_c}_activation_{act_c}_pearsons'] = []
            for threshold in manders_thresholds:
                correlations_dict[f'channel_{in_c}_activation_{act_c}_{threshold}_M1'] = []
                correlations_dict[f'channel_{in_c}_activation_{act_c}_{threshold}_M2'] = []

    # Loop over the batch
    for b in range(batch_size):
        input_img = inputs[b]  # Input image channels (C, H, W)
        activation_map = activation_maps[b]  # Activation map channels (C, H, W)

        # Add the file name to the current row
        correlations_dict['file_name'].append(file_names[b])

        # Calculate correlations for each channel pair
        for in_c in range(in_channels):
            input_channel = input_img[in_c].flatten().numpy()  # Flatten the input image channel
            input_channel = input_channel[np.isfinite(input_channel)]  # Remove NaN or inf values

            for act_c in range(act_channels):
                activation_channel = activation_map[act_c].flatten().numpy()  # Flatten the activation map channel
                activation_channel = activation_channel[np.isfinite(activation_channel)]  # Remove NaN or inf values

                # Check if there are valid (non-empty) arrays left to calculate the Pearson correlation
                if input_channel.size > 0 and activation_channel.size > 0:
                    pearson_corr, _ = pearsonr(input_channel, activation_channel)
                else:
                    pearson_corr = np.nan  # Assign NaN if there are no valid data points
                correlations_dict[f'channel_{in_c}_activation_{act_c}_pearsons'].append(pearson_corr)

                # Compute Manders correlations for each threshold
                for threshold in manders_thresholds:
                    # Get the top percentile pixels based on intensity in both channels
                    if input_channel.size > 0 and activation_channel.size > 0:
                        input_threshold = np.percentile(input_channel, threshold)
                        activation_threshold = np.percentile(activation_channel, threshold)

                        # Mask the pixels above the threshold
                        mask = (input_channel >= input_threshold) & (activation_channel >= activation_threshold)

                        # If we have enough pixels, calculate Manders correlation
                        if np.sum(mask) > 0:
                            manders_corr_M1 = np.sum(input_channel[mask] * activation_channel[mask]) / np.sum(input_channel[mask] ** 2)
                            manders_corr_M2 = np.sum(activation_channel[mask] * input_channel[mask]) / np.sum(activation_channel[mask] ** 2)
                        else:
                            manders_corr_M1 = np.nan
                            manders_corr_M2 = np.nan
                    else:
                        manders_corr_M1 = np.nan
                        manders_corr_M2 = np.nan

                    # Store the Manders correlation for this threshold
                    correlations_dict[f'channel_{in_c}_activation_{act_c}_{threshold}_M1'].append(manders_corr_M1)
                    correlations_dict[f'channel_{in_c}_activation_{act_c}_{threshold}_M2'].append(manders_corr_M2)

    # Convert the dictionary to a DataFrame
    df_correlations = pd.DataFrame(correlations_dict)

    return df_correlations

def load_settings(csv_file_path, show=False, setting_key='setting_key', setting_value='setting_value'):
    """Reload a spacr settings CSV (written by :func:`save_settings`) back into a Python dict.

    Every spacr pipeline persists its resolved settings alongside its
    outputs so that a run can be reproduced. This helper re-parses that
    CSV, coercing each value into its original Python type (``bool``,
    ``int``, ``float``, ``None``, ``list``, ``tuple``, ``dict``,
    ``str``).

    :param csv_file_path: path to the CSV file.
    :param show: display the raw DataFrame for debugging. Default ``False``.
    :param setting_key: name of the key column. Default
        ``'setting_key'``.
    :param setting_value: name of the value column. Default
        ``'setting_value'``.
    :returns: dict of parsed settings, ready to pass back into the
        original pipeline entry point.
    :raises ValueError: if the required key / value columns are missing.

    Example:
        .. code-block:: python

            from spacr.utils import load_settings
            from spacr.core import preprocess_generate_masks
            settings = load_settings('/data/plate01/settings/gen_mask_settings.csv')
            preprocess_generate_masks(settings)

    See Also:
        :func:`save_settings` — inverse operation.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    if show:
        display(df)

    # Ensure the columns 'setting_key' and 'setting_value' exist
    if setting_key not in df.columns or setting_value not in df.columns:
        raise ValueError(f"CSV file must contain {setting_key} and {setting_value} columns.")

    def parse_value(value):
        """Parse the string value into the appropriate Python data type."""
        # Handle empty values
        if pd.isna(value) or value == '':
            return None

        # Handle boolean values
        if value == 'True':
            return True
        if value == 'False':
            return False

        # Handle lists, tuples, dictionaries, and other literals
        if value.startswith(('(', '[', '{')):  # If it starts with (, [ or {, use ast.literal_eval
            try:
                parsed_value = ast.literal_eval(value)
                # If parsed_value is a dict, recursively parse its values
                if isinstance(parsed_value, dict):
                    parsed_value = {k: parse_value(v) for k, v in parsed_value.items()}
                return parsed_value
            except (ValueError, SyntaxError):
                pass  # If there's an error, return the value as-is
        
        # Handle numeric values (integers and floats)
        try:
            if '.' in value:
                return float(value)  # If it contains a dot, convert to float
            return int(value)  # Otherwise, convert to integer
        except ValueError:
            pass  # If it's not a valid number, return the value as-is

        # Return the original value if no other type matched
        return value

    # Convert the DataFrame to a dictionary, with parsing of each value
    result_dict = {key: parse_value(value) for key, value in zip(df[setting_key], df[setting_value])}

    return result_dict

def save_settings(settings, name='settings', show=False):
    """Persist a settings dict to ``<src>/settings/<name>.csv`` so a spacr run can be reproduced later.

    Called by every pipeline entry point to snapshot the resolved
    settings before real work starts. The saved copy has ``test_mode``
    and ``plot`` forced to ``False`` so that a downstream
    :func:`load_settings` -> re-run produces a full, headless run.

    :param settings: settings dict; must contain ``src``.
    :param name: base filename (no extension); ``_list`` is appended
        when ``src`` is a list. Default ``'settings'``.
    :param show: display the DataFrame before writing. Default ``False``.
    :returns: None. Writes ``<src>/settings/<name>.csv``.

    Example:
        .. code-block:: python

            from spacr.utils import save_settings
            save_settings(my_settings, name='my_experiment', show=True)

    See Also:
        :func:`load_settings` — inverse operation.
    """
    settings_2 = settings.copy()
    
    if isinstance(settings_2['src'], list):
        src = settings_2['src'][0]
        name = f"{name}_list"
    else:
        src = settings_2['src']
        
    if 'test_mode' in settings_2.keys():
        settings_2['test_mode'] = False
        
        if 'plot' in settings_2.keys():
            settings_2['plot'] = False
            
    settings_df = pd.DataFrame(list(settings_2.items()), columns=['Key', 'Value'])
    
    if show:
        display(settings_df)

    settings_csv = os.path.join(src,'settings',f'{name}.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    print(f"Saving settings to {settings_csv}")
    settings_df.to_csv(settings_csv, index=False)

def print_progress(files_processed, files_to_process, n_jobs, time_ls=None, batch_size=None, operation_type=""):
    """Print a one-line progress report with an ETA derived from mean step time.

    :param files_processed: number of items done (int or list).
    :param files_to_process: total items to do (int or list).
    :param n_jobs: parallelism used to compute ETA.
    :param time_ls: list of per-step durations (seconds) for ETA; ``None`` skips ETA.
    :param batch_size: batch size when ``time_ls`` is per batch rather than per image.
    :param operation_type: label printed alongside the progress line.
    :returns: None.
    """
    if isinstance(files_processed, list):
        files_processed = len(set(files_processed))
    if isinstance(files_to_process, list):
        files_to_process = len(set(files_to_process))
    if isinstance(batch_size, list):
        batch_size = len(batch_size)

    if not isinstance(files_processed, int):
        try:
            files_processed = int(files_processed)
        except Exception:
            files_processed = 0
    if not isinstance(files_to_process, int):
        try:
            files_to_process = int(files_to_process)
        except Exception:
            files_to_process = 0

    time_info = ""
    if time_ls is not None:
        average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
        time_left = (((files_to_process - files_processed) * average_time) / n_jobs) / 60
        if batch_size is None:
            time_info = f'Time/image: {average_time:.3f}sec, Time_left: {time_left:.3f} min.'
        else:
            average_time_img = average_time / batch_size
            time_info = f'Time/batch: {average_time:.3f}sec, Time/image: {average_time_img:.3f}sec, Time_left: {time_left:.3f} min.'
    else:
        time_info = None
    print(f'Progress: {files_processed}/{files_to_process}, operation_type: {operation_type}, {time_info}')

def reset_mp():
    """Set the multiprocessing start method appropriate for the current OS.

    Uses ``spawn`` on Windows and ``fork`` on Linux/macOS.

    :returns: None.
    """
    current_method = get_start_method()
    system = platform.system()
    
    if system == 'Windows':
        if current_method != 'spawn':
            set_start_method('spawn', force=True)
    elif system in ('Linux', 'Darwin'):  # Darwin is macOS
        if current_method != 'fork':
            set_start_method('fork', force=True)

def is_multiprocessing_process(process):
    """Return ``True`` if ``process`` cmdline contains ``multiprocessing``."""
    try:
        for cmd in process.cmdline():
            if 'multiprocessing' in cmd:
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return False

def close_file_descriptors():
    """Close file descriptors from 3 up to the soft NOFILE limit."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for fd in range(3, soft):
        try:
            os.close(fd)
        except OSError:
            pass

def close_multiprocessing_processes():
    """Terminate all detected multiprocessing child processes and close file descriptors."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            # Skip the current process
            if proc.info['pid'] == current_pid:
                continue
            
            # Check if the process is a multiprocessing process
            if is_multiprocessing_process(proc):
                proc.terminate()
                proc.wait(timeout=5)  # Wait up to 5 seconds for the process to terminate
                print(f"Terminated process {proc.info['pid']}")
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Failed to terminate process {proc.info['pid']}: {e}")

    # Close file descriptors
    close_file_descriptors()

def check_mask_folder(src,mask_fldr):
    """Return ``True`` if masks in ``src/masks/mask_fldr`` still need generating.

    :param src: experiment root containing ``masks/`` and ``stack/`` subfolders.
    :param mask_fldr: subfolder name under ``masks/``.
    :returns: ``True`` when the mask folder is missing or has fewer ``.npy`` files
        than the stack folder.
    """
    mask_folder = os.path.join(src,'masks',mask_fldr)
    stack_folder = os.path.join(src,'stack')

    if not os.path.exists(mask_folder):
        return True
    
    mask_count = sum(1 for file in os.listdir(mask_folder) if file.endswith('.npy'))
    stack_count = sum(1 for file in os.listdir(stack_folder) if file.endswith('.npy'))
    
    if mask_count == stack_count:
        print(f'All masks have been generated for {mask_fldr}')
        return False
    else:
        return True

def smooth_hull_lines(cluster_data):
    """Return the x, y coordinates of a smoothed convex-hull outline of a 2-D point set.

    :param cluster_data: 2-D array of point coordinates.
    :returns: tuple ``(x, y)`` of spline-interpolated hull coordinates (100 samples).
    """
    hull = ConvexHull(cluster_data)
    # Extract vertices of the hull
    vertices = hull.points[hull.vertices]
    # Close the loop
    vertices = np.vstack([vertices, vertices[0, :]])
    # Parameterize the vertices
    tck, u = splprep(vertices.T, u=None, s=0.0)
    # Evaluate spline at new parameter values
    new_points = splev(np.linspace(0, 1, 100), tck)
    return new_points[0], new_points[1]

def _gen_rgb_image(image, channels):
    """Return an ``(H, W, 3)`` RGB image built from selected channels of ``image``."""
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    for i, chan in enumerate(channels):
        if chan < image.shape[2]:
            rgb_image[:, :, i] = image[:, :, chan]
    return rgb_image

def _outline_and_overlay(image, rgb_image, mask_dims, outline_colors, outline_thickness):
    outlines = []
    overlayed_image = rgb_image.copy()

    def process_dim(mask_dim):
        """Return a dilated outline image of the labeled mask at ``image[..., mask_dim]``."""
        mask = np.take(image, mask_dim, axis=-1)
        outline = np.zeros_like(mask, dtype=np.uint8)  # Use uint8 for contour detection efficiency

        # Find and draw contours
        for j in np.unique(mask):
            if j == 0:
                continue  # Skip background
            contours = find_contours(mask == j, 0.5)
            # Convert contours for OpenCV format and draw directly to optimize
            cv_contours = [np.flip(contour.astype(int), axis=1) for contour in contours]
            cv2.drawContours(outline, cv_contours, -1, color=255, thickness=outline_thickness) 

        return dilation(outline, square(outline_thickness))

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        outlines = list(executor.map(process_dim, mask_dims))

    # Overlay outlines onto the RGB image
    for i, outline in enumerate(outlines):
        color = np.array(outline_colors[i % len(outline_colors)])
        for j in np.unique(outline):
            if j == 0:
                continue  # Skip background
            mask = outline == j
            overlayed_image[mask] = color  # Direct assignment with broadcasting

    return overlayed_image, outlines, image

def _convert_cq1_well_id(well_id):
    """Convert a linear well index to the CQ1 ``<row_letter><col>`` well format."""
    well_id = int(well_id)
    # ASCII code for 'A'
    ascii_A = ord('A')
    # Calculate row and column
    row, col = divmod(well_id - 1, 24)
    # Convert row to letter (A-P) and adjust col to start from 1
    row_letter = chr(ascii_A + row)
    # Format column as two digits
    well_format = f"{row_letter}{col + 1:02d}" 
    return well_format

def _get_cellpose_batch_size():
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(0)
            vram_gb = device_properties.total_memory / (1024**3)  # Convert bytes to gigabytes
        else:
            print("CUDA is not available. Please check your installation and GPU.")
            return 8
        if vram_gb < 8:
            batch_size = 8
        elif vram_gb > 8 and vram_gb < 12:
            batch_size = 16
        elif vram_gb > 12 and vram_gb < 24:
            batch_size = 48
        elif vram_gb > 24:
            batch_size = 96
        print(f"Device {0}: {device_properties.name}, VRAM: {vram_gb:.2f} GB, cellpose batch size: {batch_size}")
        return batch_size
    except Exception as e:
        return 8

def _extract_filename_metadata(filenames, src, regular_expression, metadata_type='cellvoyager'):
    
    images_by_key = defaultdict(list)

    for filename in filenames:
        match = regular_expression.match(filename)
        if match:
            try:
                try:
                    plate = match.group('plateID')
                except Exception:
                    plate = os.path.basename(src)

                well = match.group('wellID')
                if well[0].isdigit():
                    well = str(_safe_int_convert(well))
                
                field = match.group('fieldID')
                if field[0].isdigit():
                    field = str(_safe_int_convert(field))
                    
                channel = match.group('chanID')
                if channel[0].isdigit():
                    channel = str(_safe_int_convert(channel))
                    
                if 'timeID' in match.groupdict():
                    timeID = match.group('timeID')
                    if timeID[0].isdigit():
                        timeID = str(_safe_int_convert(timeID))
                else:
                    timeID = None
                        
                if 'sliceID' in match.groupdict():
                    sliceID = match.group('sliceID')
                    if sliceID[0].isdigit():
                        sliceID = str(_safe_int_convert(sliceID))
                else:
                    sliceID = None
                    
                if metadata_type =='cq1':
                    orig_well = well
                    well = _convert_cq1_well_id(well)
                    print(f'Converted Well ID: {orig_well} to {well}', end='\r', flush=True)

                key = (plate, well, field, channel, timeID, sliceID)
                file_path = os.path.join(src, filename)
                images_by_key[key].append(file_path)
                
            except IndexError:
                print(f"Could not extract information from filename {filename} using provided regex")
        else:
            print(f"Filename {filename} did not match provided regex: {regular_expression}")
            continue
        
    return images_by_key

def mask_object_count(mask):
    """Return the number of nonzero labeled objects in ``mask``."""
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels!=0])
    return num_objects

def _update_database_with_merged_info(db_path, df, table='png_list', columns=None):
    """Merge extra columns from ``df`` into ``table`` on ``prcfo`` and rewrite the table."""
    # Connect to the SQLite database
    if columns is None:
        columns = ['pathogen', 'treatment', 'host_cells', 'condition', 'prcfo']
    conn = sqlite3.connect(db_path)

    # Read the existing table into a DataFrame
    try:
        existing_df = pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception as e:
        print(f"Failed to read table {table} from database: {e}")
        conn.close()
        return
    
    if 'prcfo' not in df.columns:
        print(f'generating prcfo columns')
        try:
            df['prcfo'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str) + '_' + df['fieldID'].astype(str) + '_o' + df['object_label'].astype(int).astype(str)
        except Exception as e:
            print('Merging on cell failed, trying with cell_id')
        try:
            df['prcfo'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str) + '_' + df['fieldID'].astype(str) + '_o' + df['cell_id'].astype(int).astype(str)
        except Exception as e:
            print(e)
        
    # Merge the existing DataFrame with the new info based on the 'prcfo' column
    merged_df = pd.merge(existing_df, df[columns], on='prcfo', how='left')
    
    # Drop the existing table and replace it with the updated DataFrame
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        merged_df.to_sql(table, conn, index=False)
        print(f"Table {table} successfully updated in the database.")
    except Exception as e:
        print(f"Failed to update table {table} in the database: {e}")
    finally:
        conn.close()

def _generate_representative_images(db_path, cells=None, cell_loc=None, pathogens=None, pathogen_loc=None, treatments=None, treatment_loc=None, channel_of_interest=1, compartments = None, measurement = 'mean_intensity', nr_imgs=16, channel_indices=None, um_per_pixel=0.1, scale_bar_length_um=10, plot=False, fontsize=12, show_filename=True, channel_names=None, update_db=True):
    """Save representative-image grids per condition selected by a compartment measurement ratio."""
    
    if cells is None:
        cells = ['HeLa']
    if pathogens is None:
        pathogens = ['rh']
    if treatments is None:
        treatments = ['cm']
    if compartments is None:
        compartments = ['pathogen','cytoplasm']
    if channel_indices is None:
        channel_indices = [0,1,2]
    from .io import _read_and_join_tables, _save_figure
    from .plot import _plot_images_on_grid
    
    df = _read_and_join_tables(db_path)
    df = annotate_conditions(df, cells, cell_loc, pathogens, pathogen_loc, treatments, treatment_loc)
    
    if update_db:
        _update_database_with_merged_info(db_path, df, table='png_list', columns=['pathogen', 'treatment', 'host_cells', 'condition', 'prcfo'])
    
    if isinstance(compartments, list):
        if len(compartments) > 1:
            df['new_measurement'] = df[f'{compartments[0]}_channel_{channel_of_interest}_{measurement}']/df[f'{compartments[1]}_channel_{channel_of_interest}_{measurement}']
    else:
        df['new_measurement'] = df['cell_area']
    dfs = {condition: df_group for condition, df_group in df.groupby('condition')}
    conditions = df['condition'].dropna().unique().tolist()
    for condition in conditions:
        df = dfs[condition]
        df = _filter_closest_to_stat(df, column='new_measurement', n_rows=nr_imgs, use_median=False)
        png_paths_by_condition = df['png_path'].tolist()
        fig = _plot_images_on_grid(png_paths_by_condition, channel_indices, um_per_pixel, scale_bar_length_um, fontsize, show_filename, channel_names, plot)
        src = os.path.dirname(db_path)
        os.makedirs(src, exist_ok=True)
        _save_figure(fig=fig, src=src, text=condition)
        for channel in channel_indices:
            channel_indices=[channel]
            fig = _plot_images_on_grid(png_paths_by_condition, channel_indices, um_per_pixel, scale_bar_length_um, fontsize, show_filename, channel_names, plot)
            _save_figure(fig, src, text=f'channel_{channel}_{condition}')
            plt.close()
            
# Adjusted mapping function to infer type from location identifiers
def _map_values(row, values, locs):
    """Look up the value assigned to the row/column identifier in ``row``."""
    if locs:
        value_dict = {loc: value for value, loc_list in zip(values, locs) for loc in loc_list}
        # Determine if we're dealing with row or column based on first location identifier
        type_ = 'rowID' if locs[0][0][0] == 'r' else 'columnID'
        return value_dict.get(row[type_], None)
    return values[0] if values else None

def is_list_of_lists(var):
    """Return ``True`` if ``var`` is a list whose every element is also a list."""
    if isinstance(var, list) and all(isinstance(i, list) for i in var):
        return True
    return False

def normalize_to_dtype(array, p1=2, p2=98, percentile_list=None, new_dtype=None):
    """Percentile-normalize each channel of an image stack into the target dtype range.

    :param array: input stack of shape ``(H, W, C)``.
    :param p1: lower percentile. Default ``2``.
    :param p2: upper percentile. Default ``98``.
    :param percentile_list: per-channel ``(low, high)`` pairs; overrides ``p1``/``p2``.
    :param new_dtype: target dtype (``np.uint8``/``np.uint16`` or their string forms).
    :returns: normalized stack with the same shape as ``array``.
    """

    if new_dtype is None:
        out_range = (0, np.iinfo(array.dtype).max)
    elif new_dtype in [np.uint8, np.uint16]:
        out_range = (0, np.iinfo(new_dtype).max)
    elif new_dtype in ['uint8', 'uint16']:
        new_dtype = np.uint8 if new_dtype == 'uint8' else np.uint16
        out_range = (0, np.iinfo(new_dtype).max)
    else:
        out_range = (0, np.iinfo(array.dtype).max)

    nimg = array.shape[2]
    new_stack = np.empty_like(array, dtype=array.dtype)

    for i in range(nimg):
        img = array[:, :, i]
        non_zero_img = img[img > 0]
        if not percentile_list is None:
            percentiles = percentile_list[i]
        else:
            percentile_1 = p1
            percentile_2 = p2
        if percentile_list is None:
            if non_zero_img.size > 0:
                img_min = np.percentile(non_zero_img, percentile_1)
                img_max = np.percentile(non_zero_img, percentile_2)
            else:
                img_min = np.percentile(img, percentile_1)
                img_max = np.percentile(img, percentile_2)
        else:
            img_min = percentiles[0]
            img_max = percentiles[1]

        # Normalize to the range (0, 1) for visualization
        img = rescale_intensity(img, in_range=(img_min, img_max), out_range=out_range)
        new_stack[:, :, i] = img
    return new_stack
    
def _list_endpoint_subdirectories(base_dir):
    """Return leaf subdirectory paths under ``base_dir``, excluding any named ``figure``."""
    
    endpoint_subdirectories = []
    for root, dirs, _ in os.walk(base_dir):
        if not dirs:
            endpoint_subdirectories.append(root)
            
    endpoint_subdirectories = [path for path in endpoint_subdirectories if os.path.basename(path) != 'figure']
    return endpoint_subdirectories
    
def _generate_names(file_name, cell_id, cell_nucleus_ids, cell_pathogen_ids, source_folder, crop_mode='cell', timelapse=None):
    """Build the ``(image_name, folder_path, table_name)`` tuple for a cropped object."""
    non_zero_cell_ids = cell_id[cell_id != 0]
    cell_id_str = "multi" if non_zero_cell_ids.size > 1 else str(non_zero_cell_ids[0]) if non_zero_cell_ids.size == 1 else "none"
    cell_nucleus_ids = cell_nucleus_ids[cell_nucleus_ids != 0]
    cell_nucleus_id_str = "multi" if cell_nucleus_ids.size > 1 else str(cell_nucleus_ids[0]) if cell_nucleus_ids.size == 1 else "none"
    cell_pathogen_ids = cell_pathogen_ids[cell_pathogen_ids != 0]
    cell_pathogen_id_str = "multi" if cell_pathogen_ids.size > 1 else str(cell_pathogen_ids[0]) if cell_pathogen_ids.size == 1 else "none"
    fldr = f"{source_folder}/data/"
    img_name = ""
    if crop_mode == 'nucleus':
        img_name = f"{file_name}_{cell_id_str}_{cell_nucleus_id_str}.png"
        fldr += "single_nucleus/" if cell_nucleus_ids.size == 1 else "multiple_nucleus/" if cell_nucleus_ids.size > 1 else "no_nucleus/"
        fldr += "single_pathogen/" if cell_pathogen_ids.size == 1 else "multiple_pathogens/" if cell_pathogen_ids.size > 1 else "uninfected/"
    elif crop_mode == 'pathogen':
        img_name = f"{file_name}_{cell_id_str}_{cell_pathogen_id_str}.png"
        fldr += "single_nucleus/" if cell_nucleus_ids.size == 1 else "multiple_nucleus/" if cell_nucleus_ids.size > 1 else "no_nucleus/"
        fldr += "infected/" if cell_pathogen_ids.size >= 1 else "uninfected/"
    elif crop_mode == 'cell' or crop_mode == 'cytoplasm':
        img_name = f"{file_name}_{cell_id_str}.png"
        fldr += "single_nucleus/" if cell_nucleus_ids.size == 1 else "multiple_nucleus/" if cell_nucleus_ids.size > 1 else "no_nucleus/"
        fldr += "single_pathogen/" if cell_pathogen_ids.size == 1 else "multiple_pathogens/" if cell_pathogen_ids.size > 1 else "uninfected/"
    parts = file_name.split('_')
    plate = parts[0]
    well = parts[1] 
    
    if timelapse:
        #print("file_name:", file_name)
        #print("parts:", parts)
        timeID = parts[2]
        metadata = f'{plate}_{well}_{timeID}'
    else:
        metadata = f'{plate}_{well}'
        
    fldr = os.path.join(fldr,metadata)
    table_name = fldr.replace("/", "_")
    return img_name, fldr, table_name

def _find_bounding_box(crop_mask, _id, buffer=10):
    """Return a mask with the padded bounding box of ``_id`` filled with ``_id``."""
    object_indices = np.where(crop_mask == _id)

    # Determine the bounding box coordinates
    y_min, y_max = object_indices[0].min(), object_indices[0].max()
    x_min, x_max = object_indices[1].min(), object_indices[1].max()

    # Add buffer to the bounding box coordinates
    y_min = max(y_min - buffer, 0)
    y_max = min(y_max + buffer, crop_mask.shape[0] - 1)
    x_min = max(x_min - buffer, 0)
    x_max = min(x_max + buffer, crop_mask.shape[1] - 1)

    # Create a new mask with the same dimensions as crop_mask
    new_mask = np.zeros_like(crop_mask)

    # Fill in the bounding box area with the _id
    new_mask[y_min:y_max+1, x_min:x_max+1] = _id

    return new_mask
    
def _merge_and_save_to_database(morph_df, intensity_df, table_type, source_folder, file_name, experiment, timelapse=False):
        """Merge morphology and intensity DataFrames and append to the measurements SQLite DB."""
        morph_df = _check_integrity(morph_df)
        intensity_df = _check_integrity(intensity_df)
        if len(morph_df) > 0 and len(intensity_df) > 0:
            merged_df = pd.merge(morph_df, intensity_df, on='object_label', how='outer')
            merged_df = merged_df.rename(columns={"label_list_x": "label_list_morphology", "label_list_y": "label_list_intensity"})
            merged_df['file_name'] = file_name
            merged_df['path_name'] = os.path.join(source_folder, file_name + '.npy')
            if timelapse:
                merged_df[['plateID', 'rowID', 'columnID', 'fieldID', 'timeID', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(_map_wells(x, timelapse)))
            else:
                merged_df[['plateID', 'rowID', 'columnID', 'fieldID', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(_map_wells(x, timelapse)))
            cols = merged_df.columns.tolist()  # get the list of all columns
            if table_type == 'cell' or table_type == 'cytoplasm':
                column_list = ['object_label', 'plateID', 'rowID', 'columnID', 'fieldID', 'prcf', 'file_name', 'path_name']
            elif table_type == 'nucleus' or table_type == 'pathogen':
                column_list = ['object_label', 'cell_id', 'plateID', 'rowID', 'columnID', 'fieldID', 'prcf', 'file_name', 'path_name']
            else:
                raise ValueError(f"Invalid table_type: {table_type}")
            # Check if all columns in column_list are in cols
            missing_columns = [col for col in column_list if col not in cols]
            if len(missing_columns) == 1 and missing_columns[0] == 'cell_id':
                missing_columns = False
                column_list = ['object_label', 'plateID', 'rowID', 'columnID', 'fieldID', 'prcf', 'file_name', 'path_name']
            if missing_columns:
                raise ValueError(f"Columns missing in DataFrame: {missing_columns}")
            for i, col in enumerate(column_list):
                cols.insert(i, cols.pop(cols.index(col)))
            merged_df = merged_df[cols]  # rearrange the columns
            if len(merged_df) > 0:
                try:
                    conn = sqlite3.connect(f'{source_folder}/measurements/measurements.db', timeout=5)
                    merged_df.to_sql(table_type, conn, if_exists='append', index=False)
                except sqlite3.OperationalError as e:
                    print("SQLite error:", e)
                    
def _safe_int_convert(value, default=0):
    """Return ``int(value)`` on success, otherwise ``default``."""
    try:
        return int(value)
    except ValueError:
        print(f'Could not convert {value} to int using {default}', end='\r', flush=True)
        return default

def _map_wells(file_name, timelapse=False):
    """Parse a stack file name into ``(plate, row, column, field[, timeid], prcf)``."""
    try:
        parts = file_name.split('_')
        plate = parts[0]
        #plate = 'p' + parts[0]
        well = parts[1]
        field = 'f' + str(_safe_int_convert(parts[2]))
        if timelapse:
            timeid = 't' + str(_safe_int_convert(parts[3]))
        if well[0].isalpha():
            row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
            column = 'c' + str(int(well[1:]))
        else:
            row, column = well, well
        if timelapse:    
            prcf = '_'.join([plate, row, column, field, timeid])
        else:
            prcf = '_'.join([plate, row, column, field])
    except Exception as e:
        print(f"Error processing filename: {file_name}")
        print(f"Error: {e}")
        plate, row, column, field, timeid, prcf = 'error','error','error','error','error', 'error'
    if timelapse:
        return plate, row, column, field, timeid, prcf
    else:
        return plate, row, column, field, prcf

def _map_wells_png(file_name, timelapse=False):
    """Parse a cropped-object PNG file name into well identifiers plus ``prcfo`` and object id."""
    try:
        root, ext = os.path.splitext(file_name)
        parts = root.split('_')
        #plate = 'p' + parts[0]
        plate = parts[0]
        well = parts[1]
        field = 'f' + str(_safe_int_convert(parts[2]))
        if timelapse:
            timeid = 't' + str(_safe_int_convert(parts[3]))
        object_id = 'o' + str(_safe_int_convert(parts[-1], default='none'))
        if well[0].isalpha():
            row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
            column = 'c' + str(_safe_int_convert(well[1:]))
        else:
            row, column = well, well
        if timelapse:
            prcfo = '_'.join([plate, row, column, field, timeid, object_id])
        else:
            prcfo = '_'.join([plate, row, column, field, object_id])
    except Exception as e:
        print(f"Error processing filename: {file_name}")
        print(f"Error: {e}")
        plate, row, column, field, object_id, prcfo = 'error', 'error', 'error', 'error', 'error', 'error'
    if timelapse:
        return plate, row, column, field, timeid, prcfo, object_id
    else:
        return plate, row, column, field, prcfo, object_id
        
def _check_integrity(df):
    """Deduplicate label columns and collapse them into ``label_list``/``object_label``."""
    df.columns = [col + f'_{i}' if df.columns.tolist().count(col) > 1 and i != 0 else col for i, col in enumerate(df.columns)]
    label_cols = [col for col in df.columns if 'label' in col]
    df['label_list'] = df[label_cols].values.tolist()
    df['object_label'] = df['label_list'].apply(lambda x: x[0])
    df = df.drop(columns=label_cols)
    df['label_list'] = df['label_list'].astype(str)
    return df
    
def _get_percentiles(array, p1=2, p2=98):
    """Return per-channel ``[p1, p2]`` percentiles from nonzero pixels of an image stack."""
    nimg = array.shape[2]
    percentiles = []
    for v in range(nimg):
        img = np.squeeze(array[:, :, v])
        non_zero_img = img[img > 0]
        if non_zero_img.size > 0: # check if there are non-zero values
            img_min = np.percentile(non_zero_img, p1)  # change percentile from 0.02 to 2
            img_max = np.percentile(non_zero_img, p2)  # change percentile from 0.98 to 98
            percentiles.append([img_min, img_max])
        else:  # if there are no non-zero values, just use the image as it is
            img_min = np.percentile(img, p1)  # change percentile from 0.02 to 2
            img_max = np.percentile(img, p2)  # change percentile from 0.98 to 98
            percentiles.append([img_min, img_max])
    return percentiles

def _crop_center(img, cell_mask, new_width, new_height):
    """Crop ``img`` to ``new_width`` x ``new_height`` centered on the mask centroid."""
    # Convert all non-zero values in mask to 1
    cell_mask[cell_mask != 0] = 1
    mask_3d = np.repeat(cell_mask[:, :, np.newaxis], img.shape[2], axis=2).astype(img.dtype) # Create 3D mask
    img = np.multiply(img, mask_3d).astype(img.dtype) # Multiply image with mask to set pixel values outside of the mask to 0
    centroid = np.round(ndi.center_of_mass(cell_mask)).astype(int) # Compute centroid of the mask
    
    # Pad the image and mask to ensure the crop will not go out of bounds
    pad_width = max(new_width, new_height)
    img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant')
    cell_mask = np.pad(cell_mask, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')
    
    # Update centroid coordinates due to padding
    centroid += pad_width
    
    # Compute bounding box
    start_y = max(0, centroid[0] - new_height // 2)
    end_y = min(start_y + new_height, img.shape[0])
    start_x = max(0, centroid[1] - new_width // 2)
    end_x = min(start_x + new_width, img.shape[1])
    
    # Crop to bounding box
    img = img[start_y:end_y, start_x:end_x, :]
    return img
    
def _masks_to_masks_stack(masks):
    """Return ``masks`` as a plain list preserving iteration order."""
    mask_stack = []
    for idx, mask in enumerate(masks):
        mask_stack.append(mask)
    return mask_stack

def _get_diam(mag, obj):

    if obj == 'cell':
        diameter = 2 * mag + 80
    elif obj == 'cell_large':
        diameter = 2 * mag + 120
    elif obj == 'nucleus':
        diameter = 0.75 * mag + 45
    elif obj == 'pathogen':
        diameter = mag
    else:
        # Guard against unsupported object types — previously this fell
        # through to ``int(diameter)`` with ``diameter`` unbound, raising a
        # confusing UnboundLocalError instead of a clear message.
        raise ValueError(
            f"_get_diam: unsupported object type '{obj}'. "
            f"Expected one of: cell, cell_large, nucleus, pathogen."
        )

    return int(diameter)

def _get_object_settings(object_type, settings):
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

    elif object_type == 'nucleus':
        object_settings['model_name'] = 'nuclei'
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['restore_type'] = settings.get('nucleus_restore_type', None)

    elif object_type == 'pathogen':
        object_settings['model_name'] = 'cyto'
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['resample'] = False
        object_settings['restore_type'] = settings.get('pathogen_restore_type', None)
        object_settings['merge'] = settings['merge_pathogens']
        
    else:
        print(f'Object type: {object_type} not supported. Supported object types are : cell, nucleus and pathogen')

    if settings['verbose']:
        print(object_settings)
        
    return object_settings
    
def _pivot_counts_table(db_path):

    def _read_table_to_dataframe(db_path, table_name='object_counts'):
        """Return the given SQLite table as a DataFrame."""
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        # Read the entire table into a pandas DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        # Close the connection
        conn.close()
        return df

    def _pivot_dataframe(df):
        """Pivot count-type rows into one column per object type, NaNs filled with 0."""
        # Pivot the DataFrame
        pivoted_df = df.pivot(index='file_name', columns='count_type', values='object_count').reset_index()
        # Because the pivot operation can introduce NaN values for missing data,
        # you might want to fill those NaNs with a default value, like 0
        pivoted_df = pivoted_df.fillna(0)
        return pivoted_df

    # Read the original 'object_counts' table
    df = _read_table_to_dataframe(db_path, 'object_counts')
    # Pivot the DataFrame to have one row per filename and a column for each object type
    pivoted_df = _pivot_dataframe(df)
    # Reconnect to the SQLite database to overwrite the 'object_counts' table with the pivoted DataFrame
    conn = sqlite3.connect(db_path)
    # When overwriting, ensure that you drop the existing table or use if_exists='replace' to overwrite it
    pivoted_df.to_sql('pivoted_counts', conn, if_exists='replace', index=False)
    conn.close()
    
def _get_cellpose_channels(settings):
    """Return the channel indices to extract and the per-object-type Cellpose channel remap."""
    nucleus_ch = settings.get('cellpose_nucleus_channel')
    cell_ch = settings.get('cellpose_cell_channel')
    pathogen_ch = settings.get('cellpose_pathogen_channel')
    organelle_ch = settings.get('cellpose_organelle_channel')

    all_channels = set()
    for ch in [nucleus_ch, cell_ch, pathogen_ch, organelle_ch]:
        if ch is not None:
            all_channels.add(ch)

    channels_to_extract = sorted(all_channels)
    remap = {orig: new for new, orig in enumerate(channels_to_extract)}

    cellpose_channels = {}

    if nucleus_ch is not None:
        cellpose_channels['nucleus'] = [remap[nucleus_ch]]

    if cell_ch is not None:
        if nucleus_ch is not None:
            cellpose_channels['cell'] = [remap[cell_ch], remap[nucleus_ch]]
        else:
            cellpose_channels['cell'] = [remap[cell_ch]]

    if pathogen_ch is not None:
        cellpose_channels['pathogen'] = [remap[pathogen_ch]]

    if organelle_ch is not None:
        cellpose_channels['organelle'] = [remap[organelle_ch]]

    return channels_to_extract, cellpose_channels
    

def annotate_conditions(df, cells=None, cell_loc=None, pathogens=None, pathogen_loc=None, treatments=None, treatment_loc=None):
    """Annotate ``df`` with host cell, pathogen, treatment, and combined ``condition`` columns.

    :param df: DataFrame to annotate; must contain ``rowID``/``columnID``.
    :param cells: host cell types (str or list).
    :param cell_loc: per-cell-type list-of-lists of row/column identifiers.
    :param pathogens: pathogens (str or list).
    :param pathogen_loc: per-pathogen list-of-lists of row/column identifiers.
    :param treatments: treatments (str or list).
    :param treatment_loc: per-treatment list-of-lists of row/column identifiers.
    :returns: annotated DataFrame with ``host_cells``, ``pathogen``, ``treatment``, ``condition`` columns.
    """
    
    def _get_type(val):
        """Determine if a value maps to 'rowID' or 'columnID'."""
        if isinstance(val, str) and val.startswith('c'):
            return 'columnID'
        elif isinstance(val, str) and val.startswith('r'):
            return 'rowID'
        return None

    def _map_or_default(column_name, values, loc, df):
        """Assign or map ``values`` into ``column_name`` based on optional row/column ``loc``."""
        if isinstance(values, str) and loc is None:
            # If a single string is provided and loc is None, assign the value to all rows
            df[column_name] = values  
    
        elif isinstance(values, list) and loc is None:
            # If a list of values is provided but no loc, assign the first value to all rows
            df[column_name] = values[0]
    
        elif values is not None and loc is not None:
            # Perform location-based mapping
            value_dict = {val: key for key, loc_list in zip(values, loc) for val in loc_list}
            df[column_name] = np.nan  # Start with NaN
            for val, key in value_dict.items():
                loc_type = _get_type(val)
                if loc_type:
                    df.loc[df[loc_type] == val, column_name] = key

    # Handle cells, pathogens, and treatments using the consolidated logic
    _map_or_default('host_cells', cells, cell_loc, df)
    _map_or_default('pathogen', pathogens, pathogen_loc, df)
    _map_or_default('treatment', treatments, treatment_loc, df)

    # Conditionally fill NaN for pathogen and treatment columns if applicable
    if pathogens is not None:
        df['pathogen'].fillna(np.nan, inplace=True)
    if treatments is not None:
        df['treatment'].fillna(np.nan, inplace=True)

    # Create the 'condition' column by excluding any NaN values, safely checking if 'host_cells', 'pathogen', and 'treatment' exist
    df['condition'] = df.apply(
        lambda x: '_'.join([str(v) for v in [x.get('host_cells'), x.get('pathogen'), x.get('treatment')] if pd.notna(v)]), 
        axis=1
    )
    df.loc[df['condition'] == '', 'condition'] = pd.NA

    return df

def _split_data(df, group_by, object_type):
    """Group numeric and non-numeric columns of ``df`` separately with per-column aggregation."""

    df = df.copy()

    # Ensure 'prcft' column exists if timeID is present
    try:
        df['prcft'] = (
            df['plateID'].astype(str) + '_' +
            df['rowID'].astype(str) + '_' +
            df['columnID'].astype(str) + '_' +
            df['fieldID'].astype(str) + '_' +
            df['timeID'].astype(str)
        )
    except Exception as e:
        print('Exception', e)

    # Ensure 'prcf' column exists
    try:
        df['prcf'] = (
            df['plateID'].astype(str) + '_' +
            df['rowID'].astype(str) + '_' +
            df['columnID'].astype(str) + '_' +
            df['fieldID'].astype(str)
        )
    except Exception as e:
        print('Exception', e)

    # Create the 'prcfo' column
    df['prcfo'] = df['prcf'].astype(str) + '_' + df[object_type].astype(str)
    df = df.set_index(group_by, inplace=False)

    # Split the DataFrame into numeric and non-numeric parts
    df_numeric = df.select_dtypes(include=np.number)
    df_non_numeric = df.select_dtypes(exclude=np.number)

    # Define keywords for columns to be summed instead of averaged
    sum_keywords = [
        'area',
        'perimeter',
        'convex_area',
        'bbox_area',
        'filled_area',
        'major_axis_length',
        'minor_axis_length',
        'equivalent_diameter'
    ]

    # Create a dictionary for custom aggregation
    agg_dict = {}
    for column in df_numeric.columns:
        if any(keyword in column for keyword in sum_keywords):
            agg_dict[column] = 'sum'
        else:
            agg_dict[column] = 'mean'

    # Apply custom aggregation
    if len(agg_dict) > 0 and not df_numeric.empty:
        grouped_numeric = df_numeric.groupby(df_numeric.index).agg(agg_dict)
    else:
        grouped_numeric = pd.DataFrame(index=df.index.unique())

    if not df_non_numeric.empty:
        grouped_non_numeric = df_non_numeric.groupby(df_non_numeric.index).first()
    else:
        grouped_non_numeric = pd.DataFrame(index=df.index.unique())

    return pd.DataFrame(grouped_numeric), pd.DataFrame(grouped_non_numeric)

    
def _calculate_recruitment(df, channel):
    """Add pathogen-to-compartment recruitment ratio columns for the given intensity channel."""
    df['pathogen_cell_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'nucleus_channel_{channel}_mean_intensity']

    df['pathogen_cell_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_cytoplasm_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_nucleus_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'nucleus_channel_{channel}_mean_intensity']

    df['pathogen_outside_cell_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_outside_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_outside_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'nucleus_channel_{channel}_mean_intensity']

    df['pathogen_outside_cell_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_outside_cytoplasm_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_outside_nucleus_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'nucleus_channel_{channel}_mean_intensity']

    df['pathogen_periphery_cell_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'cell_channel_{channel}_mean_intensity']
    df['pathogen_periphery_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
    df['pathogen_periphery_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'nucleus_channel_{channel}_mean_intensity']

    channels = [0,1,2,3]
    object_type = 'pathogen'
    for chan in channels:
        df[f'{object_type}_slope_channel_{chan}'] = 1

    object_type = 'nucleus'
    for chan in channels:
        df[f'{object_type}_slope_channel_{chan}'] = 1

    #for chan in channels:
    #    df[f'nucleus_coordinates_{chan}'] = df[[f'nucleus_channel_{chan}_centroid_weighted_local-0', f'nucleus_channel_{chan}_centroid_weighted_local-1']].values.tolist()
    #    df[f'pathogen_coordinates_{chan}'] = df[[f'pathogen_channel_{chan}_centroid_weighted_local-0', f'pathogen_channel_{chan}_centroid_weighted_local-1']].values.tolist()
    #    df[f'cell_coordinates_{chan}'] = df[[f'cell_channel_{chan}_centroid_weighted_local-0', f'cell_channel_{chan}_centroid_weighted_local-1']].values.tolist()
    #    df[f'cytoplasm_coordinates_{chan}'] = df[[f'cytoplasm_channel_{chan}_centroid_weighted_local-0', f'cytoplasm_channel_{chan}_centroid_weighted_local-1']].values.tolist()
    # 
    #    df[f'pathogen_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'pathogen_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
    #                                                  (row[f'pathogen_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
    #    df[f'nucleus_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'nucleus_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
    #                                                  (row[f'nucleus_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
    return df
    
def _group_by_well(df):
    """
    Group the DataFrame by well coordinates (plate, row, col) and apply mean function to numeric columns
    and select the first value for non-numeric columns.

    Parameters:
    df (DataFrame): The input DataFrame to be grouped.

    Returns:
    DataFrame: The grouped DataFrame.
    """
    numeric_cols = df._get_numeric_data().columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns

    # Apply mean function to numeric columns and first to non-numeric
    df_grouped = df.groupby(['plateID', 'rowID', 'columnID']).agg({**{col: np.mean for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}})
    return df_grouped

###################################################
#  Classify
###################################################

class Cache:
    """LRU cache with a fixed maximum size.

    :param max_size: maximum number of entries retained; oldest is evicted on overflow.
    """

    def __init__(self, max_size):
        """Store the size limit and initialize an empty ``OrderedDict``."""
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        """Return the cached value for ``key`` and mark it most-recently-used, or ``None``."""
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        """Insert ``value`` under ``key``, evicting the least-recently-used entry if full."""
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention layer.

    :param d_k: dimensionality of key/query vectors used in the scaling factor.
    """
    def __init__(self, d_k):
        """Store ``d_k`` used to scale attention logits."""
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        """Return ``softmax(QK^T / sqrt(d_k)) V``.

        :param Q: query tensor.
        :param K: key tensor.
        :param V: value tensor.
        :returns: attention-weighted value tensor.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

class SelfAttention(nn.Module):
    """Linear-projected self-attention layer.

    :param in_channels: input feature dimension.
    :param d_k: projected key/query/value dimension.
    """
    def __init__(self, in_channels, d_k):
        """Build the Q/K/V projections and the underlying attention layer."""
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, d_k)
        self.W_k = nn.Linear(in_channels, d_k)
        self.W_v = nn.Linear(in_channels, d_k)
        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, x):
        """Return self-attention over ``x`` of shape ``(B, in_channels)``."""
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output = self.attention(Q, K, V)
        return output

# Early Fusion Block
class EarlyFusion(nn.Module):
    """1x1 convolution that fuses input channels down to 64 feature maps.

    :param in_channels: number of input channels.
    """
    def __init__(self, in_channels):
        """Create the 1x1 fusion convolution."""
        super(EarlyFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)

    def forward(self, x):
        """Return the 64-channel fused feature map."""
        x = self.conv1(x)
        return x

# Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    """Spatial attention gate that reweights features by pooled channel statistics.

    :param kernel_size: convolution kernel width used to fuse average+max pooled maps.
    """
    def __init__(self, kernel_size=7):
        """Build the fusion convolution and sigmoid gate."""
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return the spatial attention map for ``x`` in ``[0, 1]``."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
# Multi-Scale Block with Attention
class MultiScaleBlockWithAttention(nn.Module):
    """Dilated conv block followed by a 1x1 attention convolution.

    :param in_channels: input channel count.
    :param out_channels: output channel count.
    """
    def __init__(self, in_channels, out_channels):
        """Build the dilated convolution and 1x1 spatial-attention convolution."""
        super(MultiScaleBlockWithAttention, self).__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.spatial_attention = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def custom_forward(self, x):
        """Apply dilated conv + ReLU followed by the 1x1 spatial attention."""
        x1 = F.relu(self.dilated_conv1(x), inplace=True)
        x = self.spatial_attention(x1)
        return x

    def forward(self, x):
        """Forward pass; delegates to :meth:`custom_forward`."""
        return self.custom_forward(x)

# Final Classifier
class CustomCellClassifier(nn.Module):
    """Small classifier stacking :class:`EarlyFusion` and a multi-scale attention block.

    :param num_classes: output class count.
    :param pathogen_channel: reserved for downstream use; kept for API compatibility.
    :param use_attention: reserved for downstream use; kept for API compatibility.
    :param use_checkpoint: run the forward pass through ``torch.utils.checkpoint``.
    :param dropout_rate: reserved for downstream use; kept for API compatibility.
    """
    def __init__(self, num_classes, pathogen_channel, use_attention, use_checkpoint, dropout_rate):
        """Build the fusion, multi-scale, and linear classifier submodules."""
        super(CustomCellClassifier, self).__init__()
        self.early_fusion = EarlyFusion(in_channels=3)

        self.multi_scale_block_1 = MultiScaleBlockWithAttention(in_channels=64, out_channels=64)

        self.fc1 = nn.Linear(64, num_classes)
        self.use_checkpoint = use_checkpoint
        # Explicitly require gradients for all parameters
        for param in self.parameters():
            param.requires_grad = True

    def custom_forward(self, x):
        """Return the class logits for a batch ``x`` of shape ``(B, 3, H, W)``."""
        x.requires_grad = True
        x = self.early_fusion(x)
        x = self.multi_scale_block_1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        return x

    def forward(self, x):
        """Forward pass, optionally through activation checkpointing."""
        if self.use_checkpoint:
            x.requires_grad = True
            return checkpoint(self.custom_forward, x)
        else:
            return self.custom_forward(x)

class TorchModel(nn.Module):
    """
    Thin wrapper around TorchVision classification backbones that:
      1) Loads a requested backbone with (optional) pretrained weights
      2) Strips its classification head to expose features
      3) Adds a simple Linear 'spacr' classifier with `num_classes` outputs
      4) Optionally applies dropout before the final classifier
      5) Supports gradient checkpointing
    Works with most TorchVision **classification** models. Non-classification
    (detection/segmentation) models are rejected with a clear error.
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: Optional[float] = None,
        use_checkpoint: bool = False,
        num_classes: int = 2,      # >=2 => multiclass head; ==1 => binary head (BCE)
        multilabel: bool = False   # kept for external loss/metrics decisions
    ):
        """Build the backbone, strip its head, and attach the SPACR linear classifier.

        :param model_name: TorchVision classification model to load.
        :param pretrained: use ImageNet-pretrained weights when available.
        :param dropout_rate: dropout probability applied to backbone and SPACR head; ``None`` disables.
        :param use_checkpoint: enable gradient checkpointing through the backbone.
        :param num_classes: output class count; ``1`` yields a BCE-style binary head.
        :param multilabel: informational flag consumed by external loss/metrics code.
        :raises ValueError: if ``model_name`` is not a TorchVision model.
        """
        super().__init__()
        self.model_name = str(model_name)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_classes = int(num_classes)
        self.multilabel = bool(multilabel)
        self.use_dropout = (dropout_rate is not None)

        # 1) Initialize backbone
        self.base_model = self._init_base_model(pretrained=bool(pretrained))

        # 2) Special-case: keep all but last linear block for MaxViT-T
        if self.model_name == "maxvit_t" and hasattr(self.base_model, "classifier"):
            # remove final Linear only (keep preceding norm/dropout/etc.)
            seq = list(self.base_model.classifier.children())
            if len(seq) > 0:
                self.base_model.classifier = nn.Sequential(*seq[:-1])

        # 3) If a custom dropout rate is provided, push it into any existing Dropout modules
        if dropout_rate is not None:
            self._apply_dropout_rate(self.base_model, float(dropout_rate))

        # 4) Remove the original classification head so we can infer feature dim
        self._remove_head_for_features()

        # 5) Infer flattened feature dimension with a dummy forward
        self.num_ftrs = self._infer_feature_dim()

        # 6) Build SPACR head (optional dropout + linear classifier)
        if self.use_dropout:
            self.dropout = nn.Dropout(float(dropout_rate))
        self.spacr_classifier = nn.Linear(self.num_ftrs, self.num_classes)

    # ------------------------------------------------------------------ #
    # Backbone init / head removal / feature dim
    # ------------------------------------------------------------------ #
    def _get_weight_choice(self):
        """
        Return the DEFAULT weights enum if available (newer torchvision),
        otherwise None to fall back to legacy pretrained=True/False.
        """
        enum_attr = f"{self.model_name}_weights"
        for attr in dir(models):
            if attr.lower() == enum_attr.lower():
                enum = getattr(models, attr, None)
                if enum is not None and hasattr(enum, "DEFAULT"):
                    return enum.DEFAULT
        return None

    def _init_base_model(self, pretrained: bool) -> nn.Module:
        fn = models.__dict__.get(self.model_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown torchvision model: {self.model_name}")

        weights = self._get_weight_choice()
        if weights is not None:
            # Newer API
            return fn(weights=weights if pretrained else None)
        else:
            # Older API fallback
            return fn(pretrained=pretrained)

    def _apply_dropout_rate(self, module: nn.Module, p: float):
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.p = p

    def _remove_head_for_features(self):
        """
        Normalize a wide swath of TorchVision classification heads to Identity.
        Also disable auxiliary logits where present (Inception/GoogLeNet).
        """
        # Some models (Inception/GoogLeNet) expose aux heads
        if hasattr(self.base_model, "aux_logits"):
            self.base_model.aux_logits = False

        # Common conv backbones
        if hasattr(self.base_model, "fc"):           # ResNet/RegNet/ResNeXt/GoogLeNet/Inception
            self.base_model.fc = nn.Identity()
            return
        if hasattr(self.base_model, "classifier"):   # DenseNet/MobileNet/EfficientNet/ConvNeXt/SqueezeNet/MNASNet/MaxViT
            # MaxViT handled earlier; here we blank the whole thing
            if self.model_name != "maxvit_t":
                self.base_model.classifier = nn.Identity()
            return
        if hasattr(self.base_model, "_fc"):          # Older EfficientNet
            self.base_model._fc = nn.Identity()
            return
        # Vision Transformers
        if hasattr(self.base_model, "heads"):        # ViT (torchvision)
            self.base_model.heads = nn.Identity()
            return
        if hasattr(self.base_model, "head"):         # Swin
            self.base_model.head = nn.Identity()
            return
        # If none matched, we’ll still try to forward and flatten later.

    def _infer_feature_dim(self) -> int:
        """
        Forward a dummy tensor through the backbone and determine the flattened
        feature size. Uses 224×224 nominal resolution.
        """
        self.base_model.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            out = self._run_backbone_raw(x)  # raw backbone call (unwrapped)
        # Flatten if spatial
        if isinstance(out, torch.Tensor) and out.ndim > 2:
            out = torch.flatten(out, 1)
        if not isinstance(out, torch.Tensor) or out.ndim != 2:
            raise RuntimeError(
                f"Backbone produced unexpected shape/type for features: {type(out)} / {getattr(out, 'shape', None)}"
            )
        return int(out.size(1))

    # ------------------------------------------------------------------ #
    # Forward plumbing
    # ------------------------------------------------------------------ #
    def _run_backbone_raw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call the underlying backbone and unwrap common container outputs.
        Does NOT apply the new SPACR head.
        """
        def forward_fn(t):
            """Run the underlying backbone on ``t`` (used as the checkpoint target)."""
            return self.base_model(t)

        out = checkpoint(forward_fn, x) if self.use_checkpoint else forward_fn(x)

        # Unwrap common container types
        # Inception* returns namedtuple with .logits (if aux disabled we still may get a container)
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            # e.g., some models return (logits, aux) even when aux disabled; take primary
            out = out[0]
        elif isinstance(out, dict):
            # Detection/segmentation heads return dicts — not supported in this wrapper
            raise RuntimeError(
                "Selected backbone returned a dict (likely detection/segmentation). "
                "Use an image-classification backbone."
            )
        return out

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = self._run_backbone_raw(x)
        # Ensure 2D features (N, F)
        if isinstance(out, torch.Tensor) and out.ndim > 2:
            out = torch.flatten(out, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits of shape ``(N, num_classes)`` for input batch ``x``."""
        feats = self._run_backbone(x)
        if self.use_dropout:
            feats = self.dropout(feats)
        logits = self.spacr_classifier(feats)  # (N, num_classes)
        return logits

class TorchModel_v2(nn.Module):
    """TorchVision backbone with a SPACR linear head (streamlined variant of :class:`TorchModel`).

    :param model_name: TorchVision classification model to load.
    :param pretrained: use ImageNet-pretrained weights when available.
    :param dropout_rate: dropout probability applied to backbone and SPACR head; ``None`` disables.
    :param use_checkpoint: enable gradient checkpointing through the backbone.
    :param num_classes: output class count.
    :param multilabel: informational flag consumed by external loss/metrics code.
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = None,
        use_checkpoint: bool = False,
        num_classes: int = 2,          # arbitrary classes (>=2 => multiclass; 1 => binary head)
        multilabel: bool = False       # kept for external loss/metrics decisions (not used internally)
    ):
        """Build the backbone, strip its head, and attach the SPACR classifier."""
        super().__init__()
        self.model_name = model_name
        self.use_checkpoint = bool(use_checkpoint)
        self.num_classes = int(num_classes)
        self.multilabel = bool(multilabel)

        # 1) init backbone
        self.base_model = self._init_base_model(pretrained)

        # 2) special-case: keep all but the last linear block for maxvit_t
        if self.model_name == "maxvit_t" and hasattr(self.base_model, "classifier"):
            self.base_model.classifier = nn.Sequential(
                *list(self.base_model.classifier.children())[:-1]
            )

        # 3) apply custom dropout rate to any existing dropout modules in backbone
        if dropout_rate is not None:
            self._apply_dropout_rate(self.base_model, float(dropout_rate))

        # 4) discover feature dim
        self.num_ftrs = self._infer_feature_dim()

        # 5) add SPACR head
        self._init_spacr_classifier(dropout_rate)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _apply_dropout_rate(self, module: nn.Module, p: float):
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.p = p

    def _init_base_model(self, pretrained: bool) -> nn.Module:
        fn = models.__dict__.get(self.model_name, None)
        if fn is None:
            raise ValueError(f"Unknown torchvision model: {self.model_name}")

        weights = self._get_weight_choice()
        if weights is not None:
            # Newer torchvision API: weights=enum or None
            return fn(weights=weights if pretrained else None)
        else:
            # Older API fallback: pretrained=bool
            return fn(pretrained=bool(pretrained))

    def _get_weight_choice(self):
        # Return DEFAULT weights enum if available; else None
        for attr in dir(models):
            if attr.lower() == f"{self.model_name}_weights":
                return getattr(models, attr).DEFAULT
        return None

    def _remove_head_for_features(self):
        # Remove final classifier so backbone returns features
        if hasattr(self.base_model, "fc"):
            self.base_model.fc = nn.Identity()
        elif hasattr(self.base_model, "classifier"):
            if self.model_name != "maxvit_t":
                self.base_model.classifier = nn.Identity()

    def _infer_feature_dim(self) -> int:
        self._remove_head_for_features()
        self.base_model.eval()
        with torch.no_grad():
            out = self.base_model(torch.randn(1, 3, 224, 224))
        # If backbone returns spatial map, flatten to (N, C*)
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        return int(out.size(1))

    def _init_spacr_classifier(self, dropout_rate: float):
        self.use_dropout = dropout_rate is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(float(dropout_rate))
        self.spacr_classifier = nn.Linear(self.num_ftrs, self.num_classes)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # Wrap for checkpoint (expects a function)
        if self.use_checkpoint:
            return checkpoint(lambda t: self.base_model(t), x)
        return self.base_model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits of shape ``(N, num_classes)`` for input batch ``x``."""
        feats = self._run_backbone(x)
        # Ensure 2D features (N, F)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        if self.use_dropout:
            feats = self.dropout(feats)
        logits = self.spacr_classifier(feats)  # (N, C) where C==num_classes
        return logits

class FocalLossWithLogits(nn.Module):
    """Focal loss for binary, multiclass, and multilabel targets.

    Auto-selects the BCE or cross-entropy branch based on the shapes of
    ``logits`` and ``target``:
      - binary: logits ``(N,)`` or ``(N,1)``; target float ``(N,)`` in ``{0,1}``.
      - multiclass: logits ``(N,C)``; target long ``(N,)`` in ``[0..C-1]``.
      - multilabel: logits ``(N,C)``; target float ``(N,C)`` in ``{0,1}``.

    :param alpha: class-balancing factor (float or 1-D tensor of shape ``(C,)``).
    :param gamma: focusing parameter.
    :param reduction: one of ``'mean'``, ``'sum'``, ``'none'``.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """Store the focal-loss hyperparameters."""
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, target):
        """Return the focal loss value for the chosen ``reduction`` mode."""
        # Binary / multilabel (BCE-style)
        if logits.ndim == 1 or logits.size(-1) == 1 or (
            logits.ndim == 2 and target.ndim == 2 and target.size(1) == logits.size(1)
        ):
            logits = logits.view_as(target)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p = torch.sigmoid(logits)
            pt = target * p + (1 - target) * (1 - p)  # pt = p if y=1 else (1-p)
            loss = (self.alpha * (1 - pt).pow(self.gamma) * bce)
        else:
            # Multiclass CE-style: logits (N,C), target (N,) long
            if target.dtype != torch.long:
                target = target.long()
            logp = F.log_softmax(logits, dim=1)              # (N,C)
            p = torch.exp(logp)                              # (N,C)
            # gather the prob of the true class
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
            ce = F.nll_loss(logp, target, reduction="none")   # per-sample CE
            if isinstance(self.alpha, torch.Tensor):
                # class-wise alpha
                alpha = self.alpha.to(logits.device)[target]   # (N,)
            else:
                alpha = float(self.alpha)
            loss = alpha * (1 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    
class ResNet(nn.Module):
    """ResNet backbone with a two-layer SPACR binary-classification head.

    :param resnet_type: one of ``'resnet18'``/``'resnet34'``/``'resnet50'``/``'resnet101'``/``'resnet152'``.
    :param dropout_rate: dropout probability before the final linear layer; ``None`` disables.
    :param use_checkpoint: enable gradient checkpointing through the ResNet backbone.
    :param init_weights: ``'imagenet'`` for pretrained weights or ``'none'`` for random init.
    """
    def __init__(self, resnet_type='resnet50', dropout_rate=None, use_checkpoint=False, init_weights='imagenet'):
        """Select the backbone and delegate head construction to :meth:`initialize_base`."""
        super(ResNet, self).__init__()

        resnet_map = {
            'resnet18': {'func': models.resnet18, 'weights': ResNet18_Weights.IMAGENET1K_V1},
            'resnet34': {'func': models.resnet34, 'weights': ResNet34_Weights.IMAGENET1K_V1},
            'resnet50': {'func': models.resnet50, 'weights': ResNet50_Weights.IMAGENET1K_V1},
            'resnet101': {'func': models.resnet101, 'weights': ResNet101_Weights.IMAGENET1K_V1},
            'resnet152': {'func': models.resnet152, 'weights': ResNet152_Weights.IMAGENET1K_V1}
        }

        if resnet_type not in resnet_map:
            raise ValueError(f"Invalid resnet_type. Choose from {list(resnet_map.keys())}")

        self.initialize_base(resnet_map[resnet_type], dropout_rate, use_checkpoint, init_weights)

    def initialize_base(self, base_model_dict, dropout_rate, use_checkpoint, init_weights):
        """Build the backbone (with or without pretrained weights) and the two-layer head.

        :param base_model_dict: dict with keys ``func`` (model constructor) and ``weights``.
        :param dropout_rate: dropout probability applied between the two linear layers.
        :param use_checkpoint: enable gradient checkpointing through the backbone.
        :param init_weights: ``'imagenet'`` or ``'none'``.
        :raises ValueError: if ``init_weights`` is neither ``'imagenet'`` nor ``'none'``.
        """
        if init_weights == 'imagenet':
            self.resnet = base_model_dict['func'](weights=base_model_dict['weights'])
        elif init_weights == 'none':
            self.resnet = base_model_dict['func'](weights=None)
        else:
            raise ValueError("init_weights should be either 'imagenet' or 'none'")

        self.fc1 = nn.Linear(1000, 500)
        self.use_dropout = dropout_rate != None
        self.use_checkpoint = use_checkpoint

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        """Return the flattened single-logit prediction for input batch ``x``."""
        x.requires_grad = True  # Ensure that the tensor has requires_grad set to True

        if self.use_checkpoint:
            x = checkpoint(self.resnet, x)  # Use checkpointing for just the ResNet part
        else:
            x = self.resnet(x)

        x = F.relu(self.fc1(x))

        if self.use_dropout:
            x = self.dropout(x)

        logits = self.fc2(x).flatten()
        return logits

def split_my_dataset(dataset, split_ratio=0.1):
    """Randomly split ``dataset`` into ``(train, val)`` subsets.

    :param dataset: source dataset.
    :param split_ratio: fraction of samples reserved for validation.
    :returns: ``(train_subset, val_subset)``.
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split_idx = int((1 - split_ratio) * num_samples)
    random.shuffle(indices)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

def classification_metrics(all_labels, prediction_pos_probs, loss, epoch):
    """Return a one-row DataFrame of accuracy, PR-AUC, and optimal-threshold stats.

    :param all_labels: ground-truth binary labels.
    :param prediction_pos_probs: predicted positive-class probabilities.
    :param loss: loss tensor for the epoch (``.item()`` is called).
    :param epoch: epoch number used as the row index.
    :returns: DataFrame indexed by epoch with accuracy, per-class accuracy, loss,
        PR-AUC, and optimal threshold columns.
    :raises ValueError: if ``all_labels`` and ``prediction_pos_probs`` have different lengths.
    """
    
    if len(all_labels) != len(prediction_pos_probs):
        raise ValueError(f"all_labels ({len(all_labels)}) and pred_labels ({len(prediction_pos_probs)}) have different lengths")
    
    unique_labels = np.unique(all_labels)
    if len(unique_labels) >= 2:
        pr_labels = np.array(all_labels).astype(int)
        precision, recall, thresholds = precision_recall_curve(pr_labels, prediction_pos_probs, pos_label=1)
        pr_auc = auc(recall, precision)
        thresholds = np.append(thresholds, 0.0)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.nanargmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        pred_labels = [int(p > 0.5) for p in prediction_pos_probs]
    if len(unique_labels) < 2:
        optimal_threshold = 0.5
        pred_labels = [int(p > optimal_threshold) for p in prediction_pos_probs]
        pr_auc = np.nan
    data = {'label': all_labels, 'pred': pred_labels}
    df = pd.DataFrame(data)
    pc_df = df[df['label'] == 1.0]
    nc_df = df[df['label'] == 0.0]
    correct = df[df['label'] == df['pred']]
    acc_all = len(correct) / len(df)
    if len(pc_df) > 0:
        correct_pc = pc_df[pc_df['label'] == pc_df['pred']]
        acc_pc = len(correct_pc) / len(pc_df)
    else:
        acc_pc = np.nan
    if len(nc_df) > 0:
        correct_nc = nc_df[nc_df['label'] == nc_df['pred']]
        acc_nc = len(correct_nc) / len(nc_df)
    else:
        acc_nc = np.nan
    data_dict = {'accuracy': acc_all, 'neg_accuracy': acc_nc, 'pos_accuracy': acc_pc, 'loss':loss.item(),'prauc':pr_auc, 'optimal_threshold':optimal_threshold}
    data_df = pd.DataFrame(data_dict, index=[str(epoch)]) 
    return data_df
    
def compute_irm_penalty(losses, dummy_w, device):
    """Return the IRM penalty as the sum of squared gradient dot-products across environments.

    :param losses: per-environment loss tensors.
    :param dummy_w: scalar dummy weight used for gradient computation.
    :param device: torch device on which to compute the penalty.
    :returns: scalar IRM penalty value.
    """
    weighted_losses = [loss.clone().detach().requires_grad_(True).to(device) * dummy_w for loss in losses]
    gradients = [grad(w_loss, dummy_w, create_graph=True)[0] for w_loss in weighted_losses]
    irm_penalty = 0.0
    for g1, g2 in combinations(gradients, 2):
        irm_penalty += (g1.dot(g2))**2
    return irm_penalty

#def print_model_summary(base_model, channels, height, width):
#    """
#    Prints the summary of a given base model.
#
#    Args:
#        base_model (torch.nn.Module): The base model to print the summary of.
#        channels (int): The number of input channels.
#        height (int): The height of the input.
#        width (int): The width of the input.
#
#    Returns:
#        None
#    """
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    base_model.to(device)
#    summary(base_model, (channels, height, width))
#    return

def _list_torchvision_model_names() -> set[str]:
    """Robustly collect available torchvision model factory names."""
    names: set[str] = set()
    # Newer API
    try:
        names |= set(tv_models.list_models(module=tv_models))
    except Exception:
        pass
    # Fallback for older torchvision
    for n, fn in tv_models.__dict__.items():
        if not n.startswith("_") and callable(fn):
            names.add(n)
    return names


def choose_model(model_type: str,
                 device: torch.device,
                 init_weights: bool = True,
                 dropout_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 channels: int = 3,
                 height: int = 224,
                 width: int = 224,
                 chan_dict: Optional[dict[str, Any]] = None,
                 num_classes: int = 2,
                 verbose: bool = False) -> Optional[nn.Module]:
    """Instantiate a classification model by name for binary or multiclass problems.

    :param model_type: TorchVision model name (e.g. ``'resnet50'``, ``'vit_b_16'``) or ``'custom'``.
    :param device: target device (the caller moves the returned model).
    :param init_weights: load pretrained weights when available.
    :param dropout_rate: dropout probability before the classifier head (``None``/``0`` disables).
    :param use_checkpoint: enable gradient checkpointing for the backbone.
    :param channels: input channel count (pretrained backbones assume 3).
    :param height: nominal input height used for a forward sanity check.
    :param width: nominal input width used for a forward sanity check.
    :param chan_dict: optional dict forwarded to a custom model builder.
    :param num_classes: output class count; ``1`` yields a single-logit BCE head.
    :param verbose: print the model structure when ``True``.

    :returns:
        nn.Module or None if invalid.
    """

    tv_names = _list_torchvision_model_names()
    valid_names = set(tv_names) | {"custom"}

    if model_type not in valid_names:
        print(f"[choose_model] Invalid model_type '{model_type}'. "
              f"Known TorchVision models include e.g.: {sorted(list(tv_names))[:20]} ...")
        return None

    print(
        f"Model parameters: Architecture: {model_type} "
        f"init_weights: {init_weights} dropout_rate: {dropout_rate} "
        f"use_checkpoint: {use_checkpoint}", end="\r", flush=True
    )

    # --- CUSTOM BRANCH -------------------------------------------------------
    if model_type == "custom":
        raise NotImplementedError(
            "Model type 'custom' selected but no CustomCellClassifier is wired. "
            "Provide your implementation or use a TorchVision backbone."
        )

    # --- TORCHVISION CLASSIFICATION (via your TorchModel wrapper) ------------
    head_dim = max(1, int(num_classes))
    base_model = TorchModel(  # relies on your wrapper class being available in this module
        model_name=model_type,
        pretrained=bool(init_weights),
        dropout_rate=(dropout_rate if (dropout_rate and dropout_rate > 0) else None),
        use_checkpoint=use_checkpoint,
        num_classes=head_dim,
    )

    # Forward sanity-check to ensure classification logits shape
    try:
        base_model.eval()
        with torch.no_grad():
            # Keep 3 channels for sanity-check; most pretrained backbones expect 3
            dummy = torch.randn(1, 3, height, width)
            z = base_model(dummy)
            if isinstance(z, dict):
                raise RuntimeError("Selected model returned a dict, not logits.")
            if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.size(1) != head_dim:
                raise RuntimeError(
                    f"Expected logits of shape (1,{head_dim}); got {type(z)} / {getattr(z, 'shape', None)}"
                )
    except Exception as e:
        print(f"\n[choose_model] Model forward sanity-check failed: {e}")
        return None

    if verbose:
        print("\n", base_model)

    return base_model


def calculate_loss(output, target, prefer_focal=False, gamma=2.0, alpha=1.0, reduction="mean"):
    """Auto-select and return a loss for binary, multiclass, or multilabel problems.

    Dispatches based on the shapes/dtypes of ``output`` and ``target``:
      - binary: logits ``(N,1)``, float targets in ``{0,1}`` -> BCE / focal-BCE.
      - multiclass: logits ``(N,C)``, long targets ``(N,)`` -> CE / focal-CE.
      - multilabel: logits ``(N,C)``, float targets ``(N,C)`` -> BCE / focal-BCE.

    :param output: model logits.
    :param target: ground-truth labels.
    :param prefer_focal: use the focal-loss variant instead of plain CE/BCE.
    :param gamma: focal-loss focusing parameter.
    :param alpha: focal-loss class-balancing factor.
    :param reduction: one of ``'mean'``, ``'sum'``, ``'none'``.
    :returns: scalar loss tensor (or per-sample tensor when ``reduction='none'``).
    """
    # --- helpers -------------------------------------------------------------
    def _focal_bce_with_logits(logits, y, alpha=1.0, gamma=2.0, reduction="mean"):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p_t = p * y + (1 - p) * (1 - y)
        loss = alpha * (1 - p_t).pow(gamma) * ce
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    def _focal_cross_entropy(logits, y_idx, alpha=1.0, gamma=2.0, reduction="mean"):
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        log_p_t = log_p.gather(1, y_idx.view(-1,1)).squeeze(1)
        p_t = p.gather(1, y_idx.view(-1,1)).squeeze(1)
        loss = -alpha * (1 - p_t).pow(gamma) * log_p_t
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    # --- normalize shapes ----------------------------------------------------
    if output.ndim == 1:
        output = output.unsqueeze(1)  # (N,) -> (N,1)
    N, C = output.shape[0], output.shape[1]

    # --- binary (C=1) --------------------------------------------------------
    if C == 1:
        target = target.float().view(N, 1)
        if prefer_focal:
            return _focal_bce_with_logits(output, target, alpha=alpha, gamma=gamma, reduction=reduction)
        return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)

    # --- multiclass vs multilabel -------------------------------------------
    if target.dtype == torch.long and target.ndim == 1:
        # Multiclass single-label with class indices (N,)
        if prefer_focal:
            return _focal_cross_entropy(output, target, alpha=alpha, gamma=gamma, reduction=reduction)
        return F.cross_entropy(output, target, reduction=reduction)

    # Multilabel (assume float/one-hot), ensure (N,C)
    if target.ndim == 1:
        target = torch.nn.functional.one_hot(target.long(), num_classes=C).float()
    else:
        target = target.float().view(N, C)

    if prefer_focal:
        return _focal_bce_with_logits(output, target, alpha=alpha, gamma=gamma, reduction=reduction)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)

def pick_best_model(src):
    """Return the path to the ``.pth`` file in ``src`` with the highest ``epoch/acc`` tag.

    :param src: directory of checkpoint files named ``..._epoch_<N>_acc_<A>...``.
    :returns: absolute path to the top-ranked checkpoint.
    """
    all_files = os.listdir(src)
    pth_files = [f for f in all_files if f.endswith('.pth')]
    pattern = re.compile(r'_epoch_(\d+)_acc_(\d+(?:\.\d+)?)')

    def sort_key(x):
        """Return ``(accuracy, epoch)`` parsed from a checkpoint filename for sorting."""
        match = pattern.search(x)
        if not match:
            return (0.0, 0)  # Make the primary sorting key float for consistency
        g1, g2 = match.groups()
        return (float(g2), int(g1))  # Primary sort by accuracy (g2) and secondary sort by epoch (g1)
    
    sorted_files = sorted(pth_files, key=sort_key, reverse=True)
    best_model = sorted_files[0]
    return os.path.join(src, best_model)

def get_paths_from_db(df, png_df, image_type='cell_png'):
    """Return rows of ``png_df`` whose path contains ``image_type`` and whose ``prcfo`` is in ``df``.

    :param df: DataFrame indexed by ``prcfo`` identifiers.
    :param png_df: DataFrame of PNG metadata with ``png_path`` and ``prcfo`` columns.
    :param image_type: substring that must appear in ``png_path``.
    :returns: filtered subset of ``png_df``.
    """
    objects = df.index.tolist()
    filtered_df = png_df[png_df['png_path'].str.contains(image_type) & png_df['prcfo'].isin(objects)]
    return filtered_df

def save_file_lists(dst, data_set, ls):
    """Write ``ls`` as a single-column CSV named ``<data_set>.csv`` under ``dst``.

    :param dst: destination directory.
    :param data_set: column name and file stem.
    :param ls: iterable of values to persist.
    :returns: None.
    """
    df = pd.DataFrame(ls, columns=[data_set])
    df.to_csv(f'{dst}/{data_set}.csv', index=False)
    return

def augment_single_image(args):
    """Save six augmentations of one image (original, 90/180/270 rotations, H/V flips).

    :param args: ``(img_path, dst)`` tuple.
    :returns: None.
    """
    img_path, dst = args
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    filename = os.path.basename(img_path).split('.')[0]

    # Original Image
    cv2.imwrite(os.path.join(dst, f"{filename}_original.png"), img)
    
    # 90 degree rotation
    img_rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(dst, f"{filename}_rot_90.png"), img_rot_90)
    
    # 180 degree rotation
    img_rot_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(dst, f"{filename}_rot_180.png"), img_rot_180)

    # 270 degree rotation
    img_rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join(dst, f"{filename}_rot_270.png"), img_rot_270)

    # Horizontal Flip
    img_flip_hor = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(dst, f"{filename}_flip_hor.png"), img_flip_hor)

    # Vertical Flip
    img_flip_ver = cv2.flip(img, 0)
    cv2.imwrite(os.path.join(dst, f"{filename}_flip_ver.png"), img_flip_ver)

def augment_images(file_paths, dst):
    """Run :func:`augment_single_image` in parallel over ``file_paths``.

    :param file_paths: iterable of source image paths.
    :param dst: destination folder (created if missing).
    :returns: None.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    args_list = [(img_path, dst) for img_path in file_paths]

    with Pool(cpu_count()) as pool:
        pool.map(augment_single_image, args_list)
        

def suggest_training_changes(
    dst,
    train_csv=None,
    val_csv=None,
    last_k=25,
    min_epochs=10,
    gap_threshold_acc=0.05,
    plateau_eps=1e-3,
    noisy_var_ratio=0.03,
):
    """Inspect saved training/validation progress CSVs and propose concrete training changes.

    :param dst: folder where progress CSVs were saved.
    :param train_csv: explicit train-CSV path; auto-detected in ``dst`` if ``None``.
    :param val_csv: explicit val-CSV path; auto-detected in ``dst`` if ``None``.
    :param last_k: number of recent epochs used for trend and plateau checks.
    :param min_epochs: minimum epochs before most suggestions are issued.
    :param gap_threshold_acc: accuracy generalization-gap threshold (train - val).
    :param plateau_eps: absolute slope threshold used to declare a plateau.
    :param noisy_var_ratio: instability flag threshold on ``stdev/mean`` of recent val loss.
    :returns: dict with ``summary`` (key scalars), ``flags`` (short codes),
        and ``suggestions`` (ordered suggestion strings).
    """
    import os, glob, math
    import numpy as np
    import pandas as pd
    
    def _scalar(val):
        """Ensure a single float even if a Series sneaks through."""
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        return float(val)

    def _find_csv(root, hint):
        cs = sorted(glob.glob(os.path.join(root, f"*{hint}*.csv")))
        return cs[-1] if cs else None

    def _normalize_cols(df):
        # Lowercase and strip; map common variants
        m = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=m)

        # FIX: drop duplicate columns — keeps the first occurrence
        # This happens when _save_progress appends with headers repeatedly,
        # or when the same metric appears under multiple names that alias
        # to the same canonical name after normalization
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # accepted aliases
        aliases = {
            "accuracy": ["acc", "accuracy", "train_acc", "val_acc"],
            "loss": ["loss", "train_loss", "val_loss"],
            "f1_macro": ["f1_macro", "macro_f1", "f1macro", "f1"],
            "epoch": ["epoch", "epochs", "step"],
            "lr": ["lr", "learning_rate"],
        }
        name_map = {}
        for canon, opts in aliases.items():
            for o in opts:
                if o in df.columns:
                    name_map[o] = canon
        df = df.rename(columns=name_map)

        # FIX: deduplicate again after aliasing — two different original names
        # (e.g. "acc" and "accuracy") can both map to "accuracy"
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        return df

    def _poly_slope(y):
        if len(y) < 2 or np.allclose(y, y[0]):
            return 0.0
        x = np.arange(len(y), dtype=float)
        # robust to NaNs: drop them
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return 0.0
        coef = np.polyfit(x[mask], y[mask], 1)
        return float(coef[0])

    def _last_seq(series, k):
        s = np.asarray(series, dtype=float)
        return s[-min(k, len(s)):] if len(s) else np.array([])

    # --- locate CSVs ---
    train_csv = train_csv or _find_csv(dst, "train")
    val_csv = val_csv or _find_csv(dst, "val")
    out = {"summary": {}, "flags": [], "suggestions": []}

    if not train_csv or not os.path.exists(train_csv):
        out["flags"].append("missing_train_csv")
        out["suggestions"].append("Could not locate train CSV; ensure _save_progress writes a train CSV in dst.")
        return out
    if not val_csv or not os.path.exists(val_csv):
        out["flags"].append("missing_val_csv")
        out["suggestions"].append("Could not locate val CSV; enable validation logging in _save_progress.")
        return out

    tr = pd.read_csv(train_csv)
    va = pd.read_csv(val_csv)

    tr = _normalize_cols(tr)
    va = _normalize_cols(va)

    # Required columns (soft-fail if absent)
    for col in ("epoch", "loss"):
        if col not in tr.columns or col not in va.columns:
            out["flags"].append(f"missing_required_col:{col}")
            out["suggestions"].append(f"Progress CSVs lack '{col}'. Ensure _save_progress writes epoch and loss.")
            return out

    # --- core scalars ---
    best_val_idx = int(va["loss"].idxmin())
    #best_val_loss = float(va.loc[best_val_idx, "loss"])
    best_val_loss = _scalar(va.loc[best_val_idx, "loss"])

    best_epoch = int(va.loc[best_val_idx, "epoch"]) if "epoch" in va.columns else (best_val_idx + 1)

    final = {
        "train_loss": float(tr["loss"].iloc[-1]),
        "val_loss": float(va["loss"].iloc[-1]),
    }
    if "accuracy" in tr.columns:
        final["train_accuracy"] = _scalar(tr["accuracy"].iloc[-1])
    if "accuracy" in va.columns:
        final["val_accuracy"] = _scalar(va["accuracy"].iloc[-1])
    if "f1_macro" in tr.columns:
        final["train_f1_macro"] = _scalar(tr["f1_macro"].iloc[-1])
    if "f1_macro" in va.columns:
        final["val_f1_macro"] = _scalar(va["f1_macro"].iloc[-1])

    # --- trends on last_k ---
    tr_last = _last_seq(tr["loss"], last_k)
    va_last = _last_seq(va["loss"], last_k)
    slope_tr = _poly_slope(tr_last)
    slope_va = _poly_slope(va_last)

    # noise/instability
    val_mean = float(np.nanmean(va_last)) if len(va_last) else np.nan
    val_std = float(np.nanstd(va_last)) if len(va_last) else np.nan
    unstable = (len(va_last) >= max(5, last_k//2)) and np.isfinite(val_mean) and (val_std > noisy_var_ratio * max(val_mean, 1e-8))

    # generalization gap (accuracy)
    gen_gap = None
    if "accuracy" in tr.columns and "accuracy" in va.columns:
        gen_gap = _scalar(tr["accuracy"].iloc[-1]) - _scalar(va["accuracy"].iloc[-1])

    # macro-F1 NaN detection (common when a split has a single label)
    f1_nan_train = "f1_macro" in tr.columns and np.isnan(tr["f1_macro"]).mean() > 0.2
    f1_nan_val = "f1_macro" in va.columns and np.isnan(va["f1_macro"]).mean() > 0.2

    # improvement since best
    since_best = int(tr.shape[0] - (best_val_idx + 1))
    val_loss_delta_from_best = float(va["loss"].iloc[-1] - best_val_loss)

    # --- summary ---
    out["summary"].update(
        dict(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_metrics=final,
            slope_train_loss_last_k=slope_tr,
            slope_val_loss_last_k=slope_va,
            val_loss_std_last_k=val_std,
            epochs=len(tr),
            since_best=since_best,
            gen_gap_acc=gen_gap,
        )
    )

    # --- heuristics to suggest changes ---
    E = len(tr)

    # 1) Too early to judge
    if E < min_epochs:
        out["flags"].append("few_epochs")
        out["suggestions"].append(f"Only {E} epochs logged (<{min_epochs}). Consider training longer or using a warmer LR schedule.")
        # Still continue to surface other obvious issues below.

    # 2) Plateau (no meaningful val loss improvement recently)
    if len(va_last) >= max(5, last_k//2) and abs(slope_va) < plateau_eps:
        out["flags"].append("val_plateau")
        out["suggestions"].extend([
            "Validation loss plateau detected: try ReduceLROnPlateau (factor=0.1, patience=5–10) or cosine annealing with warm restarts.",
            "Add/strengthen data augmentation; if already heavy, try stochastic depth/label smoothing=0.05–0.1.",
            "If capacity may be limiting, consider a larger backbone or unfreezing more layers after a warmup.",
        ])

    # 3) Overfitting (train improving, val degrading, or large accuracy gap)
    overfit_like = False
    if slope_tr < -plateau_eps and slope_va > plateau_eps:
        overfit_like = True
    if gen_gap is not None and gen_gap > gap_threshold_acc:
        overfit_like = True
    if overfit_like:
        out["flags"].append("overfitting")
        out["suggestions"].extend([
            "Overfitting signs: increase regularization (weight_decay e.g. 0.05→0.1), enable/raise dropout (e.g. 0.2–0.5).",
            "Increase augmentation (color jitter, random crops, flips, CutMix/MixUp).",
            "Use early stopping on val loss; keep the best checkpoint (epoch with min val loss).",
            "Consider smaller head or freeze more backbone layers for longer warmup.",
        ])

    # 4) Underfitting (both losses high; train acc low and no decreasing trend)
    train_acc_low = ("accuracy" in tr.columns and final.get("train_accuracy", 0.0) < 0.70)
    losses_not_decreasing = (slope_tr > -plateau_eps and slope_va > -plateau_eps)
    if train_acc_low and losses_not_decreasing:
        out["flags"].append("underfitting")
        out["suggestions"].extend([
            "Underfitting signs: increase learning rate 2–4× or use a longer schedule (more epochs with decay).",
            "Reduce regularization (lower weight_decay), or increase model capacity (bigger backbone).",
            "Verify labels and channel order/normalization; large label noise or wrong preprocessing can cap accuracy.",
        ])

    # 5) Unstable training (high variance in recent val loss)
    if unstable:
        out["flags"].append("unstable_training")
        out["suggestions"].extend([
            "Validation loss is noisy: lower LR (e.g., ×0.5), increase batch size, or enable gradient clipping (clip_norm=1.0).",
            "Ensure deterministic preprocessing and consistent image normalization.",
        ])

    # 6) F1 NaNs (often single-class in split/batch or metric bug)
    if f1_nan_train or f1_nan_val:
        out["flags"].append("f1_nan_detected")
        out["suggestions"].extend([
            "F1(macro) shows NaN—ensure each split has ≥2 classes and use stratified sampling.",
            "If highly imbalanced, prefer class weights or focal loss (you already use focal—verify label distribution).",
        ])

    # 7) Regressed after best
    if since_best >= max(5, last_k//2) and val_loss_delta_from_best > plateau_eps:
        out["flags"].append("past_best_regression")
        out["suggestions"].extend([
            f"Validation loss has worsened by +{val_loss_delta_from_best:.4f} since best epoch {best_epoch}: adopt early stopping and keep best checkpoint.",
            "Also try ReduceLROnPlateau triggered on val loss.",
        ])

    # 8) If accuracy present but macro-F1 << accuracy -> imbalance hint
    if ("accuracy" in va.columns and "f1_macro" in va.columns
        and np.isfinite(final.get("val_accuracy", np.nan))
        and np.isfinite(final.get("val_f1_macro", np.nan))
        and (final["val_accuracy"] - final["val_f1_macro"] > 0.10)):
        out["flags"].append("class_imbalance_suspected")
        out["suggestions"].extend([
            "Accuracy ≫ macro-F1 suggests imbalance: use class weights, oversampling, or stronger focal loss (gamma 2–3, tune alpha).",
            "Track per-class metrics/confusion matrices to verify rare classes.",
        ])

    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for s in out["suggestions"]:
        if s not in seen:
            dedup.append(s); seen.add(s)
    out["suggestions"] = dedup

    return out

def _infer_indices(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return class indices (N,) from target that may be long or one-hot/float."""
    if target.dtype == torch.long:
        return target.view(-1)
    if target.ndim == 2 and target.size(1) == num_classes:
        return target.argmax(dim=1).long()
    # binary float → {0,1}
    return (target.view(-1) > 0.5).long()

def estimate_class_counts(loader, num_classes: int, src=None, classes=None) -> torch.Tensor:
    """Return per-class sample counts as a ``LongTensor`` of length ``num_classes``.

    When ``src`` and ``classes`` are provided the counts are taken from the file
    listings under ``src/<class>``, avoiding a slow DataLoader iteration on NAS.

    :param loader: fallback DataLoader iterated only when folder info is missing.
    :param num_classes: number of output classes.
    :param src: parent folder containing per-class subfolders.
    :param classes: ordered class-folder names matching ``src``.
    :returns: ``LongTensor`` of per-class counts.
    """

    # -- fast path: count files on disk instead of loading images --
    if src is not None and classes is not None:
        counts = torch.zeros(num_classes, dtype=torch.long)
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(src, cls)
            if os.path.isdir(cls_dir):
                # count only files, skip subdirectories
                n = sum(1 for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)))
                counts[i] = n
        print(f"Class counts (from folders): {dict(zip(classes, counts.tolist()))}")
        return counts

    # -- slow fallback: iterate the DataLoader (original behavior) --
    print("Warning: counting classes by iterating DataLoader (slow on NAS). "
          "Pass src and classes to avoid this.")
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y, _ in loader:
        y = y.detach()
        idx = _infer_indices(y, num_classes)
        binc = torch.bincount(idx, minlength=num_classes)
        counts[:num_classes] += binc[:num_classes]
    return counts

def build_loss(loss_type: str = "ce",
               num_classes: int = 2,
               class_counts: Optional[torch.Tensor] = None,
               label_smoothing: float = 0.0,
               focal_gamma: float = 2.0,
               focal_alpha: Optional[float] = None,
               logit_adjust_tau: float = 0.0,
               asl_gamma_pos: float = 0.0,
               asl_gamma_neg: float = 4.0,
               asl_clip: float = 0.05):
    """Return a closure ``loss_fn(logits, target)`` implementing the requested loss.

    Supported ``loss_type`` values: ``'ce'``, ``'ce_smooth'``, ``'ce_weighted'``,
    ``'focal_ce'``, ``'bce'``, ``'focal_bce'``, ``'logit_adjust_ce'``, ``'asl'``, ``'auto'``.
    ``num_classes==1`` selects binary (BCE variants); ``>=2`` selects multiclass (CE variants).

    :param loss_type: loss identifier (see above).
    :param num_classes: output class count.
    :param class_counts: per-class sample counts used to derive weights or logit adjustment.
    :param label_smoothing: label-smoothing epsilon for ``ce_smooth``.
    :param focal_gamma: focal-loss focusing parameter.
    :param focal_alpha: focal-loss class-balancing factor (float or per-class tensor).
    :param logit_adjust_tau: strength of the Menon-et-al. logit adjustment; 0 disables.
    :param asl_gamma_pos: asymmetric-loss gamma for positives.
    :param asl_gamma_neg: asymmetric-loss gamma for negatives.
    :param asl_clip: asymmetric-loss negative-probability clip.
    :returns: ``loss_fn(logits, target)`` callable returning a scalar tensor.
    :raises ValueError: if ``loss_type`` is unknown or incompatible with ``num_classes``.
    """
    lt = (loss_type or "ce").lower()

    # -------- helpers (scoped) --------
    def _infer_indices(target: torch.Tensor, C: int) -> torch.Tensor:
        # Accept indices (N,) or one-hot (N,C); return indices (N,)
        if target.ndim == 2:
            return target.argmax(dim=1).long()
        return target.long().view(-1)

    # Priors/weights from counts if provided
    class_weights = None
    logit_adjust = None
    if class_counts is not None:
        counts = class_counts.to(dtype=torch.float)
        counts = torch.clamp(counts, min=1.0)
        priors = counts / counts.sum()
        inv = 1.0 / priors
        class_weights = (inv / inv.mean()).to(dtype=torch.float)
        # Menon et al. 2020: logit adjustment
        if logit_adjust_tau > 0:
            logit_adjust = (-float(logit_adjust_tau) * priors.log()).to(dtype=torch.float)

    # ----- binary focal BCE -----
    def _focal_bce(logits, y, alpha, gamma):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        w = (1 - pt).pow(gamma)
        if alpha is not None:
            w = w * (alpha * y + (1 - alpha) * (1 - y))
        return (w * ce).mean()

    # ----- multiclass focal-CE -----
    def _focal_ce(logits, y_idx, alpha, gamma):
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        log_p_t = log_p.gather(1, y_idx.view(-1, 1)).squeeze(1)
        p_t = p.gather(1, y_idx.view(-1, 1)).squeeze(1)
        w = (1 - p_t).pow(gamma)
        if alpha is not None:
            if torch.is_tensor(alpha) and alpha.numel() > 1:
                a = alpha.to(logits.device)[y_idx]
            else:
                a = float(alpha)
            loss = -a * w * log_p_t
        else:
            loss = -w * log_p_t
        return loss.mean()

    # ----- Asymmetric Loss (multilabel-style one-vs-all) -----
    def _asl(logits, y, gpos, gneg, clip):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if clip and clip > 0:
            xs_neg = torch.clamp(xs_neg + clip, max=1.0)
        loss = y * torch.log(xs_pos.clamp_min(1e-8)) + (1 - y) * torch.log(xs_neg.clamp_min(1e-8))
        pt = xs_pos * y + xs_neg * (1 - y)
        one_sided = (1 - pt).pow(gpos * y + gneg * (1 - y))
        return -(one_sided * loss).mean()

    # Auto heuristic
    def _auto_choice() -> str:
        if num_classes >= 2:
            if class_counts is not None:
                props = (class_counts.float() / class_counts.sum().clamp_min(1))
                if props.min() < 0.10:
                    return "logit_adjust_ce"
            return "ce"
        else:
            return "bce"

    if lt == "auto":
        lt = _auto_choice()

    # -------- binary (num_classes == 1) --------
    if num_classes == 1:
        if lt in ("bce", "binary_cross_entropy_with_logits"):
            def loss_fn(logits, target):
                """Closure: compute the selected per-batch loss from ``(logits, target)``."""
                y = target.float().view(-1, 1)
                return F.binary_cross_entropy_with_logits(logits, y)
        elif lt in ("focal_bce", "focal", "focal_loss"):
            def loss_fn(logits, target):
                """Closure: compute the selected per-batch loss from ``(logits, target)``."""
                y = target.float().view(-1, 1)
                return _focal_bce(logits, y, focal_alpha, focal_gamma)
        else:
            raise ValueError(f"loss_type '{loss_type}' not valid for binary (num_classes=1)")
        return loss_fn

    # -------- multiclass (num_classes >= 2) --------
    if lt in ("ce", "cross_entropy"):
        def loss_fn(logits, target):
            """Closure: compute the selected per-batch loss from ``(logits, target)``."""
            y = _infer_indices(target, num_classes)
            w = class_weights.to(logits.device) if class_weights is not None else None
            return F.cross_entropy(logits, y, weight=w)
    elif lt in ("ce_smooth", "label_smoothing"):
        def loss_fn(logits, target):
            """Closure: compute the selected per-batch loss from ``(logits, target)``."""
            y = _infer_indices(target, num_classes)
            w = class_weights.to(logits.device) if class_weights is not None else None
            return F.cross_entropy(logits, y, weight=w, label_smoothing=float(label_smoothing))
    elif lt in ("ce_weighted",):
        if class_weights is None:
            raise ValueError("ce_weighted requires class_counts (to derive weights).")
        def loss_fn(logits, target):
            """Closure: compute the selected per-batch loss from ``(logits, target)``."""
            y = _infer_indices(target, num_classes)
            return F.cross_entropy(logits, y, weight=class_weights.to(logits.device))
    elif lt in ("focal_ce", "focal"):
        alpha = None
        if focal_alpha is not None:
            alpha = focal_alpha if torch.is_tensor(focal_alpha) else float(focal_alpha)
            if torch.is_tensor(alpha) and alpha.numel() == num_classes:
                alpha = alpha.to(torch.float)
        def loss_fn(logits, target):
            """Closure: compute the selected per-batch loss from ``(logits, target)``."""
            y = _infer_indices(target, num_classes)
            return _focal_ce(logits, y, alpha, focal_gamma)
    elif lt in ("logit_adjust_ce", "la_ce"):
        if class_counts is None:
            raise ValueError("logit_adjust_ce requires class_counts.")
        adjust = logit_adjust.to(torch.float) if logit_adjust is not None else None
        def loss_fn(logits, target):
            """Closure: compute the selected per-batch loss from ``(logits, target)``."""
            y = _infer_indices(target, num_classes)
            z = logits if adjust is None else (logits + adjust.to(logits.device))
            return F.cross_entropy(z, y)
    elif lt in ("asl", "asymmetric_loss"):
        def loss_fn(logits, target):
            """Closure: compute the selected per-batch loss from ``(logits, target)``."""
            # expect one-hot/float (N,C) or indices (N,)
            if target.ndim == 1:
                y = F.one_hot(target.long(), num_classes=num_classes).float()
            else:
                y = target.float().view(-1, num_classes)
            return _asl(logits, y, asl_gamma_pos, asl_gamma_neg, asl_clip)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    return loss_fn

def augment_classes(dst, nc, pc, generate=True,move=True):
    """Augment negative and positive class images and split them into train/test folders.

    :param dst: destination root; augmented images land under ``aug_nc``/``aug_pc`` and
        move into ``aug/{train,test}/{nc,pc}``.
    :param nc: negative-class source image paths.
    :param pc: positive-class source image paths.
    :param generate: run augmentation before moving files.
    :param move: split augmented images into train/test folders.
    :returns: None.
    """
    aug_nc = os.path.join(dst,'aug_nc')
    aug_pc = os.path.join(dst,'aug_pc')
    all_ = len(nc)+len(pc)
    if generate == True:
        os.makedirs(aug_nc, exist_ok=True)
        if __name__ == '__main__':
            augment_images(file_paths=nc, dst=aug_nc)

        os.makedirs(aug_pc, exist_ok=True)
        if __name__ == '__main__':
            augment_images(file_paths=pc, dst=aug_pc)

    if move == True:
        aug = os.path.join(dst,'aug')
        aug_train_nc = os.path.join(aug,'train/nc')
        aug_train_pc = os.path.join(aug,'train/pc')
        aug_test_nc = os.path.join(aug,'test/nc')
        aug_test_pc = os.path.join(aug,'test/pc')

        os.makedirs(aug_train_nc, exist_ok=True)
        os.makedirs(aug_train_pc, exist_ok=True)
        os.makedirs(aug_test_nc, exist_ok=True)
        os.makedirs(aug_test_pc, exist_ok=True)

        aug_nc_list = [os.path.join(aug_nc, file) for file in os.listdir(aug_nc)]
        aug_pc_list = [os.path.join(aug_pc, file) for file in os.listdir(aug_pc)]

        nc_train_data, nc_test_data = train_test_split(aug_nc_list, test_size=0.1, shuffle=True, random_state=42)
        pc_train_data, pc_test_data = train_test_split(aug_pc_list, test_size=0.1, shuffle=True, random_state=42)

        i=0
        for path in nc_train_data:
            i+=1
            shutil.move(path, os.path.join(aug_train_nc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        for path in nc_test_data:
            i+=1
            shutil.move(path, os.path.join(aug_test_nc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        for path in pc_train_data:
            i+=1
            shutil.move(path, os.path.join(aug_train_pc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        for path in pc_test_data:
            i+=1
            shutil.move(path, os.path.join(aug_test_pc, os.path.basename(path)))
            print(f'{i}/{all_}', end='\r', flush=True)
        print(f'Train nc: {len(os.listdir(aug_train_nc))}, Train pc:{len(os.listdir(aug_test_nc))}, Test nc:{len(os.listdir(aug_train_pc))}, Test pc:{len(os.listdir(aug_test_pc))}')
        return

def annotate_predictions(csv_loc):
    """Read prediction CSV and add plate/well/field/object columns plus a ``cond`` label.

    :param csv_loc: path to a predictions CSV with a ``path`` column of PNG paths.
    :returns: DataFrame enriched with parsed metadata and a ``cond`` column
        (``'screen'``/``'pc'``/``'nc'`` from the plate/well convention).
    """
    df = pd.read_csv(csv_loc)
    df['filename'] = df['path'].apply(lambda x: x.split('/')[-1])
    df[['plateID', 'well', 'fieldID', 'object']] = df['filename'].str.split('_', expand=True)
    df['object'] = df['object'].str.replace('.png', '')
    
    def assign_condition(row):
        """Return the condition label (``'screen'``/``'pc'``/``'nc'`` or ``''``) for a metadata row."""
        plate = int(row['plateID'])
        col = int(row['well'][1:])
        
        if col > 3:
            if plate in [1, 2, 3, 4]:
                return 'screen'
            elif plate in [5, 6, 7, 8]:
                return 'pc'
        elif col in [1, 2, 3]:
            return 'nc'
        else:
            return ''

    df['cond'] = df.apply(assign_condition, axis=1)
    return df

def initiate_counter(counter_, lock_):
    """Initialize shared multiprocessing ``counter`` and ``lock`` globals.

    :param counter_: shared ``multiprocessing.Value`` counter.
    :param lock_: shared ``multiprocessing.Lock`` guarding the counter.
    :returns: None.
    """
    global counter, lock
    counter = counter_
    lock = lock_

def add_images_to_tar(paths_chunk, tar_path, total_images):
    """Add ``paths_chunk`` images to ``tar_path``, updating the shared counter for progress.

    :param paths_chunk: list of image paths to add.
    :param tar_path: destination tar archive path.
    :param total_images: overall image count used to render progress.
    :returns: None.
    """
    with tarfile.open(tar_path, 'w') as tar:
        for i, img_path in enumerate(paths_chunk):
            arcname = os.path.basename(img_path)
            try:
                tar.add(img_path, arcname=arcname)
                with lock:
                    counter.value += 1
                    if counter.value % 10 == 0:  # Print every 100 updates
                        #progress = (counter.value / total_images) * 100
                        #print(f"Progress: {counter.value}/{total_images} ({progress:.2f}%)", end='\r', file=sys.stdout, flush=True)
                        print_progress(counter.value, total_images, n_jobs=1, time_ls=None, batch_size=None, operation_type="generating .tar dataset")
            except FileNotFoundError:
                print(f"File not found: {img_path}")

def generate_fraction_map(df, gene_column, min_frequency=0.0):
    """Return a wells-by-genes fraction matrix, dropping columns below ``min_frequency``.

    :param df: long-format DataFrame with ``prc``, ``count``, ``well_read_sum`` columns.
    :param gene_column: column identifying the gene/guide.
    :param min_frequency: drop columns whose maximum fraction is below this cutoff.
    :returns: DataFrame indexed by ``prc`` with per-gene fractions.
    """
    df['fraction'] = df['count']/df['well_read_sum']
    genes = df[gene_column].unique().tolist()
    wells = df['prc'].unique().tolist()
    print(len(genes),len(wells))
    independent_variables = pd.DataFrame(columns=genes, index = wells)
    for index, row in df.iterrows():
        prc = row['prc']
        gene = row[gene_column]
        fraction = row['fraction']
        independent_variables.loc[prc,gene]=fraction
    independent_variables = independent_variables.dropna(axis=1, how='all')
    independent_variables = independent_variables.dropna(axis=0, how='all')
    independent_variables['sum'] = independent_variables.sum(axis=1)
    #sums = independent_variables['sum'].unique().tolist()
    #print(sums)
    #independent_variables = independent_variables[(independent_variables['sum'] == 0.0) | (independent_variables['sum'] == 1.0)]
    independent_variables = independent_variables.fillna(0.0)
    independent_variables = independent_variables.drop(columns=[col for col in independent_variables.columns if independent_variables[col].max() < min_frequency])
    independent_variables = independent_variables.drop('sum', axis=1)
    independent_variables.index.name = 'prc'
    # NOTE: previously this unconditionally wrote the result to a hardcoded
    # developer-machine path ('/mnt/data/CellVoyager/.../iv.csv'), which raised
    # for any other environment. Removed — callers persist the returned frame
    # themselves if they need it.
    return independent_variables

def fishers_odds(df, threshold=0.5, phenotyp_col='mean_pred'):
    """Fisher's exact test per mutant column against a binarized phenotype label.

    :param df: DataFrame with per-mutant presence columns plus ``phenotyp_col``.
    :param threshold: cutoff below which ``phenotyp_col`` is called "high phenotype".
    :param phenotyp_col: name of the phenotype column.
    :returns: DataFrame with columns ``Mutant``, ``OddsRatio``, ``PValue``, ``AdjustedPValue``.
    """
    # Binning based on phenotype score (e.g., above 0.8 as high)
    df['high_phenotype'] = df[phenotyp_col] < threshold

    results = []
    mutants = df.columns[:-2]
    mutants = [item for item in mutants if item not in ['count_prc','mean_pathogen_area']]
    print(f'fishers df')
    display(df)
    # Perform Fisher's exact test for each mutant
    for mutant in mutants:
        contingency_table = pd.crosstab(df[mutant] > 0, df['high_phenotype'])
        if contingency_table.shape == (2, 2):  # Check for 2x2 shape
            odds_ratio, p_value = fisher_exact(contingency_table)
            results.append((mutant, odds_ratio, p_value))
        else:
            # Optionally handle non-2x2 tables (e.g., append NaN or other placeholders)
            results.append((mutant, float('nan'), float('nan')))
    
    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(results, columns=['Mutant', 'OddsRatio', 'PValue'])
    # Remove rows with undefined odds ratios or p-values
    filtered_results_df = results_df.dropna(subset=['OddsRatio', 'PValue'])
    
    pvalues = filtered_results_df['PValue'].values

    # Check if pvalues array is empty
    if len(pvalues) > 0:
        # Apply Benjamini-Hochberg correction
        adjusted_pvalues = multipletests(pvalues, method='fdr_bh')[1]
        # Add adjusted p-values back to the dataframe
        filtered_results_df['AdjustedPValue'] = adjusted_pvalues
        # Filter significant results
        significant_mutants = filtered_results_df[filtered_results_df['AdjustedPValue'] < 0.05]
    else:
        print("No p-values to adjust. Check your data filtering steps.")
        significant_mutants = pd.DataFrame()  # return empty DataFrame in this case
    
    return filtered_results_df

def model_metrics(model):
    """Print RMSE/MAE/Durbin-Watson and show residual/QQ/scale-location diagnostic plots.

    :param model: fitted statsmodels regression result.
    :returns: None.
    """
    # Calculate additional metrics
    rmse = np.sqrt(model.mse_resid)
    mae = np.mean(np.abs(model.resid))
    durbin_w_value = durbin_watson(model.resid)

    # Display the additional metrics
    print("\nAdditional Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Durbin-Watson: {durbin_w_value}")

    # Residual Plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    # Residual vs. Fitted
    ax[0, 0].scatter(model.fittedvalues, model.resid, edgecolors = 'k', facecolors = 'none')
    ax[0, 0].set_title('Residuals vs Fitted')
    ax[0, 0].set_xlabel('Fitted values')
    ax[0, 0].set_ylabel('Residuals')

    # Histogram
    sns.histplot(model.resid, kde=True, ax=ax[0, 1])
    ax[0, 1].set_title('Histogram of Residuals')
    ax[0, 1].set_xlabel('Residuals')

    # QQ Plot
    sm.qqplot(model.resid, fit=True, line='45', ax=ax[1, 0])
    ax[1, 0].set_title('QQ Plot')

    # Scale-Location
    standardized_resid = model.get_influence().resid_studentized_internal
    ax[1, 1].scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), edgecolors = 'k', facecolors = 'none')
    ax[1, 1].set_title('Scale-Location')
    ax[1, 1].set_xlabel('Fitted values')
    ax[1, 1].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    plt.tight_layout()
    plt.show()

def check_multicollinearity(x):
    """Checks multicollinearity of the predictors by computing the VIF."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_data

def lasso_reg(merged_df, alpha_value=0.01, reg_type='lasso'):
    """Fit Lasso or Ridge on one-hot-encoded gene/grna/plate/row/column predictors.

    :param merged_df: DataFrame with ``gene``, ``grna``, ``plateID``, ``rowID``, ``columnID``, ``pred``.
    :param alpha_value: regularization strength.
    :param reg_type: ``'lasso'`` or ``'ridge'``.
    :returns: DataFrame with ``Feature`` and ``Coefficient`` columns.
    """
    # Separate predictors and response
    X = merged_df[['gene', 'grna', 'plateID', 'rowID', 'columnID']]
    y = merged_df['pred']

    # One-hot encode the categorical predictors
    encoder = OneHotEncoder(drop='first')  # drop one category to avoid the dummy variable trap
    X_encoded = encoder.fit_transform(X).toarray()
    feature_names = encoder.get_feature_names_out(input_features=X.columns)
    
    if reg_type == 'ridge':
        # Fit ridge regression
        ridge = Ridge(alpha=alpha_value)
        ridge.fit(X_encoded, y)
        coefficients = ridge.coef_
        coeff_dict = dict(zip(feature_names, ridge.coef_))
        
    if reg_type == 'lasso':
        # Fit Lasso regression
        lasso = Lasso(alpha=alpha_value)
        lasso.fit(X_encoded, y)
        coefficients = lasso.coef_
        coeff_dict = dict(zip(feature_names, lasso.coef_))
    coeff_df = pd.DataFrame(list(coeff_dict.items()), columns=['Feature', 'Coefficient'])
    return coeff_df

def MLR(merged_df, refine_model):
    """Fit a multiple-linear regression on gene:grna interactions plus plate/row/column terms.

    :param merged_df: DataFrame with ``gene``, ``grna``, ``plate``, ``row``, ``column``, ``pred`` columns.
    :param refine_model: refit after removing outliers by residuals and Cook's distance.
    :returns: tuple ``(max_effects, max_effects_pvalues, model, df)``.
    """
    from .plot import _reg_v_plot
    
    #model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()
    model = smf.ols("pred ~ gene:grna + plate + row + column", merged_df).fit()
    # Display model metrics and summary
    model_metrics(model)

    if refine_model:
        # Filter outliers
        std_resid = model.get_influence().resid_studentized_internal
        outliers_resid = np.where(np.abs(std_resid) > 3)[0]
        (c, p) = model.get_influence().cooks_distance
        outliers_cooks = np.where(c > 4/(len(merged_df)-merged_df.shape[1]-1))[0]
        outliers = reduce(np.union1d, (outliers_resid, outliers_cooks))
        merged_df_filtered = merged_df.drop(merged_df.index[outliers])

        display(merged_df_filtered)

        # Refit the model with filtered data
        model = smf.ols("pred ~ gene + grna + gene:grna + row + column", merged_df_filtered).fit()
        print("Number of outliers detected by standardized residuals:", len(outliers_resid))
        print("Number of outliers detected by Cook's distance:", len(outliers_cooks))

        model_metrics(model)
        print(model.summary())

    # Extract interaction coefficients and determine the maximum effect size
    interaction_coeffs = {key: val for key, val in model.params.items() if "gene[T." in key and ":grna[T." in key}
    interaction_pvalues = {key: val for key, val in model.pvalues.items() if "gene[T." in key and ":grna[T." in key}

    max_effects = {}
    max_effects_pvalues = {}
    for key, val in interaction_coeffs.items():
        gene_name = key.split(":")[0].replace("gene[T.", "").replace("]", "")
        if gene_name not in max_effects or abs(max_effects[gene_name]) < abs(val):
            max_effects[gene_name] = val
            max_effects_pvalues[gene_name] = interaction_pvalues[key]

    for key in max_effects:
        print(f"Key: {key}: {max_effects[key]}, p:{max_effects_pvalues[key]}")

    df = pd.DataFrame([max_effects, max_effects_pvalues])
    df = df.transpose()
    df = df.rename(columns={df.columns[0]: 'effect', df.columns[1]: 'p'})
    df = df.sort_values(by=['effect', 'p'], ascending=[False, True])

    _reg_v_plot(df)
    
    return max_effects, max_effects_pvalues, model, df

def get_files_from_dir(dir_path, file_extension="*"):
    """Return glob matches for ``dir_path/file_extension``."""
    # ``glob`` is imported as the module here (see glob.glob usage elsewhere),
    # so it must be called as glob.glob — a bare glob(...) raised TypeError.
    return glob.glob(os.path.join(dir_path, file_extension))

def create_circular_mask(h, w, center=None, radius=None):
    """Return a boolean circular mask of shape ``(h, w)`` centered on ``center``.

    :param h: image height.
    :param w: image width.
    :param center: ``(x, y)`` center; defaults to the image middle.
    :param radius: circle radius; defaults to the largest circle fitting inside.
    :returns: boolean ndarray where ``True`` marks pixels within ``radius``.
    """
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
def apply_mask(image, output_value=0):
    """Zero out (or set to ``output_value``) pixels outside a circular mask fit to ``image``."""
    h, w = image.shape[:2]  # Assuming image is grayscale or RGB
    mask = create_circular_mask(h, w)
    
    # If the image has more than one channel, repeat the mask for each channel
    if len(image.shape) > 2:
        mask = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)
    
    # Apply the mask - set pixels outside of the mask to output_value
    masked_image = np.where(mask, image, output_value)
    return masked_image
    
def invert_image(image):
    """Return the intensity-inverted image, using the dtype max as the pivot."""
    # The maximum value depends on the image dtype (e.g., 255 for uint8)
    max_value = np.iinfo(image.dtype).max
    inverted_image = max_value - image
    return inverted_image

def resize_images_and_labels(images, labels, target_height, target_width, show_example=True):
    """Resize aligned image/label lists to ``target_height`` x ``target_width``.

    :param images: iterable of source images (2-D or 3-D).
    :param labels: matching iterable of label masks, or ``None``.
    :param target_height: output height in pixels.
    :param target_width: output width in pixels.
    :param show_example: display an example of the resized pair when ``True``.
    :returns: ``(resized_images, resized_labels)`` lists.
    """
    
    from .plot import plot_resize
    
    resized_images = []
    resized_labels = []
    if not images is None and not labels is None:
        for image, label in zip(images, labels):

            if image.ndim == 2:
                image_shape = (target_height, target_width)
            elif image.ndim == 3:
                image_shape = (target_height, target_width, image.shape[-1])
                
            resized_image = resizescikit(image, image_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
            resized_label = resizescikit(label, (target_height, target_width), order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
            
            if resized_image.shape[-1] == 1:
                resized_image = np.squeeze(resized_image)
            
            resized_images.append(resized_image)
            resized_labels.append(resized_label)
    
    elif not images is None:
        for image in images:
        
            if image.ndim == 2:
                image_shape = (target_height, target_width)
            elif image.ndim == 3:
                image_shape = (target_height, target_width, image.shape[-1])
                
            resized_image = resizescikit(image, image_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
            
            if resized_image.shape[-1] == 1:
                resized_image = np.squeeze(resized_image)
            
            resized_images.append(resized_image)
            
    elif not labels is None:
        for label in labels:
            resized_label = resizescikit(label, (target_height, target_width), order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
            resized_labels.append(resized_label)
        
    if show_example:     
        if not images is None and not labels is None:
            plot_resize(images, resized_images, labels, resized_labels)
        elif not images is None:
            plot_resize(images, resized_images, images, resized_images)
        elif not labels is None:
            plot_resize(labels, resized_labels, labels, resized_labels)
    
    return resized_images, resized_labels

def resize_labels_back(labels, orig_dims):
    """Resize a list of label masks back to their original ``(width, height)``.

    :param labels: iterable of label masks.
    :param orig_dims: matching iterable of ``(width, height)`` tuples.
    :returns: list of resized label masks.
    :raises ValueError: if lengths differ or ``orig_dims`` entries are malformed.
    """
    resized_labels = []

    if len(labels) != len(orig_dims):
        raise ValueError("The length of labels and orig_dims must match.")

    for label, dims in zip(labels, orig_dims):
        # Ensure dims is a tuple of two integers (width, height)
        if not isinstance(dims, tuple) or len(dims) != 2:
            raise ValueError("Each element in orig_dims must be a tuple of two integers representing the original dimensions (width, height)")

        resized_label = resizescikit(label, dims, order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
        resized_labels.append(resized_label)

    return resized_labels

def calculate_iou(mask1, mask2):
    """Return the intersection-over-union of two binary masks after zero-padding to a common shape."""
    mask1, mask2 = pad_to_same_shape(mask1, mask2)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0
    
def match_masks(true_masks, pred_masks, iou_threshold):
    """Greedy match each predicted mask to a still-unmatched true mask above ``iou_threshold``.

    :param true_masks: iterable of ground-truth masks.
    :param pred_masks: iterable of predicted masks.
    :param iou_threshold: minimum IoU to count as a match.
    :returns: list of ``(true_mask, pred_mask)`` matched pairs.
    """
    matches = []
    matched_true_masks_indices = set()  # Use set to store indices of matched true masks

    for pred_mask in pred_masks:
        for true_mask_index, true_mask in enumerate(true_masks):
            if true_mask_index not in matched_true_masks_indices:
                iou = calculate_iou(true_mask, pred_mask)
                if iou >= iou_threshold:
                    matches.append((true_mask, pred_mask))
                    matched_true_masks_indices.add(true_mask_index)  # Store the index of the matched true mask
                    break  # Move on to the next predicted mask
    return matches
    
def compute_average_precision(matches, num_true_masks, num_pred_masks):
    """Return ``(precision, recall)`` given match count, true count, and predicted count."""
    TP = len(matches)
    FP = num_pred_masks - TP
    FN = num_true_masks - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall

def pad_to_same_shape(mask1, mask2):
    """Zero-pad ``mask1`` and ``mask2`` to their element-wise maximum shape."""
    # Find the shape differences
    shape_diff = np.array([max(mask1.shape[0], mask2.shape[0]) - mask1.shape[0], 
                           max(mask1.shape[1], mask2.shape[1]) - mask1.shape[1]])
    pad_mask1 = ((0, shape_diff[0]), (0, shape_diff[1]))
    shape_diff = np.array([max(mask1.shape[0], mask2.shape[0]) - mask2.shape[0], 
                           max(mask1.shape[1], mask2.shape[1]) - mask2.shape[1]])
    pad_mask2 = ((0, shape_diff[0]), (0, shape_diff[1]))
    
    padded_mask1 = np.pad(mask1, pad_mask1, mode='constant', constant_values=0)
    padded_mask2 = np.pad(mask2, pad_mask2, mode='constant', constant_values=0)
    
    return padded_mask1, padded_mask2
    
def compute_ap_over_iou_thresholds(true_masks, pred_masks, iou_thresholds):
    """Return the area under the precision-recall curve swept over ``iou_thresholds``."""
    precision_recall_pairs = []
    for iou_threshold in iou_thresholds:
        matches = match_masks(true_masks, pred_masks, iou_threshold)
        precision, recall = compute_average_precision(matches, len(true_masks), len(pred_masks))
        # Check that precision and recall are within the range [0, 1]
        if not 0 <= precision <= 1 or not 0 <= recall <= 1:
            raise ValueError(f'Precision or recall out of bounds. Precision: {precision}, Recall: {recall}')
        precision_recall_pairs.append((precision, recall))

    # Sort by recall values
    precision_recall_pairs = sorted(precision_recall_pairs, key=lambda x: x[1])
    sorted_precisions = [p[0] for p in precision_recall_pairs]
    sorted_recalls = [p[1] for p in precision_recall_pairs]
    return np.trapz(sorted_precisions, x=sorted_recalls)
    
def compute_segmentation_ap(true_masks, pred_masks, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Return the COCO-style segmentation AP by matching connected components across IoU thresholds."""
    true_mask_labels = label(true_masks)
    pred_mask_labels = label(pred_masks)
    true_mask_regions = [region.image for region in regionprops(true_mask_labels)]
    pred_mask_regions = [region.image for region in regionprops(pred_mask_labels)]
    return compute_ap_over_iou_thresholds(true_mask_regions, pred_mask_regions, iou_thresholds)

def jaccard_index(mask1, mask2):
    """Return the Jaccard/IoU index of two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def dice_coefficient(mask1, mask2):
    """Return the Dice similarity of two masks, treating any nonzero value as foreground."""
    # Convert to binary masks
    mask1 = np.where(mask1 > 0, 1, 0)
    mask2 = np.where(mask2 > 0, 1, 0)

    # Calculate intersection and total
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    
    # Handle the case where both masks are empty
    if total == 0:
        return 1.0
    
    # Return the Dice coefficient
    return 2.0 * intersection / total

def extract_boundaries(mask, dilation_radius=1):
    """Return the boundary of a binary mask via morphological dilation minus erosion.

    :param mask: label or binary mask.
    :param dilation_radius: half-width of the structuring element.
    :returns: boolean boundary mask.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    struct_elem = np.ones((dilation_radius*2+1, dilation_radius*2+1))
    dilated = morphology.binary_dilation(binary_mask, footprint=struct_elem)
    eroded = morphology.binary_erosion(binary_mask, footprint=struct_elem)
    boundary = dilated ^ eroded
    return boundary

def boundary_f1_score(mask_true, mask_pred, dilation_radius=1):
    """Return the boundary F1 score between two masks with tolerance ``dilation_radius``."""
    # Assume extract_boundaries is defined to extract object boundaries with given dilation_radius
    boundary_true = extract_boundaries(mask_true, dilation_radius)
    boundary_pred = extract_boundaries(mask_pred, dilation_radius)
    
    # Calculate intersection of boundaries
    intersection = np.logical_and(boundary_true, boundary_pred)
    
    # Calculate precision and recall for boundary detection
    precision = np.sum(intersection) / (np.sum(boundary_pred) + 1e-6)
    recall = np.sum(intersection) / (np.sum(boundary_true) + 1e-6)
    
    # Calculate F1 score as harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return f1



def _remove_noninfected(stack, cell_dim, nucleus_dim, pathogen_dim):
    """Zero out cells (and their nuclei) that contain no pathogen labels."""
    if not cell_dim is None:
        cell_mask = stack[:, :, cell_dim]
    else:
        cell_mask = np.zeros_like(stack)
    if not nucleus_dim is None:
        nucleus_mask = stack[:, :, nucleus_dim]
    else:
        nucleus_mask = np.zeros_like(stack)

    if not pathogen_dim is None:
        pathogen_mask = stack[:, :, pathogen_dim]
    else:
        pathogen_mask = np.zeros_like(stack)

    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        labels_in_cell = np.unique(pathogen_mask[cell_region])
        if len(labels_in_cell) <= 1:
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0
    if not cell_dim is None:
        stack[:, :, cell_dim] = cell_mask
    if not nucleus_dim is None:
        stack[:, :, nucleus_dim] = nucleus_mask
    return stack

def _remove_outside_objects(stack, cell_dim, nucleus_dim, pathogen_dim):
    """Zero out pathogens (and their nuclei) that do not overlap any cell."""
    if not cell_dim is None:
        cell_mask = stack[:, :, cell_dim]
    else:
        return stack
    nucleus_mask = stack[:, :, nucleus_dim]
    pathogen_mask = stack[:, :, pathogen_dim]
    pathogen_labels = np.unique(pathogen_mask)[1:]
    for pathogen_label in pathogen_labels:
        pathogen_region = pathogen_mask == pathogen_label
        cell_in_pathogen_region = np.unique(cell_mask[pathogen_region])
        cell_in_pathogen_region = cell_in_pathogen_region[cell_in_pathogen_region != 0]  # Exclude background
        if len(cell_in_pathogen_region) == 0:
            pathogen_mask[pathogen_region] = 0
            corresponding_nucleus_region = nucleus_mask == pathogen_label
            nucleus_mask[corresponding_nucleus_region] = 0
    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, pathogen_dim] = pathogen_mask
    return stack

def _remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, pathogen_dim, object_dim):
    """Zero out cells containing more than one object in ``object_dim``."""
    cell_mask = stack[:, :, mask_dim]
    nucleus_mask = stack[:, :, nucleus_dim]
    pathogen_mask = stack[:, :, pathogen_dim]
    object_mask = stack[:, :, object_dim]

    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        labels_in_cell = np.unique(object_mask[cell_region])
        if len(labels_in_cell) > 2:
            cell_mask[cell_region] = 0
            nucleus_mask[cell_region] = 0
            for pathogen_label in labels_in_cell[1:]:  # Skip the first label (0)
                pathogen_mask[pathogen_mask == pathogen_label] = 0

    stack[:, :, cell_dim] = cell_mask
    stack[:, :, nucleus_dim] = nucleus_mask
    stack[:, :, pathogen_dim] = pathogen_mask
    return stack
    
def merge_touching_objects(mask, threshold=0.25):
    """Merge touching labeled objects whose shared boundary exceeds ``threshold`` of the smaller perimeter.

    :param mask: labeled mask.
    :param threshold: fraction of the smaller perimeter required to merge.
    :returns: merged label mask.
    """
    perimeters = {}
    labels = np.unique(mask)
    # Calculating perimeter of each object
    for label in labels:
        if label != 0:  # Ignore background
            edges = morphology.erosion(mask == label) ^ (mask == label)
            perimeters[label] = np.sum(edges)
    # Detect touching objects and find the shared boundary
    shared_perimeters = {}
    dilated = morphology.dilation(mask > 0)
    for label in labels:
        if label != 0:  # Ignore background
            # Find the objects that this object is touching
            dilated_label = morphology.dilation(mask == label)
            touching_labels = np.unique(mask[dilated & (dilated_label != 0) & (mask != 0)])
            for touching_label in touching_labels:
                if touching_label != label:  # Exclude the object itself
                    shared_boundary = dilated_label & morphology.dilation(mask == touching_label)
                    shared_perimeters[(label, touching_label)] = np.sum(shared_boundary)
    # Merge objects if more than 25% of their boundary is touching
    for (label1, label2), shared_perimeter in shared_perimeters.items():
        if shared_perimeter > threshold * min(perimeters[label1], perimeters[label2]):
            mask[mask == label2] = label1  # Merge label2 into label1
    return mask
    
def remove_intensity_objects(image, mask, intensity_threshold, mode):
    """Drop labeled objects whose mean intensity is on the wrong side of ``intensity_threshold``.

    :param image: intensity image.
    :param mask: labeled mask aligned to ``image``.
    :param intensity_threshold: cutoff value.
    :param mode: ``'low'`` removes below-threshold objects, ``'high'`` removes above.
    :returns: filtered label mask.
    """
    # Calculate the mean intensity of each object in the original image
    props = regionprops_table(mask, image, properties=('label', 'mean_intensity'))
    # Find the labels of the objects with mean intensity below the threshold
    if mode == 'low':
        labels_to_remove = props['label'][props['mean_intensity'] < intensity_threshold]
    if mode == 'high':
        labels_to_remove = props['label'][props['mean_intensity'] > intensity_threshold]
    # Remove these objects from the mask
    mask[np.isin(mask, labels_to_remove)] = 0
    return mask
    
def _filter_closest_to_stat(df, column, n_rows, use_median=False):
    """Return the ``n_rows`` rows of ``df`` closest to the mean or median of ``column``."""
    if use_median:
        target_value = df[column].median()
    else:
        target_value = df[column].mean()
    df['diff'] = (df[column] - target_value).abs()
    result_df = df.sort_values(by='diff').head(n_rows)
    result_df = result_df.drop(columns=['diff'])
    return result_df
    
def _find_similar_sized_images(file_list):
    """Return the largest group of image paths sharing the same cropped size/aspect ratio."""
    # Dictionary to hold image sizes and their paths
    size_to_paths = defaultdict(list)
    # Iterate over image paths to get their dimensions
    for path in file_list:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Read with unchanged color space to support different image types
        if img is not None:
            # Find indices where the image is not padded (non-zero)
            if img.ndim == 3:  # Color image
                mask = np.any(img != 0, axis=2)
            else:  # Grayscale image
                mask = img != 0
            # Find the bounding box of non-zero regions
            coords = np.argwhere(mask)
            if coords.size == 0:  # Skip images that are completely padded
                continue
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1  # Add 1 because slice end index is exclusive
            # Crop the image to remove padding
            cropped_img = img[y0:y1, x0:x1]
            # Get dimensions of the cropped image
            height, width = cropped_img.shape[:2]
            aspect_ratio = width / height
            size_key = (width, height, round(aspect_ratio, 2))  # Group by width, height, and aspect ratio
            size_to_paths[size_key].append(path)
    # Find the largest group of images with the most similar size and shape
    largest_group = max(size_to_paths.values(), key=len)
    return largest_group
    
def _relabel_parent_with_child_labels(parent_mask, child_mask):
    """Relabel parent objects to match their overlapping child labels."""
    # Label parent mask to identify unique objects
    parent_labels = label(parent_mask, background=0)
    # Use the original child mask labels directly, without relabeling
    child_labels = child_mask

    # Create a new parent mask for updated labels
    parent_mask_new = np.zeros_like(parent_mask)

    # Directly relabel parent cells based on overlapping child labels
    unique_child_labels = np.unique(child_labels)[1:]  # Skip background
    for child_label in unique_child_labels:
        child_area_mask = (child_labels == child_label)
        overlapping_parent_label = np.unique(parent_labels[child_area_mask])

        # Since each parent is assumed to overlap with exactly one nucleus,
        # directly set the parent label to the child label where overlap occurs
        for parent_label in overlapping_parent_label:
            if parent_label != 0:  # Skip background
                parent_mask_new[parent_labels == parent_label] = child_label

    # For cells containing multiple nucleus, standardize all nucleus to the first label
    # This will be done only if needed, as per your condition
    for parent_label in np.unique(parent_mask_new)[1:]:  # Skip background
        parent_area_mask = (parent_mask_new == parent_label)
        child_labels_in_parent = np.unique(child_mask[parent_area_mask])
        child_labels_in_parent = child_labels_in_parent[child_labels_in_parent != 0]  # Exclude background

        if len(child_labels_in_parent) > 1:
            # Standardize to the first child label within this parent
            first_child_label = child_labels_in_parent[0]
            for child_label in child_labels_in_parent:
                child_mask[child_mask == child_label] = first_child_label

    return parent_mask_new, child_mask
    
def _exclude_objects(cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask, uninfected=True):
    """Drop cells missing required companion objects and clear other masks outside kept cells."""
    # Remove cells with no nucleus or cytoplasm (or pathogen)
    filtered_cells = np.zeros_like(cell_mask) # Initialize a new mask to store the filtered cells.
    for cell_label in np.unique(cell_mask): # Iterate over all cell labels in the cell mask.
        if cell_label == 0: # Skip background
            continue
        cell_region = cell_mask == cell_label # Get a mask for the current cell.
        # Check existence of nucleus, cytoplasm and pathogen in the current cell.
        has_nucleus = np.any(nucleus_mask[cell_region])
        has_cytoplasm = np.any(cytoplasm_mask[cell_region])
        has_pathogen = np.any(pathogen_mask[cell_region])
        if uninfected:
            if has_nucleus and has_cytoplasm:
                filtered_cells[cell_region] = cell_label
        else:
            if has_nucleus and has_cytoplasm and has_pathogen:
                filtered_cells[cell_region] = cell_label
    # Remove objects outside of cells
    nucleus_mask = nucleus_mask * (filtered_cells > 0)
    pathogen_mask = pathogen_mask * (filtered_cells > 0)
    cytoplasm_mask = cytoplasm_mask * (filtered_cells > 0)
    return filtered_cells, nucleus_mask, pathogen_mask, cytoplasm_mask

def _merge_overlapping_objects(mask1, mask2):
    """Merge overlapping objects across two masks using a 90% overlap heuristic."""
    labeled_1 = label(mask1)
    num_1 = np.max(labeled_1)
    for m1_id in range(1, num_1 + 1):
        current_1_mask = labeled_1 == m1_id
        overlapping_2_labels = np.unique(mask2[current_1_mask])
        overlapping_2_labels = overlapping_2_labels[overlapping_2_labels != 0]
        if len(overlapping_2_labels) > 1:
            overlap_percentages = [np.sum(current_1_mask & (mask2 == m2_label)) / np.sum(current_1_mask) * 100 for m2_label in overlapping_2_labels]
            max_overlap_label = overlapping_2_labels[np.argmax(overlap_percentages)]
            max_overlap_percentage = max(overlap_percentages)
            if max_overlap_percentage >= 90:
                for m2_label in overlapping_2_labels:
                    if m2_label != max_overlap_label:
                        mask1[(current_1_mask) & (mask2 == m2_label)] = 0
            else:
                for m2_label in overlapping_2_labels[1:]:
                    mask2[mask2 == m2_label] = overlapping_2_labels[0]
    return mask1, mask2

def _filter_object(mask, min_value):
    """Zero out label values whose pixel count is below ``min_value``."""
    count = np.bincount(mask.ravel())
    to_remove = np.where(count < min_value)
    mask[np.isin(mask, to_remove)] = 0
    return mask

def _filter_cp_masks(masks, flows, filter_size, filter_intensity, minimum_size, maximum_size, remove_border_objects, merge, batch, plot, figuresize):
    """Post-process Cellpose masks: optional merge, size filter, intensity filter, border removal."""
    
    from .plot import plot_masks
    
    mask_stack = []
    for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
        
        if plot and idx == 0:
            num_objects = mask_object_count(mask)
            print(f'Number of objects before filtration: {num_objects}')
            plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)

        if merge:
            mask = merge_touching_objects(mask, threshold=0.66)
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after merging adjacent objects, : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)

        if filter_size:
            props = measure.regionprops_table(mask, properties=['label', 'area'])
            valid_labels = props['label'][np.logical_and(props['area'] > minimum_size, props['area'] < maximum_size)] 
            mask = np.isin(mask, valid_labels) * mask
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after size filtration >{minimum_size} and <{maximum_size} : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)

        if filter_intensity:
            intensity_image = image[:, :, 1]  
            props = measure.regionprops_table(mask, intensity_image=intensity_image, properties=['label', 'mean_intensity'])
            mean_intensities = np.array(props['mean_intensity']).reshape(-1, 1)

            if mean_intensities.shape[0] >= 2:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(mean_intensities)
                centroids = kmeans.cluster_centers_
            
                # Calculate the Euclidean distance between the two centroids
                dist_between_centroids = distance.euclidean(centroids[0], centroids[1])
                
                # Set a threshold for the minimum distance to consider clusters distinct
                distance_threshold = 0.25 
                
                if dist_between_centroids > distance_threshold:
                    high_intensity_cluster = np.argmax(centroids)
                    valid_labels = np.array(props['label'])[kmeans.labels_ == high_intensity_cluster]
                    mask = np.isin(mask, valid_labels) * mask

            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                props_after = measure.regionprops_table(mask, intensity_image=intensity_image, properties=['label', 'mean_intensity'])
                mean_intensities_after = np.mean(np.array(props_after['mean_intensity']))
                average_intensity_before = np.mean(mean_intensities)
                print(f'Number of objects after potential intensity clustering: {num_objects}. Mean intensity before:{average_intensity_before:.4f}. After:{mean_intensities_after:.4f}.')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)


        if remove_border_objects:
            mask = clear_border(mask)
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after removing border objects, : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        
        mask_stack.append(mask)

    return mask_stack
    
def _object_filter(df, object_type, size_range, intensity_range, mask_chans, mask_chan):
    """
    Filter the DataFrame based on object type, size range, and intensity range.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        object_type (str): The type of object to filter.
        size_range (list or None): The range of object sizes to filter.
        intensity_range (list or None): The range of object intensities to filter.
        mask_chans (list): The list of mask channels.
        mask_chan (int): The index of the mask channel to use.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    if not size_range is None:
        if isinstance(size_range, list):
            if isinstance(size_range[0], int): 
                df = df[df[f'{object_type}_area'] > size_range[0]]
                print(f'After {object_type} minimum area filter: {len(df)}')
            if isinstance(size_range[1], int):
                df = df[df[f'{object_type}_area'] < size_range[1]]
                print(f'After {object_type} maximum area filter: {len(df)}')
    if not intensity_range is None:
        if isinstance(intensity_range, list):
            if isinstance(intensity_range[0], int):
                df = df[df[f'{object_type}_channel_{mask_chans[mask_chan]}_mean_intensity'] > intensity_range[0]]
                print(f'After {object_type} minimum mean intensity filter: {len(df)}')
            if isinstance(intensity_range[1], int):
                df = df[df[f'{object_type}_channel_{mask_chans[mask_chan]}_mean_intensity'] < intensity_range[1]]
                print(f'After {object_type} maximum mean intensity filter: {len(df)}')
    return df

def _get_regex(metadata_type, img_format, custom_regex=None):
    
    print(f"Image_format: {img_format}")

    if img_format == None:
        img_format == 'tif'
    if metadata_type == 'cellvoyager':
        regex = f"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*).{img_format}"
    elif metadata_type == 'cq1':
        regex = f"W(?P<wellID>.*)F(?P<fieldID>.*)T(?P<timeID>.*)Z(?P<sliceID>.*)C(?P<chanID>.*).{img_format}"
    elif metadata_type == 'auto':
        regex = f"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)L(?P<laserID>.*)C(?P<chanID>.*).tif"     
    elif metadata_type == 'custom':
        regex = f"({custom_regex}).{img_format}"
        
    print(f'regex mode:{metadata_type} regex:{regex}')
    return regex

def _run_test_mode(src, regex, timelapse=False, test_images=10, random_test=True):
    
    if timelapse:
        test_images = 1  # Use only 1 set for timelapse to ensure full sequence inclusion
    
    test_folder_path = os.path.join(src, 'test')
    os.makedirs(test_folder_path, exist_ok=True)
    regular_expression = re.compile(regex)

    if os.path.exists(os.path.join(src, 'orig')):
        src = os.path.join(src, 'orig')
        
    all_filenames = [filename for filename in os.listdir(src) if regular_expression.match(filename)]
    print(f'Found {len(all_filenames)} files')
    images_by_set = defaultdict(list)

    for filename in all_filenames:
        match = regular_expression.match(filename)
        if match:
            plate = match.group('plateID') if 'plateID' in match.groupdict() else os.path.basename(src)
            well = match.group('wellID')
            field = match.group('fieldID')
            set_identifier = (plate, well, field)
            images_by_set[set_identifier].append(filename)
    
    # Prepare for random selection
    set_identifiers = list(images_by_set.keys())
    if random_test:
        random.seed(42)
    random.shuffle(set_identifiers)  # Randomize the order
    
    # Select a subset based on the test_images count
    selected_sets = set_identifiers[:test_images]

    # Print information about the number of sets used
    print(f'Using {len(selected_sets)} random image set(s) for test model')

    # Copy files for selected sets to the test folder
    for set_identifier in selected_sets:
        for filename in images_by_set[set_identifier]:
            shutil.copy(os.path.join(src, filename), test_folder_path)

    return test_folder_path

def _choose_model(model_name, device, object_type='cell', restore_type=None, object_settings=None):
    if object_settings is None:
        object_settings = {}
    if object_type == 'pathogen':
        if model_name == 'toxo_pv_lumen':
            diameter = object_settings['diameter']
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, 'models', 'cp', 'toxo_pv_lumen.CP_model')
            print(model_path)
            model = cp_models.CellposeModel(
                gpu=torch.cuda.is_available(),
                model_type=None,
                pretrained_model=model_path,
                diam_mean=diameter,
                device=device
            )
            print('Using Toxoplasma PV lumen model to generate pathogen masks')
            return model

    restore_list = ['denoise', 'deblur', 'upsample', None]
    if restore_type not in restore_list:
        print(f"Invalid restore type. Choose from {restore_list}, defaulting to None")
        restore_type = None

    if restore_type is None:
        if model_name == 'sam':
            model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), device=device, pretrained_model='cpsam',)
            return model
        if model_name in ['cyto', 'cyto2', 'cyto3', 'nuclei']:
            model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type=model_name, device=device)
            return model
    else:
        if object_type == 'nucleus':
            restore = f'{restore_type}_nuclei'
            model = denoise.CellposeDenoiseModel(
                gpu=torch.cuda.is_available(),
                model_type="nuclei",
                restore_type=restore,
                chan2_restore=False,
                device=device
            )
            return model
        else:
            restore = f'{restore_type}_cyto3'
            chan2_restore = (model_name == 'cyto2')
            model = denoise.CellposeDenoiseModel(
                gpu=torch.cuda.is_available(),
                model_type="cyto3",
                restore_type=restore,
                chan2_restore=chan2_restore,
                device=device
            )
            return model

class SelectChannels:
    """Callable transform that zeroes out image channels not present in ``channels``.

    :param channels: iterable of 1-based channel indices to keep (1=red, 2=green, 3=blue).
    """
    def __init__(self, channels):
        """Store the list of channels to preserve."""
        self.channels = channels

    def __call__(self, img):
        """Return ``img`` with unselected RGB channels zeroed."""
        img = img.clone()
        if 1 not in self.channels:
            img[0, :, :] = 0  # Zero out the red channel
        if 2 not in self.channels:
            img[1, :, :] = 0  # Zero out the green channel
        if 3 not in self.channels:
            img[2, :, :] = 0  # Zero out the blue channel
        return img

class SaliencyMapGenerator:
    """Generate saliency maps and predictions for a binary classifier.

    :param model: trained PyTorch model with a single-logit binary output.
    """
    def __init__(self, model):
        """Store the model to be probed."""
        self.model = model

    def compute_saliency_maps(self, X, y):
        """Return absolute-gradient saliency maps for inputs ``X`` given labels ``y``."""
        self.model.eval()
        X.requires_grad_()

        # Forward pass
        scores = self.model(X).squeeze()

        # For binary classification, target scores can be the single output
        target_scores = scores * (2 * y - 1)

        self.model.zero_grad()
        target_scores.backward(torch.ones_like(target_scores))

        saliency = X.grad.abs()
        return saliency

    def compute_saliency_and_predictions(self, X):
        """Return ``(saliency, predictions)`` computed against the model's own predicted classes."""
        self.model.eval()
        X.requires_grad_()

        # Forward pass to get predictions (logits)
        scores = self.model(X).squeeze()

        # Get predicted class (0 or 1 for binary classification)
        predictions = (scores > 0).long()

        # Compute saliency maps
        self.model.zero_grad()
        target_scores = scores * (2 * predictions - 1)
        target_scores.backward(torch.ones_like(target_scores))

        saliency = X.grad.abs()

        return saliency, predictions

    def plot_activation_grid(self, X, saliency, predictions, overlay=True, normalize=False):
        """Render a grid overlaying saliency maps on inputs with predicted-class labels."""
        N = X.shape[0]
        rows = (N + 7) // 8 
        fig, axs = plt.subplots(rows, 8, figsize=(16, rows * 2))

        for i in range(N):
            ax = axs[i // 8, i % 8]
            saliency_map = saliency[i].cpu().numpy()  # Move to CPU and convert to numpy

            if saliency_map.shape[0] == 3:  # Channels first, reshape to (H, W, 3)
                saliency_map = np.transpose(saliency_map, (1, 2, 0))

            # Normalize image channels to 2nd and 98th percentiles
            if overlay:
                img_np = X[i].permute(1, 2, 0).detach().cpu().numpy()
                if normalize:
                    img_np = self.percentile_normalize(img_np)
                ax.imshow(img_np)
                ax.imshow(saliency_map, cmap='jet', alpha=0.5)

            # Add class label in the top-left corner
            ax.text(5, 25, str(predictions[i].item()), fontsize=12, color='white', weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
            ax.axis('off')

        plt.tight_layout(pad=0)
        return fig
    
    def percentile_normalize(self, img, lower_percentile=2, upper_percentile=98):
        """Per-channel percentile-normalize ``img`` into ``[0, 1]``."""
        img_normalized = np.zeros_like(img)

        for c in range(img.shape[2]):  # Iterate over each channel
            low = np.percentile(img[:, :, c], lower_percentile)
            high = np.percentile(img[:, :, c], upper_percentile)
            img_normalized[:, :, c] = np.clip((img[:, :, c] - low) / (high - low), 0, 1)

        return img_normalized

class GradCAMGenerator:
    """Grad-CAM (and variants) map generator for binary classifiers.

    :param model: trained model to inspect.
    :param target_layer: dotted attribute path to the convolutional layer to probe.
    :param cam_type: variant identifier, e.g. ``'gradcam'``.
    """
    def __init__(self, model, target_layer, cam_type='gradcam'):
        """Store the model, resolve the target layer, and register activation/gradient hooks."""
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.cam_type = cam_type
        self.gradients = None
        self.activations = None

        # Hook the target layer
        self.target_layer_module = self.get_layer(self.model, self.target_layer)
        self.hook_layers()

    def hook_layers(self):
        """Register forward/backward hooks that capture activations and gradients."""
        # Forward hook to get activations
        def forward_hook(module, input, output):
            """Forward hook: cache the target layer's output activations."""
            self.activations = output

        # Backward hook to get gradients
        def backward_hook(module, grad_input, grad_output):
            """Backward hook: cache the gradient flowing into the target layer's output."""
            self.gradients = grad_output[0]

        self.target_layer_module.register_forward_hook(forward_hook)
        self.target_layer_module.register_backward_hook(backward_hook)

    def get_layer(self, model, target_layer):
        """Resolve a dotted attribute path into the referenced submodule."""
        # Recursively find the layer specified in target_layer
        modules = target_layer.split('.')
        layer = model
        for module in modules:
            layer = getattr(layer, module)
        return layer

    def compute_gradcam_maps(self, X, y):
        """Return the min-max normalized Grad-CAM map for a single-sample batch ``X`` and label ``y``."""
        X.requires_grad_()

        # Forward pass
        scores = self.model(X).squeeze()

        # Perform backward pass
        target_scores = scores * (2 * y - 1)
        self.model.zero_grad()
        target_scores.backward(torch.ones_like(target_scores))

        # Compute GradCAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        gradcam = torch.mean(self.activations, dim=1).squeeze()
        gradcam = F.relu(gradcam)
        gradcam = F.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=X.shape[2:], mode='bilinear')
        gradcam = gradcam.squeeze().cpu().detach().numpy()
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

        return gradcam

    def compute_gradcam_and_predictions(self, X):
        """Return ``(gradcam_maps, predictions)`` for every sample in the batch ``X``."""
        self.model.eval()
        X.requires_grad_()

        # Forward pass to get predictions (logits)
        scores = self.model(X).squeeze()

        # Get predicted class (0 or 1 for binary classification)
        predictions = (scores > 0).long()

        # Compute gradcam maps
        gradcam_maps = []
        for i in range(X.size(0)):
            gradcam_map = self.compute_gradcam_maps(X[i].unsqueeze(0), predictions[i])
            gradcam_maps.append(gradcam_map)

        return torch.tensor(gradcam_maps), predictions

    def plot_activation_grid(self, X, gradcam, predictions, overlay=True, normalize=False):
        """Render a grid overlaying Grad-CAM maps on inputs with predicted-class labels."""
        N = X.shape[0]
        rows = (N + 7) // 8
        fig, axs = plt.subplots(rows, 8, figsize=(16, rows * 2))

        for i in range(N):
            ax = axs[i // 8, i % 8]
            gradcam_map = gradcam[i].cpu().numpy()

            # Normalize image channels to 2nd and 98th percentiles
            if overlay:
                img_np = X[i].permute(1, 2, 0).detach().cpu().numpy()
                if normalize:
                    img_np = self.percentile_normalize(img_np)
                ax.imshow(img_np)
                ax.imshow(gradcam_map, cmap='jet', alpha=0.5)

            #ax.imshow(X[i].permute(1, 2, 0).detach().cpu().numpy())  # Original image
            #ax.imshow(gradcam_map, cmap='jet', alpha=0.5)  # Overlay the gradcam map

            # Add class label in the top-left corner
            ax.text(5, 25, str(predictions[i].item()), fontsize=12, color='white', weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
            ax.axis('off')

        plt.tight_layout(pad=0)
        return fig
    
    def percentile_normalize(self, img, lower_percentile=2, upper_percentile=98):
        """Per-channel percentile-normalize ``img`` into ``[0, 1]``."""
        img_normalized = np.zeros_like(img)

        for c in range(img.shape[2]):  # Iterate over each channel
            low = np.percentile(img[:, :, c], lower_percentile)
            high = np.percentile(img[:, :, c], upper_percentile)
            img_normalized[:, :, c] = np.clip((img[:, :, c] - low) / (high - low), 0, 1)

        return img_normalized

def preprocess_image(image_path, normalize=True, image_size=224, channels=None):
    """Load and preprocess ``image_path`` into a batched tensor ready for classification.

    :param image_path: path to the source image.
    :param normalize: apply ImageNet mean/std normalization.
    :param image_size: square resize dimension.
    :param channels: reserved for downstream use; kept for API compatibility.
    :returns: ``(pil_image, input_tensor)`` where the tensor has shape ``(1, 3, H, W)``.
    """
    if channels is None:
        channels = [1,2,3]
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    if normalize:
        input_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_tensor)
    input_tensor = input_tensor.unsqueeze(0)
    
    return image, input_tensor

def class_visualization(target_y, model_path, dtype, img_size=224, channels=None, l2_reg=1e-3, learning_rate=25, num_iterations=100, blur_every=10, max_jitter=16, show_every=25, class_names = None):
    """Synthesize an input image that maximizes the classifier score for ``target_y``.

    :param target_y: target class index.
    :param model_path: path to the trained model checkpoint.
    :param dtype: tensor dtype; overridden internally based on CUDA availability.
    :param img_size: square image size (pixels).
    :param channels: input channels (defaults to ``[0, 1, 2]``).
    :param l2_reg: L2 regularization weight on the pixel norm.
    :param learning_rate: gradient-ascent step size.
    :param num_iterations: optimization iteration count.
    :param blur_every: interval (iterations) between periodic Gaussian blurs.
    :param max_jitter: maximum pixel jitter applied per iteration.
    :param show_every: interval (iterations) between preview plots.
    :param class_names: display names for the classes (defaults to ``['nc', 'pc']``).
    :returns: deprocessed image as a numpy array.
    """
    if channels is None:
        channels = [0,1,2]
    if class_names is None:
        class_names = ['nc', 'pc']
    def jitter(img, ox, oy):
        """Return ``img`` shifted (rolled) by ``ox`` and ``oy`` pixels along the spatial axes."""
        return torch.roll(torch.roll(img, ox, dims=2), oy, dims=3)

    def blur_image(img, sigma=1):
        """In-place Gaussian blur of each channel of ``img`` with standard deviation ``sigma``."""
        img_np = img.cpu().numpy()
        for i in range(img_np.shape[1]):
            img_np[:, i] = gaussian_filter(img_np[:, i], sigma=sigma)
        img.copy_(torch.tensor(img_np).to(img.device))

    def deprocess(img_tensor):
        """Undo ImageNet normalization and return an ``(H, W, 3)`` numpy image in ``[0, 1]``."""
        img_tensor = img_tensor.clone()
        for c in range(3):
            img_tensor[:, c] = img_tensor[:, c] * SQUEEZENET_STD[c] + SQUEEZENET_MEAN[c]
        img_tensor = img_tensor.clamp(0, 1)
        return img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Assuming these are defined somewhere in your codebase
    SQUEEZENET_MEAN = [0.485, 0.456, 0.406]
    SQUEEZENET_STD = [0.229, 0.224, 0.225]
    
    model = torch.load(model_path)
    
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    len_chans = len(channels)
    model.type(dtype)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.randn(1, len_chans, img_size, img_size).mul_(1.0).type(dtype).requires_grad_()

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        # Forward pass
        score = model(img)
        
        if target_y == 0:
            target_score = -score
        else:
            target_score = score

        # Add regularization
        target_score = target_score - l2_reg * torch.norm(img)

        # Backward pass
        target_score.backward()

        # Gradient ascent step
        with torch.no_grad():
            img += learning_rate * img.grad / torch.norm(img.grad)
            img.grad.zero_()

        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for c in range(3):
            lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
            hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
            img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)
        
        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess(img.data.clone().cpu()))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()

    return deprocess(img.data.cpu())

def get_submodules(model, prefix=''):
    """Return all dotted submodule names of ``model`` in traversal order.

    :param model: PyTorch module to walk.
    :param prefix: optional prefix prepended to returned names.
    :returns: list of dotted submodule names.
    """
    submodules = []
    for name, module in model.named_children():
        full_name = prefix + ('.' if prefix else '') + name
        submodules.append(full_name)
        submodules.extend(get_submodules(module, full_name))
    return submodules

class GradCAM:
    """Named-hook Grad-CAM implementation for arbitrary target layers.

    :param model: trained model to inspect.
    :param target_layers: list of dotted layer names to hook.
    :param use_cuda: run the model on CUDA when available.
    """
    def __init__(self, model, target_layers=None, use_cuda=True):
        """Store the model and move it to CUDA if requested."""
        self.model = model
        self.model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def forward(self, input):
        """Return the model output for ``input``."""
        return self.model(input)

    def __call__(self, x, index=None):
        """Return the normalized CAM heatmap for input ``x``, targeting class ``index``."""
        if self.cuda:
            x = x.cuda()

        features = []
        def hook(module, input, output):
            """Forward hook: append the target layer's output to ``features``."""
            features.append(output)

        handles = []
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                handles.append(module.register_forward_hook(hook))

        output = self.forward(x)
        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = features[0].grad.cpu().data.numpy()
        target = features[0].cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.size(2), x.size(3)))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        for handle in handles:
            handle.remove()
            
        return cam

def show_cam_on_image(img, mask):
    """Return ``img`` overlaid with a jet colormap of ``mask`` as an 8-bit RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def recommend_target_layers(model):
    """Return ``([last_conv_layer], all_conv_layers)`` from ``model``.

    :param model: PyTorch module to scan for ``Conv2d`` layers.
    :returns: tuple ``(recommended, all)`` of layer-name lists.
    :raises ValueError: if the model contains no convolutional layers.
    """
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layers.append(name)
    # Choose the last conv layer as the recommended target layer
    if target_layers:
        return [target_layers[-1]], target_layers
    else:
        raise ValueError("No convolutional layers found in the model.")
    
class IntegratedGradients:
    """Compute integrated-gradients attributions for a classifier.

    :param model: trained PyTorch model.
    """
    def __init__(self, model):
        """Store the model and switch it to eval mode."""
        self.model = model
        self.model.eval()

    def generate_integrated_gradients(self, input_tensor, target_label_idx, baseline=None, num_steps=50):
        """Return integrated gradients from ``baseline`` to ``input_tensor`` for ``target_label_idx``.

        :param input_tensor: input sample tensor.
        :param target_label_idx: target class index whose logit is attributed.
        :param baseline: reference tensor (defaults to zeros of the same shape).
        :param num_steps: number of Riemann-sum interpolation steps.
        :returns: attribution ndarray with the shape of ``input_tensor``.
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        assert baseline.shape == input_tensor.shape

        # Scale input and compute gradients
        scaled_inputs = [(baseline + (float(i) / num_steps) * (input_tensor - baseline)).requires_grad_(True) for i in range(0, num_steps + 1)]
        grads = []
        for scaled_input in scaled_inputs:
            out = self.model(scaled_input)
            self.model.zero_grad()
            out[0, target_label_idx].backward(retain_graph=True)
            grads.append(scaled_input.grad.data.cpu().numpy())

        avg_grads = np.mean(grads[:-1], axis=0)
        integrated_grads = (input_tensor.cpu().data.numpy() - baseline.cpu().data.numpy()) * avg_grads
        return integrated_grads

def get_db_paths(src):
    """Return the standard ``measurements/measurements.db`` paths for one or more source roots."""
    if isinstance(src, str):
        src = [src]
    db_paths = [os.path.join(source, 'measurements/measurements.db') for source in src]
    return db_paths

def get_sequencing_paths(src):
    """Return the standard ``sequencing/sequencing_data.csv`` paths for one or more source roots."""
    if isinstance(src, str):
        src = [src]
    seq_paths = [os.path.join(source, 'sequencing/sequencing_data.csv') for source in src]
    return seq_paths

def load_image_paths(c, visualize):
    """Load the ``png_list`` table into a DataFrame indexed by ``prcfo`` and optionally filter by object.

    :param c: open sqlite3 cursor.
    :param visualize: object-type prefix (``'cell'``/``'nucleus'``/...) or falsy to keep all rows.
    :returns: DataFrame of PNG metadata indexed by ``prcfo``.
    """
    c.execute(f'SELECT * FROM png_list')
    data = c.fetchall()
    columns_info = c.execute(f'PRAGMA table_info(png_list)').fetchall()
    column_names = [col_info[1] for col_info in columns_info]
    image_paths_df = pd.DataFrame(data, columns=column_names)
    if visualize:
        object_visualize = visualize + '_png'
        image_paths_df = image_paths_df[image_paths_df['png_path'].str.contains(object_visualize)]
    image_paths_df = image_paths_df.set_index('prcfo')
    return image_paths_df

def merge_dataframes(df, image_paths_df, verbose):
    """Merge ``df`` into ``image_paths_df`` on the shared ``prcfo`` index.

    :param df: feature DataFrame with a ``prcfo`` column.
    :param image_paths_df: DataFrame indexed by ``prcfo``.
    :param verbose: display the merged DataFrame.
    :returns: merged DataFrame.
    """
    df.set_index('prcfo', inplace=True)
    df = image_paths_df.merge(df, left_index=True, right_index=True)
    if verbose:
        display(df)
    return df

def filter_columns(df, filter_by):
    """Return ``df`` restricted to columns matching ``filter_by`` (or morphology columns).

    :param df: source DataFrame.
    :param filter_by: substring required in column names, or ``'morphology'`` to drop channel columns.
    :returns: column-filtered DataFrame.
    """
    if filter_by != 'morphology':
        cols_to_include = [col for col in df.columns if filter_by in str(col)]
    else:
        cols_to_include = [col for col in df.columns if 'channel' not in str(col)]
    df = df[cols_to_include]
    return df

def reduction_and_clustering(numeric_data, n_neighbors, min_dist, metric, eps, min_samples, clustering, reduction_method='umap', verbose=False, embedding=None, n_jobs=-1, mode='fit', model=False):
    """Reduce ``numeric_data`` to 2-D and cluster the embedding.

    :param numeric_data: numeric data matrix.
    :param n_neighbors: UMAP ``n_neighbors`` or t-SNE perplexity (fraction or int).
    :param min_dist: UMAP ``min_dist``.
    :param metric: distance metric used by UMAP/DBSCAN.
    :param eps: DBSCAN ``eps``.
    :param min_samples: DBSCAN ``min_samples`` or KMeans cluster count.
    :param clustering: ``'dbscan'`` or ``'kmeans'``.
    :param reduction_method: ``'umap'`` or ``'tsne'``.
    :param verbose: print progress.
    :param embedding: precomputed embedding (skips reducer fit).
    :param n_jobs: parallel worker count.
    :param mode: ``'fit'`` to train a new reducer, otherwise transform with ``model``.
    :param model: existing reducer to reuse when ``mode != 'fit'``.
    :returns: ``(embedding, labels, reducer)``.
    :raises ValueError: on unsupported ``reduction_method`` or missing model.
    """

    if verbose:
        v = 1
    else:
        v = 0
    
    if isinstance(n_neighbors, float):
        n_neighbors = int(n_neighbors * len(numeric_data))

    if n_neighbors <= 2:
        n_neighbors = 2
    
    if mode == 'fit':
        if reduction_method == 'umap':
            reducer = umap.UMAP(n_neighbors=n_neighbors,
                                n_components=2,
                                metric=metric,
                                n_epochs=None,
                                learning_rate=1.0,
                                init='spectral',
                                min_dist=min_dist,
                                spread=1.0,
                                set_op_mix_ratio=1.0,
                                local_connectivity=1,
                                repulsion_strength=1.0,
                                negative_sample_rate=5,
                                transform_queue_size=4.0,
                                a=None,
                                b=None,
                                random_state=42,
                                metric_kwds=None,
                                angular_rp_forest=False,
                                target_n_neighbors=-1,
                                target_metric='categorical',
                                target_metric_kwds=None,
                                target_weight=0.5,
                                transform_seed=42,
                                n_jobs=n_jobs,
                                verbose=verbose)

        elif reduction_method == 'tsne':
            reducer = TSNE(n_components=2,
                        perplexity=n_neighbors,
                        early_exaggeration=12.0,
                        learning_rate=200.0,
                        # scikit-learn >=1.5 renamed TSNE's ``n_iter`` to
                        # ``max_iter``; the old name is a hard error on 1.7+.
                        max_iter=1000,
                        n_iter_without_progress=300,
                        min_grad_norm=1e-7,
                        metric=metric,
                        init='random',
                        verbose=v,
                        random_state=42,
                        method='barnes_hut',
                        angle=0.5,
                        n_jobs=n_jobs)
            
        else:
            raise ValueError(f"Unsupported reduction method: {reduction_method}. Supported methods are 'umap' and 'tsne'")
        
        embedding = reducer.fit_transform(numeric_data)
        if verbose:
            print(f'Trained and fit reducer')

    else:
        if not model is None:
            embedding = model.transform(numeric_data)
            reducer = model
            if verbose:
                print(f'Fit data to reducer')
        else:
            raise ValueError(f"Model is None. Please provide a model for transform.")

    if clustering == 'dbscan':
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
    elif clustering == 'kmeans':
        clustering_model = KMeans(n_clusters=min_samples, random_state=42)
    
    clustering_model.fit(embedding)
    labels = clustering_model.labels_ if clustering == 'dbscan' else clustering_model.predict(embedding)
    
    if verbose:
        print(f'Embedding shape: {embedding.shape}')

    return embedding, labels, reducer

def remove_noise(embedding, labels):
    """Drop rows of ``embedding`` (and ``labels``) whose label is DBSCAN noise (``-1``)."""
    non_noise_indices = labels != -1
    embedding = embedding[non_noise_indices]
    labels = labels[non_noise_indices]
    return embedding, labels

def plot_embedding(embedding, image_paths, labels, image_nr, img_zoom, colors, plot_by_cluster, plot_outlines, plot_points, plot_images, smooth_lines, black_background, figuresize, dot_size, remove_image_canvas, verbose):
    """Plot a 2-D embedding with cluster outlines, points, and optional image overlays.

    :returns: matplotlib ``Figure``.
    """
    unique_labels = np.unique(labels)
    #num_clusters = len(unique_labels[unique_labels != 0])
    colors, label_to_color_index = assign_colors(unique_labels, colors)
    cluster_centers = [np.mean(embedding[labels == cluster_label], axis=0) for cluster_label in unique_labels]
    fig, ax = setup_plot(figuresize, black_background)
    plot_clusters(ax, embedding, labels, colors, cluster_centers, plot_outlines, plot_points, smooth_lines, figuresize, dot_size, verbose)
    if not image_paths is None and plot_images:
        plot_umap_images(ax, image_paths, embedding, labels, image_nr, img_zoom, colors, plot_by_cluster, remove_image_canvas, verbose)
    plt.show()
    return fig

def generate_colors(num_clusters, black_background):
    """Return an RGBA color palette for ``num_clusters`` clusters with fixed accent colors first."""
    random_colors = np.random.rand(num_clusters + 1, 4)
    random_colors[:, 3] = 1
    specific_colors = [
        [155 / 255, 55 / 255, 155 / 255, 1],
        [55 / 255, 155 / 255, 155 / 255, 1],
        [55 / 255, 155 / 255, 255 / 255, 1],
        [255 / 255, 55 / 255, 155 / 255, 1]
    ]
    random_colors = np.vstack((specific_colors, random_colors[len(specific_colors):]))
    if not black_background:
        random_colors = np.vstack(([0, 0, 0, 1], random_colors))
    return random_colors

def assign_colors(unique_labels, random_colors):
    """Return a ``(colors, label_to_index)`` mapping keyed by ``unique_labels``."""
    normalized_colors = random_colors / 255
    colors_img = [tuple(color) for color in normalized_colors]
    colors = [tuple(color) for color in random_colors]
    label_to_color_index = {label: index for index, label in enumerate(unique_labels)}
    return colors, label_to_color_index

def setup_plot(figuresize, black_background):
    """Return a ``(fig, ax)`` with light or dark theme applied globally."""
    if black_background:
        plt.rcParams.update({'figure.facecolor': 'black', 'axes.facecolor': 'black', 'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'axes.labelcolor': 'white'})
    else:
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black'})
    fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
    return fig, ax

def plot_clusters(ax, embedding, labels, colors, cluster_centers, plot_outlines, plot_points, smooth_lines, figuresize=10, dot_size=50, verbose=False):
    """Draw cluster outlines, points, and centroid labels onto ``ax`` for a 2-D embedding.

    :param ax: Matplotlib axes to draw into.
    :param embedding: ``(N, 2)`` array of 2-D points (e.g. UMAP output).
    :param labels: length-``N`` cluster labels; ``-1`` denotes noise.
    :param colors: iterable of per-cluster colors, one per unique label.
    :param cluster_centers: iterable of ``(x, y)`` centroids, one per unique label.
    :param plot_outlines: draw a hull/smoothed outline around each cluster.
    :param plot_points: render the scatter points (otherwise plotted invisibly).
    :param smooth_lines: use a smoothed hull polyline instead of the convex hull edges.
    :param figuresize: base size in inches used to scale axis label and tick fonts. Default ``10``.
    :param dot_size: scatter marker size in points. Default ``50``.
    :param verbose: unused placeholder kept for API compatibility. Default ``False``.
    :returns: None.
    """
    unique_labels = np.unique(labels)
    for cluster_label, color, center in zip(unique_labels, colors, cluster_centers):
        cluster_data = embedding[labels == cluster_label]
        # A ConvexHull needs >=3 non-collinear points; with too few or
        # collinear points (common for tiny/degenerate clusters) Qhull raises.
        # Skip the outline in that case rather than crashing the whole plot.
        if smooth_lines:
            if cluster_data.shape[0] > 2:
                try:
                    x_smooth, y_smooth = smooth_hull_lines(cluster_data)
                    if plot_outlines:
                        plt.plot(x_smooth, y_smooth, color=color, linewidth=2)
                except Exception:
                    pass
        else:
            if cluster_data.shape[0] > 2:
                try:
                    hull = ConvexHull(cluster_data)
                    for simplex in hull.simplices:
                        if plot_outlines:
                            plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], color=color, linewidth=4)
                except Exception:
                    pass
        if plot_points:
            scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=dot_size, c=[color], alpha=0.5, label=f'Cluster {cluster_label if cluster_label != -1 else "Noise"}')
        else:
            scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=dot_size, c=[color], alpha=0, label=f'Cluster {cluster_label if cluster_label != -1 else "Noise"}')
        ax.text(center[0], center[1], str(cluster_label), fontsize=12, ha='center', va='center')
    plt.legend(loc='best', fontsize=int(figuresize * 0.75))
    plt.xlabel('UMAP Dimension 1', fontsize=int(figuresize * 0.75))
    plt.ylabel('UMAP Dimension 2', fontsize=int(figuresize * 0.75))
    plt.tick_params(axis='both', which='major', labelsize=int(figuresize * 0.75))

def plot_umap_images(ax, image_paths, embedding, labels, image_nr, img_zoom, colors, plot_by_cluster, remove_image_canvas, verbose):
    """Overlay sample images from ``image_paths`` on the UMAP embedding in ``ax``."""
    if plot_by_cluster:
        cluster_indices = {label: np.where(labels == label)[0] for label in np.unique(labels) if label != -1}
        plot_images_by_cluster(ax, image_paths, embedding, labels, image_nr, img_zoom, colors, cluster_indices, remove_image_canvas, verbose)
    else:
        indices = random.sample(range(len(embedding)), image_nr)
        for i, index in enumerate(indices):
            x, y = embedding[index]
            img = Image.open(image_paths[index])
            plot_image(ax, x, y, img, img_zoom, remove_image_canvas)

def plot_images_by_cluster(ax, image_paths, embedding, labels, image_nr, img_zoom, colors, cluster_indices, remove_image_canvas, verbose):
    """Overlay up to ``image_nr`` images per cluster on the embedding in ``ax``."""
    for cluster_label, color in zip(np.unique(labels), colors):
        if cluster_label == -1:
            continue
        indices = cluster_indices.get(cluster_label, [])
        if len(indices) > image_nr:
            indices = random.sample(list(indices), image_nr)
        for index in indices:
            x, y = embedding[index]
            img = Image.open(image_paths[index])
            plot_image(ax, x, y, img, img_zoom, remove_image_canvas)

def plot_image(ax, x, y, img, img_zoom, remove_image_canvas=True):
    """Place a zoomed thumbnail of ``img`` at ``(x, y)`` on ``ax``."""
    img = np.array(img)
    if remove_image_canvas:
        img = remove_canvas(img)
    imagebox = OffsetImage(img, zoom=img_zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

def remove_canvas(img):
    """Return ``img`` as an RGBA array whose alpha channel masks out zero pixels."""
    if img.mode in ['L', 'I']:
        img_data = np.array(img)
        img_data = img_data / np.max(img_data)
        alpha_channel = (img_data > 0).astype(float)
        img_data_rgb = np.stack([img_data] * 3, axis=-1)
        img_data_with_alpha = np.dstack([img_data_rgb, alpha_channel])
    elif img.mode == 'RGB':
        img_data = np.array(img)
        img_data = img_data / 255.0
        alpha_channel = (np.sum(img_data, axis=-1) > 0).astype(float)
        img_data_with_alpha = np.dstack([img_data, alpha_channel])
    else:
        raise ValueError(f"Unsupported image mode: {img.mode}")
    return img_data_with_alpha

def plot_clusters_grid(embedding, labels, image_nr, image_paths, colors, figuresize, black_background, verbose):
    """Plot a grid of example images per cluster label discovered in ``labels``."""
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels[unique_labels != -1])
    if num_clusters == 0:
        print("No clusters found.")
        return
    cluster_images = {label: [] for label in unique_labels if label != -1}
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels if label != -1}
    for cluster_label, indices in cluster_indices.items():
        if cluster_label == -1:
            continue
        if len(indices) > image_nr:
            indices = random.sample(list(indices), image_nr)
        for index in indices:
            img_path = image_paths[index]
            img_array = Image.open(img_path)
            img = np.array(img_array)
            cluster_images[cluster_label].append(img)
    fig = plot_grid(cluster_images, colors, figuresize, black_background, verbose)
    return fig

def plot_grid(cluster_images, colors, figuresize, black_background, verbose):
    """Render one column per cluster of representative images with colored borders and labels."""
    num_clusters = len(cluster_images)
    max_figsize = 200  # Set a maximum figure size
    if figuresize * num_clusters > max_figsize:
        figuresize = max_figsize / num_clusters

    grid_fig, grid_axes = plt.subplots(1, num_clusters, figsize=(figuresize * num_clusters, figuresize), gridspec_kw={'wspace': 0.2, 'hspace': 0})
    if num_clusters == 1:
        grid_axes = [grid_axes]  # Ensure grid_axes is always iterable
    for cluster_label, axes in zip(cluster_images.keys(), grid_axes):
        images = cluster_images[cluster_label]
        num_images = len(images)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        image_size = 0.9 / grid_size
        whitespace = (1 - grid_size * image_size) / (grid_size + 1)

        if isinstance(cluster_label, str):
            idx = list(cluster_images.keys()).index(cluster_label)
            color = colors[idx]
            if verbose:
                print(f'Lable: {cluster_label} index: {idx}')
        else:
            color = colors[cluster_label]

        axes.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes.transAxes, color=color[:3]))
        axes.axis('off')
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            x_pos = (col + 1) * whitespace + col * image_size
            y_pos = 1 - ((row + 1) * whitespace + (row + 1) * image_size)
            ax_img = axes.inset_axes([x_pos, y_pos, image_size, image_size], transform=axes.transAxes)
            ax_img.imshow(img, cmap='gray', aspect='auto')
            ax_img.axis('off')
            ax_img.set_aspect('equal')
            ax_img.set_facecolor(color[:3])
    
    # Add cluster labels beside the UMAP plot
    spacing_factor = 0.5  # Adjust this value to control the spacing between labels
    for i, (cluster_label, color) in enumerate(zip(cluster_images.keys(), colors)):
        label_y = 1 - (i + 1) * (spacing_factor / num_clusters)  # Adjust y position for each label
        grid_fig.text(1.05, label_y, f'Cluster {cluster_label}', verticalalignment='center', fontsize=figuresize, color='black' if not black_background else 'white')
        grid_fig.patches.append(plt.Rectangle((1, label_y - 0.02), 0.03, 0.03, transform=grid_fig.transFigure, color=color[:3], clip_on=False))

    plt.show()
    return grid_fig

def generate_path_list_from_db(db_path, file_metadata):
    """Return all ``png_path`` values from ``db_path`` optionally filtered by ``file_metadata`` substrings.

    :param db_path: path to the measurements SQLite DB.
    :param file_metadata: substring or list of substrings to LIKE-match against ``png_path``.
    :returns: list of PNG paths.
    """
    all_paths = []

    # Connect to the database and retrieve the image paths
    print(f"Reading DataBase: {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            if file_metadata:
                if isinstance(file_metadata, str):
                    # If file_metadata is a single string
                    cursor.execute("SELECT png_path FROM png_list WHERE png_path LIKE ?", (f"%{file_metadata}%",))
                elif isinstance(file_metadata, list):
                    # If file_metadata is a list of strings
                    query = "SELECT png_path FROM png_list WHERE " + " OR ".join(
                        ["png_path LIKE ?" for _ in file_metadata])
                    params = [f"%{meta}%" for meta in file_metadata]
                    cursor.execute(query, params)
            else:
                # If file_metadata is None or empty
                cursor.execute("SELECT png_path FROM png_list")

            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                all_paths.extend([row[0] for row in rows])

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    return all_paths

def correct_paths(df, base_path, folder='data'):
    """Rewrite PNG paths (in a DataFrame or list) so they live under ``base_path/folder``.

    :param df: DataFrame with a ``png_path`` column, or a list of paths.
    :param base_path: destination root to prepend.
    :param folder: intermediate folder name that anchors the rewrite.
    :returns: DataFrame + list, or list, mirroring the input type.
    """
    if isinstance(df, pd.DataFrame):

        if 'png_path' not in df.columns:
            print("No 'png_path' column found in the dataframe.")
            return df, None
        else:
            image_paths = df['png_path'].to_list()

    elif isinstance(df, list):
        image_paths = df
    
    adjusted_image_paths = []
    for path in image_paths:
        if base_path not in path:
            parts = path.split(f'/{folder}/')
            if len(parts) > 1:
                new_path = os.path.join(base_path, f'{folder}', parts[1])
                adjusted_image_paths.append(new_path)
            else:
                adjusted_image_paths.append(path)
        else:
            adjusted_image_paths.append(path)

    if isinstance(df, pd.DataFrame):
        df['png_path'] = adjusted_image_paths
        return df, adjusted_image_paths
    else:
        return adjusted_image_paths

def delete_folder(folder_path):
    """Recursively delete ``folder_path`` if it exists (files and subdirectories included)."""
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist or is not a directory.")

def measure_test_mode(settings):
    """Copy a random subset of source files into a ``test/merged`` folder when ``test_mode`` is on.

    :param settings: settings dict; must contain ``src``, ``test_mode``, ``test_nr``.
    :returns: settings dict with ``src`` optionally redirected to the test folder.
    """
    if settings['test_mode']:
        if not os.path.basename(settings['src']) == 'test':
            all_files = os.listdir(settings['src'])
            random_files = random.sample(all_files, settings['test_nr'])

            src = os.path.join(os.path.dirname(settings['src']),'test', 'merged')
            if os.path.exists(src):
                delete_folder(src)
            os.makedirs(src, exist_ok=True)

            for file in random_files:
                shutil.copy(os.path.join(settings['src'], file), os.path.join(src,file))

            settings['src'] = src
            print(f'Changed source folder to {src} for test mode')
        else:
            print(f'Test mode enabled, using source folder {settings["src"]}')

    return settings

def preprocess_data(df, filter_by, remove_highly_correlated, log_data, exclude, column_list=False):
    """Prepare a feature matrix by filtering, decorrelating, log-transforming, and scaling ``df``.

    :param df: input DataFrame.
    :param filter_by: channel of interest passed to :func:`filter_dataframe_features`; ``None`` disables.
    :param remove_highly_correlated: correlation cutoff (float) or ``True`` to use ``0.95``; ``False`` disables.
    :param log_data: apply ``log(x + 1e-6)`` to numeric columns.
    :param exclude: features to exclude from filtering.
    :param column_list: optional explicit column subset applied before selecting numeric columns.
    :returns: standard-scaled ``ndarray`` of numeric features.
    :raises ValueError: if no numeric columns remain after filtering.
    """
    # Apply filtering based on the `filter_by` parameter
    if filter_by is not None:
        df, _ = filter_dataframe_features(df, channel_of_interest=filter_by, exclude=exclude)
            
    if column_list:
        df = df[column_list]
    
    # Select numerical features
    numeric_data = df.select_dtypes(include=['number'])
    
    # Check if numeric_data is empty
    if numeric_data.empty:
        raise ValueError("No numeric columns available after filtering. Please check the filter_by and exclude parameters.")
    
    # Remove highly correlated columns
    if not remove_highly_correlated is False:
        if isinstance(remove_highly_correlated, float):
            numeric_data = remove_highly_correlated_columns(numeric_data, remove_highly_correlated)
        else:
            numeric_data = remove_highly_correlated_columns(numeric_data, 0.95)
    
    # Apply log transformation
    if log_data:
        numeric_data = np.log(numeric_data + 1e-6)
    
    # Fill NaN values with the column mean
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Scale the numeric data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    numeric_data = scaler.fit_transform(numeric_data)
    
    return numeric_data

def remove_low_variance_columns(df, threshold=0.01, verbose=False):
    """Drop numeric columns whose variance is below ``threshold``.

    :param df: input DataFrame.
    :param threshold: variance cutoff.
    :param verbose: print the dropped column names.
    :returns: filtered DataFrame.
    """

    numerical_cols = df.select_dtypes(include=[np.number])
    low_variance_cols = numerical_cols.var()[numerical_cols.var() < threshold].index.tolist()

    if verbose:
        print(f"Removed columns due to low variance: {low_variance_cols}")

    df = df.drop(columns=low_variance_cols)
    
    return df

def remove_highly_correlated_columns(df, threshold=0.95, verbose=False):
    """Drop numeric columns whose absolute correlation with a prior column exceeds ``threshold``.

    :param df: input DataFrame.
    :param threshold: correlation cutoff.
    :param verbose: print the dropped column names.
    :returns: decorrelated DataFrame.
    """
    numerical_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_cols.corr().abs()
    
    # Upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if verbose:
        print(f"Removed columns due to high correlation: {to_drop}")

    df = df.drop(columns=to_drop)
    
    return df

def filter_dataframe_features(df, channel_of_interest, exclude=None, remove_low_variance_features=True, remove_highly_correlated_features=True, verbose=False):
    """Restrict a features DataFrame to a channel of interest and clean up correlated/low-variance columns.

    :param df: input DataFrame.
    :param channel_of_interest: int, str, list, or ``'morphology'`` to select feature groups.
    :param exclude: features to drop from the final list.
    :param remove_low_variance_features: apply :func:`remove_low_variance_columns`.
    :param remove_highly_correlated_features: apply :func:`remove_highly_correlated_columns`.
    :param verbose: print filter details.
    :returns: ``(filtered_df, features)``.
    """

    count_and_id_columns = [col for col in df.columns if '_id' in col or 'count' in col]
    if 'pathogen_pathogen' in df.columns:
        count_and_id_columns.append('pathogen_pathogen')
    if 'cell_cell' in df.columns:
        count_and_id_columns.append('cell_cell')
    if 'nucleus_nucleus' in df.columns:
        count_and_id_columns.append('nucleus_nucleus')
    if 'cytoplasm_cytoplasm' in df.columns:
        count_and_id_columns.append('cytoplasm_cytoplasm')
    
    if verbose:
        print("Columns to remove:", count_and_id_columns)
        
    df = df.drop(columns=count_and_id_columns)
    
    if not channel_of_interest is None:
        drop_columns = ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        
        if isinstance(channel_of_interest, list):
            feature_strings = [f"channel_{channel}" for channel in channel_of_interest]
            
        elif isinstance(channel_of_interest, str):
            feature_strings = [channel_of_interest]
            
        elif isinstance(channel_of_interest, int):
            feature_string = f"channel_{channel_of_interest}"
            feature_strings = [feature_string]
        elif channel_of_interest == 'morphology':
            morphological_features = ['area', 'area_bbox', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'extent', 'perimeter', 'euler_number', 'solidity', 'zernike_0', 'zernike_1', 'zernike_2', 'zernike_3', 'zernike_4', 'zernike_5', 'zernike_6', 'zernike_7', 'zernike_8', 'zernike_9', 'zernike_10', 'zernike_11', 'zernike_12', 'zernike_13', 'zernike_14', 'zernike_15', 'zernike_16', 'zernike_17', 'zernike_18', 'zernike_19', 'zernike_20', 'zernike_21', 'zernike_22', 'zernike_23', 'zernike_24', 'area_filled', 'convex_area', 'equivalent_diameter_area', 'feret_diameter_max']
            morphological_columns = [item for item in df.columns.tolist() if any(base in item for base in morphological_features)]
            columns_to_drop = [col for col in df.columns if col not in morphological_columns]
        
        if channel_of_interest != 'morphology':
            # Remove entries from drop_columns that are also in feature_strings
            drop_columns = [col for col in drop_columns if col not in feature_strings]

            # Remove columns from the DataFrame that contain any entry from drop_columns in the column name
            columns_to_drop = [col for col in df.columns if any(drop_col in col for drop_col in drop_columns) or all(fs not in col for fs in feature_strings)]
        
        df = df.drop(columns=columns_to_drop)
        if verbose:
            print(f"Removed columns: {columns_to_drop}")
  
    if remove_low_variance_features:
        df = remove_low_variance_columns(df, threshold=0.01, verbose=verbose)
    
    if remove_highly_correlated_features:
        df = remove_highly_correlated_columns(df, threshold=0.95, verbose=verbose)
        
    # Remove columns with NaN values
    before_drop_NaN = len(df.columns)
    df = df.dropna(axis=1)
    after_drop_NaN = len(df.columns)
    print(f"Dropped {before_drop_NaN - after_drop_NaN} columns with NaN values")

    # Select numerical features
    features = df.select_dtypes(include=[np.number]).columns.tolist()

    if isinstance(exclude, list):
        features = [feature for feature in features if feature not in exclude]
    elif isinstance(exclude, str):
        features.remove(exclude)

    filtered_df = df[features]

    return filtered_df, features

# Create a function to check if images overlap
def check_overlap(current_position, other_positions, threshold):
    """Return ``True`` if ``current_position`` is within ``threshold`` of any point in ``other_positions``."""
    for other_position in other_positions:
        distance = np.linalg.norm(np.array(current_position) - np.array(other_position))
        if distance < threshold:
            return True
    return False

# Define a function to try random positions around a given point
def find_non_overlapping_position(x, y, image_positions, threshold, max_attempts=100):
    """Return a nearby ``(x, y)`` jittered position that does not collide with ``image_positions``.

    :param x: original x.
    :param y: original y.
    :param image_positions: previously placed points.
    :param threshold: minimum allowed spacing.
    :param max_attempts: retry budget before giving up.
    :returns: ``(x, y)`` tuple; original position if no non-overlapping spot is found.
    """
    offset_range = 10  # Adjust the range for random offsets
    attempts = 0
    while attempts < max_attempts:
        random_offset_x = random.uniform(-offset_range, offset_range)
        random_offset_y = random.uniform(-offset_range, offset_range)
        new_x = x + random_offset_x
        new_y = y + random_offset_y
        if not check_overlap((new_x, new_y), image_positions, threshold):
            return new_x, new_y
        attempts += 1
    return x, y  # Return the original position if no suitable position found

def search_reduction_and_clustering(numeric_data, n_neighbors, min_dist, metric, eps, min_samples, clustering, reduction_method, verbose, reduction_param=None, embedding=None, n_jobs=-1):
    """Variant of :func:`reduction_and_clustering` accepting extra reducer kwargs via ``reduction_param``.

    :param numeric_data: numeric data matrix.
    :param n_neighbors: UMAP ``n_neighbors`` or t-SNE perplexity (int or fraction).
    :param min_dist: UMAP ``min_dist``.
    :param metric: distance metric.
    :param eps: DBSCAN ``eps``.
    :param min_samples: DBSCAN ``min_samples`` or KMeans cluster count.
    :param clustering: ``'dbscan'`` or ``'kmeans'``.
    :param reduction_method: ``'umap'`` or ``'tsne'``.
    :param verbose: print progress.
    :param reduction_param: extra kwargs forwarded to the reducer.
    :param embedding: precomputed embedding to skip fitting.
    :param n_jobs: parallel worker count.
    :returns: ``(embedding, labels)``.
    :raises ValueError: on unsupported ``reduction_method`` or ``clustering``.
    """

    if isinstance(n_neighbors, float):
        n_neighbors = int(n_neighbors * len(numeric_data))
    if n_neighbors <= 1:
        n_neighbors = 2
        print(f'n_neighbors cannota be less than 2. Setting n_neighbors to {n_neighbors}')

    reduction_param = reduction_param or {}
    reduction_param = {k: v for k, v in reduction_param.items() if k not in ['perplexity', 'n_neighbors', 'min_dist', 'metric', 'method']}
    
    if reduction_method == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_jobs=n_jobs, **reduction_param)
    elif reduction_method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=n_neighbors, metric=metric, n_jobs=n_jobs, **reduction_param)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}. Supported methods are 'umap' and 'tsne'")

    if embedding is None:
        embedding = reducer.fit_transform(numeric_data)

    if clustering == 'dbscan':
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    elif clustering == 'kmeans':
        from sklearn.cluster import KMeans
        clustering_model = KMeans(n_clusters=min_samples, random_state=42)
    else:
        raise ValueError(f"Unsupported clustering method: {clustering}. Supported methods are 'dbscan' and 'kmeans'")
    clustering_model.fit(embedding)
    labels = clustering_model.labels_ if clustering == 'dbscan' else clustering_model.predict(embedding)
    if verbose:
        print(f'Embedding shape: {embedding.shape}')
    return embedding, labels

def load_image(image_path):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def extract_features(image_paths, resnet=resnet50):
    """Extract features from images using a pre-trained ResNet model."""
    model = resnet(pretrained=True)
    model = model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last classification layer

    features = []
    for image_path in image_paths:
        image = load_image(image_path)
        with torch.no_grad():
            feature = model(image).squeeze().numpy()
        features.append(feature)

    return np.array(features)

def check_normality(series):
    """Helper function to check if a feature is normally distributed."""
    k2, p = stats.normaltest(series)
    alpha = 0.05
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        return False
    return True

def random_forest_feature_importance(all_df, cluster_col='cluster'):
    """Random Forest feature importance."""
    numeric_features = all_df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_features:
        numeric_features.remove(cluster_col)

    X = all_df[numeric_features]
    y = all_df[cluster_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    return importance_df

def perform_statistical_tests(all_df, cluster_col='cluster'):
    """Perform ANOVA or Kruskal-Wallis tests depending on normality of features."""
    numeric_features = all_df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_features:
        numeric_features.remove(cluster_col)
    
    anova_results = []
    kruskal_results = []

    for feature in numeric_features:
        groups = [all_df[all_df[cluster_col] == label][feature] for label in np.unique(all_df[cluster_col])]
        
        if check_normality(all_df[feature]):
            stat, p = f_oneway(*groups)
            anova_results.append((feature, stat, p))
        else:
            stat, p = kruskal(*groups)
            kruskal_results.append((feature, stat, p))
    
    anova_df = pd.DataFrame(anova_results, columns=['Feature', 'ANOVA_Statistic', 'ANOVA_pValue'])
    kruskal_df = pd.DataFrame(kruskal_results, columns=['Feature', 'Kruskal_Statistic', 'Kruskal_pValue'])

    return anova_df, kruskal_df

def combine_results(rf_df, anova_df, kruskal_df):
    """Combine the results into a single DataFrame."""
    combined_df = rf_df.merge(anova_df, on='Feature', how='left')
    combined_df = combined_df.merge(kruskal_df, on='Feature', how='left')
    return combined_df

def cluster_feature_analysis(all_df, cluster_col='cluster'):
    """
    Perform Random Forest feature importance, ANOVA for normally distributed features,
    and Kruskal-Wallis for non-normally distributed features. Combine results into a single DataFrame.
    """
    rf_df = random_forest_feature_importance(all_df, cluster_col)
    anova_df, kruskal_df = perform_statistical_tests(all_df, cluster_col)
    combined_df = combine_results(rf_df, anova_df, kruskal_df)
    return combined_df

def _merge_cells_without_nucleus(adj_cell_mask: np.ndarray, nuclei_mask: np.ndarray):
    """
    Relabel any cell that lacks a nucleus to the ID of an adjacent
    cell that *does* contain a nucleus.

    Parameters
    ----------
    adj_cell_mask : np.ndarray
        Labelled (0 = background) cell mask after all other merging steps.
    nuclei_mask : np.ndarray
        Labelled (0 = background) nuclei mask.

    Returns
    -------
    np.ndarray
        Updated cell mask with nucleus-free cells merged into
        neighbouring nucleus-bearing cells.
    """
    out = adj_cell_mask.copy()

    # ----------------------------------------------------------------- #
    # 1 — Identify which cell IDs contain a nucleus
    nuc_labels = np.unique(nuclei_mask[nuclei_mask > 0])

    cells_with_nuc = set()
    for nuc_id in nuc_labels:
        labels, counts = np.unique(adj_cell_mask[nuclei_mask == nuc_id],
                                   return_counts=True)

        # drop background (label 0) from *both* arrays
        keep = labels > 0
        labels = labels[keep]
        counts = counts[keep]

        if labels.size:                     # at least one non-zero overlap
            cells_with_nuc.add(labels[np.argmax(counts)])

    # ----------------------------------------------------------------- #
    # 2 — Build an adjacency map between neighbouring cell IDs
    # ----------------------------------------------------------------- #
    boundaries = find_boundaries(adj_cell_mask, mode="thick")
    adj_map = defaultdict(set)

    ys, xs = np.where(boundaries)
    h, w = adj_cell_mask.shape
    for y, x in zip(ys, xs):
        src = adj_cell_mask[y, x]
        if src == 0:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    dst = adj_cell_mask[ny, nx]
                    if dst != 0 and dst != src:
                        adj_map[src].add(dst)

    # ----------------------------------------------------------------- #
    # 3 — Relabel nucleus-free cells that touch nucleus-bearing neighbours
    # ----------------------------------------------------------------- #
    cells_no_nuc = set(np.unique(adj_cell_mask)) - {0} - cells_with_nuc
    for cell_id in cells_no_nuc:
        neighbours = adj_map.get(cell_id, set()) & cells_with_nuc
        if neighbours:
            # Choose the first nucleus-bearing neighbour deterministically
            target = sorted(neighbours)[0]
            out[out == cell_id] = target

    return out.astype(np.uint16)

def _merge_cells_based_on_parasite_overlap(parasite_mask, cell_mask, nuclei_mask, organelle_mask, overlap_threshold=5, perimeter_threshold=30):
    """Merge cells that share a parasite/nucleus or a large fraction of perimeter."""
    labeled_cells = label(cell_mask)
    labeled_parasites = label(parasite_mask)
    labeled_nuclei = label(nuclei_mask)
    num_parasites = np.max(labeled_parasites)
    num_cells = np.max(labeled_cells)
    num_nuclei = np.max(labeled_nuclei)

    if organelle_mask is not None:
        labeled_organelles = label(organelle_mask)
        num_organelles = np.max(labeled_organelles)
    else:
        labeled_organelles = None
        num_organelles = 0

    # Merge cells based on parasite overlap
    for parasite_id in range(1, num_parasites + 1):
        current_parasite_mask = labeled_parasites == parasite_id
        overlapping_cell_labels = np.unique(labeled_cells[current_parasite_mask])
        overlapping_cell_labels = overlapping_cell_labels[overlapping_cell_labels != 0]
        if len(overlapping_cell_labels) > 1:
            
            # Calculate the overlap percentages
            overlap_percentages = [
                np.sum(current_parasite_mask & (labeled_cells == cell_label)) / np.sum(current_parasite_mask) * 100
                for cell_label in overlapping_cell_labels
            ]
            # Merge cells if overlap percentage is above the threshold
            for cell_label, overlap_percentage in zip(overlapping_cell_labels, overlap_percentages):
                if overlap_percentage > overlap_threshold:
                    first_label = overlapping_cell_labels[0]
                    for other_label in overlapping_cell_labels[1:]:
                        if other_label != first_label:
                            cell_mask[cell_mask == other_label] = first_label

    # Merge cells based on nucleus overlap
    for nucleus_id in range(1, num_nuclei + 1):
        current_nucleus_mask = labeled_nuclei == nucleus_id
        overlapping_cell_labels = np.unique(labeled_cells[current_nucleus_mask])
        overlapping_cell_labels = overlapping_cell_labels[overlapping_cell_labels != 0]
        if len(overlapping_cell_labels) > 1:
            
            # Calculate the overlap percentages
            overlap_percentages = [
                np.sum(current_nucleus_mask & (labeled_cells == cell_label)) / np.sum(current_nucleus_mask) * 100
                for cell_label in overlapping_cell_labels
            ]
            # Merge cells if overlap percentage is above the threshold for each cell
            if all(overlap_percentage > overlap_threshold for overlap_percentage in overlap_percentages):
                first_label = overlapping_cell_labels[0]
                for other_label in overlapping_cell_labels[1:]:
                    if other_label != first_label:
                        cell_mask[cell_mask == other_label] = first_label

    # Check for cells without nuclei and merge based on shared perimeter
    labeled_cells = label(cell_mask)  # Re-label after merging based on overlap
    cell_regions = regionprops(labeled_cells)
    for region in cell_regions:
        cell_label = region.label
        cell_mask_binary = labeled_cells == cell_label
        overlapping_nuclei = np.unique(nuclei_mask[cell_mask_binary])
        overlapping_nuclei = overlapping_nuclei[overlapping_nuclei != 0]

        if len(overlapping_nuclei) == 0:
            
            # Cell does not overlap with any nucleus
            perimeter = region.perimeter
            
            # Dilate the cell to find neighbors
            dilated_cell = binary_dilation(cell_mask_binary, structure=square(3))
            neighbor_cells = np.unique(labeled_cells[dilated_cell])
            neighbor_cells = neighbor_cells[(neighbor_cells != 0) & (neighbor_cells != cell_label)]
            
            # Calculate shared border length with neighboring cells
            shared_borders = [
                np.sum((labeled_cells == neighbor_label) & dilated_cell) for neighbor_label in neighbor_cells
            ]
            shared_border_percentages = [shared_border / perimeter * 100 for shared_border in shared_borders]
            
            # Merge with the neighbor cell with the largest shared border percentage above the threshold
            if shared_borders:
                max_shared_border_index = np.argmax(shared_border_percentages)
                max_shared_border_percentage = shared_border_percentages[max_shared_border_index]
                if max_shared_border_percentage > perimeter_threshold:
                    cell_mask[labeled_cells == cell_label] = neighbor_cells[max_shared_border_index]
    
    # Relabel the merged cell mask
    relabeled_cell_mask, _ = label(cell_mask, return_num=True)
    return relabeled_cell_mask.astype(np.uint16)


def process_mask_file_adjust_cell(file_name, parasite_folder, cell_folder, nuclei_folder, organelle_folder=None, overlap_threshold=5, perimeter_threshold=30):
    """Load one triple of parasite/cell/nuclei masks, merge cells in place, and return the elapsed time.

    :param file_name: mask file name (must exist in all folders).
    :param parasite_folder: folder of parasite masks.
    :param cell_folder: folder of cell masks (overwritten in place).
    :param nuclei_folder: folder of nuclei masks.
    :param organelle_folder: optional folder of organelle masks.
    :param overlap_threshold: fractional overlap threshold used by the merger.
    :param perimeter_threshold: shared-perimeter threshold used by the merger.
    :returns: elapsed seconds.
    :raises ValueError: if the matching cell or nuclei mask file is missing.
    """
    start = time.perf_counter()

    parasite_path = os.path.join(parasite_folder, file_name)
    cell_path = os.path.join(cell_folder, file_name)
    nuclei_path = os.path.join(nuclei_folder, file_name)

    if not (os.path.exists(cell_path) and os.path.exists(nuclei_path)):
        raise ValueError(f"Corresponding cell or nuclei mask file for {file_name} not found.")

    parasite_mask = np.load(parasite_path, allow_pickle=True)
    cell_mask = np.load(cell_path, allow_pickle=True)
    nuclei_mask = np.load(nuclei_path, allow_pickle=True)

    organelle_mask = None
    if organelle_folder is not None:
        organelle_path = os.path.join(organelle_folder, file_name)
        if os.path.exists(organelle_path):
            organelle_mask = np.load(organelle_path, allow_pickle=True)

    merged_cell_mask = _merge_cells_based_on_parasite_overlap(parasite_mask, cell_mask, nuclei_mask, organelle_mask, overlap_threshold, perimeter_threshold)

    np.save(cell_path, merged_cell_mask)

    end = time.perf_counter()
    return end - start

def adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, organelle_folder=None, overlap_threshold=5, perimeter_threshold=30, n_jobs=None):
    """Run :func:`process_mask_file_adjust_cell` in parallel across matching mask files.

    :param parasite_folder: folder of parasite masks.
    :param cell_folder: folder of cell masks (overwritten in place).
    :param nuclei_folder: folder of nuclei masks.
    :param organelle_folder: optional folder of organelle masks.
    :param overlap_threshold: fractional overlap threshold used by the merger.
    :param perimeter_threshold: shared-perimeter threshold used by the merger.
    :param n_jobs: worker count; defaults to ``cpu_count() - 2``.
    :returns: None.
    :raises ValueError: if the three folders contain different numbers of files.
    """
    parasite_files = sorted([f for f in os.listdir(parasite_folder) if f.endswith('.npy')])
    cell_files = sorted([f for f in os.listdir(cell_folder) if f.endswith('.npy')])
    nuclei_files = sorted([f for f in os.listdir(nuclei_folder) if f.endswith('.npy')])

    if not (len(parasite_files) == len(cell_files) == len(nuclei_files)):
        raise ValueError("The number of files in the folders do not match.")

    if organelle_folder is not None and os.path.exists(organelle_folder):
        organelle_files = sorted([f for f in os.listdir(organelle_folder) if f.endswith('.npy')])
        if len(organelle_files) != len(parasite_files):
            print(f'Warning: organelle mask count ({len(organelle_files)}) does not match other masks ({len(parasite_files)}). Organelle masks will be loaded per-file where available.')
    else:
        organelle_folder = None

    n_jobs = n_jobs or max(1, cpu_count() - 2)

    time_ls = []
    files_to_process = len(parasite_files)
    process_fn = partial(process_mask_file_adjust_cell,
                         parasite_folder=parasite_folder,
                         cell_folder=cell_folder,
                         nuclei_folder=nuclei_folder,
                         organelle_folder=organelle_folder,
                         overlap_threshold=overlap_threshold,
                         perimeter_threshold=perimeter_threshold)

    with Pool(n_jobs) as pool:
        for i, duration in enumerate(pool.imap_unordered(process_fn, parasite_files), 1):
            time_ls.append(duration)
            print_progress(i, files_to_process, n_jobs=n_jobs, time_ls=time_ls, batch_size=None, operation_type='adjust_cell_masks')

def process_masks(mask_folder, image_folder, channel, batch_size=50, n_clusters=2, plot=False):
    """Cluster object morphology/intensity across a mask folder and keep the largest cluster in place.

    :param mask_folder: folder of ``.npy`` masks.
    :param image_folder: matching folder of ``.npy`` intensity images.
    :param channel: channel index used for intensity measurements.
    :param batch_size: number of files to load per batch.
    :param n_clusters: number of KMeans clusters.
    :param plot: show a PCA scatter of the clustered objects.
    :returns: None.
    """
    def read_files_in_batches(folder, batch_size=50):
        """Yield sorted lists of ``.npy`` filenames from ``folder`` in chunks of ``batch_size``."""
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        files.sort()  # Sort to ensure matching order
        for i in range(0, len(files), batch_size):
            yield files[i:i + batch_size]

    def measure_morphology_and_intensity(mask, image):
        """Return a list of dicts with area/mean_intensity/perimeter/eccentricity per labeled region."""
        properties = measure.regionprops(mask, intensity_image=image)
        properties_list = [{'area': p.area, 'mean_intensity': p.mean_intensity, 'perimeter': p.perimeter, 'eccentricity': p.eccentricity} for p in properties]
        return properties_list

    def cluster_objects(properties, n_clusters=2):
        """Return a fitted ``KMeans`` object clustering the property dicts into ``n_clusters`` groups."""
        data = np.array([[p['area'], p['mean_intensity'], p['perimeter'], p['eccentricity']] for p in properties])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        return kmeans

    def remove_objects_not_in_largest_cluster(mask, labels, largest_cluster_label):
        """Return ``mask`` with all labeled regions removed except those in ``largest_cluster_label``."""
        cleaned_mask = np.zeros_like(mask)
        for region in measure.regionprops(mask):
            if labels[region.label - 1] == largest_cluster_label:
                cleaned_mask[mask == region.label] = region.label
        return cleaned_mask

    def plot_clusters(properties, labels):
        """Show a 2-D PCA scatter of the property vectors colored by cluster label."""
        data = np.array([[p['area'], p['mean_intensity'], p['perimeter'], p['eccentricity']] for p in properties])
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Object Clustering')
        plt.show()
    
    all_properties = []

    # Step 1: Accumulate properties over all files
    for batch in read_files_in_batches(mask_folder, batch_size):
        mask_files = [os.path.join(mask_folder, file) for file in batch]
        image_files = [os.path.join(image_folder, file) for file in batch]
        
        masks = [np.load(file) for file in mask_files]
        images = [np.load(file)[:, :, channel] for file in image_files]
        
        for i, mask in enumerate(masks):
            image = images[i]
            # Measure morphology and intensity
            properties = measure_morphology_and_intensity(mask, image)
            all_properties.extend(properties)

    # Step 2: Perform clustering on accumulated properties
    kmeans = cluster_objects(all_properties, n_clusters)
    labels = kmeans.labels_

    if plot:
        # Step 3: Plot clusters using PCA
        plot_clusters(all_properties, labels)

    # Step 4: Remove objects not in the largest cluster and overwrite files in batches
    label_index = 0
    for batch in read_files_in_batches(mask_folder, batch_size):
        mask_files = [os.path.join(mask_folder, file) for file in batch]
        masks = [np.load(file) for file in mask_files]
        
        for i, mask in enumerate(masks):
            batch_properties = measure_morphology_and_intensity(mask, mask)
            batch_labels = labels[label_index:label_index + len(batch_properties)]
            largest_cluster_label = np.bincount(batch_labels).argmax()
            cleaned_mask = remove_objects_not_in_largest_cluster(mask, batch_labels, largest_cluster_label)
            np.save(mask_files[i], cleaned_mask)
            label_index += len(batch_properties)

def merge_regression_res_with_metadata(results_file, metadata_file, name='_metadata'):
    """Merge regression outputs with gene metadata on the parsed ``gene`` column.

    :param results_file: path to a regression results CSV with a ``feature`` column.
    :param metadata_file: path to a gene metadata CSV with a ``Gene ID`` column.
    :param name: suffix appended to the output filename.
    :returns: merged DataFrame (also written to ``<results_file><name>.csv``).
    """
    # Read the CSV files into dataframes
    df_results = pd.read_csv(results_file)
    df_metadata = pd.read_csv(metadata_file)
    
    def extract_and_clean_gene(feature):
        """Return the gene ID parsed from a ``feature`` string like ``C(gene)[T.<id>_...]``, or ``None``."""
        # Extract the part between '[' and ']'
        match = re.search(r'\[(.*?)\]', feature)
        if match:
            gene = match.group(1)
            # Remove 'T.' if present
            gene = re.sub(r'^T\.', '', gene)
            # Remove everything after and including '_'
            gene = gene.split('_')[0]
            return gene
        return None

    # Apply the function to the feature column
    df_results['gene'] = df_results['feature'].apply(extract_and_clean_gene)
    
    df_metadata['gene'] = df_metadata['Gene ID'].apply(lambda x: x.split('_')[1] if '_' in x else None)
    
    # Drop rows where gene extraction failed
    #df_results = df_results.dropna(subset=['gene'])
    
    # Merge the two dataframes on the gene column
    merged_df = pd.merge(df_results, df_metadata, on='gene', how='left')
    
    # Generate the new file name
    base, ext = os.path.splitext(results_file)
    new_file = f"{base}{name}{ext}"
    
    # Save the merged dataframe to the new file
    merged_df.to_csv(new_file, index=False)
    
    return merged_df

def process_vision_results(df, threshold=0.5):
    """Split image paths into well identifiers and binarize the ``pred`` column.

    :param df: DataFrame with ``path`` and ``pred`` columns.
    :param threshold: cutoff used to derive ``cv_predictions``.
    :returns: enriched DataFrame with ``plateID``, ``rowID``, ``columnID``, ``fieldID``, ``prc``, ``cv_predictions``.
    """
    # Split the 'path' column using _map_wells function
    mapped_values = df['path'].apply(lambda x: _map_wells(x))
    
    df['plateID'] = mapped_values.apply(lambda x: x[0])
    df['rowID'] = mapped_values.apply(lambda x: x[1])
    df['columnID'] = mapped_values.apply(lambda x: x[2])
    df['fieldID'] = mapped_values.apply(lambda x: x[3])
    df['object'] = df['path'].str.split('_').str[3].str.split('.').str[0]
    df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
    df['cv_predictions'] = (df['pred'] >= threshold).astype(int)

    return df

def get_ml_results_paths(src, model_type='xgboost', channel_of_interest=1):
    """Return the standard set of ML output paths for the given model and channel selection.

    :param src: experiment root.
    :param model_type: model identifier (used in the results folder name).
    :param channel_of_interest: int, list, ``'morphology'``, or ``None`` (aliased to ``all_features``).
    :returns: 10-tuple of paths ``(data, permutation, feature_importance, model_metrics,
        permutation_fig, feature_importance_fig, shap_fig, plate_heatmap, settings, ml_features)``.
    :raises ValueError: if ``channel_of_interest`` has an unsupported type.
    """
    if isinstance(channel_of_interest, list):
        feature_string = "channels_" + "_".join(map(str, channel_of_interest))

    elif isinstance(channel_of_interest, int):
        feature_string = f"channel_{channel_of_interest}"

    elif channel_of_interest == 'morphology':
        feature_string = 'morphology'

    elif channel_of_interest == None:
        feature_string = 'all_features'
    else:
        raise ValueError(f"Unsupported channel_of_interest: {channel_of_interest}. Supported values are 'int', 'list', 'None', or 'morphology'.")

    res_fldr = os.path.join(src, 'results', model_type, feature_string)
    print(f'Saving results to {res_fldr}')
    os.makedirs(res_fldr, exist_ok=True)
    data_path = os.path.join(res_fldr, 'results.csv')
    permutation_path = os.path.join(res_fldr, 'permutation.csv')
    feature_importance_path = os.path.join(res_fldr, 'feature_importance.csv')
    model_metricks_path = os.path.join(res_fldr, f'{model_type}_model.csv')
    permutation_fig_path = os.path.join(res_fldr, 'permutation.pdf')
    feature_importance_fig_path = os.path.join(res_fldr, 'feature_importance.pdf')
    shap_fig_path = os.path.join(res_fldr, 'shap.pdf')
    plate_heatmap_path = os.path.join(res_fldr, 'plate_heatmap.pdf')
    settings_csv = os.path.join(res_fldr, 'ml_settings.csv')
    ml_features = os.path.join(res_fldr, 'ml_features.csv')
    return data_path, permutation_path, feature_importance_path, model_metricks_path, permutation_fig_path, feature_importance_fig_path, shap_fig_path, plate_heatmap_path, settings_csv, ml_features

def augment_image(image):
    """Return a list of PIL images covering 4 rotations x 2 horizontal reflections of ``image``."""
    augmented_images = []

    # Convert PIL image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Rotations and reflections
    transformations = [
        None,  # Original
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    ]

    for transform in transformations:
        if transform is not None:
            rotated = cv2.rotate(image, transform)
        else:
            rotated = image
        augmented_images.append(rotated)

        # Reflections
        flipped = cv2.flip(rotated, 1)
        augmented_images.append(flipped)

    # Convert numpy arrays back to PIL images
    augmented_images = [Image.fromarray(img) for img in augmented_images]
    
    return augmented_images

def augment_dataset(dataset, is_grayscale=False):
    """Expand ``dataset`` by 8x through rotation and horizontal reflection of every image tensor.

    :param dataset: iterable of ``(tensor, label, filename)``.
    :param is_grayscale: informational flag (retained for API compatibility).
    :returns: list of augmented ``(tensor, label, filename)`` tuples.
    :raises TypeError: if an image is not a ``torch.Tensor``.
    """
    augmented_dataset = []

    for img, label, filename in dataset:
        augmented_images = []

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(img)}")

        # Rotations and reflections
        angles = [0, 90, 180, 270]

        for angle in angles:
            rotated = torchvision.transforms.functional.rotate(img, angle)
            augmented_images.append(rotated)

            # Reflections
            flipped = torchvision.transforms.functional.hflip(rotated)
            augmented_images.append(flipped)

        # Add augmented images to the dataset
        for aug_img in augmented_images:
            augmented_dataset.append((aug_img, label, filename))

    return augmented_dataset


def convert_and_relabel_masks(folder_path):
    """
    Converts all int64 npy masks in a folder to uint16 with relabeling to ensure all labels are retained.

    Parameters:
    - folder_path (str): The path to the folder containing int64 npy mask files.

    Returns:
    - None
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        # Load the mask
        mask = np.load(file_path)
        #print(mask.shape)
        #print(mask.dtype)
        # Check the current dtype
        if mask.dtype != np.int64:
            print(f"Skipping {file} as it is not int64.")
            continue
        
        # Relabel the mask to ensure unique labels within uint16 range
        unique_labels = np.unique(mask)
        if unique_labels.max() > 65535:
            print(f"Warning: The mask in {file} contains values that exceed the uint16 range and will be relabeled.")

        relabeled_mask = measure.label(mask, background=0)
        
        # Check that relabeling worked correctly
        unique_relabeled = np.unique(relabeled_mask)
        if unique_relabeled.max() > 65535:
            print(f"Error: Relabeling failed for {file} as it still contains values that exceed the uint16 range.")
            continue
        
        # Convert to uint16
        relabeled_mask = relabeled_mask.astype(np.uint16)
        
        # Save the converted mask
        np.save(file_path, relabeled_mask)
        
        print(f"Converted {file} and saved as uint16_{file}")

def correct_masks(src):
    """Convert cell masks under ``src/masks/cell_mask_stack`` to uint16 and re-stack arrays.

    Relabels masks so they fit in ``uint16`` and then re-concatenates the four
    array folders under ``src`` in the layout expected downstream.

    :param src: Root folder of a spacr run containing a ``masks/`` subfolder.
    :returns: None.
    """
    from .io import _load_and_concatenate_arrays

    cell_path = os.path.join(src,'masks', 'cell_mask_stack')
    convert_and_relabel_masks(cell_path)
    _load_and_concatenate_arrays(src, [0,1,2,3], 1, 0, 2)

def count_reads_in_fastq(fastq_file):
    """Return the number of reads in a gzipped FASTQ file.

    Counts total lines and divides by four (the FASTQ record length).

    :param fastq_file: Path to a ``.fastq.gz`` file.
    :returns: Integer read count.
    """
    count = 0
    with gzip.open(fastq_file, "rt") as f:
        for _ in f:
            count += 1
    return count // 4


# Function to determine the CUDA version
def get_cuda_version():
    """Return the installed CUDA toolkit version as a digit-only string, or ``None``.

    Parses the ``nvcc --version`` output; the dots are stripped so ``11.8`` becomes ``"118"``.

    :returns: Version string without dots, or ``None`` if ``nvcc`` is missing or fails.
    """
    try:
        output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
        if 'release' in output:
            return output.split('release ')[1].split(',')[0].replace('.', '')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def all_elements_match(list1, list2):
    """Return ``True`` if every element of ``list1`` is contained in ``list2``.

    :param list1: iterable of items to test.
    :param list2: iterable acting as the reference set.
    :returns: ``True`` when ``list1`` is a subset of ``list2``, else ``False``.
    """
    return all(element in list2 for element in list1)

def prepare_batch_for_segmentation(batch):
    """Cast a batch to ``float32`` and per-image max-normalize any image whose max exceeds 1.

    :param batch: ``(N, ...)`` numpy array of images.
    :returns: The same array cast to ``float32`` with each image scaled to ``[0, 1]``.
    """
    if batch.dtype != np.float32:
        batch = batch.astype(np.float32)

    # Normalize each image in the batch
    for i in range(batch.shape[0]):
        if batch[i].max() > 1:
            batch[i] = batch[i] / batch[i].max()

    return batch

def check_index(df, elements=5, split_char='_'):
    """Validate that every index label in ``df`` splits into ``elements`` parts on ``split_char``.

    :param df: DataFrame whose index labels are compound identifiers.
    :param elements: Expected number of parts after splitting. Default ``5``.
    :param split_char: Delimiter used to split each index label. Default ``'_'``.
    :returns: None.
    :raises ValueError: if any index label does not split into ``elements`` parts.
    """
    problematic_indices = []
    for idx in df.index:
        parts = str(idx).split(split_char)
        if len(parts) != elements:
            problematic_indices.append(idx)
    if problematic_indices:
        print("Indices that cannot be separated into 5 parts:")
        for idx in problematic_indices:
            print(idx)
        raise ValueError(f"Found {len(problematic_indices)} problematic indices that do not split into {elements} parts.")
    
# Define the mapping function
def map_condition(col_value, neg='c1', pos='c2', mix='c3'):
    """Map a column-ID value to one of ``'neg'``, ``'pos'``, ``'mix'``, or ``'screen'``.

    :param col_value: Column identifier from the plate metadata.
    :param neg: Column ID that corresponds to negative controls. Default ``'c1'``.
    :param pos: Column ID that corresponds to positive controls. Default ``'c2'``.
    :param mix: Column ID that corresponds to mixed controls. Default ``'c3'``.
    :returns: Condition label; any unlisted column returns ``'screen'``.
    """
    if col_value == neg:
        return 'neg'
    elif col_value == pos:
        return 'pos'
    elif col_value == mix:
        return 'mix'
    else:
        return 'screen'
    
def download_models(repo_id="einarolafsson/models", retries=5, delay=5):
    """
    Downloads all model files from Hugging Face and stores them in the `resources/models` directory 
    within the installed `spacr` package.

    Args:
        repo_id (str): The repository ID on Hugging Face (default is 'einarolafsson/models').
        retries (int): Number of retry attempts in case of failure.
        delay (int): Delay in seconds between retries.

    Returns:
        str: The local path to the downloaded models.
    """
    # Construct the path to the `resources/models` directory in the installed `spacr` package
    package_dir = os.path.dirname(spacr_path)
    local_dir = os.path.join(package_dir, 'resources', 'models')

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    elif len(os.listdir(local_dir)) > 0:
        #print(f"Models already downloaded to: {local_dir}")
        return local_dir

    attempt = 0
    while attempt < retries:
        try:
            # List all files in the repo
            files = list_repo_files(repo_id, repo_type="dataset")
            print(f"Files in repository: {files}")  # Debugging print to check file list

            # Download each file
            for file_name in files:
                for download_attempt in range(retries):
                    try:
                        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}?download=true"
                        print(f"Downloading file from: {url}")  # Debugging

                        response = requests.get(url, stream=True)
                        print(f"HTTP response status: {response.status_code}")  # Debugging
                        response.raise_for_status()

                        # Save the file locally
                        local_file_path = os.path.join(local_dir, os.path.basename(file_name))
                        with open(local_file_path, 'wb') as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        print(f"Downloaded model file: {file_name} to {local_file_path}")
                        break  # Exit the retry loop if successful
                    except (requests.HTTPError, requests.Timeout) as e:
                        print(f"Error downloading {file_name}: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                else:
                    raise Exception(f"Failed to download {file_name} after multiple attempts.")

            return local_dir  # Return the directory where models are saved

        except (requests.HTTPError, requests.Timeout) as e:
            print(f"Error downloading files: {e}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)

    raise Exception("Failed to download model files after multiple attempts.")

def generate_cytoplasm_mask(nucleus_mask, cell_mask):
        
    """
    Generates a cytoplasm mask from nucleus and cell masks.
    
    Parameters:
    - nucleus_mask (np.array): Binary or segmented mask of the nucleus (non-zero values represent nucleus).
    - cell_mask (np.array): Binary or segmented mask of the whole cell (non-zero values represent cell).
    
    Returns:
    - cytoplasm_mask (np.array): Mask for the cytoplasm (1 for cytoplasm, 0 for nucleus and pathogens).
    """
    
    # Make sure the nucleus and cell masks are numpy arrays
    nucleus_mask = np.array(nucleus_mask)
    cell_mask = np.array(cell_mask)
    
    # Generate cytoplasm mask
    cytoplasm_mask = np.where(np.logical_or(nucleus_mask != 0), 0, cell_mask)
    
    return cytoplasm_mask

def add_column_to_database(settings):
    """
    Adds a new column to the database table by matching on a common column from the DataFrame.
    If the column already exists in the database, it adds the column with a suffix.
    NaN values will remain as NULL in the database.

    Parameters:
        settings (dict): A dictionary containing the following keys:
            csv_path (str): Path to the CSV file with the data to be added.
            db_path (str): Path to the SQLite database (or connection string for other databases).
            table_name (str): The name of the table in the database.
            update_column (str): The name of the new column in the DataFrame to add to the database.
            match_column (str): The common column used to match rows.

    Returns:
        None
    """

    # Read the DataFrame from the provided CSV path
    df = pd.read_csv(settings['csv_path'])

    # Replace 0 values with 2 in the update column
    if (df[settings['update_column']] == 0).any():
        print("Replacing all 0 values with 2 in the update column.")
        df[settings['update_column']].replace(0, 2, inplace=True)

    # Connect to the SQLite database
    conn = sqlite3.connect(settings['db_path'])
    cursor = conn.cursor()

    # Get the existing columns in the database table
    cursor.execute(f"PRAGMA table_info({settings['table_name']})")
    columns_in_db = [col[1] for col in cursor.fetchall()]

    # Add a suffix if the update column already exists in the database
    if settings['update_column'] in columns_in_db:
        suffix = 1
        new_column_name = f"{settings['update_column']}_{suffix}"
        while new_column_name in columns_in_db:
            suffix += 1
            new_column_name = f"{settings['update_column']}_{suffix}"
        print(f"Column '{settings['update_column']}' already exists. Using new column name: '{new_column_name}'")
    else:
        new_column_name = settings['update_column']

    # Add the new column with INTEGER type to the database table
    cursor.execute(f"ALTER TABLE {settings['table_name']} ADD COLUMN {new_column_name} INTEGER")
    print(f"Added new column '{new_column_name}' to the table '{settings['table_name']}'.")

    # Iterate over the DataFrame and update the new column in the database
    for index, row in df.iterrows():
        value_to_update = row[settings['update_column']]
        match_value = row[settings['match_column']]

        # Handle NaN values by converting them to None (SQLite equivalent of NULL)
        if pd.isna(value_to_update):
            value_to_update = None

        # Prepare and execute the SQL update query
        query = f"""
            UPDATE {settings['table_name']}
            SET {new_column_name} = ?
            WHERE {settings['match_column']} = ?
        """
        cursor.execute(query, (value_to_update, match_value))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

    print(f"Updated '{new_column_name}' in '{settings['table_name']}' using '{settings['match_column']}'.")

def fill_holes_in_mask(mask):
    """
    Fill holes in each object in the mask while keeping objects separated.
    
    Args:
        mask (np.ndarray): A labeled mask where each object has a unique integer value.
    
    Returns:
        np.ndarray: A mask with holes filled and original labels preserved.
    """
    # Ensure the mask is integer-labeled
    labeled_mask, num_features = ndimage.label(mask)

    # Create an empty mask to store the result
    filled_mask = np.zeros_like(labeled_mask)

    # Fill holes for each labeled object independently
    for i in range(1, num_features + 1):
        # Create a binary mask for the current object
        object_mask = (labeled_mask == i)

        # Fill holes within this object
        filled_object = binary_fill_holes(object_mask)

        # Assign the original label back to the filled object
        filled_mask[filled_object] = i

    return filled_mask

def correct_metadata_column_names(df):
    """Rename legacy metadata columns to the canonical spacr names.

    Handles the common aliases (``plate_name`` -> ``plateID``, ``col`` -> ``columnID``,
    ``row_name`` -> ``rowID``, ``grna_name`` -> ``grna``) and splits ``plate_row``
    into ``plateID`` and ``rowID``.

    :param df: DataFrame whose columns may use legacy names.
    :returns: The same DataFrame with columns renamed in place.
    """
    if 'plate_name' in df.columns:
        df = df.rename(columns={'plate_name': 'plateID'})
    if 'column_name' in df.columns:
        df = df.rename(columns={'column_name': 'columnID'})
    if 'col' in df.columns:
        df = df.rename(columns={'col': 'columnID'})
    if 'row_name' in df.columns:
        df = df.rename(columns={'row_name': 'rowID'})
    if 'grna_name' in df.columns:
        df = df.rename(columns={'grna_name': 'grna'})
    if 'plate_row' in df.columns:
        df[['plateID', 'rowID']] = df['plate_row'].str.split('_', expand=True)
    return df

def control_filelist(folder, mode='columnID', values=None):
    """Return filenames in ``folder`` whose row or column ID matches one of ``values``.

    The filename is split on ``_`` and the second token is inspected: characters
    after the first (``mode='columnID'``) or the leading character
    (``mode='rowID'``) are matched against ``values``.

    :param folder: Directory to scan.
    :param mode: ``'columnID'`` matches trailing digits, ``'rowID'`` matches leading letter.
        Default ``'columnID'``.
    :param values: Iterable of allowed ID strings. Defaults to ``['01', '02']``.
    :returns: List of matching filenames.
    """
    if values is None:
        values = ['01','02']
    files = os.listdir(folder)
    if mode == 'columnID':
        filtered_files = [file for file in files if file.split('_')[1][1:] in values]
    if mode == 'rowID':
        filtered_files = [file for file in files if file.split('_')[1][:1] in values]
    return filtered_files
    
def rename_columns_in_db(db_path):
    """Rename legacy plate-metadata columns across every table in a SQLite database.

    Renames each of ``row``/``column``/``col``/``plate``/``field``/``channel`` to
    the canonical spacr column name (``rowID``/``columnID``/``plateID``/…). Skips
    a table when the target name already exists to avoid clashes.

    :param db_path: Path to the SQLite database file to update in place.
    :returns: None.
    """
    # map old column names -> new names
    rename_map = {
        'row':      'rowID',
        'column':   'columnID',
        'col':      'columnID',
        'plate':    'plateID',
        'field':    'fieldID',
        'channel':  'chanID',
    }

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # 1) get all user tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    for table in tables:
        # 2) get column names only
        cur.execute(f"PRAGMA table_info(`{table}`);")
        cols = [row[1] for row in cur.fetchall()]

        # 3) for each old→new, if the old exists and new does not, rename it
        for old, new in rename_map.items():
            if old in cols and new not in cols:
                sql = f"ALTER TABLE `{table}` RENAME COLUMN `{old}` TO `{new}`;"
                cur.execute(sql)
                print(f"Renamed `{table}`.`{old}` → `{new}`")

    con.commit()
    con.close()    
        
def group_feature_class(df, feature_groups=None, name='compartment'):
    """Add a column tagging each feature with its compartment (or other group) label.

    Matches feature names against the tokens in ``feature_groups`` and stores the
    result in a new column ``name``. When ``name == 'channel'``, unmatched
    features are relabeled ``'morphology'``.

    :param df: DataFrame with a ``feature`` column.
    :param feature_groups: Iterable of substrings/regex tokens to look for in each
        feature name. Defaults to ``['cell', 'cytoplasm', 'nucleus', 'pathogen']``.
    :param name: Name of the column added to ``df``. Default ``'compartment'``.
    :returns: ``df`` with the new group column populated.
    """
    # Function to determine compartment based on multiple matches
    if feature_groups is None:
        feature_groups = ['cell', 'cytoplasm', 'nucleus', 'pathogen']
    def find_feature_class(feature, compartments):
        """Return the group label(s) matched in ``feature`` — joined with '-' when more than one hits."""
        matches = [compartment for compartment in compartments if re.search(compartment, feature)]
        if len(matches) > 1:
            return '-'.join(matches)
        elif matches:
            return matches[0]
        else:
            return None
        
    from .plot import spacrGraph

    df[name] = df['feature'].apply(lambda x: find_feature_class(x, feature_groups))
    
    if name == 'channel':
        df['channel'].fillna('morphology', inplace=True)
    
    # Create new DataFrame with summed importance for each compartment and channel
    importance_sum = df.groupby(name)['importance'].sum().reset_index(name=f'{name}_importance_sum')
    total_compartment_importance = importance_sum[f'{name}_importance_sum'].sum()
    importance_sum = pd.concat(
        [importance_sum,
         pd.DataFrame(
             [{name: 'all', '{name}_importance_sum': total_compartment_importance}])]
        , ignore_index=True)
    
    return df

def delete_intermedeate_files(settings):
    """Remove intermediate per-channel and stack folders under ``settings['src']``.

    Safeguarded to only run when a ``merged/`` folder is present and the ``orig/``
    backup folder exists, so raw inputs are preserved.

    :param settings: Dict with an ``'src'`` key naming the run's root folder.
    :returns: None.
    """
    path_orig = os.path.join(settings['src'], 'orig')
    path_stack = os.path.join(settings['src'], 'stack')
    merged_stack = os.path.join(settings['src'], 'merged')
    path_norm_chan_stack = os.path.join(settings['src'], 'masks')
    path_1 = os.path.join(settings['src'], '1')
    path_2 = os.path.join(settings['src'], '2')
    path_3 = os.path.join(settings['src'], '3')
    path_4 = os.path.join(settings['src'], '4')
    path_5 = os.path.join(settings['src'], '5')
    path_6 = os.path.join(settings['src'], '6')
    path_7 = os.path.join(settings['src'], '7')
    path_8 = os.path.join(settings['src'], '8')
    path_9 = os.path.join(settings['src'], '9')
    path_10 = os.path.join(settings['src'], '10')
    
    paths = [path_stack, path_norm_chan_stack, path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9, path_10]
    
    merged_len = len(merged_stack)
    stack_len = len(path_stack)
    
    if merged_len == stack_len and stack_len != 0:
        if 'src' in settings:
            if os.path.exists(settings['src']):
                if os.path.exists(path_orig):
                    for path in paths:
                        if os.path.exists(path):
                            try:
                                shutil.rmtree(path)
                                print(f"Deleted {path}")
                            except OSError as e:
                                print(f"{path} could not be deleted: {e}. Delete manually.")
                else:
                    print(f"{path_orig} does not exist.")
            else:
                print(f"{settings['src']} does not exist.")
        else:
            print("No 'src' key in settings dictionary.")
        
def filter_and_save_csv(input_csv, output_csv, column_name, upper_threshold, lower_threshold):
    """
    Reads a CSV into a DataFrame, filters rows based on a column for values > upper_threshold and < lower_threshold,
    and saves the filtered DataFrame to a new CSV file.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the filtered CSV file.
        column_name (str): Column name to apply the filters on.
        upper_threshold (float): Upper threshold for filtering (values greater than this are retained).
        lower_threshold (float): Lower threshold for filtering (values less than this are retained).

    Returns:
        None
    """
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Filter rows based on the thresholds
    filtered_df = df[(df[column_name] > upper_threshold) | (df[column_name] < lower_threshold)]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    display(filtered_df)

    print(f"Filtered DataFrame saved to {output_csv}")
    
def extract_tar_bz2_files(folder_path):
    """
    Extracts all .tar.bz2 files in the given folder into subfolders with the same name as the tar file.
    
    Parameters:
        folder_path (str): Path to the folder containing .tar.bz2 files.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid folder.")
    
    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tar.bz2'):
            file_path = os.path.join(folder_path, file_name)
            extract_folder = os.path.join(folder_path, os.path.splitext(os.path.splitext(file_name)[0])[0])
            
            # Create the subfolder for extraction if it doesn't exist
            os.makedirs(extract_folder, exist_ok=True)
            
            # Extract the tar.bz2 file
            try:
                with tarfile.open(file_path, 'r:bz2') as tar:
                    tar.extractall(path=extract_folder)
                print(f"Extracted: {file_name} -> {extract_folder}")
            except Exception as e:
                print(f"Failed to extract {file_name}: {e}")
            
            
def calculate_shortest_distance(df, object1, object2):
    """
    Calculate the shortest edge-to-edge distance between two objects (e.g., pathogen and nucleus).
    
    Parameters:
    - df: Pandas DataFrame containing measurements
    - object1: String, name of the first object (e.g., "pathogen")
    - object2: String, name of the second object (e.g., "nucleus")

    Returns:
    - df: Pandas DataFrame with a new column for shortest edge-to-edge distance.
    """

    # Compute centroid-to-centroid Euclidean distance
    centroid_distance = np.sqrt(
        (df[f'{object1}_channel_0_centroid_weighted-0'] - df[f'{object2}_channel_0_centroid_weighted-0'])**2 +
        (df[f'{object1}_channel_0_centroid_weighted-1'] - df[f'{object2}_channel_0_centroid_weighted-1'])**2
    )

    # Estimate object radii using Feret diameters
    object1_radius = df[f'{object1}_feret_diameter_max'] / 2
    object2_radius = df[f'{object2}_feret_diameter_max'] / 2

    # Compute shortest edge-to-edge distance
    shortest_distance = centroid_distance - (object1_radius + object2_radius)

    # Ensure distances are non-negative (overlapping objects should have distance 0)
    df[f'{object1}_{object2}_shortest_distance'] = np.maximum(shortest_distance, 0)

    return df

def format_path_for_system(path):
    """
    Takes a file path and reformats it to be compatible with the current operating system.
    
    Args:
        path (str): The file path to be formatted.

    Returns:
        str: The formatted path for the current operating system.
    """
    system = platform.system()
    
    # Convert Windows-style paths to Unix-style (Linux/macOS)
    if system in ["Linux", "Darwin"]:  # Darwin is macOS
        formatted_path = path.replace("\\", "/")
    
    # Convert Unix-style paths to Windows-style
    elif system == "Windows":
        formatted_path = path.replace("/", "\\")
    
    else:
        raise ValueError(f"Unsupported OS: {system}")
    
    # Normalize path to ensure consistency
    new_path = os.path.normpath(formatted_path)
    if os.path.exists(new_path):
        print(f"Found path: {new_path}")
    else:
        print(f"Path not found: {new_path}")
        
    return new_path


def normalize_src_path(src):
    """
    Ensures that the 'src' value is properly formatted as either a list of strings or a single string.

    Args:
        src (str or list): The input source path(s).

    Returns:
        list or str: A correctly formatted list if the input was a list (or string representation of a list),
                     otherwise a single string.
    """
    if isinstance(src, list):
        return src  # Already a list, return as-is

    if isinstance(src, str):
        try:
            # Check if it is a string representation of a list
            evaluated_src = ast.literal_eval(src)
            if isinstance(evaluated_src, list) and all(isinstance(item, str) for item in evaluated_src):
                return evaluated_src  # Convert to real list
        except (SyntaxError, ValueError):
            pass  # Not a valid list, treat as a string

        return src  # Return as a string if not a list

    raise ValueError(f"Invalid type for 'src': {type(src).__name__}, expected str or list")

def generate_image_path_map(root_folder, valid_extensions=("tif", "tiff", "png", "jpg", "jpeg", "bmp", "czi", "nd2", "lif")):
    """
    Recursively scans a folder and its subfolders for images, then creates a mapping of:
    {original_image_path: new_image_path}, where the new path includes all subfolder names.

    Args:
        root_folder (str): The root directory to scan for images.
        valid_extensions (tuple): Tuple of valid image file extensions.

    Returns:
        dict: A dictionary mapping original image paths to their new paths.
    """
    image_path_map = {}

    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            ext = file.lower().split('.')[-1]
            if ext in valid_extensions:
                # Get relative path of the image from root_folder
                relative_path = os.path.relpath(dirpath, root_folder)
                
                # Construct new filename: Embed folder hierarchy into the name
                folder_parts = relative_path.split(os.sep)  # Get all folder names
                folder_info = "_".join(folder_parts) if folder_parts else ""  # Join with underscores
                
                # Generate new filename
                new_filename = f"{folder_info}_{file}" if folder_info else file

                # Store in dictionary (original path -> new path)
                original_path = os.path.join(dirpath, file)
                new_path = os.path.join(root_folder, new_filename)
                image_path_map[original_path] = new_path

    return image_path_map

def copy_images_to_consolidated(image_path_map, root_folder):
    """
    Copies images from their original locations to a 'consolidated' folder,
    renaming them according to the generated dictionary.

    Args:
        image_path_map (dict): Dictionary mapping {original_path: new_path}.
        root_folder (str): The root directory where the 'consolidated' folder will be created.
    """
    
    consolidated_folder = os.path.join(root_folder, "consolidated")
    os.makedirs(consolidated_folder, exist_ok=True)  # Ensure 'consolidated' folder exists
    files_processed = 0
    files_to_process = len(image_path_map)
    time_ls= []
    
    for original_path, new_path in image_path_map.items():
        
        start = time.time()
        new_filename = os.path.basename(new_path)  # Extract only the new filename
        new_file_path = os.path.join(consolidated_folder, new_filename)  # Place in 'consolidated' folder
        
        shutil.copy2(original_path, new_file_path)  # Copy file with metadata preserved
        
        files_processed += 1
        stop = time.time()
        duration = (stop - start)
        time_ls.append(duration)
        
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'Consolidating images')
        #print(f"Copied: {original_path} -> {new_file_path}")
        
def correct_metadata(df):
    """Normalize a metadata DataFrame to the canonical spacr column names and plate ID form.

    Strips a duplicated ``pp`` prefix from plate IDs, promotes legacy
    ``*_name`` columns to their ID equivalents, and renames
    ``row``/``col``/``column``/``field`` (and their ``*_name`` variants) to
    ``rowID``/``columnID``/``fieldID``.

    :param df: Metadata DataFrame that may still use legacy naming.
    :returns: The DataFrame with canonical columns.
    """
    #if 'object' in df.columns:
    #    df['objectID'] = df['object']
    # delete these four lines in 2027
    if 'plateID' in df.columns:
        df["plateID"] = df["plateID"].str.replace(r"^pp", "p", regex=True)
        
    if 'prcfo' in df.columns:
        df["prcfo"] = df["prcfo"].str.replace(r"^pp", "p", regex=True)
        
    if 'object_name' in df.columns:
        df['objectID'] = df['object_name']
    
    if 'field_name' in df.columns:
        df['fieldID'] = df['field_name']
    
    if 'plate' in df.columns:
        df['plateID'] = df['plate']
    
    if 'plate_name' in df.columns:
        df['plateID'] = df['plate_name']
    
    if 'row' in df.columns:
        df = df.rename(columns={'row': 'rowID'})
        
    if 'row_name' in df.columns:
        df = df.rename(columns={'row_name': 'rowID'})
        
    if 'col' in df.columns:
        df = df.rename(columns={'col': 'columnID'})
        
    if 'column' in df.columns:
        df = df.rename(columns={'column': 'columnID'})
        
    if 'column_name' in df.columns:
        df = df.rename(columns={'column_name': 'columnID'})
        
    if 'field' in df.columns:
        df = df.rename(columns={'field': 'fieldID'})

    if 'field_name' in df.columns:
        df = df.rename(columns={'field_name': 'fieldID'})
    
    return df

def remove_outliers_by_group(df, group_col, value_col, method='iqr', threshold=1.5):
    """
    Removes outliers from `value_col` within each group defined by `group_col`.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): Column name to group by.
        value_col (str): Column containing values to check for outliers.
        method (str): 'iqr' or 'zscore'.
        threshold (float): Threshold multiplier for IQR (default 1.5) or z-score.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    def iqr_filter(subdf):
        """Return rows of ``subdf`` whose ``value_col`` falls within ``threshold * IQR`` of Q1/Q3."""
        q1 = subdf[value_col].quantile(0.25)
        q3 = subdf[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return subdf[(subdf[value_col] >= lower) & (subdf[value_col] <= upper)]

    def zscore_filter(subdf):
        """Return rows of ``subdf`` whose ``value_col`` is within ``threshold`` standard deviations of the mean."""
        mean = subdf[value_col].mean()
        std = subdf[value_col].std()
        return subdf[(subdf[value_col] - mean).abs() <= threshold * std]

    if method == 'iqr':
        return df.groupby(group_col, group_keys=False).apply(iqr_filter)
    elif method == 'zscore':
        return df.groupby(group_col, group_keys=False).apply(zscore_filter)
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
