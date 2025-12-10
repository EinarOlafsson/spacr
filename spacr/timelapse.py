import cv2, os, re, glob, random, btrack, sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib as mpl
from IPython.display import display
from IPython.display import Image as ipyimage
import trackpy as tp
from btrack import datasets as btrack_datasets
from skimage.measure import regionprops_table
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, linear_sum_assignment
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from multiprocessing import Pool, cpu_count
import logging

try:
    from numpy import trapz
except ImportError:
    from scipy.integrate import trapz
    
import matplotlib.pyplot as plt

import logging
from spacr.utils import debug


def _npz_to_movie(arrays, filenames, save_path, fps=10):
    """
    Convert a list of numpy arrays to a movie file.

    Args:
        arrays (List[np.ndarray]): List of numpy arrays representing frames of the movie.
        filenames (List[str]): List of filenames corresponding to each frame.
        save_path (str): Path to save the movie file.
        fps (int, optional): Frames per second of the movie. Defaults to 10.

    Returns:
        None
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if save_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter with the size of the first image
    height, width = arrays[0].shape[:2]
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for i, frame in enumerate(arrays):
        # Handle float32 images by scaling or normalizing
        if frame.dtype == np.float32:
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)

        # Convert 16-bit image to 8-bit
        elif frame.dtype == np.uint16:
            frame = cv2.convertScaleAbs(frame, alpha=(255.0/65535.0))

        # Handling 1-channel (grayscale) or 2-channel images
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in [1, 2]):
            if frame.ndim == 2 or frame.shape[2] == 1:
                # Convert grayscale to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 2:
                # Create an RGB image with the first channel as red, second as green, blue set to zero
                rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_frame[..., 0] = frame[..., 0]  # Red channel
                rgb_frame[..., 1] = frame[..., 1]  # Green channel
                frame = rgb_frame

        # For 3-channel images, ensure it's in BGR format for OpenCV
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add filenames as text on frames
        cv2.putText(frame, filenames[i], (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Movie saved to {save_path}")
    
def _scmovie(folder_paths):
        """
        Generate movies from a collection of PNG images in the given folder paths.

        Args:
            folder_paths (list): List of folder paths containing PNG images.

        Returns:
            None
        """
        folder_paths = list(set(folder_paths))
        for folder_path in folder_paths:
            movie_path = os.path.join(folder_path, 'movies')
            os.makedirs(movie_path, exist_ok=True)
            # Regular expression to parse the filename
            filename_regex = re.compile(r'(\w+)_(\w+)_(\w+)_(\d+)_(\d+).png')
            # Dictionary to hold lists of images by plate, well, field, and object number
            grouped_images = defaultdict(list)
            # Iterate over all PNG files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    match = filename_regex.match(filename)
                    if match:
                        plate, well, field, time, object_number = match.groups()
                        key = (plate, well, field, object_number)
                        grouped_images[key].append((int(time), os.path.join(folder_path, filename)))
            for key, images in grouped_images.items():
                # Sort images by time using sorted and lambda function for custom sort key
                images = sorted(images, key=lambda x: x[0])
                _, image_paths = zip(*images)
                # Determine the size to which all images should be padded
                max_height = max_width = 0
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    h, w, _ = image.shape
                    max_height, max_width = max(max_height, h), max(max_width, w)
                # Initialize VideoWriter
                plate, well, field, object_number = key
                output_filename = f"{plate}_{well}_{field}_{object_number}.mp4"
                output_path = os.path.join(movie_path, output_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(output_path, fourcc, 10, (max_width, max_height))
                # Process each image
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    h, w, _ = image.shape
                    padded_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    padded_image[:h, :w, :] = image
                    video.write(padded_image)
                video.release()
                
                
def _sort_key(file_path):
    """
    Returns a sort key for the given file path based on the pattern '(\d+)_([A-Z]\d+)_(\d+)_(\d+).npy'.
    The sort key is a tuple containing the plate, well, field, and time values extracted from the file path.
    If the file path does not match the pattern, a default sort key is returned to sort the file as "earliest" or "lowest".

    Args:
        file_path (str): The file path to extract the sort key from.

    Returns:
        tuple: The sort key tuple containing the plate, well, field, and time values.
    """
    match = re.search(r'(\d+)_([A-Z]\d+)_(\d+)_(\d+).npy', os.path.basename(file_path))
    if match:
        plate, well, field, time = match.groups()
        # Assuming plate, well, and field are to be returned as is and time converted to int for sorting
        return (plate, well, field, int(time))
    else:
        # Return a tuple that sorts this file as "earliest" or "lowest"
        return ('', '', '', 0)

def _masks_to_gif(masks, gif_folder, name, filenames, object_type):
    """
    Converts a sequence of masks into a GIF file.

    Args:
        masks (list): List of masks representing the sequence.
        gif_folder (str): Path to the folder where the GIF file will be saved.
        name (str): Name of the GIF file.
        filenames (list): List of filenames corresponding to each mask in the sequence.
        object_type (str): Type of object represented by the masks.

    Returns:
        None
    """

    from .io import _save_mask_timelapse_as_gif

    def _display_gif(path):
        with open(path, 'rb') as file:
            display(ipyimage(file.read()))

    highest_label = max(np.max(mask) for mask in masks)
    random_colors = np.random.rand(highest_label + 1, 4)
    random_colors[:, 3] = 1  # Full opacity
    random_colors[0] = [0, 0, 0, 1]  # Background color
    cmap = plt.cm.colors.ListedColormap(random_colors)
    norm = plt.cm.colors.Normalize(vmin=0, vmax=highest_label)

    save_path_gif = os.path.join(gif_folder, f'timelapse_masks_{object_type}_{name}.gif')
    _save_mask_timelapse_as_gif(masks, None, save_path_gif, cmap, norm, filenames)
    #_display_gif(save_path_gif)
    
def _timelapse_masks_to_gif(folder_path, mask_channels, object_types):
    """
    Converts a sequence of masks into a timelapse GIF file.

    Args:
        folder_path (str): The path to the folder containing the mask files.
        mask_channels (list): List of channel indices to extract masks from.
        object_types (list): List of object types corresponding to each mask channel.

    Returns:
        None
    """
    master_folder = os.path.dirname(folder_path)
    gif_folder = os.path.join(master_folder, 'movies', 'gif')
    os.makedirs(gif_folder, exist_ok=True)

    paths = glob.glob(os.path.join(folder_path, '*.npy'))
    paths.sort(key=_sort_key)

    organized_files = {}
    for file in paths:
        match = re.search(r'(\d+)_([A-Z]\d+)_(\d+)_\d+.npy', os.path.basename(file))
        if match:
            plate, well, field = match.groups()
            key = (plate, well, field)
            if key not in organized_files:
                organized_files[key] = []
            organized_files[key].append(file)

    for key, file_list in organized_files.items():
        # Generate the name for the GIF based on plate, well, field
        name = f'{key[0]}_{key[1]}_{key[2]}'
        save_path_gif = os.path.join(gif_folder, f'timelapse_masks_{name}.gif')

        for i, mask_channel in enumerate(mask_channels):
            object_type = object_types[i]
            # Initialize an empty list to store masks for the current object type
            mask_arrays = []

            for file in file_list:
                # Load only the current time series array
                array = np.load(file)
                # Append the specific channel mask to the mask_arrays list
                mask_arrays.append(array[:, :, mask_channel])

            # Convert mask_arrays list to a numpy array for processing
            mask_arrays_np = np.array(mask_arrays)
            # Generate filenames for each frame in the time series
            filenames = [os.path.basename(f) for f in file_list]
            # Create the GIF for the current time series and object type
            _masks_to_gif(mask_arrays_np, gif_folder, name, filenames, object_type)
            
def _relabel_masks_based_on_tracks(masks, tracks, mode='btrack'):
    """
    Relabels the masks based on the tracks DataFrame.

    Args:
        masks (ndarray): Input masks array with shape (num_frames, height, width).
        tracks (DataFrame): DataFrame containing track information.
        mode (str, optional): Mode for relabeling. Defaults to 'btrack'.

    Returns:
        ndarray: Relabeled masks array with the same shape and dtype as the input masks.
    """
    # Initialize an array to hold the relabeled masks with the same shape and dtype as the input masks
    relabeled_masks = np.zeros(masks.shape, dtype=masks.dtype)

    # Iterate through each frame
    for frame_number in range(masks.shape[0]):
        # Extract the mapping for the current frame from the tracks DataFrame
        frame_tracks = tracks[tracks['frame'] == frame_number]
        mapping = dict(zip(frame_tracks['original_label'], frame_tracks['track_id']))
        current_mask = masks[frame_number, :, :]

        # Apply the mapping to the current mask
        for original_label, new_label in mapping.items():
            # Where the current mask equals the original label, set it to the new label value
            relabeled_masks[frame_number][current_mask == original_label] = new_label

    return relabeled_masks

def _prepare_for_tracking(mask_array):
    frames = []
    for t, frame in enumerate(mask_array):
        props = regionprops_table(
            frame,
            properties=('label', 'centroid', 'area', 'bbox', 'eccentricity')
        )
        df = pd.DataFrame(props)
        df = df.rename(columns={
            'centroid-0': 'y',
            'centroid-1': 'x',
            'area':       'mass',
            'label':      'original_label'
        })
        df['frame'] = t
        frames.append(df[['frame','y','x','mass','original_label',
                          'bbox-0','bbox-1','bbox-2','bbox-3','eccentricity']])
    return pd.concat(frames, ignore_index=True)

def _track_by_iou(masks, iou_threshold=0.1):
    """
    Build a track table by linking masks frame→frame via IoU.
    Returns a DataFrame with columns [frame, original_label, track_id].
    """
    n_frames = masks.shape[0]
    # 1) initialize: every label in frame 0 starts its own track
    labels0 = np.unique(masks[0])[1:]
    next_track = 1
    track_map = {}  # (frame,label) -> track_id
    for L in labels0:
        track_map[(0, L)] = next_track
        next_track += 1

    # 2) iterate through frames
    for t in range(1, n_frames):
        prev, curr = masks[t-1], masks[t]
        matches = link_by_iou(prev, curr, iou_threshold=iou_threshold)
        used_curr = set()
        # a) assign matched labels to existing tracks
        for L_prev, L_curr in matches:
            tid = track_map[(t-1, L_prev)]
            track_map[(t, L_curr)] = tid
            used_curr.add(L_curr)
        # b) any label in curr not matched → new track
        for L in np.unique(curr)[1:]:
            if L not in used_curr:
                track_map[(t, L)] = next_track
                next_track += 1

    # 3) flatten into DataFrame
    records = []
    for (frame, label), tid in track_map.items():
        records.append({'frame': frame, 'original_label': label, 'track_id': tid})
    return pd.DataFrame(records)

def link_by_iou(mask_prev, mask_next, iou_threshold=0.1):
    # Get labels
    labels_prev = np.unique(mask_prev)[1:]
    labels_next = np.unique(mask_next)[1:]
    # Precompute masks as boolean
    bool_prev = {L: mask_prev==L for L in labels_prev}
    bool_next = {L: mask_next==L for L in labels_next}
    # Cost matrix = 1 - IoU
    cost = np.ones((len(labels_prev), len(labels_next)), dtype=float)
    for i, L1 in enumerate(labels_prev):
        m1 = bool_prev[L1]
        for j, L2 in enumerate(labels_next):
            m2 = bool_next[L2]
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            if union > 0:
                cost[i, j] = 1 - inter/union
    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost[i,j] <= 1 - iou_threshold:
            matches.append((labels_prev[i], labels_next[j]))
    return matches

def _find_optimal_search_range(features, initial_search_range=500, increment=10, max_attempts=49, memory=3):
    """
    Find the optimal search range for linking features.

    Args:
        features (list): List of features to be linked.
        initial_search_range (int, optional): Initial search range. Defaults to 500.
        increment (int, optional): Increment value for reducing the search range. Defaults to 10.
        max_attempts (int, optional): Maximum number of attempts to find the optimal search range. Defaults to 49.
        memory (int, optional): Memory parameter for linking features. Defaults to 3.

    Returns:
        int: The optimal search range for linking features.
    """
    optimal_search_range = initial_search_range
    for attempt in range(max_attempts):
        try:
            # Attempt to link features with the current search range
            tracks_df = tp.link(features, search_range=optimal_search_range, memory=memory)
            print(f"Success with search_range={optimal_search_range}")
            return optimal_search_range
        except Exception as e:
            #print(f"SubnetOversizeException with search_range={optimal_search_range}: {e}")
            optimal_search_range -= increment
            print(f'Retrying with displacement value: {optimal_search_range}', end='\r', flush=True)
    min_range = initial_search_range-(max_attempts*increment)
    if optimal_search_range <= min_range:
        print(f'timelapse_displacement={optimal_search_range} is too high. Lower timelapse_displacement or set to None for automatic thresholding.')
    return optimal_search_range

def _remove_objects_from_first_frame(masks, percentage=10):
        """
        Removes a specified percentage of objects from the first frame of a sequence of masks.

        Parameters:
        masks (ndarray): Sequence of masks representing the frames.
        percentage (int): Percentage of objects to remove from the first frame.

        Returns:
        ndarray: Sequence of masks with objects removed from the first frame.
        """
        first_frame = masks[0]
        unique_labels = np.unique(first_frame[first_frame != 0])
        num_labels_to_remove = max(1, int(len(unique_labels) * (percentage / 100)))
        labels_to_remove = random.sample(list(unique_labels), num_labels_to_remove)

        for label in labels_to_remove:
            masks[0][first_frame == label] = 0
        return masks

def _track_by_iou(masks, iou_threshold=0.1):
    """
    Build a track table by linking masks frame→frame via IoU.
    Returns a DataFrame with columns [frame, original_label, track_id].
    """
    n_frames = masks.shape[0]
    # 1) initialize: every label in frame 0 starts its own track
    labels0 = np.unique(masks[0])[1:]
    next_track = 1
    track_map = {}  # (frame,label) -> track_id
    for L in labels0:
        track_map[(0, L)] = next_track
        next_track += 1

    # 2) iterate through frames
    for t in range(1, n_frames):
        prev, curr = masks[t-1], masks[t]
        matches = link_by_iou(prev, curr, iou_threshold=iou_threshold)
        used_curr = set()
        # a) assign matched labels to existing tracks
        for L_prev, L_curr in matches:
            tid = track_map[(t-1, L_prev)]
            track_map[(t, L_curr)] = tid
            used_curr.add(L_curr)
        # b) any label in curr not matched → new track
        for L in np.unique(curr)[1:]:
            if L not in used_curr:
                track_map[(t, L)] = next_track
                next_track += 1

    # 3) flatten into DataFrame
    records = []
    for (frame, label), tid in track_map.items():
        records.append({'frame': frame, 'original_label': label, 'track_id': tid})
    return pd.DataFrame(records)

def _facilitate_trackin_with_adaptive_removal(masks, search_range=None, max_attempts=5, memory=3, min_mass=50, track_by_iou=False):
    """
    Facilitates object tracking with deterministic initial filtering and
    trackpy’s constant-velocity prediction.

    Args:
        masks (np.ndarray): integer‐labeled masks (frames × H × W).
        search_range (int|None): max displacement; if None, auto‐computed.
        max_attempts (int): how many times to retry with smaller search_range.
        memory (int): trackpy memory parameter.
        min_mass (float): drop any object in frame 0 with area < min_mass.

    Returns:
        masks, features_df, tracks_df

    Raises:
        RuntimeError if linking fails after max_attempts.
    """
    # 1) initial features & filter frame 0 by area
    features = _prepare_for_tracking(masks)
    f0 = features[features['frame'] == 0]
    valid = f0.loc[f0['mass'] >= min_mass, 'original_label'].unique()
    masks[0] = np.where(np.isin(masks[0], valid), masks[0], 0)

    # 2) recompute features on filtered masks
    features = _prepare_for_tracking(masks)

    # 3) default search_range = 2×sqrt(99th‑pct area)
    if search_range is None:
        a99 = f0['mass'].quantile(0.99)
        search_range = max(1, int(2 * np.sqrt(a99)))

    # 4) attempt linking, shrinking search_range on failure
    for attempt in range(1, max_attempts + 1):
        try:
            if track_by_iou:
                tracks_df = _track_by_iou(masks, iou_threshold=0.1)
            else:
                tracks_df = tp.link_df(features,search_range=search_range, memory=memory, predict=True)
                print(f"Linked on attempt {attempt} with search_range={search_range}")
            return masks, features, tracks_df

        except Exception as e:
            search_range = max(1, int(search_range * 0.8))
            print(f"Attempt {attempt} failed ({e}); reducing search_range to {search_range}")

    raise RuntimeError(
        f"Failed to track after {max_attempts} attempts; last search_range={search_range}"
    )

def _trackpy_track_cells(src, name, batch_filenames, object_type, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, mode, track_by_iou):
        """
        Track cells using the Trackpy library.

        Args:
            src (str): The source file path.
            name (str): The name of the track.
            batch_filenames (list): List of batch filenames.
            object_type (str): The type of object to track.
            masks (list): List of masks.
            timelapse_displacement (int): The displacement for timelapse tracking.
            timelapse_memory (int): The memory for timelapse tracking.
            timelapse_remove_transient (bool): Whether to remove transient objects in timelapse tracking.
            plot (bool): Whether to plot the tracks.
            save (bool): Whether to save the tracks.
            mode (str): The mode of tracking.

        Returns:
            list: The mask stack.

        """
        
        from .plot import _visualize_and_save_timelapse_stack_with_tracks
        from .utils import _masks_to_masks_stack
        
        print(f'Tracking objects with trackpy')

        if timelapse_displacement is None:
            features = _prepare_for_tracking(masks)
            timelapse_displacement = _find_optimal_search_range(features, initial_search_range=500, increment=10, max_attempts=49, memory=3)
            if timelapse_displacement is None:
                timelapse_displacement = 50

        masks, features, tracks_df = _facilitate_trackin_with_adaptive_removal(masks, search_range=timelapse_displacement, max_attempts=100, memory=timelapse_memory, track_by_iou=track_by_iou)

        tracks_df['particle'] += 1

        if timelapse_remove_transient:
            tracks_df_filter = tp.filter_stubs(tracks_df, len(masks))
        else:
            tracks_df_filter = tracks_df.copy()

        tracks_df_filter = tracks_df_filter.rename(columns={'particle': 'track_id'})
        print(f'Removed {len(tracks_df)-len(tracks_df_filter)} objects that were not present in all frames')
        masks = _relabel_masks_based_on_tracks(masks, tracks_df_filter)
        tracks_path = os.path.join(os.path.dirname(src), 'tracks')
        os.makedirs(tracks_path, exist_ok=True)
        tracks_df_filter.to_csv(os.path.join(tracks_path, f'trackpy_tracks_{object_type}_{name}.csv'), index=False)
        if plot or save:
            _visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df_filter, save, src, name, plot, batch_filenames, object_type, mode)

        mask_stack = _masks_to_masks_stack(masks)
        return mask_stack

def _filter_short_tracks(df, min_length=5):
    """Filter out tracks that are shorter than min_length.

    Args:
        df (pandas.DataFrame): The input DataFrame containing track information.
        min_length (int, optional): The minimum length of tracks to keep. Defaults to 5.

    Returns:
        pandas.DataFrame: The filtered DataFrame with only tracks longer than min_length.
    """
    track_lengths = df.groupby('track_id').size()
    long_tracks = track_lengths[track_lengths >= min_length].index
    return df[df['track_id'].isin(long_tracks)]

@debug(enabled=True)
def _btrack_track_cells(src, name, batch_filenames, object_type, plot, save, masks_3D, mode, timelapse_remove_transient, radius=100, n_jobs=10, batch_list=None, optimizer_time_limit_s=120, optimizer_mip_gap=0.01, run_optimization=True, max_objects_for_optimization=20000):
    """
    Track cells using the btrack library.

    Args:
        src (str): The source file path.
        name (str): The name of the track (npz batch name).
        batch_filenames (list[str]): Filenames for frames in this batch.
        object_type (str): The type of object to track (cell, nucleus, pathogen).
        plot (bool): Whether to plot the tracks.
        save (bool): Whether to save plots.
        masks_3D (ndarray or list): 3D label array of masks with shape (T, Y, X),
            or list of 2D (Y, X) label arrays (one per frame).
        mode (str): The tracking mode (unused here but kept for API consistency).
        timelapse_remove_transient (bool): Whether to remove short tracks.
        radius (int or None, optional): Max search radius (pixels). If None,
            it is set automatically to image_width / 20. Defaults to 100.
        n_jobs (int, optional): Number of workers for object extraction. Defaults to 10.
        batch_list (list or None, optional): List of intensity images used by Cellpose.
            Currently not used by btrack (tracking is shape-based here), but kept
            for API compatibility and possible future use.
        optimizer_time_limit_s (float or None): Time limit for GLPK in seconds
            (used only when global optimisation is actually run).
        optimizer_mip_gap (float or None): Relative MIP gap for GLPK (0.01 = 1%).
        run_optimization (bool): If False, skip global optimisation entirely.
        max_objects_for_optimization (int or None): If not None, skip global
            optimisation when the number of objects exceeds this threshold.

    Returns:
        ndarray: The relabelled mask stack (same shape as masks_3D) where labels
        are track IDs.
    """
    import os
    import logging

    import numpy as np
    import pandas as pd
    import btrack
    from btrack import datasets as btrack_datasets
    from btrack.constants import BayesianUpdates

    from .plot import _visualize_and_save_timelapse_stack_with_tracks
    from .utils import _masks_to_masks_stack, _map_wells

    logger = logging.getLogger(__name__)

    logger.debug(
        "Entering _btrack_track_cells: name=%s, object_type=%s, mode=%s",
        name,
        object_type,
        mode,
    )
    logger.debug("src=%s", src)
    logger.debug(
        "timelapse_remove_transient=%s, radius=%s, n_jobs=%s, "
        "optimizer_time_limit_s=%s, optimizer_mip_gap=%s, run_optimization=%s, "
        "max_objects_for_optimization=%s",
        timelapse_remove_transient,
        radius,
        n_jobs,
        optimizer_time_limit_s,
        optimizer_mip_gap,
        run_optimization,
        max_objects_for_optimization,
    )
    logger.debug("masks_3D type: %s", type(masks_3D))

    # ------------------------------------------------------------------
    # Normalise masks_3D to a 3D ndarray (T, Y, X)
    # ------------------------------------------------------------------
    if isinstance(masks_3D, list):
        logger.debug("masks_3D is a list with length=%d", len(masks_3D))
        if len(masks_3D) == 0:
            raise ValueError("masks_3D is an empty list; nothing to track.")
        masks_3D = [np.asarray(m) for m in masks_3D]
        shapes = {m.shape for m in masks_3D}
        logger.debug("Unique mask shapes in list: %s", shapes)
        if len(shapes) != 1:
            raise ValueError(
                f"All masks must have the same shape; got shapes={shapes}."
            )
        masks_3D = np.stack(masks_3D, axis=0)
        logger.debug("Stacked masks_3D into ndarray with shape %s", masks_3D.shape)
    else:
        masks_3D = np.asarray(masks_3D)
        logger.debug("masks_3D array shape: %s", masks_3D.shape)

    if masks_3D.ndim != 3:
        raise ValueError(
            f"masks_3D must be 3D (T, Y, X); got shape {masks_3D.shape}"
        )

    n_frames, height, width = masks_3D.shape
    logger.debug(
        "Parsed geometry: n_frames=%d, height=%d, width=%d",
        n_frames,
        height,
        width,
    )

    # Auto radius if requested
    if radius is None:
        radius = max(1, width // 20)
        logger.debug(
            "radius was None; automatically set radius=%d (width/20)", radius
        )

    # ------------------------------------------------------------------
    # btrack configuration and feature definition
    # ------------------------------------------------------------------
    CONFIG_FILE = btrack_datasets.cell_config()
    
    # Shape-based features only (robust + what your config already expects)
    FEATURES = [
        "area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "solidity",
    ]
    TRACKING_UPDATES = ["motion", "visual"]

    # ------------------------------------------------------------------
    # Convert segmentation to btrack objects
    # ------------------------------------------------------------------
    logger.debug("Converting segmentation to btrack objects...")
    objects = btrack.utils.segmentation_to_objects(
        masks_3D,
        properties=tuple(FEATURES),
        num_workers=n_jobs,
    )
    n_objects = len(objects)
    logger.info("Extracted %d objects for tracking.", n_objects)

    # ------------------------------------------------------------------
    # Run the Bayesian tracker
    # ------------------------------------------------------------------
    with btrack.BayesianTracker() as tracker:
        tracker.configure(CONFIG_FILE)

        # Use APPROXIMATE updates for large datasets (recommended by btrack docs)
        tracker.update_method = BayesianUpdates.APPROXIMATE
        tracker.max_search_radius = radius

        # Features used by the visual model
        tracker.features = FEATURES

        # Append objects and define volume
        tracker.append(objects)
        tracker.volume = ((0, width), (0, height))
        logger.debug(
            "Tracker volume set to x=(0,%d), y=(0,%d); update_method=%s",
            width,
            height,
            tracker.update_method,
        )

        # Tracking
        logger.debug("Starting tracking...")
        try:
            tracker.track(tracking_updates=TRACKING_UPDATES)
        except TypeError:
            # Fallback for older btrack APIs
            logger.debug(
                "tracker.track(tracking_updates=...) not supported; "
                "falling back to tracker.tracking_updates + track(step_size=100)."
            )
            tracker.tracking_updates = [u.upper() for u in TRACKING_UPDATES]
            tracker.track(step_size=100)

        logger.info(
            "Tracking complete. Number of tracks before optimisation: %d",
            len(tracker.tracks),
        )

        # ------------------------------------------------------------------
        # Global optimisation (GLPK) – conditionally disabled for large problems
        # ------------------------------------------------------------------
        do_optimize = bool(run_optimization)

        if max_objects_for_optimization is not None and n_objects > max_objects_for_optimization:
            logger.warning(
                "Skipping btrack global optimisation: %d objects > "
                "max_objects_for_optimization=%d. Using pre-optimisation tracks.",
                n_objects,
                max_objects_for_optimization,
            )
            do_optimize = False

        if do_optimize and len(tracker.tracks) > 0:
            # Build GLPK options from user parameters
            glpk_options = {}
            if optimizer_time_limit_s is not None and optimizer_time_limit_s > 0:
                # GLPK tm_lim is in milliseconds
                glpk_options["tm_lim"] = int(optimizer_time_limit_s * 1000)
            if optimizer_mip_gap is not None and optimizer_mip_gap > 0:
                glpk_options["mip_gap"] = float(optimizer_mip_gap)

            try:
                if glpk_options:
                    logger.info(
                        "Running GLPK optimisation with options: %s", glpk_options
                    )
                    tracker.optimize(
                        backend="glpk",
                        options={"options": glpk_options},
                    )
                else:
                    logger.info("Running GLPK optimisation with default options.")
                    tracker.optimize(backend="glpk")

                logger.info(
                    "Optimisation complete. Number of tracks after optimisation: %d",
                    len(tracker.tracks),
                )
            except Exception as e:
                # If GLPK misbehaves, fall back to pre-optimisation tracks
                logger.warning(
                    "btrack global optimisation failed or stalled (%s). "
                    "Using pre-optimisation tracks instead.",
                    e,
                    exc_info=True,
                )

        # After this point, tracker.tracks always contains the tracks we will use
        tracks = tracker.tracks

    # ------------------------------------------------------------------
    # Convert tracks to DataFrame
    # ------------------------------------------------------------------
    track_data = []
    for track in tracks:
        for t, x, y, z in zip(track.t, track.x, track.y, track.z):
            track_data.append(
                {
                    "track_id": track.ID,
                    "frame": t,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    tracks_df = pd.DataFrame(track_data)
    logger.debug("tracks_df shape: %s", tracks_df.shape)

    # Optionally remove transient tracks (very short trajectories)
    if timelapse_remove_transient and not tracks_df.empty:
        logger.debug("Removing transient tracks with min_length=%d", n_frames)
        tracks_df = _filter_short_tracks(tracks_df, min_length=n_frames)
        logger.debug("tracks_df shape after filtering: %s", tracks_df.shape)

    # ------------------------------------------------------------------
    # Map track positions back to original labels
    # ------------------------------------------------------------------
    logger.debug("Preparing objects_df from masks_3D...")
    objects_df = _prepare_for_tracking(masks_3D)
    logger.debug("objects_df shape: %s", objects_df.shape)

    # Harmonise precision before merge
    tracks_df["x"] = tracks_df["x"].round(2)
    tracks_df["y"] = tracks_df["y"].round(2)
    objects_df["x"] = objects_df["x"].round(2)
    objects_df["y"] = objects_df["y"].round(2)

    logger.debug("Merging tracks_df and objects_df on ['frame', 'x', 'y']...")
    merged_df = pd.merge(
        tracks_df,
        objects_df,
        on=["frame", "x", "y"],
        how="inner",
    )
    logger.debug("merged_df shape: %s", merged_df.shape)

    final_df = merged_df[["track_id", "frame", "x", "y", "original_label"]]
    
    try:
        final_df['file_name'] = name
        final_df[['plateID', 'rowID', 'columnID', 'fieldID', 'prcf']] = (final_df['file_name'].apply(lambda fname: pd.Series(_map_wells(fname, timelapse=False))))
        final_df['wellID'] = final_df['file_name'].str.split('_').str[1]
        
    except IndexError:
        logger.warning("Failed to parse plate, well, field from name: %s", name)
    
    # ------------------------------------------------------------------
    # Relabel masks with track IDs
    # ------------------------------------------------------------------
    logger.debug("Relabelling masks based on tracks...")
    masks = _relabel_masks_based_on_tracks(masks_3D, final_df)

    # ------------------------------------------------------------------
    # Save track table
    # ------------------------------------------------------------------
    tracks_path = os.path.join(os.path.dirname(src), "tracks")
    os.makedirs(tracks_path, exist_ok=True)
    out_csv = os.path.join(tracks_path, f"btrack_tracks_{object_type}_{name}.csv")
    logger.debug("Saving track table to %s", out_csv)
    final_df.to_csv(out_csv, index=False)

    # ------------------------------------------------------------------
    # Optional visualisation
    # ------------------------------------------------------------------
    if plot or save:
        logger.debug("Generating visualisation (plot=%s, save=%s)...", plot, save)
        _visualize_and_save_timelapse_stack_with_tracks(
            masks,
            final_df,
            save,
            src,
            name,
            plot,
            batch_filenames,
            object_type,
            mode,
        )

    # Return in your standard mask stack format
    mask_stack = _masks_to_masks_stack(masks)
    logger.debug(
        "Finished _btrack_track_cells. mask_stack shape: %s",
        getattr(mask_stack, "shape", None),
    )
    return mask_stack


def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def preprocess_pathogen_data(pathogen_df):
    # Group by identifiers and count the number of parasites
    parasite_counts = pathogen_df.groupby(['plateID', 'rowID', 'column_name', 'fieldID', 'timeid', 'pathogen_cell_id']).size().reset_index(name='parasite_count')

    # Aggregate numerical columns and take the first of object columns
    agg_funcs = {col: 'mean' if np.issubdtype(pathogen_df[col].dtype, np.number) else 'first' for col in pathogen_df.columns if col not in ['plateID', 'rowID', 'column_name', 'fieldID', 'timeid', 'pathogen_cell_id', 'parasite_count']}
    pathogen_agg = pathogen_df.groupby(['plateID', 'rowID', 'column_name', 'fieldID', 'timeid', 'pathogen_cell_id']).agg(agg_funcs).reset_index()

    # Merge the counts back into the aggregated data
    pathogen_agg = pathogen_agg.merge(parasite_counts, on=['plateID', 'rowID', 'column_name', 'fieldID', 'timeid', 'pathogen_cell_id'])
    

    # Remove the object_label column as it corresponds to the pathogen ID not the cell ID
    if 'object_label' in pathogen_agg.columns:
        pathogen_agg.drop(columns=['object_label'], inplace=True)
    
    # Change the name of pathogen_cell_id to object_label
    pathogen_agg.rename(columns={'pathogen_cell_id': 'object_label'}, inplace=True)

    return pathogen_agg

def plot_data(measurement, group, ax, label, marker='o', linestyle='-'):
    ax.plot(group['time'], group['delta_' + measurement], marker=marker, linestyle=linestyle, label=label)

def infected_vs_noninfected(result_df, measurement):
    # Separate the merged dataframe into two groups based on pathogen_count
    infected_cells_df = result_df[result_df.groupby('plate_row_column_field_object')['parasite_count'].transform('max') > 0]
    uninfected_cells_df = result_df[result_df.groupby('plate_row_column_field_object')['parasite_count'].transform('max') == 0]

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot for cells that were infected at some time
    for group_id in infected_cells_df['plate_row_column_field_object'].unique():
        group = infected_cells_df[infected_cells_df['plate_row_column_field_object'] == group_id]
        plot_data(measurement, group, axs[0], 'Infected', marker='x')

    # Plot for cells that were never infected
    for group_id in uninfected_cells_df['plate_row_column_field_object'].unique():
        group = uninfected_cells_df[uninfected_cells_df['plate_row_column_field_object'] == group_id]
        plot_data(measurement, group, axs[1], 'Uninfected')

    # Set the titles and labels
    axs[0].set_title('Cells Infected at Some Time')
    axs[1].set_title('Cells Never Infected')
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Delta ' + measurement)
        all_timepoints = sorted(result_df['time'].unique())
        ax.set_xticks(all_timepoints)
        ax.set_xticklabels(all_timepoints, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def save_figure(fig, src, figure_number):
    source = os.path.dirname(src)
    results_fldr = os.path.join(source,'results')
    os.makedirs(results_fldr, exist_ok=True)
    fig_loc = os.path.join(results_fldr, f'figure_{figure_number}.pdf')
    fig.savefig(fig_loc)
    print(f'Saved figure:{fig_loc}')

def save_results_dataframe(df, src, results_name):
    source = os.path.dirname(src)
    results_fldr = os.path.join(source,'results')
    os.makedirs(results_fldr, exist_ok=True)
    csv_loc = os.path.join(results_fldr, f'{results_name}.csv')
    df.to_csv(csv_loc, index=True)
    print(f'Saved results:{csv_loc}')

def summarize_per_well(peak_details_df):
    # Step 1: Split the 'ID' column
    split_columns = peak_details_df['ID'].str.split('_', expand=True)
    peak_details_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_number']] = split_columns

    # Step 2: Create 'well_ID' by combining 'rowID' and 'columnID'
    peak_details_df['well_ID'] = peak_details_df['rowID'] + '_' + peak_details_df['columnID']

    # Filter entries where 'amplitude' is not null
    filtered_df = peak_details_df[peak_details_df['amplitude'].notna()]

    # Preparation for Step 3: Identify numeric columns for averaging from the filtered dataframe
    numeric_cols = filtered_df.select_dtypes(include=['number']).columns

    # Step 3: Calculate summary statistics
    summary_df = filtered_df.groupby('well_ID').agg(
        peaks_per_well=('ID', 'size'),
        unique_IDs_with_amplitude=('ID', 'nunique'),  # Count unique IDs per well with non-null amplitude
        **{col: (col, 'mean') for col in numeric_cols}  # exclude 'amplitude' from averaging if it's numeric
    ).reset_index()

    # Step 3: Calculate summary statistics
    summary_df_2 = peak_details_df.groupby('well_ID').agg(
        cells_per_well=('object_number', 'nunique'),
    ).reset_index()

    summary_df['cells_per_well'] = summary_df_2['cells_per_well']
    summary_df['peaks_per_cell'] = summary_df['peaks_per_well'] / summary_df['cells_per_well']
    
    return summary_df

def summarize_per_well_inf_non_inf(peak_details_df):
    # Step 1: Split the 'ID' column
    split_columns = peak_details_df['ID'].str.split('_', expand=True)
    peak_details_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_number']] = split_columns

    # Step 2: Create 'well_ID' by combining 'rowID' and 'columnID'
    peak_details_df['well_ID'] = peak_details_df['rowID'] + '_' + peak_details_df['columnID']

    # Assume 'pathogen_count' indicates infection if > 0
    # Add an 'infected_status' column to classify cells
    peak_details_df['infected_status'] = peak_details_df['infected'].apply(lambda x: 'infected' if x > 0 else 'non_infected')

    # Preparation for Step 3: Identify numeric columns for averaging
    numeric_cols = peak_details_df.select_dtypes(include=['number']).columns

    # Step 3: Calculate summary statistics
    summary_df = peak_details_df.groupby(['well_ID', 'infected_status']).agg(
        cells_per_well=('object_number', 'nunique'),
        peaks_per_well=('ID', 'size'),
        **{col: (col, 'mean') for col in numeric_cols}
    ).reset_index()

    # Calculate peaks per cell
    summary_df['peaks_per_cell'] = summary_df['peaks_per_well'] / summary_df['cells_per_well']

    return summary_df

def analyze_calcium_oscillations(db_loc, measurement='cell_channel_1_mean_intensity', size_filter='cell_area', fluctuation_threshold=0.25, num_lines=None, peak_height=0.01, pathogen=None, cytoplasm=None, remove_transient=True, verbose=False, transience_threshold=0.9):
    # Load data
    conn = sqlite3.connect(db_loc)
    # Load cell table
    cell_df = pd.read_sql(f"SELECT * FROM {'cell'}", conn)
    
    if pathogen:
        pathogen_df = pd.read_sql("SELECT * FROM pathogen", conn)
        pathogen_df['pathogen_cell_id'] = pathogen_df['pathogen_cell_id'].astype(float).astype('Int64')
        pathogen_df = preprocess_pathogen_data(pathogen_df)
        cell_df = cell_df.merge(pathogen_df, on=['plateID', 'rowID', 'column_name', 'fieldID', 'timeid', 'object_label'], how='left', suffixes=('', '_pathogen'))
        cell_df['parasite_count'] = cell_df['parasite_count'].fillna(0)
        print(f'After pathogen merge: {len(cell_df)} objects')

    # Optionally load cytoplasm table and merge
    if cytoplasm:
        cytoplasm_df = pd.read_sql(f"SELECT * FROM {'cytoplasm'}", conn)
        # Merge on specified columns
        cell_df = cell_df.merge(cytoplasm_df, on=['plateID', 'rowID', 'column_name', 'fieldID', 'timeid', 'object_label'], how='left', suffixes=('', '_cytoplasm'))

        print(f'After cytoplasm merge: {len(cell_df)} objects')
    
    conn.close()

    # Continue with your existing processing on cell_df now containing merged data...
    # Prepare DataFrame (use cell_df instead of df)
    prcf_components = cell_df['prcf'].str.split('_', expand=True)
    cell_df['plateID'] = prcf_components[0]
    cell_df['rowID'] = prcf_components[1]
    cell_df['columnID'] = prcf_components[2]
    cell_df['fieldID'] = prcf_components[3]
    cell_df['time'] = prcf_components[4].str.extract('t(\d+)').astype(int)
    cell_df['object_number'] = cell_df['object_label']
    cell_df['plate_row_column_field_object'] = cell_df['plateID'].astype(str) + '_' + cell_df['rowID'].astype(str) + '_' + cell_df['columnID'].astype(str) + '_' + cell_df['fieldID'].astype(str) + '_' + cell_df['object_label'].astype(str)

    df = cell_df.copy()

    # Fit exponential decay model to all scaled fluorescence data
    try:
        params, _ = curve_fit(exponential_decay, df['time'], df[measurement], p0=[max(df[measurement]), 0.01, min(df[measurement])], maxfev=10000)
        df['corrected_' + measurement] = df[measurement] / exponential_decay(df['time'], *params)
    except RuntimeError as e:
        print(f"Curve fitting failed for the entire dataset with error: {e}")
        return
    if verbose:
        print(f'Analyzing: {len(df)} objects')
    
    # Normalizing corrected fluorescence for each cell
    corrected_dfs = []
    peak_details_list = []
    total_timepoints = df['time'].nunique()
    size_filter_removed = 0
    transience_removed = 0
    
    for unique_id, group in df.groupby('plate_row_column_field_object'):
        group = group.sort_values('time')
        if remove_transient:

            threshold = int(transience_threshold * total_timepoints)

            if verbose:
                print(f'Group length: {len(group)} Timelapse length: {total_timepoints}, threshold:{threshold}')

            if len(group) <= threshold:
                transience_removed += 1
                if verbose:
                    print(f'removed group {unique_id} due to transience')
                continue
        
        size_diff = group[size_filter].std() / group[size_filter].mean()

        if size_diff <= fluctuation_threshold:
            group['delta_' + measurement] = group['corrected_' + measurement].diff().fillna(0)
            corrected_dfs.append(group)
            
            # Detect peaks
            peaks, properties = find_peaks(group['delta_' + measurement], height=peak_height)

            # Set values < 0 to 0
            group_filtered = group.copy()
            group_filtered['delta_' + measurement] = group['delta_' + measurement].clip(lower=0)
            above_zero_auc = trapz(y=group_filtered['delta_' + measurement], x=group_filtered['time'])
            auc = trapz(y=group['delta_' + measurement], x=group_filtered['time'])
            is_infected = (group['parasite_count'] > 0).any()
            
            if is_infected:
                is_infected = 1
            else:
                is_infected = 0

            if len(peaks) == 0:
                peak_details_list.append({
                    'ID': unique_id,
                    'plateID': group['plateID'].iloc[0],
                    'rowID': group['rowID'].iloc[0],
                    'columnID': group['columnID'].iloc[0],
                    'fieldID': group['fieldID'].iloc[0],
                    'object_number': group['object_number'].iloc[0],
                    'time': np.nan,  # The time of the peak
                    'amplitude': np.nan,
                    'delta': np.nan,
                    'AUC': auc,
                    'AUC_positive': above_zero_auc,
                    'AUC_peak': np.nan,
                    'infected': is_infected  
                })

            # Inside the for loop where peaks are detected
            for i, peak in enumerate(peaks):

                amplitude = properties['peak_heights'][i]
                peak_time = group['time'].iloc[peak]
                pathogen_count_at_peak = group['parasite_count'].iloc[peak]

                start_idx = max(peak - 1, 0)
                end_idx = min(peak + 1, len(group) - 1)

                # Using indices to slice for AUC calculation
                peak_segment_y = group['delta_' + measurement].iloc[start_idx:end_idx + 1]
                peak_segment_x = group['time'].iloc[start_idx:end_idx + 1]
                peak_auc = trapz(y=peak_segment_y, x=peak_segment_x)

                peak_details_list.append({
                    'ID': unique_id,
                    'plateID': group['plateID'].iloc[0],
                    'rowID': group['rowID'].iloc[0],
                    'columnID': group['columnID'].iloc[0],
                    'fieldID': group['fieldID'].iloc[0],
                    'object_number': group['object_number'].iloc[0],
                    'time': peak_time,  # The time of the peak
                    'amplitude': amplitude,
                    'delta': group['delta_' + measurement].iloc[peak],
                    'AUC': auc,
                    'AUC_positive': above_zero_auc,
                    'AUC_peak': peak_auc,
                    'infected': pathogen_count_at_peak  
                })
        else:
            size_filter_removed += 1

    if verbose:
        print(f'Removed {size_filter_removed} objects due to size filter fluctuation')
        print(f'Removed {transience_removed} objects due to transience')

    if len(corrected_dfs) > 0:
        result_df = pd.concat(corrected_dfs)
    else:
        print("No suitable cells found for analysis")
        return
    
    peak_details_df = pd.DataFrame(peak_details_list)
    summary_df = summarize_per_well(peak_details_df)
    summary_df_inf_non_inf = summarize_per_well_inf_non_inf(peak_details_df)

    save_results_dataframe(df=peak_details_df, src=db_loc, results_name='peak_details')
    save_results_dataframe(df=result_df, src=db_loc, results_name='results')
    save_results_dataframe(df=summary_df, src=db_loc, results_name='well_results')
    save_results_dataframe(df=summary_df_inf_non_inf, src=db_loc, results_name='well_results_inf_non_inf')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sampled_groups = result_df['plate_row_column_field_object'].unique()
    if num_lines is not None and 0 < num_lines < len(sampled_groups):
        sampled_groups = np.random.choice(sampled_groups, size=num_lines, replace=False)

    for group_id in sampled_groups:
        group = result_df[result_df['plate_row_column_field_object'] == group_id]
        ax.plot(group['time'], group['delta_' + measurement], marker='o', linestyle='-')

    ax.set_xticks(sorted(df['time'].unique()))
    ax.set_xticklabels(sorted(df['time'].unique()), rotation=45, ha="right")
    ax.set_title(f'Normalized Delta of {measurement} Over Time (Corrected for Photobleaching)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Delta ' + measurement)
    plt.tight_layout()
    
    plt.show()

    save_figure(fig, src=db_loc, figure_number=1)
    
    if pathogen:
        infected_vs_noninfected(result_df, measurement)
        save_figure(fig, src=db_loc, figure_number=2)

        # Identify cells with and without pathogens
        infected_cells = result_df[result_df.groupby('plate_row_column_field_object')['parasite_count'].transform('max') > 0]['plate_row_column_field_object'].unique()
        noninfected_cells = result_df[result_df.groupby('plate_row_column_field_object')['parasite_count'].transform('max') == 0]['plate_row_column_field_object'].unique()

        # Peaks in infected and noninfected cells
        infected_peaks = peak_details_df[peak_details_df['ID'].isin(infected_cells)]
        noninfected_peaks = peak_details_df[peak_details_df['ID'].isin(noninfected_cells)]

        # Calculate the average number of peaks per cell
        avg_inf_peaks_per_cell = len(infected_peaks) / len(infected_cells) if len(infected_cells) > 0 else 0
        avg_non_inf_peaks_per_cell = len(noninfected_peaks) / len(noninfected_cells) if len(noninfected_cells) > 0 else 0

        print(f'Average number of peaks per infected cell: {avg_inf_peaks_per_cell:.2f}')
        print(f'Average number of peaks per non-infected cell: {avg_non_inf_peaks_per_cell:.2f}')
    print(f'done')
    return result_df, peak_details_df, fig

def _generate_mask_random_cmap(mask):
    """
    Generate a random colormap based on the unique labels in the given mask.

    Parameters
    ----------
    mask : ndarray
        2D label mask. Background must be 0, objects > 0.

    Returns
    -------
    mpl.colors.ListedColormap
        Random colormap with a fixed black background (label 0).
    """
    unique_labels = np.unique(mask)
    # Only count non-zero labels as objects
    num_objects = np.sum(unique_labels != 0)
    # +1 so index 0 is background
    random_colors = np.random.rand(num_objects + 1, 4)
    random_colors[:, 3] = 1.0  # full alpha
    # background = black, fully opaque
    random_colors[0, :] = [0.0, 0.0, 0.0, 1.0]
    return mpl.colors.ListedColormap(random_colors)

def create_results_figure():
    """
    Create a Figure with 3 subplots arranged as:
      - PCA (top-left)
      - XGBoost (top-right)
      - Histogram (bottom spanning both columns)
    Returns
    -------
    fig : Figure
    ax_pca, ax_xgb, ax_hist : matplotlib.axes.Axes
    """
    fig = Figure(figsize=(7, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax_pca = fig.add_subplot(gs[0, 0])
    ax_xgb = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, :])

    return fig, ax_pca, ax_xgb, ax_hist

def _make_intensity_motility_panel(
    all_df,
    infection_col,
    track_df,
    per_well_tracks,
    n_channels,
    motility_dir,
    pixels_per_um,
    seconds_per_frame,
    vel_unit,
    settings,
    label_tag,
):
    """
    Make panels for infection and motility.

    Behaviour:
      - "mask_*" label_tag:
            classic panel with
                * per-channel mean intensity (infected vs uninfected)
                * (optional) pathogen-channel p75 intensity bar
                * (optional) pathogen/cytoplasm intensity ratio bar
                * all-tracks motility plot (absolute FOV)
                * motility origin plots (infected / uninfected)
                * optional small QC image (feature importance PNG)

      - "adjusted_*" label_tag:
            same as mask panel, plus method-specific QC subplots appended:
                * histogram (if strategy == "histogram")
                * PCA/UMAP/t-SNE embedding (if strategy == "pca"/"umap"/"tsne")
                * XGBoost:
                    - probability separation histogram
                    - feature-importance barplot
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg  # used for the small QC PNG in mask panel

    if all_df.empty or track_df.empty or not per_well_tracks:
        print(f"[_make_intensity_motility_panel] No data for panel '{label_tag}', skipping.")
        return

    os.makedirs(motility_dir, exist_ok=True)
    key_cols = ["plateID", "wellID", "fieldID", "cellID"]

    # ------------------------------------------------------------------
    # Panel type / strategy / QC payload availability
    # ------------------------------------------------------------------
    label_lower = str(label_tag).lower()
    is_mask_panel = label_lower.startswith("mask")
    is_adjusted_panel = label_lower.startswith("adjusted")

    qc_strategy = str(settings.get("infection_intensity_strategy", "none")).lower()
    method_label = qc_strategy if qc_strategy else "none"
    panel_label = "mask" if is_mask_panel else ("adjusted" if is_adjusted_panel else label_tag)

    qc_graphs_enabled = bool(settings.get("infection_intensity_qc_graphs", True))
    qc_panel_type = settings.get("infection_intensity_qc_panel_type", None)
    qc_panel_path = settings.get("infection_intensity_qc_panel_path", None)

    # Global QC payloads (built in QC helpers)
    hist_data = settings.get("infection_hist_data", None)
    pca_data = settings.get("infection_pca_data", None)
    xgb_data = settings.get("infection_xgb_importance", None)

    has_hist = hist_data is not None
    has_pca = pca_data is not None
    has_xgb = xgb_data is not None

    # Mask panel: small embedded QC PNG if available
    qc_panel_needed_mask = (
        is_mask_panel
        and qc_graphs_enabled
        and isinstance(qc_panel_path, str)
        and qc_panel_path
        and os.path.exists(qc_panel_path)
    )

    # Adjusted panel: method-specific QC axes
    qc_axes_count = 0
    if is_adjusted_panel and qc_graphs_enabled:
        # Histogram: we *always* allocate one QC axis and can compute from df_well
        if qc_strategy == "histogram":
            qc_axes_count = 1
        # PCA/UMAP/t-SNE: need pca_data for embedding
        elif qc_strategy in {"pca", "umap", "tsne"} and has_pca:
            qc_axes_count = 1
        # XGBoost: probability separation + feature importance
        elif qc_strategy == "xgboost" and has_xgb:
            qc_axes_count = 2

    # Motility axis limits: driven by motility_xlim / motility_ylim
    origin_xlim = settings.get("motility_xlim", settings.get("motility_origin_xlim"))
    origin_ylim = settings.get("motility_ylim", settings.get("motility_origin_ylim"))

    # Coordinate scaling
    if pixels_per_um is not None and pixels_per_um > 0:
        coord_scale = 1.0 / float(pixels_per_um)
        coord_label_x = "x (µm)"
        coord_label_y = "y (µm)"
    else:
        coord_scale = 1.0
        coord_label_x = "x (pixels)"
        coord_label_y = "y (pixels)"

    pathogen_chan = settings.get("pathogen_channel", None)

    # ------------------------------------------------------------------
    # Helpers for QC subplots (used in adjusted panel)
    # ------------------------------------------------------------------
    def _plot_hist_qc(ax, source):
        """
        Draw infected vs uninfected intensity histogram.

        `source` can be:
          - a dict payload (settings['infection_hist_data']) with keys:
                'intensities_inf', 'intensities_uninf', 'bin_edges',
                'thr_val' (optional), 'intensity_col' (optional)
          - or a DataFrame (df_well), in which case the histogram
                is computed on the fly using a reasonable intensity
                column and `infection_intensity_n_bins`.
        """
        try:
            # Case 1: payload dict from settings
            if isinstance(source, dict):
                intens_inf = np.asarray(source["intensities_inf"], dtype=float)
                intens_uninf = np.asarray(source["intensities_uninf"], dtype=float)
                bin_edges = np.asarray(source["bin_edges"], dtype=float)
                thr_val = float(source.get("thr_val", np.nan))
                intensity_col = source.get("intensity_col", "intensity")
            else:
                # Case 2: compute from per-well DataFrame
                df_vals = source

                # Decide which intensity column to use
                intensity_col = settings.get("infection_hist_intensity_col", None)
                if not intensity_col or intensity_col not in df_vals.columns:
                    cand_cols = []
                    if pathogen_chan is not None:
                        cand_cols.extend(
                            [
                                f"cell_mean_intensity_ch{pathogen_chan}",
                                f"cell_p75_intensity_ch{pathogen_chan}",
                                f"pathogen_mean_intensity_ch{pathogen_chan}",
                            ]
                        )
                    # Fallback: any cell_mean_intensity_ch*
                    cand_cols.extend(
                        [c for c in df_vals.columns if c.startswith("cell_mean_intensity_ch")]
                    )
                    for c in cand_cols:
                        if c in df_vals.columns:
                            intensity_col = c
                            break

                if not intensity_col or intensity_col not in df_vals.columns:
                    ax.set_visible(False)
                    return

                # Collapse to one value per cell-track
                cell_level = (
                    df_vals[key_cols + [intensity_col, infection_col]]
                    .groupby(key_cols, dropna=False)
                    .agg({intensity_col: "mean", infection_col: "max"})
                    .reset_index()
                )
                cell_level = cell_level.replace([np.inf, -np.inf], np.nan)
                cell_level = cell_level.dropna(subset=[intensity_col])
                if cell_level.empty:
                    ax.set_visible(False)
                    return

                mask_inf = cell_level[infection_col].astype(bool)
                intens_inf = cell_level.loc[mask_inf, intensity_col].to_numpy()
                intens_uninf = cell_level.loc[~mask_inf, intensity_col].to_numpy()

                if intens_inf.size == 0 and intens_uninf.size == 0:
                    ax.set_visible(False)
                    return

                all_vals = np.concatenate(
                    [arr for arr in (intens_inf, intens_uninf) if arr.size]
                )
                n_bins = int(settings.get("infection_intensity_n_bins", 64) or 64)
                bin_edges = np.histogram_bin_edges(all_vals, bins=n_bins)

                thr_val = settings.get(
                    "infection_hist_thr_val",
                    settings.get("infection_intensity_threshold", np.nan),
                )
                thr_val = float(thr_val) if thr_val is not None else np.nan

            # Now plot
            ax.hist(
                intens_uninf,
                bins=bin_edges,
                alpha=0.5,
                color="green",
                label="Uninfected",
            )
            ax.hist(
                intens_inf,
                bins=bin_edges,
                alpha=0.5,
                color="red",
                label="Infected",
            )
            if np.isfinite(thr_val):
                ax.axvline(thr_val, color="black", linestyle="--", linewidth=1)

            ax.set_xlabel(intensity_col)
            ax.set_ylabel("Count")
            ax.set_title("Pathogen-channel intensity\n(adjusted labels)")
            ax.legend(fontsize=7)
        except Exception as e:
            print(f"[_make_intensity_motility_panel] Histogram payload invalid: {e}")
            ax.set_visible(False)

    def _plot_pca_qc(ax, pdata):
        import numpy as np
    
        try:
            coords = np.asarray(pdata["coords"], dtype=float)
            labels = np.asarray(pdata["labels"], dtype=bool)
        except Exception as e:
            print(f"[_make_intensity_motility_panel] PCA/embedding payload invalid: {e}")
            ax.set_visible(False)
            return
    
        if coords.ndim != 2 or coords.shape[1] < 2:
            ax.set_visible(False)
            return
    
        # Method label stored by _infection_qc_pca_clustering: 'PCA', 'UMAP', or 't-SNE'
        method_label = str(pdata.get("method_label", "PCA"))
    
        x = coords[:, 0]
        y = coords[:, 1]
    
        ax.scatter(
            x[~labels],
            y[~labels],
            s=5,
            alpha=0.4,
            color="green",
            label="Uninfected",
        )
        ax.scatter(
            x[labels],
            y[labels],
            s=5,
            alpha=0.4,
            color="red",
            label="Infected",
        )
    
        # Generic axis labels that respect the embedding method
        ax.set_xlabel(f"{method_label} 1")
        ax.set_ylabel(f"{method_label} 2")
    
        # Title also reflects method
        ax.set_title(f"{method_label} of features\n(adjusted labels)")
    
        ax.legend(fontsize=7)

    def _plot_xgb_importance_qc(ax, xdata):
        try:
            feat_names = xdata["feature_names"]
            feat_vals = xdata["feature_importances"]
        except Exception as e:
            print(f"[_make_intensity_motility_panel] XGB importance payload invalid: {e}")
            ax.set_visible(False)
            return

        if not feat_names:
            ax.set_visible(False)
            return

        y_pos = np.arange(len(feat_names))
        ax.barh(y_pos, feat_vals)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (gain)")
        ax.set_title("XGBoost feature importance")

    def _plot_xgb_prob_qc(ax, df_prob):
        """
        Per-cell probability distribution by adjusted infection label.

        Uses settings['infection_xgb_proba_column'] if available,
        otherwise falls back through a few common column names.
        """
        # Resolve probability column
        prob_col_candidates = []

        cfg_col = settings.get("infection_xgb_proba_column", None)
        if isinstance(cfg_col, str) and cfg_col:
            prob_col_candidates.append(cfg_col)

        prob_col_candidates.extend(
            [
                "infection_prob",
                "infection_xgb_proba",
                "xgb_prob",
            ]
        )

        prob_col = None
        for c in prob_col_candidates:
            if c in df_prob.columns:
                prob_col = c
                break

        if prob_col is None:
            ax.set_visible(False)
            return

        # per-cell probabilities
        cell_probs = (
            df_prob[key_cols + [prob_col, infection_col]]
            .groupby(key_cols, dropna=False)
            .agg({prob_col: "mean", infection_col: "max"})
            .reset_index()
        )
        cell_probs = cell_probs.replace([np.inf, -np.inf], np.nan)
        cell_probs = cell_probs.dropna(subset=[prob_col])
        if cell_probs.empty:
            ax.set_visible(False)
            return

        mask_inf = cell_probs[infection_col].astype(bool)
        probs_inf = cell_probs.loc[mask_inf, prob_col].to_numpy()
        probs_uninf = cell_probs.loc[~mask_inf, prob_col].to_numpy()

        bins = np.linspace(0.0, 1.0, 21)
        if probs_uninf.size:
            ax.hist(
                probs_uninf,
                bins=bins,
                alpha=0.5,
                color="green",
                label="Uninfected",
            )
        if probs_inf.size:
            ax.hist(
                probs_inf,
                bins=bins,
                alpha=0.5,
                color="red",
                label="Infected",
            )
        ax.set_xlabel("XGBoost infection probability")
        ax.set_ylabel("Cells")
        ax.set_title("Probability separation (adjusted labels)")
        ax.legend(fontsize=7)

    def _plot_inf_uninf_bar(ax, df_vals, value_col, title, ylabel):
        """
        Helper to plot infected vs uninfected distributions for the given column,
        using violin plots (with mean markers) instead of barplots.
        """
        if value_col not in df_vals.columns:
            ax.set_visible(False)
            return

        # Collapse to one value per cell-track
        cell_level = (
            df_vals[key_cols + [value_col, infection_col]]
            .groupby(key_cols, dropna=False)
            .agg({value_col: "mean", infection_col: "max"})
            .reset_index()
        )
        cell_level = cell_level.replace([np.inf, -np.inf], np.nan)
        cell_level = cell_level.dropna(subset=[value_col])

        if cell_level.empty:
            ax.set_visible(False)
            return

        mask_inf = cell_level[infection_col].astype(bool)
        vals_inf = cell_level.loc[mask_inf, value_col].to_numpy()
        vals_uninf = cell_level.loc[~mask_inf, value_col].to_numpy()

        data = []
        positions = []
        colors = []
        labels_xtick = []

        pos = 0
        if vals_inf.size:
            data.append(vals_inf)
            positions.append(pos)
            colors.append("red")
            labels_xtick.append("Inf")
            pos += 1
        if vals_uninf.size:
            data.append(vals_uninf)
            positions.append(pos)
            colors.append("green")
            labels_xtick.append("Uninf")

        if not data:
            ax.set_visible(False)
            return

        # Violin plots
        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        # Color each violin (Inf = red, Uninf = green)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.6)

        # Overlay means as black points
        means = [float(np.nanmean(d)) for d in data]
        ax.scatter(positions, means, color="black", s=10, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_xtick)
        # If all values are non-negative, anchor at 0
        flat = np.concatenate(data)
        if np.nanmin(flat) >= 0:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=0, top=ymax)

        ax.set_title(title)
        ax.set_ylabel(ylabel)

    # ------------------------------------------------------------------
    # One figure per (plateID, wellID)
    # ------------------------------------------------------------------
    if not {"plateID", "wellID"}.issubset(all_df.columns):
        print(
            "[_make_intensity_motility_panel] Missing 'plateID'/'wellID' columns; "
            "cannot make per-well panels."
        )
        return

    unique_wells = (
        all_df[["plateID", "wellID"]]
        .dropna()
        .drop_duplicates()
        .to_records(index=False)
    )

    for plate_id, well_id in unique_wells:
        # Subset data for this well
        df_well = all_df[
            (all_df["plateID"] == plate_id) & (all_df["wellID"] == well_id)
        ]
        track_df_well = track_df[
            (track_df["plateID"] == plate_id) & (track_df["wellID"] == well_id)
        ]

        # Collect tracks for this well from per_well_tracks
        well_tracks = []
        for tracks in per_well_tracks.values():
            for tr in tracks:
                if tr.get("plateID") == plate_id and tr.get("wellID") == well_id:
                    well_tracks.append(tr)

        if df_well.empty or track_df_well.empty or not well_tracks:
            print(
                f"[_make_intensity_motility_panel] No data for plate={plate_id}, "
                f"well={well_id}; skipping."
            )
            continue

        # Determine which channels are available *for this well*
        available_channels = [
            ch
            for ch in range(n_channels)
            if f"cell_mean_intensity_ch{ch}" in df_well.columns
        ]
        if not available_channels:
            print(
                f"[_make_intensity_motility_panel] No cell_mean_intensity_ch* "
                f"columns for plate={plate_id}, well={well_id}; skipping."
            )
            continue

        # Extra intensity plots for pathogen channel
        has_p75_path = False
        has_rel_int = False
        if pathogen_chan is not None:
            p75_col = f"cell_p75_intensity_ch{pathogen_chan}"
            if p75_col in df_well.columns:
                has_p75_path = True
            path_col = f"pathogen_mean_intensity_ch{pathogen_chan}"
            cyto_col = f"cytoplasm_mean_intensity_ch{pathogen_chan}"
            if path_col in df_well.columns and cyto_col in df_well.columns:
                has_rel_int = True

        extra_int_plots = (1 if has_p75_path else 0) + (1 if has_rel_int else 0)
        n_int_plots = len(available_channels) + extra_int_plots

        # +3 for: all-tracks motility, infected origin, uninfected origin
        # +1 for small QC PNG in mask panel,
        # +qc_axes_count for adjusted panel QC subplots
        n_cols = n_int_plots + 3 + (1 if qc_panel_needed_mask else 0) + qc_axes_count

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).ravel()

        axis_idx = 0

        # ----- intensity violins per channel (per well) -----
        for ch in available_channels:
            col_int = f"cell_mean_intensity_ch{ch}"
            ax = axes[axis_idx]
            axis_idx += 1
            _plot_inf_uninf_bar(
                ax,
                df_well,
                value_col=col_int,
                title=f"Ch {ch} mean",
                ylabel="Mean cell intensity",
            )

            # If this is the pathogen channel, append p75 and ratio plots if available
            if pathogen_chan is not None and ch == pathogen_chan:
                if has_p75_path:
                    ax_p75 = axes[axis_idx]
                    axis_idx += 1
                    p75_col = f"cell_p75_intensity_ch{pathogen_chan}"
                    _plot_inf_uninf_bar(
                        ax_p75,
                        df_well,
                        value_col=p75_col,
                        title=f"Ch {pathogen_chan} p75",
                        ylabel="Cell p75 intensity",
                    )
                if has_rel_int:
                    ax_rel = axes[axis_idx]
                    axis_idx += 1
                    path_col = f"pathogen_mean_intensity_ch{pathogen_chan}"
                    cyto_col = f"cytoplasm_mean_intensity_ch{pathogen_chan}"
                    df_ratio = df_well[
                        key_cols + [path_col, cyto_col, infection_col]
                    ].copy()
                    df_ratio["rel_intensity"] = df_ratio[path_col] / df_ratio[
                        cyto_col
                    ].replace(0, np.nan)
                    _plot_inf_uninf_bar(
                        ax_rel,
                        df_ratio,
                        value_col="rel_intensity",
                        title=f"Ch {pathogen_chan} pathogen/cytoplasm",
                        ylabel="Intensity ratio",
                    )

        # ----- all-tracks FOV plot (absolute coordinates) -----
        def _plot_all_tracks(ax):
            if not well_tracks:
                ax.set_visible(False)
                return

            xs_all = []
            ys_all = []
            n_inf_tr = 0
            n_uninf_tr = 0

            for tr in well_tracks:
                x_px = np.asarray(tr["x_px"], dtype=float)
                y_px = np.asarray(tr["y_px"], dtype=float)
                if x_px.size < 2:
                    continue
                x = x_px * coord_scale
                y = y_px * coord_scale
                infected_tr = bool(tr.get("infected", False))
                color = "red" if infected_tr else "green"
                ax.plot(x, y, color=color, alpha=0.15, linewidth=0.5)
                ax.scatter(x[-1], y[-1], color=color, s=5)
                xs_all.append(x)
                ys_all.append(y)
                if infected_tr:
                    n_inf_tr += 1
                else:
                    n_uninf_tr += 1

            if not xs_all:
                ax.set_visible(False)
                return

            xs_all = np.concatenate(xs_all)
            ys_all = np.concatenate(ys_all)
            ax.set_aspect("equal", "box")
            ax.set_xlabel(coord_label_x)
            ax.set_ylabel(coord_label_y)
            # auto limits from data
            x_margin = 0.05 * (xs_all.max() - xs_all.min() + 1e-9)
            y_margin = 0.05 * (ys_all.max() - ys_all.min() + 1e-9)
            ax.set_xlim(xs_all.min() - x_margin, xs_all.max() + x_margin)
            ax.set_ylim(ys_all.min() - y_margin, ys_all.max() + y_margin)

            mask_inf = track_df_well["infected"].astype(bool)
            v_inf = track_df_well.loc[mask_inf, "velocity"].to_numpy()
            v_uninf = track_df_well.loc[~mask_inf, "velocity"].to_numpy()
            mean_inf_v = float(np.nanmean(v_inf)) if v_inf.size else np.nan
            mean_uninf_v = float(np.nanmean(v_uninf)) if v_uninf.size else np.nan

            txt_lines = []
            txt_lines.append(f"Infected ({mean_inf_v:.2f} {vel_unit})")
            txt_lines.append(f"Uninfected ({mean_uninf_v:.2f} {vel_unit})")
            if pixels_per_um is not None and pixels_per_um > 0:
                txt_lines.append(f"1 µm = {pixels_per_um:.2f} px")
            if seconds_per_frame is not None:
                txt_lines.append(f"1 frame = {seconds_per_frame:.0f} s")

            txt = "\n".join(txt_lines)
            ax.text(
                0.98,
                0.02,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
            )

        ax_all = axes[axis_idx]
        axis_idx += 1
        _plot_all_tracks(ax_all)

        # ----- motility origin plots (infected vs uninfected) for this well -----
        def _plot_origin(ax, want_infected: bool):
            n_tr = 0
            color = "red" if want_infected else "green"

            for tr in well_tracks:
                if bool(tr.get("infected", False)) != want_infected:
                    continue
                x_px = np.asarray(tr["x_px"], dtype=float)
                y_px = np.asarray(tr["y_px"], dtype=float)
                if x_px.size < 2:
                    continue
                x = (x_px - x_px[0]) * coord_scale
                y = (y_px - y_px[0]) * coord_scale
                ax.plot(x, y, color=color, alpha=0.15, linewidth=0.5)
                ax.scatter(x[-1], y[-1], color=color, s=5)
                n_tr += 1

            ax.set_aspect("equal", "box")
            ax.set_xlabel(coord_label_x)
            ax.set_ylabel(coord_label_y)
            if origin_xlim is not None and len(origin_xlim) == 2:
                ax.set_xlim(origin_xlim)
            if origin_ylim is not None and len(origin_ylim) == 2:
                ax.set_ylim(origin_ylim)

            mask = track_df_well["infected"].astype(bool)
            if not want_infected:
                mask = ~mask
            v = track_df_well.loc[mask, "velocity"].to_numpy()
            mean_v = float(np.nanmean(v)) if v.size else np.nan

            label = "Infected" if want_infected else "Uninfected"
            ax.set_title(f"{label}\n(n={n_tr}, v={mean_v:.2f} {vel_unit})")

        # infected origin plot
        ax_inf = axes[axis_idx]
        axis_idx += 1
        _plot_origin(ax_inf, True)

        # uninfected origin plot
        ax_uninf = axes[axis_idx]
        axis_idx += 1
        _plot_origin(ax_uninf, False)

        # ----- optional small QC PNG (mask panel only) -----
        if qc_panel_needed_mask and axis_idx < len(axes):
            ax_qc = axes[axis_idx]
            axis_idx += 1
            try:
                img = mpimg.imread(qc_panel_path)
                ax_qc.imshow(img)
                ax_qc.axis("off")

                tmap = {
                    "histogram": "Intensity histogram",
                    "pca": "PCA/UMAP clustering",
                    "xgboost": "XGBoost feature importance",
                }
                ttl = tmap.get(str(qc_panel_type).lower(), "Infection QC")
                ax_qc.set_title(ttl, fontsize=9)
            except Exception as e:
                print(
                    f"[_make_intensity_motility_panel] Could not embed QC plot "
                    f"from {qc_panel_path}: {e}"
                )
                ax_qc.set_visible(False)

        # ----- adjusted panel QC subplots: method-specific -----
        if is_adjusted_panel and qc_graphs_enabled:
            if qc_strategy == "histogram" and axis_idx < len(axes):
                ax_hist = axes[axis_idx]
                axis_idx += 1
                # Prefer global payload if present; otherwise compute from per-well df
                src = hist_data if hist_data is not None else df_well
                _plot_hist_qc(ax_hist, src)

            elif qc_strategy in {"pca", "umap", "tsne"} and has_pca and axis_idx < len(axes):
                ax_pca = axes[axis_idx]
                axis_idx += 1
                _plot_pca_qc(ax_pca, pca_data)

            elif qc_strategy == "xgboost" and has_xgb:
                if axis_idx < len(axes):
                    ax_prob = axes[axis_idx]
                    axis_idx += 1
                    _plot_xgb_prob_qc(ax_prob, df_well)
                if axis_idx < len(axes):
                    ax_xgb = axes[axis_idx]
                    axis_idx += 1
                    _plot_xgb_importance_qc(ax_xgb, xgb_data)

        # Plate/well tag for title & filename
        meta_tag = f"{plate_id}_{well_id}"

        fig.suptitle(
            f"Infection panel – {panel_label} labels – method={method_label}\n{meta_tag}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        # Filenames:
        #   mask/original: plate1_A03.pdf
        #   adjusted:      plate1_A03_xgboost_adjusted.pdf
        if is_adjusted_panel:
            out_name = f"{meta_tag}_{method_label}_adjusted.pdf"
        elif is_mask_panel:
            out_name = f"{meta_tag}.pdf"
        else:
            # fallback for any unexpected label_tag
            out_name = f"{meta_tag}_{label_tag}_{method_label}.pdf"

        out_path = os.path.join(motility_dir, out_name)
        fig.savefig(out_path)  # PDF inferred from extension
        plt.close(fig)
        print(
            f"[summarise_tracks_from_merged] Saved per-well intensity+motility panel "
            f"({panel_label}, method={method_label}) for plate={plate_id}, well={well_id} "
            f"to {out_path}"
        )

def _infer_plate_well_meta_tag(df):
    """
    Infer a compact 'plate_well' tag for filenames from a DataFrame that has
    plateID / wellID columns.

    Examples
    --------
    plate1 + A02   -> 'plate1_A02'
    plate1 + many  -> 'plate1_MULTI_WELLS'
    many  + A02    -> 'MULTI_PLATES_A02'
    many  + many   -> 'MULTI_PLATES_MULTI_WELLS'
    """
    plates = sorted(df["plateID"].dropna().unique()) if "plateID" in df.columns else []
    wells = sorted(df["wellID"].dropna().unique()) if "wellID" in df.columns else []

    if len(plates) == 1 and len(wells) == 1:
        return f"{plates[0]}_{wells[0]}"
    elif len(plates) == 1 and len(wells) > 1:
        return f"{plates[0]}_MULTI_WELLS"
    elif len(plates) > 1 and len(wells) == 1:
        return f"MULTI_PLATES_{wells[0]}"
    else:
        return "MULTI_PLATES_MULTI_WELLS"

def _compute_cell_mean_intensity_per_channel(
    mask_stack,
    intensity_stack,
    channel_index,
):
    """
    Compute per-frame, per-cell mean intensity for a given channel.

    Parameters
    ----------
    mask_stack : ndarray
        Label image stack of shape (T, Y, X) for cells (track_id labels).
    intensity_stack : ndarray
        Intensity stack of shape (T, Y, X, C).
    channel_index : int
        Channel index in intensity_stack to use.

    Returns
    -------
    DataFrame
        Columns: ['frame', 'track_id', f'cell_mean_intensity_ch{channel_index}']
    """
    import numpy as np
    import pandas as pd

    if intensity_stack is None:
        print(
            f"[cell_mean_intensity] channel {channel_index}: "
            "intensity_stack is None, skipping."
        )
        return pd.DataFrame(
            columns=["frame", "track_id", f"cell_mean_intensity_ch{channel_index}"]
        )

    if channel_index is None or channel_index < 0 or channel_index >= intensity_stack.shape[-1]:
        print(
            f"[cell_mean_intensity] channel {channel_index}: "
            "invalid channel index for intensity_stack, skipping."
        )
        return pd.DataFrame(
            columns=["frame", "track_id", f"cell_mean_intensity_ch{channel_index}"]
        )

    T = mask_stack.shape[0]
    dfs = []
    col_name = f"cell_mean_intensity_ch{channel_index}"

    for frame in range(T):
        labels = mask_stack[frame]
        if not np.any(labels):
            continue

        intensity_image = intensity_stack[frame, :, :, channel_index]
        props_table = regionprops_table(
            labels,
            intensity_image=intensity_image,
            properties=("label", "mean_intensity"),
        )
        frame_df = pd.DataFrame(props_table)
        frame_df = frame_df.rename(
            columns={
                "label": "track_id",
                "mean_intensity": col_name,
            }
        )
        frame_df["frame"] = frame
        dfs.append(frame_df)

    if not dfs:
        print(
            f"[cell_mean_intensity] channel {channel_index}: "
            f"no objects found in any of {T} frames."
        )
        return pd.DataFrame(columns=["frame", "track_id", col_name])

    out_df = pd.concat(dfs, ignore_index=True)
    n_rows = out_df.shape[0]
    n_frames_detected = out_df["frame"].nunique()
    n_objs = out_df["track_id"].nunique()
    print(
        f"[cell_mean_intensity] channel {channel_index}: "
        f"frames_with_objects={n_frames_detected}/{T}, "
        f"unique_track_id={n_objs}, rows={n_rows}"
    )
    return out_df


def _reorient_merged_array(arr, n_channels, max_extra_masks=3):
    """
    Ensure merged array has shape (planes, H, W) with planes as the first axis.

    Handles both (planes, H, W) and (H, W, planes) layouts by detecting which
    axis likely corresponds to the small "planes" dimension (~n_channels + masks).
    """
    import numpy as np

    if arr.ndim != 3:
        raise ValueError(
            f"_reorient_merged_array expected 3D array, got ndim={arr.ndim}"
        )

    target_min = n_channels
    target_max = n_channels + max_extra_masks
    shape = arr.shape

    plane_axis = None
    for ax, dim in enumerate(shape):
        if target_min <= dim <= target_max:
            plane_axis = ax
            break

    if plane_axis is None:
        # Fallback: choose the smallest axis as planes
        plane_axis = int(np.argmin(shape))

    if plane_axis != 0:
        arr = np.moveaxis(arr, plane_axis, 0)

    planes, H, W = arr.shape
    return arr, planes, H, W


def _parse_merged_filename(fname):
    """
    Parse a merged .npy filename of the form:
        plate_well_field_time.npy

    Returns a dict with:
        plateID, wellID, rowID, columnID, fieldID, timeID, prcf, prcft, filename
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")

    plateID = parts[0] if len(parts) > 0 else ""
    wellID = parts[1] if len(parts) > 1 else ""
    fieldID = parts[2] if len(parts) > 2 else "1"
    time_str = parts[3] if len(parts) > 3 else "0"

    # Extract numeric time index, tolerate formats like "t000"
    digits = "".join(ch for ch in time_str if ch.isdigit())
    timeID = int(digits) if digits else 0

    rowID = wellID[0] if wellID else ""
    col_part = "".join(ch for ch in wellID[1:] if ch.isdigit())
    columnID = int(col_part) if col_part else 0

    prcf = f"{plateID}_{wellID}_{fieldID}"
    prcft = f"{prcf}_{timeID}"

    meta = dict(
        plateID=plateID,
        wellID=wellID,
        rowID=rowID,
        columnID=columnID,
        fieldID=fieldID,
        timeID=timeID,
        prcf=prcf,
        prcft=prcft,
        filename=os.path.basename(fname),
    )
    return meta

def _compute_parent_child_overlaps(
    parent_masks,
    child_masks,
    parent_label_col,
    child_label_col,
):
    """
    For each frame, find which child labels overlap which parent labels.

    Returns columns: 'frame', parent_label_col, child_label_col
    """
    T = parent_masks.shape[0]
    records = []

    for frame in range(T):
        p = parent_masks[frame]
        c = child_masks[frame]
        m = (p > 0) & (c > 0)
        if not np.any(m):
            continue

        p_flat = p[m].ravel()
        c_flat = c[m].ravel()
        pairs = np.stack([p_flat, c_flat], axis=1)
        unique_pairs = np.unique(pairs, axis=0)

        for parent_label, child_label in unique_pairs:
            records.append(
                {
                    "frame": frame,
                    parent_label_col: int(parent_label),
                    child_label_col: int(child_label),
                }
            )

    if not records:
        return pd.DataFrame(columns=["frame", parent_label_col, child_label_col])

    return pd.DataFrame.from_records(records)


def _summarise_child_features_per_parent(
    overlaps_df,
    child_props_df,
    parent_label_col,
    child_label_col,
    count_col_name,
):
    """
    Summarise child object features per parent object.

    - Counts distinct children -> count_col_name
    - Aggregates numeric child features per parent:

      * '*area*'      -> sum
      * '*intensity*' -> mean
      * '*dist*'/'*distance*' -> min
      * everything else -> mean
    """
    if overlaps_df.empty or child_props_df.empty:
        return pd.DataFrame(columns=["frame", parent_label_col, count_col_name])

    df = overlaps_df.merge(child_props_df, on=["frame", child_label_col], how="left")
    if df.empty:
        return pd.DataFrame(columns=["frame", parent_label_col, count_col_name])

    group_cols = ["frame", parent_label_col]

    counts = (
        df.groupby(group_cols)[child_label_col]
        .nunique()
        .reset_index()
        .rename(columns={child_label_col: count_col_name})
    )

    numeric_cols = [
        c
        for c in df.columns
        if c not in group_cols + [child_label_col]
        and np.issubdtype(df[c].dtype, np.number)
    ]
    if not numeric_cols:
        return counts

    def _agg_for_feature(col_name: str) -> str:
        name = col_name.lower()
        if "area" in name:
            return "sum"
        if "intensity" in name:
            return "mean"
        if "distance" in name or "dist" in name:
            return "min"
        return "mean"

    agg_dict = {c: _agg_for_feature(c) for c in numeric_cols}
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    summary = agg_df.merge(counts, on=group_cols, how="left")
    return summary


def _load_intensity_stack_from_merged(
    src,
    filenames,
    n_channels,
    height,
    width,
    dtype=np.float32,
):
    """
    Load intensity channels from merged/*.npy into a (T, H, W, C) stack.

    Supports merged arrays stored either as (planes, H, W) or (H, W, planes).
    The first n_channels planes are intensities and any remaining planes are masks.
    """
    import os
    import numpy as np

    merged_dir = os.path.join(src, "merged")
    T = len(filenames)

    if not os.path.isdir(merged_dir) or n_channels is None or n_channels <= 0:
        return np.zeros((T, height, width, 0), dtype=dtype)

    stack = np.zeros((T, height, width, n_channels), dtype=dtype)

    for t, fn in enumerate(filenames):
        base = os.path.splitext(os.path.basename(fn))[0]
        candidates = [
            os.path.join(merged_dir, base + ".npy"),
            os.path.join(merged_dir, fn),
            os.path.join(merged_dir, fn + ".npy"),
        ]
        arr = None
        for path in candidates:
            if os.path.exists(path):
                arr = np.load(path)
                break

        if arr is None or arr.ndim != 3:
            continue

        # Standardise to (planes, H, W)
        try:
            arr, planes, H_img, W_img = _reorient_merged_array(
                arr, n_channels=n_channels
            )
        except ValueError:
            continue

        if H_img != height or W_img != width:
            # Skip unexpected size
            print(
                f"[_load_intensity_stack_from_merged] Skipping {fn}: "
                f"reoriented size=({planes}, {H_img}, {W_img}), "
                f"expected H={height}, W={width}"
            )
            continue

        use_planes = min(n_channels, planes)
        if use_planes <= 0:
            continue

        img = arr[:use_planes].transpose(1, 2, 0)  # (H, W, C)
        C = img.shape[2]
        stack[t, :, :, :C] = img

    return stack


def _load_masks_from_merged(
    src,
    filenames,
    n_channels,
    height,
    width,
    nucleus_chan=None,
    pathogen_chan=None,
    dtype=None,
):
    """
    Load cell / nucleus / pathogen masks from merged/*.npy.

    Supports merged arrays stored either as (planes, H, W) or (H, W, planes).

    Layout per merged array after reorientation (planes, H, W):

        0 .. n_channels-1          → intensity channels
        n_channels                 → cell_mask (always present)
        n_channels + 1 (optional)  → nucleus_mask or pathogen_mask
        n_channels + 2 (optional)  → pathogen_mask (when both nuc+pathogen exist)

    The exact interpretation of mask planes depends on whether
    `nucleus_chan` and/or `pathogen_chan` are None.
    """
    import os
    import numpy as np

    if dtype is None:
        dtype = np.int32

    merged_dir = os.path.join(src, "merged")
    T = len(filenames)

    cell_masks = np.zeros((T, height, width), dtype=dtype)
    nucleus_masks = np.zeros((T, height, width), dtype=dtype)
    pathogen_masks = np.zeros((T, height, width), dtype=dtype)

    if not os.path.isdir(merged_dir):
        return cell_masks, nucleus_masks, pathogen_masks

    for t, fn in enumerate(filenames):
        base = os.path.splitext(os.path.basename(fn))[0]
        candidates = [
            os.path.join(merged_dir, base + ".npy"),
            os.path.join(merged_dir, fn),
            os.path.join(merged_dir, fn + ".npy"),
        ]
        arr = None
        for path in candidates:
            if os.path.exists(path):
                arr = np.load(path)
                break

        if arr is None or arr.ndim != 3:
            continue

        # Standardise to (planes, H, W)
        try:
            arr, planes, H_img, W_img = _reorient_merged_array(
                arr, n_channels=n_channels
            )
        except ValueError:
            continue

        if H_img != height or W_img != width:
            print(
                f"[_load_masks_from_merged] Skipping {fn}: "
                f"reoriented size=({planes}, {H_img}, {W_img}), "
                f"expected H={height}, W={width}"
            )
            continue

        if planes <= n_channels:
            # Only intensity planes, no masks
            continue

        n_masks = planes - n_channels

        # First mask plane is always cell
        cell_masks[t] = arr[n_channels].astype(dtype)

        # Second mask plane (if present) is nucleus OR pathogen depending on settings
        if n_masks >= 2:
            if nucleus_chan is not None and pathogen_chan is None:
                nucleus_masks[t] = arr[n_channels + 1].astype(dtype)
            elif nucleus_chan is None and pathogen_chan is not None:
                pathogen_masks[t] = arr[n_channels + 1].astype(dtype)
            elif nucleus_chan is not None and pathogen_chan is not None:
                # both requested → expect nucleus here
                nucleus_masks[t] = arr[n_channels + 1].astype(dtype)

        # Third mask plane (if present) is pathogen when both nuc+pathogen exist
        if n_masks >= 3 and pathogen_chan is not None:
            pathogen_masks[t] = arr[n_channels + 2].astype(dtype)

    return cell_masks, nucleus_masks, pathogen_masks

def _compute_regionprops_stack(
    mask_stack,
    intensity_stack,
    channel_index,
    object_prefix,
    label_as_track_id=False,
):
    """
    Compute regionprops over a (T, Y, X) label stack.

    Parameters
    ----------
    mask_stack : ndarray
        Label image stack of shape (T, Y, X).
    intensity_stack : ndarray or None
        Intensity stack of shape (T, Y, X, C) or None.
    channel_index : int or None
        Channel index in intensity_stack to use for intensity props.
    object_prefix : str
        Prefix for column names ("cell", "nucleus", "pathogen", "cytoplasm").
    label_as_track_id : bool
        If True, rename 'label' to 'track_id',
        otherwise to f"{object_prefix}_label".

    Returns
    -------
    DataFrame
        One row per object per frame with prefixed column names.
    """
    import numpy as np
    import pandas as pd

    T, H, W = mask_stack.shape
    use_intensity = (
        intensity_stack is not None
        and channel_index is not None
        and 0 <= channel_index < intensity_stack.shape[-1]
    )

    # Avoid properties that rely on normalized central moments
    geom_props = [
        "label",
        "area",
        "bbox_area",
        "equivalent_diameter",
        "perimeter",
        "perimeter_crofton",
        "solidity",
        "centroid",
    ]
    intensity_props = [
        "max_intensity",
        "mean_intensity",
        "min_intensity",
    ]
    props = geom_props + intensity_props if use_intensity else geom_props

    label_col_name = "track_id" if label_as_track_id else f"{object_prefix}_label"

    dfs = []
    for frame in range(T):
        labels = mask_stack[frame]
        if not np.any(labels):
            continue

        if use_intensity:
            intensity_image = intensity_stack[frame, :, :, channel_index]
            props_table = regionprops_table(
                labels,
                intensity_image=intensity_image,
                properties=props,
            )
        else:
            props_table = regionprops_table(labels, properties=props)

        frame_df = pd.DataFrame(props_table)
        frame_df = frame_df.rename(columns={"label": label_col_name})
        frame_df["frame"] = frame

        feature_cols = [
            c for c in frame_df.columns if c not in ("frame", label_col_name)
        ]
        frame_df = frame_df.rename(
            columns={c: f"{object_prefix}_{c}" for c in feature_cols}
        )
        dfs.append(frame_df)

    if not dfs:
        print(f"[regionprops] {object_prefix}: no objects found in any of {T} frames.")
        return pd.DataFrame(columns=["frame", label_col_name])

    out_df = pd.concat(dfs, ignore_index=True)

    n_rows = out_df.shape[0]
    n_frames_detected = out_df["frame"].nunique()
    n_objs = out_df[label_col_name].nunique()
    print(
        f"[regionprops] {object_prefix}: frames_with_objects="
        f"{n_frames_detected}/{T}, unique_{label_col_name}={n_objs}, rows={n_rows}"
    )

    return out_df


def _process_merged_group(args):
    """
    Worker: process one (plate, well, field) group of merged .npy files.

    Returns per-cell-per-frame DataFrame with:
      - metadata
      - cell features
      - aggregated nucleus / pathogen / cytoplasm features
      - per-channel cell mean intensities (cell_mean_intensity_ch{c})
    """
    import numpy as np
    import pandas as pd
    import os

    (
        src,
        file_basenames,
        n_channels,
        cell_chan,
        nucleus_chan,
        pathogen_chan,
    ) = args

    if not file_basenames:
        print("[_process_merged_group] Empty file_basenames list.")
        return pd.DataFrame()

    merged_dir = os.path.join(src, "merged")

    # sort filenames by timeID
    metas = []
    for bn in file_basenames:
        meta = _parse_merged_filename(bn)
        metas.append(meta)
    metas_sorted = sorted(metas, key=lambda m: m["timeID"])
    sorted_basenames = [m["filename"] for m in metas_sorted]

    key = (
        metas_sorted[0]["plateID"],
        metas_sorted[0]["wellID"],
        metas_sorted[0]["fieldID"],
    )
    print(f"[_process_merged_group] Start group {key}, files={len(sorted_basenames)}")

    # infer size from first file (respecting orientation)
    first_path = os.path.join(merged_dir, sorted_basenames[0])
    first_arr_raw = np.load(first_path)
    if first_arr_raw.ndim != 3:
        print(
            f"[_process_merged_group] First array for group {key} is not 3D, "
            "skipping."
        )
        return pd.DataFrame()

    try:
        first_arr, planes, H, W = _reorient_merged_array(
            first_arr_raw, n_channels=n_channels
        )
    except ValueError:
        print(
            f"[_process_merged_group] Group {key}: could not reorient first array "
            f"with shape={first_arr_raw.shape}, skipping."
        )
        return pd.DataFrame()

    base_dtype = first_arr.dtype
    print(
        f"[_process_merged_group] Group {key}: first array original_shape="
        f"{first_arr_raw.shape}, reoriented_shape=({planes}, {H}, {W}), "
        f"dtype={base_dtype}"
    )

    # load stacks
    intensity_stack = _load_intensity_stack_from_merged(
        src=src,
        filenames=sorted_basenames,
        n_channels=n_channels,
        height=H,
        width=W,
        dtype=base_dtype,
    )

    cell_masks, nucleus_masks, pathogen_masks = _load_masks_from_merged(
        src=src,
        filenames=sorted_basenames,
        n_channels=n_channels,
        height=H,
        width=W,
        nucleus_chan=nucleus_chan,
        pathogen_chan=pathogen_chan,
        dtype=np.int32,
    )

    T = cell_masks.shape[0]
    if T == 0 or not np.any(cell_masks):
        print(f"[_process_merged_group] Group {key}: no cell masks found, skipping.")
        return pd.DataFrame()

    print(
        f"[_process_merged_group] Group {key}: frames={T}, "
        f"any_nucleus={np.any(nucleus_masks)}, any_pathogen={np.any(pathogen_masks)}"
    )

    # cytoplasm = cell minus (nucleus union pathogen)
    has_nucleus = np.any(nucleus_masks)
    has_pathogen = np.any(pathogen_masks)
    cytoplasm_masks = None
    if has_nucleus or has_pathogen:
        cytoplasm_masks = cell_masks.copy()
        if has_nucleus:
            cytoplasm_masks[nucleus_masks > 0] = 0
        if has_pathogen:
            cytoplasm_masks[pathogen_masks > 0] = 0

    # regionprops for cell geometry (+ intensities in cell_chan)
    cell_props_df = _compute_regionprops_stack(
        mask_stack=cell_masks,
        intensity_stack=intensity_stack,
        channel_index=cell_chan,
        object_prefix="cell",
        label_as_track_id=True,
    )
    nucleus_props_df = _compute_regionprops_stack(
        mask_stack=nucleus_masks,
        intensity_stack=intensity_stack,
        channel_index=nucleus_chan,
        object_prefix="nucleus",
        label_as_track_id=False,
    )
    pathogen_props_df = _compute_regionprops_stack(
        mask_stack=pathogen_masks,
        intensity_stack=intensity_stack,
        channel_index=pathogen_chan,
        object_prefix="pathogen",
        label_as_track_id=False,
    )

    cytoplasm_props_df = pd.DataFrame()
    if cytoplasm_masks is not None and np.any(cytoplasm_masks):
        cytoplasm_props_df = _compute_regionprops_stack(
            mask_stack=cytoplasm_masks,
            intensity_stack=intensity_stack,
            channel_index=cell_chan,  # use same channel as cell by default
            object_prefix="cytoplasm",
            label_as_track_id=False,
        )

    # --- per-channel intensity percentiles for each compartment ---
    percentile_dfs_cell = []
    percentile_dfs_nucleus = []
    percentile_dfs_pathogen = []
    percentile_dfs_cytoplasm = []

    for ch in range(n_channels):
        # cell: track_id labels
        df_p = _compute_intensity_percentiles_per_channel(
            mask_stack=cell_masks,
            intensity_stack=intensity_stack,
            channel_index=ch,
            object_prefix="cell",
            label_as_track_id=True,
        )
        if not df_p.empty:
            percentile_dfs_cell.append(df_p)

        # nucleus
        if np.any(nucleus_masks):
            df_p_n = _compute_intensity_percentiles_per_channel(
                mask_stack=nucleus_masks,
                intensity_stack=intensity_stack,
                channel_index=ch,
                object_prefix="nucleus",
                label_as_track_id=False,
            )
            if not df_p_n.empty:
                percentile_dfs_nucleus.append(df_p_n)

        # pathogen
        if np.any(pathogen_masks):
            df_p_pa = _compute_intensity_percentiles_per_channel(
                mask_stack=pathogen_masks,
                intensity_stack=intensity_stack,
                channel_index=ch,
                object_prefix="pathogen",
                label_as_track_id=False,
            )
            if not df_p_pa.empty:
                percentile_dfs_pathogen.append(df_p_pa)

        # cytoplasm
        if cytoplasm_masks is not None and np.any(cytoplasm_masks):
            df_p_cy = _compute_intensity_percentiles_per_channel(
                mask_stack=cytoplasm_masks,
                intensity_stack=intensity_stack,
                channel_index=ch,
                object_prefix="cytoplasm",
                label_as_track_id=False,
            )
            if not df_p_cy.empty:
                percentile_dfs_cytoplasm.append(df_p_cy)

    # merge percentile features into base props
    if percentile_dfs_cell:
        tmp = percentile_dfs_cell[0]
        for df_p in percentile_dfs_cell[1:]:
            tmp = tmp.merge(df_p, on=["frame", "track_id"], how="outer")
        cell_props_df = cell_props_df.merge(
            tmp, on=["frame", "track_id"], how="left"
        )

    if np.any(nucleus_masks) and not nucleus_props_df.empty and percentile_dfs_nucleus:
        tmp = percentile_dfs_nucleus[0]
        for df_p in percentile_dfs_nucleus[1:]:
            tmp = tmp.merge(df_p, on=["frame", "nucleus_label"], how="outer")
        nucleus_props_df = nucleus_props_df.merge(
            tmp, on=["frame", "nucleus_label"], how="left"
        )

    if np.any(pathogen_masks) and not pathogen_props_df.empty and percentile_dfs_pathogen:
        tmp = percentile_dfs_pathogen[0]
        for df_p in percentile_dfs_pathogen[1:]:
            tmp = tmp.merge(df_p, on=["frame", "pathogen_label"], how="outer")
        pathogen_props_df = pathogen_props_df.merge(
            tmp, on=["frame", "pathogen_label"], how="left"
        )

    if (
        cytoplasm_masks is not None
        and np.any(cytoplasm_masks)
        and not cytoplasm_props_df.empty
        and percentile_dfs_cytoplasm
    ):
        tmp = percentile_dfs_cytoplasm[0]
        for df_p in percentile_dfs_cytoplasm[1:]:
            tmp = tmp.merge(df_p, on=["frame", "cytoplasm_label"], how="outer")
        cytoplasm_props_df = cytoplasm_props_df.merge(
            tmp, on=["frame", "cytoplasm_label"], how="left"
        )


    if cell_props_df.empty:
        print(f"[_process_merged_group] Group {key}: cell_props_df empty, skipping.")
        return pd.DataFrame()

    # --- per-channel cell mean intensities (one column per channel) ---
    per_channel_intensity_dfs = []
    for ch in range(n_channels):
        df_ch = _compute_cell_mean_intensity_per_channel(
            mask_stack=cell_masks,
            intensity_stack=intensity_stack,
            channel_index=ch,
        )
        if not df_ch.empty:
            per_channel_intensity_dfs.append(df_ch)

    cell_intensity_df = None
    if per_channel_intensity_dfs:
        cell_intensity_df = per_channel_intensity_dfs[0]
        for df_ch in per_channel_intensity_dfs[1:]:
            cell_intensity_df = cell_intensity_df.merge(
                df_ch,
                on=["frame", "track_id"],
                how="outer",
            )
        added_cols = [
            c
            for c in cell_intensity_df.columns
            if c.startswith("cell_mean_intensity_ch")
        ]
        print(
            f"[_process_merged_group] Group {key}: added per-channel cell "
            f"intensity columns: {added_cols}"
        )

    # overlaps and summaries
    nucleus_summary = None
    if has_nucleus:
        overlaps_cn = _compute_parent_child_overlaps(
            parent_masks=cell_masks,
            child_masks=nucleus_masks,
            parent_label_col="track_id",
            child_label_col="nucleus_label",
        )
        if not overlaps_cn.empty and not nucleus_props_df.empty:
            nucleus_summary = _summarise_child_features_per_parent(
                overlaps_df=overlaps_cn,
                child_props_df=nucleus_props_df,
                parent_label_col="track_id",
                child_label_col="nucleus_label",
                count_col_name="n_nuclei",
            )
            print(
                f"[_process_merged_group] Group {key}: nucleus_summary rows="
                f"{len(nucleus_summary)}"
            )

    pathogen_summary = None
    if has_pathogen:
        overlaps_cp = _compute_parent_child_overlaps(
            parent_masks=cell_masks,
            child_masks=pathogen_masks,
            parent_label_col="track_id",
            child_label_col="pathogen_label",
        )
        if not overlaps_cp.empty and not pathogen_props_df.empty:
            pathogen_summary = _summarise_child_features_per_parent(
                overlaps_df=overlaps_cp,
                child_props_df=pathogen_props_df,
                parent_label_col="track_id",
                child_label_col="pathogen_label",
                count_col_name="n_pathogens",
            )
            print(
                f"[_process_merged_group] Group {key}: pathogen_summary rows="
                f"{len(pathogen_summary)}"
            )

    cytoplasm_summary = None
    if (
        cytoplasm_masks is not None
        and np.any(cytoplasm_masks)
        and not cytoplasm_props_df.empty
    ):
        overlaps_cc = _compute_parent_child_overlaps(
            parent_masks=cell_masks,
            child_masks=cytoplasm_masks,
            parent_label_col="track_id",
            child_label_col="cytoplasm_label",
        )
        if not overlaps_cc.empty:
            cytoplasm_summary = _summarise_child_features_per_parent(
                overlaps_df=overlaps_cc,
                child_props_df=cytoplasm_props_df,
                parent_label_col="track_id",
                child_label_col="cytoplasm_label",
                count_col_name="n_cytoplasm",
            )
            print(
                f"[_process_merged_group] Group {key}: cytoplasm_summary rows="
                f"{len(cytoplasm_summary)}"
            )

    enriched_df = cell_props_df.copy()

    if nucleus_summary is not None and not nucleus_summary.empty:
        enriched_df = enriched_df.merge(
            nucleus_summary,
            on=["frame", "track_id"],
            how="left",
        )
    if pathogen_summary is not None and not pathogen_summary.empty:
        enriched_df = enriched_df.merge(
            pathogen_summary,
            on=["frame", "track_id"],
            how="left",
        )
    if cytoplasm_summary is not None and not cytoplasm_summary.empty:
        enriched_df = enriched_df.merge(
            cytoplasm_summary,
            on=["frame", "track_id"],
            how="left",
        )

    if cell_intensity_df is not None:
        enriched_df = enriched_df.merge(
            cell_intensity_df,
            on=["frame", "track_id"],
            how="left",
        )

    # attach metadata (plate, well, field, timeID, etc.)
    meta_records = []
    for local_frame_idx, meta in enumerate(metas_sorted):
        rec = {"frame": local_frame_idx}
        rec.update(meta)
        meta_records.append(rec)
    meta_df = pd.DataFrame(meta_records)

    enriched_df = enriched_df.merge(meta_df, on="frame", how="left")
    enriched_df["cellID"] = enriched_df["track_id"]

    n_tracks = (
        enriched_df[["plateID", "wellID", "fieldID", "cellID"]]
        .drop_duplicates()
        .shape[0]
    )
    print(
        f"[_process_merged_group] Group {key}: enriched_df rows={len(enriched_df)}, "
        f"unique_tracks={n_tracks}"
    )

    return enriched_df


def _smooth_tracks_and_features(df, max_displacement=50.0, zscore_thresh=3.0):
    """
    Smooth cell tracks and a small set of scalar features.

    - Fixes single-frame "teleport" glitches in centroid position.
    - Optionally drops tracks with impossible jumps.
    - Smooths a subset of scalar cell_* features using a z-score heuristic.
    """
    import numpy as np
    import pandas as pd

    if df.empty:
        print("[_smooth_tracks_and_features] Input DataFrame is empty.")
        return df

    n_rows_before = len(df)
    n_tracks_before = df[["plateID", "wellID", "fieldID", "cellID"]].drop_duplicates().shape[0]

    df = df.sort_values(
        ["plateID", "wellID", "fieldID", "cellID", "frame"]
    ).reset_index(drop=True)

    y_col = "cell_centroid-0"
    x_col = "cell_centroid-1"
    if y_col not in df.columns or x_col not in df.columns:
        print("[_smooth_tracks_and_features] Centroid columns missing, nothing to smooth.")
        return df

    drop_indices = set()
    updates = {}

    # Only smooth scalar features with well-defined numeric dtype
    candidate_cols = [
        "cell_area",
        "cell_bbox_area",
        "cell_equivalent_diameter",
        "cell_perimeter",
        "cell_perimeter_crofton",
        "cell_solidity",
        "cell_mean_intensity",
        "cell_max_intensity",
        "cell_min_intensity",
    ]
    cell_feature_cols = [c for c in candidate_cols if c in df.columns]

    # Ensure we are not writing floats into int columns (avoid FutureWarning)
    for col in [y_col, x_col] + cell_feature_cols:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.floating):
            df[col] = df[col].astype(float)

    grouped = df.groupby(["plateID", "wellID", "fieldID", "cellID"], sort=False)

    n_tracks_processed = 0
    n_tracks_dropped = 0
    n_glitches_fixed = 0

    for (plateID, wellID, fieldID, cellID), g in grouped:
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue

        n_tracks_processed += 1

        y = g[y_col].to_numpy(dtype=float)
        x = g[x_col].to_numpy(dtype=float)
        n = len(idx)
        glitch_frames = set()

        # --- 1) detect and interpolate single-frame centroid glitches ---
        if n >= 3:
            for i_local in range(1, n - 1):
                y_prev, y_curr, y_next = y[i_local - 1], y[i_local], y[i_local + 1]
                x_prev, x_curr, x_next = x[i_local - 1], x[i_local], x[i_local + 1]

                d_prev = np.hypot(y_curr - y_prev, x_curr - x_prev)
                d_next = np.hypot(y_curr - y_next, x_curr - x_next)
                d_neigh = np.hypot(y_next - y_prev, x_next - x_prev)

                if (
                    d_prev > max_displacement
                    and d_next > max_displacement
                    and d_neigh <= max_displacement
                ):
                    glitch_frames.add(i_local)

            # interpolate centroid + scalar features at glitch frames
            for i_local in glitch_frames:
                if i_local <= 0 or i_local >= n - 1:
                    continue

                n_glitches_fixed += 1

                y_new = 0.5 * (y[i_local - 1] + y[i_local + 1])
                x_new = 0.5 * (x[i_local - 1] + x[i_local + 1])
                y[i_local] = y_new
                x[i_local] = x_new

                for col in cell_feature_cols:
                    s = g[col].to_numpy(dtype=float)
                    if len(s) < 3:
                        continue
                    s_new = 0.5 * (s[i_local - 1] + s[i_local + 1])
                    updates.setdefault(col, {})[idx[i_local]] = s_new

            # --- 2) drop tracks with big jumps not explainable as glitches ---
            drop_track = False
            for i_local in range(1, n):
                d = np.hypot(y[i_local] - y[i_local - 1], x[i_local] - x[i_local - 1])
                if (
                    d > max_displacement
                    and i_local not in glitch_frames
                    and (i_local - 1) not in glitch_frames
                ):
                    drop_track = True
                    break

            if drop_track:
                n_tracks_dropped += 1
                drop_indices.update(idx.tolist())
                continue

        # write back smoothed centroid
        for i_local, global_idx in enumerate(idx):
            if y[i_local] != g[y_col].iloc[i_local]:
                updates.setdefault(y_col, {})[global_idx] = y[i_local]
            if x[i_local] != g[x_col].iloc[i_local]:
                updates.setdefault(x_col, {})[global_idx] = x[i_local]

        # --- 3) z-score based smoothing of scalar features ---
        if len(idx) < 3 or not cell_feature_cols:
            continue

        for col in cell_feature_cols:
            s = g[col].to_numpy(dtype=float)
            if np.all(~np.isfinite(s)):
                continue

            mean = np.nanmean(s)
            std = np.nanstd(s)
            if not np.isfinite(std) or std == 0:
                continue

            z = (s - mean) / std
            for i_local in range(1, n - 1):
                if not np.isfinite(z[i_local]) or abs(z[i_local]) <= zscore_thresh:
                    continue
                if (
                    abs(z[i_local - 1]) <= zscore_thresh / 2
                    and abs(z[i_local + 1]) <= zscore_thresh / 2
                ):
                    new_val = 0.5 * (s[i_local - 1] + s[i_local + 1])
                    updates.setdefault(col, {})[idx[i_local]] = new_val

    # apply all updates in one go
    for col, mapping in updates.items():
        df.loc[list(mapping.keys()), col] = list(mapping.values())

    if drop_indices:
        df = df.drop(index=list(drop_indices)).reset_index(drop=True)

    n_rows_after = len(df)
    n_tracks_after = df[["plateID", "wellID", "fieldID", "cellID"]].drop_duplicates().shape[0]

    print(
        "[_smooth_tracks_and_features] rows_before="
        f"{n_rows_before}, rows_after={n_rows_after}, "
        f"tracks_before={n_tracks_before}, tracks_after={n_tracks_after}, "
        f"tracks_processed={n_tracks_processed}, tracks_dropped={n_tracks_dropped}, "
        f"glitches_fixed={n_glitches_fixed}"
    )

    return df

def _debug_plot_merged_planes(src, sample_filename, n_channels, nucleus_chan, pathogen_chan, out_dir):
    """
    Debug-plot a single merged .npy file.

    The plot is saved as a PDF and contains:
      - one panel per raw intensity channel (normalized 2–98 percent)
      - one panel per mask plane (random colormap)
      - one panel showing merged intensity channels with all masks overlaid
        using a random colormap with alpha=0.6.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl  # needed by _generate_mask_random_cmap if defined elsewhere

    merged_path = os.path.join(src, "merged", sample_filename)
    if not os.path.isfile(merged_path):
        print(f"[_debug_plot_merged_planes] File not found: {merged_path}")
        return

    arr = np.load(merged_path)
    original_shape = arr.shape

    # Re-orient to (planes, y, x)
    if arr.ndim == 3:
        # (Y, X, planes) -> (planes, Y, X)
        if arr.shape[-1] != n_channels and arr.shape[0] == n_channels:
            planes = arr
        else:
            planes = np.moveaxis(arr, -1, 0)
    elif arr.ndim == 4:
        # Take first timepoint; assume (T, Y, X, planes) or similar
        if arr.shape[-1] >= n_channels:
            planes = np.moveaxis(arr[0], -1, 0)
        else:
            # fallback: collapse time into planes
            planes = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    else:
        # Fallback, try to interpret leading axis as planes
        planes = arr

    reoriented_shape = planes.shape
    print(
        f"[_debug_plot_merged_planes] Sample '{sample_filename}': "
        f"original_shape={original_shape}, reoriented_shape={reoriented_shape}"
    )

    if planes.ndim != 3:
        print(
            f"[_debug_plot_merged_planes] Expected 3D array after reorientation, "
            f"got shape={planes.shape}; skipping."
        )
        return

    n_planes = planes.shape[0]
    if n_planes < n_channels:
        n_channels = n_planes

    intensity_planes = planes[:n_channels].astype(float)
    mask_planes = planes[n_channels:]
    n_masks = mask_planes.shape[0]

    # Normalize intensity channels to 2–98 percentiles
    norm_intensity = []
    for ch_idx in range(n_channels):
        p = intensity_planes[ch_idx].astype(float)
        lo = np.percentile(p, 2)
        hi = np.percentile(p, 98)
        if hi <= lo:
            p_norm = np.zeros_like(p, dtype=float)
        else:
            p_norm = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
        norm_intensity.append(p_norm)
    norm_intensity = np.asarray(norm_intensity)

    if norm_intensity.size == 0:
        print("[_debug_plot_merged_planes] No intensity channels to plot; skipping.")
        return

    H, W = norm_intensity[0].shape

    # Build RGB merge of intensity channels (up to 3)
    merged_rgb = np.zeros((H, W, 3), dtype=float)
    if n_channels >= 1:
        merged_rgb[..., 0] = norm_intensity[0]  # red
    if n_channels >= 2:
        merged_rgb[..., 1] = norm_intensity[1]  # green
    if n_channels >= 3:
        merged_rgb[..., 2] = norm_intensity[2]  # blue

    # Combined mask for overlay
    combined_mask = None
    if n_masks > 0:
        combined_mask = np.zeros((H, W), dtype=int)
        offset = 0
        for m in mask_planes:
            m_int = m.astype(int)
            if m_int.max() <= 0:
                continue
            nonzero = m_int > 0
            combined_mask[nonzero] = m_int[nonzero] + offset
            offset += int(m_int.max())
        if offset == 0:
            combined_mask = None

    # Figure layout: channels + masks + merged overlay
    extra = 1 if combined_mask is not None else 0
    n_cols = n_channels + n_masks + extra

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3 * n_cols, 3),
        dpi=150,
        squeeze=False,
    )
    axes = axes[0]

    col_idx = 0

    # Intensity channels
    for ch_idx in range(n_channels):
        ax = axes[col_idx]
        col_idx += 1
        ax.imshow(norm_intensity[ch_idx], cmap="gray")
        ax.set_title(f"Ch {ch_idx} (2–98% norm)")
        ax.axis("off")

    # Individual mask planes with random cmap
    for m_idx in range(n_masks):
        ax = axes[col_idx]
        col_idx += 1
        mask_plane = mask_planes[m_idx]
        try:
            random_cmap = _generate_mask_random_cmap(mask_plane)
        except NameError:
            # Fallback: create a simple random colormap here
            unique_labels = np.unique(mask_plane)
            unique_labels = unique_labels[unique_labels != 0]
            n_labels = len(unique_labels)
            rng = np.random.default_rng(seed=42)
            colors = np.ones((n_labels + 1, 4))
            colors[1:, :3] = rng.random((n_labels, 3))
            random_cmap = mpl.colors.ListedColormap(colors)
        ax.imshow(mask_plane, cmap=random_cmap, interpolation="nearest")
        ax.set_title(f"Mask {m_idx}")
        ax.axis("off")

    # Merged channels + combined masks
    if combined_mask is not None:
        ax = axes[col_idx]
        try:
            merged_cmap = _generate_mask_random_cmap(combined_mask)
        except NameError:
            unique_labels = np.unique(combined_mask)
            unique_labels = unique_labels[unique_labels != 0]
            n_labels = len(unique_labels)
            rng = np.random.default_rng(seed=123)
            colors = np.ones((n_labels + 1, 4))
            colors[1:, :3] = rng.random((n_labels, 3))
            merged_cmap = mpl.colors.ListedColormap(colors)
        ax.imshow(merged_rgb)
        ax.imshow(combined_mask, cmap=merged_cmap, alpha=0.6, interpolation="nearest")
        ax.set_title("Merged channels + masks")
        ax.axis("off")

    fig.tight_layout()
    base = os.path.splitext(sample_filename)[0]
    out_path = os.path.join(out_dir, f"merged_planes_{base}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[_debug_plot_merged_planes] Saved merged plane debug figure to {out_path}"
    )

def _infection_qc_pca_clustering(
    all_df,
    settings,
    infection_col,
    pathogen_chan,
    motility_dir,
):
    """
    Embedding-based infection intensity QC (PCA/UMAP/t-SNE).

    Steps
    -----
    1. Aggregate to per-cell features (cell_* columns).
    2. Select morphology + pathogen-channel intensity features.
    3. Optionally transform features to improve structure:
         - log1p on intensity-like features (if enabled)
         - up-weight pathogen-channel features (if configured)
    4. Embed to 2D using PCA / UMAP / t-SNE
       (controlled by settings['infection_intensity_strategy']:
        'pca', 'umap', or 'tsne').
       For UMAP and t-SNE, an internal hyperparameter search is
       performed (if enabled) to maximize a separation score based on:
         - distance between the two clusters in the embedding
         - how well clusters separate GT infected vs GT uninfected.
    5. KMeans clustering (2 clusters) in the embedded space.
    6. Define "ground-truth" subsets based on pathogen-channel intensity:
         - uninfected_gt: lowest 25% of intensities in the UNINFECTED group
         - infected_gt  : highest 25% of intensities in the INFECTED group
    7. Assign clusters as "infected" vs "uninfected" based on which
       ground-truth class dominates each cluster.
    8. Depending on infection_intensity_mode:
         - 'relabel': adjusted_infected = cluster assignment.
         - 'remove' : drop cells whose original label disagrees with the
                     cluster assignment.
    9. Map adjusted_infected back to all_df (frame level).

    Hyperparameter search
    ---------------------
    UMAP:
      - Enabled if settings.get('infection_pca_umap_search', True) is True.
      - Grid (overrideable via settings):
          settings['infection_pca_umap_n_neighbors_grid'] (default [5, 10, 15, 30])
          settings['infection_pca_umap_min_dist_grid']    (default [0.0, 0.05, 0.1, 0.3])
      - Score = centroid_distance * |GT-infected fraction difference between clusters|.

    t-SNE:
      - Enabled if settings.get('infection_pca_tsne_search', True) is True.
      - Grid (overrideable via settings):
          settings['infection_pca_tsne_perplexity_grid']      (default [15.0, 30.0, 45.0])
          settings['infection_pca_tsne_learning_rate_grid']   (default [200.0, 500.0])
        (perplexity is clamped to < (n_samples-1)/3.)
      - Same score definition as for UMAP.

    PCA “structure” helpers
    -----------------------
    - Optional log1p transform on intensity-like features
      (if settings.get('infection_pca_log_intensity', True) is True).
    - Optional up-weighting of pathogen-channel features after
      standardization:
          settings['infection_pca_pathogen_weight'] (default 1.0)

    Side effects
    ------------
    - settings['infection_pca_data'] with:
        {
          'coords': coords (n_cells x 2),
          'labels': adjusted_infected (bool),
          'cluster_labels': cluster_ids (0 or 1),
          'method_label': 'PCA' / 'UMAP' / 't-SNE',
          'infected_cluster': int,
          'uninfected_cluster': int,
          'initial_infected_frac_infected_cluster': float (0-1),
          'initial_infected_frac_uninfected_cluster': float (0-1),
          'gt_sep_score': float,
          'silhouette_score': float or None,
          'centroid_distance': float,
          'embedding_params': dict (e.g. {'n_neighbors': 15, 'min_dist': 0.1}),
        }
    - settings['infection_intensity_qc_panel_type'] = 'pca'
    - settings['infection_intensity_qc_panel_path'] = None
    """
    import os
    import numpy as np
    import pandas as pd

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    # Optional imports for alternative embeddings
    try:
        from sklearn.manifold import TSNE
    except Exception:  # optional
        TSNE = None

    try:
        import umap  # type: ignore
    except Exception:  # optional
        umap = None

    # ------------------------------------------------------------------
    # Helper: evaluate an embedding + clustering
    # ------------------------------------------------------------------
    def _evaluate_embedding(coords, cluster_labels, y_orig, gt_uninf, gt_inf):
        """
        Compute:
          - infected/uninfected cluster mapping using GT sets
          - GT separation score
          - silhouette score
          - centroid distance between the two clusters
          - original infected fractions in each cluster
          - overall score = centroid_distance * GT separation
        """
        # GT-based fractional infection per cluster
        frac_inf_gt = []
        for k in (0, 1):
            mask_k = cluster_labels == k
            n_inf_k = int(np.sum(mask_k & gt_inf))
            n_uninf_k = int(np.sum(mask_k & gt_uninf))
            tot_k = n_inf_k + n_uninf_k
            if tot_k == 0:
                frac_inf_gt.append(0.5)
            else:
                frac_inf_gt.append(n_inf_k / float(tot_k))

        infected_cluster = int(np.argmax(frac_inf_gt))
        uninfected_cluster = 1 - infected_cluster

        gt_sep_score = abs(frac_inf_gt[0] - frac_inf_gt[1])

        # Centroid distance in embedding space
        centroids = []
        for k in (0, 1):
            mask_k = cluster_labels == k
            if mask_k.any():
                centroids.append(coords[mask_k].mean(axis=0))
            else:
                centroids.append(np.zeros(coords.shape[1], dtype=float))
        centroid_distance = float(np.linalg.norm(centroids[0] - centroids[1]))

        # Silhouette in embedding space
        sil = None
        if coords.shape[0] > 10 and len(np.unique(cluster_labels)) > 1:
            try:
                sil = float(silhouette_score(coords, cluster_labels))
            except Exception:
                sil = None

        # Original infected fractions in each cluster
        mask_inf_cluster = cluster_labels == infected_cluster
        mask_uninf_cluster = cluster_labels == uninfected_cluster
        frac_inf_infected_cluster = (
            float(y_orig[mask_inf_cluster].mean()) if mask_inf_cluster.any() else 0.0
        )
        frac_inf_uninfected_cluster = (
            float(y_orig[mask_uninf_cluster].mean()) if mask_uninf_cluster.any() else 0.0
        )

        # Objective: distance * GT separation
        score = centroid_distance * gt_sep_score

        return {
            "score": score,
            "infected_cluster": infected_cluster,
            "uninfected_cluster": uninfected_cluster,
            "gt_sep_score": gt_sep_score,
            "silhouette_score": sil,
            "centroid_distance": centroid_distance,
            "frac_inf_infected_cluster": frac_inf_infected_cluster,
            "frac_inf_uninfected_cluster": frac_inf_uninfected_cluster,
        }

    # ------------------------------------------------------------------
    # Helper: UMAP with hyperparameter search
    # ------------------------------------------------------------------
    def _search_umap(X_scaled, y_orig, gt_uninf, gt_inf, settings_local):
        if umap is None:
            raise RuntimeError("umap-learn is not installed.")

        random_state = int(settings_local.get("infection_pca_random_state", 0))
        do_search = bool(settings_local.get("infection_pca_umap_search", True))

        # No search: single run with configured/default params
        if not do_search:
            n_neighbors = int(settings_local.get("infection_pca_umap_n_neighbors", 15))
            min_dist = float(settings_local.get("infection_pca_umap_min_dist", 0.1))
            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
            coords = reducer.fit_transform(X_scaled)
            kmeans = KMeans(
                n_clusters=2, random_state=random_state, n_init="auto"
            )
            cluster_labels = kmeans.fit_predict(coords)
            stats = _evaluate_embedding(coords, cluster_labels, y_orig, gt_uninf, gt_inf)
            return coords, cluster_labels, stats, {"n_neighbors": n_neighbors, "min_dist": min_dist}

        # With search: small grid over n_neighbors and min_dist
        nn_grid = settings_local.get(
            "infection_pca_umap_n_neighbors_grid", [5, 10, 15, 30]
        )
        md_grid = settings_local.get(
            "infection_pca_umap_min_dist_grid", [0.0, 0.05, 0.1, 0.3]
        )

        best = None
        for nn in nn_grid:
            for md in md_grid:
                try:
                    reducer = umap.UMAP(
                        n_components=2,
                        random_state=random_state,
                        n_neighbors=int(nn),
                        min_dist=float(md),
                    )
                    coords = reducer.fit_transform(X_scaled)
                    kmeans = KMeans(
                        n_clusters=2, random_state=random_state, n_init="auto"
                    )
                    cluster_labels = kmeans.fit_predict(coords)
                    stats = _evaluate_embedding(
                        coords, cluster_labels, y_orig, gt_uninf, gt_inf
                    )
                    if (best is None) or (stats["score"] > best["stats"]["score"]):
                        best = {
                            "coords": coords,
                            "cluster_labels": cluster_labels,
                            "stats": stats,
                            "params": {"n_neighbors": int(nn), "min_dist": float(md)},
                        }
                except Exception as e:
                    print(
                        f"[infection_intensity_qc:PCA] UMAP trial failed for "
                        f"n_neighbors={nn}, min_dist={md}: {e}"
                    )
                    continue

        if best is None:
            raise RuntimeError("UMAP hyperparameter search failed for all trials.")

        return best["coords"], best["cluster_labels"], best["stats"], best["params"]

    # ------------------------------------------------------------------
    # Helper: t-SNE with hyperparameter search
    # ------------------------------------------------------------------
    def _search_tsne(X_scaled, y_orig, gt_uninf, gt_inf, settings_local):
        if TSNE is None:
            raise RuntimeError("sklearn.manifold.TSNE is not available.")

        random_state = int(settings_local.get("infection_pca_random_state", 0))
        do_search = bool(settings_local.get("infection_pca_tsne_search", True))
        n_samples = X_scaled.shape[0]
        max_perp = max(5.0, (n_samples - 1) / 3.0)

        # Utility: run one t-SNE
        def _run_tsne(perplexity, learning_rate):
            tsne = TSNE(
                n_components=2,
                random_state=random_state,
                init="pca",
                learning_rate=learning_rate,
                perplexity=perplexity,
            )
            coords_ = tsne.fit_transform(X_scaled)
            kmeans_ = KMeans(
                n_clusters=2, random_state=random_state, n_init="auto"
            )
            cluster_labels_ = kmeans_.fit_predict(coords_)
            stats_ = _evaluate_embedding(
                coords_, cluster_labels_, y_orig, gt_uninf, gt_inf
            )
            return coords_, cluster_labels_, stats_

        # No search: single run with configured/default params
        if not do_search:
            base_perp = float(settings_local.get("infection_pca_tsne_perplexity", 30.0))
            perplexity = min(base_perp, max_perp)
            if perplexity <= 0:
                perplexity = max_perp
            coords, cluster_labels, stats = _run_tsne(perplexity, learning_rate="auto")
            return coords, cluster_labels, stats, {"perplexity": perplexity, "learning_rate": "auto"}

        # With search: grid over perplexity and learning_rate
        perp_grid = settings_local.get(
            "infection_pca_tsne_perplexity_grid", [15.0, 30.0, 45.0]
        )
        lr_grid = settings_local.get(
            "infection_pca_tsne_learning_rate_grid", [200.0, 500.0]
        )
        perp_candidates = [
            float(p) for p in perp_grid if float(p) < max_perp and float(p) > 0
        ]
        if not perp_candidates:
            perp_candidates = [min(30.0, max_perp)]

        best = None
        for perp in perp_candidates:
            for lr in lr_grid:
                try:
                    coords, cluster_labels, stats = _run_tsne(perp, float(lr))
                    if (best is None) or (stats["score"] > best["stats"]["score"]):
                        best = {
                            "coords": coords,
                            "cluster_labels": cluster_labels,
                            "stats": stats,
                            "params": {"perplexity": float(perp), "learning_rate": float(lr)},
                        }
                except Exception as e:
                    print(
                        f"[infection_intensity_qc:PCA] t-SNE trial failed for "
                        f"perplexity={perp}, learning_rate={lr}: {e}"
                    )
                    continue

        if best is None:
            raise RuntimeError("t-SNE hyperparameter search failed for all trials.")

        return best["coords"], best["cluster_labels"], best["stats"], best["params"]

    # ------------------------------------------------------------------
    # Main body
    # ------------------------------------------------------------------
    if all_df.empty:
        print("[infection_intensity_qc:PCA] all_df is empty; skipping embedding QC.")
        return all_df, infection_col

    if infection_col not in all_df.columns:
        print(
            f"[infection_intensity_qc:PCA] infection_col {infection_col!r} missing; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    mode = str(settings.get("infection_intensity_mode", "relabel")).lower()
    if mode not in {"relabel", "remove"}:
        print(
            f"[infection_intensity_qc:PCA] Unsupported mode={mode!r}; "
            "expected 'relabel' or 'remove'. Skipping embedding QC."
        )
        return all_df, infection_col

    # ------------------------
    # 🔴 Key change is here 🔴
    # ------------------------
    # Use infection_intensity_strategy to define embedding method
    strategy = str(settings.get("infection_intensity_strategy", "pca")).lower()
    if strategy in {"pca", "umap", "tsne"}:
        embed_method = strategy
    else:
        embed_method = "pca"

    # keep settings in sync so downstream code can use this if needed
    settings["infection_pca_method"] = embed_method

    if embed_method not in {"pca", "umap", "tsne"}:
        embed_method = "pca"

    key_cols = ["plateID", "wellID", "fieldID", "cellID"]
    for col in key_cols:
        if col not in all_df.columns:
            raise KeyError(
                f"[infection_intensity_qc:PCA] Required column {col!r} not in all_df."
            )

    # Drop any existing adjusted_infected to avoid _x/_y columns on merge
    cols_to_drop = [
        c
        for c in all_df.columns
        if c == "adjusted_infected" or c.startswith("adjusted_infected_")
    ]
    if cols_to_drop:
        all_df = all_df.drop(columns=cols_to_drop)

    # ------------------------------------------------------------------
    # Build per-cell feature table
    # ------------------------------------------------------------------
    numeric_cols = [
        c
        for c in all_df.columns
        if c.startswith("cell_")
        and c not in {"cellID"}
        and pd.api.types.is_numeric_dtype(all_df[c])
    ]
    if not numeric_cols:
        print(
            "[infection_intensity_qc:PCA] No numeric cell_* features found; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    cols_for_group = key_cols + numeric_cols + [infection_col]
    tmp = all_df[cols_for_group].copy()
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)

    group = tmp.groupby(key_cols, observed=True)
    cell_level = group[numeric_cols].median(numeric_only=True).reset_index()

    # any cell that was ever infected in the time series is treated as infected
    inf_any = group[infection_col].max().reset_index()
    cell_level = cell_level.merge(inf_any, on=key_cols, how="left", suffixes=("", "_y"))

    if infection_col not in cell_level.columns:
        for cand in (f"{infection_col}_y", f"{infection_col}_x"):
            if cand in cell_level.columns:
                cell_level[infection_col] = cell_level[cand]
                break

    if infection_col not in cell_level.columns:
        print(
            f"[infection_intensity_qc:PCA] Could not recover infection_col={infection_col!r} "
            "after aggregation; skipping embedding QC."
        )
        return all_df, infection_col

    cell_level[infection_col] = cell_level[infection_col].fillna(0).astype(bool)

    # ------------------------------------------------------------------
    # Decide pathogen-channel intensity column (needed for ground truth)
    # ------------------------------------------------------------------
    intensity_col = None
    if pathogen_chan is not None:
        cand_int = [
            f"cell_p95_intensity_ch{pathogen_chan}",
            f"cell_max_intensity_ch{pathogen_chan}",
            f"cell_mean_intensity_ch{pathogen_chan}",
        ]
        for c in cand_int:
            if c in cell_level.columns:
                intensity_col = c
                break

    if intensity_col is None:
        print(
            "[infection_intensity_qc:PCA] No pathogen-channel cell_* intensity column "
            "found; skipping embedding QC."
        )
        return all_df, infection_col

    # ------------------------------------------------------------------
    # Select morphology + pathogen-channel features
    #   - morphology: cell_* columns without 'ch' (no per-channel intensity)
    #   - pathogen:   cell_* columns that mention ch{pathogen_chan}
    # ------------------------------------------------------------------
    morph_cols = [
        c
        for c in numeric_cols
        if c.startswith("cell_") and ("ch" not in c.lower())
    ]
    path_cols = [
        c
        for c in numeric_cols
        if c.startswith("cell_") and f"ch{pathogen_chan}" in c.lower()
    ]

    feature_cols = sorted(set(morph_cols + path_cols))
    if intensity_col not in feature_cols and intensity_col in cell_level.columns:
        feature_cols.append(intensity_col)

    # Drop degenerate features
    clean_feature_cols = []
    for c in feature_cols:
        s = cell_level[c]
        if s.notna().sum() < 10:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        clean_feature_cols.append(c)
    feature_cols = clean_feature_cols

    if not feature_cols:
        print(
            "[infection_intensity_qc:PCA] No usable morphology + pathogen features; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    # ------------------------------------------------------------------
    # Prepare feature matrix + ground-truth subsets
    # ------------------------------------------------------------------
    # Optional log1p transform on intensity-like features to sharpen structure
    log_intensity = bool(settings.get("infection_pca_log_intensity", True))
    cell_for_X = cell_level.copy()
    if log_intensity:
        for c in feature_cols:
            cl = c.lower()
            if ("intensity" in cl) or ("p75" in cl) or ("p95" in cl) or ("max" in cl):
                vals = cell_for_X[c].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                if finite.any() and np.nanmin(vals[finite]) >= 0:
                    vals[finite] = np.log1p(vals[finite])
                    cell_for_X[c] = vals

    X = cell_for_X[feature_cols].to_numpy(dtype=float)
    y_orig = cell_level[infection_col].astype(bool).to_numpy()

    # Remove rows with all NaNs
    finite_counts = np.isfinite(X).sum(axis=1)
    mask_rows = finite_counts > 0
    if not mask_rows.any():
        print(
            "[infection_intensity_qc:PCA] No rows with finite features; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    X = X[mask_rows]
    cell_level = cell_level.loc[mask_rows].reset_index(drop=True)
    y_orig = y_orig[mask_rows]

    # Median imputation per feature
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        if not m.any():
            X[:, j] = 0.0
        else:
            med = np.nanmedian(col[m])
            col[~m] = med
            X[:, j] = col

    if X.shape[0] < 10:
        print(
            "[infection_intensity_qc:PCA] Fewer than 10 cells after filtering; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    # Optional subsampling for speed
    max_cells = int(settings.get("infection_pca_max_cells", 50000))
    if X.shape[0] > max_cells:
        rng = np.random.default_rng(0)
        idx = rng.choice(np.arange(X.shape[0]), size=max_cells, replace=False)
        X = X[idx]
        cell_level = cell_level.iloc[idx].reset_index(drop=True)
        y_orig = y_orig[idx]

    # ------------------------------------------------------------------
    # Build intensity-based ground-truth subsets
    # ------------------------------------------------------------------
    intens = cell_level[intensity_col].to_numpy(dtype=float)
    mask_finite_int = np.isfinite(intens)
    intens = intens[mask_finite_int]
    y_int = y_orig[mask_finite_int]

    if intens.size < 40 or np.sum(y_int) < 10 or np.sum(~y_int) < 10:
        print(
            "[infection_intensity_qc:PCA] Not enough cells with finite intensity in "
            "both infected/uninfected for ground-truth definition; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    inf_vals = intens[y_int]
    uninf_vals = intens[~y_int]

    if inf_vals.size < 10 or uninf_vals.size < 10:
        print(
            "[infection_intensity_qc:PCA] Too few intensity values per class; "
            "skipping embedding QC."
        )
        return all_df, infection_col

    thr_uninf = float(np.nanpercentile(uninf_vals, 25.0))
    thr_inf = float(np.nanpercentile(inf_vals, 75.0))

    # Boolean masks in the full (post-subsample) cell_level
    intens_full = cell_level[intensity_col].to_numpy(dtype=float)
    mask_finite_full = np.isfinite(intens_full)
    gt_uninf = mask_finite_full & (~y_orig) & (intens_full <= thr_uninf)
    gt_inf = mask_finite_full & (y_orig) & (intens_full >= thr_inf)

    n_gt_uninf = int(gt_uninf.sum())
    n_gt_inf = int(gt_inf.sum())
    print(
        "[infection_intensity_qc:PCA] Ground-truth sets: "
        f"uninfected_gt={n_gt_uninf}, infected_gt={n_gt_inf} "
        f"(thr_uninf={thr_uninf:.3f}, thr_inf={thr_inf:.3f})."
    )

    if n_gt_uninf < 10 or n_gt_inf < 10:
        print(
            "[infection_intensity_qc:PCA] Very small ground-truth subsets; "
            "embedding QC may be unstable."
        )

    # ------------------------------------------------------------------
    # Embedding (PCA / UMAP / t-SNE) with optional hyperparameter search
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional: up-weight pathogen-channel features to emphasize infection signal
    path_weight = float(settings.get("infection_pca_pathogen_weight", 1.0))
    if path_weight != 1.0 and path_cols:
        path_idx = [feature_cols.index(c) for c in feature_cols if c in path_cols]
        if path_idx:
            X_scaled[:, path_idx] *= path_weight

    random_state = int(settings.get("infection_pca_random_state", 0))
    method_label = "PCA"
    embedding_params = {}
    coords = None
    cluster_labels = None
    eval_stats = None

    if embed_method == "umap" and umap is not None:
        coords, cluster_labels, eval_stats, embedding_params = _search_umap(
            X_scaled, y_orig, gt_uninf, gt_inf, settings
        )
        method_label = "UMAP"
        print(
            "[infection_intensity_qc:PCA] UMAP best params: "
            f"{embedding_params}, score={eval_stats['score']:.4f}"
        )

    elif embed_method == "tsne" and TSNE is not None:
        coords, cluster_labels, eval_stats, embedding_params = _search_tsne(
            X_scaled, y_orig, gt_uninf, gt_inf, settings
        )
        method_label = "t-SNE"
        print(
            "[infection_intensity_qc:PCA] t-SNE best params: "
            f"{embedding_params}, score={eval_stats['score']:.4f}"
        )

    else:
        # PCA (no hyperparameter search, but benefits from log/weighting above)
        if embed_method in {"umap", "tsne"}:
            print(
                f"[infection_intensity_qc:PCA] Requested method={embed_method!r} "
                "not available; falling back to PCA."
            )
        pca = PCA(
            n_components=2,
            random_state=random_state,
        )
        coords = pca.fit_transform(X_scaled)
        kmeans = KMeans(
            n_clusters=2,
            random_state=random_state,
            n_init="auto",
        )
        cluster_labels = kmeans.fit_predict(coords)
        eval_stats = _evaluate_embedding(coords, cluster_labels, y_orig, gt_uninf, gt_inf)
        method_label = "PCA"
        embedding_params = {}

    # Unpack evaluation stats
    infected_cluster = int(eval_stats["infected_cluster"])
    uninfected_cluster = int(eval_stats["uninfected_cluster"])
    gt_sep_score = float(eval_stats["gt_sep_score"])
    sil_score = eval_stats["silhouette_score"]
    centroid_distance = float(eval_stats["centroid_distance"])
    frac_inf_infected_cluster = float(eval_stats["frac_inf_infected_cluster"])
    frac_inf_uninfected_cluster = float(eval_stats["frac_inf_uninfected_cluster"])

    min_gt_sep = float(settings.get("infection_pca_min_gt_separation", 0.2))
    min_sil = float(settings.get("infection_pca_min_silhouette", 0.05))

    if gt_sep_score < min_gt_sep or (sil_score is not None and sil_score < min_sil):
        print(
            "[infection_intensity_qc:PCA] WARNING: weak cluster structure "
            f"(gt_sep_score={gt_sep_score:.3f}, silhouette={sil_score}). "
            "To improve separation you can try:\n"
            "  - Tightening infection ground-truth thresholds (e.g. more extreme percentiles)\n"
            "  - Reducing noise features, especially non-morphology/non-pathogen\n"
            "  - Adjusting UMAP/t-SNE grids to favor more local structure\n"
            "  - Increasing infection_pca_pathogen_weight to emphasize pathogen features."
        )

    print(
        "[infection_intensity_qc:PCA] Cluster infected fractions (original labels): "
        f"infected_cluster={frac_inf_infected_cluster:.3f}, "
        f"uninfected_cluster={frac_inf_uninfected_cluster:.3f}, "
        f"centroid_distance={centroid_distance:.3f}, gt_sep={gt_sep_score:.3f}."
    )

    # ------------------------------------------------------------------
    # Build cluster-based infection call
    # ------------------------------------------------------------------
    cluster_infected = (cluster_labels == infected_cluster)

    removed_ids = set()

    if mode == "relabel":
        # labels follow cluster
        adjusted = cluster_infected.astype(bool)
        n_changed = int((adjusted != y_orig).sum())
        print(
            "[infection_intensity_qc:PCA] Relabel mode: adjusted infection labels for "
            f"{n_changed} cells based on {method_label} clusters."
        )
        cell_level["adjusted_infected"] = adjusted.astype(bool)

    else:  # mode == "remove"
        consistent = cluster_infected == y_orig
        to_remove = ~consistent
        if to_remove.any():
            removed = cell_level.loc[to_remove, key_cols]
            removed_ids = {
                (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
                for _, r in removed.iterrows()
            }
            cell_level = cell_level.loc[consistent].copy()
            cluster_infected = cluster_infected[consistent]
            y_orig = y_orig[consistent]
            coords = coords[consistent]
            cluster_labels = cluster_labels[consistent]
            print(
                "[infection_intensity_qc:PCA] Remove mode: removed "
                f"{len(removed_ids)} cells with cluster vs label disagreement."
            )
        cell_level["adjusted_infected"] = y_orig.astype(bool)

    # ------------------------------------------------------------------
    # Map adjusted infection back to all_df (frame level)
    # ------------------------------------------------------------------
    # Ensure key dtypes match
    for col in key_cols:
        all_df[col] = all_df[col].astype(cell_level[col].dtype)

    all_df = all_df.merge(
        cell_level[key_cols + ["adjusted_infected"]],
        on=key_cols,
        how="left",
        validate="m:1",
    )

    if removed_ids:
        mask_drop = all_df.apply(
            lambda r: (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
            in removed_ids,
            axis=1,
        )
        all_df = all_df.loc[~mask_drop].reset_index(drop=True)

    # Any rows that did not get an adjusted label inherit the original
    mask_missing = all_df["adjusted_infected"].isna()
    if mask_missing.any():
        all_df.loc[mask_missing, "adjusted_infected"] = (
            all_df.loc[mask_missing, infection_col].astype(bool)
        )
    all_df["adjusted_infected"] = all_df["adjusted_infected"].astype(bool)

    infection_col = "adjusted_infected"

    # ------------------------------------------------------------------
    # Store payload for combined panel
    # ------------------------------------------------------------------
    settings["infection_pca_data"] = {
        "coords": coords,
        "labels": cell_level["adjusted_infected"].astype(bool).to_numpy(),
        "cluster_labels": cluster_labels,
        "method_label": method_label,  # <- used for axis titles / panel labels
        "infected_cluster": int(infected_cluster),
        "uninfected_cluster": int(uninfected_cluster),
        "initial_infected_frac_infected_cluster": frac_inf_infected_cluster,
        "initial_infected_frac_uninfected_cluster": frac_inf_uninfected_cluster,
        "gt_sep_score": gt_sep_score,
        "silhouette_score": sil_score,
        "centroid_distance": centroid_distance,
        "embedding_params": embedding_params,
    }

    settings["infection_intensity_qc_panel_type"] = "pca"
    settings["infection_intensity_qc_panel_path"] = None

    # ------------------------------------------------------------------
    # Optional debug plot
    # ------------------------------------------------------------------
    try:
        if motility_dir is not None:
            import matplotlib.pyplot as plt

            os.makedirs(motility_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(4, 4))

            # Masks for remaining cells
            mask_uninf_cluster_plot = cluster_labels == uninfected_cluster
            mask_inf_cluster_plot = cluster_labels == infected_cluster

            # Plot clusters with transparency and filled markers
            ax.scatter(
                coords[mask_uninf_cluster_plot, 0],
                coords[mask_uninf_cluster_plot, 1],
                s=2,
                alpha=0.6,
                color="green",
                label=(
                    f"Uninfected cluster "
                    f"({frac_inf_uninfected_cluster*100:.1f}% infected at start)"
                ),
            )
            ax.scatter(
                coords[mask_inf_cluster_plot, 0],
                coords[mask_inf_cluster_plot, 1],
                s=2,
                alpha=0.6,
                color="red",
                label=(
                    f"Infected cluster "
                    f"({frac_inf_infected_cluster*100:.1f}% infected at start)"
                ),
            )

            # Axis titles and main title reflect method
            ax.set_xlabel(f"{method_label} 1")
            ax.set_ylabel(f"{method_label} 2")

            title = f"{method_label} infection QC"
            if embedding_params:
                param_str = ", ".join(
                    f"{k}={v}" for k, v in embedding_params.items()
                )
                title += f"\n{param_str}"
            if sil_score is not None:
                title += f"\nGT-sep={gt_sep_score:.2f}, sil={sil_score:.2f}"
            else:
                title += f"\nGT-sep={gt_sep_score:.2f}"

            ax.set_title(title)
            ax.legend(fontsize=7, loc="best")

            out_png = os.path.join(
                motility_dir, f"infection_{embed_method}_qc_embedding.png"
            )
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"[infection_intensity_qc:PCA] Failed to save embedding QC plot: {e}")

    return all_df, infection_col


def _apply_infection_intensity_qc(
    all_df,
    settings,
    infection_col,
    pathogen_chan,
    motility_dir,
):
    """
    Dispatch to different infection QC strategies based on
    settings['infection_intensity_strategy'] and
    settings['infection_intensity_qc_scope'].

    Strategies
    ----------
        'histogram' / 'hist' / 'histagram'
            1D intensity histogram thresholding

        'pca' / 'umap' / 'tsne'
            PCA/UMAP/TSNE + clustering (via _infection_qc_pca_clustering)

        'xgboost' / 'xgb'
            Supervised XGBoost classifier on extreme intensities

    Scope
    -----
        settings['infection_intensity_qc_scope']:

        'combined' (default)
            Run QC once on all_df (old behaviour).

        'plate'
            Run QC separately per plateID.

        'well'
            Run QC separately per (plateID, wellID).

        'none' / 'off'
            Skip QC entirely and return original labels.

    If settings['infection_intensity_qc'] is False or pathogen_chan is None,
    this function is a no-op and returns the input as-is.

    Returns
    -------
    all_df : DataFrame
        Frame-level measurements with possibly updated 'infection_col'.
    infection_col : str
        Name of the column in all_df that encodes the (possibly adjusted)
        infection status.
    """
    import os
    import numpy as np
    import pandas as pd

    # Reset QC payloads by default; strategy helpers will overwrite if used
    settings["infection_hist_data"] = None
    settings["infection_pca_data"] = None
    settings["infection_xgb_importance"] = None
    settings["infection_intensity_qc_panel_type"] = None
    settings["infection_intensity_qc_panel_path"] = None

    # If QC is disabled or there is no pathogen channel, do nothing.
    infection_intensity_qc = bool(settings.get("infection_intensity_qc", False))
    if (not infection_intensity_qc) or (pathogen_chan is None):
        print("[infection_intensity_qc] QC disabled or no pathogen channel; skipping.")
        return all_df, infection_col

    # Make sure output directory exists for plots
    os.makedirs(motility_dir, exist_ok=True)

    strategy = str(settings.get("infection_intensity_strategy", "histogram")).lower()

    # Strategy → QC helper
    if strategy in {"hist", "histogram", "histagram"}:
        qc_func = _infection_qc_histogram
    elif strategy in {"xgboost", "xgb"}:
        qc_func = _infection_qc_xgboost
    elif strategy in {"pca", "umap", "tsne"}:
        qc_func = _infection_qc_pca_clustering
    else:
        print(
            "[infection_intensity_qc] Unknown strategy "
            f"{strategy!r}; falling back to 'histogram'."
        )
        qc_func = _infection_qc_histogram

    # Scope: combined (default), per-plate, per-well, or none
    scope = str(settings.get("infection_intensity_qc_scope", "combined") or "combined").lower()

    if scope in {"none", "off"}:
        # Explicit request to skip QC
        return all_df, infection_col

    if scope in {"combined", "global", "all"}:
        # --- old behaviour: single global run ---
        local_settings = dict(settings)  # shallow copy for QC helper
        df_qc, inf_col_out = qc_func(
            all_df=all_df,
            settings=local_settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

        # propagate QC payloads back
        settings["infection_hist_data"] = local_settings.get("infection_hist_data")
        settings["infection_pca_data"] = local_settings.get("infection_pca_data")
        settings["infection_xgb_importance"] = local_settings.get("infection_xgb_importance")
        settings["infection_intensity_qc_panel_type"] = local_settings.get(
            "infection_intensity_qc_panel_type"
        )
        settings["infection_intensity_qc_panel_path"] = local_settings.get(
            "infection_intensity_qc_panel_path"
        )

        # normalise adjusted_infected if present
        if "adjusted_infected" in df_qc.columns:
            if df_qc["adjusted_infected"].isna().any():
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].fillna(
                    df_qc[infection_col]
                )
            try:
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].astype(int)
            except Exception:
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].astype(bool)
            inf_col_out = "adjusted_infected"

        return df_qc, inf_col_out

    # Grouped scopes
    if scope in {"plate", "per_plate", "plateid"}:
        group_cols = ["plateID"]
    elif scope in {"well", "per_well"}:
        group_cols = ["plateID", "wellID"]
    else:
        print(
            f"[_apply_infection_intensity_qc] Unknown scope={scope!r}; "
            "using 'combined' behaviour."
        )
        local_settings = dict(settings)
        df_qc, inf_col_out = qc_func(
            all_df=all_df,
            settings=local_settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

        settings["infection_hist_data"] = local_settings.get("infection_hist_data")
        settings["infection_pca_data"] = local_settings.get("infection_pca_data")
        settings["infection_xgb_importance"] = local_settings.get("infection_xgb_importance")
        settings["infection_intensity_qc_panel_type"] = local_settings.get(
            "infection_intensity_qc_panel_type"
        )
        settings["infection_intensity_qc_panel_path"] = local_settings.get(
            "infection_intensity_qc_panel_path"
        )

        if "adjusted_infected" in df_qc.columns:
            if df_qc["adjusted_infected"].isna().any():
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].fillna(
                    df_qc[infection_col]
                )
            try:
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].astype(int)
            except Exception:
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].astype(bool)
            inf_col_out = "adjusted_infected"

        return df_qc, inf_col_out

    # If requested group columns are missing, fall back to combined
    if not set(group_cols).issubset(all_df.columns):
        print(
            f"[_apply_infection_intensity_qc] Requested scope={scope!r} but "
            f"missing grouping columns {group_cols}; falling back to combined QC."
        )
        local_settings = dict(settings)
        df_qc, inf_col_out = qc_func(
            all_df=all_df,
            settings=local_settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

        settings["infection_hist_data"] = local_settings.get("infection_hist_data")
        settings["infection_pca_data"] = local_settings.get("infection_pca_data")
        settings["infection_xgb_importance"] = local_settings.get("infection_xgb_importance")
        settings["infection_intensity_qc_panel_type"] = local_settings.get(
            "infection_intensity_qc_panel_type"
        )
        settings["infection_intensity_qc_panel_path"] = local_settings.get(
            "infection_intensity_qc_panel_path"
        )

        if "adjusted_infected" in df_qc.columns:
            if df_qc["adjusted_infected"].isna().any():
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].fillna(
                    df_qc[infection_col]
                )
            try:
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].astype(int)
            except Exception:
                df_qc["adjusted_infected"] = df_qc["adjusted_infected"].astype(bool)
            inf_col_out = "adjusted_infected"

        return df_qc, inf_col_out

    # ------------------------------------------------------------------
    # Group-wise QC: per-plate or per-well
    # ------------------------------------------------------------------
    parts = []
    any_adjusted = False
    first_payload_settings = None

    for g_key, df_group in all_df.groupby(group_cols, sort=False):
        if df_group.empty:
            continue

        local_settings = dict(settings)
        df_group_qc, inf_col_group = qc_func(
            all_df=df_group,
            settings=local_settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

        if "adjusted_infected" in df_group_qc.columns and df_group_qc["adjusted_infected"].notna().any():
            any_adjusted = True

        if first_payload_settings is None:
            first_payload_settings = local_settings

        parts.append(df_group_qc)

    if not parts:
        # nothing processed → return original
        return all_df, infection_col

    all_df_qc = pd.concat(parts, axis=0, ignore_index=True)

    # Use QC payloads (histogram/PCA/XGB) from the first processed group
    if first_payload_settings is not None:
        settings["infection_hist_data"] = first_payload_settings.get("infection_hist_data")
        settings["infection_pca_data"] = first_payload_settings.get("infection_pca_data")
        settings["infection_xgb_importance"] = first_payload_settings.get("infection_xgb_importance")
        settings["infection_intensity_qc_panel_type"] = first_payload_settings.get(
            "infection_intensity_qc_panel_type"
        )
        settings["infection_intensity_qc_panel_path"] = first_payload_settings.get(
            "infection_intensity_qc_panel_path"
        )

    if any_adjusted and "adjusted_infected" in all_df_qc.columns:
        if all_df_qc["adjusted_infected"].isna().any():
            all_df_qc["adjusted_infected"] = all_df_qc["adjusted_infected"].fillna(
                all_df_qc[infection_col]
            )
        try:
            all_df_qc["adjusted_infected"] = all_df_qc["adjusted_infected"].astype(int)
        except Exception:
            all_df_qc["adjusted_infected"] = all_df_qc["adjusted_infected"].astype(bool)
        infection_col_out = "adjusted_infected"
    else:
        infection_col_out = infection_col

    return all_df_qc, infection_col_out


def _compute_velocities_and_well_summary(
    all_df,
    settings,
    infection_col,
    pixels_per_um,
    seconds_per_frame,
):
    """
    Compute per-track velocities, straightness and per-well motility summary.

    Returns
    -------
    track_df : DataFrame
    per_well_tracks : dict[(plateID, wellID) -> list of track dicts]
    well_summary_df : DataFrame
    vel_unit : str
    """
    import numpy as np
    import pandas as pd

    y_col = "cell_centroid-0"
    x_col = "cell_centroid-1"

    track_df = pd.DataFrame()
    well_summary_df = pd.DataFrame()
    per_well_tracks = {}
    vel_unit = "px/frame"

    if y_col not in all_df.columns or x_col not in all_df.columns:
        print(
            "[summarise_tracks_from_merged] Centroid columns missing; "
            "motility summary and plots will not be generated."
        )
        return track_df, per_well_tracks, well_summary_df, vel_unit

    gtracks = all_df.groupby(["plateID", "wellID", "fieldID", "cellID"])
    track_records = []

    for (plateID, wellID, fieldID, cellID), g in gtracks:
        g = g.sort_values("frame")
        x_px = g[x_col].to_numpy(dtype=float)
        y_px = g[y_col].to_numpy(dtype=float)
        if len(x_px) < 2:
            continue

        dx = np.diff(x_px)
        dy = np.diff(y_px)
        d = np.hypot(dx, dy)
        if d.size == 0 or not np.isfinite(d).any():
            continue

        v_px = float(np.nanmean(d))
        path_length = float(np.nansum(d))
        net_dx = float(x_px[-1] - x_px[0])
        net_dy = float(y_px[-1] - y_px[0])
        net_disp = float(np.hypot(net_dx, net_dy))
        if path_length > 0 and np.isfinite(net_disp):
            straightness = net_disp / path_length
        else:
            straightness = np.nan

        infected_track = bool(g[infection_col].any())

        track_records.append(
            {
                "plateID": plateID,
                "wellID": wellID,
                "fieldID": fieldID,
                "cellID": cellID,
                "infected": infected_track,
                "v_px_per_frame": v_px,
                "straightness": straightness,
            }
        )

        key_well = (plateID, wellID)
        per_well_tracks.setdefault(key_well, []).append(
            {
                "plateID": plateID,
                "wellID": wellID,
                "fieldID": fieldID,
                "cellID": cellID,
                "infected": infected_track,
                "x_px": x_px,
                "y_px": y_px,
                "v_px_per_frame": v_px,
                "straightness": straightness,
            }
        )

    if not track_records:
        print(
            "[summarise_tracks_from_merged] No tracks with >=2 frames; "
            "skipping motility summary and plots."
        )
        return track_df, per_well_tracks, well_summary_df, vel_unit

    track_df = pd.DataFrame(track_records)

    use_physical_units = (
        pixels_per_um is not None and seconds_per_frame is not None
    )
    if use_physical_units:
        pixels_per_um = float(pixels_per_um)
        seconds_per_frame = float(seconds_per_frame)
        factor = (1.0 / pixels_per_um) * (60.0 / seconds_per_frame)
        vel_unit = "µm/min"
    else:
        factor = 1.0
        vel_unit = "px/frame"

    track_df["velocity"] = track_df["v_px_per_frame"] * factor
    track_df["velocity_unit"] = vel_unit

    # Straightness-based artifact detection / filtering
    if "straightness" in track_df.columns:
        straightness_threshold = float(
            settings.get("straightness_threshold", 0.95)
        )
        straightness_filter = bool(settings.get("straightness_filter", False))
        n_tracks_before = track_df.shape[0]
        n_high = int((track_df["straightness"] >= straightness_threshold).sum())
        print(
            "[summarise_tracks_from_merged] Straightness metric: "
            f"{n_high} of {n_tracks_before} tracks have straightness "
            f">= {straightness_threshold:.2f} "
            "(net displacement / path length)."
        )

        if straightness_filter and n_high > 0:
            drop_mask = track_df["straightness"] >= straightness_threshold
            dropped = track_df.loc[
                drop_mask, ["plateID", "wellID", "fieldID", "cellID"]
            ].copy()
            drop_keys = set(
                zip(
                    dropped["plateID"],
                    dropped["wellID"],
                    dropped["fieldID"],
                    dropped["cellID"],
                )
            )

            track_df = track_df.loc[~drop_mask].reset_index(drop=True)
            print(
                "[summarise_tracks_from_merged] Straightness filter "
                f"removed {n_high} overly straight tracks "
                f"(threshold={straightness_threshold:.2f})."
            )

            # Filter per_well_tracks accordingly
            for well_key, track_list in list(per_well_tracks.items()):
                filtered_list = [
                    tr
                    for tr in track_list
                    if (
                        tr["plateID"],
                        tr["wellID"],
                        tr["fieldID"],
                        tr["cellID"],
                    )
                    not in drop_keys
                ]
                if filtered_list:
                    per_well_tracks[well_key] = filtered_list
                else:
                    del per_well_tracks[well_key]

    if track_df.empty:
        print(
            "[summarise_tracks_from_merged] No tracks left after "
            "straightness filtering; skipping motility summary and plots."
        )
        return track_df, per_well_tracks, well_summary_df, vel_unit

    well_records = []
    for (plateID, wellID), g in track_df.groupby(["plateID", "wellID"]):
        n_tracks_well = len(g)
        n_inf_well = int(g["infected"].sum())
        n_uninf_well = n_tracks_well - n_inf_well

        mean_all = float(g["velocity"].mean()) if n_tracks_well > 0 else np.nan
        mean_inf = (
            float(g.loc[g["infected"], "velocity"].mean())
            if n_inf_well > 0
            else np.nan
        )
        mean_uninf = (
            float(g.loc[~g["infected"], "velocity"].mean())
            if n_uninf_well > 0
            else np.nan
        )

        well_records.append(
            dict(
                plateID=plateID,
                wellID=wellID,
                n_tracks=n_tracks_well,
                n_infected_tracks=n_inf_well,
                n_uninfected_tracks=n_uninf_well,
                mean_velocity_all=mean_all,
                mean_velocity_infected=mean_inf,
                mean_velocity_uninfected=mean_uninf,
                velocity_unit=vel_unit,
            )
        )

    if well_records:
        well_summary_df = pd.DataFrame(well_records)

    print(
        "[summarise_tracks_from_merged] Computed per-track velocities "
        f"in units: {vel_unit}"
    )

    return track_df, per_well_tracks, well_summary_df, vel_unit


def _save_measurements_and_well_summary(
    all_df,
    well_summary_df,
    src,
    db_table_name,
):
    """
    Save per-frame measurements and well-level motility summary to SQLite.
    Returns (measurements_dir, db_path).
    """
    import os
    import sqlite3

    measurements_dir = os.path.join(src, "measurements")
    os.makedirs(measurements_dir, exist_ok=True)
    db_path = os.path.join(measurements_dir, "measurements.db")

    with sqlite3.connect(db_path) as conn:
        all_df.to_sql(db_table_name, conn, if_exists="replace", index=False)
        print(
            f"[summarise_tracks_from_merged] Saved measurements to "
            f"{db_path} (table='{db_table_name}')"
        )

        if not well_summary_df.empty:
            well_table_name = db_table_name + "_well_motility"
            well_summary_df.to_sql(
                well_table_name,
                conn,
                if_exists="replace",
                index=False,
            )
            print(
                f"[summarise_tracks_from_merged] Saved well-level motility "
                f"summary to {db_path} (table='{well_table_name}')"
            )
        else:
            print(
                "[summarise_tracks_from_merged] No well-level motility "
                "summary table was created."
            )

    return measurements_dir, db_path


def _feature_velocity_correlations(all_df, track_df, measurements_dir):
    """
    Correlate per-track velocity with median per-track features (all / infected / uninfected).
    Saves CSV to measurements_dir/velocity_feature_correlations.csv
    """
    import numpy as np
    import os
    import pandas as pd

    if track_df.empty:
        return

    try:
        group_cols = ["plateID", "wellID", "fieldID", "cellID"]

        numeric_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
        for col_rm in ("frame", "timeID", "cellID"):
            if col_rm in numeric_cols:
                numeric_cols.remove(col_rm)

        if not numeric_cols:
            print(
                "[summarise_tracks_from_merged] No numeric feature columns "
                "available for correlation analysis."
            )
            return

        agg_features = (
            all_df[group_cols + numeric_cols]
            .groupby(group_cols, dropna=False)
            .median()
            .reset_index()
        )

        track_features = track_df.merge(agg_features, on=group_cols, how="left")

        exclude_cols = set(
            group_cols
            + ["infected", "v_px_per_frame", "velocity", "velocity_unit"]
        )
        candidate_cols = [
            c
            for c in track_features.columns
            if c not in exclude_cols
            and np.issubdtype(track_features[c].dtype, np.number)
        ]

        if not candidate_cols:
            print(
                "[summarise_tracks_from_merged] No numeric feature columns "
                "available for correlation analysis."
            )
            return

        def _corr_subset(mask, label):
            sub = track_features.loc[mask, candidate_cols + ["velocity"]].copy()
            sub = sub[np.isfinite(sub["velocity"])]
            if sub.shape[0] < 5:
                print(
                    "[summarise_tracks_from_merged] "
                    f"Not enough tracks for correlation ({label})."
                )
                return None
            corr_series = sub.corr(method="pearson")["velocity"].drop("velocity")
            corr_df = (
                corr_series.rename("pearson_r")
                .to_frame()
                .reset_index()
                .rename(columns={"index": "feature"})
            )
            corr_df["n_tracks"] = sub.shape[0]
            corr_df["group"] = label
            return corr_df

        mask_all = np.isfinite(track_features["velocity"])
        results = []

        res_all = _corr_subset(mask_all, "all")
        if res_all is not None:
            results.append(res_all)

        mask_inf = track_features["infected"].astype(bool) & mask_all
        res_inf = _corr_subset(mask_inf, "infected")
        if res_inf is not None:
            results.append(res_inf)

        mask_uninf = (~track_features["infected"].astype(bool)) & mask_all
        res_uninf = _corr_subset(mask_uninf, "uninfected")
        if res_uninf is not None:
            results.append(res_uninf)

        if not results:
            return

        corr_all = pd.concat(results, ignore_index=True)
        corr_all["abs_pearson_r"] = corr_all["pearson_r"].abs()
        corr_all = corr_all.sort_values(
            ["group", "abs_pearson_r"], ascending=[True, False]
        )

        corr_out = os.path.join(measurements_dir, "velocity_feature_correlations.csv")
        corr_all.to_csv(corr_out, index=False)
        print(
            "[summarise_tracks_from_merged] Saved velocity–feature "
            f"correlations to {corr_out}"
        )

    except Exception as e:
        print(
            "[summarise_tracks_from_merged] Feature–velocity correlation "
            f"analysis failed with error: {e}"
        )


def _make_intensity_sanity_plots(all_df, infection_col, n_channels, motility_dir):
    """
    Per-channel intensity sanity-check plots (infected vs uninfected).
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    if all_df.empty:
        return

    keys = ["plateID", "wellID", "fieldID", "cellID"]
    os.makedirs(motility_dir, exist_ok=True)

    for ch in range(n_channels):
        col_int = f"cell_mean_intensity_ch{ch}"
        if col_int not in all_df.columns:
            continue

        cell_level_int = (
            all_df[keys + [col_int, infection_col]]
            .groupby(keys, dropna=False)
            .agg(
                {
                    col_int: "mean",
                    infection_col: "max",
                }
            )
            .reset_index()
        )
        cell_level_int = cell_level_int.replace([np.inf, -np.inf], np.nan)
        cell_level_int = cell_level_int.dropna(subset=[col_int])

        if cell_level_int.empty:
            print(
                f"[summarise_tracks_from_merged] No data for intensity "
                f"channel {ch}, skipping sanity plot."
            )
            continue

        mask_inf = cell_level_int[infection_col].astype(bool)
        vals_inf = cell_level_int.loc[mask_inf, col_int].to_numpy()
        vals_uninf = cell_level_int.loc[~mask_inf, col_int].to_numpy()

        mean_inf = float(np.nanmean(vals_inf)) if vals_inf.size else np.nan
        std_inf = (
            float(np.nanstd(vals_inf, ddof=1)) if vals_inf.size > 1 else np.nan
        )
        mean_uninf = float(np.nanmean(vals_uninf)) if vals_uninf.size else np.nan
        std_uninf = (
            float(np.nanstd(vals_uninf, ddof=1))
            if vals_uninf.size > 1
            else np.nan
        )

        x_pos = np.arange(2)
        heights = [mean_inf, mean_uninf]
        errors = [std_inf, std_uninf]

        fig_ch, ax_ch = plt.subplots(figsize=(4, 4))
        ax_ch.bar(
            x_pos,
            heights,
            yerr=errors,
            capsize=5,
            color=["red", "green"],
            alpha=0.7,
        )
        ax_ch.set_xticks(x_pos)
        ax_ch.set_xticklabels(["Infected", "Uninfected"])
        ax_ch.set_ylabel(f"Mean cell intensity (channel {ch})")
        ax_ch.set_title(f"Intensity vs infection – channel {ch}")
        ax_ch.set_ylim(bottom=0)
        plt.tight_layout()
        out_ch = os.path.join(
            motility_dir, f"intensity_channel{ch}_infected_vs_uninfected.png"
        )
        fig_ch.savefig(out_ch, dpi=300)
        plt.close(fig_ch)
        print(
            f"[summarise_tracks_from_merged] Saved intensity sanity plot "
            f"for channel {ch} to {out_ch}"
        )


def _make_motility_plots(
    track_df,
    per_well_tracks,
    well_summary_df,
    motility_dir,
    pixels_per_um,
    seconds_per_frame,
    vel_unit,
    settings,
):
    """
    Motility plots (combined + per-well) with compact text box.

    Axis control via settings:
        - motility_xlim / motility_ylim: applied to absolute-coordinate plots
        - motility_origin_xlim / motility_origin_ylim: applied to origin plots
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from matplotlib import patches

    if track_df.empty or not per_well_tracks:
        print(
            "[summarise_tracks_from_merged] No per-track velocities available; "
            "motility plots were not generated."
        )
        return

    def _fmt_vel(val):
        return "n/a" if not np.isfinite(val) else f"{val:.2f}"

    def _apply_axis_limits(ax, xlim, ylim):
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

    abs_xlim = settings.get("motility_xlim", None)
    abs_ylim = settings.get("motility_ylim", None)
    origin_xlim = settings.get("motility_origin_xlim", None)
    origin_ylim = settings.get("motility_origin_ylim", None)

    if pixels_per_um is not None:
        unit_line1 = f"1 µm = {float(pixels_per_um):.2f} px"
        coord_label_x = "x (µm)"
        coord_label_y = "y (µm)"
        coord_scale = 1.0 / float(pixels_per_um)
    else:
        unit_line1 = "1 µm = ? px"
        coord_label_x = "x (pixels)"
        coord_label_y = "y (pixels)"
        coord_scale = 1.0

    if seconds_per_frame is not None:
        unit_line2 = f"1 frame = {float(seconds_per_frame):g} s"
    else:
        unit_line2 = "1 frame = ? s"

    box_x0 = 0.64
    box_y0 = 0.69
    box_width = 0.30
    box_height = 0.23
    text_x = box_x0 + 0.02
    y_top = box_y0 + box_height - 0.03
    line_spacing = 0.07
    fontsize_main = 8
    fontsize_units = 7

    os.makedirs(motility_dir, exist_ok=True)

    # Combined plot over all wells
    fig_all, ax_all = plt.subplots(figsize=(6, 6))

    for tracks in per_well_tracks.values():
        for tr in tracks:
            x = tr["x_px"] * coord_scale
            y = tr["y_px"] * coord_scale
            infected_track = tr["infected"]
            color = "red" if infected_track else "green"
            ax_all.plot(x, y, color=color, alpha=0.2, linewidth=0.5)
            ax_all.scatter(x[-1], y[-1], color=color, s=5)

    vel_all = track_df["velocity"].to_numpy()
    vel_inf = track_df.loc[track_df["infected"], "velocity"].to_numpy()
    vel_uninf = track_df.loc[~track_df["infected"], "velocity"].to_numpy()

    mean_vel_all = float(np.nanmean(vel_all)) if vel_all.size else np.nan
    mean_vel_inf = float(np.nanmean(vel_inf)) if vel_inf.size else np.nan
    mean_vel_uninf = float(np.nanmean(vel_uninf)) if vel_uninf.size else np.nan

    print(
        "[summarise_tracks_from_merged] Velocity stats "
        f"({vel_unit}): all={mean_vel_all:.3f} "
        f"(n={vel_all.size} tracks with >=2 frames), "
        f"infected={mean_vel_inf:.3f} (n={vel_inf.size}), "
        f"uninfected={mean_vel_uninf:.3f} (n={vel_uninf.size})"
    )

    ax_all.set_aspect("equal", "box")
    ax_all.set_xlabel(coord_label_x)
    ax_all.set_ylabel(coord_label_y)
    _apply_axis_limits(ax_all, abs_xlim, abs_ylim)

    bbox_all = patches.FancyBboxPatch(
        (box_x0, box_y0),
        box_width,
        box_height,
        transform=ax_all.transAxes,
        facecolor="white",
        edgecolor="black",
        boxstyle="round,pad=0.02",
        alpha=0.8,
    )
    ax_all.add_patch(bbox_all)

    ax_all.text(
        text_x,
        y_top,
        f"Infected ({_fmt_vel(mean_vel_inf)} {vel_unit})",
        color="red",
        transform=ax_all.transAxes,
        fontsize=fontsize_main,
        va="top",
    )
    ax_all.text(
        text_x,
        y_top - line_spacing,
        f"Uninfected ({_fmt_vel(mean_vel_uninf)} {vel_unit})",
        color="green",
        transform=ax_all.transAxes,
        fontsize=fontsize_main,
        va="top",
    )
    ax_all.text(
        text_x,
        y_top - 2 * line_spacing,
        unit_line1,
        color="black",
        transform=ax_all.transAxes,
        fontsize=fontsize_units,
        va="top",
    )
    ax_all.text(
        text_x,
        y_top - 3 * line_spacing,
        unit_line2,
        color="black",
        transform=ax_all.transAxes,
        fontsize=fontsize_units,
        va="top",
    )

    plt.tight_layout()
    out_png_all = os.path.join(motility_dir, "motility_all_tracks.png")
    fig_all.savefig(out_png_all, dpi=300)
    plt.close(fig_all)
    print(
        f"[summarise_tracks_from_merged] Saved combined motility plot to "
        f"{out_png_all}"
    )

    # Per-well plots
    well_summary_map = {}
    if not well_summary_df.empty:
        for _, row in well_summary_df.iterrows():
            well_summary_map[(row["plateID"], row["wellID"])] = row

    for (plateID, wellID), tracks in per_well_tracks.items():
        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        has_infected = False
        has_uninfected = False

        for tr in tracks:
            x = tr["x_px"] * coord_scale
            y = tr["y_px"] * coord_scale
            infected_track = tr["infected"]
            color = "red" if infected_track else "green"
            if infected_track:
                has_infected = True
            else:
                has_uninfected = True
            ax_w.plot(x, y, color=color, alpha=0.2, linewidth=0.5)
            ax_w.scatter(x[-1], y[-1], color=color, s=5)

        ax_w.set_aspect("equal", "box")
        ax_w.set_xlabel(coord_label_x)
        ax_w.set_ylabel(coord_label_y)
        _apply_axis_limits(ax_w, abs_xlim, abs_ylim)

        mean_inf_w = np.nan
        mean_uninf_w = np.nan
        summary_row = well_summary_map.get((plateID, wellID))
        if summary_row is not None:
            mean_inf_w = summary_row["mean_velocity_infected"]
            mean_uninf_w = summary_row["mean_velocity_uninfected"]

        bbox_w = patches.FancyBboxPatch(
            (box_x0, box_y0),
            box_width,
            box_height,
            transform=ax_w.transAxes,
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.02",
            alpha=0.8,
        )
        ax_w.add_patch(bbox_w)

        ax_w.text(
            text_x,
            y_top,
            f"Infected ({_fmt_vel(mean_inf_w)} {vel_unit})",
            color="red",
            transform=ax_w.transAxes,
            fontsize=fontsize_main,
            va="top",
        )
        ax_w.text(
            text_x,
            y_top - line_spacing,
            f"Uninfected ({_fmt_vel(mean_uninf_w)} {vel_unit})",
            color="green",
            transform=ax_w.transAxes,
            fontsize=fontsize_main,
            va="top",
        )
        ax_w.text(
            text_x,
            y_top - 2 * line_spacing,
            unit_line1,
            color="black",
            transform=ax_w.transAxes,
            fontsize=fontsize_units,
            va="top",
        )
        ax_w.text(
            text_x,
            y_top - 3 * line_spacing,
            unit_line2,
            color="black",
            transform=ax_w.transAxes,
            fontsize=fontsize_units,
            va="top",
        )

        plt.tight_layout()
        out_well = os.path.join(
            motility_dir, f"motility_{plateID}_{wellID}_all_tracks.png"
        )
        fig_w.savefig(out_well, dpi=300)
        plt.close(fig_w)
        print(
            f"[summarise_tracks_from_merged] Saved per-well motility plot "
            f"to {out_well}"
        )

        # infected-only, re-centred to (0,0)
        if has_infected:
            fig_inf, ax_inf = plt.subplots(figsize=(6, 6))
            for tr in tracks:
                if not tr["infected"]:
                    continue
                x = (tr["x_px"] - tr["x_px"][0]) * coord_scale
                y = (tr["y_px"] - tr["y_px"][0]) * coord_scale
                ax_inf.plot(x, y, color="red", alpha=0.2, linewidth=0.5)
                ax_inf.scatter(x[-1], y[-1], color="red", s=5)
            ax_inf.set_aspect("equal", "box")
            ax_inf.set_xlabel(coord_label_x)
            ax_inf.set_ylabel(coord_label_y)
            _apply_axis_limits(ax_inf, origin_xlim, origin_ylim)
            plt.tight_layout()
            out_inf = os.path.join(
                motility_dir, f"motility_{plateID}_{wellID}_infected_origin.png"
            )
            fig_inf.savefig(out_inf, dpi=300)
            plt.close(fig_inf)
            print(
                f"[summarise_tracks_from_merged] Saved per-well infected "
                f"origin plot to {out_inf}"
            )

        # uninfected-only, re-centred to (0,0)
        if has_uninfected:
            fig_uninf, ax_uninf = plt.subplots(figsize=(6, 6))
            for tr in tracks:
                if tr["infected"]:
                    continue
                x = (tr["x_px"] - tr["x_px"][0]) * coord_scale
                y = (tr["y_px"] - tr["y_px"][0]) * coord_scale
                ax_uninf.plot(x, y, color="green", alpha=0.2, linewidth=0.5)
                ax_uninf.scatter(x[-1], y[-1], color="green", s=5)
            ax_uninf.set_aspect("equal", "box")
            ax_uninf.set_xlabel(coord_label_x)
            ax_uninf.set_ylabel(coord_label_y)
            _apply_axis_limits(ax_uninf, origin_xlim, origin_ylim)
            plt.tight_layout()
            out_uninf = os.path.join(
                motility_dir, f"motility_{plateID}_{wellID}_uninfected_origin.png"
            )
            fig_uninf.savefig(out_uninf, dpi=300)
            plt.close(fig_uninf)
            print(
                f"[summarise_tracks_from_merged] Saved per-well uninfected "
                f"origin plot to {out_uninf}"
            )

def _select_infection_feature_columns(all_df, pathogen_chan):
    """
    Select numeric feature columns for infection QC:
    - numeric columns
    - drop obvious IDs / motility metrics
    - drop centroid (coordinate) features
    - keep intensity features only for the pathogen channel
    - drop near-constant or almost-empty columns at cell level
    """
    import numpy as np

    numeric_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "frame",
        "timeID",
        "cellID",
        "n_pathogens",
        "v_px_per_frame",
        "velocity",
        "straightness",
    }
    # drop any debug / temporary numeric cols if present
    exclude |= {c for c in numeric_cols if c.endswith("_idx")}

    # drop centroid features (absolute coordinates)
    exclude |= {c for c in numeric_cols if "centroid" in c.lower()}

    # exclude intensity columns for non-pathogen channels
    if pathogen_chan is not None:
        for c in numeric_cols:
            if "intensity_ch" in c:
                try:
                    digits = "".join(ch for ch in c.split("ch")[-1] if ch.isdigit())
                    if digits != "":
                        ch_idx = int(digits)
                        if ch_idx != pathogen_chan:
                            exclude.add(c)
                except Exception:
                    # if parsing fails, keep column
                    pass

    feature_cols = [c for c in numeric_cols if c not in exclude]
    if not feature_cols:
        return []

    key_cols = ["plateID", "wellID", "fieldID", "cellID"]
    agg_cols = [c for c in feature_cols if c in all_df.columns]
    if not agg_cols:
        return []

    # Build per-cell table to filter out useless columns
    cell_level = (
        all_df[key_cols + agg_cols]
        .groupby(key_cols, dropna=False)
        .median()
        .reset_index()
    )

    filtered = []
    for c in agg_cols:
        arr = cell_level[c].to_numpy(dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() < 10:
            continue
        if np.nanstd(arr[finite]) < 1e-6:
            continue
        filtered.append(c)

    return filtered

def _compute_intensity_percentiles_per_channel(
    mask_stack,
    intensity_stack,
    channel_index,
    object_prefix,
    percentiles=(1, 5, 10, 25, 75, 95, 99),
    label_as_track_id=False,
):
    """
    Compute per-frame, per-object intensity percentiles for a given channel.

    Parameters
    ----------
    mask_stack : ndarray
        Label image stack of shape (T, Y, X).
    intensity_stack : ndarray
        Intensity stack of shape (T, Y, X, C).
    channel_index : int
        Channel index in intensity_stack.
    object_prefix : str
        Prefix for column names ("cell", "nucleus", "pathogen", "cytoplasm").
    percentiles : tuple of int
        Percentiles to compute (0–100).
    label_as_track_id : bool
        If True, rename 'label' -> 'track_id'; otherwise
        'label' -> f"{object_prefix}_label".

    Returns
    -------
    DataFrame
        Columns: ['frame', label_col, f'{object_prefix}_pXX_intensity_ch{channel_index}', ...]
    """
    import numpy as np
    import pandas as pd

    if intensity_stack is None:
        return pd.DataFrame(
            columns=["frame", "track_id" if label_as_track_id else f"{object_prefix}_label"]
        )

    if channel_index is None or channel_index < 0 or channel_index >= intensity_stack.shape[-1]:
        return pd.DataFrame(
            columns=["frame", "track_id" if label_as_track_id else f"{object_prefix}_label"]
        )

    T = mask_stack.shape[0]
    dfs = []
    label_col_name = "track_id" if label_as_track_id else f"{object_prefix}_label"

    perc = np.array(percentiles, dtype=float)

    for frame in range(T):
        labels = mask_stack[frame]
        if not np.any(labels):
            continue

        intensity_image = intensity_stack[frame, :, :, channel_index]
        # unique labels > 0
        obj_labels = np.unique(labels)
        obj_labels = obj_labels[obj_labels > 0]
        if obj_labels.size == 0:
            continue

        records = []
        for lab in obj_labels:
            mask = labels == lab
            vals = intensity_image[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            pvals = np.percentile(vals, perc)
            rec = {"frame": frame, label_col_name: int(lab)}
            for p, v in zip(perc, pvals):
                col_name = f"{object_prefix}_p{int(p):02d}_intensity_ch{channel_index}"
                rec[col_name] = float(v)
            records.append(rec)

        if records:
            dfs.append(pd.DataFrame.from_records(records))

    if not dfs:
        return pd.DataFrame(columns=["frame", label_col_name])

    out_df = pd.concat(dfs, ignore_index=True)
    return out_df

def _make_adjusted_qc_panel(
    all_df,
    infection_col,
    motility_dir,
    settings,
    label_tag,
):
    """
    Build a QC results panel for adjusted labels using 3 subplots:

        - top-left: PCA (adjusted_infected)
        - top-right: XGBoost feature importance
        - bottom: pathogen-channel intensity histogram

    Uses payloads stored in `settings` by the QC functions:
        settings["infection_hist_data"]
        settings["infection_pca_data"]
        settings["infection_xgb_importance"]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(motility_dir, exist_ok=True)
    meta_tag = _infer_plate_well_meta_tag(all_df)

    # Create figure with desired layout
    fig, ax_pca, ax_xgb, ax_hist = create_results_figure()

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------
    hist_data = settings.get("infection_hist_data") or {}
    vals_inf = np.asarray(hist_data.get("intensities_inf", []), dtype=float)
    vals_uninf = np.asarray(hist_data.get("intensities_uninf", []), dtype=float)
    bin_edges = np.asarray(hist_data.get("bin_edges", []), dtype=float)
    thr_val = hist_data.get("thr_val", None)
    pathogen_chan = hist_data.get("pathogen_chan", None)
    do_log = bool(hist_data.get("log_transform", False))

    if vals_inf.size + vals_uninf.size > 0 and bin_edges.size > 0:
        ax_hist.hist(
            vals_uninf,
            bins=bin_edges,
            alpha=0.5,
            color="green",
            label="Uninfected",
        )
        ax_hist.hist(
            vals_inf,
            bins=bin_edges,
            alpha=0.5,
            color="red",
            label="Infected",
        )
        if thr_val is not None:
            ax_hist.axvline(
                thr_val,
                linestyle="--",
                linewidth=2,
                color="black",
                label=f"thr={thr_val:.2f}",
            )
        if pathogen_chan is not None:
            if do_log:
                ax_hist.set_xlabel(f"log10 intensity (channel {pathogen_chan})")
            else:
                ax_hist.set_xlabel(f"Intensity (channel {pathogen_chan})")
        else:
            ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Cell count")
        ax_hist.set_title("Pathogen-channel intensity histogram")
        ax_hist.legend(loc="best")
    else:
        ax_hist.text(
            0.5,
            0.5,
            "No histogram data",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_hist.axis("off")

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    pca_data = settings.get("infection_pca_data") or {}
    coords = pca_data.get("coords", None)
    labels = pca_data.get("labels", None)
    method_label = pca_data.get("method_label", "PCA")

    if coords is not None and labels is not None:
        coords = np.asarray(coords, dtype=float)
        labels = np.asarray(labels, dtype=bool)
        if coords.ndim == 2 and coords.shape[0] == labels.shape[0] and coords.shape[1] >= 2:
            x = coords[:, 0]
            y = coords[:, 1]
            ax_pca.scatter(
                x[~labels],
                y[~labels],
                s=8,
                c="green",
                alpha=0.5,
                label="Uninfected",
            )
            ax_pca.scatter(
                x[labels],
                y[labels],
                s=8,
                c="red",
                alpha=0.5,
                label="Infected",
            )
            ax_pca.set_xlabel("component 1")
            ax_pca.set_ylabel("component 2")
            ax_pca.set_title(f"{method_label} embedding")
            ax_pca.legend(loc="best")
        else:
            ax_pca.text(
                0.5,
                0.5,
                "No PCA data",
                ha="center",
                va="center",
                transform=ax_pca.transAxes,
            )
            ax_pca.axis("off")
    else:
        ax_pca.text(
            0.5,
            0.5,
            "No PCA data",
            ha="center",
            va="center",
            transform=ax_pca.transAxes,
        )
        ax_pca.axis("off")

    # ------------------------------------------------------------------
    # XGBoost feature importance
    # ------------------------------------------------------------------
    xgb_data = settings.get("infection_xgb_importance") or {}
    feat_names = xgb_data.get("feature_names") or []
    feat_vals = xgb_data.get("feature_importances") or []

    if feat_names and feat_vals and len(feat_names) == len(feat_vals):
        feat_names = list(feat_names)
        feat_vals = np.asarray(feat_vals, dtype=float)
        y_pos = np.arange(len(feat_names))
        ax_xgb.barh(y_pos, feat_vals)
        ax_xgb.set_yticks(y_pos)
        ax_xgb.set_yticklabels(feat_names)
        ax_xgb.invert_yaxis()
        ax_xgb.set_xlabel("Importance (gain)")
        ax_xgb.set_title("XGBoost feature importance")
    else:
        ax_xgb.text(
            0.5,
            0.5,
            "No XGBoost importance data",
            ha="center",
            va="center",
            transform=ax_xgb.transAxes,
        )
        ax_xgb.axis("off")

    fig.suptitle(
        f"Infection QC panel – {label_tag} labels\n{meta_tag}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_name = f"infection_qc_panel_{label_tag}_{meta_tag}.png"
    out_path = os.path.join(motility_dir, out_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(
        f"[summarise_tracks_from_merged] Saved infection QC results panel "
        f"({label_tag}) to {out_path}"
    )


def _load_measurements_from_db(db_path, db_table_name):
    """
    Load per-cell measurements from an existing SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (measurements.db).
    db_table_name : str
        Name of the table that stores per-cell measurements.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the measurements, or an empty DataFrame if the
        database/table is missing or unreadable.
    """
    import os
    import sqlite3
    import pandas as pd

    if not os.path.isfile(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT * FROM {db_table_name}"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(
            "[summarise_tracks_from_merged] Could not load existing measurements "
            f"from {db_path} (table='{db_table_name}'): {e}"
        )
        df = pd.DataFrame()
    finally:
        conn.close()

    return df

def _infection_qc_histogram(
    all_df,
    settings,
    infection_col,
    pathogen_chan,
    motility_dir,
):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Prefer 95th percentile of pathogen channel; fall back to mean if needed
    cand_cols = [
        f"cell_p95_intensity_ch{pathogen_chan}",
        f"cell_mean_intensity_ch{pathogen_chan}",
    ]
    intensity_col = None
    for c in cand_cols:
        if c in all_df.columns:
            intensity_col = c
            break

    # Initialize payload slot
    settings["infection_hist_data"] = None

    if intensity_col is None:
        print(
            f"[infection_intensity_qc] None of {cand_cols} found; "
            f"skipping intensity-based relabelling."
        )
        settings["infection_intensity_qc_panel_type"] = "histogram"
        settings["infection_intensity_qc_panel_path"] = None
        return all_df, infection_col

    if intensity_col not in all_df.columns:
        print(
            f"[infection_intensity_qc] Column {intensity_col!r} not found; "
            f"skipping intensity-based relabelling."
        )
        settings["infection_intensity_qc_panel_type"] = "histogram"
        settings["infection_intensity_qc_panel_path"] = None
        return all_df, infection_col

    # --- IMPORTANT: drop any existing adjusted_infected when reusing DB ---
    # This prevents merge from creating adjusted_infected_x / adjusted_infected_y
    # and guarantees we recompute labels fresh each run.
    cols_to_drop = [
        c
        for c in all_df.columns
        if c == "adjusted_infected" or c.startswith("adjusted_infected_")
    ]
    if cols_to_drop:
        all_df = all_df.drop(columns=cols_to_drop)

    key_cols = ["plateID", "wellID", "fieldID", "cellID"]
    cell_level = (
        all_df[key_cols + [intensity_col, infection_col]]
        .groupby(key_cols, dropna=False)
        .agg({intensity_col: "mean", infection_col: "max"})
        .reset_index()
    )

    cell_level = cell_level.replace([np.inf, -np.inf], np.nan)
    cell_level = cell_level.dropna(subset=[intensity_col])

    if len(cell_level) < 20 or cell_level[intensity_col].nunique() < 2:
        print(
            "[infection_intensity_qc] Too few cells or no intensity variation; "
            "skipping intensity-based relabelling."
        )
        settings["infection_intensity_qc_panel_type"] = "histogram"
        settings["infection_intensity_qc_panel_path"] = None
        return all_df, infection_col

    intensities = cell_level[intensity_col].to_numpy(dtype=float)
    mask_labels = cell_level[infection_col].to_numpy(dtype=bool)

    # Optional log-transform to help separate populations
    do_log = bool(settings.get("infection_intensity_log", False))
    if do_log:
        eps = np.nanmax([np.nanmin(intensities[intensities > 0]) * 0.5, 1e-6])
        intensities = np.log10(intensities + eps)

    n_bins = int(settings.get("infection_intensity_n_bins", 64))
    n_bins = max(10, min(n_bins, 256))

    counts_all, bin_edges = np.histogram(intensities, bins=n_bins)
    counts_inf, _ = np.histogram(intensities[mask_labels], bins=bin_edges)

    denom = np.maximum(counts_all, 1)
    frac_inf = counts_inf.astype(float) / denom.astype(float)

    # Target fraction of infected in a bin
    target_frac = float(settings.get("infection_intensity_frac_infected", 0.7))
    target_frac = max(0.5, min(target_frac, 0.95))

    # Fallback percentile (now default 25th)
    hist_pct = float(settings.get("infection_hist_percentile", 25.0))
    hist_pct = max(0.0, min(hist_pct, 100.0))

    # First bin (low→high) where infected ≥ target_frac
    thresh_idx = None
    for i, frac in enumerate(frac_inf):
        if frac >= target_frac:
            thresh_idx = i
            break

    if thresh_idx is None:
        # fallback: hist_pct percentile of all cells (after optional log)
        thr_val = float(np.nanpercentile(intensities, hist_pct))
        print(
            "[infection_intensity_qc] Could not find bin with infected ≥ "
            f"{target_frac:.2f}; using {hist_pct:.1f}th percentile of all cells "
            f"({thr_val:.2f}) as threshold."
        )
    else:
        thr_val = float(bin_edges[thresh_idx])
        print(
            "[infection_intensity_qc] Automatic intensity threshold at first bin "
            f"where infected ≥ {target_frac:.2f}: {thr_val:.2f} (bin {thresh_idx})"
        )

    cell_level["intensity_positive"] = cell_level[intensity_col] >= thr_val

    mode = str(settings.get("infection_intensity_mode", "relabel")).lower()
    if mode not in {"relabel", "remove"}:
        mode = "relabel"

    removed_ids = None
    if mode == "relabel":
        cell_level["adjusted_infected"] = cell_level["intensity_positive"].astype(bool)
        n_changed = int(
            (
                cell_level["adjusted_infected"]
                != cell_level[infection_col].astype(bool)
            ).sum()
        )
        print(
            "[infection_intensity_qc] Adjusted infection labels for "
            f"{n_changed} cells (mode=relabel)."
        )
    else:
        consistent = (
            cell_level[infection_col].astype(bool)
            == cell_level["intensity_positive"].astype(bool)
        )
        removed = cell_level.loc[
            ~consistent, ["plateID", "wellID", "fieldID", "cellID"]
        ]
        removed_ids = set(
            zip(
                removed["plateID"],
                removed["wellID"],
                removed["fieldID"],
                removed["cellID"],
            )
        )
        cell_level = cell_level.loc[consistent].copy()
        cell_level["adjusted_infected"] = cell_level["intensity_positive"].astype(bool)
        print(
            "[infection_intensity_qc] Removed "
            f"{len(removed_ids)} cells with conflicting mask vs intensity labels "
            "(mode=remove)."
        )

    # merge back
    all_df = all_df.merge(
        cell_level[key_cols + ["adjusted_infected"]],
        on=key_cols,
        how="left",
    )
    if removed_ids:
        mask_keep = ~all_df.apply(
            lambda r: (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
            in removed_ids,
            axis=1,
        )
        all_df = all_df.loc[mask_keep].reset_index(drop=True)

    # Now adjusted_infected definitely exists and comes from histogram QC;
    # fill any NaNs (cells not in cell_level) from the original infection_col.
    all_df["adjusted_infected"] = all_df["adjusted_infected"].fillna(
        all_df[infection_col]
    )
    infection_col = "adjusted_infected"

    # Decide whether to make / save QC graph
    make_graphs = bool(settings.get("infection_intensity_qc_graphs", True))
    meta_tag = _infer_plate_well_meta_tag(all_df)
    hist_path = None

    # Prepare payload (even if we don't generate PNG)
    vals_inf = intensities[mask_labels]
    vals_uninf = intensities[~mask_labels]
    hist_payload = {
        "intensities_inf": vals_inf,
        "intensities_uninf": vals_uninf,
        "bin_edges": bin_edges,
        "thr_val": thr_val,
        "pathogen_chan": pathogen_chan,
        "log_transform": do_log,
        "intensity_col": intensity_col,
    }
    settings["infection_hist_data"] = hist_payload

    if make_graphs:
        # Plot histogram
        os.makedirs(motility_dir, exist_ok=True)

        fig_h, ax_h = plt.subplots(figsize=(6, 4))
        ax_h.hist(
            vals_uninf,
            bins=bin_edges,
            alpha=0.5,
            color="green",
            label="Uninfected (mask-based)",
        )
        ax_h.hist(
            vals_inf,
            bins=bin_edges,
            alpha=0.5,
            color="red",
            label="Infected (mask-based)",
        )
        ax_h.axvline(
            thr_val,
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"Threshold = {thr_val:.1f}",
        )
        if do_log:
            ax_h.set_xlabel(f"log10 intensity metric (channel {pathogen_chan})")
        else:
            ax_h.set_xlabel(f"Intensity metric (channel {pathogen_chan})")
        ax_h.set_ylabel("Cell count")
        ax_h.set_title(
            f"Pathogen-channel intensity histogram (thr={thr_val:.1f}, "
            f"{hist_pct:.1f}th pct fallback)"
        )
        ax_h.legend(loc="best")

        hist_filename = f"infection_intensity_histogram_{meta_tag}.png"
        hist_path = os.path.join(motility_dir, hist_filename)
        fig_h.tight_layout()
        fig_h.savefig(hist_path, dpi=200)
        plt.close(fig_h)

        print(f"[infection_intensity_qc] Saved histogram to: {hist_path}")
    else:
        print(
            "[infection_intensity_qc] infection_intensity_qc_graphs=False; "
            "skipping histogram plot."
        )

    # Let the panel know what QC plot to embed (if present)
    settings["infection_intensity_qc_panel_type"] = "histogram"
    settings["infection_intensity_qc_panel_path"] = hist_path

    return all_df, infection_col

def _infection_qc_xgboost(all_df, settings, infection_col, pathogen_chan, motility_dir):
    """
    Use an XGBoost classifier to refine infection calling based on per-object features.

    Key behaviour
    -------------
    - Training:
        * per-cell (or per-object) medians across frames
        * {tracked_object}_* morphology + {tracked_object}_pathogen-channel intensity features
        * training labels from pathogen-channel intensity extremes:
            - bottom 25% of UNINFECTED → strong negatives
            - top    25% of INFECTED   → strong positives
        * training data curated per well:
            - wells with both classes in the extreme set:
                - if both classes have >= infection_xgb_min_cells_per_class examples:
                    → balanced sampling per class within that well
                - else (small wells with both classes):
                    → keep all extreme examples from that well
            - wells with only one class in the extreme set are skipped
            - if no well has both classes in the extreme set → skip XGBoost QC

    - Prediction:
        * get P(infected) for all cells
        * "mode":
            - 'relabel': start from original labels, override only when model
                         is confident; ambiguous are still dropped if requested.
            - 'remove' : drop strong label–model disagreements AND ambiguous if
                         requested.

    - Ambiguous band removal:
        * if infection_xgb_drop_ambiguous is True (default),
          drop cells with proba in [infection_xgb_ambiguous_low,
                                    infection_xgb_ambiguous_high]
          (defaults: 0.25 and 0.75).

    Additionally, this function stores three QC payloads in `settings`:

        settings["infection_hist_data"]        : dict for histogram panel
        settings["infection_pca_data"]         : dict for PCA panel
        settings["infection_xgb_importance"]   : dict for feature-importance panel

    Returns
    -------
    all_df, infection_col='adjusted_infected'
    """
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        import xgboost as xgb
    except ImportError:
        print("[_infection_qc_xgboost] XGBoost not installed; using histogram QC.")
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    # init payload slots
    settings["infection_hist_data"] = None
    settings["infection_pca_data"] = None
    settings["infection_xgb_importance"] = None

    # IMPORTANT: drop any existing adjusted_* / infection_prob* from DB reuse
    cols_to_drop = [
        c
        for c in all_df.columns
        if c == "adjusted_infected"
        or c.startswith("adjusted_infected_")
        or c == "infection_prob"
        or c.startswith("infection_prob_")
    ]
    if cols_to_drop:
        all_df = all_df.drop(columns=cols_to_drop)

    orig_infection_col = infection_col

    if pathogen_chan is None:
        print("[_infection_qc_xgboost] pathogen_chan is None; using histogram QC.")
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    # Ensure n_pathogens exists and has 0 instead of NaN
    if "n_pathogens" in all_df.columns:
        all_df["n_pathogens"] = all_df["n_pathogens"].fillna(0)

    # Recover infection column if missing
    if orig_infection_col not in all_df.columns:
        infect_like = [c for c in all_df.columns if "infect" in c.lower()]
        if infect_like:
            new_col = infect_like[0]
            print(
                f"[_infection_qc_xgboost] Column {orig_infection_col!r} not found; "
                f"using {new_col!r} instead."
            )
            orig_infection_col = new_col
        elif "n_pathogens" in all_df.columns:
            orig_infection_col = "_infected_from_n_pathogens"
            all_df[orig_infection_col] = (all_df["n_pathogens"] > 0).astype(int)
            print(
                "[_infection_qc_xgboost] Column 'infected' not found; created "
                f"{orig_infection_col!r} from 'n_pathogens > 0'."
            )
        else:
            print(
                "[_infection_qc_xgboost] No infection label and no 'n_pathogens'; "
                "using histogram QC instead."
            )
            return _infection_qc_histogram(
                all_df=all_df,
                settings=settings,
                infection_col=infection_col,
                pathogen_chan=pathogen_chan,
                motility_dir=motility_dir,
            )

    key_cols = ["plateID", "wellID", "fieldID", "cellID"]
    for col in key_cols:
        if col not in all_df.columns:
            raise KeyError(f"[_infection_qc_xgboost] Required column {col!r} not in all_df.")

    # ------------------------------------------------------------------
    # Aggregate to per-object level (median across frames)
    # ------------------------------------------------------------------
    agg_cols = [
        c
        for c in all_df.columns
        if c not in (key_cols + ["frame", "timeID", orig_infection_col])
    ]

    group = all_df.groupby(key_cols, observed=True)
    cell_level = group[agg_cols].median(numeric_only=True).reset_index()

    infection_any = (
        all_df.groupby(key_cols, observed=True)[orig_infection_col]
        .max()
        .reset_index()
    )
    cell_level = cell_level.merge(
        infection_any, on=key_cols, how="left", suffixes=("", "_y")
    )

    if orig_infection_col not in cell_level.columns:
        for cand in (f"{orig_infection_col}_y", f"{orig_infection_col}_x"):
            if cand in cell_level.columns:
                cell_level[orig_infection_col] = cell_level[cand]
                break

    if orig_infection_col not in cell_level.columns:
        raise KeyError(
            f"[_infection_qc_xgboost] Infection column {orig_infection_col!r} "
            "missing from per-cell table after aggregation/merge."
        )

    cell_level[orig_infection_col] = (
        cell_level[orig_infection_col]
        .fillna(0)
        .astype(bool)
    )

    if "n_pathogens" in cell_level.columns:
        cell_level["n_pathogens"] = cell_level["n_pathogens"].fillna(0)

    # ------------------------------------------------------------------
    # Decide which object type's features to use (tracked_object)
    # ------------------------------------------------------------------
    tracked_object = str(settings.get("tracked_object", "cell")).strip().lower()
    if tracked_object not in {"cell", "nucleus", "pathogen"}:
        print(
            f"[_infection_qc_xgboost] Unknown tracked_object={tracked_object!r}; "
            "falling back to 'cell'."
        )
        tracked_object = "cell"
    obj_prefix = f"{tracked_object}_"

    # ------------------------------------------------------------------
    # Decide pathogen-channel intensity column for this tracked_object
    # ------------------------------------------------------------------
    intensity_candidates = [
        f"{obj_prefix}p95_intensity_ch{pathogen_chan}",
        f"{obj_prefix}max_intensity_ch{pathogen_chan}",
        f"{obj_prefix}mean_intensity_ch{pathogen_chan}",
    ]
    intensity_col = None
    for c in intensity_candidates:
        if c in cell_level.columns:
            intensity_col = c
            break

    if intensity_col is None:
        print(
            "[_infection_qc_xgboost] No pathogen-channel intensity column found for "
            f"tracked_object={tracked_object!r} "
            f"(tried: {intensity_candidates}); using histogram QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    # ------------------------------------------------------------------
    # Build feature set: {tracked_object}_* only, excluding centroids,
    # non-pathogen channels, degenerate features
    # ------------------------------------------------------------------
    numeric_cols = cell_level.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    feature_cols = []
    pattern_obj = re.compile(rf"^{re.escape(obj_prefix)}")
    pattern_ch = re.compile(r"ch(\d+)\b")

    for c in numeric_cols:
        if c == orig_infection_col:
            continue
        if c in {"frame", "timeID"}:
            continue
        if "centroid" in c.lower():
            continue
        if not pattern_obj.match(c):
            continue

        m = pattern_ch.search(c)
        if m:
            ch_idx = int(m.group(1))
            if ch_idx != int(pathogen_chan):
                continue
            feature_cols.append(c)
        else:
            feature_cols.append(c)

    if intensity_col in cell_level.columns and intensity_col not in feature_cols:
        feature_cols.append(intensity_col)

    # Drop degenerate features
    clean_feature_cols = []
    for c in feature_cols:
        s = cell_level[c]
        if s.notna().sum() < 10:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        clean_feature_cols.append(c)
    feature_cols = clean_feature_cols

    if not feature_cols:
        print(
            "[_infection_qc_xgboost] No usable "
            f"{tracked_object}_* feature columns; using histogram QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    # ------------------------------------------------------------------
    # Global sanity check: do we even have enough infected/uninfected cells?
    # ------------------------------------------------------------------
    infected_cells = cell_level[cell_level[orig_infection_col]]
    uninfected_cells = cell_level[~cell_level[orig_infection_col]]

    if len(infected_cells) < 10 or len(uninfected_cells) < 10:
        print(
            "[_infection_qc_xgboost] Too few infected or uninfected cells overall; "
            "using histogram QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    # ------------------------------------------------------------------
    # Define confident training sets using intensity quartiles
    # ------------------------------------------------------------------
    inf_int = infected_cells[intensity_col].to_numpy(dtype=float)
    uninf_int = uninfected_cells[intensity_col].to_numpy(dtype=float)

    inf_int = inf_int[np.isfinite(inf_int)]
    uninf_int = uninf_int[np.isfinite(uninf_int)]

    if inf_int.size == 0 or uninf_int.size == 0:
        print(
            "[_infection_qc_xgboost] No finite intensities for infected/uninfected; "
            "using histogram QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    high_thr_inf = np.nanpercentile(inf_int, 75.0)
    low_thr_uninf = np.nanpercentile(uninf_int, 25.0)

    hi_inf = infected_cells[infected_cells[intensity_col] >= high_thr_inf].copy()
    lo_uninf = uninfected_cells[uninfected_cells[intensity_col] <= low_thr_uninf].copy()

    if hi_inf.empty or lo_uninf.empty:
        print(
            "[_infection_qc_xgboost] Could not define confident high/low quartiles; "
            "using histogram QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    print(
        "[_infection_qc_xgboost] Extreme-intensity candidates: "
        f"infected={len(hi_inf)}, uninfected={len(lo_uninf)} "
        f"(tracked_object={tracked_object}, intensity_col={intensity_col}, "
        f"low_thr_uninf={low_thr_uninf:.3f}, high_thr_inf={high_thr_inf:.3f})"
    )

    # ------------------------------------------------------------------
    # Curate XGBoost training data per well
    # ------------------------------------------------------------------
    hi_inf["xgb_label"] = 1
    lo_uninf["xgb_label"] = 0
    train_candidates = pd.concat([hi_inf, lo_uninf], axis=0)

    min_per_class = int(settings.get("infection_xgb_min_cells_per_class", 10))
    rng_seed = settings.get("infection_xgb_random_state", 42)
    try:
        rng = np.random.default_rng(int(rng_seed))
    except Exception:
        rng = np.random.default_rng(42)

    train_idx_list = []
    y_train_list = []
    wells_used = set()
    wells_single_class = []

    grouped = train_candidates.groupby(["plateID", "wellID"], observed=True)

    for (plate_id, well_id), df_w in grouped:
        pos_df = df_w[df_w["xgb_label"] == 1]
        neg_df = df_w[df_w["xgb_label"] == 0]
        n_pos = len(pos_df)
        n_neg = len(neg_df)

        if n_pos == 0 or n_neg == 0:
            # wells with only one class → skip for training
            wells_single_class.append((plate_id, well_id))
            continue

        pos_idx = pos_df.index.to_numpy()
        neg_idx = neg_df.index.to_numpy()

        if n_pos >= min_per_class and n_neg >= min_per_class:
            # wells with enough data per class → balanced sampling within well
            n_per_class = min(n_pos, n_neg)
            if n_pos > n_per_class:
                pos_sel = rng.choice(pos_idx, size=n_per_class, replace=False)
            else:
                pos_sel = pos_idx
            if n_neg > n_per_class:
                neg_sel = rng.choice(neg_idx, size=n_per_class, replace=False)
            else:
                neg_sel = neg_idx
        else:
            # small wells with both classes → keep all extreme examples
            pos_sel = pos_idx
            neg_sel = neg_idx

        if pos_sel.size == 0 or neg_sel.size == 0:
            wells_single_class.append((plate_id, well_id))
            continue

        train_idx_list.extend(pos_sel.tolist())
        y_train_list.extend([1] * pos_sel.size)
        train_idx_list.extend(neg_sel.tolist())
        y_train_list.extend([0] * neg_sel.size)
        wells_used.add((plate_id, well_id))

    if not wells_used:
        print(
            "[_infection_qc_xgboost] No wells with both infected and uninfected "
            "extreme-intensity examples; skipping XGBoost QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    train_idx = np.array(train_idx_list, dtype=int)
    y_train = np.array(y_train_list, dtype=int)

    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())
    print(
        "[_infection_qc_xgboost] Training set (after per-well curation): "
        f"wells used={len(wells_used)}, positives={n_pos_train}, "
        f"negatives={n_neg_train}, min_per_class={min_per_class}."
    )
    if wells_single_class:
        print(
            "[_infection_qc_xgboost] Wells skipped due to single class in extreme set: "
            + ", ".join([f"{p}_{w}" for (p, w) in wells_single_class])
        )

    # ------------------------------------------------------------------
    # Build feature matrix + median imputation
    # ------------------------------------------------------------------
    X_all = cell_level[feature_cols].to_numpy(dtype=float)
    for j in range(X_all.shape[1]):
        col = X_all[:, j]
        mask = np.isfinite(col)
        if not mask.any():
            X_all[:, j] = 0.0
        else:
            med = np.nanmedian(col[mask])
            col[~mask] = med
            X_all[:, j] = col

    X_train = X_all[train_idx]

    # ------------------------------------------------------------------
    # Remove highly correlated features
    # ------------------------------------------------------------------
    if X_train.shape[1] > 1:
        corr = np.corrcoef(X_train, rowvar=False)
        corr_thr = float(settings.get("infection_xgb_corr_threshold", 0.95))
        keep = np.ones(corr.shape[0], dtype=bool)

        for i in range(corr.shape[0]):
            if not keep[i]:
                continue
            for j in range(i + 1, corr.shape[0]):
                if keep[j] and abs(corr[i, j]) >= corr_thr:
                    keep[j] = False

        if not keep.any():
            print(
                "[_infection_qc_xgboost] All features flagged as highly correlated; "
                "keeping original feature set."
            )
        else:
            removed = [f for f, k in zip(feature_cols, keep) if not k]
            if removed:
                print(
                    "[_infection_qc_xgboost] Removing highly correlated features "
                    f"(>|{corr_thr:.2f}|): " + ", ".join(removed)
                )
            feature_cols = [f for f, k in zip(feature_cols, keep) if k]
            X_all = X_all[:, keep]
            X_train = X_train[:, keep]

    used_feature_cols = feature_cols

    if X_train.shape[1] == 0:
        print(
            "[_infection_qc_xgboost] No usable feature columns after correlation "
            "filtering; using histogram QC."
        )
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    print(
        f"[_infection_qc_xgboost] Using {len(used_feature_cols)} "
        f"{tracked_object}_* features:"
    )
    print("   " + ", ".join(used_feature_cols))

    # ------------------------------------------------------------------
    # Train XGBoost
    # ------------------------------------------------------------------
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=used_feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(settings.get("infection_xgb_max_depth", 3)),
        "eta": float(settings.get("infection_xgb_learning_rate", 0.1)),
        "subsample": float(settings.get("infection_xgb_subsample", 0.8)),
        "colsample_bytree": float(settings.get("infection_xgb_colsample_bytree", 0.8)),
        "lambda": float(settings.get("infection_xgb_reg_lambda", 1.0)),
        "alpha": 0.0,
        "verbosity": 0,
        "nthread": int(settings.get("infection_xgb_n_jobs", -1)),
    }
    num_round = int(settings.get("infection_xgb_n_estimators", 200))
    bst = xgb.train(params, dtrain, num_boost_round=num_round)

    # ------------------------------------------------------------------
    # Predict all cells
    # ------------------------------------------------------------------
    dall = xgb.DMatrix(X_all, feature_names=used_feature_cols)
    probs = bst.predict(dall)

    prob_thr = float(settings.get("infection_xgb_proba_threshold", 0.5))
    margin = float(settings.get("infection_xgb_margin", 0.0))
    margin = max(0.0, min(margin, 0.49))

    orig_arr = cell_level[orig_infection_col].astype(int).to_numpy()
    pred_arr = (probs >= prob_thr).astype(int)

    mode = str(settings.get("infection_intensity_mode", "relabel")).lower()
    if mode not in {"relabel", "remove"}:
        mode = "relabel"

    removed_ids = set()
    ambiguous_ids = set()

    if mode == "relabel":
        adjusted = orig_arr.copy()
        hi_conf = probs >= (prob_thr + margin)
        lo_conf = probs <= (prob_thr - margin)

        adjusted[hi_conf] = 1
        adjusted[lo_conf] = 0

        n_changed = int((adjusted != orig_arr).sum())
        print(
            "[_infection_qc_xgboost] Relabel mode: adjusted infection labels for "
            f"{n_changed} cells (prob_thr={prob_thr:.2f}, margin={margin:.2f})."
        )
        cell_level["adjusted_infected"] = adjusted
        cell_level["infection_prob"] = probs
    else:
        if margin > 0:
            ambig = np.abs(probs - prob_thr) < margin
        else:
            ambig = np.zeros_like(probs, dtype=bool)

        disagree = pred_arr != orig_arr
        to_remove = disagree & ~ambig

        removed = cell_level.loc[to_remove, key_cols]
        removed_ids = {
            (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
            for _, r in removed.iterrows()
        }

        cell_level = cell_level.loc[~to_remove].copy()
        kept_idx = (~to_remove).nonzero()[0]

        adjusted = orig_arr[kept_idx].copy()
        pred_kept = pred_arr[kept_idx]
        ambig_kept = ambig[kept_idx]

        adjusted[~ambig_kept] = pred_kept[~ambig_kept]

        cell_level["adjusted_infected"] = adjusted
        cell_level["infection_prob"] = probs[~to_remove]

        print(
            "[_infection_qc_xgboost] Remove mode: removed "
            f"{len(removed_ids)} cells with strong model vs label disagreement "
            f"(prob_thr={prob_thr:.2f}, margin={margin:.2f})."
        )

    # ------------------------------------------------------------------
    # Drop ambiguous band (probability in [low, high])
    # ------------------------------------------------------------------
    drop_amb = bool(settings.get("infection_xgb_drop_ambiguous", True))
    amb_low = float(settings.get("infection_xgb_ambiguous_low", 0.25))
    amb_high = float(settings.get("infection_xgb_ambiguous_high", 0.75))
    amb_low = max(0.0, min(amb_low, 1.0))
    amb_high = max(0.0, min(amb_high, 1.0))
    if amb_low > amb_high:
        amb_low, amb_high = amb_high, amb_low

    if drop_amb and "infection_prob" in cell_level.columns:
        amb_mask = (
            (cell_level["infection_prob"] >= amb_low)
            & (cell_level["infection_prob"] <= amb_high)
        )
        if amb_mask.any():
            amb = cell_level.loc[amb_mask, key_cols]
            ambiguous_ids = {
                (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
                for _, r in amb.iterrows()
            }
            cell_level = cell_level.loc[~amb_mask].copy()
            print(
                "[_infection_qc_xgboost] Dropped "
                f"{len(ambiguous_ids)} cells with ambiguous XGBoost probability "
                f"in [{amb_low:.2f}, {amb_high:.2f}]."
            )

    # ------------------------------------------------------------------
    # Map adjusted calls back to all_df
    # ------------------------------------------------------------------
    for col in key_cols:
        all_df[col] = all_df[col].astype(cell_level[col].dtype)

    # (any stale adjusted_infected/infection_prob already dropped above)

    all_df = all_df.merge(
        cell_level[key_cols + ["adjusted_infected", "infection_prob"]],
        on=key_cols,
        how="left",
        validate="m:1",
    )

    # Combine removed_sets (disagreement + ambiguous)
    ids_to_remove = set()
    if removed_ids:
        ids_to_remove |= removed_ids
    if ambiguous_ids:
        ids_to_remove |= ambiguous_ids

    if ids_to_remove:
        mask_drop = all_df.apply(
            lambda r: (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
            in ids_to_remove,
            axis=1,
        )
        all_df = all_df.loc[~mask_drop].reset_index(drop=True)

    mask_missing = all_df["adjusted_infected"].isna()
    if mask_missing.any():
        all_df.loc[mask_missing, "adjusted_infected"] = (
            all_df.loc[mask_missing, orig_infection_col].astype(int)
        )
    all_df["adjusted_infected"] = all_df["adjusted_infected"].astype(int)

    infection_col = "adjusted_infected"

    try:
        n_inf = int(all_df[infection_col].sum())
        n_uninf = int((1 - all_df[infection_col]).sum())
        print(
            f"[_infection_qc_xgboost] Final infection counts (frame-level): "
            f"infected={n_inf}, uninfected={n_uninf}"
        )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Prepare QC payloads for the combined adjusted panel
    #   - histogram of intensity (adjusted labels)
    #   - PCA embedding of used features (adjusted labels)
    #   - XGBoost feature importances (gain)
    # ------------------------------------------------------------------
    try:
        # --- histogram payload ---
        if intensity_col in cell_level.columns:
            intens = cell_level[intensity_col].to_numpy(dtype=float)
            labels_adj = cell_level["adjusted_infected"].astype(bool).to_numpy()
            mask_fin = np.isfinite(intens)
            intens = intens[mask_fin]
            labels_adj = labels_adj[mask_fin]
            vals_inf = intens[labels_adj]
            vals_uninf = intens[~labels_adj]
            if intens.size >= 10:
                n_bins = int(settings.get("infection_intensity_n_bins", 64))
                n_bins = max(10, min(n_bins, 256))
                _, bin_edges = np.histogram(intens, bins=n_bins)
                # Use midpoint between training thresholds as a visual threshold
                thr_val = float(0.5 * (low_thr_uninf + high_thr_inf))
                hist_payload = {
                    "intensities_inf": vals_inf,
                    "intensities_uninf": vals_uninf,
                    "bin_edges": bin_edges,
                    "thr_val": thr_val,
                    "pathogen_chan": pathogen_chan,
                    "log_transform": False,
                    "intensity_col": intensity_col,
                    "tracked_object": tracked_object,
                }
                settings["infection_hist_data"] = hist_payload

        # --- PCA payload ---
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if used_feature_cols:
            X_panel = cell_level[used_feature_cols].to_numpy(dtype=float)
            for j in range(X_panel.shape[1]):
                col = X_panel[:, j]
                m = np.isfinite(col)
                if not m.any():
                    X_panel[:, j] = 0.0
                else:
                    med = np.nanmedian(col[m])
                    col[~m] = med
                    X_panel[:, j] = col

            scaler = StandardScaler()
            X_scaled_panel = scaler.fit_transform(X_panel)
            pca = PCA(
                n_components=2,
                random_state=int(settings.get("infection_pca_random_state", 0)),
            )
            coords = pca.fit_transform(X_scaled_panel)
            labels_adj_panel = cell_level["adjusted_infected"].astype(bool).to_numpy()
            pca_payload = {
                "coords": coords,
                "labels": labels_adj_panel,
                "method_label": "PCA",
                "tracked_object": tracked_object,
            }
            settings["infection_pca_data"] = pca_payload
    except Exception as e:
        print(f"[_infection_qc_xgboost] Could not compute histogram/PCA payloads: {e}")

    # ------------------------------------------------------------------
    # Feature importance payload (no PNG; panels draw from this)
    # ------------------------------------------------------------------
    try:
        importance_dict = bst.get_score(importance_type="gain") or {}
        feat_names = used_feature_cols
        feat_vals = [importance_dict.get(f, 0.0) for f in feat_names]

        # sort and truncate
        sorted_pairs = sorted(
            zip(feat_names, feat_vals),
            key=lambda x: x[1],
            reverse=True,
        )
        feat_names = [p[0] for p in sorted_pairs]
        feat_vals = [p[1] for p in sorted_pairs]

        top_k = int(settings.get("infection_xgb_top_features", 20))
        feat_names = feat_names[:top_k]
        feat_vals = feat_vals[:top_k]

        settings["infection_xgb_importance"] = {
            "feature_names": feat_names,
            "feature_importances": feat_vals,
            "tracked_object": tracked_object,
        }

        if feat_names:
            print("[_infection_qc_xgboost] Top XGBoost features (gain):")
            for name, val in zip(feat_names, feat_vals):
                print(f"   {name}: {val:.4g}")
    except Exception as e:
        print(f"[_infection_qc_xgboost] Could not compute feature importances: {e}")

    # Mark QC type for panels; no embedded PNG (mask panel draws nothing)
    settings["infection_intensity_qc_panel_type"] = "xgboost"
    settings["infection_intensity_qc_panel_path"] = None

    return all_df, infection_col

def automated_motility_assay(settings):
    """
    End-to-end:

    1. Read merged/*.npy (plate_well_field_time.npy)
    2. Build intensity + cell/nucleus/pathogen masks, derive cytoplasm
    3. Per cell & frame: metadata + cell regionprops
    4. Aggregate child (nucleus/pathogen/cytoplasm) features per cell
    5. Concatenate across all merged files
    6. Clean impossible jumps + measurement glitches
    7. Save per-cell measurements to SQLite DB
       (measurements/measurements.db, table=db_table_name)
       **This table is always the original, pre-QC measurements.**
    8. Compute per-track velocities (after smoothing)
    9. Save a well-level motility summary table in the same DB
    10. Generate *panel* plots combining intensity + motility:
        - original (mask-based) infection labels
        - adjusted infection labels (if QC modifies labels)
    11. Optional infection intensity QC based on pathogen channel.

    New-relevant settings (all optional):

        # Infection QC / strategy
        'infection_intensity_qc': True/False
        'infection_intensity_strategy': one of
            {'xgboost', 'histogram', 'pca', 'umap', 'tsne'}
        'infection_intensity_mode': {'relabel', 'remove'}   # existing

        # XGBoost ambiguous-band filtering (track-level)
        'infection_xgb_drop_ambiguous': True/False (default True)
        'infection_xgb_ambiguous_low': 0.25  (default)
        'infection_xgb_ambiguous_high': 0.75 (default)
        'infection_xgb_proba_column': 'name_of_proba_col'   # optional override

        # Histogram strategy
        'infection_hist_percentile': 25  # used inside _apply_infection_intensity_qc

        # Panel toggles
        'make_mask_panel': True/False (default True)
        'make_adjusted_panel': True/False (default True)

        # Plot ranges (unchanged)
        - 'motility_xlim', 'motility_ylim'
        - 'motility_origin_xlim', 'motility_origin_ylim'

        # Measurements reuse
        'reuse_existing_measurements': True/False (default True)
    """
    import matplotlib.pyplot as plt  # noqa: F401 (used in helpers)
    from matplotlib import patches  # noqa: F401 (used in helpers)
    import numpy as np
    import pandas as pd
    import os
    from multiprocessing import Pool, cpu_count
    import sqlite3
    
    from .settings import get_automated_motility_assay_default_settings

    settings = get_automated_motility_assay_default_settings(settings)

    src = settings["src"]
    db_table_name = settings["db_table_name"]
    n_jobs = settings["n_jobs"]
    max_displacement = settings["max_displacement"]
    zscore_thresh = settings["zscore_thresh"]

    # ------------------------------------------------------------------
    # Optional reuse of existing measurements from SQLite
    #   → this table is treated as the ORIGINAL, pre-QC dataset.
    # ------------------------------------------------------------------
    reuse_existing = settings.get("reuse_existing_measurements", True)
    measurements_dir = os.path.join(src, "measurements")
    os.makedirs(measurements_dir, exist_ok=True)
    db_path = os.path.join(measurements_dir, "measurements.db")

    all_df = None
    loaded_from_db = False

    if reuse_existing and os.path.exists(db_path):
        try:
            print(
                f"[summarise_tracks_from_merged] Attempting to reuse existing "
                f"measurements from {db_path} (table='{db_table_name}')."
            )

            with sqlite3.connect(db_path) as conn:
                # If the table does not exist, this will raise and fall back to recompute
                all_df = pd.read_sql_query(f"SELECT * FROM {db_table_name}", conn)

            if (
                all_df is not None
                and not all_df.empty
                and {"plateID", "wellID", "fieldID", "cellID", "frame"}.issubset(
                    all_df.columns
                )
            ):
                n_frames_db = all_df["frame"].nunique()
                n_tracks_db = (
                    all_df[["plateID", "wellID", "fieldID", "cellID"]]
                    .drop_duplicates()
                    .shape[0]
                )
                print(
                    "[summarise_tracks_from_merged] Loaded ORIGINAL measurements "
                    f"from DB: shape={all_df.shape}, frames={n_frames_db}, "
                    f"tracks={n_tracks_db}. Skipping regionprops/intensity "
                    "computation from merged .npy files."
                )
                loaded_from_db = True
            else:
                print(
                    "[summarise_tracks_from_merged] Loaded table is empty or missing "
                    "required columns; recomputing from merged .npy files."
                )
                all_df = None
        except Exception as e:
            print(
                "[summarise_tracks_from_merged] Failed to reuse existing measurements "
                f"({e}); recomputing from merged .npy files."
            )
            all_df = None

    # ------------------------------------------------------------------
    # Read merged files & basic metadata (if not reusing DB)
    # ------------------------------------------------------------------
    merged_dir = os.path.join(src, "merged")
    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(f"No merged directory at: {merged_dir}")

    all_files = [f for f in os.listdir(merged_dir) if f.endswith(".npy")]
    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {merged_dir}")

    print(
        f"[summarise_tracks_from_merged] Found {len(all_files)} merged .npy files "
        f"in {merged_dir}"
    )

    # group by (plateID, wellID, fieldID)
    groups = {}
    for fname in all_files:
        meta = _parse_merged_filename(fname)
        key = (meta["plateID"], meta["wellID"], meta["fieldID"])
        groups.setdefault(key, []).append(fname)

    print(
        "[summarise_tracks_from_merged] Number of (plate, well, field) groups: "
        f"{len(groups)}"
    )

    cell_chan = settings.get("cell_channel", None)
    nucleus_chan = settings.get("nucleus_channel", None)
    pathogen_chan = settings.get("pathogen_channel", None)

    channels_list = settings.get("channels", [])

    pixels_per_um = settings.get("pixels_per_um", None)
    seconds_per_frame = settings.get("seconds_per_frame", None)

    n_channels = len(channels_list) if isinstance(channels_list, (list, tuple)) else None
    if n_channels is None or n_channels <= 0:
        raise ValueError(
            "settings['channels'] must be a non-empty list of channels used "
            "in merged arrays."
        )

    print(
        f"[summarise_tracks_from_merged] Channels={channels_list}, "
        f"cell_chan={cell_chan}, nucleus_chan={nucleus_chan}, "
        f"pathogen_chan={pathogen_chan}"
    )

    motility_dir = os.path.join(src, "motility_plots")
    os.makedirs(motility_dir, exist_ok=True)

    # Debug-plot one sample merged array with channel/mask labels
    sample_filename = sorted(all_files)[0]
    print(
        "[summarise_tracks_from_merged] Debug plotting planes for sample file: "
        f"{sample_filename}"
    )
    _debug_plot_merged_planes(
        src=src,
        sample_filename=sample_filename,
        n_channels=n_channels,
        nucleus_chan=nucleus_chan,
        pathogen_chan=pathogen_chan,
        out_dir=motility_dir,
    )

    # ------------------------------------------------------------------
    # Build measurements if not reusing from DB
    # ------------------------------------------------------------------
    if not loaded_from_db:
        worker_args = []
        for key, file_basenames in groups.items():
            worker_args.append(
                (src, file_basenames, n_channels, cell_chan, nucleus_chan, pathogen_chan)
            )

        if n_jobs is None:
            n_jobs = max(cpu_count() - 1, 1)
        print(f"[summarise_tracks_from_merged] Using n_jobs={n_jobs}")

        if n_jobs == 1:
            dfs = [_process_merged_group(args) for args in worker_args]
        else:
            with Pool(processes=n_jobs) as pool:
                dfs = pool.map(_process_merged_group, worker_args)

        all_df = (
            pd.concat([df for df in dfs if not df.empty], ignore_index=True)
            if dfs
            else pd.DataFrame()
        )
        if all_df.empty:
            raise RuntimeError("No measurements were produced from merged .npy files.")

        print(
            "[summarise_tracks_from_merged] Combined raw measurements: "
            f"shape={all_df.shape}, frames={all_df['frame'].nunique()}"
        )
        n_tracks_raw = (
            all_df[["plateID", "wellID", "fieldID", "cellID"]]
            .drop_duplicates()
            .shape[0]
        )
        print(
            "[summarise_tracks_from_merged] Unique tracks before smoothing: "
            f"{n_tracks_raw}"
        )

        # Clean tracks
        all_df = _smooth_tracks_and_features(
            all_df,
            max_displacement=max_displacement,
            zscore_thresh=zscore_thresh,
        )

        n_tracks_smoothed = (
            all_df[["plateID", "wellID", "fieldID", "cellID"]]
            .drop_duplicates()
            .shape[0]
        )
        print(
            "[summarise_tracks_from_merged] After smoothing: "
            f"shape={all_df.shape}, frames={all_df['frame'].nunique()}, "
            f"tracks={n_tracks_smoothed}"
        )
    else:
        # Already loaded smoothed measurements from DB (treated as ORIGINAL)
        n_frames_db = all_df["frame"].nunique()
        n_tracks_db = (
            all_df[["plateID", "wellID", "fieldID", "cellID"]]
            .drop_duplicates()
            .shape[0]
        )
        print(
            "[summarise_tracks_from_merged] Reusing ORIGINAL smoothed measurements "
            f"from DB: shape={all_df.shape}, frames={n_frames_db}, "
            f"tracks={n_tracks_db}"
        )

    # ------------------------------------------------------------------
    # Infection status per track (mask-based)
    #   - This is part of the ORIGINAL dataset.
    # ------------------------------------------------------------------
    if "infected" in all_df.columns:
        # Reuse existing infection labels (DB-reused or previous run),
        # but ensure no NaNs and correct dtype.
        all_df["infected"] = all_df["infected"].fillna(False).astype(bool)
    elif "n_pathogens" in all_df.columns:
        tmp = all_df[["plateID", "wellID", "fieldID", "cellID", "n_pathogens"]].copy()
        tmp["n_pathogens"] = tmp["n_pathogens"].fillna(0)
        infected = (
            tmp.groupby(["plateID", "wellID", "fieldID", "cellID"])["n_pathogens"]
            .max()
            .gt(0)
        )
        infected = infected.reset_index()
        infected = infected.rename(columns={"n_pathogens": "infected"})
        infected["infected"] = infected["infected"].astype(bool)

        all_df = all_df.merge(
            infected[["plateID", "wellID", "fieldID", "cellID", "infected"]],
            on=["plateID", "wellID", "fieldID", "cellID"],
            how="left",
        )
        all_df["infected"] = all_df["infected"].fillna(False).astype(bool)
    else:
        all_df["infected"] = False

    n_infected_tracks = (
        all_df[all_df["infected"]][["plateID", "wellID", "fieldID", "cellID"]]
        .drop_duplicates()
        .shape[0]
    )
    n_uninfected_tracks = (
        all_df[~all_df["infected"]][["plateID", "wellID", "fieldID", "cellID"]]
        .drop_duplicates()
        .shape[0]
    )
    print(
        "[summarise_tracks_from_merged] Tracks (mask-based): "
        f"infected={n_infected_tracks}, uninfected={n_uninfected_tracks}"
    )

    # ------------------------------------------------------------------
    # SNAPSHOT: ORIGINAL measurements (pre-QC) to be stored in SQLite.
    # This copy is never overridden by adjusted/QC'd data.
    # ------------------------------------------------------------------
    all_df_original = all_df.copy(deep=True)

    # ------------------------------------------------------------------
    # Optional infection-intensity QC (may create 'adjusted_infected')
    #   - This operates on all_df only (not on all_df_original).
    # ------------------------------------------------------------------
    infection_col = "infected"
    all_df, infection_col = _apply_infection_intensity_qc(
        all_df=all_df,
        settings=settings,
        infection_col=infection_col,
        motility_dir=motility_dir,
        pathogen_chan=pathogen_chan,
    )

    # ------------------------------------------------------------------
    # XGBoost ambiguous-band filtering (track-level)
    # ------------------------------------------------------------------
    if (
        settings.get("infection_intensity_qc", False)
        and str(settings.get("infection_intensity_strategy", "")).lower() == "xgboost"
        and settings.get("infection_xgb_drop_ambiguous", True)
    ):
        low = settings.get("infection_xgb_ambiguous_low", 0.25)
        high = settings.get("infection_xgb_ambiguous_high", 0.75)

        # Try to locate a probability column created by the QC step
        xgb_proba_col = settings.get("infection_xgb_proba_column", None)
        if xgb_proba_col is None:
            cand_cols = [
                c
                for c in all_df.columns
                if "xgb" in c.lower()
                and (
                    "proba" in c.lower()
                    or "prob" in c.lower()
                    or "score" in c.lower()
                )
            ]
            if not cand_cols:
                cand_cols = [
                    c
                    for c in all_df.columns
                    if "infection" in c.lower()
                    and (
                        "proba" in c.lower()
                        or "prob" in c.lower()
                        or "score" in c.lower()
                    )
                ]
            if cand_cols:
                xgb_proba_col = cand_cols[0]

        if xgb_proba_col and xgb_proba_col in all_df.columns:
            track_keys = ["plateID", "wellID", "fieldID", "cellID"]
            track_scores = (
                all_df[track_keys + [xgb_proba_col]]
                .groupby(track_keys)[xgb_proba_col]
                .mean()
                .reset_index()
            )

            ambiguous = track_scores[
                (track_scores[xgb_proba_col] > low)
                & (track_scores[xgb_proba_col] < high)
            ][track_keys]

            if not ambiguous.empty:
                before = all_df.shape[0]
                all_df = all_df.merge(
                    ambiguous.assign(_ambiguous_flag=1),
                    on=track_keys,
                    how="left",
                )
                all_df = all_df[all_df["_ambiguous_flag"].isna()].drop(
                    columns=["_ambiguous_flag"]
                )
                after = all_df.shape[0]
                print(
                    "[summarise_tracks_from_merged] Dropped "
                    f"{before - after} rows from {ambiguous.shape[0]} ambiguous "
                    f"XGBoost tracks ({low} < proba < {high})."
                )
        else:
            print(
                "[summarise_tracks_from_merged] WARNING: "
                "infection_xgb_drop_ambiguous is True, but no XGBoost "
                "probability/score column was found. Skipping ambiguous-track "
                "filtering."
            )

    # ------------------------------------------------------------------
    # Save ADJUSTED frame-level measurements to CSV ONLY.
    # This NEVER overwrites the SQLite original table.
    # ------------------------------------------------------------------
    try:
        qc_strategy = str(settings.get("infection_intensity_strategy", "none")).lower()
        if qc_strategy in {"", "none", "null"}:
            adjusted_basename = f"{db_table_name}_adjusted.csv"
        else:
            adjusted_basename = f"{db_table_name}_adjusted_{qc_strategy}.csv"

        adjusted_csv_path = os.path.join(measurements_dir, adjusted_basename)
        all_df.to_csv(adjusted_csv_path, index=False)
        print(
            "[summarise_tracks_from_merged] Saved ADJUSTED frame-level measurements "
            f"to CSV: {adjusted_csv_path}"
        )
    except Exception as e:
        print(
            f"[summarise_tracks_from_merged] WARNING: failed to save adjusted CSV "
            f"({e})"
        )

    # ------------------------------------------------------------------
    # Compute per-track velocities + per-well summary
    # ------------------------------------------------------------------
    (
        track_df_mask,
        per_well_tracks_mask,
        well_summary_mask,
        vel_unit_mask,
    ) = _compute_velocities_and_well_summary(
        all_df=all_df,
        settings=settings,
        infection_col="infected",
        pixels_per_um=pixels_per_um,
        seconds_per_frame=seconds_per_frame,
    )

    (
        track_df,
        per_well_tracks,
        well_summary_df,
        vel_unit,
    ) = _compute_velocities_and_well_summary(
        all_df=all_df,
        settings=settings,
        infection_col=infection_col,
        pixels_per_um=pixels_per_um,
        seconds_per_frame=seconds_per_frame,
    )

    # ------------------------------------------------------------------
    # Save to DB:
    #   - all_df_original (pre-QC snapshot) is written to db_table_name
    #   - well_summary_df is written as usual by _save_measurements_and_well_summary
    #   → adjusted labels NEVER touch the canonical measurements table.
    # ------------------------------------------------------------------
    measurements_dir, db_path = _save_measurements_and_well_summary(
        all_df=all_df_original,
        well_summary_df=well_summary_df,
        src=src,
        db_table_name=db_table_name,
    )

    # Feature–velocity correlation analysis (final labels, adjusted view)
    _feature_velocity_correlations(all_df, track_df, measurements_dir)

    # ------------------------------------------------------------------
    # Combined intensity + motility panels
    # ------------------------------------------------------------------
    qc_strategy = str(settings.get("infection_intensity_strategy", "none")).lower()

    # Intensity + motility panel for mask-based labels
    if settings.get("make_mask_panel", True):
        _make_intensity_motility_panel(
            all_df=all_df,
            infection_col="infected",
            track_df=track_df_mask,
            per_well_tracks=per_well_tracks_mask,
            n_channels=n_channels,
            motility_dir=motility_dir,
            pixels_per_um=pixels_per_um,
            seconds_per_frame=seconds_per_frame,
            vel_unit=vel_unit_mask,
            settings=settings,
            # encode both label type and QC strategy in the tag
            label_tag=f"mask_{qc_strategy}",
        )

    # Intensity + motility panel for adjusted labels (if distinct)
    if (
        settings.get("make_adjusted_panel", True)
        and infection_col in all_df.columns
        and infection_col != "infected"
    ):
        _make_intensity_motility_panel(
            all_df=all_df,
            infection_col=infection_col,
            track_df=track_df,
            per_well_tracks=per_well_tracks,
            n_channels=n_channels,
            motility_dir=motility_dir,
            pixels_per_um=pixels_per_um,
            seconds_per_frame=seconds_per_frame,
            vel_unit=vel_unit,
            settings=settings,
            label_tag=f"adjusted_{qc_strategy}",
        )

    return all_df