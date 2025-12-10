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
                * PCA/UMAP/t-SNE embedding (if strategy in {"pca","umap","tsne"})
                * XGBoost:
                    - probability separation histogram
                    - feature-importance barplot

    Supports global vs per-well QC scope via settings['infection_intensity_qc_scope']:
      - "global" (default): single QC payload used for all wells
      - "per_well": expects dicts keyed by (plateID, wellID) or "plateID_wellID"
                    for histogram / PCA-like strategies.
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
    # Panel / QC configuration
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
    qc_scope = str(settings.get("infection_intensity_qc_scope", "global")).lower()

    # Global / per-well QC payload containers
    hist_global = settings.get("infection_hist_data", None)
    hist_per_well = settings.get("infection_hist_data_per_well", None)

    pca_global = settings.get("infection_pca_data", None)
    pca_per_well = settings.get("infection_pca_data_per_well", None)

    xgb_data = settings.get("infection_xgb_importance", None)

    # Mask panel: small embedded QC PNG if available
    qc_panel_needed_mask = (
        is_mask_panel
        and qc_graphs_enabled
        and isinstance(qc_panel_path, str)
        and qc_panel_path
        and os.path.exists(qc_panel_path)
    )

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
    def _plot_hist_qc(ax, hdata):
        if not hdata:
            ax.set_visible(False)
            return
        try:
            intens_inf = np.asarray(hdata["intensities_inf"], dtype=float)
            intens_uninf = np.asarray(hdata["intensities_uninf"], dtype=float)
            bin_edges = np.asarray(hdata["bin_edges"], dtype=float)
            thr_val = float(hdata.get("thr_val", np.nan))
            intensity_col = hdata.get("intensity_col", "intensity")
        except Exception as e:
            print(f"[_make_intensity_motility_panel] Histogram payload invalid: {e}")
            ax.set_visible(False)
            return

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

    def _plot_pca_qc(ax, pdata):
        if not pdata:
            ax.set_visible(False)
            return
        try:
            coords = np.asarray(pdata["coords"], dtype=float)
            labels = np.asarray(pdata["labels"], dtype=bool)
        except Exception as e:
            print(f"[_make_intensity_motility_panel] PCA/UMAP/t-SNE payload invalid: {e}")
            ax.set_visible(False)
            return

        if coords.shape[1] < 2:
            ax.set_visible(False)
            return

        x = coords[:, 0]
        y = coords[:, 1]

        method = str(pdata.get("method_label", "") or "").strip()
        strategy_name = str(pdata.get("strategy", "") or "").strip().lower()
        # Fallback order: strategy -> method_label -> generic "PCA"
        if strategy_name in {"pca", "umap", "tsne"}:
            base = strategy_name.upper() if strategy_name != "tsne" else "t-SNE"
        elif method:
            base = method
        else:
            base = "PCA"

        # Plot
        ax.scatter(x[~labels], y[~labels], s=5, alpha=0.4, color="green", label="Uninfected")
        ax.scatter(x[labels], y[labels], s=5, alpha=0.4, color="red", label="Infected")
        ax.set_xlabel(f"{base} 1")
        ax.set_ylabel(f"{base} 2")
        ax.set_title(f"{base} of features\n(adjusted labels)")
        ax.legend(fontsize=7)

    def _plot_xgb_importance_qc(ax, xdata):
        if not xdata:
            ax.set_visible(False)
            return
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
        # per-cell probability distribution by adjusted infection label
        if "infection_prob" not in df_prob.columns:
            ax.set_visible(False)
            return

        cell_probs = (
            df_prob[key_cols + ["infection_prob", infection_col]]
            .groupby(key_cols, dropna=False)
            .agg({"infection_prob": "mean", infection_col: "max"})
            .reset_index()
        )
        cell_probs = cell_probs.replace([np.inf, -np.inf], np.nan)
        cell_probs = cell_probs.dropna(subset=["infection_prob"])
        if cell_probs.empty:
            ax.set_visible(False)
            return

        mask_inf = cell_probs[infection_col].astype(bool)
        probs_inf = cell_probs.loc[mask_inf, "infection_prob"].to_numpy()
        probs_uninf = cell_probs.loc[~mask_inf, "infection_prob"].to_numpy()

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
    # Per-well panels
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

        # ------------------------------------------------------------------
        # QC payload selection for THIS WELL
        # ------------------------------------------------------------------
        well_key_tuple = (plate_id, well_id)
        well_key_str = f"{plate_id}_{well_id}"

        # Histogram payload
        hdata = None
        if qc_scope == "per_well" and isinstance(hist_per_well, dict):
            hdata = hist_per_well.get(well_key_tuple)
            if hdata is None:
                hdata = hist_per_well.get(well_key_str)
        if hdata is None:
            hdata = hist_global

        # PCA/UMAP/t-SNE payload
        pdata = None
        if qc_scope == "per_well" and isinstance(pca_per_well, dict):
            pdata = pca_per_well.get(well_key_tuple)
            if pdata is None:
                pdata = pca_per_well.get(well_key_str)
        if pdata is None:
            pdata = pca_global

        has_hist = hdata is not None
        has_pca = pdata is not None
        has_xgb = xgb_data is not None

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

        # QC axes count for THIS WELL
        qc_axes_count = 0
        if is_adjusted_panel and qc_graphs_enabled:
            if qc_strategy == "histogram" and has_hist:
                qc_axes_count = 1
            elif qc_strategy in {"pca", "umap", "tsne"} and has_pca:
                qc_axes_count = 1
            elif qc_strategy in {"xgboost", "xgb"} and has_xgb:
                # probability separation + feature importance
                qc_axes_count = 2

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
                    "umap": "PCA/UMAP clustering",
                    "tsne": "t-SNE clustering",
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
            if qc_strategy == "histogram" and has_hist and axis_idx < len(axes):
                ax_hist = axes[axis_idx]
                axis_idx += 1
                _plot_hist_qc(ax_hist, hdata)

            elif qc_strategy in {"pca", "umap", "tsne"} and has_pca and axis_idx < len(axes):
                ax_pca = axes[axis_idx]
                axis_idx += 1
                _plot_pca_qc(ax_pca, pdata)

            elif qc_strategy in {"xgboost", "xgb"} and has_xgb:
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
            f"Infection panel – {panel_label} labels – method={method_label}, scope={qc_scope}\n{meta_tag}",
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
            f"({panel_label}, method={method_label}, scope={qc_scope}) for plate={plate_id}, well={well_id} "
            f"to {out_path}"
        )

def _apply_infection_intensity_qc(
    all_df,
    settings,
    infection_col,
    pathogen_chan,
    motility_dir,
):
    """
    Dispatch to different infection QC strategies based on
    settings['infection_intensity_strategy']:

        'histogram' / 'hist' / 'histagram'
            1D intensity histogram thresholding

        'pca'
            PCA or UMAP + k-means clustering

        'xgboost'
            Supervised XGBoost classifier on extreme intensities

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

    # If QC is disabled or there is no pathogen channel, do nothing.
    infection_intensity_qc = bool(settings.get("infection_intensity_qc", False))
    if (not infection_intensity_qc) or (pathogen_chan is None):
        print("[infection_intensity_qc] QC disabled or no pathogen channel; skipping.")
        return all_df, infection_col

    # Make sure output directory exists for plots
    os.makedirs(motility_dir, exist_ok=True)

    strategy = str(settings.get("infection_intensity_strategy", "histogram")).lower()

    if strategy in {"hist", "histogram", "histagram"}:
        return _infection_qc_histogram(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    if strategy in {"pca", "umap", "tsne"}:
        return _infection_qc_pca_clustering(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )
    if strategy in {"xgboost", "xgb"}:
        return _infection_qc_xgboost(
            all_df=all_df,
            settings=settings,
            infection_col=infection_col,
            pathogen_chan=pathogen_chan,
            motility_dir=motility_dir,
        )

    # Fallback if unknown strategy name
    print(
        "[infection_intensity_qc] Unknown strategy "
        f"{strategy!r}; falling back to 'histogram'."
    )
    return _infection_qc_histogram(
        all_df=all_df,
        settings=settings,
        infection_col=infection_col,
        pathogen_chan=pathogen_chan,
        motility_dir=motility_dir,
    )

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

def get_automated_motility_assay_default_settings(settings):
    if settings is None:
        settings = {}

    # array settings
    settings.setdefault('channels', [0, 1, 2, 3])
    settings.setdefault('cell_channel', 2)
    settings.setdefault('nucleus_channel', 0)
    settings.setdefault('pathogen_channel', 1)
    settings.setdefault('tracked_object', 'cell')
    settings.setdefault('reuse_existing_measurements', True)

    # filter settings
    settings.setdefault('n_jobs', -1)
    settings.setdefault('max_displacement', 50.0)
    settings.setdefault('zscore_thresh', 3.0)
    settings.setdefault('straightness_filter', False)
    settings.setdefault('straightness_threshold', 0.95)
    settings.setdefault('infection_intensity_qc', True)
    settings.setdefault('infection_intensity_strategy', 'xgb')  # 'pca' | 'umap' | 'tsne' | 'histogram' | 'xgb'
    settings.setdefault('infection_intensity_mode', "relabel")  # or 'remove'
    settings.setdefault('infection_intensity_qc_panel_path', None)
    settings.setdefault('infection_intensity_qc_graphs', True)
    settings.setdefault('db_table_name', "timelapse_object_measurements")
    settings.setdefault('infection_intensity_n_bins', 64)

    # motility plot settings
    settings.setdefault('pixels_per_um', 1.78)
    settings.setdefault('seconds_per_frame', 60)
    settings.setdefault('motility_xlim', (100, -100))
    settings.setdefault('motility_ylim', (100, -100))

    # xgboost settings
    settings.setdefault('infection_xgb_n_estimators', 200)
    settings.setdefault('infection_xgb_max_depth', 3)
    settings.setdefault('infection_xgb_learning_rate', 0.1)
    settings.setdefault('infection_xgb_subsample', 0.8)
    settings.setdefault('infection_xgb_colsample_bytree', 0.8)
    settings.setdefault('infection_xgb_reg_lambda', 1.0)
    settings.setdefault('infection_xgb_random_state', 42)
    settings.setdefault('infection_xgb_n_jobs', -1)
    settings.setdefault('infection_xgb_proba_threshold', 0.5)
    settings.setdefault('infection_xgb_margin', 0.15)
    settings.setdefault('infection_xgb_top_features', 20)
    settings.setdefault('infection_xgb_proba_column', 'infection_xgb_proba')
    settings.setdefault('infection_xgb_drop_ambiguous', True)
    settings.setdefault('infection_xgb_ambiguous_low', 0.25)
    settings.setdefault('infection_xgb_ambiguous_high', 0.75)
    settings.setdefault('infection_xgb_min_cells_per_class', 10)

    # PCA / embedding-common settings
    settings.setdefault('infection_pca_n_clusters', 2)
    settings.setdefault('infection_pca_random_state', 42)
    settings.setdefault('infection_pca_pathogen_weight', 2.0)
    settings.setdefault('infection_pca_log_intensity', False)
    settings.setdefault('infection_pca_max_cells', 50000)
    settings.setdefault('infection_pca_min_gt_separation', 0.2)
    settings.setdefault('infection_pca_min_silhouette', 0.05)

    # UMAP
    settings.setdefault('infection_pca_umap_search', True)
    settings.setdefault('infection_pca_umap_n_neighbors_grid', [5, 10, 15, 30])
    settings.setdefault('infection_pca_umap_min_dist_grid', [0.0, 0.05, 0.1, 0.3])
    # used if infection_pca_umap_search == False
    settings.setdefault('infection_pca_umap_n_neighbors', 15)
    settings.setdefault('infection_pca_umap_min_dist', 0.1)

    # t-SNE
    settings.setdefault('infection_pca_tsne_search', True)
    settings.setdefault('infection_pca_tsne_perplexity_grid', [15.0, 30.0, 45.0])
    settings.setdefault('infection_pca_tsne_learning_rate_grid', [200.0, 500.0])
    # used if infection_pca_tsne_search == False
    settings.setdefault('infection_pca_tsne_perplexity', 30.0)
    settings.setdefault('infection_intensity_qc_scope', "well")
    
    return settings

import numpy as np
import pandas as pd

def _infection_qc_histogram(
    all_df,
    settings,
    infection_col,
    pathogen_chan,
    motility_dir,
):
    """
    Intensity-based infection QC using a 1D histogram threshold on a
    pathogen-channel cell intensity feature.

    Supports global vs per-well scope via
    settings['infection_intensity_qc_scope'] ('global' | 'per_well').
    """
    import os
    import numpy as np
    import pandas as pd

    if all_df.empty:
        print("[infection_intensity_qc:hist] all_df is empty; skipping histogram QC.")
        return all_df, infection_col

    if infection_col not in all_df.columns:
        print(
            f"[infection_intensity_qc:hist] infection_col {infection_col!r} missing; "
            "skipping histogram QC."
        )
        return all_df, infection_col

    # Scope dispatch: global vs per-well
    qc_scope = str(settings.get("infection_intensity_qc_scope", "global")).lower()
    if qc_scope not in {"global", "per_well"}:
        qc_scope = "global"

    key_cols = ["plateID", "wellID", "fieldID", "cellID"]

    if qc_scope == "per_well":
        if not {"plateID", "wellID"}.issubset(all_df.columns):
            print(
                "[infection_intensity_qc:hist] per-well scope requested but plateID/wellID "
                "missing; falling back to global scope."
            )
        else:
            grouped = all_df.groupby(["plateID", "wellID"], sort=False, dropna=False)
            if grouped.ngroups > 0:
                hist_per_well = {}
                dfs_out = []
                original_scope = settings.get("infection_intensity_qc_scope", "global")
                settings["infection_intensity_qc_scope"] = "global"
                last_infection_col = infection_col

                for (plate_id, well_id), df_well in grouped:
                    df_well = df_well.copy()
                    df_well_out, inf_col_local = _infection_qc_histogram(
                        df_well,
                        settings,
                        infection_col,
                        pathogen_chan,
                        motility_dir,
                    )
                    hist_payload = settings.get("infection_hist_data", None)
                    if hist_payload is not None:
                        hist_per_well[(plate_id, well_id)] = hist_payload
                    dfs_out.append(df_well_out)
                    last_infection_col = inf_col_local

                settings["infection_intensity_qc_scope"] = original_scope
                settings["infection_hist_data_per_well"] = hist_per_well
                settings["infection_hist_data"] = None

                all_df = pd.concat(dfs_out, ignore_index=True)
                return all_df, last_infection_col

    mode = str(settings.get("infection_intensity_mode", "relabel")).lower()
    if mode not in {"relabel", "remove"}:
        print(
            f"[infection_intensity_qc:hist] Unsupported mode={mode!r}; "
            "expected 'relabel' or 'remove'. Skipping histogram QC."
        )
        return all_df, infection_col

    # Ensure required key columns exist
    for col in key_cols:
        if col not in all_df.columns:
            raise KeyError(
                f"[infection_intensity_qc:hist] Required column {col!r} not in all_df."
            )

    # Decide pathogen-channel intensity column
    intensity_col = None
    if pathogen_chan is not None:
        cand_int = [
            f"cell_p95_intensity_ch{pathogen_chan}",
            f"cell_max_intensity_ch{pathogen_chan}",
            f"cell_mean_intensity_ch{pathogen_chan}",
        ]
        for c in cand_int:
            if c in all_df.columns:
                intensity_col = c
                break

    if intensity_col is None:
        print(
            "[infection_intensity_qc:hist] No pathogen-channel cell_* intensity column "
            "found; skipping histogram QC."
        )
        return all_df, infection_col

    # Build per-cell table
    cols_for_group = key_cols + [intensity_col, infection_col]
    tmp = all_df[cols_for_group].copy()
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)

    group = tmp.groupby(key_cols, observed=True)
    cell_level = group[[intensity_col]].median(numeric_only=True).reset_index()
    inf_any = group[infection_col].max().reset_index()
    cell_level = cell_level.merge(inf_any, on=key_cols, how="left", suffixes=("", "_y"))

    if infection_col not in cell_level.columns:
        for cand in (f"{infection_col}_y", f"{infection_col}_x"):
            if cand in cell_level.columns:
                cell_level[infection_col] = cell_level[cand]
                break

    if infection_col not in cell_level.columns:
        print(
            f"[infection_intensity_qc:hist] Could not recover infection_col={infection_col!r} "
            "after aggregation; skipping histogram QC."
        )
        return all_df, infection_col

    cell_level[infection_col] = cell_level[infection_col].fillna(0).astype(bool)

    intens = cell_level[intensity_col].to_numpy(dtype=float)
    mask_finite = np.isfinite(intens)
    intens = intens[mask_finite]
    y_orig = cell_level.loc[mask_finite, infection_col].astype(bool).to_numpy()

    if intens.size < 40 or np.sum(y_orig) < 10 or np.sum(~y_orig) < 10:
        print(
            "[infection_intensity_qc:hist] Not enough cells with finite intensity "
            "and both classes represented; skipping histogram QC."
        )
        return all_df, infection_col

    n_bins = int(settings.get("infection_intensity_n_bins", 64))

    # Candidate thresholds from quantiles
    q_grid = np.linspace(0.05, 0.95, 50)
    thr_candidates = np.quantile(intens, q_grid)

    best_thr = None
    best_score = -np.inf

    for thr in np.unique(thr_candidates):
        pred = intens >= thr
        # True Positive Rate, True Negative Rate
        pos_mask = y_orig
        neg_mask = ~y_orig
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        tpr = (pred[pos_mask]).mean()
        tnr = (~pred[neg_mask]).mean()
        score = tpr + tnr - 1.0  # Youden's J
        if score > best_score:
            best_score = score
            best_thr = thr

    if best_thr is None:
        print(
            "[infection_intensity_qc:hist] Could not find a usable threshold; "
            "skipping histogram QC."
        )
        return all_df, infection_col

    thr_val = float(best_thr)

    print(
        "[infection_intensity_qc:hist] Selected intensity threshold "
        f"{thr_val:.4f} (Youden J={best_score:.3f})."
    )

    # Build histogram payload for panel plotting
    bin_edges = np.histogram_bin_edges(intens, bins=n_bins)
    intens_inf = intens[y_orig]
    intens_uninf = intens[~y_orig]

    settings["infection_hist_data"] = {
        "intensities_inf": intens_inf,
        "intensities_uninf": intens_uninf,
        "bin_edges": bin_edges,
        "thr_val": thr_val,
        "intensity_col": intensity_col,
    }

    settings["infection_intensity_qc_panel_path"] = None

    # Apply threshold at cell-level
    model_call = intens >= thr_val
    removed_ids = set()

    if mode == "relabel":
        adjusted = model_call.astype(bool)
        n_changed = int((adjusted != y_orig).sum())
        print(
            "[infection_intensity_qc:hist] Relabel mode: adjusted infection labels for "
            f"{n_changed} cells based on histogram threshold."
        )
        cell_level = cell_level.loc[mask_finite].reset_index(drop=True)
        cell_level["adjusted_infected"] = adjusted.astype(bool)

    else:  # mode == "remove"
        consistent = model_call == y_orig
        to_remove = ~consistent
        cell_level = cell_level.loc[mask_finite].reset_index(drop=True)
        if to_remove.any():
            removed = cell_level.loc[to_remove, key_cols]
            removed_ids = {
                (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
                for _, r in removed.iterrows()
            }
            cell_level = cell_level.loc[consistent].copy()
            print(
                "[infection_intensity_qc:hist] Remove mode: removed "
                f"{len(removed_ids)} cells with label vs threshold disagreement."
            )
        cell_level["adjusted_infected"] = cell_level[infection_col].astype(bool)

    # Map adjusted infection back to all_df
    for col in key_cols:
        all_df[col] = all_df[col].astype(cell_level[col].dtype)

    # Drop any pre-existing adjusted_infected columns
    cols_to_drop = [
        c
        for c in all_df.columns
        if c == "adjusted_infected" or c.startswith("adjusted_infected_")
    ]
    if cols_to_drop:
        all_df = all_df.drop(columns=cols_to_drop)

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

    mask_missing = all_df["adjusted_infected"].isna()
    if mask_missing.any():
        all_df.loc[mask_missing, "adjusted_infected"] = (
            all_df.loc[mask_missing, infection_col].astype(bool)
        )
    all_df["adjusted_infected"] = all_df["adjusted_infected"].astype(bool)

    infection_col = "adjusted_infected"

    # Optional debug plot
    try:
        if motility_dir is not None:
            import matplotlib.pyplot as plt

            os.makedirs(motility_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(
                intens_uninf,
                bins=bin_edges,
                alpha=0.6,
                label="Uninfected (orig)",
                density=True,
            )
            ax.hist(
                intens_inf,
                bins=bin_edges,
                alpha=0.6,
                label="Infected (orig)",
                density=True,
            )
            ax.axvline(thr_val, linestyle="--", linewidth=1.5)
            ax.set_xlabel(intensity_col)
            ax.set_ylabel("Density")
            ax.set_title("Infection QC – histogram threshold")
            ax.legend(fontsize=7, loc="best")

            out_png = os.path.join(motility_dir, "infection_histogram_qc.png")
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"[infection_intensity_qc:hist] Failed to save histogram QC plot: {e}")

    return all_df, infection_col

def _infection_qc_pca_clustering(
    all_df,
    settings,
    infection_col,
    pathogen_chan,
    motility_dir,
):
    """
    Embedding-based infection intensity QC (PCA/UMAP/t-SNE) with optional
    global vs per-well scope controlled by
    settings['infection_intensity_qc_scope'] ('global' | 'per_well').
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
                        n_clusters=2,
                        random_state=random_state,
                        n_init="auto"
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

    # Scope dispatch: global vs per-well
    qc_scope = str(settings.get("infection_intensity_qc_scope", "global")).lower()
    if qc_scope not in {"global", "per_well"}:
        qc_scope = "global"

    if qc_scope == "per_well":
        if not {"plateID", "wellID"}.issubset(all_df.columns):
            print(
                "[infection_intensity_qc:PCA] per-well scope requested but plateID/wellID "
                "missing; falling back to global scope."
            )
        else:
            # run embedding QC independently per (plateID, wellID)
            grouped = all_df.groupby(["plateID", "wellID"], sort=False, dropna=False)
            if grouped.ngroups > 0:
                pca_per_well = {}
                dfs_out = []
                original_scope = settings.get("infection_intensity_qc_scope", "global")
                settings["infection_intensity_qc_scope"] = "global"
                last_infection_col = infection_col

                for (plate_id, well_id), df_well in grouped:
                    df_well = df_well.copy()
                    df_well_out, inf_col_local = _infection_qc_pca_clustering(
                        df_well,
                        settings,
                        infection_col,
                        pathogen_chan,
                        motility_dir,
                    )
                    pca_payload = settings.get("infection_pca_data", None)
                    if pca_payload is not None:
                        pca_per_well[(plate_id, well_id)] = pca_payload
                    dfs_out.append(df_well_out)
                    last_infection_col = inf_col_local

                # restore scope and store per-well payloads
                settings["infection_intensity_qc_scope"] = original_scope
                settings["infection_pca_data_per_well"] = pca_per_well
                settings["infection_pca_data"] = None

                all_df = pd.concat(dfs_out, ignore_index=True)
                return all_df, last_infection_col

    mode = str(settings.get("infection_intensity_mode", "relabel")).lower()
    if mode not in {"relabel", "remove"}:
        print(
            f"[infection_intensity_qc:PCA] Unsupported mode={mode!r}; "
            "expected 'relabel' or 'remove'. Skipping embedding QC."
        )
        return all_df, infection_col

    # Decide embedding method from infection_intensity_strategy
    embed_method = str(settings.get("infection_intensity_strategy", "pca")).lower()
    if embed_method not in {"pca", "umap", "tsne"}:
        print(
            "[infection_intensity_qc:PCA] infection_intensity_strategy "
            f"{embed_method!r} is not an embedding method; using 'pca'."
        )
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
            embed_method = "pca"  # for consistent filenames etc.
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
        "strategy": embed_method,
    }

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

def _infection_qc_xgboost(all_df, settings, infection_col, pathogen_chan, motility_dir=None):
    """
    XGBoost-based infection intensity QC.

    Behaviour
    ---------
    - Aggregate to per-cell features (median over frames).
    - Use ONLY:
        * cell morphology features (cell_* without 'ch')
        * pathogen-channel cell_* features (matching ch{pathogen_chan})
    - Train an XGBoost classifier to separate infected vs uninfected.
    - Depending on infection_intensity_scope:
        * 'global'  : single model on all cells.
        * 'per_well': separate model per (plateID, wellID).
    - Depending on infection_intensity_mode:
        * 'relabel': labels follow the classifier (with a margin).
        * 'remove' : cells inconsistent with the classifier are removed.

    Side effects
    ------------
    - Adds per-cell probability column to all_df:
        settings['infection_xgb_proba_column'] (default 'infection_xgb_proba')
      where the probability is **always** "probability of being infected"
      (higher pathogen intensity), regardless of original encoding.
    - Adds adjusted infection label:
        all_df['adjusted_infected'] (bool)
      and returns infection_col = 'adjusted_infected'.
    - Stores feature-importance payload for the adjusted panel:
        settings['infection_xgb_importance'] = {
            'feature_names': [...],
            'feature_importances': [...],
            'scope': 'global' | 'per_well',
        }
    - Sets QC panel metadata (plots are drawn inside the panel function):
        settings['infection_intensity_qc_panel_type'] = 'xgboost'
        settings['infection_intensity_qc_panel_path'] = None
    """
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier

    if all_df.empty:
        print("[infection_intensity_qc:xgb] all_df is empty; skipping XGBoost QC.")
        return all_df, infection_col

    if infection_col not in all_df.columns:
        print(
            f"[infection_intensity_qc:xgb] infection_col {infection_col!r} missing; "
            "skipping XGBoost QC."
        )
        return all_df, infection_col

    # Mode: relabel vs remove
    mode = str(settings.get("infection_intensity_mode", "relabel")).lower()
    if mode not in {"relabel", "remove"}:
        print(
            f"[infection_intensity_qc:xgb] Unsupported mode={mode!r}; "
            "expected 'relabel' or 'remove'. Skipping XGBoost QC."
        )
        return all_df, infection_col

    # Scope: global vs per-well
    scope = str(settings.get("infection_intensity_scope", "global")).lower()
    if scope not in {"global", "per_well"}:
        scope = "global"

    key_cols = ["plateID", "wellID", "fieldID", "cellID"]
    for col in key_cols:
        if col not in all_df.columns:
            raise KeyError(
                f"[infection_intensity_qc:xgb] Required column {col!r} not in all_df."
            )

    # Drop any existing adjusted_infected to avoid *_x / *_y merges later
    cols_to_drop = [
        c for c in all_df.columns
        if c == "adjusted_infected" or c.startswith("adjusted_infected_")
    ]
    if cols_to_drop:
        all_df = all_df.drop(columns=cols_to_drop)

    proba_col = settings.get("infection_xgb_proba_column", "infection_xgb_proba")

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
            "[infection_intensity_qc:xgb] No numeric cell_* features found; "
            "skipping XGBoost QC."
        )
        return all_df, infection_col

    cols_for_group = key_cols + numeric_cols + [infection_col]
    tmp = all_df[cols_for_group].copy()
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)

    group = tmp.groupby(key_cols, observed=True)

    # median per cell over frames
    cell_level = group[numeric_cols].median(numeric_only=True).reset_index()

    # any frame infected => cell infected
    inf_any = group[infection_col].max().reset_index()
    cell_level = cell_level.merge(inf_any, on=key_cols, how="left", suffixes=("", "_y"))

    if infection_col not in cell_level.columns:
        for cand in (f"{infection_col}_y", f"{infection_col}_x"):
            if cand in cell_level.columns:
                cell_level[infection_col] = cell_level[cand]
                break

    if infection_col not in cell_level.columns:
        print(
            f"[infection_intensity_qc:xgb] Could not recover infection_col={infection_col!r} "
            "after aggregation; skipping XGBoost QC."
        )
        return all_df, infection_col

    cell_level[infection_col] = cell_level[infection_col].fillna(0).astype(bool)

    # ------------------------------------------------------------------
    # Feature selection: morphology + pathogen-channel ONLY
    # ------------------------------------------------------------------
    morph_cols = [
        c
        for c in numeric_cols
        if c.startswith("cell_") and ("ch" not in c.lower())
    ]
    path_cols = []
    if pathogen_chan is not None:
        token = f"ch{pathogen_chan}".lower()
        path_cols = [
            c
            for c in numeric_cols
            if c.startswith("cell_") and (token in c.lower())
        ]

    feature_cols = sorted(set(morph_cols + path_cols))

    if not feature_cols:
        print(
            "[infection_intensity_qc:xgb] No morphology + pathogen-channel features found; "
            "skipping XGBoost QC."
        )
        return all_df, infection_col

    # Pathogen intensity column (for orientation sanity check)
    intensity_col = None
    if pathogen_chan is not None:
        for cand in [
            f"cell_p95_intensity_ch{pathogen_chan}",
            f"cell_max_intensity_ch{pathogen_chan}",
            f"cell_mean_intensity_ch{pathogen_chan}",
        ]:
            if cand in cell_level.columns:
                intensity_col = cand
                break

    # Optional log1p on intensity-like features
    log_intensity = bool(
        settings.get(
            "infection_xgb_log_intensity",
            settings.get("infection_pca_log_intensity", False),
        )
    )
    cell_for_X = cell_level.copy()
    if log_intensity:
        for c in feature_cols:
            cl = c.lower()
            if (
                ("intensity" in cl)
                or ("p75" in cl)
                or ("p95" in cl)
                or ("max" in cl)
            ):
                vals = cell_for_X[c].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                if finite.any() and np.nanmin(vals[finite]) >= 0:
                    vals[finite] = np.log1p(vals[finite])
                    cell_for_X[c] = vals

    X_full = cell_for_X[feature_cols].to_numpy(dtype=float)
    y_full_raw = cell_level[infection_col].astype(bool).to_numpy()

    # intensity vector for orientation check
    intensity_vec = None
    if intensity_col is not None:
        intensity_vec = cell_level[intensity_col].to_numpy(dtype=float)

    # Impute feature-wise medians
    for j in range(X_full.shape[1]):
        col = X_full[:, j]
        m = np.isfinite(col)
        if not m.any():
            X_full[:, j] = 0.0
        else:
            med = np.nanmedian(col[m])
            col[~m] = med
            X_full[:, j] = col

    if X_full.shape[0] < 10:
        print(
            "[infection_intensity_qc:xgb] Fewer than 10 cells after filtering; "
            "skipping XGBoost QC."
        )
        return all_df, infection_col

    # Optional subsampling for speed (global limit)
    max_cells = int(
        settings.get(
            "infection_xgb_max_cells",
            settings.get("infection_pca_max_cells", 50000),
        )
    )
    if X_full.shape[0] > max_cells:
        rng = np.random.default_rng(0)
        idx = rng.choice(np.arange(X_full.shape[0]), size=max_cells, replace=False)
        X_full = X_full[idx]
        cell_level = cell_level.iloc[idx].reset_index(drop=True)
        y_full_raw = y_full_raw[idx]
        if intensity_vec is not None:
            intensity_vec = intensity_vec[idx]

    # ------------------------------------------------------------------
    # Orientation sanity check:
    # ensure the "positive" class corresponds to higher pathogen intensity.
    # If mean intensity(infected=True) < mean intensity(infected=False),
    # flip labels for training and later flip probabilities.
    # ------------------------------------------------------------------
    y_full = y_full_raw.copy()
    flip_positive = False
    if intensity_vec is not None:
        mask_inf = y_full_raw & np.isfinite(intensity_vec)
        mask_uninf = (~y_full_raw) & np.isfinite(intensity_vec)
        if mask_inf.sum() >= 5 and mask_uninf.sum() >= 5:
            mean_inf_int = float(np.nanmean(intensity_vec[mask_inf]))
            mean_uninf_int = float(np.nanmean(intensity_vec[mask_uninf]))
            if mean_inf_int < mean_uninf_int:
                flip_positive = True
                y_full = ~y_full_raw
                print(
                    "[infection_intensity_qc:xgb] Detected lower pathogen intensity in "
                    "cells labeled infected=True than in infected=False; "
                    "flipping labels for XGBoost so that the positive class "
                    "corresponds to higher pathogen intensity."
                )

    # ------------------------------------------------------------------
    # Helper: train a single XGBoost model & compute *infected* probabilities
    # ------------------------------------------------------------------
    def _train_xgb(X, y, intensity_subset=None):
        """
        Train XGB where y is already oriented such that y==1 should
        correspond to 'infected' (higher pathogen intensity).
        Returns probabilities P(infected).
        """
        n_estimators = int(settings.get("infection_xgb_n_estimators", 200))
        max_depth = int(settings.get("infection_xgb_max_depth", 3))
        learning_rate = float(settings.get("infection_xgb_learning_rate", 0.1))
        subsample = float(settings.get("infection_xgb_subsample", 0.8))
        colsample_bytree = float(settings.get("infection_xgb_colsample_bytree", 0.8))
        reg_lambda = float(settings.get("infection_xgb_reg_lambda", 1.0))
        random_state = int(settings.get("infection_xgb_random_state", 42))
        n_jobs = int(settings.get("infection_xgb_n_jobs", -1))

        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
        clf.fit(X, y.astype(int))
        proba_pos = clf.predict_proba(X)[:, 1]  # probability for class 1 in y

        # At this point, y has already been oriented so that 1 == infected.
        # No further flipping here.
        return clf, proba_pos

    # Threshold logic for relabeling
    thr = float(settings.get("infection_xgb_proba_threshold", 0.5))
    margin = float(settings.get("infection_xgb_margin", 0.0))
    low_cut = thr - margin
    high_cut = thr + margin

    def _apply_thresholds(p, y_orig_oriented):
        """
        p: P(infected)
        y_orig_oriented: original labels used for training (after orientation fix).
        """
        p = np.asarray(p, dtype=float)
        y_new = y_orig_oriented.copy()
        # confidently uninfected
        y_new[p <= low_cut] = False
        # confidently infected
        y_new[p >= high_cut] = True
        # mid-range keep original label
        return y_new

    min_cells_class = int(settings.get("infection_xgb_min_cells_per_class", 10))
    amb_low = float(settings.get("infection_xgb_ambiguous_low", 0.25))
    amb_high = float(settings.get("infection_xgb_ambiguous_high", 0.75))
    drop_ambig = bool(settings.get("infection_xgb_drop_ambiguous", True))

    # ------------------------------------------------------------------
    # Train models (global or per-well)
    # ------------------------------------------------------------------
    cell_level["adjusted_infected"] = cell_level[infection_col].astype(bool)
    cell_level[proba_col] = np.nan

    feature_importance_sum = None
    n_importance_models = 0

    if scope == "global":
        X = X_full
        y = y_full
        n_pos = int(y.sum())
        n_neg = int((~y).sum())

        if n_pos < min_cells_class or n_neg < min_cells_class:
            print(
                "[infection_intensity_qc:xgb] GLOBAL: too few cells per class "
                f"(pos={n_pos}, neg={n_neg}); skipping XGBoost QC."
            )
        else:
            clf, proba_inf = _train_xgb(X, y, intensity_vec)
            cell_level[proba_col] = proba_inf
            new_labels = _apply_thresholds(proba_inf, y)

            if mode == "relabel":
                cell_level["adjusted_infected"] = new_labels.astype(bool)
            else:  # 'remove' handled later using disagreement
                cell_level["_xgb_consistent"] = (new_labels == y)

            if hasattr(clf, "feature_importances_"):
                fi = clf.feature_importances_
                feature_importance_sum = fi.copy()
                n_importance_models = 1

            print(
                "[infection_intensity_qc:xgb] XGBoost trained on "
                f"{len(y)} cells (pos={n_pos}, neg={n_neg}); "
                f"thr={thr:.2f}, amb_low={amb_low:.2f}, amb_high={amb_high:.2f}, "
                f"drop_ambiguous={drop_ambig}."
            )
    else:
        # per-well models
        grouped = cell_level.groupby(["plateID", "wellID"], sort=False)

        for (plate_id, well_id), df_g in grouped:
            idx = df_g.index.to_numpy()
            X = X_full[idx]
            y = y_full[idx]
            n_pos = int(y.sum())
            n_neg = int((~y).sum())

            if n_pos < min_cells_class or n_neg < min_cells_class:
                print(
                    "[infection_intensity_qc:xgb] Skipping well "
                    f"{plate_id}_{well_id}: too few cells per class "
                    f"(pos={n_pos}, neg={n_neg})."
                )
                continue

            intens_subset = None
            if intensity_vec is not None:
                intens_subset = intensity_vec[idx]

            clf, proba_inf = _train_xgb(X, y, intens_subset)
            cell_level.loc[idx, proba_col] = proba_inf
            new_labels = _apply_thresholds(proba_inf, y)

            if mode == "relabel":
                cell_level.loc[idx, "adjusted_infected"] = new_labels.astype(bool)
            else:
                cell_level.loc[idx, "_xgb_consistent"] = (new_labels == y)

            if hasattr(clf, "feature_importances_"):
                fi = clf.feature_importances_
                if feature_importance_sum is None:
                    feature_importance_sum = np.zeros_like(fi, dtype=float)
                feature_importance_sum += fi
                n_importance_models += 1

            print(
                "[infection_intensity_qc:xgb] XGBoost trained on "
                f"{len(y)} cells (pos={n_pos}, neg={n_neg}) for well "
                f"{plate_id}_{well_id}; thr={thr:.2f}, "
                f"amb_low={amb_low:.2f}, amb_high={amb_high:.2f}, "
                f"drop_ambiguous={drop_ambig}."
            )

    # ------------------------------------------------------------------
    # Remove inconsistent cells in 'remove' mode
    # ------------------------------------------------------------------
    removed_ids = set()
    if mode == "remove" and "_xgb_consistent" in cell_level.columns:
        consistent_mask = cell_level["_xgb_consistent"].astype(bool)
        to_remove = ~consistent_mask
        if to_remove.any():
            removed = cell_level.loc[to_remove, key_cols]
            removed_ids = {
                (r["plateID"], r["wellID"], r["fieldID"], r["cellID"])
                for _, r in removed.iterrows()
            }
            cell_level = cell_level.loc[consistent_mask].copy()
            print(
                "[infection_intensity_qc:xgb] Remove mode: removed "
                f"{len(removed_ids)} cells with classifier vs label disagreement."
            )
        cell_level["adjusted_infected"] = cell_level[infection_col].astype(bool)

    # ------------------------------------------------------------------
    # Map adjusted infection + probabilities back to frame-level all_df
    # ------------------------------------------------------------------
    for col in key_cols:
        all_df[col] = all_df[col].astype(cell_level[col].dtype)

    merge_cols = key_cols + ["adjusted_infected", proba_col]
    all_df = all_df.merge(
        cell_level[merge_cols],
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
    # Store feature-importance payload for the panel QC
    # ------------------------------------------------------------------
    if feature_importance_sum is not None and n_importance_models > 0:
        fi_mean = feature_importance_sum / float(n_importance_models)
        order = np.argsort(fi_mean)[::-1]
        top_k = int(settings.get("infection_xgb_top_features", 20))
        top_idx = order[:top_k]

        feat_names = [feature_cols[i] for i in top_idx]
        feat_vals = fi_mean[top_idx].tolist()

        settings["infection_xgb_importance"] = {
            "feature_names": feat_names,
            "feature_importances": feat_vals,
            "scope": scope,
        }
    else:
        settings["infection_xgb_importance"] = {
            "feature_names": [],
            "feature_importances": [],
            "scope": scope,
        }

    # We now draw QC plots directly in the adjusted panel
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

    if settings['infection_intensity_qc_scope'] == "well":
        settings['infection_intensity_qc_scope'] = "per_well"
        
    if settings['infection_intensity_qc_scope'] == "all":
        settings['infection_intensity_qc_scope'] = "global"
        
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