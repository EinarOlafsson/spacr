import cv2, os, re, glob, random, btrack
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display
from IPython.display import Image as ipyimage
import trackpy as tp
from btrack import datasets as btrack_datasets
from skimage.measure import regionprops

from .logger import log_function_call

#from .plot import _visualize_and_save_timelapse_stack_with_tracks
#from .utils import _masks_to_masks_stack

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

def _save_mask_timelapse_as_gif(masks, path, cmap, norm, filenames):
    """
    Save a timelapse of masks as a GIF.

    Parameters:
    masks (list): List of mask frames.
    path (str): Path to save the GIF.
    cmap: Colormap for displaying the masks.
    norm: Normalization for the masks.
    filenames (list): List of filenames corresponding to each mask frame.

    Returns:
    None
    """
    def _update(frame):
        """
        Update the plot with the given frame.

        Parameters:
        frame (int): The frame number to update the plot with.

        Returns:
        None
        """
        nonlocal filename_text_obj
        if filename_text_obj is not None:
            filename_text_obj.remove()
        ax.clear()
        ax.axis('off')
        current_mask = masks[frame]
        ax.imshow(current_mask, cmap=cmap, norm=norm)
        ax.set_title(f'Frame: {frame}', fontsize=24, color='white')
        filename_text = filenames[frame]
        filename_text_obj = fig.text(0.5, 0.01, filename_text, ha='center', va='center', fontsize=20, color='white')
        for label_value in np.unique(current_mask):
            if label_value == 0: continue  # Skip background
            y, x = np.mean(np.where(current_mask == label_value), axis=1)
            ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

    fig, ax = plt.subplots(figsize=(50, 50), facecolor='black')
    ax.set_facecolor('black')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    filename_text_obj = None
    anim = FuncAnimation(fig, _update, frames=len(masks), blit=False)
    anim.save(path, writer='pillow', fps=2, dpi=80)  # Adjust DPI for size/quality
    plt.close(fig)
    print(f'Saved timelapse to {path}')

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
    _save_mask_timelapse_as_gif(masks, save_path_gif, cmap, norm, filenames)
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
    """
    Prepare the mask array for object tracking.

    Args:
        mask_array (ndarray): Array of binary masks representing objects.

    Returns:
        DataFrame: DataFrame containing information about each object in the mask array.
            The DataFrame has the following columns:
            - frame: The frame number.
            - y: The y-coordinate of the object's centroid.
            - x: The x-coordinate of the object's centroid.
            - mass: The area of the object.
            - original_label: The original label of the object.

    """
    frames = []
    for t, frame in enumerate(mask_array):
        props = regionprops(frame)
        for obj in props:
            # Include 'label' in the dictionary to capture the original label of the object
            frames.append({
                'frame': t, 
                'y': obj.centroid[0], 
                'x': obj.centroid[1], 
                'mass': obj.area,
                'original_label': obj.label  # Capture the original label
            })
    return pd.DataFrame(frames)

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

def _facilitate_trackin_with_adaptive_removal(masks, search_range=500, max_attempts=100, memory=3):
    """
    Facilitates object tracking with adaptive removal.

    Args:
        masks (numpy.ndarray): Array of binary masks representing objects in each frame.
        search_range (int, optional): Maximum distance objects can move between frames. Defaults to 500.
        max_attempts (int, optional): Maximum number of attempts to track objects. Defaults to 100.
        memory (int, optional): Number of frames to remember when linking tracks. Defaults to 3.

    Returns:
        tuple: A tuple containing the updated masks, features, and tracks_df.
            masks (numpy.ndarray): Updated array of binary masks.
            features (pandas.DataFrame): DataFrame containing features for object tracking.
            tracks_df (pandas.DataFrame): DataFrame containing the tracked object trajectories.

    Raises:
        Exception: If tracking fails after the maximum number of attempts.

    """
    attempts = 0
    first_frame = masks[0]
    starting_objects = np.unique(first_frame[first_frame != 0])
    while attempts < max_attempts:
        try:
            masks = _remove_objects_from_first_frame(masks, 10)
            first_frame = masks[0]
            objects = np.unique(first_frame[first_frame != 0])
            print(len(objects))
            features = _prepare_for_tracking(masks)
            tracks_df = tp.link(features, search_range=search_range, memory=memory)
            print(f"Success with {len(objects)} objects, started with {len(starting_objects)} objects")
            return masks, features, tracks_df
        except Exception as e:  # Consider catching a more specific exception if possible
            print(f"Retrying with fewer objects. Exception: {e}", flush=True)
        finally:
            attempts += 1
    print(f"Failed to track objects after {max_attempts} attempts. Consider adjusting parameters.")
    return None, None, None

def _trackpy_track_cells(src, name, batch_filenames, object_type, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, mode):
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
        
        if timelapse_displacement is None:
            features = _prepare_for_tracking(masks)
            timelapse_displacement = _find_optimal_search_range(features, initial_search_range=500, increment=10, max_attempts=49, memory=3)
            if timelapse_displacement is None:
                timelapse_displacement = 50

        masks, features, tracks_df = _facilitate_trackin_with_adaptive_removal(masks, search_range=timelapse_displacement, max_attempts=100, memory=timelapse_memory)

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

def _btrack_track_cells(src, name, batch_filenames, object_type, plot, save, masks_3D, mode, timelapse_remove_transient, radius=100, workers=10):
    """
    Track cells using the btrack library.

    Args:
        src (str): The source file path.
        name (str): The name of the track.
        batch_filenames (list): List of batch filenames.
        object_type (str): The type of object to track.
        plot (bool): Whether to plot the tracks.
        save (bool): Whether to save the tracks.
        masks_3D (ndarray): 3D array of masks.
        mode (str): The tracking mode.
        timelapse_remove_transient (bool): Whether to remove transient tracks.
        radius (int, optional): The maximum search radius. Defaults to 100.
        workers (int, optional): The number of workers. Defaults to 10.

    Returns:
        ndarray: The mask stack.

    """
    
    from .plot import _visualize_and_save_timelapse_stack_with_tracks
    from .utils import _masks_to_masks_stack
    
    CONFIG_FILE = btrack_datasets.cell_config()
    frame, width, height = masks_3D.shape

    FEATURES = ["area", "major_axis_length", "minor_axis_length", "orientation", "solidity"]
    objects = btrack.utils.segmentation_to_objects(masks_3D, properties=tuple(FEATURES), num_workers=workers)

    # initialise a tracker session using a context manager
    with btrack.BayesianTracker() as tracker:
        tracker.configure(CONFIG_FILE) # configure the tracker using a config file
        tracker.max_search_radius = radius
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        #tracker.tracking_updates = ["MOTION"]
        tracker.features = FEATURES
        tracker.append(objects) # append the objects to be tracked
        tracker.volume=((0, height), (0, width)) # set the tracking volume
        tracker.track(step_size=100) # track them (in interactive mode)
        tracker.optimize() # generate hypotheses and run the global optimizer
        #data, properties, graph = tracker.to_napari() # get the tracks in a format for napari visualization
        tracks = tracker.tracks # store the tracks
        #cfg = tracker.configuration # store the configuration

    # Process the track data to create a DataFrame
    track_data = []
    for track in tracks:
        for t, x, y, z in zip(track.t, track.x, track.y, track.z):
            track_data.append({
                'track_id': track.ID,
                'frame': t,
                'x': x,
                'y': y,
                'z': z
            })
    # Convert track data to a DataFrame
    tracks_df = pd.DataFrame(track_data)
    if timelapse_remove_transient:
        tracks_df = _filter_short_tracks(tracks_df, min_length=len(masks_3D))

    objects_df = _prepare_for_tracking(masks_3D)

    # Optional: If necessary, round 'x' and 'y' to ensure matching precision
    tracks_df['x'] = tracks_df['x'].round(decimals=2)
    tracks_df['y'] = tracks_df['y'].round(decimals=2)
    objects_df['x'] = objects_df['x'].round(decimals=2)
    objects_df['y'] = objects_df['y'].round(decimals=2)

    # Merge the DataFrames on 'frame', 'x', and 'y'
    merged_df = pd.merge(tracks_df, objects_df, on=['frame', 'x', 'y'], how='inner')
    final_df = merged_df[['track_id', 'frame', 'x', 'y', 'original_label']]

    masks = _relabel_masks_based_on_tracks(masks_3D, final_df)
    tracks_path = os.path.join(os.path.dirname(src), 'tracks')
    os.makedirs(tracks_path, exist_ok=True)
    final_df.to_csv(os.path.join(tracks_path, f'btrack_tracks_{object_type}_{name}.csv'), index=False)
    if plot or save:
        _visualize_and_save_timelapse_stack_with_tracks(masks, final_df, save, src, name, plot, batch_filenames, object_type, mode)

    mask_stack = _masks_to_masks_stack(masks)
    return mask_stack
