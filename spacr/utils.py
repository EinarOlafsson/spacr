
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

from collections import defaultdict, OrderedDict
from PIL import Image
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from functools import reduce
from IPython.display import display

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

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl

from scipy import stats
import scipy.ndimage as ndi
from scipy.spatial import distance
from scipy.stats import fisher_exact, f_oneway, kruskal
from scipy.ndimage.filters import gaussian_filter
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

def _generate_mask_random_cmap(mask):
    """
    Generate a random colormap based on the unique labels in the given mask.

    Args:
    mask (ndarray): The mask array containing unique labels.

    Returns:
    ListedColormap: A random colormap generated based on the unique labels in the mask.
    """
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def filepaths_to_database(img_paths, settings, source_folder, crop_mode):
    """
    Insert image paths and metadata into a SQLite database.

    Args:
        img_paths (list of str): Image file paths to insert.
        settings (dict): Configuration dictionary. Must contain a 'timelapse' key (bool).
        source_folder (str): Folder containing the SQLite database.
        crop_mode (str): One of 'cell', 'nucleus', 'pathogen', or 'cytoplasm'.

    Raises:
        sqlite3.OperationalError: On database connection or write errors.
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
    """
    Store image paths and metadata in a SQLite database.

    Args:
        img_paths (list of str): Paths to image files.
        source_folder (str): Folder where the database is stored.
        settings (dict): Must include 'dataset' and 'cam_type' keys.

    Raises:
        sqlite3.OperationalError: On database errors.
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
    """
    Save activation correlation data to a SQLite database.

    Args:
        df (pd.DataFrame): DataFrame with correlation data. Must include 'file_name'.
        img_paths (list): List of image file paths.
        source_folder (str): Folder where the database is stored.
        settings (dict): Must include 'dataset' and 'cam_type' keys.

    Raises:
        sqlite3.OperationalError: On database errors.
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

def calculate_activation_correlations(inputs, activation_maps, file_names, manders_thresholds=[15, 50, 75]):
    """
    Calculates Pearson and Manders correlations between input image channels and activation map channels.
    
    Args:
        inputs: A batch of input images, Tensor of shape (batch_size, channels, height, width)
        activation_maps: A batch of activation maps, Tensor of shape (batch_size, channels, height, width)
        file_names: List of file names corresponding to each image in the batch.
        manders_thresholds: List of intensity percentiles to calculate Manders correlation.
        
    Returns:
        df_correlations: A DataFrame with columns for pairwise correlations (Pearson and Manders) 
                         between input channels and activation map channels.
    """
    
    # Ensure tensors are detached and moved to CPU before converting to numpy
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
    """
    Convert a CSV file with 'settings_key' and 'settings_value' columns into a dictionary.
    Handles special cases where values are lists, tuples, booleans, None, integers, floats, and nested dictionaries.

    Args:
        csv_file_path (str): The path to the CSV file.
        show (bool): Whether to display the dataframe (for debugging).
        setting_key (str): The name of the column that contains the setting keys.
        setting_value (str): The name of the column that contains the setting values.

    Returns:
        dict: A dictionary where 'settings_key' are the keys and 'settings_value' are the values.
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
    """
    Save a dictionary of settings to a CSV file.

    This function takes a dictionary of settings, processes it, and saves it 
    as a CSV file in a specified directory. It also provides an option to 
    display the settings as a DataFrame.

    Args:
        settings (dict): A dictionary containing the settings to be saved.
        name (str, optional): The base name for the output CSV file. Defaults to 'settings'.
        show (bool, optional): If True, displays the settings as a DataFrame. Defaults to False.

    Raises:
        KeyError: If the 'src' key is not present in the settings dictionary.

    Notes:
        - If the 'src' key in the settings dictionary is a list, the first element 
          is used as the source directory, and the file name is appended with '_list'.
        - If the 'test_mode' key exists in the settings dictionary, it is set to False.
        - If the 'plot' key exists in the settings dictionary, it is set to False.
        - A directory named 'settings' is created inside the source directory if it does not exist.
        - The settings are saved as a CSV file in the 'settings' directory.
        - The file path where the settings are saved is printed.

    Example:
        >>> settings = {
        ...     'src': '/path/to/source',
        ...     'test_mode': True,
        ...     'plot': True,
        ...     'param1': 42
        ... }
        >>> save_settings(settings, name='experiment1', show=True)
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
    """
    Prints the progress of a file processing operation along with estimated time information.

    Args:
        files_processed (int or list): The number of files processed so far. If a list is provided, 
            its unique length will be used.
        files_to_process (int or list): The total number of files to process. If a list is provided, 
            its unique length will be used.
        n_jobs (int): The number of parallel jobs being used for processing.
        time_ls (list, optional): A list of time durations for processing batches or files. Used to 
            calculate average time and estimated time left. Defaults to None.
        batch_size (int or list, optional): The size of each batch being processed. If a list is 
            provided, its length will be used. Defaults to None.
        operation_type (str, optional): A string describing the type of operation being performed. 
            Defaults to an empty string.

    Returns:
        None: This function prints the progress and time information to the console.

    Notes:
        - If `time_ls` is provided, the function calculates the average time per batch or file and 
          estimates the time remaining for the operation.
        - If `batch_size` is provided, the function calculates the average time per image within 
          a batch.
        - Handles cases where inputs are lists or non-integer types by converting them to integers 
          or calculating their lengths.
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
        except:
            files_processed = 0
    if not isinstance(files_to_process, int):
        try:
            files_to_process = int(files_to_process)
        except:
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
    """
    Resets the multiprocessing start method based on the operating system.
    On Windows, the start method is set to 'spawn' if it is not already set.
    On Linux and macOS (Darwin), the start method is set to 'fork' if it is not already set.
    This function ensures compatibility with the multiprocessing module by
    enforcing the appropriate start method for the current platform.
    Note:
        - The function uses `get_start_method` to retrieve the current start method.
        - The `set_start_method` function is called with `force=True` to override
          the existing start method if necessary.
    Raises:
        ValueError: If an invalid start method is encountered or if the start
                    method cannot be set for some reason.
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
    """
    Check if the given process is a multiprocessing process.

    This function examines the command-line arguments of the provided process
    to determine if it is associated with Python's multiprocessing module.

    Args:
        process (psutil.Process): A psutil Process object representing the process to check.

    Returns:
        bool: True if the process is a multiprocessing process, False otherwise.

    Raises:
        psutil.NoSuchProcess: If the process no longer exists.
        psutil.AccessDenied: If access to the process information is denied.
        psutil.ZombieProcess: If the process is a zombie process.
    """
    """ Check if the process is a multiprocessing process. """
    try:
        for cmd in process.cmdline():
            if 'multiprocessing' in cmd:
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return False

def close_file_descriptors():
    """
    Closes all open file descriptors starting from 3 up to the soft limit 
    of the maximum number of file descriptors allowed for the process.

    This function is useful for cleaning up resources by ensuring that 
    no unnecessary file descriptors remain open. It skips standard input, 
    output, and error (file descriptors 0, 1, and 2).

    Exceptions during the closing of file descriptors are caught and ignored.

    Note:
        The function uses the `resource` module to retrieve the soft limit 
        for the maximum number of file descriptors.

    Raises:
        OSError: If an error occurs while closing a file descriptor, it is 
        caught and ignored.
    """
    """ Close file descriptors and shared memory objects. """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for fd in range(3, soft):
        try:
            os.close(fd)
        except OSError:
            pass

def close_multiprocessing_processes():
    """
    Close all multiprocessing processes and clean up associated resources.
    This function iterates through all running processes and terminates any
    that are identified as multiprocessing processes, excluding the current
    process. It waits for up to 5 seconds for each process to terminate and
    logs the termination status. Additionally, it handles exceptions that may
    occur during the termination process, such as access denial or the process
    no longer existing.
    After terminating the processes, it ensures that any open file descriptors
    are properly closed.
    Note:
        This function relies on the `psutil` library to inspect and manage
        processes, and assumes the existence of helper functions:
        - `is_multiprocessing_process(proc)`: Determines if a process is a
          multiprocessing process.
        - `close_file_descriptors()`: Closes any open file descriptors.
    """
    """ Close all multiprocessing processes. """
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
    def check_mask_folder(src, mask_fldr):
        """
        Checks the status of a mask folder within a given source directory.
        This function verifies whether the specified mask folder exists and 
        whether the number of mask files matches the number of stack files 
        in the corresponding stack folder. It returns a boolean indicating 
        whether further processing is needed.
        Args:
            src (str): The source directory containing the 'masks' and 'stack' subdirectories.
            mask_fldr (str): The name of the mask folder to check within the 'masks' directory.
        Returns:
            bool: 
                - True if the mask folder does not exist or if the number of mask files 
                  does not match the number of stack files.
                - False if the mask folder exists and all masks have been generated.
        Side Effects:
            Prints a message if all masks have been generated for the specified mask folder.
        Example:
            >>> check_mask_folder('/home/user/data', 'experiment1')
            All masks have been generated for experiment1
            False
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
    """
    Smooths the convex hull of a set of 2D points using spline interpolation.

    Args:
        cluster_data (numpy.ndarray): A 2D array of shape (n_points, 2) representing 
                                       the coordinates of the points in the cluster.

    Returns:
        tuple: Two 1D numpy arrays representing the x and y coordinates of the smoothed 
               convex hull, respectively. Each array contains 100 points.
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
    """
    Generate an RGB image from the specified channels of the input image.

    Args:
        image (ndarray): The input image.
        channels (list): List of channel indices to use for RGB.

    Returns:
        rgb_image (ndarray): The generated RGB image.
    """
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    for i, chan in enumerate(channels):
        if chan < image.shape[2]:
            rgb_image[:, :, i] = image[:, :, chan]
    return rgb_image

def _outline_and_overlay(image, rgb_image, mask_dims, outline_colors, outline_thickness):
    """
    Generate outlines for specified mask dimensions and overlay them onto an RGB image.

    This function processes a multi-dimensional mask to extract contours for each specified
    dimension, creates outlines with specified thickness, and overlays these outlines onto
    the provided RGB image using the given colors.

    Args:
        image (numpy.ndarray): The input image containing multi-dimensional masks.
        rgb_image (numpy.ndarray): The RGB image onto which the outlines will be overlaid.
        mask_dims (list[int]): A list of dimensions (indices) in the mask to process.
        outline_colors (list[tuple[int, int, int]]): A list of RGB color tuples for the outlines.
        outline_thickness (int): The thickness of the outlines to be drawn.

    Returns:
        tuple:
            - overlayed_image (numpy.ndarray): The RGB image with outlines overlaid.
            - outlines (list[numpy.ndarray]): A list of binary masks representing the outlines
              for each processed dimension.
            - image (numpy.ndarray): The original input image (unchanged).
    """
    outlines = []
    overlayed_image = rgb_image.copy()

    def process_dim(mask_dim):
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
    """
    Converts a well ID to the CQ1 well format.

    Args:
        well_id (int): The well ID to be converted.

    Returns:
        str: The well ID in CQ1 well format.

    """
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
    """
    Determines the appropriate batch size for Cellpose based on the available GPU's VRAM.

    This function checks if CUDA is available and retrieves the VRAM (Video RAM) of the first GPU device.
    Based on the VRAM size, it assigns a batch size for Cellpose processing. If CUDA is not available
    or an error occurs, a default batch size of 8 is returned.

    Returns:
        int: The batch size for Cellpose. Possible values are:
             - 8 (default or if VRAM < 8 GB)
             - 16 (if 8 GB <= VRAM < 12 GB)
             - 48 (if 12 GB <= VRAM < 24 GB)
             - 96 (if VRAM >= 24 GB)
    """
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
    """
    Extracts metadata from a list of filenames based on a provided regular expression
    and organizes the files into a dictionary grouped by metadata keys.
    Args:
        filenames (list of str): A list of filenames to process.
        src (str): The source directory containing the files.
        regular_expression (re.Pattern): A compiled regular expression used to extract metadata
            from the filenames. The regex should define named groups such as 'plateID', 'wellID',
            'fieldID', 'chanID', 'timeID', and 'sliceID' as applicable.
        metadata_type (str, optional): The type of metadata to process. Defaults to 'cellvoyager'.
            If set to 'cq1', well IDs will be converted using `_convert_cq1_well_id`.
    Returns:
        defaultdict: A dictionary where keys are tuples of metadata values 
        (plate, well, field, channel, timeID, sliceID), and values are lists of file paths
        corresponding to those metadata keys.
    Raises:
        IndexError: If the filename matches the regex but does not contain the expected groups.
        Prints warnings for filenames that do not match the regex or fail to extract metadata.
    Notes:
        - The function assumes that the provided regular expression includes named groups
          for extracting metadata.
        - If a metadata group (e.g., 'timeID' or 'sliceID') is not present in the regex,
          its value will be set to `None`.
        - The function handles numeric metadata values by converting them to integers
          where applicable.
    """
    
    images_by_key = defaultdict(list)

    for filename in filenames:
        match = regular_expression.match(filename)
        if match:
            try:
                try:
                    plate = match.group('plateID')
                except:
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
                    orig_wellID = wellID
                    wellID = _convert_cq1_well_id(wellID)
                    print(f'Converted Well ID: {orig_wellID} to {wellID}', end='\r', flush=True)

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
    """
    Counts the number of objects in a given mask.

    Args:
    - mask: numpy.ndarray. The mask containing object labels.

    Returns:
    - int. The number of objects in the mask.
    """
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels!=0])
    return num_objects

def _update_database_with_merged_info(db_path, df, table='png_list', columns=['pathogen', 'treatment', 'host_cells', 'condition', 'prcfo']):
    """
    Merges additional columns into the png_list table in the SQLite database and updates it.

    Args:
        db_path (str): The path to the SQLite database file.
        df (pd.DataFrame): DataFrame containing the additional info to be merged.
        table (str): Name of the table to update in the database. Defaults to 'png_list'.
    """
    # Connect to the SQLite database
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

def _generate_representative_images(db_path, cells=['HeLa'], cell_loc=None, pathogens=['rh'], pathogen_loc=None, treatments=['cm'], treatment_loc=None, channel_of_interest=1, compartments = ['pathogen','cytoplasm'], measurement = 'mean_intensity', nr_imgs=16, channel_indices=[0,1,2], um_per_pixel=0.1, scale_bar_length_um=10, plot=False, fontsize=12, show_filename=True, channel_names=None, update_db=True):
    
    """
    Generates representative images based on the provided parameters.

    Args:
        db_path (str): The path to the SQLite database file.
        cells (list, optional): The list of host cell types. Defaults to ['HeLa'].
        cell_loc (list, optional): The list of location identifiers for host cells. Defaults to None.
        pathogens (list, optional): The list of pathogens. Defaults to ['rh'].
        pathogen_loc (list, optional): The list of location identifiers for pathogens. Defaults to None.
        treatments (list, optional): The list of treatments. Defaults to ['cm'].
        treatment_loc (list, optional): The list of location identifiers for treatments. Defaults to None.
        channel_of_interest (int, optional): The index of the channel of interest. Defaults to 1.
        compartments (list or str, optional): The compartments to compare. Defaults to ['pathogen', 'cytoplasm'].
        measurement (str, optional): The measurement to compare. Defaults to 'mean_intensity'.
        nr_imgs (int, optional): The number of representative images to generate. Defaults to 16.
        channel_indices (list, optional): The indices of the channels to include in the representative images. Defaults to [0, 1, 2].
        um_per_pixel (float, optional): The scale factor for converting pixels to micrometers. Defaults to 0.1.
        scale_bar_length_um (float, optional): The length of the scale bar in micrometers. Defaults to 10.
        plot (bool, optional): Whether to plot the representative images. Defaults to False.
        fontsize (int, optional): The font size for the plot. Defaults to 12.
        show_filename (bool, optional): Whether to show the filename on the plot. Defaults to True.
        channel_names (list, optional): The names of the channels. Defaults to None.

    Returns:
        None
    """
    
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
    """
    Maps values to a specific location in the row or column based on the given locs.

    Args:
        row (dict): The row dictionary containing the location identifier.
        values (list): The list of values to be mapped.
        locs (list): The list of location identifiers.

    Returns:
        The mapped value corresponding to the given row or column location, or None if not found.
    """
    if locs:
        value_dict = {loc: value for value, loc_list in zip(values, locs) for loc in loc_list}
        # Determine if we're dealing with row or column based on first location identifier
        type_ = 'rowID' if locs[0][0][0] == 'r' else 'columnID'
        return value_dict.get(row[type_], None)
    return values[0] if values else None

def is_list_of_lists(var):
    if isinstance(var, list) and all(isinstance(i, list) for i in var):
        return True
    return False

def normalize_to_dtype(array, p1=2, p2=98, percentile_list=None, new_dtype=None):
    """
    Normalize each image in the stack to its own percentiles.

    Args:
    - array: numpy array
    The input stack to be normalized.
    - p1: int, optional
    The lower percentile value for normalization. Default is 2.
    - p2: int, optional
    The upper percentile value for normalization. Default is 98.
    - percentile_list: list, optional
    A list of pre-calculated percentiles for each image in the stack. Default is None.
    
    Returns:
    - new_stack: numpy array
    The normalized stack with the same shape as the input stack.
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
    """
    Returns a list of subdirectories within the given base directory.

    Args:
        base_dir (str): The base directory to search for subdirectories.

    Returns:
        list: A list of subdirectories within the base directory.
    """
    
    endpoint_subdirectories = []
    for root, dirs, _ in os.walk(base_dir):
        if not dirs:
            endpoint_subdirectories.append(root)
            
    endpoint_subdirectories = [path for path in endpoint_subdirectories if os.path.basename(path) != 'figure']
    return endpoint_subdirectories
    
def _generate_names(file_name, cell_id, cell_nucleus_ids, cell_pathogen_ids, source_folder, crop_mode='cell'):
    """
    Generate names for the image, folder, and table based on the given parameters.

    Args:
        file_name (str): The name of the file.
        cell_id (numpy.ndarray): An array of cell IDs.
        cell_nucleus_ids (numpy.ndarray): An array of cell nucleus IDs.
        cell_pathogen_ids (numpy.ndarray): An array of cell pathogen IDs.
        source_folder (str): The source folder path.
        crop_mode (str, optional): The crop mode. Defaults to 'cell'.

    Returns:
        tuple: A tuple containing the image name, folder path, and table name.
    """
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
    metadata = f'{plate}_{well}'
    fldr = os.path.join(fldr,metadata)
    table_name = fldr.replace("/", "_")
    return img_name, fldr, table_name

def _find_bounding_box(crop_mask, _id, buffer=10):
    """
    Find the bounding box coordinates for a given object ID in a crop mask.

    Args:
    crop_mask (ndarray): The crop mask containing object IDs.
    _id (int): The object ID to find the bounding box for.
    buffer (int, optional): The buffer size to add to the bounding box coordinates. Defaults to 10.

    Returns:
    ndarray: A new mask with the same dimensions as crop_mask, where the bounding box area is filled with the object ID.
    """
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
        """
        Merges morphology and intensity dataframes, renames columns, adds additional columns, rearranges columns,
        and saves the merged dataframe to a SQLite database.

        Args:
            morph_df (pd.DataFrame): Dataframe containing morphology data.
            intensity_df (pd.DataFrame): Dataframe containing intensity data.
            table_type (str): Type of table to save the merged dataframe to.
            source_folder (str): Path to the source folder.
            file_name (str): Name of the file.
            experiment (str): Name of the experiment.
            timelapse (bool, optional): Indicates if the data is from a timelapse experiment. Defaults to False.

        Raises:
            ValueError: If an invalid table_type is provided or if columns are missing in the dataframe.

        """
        morph_df = _check_integrity(morph_df)
        intensity_df = _check_integrity(intensity_df)
        if len(morph_df) > 0 and len(intensity_df) > 0:
            merged_df = pd.merge(morph_df, intensity_df, on='object_label', how='outer')
            merged_df = merged_df.rename(columns={"label_list_x": "label_list_morphology", "label_list_y": "label_list_intensity"})
            merged_df['file_name'] = file_name
            merged_df['path_name'] = os.path.join(source_folder, file_name + '.npy')
            if timelapse:
                merged_df[['plateID', 'rowID', 'columnID', 'fieldID', 'timeid', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(_map_wells(x, timelapse)))
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
    """
    Converts the given value to an integer if possible, otherwise returns the default value.

    Args:
        value: The value to be converted to an integer.
        default: The default value to be returned if the conversion fails. Default is 0.

    Returns:
        The converted integer value if successful, otherwise the default value.
    """
    try:
        return int(value)
    except ValueError:
        print(f'Could not convert {value} to int using {default}', end='\r', flush=True)
        return default

def _map_wells(file_name, timelapse=False):
    """
    Maps the components of a file name to plate, row, column, field, and timeid (if timelapse is True).

    Args:
        file_name (str): The name of the file.
        timelapse (bool, optional): Indicates whether the file is part of a timelapse sequence. Defaults to False.

    Returns:
        tuple: A tuple containing the mapped values for plate, row, column, field, and timeid (if timelapse is True).
    """
    try:
        parts = file_name.split('_')
        plate = 'p' + parts[0]
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
    """
    Maps the components of a file name to their corresponding values.

    Args:
        file_name (str): The name of the file.
        timelapse (bool, optional): Indicates whether the file is part of a timelapse sequence. Defaults to False.

    Returns:
        tuple: A tuple containing the mapped components of the file name.

    Raises:
        None

    """
    try:
        root, ext = os.path.splitext(file_name)
        parts = root.split('_')
        plate = 'p' + parts[0]
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
    """
    Check the integrity of the DataFrame and perform necessary modifications.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame with integrity checks and modifications applied.
    """
    df.columns = [col + f'_{i}' if df.columns.tolist().count(col) > 1 and i != 0 else col for i, col in enumerate(df.columns)]
    label_cols = [col for col in df.columns if 'label' in col]
    df['label_list'] = df[label_cols].values.tolist()
    df['object_label'] = df['label_list'].apply(lambda x: x[0])
    df = df.drop(columns=label_cols)
    df['label_list'] = df['label_list'].astype(str)
    return df
    
def _get_percentiles(array, p1=2, p2=98):
    """
    Calculate the percentiles of each image in the given array.

    Args:
    - array: numpy.ndarray
        The input array containing the images.
    - q1: float, optional
        The lower percentile value to calculate. Default is 2.
    - q2: float, optional
        The upper percentile value to calculate. Default is 98.

    Returns:
    - percentiles: list
        A list of tuples, where each tuple contains the minimum and maximum
        values of the corresponding image in the array.
    """
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
    """
    Crop the image around the center of the cell mask.

    Args:
    - img: numpy.ndarray
        The input image.
    - cell_mask: numpy.ndarray
        The binary mask of the cell.
    - new_width: int
        The desired width of the cropped image.
    - new_height: int
        The desired height of the cropped image.

    Returns:
    - img: numpy.ndarray
        The cropped image.
    """
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
    """
    Convert a list of masks into a stack of masks.

    Args:
        masks (list): A list of masks.

    Returns:
        list: A stack of masks.
    """
    mask_stack = []
    for idx, mask in enumerate(masks):
        mask_stack.append(mask)
    return mask_stack

def _get_diam(mag, obj):

    if obj == 'cell':
        diamiter = 2 * mag + 80
        
    if obj == 'cell_large':
        diamiter = 2 * mag + 120
                                
    if obj == 'nucleus':
        diamiter = 0.75 * mag + 45
                                
    if obj == 'pathogen':
        diamiter = mag
                                
    return int(diamiter)

def _get_object_settings(object_type, settings):
    """
    Generate and return a dictionary of settings for a specific object type.
    This function configures object-specific settings based on the provided
    `object_type` and `settings`. It supports three object types: 'cell',
    'nucleus', and 'pathogen'. If an unsupported object type is provided, 
    a message is printed, and no settings are returned.
    Args:
        object_type (str): The type of object to configure settings for. 
                           Supported values are 'cell', 'nucleus', and 'pathogen'.
        settings (dict): A dictionary containing global settings, which may include:
            - 'magnification' (float): Magnification level used to calculate diameter.
            - 'nucleus_channel' (optional): Specifies if a nucleus channel is present.
            - 'cell_restore_type' (optional): Restore type for cell objects.
            - 'nucleus_restore_type' (optional): Restore type for nucleus objects.
            - 'pathogen_restore_type' (optional): Restore type for pathogen objects.
            - 'merge_pathogens' (bool): Whether to merge pathogens.
            - 'verbose' (bool): Whether to print the generated settings.
    Returns:
        dict: A dictionary containing the configured settings for the specified object type.
              The dictionary includes keys such as 'diameter', 'minimum_size', 'maximum_size',
              'merge', 'resample', 'remove_border_objects', 'model_name', 'filter_size',
              'filter_intensity', and 'restore_type'.
    Notes:
        - The 'diameter' is calculated using the `_get_diam` function based on the 
          magnification level and object type.
        - The 'minimum_size' and 'maximum_size' are derived from the diameter.
        - The 'model_name' varies depending on the object type and other settings.
        - If `settings['verbose']` is True, the generated settings are printed.
    """
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
        """
        Read a table from an SQLite database into a pandas DataFrame.

        Args:
        - db_path (str): The path to the SQLite database file.
        - table_name (str): The name of the table to read. Default is 'object_counts'.

        Returns:
        - df (pandas.DataFrame): The table data as a pandas DataFrame.
        """
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        # Read the entire table into a pandas DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        # Close the connection
        conn.close()
        return df

    def _pivot_dataframe(df):

        """
        Pivot the DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The pivoted DataFrame with filled NaN values.
        """
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
    
def _get_cellpose_channels_v2(src, nucleus_channel, pathogen_channel, cell_channel):
    """
    Generate a dictionary of Cellpose channels based on the provided input channels.

    This function constructs a mapping of Cellpose channels for nucleus, pathogen, 
    and cell segmentation. It also checks for the existence of precomputed mask 
    files and issues a warning if any channel is `None` while masks already exist.

    Args:
        src (str): The source directory containing the 'masks' folder with precomputed 
                   mask files ('cell_mask_stack', 'nucleus_mask_stack', 'pathogen_mask_stack').
        nucleus_channel (int or None): The channel index for the nucleus. If `None`, 
                                       the nucleus channel is not included.
        pathogen_channel (int or None): The channel index for the pathogen. If `None`, 
                                        the pathogen channel is not included.
        cell_channel (int or None): The channel index for the cell. If `None`, the cell 
                                    channel is not included.

    Returns:
        dict: A dictionary where keys are 'nucleus', 'pathogen', and 'cell', and values 
              are lists of two integers representing the channels to be used for each 
              type. If a channel is not provided (`None`), it is excluded from the dictionary.

    Warnings:
        Prints a warning if any of the channels (`nucleus_channel`, `pathogen_channel`, 
        `cell_channel`) is `None` while precomputed mask files already exist in the 
        specified source directory.

    Notes:
        - The nucleus and pathogen channels are always duplicated as [channel, channel].
        - The cell channel prefers the nucleus channel as the first channel if available; 
          otherwise, it duplicates the cell channel as [cell_channel, cell_channel].
    """
    cell_mask_path = os.path.join(src, 'masks', 'cell_mask_stack')
    nucleus_mask_path = os.path.join(src, 'masks', 'nucleus_mask_stack')
    pathogen_mask_path = os.path.join(src, 'masks', 'pathogen_mask_stack')

    if any(os.path.exists(p) for p in [cell_mask_path, nucleus_mask_path, pathogen_mask_path]):
        if any(c is None for c in [nucleus_channel, pathogen_channel, cell_channel]):
            print('Warning: Cellpose masks already exist. Unexpected behaviour if any channel is None while masks exist.')

    cellpose_channels = {}

    # Nucleus: always duplicated single channel
    if nucleus_channel is not None:
        cellpose_channels['nucleus'] = [nucleus_channel, nucleus_channel]

    # Pathogen: always duplicated single channel
    if pathogen_channel is not None:
        cellpose_channels['pathogen'] = [pathogen_channel, pathogen_channel]

    # Cell: prefer nucleus as second if available
    if cell_channel is not None:
        if nucleus_channel is not None:
            cellpose_channels['cell'] = [nucleus_channel, cell_channel]
        else:
            cellpose_channels['cell'] = [cell_channel, cell_channel]

    return cellpose_channels
    
def _get_cellpose_channels_v1(src, nucleus_channel, pathogen_channel, cell_channel):
    """
    Generates a dictionary mapping object types (nucleus, pathogen, cell) to their respective 
    Cellpose channel configurations based on the provided input channels.
    This function checks for the existence of pre-generated Cellpose masks in the specified 
    source directory and warns the user if any of the input channels are set to None while 
    masks already exist. The function then constructs a dictionary of Cellpose channel 
    configurations for the specified object types.
    Args:
        src (str): The source directory path where the Cellpose masks are stored.
        nucleus_channel (int or None): The channel index for the nucleus. If None, the nucleus 
            is not included in the configuration.
        pathogen_channel (int or None): The channel index for the pathogen. If None, the pathogen 
            is not included in the configuration.
        cell_channel (int or None): The channel index for the cell. If None, the cell is not 
            included in the configuration.
    Returns:
        dict: A dictionary where keys are object types ('nucleus', 'pathogen', 'cell') and 
        values are lists specifying the Cellpose channel configuration for each object type.
    Warnings:
        Prints a warning if any of the input channels are set to None while Cellpose masks 
        already exist in the specified source directory.
    Notes:
        - The function assumes that the Cellpose masks are stored in subdirectories named 
          'masks/cell_mask_stack', 'masks/nucleus_mask_stack', and 'masks/pathogen_mask_stack' 
          within the source directory.
        - The channel configuration is represented as a list of two integers, where the first 
          integer is always 0, and the second integer depends on the presence of other channels.
    """

    cell_mask_path = os.path.join(src, 'masks', 'cell_mask_stack')
    nucleus_mask_path = os.path.join(src, 'masks', 'nucleus_mask_stack')
    pathogen_mask_path = os.path.join(src, 'masks', 'pathogen_mask_stack')


    if os.path.exists(cell_mask_path) or os.path.exists(nucleus_mask_path) or os.path.exists(pathogen_mask_path):
        if nucleus_channel is None or nucleus_channel is None or nucleus_channel is None:
            print('Warning: Cellpose masks already exist. Unexpected behaviour when setting any object dimention to None when the object masks have been created.')
        
    cellpose_channels = {}
    if not nucleus_channel is None:
        cellpose_channels['nucleus'] = [0,0]
        
    if not pathogen_channel is None:
        if not nucleus_channel is None:
            if not pathogen_channel is None:
                cellpose_channels['pathogen'] = [0,2]
            else:
                cellpose_channels['pathogen'] = [0,1]
        else:
            cellpose_channels['pathogen'] = [0,0]
        
    if not cell_channel is None:
        if not nucleus_channel is None:
            cellpose_channels['cell'] = [0,1]
        else:
            cellpose_channels['cell'] = [0,0]
            
    return cellpose_channels

def _get_cellpose_channels(src, nucleus_channel, pathogen_channel, cell_channel):
    """
    Generate a mapping of channels for Cellpose segmentation based on the provided input channels.

    This function determines the logical order of channels for nucleus, cell, and pathogen,
    removes duplicates while preserving the first occurrence, and maps the channels to indices
    capped at a maximum value of 2. It then assigns these indices to the respective Cellpose
    segmentation categories.

    Args:
        src (str): The source directory containing the mask files.
        nucleus_channel (int or None): The channel index for the nucleus. If None, the nucleus
            is not included in the mapping.
        pathogen_channel (int or None): The channel index for the pathogen. If None, the pathogen
            is not included in the mapping.
        cell_channel (int or None): The channel index for the cell. If None, the cell is not
            included in the mapping.

    Returns:
        dict: A dictionary mapping segmentation categories ('nucleus', 'cell', 'pathogen') to
        their respective channel indices. Each category is assigned a list of two indices,
        which are either identical or derived from the logical order of the input channels.

    Warnings:
        If any of the mask files already exist in the `src` directory and any of the input
        channels (`nucleus_channel`, `pathogen_channel`, `cell_channel`) are None, a warning
        is printed indicating potential unexpected behavior.
    """
    cell_mask_path = os.path.join(src, 'masks', 'cell_mask_stack')
    nucleus_mask_path = os.path.join(src, 'masks', 'nucleus_mask_stack')
    pathogen_mask_path = os.path.join(src, 'masks', 'pathogen_mask_stack')

    if any(os.path.exists(p) for p in [cell_mask_path, nucleus_mask_path, pathogen_mask_path]):
        if any(c is None for c in [nucleus_channel, pathogen_channel, cell_channel]):
            print('Warning: Cellpose masks already exist. Unexpected behaviour if any channel is None while masks exist.')

    # Enforced order: nucleus → cell → pathogen
    logical_order = []
    if nucleus_channel is not None:
        logical_order.append(('nucleus', nucleus_channel))
    if cell_channel is not None:
        logical_order.append(('cell', cell_channel))
    if pathogen_channel is not None:
        logical_order.append(('pathogen', pathogen_channel))

    # Remove duplicate values while preserving first occurrence
    seen = set()
    ordered_channels = []
    for name, ch in logical_order:
        if ch not in seen:
            ordered_channels.append(ch)
            seen.add(ch)

    # Map actual channel values to capped indices
    remap = {ch: min(i, 2) for i, ch in enumerate(ordered_channels)}

    # Final channel assignments
    cellpose_channels = {}

    if nucleus_channel is not None:
        c = remap[nucleus_channel]
        cellpose_channels['nucleus'] = [c, c]

    if cell_channel is not None:
        c = remap[cell_channel]
        if nucleus_channel is not None:
            n = remap[nucleus_channel]
            cellpose_channels['cell'] = [n, c]
        else:
            cellpose_channels['cell'] = [c, c]

    if pathogen_channel is not None:
        c = remap[pathogen_channel]
        cellpose_channels['pathogen'] = [c, c]

    return cellpose_channels

def annotate_conditions(df, cells=None, cell_loc=None, pathogens=None, pathogen_loc=None, treatments=None, treatment_loc=None):
    """
    Annotates conditions in a DataFrame based on specified criteria and combines them into a 'condition' column.
    NaN is used for missing values, and they are excluded from the 'condition' column.

    Args:
        df (pandas.DataFrame): The DataFrame to annotate.
        cells (list/str, optional): Host cell types. Defaults to None.
        cell_loc (list of lists, optional): Values for each host cell type. Defaults to None.
        pathogens (list/str, optional): Pathogens. Defaults to None.
        pathogen_loc (list of lists, optional): Values for each pathogen. Defaults to None.
        treatments (list/str, optional): Treatments. Defaults to None.
        treatment_loc (list of lists, optional): Values for each treatment. Defaults to None.

    Returns:
        pandas.DataFrame: Annotated DataFrame with a combined 'condition' column.
    """
    
    def _get_type(val):
        """Determine if a value maps to 'rowID' or 'columnID'."""
        if isinstance(val, str) and val.startswith('c'):
            return 'columnID'
        elif isinstance(val, str) and val.startswith('r'):
            return 'rowID'
        return None

    def _map_or_default(column_name, values, loc, df):
        """
        Consolidates the logic for mapping values or assigning defaults when loc is None.
    
        Args:
            column_name (str): The column in the DataFrame to annotate.
            values (list/str): The list of values or a single string to annotate.
            loc (list of lists): Location mapping for the values, or None if not used.
            df (pandas.DataFrame): The DataFrame to modify.
        """
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
    """
    Splits the input dataframe into numeric and non-numeric parts, groups them by the specified column,
    and returns the grouped dataframes with conditional aggregation.

    Args:
    df (pandas.DataFrame): The input dataframe.
    group_by (str): The column name to group the dataframes by.
    object_type (str): The column name to concatenate with 'prcf' to create a new column 'prcfo'.

    Returns:
    grouped_numeric (pandas.DataFrame): The grouped dataframe containing numeric columns with conditional aggregation.
    grouped_non_numeric (pandas.DataFrame): The grouped dataframe containing non-numeric columns.
    """
    
    # Ensure 'prcf' column exists by concatenating specific columns
    if 'prcf' not in df.columns:
        try:
            df['prcf'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str) + '_' + df['fieldID'].astype(str)
        except Exception as e:
            print(e)    
    
    # Create the 'prcfo' column
    df['prcfo'] = df['prcf'] + '_' + df[object_type]
    df = df.set_index(group_by, inplace=False)

    # Split the DataFrame into numeric and non-numeric parts
    df_numeric = df.select_dtypes(include=np.number)
    df_non_numeric = df.select_dtypes(exclude=np.number)

    # Define keywords for columns to be summed instead of averaged
    sum_keywords = ['area', 'perimeter', 'convex_area', 'bbox_area', 'filled_area', 'major_axis_length', 'minor_axis_length', 'equivalent_diameter']

    # Create a dictionary for custom aggregation
    agg_dict = {}
    for column in df_numeric.columns:
        if any(keyword in column for keyword in sum_keywords):
            agg_dict[column] = 'sum'
        else:
            agg_dict[column] = 'mean'

    # Apply custom aggregation
    grouped_numeric = df_numeric.groupby(df_numeric.index).agg(agg_dict)
    grouped_non_numeric = df_non_numeric.groupby(df_non_numeric.index).first()

    return pd.DataFrame(grouped_numeric), pd.DataFrame(grouped_non_numeric)
    
def _calculate_recruitment(df, channel):
    """
    Calculate recruitment metrics based on intensity values in different channels.

    Args:
        df (pandas.DataFrame): The input DataFrame containing intensity values in different channels.
        channel (int): The channel number.

    Returns:
        pandas.DataFrame: The DataFrame with calculated recruitment metrics.

    """
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

    Args:
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
    """
    A class representing a cache with a maximum size.

    Args:
        max_size (int): The maximum size of the cache.
    """

    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

class ScaledDotProductAttention_v1(nn.Module):
    """
    Scaled Dot-Product Attention module.

    Args:
        d_k (int): The dimension of the key and query vectors.
    """
    def __init__(self, d_k):
        super(ScaledDotProductAttention_v1, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        """
        Performs the forward pass of the attention mechanism.

        Args:
            Q (torch.Tensor): The query tensor of shape (batch_size, seq_len_q, d_k).
            K (torch.Tensor): The key tensor of shape (batch_size, seq_len_k, d_k).
            V (torch.Tensor): The value tensor of shape (batch_size, seq_len_v, d_k).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len_q, d_k).
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

class SelfAttention_v1(nn.Module):
    """
    Self-Attention module that applies scaled dot-product attention mechanism.

    Args:
        in_channels (int): Number of input channels.
        d_k (int): Dimensionality of the key and query vectors.
    """

    def __init__(self, in_channels, d_k):
        super(SelfAttention_v1, self).__init__()
        self.W_q = nn.Linear(in_channels, d_k)
        self.W_k = nn.Linear(in_channels, d_k)
        self.W_v = nn.Linear(in_channels, d_k)
        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, x):
        """
        Forward pass of the SelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, d_k).
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output = self.attention(Q, K, V)
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        """
        Initializes the ScaledDotProductAttention module.

        Args:
            d_k (int): The dimension of the key and query vectors.

        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        """
        Performs the forward pass of the ScaledDotProductAttention module.

        Args:
            Q (torch.Tensor): The query tensor.
            K (torch.Tensor): The key tensor.
            V (torch.Tensor): The value tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

class SelfAttention(nn.Module):
    """
    Self-Attention module that applies scaled dot-product attention mechanism.
    
    Args:
        in_channels (int): Number of input channels.
        d_k (int): Dimensionality of the key and query vectors.
    """
    def __init__(self, in_channels, d_k):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, d_k)
        self.W_k = nn.Linear(in_channels, d_k)
        self.W_v = nn.Linear(in_channels, d_k)
        self.attention = ScaledDotProductAttention(d_k)
    
    def forward(self, x):
        """
        Forward pass of the SelfAttention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels).
        
        Returns:
            torch.Tensor: Output tensor after applying self-attention mechanism.
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output = self.attention(Q, K, V)
        return output

# Early Fusion Block
class EarlyFusion(nn.Module):
    """
    Early Fusion module for image classification.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(EarlyFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)
        
    def forward(self, x):
        """
        Forward pass of the Early Fusion module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 64, height, width).
        """
        x = self.conv1(x)
        return x

# Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Initializes the SpatialAttention module.

        Args:
            kernel_size (int): The size of the convolutional kernel. Default is 7.
        """
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs forward pass of the SpatialAttention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying spatial attention.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
# Multi-Scale Block with Attention
class MultiScaleBlockWithAttention(nn.Module):
    """
    A PyTorch module implementing a multi-scale convolutional block with spatial attention.

    This module applies a dilated convolution followed by a spatial attention mechanism
    to enhance input feature maps.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the MultiScaleBlockWithAttention module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after convolution.
        """
        super(MultiScaleBlockWithAttention, self).__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.spatial_attention = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def custom_forward(self, x):
        """
        Apply dilated convolution followed by spatial attention.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after attention-enhanced feature transformation.
        """
        x1 = F.relu(self.dilated_conv1(x), inplace=True)
        x = self.spatial_attention(x1)
        return x

    def forward(self, x):
        """
        Standard forward pass that delegates to `custom_forward`.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.custom_forward(x)

# Final Classifier
class CustomCellClassifier(nn.Module):
    """
    A custom neural network module for single-cell classification using early fusion and multi-scale attention.

    This architecture supports optional gradient checkpointing for reduced memory usage during training.
    """
    def __init__(self, num_classes, pathogen_channel, use_attention, use_checkpoint, dropout_rate):
        """
        Initialize the classifier with early fusion and attention blocks.

        Args:
            num_classes (int): Number of classification categories.
            pathogen_channel (int): Unused; reserved for future feature fusion.
            use_attention (bool): Unused; reserved for attention gating.
            use_checkpoint (bool): Whether to use checkpointing to reduce memory.
            dropout_rate (float): Unused; reserved for future regularization.
        """
        super(CustomCellClassifier, self).__init__()
        self.early_fusion = EarlyFusion(in_channels=3)
        
        self.multi_scale_block_1 = MultiScaleBlockWithAttention(in_channels=64, out_channels=64)
        
        self.fc1 = nn.Linear(64, num_classes)
        self.use_checkpoint = use_checkpoint
        # Explicitly require gradients for all parameters
        for param in self.parameters():
            param.requires_grad = True
        
    def custom_forward(self, x):
        """
        Perform a standard forward pass without gradient checkpointing.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Logits for each class.
        """
        x.requires_grad = True 
        x = self.early_fusion(x)
        x = self.multi_scale_block_1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        return x

    def forward(self, x):
        """
        Forward pass with optional gradient checkpointing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        if self.use_checkpoint:
            x.requires_grad = True 
            return checkpoint(self.custom_forward, x)
        else:
            return self.custom_forward(x)

#CNN and Transformer class, pick any Torch model.
class TorchModel(nn.Module):
    """
    A PyTorch wrapper for pretrained torchvision models with a custom SPACR classifier head.

    This class supports custom dropout insertion and optional gradient checkpointing
    for memory efficiency during training.
    """
    def __init__(self, model_name='resnet50', pretrained=True, dropout_rate=None, use_checkpoint=False):
        """
        Initialize the TorchModel with optional dropout and checkpointing.

        Args:
            model_name (str): The model architecture to load from torchvision.models.
            pretrained (bool): Whether to initialize with pretrained weights.
            dropout_rate (float or None): Dropout rate for the classifier head.
            use_checkpoint (bool): Whether to enable gradient checkpointing.
        """
        super(TorchModel, self).__init__()
        self.model_name = model_name
        self.use_checkpoint = use_checkpoint
        self.base_model = self.init_base_model(pretrained)
        
        # Retain layers up to and including the (5): Linear layer for model 'maxvit_t'
        if model_name == 'maxvit_t':
            self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-1])
        
        if dropout_rate is not None:
            self.apply_dropout_rate(self.base_model, dropout_rate)
            
        self.num_ftrs = self.get_num_ftrs()
        self.init_spacr_classifier(dropout_rate)

    def apply_dropout_rate(self, model, dropout_rate):
        """
        Recursively set dropout probability for all nn.Dropout layers in the model.

        Args:
            model (nn.Module): The model to modify.
            dropout_rate (float): New dropout probability.
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    def init_base_model(self, pretrained):
        """
        Load the base model from torchvision.models.

        Args:
            pretrained (bool): Whether to load pretrained weights.

        Returns:
            nn.Module: The base feature extractor.
        """
        model_func = models.__dict__.get(self.model_name, None)
        if not model_func:
            raise ValueError(f"Model {self.model_name} is not recognized.")
        weight_choice = self.get_weight_choice()
        if weight_choice is not None:
            return model_func(weights=weight_choice)
        else:
            return model_func(pretrained=pretrained)

    def get_weight_choice(self):
        """
        Get the default weights enum for the selected model.

        Returns:
            Optional[torchvision.models.WeightsEnum]: Default weights or None.
        """
        weight_enum = None
        for attr_name in dir(models):
            if attr_name.lower() == f"{self.model_name}_weights".lower():
                weight_enum = getattr(models, attr_name)
                break
        return weight_enum.DEFAULT if weight_enum else None

    def get_num_ftrs(self):
        """
        Determine output feature dimensionality from the base model.

        Returns:
            int: Feature vector size.
        """
        if hasattr(self.base_model, 'fc'):
            self.base_model.fc = nn.Identity()
        elif hasattr(self.base_model, 'classifier'):
            if self.model_name != 'maxvit_t':
                self.base_model.classifier = nn.Identity()

        # Forward a dummy input and check output size
        dummy_input = torch.randn(1, 3, 224, 224)
        output = self.base_model(dummy_input)
        return output.size(1)

    def init_spacr_classifier(self, dropout_rate):
        """
        Create the final classification layer and optional dropout.

        Args:
            dropout_rate (float or None): Dropout probability. If None, dropout is skipped.
        """
        self.use_dropout = dropout_rate is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        self.spacr_classifier = nn.Linear(self.num_ftrs, 1)

    def forward(self, x):
        """
        Forward pass through base model, optional dropout, and final classifier.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (N,).
        """
        if self.use_checkpoint:
            x = checkpoint(self.base_model, x)
        else:
            x = self.base_model(x)
        if self.use_dropout:
            x = self.dropout(x)
        logits = self.spacr_classifier(x).flatten()
        return logits

class FocalLossWithLogits(nn.Module):
    """
    Focal Loss with logits for binary classification.

    This loss function is especially useful for addressing class imbalance by focusing more
    on hard-to-classify examples.

    Args:
        alpha (float): Balancing factor for positive/negative examples. Default is 1.
        gamma (float): Focusing parameter that down-weights easy examples. Default is 2.
    """

    def __init__(self, alpha=1, gamma=2):
        """
        Initialize the focal loss.

        Args:
            alpha (float): Balancing factor for positive/negative examples. Default is 1.
            gamma (float): Focusing parameter to down-weight well-classified examples. Default is 2.
        """
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        """
        Compute the focal loss between logits and targets.

        Args:
            logits (torch.Tensor): Predicted unnormalized scores (logits).
            target (torch.Tensor): Ground truth binary labels (same shape as logits).

        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()
    
class ResNet(nn.Module):
    """
    A wrapper around torchvision ResNet models with optional dropout, checkpointing, 
    and a custom classifier head.

    Supported ResNet variants: resnet18, resnet34, resnet50, resnet101, resnet152.
    """
    def __init__(self, resnet_type='resnet50', dropout_rate=None, use_checkpoint=False, init_weights='imagenet'):
        """
        Initialize the ResNet model wrapper.

        Args:
            resnet_type (str): Which ResNet variant to use. Options: 'resnet18', 'resnet34', etc.
            dropout_rate (float or None): Dropout rate to apply before the final layer.
            use_checkpoint (bool): Whether to enable gradient checkpointing.
            init_weights (str): Either 'imagenet' to load pretrained weights or 'none'.
        """
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
        """
        Initialize the base model and classifier layers.

        Args:
            base_model_dict (dict): Contains model constructor and weight enum.
            dropout_rate (float or None): Dropout rate to use.
            use_checkpoint (bool): Whether to use gradient checkpointing.
            init_weights (str): Weight initialization mode.
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
        """
        Forward pass through ResNet and classification layers.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (N,).
        """
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
    """
    Splits a dataset into training and validation subsets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        split_ratio (float, optional): The ratio of validation samples to total samples. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the training dataset and validation dataset.
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
    """
    Calculate classification metrics for binary classification.

    Args:
    - all_labels (list): List of true labels.
    - prediction_pos_probs (list): List of predicted positive probabilities.
    - loader_name (str): Name of the data loader.
    - loss (float): Loss value.
    - epoch (int): Epoch number.

    Returns:
    - data_df (DataFrame): DataFrame containing the calculated metrics.
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
    """
    Computes the Invariant Risk Minimization (IRM) penalty.

    Args:
        losses (list): A list of losses.
        dummy_w (torch.Tensor): A dummy weight tensor.
        device (torch.device): The device to perform computations on.

    Returns:
        float: The computed IRM penalty.
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

def choose_model(model_type, device, init_weights=True, dropout_rate=0, use_checkpoint=False, channels=3, height=224, width=224, chan_dict=None, num_classes=2, verbose=False):
    """
    Choose a model for classification.

    Args:
        model_type (str): The type of model to choose. Can be one of the pre-defined TorchVision models or 'custom' for a custom model.
        device (str): The device to use for model inference.
        init_weights (bool, optional): Whether to initialize the model with pre-trained weights. Defaults to True.
        dropout_rate (float, optional): The dropout rate to use in the model. Defaults to 0.
        use_checkpoint (bool, optional): Whether to use checkpointing during model training. Defaults to False.
        channels (int, optional): The number of input channels for the model. Defaults to 3.
        height (int, optional): The height of the input images for the model. Defaults to 224.
        width (int, optional): The width of the input images for the model. Defaults to 224.
        chan_dict (dict, optional): A dictionary containing channel information for custom models. Defaults to None.
        num_classes (int, optional): The number of output classes for the model. Defaults to 2.

    Returns:
        torch.nn.Module: The chosen model.
    """

    torch_model_types = torchvision.models.list_models(module=torchvision.models)
    model_types = torch_model_types + ['custom']
    
    if not chan_dict is None:
        pathogen_channel = chan_dict['pathogen_channel']
        nucleus_channel = chan_dict['nucleus_channel']
        protein_channel = chan_dict['protein_channel']
    
    if model_type not in model_types:
        print(f'Invalid model_type: {model_type}. Compatible model_types: {model_types}')
        return

    print(f'Model parameters: Architecture: {model_type} init_weights: {init_weights} dropout_rate: {dropout_rate} use_checkpoint: {use_checkpoint}', end='\r', flush=True)
    
    if model_type == 'custom':
        
        base_model = CustomCellClassifier(num_classes, pathogen_channel=pathogen_channel, use_attention=True, use_checkpoint=use_checkpoint, dropout_rate=dropout_rate)
        #base_model = CustomCellClassifier(num_classes=2, pathogen_channel=pathogen_channel, nucleus_channel=nucleus_channel, protein_channel=protein_channel, dropout_rate=dropout_rate, use_checkpoint=use_checkpoint)
    elif model_type in torch_model_types:
        base_model = TorchModel(model_name=model_type, pretrained=init_weights, dropout_rate=dropout_rate)
    else:
        print(f'Compatible model_types: {model_types}')
        raise ValueError(f"Invalid model_type: {model_type}")
    if verbose:
        print(base_model)
    
    return base_model

def calculate_loss(output, target, loss_type='binary_cross_entropy_with_logits'):
    """
    Calculates the loss between the model output and the target based on the specified loss type.

    Args:
        output (Tensor): The predicted output from the model.
        target (Tensor): The ground truth target values.
        loss_type (str, optional): The type of loss function to use. 
            Supported values are:
            - 'binary_cross_entropy_with_logits': Uses binary cross-entropy loss with logits.
            - 'focal_loss': Uses focal loss with logits. Defaults to 'binary_cross_entropy_with_logits'.

    Returns:
        Tensor: The computed loss value.

    Raises:
        ValueError: If an unsupported loss_type is provided.
    """
    if loss_type == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(output, target)
    elif loss_type == 'focal_loss':
        focal_loss_fn = FocalLossWithLogits(alpha=1, gamma=2)
        loss = focal_loss_fn(output, target)
    return loss

def pick_best_model(src):
    """
    Selects the best model file from a given directory based on accuracy and epoch.
    This function scans the specified directory for files with a `.pth` extension,
    extracts accuracy and epoch information from their filenames using a predefined
    pattern, and selects the file with the highest accuracy. If multiple files have
    the same accuracy, the one with the highest epoch is selected.
    Args:
        src (str): The path to the directory containing the model files.
    Returns:
        str: The full path to the best model file based on accuracy and epoch.
    Notes:
        - The filenames are expected to follow the pattern `_epoch_<epoch>_acc_<accuracy>.pth`,
          where `<epoch>` is an integer and `<accuracy>` is a float.
        - If no files match the pattern, the function may raise an IndexError when
          attempting to access the first element of the sorted list.
    """
    all_files = os.listdir(src)
    pth_files = [f for f in all_files if f.endswith('.pth')]
    pattern = re.compile(r'_epoch_(\d+)_acc_(\d+(?:\.\d+)?)')

    def sort_key(x):
        match = pattern.search(x)
        if not match:
            return (0.0, 0)  # Make the primary sorting key float for consistency
        g1, g2 = match.groups()
        return (float(g2), int(g1))  # Primary sort by accuracy (g2) and secondary sort by epoch (g1)
    
    sorted_files = sorted(pth_files, key=sort_key, reverse=True)
    best_model = sorted_files[0]
    return os.path.join(src, best_model)

def get_paths_from_db(df, png_df, image_type='cell_png'):
    """
    Filters and retrieves paths from a DataFrame based on specified criteria.

    Args:
        df (pd.DataFrame): A DataFrame whose index contains the objects of interest.
        png_df (pd.DataFrame): A DataFrame containing a 'png_path' column and a 'prcfo' column.
        image_type (str, optional): A string to filter the 'png_path' column. Defaults to 'cell_png'.

    Returns:
        pd.DataFrame: A filtered DataFrame containing rows from `png_df` where the 'png_path' 
                      column contains the `image_type` string and the 'prcfo' column matches 
                      the index of `df`.
    """
    objects = df.index.tolist()
    filtered_df = png_df[png_df['png_path'].str.contains(image_type) & png_df['prcfo'].isin(objects)]
    return filtered_df

def save_file_lists(dst, data_set, ls):
    """
    Saves a list of file paths or data entries to a CSV file.

    Args:
        dst (str): The destination directory where the CSV file will be saved.
        data_set (str): The name of the dataset, which will also be used as the column name in the CSV file and the filename.
        ls (list): A list of file paths or data entries to be saved.

    Returns:
        None
    """
    df = pd.DataFrame(ls, columns=[data_set])  
    df.to_csv(f'{dst}/{data_set}.csv', index=False)
    return

def augment_single_image(args):
    """
    Augment a single image by applying various transformations and saving the results.

    This function reads an image from the specified file path, applies a series of 
    transformations (original, rotations, and flips), and saves the transformed images 
    to the destination directory with appropriate filenames.

    Args:
        args (tuple): A tuple containing:
            img_path (str): The file path to the input image.
            dst (str): The destination directory where the augmented images will be saved.

    Notes:
        The following transformations are applied to the input image:
        - Original image (no transformation)
        - 90-degree clockwise rotation
        - 180-degree rotation
        - 270-degree clockwise rotation
        - Horizontal flip
        - Vertical flip

    Side Effects:
        Saves the augmented images to the specified destination directory.
        Filenames indicate the type of transformation applied.
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
    """
    Augments a list of images and saves the augmented images to the specified destination directory.

    Args:
        file_paths (list of str): A list of file paths to the images to be augmented.
        dst (str): The destination directory where the augmented images will be saved. 
                   If the directory does not exist, it will be created.

    Returns:
        None
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    args_list = [(img_path, dst) for img_path in file_paths]

    with Pool(cpu_count()) as pool:
        pool.map(augment_single_image, args_list)

def augment_classes(dst, nc, pc, generate=True,move=True):
    def augment_classes(dst, nc, pc, generate=True, move=True):
        """
        Augments and organizes image datasets into training and testing directories.

        Args:
        -----------
        dst : str
            The destination directory where augmented data will be stored.
        nc : list
            List of file paths for the "negative class" images to be augmented.
        pc : list
            List of file paths for the "positive class" images to be augmented.
        generate : bool, optional
            If True, generates augmented images for both classes and saves them in 
            separate directories (`aug_nc` for negative class and `aug_pc` for positive class).
            Default is True.
        move : bool, optional
            If True, splits the augmented images into training and testing datasets, 
            and organizes them into subdirectories under `aug/train` and `aug/test`.
            Default is True.

        Returns:
        --------
        None

        Notes:
        ------
        - The function uses `train_test_split` to split the augmented images into 
          training (90%) and testing (10%) datasets.
        - The progress of moving files is displayed in the console.
        - The final counts of training and testing images for each class are printed.

        Example Directory Structure:
        ----------------------------
        After execution, the directory structure will look like this:
        dst/
        ├── aug_nc/       # Augmented negative class images
        ├── aug_pc/       # Augmented positive class images
        ├── aug/
            ├── train/
                ├── nc/  # Training negative class images
                ├── pc/  # Training positive class images
            ├── test/
                ├── nc/  # Testing negative class images
                ├── pc/  # Testing positive class images
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
    """
    Reads a CSV file containing image metadata, processes the data to extract
    additional information, and assigns a condition label to each row based on
    specific rules.
    Args:
        csv_loc (str): The file path to the CSV file containing the metadata.
    Returns:
        pandas.DataFrame: A DataFrame with the following additional columns:
            - 'filename': Extracted filename from the 'path' column.
            - 'plateID': Plate ID extracted from the filename.
            - 'well': Well information extracted from the filename.
            - 'fieldID': Field ID extracted from the filename.
            - 'object': Object ID extracted from the filename (with '.png' removed).
            - 'cond': Assigned condition label ('screen', 'pc', 'nc', or '') based on rules.
    """
    df = pd.read_csv(csv_loc)
    df['filename'] = df['path'].apply(lambda x: x.split('/')[-1])
    df[['plateID', 'well', 'fieldID', 'object']] = df['filename'].str.split('_', expand=True)
    df['object'] = df['object'].str.replace('.png', '')
    
    def assign_condition(row):
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
    """
    Initializes global variables `counter` and `lock` with the provided arguments.

    This function sets the global variables `counter` and `lock` to the values
    passed as `counter_` and `lock_`, respectively. It is typically used to
    share a counter and a lock object across multiple threads or processes.

    Args:
        counter_ (Any): The counter object to be assigned to the global `counter`.
        lock_ (Any): The lock object to be assigned to the global `lock`.
    """
    global counter, lock
    counter = counter_
    lock = lock_

def add_images_to_tar(paths_chunk, tar_path, total_images):
    """
    Adds a chunk of image files to a tar archive.

    Args:
        paths_chunk (list of str): A list of file paths to the images to be added to the tar archive.
        tar_path (str): The path where the tar archive will be created or overwritten.
        total_images (int): The total number of images being processed, used for progress tracking.

    Behavior:
        - Opens a tar archive at the specified `tar_path` in write mode.
        - Iterates through the provided `paths_chunk` and adds each image to the tar archive.
        - Tracks progress using a shared counter and prints progress updates every 10 images.
        - Handles missing files gracefully by printing a warning message if a file is not found.

    Notes:
        - This function assumes the existence of a global `lock` object for thread-safe counter updates.
        - The `counter` object is expected to be a shared multiprocessing.Value or similar.
        - The `print_progress` function is used to display progress updates.

    Exceptions:
        - Prints a warning message if a file in `paths_chunk` is not found, but continues processing other files.

    Example:
        add_images_to_tar(['/path/to/image1.jpg', '/path/to/image2.jpg'], '/path/to/archive.tar', 100)
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
    """
    Generates a fraction map from a given DataFrame and writes it to a CSV file.

    This function calculates the fraction of counts for each gene and well, 
    organizes the data into a pivot table-like structure, and filters out 
    columns based on a minimum frequency threshold. The resulting DataFrame 
    is saved as a CSV file.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data. It must include 
            the columns 'count', 'well_read_sum', 'prc', and the specified 
            `gene_column`.
        gene_column (str): The name of the column in `df` that contains gene 
            identifiers.
        min_frequency (float, optional): The minimum frequency threshold for 
            filtering columns. Columns with a maximum value below this 
            threshold are dropped. Defaults to 0.0.

    Returns:
        pd.DataFrame: A DataFrame containing the fraction map, with wells as 
        rows and genes as columns. Cells contain the fraction values, and 
        missing values are filled with 0.0.
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
    loc = '/mnt/data/CellVoyager/20x/tsg101/crispr_screen/all/measurements/iv.csv'
    independent_variables.to_csv(loc, index=True, header=True, mode='w')
    return independent_variables

def fishers_odds(df, threshold=0.5, phenotyp_col='mean_pred'):
    """
    Perform Fisher's exact test to evaluate the association between mutants and a binned phenotype score.

    Args:
        df (pandas.DataFrame): DataFrame containing binary mutant indicators and a phenotype score column.
        threshold (float, optional): Threshold to bin the phenotype score. Defaults to 0.5.
        phenotyp_col (str, optional): Name of the column containing phenotype scores. Defaults to 'mean_pred'.

    Returns:
        pandas.DataFrame: DataFrame with columns 'Mutant', 'OddsRatio', 'PValue', and 'AdjustedPValue'.
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
    """
    Calculate and display additional metrics and generate diagnostic plots for a given model.

    Args:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        A fitted regression model object from the statsmodels library.

    Metrics Calculated:
    --------------------
    - Root Mean Squared Error (RMSE): Measures the standard deviation of residuals.
    - Mean Absolute Error (MAE): Measures the average magnitude of residuals.
    - Durbin-Watson: Tests for the presence of autocorrelation in residuals.

    Diagnostic Plots:
    ------------------
    1. Residuals vs. Fitted: Scatter plot to check for non-linearity or unequal error variance.
    2. Histogram of Residuals: Distribution of residuals to check for normality.
    3. QQ Plot: Quantile-Quantile plot to assess if residuals follow a normal distribution.
    4. Scale-Location: Scatter plot of standardized residuals to check for homoscedasticity.

    Notes:
    ------
    - This function uses matplotlib and seaborn for plotting.
    - Ensure that the input model is fitted and contains the necessary attributes like `resid`, 
      `fittedvalues`, and `mse_resid`.

    Returns:
    --------
    None
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
    """
    Checks multicollinearity of the predictors by computing the Variance Inflation Factor (VIF).

    Args:
        x (pd.DataFrame): A DataFrame containing the predictor variables.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'Variable': The names of the predictor variables.
            - 'VIF': The Variance Inflation Factor for each predictor variable.
              A VIF value greater than 10 indicates high multicollinearity.
    """
    """Checks multicollinearity of the predictors by computing the VIF."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_data

def lasso_reg(merged_df, alpha_value=0.01, reg_type='lasso'):
    """
    Perform Lasso or Ridge regression on the input DataFrame.

    Args:
        merged_df (pandas.DataFrame): DataFrame with columns 'gene', 'grna', 'plateID', 'rowID', 'columnID', and 'pred'.
        alpha_value (float, optional): Regularization strength. Defaults to 0.01.
        reg_type (str, optional): Type of regression to perform, either 'lasso' or 'ridge'. Defaults to 'lasso'.

    Returns:
        pandas.DataFrame: DataFrame with 'Feature' and 'Coefficient' columns.
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
    """
    Perform multiple linear regression (MLR) and extract interaction coefficients.

    Args:
        merged_df (pd.DataFrame): Input DataFrame with data for regression analysis.
        refine_model (bool): Whether to remove outliers before refitting the model.

    Returns:
        tuple: Contains:
            - max_effects (dict): Maximum interaction effect size per gene.
            - max_effects_pvalues (dict): Corresponding p-values.
            - model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted regression model.
            - df (pd.DataFrame): DataFrame with sorted interaction effects and p-values.
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
    """
    Retrieves a list of files from the specified directory that match the given file extension.

    Args:
        dir_path (str): The path to the directory from which to retrieve files.
        file_extension (str, optional): The file extension to filter files by. Defaults to "*" 
                                         (matches all files).

    Returns:
        list: A list of file paths matching the specified file extension in the given directory.
    """
    return glob(os.path.join(dir_path, file_extension))
    
def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask for a 2D array with the specified dimensions.

    Args:
        h (int): The height of the 2D array.
        w (int): The width of the 2D array.
        center (tuple, optional): The (x, y) coordinates of the circle's center. 
                                  Defaults to the center of the array.
        radius (int, optional): The radius of the circle. Defaults to the smallest 
                                distance from the center to the array's edges.

    Returns:
        numpy.ndarray: A boolean 2D array where `True` represents the pixels 
                       inside the circle and `False` represents the pixels outside.
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
    """
    Apply a circular mask to an image, setting pixels outside the mask to a specified value.

    Args:
        image (np.ndarray): Input image (2D grayscale or 3D RGB array).
        output_value (int, optional): Value for pixels outside the mask. Defaults to 0.

    Returns:
        np.ndarray: Image with circular mask applied.
    """
    h, w = image.shape[:2]  # Assuming image is grayscale or RGB
    mask = create_circular_mask(h, w)
    
    # If the image has more than one channel, repeat the mask for each channel
    if len(image.shape) > 2:
        mask = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)
    
    # Apply the mask - set pixels outside of the mask to output_value
    masked_image = np.where(mask, image, output_value)
    return masked_image
    
def invert_image(image):
    """
    Inverts the pixel values of an image.

    The function calculates the inverted image by subtracting each pixel value 
    from the maximum possible value for the image's data type. For example, 
    for an image with dtype `uint8`, the maximum value is 255.

    Args:
        image (numpy.ndarray): The input image to be inverted. The image should 
        be a NumPy array with a valid integer data type (e.g., uint8, uint16).

    Returns:
        numpy.ndarray: The inverted image, with the same shape and dtype as the input.
    """
    # The maximum value depends on the image dtype (e.g., 255 for uint8)
    max_value = np.iinfo(image.dtype).max
    inverted_image = max_value - image
    return inverted_image  

def resize_images_and_labels(images, labels, target_height, target_width, show_example=True):
    """
    Resize images and labels to the specified target dimensions.

    Args:
        images (list or None): List of 2D or 3D numpy arrays representing input images.
                               If None, only labels will be resized.
        labels (list or None): List of 2D numpy arrays representing label masks.
                               If None, only images will be resized.
        target_height (int): Desired height of output arrays.
        target_width (int): Desired width of output arrays.
        show_example (bool, optional): Whether to display an example of original vs resized output.
                                       Defaults to True.

    Returns:
        tuple: A tuple (resized_images, resized_labels), where:
            - resized_images (list): List of resized image arrays. Empty if `images` is None.
            - resized_labels (list): List of resized label arrays. Empty if `labels` is None.

    Raises:
        ValueError: If both `images` and `labels` are None.

    Notes:
        - Uses `resizescikit` for resizing.
        - Applies anti-aliasing when resizing images.
        - Uses nearest-neighbor interpolation (`order=0`) for labels to preserve class values.
        - Visualization of the resizing process is shown using `plot_resize` if `show_example` is True.
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
    """
    Resize a list of label arrays back to their original dimensions.

    Args:
        labels (list of numpy.ndarray): A list of label arrays to be resized.
        orig_dims (list of tuple): A list of tuples where each tuple contains 
            two integers representing the original dimensions (width, height) 
            of the corresponding label array.

    Returns:
        list of numpy.ndarray: A list of resized label arrays with dimensions 
        matching the corresponding tuples in `orig_dims`.

    Raises:
        ValueError: If the length of `labels` and `orig_dims` do not match.
        ValueError: If any element in `orig_dims` is not a tuple of two integers.

    Notes:
        - The resizing operation uses nearest-neighbor interpolation (order=0).
        - The `preserve_range` parameter ensures that the data range of the 
          input is preserved during resizing.
        - Anti-aliasing is disabled for this operation.
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
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    The IoU is a measure of the overlap between two binary masks, defined as the 
    ratio of the intersection area to the union area of the masks.

    Args:
        mask1 (numpy.ndarray): The first binary mask. Must be a 2D array.
        mask2 (numpy.ndarray): The second binary mask. Must be a 2D array.

    Returns:
        float: The IoU value, ranging from 0 to 1. Returns 0 if the union of the 
        masks is empty.
    """
    mask1, mask2 = pad_to_same_shape(mask1, mask2)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0
    
def match_masks(true_masks, pred_masks, iou_threshold):
    """
    Matches predicted masks to ground truth masks based on Intersection over Union (IoU) threshold.

    Args:
        true_masks (list): A list of ground truth masks.
        pred_masks (list): A list of predicted masks.
        iou_threshold (float): The IoU threshold for determining a match between a true mask and a predicted mask.

    Returns:
        list: A list of tuples where each tuple contains a matched pair of (true_mask, pred_mask).
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
    """
    Computes the precision and recall based on the provided matches, number of true masks, 
    and number of predicted masks.

    Args:
        matches (list): A list of matched predictions to ground truth masks.
        num_true_masks (int): The total number of ground truth masks.
        num_pred_masks (int): The total number of predicted masks.

    Returns:
        tuple: A tuple containing:
            - precision (float): The precision value, calculated as TP / (TP + FP).
            - recall (float): The recall value, calculated as TP / (TP + FN).
    """
    TP = len(matches)
    FP = num_pred_masks - TP
    FN = num_true_masks - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall

def pad_to_same_shape(mask1, mask2):
    """
    Pads two 2D arrays (masks) to the same shape by adding zero-padding to the 
    right and bottom of each array as needed.
    Args:
        mask1 (numpy.ndarray): The first 2D array to be padded.
        mask2 (numpy.ndarray): The second 2D array to be padded.
    Returns:
        tuple: A tuple containing two 2D numpy arrays:
            - padded_mask1 (numpy.ndarray): The first array padded to match the shape of the larger array.
            - padded_mask2 (numpy.ndarray): The second array padded to match the shape of the larger array.
    Notes:
        - The padding is applied with constant values of 0.
        - The function assumes that both inputs are 2D arrays.
    """
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
    """
    Compute the Average Precision (AP) over a range of Intersection over Union (IoU) thresholds.

    This function calculates the precision-recall pairs for each IoU threshold, validates that
    precision and recall values are within the range [0, 1], and computes the Average Precision
    by integrating the precision-recall curve using the trapezoidal rule.

    Args:
        true_masks (list or array-like): Ground truth masks.
        pred_masks (list or array-like): Predicted masks.
        iou_thresholds (list or array-like): A list of IoU thresholds to evaluate.

    Returns:
        float: The computed Average Precision (AP) over the specified IoU thresholds.

    Raises:
        ValueError: If precision or recall values are out of the valid range [0, 1].
    """
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
    """
    Compute the Average Precision (AP) for segmentation masks over a range of IoU thresholds.

    This function calculates the AP by comparing the ground truth masks (`true_masks`) with the
    predicted masks (`pred_masks`) across multiple Intersection over Union (IoU) thresholds.

    Args:
        true_masks (ndarray): A binary array representing the ground truth segmentation masks.
                              Each connected component is treated as a separate object.
        pred_masks (ndarray): A binary array representing the predicted segmentation masks.
                              Each connected component is treated as a separate object.
        iou_thresholds (ndarray, optional): A 1D array of IoU thresholds to evaluate AP over.
                                            Defaults to `np.linspace(0.5, 0.95, 10)`.

    Returns:
        float: The computed Average Precision (AP) over the specified IoU thresholds.
    """
    true_mask_labels = label(true_masks)
    pred_mask_labels = label(pred_masks)
    true_mask_regions = [region.image for region in regionprops(true_mask_labels)]
    pred_mask_regions = [region.image for region in regionprops(pred_mask_labels)]
    return compute_ap_over_iou_thresholds(true_mask_regions, pred_mask_regions, iou_thresholds)

def jaccard_index(mask1, mask2):
    """
    Computes the Jaccard Index (Intersection over Union) between two binary masks.

    The Jaccard Index is a measure of similarity between two sets, defined as the size
    of the intersection divided by the size of the union of the sets.

    Args:
        mask1 (numpy.ndarray): A binary mask (e.g., a 2D array of boolean or 0/1 values).
        mask2 (numpy.ndarray): Another binary mask of the same shape as `mask1`.

    Returns:
        float: The Jaccard Index, a value between 0 and 1, where 1 indicates perfect overlap
               and 0 indicates no overlap.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def dice_coefficient(mask1, mask2):
    """
    Compute the Dice coefficient, a measure of overlap between two binary masks.

    The Dice coefficient is defined as:
        ``2 * |A ∩ B| / (|A| + |B|)``

    where A and B are binary masks, ``|A ∩ B|`` is the number of overlapping non-zero elements,
    and ``|A| + |B|`` is the total number of non-zero elements in both masks.

    Args:
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns:
        float: Dice coefficient between 0.0 (no overlap) and 1.0 (perfect overlap).
               Returns 1.0 if both masks are empty.
    """
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
    """
    Extracts the boundaries of a binary mask by applying morphological dilation 
    and erosion operations and computing their difference.

    Args:
        mask (numpy.ndarray): A 2D array representing the input mask. Non-zero 
                              values are considered part of the mask.
        dilation_radius (int, optional): The radius of the structuring element 
                                         used for dilation and erosion. 
                                         Defaults to 1.

    Returns:
        numpy.ndarray: A binary array of the same shape as the input mask, 
                       where the boundary pixels are marked as 1 and all 
                       other pixels are 0.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    struct_elem = np.ones((dilation_radius*2+1, dilation_radius*2+1))
    dilated = morphology.binary_dilation(binary_mask, footprint=struct_elem)
    eroded = morphology.binary_erosion(binary_mask, footprint=struct_elem)
    boundary = dilated ^ eroded
    return boundary

def boundary_f1_score(mask_true, mask_pred, dilation_radius=1):
    """
    Calculate the boundary F1 score between two binary masks.

    The boundary F1 score evaluates the alignment of object boundaries between
    predicted and ground truth masks. It computes the harmonic mean of boundary
    precision and recall after dilating the boundaries.

    Args:
        mask_true (np.ndarray): Ground truth binary mask. Non-zero values are considered foreground.
        mask_pred (np.ndarray): Predicted binary mask. Non-zero values are considered foreground.
        dilation_radius (int, optional): Radius used for boundary dilation to allow tolerance in matching. Defaults to 1.

    Returns:
        float: Boundary F1 score between 0 (no boundary match) and 1 (perfect boundary alignment).
    """
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
    """
    Remove non-infected cells from the stack based on the provided dimensions.

    Args:
        stack (ndarray): The stack of images.
        cell_dim (int or None): The dimension index for the cell mask. If None, a zero-filled mask will be used.
        nucleus_dim (int or None): The dimension index for the nucleus mask. If None, a zero-filled mask will be used.
        pathogen_dim (int or None): The dimension index for the pathogen mask. If None, a zero-filled mask will be used.

    Returns:
        ndarray: The updated stack with non-infected cells removed.
    """
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
    """
    Remove outside objects from the stack based on the provided dimensions.

    Args:
        stack (ndarray): The stack of images.
        cell_dim (int): The dimension index of the cell mask in the stack.
        nucleus_dim (int): The dimension index of the nucleus mask in the stack.
        pathogen_dim (int): The dimension index of the pathogen mask in the stack.

    Returns:
        ndarray: The updated stack with outside objects removed.
    """
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
    """
    Remove multi-object cells from the stack.

    Args:
        stack (ndarray): The stack of images.
        mask_dim (int): The dimension of the mask in the stack.
        cell_dim (int): The dimension of the cell in the stack.
        nucleus_dim (int): The dimension of the nucleus in the stack.
        pathogen_dim (int): The dimension of the pathogen in the stack.
        object_dim (int): The dimension of the object in the stack.

    Returns:
        ndarray: The updated stack with multi-object cells removed.
    """
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
    """
    Merges touching objects in a binary mask based on the percentage of their shared boundary.

    Args:
        mask (ndarray): Binary mask representing objects.
        threshold (float, optional): Threshold value for merging objects. Defaults to 0.25.

    Returns:
        ndarray: Merged mask.

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
    """
    Removes objects from the mask based on their mean intensity in the original image.

    Args:
        image (ndarray): The original image.
        mask (ndarray): The mask containing labeled objects.
        intensity_threshold (float): The threshold value for mean intensity.
        mode (str): The mode for intensity comparison. Can be 'low' or 'high'.

    Returns:
        ndarray: The updated mask with objects removed.

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
    """
    Filter the DataFrame to include the closest rows to a statistical measure.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name to calculate the statistical measure.
        n_rows (int): The number of closest rows to include in the result.
        use_median (bool, optional): Whether to use the median or mean as the statistical measure. 
            Defaults to False (mean).

    Returns:
        pandas.DataFrame: The filtered DataFrame with the closest rows to the statistical measure.
    """
    if use_median:
        target_value = df[column].median()
    else:
        target_value = df[column].mean()
    df['diff'] = (df[column] - target_value).abs()
    result_df = df.sort_values(by='diff').head(n_rows)
    result_df = result_df.drop(columns=['diff'])
    return result_df
    
def _find_similar_sized_images(file_list):
    """
    Find the largest group of images with the most similar size and shape.

    Args:
        file_list (list): List of file paths to the images.

    Returns:
        list: List of file paths belonging to the largest group of images with the most similar size and shape.
    """
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
    """
    Relabels the parent mask based on overlapping child labels.

    Args:
        parent_mask (ndarray): Binary mask representing the parent objects.
        child_mask (ndarray): Binary mask representing the child objects.

    Returns:
        tuple: A tuple containing the relabeled parent mask and the original child mask.

    """
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
    """
    Exclude objects from the masks based on certain criteria.

    Args:
        cell_mask (ndarray): Mask representing cells.
        nucleus_mask (ndarray): Mask representing nucleus.
        pathogen_mask (ndarray): Mask representing pathogens.
        cytoplasm_mask (ndarray): Mask representing cytoplasm.
        uninfected (bool, optional): Whether to include uninfected cells. Defaults to True.

    Returns:
        tuple: A tuple containing the filtered cell mask, nucleus mask, pathogen mask, and cytoplasm mask.
    """
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
    """
    Merge overlapping objects in two masks.

    Args:
        mask1 (ndarray): First mask.
        mask2 (ndarray): Second mask.

    Returns:
        tuple: A tuple containing the merged masks (mask1, mask2).
    """
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
    """
    Filter objects in a mask based on their frequency.

    Args:
        mask (ndarray): The input mask.
        min_value (int): The minimum frequency threshold.

    Returns:
        ndarray: The filtered mask.
    """
    count = np.bincount(mask.ravel())
    to_remove = np.where(count < min_value)
    mask[np.isin(mask, to_remove)] = 0
    return mask

def _filter_cp_masks(masks, flows, filter_size, filter_intensity, minimum_size, maximum_size, remove_border_objects, merge, batch, plot, figuresize):
    
    """
    Filter the masks based on various criteria such as size, border objects, merging, and intensity.

    Args:
        masks (list): List of masks.
        flows (list): List of flows.
        filter_size (bool): Flag indicating whether to filter based on size.
        filter_intensity (bool): Flag indicating whether to filter based on intensity.
        minimum_size (int): Minimum size of objects to keep.
        maximum_size (int): Maximum size of objects to keep.
        remove_border_objects (bool): Flag indicating whether to remove border objects.
        merge (bool): Flag indicating whether to merge adjacent objects.
        batch (ndarray): Batch of images.
        plot (bool): Flag indicating whether to plot the masks.
        figuresize (tuple): Size of the figure.

    Returns:
        list: List of filtered masks.
    """
    
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
    """
    Generates a regex pattern based on the specified metadata type and image format.
    Args:
        metadata_type (str): The type of metadata to generate the regex for. 
            Supported values are:
            - 'cellvoyager': Generates a regex for CellVoyager metadata format.
            - 'cq1': Generates a regex for CQ1 metadata format.
            - 'auto': Generates a regex for automatic detection of metadata.
            - 'custom': Uses a custom regex provided by the user.
        img_format (str): The image file format (e.g., 'tif'). If None, defaults to 'tif'.
        custom_regex (str, optional): A custom regex pattern to use when metadata_type is 'custom'.
    Returns:
        str: The generated regex pattern.
    Notes:
        - Prints the image format and the generated regex for debugging purposes.
        - If `img_format` is None, it defaults to 'tif'.
    """
    
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
        regex = f"({custom_regex}){img_format}"
        
    print(f'regex mode:{metadata_type} regex:{regex}')
    return regex

def _run_test_mode(src, regex, timelapse=False, test_images=10, random_test=True):
    """
    Prepares a test dataset by selecting and copying a subset of images from the source directory 
    based on a regular expression and other criteria.
    Args:
        src (str): The source directory containing the images.
        regex (str): A regular expression to match filenames. The regex should include named groups 
                     'plateID', 'wellID', and 'fieldID' if applicable.
        timelapse (bool, optional): If True, limits the selection to one set of images to ensure 
                                    full sequence inclusion. Defaults to False.
        test_images (int, optional): The number of image sets to include in the test dataset. 
                                     Defaults to 10.
        random_test (bool, optional): If True, randomizes the selection of image sets. Defaults to True.
    Returns:
        str: The path to the test folder containing the selected images.
    Notes:
        - If a subdirectory named 'orig' exists within the source directory, it will be used as the 
          source for image selection.
        - The selected images are grouped by a combination of plate, well, and field identifiers 
          extracted from the filename using the provided regex.
        - The function ensures the test folder is created and populated with the selected images.
    """
    
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

def _choose_model(model_name, device, object_type='cell', restore_type=None, object_settings={}):
    """
    Selects and initializes a model based on the provided parameters.

    Args:
    -----------
    model_name : str
        The name of the model to use. Supported values include:
        - 'toxo_pv_lumen' (specific to 'pathogen' object_type)
        - 'sam', 'cyto', 'cyto2', 'cyto3', 'nuclei' (for default models)
    device : torch.device
        The device on which the model will run (e.g., CPU or GPU).
    object_type : str, optional
        The type of object the model is intended for. Default is 'cell'.
        Supported values:
        - 'pathogen': Uses a specific model for Toxoplasma PV lumen.
        - 'nucleus': Uses denoising or restoration models.
        - 'cell': Uses default Cellpose models.
    restore_type : str or None, optional
        The type of restoration to apply. Supported values:
        - 'denoise', 'deblur', 'upsample', or None (default).
        If None, standard models are used.
    object_settings : dict, optional
        Additional settings for specific object types. For example:
        - For 'pathogen', requires 'diameter' key to specify object diameter.

    Returns:
    --------
    model : CellposeModel or CellposeDenoiseModel
        The initialized model based on the provided parameters.

    Notes:
    ------
    - For 'pathogen' object_type with 'toxo_pv_lumen' model_name, a custom
      pretrained model is loaded from the 'models/cp' directory.
    - For 'nucleus' object_type, restoration models are initialized with
      specific restore types.
    - If an invalid restore_type is provided, it defaults to None with a warning.
    - GPU usage is determined automatically based on availability.

    Raises:
    -------
    KeyError:
        If 'diameter' is not provided in `object_settings` when using the
        'toxo_pv_lumen' model for 'pathogen' object_type.
    """
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
    """
    Selectively retain specific color channels in a 3-channel image tensor.

    This transformation zeros out unselected channels based on a list of 1-based
    indices corresponding to RGB channels.

    Args:
        channels (list of int): List of 1-based indices for channels to retain.
                                Valid values are:
                                - 1: Red
                                - 2: Green
                                - 3: Blue

    Example:
        >>> select_channels = SelectChannels([1, 3])
        >>> modified_img = select_channels(img)

    Note:
        The input image must be a PyTorch tensor with shape (3, H, W).
    """
    def __init__(self, channels):
        """
        Initialize the SelectChannels instance.

        Args:
            channels (list of int): Channels (1-based) to retain.
        """
        self.channels = channels
    
    def __call__(self, img):
        """
        Apply the channel selection to the input image tensor.

        Args:
            img (torch.Tensor): A 3-channel image tensor of shape (3, H, W).

        Returns:
            torch.Tensor: A tensor where unselected channels are set to zero.
        """
        img = img.clone()
        if 1 not in self.channels:
            img[0, :, :] = 0  # Zero out the red channel
        if 2 not in self.channels:
            img[1, :, :] = 0  # Zero out the green channel
        if 3 not in self.channels:
            img[2, :, :] = 0  # Zero out the blue channel
        return img
    
def preprocess_image_v1(image_path, image_size=224, channels=[1,2,3], normalize=True):
    """
    Preprocess an image for input into a machine learning model.

    Args:
        image_path (str): The file path to the image to be processed.
        image_size (int, optional): The size to which the image will be center-cropped. Defaults to 224.
        channels (list, optional): A list of channel indices to select from the image. Defaults to [1, 2, 3].
        normalize (bool, optional): Whether to normalize the image using mean and standard deviation. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - image (PIL.Image.Image): The original image loaded as a PIL Image object.
            - input_tensor (torch.Tensor): The preprocessed image as a PyTorch tensor with an added batch dimension.
    """

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            SelectChannels(channels),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            SelectChannels(channels)])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor

class SaliencyMapGenerator:
    """
    SaliencyMapGenerator is a utility class for generating saliency maps and visualizing model predictions
    for PyTorch models in binary classification tasks.
    """
    def __init__(self, model):
        """
        Initialize the SaliencyMapGenerator.
        """
        self.model = model

    def compute_saliency_maps(self, X, y):
        """
        Compute saliency maps for the given inputs and target labels.

        Args:
            X (torch.Tensor): Input tensor with requires_grad enabled.
            y (torch.Tensor): Ground truth labels for the input samples.

        Returns:
            torch.Tensor: The computed saliency maps, same shape as input.
        """
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
        """
        Compute saliency maps and predictions for the given input batch.

        Args:
            X (torch.Tensor): Input tensor with requires_grad enabled.

        Returns:
            tuple:
                - torch.Tensor: Saliency maps for the input.
                - torch.Tensor: Predicted class labels.
        """
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
        """
        Plot a grid of input images with overlaid saliency maps and predicted labels.

        Args:
            X (torch.Tensor): Input tensor of images (N, C, H, W).
            saliency (torch.Tensor): Corresponding saliency maps (N, C, H, W).
            predictions (torch.Tensor): Predicted class labels (N,).
            overlay (bool): Whether to overlay saliency maps on input images. Default is True.
            normalize (bool): Whether to normalize input images by percentiles. Default is False.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object showing the grid.
        """
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
        """
        Normalize an image's intensity per channel using percentile clipping.

        Args:
            img (np.ndarray): Image of shape (H, W, C) to be normalized.
            lower_percentile (int): Lower percentile for intensity clipping. Default is 2.
            upper_percentile (int): Upper percentile for intensity clipping. Default is 98.

        Returns:
            np.ndarray: Percentile-normalized image.
        """
        img_normalized = np.zeros_like(img)

        for c in range(img.shape[2]):  # Iterate over each channel
            low = np.percentile(img[:, :, c], lower_percentile)
            high = np.percentile(img[:, :, c], upper_percentile)
            img_normalized[:, :, c] = np.clip((img[:, :, c] - low) / (high - low), 0, 1)

        return img_normalized

class GradCAMGenerator:
    """
    GradCAMGenerator generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations
    for CNN-based PyTorch models, supporting binary classification tasks.
    """
    def __init__(self, model, target_layer, cam_type='gradcam'):
        """
        Initialize the GradCAMGenerator and register hooks.

        Args:
            model (torch.nn.Module): A trained PyTorch model.
            target_layer (str): The name of the layer to compute Grad-CAM on.
            cam_type (str): Type of CAM method (default is 'gradcam').
        """
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
        """
        Register forward and backward hooks to capture activations and gradients
        from the specified target layer during inference and backpropagation.
        """
        # Forward hook to get activations
        def forward_hook(module, input, output):
            self.activations = output

        # Backward hook to get gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer_module.register_forward_hook(forward_hook)
        self.target_layer_module.register_backward_hook(backward_hook)

    def get_layer(self, model, target_layer):
        """
        Recursively retrieve a layer object from a nested model.

        Args:
            model (torch.nn.Module): The model containing the layer.
            target_layer (str): Dot-separated string path to the desired layer.

        Returns:
            torch.nn.Module: The resolved layer module.
        """
        # Recursively find the layer specified in target_layer
        modules = target_layer.split('.')
        layer = model
        for module in modules:
            layer = getattr(layer, module)
        return layer

    def compute_gradcam_maps(self, X, y):
        """
        Compute Grad-CAM heatmaps for an input batch and target labels.

        Args:
            X (torch.Tensor): Input tensor of shape (N, C, H, W).
            y (torch.Tensor): Target labels (0 or 1) for each sample.

        Returns:
            np.ndarray: Grad-CAM heatmaps normalized to [0, 1] for each input.
        """
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
        """
        Compute Grad-CAM heatmaps and class predictions for a batch.

        Args:
            X (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Grad-CAM heatmaps for each sample.
                - Predicted class labels (0 or 1).
        """
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
        """
        Plot a grid of input images overlaid with Grad-CAM heatmaps.

        Args:
            X (torch.Tensor): Input image batch (N, C, H, W).
            gradcam (torch.Tensor): Grad-CAM heatmaps (N, H, W).
            predictions (torch.Tensor): Predicted class labels.
            overlay (bool): Whether to overlay Grad-CAM on input images.
            normalize (bool): Whether to normalize image intensities by percentiles.

        Returns:
            matplotlib.figure.Figure: The generated grid figure.
        """
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
        """
        Normalize each channel of the input image to the specified percentiles.

        Args:
            img (np.ndarray): Image array (H, W, C).
            lower_percentile (int): Lower clipping percentile.
            upper_percentile (int): Upper clipping percentile.

        Returns:
            np.ndarray: Percentile-normalized image.
        """
        img_normalized = np.zeros_like(img)

        for c in range(img.shape[2]):  # Iterate over each channel
            low = np.percentile(img[:, :, c], lower_percentile)
            high = np.percentile(img[:, :, c], upper_percentile)
            img_normalized[:, :, c] = np.clip((img[:, :, c] - low) / (high - low), 0, 1)

        return img_normalized

def preprocess_image(image_path, normalize=True, image_size=224, channels=[1,2,3]):
    """
    Preprocess an image for input into a machine learning model.

    Args:
        image_path (str): Path to the input image file.
        normalize (bool, optional): If True, apply ImageNet normalization 
            (mean and std). Defaults to True.
        image_size (int, optional): Target size (height and width) for resizing. 
            Defaults to 224.
        channels (list of int, optional): 1-based channel indices to retain (e.g., [1, 2, 3] 
            for RGB). Defaults to [1, 2, 3].

    Returns:
        tuple:
            - PIL.Image.Image: The original image.
            - torch.Tensor: The processed image tensor suitable for model input.
    """
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

def class_visualization(target_y, model_path, dtype, img_size=224, channels=[0,1,2], l2_reg=1e-3, learning_rate=25, num_iterations=100, blur_every=10, max_jitter=16, show_every=25, class_names = ['nc', 'pc']):
    def class_visualization(target_y, model_path, dtype, img_size=224, channels=[0,1,2], l2_reg=1e-3, learning_rate=25, num_iterations=100, blur_every=10, max_jitter=16, show_every=25, class_names=['nc', 'pc']):
        """
        Generate a class visualization for a given target class using gradient ascent.
        Args:
            target_y (int): The target class index for which the visualization is generated.
            model_path (str): Path to the pre-trained model file.
            dtype (torch.dtype): Data type for the tensors (e.g., torch.FloatTensor or torch.cuda.FloatTensor).
            img_size (int, optional): Size of the square input image. Default is 224.
            channels (list, optional): List of channel indices to use in the input image. Default is [0, 1, 2].
            l2_reg (float, optional): L2 regularization strength to prevent overfitting. Default is 1e-3.
            learning_rate (float, optional): Learning rate for gradient ascent. Default is 25.
            num_iterations (int, optional): Number of iterations for gradient ascent. Default is 100.
            blur_every (int, optional): Frequency (in iterations) to apply Gaussian blur as a regularizer. Default is 10.
            max_jitter (int, optional): Maximum pixel jitter for random image translation. Default is 16.
            show_every (int, optional): Frequency (in iterations) to display the intermediate visualization. Default is 25.
            class_names (list, optional): List of class names corresponding to the target indices. Default is ['nc', 'pc'].
        Returns:
            numpy.ndarray: The final deprocessed image as a NumPy array.
        Notes:
            - The function performs gradient ascent on the input image to maximize the score of the target class.
            - Regularization techniques such as L2 regularization, clamping, and Gaussian blur are applied to improve visualization quality.
            - The function assumes the model is compatible with the input image size and channels.
        """
    
    def jitter(img, ox, oy):
        # Randomly jitter the image
        return torch.roll(torch.roll(img, ox, dims=2), oy, dims=3)

    def blur_image(img, sigma=1):
        # Apply Gaussian blur to the image
        img_np = img.cpu().numpy()
        for i in range(img_np.shape[1]):
            img_np[:, i] = gaussian_filter(img_np[:, i], sigma=sigma)
        img.copy_(torch.tensor(img_np).to(img.device))

    def deprocess(img_tensor):
        # Convert the tensor image to a numpy array for visualization
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
    """
    Recursively retrieves the names of all submodules in a given model.

    Args:
        model (torch.nn.Module): The model whose submodules are to be retrieved.
        prefix (str, optional): A prefix to prepend to the names of the submodules. 
            Defaults to an empty string.

    Returns:
        list of str: A list of strings representing the full names of all submodules 
        in the model, including nested submodules.
    """
    submodules = []
    for name, module in model.named_children():
        full_name = prefix + ('.' if prefix else '') + name
        submodules.append(full_name)
        submodules.extend(get_submodules(module, full_name))
    return submodules

class GradCAM:
    """
    Compute Grad-CAM (Gradient-weighted Class Activation Mapping) for a given model and target layer(s).
    """
    def __init__(self, model, target_layers=None, use_cuda=True):
        """
        Initialize the GradCAM object.

        Args:
            model (nn.Module): The model for which Grad-CAM will be computed.
            target_layers (list of str): Names of layers to register hooks on.
            use_cuda (bool): Whether to use CUDA (GPU) for computation.
        """
        self.model = model
        self.model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def forward(self, input):
        """
        Run a forward pass through the model.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(input)

    def __call__(self, x, index=None):
        """
        Compute the Grad-CAM heatmap for an input image.

        Args:
            x (torch.Tensor): Input tensor of shape (1, C, H, W).
            index (int or None): Class index to compute gradients for. If None, uses the predicted class.

        Returns:
            numpy.ndarray: Normalized Grad-CAM heatmap of shape (H, W).
        """
        if self.cuda:
            x = x.cuda()

        features = []
        def hook(module, input, output):
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
    """
    Overlay a heatmap generated from a mask onto an image.

    This function applies a color map to the mask, combines it with the input
    image, and normalizes the result to create a visually interpretable
    representation of the mask overlaid on the image.

    Args:
        img (numpy.ndarray): The input image as a NumPy array with pixel values
            normalized between 0 and 1.
        mask (numpy.ndarray): The mask to overlay on the image, with values
            normalized between 0 and 1.

    Returns:
        numpy.ndarray: The resulting image with the heatmap overlay, as a
        NumPy array with pixel values in the range [0, 255].
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def recommend_target_layers(model):
    """
    Identifies and recommends target layers in a given model for further processing.

    This function iterates through all the modules in the provided model and collects
    the names of all 2D convolutional layers (`torch.nn.Conv2d`). It then recommends
    the last convolutional layer as the primary target layer.

    Args:
        model (torch.nn.Module): The neural network model to analyze.

    Returns:
        tuple:
            - list: A list containing the name of the recommended target layer (last Conv2d layer).
            - list: A list of all Conv2d layer names found in the model.

    Raises:
        ValueError: If no convolutional layers (`torch.nn.Conv2d`) are found in the model.
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
    """
    Compute Integrated Gradients for model interpretability.

    This class implements the Integrated Gradients method to attribute the prediction
    of a neural network to its input features. It approximates the integral of gradients
    along a straight path from a baseline to the input.
    """
    

    def __init__(self, model):
        """
        Initialize the IntegratedGradients instance.

        Args:
            model (torch.nn.Module): A trained PyTorch model.
        """
        self.model = model
        self.model.eval()

    def generate_integrated_gradients(self, input_tensor, target_label_idx, baseline=None, num_steps=50):
        """
        Compute the integrated gradients for a given input and target class.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (1, C, H, W) or similar.
            target_label_idx (int): Index of the target class for which gradients are computed.
            baseline (torch.Tensor, optional): Baseline tensor with the same shape as input. Defaults to zeros.
            num_steps (int, optional): Number of steps in the Riemann approximation of the integral. Defaults to 50.

        Returns:
            np.ndarray: Integrated gradients as a NumPy array with the same shape as `input_tensor`.
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
    """
    Generate a list of database file paths based on the given source(s).

    This function takes a single source path or a list of source paths and 
    constructs the corresponding paths to the 'measurements.db' file located 
    in the 'measurements' subdirectory of each source.

    Args:
        src (str or list of str): A single source path as a string or a list 
            of source paths.

    Returns:
        list of str: A list of file paths pointing to 'measurements/measurements.db' 
        for each source in the input.
    """
    if isinstance(src, str):
        src = [src]
    db_paths = [os.path.join(source, 'measurements/measurements.db') for source in src]
    return db_paths

def get_sequencing_paths(src):
    """
    Generate a list of file paths pointing to sequencing data CSV files.

    This function takes a single source path or a list of source paths and 
    constructs the full file paths to the 'sequencing/sequencing_data.csv' 
    file located within each source directory.

    Args:
        src (str or list of str): A single source directory path as a string 
            or a list of source directory paths.

    Returns:
        list of str: A list of full file paths to the 'sequencing_data.csv' 
        files for each source directory.
    """
    if isinstance(src, str):
        src = [src]
    seq_paths = [os.path.join(source, 'sequencing/sequencing_data.csv') for source in src]
    return seq_paths

def load_image_paths(c, visualize):
    """
    Loads image paths from a database table and optionally filters them based on a visualization keyword.

    Args:
        c (sqlite3.Cursor): A database cursor object used to execute SQL queries.
        visualize (str): A keyword to filter image paths. If provided, only rows where the 'png_path' column
                         contains the keyword followed by '_png' will be included. If None or empty, no filtering
                         is applied.

    Returns:
        pandas.DataFrame: A DataFrame containing the image paths and other associated data from the 'png_list' table.
                          The DataFrame is indexed by the 'prcfo' column.
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
    """
    Merges two pandas DataFrames on their indices and optionally displays the result.

    Parameters:
        df (pandas.DataFrame): The main DataFrame to be merged. It must have a column named 'prcfo',
                               which will be set as the index before merging.
        image_paths_df (pandas.DataFrame): The DataFrame containing image paths to be merged with `df`.
        verbose (bool): If True, the resulting merged DataFrame will be displayed.

    Returns:
        pandas.DataFrame: The merged DataFrame with the indices aligned.
    """
    df.set_index('prcfo', inplace=True)
    df = image_paths_df.merge(df, left_index=True, right_index=True)
    if verbose:
        display(df)
    return df

def remove_highly_correlated_columns_v1(df, threshold):
    """
    Removes columns from a DataFrame that are highly correlated with other columns.

    This function calculates the correlation matrix of the given DataFrame, identifies
    columns with a correlation higher than the specified threshold, and removes them
    to reduce multicollinearity.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        threshold (float): The correlation threshold above which columns are considered
            highly correlated and will be removed.

    Returns:
        pandas.DataFrame: A DataFrame with highly correlated columns removed.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {'A': [1, 2, 3], 'B': [2, 4, 6], 'C': [7, 8, 9]}
        >>> df = pd.DataFrame(data)
        >>> remove_highly_correlated_columns_v1(df, threshold=0.9)
           A  C
    """
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(to_drop, axis=1)

def filter_columns(df, filter_by):
    """
    Filters the columns of a DataFrame based on a specified criterion.

    Args:
        df (pandas.DataFrame): The input DataFrame whose columns are to be filtered.
        filter_by (str): The criterion for filtering columns. If 'morphology', 
                         columns containing 'channel' in their names are excluded. 
                         Otherwise, only columns containing the specified string 
                         are included.

    Returns:
        pandas.DataFrame: A DataFrame containing only the filtered columns.
    """
    if filter_by != 'morphology':
        cols_to_include = [col for col in df.columns if filter_by in str(col)]
    else:
        cols_to_include = [col for col in df.columns if 'channel' not in str(col)]
    df = df[cols_to_include]
    return df

def reduction_and_clustering(numeric_data, n_neighbors, min_dist, metric, eps, min_samples, clustering, reduction_method='umap', verbose=False, embedding=None, n_jobs=-1, mode='fit', model=False):
    """
    Perform dimensionality reduction and clustering on the given data.
    
    Args:
    numeric_data (np.ndarray): Numeric data for embedding and clustering.
    n_neighbors (int or float): Number of neighbors for UMAP or perplexity for t-SNE.
    min_dist (float): Minimum distance for UMAP.
    metric (str): Metric for UMAP and DBSCAN.
    eps (float): Epsilon for DBSCAN.
    min_samples (int): Minimum samples for DBSCAN or number of clusters for KMeans.
    clustering (str): Clustering method ('DBSCAN' or 'KMeans').
    reduction_method (str): Dimensionality reduction method ('UMAP' or 'tSNE').
    verbose (bool): Whether to print verbose output.
    embedding (np.ndarray, optional): Precomputed embedding. Default is None.
    return_model (bool): Whether to return the reducer model. Default is False.
    
    Returns:
    tuple: embedding, labels (and optionally the reducer model)
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
                        n_iter=1000,
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
    """
    Removes noise from the given embedding and labels by filtering out elements
    where the corresponding label is -1.

    Args:
        embedding (numpy.ndarray): The embedding array, where each row corresponds
            to a data point.
        labels (numpy.ndarray): The array of labels corresponding to the embedding,
            where a label of -1 indicates noise.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The filtered embedding array with noise removed.
            - numpy.ndarray: The filtered labels array with noise removed.
    """
    non_noise_indices = labels != -1
    embedding = embedding[non_noise_indices]
    labels = labels[non_noise_indices]
    return embedding, labels

def plot_embedding(embedding, image_paths, labels, image_nr, img_zoom, colors, plot_by_cluster, plot_outlines, plot_points, plot_images, smooth_lines, black_background, figuresize, dot_size, remove_image_canvas, verbose):
    """
    Plots a 2D embedding with optional images, clusters, and customization options.

    Args:
        embedding (np.ndarray): A 2D array of shape (n_samples, 2) representing the embedding coordinates.
        image_paths (list or None): A list of file paths to images corresponding to the data points, or None if no images are used.
        labels (np.ndarray): An array of cluster labels for each data point.
        image_nr (int): The number of images to display on the plot.
        img_zoom (float): The zoom factor for the displayed images.
        colors (list or None): A list of colors to use for clusters, or None to use default colors.
        plot_by_cluster (bool): Whether to plot images grouped by cluster.
        plot_outlines (bool): Whether to draw outlines around clusters.
        plot_points (bool): Whether to plot individual data points.
        plot_images (bool): Whether to overlay images on the embedding.
        smooth_lines (bool): Whether to draw smooth lines between cluster centers.
        black_background (bool): Whether to use a black background for the plot.
        figuresize (tuple): The size of the figure in inches (width, height).
        dot_size (float): The size of the dots representing data points.
        remove_image_canvas (bool): Whether to remove the canvas around the images.
        verbose (bool): Whether to print verbose output during the plotting process.

    Returns:
        matplotlib.figure.Figure: The generated plot as a Matplotlib figure object.
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
    """
    Generate a set of RGBA colors for visualization purposes.

    This function generates a list of random RGBA colors, appends specific predefined colors,
    and optionally includes a black background color.

    Args:
        num_clusters (int): The number of clusters for which colors need to be generated.
                            Additional random colors will be generated beyond the predefined ones.
        black_background (bool): If True, a black background color ([0, 0, 0, 1]) will be included
                                 at the beginning of the color list.

    Returns:
        numpy.ndarray: A 2D array of shape (num_colors, 4), where each row represents an RGBA color.
                       The first dimension corresponds to the total number of colors generated.
    """
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
    """
    Assigns colors to unique labels and creates a mapping from labels to color indices.

    Args:
        unique_labels (list or iterable): A collection of unique labels for which colors need to be assigned.
        random_colors (numpy.ndarray or list): An array or list of RGB color values, where each color is represented 
            as a triplet of integers in the range [0, 255].

    Returns:
        tuple: A tuple containing:
            - colors (list of tuple): A list of RGB color tuples in the original [0, 255] range.
            - label_to_color_index (dict): A dictionary mapping each unique label to its corresponding color index.
    """
    normalized_colors = random_colors / 255
    colors_img = [tuple(color) for color in normalized_colors]
    colors = [tuple(color) for color in random_colors]
    label_to_color_index = {label: index for index, label in enumerate(unique_labels)}
    return colors, label_to_color_index

def setup_plot(figuresize, black_background):
    """
    Sets up a matplotlib plot with specified figure size and background color.

    Args:
        figuresize (float): The size of the figure in inches (used for both width and height).
        black_background (bool): If True, sets the plot to have a black background with white text and labels.
                                 If False, sets the plot to have a white background with black text and labels.

    Returns:
        tuple: A tuple containing the figure (`matplotlib.figure.Figure`) and axes (`matplotlib.axes._axes.Axes`) objects.
    """
    if black_background:
        plt.rcParams.update({'figure.facecolor': 'black', 'axes.facecolor': 'black', 'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'axes.labelcolor': 'white'})
    else:
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black'})
    fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
    return fig, ax

def plot_clusters(ax, embedding, labels, colors, cluster_centers, plot_outlines, plot_points, smooth_lines, figuresize=10, dot_size=50, verbose=False):
    """
    Plots clusters on a 2D embedding using matplotlib.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
        embedding (numpy.ndarray): A 2D array of shape (n_samples, 2) representing the embedding coordinates.
        labels (numpy.ndarray): An array of cluster labels for each point in the embedding.
        colors (list): A list of colors corresponding to each cluster.
        cluster_centers (numpy.ndarray): A 2D array of shape (n_clusters, 2) representing the coordinates of cluster centers.
        plot_outlines (bool): Whether to plot the outlines of clusters using convex hulls or smoothed lines.
        plot_points (bool): Whether to plot individual points in the clusters.
        smooth_lines (bool): Whether to use smoothed lines for cluster outlines instead of convex hulls.
        figuresize (int, optional): The size of the figure. Defaults to 10.
        dot_size (int, optional): The size of the points in the scatter plot. Defaults to 50.
        verbose (bool, optional): Whether to print additional information for debugging. Defaults to False.

    Returns:
        None

    Notes:
        - This function assumes that the embedding is 2D.
        - Cluster labels should be integers, with -1 typically representing noise.
        - The function uses matplotlib for plotting and assumes that the required libraries (e.g., numpy, matplotlib) are imported.
    """
    unique_labels = np.unique(labels)
    for cluster_label, color, center in zip(unique_labels, colors, cluster_centers):
        cluster_data = embedding[labels == cluster_label]
        if smooth_lines:
            if cluster_data.shape[0] > 2:
                x_smooth, y_smooth = smooth_hull_lines(cluster_data)
                if plot_outlines:
                    plt.plot(x_smooth, y_smooth, color=color, linewidth=2)
        else:
            if cluster_data.shape[0] > 2:
                hull = ConvexHull(cluster_data)
                for simplex in hull.simplices:
                    if plot_outlines:
                        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], color=color, linewidth=4)
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
    """
    Plots UMAP embeddings with associated images on a given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis on which to plot the images.
        image_paths (list of str): List of file paths to the images to be plotted.
        embedding (numpy.ndarray): 2D array of UMAP embeddings with shape (n_samples, 2).
        labels (numpy.ndarray): Array of cluster labels for each embedding point.
        image_nr (int): Number of images to plot.
        img_zoom (float): Zoom factor for the images.
        colors (list): List of colors for each cluster.
        plot_by_cluster (bool): If True, plot images grouped by cluster; otherwise, plot randomly sampled images.
        remove_image_canvas (bool): If True, remove the image canvas (background) when plotting.
        verbose (bool): If True, print additional information during execution.

    Returns:
        None
    """
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
    """
    Plots images on a given axis based on their cluster assignments and embeddings.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis on which to plot the images.
        image_paths (list of str): List of file paths to the images to be plotted.
        embedding (array-like): 2D array of shape (n_samples, 2) containing the x and y coordinates for each image.
        labels (array-like): Array of cluster labels for each image. -1 indicates noise or unclustered points.
        image_nr (int): Maximum number of images to display per cluster.
        img_zoom (float): Zoom factor for the displayed images.
        colors (list of str): List of colors corresponding to each cluster.
        cluster_indices (dict): Dictionary mapping cluster labels to lists of indices of images in each cluster.
        remove_image_canvas (bool): If True, removes the canvas (border) around the plotted images.
        verbose (bool): If True, prints additional information during execution.

    Returns:
        None
    """
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
    """
    Plots an image on a given matplotlib axis at specified coordinates.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the image.
        x (float): The x-coordinate where the image will be placed.
        y (float): The y-coordinate where the image will be placed.
        img (numpy.ndarray or array-like): The image data to be plotted.
        img_zoom (float): The zoom factor for the image.
        remove_image_canvas (bool, optional): If True, removes the canvas 
            (e.g., padding or borders) from the image before plotting. 
            Defaults to True.

    Returns:
        None
    """
    img = np.array(img)
    if remove_image_canvas:
        img = remove_canvas(img)
    imagebox = OffsetImage(img, zoom=img_zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

def remove_canvas(img):
    """
    Converts an image to a normalized RGBA format by adding an alpha channel.

    This function processes images in either grayscale ('L', 'I') or RGB ('RGB') mode.
    For grayscale images, the pixel values are normalized, and an alpha channel is
    created based on non-zero pixel values. For RGB images, the pixel values are
    normalized to the range [0, 1], and an alpha channel is created based on the
    presence of non-zero pixel values across all channels.

    Args:
        img (PIL.Image.Image): The input image to process. Must be in 'L', 'I', or 'RGB' mode.

    Returns:
        numpy.ndarray: A 4-channel RGBA image as a NumPy array, where the first three
        channels represent the normalized RGB values, and the fourth channel represents
        the alpha channel.

    Raises:
        ValueError: If the input image mode is not 'L', 'I', or 'RGB'.
    """
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
    """
    Plot a grid of images for each cluster based on the given labels and embeddings.

    Args:
        embedding (np.ndarray): Embedding of data points for visualization.
        labels (np.ndarray): Cluster labels for each data point. A value of -1 indicates noise or outliers.
        image_nr (int): Maximum number of images to display per cluster.
        image_paths (list of str): File paths to images corresponding to the data points.
        colors (list of str): List of colors for each cluster.
        figuresize (tuple): Size of the figure (width, height).
        black_background (bool): Whether to use a black background.
        verbose (bool): Whether to print progress information.

    Returns:
        matplotlib.figure.Figure or None: The generated figure, or None if no valid clusters are found.

    Notes:
        - Clusters larger than `image_nr` are randomly subsampled.
        - If all labels are -1, the function returns None.
        - Relies on an external `plot_grid` function for grid rendering.
    """
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
    """
    Plot a grid of images grouped by cluster with optional background and labels.

    Args:
        cluster_images (dict): Dictionary mapping cluster labels to lists of images.
        colors (list): List of RGB tuples specifying colors for each cluster.
        figuresize (float): Base figure size; actual size scales with the number of clusters.
        black_background (bool): If True, use a black background; otherwise, use white.
        verbose (bool): If True, print cluster labels and index info during plotting.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the image grid.

    Notes:
        - Grid size is dynamically adjusted per cluster.
        - Cluster labels are shown alongside image grids using corresponding colors.
        - A maximum figure size limit prevents overly large plots.
    """
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
    """
    Generate a list of file paths from a SQLite database using optional metadata filters.

    Args:
        db_path (str): Path to the SQLite database.
        file_metadata (str | list[str] | None): Filter criteria for file paths.
            - str: Only include paths containing the string.
            - list of str: Include paths containing any of the strings.
            - None or empty: Include all paths.

    Returns:
        list[str] or None: List of matching file paths, or None if an error occurs.

    Raises:
        sqlite3.Error: If a database operation fails.
        Exception: For any other unexpected error.

    Notes:
        - Paths are fetched from the 'png_list' table using the 'png_path' column.
        - Results are retrieved in batches of 1000 rows for efficiency.
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
    """
    Adjust file paths to include the specified base directory and folder.

    Args:
        df (pandas.DataFrame or list): Input containing file paths.
            - If a DataFrame, it must have a 'png_path' column.
            - If a list, it should contain file path strings.
        base_path (str): Base directory to prepend if not already present.
        folder (str, optional): Folder name to insert into paths (default: 'data').

    Returns:
        tuple or list: 
            - If input is a DataFrame: (updated DataFrame, list of adjusted paths).
            - If input is a list: list of adjusted paths.

    Raises:
        ValueError: If the DataFrame does not contain a 'png_path' column.

    Notes:
        Paths already containing the base path are not modified.
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
    """
    Deletes a folder and all of its contents, including subdirectories and files.

    Args:
        folder_path (str): The path to the folder to be deleted.

    Behavior:
        - If the specified folder exists and is a directory, it recursively deletes all files 
          and subdirectories within it, and then removes the folder itself.
        - If the folder does not exist or is not a directory, a message is printed indicating this.

    Prints:
        - A confirmation message if the folder is successfully deleted.
        - An error message if the folder does not exist or is not a directory.

    Example:
        delete_folder('/path/to/folder')
    """
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
    """
    Adjusts the source folder in the settings dictionary for test mode.

    If `test_mode` is enabled in the `settings` dictionary, this function:
    - Checks if the current source folder (`settings['src']`) is not already set to 'test'.
    - Selects a random subset of files from the source folder based on `settings['test_nr']`.
    - Copies the selected files into a new 'test/merged' directory.
    - Updates the `settings['src']` to point to the new 'test/merged' directory.
    - Prints a message indicating the change in the source folder.

    If the source folder is already set to 'test', it simply prints a message confirming the test mode.

    Args:
        settings (dict): A dictionary containing configuration settings. 
                        Expected keys:
                        - 'test_mode' (bool): Whether test mode is enabled.
                        - 'src' (str): Path to the source folder.
                        - 'test_nr' (int): Number of files to select for test mode.

    Returns:
        dict: The updated settings dictionary with the modified source folder if test mode is enabled.
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
    """
    Preprocesses the given dataframe by applying filtering, removing highly correlated columns,
    applying log transformation, filling NaN values, and scaling the numeric data.

    Args:
    df (pandas.DataFrame): The input dataframe.
    filter_by (str or None): The channel of interest to filter the dataframe by.
    remove_highly_correlated (bool or float): Whether to remove highly correlated columns.
    If a float is provided, it represents the correlation threshold.
    log_data (bool): Whether to apply log transformation to the numeric data.
    exclude (list or None): List of features to exclude from the filtering process.
    verbose (bool): Whether to print verbose output during preprocessing.

    Returns:
    numpy.ndarray: The preprocessed numeric data.

    Raises:
    ValueError: If no numeric columns are available after filtering.

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
    """
    Removes columns from the dataframe that have low variance.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    threshold (float): The variance threshold below which columns will be removed.

    Returns:
    pandas.DataFrame: The DataFrame with low variance columns removed.
    """

    numerical_cols = df.select_dtypes(include=[np.number])
    low_variance_cols = numerical_cols.var()[numerical_cols.var() < threshold].index.tolist()

    if verbose:
        print(f"Removed columns due to low variance: {low_variance_cols}")

    df = df.drop(columns=low_variance_cols)
    
    return df

def remove_highly_correlated_columns(df, threshold=0.95, verbose=False):
    """
    Removes columns from the dataframe that are highly correlated with one another.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    threshold (float): The correlation threshold above which columns will be removed.

    Returns:
    pandas.DataFrame: The DataFrame with highly correlated columns removed.
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
    
    """
    Filter the dataframe `df` based on the specified `channel_of_interest` and `exclude` parameters.

    Args:
    - df (pandas.DataFrame): The input dataframe to be filtered.
    - channel_of_interest (str, int, list, None): The channel(s) of interest to filter the dataframe. If None, no filtering is applied. If 'morphology', only morphology features are included.If an integer, only the specified channel is included. If a list, only the specified channels are included.If a string, only the specified channel is included.
    - exclude (str, list, None): The feature(s) to exclude from the filtered dataframe. If None, no features are excluded. If a string, the specified feature is excluded.If a list, the specified features are excluded.

    Returns:
    - filtered_df (pandas.DataFrame): The filtered dataframe based on the specified parameters.
    - features (list): The list of selected features after filtering.

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
    """
    Checks if the current position overlaps with any of the other positions 
    within a specified threshold distance.

    Args:
        current_position (iterable): The current position as a list, tuple, or array of coordinates.
        other_positions (iterable): A collection of positions to compare against, 
                                    where each position is a list, tuple, or array of coordinates.
        threshold (float): The distance threshold below which two positions are considered overlapping.

    Returns:
        bool: True if the current position overlaps with any of the other positions, 
              False otherwise.
    """
    for other_position in other_positions:
        distance = np.linalg.norm(np.array(current_position) - np.array(other_position))
        if distance < threshold:
            return True
    return False

# Define a function to try random positions around a given point
def find_non_overlapping_position(x, y, image_positions, threshold, max_attempts=100):
    """
    Finds a new position near the given coordinates (x, y) that does not overlap
    with any of the positions in the provided image_positions list, based on a
    specified threshold.

    Args:
        x (float): The x-coordinate of the initial position.
        y (float): The y-coordinate of the initial position.
        image_positions (list of tuples): A list of (x, y) tuples representing
            existing positions to avoid overlapping with.
        threshold (float): The minimum distance required to avoid overlap.
        max_attempts (int, optional): The maximum number of attempts to find a
            non-overlapping position. Defaults to 100.

    Returns:
        tuple: A tuple (new_x, new_y) representing the new non-overlapping
        position. If no suitable position is found within the maximum attempts,
        the original position (x, y) is returned.
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
    """
    Perform dimensionality reduction and clustering on the given data.
    
    Args:
    numeric_data (np.array): Numeric data to process.
    n_neighbors (int): Number of neighbors for UMAP or perplexity for tSNE.
    min_dist (float): Minimum distance for UMAP.
    metric (str): Metric for UMAP, tSNE, and DBSCAN.
    eps (float): Epsilon for DBSCAN clustering.
    min_samples (int): Minimum samples for DBSCAN or number of clusters for KMeans.
    clustering (str): Clustering method ('DBSCAN' or 'KMeans').
    reduction_method (str): Dimensionality reduction method ('UMAP' or 'tSNE').
    verbose (bool): Whether to print verbose output.
    reduction_param (dict): Additional parameters for the reduction method.
    embedding (np.array): Precomputed embedding (optional).
    n_jobs (int): Number of parallel jobs to run.

    Returns:
    embedding (np.array): Embedding of the data.
    labels (np.array): Cluster labels.
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
    """
    Load and transform an image into a normalized tensor.

    Applies resizing to 224x224, converts to a tensor, and normalizes using
    ImageNet mean and standard deviation.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Transformed image tensor with shape (1, 3, 224, 224).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def extract_features(image_paths, resnet=resnet50):
    """
    Extracts features from a list of image paths using a pre-trained ResNet model.

    Args:
        image_paths (list of str): A list of file paths to the images from which features are to be extracted.
        resnet (torchvision.models, optional): A ResNet model class to use for feature extraction. 
            Defaults to torchvision.models.resnet50.

    Returns:
        numpy.ndarray: A 2D array where each row corresponds to the extracted features of an image.
    """
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
    """
    Test whether a given data series follows a normal distribution.

    This function uses the D'Agostino and Pearson's test to check the null 
    hypothesis that the data comes from a normal distribution. If the p-value 
    is less than the significance level (alpha), the null hypothesis is rejected.

    Args:
        series (array-like): The data series to test for normality.

    Returns:
        bool: True if the data follows a normal distribution (p >= alpha), 
              False otherwise.
    """
    k2, p = stats.normaltest(series)
    alpha = 0.05
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        return False
    return True

def random_forest_feature_importance(all_df, cluster_col='cluster'):
    """
    Computes feature importance using a Random Forest Classifier.

    This function takes a DataFrame, selects numeric features, and computes
    the importance of each feature in predicting the specified cluster column
    using a Random Forest Classifier. The results are returned as a sorted
    DataFrame of feature importances.

    Args:
        all_df (pd.DataFrame): The input DataFrame containing the data.
        cluster_col (str, optional): The name of the column representing the 
            target variable (cluster). Defaults to 'cluster'.

    Returns:
        pd.DataFrame: A DataFrame containing the features and their corresponding
        importance scores, sorted in descending order of importance.

    Notes:
        - The function assumes that the target column (`cluster_col`) is numeric.
        - Standard scaling is applied to the numeric features before fitting the model.
        - The Random Forest Classifier is initialized with 100 estimators and a 
          random state of 42 for reproducibility.
    """
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
    """
    Perform ANOVA and Kruskal-Wallis tests on numeric features grouped by clusters.

    This function evaluates whether numeric features differ significantly across groups 
    defined by the `cluster_col`.

    Args:
        all_df (pd.DataFrame): DataFrame containing numeric features and cluster assignments.
        cluster_col (str, optional): Name of the column indicating cluster/group labels. Defaults to 'cluster'.

    Returns:
        tuple:
            - anova_df (pd.DataFrame): ANOVA test results with columns 
              ['Feature', 'ANOVA_Statistic', 'ANOVA_pValue'].
            - kruskal_df (pd.DataFrame): Kruskal-Wallis test results with columns 
              ['Feature', 'Kruskal_Statistic', 'Kruskal_pValue'].

    Notes:
        - Normality of each feature is assessed using `check_normality`.
        - ANOVA is used for normally distributed features.
        - Kruskal-Wallis is used for non-normal features.
        - Assumes `check_normality`, `scipy.stats.f_oneway`, and `scipy.stats.kruskal` are available.
    """
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
    """
    Combines results from multiple DataFrames into a single DataFrame.

    This function merges three DataFrames (`rf_df`, `anova_df`, and `kruskal_df`) 
    on the 'Feature' column using a left join. The resulting DataFrame contains 
    all features from `rf_df` and their corresponding data from `anova_df` and 
    `kruskal_df` where available.

    Args:
        rf_df (pd.DataFrame): A DataFrame containing features and their associated 
                              data from a random forest analysis.
        anova_df (pd.DataFrame): A DataFrame containing features and their associated 
                                 data from an ANOVA analysis.
        kruskal_df (pd.DataFrame): A DataFrame containing features and their associated 
                                   data from a Kruskal-Wallis analysis.

    Returns:
        pd.DataFrame: A combined DataFrame with features and their associated data 
                      from all three input DataFrames.
    """
    combined_df = rf_df.merge(anova_df, on='Feature', how='left')
    combined_df = combined_df.merge(kruskal_df, on='Feature', how='left')
    return combined_df

def cluster_feature_analysis(all_df, cluster_col='cluster'):
    """
    Perform feature analysis for clustering by combining results from 
    random forest feature importance and statistical tests.

    This function calculates feature importance using a random forest model, 
    performs statistical tests (ANOVA and Kruskal-Wallis) to assess the 
    significance of features, and combines the results into a single DataFrame.

    Args:
        all_df (pd.DataFrame): The input DataFrame containing features and cluster labels.
        cluster_col (str, optional): The name of the column representing cluster labels. 
                                     Defaults to 'cluster'.

    Returns:
        pd.DataFrame: A DataFrame combining the results of random forest feature 
                      importance and statistical tests for feature analysis.
    """
    rf_df = random_forest_feature_importance(all_df, cluster_col)
    anova_df, kruskal_df = perform_statistical_tests(all_df, cluster_col)
    combined_df = combine_results(rf_df, anova_df, kruskal_df)
    return combined_df

def _merge_cells_based_on_parasite_overlap(parasite_mask, cell_mask, nuclei_mask, overlap_threshold=5, perimeter_threshold=30):
    
    """
    Merge cells in cell_mask if a parasite in parasite_mask overlaps with more than one cell,
    and if cells share more than a specified perimeter percentage.

    Args:
        parasite_mask (ndarray): Mask of parasites.
        cell_mask (ndarray): Mask of cells.
        nuclei_mask (ndarray): Mask of nuclei.
        overlap_threshold (float): The percentage threshold for merging cells based on parasite overlap.
        perimeter_threshold (float): The percentage threshold for merging cells based on shared perimeter.

    Returns:
        ndarray: The modified cell mask (cell_mask) with unique labels.
    """
    labeled_cells = label(cell_mask)
    labeled_parasites = label(parasite_mask)
    labeled_nuclei = label(nuclei_mask)
    num_parasites = np.max(labeled_parasites)
    num_cells = np.max(labeled_cells)
    num_nuclei = np.max(labeled_nuclei)

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

def _merge_cells_without_nucleus(adj_cell_mask: np.ndarray, nuclei_mask: np.ndarray):
    """
    Relabel any cell that lacks a nucleus to the ID of an adjacent
    cell that *does* contain a nucleus.

    Args
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

def process_mask_file_adjust_cell(file_name, parasite_folder, cell_folder, nuclei_folder, overlap_threshold, perimeter_threshold):
    """
    Processes and adjusts a cell mask file based on parasite overlap and perimeter thresholds.

    This function loads parasite, cell, and nuclei mask files, merges cells based on parasite overlap
    and perimeter thresholds, and saves the updated cell mask back to the file system.

    Args:
        file_name (str): The name of the mask file to process.
        parasite_folder (str): The directory containing parasite mask files.
        cell_folder (str): The directory containing cell mask files.
        nuclei_folder (str): The directory containing nuclei mask files.
        overlap_threshold (float): The threshold for parasite overlap to merge cells.
        perimeter_threshold (float): The threshold for cell perimeter to merge cells.

    Returns:
        float: The time taken to process the mask file, in seconds.

    Raises:
        ValueError: If the corresponding cell or nuclei mask file for the given file_name is not found.
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

    merged_cell_mask = _merge_cells_based_on_parasite_overlap(parasite_mask, cell_mask, nuclei_mask, overlap_threshold, perimeter_threshold)
    #merged_cell_mask = _merge_cells_without_nucleus(merged_cell_mask, nuclei_mask)

    np.save(cell_path, merged_cell_mask)

    end = time.perf_counter()
    return end - start

def adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, overlap_threshold=5, perimeter_threshold=30, n_jobs=None):
    """
    Adjusts cell masks based on parasite, cell, and nuclei data files.
    This function processes `.npy` files from the specified folders to adjust cell masks
    by considering overlap and perimeter thresholds. It uses multiprocessing to parallelize
    the processing of files.
    Args:
        parasite_folder (str): Path to the folder containing parasite `.npy` files.
        cell_folder (str): Path to the folder containing cell `.npy` files.
        nuclei_folder (str): Path to the folder containing nuclei `.npy` files.
        overlap_threshold (int, optional): Threshold for overlap adjustment. Defaults to 5.
        perimeter_threshold (int, optional): Threshold for perimeter adjustment. Defaults to 30.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to the number of CPU cores minus 2.
    Raises:
        ValueError: If the number of files in the parasite, cell, and nuclei folders do not match.
    Notes:
        - The function assumes that the files in the folders are named in a way that allows
          them to be sorted and matched correctly.
        - Progress is printed to the console during processing.
    Returns:
        None
    """
    
    parasite_files = sorted([f for f in os.listdir(parasite_folder) if f.endswith('.npy')])
    cell_files = sorted([f for f in os.listdir(cell_folder) if f.endswith('.npy')])
    nuclei_files = sorted([f for f in os.listdir(nuclei_folder) if f.endswith('.npy')])

    if not (len(parasite_files) == len(cell_files) == len(nuclei_files)):
        raise ValueError("The number of files in the folders do not match.")

    n_jobs = n_jobs or max(1, cpu_count() - 2)

    time_ls = []
    files_to_process = len(parasite_files)
    process_fn = partial(process_mask_file_adjust_cell,
                         parasite_folder=parasite_folder,
                         cell_folder=cell_folder,
                         nuclei_folder=nuclei_folder,
                         overlap_threshold=overlap_threshold,
                         perimeter_threshold=perimeter_threshold)

    with Pool(n_jobs) as pool:
        for i, duration in enumerate(pool.imap_unordered(process_fn, parasite_files), 1):
            time_ls.append(duration)
            print_progress(i, files_to_process, n_jobs=n_jobs, time_ls=time_ls, batch_size=None, operation_type='adjust_cell_masks')

def adjust_cell_masks_v1(parasite_folder, cell_folder, nuclei_folder, overlap_threshold=5, perimeter_threshold=30):

    """
    Process all npy files in the given folders. Merge and relabel cells in cell masks
    based on parasite overlap and cell perimeter sharing conditions.

    Args:
        parasite_folder (str): Path to the folder containing parasite masks.
        cell_folder (str): Path to the folder containing cell masks.
        nuclei_folder (str): Path to the folder containing nuclei masks.
        overlap_threshold (float): The percentage threshold for merging cells based on parasite overlap.
        perimeter_threshold (float): The percentage threshold for merging cells based on shared perimeter.
    """

    parasite_files = sorted([f for f in os.listdir(parasite_folder) if f.endswith('.npy')])
    cell_files = sorted([f for f in os.listdir(cell_folder) if f.endswith('.npy')])
    nuclei_files = sorted([f for f in os.listdir(nuclei_folder) if f.endswith('.npy')])
    
    # Ensure there are matching files in all folders
    if not (len(parasite_files) == len(cell_files) == len(nuclei_files)):
        raise ValueError("The number of files in the folders do not match.")
    
    # Match files by name
    time_ls = []
    files_to_process = len(parasite_files)
    for files_processed, file_name in enumerate(parasite_files):
        start = time.time()
        parasite_path = os.path.join(parasite_folder, file_name)
        cell_path = os.path.join(cell_folder, file_name)
        nuclei_path = os.path.join(nuclei_folder, file_name)
        # Check if the corresponding cell and nuclei mask files exist
        if not (os.path.exists(cell_path) and os.path.exists(nuclei_path)):
            raise ValueError(f"Corresponding cell or nuclei mask file for {file_name} not found.")
        # Load the masks
        parasite_mask = np.load(parasite_path, allow_pickle=True)
        cell_mask = np.load(cell_path, allow_pickle=True)
        nuclei_mask = np.load(nuclei_path, allow_pickle=True)
        
        # Merge and relabel cells
        merged_cell_mask = _merge_cells_based_on_parasite_overlap(parasite_mask, cell_mask, nuclei_mask, overlap_threshold, perimeter_threshold)
        
        #merged_cell_mask = _merge_cells_without_nucleus(merged_cell_mask, nuclei_mask)
    
        # Overwrite the original cell mask file with the merged result
        np.save(cell_path, merged_cell_mask)
        
        stop = time.time()
        duration = (stop - start)
        time_ls.append(duration)
        files_processed += 1
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'adjust_cell_masks')

def process_masks(mask_folder, image_folder, channel, batch_size=50, n_clusters=2, plot=False):
    """
    Processes mask files by measuring object properties, clustering objects, and removing objects 
    not belonging to the largest cluster.
    Args:
        mask_folder (str): Path to the folder containing mask files (.npy format).
        image_folder (str): Path to the folder containing corresponding image files (.npy format).
        channel (int): The channel index to extract from the image files.
        batch_size (int, optional): Number of files to process in each batch. Defaults to 50.
        n_clusters (int, optional): Number of clusters for KMeans clustering. Defaults to 2.
        plot (bool, optional): Whether to plot the clustering results using PCA. Defaults to False.
    Returns:
        None: The function modifies the mask files in place by removing objects not in the largest cluster.
    Notes:
        - The mask files are expected to be in .npy format and contain labeled regions.
        - The image files are expected to be in .npy format and have the same names as the mask files.
        - The function assumes that the mask and image files are sorted in the same order.
        - The clustering is performed on accumulated object properties across all files.
        - The largest cluster is determined based on the number of objects in each cluster.
    """
    
    def read_files_in_batches(folder, batch_size=50):
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        files.sort()  # Sort to ensure matching order
        for i in range(0, len(files), batch_size):
            yield files[i:i + batch_size]

    def measure_morphology_and_intensity(mask, image):
        properties = measure.regionprops(mask, intensity_image=image)
        properties_list = [{'area': p.area, 'mean_intensity': p.mean_intensity, 'perimeter': p.perimeter, 'eccentricity': p.eccentricity} for p in properties]
        return properties_list

    def cluster_objects(properties, n_clusters=2):
        data = np.array([[p['area'], p['mean_intensity'], p['perimeter'], p['eccentricity']] for p in properties])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        return kmeans

    def remove_objects_not_in_largest_cluster(mask, labels, largest_cluster_label):
        cleaned_mask = np.zeros_like(mask)
        for region in measure.regionprops(mask):
            if labels[region.label - 1] == largest_cluster_label:
                cleaned_mask[mask == region.label] = region.label
        return cleaned_mask

    def plot_clusters(properties, labels):
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
    """
    Merge regression results with metadata using gene identifiers.

    Reads regression results and metadata from two CSV files, extracts gene identifiers,
    merges the data on gene names, and saves the merged DataFrame.

    Args:
        results_file (str): Path to the regression results CSV. Must contain a 'feature' column.
        metadata_file (str): Path to the metadata CSV. Must contain a 'Gene ID' column.
        name (str, optional): Suffix for the output file name. Defaults to '_metadata'.

    Returns:
        pandas.DataFrame: Merged DataFrame with regression results and metadata.

    Notes:
        - Extracts gene names from 'feature' (format: '[gene]') and 'Gene ID' (format: 'prefix_gene').
        - Gene extraction failures result in NaNs in the merge but are not dropped.
        - Output CSV is saved next to the input `results_file`, with the suffix appended.
    """
    # Read the CSV files into dataframes
    df_results = pd.read_csv(results_file)
    df_metadata = pd.read_csv(metadata_file)
    
    def extract_and_clean_gene(feature):
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
    """
    Process vision results by extracting metadata from file paths and thresholding predictions.

    Args:
        df (pd.DataFrame): DataFrame with vision results. Must include 'path' and 'pred' columns.
        threshold (float, optional): Threshold for classifying predictions. Defaults to 0.5.

    Returns:
        pd.DataFrame: Modified DataFrame with added columns:
            - 'plateID', 'rowID', 'columnID', 'fieldID', 'object': extracted from 'path'
            - 'prc': combination of 'plateID', 'rowID', and 'columnID'
            - 'cv_predictions': binary classification based on `threshold`
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
    """
    Generate file paths for machine learning result outputs based on model type and feature selection.

    Args:
        src (str): Base directory where the results folder structure will be created.
        model_type (str, optional): Type of ML model used (e.g., 'xgboost', 'random_forest'). Defaults to 'xgboost'.
        channel_of_interest (int | list[int] | str | None, optional): Feature set specification:
            - int: Single channel (e.g., 1)
            - list[int]: Multiple channels (e.g., [1, 2, 3])
            - 'morphology': Use only morphology features
            - None: Use all features
            Defaults to 1.

    Returns:
        tuple[str, ...]: Tuple of 10 file paths:
            - data_path: CSV with predictions or main results
            - permutation_path: CSV with permutation importances
            - feature_importance_path: CSV with model feature importances
            - model_metricks_path: CSV with metrics from trained model
            - permutation_fig_path: PDF plot of permutation importances
            - feature_importance_fig_path: PDF plot of feature importances
            - shap_fig_path: PDF SHAP summary plot
            - plate_heatmap_path: PDF visualization of plate layout
            - settings_csv: CSV with ML settings and parameters
            - ml_features: CSV with extracted feature data used for training

    Raises:
        ValueError: If `channel_of_interest` is not an int, list, None, or 'morphology'.

    Example:
        >>> get_ml_results_paths('/home/user/data', model_type='random_forest', channel_of_interest=[1, 2])
        (
            '/home/user/data/results/random_forest/channels_1_2/results.csv',
            '/home/user/data/results/random_forest/channels_1_2/permutation.csv',
            '/home/user/data/results/random_forest/channels_1_2/feature_importance.csv',
            '/home/user/data/results/random_forest/channels_1_2/random_forest_model.csv',
            '/home/user/data/results/random_forest/channels_1_2/permutation.pdf',
            '/home/user/data/results/random_forest/channels_1_2/feature_importance.pdf',
            '/home/user/data/results/random_forest/channels_1_2/shap.pdf',
            '/home/user/data/results/random_forest/channels_1_2/plate_heatmap.pdf',
            '/home/user/data/results/random_forest/channels_1_2/ml_settings.csv',
            '/home/user/data/results/random_forest/channels_1_2/ml_features.csv'
        )
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
    """
    Perform data augmentation by rotating and reflecting the image.
    
    Args:
    - image (PIL Image or numpy array): The input image.

    Returns:
    - augmented_images (list): A list of augmented images.
    """
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
    """
    Perform data augmentation on the entire dataset by rotating and reflecting the images.

    Args:
    - dataset (list of tuples): The input dataset, each entry is a tuple (image, label, filename).
    - is_grayscale (bool): Flag indicating if the images are grayscale.

    Returns:
    - augmented_dataset (list of tuples): A dataset with augmented (image, label, filename) tuples.
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

    Args:
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
    """
    Corrects and processes mask files located in the specified source directory.

    This function performs the following steps:
    1. Constructs the file path for the cell mask stack within the 'masks' subdirectory of the source.
    2. Converts and relabels the masks using the `convert_and_relabel_masks` function.
    3. Loads and concatenates arrays from the source directory using the `_load_and_concatenate_arrays` function.

    Args:
        src (str): The path to the source directory containing the mask files.

    Returns:
        None
    """

    from .io import _load_and_concatenate_arrays

    cell_path = os.path.join(src,'masks', 'cell_mask_stack')
    convert_and_relabel_masks(cell_path)
    _load_and_concatenate_arrays(src, [0,1,2,3], 1, 0, 2)

def count_reads_in_fastq(fastq_file):
    """
    Counts the number of reads in a FASTQ file.

    A FASTQ file contains sequencing reads, where each read is represented
    by four lines: a header, the sequence, a separator, and the quality scores.
    This function calculates the total number of reads by dividing the total
    number of lines in the file by 4.

    Args:
        fastq_file (str): Path to the FASTQ file, which can be gzip-compressed.

    Returns:
        int: The number of reads in the FASTQ file.
    """
    count = 0
    with gzip.open(fastq_file, "rt") as f:
        for _ in f:
            count += 1
    return count // 4

# Function to determine the CUDA version
def get_cuda_version():
    """
    Retrieves the installed CUDA version by invoking the `nvcc --version` command.

    Returns:
        str: The CUDA version as a string with dots removed (e.g., '110' for version 11.0),
             or None if the `nvcc` command is not found or an error occurs.
    """
    try:
        output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
        if 'release' in output:
            return output.split('release ')[1].split(',')[0].replace('.', '')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def all_elements_match(list1, list2):
    """
    Check if all elements in the first list are present in the second list.

    Args:
        list1 (list): The first list containing elements to check.
        list2 (list): The second list to check against.

    Returns:
        bool: True if all elements in list1 are found in list2, False otherwise.
    """
    # Check if all elements in list1 are in list2
    return all(element in list2 for element in list1)

def prepare_batch_for_segmentation(batch):
    """
    Prepare a batch of images for segmentation by ensuring correct data type and normalization.

    Args:
        batch (np.ndarray): Batch of images with shape (N, H, W, C), where:
            - N: number of images
            - H, W: height and width
            - C: number of channels (e.g., 1 for grayscale, 3 for RGB)

    Returns:
        np.ndarray: Batch with dtype `float32`, normalized to [0, 1] if needed.

    Notes:
        - Converts to `float32` if not already.
        - Each image is divided by its own maximum pixel value if that value > 1.
    """
    # Ensure the batch is of dtype float32
    if batch.dtype != np.float32:
        batch = batch.astype(np.float32)
    
    # Normalize each image in the batch
    for i in range(batch.shape[0]):
        if batch[i].max() > 1:
            batch[i] = batch[i] / batch[i].max()
    
    return batch

def check_index(df, elements=5, split_char='_'):
    """
    Checks the indices of a DataFrame to ensure they can be split into a specified number of parts.

    Args:
        df (pandas.DataFrame): The DataFrame whose indices are to be checked.
        elements (int, optional): The expected number of parts after splitting an index. Defaults to 5.
        split_char (str, optional): The character used to split the index. Defaults to '_'.

    Raises:
        ValueError: If any index cannot be split into the specified number of parts, 
                    a ValueError is raised listing the problematic indices.

    Example:
        >>> import pandas as pd
        >>> data = {'col1': [1, 2, 3]}
        >>> df = pd.DataFrame(data, index=['a_b_c_d_e', 'f_g_h_i', 'j_k_l_m_n'])
        >>> check_index(df)
        ValueError: Found 1 problematic indices that do not split into 5 parts.
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
    """
    Maps a given column value to a specific condition label.

    Args:
        col_value (str): The value to be mapped.
        neg (str, optional): The value representing the 'neg' condition. Defaults to 'c1'.
        pos (str, optional): The value representing the 'pos' condition. Defaults to 'c2'.
        mix (str, optional): The value representing the 'mix' condition. Defaults to 'c3'.

    Returns:
        str: A string representing the mapped condition:
             - 'neg' if col_value matches the neg parameter.
             - 'pos' if col_value matches the pos parameter.
             - 'mix' if col_value matches the mix parameter.
             - 'screen' if col_value does not match any of the above.
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
    
    Args:
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

    Args:
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
    """
    Standardize column names in a metadata DataFrame.

    This function renames commonly used but inconsistent metadata columns to a standardized format.

    Renaming rules:
        - 'plate_name' → 'plateID'
        - 'column_name' or 'col' → 'columnID'
        - 'row_name' → 'rowID'
        - 'grna_name' → 'grna'
        - If 'plate_row' exists, it is split into 'plateID' and 'rowID' using '_' as a delimiter.

    Args:
        df (pd.DataFrame): Input DataFrame with metadata columns.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
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

def control_filelist(folder, mode='columnID', values=['01','02']):
    def control_filelist(folder, mode='columnID', values=['01', '02']):
        """
        Filters a list of files in a given folder based on a specified mode and values.

        Args:
            folder (str): The path to the folder containing the files to be filtered.
            mode (str, optional): The filtering mode. Can be 'columnID' or 'rowID'.
                - 'columnID': Filters files where the second part of the filename (split by '_') 
                  has a substring starting from the second character that matches one of the values.
                - 'rowID': Filters files where the second part of the filename (split by '_') 
                  has a substring starting from the first character that matches one of the values.
                Defaults to 'columnID'.
            values (list of str, optional): A list of string values to filter the files by. Defaults to ['01', '02'].

        Returns:
            list of str: A list of filenames that match the filtering criteria.

        Raises:
            IndexError: If the filenames in the folder do not conform to the expected format 
                        (e.g., missing the second part after splitting by '_').
        """
    files = os.listdir(folder)
    if mode == 'columnID':
        filtered_files = [file for file in files if file.split('_')[1][1:] in values]
    if mode == 'rowID':
        filtered_files = [file for file in files if file.split('_')[1][:1] in values]
    return filtered_files
    
def rename_columns_in_db(db_path):
    """
    Renames specific columns in all user tables of a SQLite database based on a predefined mapping.

    This function connects to the SQLite database at the given path, retrieves all user-defined tables,
    and renames columns in those tables according to the `rename_map` dictionary. If a column with the
    old name exists and the new name does not already exist in the same table, the column is renamed.

    Args:
        db_path (str): The file path to the SQLite database.

    Behavior:
        - Retrieves all user-defined tables in the database.
        - For each table, checks the column names against the `rename_map`.
        - Renames columns as specified in the `rename_map` if conditions are met.
        - Commits the changes to the database.

    Notes:
        - The `rename_map` dictionary defines the mapping of old column names to new column names.
        - If a column with the new name already exists in a table, the old column will not be renamed.
        - The function uses SQLite's `ALTER TABLE ... RENAME COLUMN` syntax, which requires SQLite version 3.25.0 or higher.

    Example:
        rename_columns_in_db("/path/to/database.db")

    Raises:
        sqlite3.OperationalError: If there are issues executing SQL commands, such as unsupported SQLite versions.
    """
    # map old column names → new names
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
        
def group_feature_class(df, feature_groups=['cell', 'cytoplasm', 'nucleus', 'pathogen'], name='compartment'):
    """
    Classify and group features by category, then compute summed importance for each group.

    This function adds a new column to the DataFrame to classify features based on regex matching 
    against the given `feature_groups`. It then computes the total importance per group.

    Args:
        df (pd.DataFrame): Input DataFrame with at least the columns `'feature'` and `'importance'`.
        feature_groups (list of str, optional): List of feature group identifiers (used in regex matching).
            Defaults to ['cell', 'cytoplasm', 'nucleus', 'pathogen'].
        name (str, optional): Name of the new column used to store group classifications. 
            If set to `'channel'`, missing values in that column will be filled with `'morphology'`.
            Defaults to 'compartment'.

    Returns:
        pd.DataFrame: Modified DataFrame including:
            - A new column with group classifications (based on `name` argument).
            - A summary with summed importance for each group and a row for total importance.

    Notes:
        - A feature matching multiple groups will have their labels joined with a hyphen (e.g., "cell-nucleus").
        - Classification is done via regex search on the `'feature'` column.
    """

    # Function to determine compartment based on multiple matches
    def find_feature_class(feature, compartments):
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
    """
    Delete intermediate files and directories specified in the settings dictionary.

    This function removes a predefined set of subdirectories under the path given by
    `settings['src']`. If deletion fails, an error message is printed.

    Args:
        settings (dict): Must include the key `'src'`, the base directory containing
                         intermediate files and subdirectories.

    Behavior:
        - Verifies that `'src'` and its `orig/` subdirectory exist.
        - Deletes the following under `src` if they exist:
            * 'stack'
            * 'masks'
            * directories '1' through '10'
        - Compares lengths of `merged_stack` and `path_stack` to decide on deletion.
        - Prints success or error messages for each directory.

    Notes:
        - If `src` or `orig/` is missing, the function exits early.
        - Any deletion exceptions are caught and reported; no exceptions are raised.

    Example:
        >>> settings = {'src': '/path/to/source'}
        >>> delete_intermedeate_files(settings)
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

    Args:
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
    
    Args:
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
    
    Args:
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
    """
    Corrects and standardizes the metadata column names in a DataFrame.
    This function checks for specific column names in the input DataFrame and 
    renames or maps them to standardized names for consistency. The following 
    transformations are applied:
    - If 'object_name' exists, it is mapped to 'objectID'.
    - If 'field_name' exists, it is mapped to 'fieldID'.
    - If 'plate' or 'plate_name' exists, they are mapped to 'plateID'.
    - If 'row' or 'row_name' exists, they are renamed to 'rowID'.
    - If 'col', 'column', or 'column_name' exists, they are renamed to 'columnID'.
    - If 'field' or 'field_name' exists, they are renamed to 'fieldID'.
    Args:
        df (pandas.DataFrame): The input DataFrame containing metadata columns.
    Returns:
        pandas.DataFrame: The DataFrame with standardized metadata column names.
    """
    
    #if 'object' in df.columns:
    #    df['objectID'] = df['object']
        
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

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): Column name to group by.
        value_col (str): Column containing values to check for outliers.
        method (str): 'iqr' or 'zscore'.
        threshold (float): Threshold multiplier for IQR (default 1.5) or z-score.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    def iqr_filter(subdf):
        q1 = subdf[value_col].quantile(0.25)
        q3 = subdf[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return subdf[(subdf[value_col] >= lower) & (subdf[value_col] <= upper)]

    def zscore_filter(subdf):
        mean = subdf[value_col].mean()
        std = subdf[value_col].std()
        return subdf[(subdf[value_col] - mean).abs() <= threshold * std]

    if method == 'iqr':
        return df.groupby(group_col, group_keys=False).apply(iqr_filter)
    elif method == 'zscore':
        return df.groupby(group_col, group_keys=False).apply(zscore_filter)
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
