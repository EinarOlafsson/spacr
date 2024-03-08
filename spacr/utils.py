import os, re, sqlite3, gc, torch, torchvision, time, random, string, shutil, cv2, tarfile, glob

import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops_table, regionprops
import skimage.measure as measure
from collections import defaultdict
from PIL import Image
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from collections import OrderedDict
from functools import reduce
from IPython.display import display, clear_output
from multiprocessing import Pool, cpu_count
from skimage.transform import resize as resizescikit
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Subset
from torch.autograd import grad
from torchvision import models
from skimage.segmentation import clear_border
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.stats import fisher_exact
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.exposure import rescale_intensity
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

from .logger import log_function_call

#from .io import _read_and_join_tables, _save_figure
#from .timelapse import _btrack_track_cells, _trackpy_track_cells
#from .plot import _plot_images_on_grid, plot_masks, _plot_histograms_and_stats, plot_resize, _plot_plates, _reg_v_plot, plot_masks
#from .core import identify_masks

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

def _extract_filename_metadata(filenames, src, images_by_key, regular_expression, metadata_type='cellvoyager', pick_slice=False, skip_mode='01'):
    for filename in filenames:
        match = regular_expression.match(filename)
        if match:
            try:
                try:
                    plate = match.group('plateID')
                except:
                    plate = os.path.basename(src)

                well = match.group('wellID')
                field = match.group('fieldID')
                channel = match.group('chanID')
                mode = None

                if well[0].isdigit():
                    well = str(_safe_int_convert(well))
                if field[0].isdigit():
                    field = str(_safe_int_convert(field))
                if channel[0].isdigit():
                    channel = str(_safe_int_convert(channel))

                if metadata_type =='cq1':
                    orig_wellID = wellID
                    wellID = _convert_cq1_well_id(wellID)
                    clear_output(wait=True)
                    print(f'Converted Well ID: {orig_wellID} to {wellID}', end='\r', flush=True)

                if pick_slice:
                    try:
                        mode = match.group('AID')
                    except IndexError:
                        sliceid = '00'

                    if mode == skip_mode:
                        continue      
                        
                key = (plate, well, field, channel, mode)
                with Image.open(os.path.join(src, filename)) as img:
                    images_by_key[key].append(np.array(img))
            except IndexError:
                print(f"Could not extract information from filename {filename} using provided regex")
        else:
            print(f"Filename {filename} did not match provided regex")
            continue
        
    return images_by_key

def mask_object_count(mask):
    """
    Counts the number of objects in a given mask.

    Parameters:
    - mask: numpy.ndarray
        The mask containing object labels.

    Returns:
    - int
        The number of objects in the mask.
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
            df['prcfo'] = df['plate'].astype(str) + '_' + df['row'].astype(str) + '_' + df['col'].astype(str) + '_' + df['field'].astype(str) + '_o' + df['object_label'].astype(int).astype(str)
        except Exception as e:
            print('Merging on cell failed, trying with cell_id')
        try:
            df['prcfo'] = df['plate'].astype(str) + '_' + df['row'].astype(str) + '_' + df['col'].astype(str) + '_' + df['field'].astype(str) + '_o' + df['cell_id'].astype(int).astype(str)
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
    df = _annotate_conditions(df, cells, cell_loc, pathogens, pathogen_loc, treatments,treatment_loc)
    
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
        type_ = 'row' if locs[0][0][0] == 'r' else 'col'
        return value_dict.get(row[type_], None)
    return values[0] if values else None

def _annotate_conditions(df, cells=['HeLa'], cell_loc=None, pathogens=['rh'], pathogen_loc=None, treatments=['cm'], treatment_loc=None):
    """
    Annotates conditions in the given DataFrame based on the provided parameters.

    Args:
        df (pandas.DataFrame): The DataFrame to annotate.
        cells (list, optional): The list of host cell types. Defaults to ['HeLa'].
        cell_loc (list, optional): The list of location identifiers for host cells. Defaults to None.
        pathogens (list, optional): The list of pathogens. Defaults to ['rh'].
        pathogen_loc (list, optional): The list of location identifiers for pathogens. Defaults to None.
        treatments (list, optional): The list of treatments. Defaults to ['cm'].
        treatment_loc (list, optional): The list of location identifiers for treatments. Defaults to None.

    Returns:
        pandas.DataFrame: The annotated DataFrame with the 'host_cells', 'pathogen', 'treatment', and 'condition' columns.
    """


    # Apply mappings or defaults
    df['host_cells'] = [cells[0]] * len(df) if cell_loc is None else df.apply(_map_values, args=(cells, cell_loc), axis=1)
    df['pathogen'] = [pathogens[0]] * len(df) if pathogen_loc is None else df.apply(_map_values, args=(pathogens, pathogen_loc), axis=1)
    df['treatment'] = [treatments[0]] * len(df) if treatment_loc is None else df.apply(_map_values, args=(treatments, treatment_loc), axis=1)

    # Construct condition column
    df['condition'] = df.apply(lambda row: '_'.join(filter(None, [row.get('pathogen'), row.get('treatment')])), axis=1)
    df['condition'] = df['condition'].apply(lambda x: x if x else 'none')
    return df
    
def normalize_to_dtype(array, q1=2,q2=98, percentiles=None):
    """
    Normalize the input array to a specified data type.

    Parameters:
    - array: numpy array
        The input array to be normalized.
    - q1: int, optional
        The lower percentile value for normalization. Default is 2.
    - q2: int, optional
        The upper percentile value for normalization. Default is 98.
    - percentiles: list of tuples, optional
        A list of tuples containing the percentile values for each image in the array.
        If provided, the percentiles for each image will be used instead of q1 and q2.

    Returns:
    - new_stack: numpy array
        The normalized array with the same shape as the input array.
    """
    nimg = array.shape[2]
    new_stack = np.empty_like(array)
    for i,v in enumerate(range(nimg)):
        img = np.squeeze(array[:, :, v])
        non_zero_img = img[img > 0]
        if non_zero_img.size > 0: # check if there are non-zero values
            img_min = np.percentile(non_zero_img, q1)  # change percentile from 0.02 to 2
            img_max = np.percentile(non_zero_img, q2)  # change percentile from 0.98 to 98
            img = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')
        else:  # if there are no non-zero values, just use the image as it is
            if percentiles==None:
                img_min, img_max = img.min(), img.max()
            else:
                img_min, img_max = percentiles[i]
            img = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')
        img = np.expand_dims(img, axis=2)
        new_stack[:, :, v] = img[:, :, 0]
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

    Parameters:
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
                merged_df[['plate', 'row', 'col', 'field', 'timeid', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(_map_wells(x, timelapse)))
            else:
                merged_df[['plate', 'row', 'col', 'field', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(_map_wells(x, timelapse)))
            cols = merged_df.columns.tolist()  # get the list of all columns
            if table_type == 'cell' or table_type == 'cytoplasm':
                column_list = ['object_label', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
            elif table_type == 'nucleus' or table_type == 'pathogen':
                column_list = ['object_label', 'cell_id', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
            else:
                raise ValueError(f"Invalid table_type: {table_type}")
            # Check if all columns in column_list are in cols
            missing_columns = [col for col in column_list if col not in cols]
            if len(missing_columns) == 1 and missing_columns[0] == 'cell_id':
                missing_columns = False
                column_list = ['object_label', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
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
        return plate, row, column, field, timeid, prcfo, object_id,
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
    
def _get_percentiles(array, q1=2, q2=98):
    """
    Calculate the percentiles of each image in the given array.

    Parameters:
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
            img_min = np.percentile(non_zero_img, q1)  # change percentile from 0.02 to 2
            img_max = np.percentile(non_zero_img, q2)  # change percentile from 0.98 to 98
            percentiles.append([img_min, img_max])
        else:  # if there are no non-zero values, just use the image as it is
            img_min, img_max = img.min(), img.max()
            percentiles.append([img_min, img_max])
    return percentiles

def _crop_center(img, cell_mask, new_width, new_height, normalize=(2,98)):
    """
    Crop the image around the center of the cell mask.

    Parameters:
    - img: numpy.ndarray
        The input image.
    - cell_mask: numpy.ndarray
        The binary mask of the cell.
    - new_width: int
        The desired width of the cropped image.
    - new_height: int
        The desired height of the cropped image.
    - normalize: tuple, optional
        The normalization range for the image pixel values. Default is (2, 98).

    Returns:
    - img: numpy.ndarray
        The cropped image.
    """
    # Convert all non-zero values in mask to 1
    cell_mask[cell_mask != 0] = 1
    mask_3d = np.repeat(cell_mask[:, :, np.newaxis], img.shape[2], axis=2).astype(img.dtype) # Create 3D mask
    img = np.multiply(img, mask_3d).astype(img.dtype) # Multiply image with mask to set pixel values outside of the mask to 0
    #centroid = np.round(ndi.measurements.center_of_mass(cell_mask)).astype(int) # Compute centroid of the mask
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
        if mag == 20:
            scale = 6
        if mag == 40:
            scale = 4.5
        if mag == 60:
            scale = 3
    elif obj == 'nucleus':
        if mag == 20:
            scale = 3
        if mag == 40:
            scale = 2
        if mag == 60:
            scale = 1.5
    elif obj == 'pathogen':
        if mag == 20:
            scale = 1.5
        if mag == 40:
            scale = 1
        if mag == 60:
            scale = 1.25
    elif obj == 'pathogen_nucleus':
        if mag == 20:
            scale = 0.25
        if mag == 40:
            scale = 0.2
        if mag == 60:
            scale = 0.2
    else:
        raise ValueError("Invalid object type")
    diamiter = mag*scale
    return diamiter

def _get_object_settings(object_type, settings):
    
    object_settings = {}
    object_settings['refine_masks'] = False
    object_settings['filter_size'] = False
    object_settings['filter_dimm'] = False
    print(object_type)
    object_settings['diameter'] = _get_diam(settings['magnification'], obj=object_type)
    object_settings['remove_border_objects'] = False
    object_settings['minimum_size'] = (object_settings['diameter']**2)/10
    object_settings['maximum_size'] = object_settings['minimum_size']*50
    object_settings['merge'] = False
    object_settings['net_avg'] = True
    object_settings['resample'] = True
    object_settings['model_name'] = 'cyto'
    
    if object_type == 'cell':
        if settings['nucleus_channel'] is None:
            object_settings['model_name'] = 'cyto'
        else:
            object_settings['model_name'] = 'cyto2'
        
    elif object_type == 'nucleus':
        object_settings['model_name'] = 'nuclei'

    elif object_type == 'pathogen':
        object_settings['model_name'] = 'cyto3'

    elif object_type == 'pathogen_nucleus':
        object_settings['filter_size'] = True
        object_settings['model_name'] = 'cyto'
        
    else:
        print(f'Object type: {object_type} not supported. Supported object types are : cell, nucleus and pathogen')
        print(f'using settings: {object_settings}')
        
    return object_settings
    
def _pivot_counts_table(db_path):

    def _read_table_to_dataframe(db_path, table_name='object_counts'):
        """
        Read a table from an SQLite database into a pandas DataFrame.

        Parameters:
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
    
def _get_cellpose_channels_v1(mask_channels, nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim):
    cellpose_channels = {}
    if nucleus_chann_dim in mask_channels:
        cellpose_channels['nucleus'] = [0, mask_channels.index(nucleus_chann_dim)]
    if pathogen_chann_dim in mask_channels:
        cellpose_channels['pathogen'] = [0, mask_channels.index(pathogen_chann_dim)]
    if cell_chann_dim in mask_channels:
        cellpose_channels['cell'] = [0, mask_channels.index(cell_chann_dim)]
    return cellpose_channels

def _get_cellpose_channels_v1(cell_channel, nucleus_channel, pathogen_channel):
    # Initialize a dictionary to hold the new indices for the specified channels
    cellpose_channels = {}

    # Initialize a list to keep track of the channels in their new order
    new_channel_order = []

    # Add each channel to the new order list if it is not None
    if cell_channel is not None:
        new_channel_order.append(('cell', cell_channel))
    if nucleus_channel is not None:
        new_channel_order.append(('nucleus', nucleus_channel))
    if pathogen_channel is not None:
        new_channel_order.append(('pathogen', pathogen_channel))

    # Sort the list based on the original channel indices to maintain the original order
    new_channel_order.sort(key=lambda x: x[1])
    print(new_channel_order)
    # Assign new indices based on the sorted order
    for new_index, (channel_name, _) in enumerate(new_channel_order):
        cellpose_channels[channel_name] = [new_index, 0]
        
    if cell_channel is not None and nucleus_channel is not None:
        cellpose_channels['cell'][1] = cellpose_channels['nucleus'][0]
        
    return cellpose_channels

def _get_cellpose_channels(nucleus_channel, pathogen_channel, cell_channel):
    cellpose_channels = {}
    if not nucleus_channel is None:
        cellpose_channels['nucleus'] = [0,0]
        
    if not pathogen_channel is None:
        if not nucleus_channel is None:
            cellpose_channels['pathogen'] = [0,1]
        else:
            cellpose_channels['pathogen'] = [0,0]
        
    if not cell_channel is None:
        if not nucleus_channel is None:
            if not pathogen_channel is None:
                cellpose_channels['cell'] = [0,2]
            else:
                cellpose_channels['cell'] = [0,1]
        elif not pathogen_channel is None:
            cellpose_channels['cell'] = [0,1]
        else:
            cellpose_channels['cell'] = [0,0]
    return cellpose_channels
    
def annotate_conditions(df, cells=['HeLa'], cell_loc=None, pathogens=['rh'], pathogen_loc=None, treatments=['cm'], treatment_loc=None, types = ['col','col','col']):
    """
    Annotates conditions in a DataFrame based on specified criteria.

    Args:
        df (pandas.DataFrame): The DataFrame to annotate.
        cells (list, optional): List of host cell types. Defaults to ['HeLa'].
        cell_loc (list, optional): List of corresponding values for each host cell type. Defaults to None.
        pathogens (list, optional): List of pathogens. Defaults to ['rh'].
        pathogen_loc (list, optional): List of corresponding values for each pathogen. Defaults to None.
        treatments (list, optional): List of treatments. Defaults to ['cm'].
        treatment_loc (list, optional): List of corresponding values for each treatment. Defaults to None.
        types (list, optional): List of column types for host cells, pathogens, and treatments. Defaults to ['col','col','col'].

    Returns:
        pandas.DataFrame: The annotated DataFrame.
    """

    # Function to apply to each row
    def _map_values(row, dict_, type_='col'):
        """
        Maps the values in a row to corresponding keys in a dictionary.

        Args:
            row (dict): The row containing the values to be mapped.
            dict_ (dict): The dictionary containing the mapping values.
            type_ (str, optional): The type of mapping to perform. Defaults to 'col'.

        Returns:
            str: The mapped value if found, otherwise None.
        """
        for values, cols in dict_.items():
            if row[type_] in cols:
                return values
        return None

    if cell_loc is None:
        df['host_cells'] = cells[0]
    else:
        cells_dict = dict(zip(cells, cell_loc))
        df['host_cells'] = df.apply(lambda row: _map_values(row, cells_dict, type_=types[0]), axis=1)
    if pathogen_loc is None:
        if pathogens != None:
            df['pathogen'] = 'none'
    else:
        pathogens_dict = dict(zip(pathogens, pathogen_loc))
        df['pathogen'] = df.apply(lambda row: _map_values(row, pathogens_dict, type_=types[1]), axis=1)
    if treatment_loc is None:
        df['treatment'] = 'cm'
    else:
        treatments_dict = dict(zip(treatments, treatment_loc))
        df['treatment'] = df.apply(lambda row: _map_values(row, treatments_dict, type_=types[2]), axis=1)
    if pathogens != None:
        df['condition'] = df['pathogen']+'_'+df['treatment']
    else:
        df['condition'] = df['treatment']
    return df
    

    
def _split_data(df, group_by, object_type):
    """
    Splits the input dataframe into numeric and non-numeric parts, groups them by the specified column,
    and returns the grouped dataframes.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    group_by (str): The column name to group the dataframes by.
    object_type (str): The column name to concatenate with 'prcf' to create a new column 'prcfo'.

    Returns:
    grouped_numeric (pandas.DataFrame): The grouped dataframe containing numeric columns.
    grouped_non_numeric (pandas.DataFrame): The grouped dataframe containing non-numeric columns.
    """
    df['prcfo'] = df['prcf'] + '_' + df[object_type]
    df = df.set_index(group_by, inplace=False)

    df_numeric = df.select_dtypes(include=np.number)
    df_non_numeric = df.select_dtypes(exclude=np.number)

    grouped_numeric = df_numeric.groupby(df_numeric.index).mean()
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

    for chan in channels:
        df[f'nucleus_coordinates_{chan}'] = df[[f'nucleus_channel_{chan}_centroid_weighted_local-0', f'nucleus_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'pathogen_coordinates_{chan}'] = df[[f'pathogen_channel_{chan}_centroid_weighted_local-0', f'pathogen_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'cell_coordinates_{chan}'] = df[[f'cell_channel_{chan}_centroid_weighted_local-0', f'cell_channel_{chan}_centroid_weighted_local-1']].values.tolist()
        df[f'cytoplasm_coordinates_{chan}'] = df[[f'cytoplasm_channel_{chan}_centroid_weighted_local-0', f'cytoplasm_channel_{chan}_centroid_weighted_local-1']].values.tolist()

        df[f'pathogen_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'pathogen_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                      (row[f'pathogen_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
        df[f'nucleus_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'nucleus_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                      (row[f'nucleus_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
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
    df_grouped = df.groupby(['plate', 'row', 'col']).agg({**{col: np.mean for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}})
    return df_grouped
    



###################################################
#  Classify
###################################################

class Cache:
    """
    A class representing a cache with a maximum size.

    Attributes:
        max_size (int): The maximum size of the cache.
        cache (OrderedDict): The cache data structure.
    """

    def _init__(self, max_size):
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

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.

    Args:
        d_k (int): The dimension of the key and query vectors.

    Attributes:
        d_k (int): The dimension of the key and query vectors.

    Methods:
        forward(Q, K, V): Performs the forward pass of the attention mechanism.

    """

    def _init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
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

class SelfAttention(nn.Module):
    """
    Self-Attention module that applies scaled dot-product attention mechanism.
    
    Args:
        in_channels (int): Number of input channels.
        d_k (int): Dimensionality of the key and query vectors.
    """

    def _init__(self, in_channels, d_k):
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
            torch.Tensor: Output tensor of shape (batch_size, d_k).
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output = self.attention(Q, K, V)
        return output

class ScaledDotProductAttention(nn.Module):
    def _init__(self, d_k):
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
    def _init__(self, in_channels, d_k):
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
    def _init__(self, in_channels):
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
    def _init__(self, kernel_size=7):
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
    Multi-scale block with attention module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        dilated_conv1 (nn.Conv2d): Dilated convolution layer.
        spatial_attention (nn.Conv2d): Spatial attention layer.

    Methods:
        custom_forward: Custom forward method for the module.
        forward: Forward method for the module.
    """

    def _init__(self, in_channels, out_channels):
        super(MultiScaleBlockWithAttention, self).__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.spatial_attention = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def custom_forward(self, x):
        """
        Custom forward method for the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = F.relu(self.dilated_conv1(x), inplace=True)
        x = self.spatial_attention(x1)
        return x

    def forward(self, x):
        """
        Forward method for the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return checkpoint(self.custom_forward, x)

# Final Classifier
class CustomCellClassifier(nn.Module):
    def _init__(self, num_classes, pathogen_channel, use_attention, use_checkpoint, dropout_rate):
        super(CustomCellClassifier, self).__init__()
        self.early_fusion = EarlyFusion(in_channels=3)
        
        self.multi_scale_block_1 = MultiScaleBlockWithAttention(in_channels=64, out_channels=64)
        
        self.fc1 = nn.Linear(64, num_classes)
        self.use_checkpoint = use_checkpoint
        # Explicitly require gradients for all parameters
        for param in self.parameters():
            param.requires_grad = True
        
    def custom_forward(self, x):
        x.requires_grad = True 
        x = self.early_fusion(x)
        x = self.multi_scale_block_1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        return x

    def forward(self, x):
        if self.use_checkpoint:
            x.requires_grad = True 
            return checkpoint(self.custom_forward, x)
        else:
            return self.custom_forward(x)

#CNN and Transformer class, pick any Torch model.
class TorchModel(nn.Module):
    def _init__(self, model_name='resnet50', pretrained=True, dropout_rate=None, use_checkpoint=False):
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
        """Apply dropout rate to all dropout layers in the model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    def init_base_model(self, pretrained):
        """Initialize the base model from torchvision.models."""
        model_func = models.__dict__.get(self.model_name, None)
        if not model_func:
            raise ValueError(f"Model {self.model_name} is not recognized.")
        weight_choice = self.get_weight_choice()
        if weight_choice is not None:
            return model_func(weights=weight_choice)
        else:
            return model_func(pretrained=pretrained)

    def get_weight_choice(self):
        """Get weight choice if it exists for the model."""
        weight_enum = None
        for attr_name in dir(models):
            if attr_name.lower() == f"{self.model_name}_weights".lower():
                weight_enum = getattr(models, attr_name)
                break
        return weight_enum.DEFAULT if weight_enum else None

    def get_num_ftrs(self):
        """Determine the number of features output by the base model."""
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
        """Initialize the SPACR classifier."""
        self.use_dropout = dropout_rate is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        self.spacr_classifier = nn.Linear(self.num_ftrs, 1)

    def forward(self, x):
        """Define the forward pass of the model."""
        if self.use_checkpoint:
            x = checkpoint(self.base_model, x)
        else:
            x = self.base_model(x)
        if self.use_dropout:
            x = self.dropout(x)
        logits = self.spacr_classifier(x).flatten()
        return logits

class FocalLossWithLogits(nn.Module):
    def _init__(self, alpha=1, gamma=2):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()
    
class ResNet(nn.Module):
    def _init__(self, resnet_type='resnet50', dropout_rate=None, use_checkpoint=False, init_weights='imagenet'):
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

def classification_metrics(all_labels, prediction_pos_probs, loader_name, loss, epoch):
    """
    Calculate classification metrics for binary classification.

    Parameters:
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
    data_df = pd.DataFrame(data_dict, index=[str(epoch)+'_'+loader_name]) 
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

def choose_model(model_type, device, init_weights=True, dropout_rate=0, use_checkpoint=False, channels=3, height=224, width=224, chan_dict=None, num_classes=2):
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

    print(f'\rModel parameters: Architecture: {model_type} init_weights: {init_weights} dropout_rate: {dropout_rate} use_checkpoint: {use_checkpoint}', end='\r', flush=True)
    
    if model_type == 'custom':
        
        base_model = CustomCellClassifier(num_classes, pathogen_channel=pathogen_channel, use_attention=True, use_checkpoint=use_checkpoint, dropout_rate=dropout_rate)
        #base_model = CustomCellClassifier(num_classes=2, pathogen_channel=pathogen_channel, nucleus_channel=nucleus_channel, protein_channel=protein_channel, dropout_rate=dropout_rate, use_checkpoint=use_checkpoint)
    elif model_type in torch_model_types:
        base_model = TorchModel(model_name=model_type, pretrained=init_weights, dropout_rate=dropout_rate)
    else:
        print(f'Compatible model_types: {model_types}')
        raise ValueError(f"Invalid model_type: {model_type}")

    print(base_model)
    
    return base_model

def calculate_loss(output, target, loss_type='binary_cross_entropy_with_logits'):
    if loss_type == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(output, target)
    elif loss_type == 'focal_loss':
        focal_loss_fn = FocalLossWithLogits(alpha=1, gamma=2)
        loss = focal_loss_fn(output, target)
    return loss

def pick_best_model(src):
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
    objects = df.index.tolist()
    filtered_df = png_df[png_df['png_path'].str.contains(image_type) & png_df['prcfo'].isin(objects)]
    return filtered_df

def save_file_lists(dst, data_set, ls):
    df = pd.DataFrame(ls, columns=[data_set])  
    df.to_csv(f'{dst}/{data_set}.csv', index=False)
    return

def augment_single_image(args):
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
    if not os.path.exists(dst):
        os.makedirs(dst)

    args_list = [(img_path, dst) for img_path in file_paths]

    with Pool(cpu_count()) as pool:
        pool.map(augment_single_image, args_list)

def augment_classes(dst, nc, pc, generate=True,move=True):
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
    df = pd.read_csv(csv_loc)
    df['filename'] = df['path'].apply(lambda x: x.split('/')[-1])
    df[['plate', 'well', 'field', 'object']] = df['filename'].str.split('_', expand=True)
    df['object'] = df['object'].str.replace('.png', '')
    
    def assign_condition(row):
        plate = int(row['plate'])
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

def init_globals(counter_, lock_):
    global counter, lock
    counter = counter_
    lock = lock_

def add_images_to_tar(args):
    global counter, lock, total_images
    paths_chunk, tar_path = args
    with tarfile.open(tar_path, 'w') as tar:
        for img_path in paths_chunk:
            arcname = os.path.basename(img_path)
            try:
                tar.add(img_path, arcname=arcname)
                with lock:
                    counter.value += 1
                    print(f"\rProcessed: {counter.value}/{total_images}", end='', flush=True)
            except FileNotFoundError:
                print(f"File not found: {img_path}")
    return tar_path

def generate_fraction_map(df, gene_column, min_frequency=0.0):
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

def generate_dependent_variable(df, dv_loc, pc_min=0.95, nc_max=0.05, agg_type='mean'):
    
    from .plot import _plot_histograms_and_stats, _plot_plates
    
    def qstring_to_float(qstr):
        number = int(qstr[1:])  # Remove the "q" and convert the rest to an integer
        return number / 100.0
    
    print("Unique values in plate:", df['plate'].unique())
    dv_cell_loc = f'{dv_loc}/dv_cell.csv'
    dv_well_loc = f'{dv_loc}/dv_well.csv'
    
    df['pred'] = 1-df['pred'] #if you swiched pc and nc
    df = df[(df['pred'] <= nc_max) | (df['pred'] >= pc_min)]
    
    if 'prc' not in df.columns:
        df['prc'] = df['plate'] + '_' + df['row'] + '_' + df['col']
    
    if agg_type.startswith('q'):
        val = qstring_to_float(agg_type)
        agg_type = lambda x: x.quantile(val)
    
    # Aggregating for mean prediction and total count
    df_grouped = df.groupby('prc').agg(
        pred=('pred', agg_type),
        recruitment=('recruitment', agg_type),
        count_prc=('prc', 'size'),
        #count_above_95=('pred', lambda x: (x > 0.95).sum()),
        mean_pathogen_area=('pathogen_area', 'mean')
    )
    
    df_cell = df[['prc', 'pred', 'pathogen_area', 'recruitment']]
    
    df_cell.to_csv(dv_cell_loc, index=True, header=True, mode='w')
    df_grouped.to_csv(dv_well_loc, index=True, header=True, mode='w')  # Changed from loc to dv_loc
    display(df)
    _plot_histograms_and_stats(df)
    df_grouped = df_grouped.sort_values(by='count_prc', ascending=True)
    display(df_grouped)
    print('pred')
    _plot_plates(df=df_cell, variable='pred', grouping='mean', min_max='allq', cmap='viridis')
    print('recruitment')
    _plot_plates(df=df_cell, variable='recruitment', grouping='mean', min_max='allq', cmap='viridis')
    
    return df_grouped

def lasso_reg(merged_df, alpha_value=0.01, reg_type='lasso'):
    # Separate predictors and response
    X = merged_df[['gene', 'grna', 'plate', 'row', 'column']]
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

#def normalize_to_dtype(array, q1=2, q2=98, percentiles=None):
#    if len(array.shape) == 2:
#        array = np.expand_dims(array, axis=-1)
#    num_channels = array.shape[-1]
#    new_stack = np.empty_like(array)
#    for channel in range(num_channels):
#        img = array[..., channel]
#        non_zero_img = img[img > 0]
#        if non_zero_img.size > 0:
#            img_min = np.percentile(non_zero_img, q1)
#            img_max = np.percentile(non_zero_img, q2)
#        else:
#            img_min, img_max = (percentiles[channel] if percentiles and channel < len(percentiles)
#                                else (img.min(), img.max()))
#        new_stack[..., channel] = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')
#    if new_stack.shape[-1] == 1:
#        new_stack = np.squeeze(new_stack, axis=-1)
#    return new_stack

def get_files_from_dir(dir_path, file_extension="*"):
    return glob(os.path.join(dir_path, file_extension))
    
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
def apply_mask(image, output_value=0):
    h, w = image.shape[:2]  # Assuming image is grayscale or RGB
    mask = create_circular_mask(h, w)
    
    # If the image has more than one channel, repeat the mask for each channel
    if len(image.shape) > 2:
        mask = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)
    
    # Apply the mask - set pixels outside of the mask to output_value
    masked_image = np.where(mask, image, output_value)
    return masked_image
    
def invert_image(image):
    # The maximum value depends on the image dtype (e.g., 255 for uint8)
    max_value = np.iinfo(image.dtype).max
    inverted_image = max_value - image
    return inverted_image  

def resize_images_and_labels(images, labels, target_height, target_width, show_example=True):
    
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
    mask1, mask2 = pad_to_same_shape(mask1, mask2)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0
    
def match_masks(true_masks, pred_masks, iou_threshold):
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
    TP = len(matches)
    FP = num_pred_masks - TP
    FN = num_true_masks - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall

def pad_to_same_shape(mask1, mask2):
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
    true_mask_labels = label(true_masks)
    pred_mask_labels = label(pred_masks)
    true_mask_regions = [region.image for region in regionprops(true_mask_labels)]
    pred_mask_regions = [region.image for region in regionprops(pred_mask_labels)]
    return compute_ap_over_iou_thresholds(true_mask_regions, pred_mask_regions, iou_thresholds)

def jaccard_index(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def dice_coefficient(mask1, mask2):
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
    binary_mask = (mask > 0).astype(np.uint8)
    struct_elem = np.ones((dilation_radius*2+1, dilation_radius*2+1))
    dilated = binary_dilation(binary_mask, footprint=struct_elem)
    eroded = binary_erosion(binary_mask, footprint=struct_elem)
    boundary = dilated ^ eroded
    return boundary

def boundary_f1_score(mask_true, mask_pred, dilation_radius=1):
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
    
def _exclude_objects(cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask, include_uninfected=True):
    """
    Exclude objects from the masks based on certain criteria.

    Args:
        cell_mask (ndarray): Mask representing cells.
        nucleus_mask (ndarray): Mask representing nucleus.
        pathogen_mask (ndarray): Mask representing pathogens.
        cytoplasm_mask (ndarray): Mask representing cytoplasm.
        include_uninfected (bool, optional): Whether to include uninfected cells. Defaults to True.

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
        if include_uninfected:
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

def _filter_cp_masks(masks, flows, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize):
    """
    Filter the masks based on various criteria such as size, border objects, merging, and intensity.

    Args:
        masks (list): List of masks.
        flows (list): List of flows.
        refine_masks (bool): Flag indicating whether to refine masks.
        filter_size (bool): Flag indicating whether to filter based on size.
        minimum_size (int): Minimum size of objects to keep.
        maximum_size (int): Maximum size of objects to keep.
        remove_border_objects (bool): Flag indicating whether to remove border objects.
        merge (bool): Flag indicating whether to merge adjacent objects.
        filter_dimm (bool): Flag indicating whether to filter based on intensity.
        batch (ndarray): Batch of images.
        moving_avg_q1 (float): Moving average of the first quartile of object intensities.
        moving_avg_q3 (float): Moving average of the third quartile of object intensities.
        moving_count (int): Count of moving averages.
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

        if filter_size:
            props = measure.regionprops_table(mask, properties=['label', 'area'])  # Measure properties of labeled image regions.
            valid_labels = props['label'][np.logical_and(props['area'] > minimum_size, props['area'] < maximum_size)]  # Select labels of valid size.
            masks[idx] = np.isin(mask, valid_labels) * mask  # Keep only valid objects.
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after size filtration >{minimum_size} and <{maximum_size} : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        if remove_border_objects:
            mask = clear_border(mask)
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after removing border objects, : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        if merge:
            mask = merge_touching_objects(mask, threshold=0.25)
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Number of objects after merging adjacent objects, : {num_objects}')
                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        if filter_dimm:
            unique_labels = np.unique(mask)
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                continue
            object_intensities = [np.mean(batch[idx, :, :, 1][mask == label]) for label in unique_labels if label != 0]
            object_q1s = [np.percentile(intensities, 25) for intensities in object_intensities if intensities.size > 0]
            object_q3s = [np.percentile(intensities, 75) for intensities in object_intensities if intensities.size > 0]
            if object_q1s:
                object_q1_mean = np.mean(object_q1s)
                object_q3_mean = np.mean(object_q3s)
                moving_avg_q1 = (moving_avg_q1 * moving_count + object_q1_mean) / (moving_count + 1)
                moving_avg_q3 = (moving_avg_q3 * moving_count + object_q3_mean) / (moving_count + 1)
                moving_count += 1
            mask = remove_intensity_objects(batch[idx, :, :, 1], mask, intensity_threshold=moving_avg_q1, mode='low')
            mask = remove_intensity_objects(batch[idx, :, :, 1], mask, intensity_threshold=moving_avg_q3, mode='high')
            if plot and idx == 0:
                num_objects = mask_object_count(mask)
                print(f'Objects after intensity filtration > {moving_avg_q1} and <{moving_avg_q3}: {num_objects}')
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

###################################################
#  Classify
###################################################