import seaborn as sns
import os, random, sqlite3, re, shap, string
import pandas as pd
import numpy as np

from skimage.measure import regionprops, label
from skimage.transform import resize as sk_resize, rotate
from skimage.exposure import rescale_intensity

import cellpose
from cellpose import models as cp_models
from cellpose import train as train_cp
from cellpose import models as cp_models
from cellpose import io as cp_io
from cellpose import train as train_cp
from cellpose.metrics import aggregated_jaccard_index

from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from math import pi
from scipy.stats import chi2_contingency, pearsonr
from scipy.spatial.distance import cosine

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
from natsort import natsorted

import torch
from torch.utils.data import Dataset
from spacr.settings import get_train_cellpose_default_settings
from spacr.utils import save_settings, invert_image



class CellposeLazyDataset(Dataset):
    def __init__(self, image_files, label_files, settings, randomize=True, augment=False):
        combined = list(zip(image_files, label_files))
        if randomize:
            random.shuffle(combined)
        self.image_files, self.label_files = zip(*combined)
        self.normalize = settings['normalize']
        self.percentiles = settings.get('percentiles', [2, 99])
        self.target_size = settings['target_size']
        self.augment = augment

    def __len__(self):
        return len(self.image_files) * (8 if self.augment else 1)

    def apply_augmentation(self, image, label, aug_idx):
        if aug_idx == 1:
            return rotate(image, 90, resize=False, preserve_range=True), rotate(label, 90, resize=False, preserve_range=True)
        elif aug_idx == 2:
            return rotate(image, 180, resize=False, preserve_range=True), rotate(label, 180, resize=False, preserve_range=True)
        elif aug_idx == 3:
            return rotate(image, 270, resize=False, preserve_range=True), rotate(label, 270, resize=False, preserve_range=True)
        elif aug_idx == 4:
            return np.fliplr(image), np.fliplr(label)
        elif aug_idx == 5:
            return np.flipud(image), np.flipud(label)
        elif aug_idx == 6:
            return np.fliplr(rotate(image, 90, resize=False, preserve_range=True)), np.fliplr(rotate(label, 90, resize=False, preserve_range=True))
        elif aug_idx == 7:
            return np.flipud(rotate(image, 90, resize=False, preserve_range=True)), np.flipud(rotate(label, 90, resize=False, preserve_range=True))
        return image, label

    def __getitem__(self, idx):
        base_idx = idx // 8 if self.augment else idx
        aug_idx = idx % 8 if self.augment else 0

        image = cp_io.imread(self.image_files[base_idx])
        label = cp_io.imread(self.label_files[base_idx])

        if image.ndim == 3:
            image = image.mean(axis=-1)

        if image.max() > 1:
            image = image / image.max()

        if self.normalize:
            lower_p, upper_p = np.percentile(image, self.percentiles)
            image = rescale_intensity(image, in_range=(lower_p, upper_p), out_range=(0, 1))

        image, label = self.apply_augmentation(image, label, aug_idx)

        image_shape = (self.target_size, self.target_size)
        image = sk_resize(image, image_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
        label = sk_resize(label, image_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        return image, label


def plot_cellpose_batch(images, labels):
    from spacr.plot import generate_mask_random_cmap

    cmap_lbl = generate_mask_random_cmap(labels)
    batch_size = len(images)
    fig, axs = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
    for i in range(batch_size):
        axs[0, i].imshow(images[i], cmap='gray')
        axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].axis('off')
        axs[1, i].imshow(labels[i], cmap=cmap_lbl, interpolation='nearest')
        axs[1, i].set_title(f'Label {i+1}')
        axs[1, i].axis('off')
    plt.show()
    
    
def plot_cellpose_test_v1(images, labels, masks_pred, flows):
    from spacr.plot import generate_mask_random_cmap
    
    fig, axs = plt.subplots(len(images), 4, figsize=(16, 4 * len(images)), gridspec_kw={'wspace':0.1, 'hspace':0.1})
    
    for i in range(len(images)):
        cmap_lbl = generate_mask_random_cmap(labels[i])
        cmap_msk = generate_mask_random_cmap(masks_pred[i])
        
        axs[i, 0].imshow(images[i], cmap='gray')
        axs[i, 0].set_title(f'Image {i+1}')
        axs[i, 0].axis('off')
        axs[i, 1].imshow(labels[i], cmap=cmap_lbl, interpolation='nearest')
        axs[i, 1].set_title(f'True Label {i+1}')
        axs[i, 1].axis('off')
        axs[i, 2].imshow(masks_pred[i], cmap=cmap_msk, interpolation='nearest')
        axs[i, 2].set_title(f'Predicted Mask {i+1}')
        axs[i, 2].axis('off')
        axs[i, 3].imshow(flows[i][0], cmap='gray')
        axs[i, 3].set_title(f'Flows {i+1}')
        axs[i, 3].axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.show()
    
def plot_cellpose_test(images, labels, masks_pred, flows, save_dir=None, prefix='cellpose_result'):
    """
    Plot image, label, prediction, and flow for each sample in its own figure.

    Parameters:
    images (list): List of input images.
    labels (list): List of ground truth masks.
    masks_pred (list): List of predicted masks.
    flows (list): List of flow images (use flow[0] for each).
    save_dir (str or None): Directory to save figures. If None, figures are not saved.
    prefix (str): Prefix for saved figure filenames.
    """
    from spacr.plot import generate_mask_random_cmap

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for i in range(len(images)):
        
        width = sum(x is not None for x in [images, labels, masks_pred, flows])    
        fig, axs = plt.subplots(1, width, figsize=(16, width), gridspec_kw={'wspace':0.1, 'hspace':0.1})
        j = 0
        if not images is None:
            axs[j].imshow(images[i], cmap='gray')
            axs[j].set_title(f'Image {i+1}')
            axs[j].axis('off')
        
        if not labels is None:
            j += 1
            cmap_lbl = generate_mask_random_cmap(labels[i])
            axs[j].imshow(labels[i], cmap=cmap_lbl, interpolation='nearest')
            axs[j].set_title(f'True Label {i+1}')
            axs[j].axis('off')

        if not masks_pred is None:
            j += 1
            cmap_msk = generate_mask_random_cmap(masks_pred[i])
            axs[j].imshow(masks_pred[i], cmap=cmap_msk, interpolation='nearest')
            axs[j].set_title(f'Predicted Mask {i+1}')
            axs[j].axis('off')

        if not flows is None:
            j += 1
            axs[j].imshow(flows[i][0], cmap='gray')
            axs[j].set_title(f'Flows {i+1}')
            axs[j].axis('off')

        plt.tight_layout(pad=0.01)

        if save_dir:
            save_path = os.path.join(save_dir, f"{prefix}_{i+1:03d}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def train_cellpose(settings):
    settings = get_train_cellpose_default_settings(settings)
    img_src = os.path.join(settings['src'], 'train', 'images')
    mask_src = os.path.join(settings['src'], 'train', 'masks')
    target_size = settings['target_size']

    model_name = f"{settings['model_name']}_cyto_e{settings['n_epochs']}_X{target_size}_Y{target_size}.CP_model"
    model_save_path = os.path.join(settings['src'], 'models', 'cellpose_model')
    os.makedirs(model_save_path, exist_ok=True)

    save_settings(settings, name=model_name)

    model = cp_models.CellposeModel(gpu=True, model_type='cyto', diam_mean=30, pretrained_model='cyto')
    cp_channels = [0, 0]

    train_image_files = sorted([os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')])
    train_label_files = sorted([os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')])

    train_dataset = CellposeLazyDataset(train_image_files, train_label_files, settings, randomize=True, augment=settings['augment'])

    n_aug = 8 if settings['augment'] else 1
    max_base_images = len(train_dataset) // n_aug if settings['augment'] else len(train_dataset)
    n_base = min(settings['batch_size'], max_base_images)

    unique_base_indices = list(range(max_base_images))
    random.shuffle(unique_base_indices)
    selected_indices = unique_base_indices[:n_base]

    images, labels = [], []
    for idx in selected_indices:
        for aug_idx in range(n_aug):
            i = idx * n_aug + aug_idx if settings['augment'] else idx
            img, lbl = train_dataset[i]
            images.append(img)
            labels.append(lbl)
    try:
        plot_cellpose_batch(images, labels)
    except:
        print(f"could not print batch images")
        
    print(f"Training model with {len(images)} ber patch for {settings['n_epochs']} Epochs")

    train_cp.train_seg(model.net,
                       train_data=images,
                       train_labels=labels,
                       channels=cp_channels,
                       save_path=model_save_path,
                       n_epochs=settings['n_epochs'],
                       batch_size=settings['batch_size'],
                       learning_rate=settings['learning_rate'],
                       weight_decay=settings['weight_decay'],
                       model_name=model_name,
                       save_every=max(1, (settings['n_epochs'] // 10)),
                       rescale=False)

    print(f"Model saved at: {model_save_path}/{model_name}")
    
def test_cellpose_model(settings):
    
    from spacr.plot import generate_mask_random_cmap
    
    test_image_folder = os.path.join(settings['src'], 'test', 'images')
    test_label_folder = os.path.join(settings['src'], 'test', 'masks')

    test_image_files = sorted([os.path.join(test_image_folder, f) for f in os.listdir(test_image_folder) if f.endswith('.tif')])
    test_label_files = sorted([os.path.join(test_label_folder, f) for f in os.listdir(test_label_folder) if f.endswith('.tif')])

    test_dataset = CellposeLazyDataset(test_image_files, test_label_files, settings, randomize=False, augment=False)
    
    images, labels = map(list, zip(*[test_dataset[i] for i in range(len(test_dataset))]))

    model = cp_models.CellposeModel(gpu=True, pretrained_model=settings['model_path'])
    
    masks_pred, flows, _ = model.eval(x=images, 
                                      channels=[0, 0],
                                      normalize=False,
                                      diameter=30,
                                      flow_threshold=settings['FT'],
                                      cellprob_threshold=settings['CP_probability'])

    scores = [float(aggregated_jaccard_index([label], [pred])) for label, pred in zip(labels, masks_pred)]
    
    df_results = pd.DataFrame({
        'label_image': [os.path.basename(f) for f in test_label_files],
        'Jaccard': scores
    })
    
    if settings['save']:
        results_dir = os.path.join(settings['src'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        df_results.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        plot_cellpose_test(images, labels, masks_pred, flows, save_dir=results_dir, prefix='cellpose_result')        
        
def generate_cellpose_masks(settings):
    
    """
    Generate masks for a folder of images using a pretrained Cellpose model.

    Parameters:
    - image_dir (str): Path to directory containing input images (.tif).
    - model_path (str): Path to pretrained Cellpose model (.CP_model).
    - save_dir (str or None): If specified, saves masks to this directory.
    - settings (dict or None): Optional dictionary with keys:
        'target_size': int (rescale input),
        'normalize': bool,
        'percentiles': tuple of (lower, upper) for intensity rescaling.
    """
    
    image_dir = settings['src']
    model_path = settings['model_path']
    save_dir = os.path.join(settings['src'],  'masks')
    
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    if not image_files:
        raise ValueError(f"No .tif files found in {image_dir}")

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    normalize = settings.get('normalize', True) if settings else True
    percentiles = settings.get('percentiles', (2, 99)) if settings else (2, 99)
    target_size = settings.get('target_size', None) if settings else None

    images = []
    for path in image_files:
        img = cp_io.imread(path)
        if img.ndim == 3:
            img = img.mean(axis=-1)

        if img.max() > 1:
            img = img / img.max()

        if normalize:
            p_low, p_high = np.percentile(img, percentiles)
            img = rescale_intensity(img, in_range=(p_low, p_high), out_range=(0, 1))

        if target_size:
            img = sk_resize(img, (target_size, target_size), preserve_range=True, anti_aliasing=True).astype(np.float32)

        images.append(img)

    model = cp_models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks, flows, _ = model.eval(x=images,
                                  channels=[0, 0],
                                  normalize=False,
                                  diameter=30,
                                  flow_threshold=0.4,
                                  cellprob_threshold=0.0)
         
    if settings['save']:
        results_dir = os.path.join(settings['src'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        plot_cellpose_test(images, None, masks, flows, save_dir=results_dir, prefix='cellpose_result')
        
        for path, mask in zip(image_files, masks):
            base = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(save_dir, base + "_mask.tif")
            cp_io.imsave(save_path, mask.astype(np.uint16))
    return


def analyze_percent_positive(settings):
    from spacr.io import _read_and_merge_data
    from spacr.utils import save_settings
    from .settings import default_settings_analyze_percent_positive
    
    settings = default_settings_analyze_percent_positive(settings)
    
    def translate_well_in_df(csv_loc):
        # Load and extract metadata
        df = pd.read_csv(csv_loc)
        df[['plate', 'well']] = df['Renamed TIFF'].str.replace('.tif', '', regex=False).str.split('_', expand=True)[[0, 1]]
        df['plate_well'] = df['plate'] + '_' + df['well']

        # Retain one row per plate_well
        df_2 = df.drop_duplicates(subset='plate_well').copy()

        # Translate well to row and column
        df_2['row_name'] = 'r' + df_2['well'].str[0].map(lambda x: str(string.ascii_uppercase.index(x) + 1))
        df_2['column_name'] = 'c' + df_2['well'].str[1:].astype(int).astype(str)

        # Optional: add prcf ID (plate_row_column_field)
        df_2['field'] = 'f1'  # default or extract from filename if needed
        df_2['prc'] = 'p' + df_2['plate'].str.extract(r'(\d+)')[0] + '_' + df_2['row_name'] + '_' + df_2['column_name']

        return df_2
    
    def annotate_and_summarize(df, value_col, condition_col, well_col, threshold, annotation_col='annotation'):
        """
        Annotate and summarize a DataFrame based on a threshold.

        Parameters:
        - df: pandas.DataFrame
        - value_col: str, column name to apply threshold on
        - condition_col: str, column name for experimental condition
        - well_col: str, column name for wells
        - threshold: float, threshold value for annotation
        - annotation_col: str, name of the new annotation column

        Returns:
        - df: annotated DataFrame
        - summary_df: DataFrame with counts and fractions per condition and well
        """
        # Annotate
        df[annotation_col] = np.where(df[value_col] > threshold, 'above', 'below')

        # Count per condition and well
        count_df = df.groupby([condition_col, well_col, annotation_col]).size().unstack(fill_value=0)

        # Calculate total and fractions
        count_df['total'] = count_df.sum(axis=1)
        count_df['fraction_above'] = count_df.get('above', 0) / count_df['total']
        count_df['fraction_below'] = count_df.get('below', 0) / count_df['total']

        return df, count_df.reset_index()
    
    save_settings(settings, name='analyze_percent_positive', show=False)
    
    df, _ = _read_and_merge_data(locs=[settings['src']+'/measurements/measurements.db'], 
                             tables=settings['tables'], 
                             verbose=True, 
                             nuclei_limit=None, 
                             pathogen_limit=None)

    df['condition'] = 'none'
    
    if not settings['filter_1'] is None:
        df = df[df[settings['filter_1'][0]]>settings['filter_1'][1]]
    
    condition_col = 'condition'
    well_col = 'prc'
    
    df, count_df = annotate_and_summarize(df, settings['value_col'], condition_col, well_col, settings['threshold'], annotation_col='annotation')
    count_df[['plate', 'row_name', 'column_name']] = count_df['prc'].str.split('_', expand=True)
    
    csv_loc = os.path.join(settings['src'], 'rename_log.csv')
    csv_out_loc = os.path.join(settings['src'], 'result.csv')
    translate_df = translate_well_in_df(csv_loc)
    
    merged = pd.merge(count_df, translate_df, on=['row_name', 'column_name'], how='inner')

    merged = merged[['plate_y', 'well', 'plate_well','field','row_name','column_name','prc_x','Original File','Renamed TIFF','above','below','fraction_above','fraction_below']]
    merged[[f'part{i}' for i in range(merged['Original File'].str.count('_').max() + 1)]] = merged['Original File'].str.split('_', expand=True)
    merged.to_csv(csv_out_loc, index=False)
    display(merged)
    return merged

def analyze_recruitment(settings):
    """
    Analyze recruitment data by grouping the DataFrame by well coordinates and plotting controls and recruitment data.

    Parameters:
    settings (dict): settings.

    Returns:
    None
    """
    
    from .io import _read_and_merge_data, _results_to_csv
    from .plot import plot_image_mask_overlay, _plot_controls, _plot_recruitment
    from .utils import _object_filter, annotate_conditions, _calculate_recruitment, _group_by_well, save_settings
    from .settings import get_analyze_recruitment_default_settings

    settings = get_analyze_recruitment_default_settings(settings=settings)
    save_settings(settings, name='recruitment')

    print(f"Cell(s): {settings['cell_types']}, in {settings['cell_plate_metadata']}")
    print(f"Pathogen(s): {settings['pathogen_types']}, in {settings['pathogen_plate_metadata']}")
    print(f"Treatment(s): {settings['treatments']}, in {settings['treatment_plate_metadata']}")
    
    mask_chans=[settings['nucleus_chann_dim'], settings['pathogen_chann_dim'], settings['cell_chann_dim']]
    
    sns.color_palette("mako", as_cmap=True)
    print(f"channel:{settings['channel_of_interest']} = {settings['target']}")
    
    df, _ = _read_and_merge_data(locs=[settings['src']+'/measurements/measurements.db'], 
                                 tables=['cell', 'nucleus', 'pathogen','cytoplasm'], 
                                 verbose=True, 
                                 nuclei_limit=settings['nuclei_limit'], 
                                 pathogen_limit=settings['pathogen_limit'])
        
    df = annotate_conditions(df, 
                             cells=settings['cell_types'], 
                             cell_loc=settings['cell_plate_metadata'], 
                             pathogens=settings['pathogen_types'],
                             pathogen_loc=settings['pathogen_plate_metadata'],
                             treatments=settings['treatments'], 
                             treatment_loc=settings['treatment_plate_metadata'])
      
    df = df.dropna(subset=['condition'])
    print(f'After dropping non-annotated wells: {len(df)} rows')

    files = df['file_name'].tolist()
    print(f'found: {len(files)} files')

    files = [item + '.npy' for item in files]
    random.shuffle(files)

    _max = 10**100
    if settings['cell_size_range'] is None:
        settings['cell_size_range'] = [0,_max]
    if settings['nucleus_size_range'] is None:
        settings['nucleus_size_range'] = [0,_max]
    if settings['pathogen_size_range'] is None:
        settings['pathogen_size_range'] = [0,_max]

    if settings['plot']:
        merged_path = os.path.join(settings['src'],'merged')
        if os.path.exists(merged_path):
            try:
                for idx, file in enumerate(os.listdir(merged_path)):
                    file_path = os.path.join(merged_path,file)
                    if idx <= settings['plot_nr']:
                        plot_image_mask_overlay(file_path, 
                                                settings['channel_dims'],
                                                settings['cell_chann_dim'],
                                                settings['nucleus_chann_dim'],
                                                settings['pathogen_chann_dim'],
                                                figuresize=10,
                                                normalize=True,
                                                thickness=3,
                                                save_pdf=True)
            except Exception as e:
                print(f'Failed to plot images with outlines, Error: {e}')
        
    if not settings['cell_chann_dim'] is None:
        df = _object_filter(df, 'cell', settings['cell_size_range'], settings['cell_intensity_range'], mask_chans, 0)
        if not settings['target_intensity_min'] is None or not settings['target_intensity_min'] is 0:
            df = df[df[f"cell_channel_{settings['channel_of_interest']}_percentile_95"] > settings['target_intensity_min']]
            print(f"After channel {settings['channel_of_interest']} filtration", len(df))
    if not settings['nucleus_chann_dim'] is None:
        df = _object_filter(df, 'nucleus', settings['nucleus_size_range'], settings['nucleus_intensity_range'], mask_chans, 1)
    if not settings['pathogen_chann_dim'] is None:
        df = _object_filter(df, 'pathogen', settings['pathogen_size_range'], settings['pathogen_intensity_range'], mask_chans, 2)
       
    df['recruitment'] = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"]/df[f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]
    
    for chan in settings['channel_dims']:
        df = _calculate_recruitment(df, channel=chan)
    print(f'calculated recruitment for: {len(df)} rows')
    
    df_well = _group_by_well(df)
    print(f'found: {len(df_well)} wells')
    
    df_well = df_well[df_well['cells_per_well'] >= settings['cells_per_well']]
    prc_list = df_well['prc'].unique().tolist()
    df = df[df['prc'].isin(prc_list)]
    print(f"After cells per well filter: {len(df)} cells in {len(df_well)} wells left wth threshold {settings['cells_per_well']}")
    
    if settings['plot_control']:
        _plot_controls(df, mask_chans, settings['channel_of_interest'], figuresize=5)

    print(f'PV level: {len(df)} rows')
    _plot_recruitment(df, 'by PV', settings['channel_of_interest'], columns=[], figuresize=settings['figuresize'])
    print(f'well level: {len(df_well)} rows')
    _plot_recruitment(df_well, 'by well', settings['channel_of_interest'], columns=[], figuresize=settings['figuresize'])
    cells,wells = _results_to_csv(settings['src'], df, df_well)

    return [cells,wells]

def analyze_plaques(settings):

    from .cellpose import identify_masks_finetune
    from .settings import get_analyze_plaque_settings
    from .utils import save_settings, download_models
    from spacr import __file__ as spacr_path

    download_models()
    package_dir = os.path.dirname(spacr_path)
    models_dir = os.path.join(package_dir, 'resources', 'models', 'cp')
    model_path = os.path.join(models_dir, 'toxo_plaque_cyto_e25000_X1120_Y1120.CP_model')
    settings['custom_model'] = model_path
    print('custom_model',settings['custom_model'])

    settings = get_analyze_plaque_settings(settings)
    save_settings(settings, name='analyze_plaques', show=True)
    settings['dst'] = os.path.join(settings['src'], 'masks')

    if settings['masks']:
        identify_masks_finetune(settings)
        folder = settings['dst']
    else:
        folder = settings['dst']

    summary_data = []
    details_data = []
    stats_data = []
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if filepath.endswith('.tif') and os.path.isfile(filepath):
            print(f"Analyzing: {filepath}")
            image = cellpose.io.imread(filepath)
            labeled_image = label(image)
            regions = regionprops(labeled_image)
            
            object_count = len(regions)
            sizes = [region.area for region in regions]
            average_size = np.mean(sizes) if sizes else 0
            std_dev_size = np.std(sizes) if sizes else 0
            
            summary_data.append({'file': filename, 'object_count': object_count, 'average_size': average_size})
            stats_data.append({'file': filename, 'plaque_count': object_count, 'average_size': average_size, 'std_dev_size': std_dev_size})
            for size in sizes:
                details_data.append({'file': filename, 'plaque_size': size})
    
    # Convert lists to pandas DataFrames
    summary_df = pd.DataFrame(summary_data)
    details_df = pd.DataFrame(details_data)
    stats_df = pd.DataFrame(stats_data)
    
    # Save DataFrames to a SQLite database
    db_name = os.path.join(folder, 'plaques_analysis.db')
    conn = sqlite3.connect(db_name)
    
    summary_df.to_sql('summary', conn, if_exists='replace', index=False)
    details_df.to_sql('details', conn, if_exists='replace', index=False)
    stats_df.to_sql('stats', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"Analysis completed and saved to database '{db_name}'.")
    


def train_cellpose_v2(settings):
    
    from spacr.io import _load_normalized_images_and_labels, _load_images_and_labels
    from spacr.settings import get_train_cellpose_default_settings
    from spacr.utils import save_settings
    
    settings = get_train_cellpose_default_settings(settings)
    
    img_src = settings['src']
    
    img_src = os.path.join(settings['src'],'train', 'images')
    mask_src = os.path.join(settings['src'], 'train', 'masks')
    test_img_src = os.path.join(settings['src'],'test', 'images')
    test_mask_src = os.path.join(settings['src'], 'test', 'masks')
    
    if settings['resize']:
        target_dimensions = settings['width_dimensions']

    if settings['test']:
        if os.path.exists(test_img_src) and os.path.exists(test_mask_src):
            print(f"Found test set")
        else:
            print(f"could not find test folders: {test_img_src} and {test_mask_src}")
            return

    test_images, test_masks, test_image_names, test_mask_names = None,None,None,None

    if settings['from_scratch']:
        model_name=f"scratch_{settings['model_name']}_{settings['model_type']}_e{settings['n_epochs']}_X{target_dimensions}_Y{target_dimensions}.CP_model"
    else:
        if settings['resize']:
            model_name=f"{settings['model_name']}_{settings['model_type']}_e{settings['n_epochs']}_X{target_dimensions}_Y{target_dimensions}.CP_model"
        else:
            model_name=f"{settings['model_name']}_{settings['model_type']}_e{settings['n_epochs']}.CP_model"

    model_save_path = os.path.join(settings['src'], 'models', 'cellpose_model')
    
    print(model_save_path)
    
    os.makedirs(model_save_path, exist_ok=True)
    save_settings(settings, name=f"{model_name}")
    
    if settings['from_scratch']:
        model = cp_models.CellposeModel(gpu=True, model_type=settings['model_type'], diam_mean=settings['diameter'], pretrained_model=None)
    else:
        model = cp_models.CellposeModel(gpu=True, model_type=settings['model_type'])
        
    if settings['normalize']:

        image_files = [os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')]
        label_files = [os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')]
        images, masks, image_names, mask_names, orig_dims = _load_normalized_images_and_labels(image_files, 
                                                                                               label_files, 
                                                                                               settings['channels'], 
                                                                                               settings['percentiles'],  
                                                                                               settings['invert'], 
                                                                                               settings['verbose'], 
                                                                                               settings['remove_background'], 
                                                                                               settings['background'], 
                                                                                               settings['Signal_to_noise'], 
                                                                                               settings['target_dimensions'], 
                                                                                               settings['target_dimensions'])        
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if settings['test']:
            test_image_files = [os.path.join(test_img_src, f) for f in os.listdir(test_img_src) if f.endswith('.tif')]
            test_label_files = [os.path.join(test_mask_src, f) for f in os.listdir(test_mask_src) if f.endswith('.tif')]
            test_images, test_masks, test_image_names, test_mask_names = _load_normalized_images_and_labels(test_image_files, 
                                                                                                            test_label_files, 
                                                                                                            settings['channels'], 
                                                                                                            settings['percentiles'],  
                                                                                                            settings['invert'], 
                                                                                                            settings['verbose'], 
                                                                                                            settings['remove_background'], 
                                                                                                            settings['background'], 
                                                                                                            settings['Signal_to_noise'], 
                                                                                                            settings['target_dimensions'], 
                                                                                                            settings['target_dimensions'])
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
            
    else:
        image_files = sorted([os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')])
        label_files = sorted([os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')])
        images, masks, image_names, mask_names = _load_images_and_labels(image_files, label_files, settings['invert'])
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if settings['test']:
            test_image_files = sorted([os.path.join(test_img_src, f) for f in os.listdir(test_img_src) if f.endswith('.tif')])
            test_label_files = sorted([os.path.join(test_mask_src, f) for f in os.listdir(test_mask_src) if f.endswith('.tif')])
            test_images, test_masks, test_image_names, test_mask_names = _load_images_and_labels(test_image_files, test_label_files, settings['invert'])
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
    
    #if resize:
    #    images, masks = resize_images_and_labels(images, masks, target_height, target_width, show_example=True)

    if settings['model_type'] == 'cyto':
        cp_channels = [0,1]
    if settings['model_type'] == 'cyto2':
        cp_channels = [0,2]
    if settings['model_type'] == 'cyto3':
        cp_channels = [0,2]
    if settings['model_type'] == 'nucleus':
        cp_channels = [0,0]
    if settings['grayscale']:
        cp_channels = [0,0]
        images = [np.squeeze(img) if img.ndim == 3 and 1 in img.shape else img for img in images]
    
    masks = [np.squeeze(mask) if mask.ndim == 3 and 1 in mask.shape else mask for mask in masks]

    print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {masks[0].shape}, image type: masks[0].shape')
    save_every = int(settings['n_epochs']/10)
    if save_every < 10:
        save_every = settings['n_epochs']
        
    if settings['test'] and (not test_images or not test_masks):
        print("WARNING: Test data missing or empty. Proceeding without test set.")
        test_images, test_masks = None, None
        test_image_names, test_mask_names = None, None

    train_cp.train_seg(model.net,
                    train_data=images,
                    train_labels=masks,
                    train_files=image_names,
                    train_labels_files=mask_names,
                    train_probs=None,
                    test_data=test_images,
                    test_labels=test_masks,
                    test_files=test_image_names,
                    test_labels_files=test_mask_names, 
                    test_probs=None,
                    load_files=True,
                    batch_size=settings['batch_size'],
                    learning_rate=settings['learning_rate'],
                    n_epochs=settings['n_epochs'],
                    weight_decay=settings['weight_decay'],
                    momentum=0.9,
                    SGD=False,
                    channels=cp_channels,
                    channel_axis=None,
                    normalize=False, 
                    compute_flows=False,
                    save_path=model_save_path,
                    save_every=save_every,
                    nimg_per_epoch=None,
                    nimg_test_per_epoch=None,
                    rescale=settings['rescale'],
                    min_train_masks=1,
                    model_name=settings['model_name'])

    return print(f"Model saved at: {model_save_path}/{model_name}")

def train_cellpose_v1(settings):
    
    from .io import _load_normalized_images_and_labels, _load_images_and_labels
    from .settings import get_train_cellpose_default_settings
    from .utils import save_settings

    settings = get_train_cellpose_default_settings(settings)

    img_src = settings['img_src'] 
    mask_src = os.path.join(img_src, 'masks')
    test_img_src = settings['test_img_src']
    test_mask_src = settings['test_mask_src']

    if settings['resize']:
        target_height = settings['width_height'][1]
        target_width = settings['width_height'][0]

    if settings['test']:
        test_img_src = os.path.join(os.path.dirname(settings['img_src']), 'test')
        test_mask_src = os.path.join(settings['test_img_src'], 'mask')

    test_images, test_masks, test_image_names, test_mask_names = None,None,None,None
    print(settings)

    if settings['from_scratch']:
        model_name=f"scratch_{settings['model_name']}_{settings['model_type']}_e{settings['n_epochs']}_X{target_width}_Y{target_height}.CP_model"
    else:
        if settings['resize']:
            model_name=f"{settings['model_name']}_{settings['model_type']}_e{settings['n_epochs']}_X{target_width}_Y{target_height}.CP_model"
        else:
            model_name=f"{settings['model_name']}_{settings['model_type']}_e{settings['n_epochs']}.CP_model"

    model_save_path = os.path.join(settings['mask_src'], 'models', 'cellpose_model')
    print(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

    save_settings(settings, name=model_name)
    
    if settings['from_scratch']:
        model = cp_models.CellposeModel(gpu=True, model_type=settings['model_type'], diam_mean=settings['diameter'], pretrained_model=None)
    else:
        model = cp_models.CellposeModel(gpu=True, model_type=settings['model_type'])
        
    if settings['normalize']:

        image_files = [os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')]
        label_files = [os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')]
        images, masks, image_names, mask_names, orig_dims = _load_normalized_images_and_labels(image_files, 
                                                                                               label_files, 
                                                                                               settings['channels'], 
                                                                                               settings['percentiles'],  
                                                                                               settings['invert'], 
                                                                                               settings['verbose'], 
                                                                                               settings['remove_background'], 
                                                                                               settings['background'], 
                                                                                               settings['Signal_to_noise'], 
                                                                                               settings['target_height'], 
                                                                                               settings['target_width'])        
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if settings['test']:
            test_image_files = [os.path.join(test_img_src, f) for f in os.listdir(test_img_src) if f.endswith('.tif')]
            test_label_files = [os.path.join(test_mask_src, f) for f in os.listdir(test_mask_src) if f.endswith('.tif')]
            test_images, test_masks, test_image_names, test_mask_names = _load_normalized_images_and_labels(test_image_files, 
                                                                                                            test_label_files, 
                                                                                                            settings['channels'], 
                                                                                                            settings['percentiles'],  
                                                                                                            settings['invert'], 
                                                                                                            settings['verbose'], 
                                                                                                            settings['remove_background'], 
                                                                                                            settings['background'], 
                                                                                                            settings['Signal_to_noise'], 
                                                                                                            settings['target_height'], 
                                                                                                            settings['target_width'])
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
            
    else:
        images, masks, image_names, mask_names = _load_images_and_labels(img_src, mask_src, settings['invert'])
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if settings['test']:
            test_images, test_masks, test_image_names, test_mask_names = _load_images_and_labels(test_img_src, 
                                                                                                 test_mask_src, 
                                                                                                 settings['invert'])
            
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
    
    #if resize:
    #    images, masks = resize_images_and_labels(images, masks, target_height, target_width, show_example=True)

    if settings['model_type'] == 'cyto':
        cp_channels = [0,1]
    if settings['model_type'] == 'cyto2':
        cp_channels = [0,2]
    if settings['model_type'] == 'nucleus':
        cp_channels = [0,0]
    if settings['grayscale']:
        cp_channels = [0,0]
        images = [np.squeeze(img) if img.ndim == 3 and 1 in img.shape else img for img in images]
    
    masks = [np.squeeze(mask) if mask.ndim == 3 and 1 in mask.shape else mask for mask in masks]

    print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {masks[0].shape}, image type: masks[0].shape')
    save_every = int(settings['n_epochs']/10)
    if save_every < 10:
        save_every = settings['n_epochs']

    train_cp.train_seg(model.net,
                    train_data=images,
                    train_labels=masks,
                    train_files=image_names,
                    train_labels_files=mask_names,
                    train_probs=None,
                    test_data=test_images,
                    test_labels=test_masks,
                    test_files=test_image_names,
                    test_labels_files=test_mask_names, 
                    test_probs=None,
                    load_files=True,
                    batch_size=settings['batch_size'],
                    learning_rate=settings['learning_rate'],
                    n_epochs=settings['n_epochs'],
                    weight_decay=settings['weight_decay'],
                    momentum=0.9,
                    SGD=False,
                    channels=cp_channels,
                    channel_axis=None,
                    normalize=False, 
                    compute_flows=False,
                    save_path=model_save_path,
                    save_every=save_every,
                    nimg_per_epoch=None,
                    nimg_test_per_epoch=None,
                    rescale=settings['rescale'],
                    #scale_range=None,
                    #bsize=224,
                    min_train_masks=1,
                    model_name=settings['model_name'])

    return print(f"Model saved at: {model_save_path}/{model_name}")

def count_phenotypes(settings):
    from .io import _read_db

    if not settings['src'].endswith('/measurements/measurements.db'):
        settings['src'] = os.path.join(settings['src'], 'measurements/measurements.db')

    df = _read_db(loc=settings['src'], tables=['png_list'])

    unique_values_count = df[settings['annotation_column']].nunique(dropna=True)
    print(f"Unique values in {settings['annotation_column']} (excluding NaN): {unique_values_count}")

    # Count unique values in 'value' column, grouped by 'plate', 'row_name', 'column'
    grouped_unique_count = df.groupby(['plate', 'row_name', 'column'])[settings['annotation_column']].nunique(dropna=True).reset_index(name='unique_count')
    display(grouped_unique_count)

    save_path = os.path.join(settings['src'], 'phenotype_counts.csv')

    # Group by plate, row, and column, then count the occurrences of each unique value
    grouped_counts = df.groupby(['plate', 'row_name', 'column', 'value']).size().reset_index(name='count')

    # Pivot the DataFrame so that unique values are columns and their counts are in the rows
    pivot_df = grouped_counts.pivot_table(index=['plate', 'row_name', 'column'], columns='value', values='count', fill_value=0)

    # Flatten the multi-level columns
    pivot_df.columns = [f"value_{int(col)}" for col in pivot_df.columns]

    # Reset the index so that plate, row, and column form a combined index
    pivot_df.index = pivot_df.index.map(lambda x: f"{x[0]}_{x[1]}_{x[2]}")

    # Saving the DataFrame to a SQLite .db file
    output_dir = os.path.join('src', 'results')  # Replace 'src' with the actual base directory
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.dirname(settings['src'])
    output_path = os.path.join(output_dir, 'phenotype_counts.csv')

    pivot_df.to_csv(output_path)

    return

def compare_reads_to_scores(reads_csv, scores_csv, empirical_dict={'r1':(90,10),'r2':(90,10),'r3':(80,20),'r4':(80,20),'r5':(70,30),'r6':(70,30),'r7':(60,40),'r8':(60,40),'r9':(50,50),'r10':(50,50),'r11':(40,60),'r12':(40,60),'r13':(30,70),'r14':(30,70),'r15':(20,80),'r16':(20,80)},
                            pc_grna='TGGT1_220950_1', nc_grna='TGGT1_233460_4', 
                            y_columns=['class_1_fraction', 'TGGT1_220950_1_fraction', 'nc_fraction'], 
                            column='column', value='c3', plate=None, save_paths=None):

    def calculate_well_score_fractions(df, class_columns='cv_predictions'):
        if all(col in df.columns for col in ['plate', 'row_name', 'column']):
            df['prc'] = df['plate'] + '_' + df['row_name'] + '_' + df['column']
        else:
            raise ValueError("Cannot find 'plate', 'row_name', or 'column' in df.columns")
        prc_summary = df.groupby(['plate', 'row_name', 'column', 'prc']).size().reset_index(name='total_rows')
        well_counts = (df.groupby(['plate', 'row_name', 'column', 'prc', class_columns])
                       .size()
                       .unstack(fill_value=0)
                       .reset_index()
                       .rename(columns={0: 'class_0', 1: 'class_1'}))
        summary_df = pd.merge(prc_summary, well_counts, on=['plate', 'row_name', 'column', 'prc'], how='left')
        summary_df['class_0_fraction'] = summary_df['class_0'] / summary_df['total_rows']
        summary_df['class_1_fraction'] = summary_df['class_1'] / summary_df['total_rows']
        return summary_df
        
    def plot_line(df, x_column, y_columns, group_column=None, xlabel=None, ylabel=None, 
                  title=None, figsize=(10, 6), save_path=None, theme='deep'):
        """
        Create a line plot that can handle multiple y-columns, each becoming a separate line.
        """

        def _set_theme(theme):
            """Set the Seaborn theme and reorder colors if necessary."""

            def __set_reordered_theme(theme='deep', order=None, n_colors=100, show_theme=False):
                """Set and reorder the Seaborn color palette."""
                palette = sns.color_palette(theme, n_colors)
                if order:
                    reordered_palette = [palette[i] for i in order]
                else:
                    reordered_palette = palette
                if show_theme:
                    sns.palplot(reordered_palette)
                    plt.show()
                return reordered_palette

            integer_list = list(range(1, 81))
            color_order = [7, 9, 4, 0, 3, 6, 2] + integer_list
            sns_palette = __set_reordered_theme(theme, color_order, 100)
            return sns_palette

        sns_palette = _set_theme(theme)

        # Sort the DataFrame based on the x_column
        df = df.loc[natsorted(df.index, key=lambda x: df.loc[x, x_column])]
        
        fig, ax = plt.subplots(figsize=figsize)

        # Handle multiple y-columns, each as a separate line
        if isinstance(y_columns, list):
            for idx, y_col in enumerate(y_columns):
                sns.lineplot(
                    data=df, x=x_column, y=y_col, ax=ax, label=y_col, 
                    color=sns_palette[idx % len(sns_palette)], linewidth=1
                )
        else:
            sns.lineplot(
                data=df, x=x_column, y=y_columns, hue=group_column, ax=ax, 
                palette=sns_palette, linewidth=2
            )

        # Set axis labels and title
        ax.set_xlabel(xlabel if xlabel else x_column)
        ax.set_ylabel(ylabel if ylabel else 'Value')
        ax.set_title(title if title else 'Line Plot')

        # Remove top and right spines
        sns.despine(ax=ax)

        # Ensure legend only appears when needed and place it to the right
        if group_column or isinstance(y_columns, list):
            ax.legend(title='Legend', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()
        return fig
    
    def calculate_grna_fraction_ratio(df, grna1='TGGT1_220950_1', grna2='TGGT1_233460_4'):
        # Filter relevant grna_names within each prc and group them
        grouped = df[df['grna_name'].isin([grna1, grna2])] \
            .groupby(['prc', 'grna_name']) \
            .agg({'fraction': 'sum', 'count': 'sum'}) \
            .unstack(fill_value=0)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped['fraction_ratio'] = grouped[f'fraction_{grna1}'] / grouped[f'fraction_{grna2}']
        grouped = grouped.assign(
            fraction_ratio=lambda x: x['fraction_ratio'].replace([float('inf'), -float('inf')], 0)
        ).fillna({'fraction_ratio': 0})
        grouped = grouped.rename(columns={
            f'count_{grna1}': f'{grna1}_count',
            f'count_{grna2}': f'{grna2}_count'
        })
        result = grouped.reset_index()[['prc', f'{grna1}_count', f'{grna2}_count', 'fraction_ratio']]
        result['total_reads'] = result[f'{grna1}_count'] + result[f'{grna2}_count']
        result[f'{grna1}_fraction'] = result[f'{grna1}_count'] / result['total_reads']
        result[f'{grna2}_fraction'] = result[f'{grna2}_count'] / result['total_reads']
        return result

    def calculate_well_read_fraction(df, count_column='count'):
        if all(col in df.columns for col in ['plate', 'row_name', 'column']):
            df['prc'] = df['plate'] + '_' + df['row_name'] + '_' + df['column']
        else:
            raise ValueError("Cannot find plate, row or column in df.columns")
        grouped_df = df.groupby('prc')[count_column].sum().reset_index()
        grouped_df = grouped_df.rename(columns={count_column: 'total_counts'})
        df = pd.merge(df, grouped_df, on='prc')
        df['fraction'] = df['count'] / df['total_counts']
        return df
    
    if isinstance(reads_csv, list):
        if len(reads_csv) == len(scores_csv):
            reads_ls = []
            scores_ls = []
            for i, reads_csv_temp in enumerate(reads_csv):
                reads_df_temp = pd.read_csv(reads_csv_temp)
                scores_df_temp = pd.read_csv(scores_csv[i])
                reads_df_temp['plate'] = f"plate{i+1}"
                scores_df_temp['plate'] = f"plate{i+1}"
                
                if 'column_name' in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'column_name': 'column'})
                if 'column_name' in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'column_name': 'column'})
                if 'column_name' in scores_df_temp.columns:
                    scores_df_temp = scores_df_temp.rename(columns={'column_name': 'column'})
                if 'column_name' in scores_df_temp.columns:
                    scores_df_temp = scores_df_temp.rename(columns={'column_name': 'column'})
                if 'row_name' in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'row_name': 'row_name'})
                if 'row_name' in scores_df_temp.columns:
                    scores_df_temp = scores_df_temp.rename(columns={'row_name': 'row_name'})
                    
                reads_ls.append(reads_df_temp)
                scores_ls.append(scores_df_temp)
                    
            reads_df = pd.concat(reads_ls, axis=0)
            scores_df = pd.concat(scores_ls, axis=0)
            print(f"Reads: {len(reads_df)} Scores: {len(scores_df)}")
        else:
            print(f"reads_csv and scores_csv must contain the same number of elements if reads_csv is a list")
    else:
        reads_df = pd.read_csv(reads_csv)
        scores_df = pd.read_csv(scores_csv)
        if plate != None:
            reads_df['plate'] = plate
            scores_df['plate'] = plate
        
    reads_df = calculate_well_read_fraction(reads_df)
    scores_df = calculate_well_score_fractions(scores_df)
    reads_col_df = reads_df[reads_df[column]==value]
    scores_col_df = scores_df[scores_df[column]==value]
    
    reads_col_df = calculate_grna_fraction_ratio(reads_col_df, grna1=pc_grna, grna2=nc_grna)
    df = pd.merge(reads_col_df, scores_col_df, on='prc')
    
    df_emp = pd.DataFrame([(key, val[0], val[1], val[0] / (val[0] + val[1]), val[1] / (val[0] + val[1])) for key, val in empirical_dict.items()],columns=['key', 'value1', 'value2', 'pc_fraction', 'nc_fraction'])
    
    df = pd.merge(df, df_emp, left_on='row_name', right_on='key')
    
    if any in y_columns not in df.columns:
        print(f"columns in dataframe:")
        for col in df.columns:
            print(col)
        return
    display(df)
    fig_1 = plot_line(df, x_column = 'pc_fraction', y_columns=y_columns, group_column=None, xlabel=None, ylabel='Fraction', title=None, figsize=(10, 6), save_path=save_paths[0])
    fig_2 = plot_line(df, x_column = 'nc_fraction', y_columns=y_columns, group_column=None, xlabel=None, ylabel='Fraction', title=None, figsize=(10, 6), save_path=save_paths[1])
    
    return [fig_1, fig_2]

def interperate_vision_model(settings={}):
    
    from .io import _read_and_merge_data

    def generate_comparison_columns(df, compartments=['cell', 'nucleus', 'pathogen', 'cytoplasm']):

        comparison_dict = {}

        # Get columns by compartment
        compartment_columns = {comp: [col for col in df.columns if col.startswith(comp)] for comp in compartments}

        for comp0, comp0_columns in compartment_columns.items():
            for comp0_col in comp0_columns:
                related_cols = []
                base_col_name = comp0_col.replace(comp0, '')  # Base feature name without compartment prefix

                # Look for matching columns in other compartments
                for prefix, prefix_columns in compartment_columns.items():
                    if prefix == comp0:  # Skip same-compartment comparisons
                        continue
                    # Check if related column exists in other compartment
                    related_col = prefix + base_col_name
                    if related_col in df.columns:
                        related_cols.append(related_col)
                        new_col_name = f"{prefix}_{comp0}{base_col_name}"  # Format: prefix_comp0_base

                        # Calculate ratio and handle infinite or NaN values
                        df[new_col_name] = df[related_col] / df[comp0_col]
                        df[new_col_name].replace([float('inf'), -float('inf')], pd.NA, inplace=True)  # Replace inf values with NA
                        df[new_col_name].fillna(0, inplace=True)  # Replace NaN values with 0 for ease of further calculations

                # Generate all-to-all comparisons
                if related_cols:
                    comparison_dict[comp0_col] = related_cols
                    for i, rel_col_1 in enumerate(related_cols):
                        for rel_col_2 in related_cols[i + 1:]:
                            # Create a new column name for each pairwise comparison
                            comp1, comp2 = rel_col_1.split('_')[0], rel_col_2.split('_')[0]
                            new_col_name_all = f"{comp1}_{comp2}{base_col_name}"

                            # Calculate pairwise ratio and handle infinite or NaN values
                            df[new_col_name_all] = df[rel_col_1] / df[rel_col_2]
                            df[new_col_name_all].replace([float('inf'), -float('inf')], pd.NA, inplace=True)  # Replace inf with NA
                            df[new_col_name_all].fillna(0, inplace=True)  # Replace NaN with 0

        return df, comparison_dict

    def group_feature_class(df, feature_groups=['cell', 'cytoplasm', 'nucleus', 'pathogen'], name='compartment', include_all=False):

        # Function to determine compartment based on multiple matches
        def find_feature_class(feature, compartments):
            matches = [compartment for compartment in compartments if re.search(compartment, feature)]
            if len(matches) > 1:
                return '-'.join(matches)
            elif matches:
                return matches[0]
            else:
                return None

        from spacr.plot import spacrGraph

        df[name] = df['feature'].apply(lambda x: find_feature_class(x, feature_groups))

        if name == 'channel':
            df['channel'].fillna('morphology', inplace=True)

        # Create new DataFrame with summed importance for each compartment and channel
        importance_sum = df.groupby(name)['importance'].sum().reset_index(name=f'{name}_importance_sum')
        
        if include_all:
            total_compartment_importance = importance_sum[f'{name}_importance_sum'].sum()
            importance_sum = pd.concat(
                [importance_sum,
                 pd.DataFrame(
                     [{name: 'all', f'{name}_importance_sum': total_compartment_importance}])]
                , ignore_index=True)

        return importance_sum

    # Function to create radar plot for individual and combined values
    def create_extended_radar_plot(values, labels, title):
        values = list(values) + [values[0]]  # Close the loop for radar chart
        angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
        plt.title(title, pad=20)
        plt.show()

    def extract_compartment_channel(feature_name):
        # Identify compartment as the first part before an underscore
        compartment = feature_name.split('_')[0]
        
        if compartment == 'cells':
            compartment = 'cell'

        # Identify channels based on substring presence
        channels = []
        if 'channel_0' in feature_name:
            channels.append('channel_0')
        if 'channel_1' in feature_name:
            channels.append('channel_1')
        if 'channel_2' in feature_name:
            channels.append('channel_2')
        if 'channel_3' in feature_name:
            channels.append('channel_3')

        # If multiple channels are found, join them with a '+'
        if channels:
            channel = ' + '.join(channels)
        else:
            channel = 'morphology'  # Use 'morphology' if no channel identifier is found

        return (compartment, channel)

    def read_and_preprocess_data(settings):

        df, _ = _read_and_merge_data(
            locs=[settings['src']+'/measurements/measurements.db'], 
            tables=settings['tables'], 
            verbose=True, 
            nuclei_limit=settings['nuclei_limit'], 
            pathogen_limit=settings['pathogen_limit']
        )
                
        df, _dict = generate_comparison_columns(df, compartments=['cell', 'nucleus', 'pathogen', 'cytoplasm'])
        print(f"Expanded dataframe to {len(df.columns)} columns with relative features")
        scores_df = pd.read_csv(settings['scores'])

        # Clean and align columns for merging
        df['object_label'] = df['object_label'].str.replace('o', '')

        if 'row_name' not in scores_df.columns:
            scores_df['row_name'] = scores_df['row']

        if 'column_name' not in scores_df.columns:
            scores_df['column_name'] = scores_df['col']

        if 'object_label' not in scores_df.columns:
            scores_df['object_label'] = scores_df['object']

        # Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type
        df['object_label'] = df['object_label'].str.replace('o', '').astype(str)

        # Ensure 'object_label' in scores_df is also a string
        scores_df['object_label'] = scores_df['object'].astype(str)

        # Ensure all join columns have the same data type in both DataFrames
        df[['plate', 'row_name', 'column_name', 'field', 'object_label']] = df[['plate', 'row_name', 'column_name', 'field', 'object_label']].astype(str)
        scores_df[['plate', 'row_name', 'column_name', 'field', 'object_label']] = scores_df[['plate', 'row_name', 'column_name', 'field', 'object_label']].astype(str)

        # Select only the necessary columns from scores_df for merging
        scores_df = scores_df[['plate', 'row_name', 'column_name', 'field', 'object_label', settings['score_column']]]

        # Now merge DataFrames
        merged_df = pd.merge(df, scores_df, on=['plate', 'row_name', 'column_name', 'field', 'object_label'], how='inner')

        # Separate numerical features and the score column
        X = merged_df.select_dtypes(include='number').drop(columns=[settings['score_column']])
        y = merged_df[settings['score_column']]

        return X, y, merged_df
    
    X, y, merged_df = read_and_preprocess_data(settings)
    
    output = {}
    
    # Step 1: Feature Importance using Random Forest
    if settings['feature_importance'] or settings['feature_importance']:
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X, y)
        
        if settings['feature_importance']:
            print(f"Feature Importance ...")
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            top_feature_importance_df = feature_importance_df.head(settings['top_features'])

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.barh(top_feature_importance_df['feature'], top_feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f"Top {settings['top_features']} Features - Feature Importance")
            plt.gca().invert_yaxis()
            plt.show()
            
        output['feature_importance'] = feature_importance_df
        fi_compartment_df = group_feature_class(feature_importance_df, feature_groups=settings['tables'], name='compartment', include_all=settings['include_all'])
        fi_channel_df = group_feature_class(feature_importance_df, feature_groups=settings['channels'], name='channel', include_all=settings['include_all'])
        
        output['feature_importance_compartment'] = fi_compartment_df
        output['feature_importance_channel'] = fi_channel_df
    
    # Step 2: Permutation Importance
    if settings['permutation_importance']:
        print(f"Permutation Importance ...")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=settings['n_jobs'])
        perm_importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean})
        perm_importance_df = perm_importance_df.sort_values(by='importance', ascending=False)
        top_perm_importance_df = perm_importance_df.head(settings['top_features'])

        # Plot Permutation Importance
        plt.figure(figsize=(10, 6))
        plt.barh(top_perm_importance_df['feature'], top_perm_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f"Top {settings['top_features']} Features - Permutation Importance")
        plt.gca().invert_yaxis()
        plt.show()
            
        output['permutation_importance'] = perm_importance_df
    
    # Step 3: SHAP Analysis
    if settings['shap']:
        print(f"SHAP Analysis ...")

        # Select top N features based on Random Forest importance and fit the model on these features only
        top_features = feature_importance_df.head(settings['top_features'])['feature']
        X_top = X[top_features]

        # Refit the model on this subset of features
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X_top, y)

        # Sample a smaller subset of rows to speed up SHAP
        if settings['shap_sample']:
            sample = int(len(X_top) / 100)
            X_sample = X_top.sample(min(sample, len(X_top)), random_state=42)
        else:
            X_sample = X_top

        # Initialize SHAP explainer with the same subset of features
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample, max_evals=1500)

        # Plot SHAP summary for the selected sample and top features
        shap.summary_plot(shap_values, X_sample, max_display=settings['top_features'])

        # Convert SHAP values to a DataFrame for easier manipulation
        shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        
        # Apply the function to create MultiIndex columns with compartment and channel
        shap_df.columns = pd.MultiIndex.from_tuples(
            [extract_compartment_channel(feat) for feat in shap_df.columns], 
            names=['compartment', 'channel']
        )
        
        # Aggregate SHAP values by compartment and channel
        compartment_mean = shap_df.abs().groupby(level='compartment', axis=1).mean().mean(axis=0)
        channel_mean = shap_df.abs().groupby(level='channel', axis=1).mean().mean(axis=0)

        # Calculate combined importance for each pair of compartments and channels
        combined_compartment = {}
        for i, comp1 in enumerate(compartment_mean.index):
            for comp2 in compartment_mean.index[i+1:]:
                combined_compartment[f"{comp1} + {comp2}"] = shap_df.loc[:, (comp1, slice(None))].abs().mean().mean() + \
                                                              shap_df.loc[:, (comp2, slice(None))].abs().mean().mean()
        
        combined_channel = {}
        for i, chan1 in enumerate(channel_mean.index):
            for chan2 in channel_mean.index[i+1:]:
                combined_channel[f"{chan1} + {chan2}"] = shap_df.loc[:, (slice(None), chan1)].abs().mean().mean() + \
                                                          shap_df.loc[:, (slice(None), chan2)].abs().mean().mean()

        # Prepare values and labels for radar charts
        all_compartment_importance = list(compartment_mean.values) + list(combined_compartment.values())
        all_compartment_labels = list(compartment_mean.index) + list(combined_compartment.keys())

        all_channel_importance = list(channel_mean.values) + list(combined_channel.values())
        all_channel_labels = list(channel_mean.index) + list(combined_channel.keys())

        # Create radar plots for compartments and channels
        #create_extended_radar_plot(all_compartment_importance, all_compartment_labels, "SHAP Importance by Compartment (Individual and Combined)")
        #create_extended_radar_plot(all_channel_importance, all_channel_labels, "SHAP Importance by Channel (Individual and Combined)")
        
        output['shap'] = shap_df
        
    if settings['save']:
        dst = os.path.join(settings['src'], 'results')
        os.makedirs(dst, exist_ok=True)
        for key, df in output.items(): 
            save_path = os.path.join(dst, f"{key}.csv")
            df.to_csv(save_path)
            print(f"Saved {save_path}")
        
    return output

def _plot_proportion_stacked_bars(settings, df, group_column, bin_column, prc_column='prc', level='object'):
    # Always calculate chi-squared on raw data
    raw_counts = df.groupby([group_column, bin_column]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(raw_counts)
    print(f"Chi-squared test statistic (raw data): {chi2:.4f}")
    print(f"p-value (raw data): {p:.4e}")

    # Extract bin labels and indices for formatting the legend in the correct order
    bin_labels = df[bin_column].cat.categories if pd.api.types.is_categorical_dtype(df[bin_column]) else sorted(df[bin_column].unique())
    bin_indices = range(1, len(bin_labels) + 1)
    legend_labels = [f"{index}: {label}" for index, label in zip(bin_indices, bin_labels)]

    # Plot based on level setting
    if level == 'well':
        # Aggregate by well for mean ± SD visualization
        well_proportions = (
            df.groupby([group_column, prc_column, bin_column])
            .size()
            .groupby(level=[0, 1])
            .apply(lambda x: x / x.sum())
            .unstack(fill_value=0)
        )
        mean_proportions = well_proportions.groupby(group_column).mean()
        std_proportions = well_proportions.groupby(group_column).std()

        ax = mean_proportions.plot(
            kind='bar', stacked=True, yerr=std_proportions, capsize=5, colormap='viridis', figsize=(12, 8)
        )
        plt.title('Proportion of Volume Bins by Group (Mean ± SD across wells)')
    else:
        # Object-level plotting without aggregation
        group_counts = df.groupby([group_column, bin_column]).size()
        group_totals = group_counts.groupby(level=0).sum()
        proportions = group_counts / group_totals
        proportion_df = proportions.unstack(fill_value=0)

        ax = proportion_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
        plt.title('Proportion of Volume Bins by Group')

    plt.xlabel('Group')
    plt.ylabel('Proportion')

    # Update legend with formatted labels, maintaining correct order
    volume_unit = "px³" if settings['um_per_px'] is None else "µm³"
    plt.legend(legend_labels, title=f'Volume Range ({volume_unit})', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    fig = plt.gcf() 
    return chi2, p, dof, expected, raw_counts, fig

def analyze_endodyogeny(settings):
    
    from .utils import annotate_conditions, save_settings
    from .io import _read_and_merge_data
    from .settings import set_analyze_endodyogeny_defaults
    from .plot import plot_proportion_stacked_bars

    def _calculate_volume_bins(df, compartment='pathogen', min_area_bin=500, max_bins=None, verbose=False):
        area_column = f'{compartment}_area'
        df[f'{compartment}_volume'] = df[area_column] ** 1.5
        min_volume_bin = min_area_bin ** 1.5
        max_volume = df[f'{compartment}_volume'].max()

        # Generate bin edges as floats, and filter out any duplicate edges
        bins = [min_volume_bin * (2 ** i) for i in range(int(np.ceil(np.log2(max_volume / min_volume_bin)) + 1))]
        bins = sorted(set(bins))  # Ensure bin edges are unique

        # Create bin labels as ranges with decimal precision for float values (e.g., "500.0-1000.0")
        bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
        if verbose:
            print('Volume bins:', bins)
            print('Volume bin labels:', bin_labels)

        # Apply the bins to create a new column with the binned labels
        df[f'{compartment}_volume_bin'] = pd.cut(df[f'{compartment}_volume'], bins=bins, labels=bin_labels, right=False)
        
        # Create a bin index column (numeric version of bins)
        df['bin_index'] = pd.cut(df[f'{compartment}_volume'], bins=bins, labels=range(1, len(bins)), right=False).astype(int)

        # Adjust bin indices and labels based on max_bins
        if max_bins is not None:
            df.loc[df['bin_index'] > max_bins, 'bin_index'] = max_bins
            
            # Update bin labels to reflect capped bins
            bin_labels = bin_labels[:max_bins - 1] + [f">{bins[max_bins - 1]:.2f}"]
            df[f'{compartment}_volume_bin'] = df['bin_index'].map(
                {i + 1: label for i, label in enumerate(bin_labels)}
            )

        if verbose:
            print(df[[f'{compartment}_volume', f'{compartment}_volume_bin', 'bin_index']].head())

        return df

    settings = set_analyze_endodyogeny_defaults(settings)
    save_settings(settings, name='analyze_endodyogeny', show=True)
    output = {}

    # Process data
    if not isinstance(settings['src'], list):
        settings['src'] = [settings['src']]
    
    locs = []
    for s in settings['src']:
        loc = os.path.join(s, 'measurements/measurements.db')
        locs.append(loc)
        
    if 'png_list' not in settings['tables']:
        settings['tables'] = settings['tables'] + ['png_list']
    
    df, _ = _read_and_merge_data(
        locs, 
        tables=settings['tables'], 
        verbose=settings['verbose'], 
        nuclei_limit=settings['nuclei_limit'], 
        pathogen_limit=settings['pathogen_limit'],
        change_plate=settings['change_plate']
    )
    
    if not settings['um_per_px'] is None:
        df[f"{settings['compartment']}_area"] = df[f"{settings['compartment']}_area"] * (settings['um_per_px'] ** 2)
        settings['min_area_bin'] = settings['min_area_bin'] * (settings['um_per_px'] ** 2)
    
    df = df[df[f"{settings['compartment']}_area"] >= settings['min_area_bin']]
    
    df = annotate_conditions(
        df=df, 
        cells=settings['cell_types'], 
        cell_loc=settings['cell_plate_metadata'], 
        pathogens=settings['pathogen_types'],
        pathogen_loc=settings['pathogen_plate_metadata'],
        treatments=settings['treatments'], 
        treatment_loc=settings['treatment_plate_metadata']
    )
    
    if settings['group_column'] not in df.columns:
        print(f"{settings['group_column']} not found in DataFrame, please choose from:")
        for col in df.columns:
            print(col)
    
    df = df.dropna(subset=[settings['group_column']])
    df = _calculate_volume_bins(df, settings['compartment'], settings['min_area_bin'], settings['max_bins'], settings['verbose'])
    output['data'] = df
    
    
    if settings['level'] == 'plate':
        prc_column = 'plate'
    else:
        prc_column = 'prc'
    
    # Perform chi-squared test and plot
    results_df, pairwise_results_df, fig = plot_proportion_stacked_bars(settings, df, settings['group_column'], bin_column=f"{settings['compartment']}_volume_bin", prc_column=prc_column, level=settings['level'], cmap=settings['cmap'])
    
    # Extract bin labels and indices for formatting the legend in the correct order
    bin_labels = df[f"{settings['compartment']}_volume_bin"].cat.categories if pd.api.types.is_categorical_dtype(df[f"{settings['compartment']}_volume_bin"]) else sorted(df[f"{settings['compartment']}_volume_bin"].unique())
    bin_indices = range(1, len(bin_labels) + 1)
    legend_labels = [f"{index}: {label}" for index, label in zip(bin_indices, bin_labels)]
    
    # Update legend with formatted labels, maintaining correct order
    volume_unit = "px³" if settings['um_per_px'] is None else "µm³"
    plt.legend(legend_labels, title=f'Volume Range ({volume_unit})', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    
    output['chi_squared'] = results_df

    if settings['save']:
        # Save DataFrame to CSV
        output_dir = os.path.join(settings['src'][0], 'results', 'analyze_endodyogeny')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'chi_squared_results.csv')
        output_path_data = os.path.join(output_dir, 'data.csv')
        output_path_pairwise = os.path.join(output_dir, 'chi_squared_results.csv')
        output_path_fig = os.path.join(output_dir, 'chi_squared_results.pdf')
        fig.savefig(output_path_fig, dpi=300, bbox_inches='tight')
        results_df.to_csv(output_path, index=False)
        df.to_csv(output_path_data, index=False)
        pairwise_results_df.to_csv(output_path_pairwise, index=False)
        print(f"Chi-squared results saved to {output_path}")
        
    plt.show()     

    return output

def analyze_class_proportion(settings):
    
    from .utils import annotate_conditions, save_settings
    from .io import _read_and_merge_data
    from .settings import set_analyze_class_proportion_defaults
    from .plot import plot_plates, plot_proportion_stacked_bars
    from .sp_stats import perform_normality_tests, perform_levene_test, perform_statistical_tests, perform_posthoc_tests
    
    settings = set_analyze_class_proportion_defaults(settings)
    save_settings(settings, name='analyze_class_proportion', show=True)
    output = {}

    # Process data
    if not isinstance(settings['src'], list):
        settings['src'] = [settings['src']]
    
    locs = []
    for s in settings['src']:
        loc = os.path.join(s, 'measurements/measurements.db')
        locs.append(loc)
        
    if 'png_list' not in settings['tables']:
        settings['tables'] = settings['tables'] + ['png_list']
            
    df, _ = _read_and_merge_data(
        locs, 
        tables=settings['tables'], 
        verbose=settings['verbose'], 
        nuclei_limit=settings['nuclei_limit'], 
        pathogen_limit=settings['pathogen_limit']
    )
        
    df = annotate_conditions(
        df=df, 
        cells=settings['cell_types'], 
        cell_loc=settings['cell_plate_metadata'], 
        pathogens=settings['pathogen_types'],
        pathogen_loc=settings['pathogen_plate_metadata'],
        treatments=settings['treatments'], 
        treatment_loc=settings['treatment_plate_metadata']
    )
    
    if settings['group_column'] not in df.columns:
        print(f"{settings['group_column']} not found in DataFrame, please choose from:")
        for col in df.columns:
            print(col)
    
    df[settings['class_column']] = df[settings['class_column']].fillna(0)
    output['data'] = df
    
    # Perform chi-squared test and plot
    results_df, pairwise_results, fig = plot_proportion_stacked_bars(settings, df, settings['group_column'], bin_column=settings['class_column'], level=settings['level'])
    
    output['chi_squared'] = results_df
    
    if settings['save']:
        output_dir = os.path.join(settings['src'][0], 'results', 'analyze_class_proportion')
        os.makedirs(output_dir, exist_ok=True)
        output_path_chi = os.path.join(output_dir, 'class_chi_squared_results.csv')
        output_path_chi_pairwise = os.path.join(output_dir, 'class_frequency_test.csv')
        output_path_data = os.path.join(output_dir, 'class_chi_squared_data.csv')
        output_path_fig = os.path.join(output_dir, 'class_chi_squared.pdf')
        fig.savefig(output_path_fig, dpi=300, bbox_inches='tight')
        results_df.to_csv(output_path_chi, index=False)
        pairwise_results.to_csv(output_path_chi_pairwise, index=False)
        df.to_csv(output_path_data, index=False)
        print(f"Chi-squared results saved to {output_path_chi}")
        print(f"Annotated data saved to {output_path_data}")

    plt.show()
    
    fig2 = plot_plates(df, variable=settings['class_column'], grouping='mean', min_max='allq', cmap='viridis', min_count=0, verbose=True, dst=None)
    if settings['save']:
        output_path_fig2 = os.path.join(output_dir, 'class_heatmap.pdf')
        fig2.savefig(output_path_fig2, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Perform normality, variance, and statistical tests
    is_normal, normality_results = perform_normality_tests(df, settings['group_column'], [settings['class_column']])
    variance_stat, variance_p = perform_levene_test(df, settings['group_column'], settings['class_column'])

    print(f"Levene's test statistic: {variance_stat:.4f}, p-value: {variance_p:.4e}")
    variance_results = {
        'Test Statistic': variance_stat,
        'p-value': variance_p,
        'Test Name': "Levene's Test"
    }

    test_results = perform_statistical_tests(df, settings['group_column'], [settings['class_column']])
    posthoc_results = perform_posthoc_tests(
        df, settings['group_column'], settings['class_column'], is_normal=is_normal
    )

    # Save additional results
    if settings['save']:
        pd.DataFrame(normality_results).to_csv(os.path.join(output_dir, 'normality_results.csv'), index=False)
        pd.DataFrame([variance_results]).to_csv(os.path.join(output_dir, 'variance_results.csv'), index=False)
        pd.DataFrame(test_results).to_csv(os.path.join(output_dir, 'statistical_test_results.csv'), index=False)
        pd.DataFrame(posthoc_results).to_csv(os.path.join(output_dir, 'posthoc_results.csv'), index=False)
        print("Statistical analysis results saved.")

    return output

def generate_score_heatmap(settings):
    
    def group_cv_score(csv, plate=1, column='c3', data_column='pred'):
        
        df = pd.read_csv(csv)
        if 'col' in df.columns:
            df = df[df['col']==column]
        elif 'column' in df.columns:
            df['col'] = df['column']
            df = df[df['col']==column]
        if not plate is None:
            df['plate'] = f"plate{plate}"
        grouped_df = df.groupby(['plate', 'row', 'col'])[data_column].mean().reset_index()
        grouped_df['prc'] = grouped_df['plate'].astype(str) + '_' + grouped_df['row'].astype(str) + '_' + grouped_df['col'].astype(str)
        return grouped_df

    def calculate_fraction_mixed_condition(csv, plate=1, column='c3', control_sgrnas = ['TGGT1_220950_1', 'TGGT1_233460_4']):
        df = pd.read_csv(csv)  
        df = df[df['column_name']==column]
        if plate not in df.columns:
            df['plate'] = f"plate{plate}"
        df = df[df['grna_name'].str.match(f'^{control_sgrnas[0]}$|^{control_sgrnas[1]}$')]
        grouped_df = df.groupby(['plate', 'row_name', 'column_name'])['count'].sum().reset_index()
        grouped_df = grouped_df.rename(columns={'count': 'total_count'})
        merged_df = pd.merge(df, grouped_df, on=['plate', 'row_name', 'column_name'])
        merged_df['fraction'] = merged_df['count'] / merged_df['total_count']
        merged_df['prc'] = merged_df['plate'].astype(str) + '_' + merged_df['row_name'].astype(str) + '_' + merged_df['column_name'].astype(str)
        return merged_df

    def plot_multi_channel_heatmap(df, column='c3', cmap='coolwarm'):
        """
        Plot a heatmap with multiple channels as columns.

        Parameters:
        - df: DataFrame with scores for different channels.
        - column: Column to filter by (default is 'c3').
        """
        # Extract row number and convert to integer for sorting
        df['row_num'] = df['row'].str.extract(r'(\d+)').astype(int)

        # Filter and sort by plate, row, and column
        df = df[df['col'] == column]
        df = df.sort_values(by=['plate', 'row_num', 'col'])

        # Drop temporary 'row_num' column after sorting
        df = df.drop('row_num', axis=1)

        # Create a new column combining plate, row, and column for the index
        df['plate_row_col'] = df['plate'] + '-' + df['row'] + '-' + df['col']

        # Set 'plate_row_col' as the index
        df.set_index('plate_row_col', inplace=True)

        # Extract only numeric data for the heatmap
        heatmap_data = df.select_dtypes(include=[float, int])

        # Plot heatmap with square boxes, no annotations, and 'viridis' colormap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            cbar=True,
            square=True,
            annot=False
        )

        plt.title("Heatmap of Prediction Scores for All Channels")
        plt.xlabel("Channels")
        plt.ylabel("Plate-Row-Column")
        plt.tight_layout()

        # Save the figure object and return it
        fig = plt.gcf()
        plt.show()

        return fig


    def combine_classification_scores(folders, csv_name, data_column, plate=1, column='c3'):
        # Ensure `folders` is a list
        if isinstance(folders, str):
            folders = [folders]

        ls = []  # Initialize ls to store found CSV file paths

        # Iterate over the provided folders
        for folder in folders:
            sub_folders = os.listdir(folder)  # Get sub-folder list
            for sub_folder in sub_folders:  # Iterate through sub-folders
                path = os.path.join(folder, sub_folder)  # Join the full path

                if os.path.isdir(path):  # Check if it’s a directory
                    csv = os.path.join(path, csv_name)  # Join path to the CSV file
                    if os.path.exists(csv):  # If CSV exists, add to list
                        ls.append(csv)
                    else:
                        print(f'No such file: {csv}')

        # Initialize combined DataFrame
        combined_df = None
        print(f'Found {len(ls)} CSV files')

        # Loop through all collected CSV files and process them
        for csv_file in ls:
            df = pd.read_csv(csv_file)  # Read CSV into DataFrame
            df = df[df['col']==column]
            if not plate is None:
                df['plate'] = f"plate{plate}"
            # Group the data by 'plate', 'row', and 'col'
            grouped_df = df.groupby(['plate', 'row', 'col'])[data_column].mean().reset_index()
            # Use the CSV filename to create a new column name
            folder_name = os.path.dirname(csv_file).replace(".csv", "")
            new_column_name = os.path.basename(f"{folder_name}_{data_column}")
            print(new_column_name)
            grouped_df = grouped_df.rename(columns={data_column: new_column_name})

            # Merge into the combined DataFrame
            if combined_df is None:
                combined_df = grouped_df
            else:
                combined_df = pd.merge(combined_df, grouped_df, on=['plate', 'row', 'col'], how='outer')
        combined_df['prc'] = combined_df['plate'].astype(str) + '_' + combined_df['row'].astype(str) + '_' + combined_df['col'].astype(str)
        return combined_df
    
    def calculate_mae(df):
        """
        Calculate the MAE between each channel's predictions and the fraction column for all rows.
        """
        # Extract numeric columns excluding 'fraction' and 'prc'
        channels = df.drop(columns=['fraction', 'prc']).select_dtypes(include=[float, int])

        mae_data = []

        # Compute MAE for each channel with 'fraction' for all rows
        for column in channels.columns:
            for index, row in df.iterrows():
                mae = mean_absolute_error([row['fraction']], [row[column]])
                mae_data.append({'Channel': column, 'MAE': mae, 'Row': row['prc']})

        # Convert the list of dictionaries to a DataFrame
        mae_df = pd.DataFrame(mae_data)
        return mae_df

    result_df = combine_classification_scores(settings['folders'], settings['csv_name'], settings['data_column'], settings['plate'], settings['column'], )
    df = calculate_fraction_mixed_condition(settings['csv'], settings['plate'], settings['column'], settings['control_sgrnas'])
    df = df[df['grna_name']==settings['fraction_grna']]
    fraction_df = df[['fraction', 'prc']]
    merged_df = pd.merge(fraction_df, result_df, on=['prc'])
    cv_df = group_cv_score(settings['cv_csv'], settings['plate'], settings['column'], settings['data_column_cv'])
    cv_df = cv_df[[settings['data_column_cv'], 'prc']]
    merged_df = pd.merge(merged_df, cv_df, on=['prc'])
    
    fig = plot_multi_channel_heatmap(merged_df, settings['column'], settings['cmap'])
    if 'row_number' in merged_df.columns:
        merged_df = merged_df.drop('row_num', axis=1)
    mae_df = calculate_mae(merged_df)
    if 'row_number' in mae_df.columns:
        mae_df = mae_df.drop('row_num', axis=1)
        
    if not settings['dst'] is None:
        mae_dst = os.path.join(settings['dst'], f"mae_scores_comparison_plate_{settings['plate']}.csv")
        merged_dst = os.path.join(settings['dst'], f"scores_comparison_plate_{settings['plate']}_data.csv")
        heatmap_save = os.path.join(settings['dst'], f"scores_comparison_plate_{settings['plate']}.pdf")
        mae_df.to_csv(mae_dst, index=False)
        merged_df.to_csv(merged_dst, index=False)
        fig.savefig(heatmap_save, format='pdf', dpi=600, bbox_inches='tight')
    return merged_df

def post_regression_analysis(csv_file, grna_dict, grna_list, save=False):
    
    def _analyze_and_visualize_grna_correlation(df, grna_list, save_folder, save=False):
        """
        Analyze and visualize the correlation matrix of gRNAs based on their fractions and overlap.

        Parameters:
        df (pd.DataFrame): DataFrame with columns ['grna', 'fraction', 'prc'].
        grna_list (list): List of gRNAs to include in the correlation analysis.
        save_folder (str): Path to the folder where figures and data will be saved.

        Returns:
        pd.DataFrame: Correlation matrix of the gRNAs.
        """
        # Filter the DataFrame to include only rows with gRNAs in the list
        filtered_df = df[df['grna'].isin(grna_list)]

        # Pivot the data to create a prc-by-gRNA matrix, using fractions as values
        pivot_df = filtered_df.pivot_table(index='prc', columns='grna', values='fraction', aggfunc='sum').fillna(0)

        # Compute the correlation matrix
        correlation_matrix = pivot_df.corr()
        
        if save:
            # Save the correlation matrix
            correlation_matrix.to_csv(os.path.join(save_folder, 'correlation_matrix.csv'))
        
        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
        plt.title('gRNA Correlation Matrix')
        plt.xlabel('gRNAs')
        plt.ylabel('gRNAs')
        plt.tight_layout()
        
        if save:
            correlation_fig_path = os.path.join(save_folder, 'correlation_matrix_heatmap.pdf')
            plt.savefig(correlation_fig_path, dpi=300)
        
        plt.show()

        return correlation_matrix

    def _compute_effect_sizes(correlation_matrix, grna_dict, save_folder, save=False):
        """
        Compute and visualize the effect sizes of gRNAs given fixed effect sizes for a subset of gRNAs.

        Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix of gRNAs.
        grna_dict (dict): Dictionary of gRNAs with fixed effect sizes {grna_name: effect_size}.
        save_folder (str): Path to the folder where figures and data will be saved.

        Returns:
        pd.Series: Effect sizes of all gRNAs.
        """
        # Ensure the matrix is symmetric and normalize values to 0-1
        corr_matrix = correlation_matrix.copy()
        corr_matrix = (corr_matrix - corr_matrix.min().min()) / (corr_matrix.max().max() - corr_matrix.min().min())

        # Initialize the effect sizes with dtype float
        effect_sizes = pd.Series(0.0, index=corr_matrix.index)

        # Set the effect sizes for the specified gRNAs
        for grna, size in grna_dict.items():
            effect_sizes[grna] = size

        # Propagate the effect sizes
        for grna in corr_matrix.index:
            if grna not in grna_dict:
                # Weighted sum of correlations with the fixed gRNAs
                effect_sizes[grna] = np.dot(corr_matrix.loc[grna], effect_sizes) / np.sum(corr_matrix.loc[grna])
        
        if save:
            # Save the effect sizes
            effect_sizes.to_csv(os.path.join(save_folder, 'effect_sizes.csv'))

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=effect_sizes.index, y=effect_sizes.values, palette="viridis", hue=None, legend=False)

        #for i, val in enumerate(effect_sizes.values):
        #    plt.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
        plt.title("Effect Sizes of gRNAs")
        plt.xlabel("gRNAs")
        plt.ylabel("Effect Size")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            effect_sizes_fig_path = os.path.join(save_folder, 'effect_sizes_barplot.pdf')
            plt.savefig(effect_sizes_fig_path, dpi=300)
        
        plt.show()

        return effect_sizes
    
    # Ensure the save folder exists
    save_folder = os.path.join(os.path.dirname(csv_file), 'post_regression_analysis_results')
    os.makedirs(save_folder, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Perform analysis
    correlation_matrix = _analyze_and_visualize_grna_correlation(df, grna_list, save_folder, save)
    effect_sizes = _compute_effect_sizes(correlation_matrix, grna_dict, save_folder, save)
