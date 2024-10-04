import seaborn as sns
import os, random, sqlite3
import pandas as pd
import numpy as np
import cellpose
from skimage.measure import regionprops, label
from cellpose import models as cp_models
from cellpose import train as train_cp
from IPython.display import display

def analyze_recruitment(settings={}):
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
    
    df, _ = _read_and_merge_data(db_loc=[settings['src']+'/measurements/measurements.db'], 
                                 tables=['cell', 'nucleus', 'pathogen','cytoplasm'], 
                                 verbose=True, 
                                 nuclei_limit=settings['nuclei_limit'], 
                                 pathogen_limit=settings['pathogen_limit'], 
                                 uninfected=settings['uninfected'])
    
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
        if not settings['target_intensity_min'] is None:
            df = df[df[f"cell_channel_{settings['channel_of_interest']}_percentile_95'] > settings['target_intensity_min"]]
            print(f"After channel {settings['channel_of_interest']} filtration", len(df))
    if not settings['nucleus_chann_dim'] is None:
        df = _object_filter(df, 'nucleus', settings['nucleus_size_range'], settings['nucleus_intensity_range'], mask_chans, 1)
    if not settings['pathogen_chann_dim'] is None:
        df = _object_filter(df, 'pathogen', settings['pathogen_size_range'], settings['pathogen_intensity_range'], mask_chans, 2)
       
    df['recruitment'] = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity']/df[f'cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]
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
    _plot_recruitment(df, 'by PV', settings['channel_of_interest'], settings['target'], settings['figuresize'])
    print(f'well level: {len(df_well)} rows')
    _plot_recruitment(df_well, 'by well', settings['channel_of_interest'], settings['target'], settings['figuresize'])
    cells,wells = _results_to_csv(settings['src'], df, df_well)

    return [cells,wells]

def analyze_plaques(folder):
    summary_data = []
    details_data = []
    stats_data = []
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            # Assuming each file is a NumPy array file (.npy) containing a 16-bit labeled image
            #image = np.load(filepath)
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

def train_cellpose(settings):
    
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
                                                                                               settings['circular'], 
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
                                                                                                            settings['circular'], 
                                                                                                            settings['invert'], 
                                                                                                            settings['verbose'], 
                                                                                                            settings['remove_background'], 
                                                                                                            settings['background'], 
                                                                                                            settings['Signal_to_noise'], 
                                                                                                            settings['target_height'], 
                                                                                                            settings['target_width'])
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
            
    else:
        images, masks, image_names, mask_names = _load_images_and_labels(img_src, mask_src, settings['circular'], settings['invert'])
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if settings['test']:
            test_images, test_masks, test_image_names, test_mask_names = _load_images_and_labels(test_img_src, 
                                                                                                 test_mask_src, 
                                                                                                 settings['circular'], 
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
                    #rgb=False,
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

    # Count unique values in 'value' column, grouped by 'plate', 'row', 'column'
    grouped_unique_count = df.groupby(['plate', 'row', 'column'])[settings['annotation_column']].nunique(dropna=True).reset_index(name='unique_count')
    display(grouped_unique_count)

    save_path = os.path.join(settings['src'], 'phenotype_counts.csv')

    # Group by plate, row, and column, then count the occurrences of each unique value
    grouped_counts = df.groupby(['plate', 'row', 'column', 'value']).size().reset_index(name='count')

    # Pivot the DataFrame so that unique values are columns and their counts are in the rows
    pivot_df = grouped_counts.pivot_table(index=['plate', 'row', 'column'], columns='value', values='count', fill_value=0)

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
