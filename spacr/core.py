import os, sqlite3, gc, torch, time, random, shutil, cv2, tarfile, datetime, shap, string

# image and array processing
import numpy as np
import pandas as pd

from cellpose import train
import cellpose
from cellpose import models as cp_models
from cellpose.models import CellposeModel

import statsmodels.formula.api as smf
import statsmodels.api as sm
from functools import reduce
from IPython.display import display
from multiprocessing import Pool, cpu_count, Value, Lock

import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import skimage.measure as measure
from skimage.transform import resize as resizescikit
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier
from .logger import log_function_call

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

from scipy.spatial.distance import cosine, euclidean, mahalanobis, cityblock, minkowski, chebyshev, hamming, jaccard, braycurtis
from sklearn.preprocessing import StandardScaler
import shap

def analyze_plaques(folder):
    summary_data = []
    details_data = []
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            # Assuming each file is a NumPy array file (.npy) containing a 16-bit labeled image
            image = np.load(filepath)
            
            labeled_image = label(image)
            regions = regionprops(labeled_image)
            
            object_count = len(regions)
            sizes = [region.area for region in regions]
            average_size = np.mean(sizes) if sizes else 0
            
            summary_data.append({'file': filename, 'object_count': object_count, 'average_size': average_size})
            for size in sizes:
                details_data.append({'file': filename, 'plaque_size': size})
    
    # Convert lists to pandas DataFrames
    summary_df = pd.DataFrame(summary_data)
    details_df = pd.DataFrame(details_data)
    
    # Save DataFrames to a SQLite database
    db_name = 'plaques_analysis.db'
    conn = sqlite3.connect(db_name)
    
    summary_df.to_sql('summary', conn, if_exists='replace', index=False)
    details_df.to_sql('details', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"Analysis completed and saved to database '{db_name}'.")

def generate_cp_masks(settings):
    
    src = settings['src']
    model_name = settings['model_name']
    channels = settings['channels']
    diameter = settings['diameter']
    regex = '.tif'
    #flow_threshold = 30
    cellprob_threshold = settings['cellprob_threshold']
    figuresize = 25
    cmap = 'inferno'
    verbose = settings['verbose']
    plot = settings['plot']
    save = settings['save']
    custom_model = settings['custom_model']
    signal_thresholds = 1000
    normalize = settings['normalize']
    resize = settings['resize']
    target_height = settings['width_height'][1]
    target_width = settings['width_height'][0]
    rescale = settings['rescale']
    resample = settings['resample']
    net_avg = settings['net_avg']
    invert = settings['invert']
    circular = settings['circular']
    percentiles = settings['percentiles']
    overlay = settings['overlay']
    grayscale = settings['grayscale']
    flow_threshold = settings['flow_threshold']
    batch_size = settings['batch_size']
    
    dst = os.path.join(src,'masks')
    os.makedirs(dst, exist_ok=True)
	   
    identify_masks(src, dst, model_name, channels, diameter, batch_size, flow_threshold, cellprob_threshold, figuresize, cmap, verbose, plot, save, custom_model, signal_thresholds, normalize, resize, target_height, target_width, rescale, resample, net_avg, invert, circular, percentiles, overlay, grayscale)

def train_cellpose(settings):
    
    from .io import _load_normalized_images_and_labels, _load_images_and_labels
    from .utils import resize_images_and_labels

    img_src = settings['img_src'] 
    mask_src = os.path.join(img_src, 'mask')
    
    model_name = settings['model_name']
    model_type = settings['model_type']
    learning_rate = settings['learning_rate']
    weight_decay = settings['weight_decay']
    batch_size = settings['batch_size']
    n_epochs = settings['n_epochs']
    from_scratch = settings['from_scratch']
    diameter = settings['diameter']
    verbose = settings['verbose']

    channels = [0,0]
    signal_thresholds = 1000
    normalize = True
    percentiles = [2,98]
    circular = False
    invert = False
    resize = False
    settings['width_height'] = [1000,1000]
    target_height = settings['width_height'][1]
    target_width = settings['width_height'][0]
    rescale = False
    grayscale = True
    test = False

    if test:
        test_img_src = os.path.join(os.path.dirname(img_src), 'test')
        test_mask_src = os.path.join(test_img_src, 'mask')

    test_images, test_masks, test_image_names, test_mask_names = None,None,None,None,
    print(settings)

    if from_scratch:
        model_name=f'scratch_{model_name}_{model_type}_e{n_epochs}_X{target_width}_Y{target_height}.CP_model'
    else:
        if resize:
            model_name=f'{model_name}_{model_type}_e{n_epochs}_X{target_width}_Y{target_height}.CP_model'
        else:
            model_name=f'{model_name}_{model_type}_e{n_epochs}.CP_model'

    model_save_path = os.path.join(mask_src, 'models', 'cellpose_model')
    print(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(model_save_path,f'{model_name}_settings.csv')
    settings_df.to_csv(settings_csv, index=False)
    
    if from_scratch:
        model = cp_models.CellposeModel(gpu=True, model_type=model_type, diam_mean=diameter, pretrained_model=None)
    else:
        model = cp_models.CellposeModel(gpu=True, model_type=model_type)
        
    if normalize:

        image_files = [os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')]
        label_files = [os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')]
        images, masks, image_names, mask_names = _load_normalized_images_and_labels(image_files, label_files, signal_thresholds, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=verbose)
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if test:
            test_image_files = [os.path.join(test_img_src, f) for f in os.listdir(test_img_src) if f.endswith('.tif')]
            test_label_files = [os.path.join(test_mask_src, f) for f in os.listdir(test_mask_src) if f.endswith('.tif')]
            test_images, test_masks, test_image_names, test_mask_names = _load_normalized_images_and_labels(image_files=test_image_files, label_files=test_label_files, signal_thresholds=signal_thresholds, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=verbose)
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
            
        
    else:
        images, masks, image_names, mask_names = _load_images_and_labels(img_src, mask_src, circular, invert)
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
        
        if test:
            test_images, test_masks, test_image_names, test_mask_names = _load_images_and_labels(img_src=test_img_src, mask_src=test_mask_src, circular=circular, invert=circular)
            test_images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in test_images]
    
    if resize:
        images, masks = resize_images_and_labels(images, masks, target_height, target_width, show_example=True)

    if model_type == 'cyto':
        cp_channels = [0,1]
    if model_type == 'cyto2':
        cp_channels = [0,2]
    if model_type == 'nucleus':
        cp_channels = [0,0]
    if grayscale:
        cp_channels = [0,0]
        images = [np.squeeze(img) if img.ndim == 3 and 1 in img.shape else img for img in images]
    
    masks = [np.squeeze(mask) if mask.ndim == 3 and 1 in mask.shape else mask for mask in masks]

    print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {masks[0].shape}, image type: masks[0].shape')
    save_every = int(n_epochs/10)
    if save_every < 10:
        save_every = n_epochs

    train.train_seg(model.net,
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
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    weight_decay=weight_decay,
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
                    rescale=rescale,
                    #scale_range=None,
                    #bsize=224,
                    min_train_masks=1,
                    model_name=model_name)

    return print(f"Model saved at: {model_save_path}/{model_name}")

def train_cellpose_v1(settings):
    
    from .io import _load_normalized_images_and_labels, _load_images_and_labels
    from .utils import resize_images_and_labels

    img_src = settings['img_src'] 
    
    mask_src = os.path.join(img_src, 'mask')
    
    model_name = settings['model_name']
    model_type = settings['model_type']
    learning_rate = settings['learning_rate']
    weight_decay = settings['weight_decay']
    batch_size = settings['batch_size']
    n_epochs = settings['n_epochs']
    verbose = settings['verbose']

    signal_thresholds = 100 #settings['signal_thresholds']

    channels = settings['channels']
    from_scratch = settings['from_scratch']
    diameter = settings['diameter']
    resize = settings['resize']
    rescale = settings['rescale']
    normalize = settings['normalize']
    target_height = settings['width_height'][1]
    target_width = settings['width_height'][0]
    circular = settings['circular']
    invert = settings['invert']
    percentiles = settings['percentiles']
    grayscale = settings['grayscale']

    if model_type == 'cyto':
        settings['diameter'] = 30
        diameter = settings['diameter']
        print(f'Cyto model must have diamiter 30. Diameter set the 30')

    if model_type == 'nuclei':
        settings['diameter'] = 17
        diameter = settings['diameter']
        print(f'Nuclei model must have diamiter 17. Diameter set the 17')

    print(settings)

    if from_scratch:
        model_name=f'scratch_{model_name}_{model_type}_e{n_epochs}_X{target_width}_Y{target_height}.CP_model'
    else:
        model_name=f'{model_name}_{model_type}_e{n_epochs}_X{target_width}_Y{target_height}.CP_model'

    model_save_path = os.path.join(mask_src, 'models', 'cellpose_model')
    print(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(model_save_path,f'{model_name}_settings.csv')
    settings_df.to_csv(settings_csv, index=False)
    
    if not from_scratch:
        model = cp_models.CellposeModel(gpu=True, model_type=model_type)

    else:
        model = cp_models.CellposeModel(gpu=True, model_type=model_type, pretrained_model=None)
            
    if normalize:
        image_files = [os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')]
        label_files = [os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')]

        images, masks, image_names, mask_names = _load_normalized_images_and_labels(image_files, label_files, signal_thresholds, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=verbose)
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
    else:
        images, masks, image_names, mask_names = _load_images_and_labels(img_src, mask_src, circular, invert)
        images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
    
    if resize:
        images, masks = resize_images_and_labels(images, masks, target_height, target_width, show_example=True)

    if model_type == 'cyto':
        cp_channels = [0,1]
    if model_type == 'cyto2':
        cp_channels = [0,2]
    if model_type == 'nucleus':
        cp_channels = [0,0]
    if grayscale:
        cp_channels = [0,0]
        images = [np.squeeze(img) if img.ndim == 3 and 1 in img.shape else img for img in images]
    
    masks = [np.squeeze(mask) if mask.ndim == 3 and 1 in mask.shape else mask for mask in masks]

    print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {masks[0].shape}, image type: masks[0].shape')
    save_every = int(n_epochs/10)
    if save_every < 10:
        save_every = n_epochs


    #print('cellpose image input dtype', images[0].dtype)
    #print('cellpose mask input dtype', masks[0].dtype)
    
    # Train the model
    #model.train(train_data=images, #(list of arrays (2D or 3D)) – images for training

    #model.train(train_data=images, #(list of arrays (2D or 3D)) – images for training
    #            train_labels=masks, #(list of arrays (2D or 3D)) – labels for train_data, where 0=no masks; 1,2,…=mask labels can include flows as additional images
    #            train_files=image_names, #(list of strings) – file names for images in train_data (to save flows for future runs)
    #            channels=cp_channels, #(list of ints (default, None)) – channels to use for training
    #            normalize=False, #(bool (default, True)) – normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
    #            save_path=model_save_path, #(string (default, None)) – where to save trained model, if None it is not saved
    #            save_every=save_every, #(int (default, 100)) – save network every [save_every] epochs
    #            learning_rate=learning_rate, #(float or list/np.ndarray (default, 0.2)) – learning rate for training, if list, must be same length as n_epochs
    #            n_epochs=n_epochs, #(int (default, 500)) – how many times to go through whole training set during training
    #            weight_decay=weight_decay, #(float (default, 0.00001)) –
    #            SGD=True, #(bool (default, True)) – use SGD as optimization instead of RAdam
    #            batch_size=batch_size, #(int (optional, default 8)) – number of 224x224 patches to run simultaneously on the GPU (can make smaller or bigger depending on GPU memory usage)
    #            nimg_per_epoch=None, #(int (optional, default None)) – minimum number of images to train on per epoch, with a small training set (< 8 images) it may help to set to 8
    #            rescale=rescale, #(bool (default, True)) – whether or not to rescale images to diam_mean during training, if True it assumes you will fit a size model after training or resize your images accordingly, if False it will try to train the model to be scale-invariant (works worse)
    #            min_train_masks=1, #(int (default, 5)) – minimum number of masks an image must have to use in training set
    #            model_name=model_name) #(str (default, None)) – name of network, otherwise saved with name as params + training start time 


    train.train_seg(model.net,
                    train_data=images,
                    train_labels=masks,
                    train_files=image_names,
                    train_labels_files=None,
                    train_probs=None,
                    test_data=None,
                    test_labels=None,
                    test_files=None,
                    test_labels_files=None, 
                    test_probs=None,
                    load_files=True,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    weight_decay=weight_decay,
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
                    rescale=rescale,
                    #scale_range=None,
                    #bsize=224,
                    min_train_masks=1,
                    model_name=model_name)

    #model_save_path = train.train_seg(model.net,
    #                                  train_data=images,
    #                                  train_files=image_names,
    #                                  train_labels=masks,
    #                                  channels=cp_channels,
    #                                  normalize=False,
    #                                  save_every=save_every,
    #                                  learning_rate=learning_rate,
    #                                  n_epochs=n_epochs,
    #                                  #test_data=test_images,
    #                                  #test_labels=test_labels,
    #                                  weight_decay=weight_decay,
    #                                  SGD=True,
    #                                  batch_size=batch_size, 
    #                                  nimg_per_epoch=None,
    #                                  rescale=rescale,
    #                                  min_train_masks=1,
    #                                  model_name=model_name)
 

    return print(f"Model saved at: {model_save_path}/{model_name}")

def analyze_data_reg(sequencing_loc, dv_loc, agg_type = 'mean', dv_col='pred', transform=None, min_cell_count=50, min_reads=100, min_wells=2, max_wells=1000, min_frequency=0.0,remove_outlier_genes=False, refine_model=False,by_plate=False, regression_type='mlr', alpha_value=0.01, fishers=False, fisher_threshold=0.9):
    
    from .plot import _reg_v_plot
    from .utils import generate_fraction_map, MLR, fishers_odds, lasso_reg
    
    def qstring_to_float(qstr):
        number = int(qstr[1:])  # Remove the "q" and convert the rest to an integer
        return number / 100.0
    
    columns_list = ['c1', 'c2', 'c3']
    plate_list = ['p1','p3','p4']
    
    dv_df = pd.read_csv(dv_loc)#, index_col='prc')    
    
    if agg_type.startswith('q'):
        val = qstring_to_float(agg_type)
        agg_type = lambda x: x.quantile(val)
    
    # Aggregating for mean prediction, total count and count of values > 0.95
    dv_df = dv_df.groupby('prc').agg(
        pred=(dv_col, agg_type),
        count_prc=('prc', 'size'),
        mean_pathogen_area=('pathogen_area', 'mean')
    )
    
    dv_df = dv_df[dv_df['count_prc'] >= min_cell_count]
    sequencing_df = pd.read_csv(sequencing_loc)
    

    reads_df, stats_dict = process_reads(df=sequencing_df,
                                         min_reads=min_reads,
                                         min_wells=min_wells,
                                         max_wells=max_wells,
                                         gene_column='gene',
                                         remove_outliers=remove_outlier_genes)
    
    reads_df['value'] = reads_df['count']/reads_df['well_read_sum']
    reads_df['gene_grna'] = reads_df['gene']+'_'+reads_df['grna']
    
    display(reads_df)
    
    df_long = reads_df
    
    df_long = df_long[df_long['value'] > min_frequency] # removes gRNAs under a certain proportion
    #df_long = df_long[df_long['value']<1.0] # removes gRNAs in wells with only one gRNA

    # Extract gene and grna info from gene_grna column
    df_long["gene"] = df_long["grna"].str.split("_").str[1]
    df_long["grna"] = df_long["grna"].str.split("_").str[2]
    
    agg_df = df_long.groupby('prc')['count'].sum().reset_index()
    agg_df = agg_df.rename(columns={'count': 'count_sum'})
    df_long = pd.merge(df_long, agg_df, on='prc', how='left')
    df_long['value'] = df_long['count']/df_long['count_sum']
    
    merged_df = df_long.merge(dv_df, left_on='prc', right_index=True)
    merged_df = merged_df[merged_df['value'] > 0]
    merged_df['plate'] = merged_df['prc'].str.split('_').str[0]
    merged_df['row'] = merged_df['prc'].str.split('_').str[1]
    merged_df['column'] = merged_df['prc'].str.split('_').str[2]
    
    merged_df = merged_df[~merged_df['column'].isin(columns_list)]
    merged_df = merged_df[merged_df['plate'].isin(plate_list)]
    
    if transform == 'log':
        merged_df['pred'] = np.log(merged_df['pred'] + 1e-10)
    
    # Printing the unique values in 'col' and 'plate' columns
    print("Unique values in col:", merged_df['column'].unique())
    print("Unique values in plate:", merged_df['plate'].unique())
    display(merged_df)

    if fishers:
        iv_df = generate_fraction_map(df=reads_df, 
                                      gene_column='grna', 
                                      min_frequency=min_frequency)

        fishers_df = iv_df.join(dv_df, on='prc', how='inner')
        
        significant_mutants = fishers_odds(df=fishers_df, threshold=fisher_threshold, phenotyp_col='pred')
        significant_mutants = significant_mutants.sort_values(by='OddsRatio', ascending=False) 
        display(significant_mutants)
        
    if regression_type == 'mlr':
        if by_plate:
            merged_df2 = merged_df.copy()
            for plate in merged_df2['plate'].unique():
                merged_df = merged_df2[merged_df2['plate'] == plate]
                print(f'merged_df: {len(merged_df)}, plate: {plate}')
                if len(merged_df) <100:
                    break
                
                max_effects, max_effects_pvalues, model, df = MLR(merged_df, refine_model)
        else:
            
            max_effects, max_effects_pvalues, model, df = MLR(merged_df, refine_model)
        return max_effects, max_effects_pvalues, model, df
            
    if regression_type == 'ridge' or regression_type == 'lasso':
        coeffs = lasso_reg(merged_df, alpha_value=alpha_value, reg_type=regression_type)
        return coeffs
    
    if regression_type == 'mixed':
        model = smf.mixedlm("pred ~ gene_grna - 1", merged_df, groups=merged_df["plate"], re_formula="~1")
        result = model.fit(method="bfgs")
        print(result.summary())

        # Print AIC and BIC
        print("AIC:", result.aic)
        print("BIC:", result.bic)
    

        results_df = pd.DataFrame({
            'effect': result.params,
            'Standard Error': result.bse,
            'T-Value': result.tvalues,
            'p': result.pvalues
        })
        
        display(results_df)
        _reg_v_plot(df=results_df)
        
        std_resid = result.resid

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram of Residuals
        axes[0].hist(std_resid, bins=50, edgecolor='k')
        axes[0].set_xlabel('Residuals')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Histogram of Residuals')

        # Boxplot of Residuals
        axes[1].boxplot(std_resid)
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Boxplot of Residuals')

        # QQ Plot
        sm.qqplot(std_resid, line='45', ax=axes[2])
        axes[2].set_title('QQ Plot')

        # Show plots
        plt.tight_layout()
        plt.show()
        
        return result

def analyze_data_reg(sequencing_loc, dv_loc, agg_type = 'mean', min_cell_count=50, min_reads=100, min_wells=2, max_wells=1000, remove_outlier_genes=False, refine_model=False, by_plate=False, threshold=0.5, fishers=False):
    
    from .plot import _reg_v_plot
    from .utils import generate_fraction_map, fishers_odds, model_metrics
    
    def qstring_to_float(qstr):
        number = int(qstr[1:])  # Remove the "q" and convert the rest to an integer
        return number / 100.0
    
    columns_list = ['c1', 'c2', 'c3', 'c15']
    plate_list = ['p1','p2','p3','p4']
    
    dv_df = pd.read_csv(dv_loc)#, index_col='prc')    
    
    if agg_type.startswith('q'):
        val = qstring_to_float(agg_type)
        agg_type = lambda x: x.quantile(val)
    
    # Aggregating for mean prediction, total count and count of values > 0.95
    dv_df = dv_df.groupby('prc').agg(
        pred=('pred', agg_type),
        count_prc=('prc', 'size'),
        #count_above_95=('pred', lambda x: (x > 0.95).sum()),
        mean_pathogen_area=('pathogen_area', 'mean')
    )
    
    dv_df = dv_df[dv_df['count_prc'] >= min_cell_count]
    sequencing_df = pd.read_csv(sequencing_loc)

    reads_df, stats_dict = process_reads(df=sequencing_df,
                                         min_reads=min_reads,
                                         min_wells=min_wells,
                                         max_wells=max_wells,
                                         gene_column='gene',
                                         remove_outliers=remove_outlier_genes)

    iv_df = generate_fraction_map(df=reads_df, 
                                  gene_column='grna', 
                                  min_frequency=0.0)

    # Melt the iv_df to long format
    df_long = iv_df.reset_index().melt(id_vars=["prc"], 
                                       value_vars=iv_df.columns, 
                                       var_name="gene_grna", 
                                       value_name="value")

    # Extract gene and grna info from gene_grna column
    df_long["gene"] = df_long["gene_grna"].str.split("_").str[1]
    df_long["grna"] = df_long["gene_grna"].str.split("_").str[2]

    merged_df = df_long.merge(dv_df, left_on='prc', right_index=True)
    merged_df = merged_df[merged_df['value'] > 0]
    merged_df['plate'] = merged_df['prc'].str.split('_').str[0]
    merged_df['row'] = merged_df['prc'].str.split('_').str[1]
    merged_df['column'] = merged_df['prc'].str.split('_').str[2]
    
    merged_df = merged_df[~merged_df['column'].isin(columns_list)]
    merged_df = merged_df[merged_df['plate'].isin(plate_list)]
    
    # Printing the unique values in 'col' and 'plate' columns
    print("Unique values in col:", merged_df['column'].unique())
    print("Unique values in plate:", merged_df['plate'].unique())
    
    if not by_plate:
        if fishers:
            fishers_odds(df=merged_df, threshold=threshold, phenotyp_col='pred')
    
    if by_plate:
        merged_df2 = merged_df.copy()
        for plate in merged_df2['plate'].unique():
            merged_df = merged_df2[merged_df2['plate'] == plate]
            print(f'merged_df: {len(merged_df)}, plate: {plate}')
            if len(merged_df) <100:
                break
            display(merged_df)

            model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()
            #model = smf.ols("pred ~ infection_time + gene + grna + gene:grna + plate + row + column", merged_df).fit()
            
            # Display model metrics and summary
            model_metrics(model)
            #print(model.summary())

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
            
            if fishers:
                fishers_odds(df=merged_df, threshold=threshold, phenotyp_col='pred')
    else:
        display(merged_df)

        model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()

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
            model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df_filtered).fit()
            print("Number of outliers detected by standardized residuals:", len(outliers_resid))
            print("Number of outliers detected by Cook's distance:", len(outliers_cooks))

            model_metrics(model)

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
        
        if fishers:
            fishers_odds(df=merged_df, threshold=threshold, phenotyp_col='pred')

    return max_effects, max_effects_pvalues, model, df

def regression_analasys(dv_df,sequencing_loc, min_reads=75, min_wells=2, max_wells=0, model_type = 'mlr', min_cells=100, transform='logit', min_frequency=0.05, gene_column='gene', effect_size_threshold=0.25, fishers=True, clean_regression=False, VIF_threshold=10):
    
    from .utils import generate_fraction_map, fishers_odds, model_metrics, check_multicollinearity
    
    sequencing_df = pd.read_csv(sequencing_loc)
    columns_list = ['c1','c2','c3', 'c15']
    sequencing_df = sequencing_df[~sequencing_df['col'].isin(columns_list)]

    reads_df, stats_dict = process_reads(df=sequencing_df,
                                   min_reads=min_reads,
                                   min_wells=min_wells,
                                   max_wells=max_wells,
                                   gene_column='gene')
    
    display(reads_df)
    
    iv_df = generate_fraction_map(df=reads_df, 
                              gene_column=gene_column, 
                              min_frequency=min_frequency)
    
    display(iv_df)
    
    dv_df = dv_df[dv_df['count_prc']>min_cells]
    display(dv_df)
    merged_df = iv_df.join(dv_df, on='prc', how='inner')
    display(merged_df)
    fisher_df = merged_df.copy()
    
    merged_df.reset_index(inplace=True)
    merged_df[['plate', 'row', 'col']] = merged_df['prc'].str.split('_', expand=True)
    merged_df = merged_df.drop(columns=['prc'])
    merged_df.dropna(inplace=True)
    merged_df = pd.get_dummies(merged_df, columns=['plate', 'row', 'col'], drop_first=True)
    
    y = merged_df['mean_pred']
    
    if model_type == 'mlr':
        merged_df = merged_df.drop(columns=['count_prc'])
        
    elif model_type == 'wls':
        weights = merged_df['count_prc']
    
    elif model_type == 'glm':
        merged_df = merged_df.drop(columns=['count_prc'])
    
    if transform == 'logit':
    # logit transformation
        epsilon = 1e-15
        y = np.log(y + epsilon) - np.log(1 - y + epsilon)
    
    elif transform == 'log':
    # log transformation
        y = np.log10(y+1)
    
    elif transform == 'center':
    # Centering the y around 0
        y_mean = y.mean()
        y = y - y_mean
    
    x = merged_df.drop('mean_pred', axis=1)
    x = x.select_dtypes(include=[np.number])
    #x = sm.add_constant(x)
    x['const'] = 0.0

    if model_type == 'mlr':
        model = sm.OLS(y, x).fit()
        model_metrics(model)

        # Check for Multicollinearity
        vif_data = check_multicollinearity(x.drop('const', axis=1))  # assuming you've added a constant to x
        high_vif_columns = vif_data[vif_data["VIF"] > VIF_threshold]["Variable"].values  # VIF threshold of 10 is common, but this can vary based on context

        print(f"Columns with high VIF: {high_vif_columns}")
        x = x.drop(columns=high_vif_columns)  # dropping columns with high VIF

        if clean_regression:
            # 1. Filter by standardized residuals
            std_resid = model.get_influence().resid_studentized_internal
            outliers_resid = np.where(np.abs(std_resid) > 3)[0]

            # 2. Filter by leverage
            influence = model.get_influence().hat_matrix_diag
            outliers_lev = np.where(influence > 2*(x.shape[1])/len(y))[0]

            # 3. Filter by Cook's distance
            (c, p) = model.get_influence().cooks_distance
            outliers_cooks = np.where(c > 4/(len(y)-x.shape[1]-1))[0]

            # Combine all identified outliers
            outliers = reduce(np.union1d, (outliers_resid, outliers_lev, outliers_cooks))

            # Filter out outliers
            x_clean = x.drop(x.index[outliers])
            y_clean = y.drop(y.index[outliers])

            # Re-run the regression with the filtered data
            model = sm.OLS(y_clean, x_clean).fit()
            model_metrics(model)
    
    elif model_type == 'wls':
        model = sm.WLS(y, x, weights=weights).fit()
    
    elif model_type == 'glm':
        model = sm.GLM(y, x, family=sm.families.Binomial()).fit()

    print(model.summary())
    
    results_summary = model.summary()
        
    results_as_html = results_summary.tables[1].as_html()
    results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    results_df = results_df.sort_values(by='coef', ascending=False)
    
    if model_type == 'mlr':
        results_df['p'] = results_df['P>|t|']
    elif model_type == 'wls':
        results_df['p'] = results_df['P>|t|']
    elif model_type == 'glm':    
        results_df['p'] = results_df['P>|z|']
    
    results_df['type'] = 1
    results_df.loc[results_df['p'] == 0.000, 'p'] = 0.005
    results_df['-log10(p)'] = -np.log10(results_df['p'])
    
    display(results_df)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    # Plot histogram on ax1
    sns.histplot(data=y, kde=False, element="step", ax=ax1, color='teal')
    ax1.set_xlim([0, 1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Prepare data for volcano plot on ax2
    results_df['-log10(p)'] = -np.log10(results_df['p'])

    # Assuming the 'type' column is in the merged_df
    sc = ax2.scatter(results_df['coef'], results_df['-log10(p)'], c=results_df['type'], cmap='coolwarm')
    ax2.set_title('Volcano Plot')
    ax2.set_xlabel('Coefficient')
    ax2.set_ylabel('-log10(P-value)')

    # Adjust colorbar
    cbar = plt.colorbar(sc, ax=ax2, ticks=[-1, 1])
    cbar.set_label('Sign of Coefficient')
    cbar.set_ticklabels(['-ve', '+ve'])

    # Add text for specified points
    for idx, row in results_df.iterrows():
        if row['p'] < 0.05 and row['coef'] > effect_size_threshold:
            ax2.text(row['coef'], -np.log10(row['p']), idx, fontsize=8, ha='center', va='bottom', color='black')

    ax2.axhline(y=-np.log10(0.05), color='gray', linestyle='--')

    plt.show()
    
    #if model_type == 'mlr':
    #    show_residules(model)
    
    if fishers:
        threshold = 2*effect_size_threshold
        fishers_odds(df=fisher_df, threshold=threshold, phenotyp_col='mean_pred')
    
    return

def merge_pred_mes(src,
                   pred_loc,
                   target='protein of interest', 
                   cell_dim=4, 
                   nucleus_dim=5, 
                   pathogen_dim=6,
                   channel_of_interest=1,
                   pathogen_size_min=0, 
                   nucleus_size_min=0, 
                   cell_size_min=0, 
                   pathogen_min=0, 
                   nucleus_min=0, 
                   cell_min=0, 
                   target_min=0, 
                   mask_chans=[0,1,2], 
                   filter_data=False,
                   include_noninfected=False,
                   include_multiinfected=False,
                   include_multinucleated=False, 
                   cells_per_well=10, 
                   save_filtered_filelist=False,
                   verbose=False):
    
    from .io import _read_and_merge_data
    from .plot import _plot_histograms_and_stats
    
    mask_chans=[cell_dim,nucleus_dim,pathogen_dim]
    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    df, object_dfs = _read_and_merge_data(db_loc,
                                         tables,
                                         verbose=True,
                                         include_multinucleated=include_multinucleated,
                                         include_multiinfected=include_multiinfected,
                                         include_noninfected=include_noninfected)
    if filter_data:
        df = df[df['cell_area'] > cell_size_min]
        df = df[df[f'cell_channel_{mask_chans[2]}_mean_intensity'] > cell_min]
        print(f'After cell filtration {len(df)}')
        df = df[df['nucleus_area'] > nucleus_size_min]
        df = df[df[f'nucleus_channel_{mask_chans[0]}_mean_intensity'] > nucleus_min]
        print(f'After nucleus filtration {len(df)}')
        df = df[df['pathogen_area'] > pathogen_size_min]
        df=df[df[f'pathogen_channel_{mask_chans[1]}_mean_intensity'] > pathogen_min]
        print(f'After pathogen filtration {len(df)}')
        df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_min]
        print(f'After channel {channel_of_interest} filtration', len(df))

    df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    
    pred_df = annotate_results(pred_loc=pred_loc)
    
    if verbose:
        _plot_histograms_and_stats(df=pred_df)
        
    pred_df.set_index('prcfo', inplace=True)
    pred_df = pred_df.drop(columns=['plate', 'row', 'col', 'field'])

    joined_df = df.join(pred_df, how='inner')
    
    if verbose:
        _plot_histograms_and_stats(df=joined_df)
        
    #dv = joined_df.copy()
    #if 'prc' not in dv.columns:
    #dv['prc'] = dv['plate'] + '_' + dv['row'] + '_' + dv['col']
    #dv = dv[['pred']].groupby('prc').mean()
    #dv.set_index('prc', inplace=True)
    
    #loc = '/mnt/data/CellVoyager/20x/tsg101/crispr_screen/all/measurements/dv.csv'
    #dv.to_csv(loc, index=True, header=True, mode='w')

    return joined_df

def process_reads(df, min_reads, min_wells, max_wells, gene_column, remove_outliers=False):
    print('start',len(df))
    df = df[df['count'] >= min_reads]
    print('after filtering min reads',min_reads, len(df))
    reads_ls = df['count']
    stats_dict = {}
    stats_dict['screen_reads_mean'] = np.mean(reads_ls)
    stats_dict['screen_reads_sd'] = np.std(reads_ls)
    stats_dict['screen_reads_var'] = np.var(reads_ls)
    
    well_read_sum = pd.DataFrame(df.groupby(['prc']).sum())
    well_read_sum = well_read_sum.rename({'count': 'well_read_sum'}, axis=1)
    well_sgRNA_count = pd.DataFrame(df.groupby(['prc']).count()[gene_column])
    well_sgRNA_count = well_sgRNA_count.rename({gene_column: 'gRNAs_per_well'}, axis=1)
    well_seq = pd.merge(well_read_sum, well_sgRNA_count, how='inner', suffixes=('', '_right'), left_index=True, right_index=True)
    gRNA_well_count = pd.DataFrame(df.groupby([gene_column]).count()['prc'])
    gRNA_well_count = gRNA_well_count.rename({'prc': 'gRNA_well_count'}, axis=1)
    df = pd.merge(df, well_seq, on='prc', how='inner', suffixes=('', '_right'))
    df = pd.merge(df, gRNA_well_count, on=gene_column, how='inner', suffixes=('', '_right'))

    df = df[df['gRNA_well_count'] >= min_wells]
    df = df[df['gRNA_well_count'] <= max_wells]
    
    if remove_outliers:
        clf = IsolationForest(contamination='auto', random_state=42, n_jobs=20)
        #clf.fit(df.select_dtypes(include=['int', 'float']))
        clf.fit(df[["gRNA_well_count", "count"]])
        outlier_array = clf.predict(df[["gRNA_well_count", "count"]])
        #outlier_array = clf.predict(df.select_dtypes(include=['int', 'float']))
        outlier_df = pd.DataFrame(outlier_array, columns=['outlier'])
        df['outlier'] =  outlier_df['outlier']
        outliers = pd.DataFrame(df[df['outlier']==-1])
        df = pd.DataFrame(df[df['outlier']==1])
        print('removed',len(outliers), 'outliers', 'inlers',len(df))
    
    columns_to_drop = ['gRNA_well_count','gRNAs_per_well', 'well_read_sum']#, 'outlier']
    df = df.drop(columns_to_drop, axis=1)

    plates = ['p1', 'p2', 'p3', 'p4']
    df = df[df.plate.isin(plates) == True]
    print('after filtering out p5,p6,p7,p8',len(df))

    gRNA_well_count = pd.DataFrame(df.groupby([gene_column]).count()['prc'])
    gRNA_well_count = gRNA_well_count.rename({'prc': 'gRNA_well_count'}, axis=1)
    df = pd.merge(df, gRNA_well_count, on=gene_column, how='inner', suffixes=('', '_right'))
    well_read_sum = pd.DataFrame(df.groupby(['prc']).sum())
    well_read_sum = well_read_sum.rename({'count': 'well_read_sum'}, axis=1)
    well_sgRNA_count = pd.DataFrame(df.groupby(['prc']).count()[gene_column])
    well_sgRNA_count = well_sgRNA_count.rename({gene_column: 'gRNAs_per_well'}, axis=1)
    well_seq = pd.merge(well_read_sum, well_sgRNA_count, how='inner', suffixes=('', '_right'), left_index=True, right_index=True)
    df = pd.merge(df, well_seq, on='prc', how='inner', suffixes=('', '_right'))

    columns_to_drop = [col for col in df.columns if col.endswith('_right')]
    columns_to_drop2 = [col for col in df.columns if col.endswith('0')]
    columns_to_drop = columns_to_drop + columns_to_drop2
    df = df.drop(columns_to_drop, axis=1)
    return df, stats_dict

def annotate_results(pred_loc):
    
    from .utils import _map_wells_png
    
    df = pd.read_csv(pred_loc)
    df = df.copy()
    pc_col_list = ['c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24']
    pc_plate_list = ['p6','p7','p8', 'p9']
        
    nc_col_list = ['c1','c2','c3']
    nc_plate_list = ['p1','p2','p3','p4','p6','p7','p8', 'p9']
    
    screen_col_list = ['c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24']
    screen_plate_list = ['p1','p2','p3','p4']
    
    df[['plate', 'row', 'col', 'field', 'cell_id', 'prcfo']] = df['path'].apply(lambda x: pd.Series(_map_wells_png(x)))
    
    df.loc[(df['col'].isin(pc_col_list)) & (df['plate'].isin(pc_plate_list)), 'condition'] = 'pc'
    df.loc[(df['col'].isin(nc_col_list)) & (df['plate'].isin(nc_plate_list)), 'condition'] = 'nc'
    df.loc[(df['col'].isin(screen_col_list)) & (df['plate'].isin(screen_plate_list)), 'condition'] = 'screen'

    df = df.dropna(subset=['condition'])
    display(df)
    return df

def generate_dataset(src, file_metadata=None, experiment='TSG101_screen', sample=None):
    
    from .utils import initiate_counter, add_images_to_tar
    
    db_path = os.path.join(src, 'measurements', 'measurements.db')
    dst = os.path.join(src, 'datasets')
    all_paths = []

    # Connect to the database and retrieve the image paths
    print(f'Reading DataBase: {db_path}')
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            if file_metadata:
                if isinstance(file_metadata, str):
                    cursor.execute("SELECT png_path FROM png_list WHERE png_path LIKE ?", (f"%{file_metadata}%",))
            else:
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

    if isinstance(sample, int):
        selected_paths = random.sample(all_paths, sample)
        print(f'Random selection of {len(selected_paths)} paths')
    else:
        selected_paths = all_paths
        random.shuffle(selected_paths)
        print(f'All paths: {len(selected_paths)} paths')

    total_images = len(selected_paths)
    print(f'Found {total_images} images')

    # Create a temp folder in dst
    temp_dir = os.path.join(dst, "temp_tars")
    os.makedirs(temp_dir, exist_ok=True)

    # Chunking the data
    num_procs = max(2, cpu_count() - 2)
    chunk_size = len(selected_paths) // num_procs
    remainder = len(selected_paths) % num_procs

    paths_chunks = []
    start = 0
    for i in range(num_procs):
        end = start + chunk_size + (1 if i < remainder else 0)
        paths_chunks.append(selected_paths[start:end])
        start = end

    temp_tar_files = [os.path.join(temp_dir, f'temp_{i}.tar') for i in range(num_procs)]

    print(f'Generating temporary tar files in {dst}')

    # Initialize shared counter and lock
    counter = Value('i', 0)
    lock = Lock()

    with Pool(processes=num_procs, initializer=initiate_counter, initargs=(counter, lock)) as pool:
        pool.starmap(add_images_to_tar, [(paths_chunks[i], temp_tar_files[i], total_images) for i in range(num_procs)])

    # Combine the temporary tar files into a final tar
    date_name = datetime.date.today().strftime('%y%m%d')
    if not file_metadata is None:
        tar_name = f'{date_name}_{experiment}_{file_metadata}.tar'
    else:
        tar_name = f'{date_name}_{experiment}.tar'
    tar_name = os.path.join(dst, tar_name)
    if os.path.exists(tar_name):
        number = random.randint(1, 100)
        tar_name_2 = f'{date_name}_{experiment}_{file_metadata}_{number}.tar'
        print(f'Warning: {os.path.basename(tar_name)} exists, saving as {os.path.basename(tar_name_2)} ')
        tar_name = os.path.join(dst, tar_name_2)

    print(f'Merging temporary files')

    with tarfile.open(tar_name, 'w') as final_tar:
        for temp_tar_path in temp_tar_files:
            with tarfile.open(temp_tar_path, 'r') as temp_tar:
                for member in temp_tar.getmembers():
                    file_obj = temp_tar.extractfile(member)
                    final_tar.addfile(member, file_obj)
            os.remove(temp_tar_path)

    # Delete the temp folder
    shutil.rmtree(temp_dir)
    print(f"\nSaved {total_images} images to {tar_name}")

def apply_model_to_tar(tar_path, model_path, file_type='cell_png', image_size=224, batch_size=64, normalize=True, preload='images', num_workers=10, verbose=False):
    
    from .io import TarImageDataset, DataLoader
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    if verbose:
        print(f'Loading model from {model_path}')
        print(f'Loading dataset from {tar_path}')
        
    model = torch.load(model_path)
    
    dataset = TarImageDataset(tar_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0] 
    dataset_name = os.path.splitext(os.path.basename(tar_path))[0]  
    date_name = datetime.date.today().strftime('%y%m%d')
    dst = os.path.dirname(tar_path)
    result_loc = f'{dst}/{date_name}_{dataset_name}_{model_name}_result.csv'

    model.eval()
    model = model.to(device)
    
    if verbose:
        print(model)
        print(f'Generated dataset with {len(dataset)} images')
        print(f'Generating loader from {len(data_loader)} batches')
        print(f'Results wil be saved in: {result_loc}')
        print(f'Model is in eval mode')
        print(f'Model loaded to device')
        
    prediction_pos_probs = []
    filenames_list = []
    gc.collect()
    with torch.no_grad():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            images = batch_images.to(torch.float).to(device)
            outputs = model(images)
            batch_prediction_pos_prob = torch.sigmoid(outputs).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            filenames_list.extend(filenames)
            print(f'batch: {batch_idx}/{len(data_loader)}', end='\r', flush=True)

    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    df = pd.DataFrame(data, index=None)
    df.to_csv(result_loc, index=True, header=True, mode='w')
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    return df

def apply_model(src, model_path, image_size=224, batch_size=64, normalize=True, num_workers=10):
    
    from .io import NoClassDataset
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    model = torch.load(model_path)
    print(model)
    
    print(f'Loading dataset in {src} with {len(src)} images')
    dataset = NoClassDataset(data_dir=src, transform=transform, shuffle=True, load_to_memory=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f'Loaded {len(src)} images')
    
    result_loc = os.path.splitext(model_path)[0]+datetime.date.today().strftime('%y%m%d')+'_'+os.path.splitext(model_path)[1]+'_test_result.csv'
    print(f'Results wil be saved in: {result_loc}')
    
    model.eval()
    model = model.to(device)
    prediction_pos_probs = []
    filenames_list = []
    with torch.no_grad():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            images = batch_images.to(torch.float).to(device)
            outputs = model(images)
            batch_prediction_pos_prob = torch.sigmoid(outputs).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            filenames_list.extend(filenames)
            print(f'\rbatch: {batch_idx}/{len(data_loader)}', end='\r', flush=True)
    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    df = pd.DataFrame(data, index=None)
    df.to_csv(result_loc, index=True, header=True, mode='w')
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    return df


def generate_training_data_file_list(src, 
                        target='protein of interest', 
                        cell_dim=4, 
                        nucleus_dim=5, 
                        pathogen_dim=6,
                        channel_of_interest=1,
                        pathogen_size_min=0, 
                        nucleus_size_min=0, 
                        cell_size_min=0, 
                        pathogen_min=0, 
                        nucleus_min=0, 
                        cell_min=0, 
                        target_min=0, 
                        mask_chans=[0,1,2], 
                        filter_data=False,
                        include_noninfected=False,
                        include_multiinfected=False,
                        include_multinucleated=False, 
                        cells_per_well=10, 
                        save_filtered_filelist=False):
    
    from .io import _read_and_merge_data
    
    mask_dims=[cell_dim,nucleus_dim,pathogen_dim]
    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    df, object_dfs = _read_and_merge_data(db_loc,
                                         tables,
                                         verbose=True,
                                         include_multinucleated=include_multinucleated,
                                         include_multiinfected=include_multiinfected,
                                         include_noninfected=include_noninfected)

    if filter_data:
        df = df[df['cell_area'] > cell_size_min]
        df = df[df[f'cell_channel_{mask_chans[2]}_mean_intensity'] > cell_min]
        print(f'After cell filtration {len(df)}')
        df = df[df['nucleus_area'] > nucleus_size_min]
        df = df[df[f'nucleus_channel_{mask_chans[0]}_mean_intensity'] > nucleus_min]
        print(f'After nucleus filtration {len(df)}')
        df = df[df['pathogen_area'] > pathogen_size_min]
        df=df[df[f'pathogen_channel_{mask_chans[1]}_mean_intensity'] > pathogen_min]
        print(f'After pathogen filtration {len(df)}')
        df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_min]
        print(f'After channel {channel_of_interest} filtration', len(df))

    df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    return df

def training_dataset_from_annotation(db_path, dst, annotation_column='test', annotated_classes=(1, 2)):
    all_paths = []
    
    # Connect to the database and retrieve the image paths and annotations
    print(f'Reading DataBase: {db_path}')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Prepare the query with parameterized placeholders for annotated_classes
        placeholders = ','.join('?' * len(annotated_classes))
        query = f"SELECT png_path, {annotation_column} FROM png_list WHERE {annotation_column} IN ({placeholders})"
        cursor.execute(query, annotated_classes)

        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for row in rows:
                all_paths.append(row)

    # Filter paths based on annotation
    class_paths = []
    for class_ in annotated_classes:
        class_paths_temp = [path for path, annotation in all_paths if annotation == class_]
        class_paths.append(class_paths_temp)

    print(f'Generated a list of lists from annotation of {len(class_paths)} classes')
    return class_paths

def generate_dataset_from_lists(dst, class_data, classes, test_split=0.1):
    # Make sure that the length of class_data matches the length of classes
    if len(class_data) != len(classes):
        raise ValueError("class_data and classes must have the same length.")

    total_files = sum(len(data) for data in class_data)
    processed_files = 0
    
    for cls, data in zip(classes, class_data):
        # Create directories
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Split the data
        train_data, test_data = train_test_split(data, test_size=test_split, shuffle=True, random_state=42)
        
        # Copy train files
        for path in train_data:
            shutil.copy(path, os.path.join(train_class_dir, os.path.basename(path)))
            processed_files += 1
            print(f'{processed_files}/{total_files}', end='\r', flush=True)

        # Copy test files
        for path in test_data:
            shutil.copy(path, os.path.join(test_class_dir, os.path.basename(path)))
            processed_files += 1
            print(f'{processed_files}/{total_files}', end='\r', flush=True)

    # Print summary
    for cls in classes:
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        print(f'Train class {cls}: {len(os.listdir(train_class_dir))}, Test class {cls}: {len(os.listdir(test_class_dir))}')

    return

def generate_training_dataset(src, mode='annotation', annotation_column='test', annotated_classes=[1,2], classes=['nc','pc'], size=200, test_split=0.1, class_metadata=[['c1'],['c2']], metadata_type_by='col', channel_of_interest=3, custom_measurement=None, tables=None, png_type='cell_png'):
    
    from .io import _read_and_merge_data, _read_db
    from .utils import get_paths_from_db, annotate_conditions
    
    db_path = os.path.join(src, 'measurements','measurements.db')
    dst = os.path.join(src, 'datasets', 'training')

    if os.path.exists(dst):
        for i in range(1, 1000):
            dst = os.path.join(src, 'datasets', f'training_{i}')
            if not os.path.exists(dst):
                print(f'Creating new directory for training: {dst}')
                break
                
    if mode == 'annotation':
        class_paths_ls_2 = []
        class_paths_ls = training_dataset_from_annotation(db_path, dst, annotation_column, annotated_classes=annotated_classes)
        for class_paths in class_paths_ls:
            class_paths_temp = random.sample(class_paths, size)
            class_paths_ls_2.append(class_paths_temp)
        class_paths_ls = class_paths_ls_2

    elif mode == 'metadata':
        class_paths_ls = []
        class_len_ls = []
        [df] = _read_db(db_loc=db_path, tables=['png_list'])
        df['metadata_based_class'] = pd.NA
        for i, class_ in enumerate(classes):
            ls = class_metadata[i]
            df.loc[df[metadata_type_by].isin(ls), 'metadata_based_class'] = class_
            
        for class_ in classes:
            if size == None:
                c_s = []
                for c in classes:
                    c_s_t_df = df[df['metadata_based_class'] == c]
                    c_s.append(len(c_s_t_df))
                    print(f'Found {len(c_s_t_df)} images for class {c}')
                size = min(c_s)
                print(f'Using the smallest class size: {size}')

            class_temp_df = df[df['metadata_based_class'] == class_]
            class_len_ls.append(len(class_temp_df))
            print(f'Found {len(class_temp_df)} images for class {class_}')
            class_paths_temp = random.sample(class_temp_df['png_path'].tolist(), size)
            class_paths_ls.append(class_paths_temp)
    
    elif mode == 'recruitment':
        class_paths_ls = []
        if not isinstance(tables, list):
            tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
        
        df, _ = _read_and_merge_data(locs=[db_path],
                                    tables=tables,
                                    verbose=False,
                                    include_multinucleated=True,
                                    include_multiinfected=True,
                                    include_noninfected=True)
        
        print('length df 1', len(df))
        
        df = annotate_conditions(df, cells=['HeLa'], cell_loc=None, pathogens=['pathogen'], pathogen_loc=None, treatments=classes, treatment_loc=class_metadata, types = ['col','col',metadata_type_by])
        print('length df 2', len(df))
        [png_list_df] = _read_db(db_loc=db_path, tables=['png_list'])
	    
        if custom_measurement != None:
        
            if not isinstance(custom_measurement, list):
                 print(f'custom_measurement should be a list, add [ measurement_1,  measurement_2 ] or [ measurement ]')
                 return
        	
            if isinstance(custom_measurement, list):
                if len(custom_measurement) == 2:
                    print(f'Classes will be defined by the Q1 and Q3 quantiles of recruitment ({custom_measurement[0]}/{custom_measurement[1]})')
                    df['recruitment'] = df[f'{custom_measurement[0]}']/df[f'{custom_measurement[1]}']
                if len(custom_measurement) == 1:
                    print(f'Classes will be defined by the Q1 and Q3 quantiles of recruitment ({custom_measurement[0]})')
                    df['recruitment'] = df[f'{custom_measurement[0]}']
        else:
            print(f'Classes will be defined by the Q1 and Q3 quantiles of recruitment (pathogen/cytoplasm for channel {channel_of_interest})')
            df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
		
        q25 = df['recruitment'].quantile(0.25)
        q75 = df['recruitment'].quantile(0.75)
        df_lower = df[df['recruitment'] <= q25]
        df_upper = df[df['recruitment'] >= q75]
        
        class_paths_lower = get_paths_from_db(df=df_lower, png_df=png_list_df, image_type=png_type)
        
        class_paths_lower = random.sample(class_paths_lower['png_path'].tolist(), size)
        class_paths_ls.append(class_paths_lower)
        
        class_paths_upper = get_paths_from_db(df=df_upper, png_df=png_list_df, image_type=png_type)
        class_paths_upper = random.sample(class_paths_upper['png_path'].tolist(), size)
        class_paths_ls.append(class_paths_upper)
    
    generate_dataset_from_lists(dst, class_data=class_paths_ls, classes=classes, test_split=0.1)
    
    return

def generate_loaders_v1(src, train_mode='erm', mode='train', image_size=224, batch_size=32, classes=['nc','pc'], num_workers=None, validation_split=0.0, max_show=2, pin_memory=False, normalize=False, verbose=False):
    """
    Generate data loaders for training and validation/test datasets.

    Parameters:
    - src (str): The source directory containing the data.
    - train_mode (str): The training mode. Options are 'erm' (Empirical Risk Minimization) or 'irm' (Invariant Risk Minimization).
    - mode (str): The mode of operation. Options are 'train' or 'test'.
    - image_size (int): The size of the input images.
    - batch_size (int): The batch size for the data loaders.
    - classes (list): The list of classes to consider.
    - num_workers (int): The number of worker threads for data loading.
    - validation_split (float): The fraction of data to use for validation when train_mode is 'erm'.
    - max_show (int): The maximum number of images to show when verbose is True.
    - pin_memory (bool): Whether to pin memory for faster data transfer.
    - normalize (bool): Whether to normalize the input images.
    - verbose (bool): Whether to print additional information and show images.

    Returns:
    - train_loaders (list): List of data loaders for training datasets.
    - val_loaders (list): List of data loaders for validation datasets.
    - plate_names (list): List of plate names (only applicable when train_mode is 'irm').
    """
    
    from .io import MyDataset
    from .plot import _imshow
    
    plate_to_filenames = defaultdict(list)
    plate_to_labels = defaultdict(list)
    train_loaders = []
    val_loaders = []
    plate_names = []

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    if mode == 'train':
        data_dir = os.path.join(src, 'train')
        shuffle = True
        print(f'Generating Train and validation datasets')
        
    elif mode == 'test':
        data_dir = os.path.join(src, 'test')
        val_loaders = []
        validation_split=0.0
        shuffle = True
        print(f'Generating test dataset')
    
    else:
        print(f'mode:{mode} is not valid, use mode = train or test')
        return
    
    if train_mode == 'erm':
        data = MyDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
        #train_loaders = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        if validation_split > 0:
            train_size = int((1 - validation_split) * len(data))
            val_size = len(data) - train_size

            print(f'Train data:{train_size}, Validation data:{val_size}')

            train_dataset, val_dataset = random_split(data, [train_size, val_size])

            train_loaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
            val_loaders = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        else:
            train_loaders = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        
    elif train_mode == 'irm':
        data = MyDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
        
        for filename, label in zip(data.filenames, data.labels):
            plate = data.get_plate(filename)
            plate_to_filenames[plate].append(filename)
            plate_to_labels[plate].append(label)

        for plate, filenames in plate_to_filenames.items():
            labels = plate_to_labels[plate]
            plate_data = MyDataset(data_dir, classes, specific_files=filenames, specific_labels=labels, transform=transform, shuffle=False, pin_memory=pin_memory)
            plate_names.append(plate)

            if validation_split > 0:
                train_size = int((1 - validation_split) * len(plate_data))
                val_size = len(plate_data) - train_size

                print(f'Train data:{train_size}, Validation data:{val_size}')

                train_dataset, val_dataset = random_split(plate_data, [train_size, val_size])

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)

                train_loaders.append(train_loader)
                val_loaders.append(val_loader)
            else:
                train_loader = DataLoader(plate_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
                train_loaders.append(train_loader)
                val_loaders.append(None)
    
    else:
        print(f'train_mode:{train_mode} is not valid, use: train_mode = irm or erm')
        return

    if verbose:
        if train_mode == 'erm':
            for idx, (images, labels, filenames) in enumerate(train_loaders):
                if idx >= max_show:
                    break
                images = images.cpu()
                label_strings = [str(label.item()) for label in labels]
                _imshow(images, label_strings, nrow=20, fontsize=12)

        elif train_mode == 'irm':
            for plate_name, train_loader in zip(plate_names, train_loaders):
                print(f'Plate: {plate_name} with {len(train_loader.dataset)} images')
                for idx, (images, labels, filenames) in enumerate(train_loader):
                    if idx >= max_show:
                        break
                    images = images.cpu()
                    label_strings = [str(label.item()) for label in labels]
                    _imshow(images, label_strings, nrow=20, fontsize=12)
    
    return train_loaders, val_loaders, plate_names

def generate_loaders(src, train_mode='erm', mode='train', image_size=224, batch_size=32, classes=['nc','pc'], num_workers=None, validation_split=0.0, max_show=2, pin_memory=False, normalize=False, channels=[1, 2, 3], verbose=False):
    
    """
    Generate data loaders for training and validation/test datasets.

    Parameters:
    - src (str): The source directory containing the data.
    - train_mode (str): The training mode. Options are 'erm' (Empirical Risk Minimization) or 'irm' (Invariant Risk Minimization).
    - mode (str): The mode of operation. Options are 'train' or 'test'.
    - image_size (int): The size of the input images.
    - batch_size (int): The batch size for the data loaders.
    - classes (list): The list of classes to consider.
    - num_workers (int): The number of worker threads for data loading.
    - validation_split (float): The fraction of data to use for validation when train_mode is 'erm'.
    - max_show (int): The maximum number of images to show when verbose is True.
    - pin_memory (bool): Whether to pin memory for faster data transfer.
    - normalize (bool): Whether to normalize the input images.
    - verbose (bool): Whether to print additional information and show images.
    - channels (list): The list of channels to retain. Options are [1, 2, 3] for all channels, [1, 2] for blue and green, etc.

    Returns:
    - train_loaders (list): List of data loaders for training datasets.
    - val_loaders (list): List of data loaders for validation datasets.
    - plate_names (list): List of plate names (only applicable when train_mode is 'irm').
    """

    from .io import MyDataset
    from .plot import _imshow
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    from collections import defaultdict
    import os
    import random
    from PIL import Image
    from torchvision.transforms import ToTensor
    from .utils import SelectChannels

    chans = []

    if 'r' in channels:
        chans.append(1)
    if 'g' in channels:
        chans.append(2)
    if 'b' in channels:
        chans.append(3)

    channels = chans

    if verbose:
        print(f'Training a network on channels: {channels}')
        print(f'Channel 1: Red, Channel 2: Green, Channel 3: Blue')

    plate_to_filenames = defaultdict(list)
    plate_to_labels = defaultdict(list)
    train_loaders = []
    val_loaders = []
    plate_names = []

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

    if mode == 'train':
        data_dir = os.path.join(src, 'train')
        shuffle = True
        print('Generating Train and validation datasets')
    elif mode == 'test':
        data_dir = os.path.join(src, 'test')
        val_loaders = []
        validation_split = 0.0
        shuffle = True
        print('Generating test dataset')
    else:
        print(f'mode:{mode} is not valid, use mode = train or test')
        return

    if train_mode == 'erm':
        data = MyDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
        if validation_split > 0:
            train_size = int((1 - validation_split) * len(data))
            val_size = len(data) - train_size

            print(f'Train data:{train_size}, Validation data:{val_size}')

            train_dataset, val_dataset = random_split(data, [train_size, val_size])

            train_loaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
            val_loaders = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        else:
            train_loaders = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
        
    elif train_mode == 'irm':
        data = MyDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
        
        for filename, label in zip(data.filenames, data.labels):
            plate = data.get_plate(filename)
            plate_to_filenames[plate].append(filename)
            plate_to_labels[plate].append(label)

        for plate, filenames in plate_to_filenames.items():
            labels = plate_to_labels[plate]
            plate_data = MyDataset(data_dir, classes, specific_files=filenames, specific_labels=labels, transform=transform, shuffle=False, pin_memory=pin_memory)
            plate_names.append(plate)

            if validation_split > 0:
                train_size = int((1 - validation_split) * len(plate_data))
                val_size = len(plate_data) - train_size

                print(f'Train data:{train_size}, Validation data:{val_size}')

                train_dataset, val_dataset = random_split(plate_data, [train_size, val_size])

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)

                train_loaders.append(train_loader)
                val_loaders.append(val_loader)
            else:
                train_loader = DataLoader(plate_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers if num_workers is not None else 0, pin_memory=pin_memory)
                train_loaders.append(train_loader)
                val_loaders.append(None)
    
    else:
        print(f'train_mode:{train_mode} is not valid, use: train_mode = irm or erm')
        return

    if verbose:
        if train_mode == 'erm':
            for idx, (images, labels, filenames) in enumerate(train_loaders):
                if idx >= max_show:
                    break
                images = images.cpu()
                label_strings = [str(label.item()) for label in labels]
                _imshow(images, label_strings, nrow=20, fontsize=12)
        elif train_mode == 'irm':
            for plate_name, train_loader in zip(plate_names, train_loaders):
                print(f'Plate: {plate_name} with {len(train_loader.dataset)} images')
                for idx, (images, labels, filenames) in enumerate(train_loader):
                    if idx >= max_show:
                        break
                    images = images.cpu()
                    label_strings = [str(label.item()) for label in labels]
                    _imshow(images, label_strings, nrow=20, fontsize=12)
    
    return train_loaders, val_loaders, plate_names

def analyze_recruitment(src, metadata_settings, advanced_settings):
    """
    Analyze recruitment data by grouping the DataFrame by well coordinates and plotting controls and recruitment data.

    Parameters:
    src (str): The source of the recruitment data.
    metadata_settings (dict): The settings for metadata.
    advanced_settings (dict): The advanced settings for recruitment analysis.

    Returns:
    None
    """
    
    from .io import _read_and_merge_data, _results_to_csv
    from .plot import plot_merged, _plot_controls, _plot_recruitment
    from .utils import _object_filter, annotate_conditions, _calculate_recruitment, _group_by_well
    
    settings_dict = {**metadata_settings, **advanced_settings}
    settings_df = pd.DataFrame(list(settings_dict.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','analyze_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)

    # metadata settings
    target = metadata_settings['target']
    cell_types = metadata_settings['cell_types']
    cell_plate_metadata = metadata_settings['cell_plate_metadata']
    pathogen_types = metadata_settings['pathogen_types']
    pathogen_plate_metadata = metadata_settings['pathogen_plate_metadata']
    treatments = metadata_settings['treatments']
    treatment_plate_metadata = metadata_settings['treatment_plate_metadata']
    metadata_types = metadata_settings['metadata_types']
    channel_dims = metadata_settings['channel_dims']
    cell_chann_dim = metadata_settings['cell_chann_dim']
    cell_mask_dim = metadata_settings['cell_mask_dim']
    nucleus_chann_dim = metadata_settings['nucleus_chann_dim']
    nucleus_mask_dim = metadata_settings['nucleus_mask_dim']
    pathogen_chann_dim = metadata_settings['pathogen_chann_dim']
    pathogen_mask_dim = metadata_settings['pathogen_mask_dim']
    channel_of_interest = metadata_settings['channel_of_interest']
    
    # Advanced settings
    plot = advanced_settings['plot']
    plot_nr = advanced_settings['plot_nr']
    plot_control = advanced_settings['plot_control']
    figuresize = advanced_settings['figuresize']
    remove_background = advanced_settings['remove_background']
    backgrounds = advanced_settings['backgrounds']
    include_noninfected = advanced_settings['include_noninfected']
    include_multiinfected = advanced_settings['include_multiinfected']
    include_multinucleated = advanced_settings['include_multinucleated']
    cells_per_well = advanced_settings['cells_per_well']
    pathogen_size_range = advanced_settings['pathogen_size_range']
    nucleus_size_range = advanced_settings['nucleus_size_range']
    cell_size_range = advanced_settings['cell_size_range']
    pathogen_intensity_range = advanced_settings['pathogen_intensity_range']
    nucleus_intensity_range = advanced_settings['nucleus_intensity_range']
    cell_intensity_range = advanced_settings['cell_intensity_range']
    target_intensity_min = advanced_settings['target_intensity_min']
    
    print(f'Cell(s): {cell_types}, in {cell_plate_metadata}')
    print(f'Pathogen(s): {pathogen_types}, in {pathogen_plate_metadata}')
    print(f'Treatment(s): {treatments}, in {treatment_plate_metadata}')
    
    mask_dims=[cell_mask_dim,nucleus_mask_dim,pathogen_mask_dim]
    mask_chans=[nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim]

    if isinstance(metadata_types, str):
        metadata_types = [metadata_types, metadata_types, metadata_types]
    if isinstance(metadata_types, list):
        if len(metadata_types) < 3:
            metadata_types = [metadata_types[0], metadata_types[0], metadata_types[0]]
            print(f'WARNING: setting metadata types to first element times 3: {metadata_types}. To avoid this behaviour, set metadata_types to a list with 3 elements. Elements should be col row or plate.')
        else:
            metadata_types = metadata_types
    
    if isinstance(backgrounds, (int,float)):
        backgrounds = [backgrounds, backgrounds, backgrounds, backgrounds]

    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = channel_dims
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()
    
    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    df, _ = _read_and_merge_data(db_loc, 
                                         tables, 
                                         verbose=True, 
                                         include_multinucleated=include_multinucleated, 
                                         include_multiinfected=include_multiinfected, 
                                         include_noninfected=include_noninfected)
    
    df = annotate_conditions(df, 
                             cells=cell_types, 
                             cell_loc=cell_plate_metadata, 
                             pathogens=pathogen_types,
                             pathogen_loc=pathogen_plate_metadata,
                             treatments=treatments, 
                             treatment_loc=treatment_plate_metadata,
                             types=metadata_types)
    
    df = df.dropna(subset=['condition'])
    print(f'After dropping non-annotated wells: {len(df)} rows')
    files = df['file_name'].tolist()
    print(f'found: {len(files)} files')
    files = [item + '.npy' for item in files]
    random.shuffle(files)

    _max = 10**100

    if cell_size_range is None and nucleus_size_range is None and pathogen_size_range is None:
        filter_min_max = None
    else:
        if cell_size_range is None:
            cell_size_range = [0,_max]
        if nucleus_size_range is None:
            nucleus_size_range = [0,_max]
        if pathogen_size_range is None:
            pathogen_size_range = [0,_max]

        filter_min_max = [[cell_size_range[0],cell_size_range[1]],[nucleus_size_range[0],nucleus_size_range[1]],[pathogen_size_range[0],pathogen_size_range[1]]]

    if plot:
        plot_settings = {'include_noninfected':include_noninfected, 
                         'include_multiinfected':include_multiinfected,
                         'include_multinucleated':include_multinucleated,
                         'remove_background':remove_background,
                         'filter_min_max':filter_min_max,
                         'channel_dims':channel_dims,
                         'backgrounds':backgrounds,
                         'cell_mask_dim':mask_dims[0],
                         'nucleus_mask_dim':mask_dims[1],
                         'pathogen_mask_dim':mask_dims[2],
                         'overlay_chans':overlay_channels,
                         'outline_thickness':3,
                         'outline_color':'gbr',
                         'overlay_chans':overlay_channels,
                         'overlay':True,
                         'normalization_percentiles':[1,99],
                         'normalize':True,
                         'print_object_number':True,
                         'nr':plot_nr,
                         'figuresize':20,
                         'cmap':'inferno',
                         'verbose':False}
        
    if os.path.exists(os.path.join(src,'merged')):
        try:
            plot_merged(src=os.path.join(src,'merged'), settings=plot_settings)
        except Exception as e:
            print(f'Failed to plot images with outlines, Error: {e}')
        
    if not cell_chann_dim is None:
        df = _object_filter(df, object_type='cell', size_range=cell_size_range, intensity_range=cell_intensity_range, mask_chans=mask_chans, mask_chan=0)
        if not target_intensity_min is None:
            df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_intensity_min]
            print(f'After channel {channel_of_interest} filtration', len(df))
    if not nucleus_chann_dim is None:
        df = _object_filter(df, object_type='nucleus', size_range=nucleus_size_range, intensity_range=nucleus_intensity_range, mask_chans=mask_chans, mask_chan=1)
    if not pathogen_chann_dim is None:
        df = _object_filter(df, object_type='pathogen', size_range=pathogen_size_range, intensity_range=pathogen_intensity_range, mask_chans=mask_chans, mask_chan=2)
       
    df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    for chan in channel_dims:
        df = _calculate_recruitment(df, channel=chan)
    print(f'calculated recruitment for: {len(df)} rows')
    df_well = _group_by_well(df)
    print(f'found: {len(df_well)} wells')
    
    df_well = df_well[df_well['cells_per_well'] >= cells_per_well]
    prc_list = df_well['prc'].unique().tolist()
    df = df[df['prc'].isin(prc_list)]
    print(f'After cells per well filter: {len(df)} cells in {len(df_well)} wells left wth threshold {cells_per_well}')
    
    if plot_control:
        _plot_controls(df, mask_chans, channel_of_interest, figuresize=5)

    print(f'PV level: {len(df)} rows')
    _plot_recruitment(df=df, df_type='by PV', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    print(f'well level: {len(df_well)} rows')
    _plot_recruitment(df=df_well, df_type='by well', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    cells,wells = _results_to_csv(src, df, df_well)
    return [cells,wells]

def preprocess_generate_masks(src, settings={}):

    from .io import preprocess_img_data, _load_and_concatenate_arrays
    from .plot import plot_merged, plot_arrays
    from .utils import _pivot_counts_table

    settings['plot'] = False
    settings['fps'] = 2
    settings['remove_background'] = True
    settings['lower_quantile'] = 0.02
    settings['merge'] = False
    settings['normalize_plots'] = True
    settings['all_to_mip'] = False
    settings['pick_slice'] = False
    settings['skip_mode'] = src
    settings['workers'] = os.cpu_count()-4
    settings['verbose'] = True
    settings['examples_to_plot'] = 1
    settings['src'] = src
    settings['upscale'] = False
    settings['upscale_factor'] = 2.0

    settings['randomize'] = True
    settings['timelapse'] = False
    settings['timelapse_displacement'] = None
    settings['timelapse_memory'] = 3
    settings['timelapse_frame_limits'] = None
    settings['timelapse_remove_transient'] = False
    settings['timelapse_mode'] = 'trackpy'
    settings['timelapse_objects'] = ['cells']

    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','preprocess_generate_masks_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)
    
    if settings['timelapse']:
        settings['randomize'] = False
    
    if settings['preprocess']:
        if not settings['masks']:
            print(f'WARNING: channels for mask generation are defined when preprocess = True')
    
    if isinstance(settings['merge'], bool):
        settings['merge'] = [settings['merge']]*3
    if isinstance(settings['save'], bool):
        settings['save'] = [settings['save']]*3

    if settings['preprocess']:
        settings, src = preprocess_img_data(settings)
        
    if settings['masks']:
        mask_src = os.path.join(src, 'norm_channel_stack')
        if settings['cell_channel'] != None:
            generate_cellpose_masks(src=mask_src, settings=settings, object_type='cell')
            
        if settings['nucleus_channel'] != None:
            generate_cellpose_masks(src=mask_src, settings=settings, object_type='nucleus')
            
        if settings['pathogen_channel'] != None:
            generate_cellpose_masks(src=mask_src, settings=settings, object_type='pathogen')
            
        if os.path.exists(os.path.join(src,'measurements')):
            _pivot_counts_table(db_path=os.path.join(src,'measurements', 'measurements.db'))

        #Concatinate stack with masks
        _load_and_concatenate_arrays(src, settings['channels'], settings['cell_channel'], settings['nucleus_channel'], settings['pathogen_channel'])
        
        if settings['plot']:
            if not settings['timelapse']:
                plot_dims = len(settings['channels'])
                overlay_channels = [2,1,0]
                cell_mask_dim = nucleus_mask_dim = pathogen_mask_dim = None
                plot_counter = plot_dims

                if settings['cell_channel'] is not None:
                    cell_mask_dim = plot_counter
                    plot_counter += 1

                if settings['nucleus_channel'] is not None:
                    nucleus_mask_dim = plot_counter
                    plot_counter += 1

                if settings['pathogen_channel'] is not None:
                    pathogen_mask_dim = plot_counter
                    
                overlay_channels = [settings['nucleus_channel'], settings['pathogen_channel'], settings['cell_channel']]
                overlay_channels = [element for element in overlay_channels if element is not None]

                plot_settings = {'include_noninfected':True, 
                                 'include_multiinfected':True,
                                 'include_multinucleated':True,
                                 'remove_background':False,
                                 'filter_min_max':None,
                                 'channel_dims':settings['channels'],
                                 'backgrounds':[100,100,100,100],
                                 'cell_mask_dim':cell_mask_dim,
                                 'nucleus_mask_dim':nucleus_mask_dim,
                                 'pathogen_mask_dim':pathogen_mask_dim,
                                 'outline_thickness':3,
                                 'outline_color':'gbr',
                                 'overlay_chans':overlay_channels,
                                 'overlay':True,
                                 'normalization_percentiles':[1,99],
                                 'normalize':True,
                                 'print_object_number':True,
                                 'nr':settings['examples_to_plot'],
                                 'figuresize':20,
                                 'cmap':'inferno',
                                 'verbose':False}
                
                if settings['test_mode'] == True:
                    plot_settings['nr'] = len(os.path.join(src,'merged'))

                try:
                    fig = plot_merged(src=os.path.join(src,'merged'), settings=plot_settings)
                except Exception as e:
                    print(f'Failed to plot image mask overly. Error: {e}')
            else:
                plot_arrays(src=os.path.join(src,'merged'), figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99)
            
    torch.cuda.empty_cache()
    gc.collect()
    print("Successfully completed run")
    return

def identify_masks_finetune(settings):
    
    from .plot import print_mask_and_flows
    from .utils import get_files_from_dir, resize_images_and_labels
    from .io import _load_normalized_images_and_labels, _load_images_and_labels
    
    src=settings['src']
    dst=settings['dst']
    model_name=settings['model_name']
    diameter=settings['diameter']
    batch_size=settings['batch_size']
    flow_threshold=settings['flow_threshold']
    cellprob_threshold=settings['cellprob_threshold']

    verbose=settings['verbose']
    plot=settings['plot']
    save=settings['save']
    custom_model=settings['custom_model']
    overlay=settings['overlay']

    figuresize=25
    cmap='inferno'
    channels = [0,0]
    signal_thresholds = 1000
    normalize = True
    percentiles = [2,98]
    circular = False
    invert = False
    resize = False
    settings['width_height'] = [1000,1000]
    target_height = settings['width_height'][1]
    target_width = settings['width_height'][0]
    rescale = False
    resample = False
    grayscale = True
    test = False

    os.makedirs(dst, exist_ok=True)

    if not custom_model is None:
        if not os.path.exists(custom_model):
            print(f'Custom model not found: {custom_model}')
            return 

    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if custom_model == None:
        model = cp_models.CellposeModel(gpu=True, model_type=model_name, device=device)
        print(f'Loaded model: {model_name}')
    else:
        model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type=None, pretrained_model=custom_model, diam_mean=diameter, device=device)
        print("Pretrained Model Loaded:", model.pretrained_model)

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [1,0] if model_name == 'cyto' else [2, 0]
    
    if grayscale:
        chans=[0, 0]
    
    print(f'Using channels: {chans} for model of type {model_name}')
    
    if verbose == True:
        print(f'Cellpose settings: Model: {model_name}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
        
    all_image_files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.tif')]

    random.shuffle(all_image_files)
    
    time_ls = []
    for i in range(0, len(all_image_files), batch_size):
        image_files = all_image_files[i:i+batch_size]

        if normalize:
            images, _, image_names, _ = _load_normalized_images_and_labels(image_files=image_files, label_files=None, signal_thresholds=signal_thresholds, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=plot)
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[0], image.shape[1]) for image in images]
        else:
            images, _, image_names, _ = _load_images_and_labels(image_files=image_files, label_files=None, circular=circular, invert=invert) 
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
                         rescale=rescale,
                         resample=resample,
                         progress=True)

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
            average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
            print(f'Processing {file_index+1}/{len(images)} images : Time/image {average_time:.3f} sec', end='\r', flush=True)
            if plot:
                if resize:
                    stack = resizescikit(stack, dims, preserve_range=True, anti_aliasing=False).astype(stack.dtype)
                print_mask_and_flows(stack, mask, flows, overlay=overlay)
            if save:
                output_filename = os.path.join(dst, image_names[file_index])
                cv2.imwrite(output_filename, mask)
    return

def identify_masks(src, object_type, model_name, batch_size, channels, diameter, minimum_size, maximum_size, filter_intensity, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', refine_masks=True, filter_size=True, filter_dimm=True, remove_border_objects=False, verbose=False, plot=False, merge=False, save=True, start_at=0, file_type='.npz', net_avg=True, resample=True, timelapse=False, timelapse_displacement=None, timelapse_frame_limits=None, timelapse_memory=3, timelapse_remove_transient=False, timelapse_mode='btrack', timelapse_objects='cell'):
    """
    Identify masks from the source images.

    Args:
        src (str): Path to the source images.
        object_type (str): Type of object to identify.
        model_name (str): Name of the model to use for identification.
        batch_size (int): Number of images to process in each batch.
        channels (list): List of channel names.
        diameter (float): Diameter of the objects to identify.
        minimum_size (int): Minimum size of objects to keep.
        maximum_size (int): Maximum size of objects to keep.
        flow_threshold (int, optional): Threshold for flow detection. Defaults to 30.
        cellprob_threshold (int, optional): Threshold for cell probability. Defaults to 1.
        figuresize (int, optional): Size of the figure. Defaults to 25.
        cmap (str, optional): Colormap for plotting. Defaults to 'inferno'.
        refine_masks (bool, optional): Flag indicating whether to refine masks. Defaults to True.
        filter_size (bool, optional): Flag indicating whether to filter based on size. Defaults to True.
        filter_dimm (bool, optional): Flag indicating whether to filter based on intensity. Defaults to True.
        remove_border_objects (bool, optional): Flag indicating whether to remove border objects. Defaults to False.
        verbose (bool, optional): Flag indicating whether to display verbose output. Defaults to False.
        plot (bool, optional): Flag indicating whether to plot the masks. Defaults to False.
        merge (bool, optional): Flag indicating whether to merge adjacent objects. Defaults to False.
        save (bool, optional): Flag indicating whether to save the masks. Defaults to True.
        start_at (int, optional): Index to start processing from. Defaults to 0.
        file_type (str, optional): File type for saving the masks. Defaults to '.npz'.
        net_avg (bool, optional): Flag indicating whether to use network averaging. Defaults to True.
        resample (bool, optional): Flag indicating whether to resample the images. Defaults to True.
        timelapse (bool, optional): Flag indicating whether to generate a timelapse. Defaults to False.
        timelapse_displacement (float, optional): Displacement threshold for timelapse. Defaults to None.
        timelapse_frame_limits (tuple, optional): Frame limits for timelapse. Defaults to None.
        timelapse_memory (int, optional): Memory for timelapse. Defaults to 3.
        timelapse_remove_transient (bool, optional): Flag indicating whether to remove transient objects in timelapse. Defaults to False.
        timelapse_mode (str, optional): Mode for timelapse. Defaults to 'btrack'.
        timelapse_objects (str, optional): Objects to track in timelapse. Defaults to 'cell'.

    Returns:
        None
    """

    from .utils import _masks_to_masks_stack, _filter_cp_masks, _get_cellpose_batch_size
    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .timelapse import _npz_to_movie, _btrack_track_cells, _trackpy_track_cells
    from .plot import plot_masks
    
    #Note add logic that handles batches of size 1 as these will break the code batches must all be > 2 images
    gc.collect()
    
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cp_models.Cellpose(gpu=True, model_type=model_name, device=device)

    if file_type == '.npz':
        paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    else:
        paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.png')]
        if timelapse:
            print(f'timelaps is only compatible with npz files')
            return

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [2,0] if model_name == 'cyto' else [2, 0]
    
    if verbose == True:
        print(f'source: {src}')
        print()
        print(f'Settings: object_type: {object_type}, minimum_size: {minimum_size}, maximum_size:{maximum_size}, figuresize:{figuresize}, cmap:{cmap}, , net_avg:{net_avg}, resample:{resample}')
        print()
        print(f'Cellpose settings: Model: {model_name}, batch_size: {batch_size}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
        print()
        print(f'Bool Settings: verbose:{verbose}, plot:{plot}, merge:{merge}, save:{save}, start_at:{start_at}, file_type:{file_type}, timelapse:{timelapse}')
        print()
        
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    time_ls = []
    for file_index, path in enumerate(paths):

        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        if file_type == '.npz':
            if start_at: 
                print(f'starting at file index:{start_at}')
                if file_index < start_at:
                    continue
            output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
            os.makedirs(output_folder, exist_ok=True)
            overall_average_size = 0
            with np.load(path) as data:
                stack = data['data']
                filenames = data['filenames']
            if timelapse:
                if len(stack) != batch_size:
                    print(f'Changed batch_size:{batch_size} to {len(stack)}, data length:{len(stack)}')
                    batch_size = len(stack)
                    if isinstance(timelapse_frame_limits, list):
                        if len(timelapse_frame_limits) >= 2:
                            stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1], :, :, :].astype(stack.dtype)
                            filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                            batch_size = len(stack)
                            print(f'Cut batch an indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')

            for i in range(0, stack.shape[0], batch_size):
                mask_stack = []
                start = time.time()

                if stack.shape[3] == 1:
                    batch = stack[i: i+batch_size, :, :, [0,0]].astype(stack.dtype)
                else:
                    batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)
                    
                batch_filenames = filenames[i: i+batch_size].tolist()
                        
                if not plot:
                    batch, batch_filenames = _check_masks(batch, batch_filenames, output_folder)
                if batch.size == 0:
                    print(f'Processing: {file_index}/{len(paths)}: Images/N100pz {batch.shape[0]}')
                    #print(f'Processing {file_index}/{len(paths)}: Images/N100pz {batch.shape[0]}', end='\r', flush=True)
                    continue
                if batch.max() > 1:
                    batch = batch / batch.max()
                    
                if timelapse:
                    stitch_threshold=100.0
                    movie_path = os.path.join(os.path.dirname(src), 'movies')
                    os.makedirs(movie_path, exist_ok=True)
                    save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                    _npz_to_movie(batch, batch_filenames, save_path, fps=2)
                else:
                    stitch_threshold=0.0
                    
                cellpose_batch_size = _get_cellpose_batch_size()
                
                #model = cellpose.denoise.DenoiseModel(model_type=f"denoise_{model_name}", gpu=True)
                   
                masks, flows, _, _ = model.eval(x=batch,
                                                batch_size=cellpose_batch_size,
                                                normalize=False,
                                                channels=chans,
                                                channel_axis=3,
                                                diameter=diameter,
                                                flow_threshold=flow_threshold,
                                                cellprob_threshold=cellprob_threshold,
                                                rescale=None,
                                                resample=resample,
                                                stitch_threshold=stitch_threshold,
                                                progress=None)
                
                print('Masks shape',masks.shape)                
                if timelapse:
                    _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
                    if object_type in timelapse_objects:
                        if timelapse_mode == 'btrack':
                            if not timelapse_displacement is None:
                                radius = timelapse_displacement
                            else:
                                radius = 100

                            workers = os.cpu_count()-2
                            if workers < 1:
                                workers = 1

                            mask_stack = _btrack_track_cells(src, name, batch_filenames, object_type, plot, save, masks_3D=masks, mode=timelapse_mode, timelapse_remove_transient=timelapse_remove_transient, radius=radius, workers=workers)
                        if timelapse_mode == 'trackpy':
                            mask_stack = _trackpy_track_cells(src, name, batch_filenames, object_type, masks, timelapse_displacement, timelapse_memory, timelapse_remove_transient, plot, save, timelapse_mode)
                            
                    else:
                        mask_stack = _masks_to_masks_stack(masks)

                else:
                    _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                    mask_stack = _filter_cp_masks(masks, flows, filter_size, filter_intensity, minimum_size, maximum_size, remove_border_objects, merge, batch, plot, figuresize)
                    _save_object_counts_to_database(mask_stack, object_type, batch_filenames, count_loc, added_string='_after_filtration')

                if not np.any(mask_stack):
                    average_obj_size = 0
                else:
                    average_obj_size = _get_avg_object_size(mask_stack)

                average_sizes.append(average_obj_size) 
                overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
            
            stop = time.time()
            duration = (stop - start)
            time_ls.append(duration)
            average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
            time_in_min = average_time/60
            time_per_mask = average_time/batch_size
            print(f'Processing: {len(paths)}  files with {batch_size} imgs: {(file_index+1)*(batch_size+1)}/{(len(paths))*(batch_size+1)}: Time/batch {time_in_min:.3f} min: Time/mask {time_per_mask:.3f}sec: {object_type} size: {overall_average_size:.3f} px2')
            #print(f'Processing {len(paths)}  files with {batch_size} imgs: {(file_index+1)*(batch_size+1)}/{(len(paths))*(batch_size+1)}: Time/batch {time_in_min:.3f} min: Time/mask {time_per_mask:.3f}sec: {object_type} size: {overall_average_size:.3f} px2', end='\r', flush=True)
            if not timelapse:
                if plot:
                    plot_masks(batch, mask_stack, flows, figuresize=figuresize, cmap=cmap, nr=batch_size, file_type='.npz')
        if save:
            if file_type == '.npz':
                for mask_index, mask in enumerate(mask_stack):
                    output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                    np.save(output_filename, mask)
            mask_stack = []
            batch_filenames = []
        gc.collect()
    return

def all_elements_match(list1, list2):
    # Check if all elements in list1 are in list2
    return all(element in list2 for element in list1)

def generate_cellpose_masks(src, settings, object_type):
    
    from .utils import _masks_to_masks_stack, _filter_cp_masks, _get_cellpose_batch_size, _get_object_settings, _get_cellpose_channels, _choose_model, mask_object_count
    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .timelapse import _npz_to_movie, _btrack_track_cells, _trackpy_track_cells
    from .plot import plot_masks
    
    gc.collect()
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
        
    figuresize=25
    timelapse = settings['timelapse']
    
    if timelapse:
        timelapse_displacement = settings['timelapse_displacement']
        timelapse_frame_limits = settings['timelapse_frame_limits']
        timelapse_memory = settings['timelapse_memory']
        timelapse_remove_transient = settings['timelapse_remove_transient']
        timelapse_mode = settings['timelapse_mode']
        timelapse_objects = settings['timelapse_objects']
    
    batch_size = settings['batch_size']
    cellprob_threshold = settings[f'{object_type}_CP_prob']
    flow_threshold = 30
    
    object_settings = _get_object_settings(object_type, settings)
    model_name = object_settings['model_name']
    
    cellpose_channels = _get_cellpose_channels(src, settings['nucleus_channel'], settings['pathogen_channel'], settings['cell_channel'])
    if settings['verbose']:
        print(cellpose_channels)

    channels = cellpose_channels[object_type]
    cellpose_batch_size = _get_cellpose_batch_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _choose_model(model_name, device, object_type='cell', restore_type=None)
    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [2,0] if model_name == 'cyto' else [2, 0] if model_name == 'cyto3' else [2, 0]
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]    
    
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    time_ls = []
    for file_index, path in enumerate(paths):
        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
        os.makedirs(output_folder, exist_ok=True)
        overall_average_size = 0
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
        if settings['timelapse']:

            trackable_objects = ['cell','nucleus','pathogen']
            if not all_elements_match(settings['timelapse_objects'], trackable_objects):
                print(f'timelapse_objects {settings["timelapse_objects"]} must be a subset of {trackable_objects}')
                return

            if len(stack) != batch_size:
                print(f'Changed batch_size:{batch_size} to {len(stack)}, data length:{len(stack)}')
                settings['timelapse_batch_size'] = len(stack)
                batch_size = len(stack)
                if isinstance(timelapse_frame_limits, list):
                    if len(timelapse_frame_limits) >= 2:
                        stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1], :, :, :].astype(stack.dtype)
                        filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                        batch_size = len(stack)
                        print(f'Cut batch at indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')

        for i in range(0, stack.shape[0], batch_size):
            mask_stack = []
            start = time.time()

            if stack.shape[3] == 1:
                batch = stack[i: i+batch_size, :, :, [0,0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)

            batch_filenames = filenames[i: i+batch_size].tolist()

            if not settings['plot']:
                batch, batch_filenames = _check_masks(batch, batch_filenames, output_folder)
            if batch.size == 0:
                print(f'Processing {file_index}/{len(paths)}: Images/npz {batch.shape[0]}')
                continue
            if batch.max() > 1:
                batch = batch / batch.max()

            if timelapse:
                stitch_threshold=100.0
                movie_path = os.path.join(os.path.dirname(src), 'movies')
                os.makedirs(movie_path, exist_ok=True)
                save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                _npz_to_movie(batch, batch_filenames, save_path, fps=2)
            else:
                stitch_threshold=0.0

            print('batch.shape',batch.shape)
            masks, flows, _, _ = model.eval(x=batch,
                                            batch_size=cellpose_batch_size,
                                            normalize=False,
                                            channels=chans,
                                            channel_axis=3,
                                            diameter=object_settings['diameter'],
                                            flow_threshold=flow_threshold,
                                            cellprob_threshold=cellprob_threshold,
                                            rescale=None,
                                            resample=object_settings['resample'],
                                            stitch_threshold=stitch_threshold)
            
            if timelapse:
                if settings['plot']:
                    for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
                        if idx == 0:
                            num_objects = mask_object_count(mask)
                            print(f'Number of objects: {num_objects}')
                            plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)

                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
                if object_type in timelapse_objects:
                    if timelapse_mode == 'btrack':
                        if not timelapse_displacement is None:
                            radius = timelapse_displacement
                        else:
                            radius = 100

                        workers = os.cpu_count()-2
                        if workers < 1:
                            workers = 1

                        mask_stack = _btrack_track_cells(src=src,
                                                         name=name,
                                                         batch_filenames=batch_filenames,
                                                         object_type=object_type,
                                                         plot=settings['plot'],
                                                         save=settings['save'],
                                                         masks_3D=masks,
                                                         mode=timelapse_mode,
                                                         timelapse_remove_transient=timelapse_remove_transient,
                                                         radius=radius,
                                                         workers=workers)
                    if timelapse_mode == 'trackpy':
                        mask_stack = _trackpy_track_cells(src=src,
                                                          name=name,
                                                          batch_filenames=batch_filenames,
                                                          object_type=object_type,
                                                          masks=masks,
                                                          timelapse_displacement=timelapse_displacement,
                                                          timelapse_memory=timelapse_memory,
                                                          timelapse_remove_transient=timelapse_remove_transient,
                                                          plot=settings['plot'],
                                                          save=settings['save'],
                                                          mode=timelapse_mode)
                else:
                    mask_stack = _masks_to_masks_stack(masks)
            else:
                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                if object_settings['merge'] and not settings['filter']:
                    mask_stack = _filter_cp_masks(masks=masks,
                                                flows=flows,
                                                filter_size=False,
                                                filter_intensity=False,
                                                minimum_size=object_settings['minimum_size'],
                                                maximum_size=object_settings['maximum_size'],
                                                remove_border_objects=False,
                                                merge=object_settings['merge'],
                                                batch=batch,
                                                plot=settings['plot'],
                                                figuresize=figuresize)

                if settings['filter']:
                    mask_stack = _filter_cp_masks(masks=masks,
                                                flows=flows,
                                                filter_size=object_settings['filter_size'],
                                                filter_intensity=object_settings['filter_intensity'],
                                                minimum_size=object_settings['minimum_size'],
                                                maximum_size=object_settings['maximum_size'],
                                                remove_border_objects=object_settings['remove_border_objects'],
                                                merge=object_settings['merge'],
                                                batch=batch,
                                                plot=settings['plot'],
                                                figuresize=figuresize)
                    
                    _save_object_counts_to_database(mask_stack, object_type, batch_filenames, count_loc, added_string='_after_filtration')
                else:
                    mask_stack = _masks_to_masks_stack(masks)

                    if settings['plot']:
                        for idx, (mask, flow, image) in enumerate(zip(masks, flows[0], batch)):
                            if idx == 0:
                                num_objects = mask_object_count(mask)
                                print(f'Number of objects, : {num_objects}')
                                plot_masks(batch=image, masks=mask, flows=flow, cmap='inferno', figuresize=figuresize, nr=1, file_type='.npz', print_object_number=True)
        
            if not np.any(mask_stack):
                average_obj_size = 0
            else:
                average_obj_size = _get_avg_object_size(mask_stack)

            average_sizes.append(average_obj_size) 
            overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0

        stop = time.time()
        duration = (stop - start)
        time_ls.append(duration)
        average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
        time_in_min = average_time/60
        time_per_mask = average_time/batch_size
        print(f'Processing {len(paths)}  files with {batch_size} imgs: {(file_index+1)*(batch_size+1)}/{(len(paths))*(batch_size+1)}: Time/batch {time_in_min:.3f} min: Time/mask {time_per_mask:.3f}sec: {object_type} size: {overall_average_size:.3f} px2')
        if not timelapse:
            if settings['plot']:
                plot_masks(batch, mask_stack, flows, figuresize=figuresize, cmap='inferno', nr=batch_size)
        if settings['save']:
            for mask_index, mask in enumerate(mask_stack):
                output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                np.save(output_filename, mask)
            mask_stack = []
            batch_filenames = []
        gc.collect()
    torch.cuda.empty_cache()
    return

def generate_masks_from_imgs(src, model, model_name, batch_size, diameter, cellprob_threshold, grayscale, save, normalize, channels, percentiles, circular, invert, plot, resize, target_height, target_width, verbose):
    from .io import _load_images_and_labels, _load_normalized_images_and_labels
    from .utils import resize_images_and_labels, resizescikit
    from .plot import print_mask_and_flows

    dst = os.path.join(src, model_name)
    os.makedirs(dst, exist_ok=True)
    
    flow_threshold = 30
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
            images, _, image_names, _ = _load_normalized_images_and_labels(image_files=image_files, label_files=None, signal_thresholds=100, channels=channels, percentiles=percentiles,  circular=circular, invert=invert, visualize=plot)
            images = [np.squeeze(img) if img.shape[-1] == 1 else img for img in images]
            orig_dims = [(image.shape[0], image.shape[1]) for image in images]
        else:
            images, _, image_names, _ = _load_images_and_labels(image_files=image_files, label_files=None, circular=circular, invert=invert) 
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
                         progress=True)

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
            average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
            print(f'Processing {file_index+1}/{len(images)} images : Time/image {average_time:.3f} sec', end='\r', flush=True)
            if plot:
                if resize:
                    stack = resizescikit(stack, dims, preserve_range=True, anti_aliasing=False).astype(stack.dtype)
                print_mask_and_flows(stack, mask, flows, overlay=True)
            if save:
                output_filename = os.path.join(dst, image_names[file_index])
                cv2.imwrite(output_filename, mask)


def check_cellpose_models(settings):
    
    src = settings['src']
    batch_size = settings['batch_size']
    cellprob_threshold = settings['cellprob_threshold']
    save = settings['save']
    normalize = settings['normalize']
    channels = settings['channels']
    percentiles = settings['percentiles']
    circular = settings['circular']
    invert = settings['invert']
    plot = settings['plot']
    diameter = settings['diameter']
    resize = settings['resize']
    grayscale = settings['grayscale']
    verbose = settings['verbose']
    target_height = settings['width_height'][0]
    target_width = settings['width_height'][1]
    
    cellpose_models = ['cyto', 'nuclei', 'cyto2', 'cyto3']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for model_name in cellpose_models:

        model = cp_models.CellposeModel(gpu=True, model_type=model_name, device=device)
        print(f'Using {model_name}')
        generate_masks_from_imgs(src, model, model_name, batch_size, diameter, cellprob_threshold, grayscale, save, normalize, channels, percentiles, circular, invert, plot, resize, target_height, target_width, verbose)
    
    return

def compare_masks_v1(dir1, dir2, dir3, verbose=False):
    
    from .io import _read_mask
    from .plot import visualize_masks, plot_comparison_results
    from .utils import extract_boundaries, boundary_f1_score, compute_segmentation_ap, jaccard_index, dice_coefficient
    
    filenames = os.listdir(dir1)
    results = []
    cond_1 = os.path.basename(dir1)
    cond_2 = os.path.basename(dir2)
    cond_3 = os.path.basename(dir3)

    for index, filename in enumerate(filenames):
        print(f'Processing image:{index+1}', end='\r', flush=True)
        path1, path2, path3 = os.path.join(dir1, filename), os.path.join(dir2, filename), os.path.join(dir3, filename)

        print(path1)
        print(path2)
        print(path3)
        
        if os.path.exists(path2) and os.path.exists(path3):
            
            mask1, mask2, mask3 = _read_mask(path1), _read_mask(path2), _read_mask(path3)
            boundary_true1, boundary_true2, boundary_true3 = extract_boundaries(mask1), extract_boundaries(mask2), extract_boundaries(mask3)
            
            
            true_masks, pred_masks = [mask1], [mask2, mask3]  # Assuming mask1 is the ground truth for simplicity
            true_labels, pred_labels_1, pred_labels_2 = label(mask1), label(mask2), label(mask3)
            average_precision_0, average_precision_1 = compute_segmentation_ap(mask1, mask2), compute_segmentation_ap(mask1, mask3)
            ap_scores = [average_precision_0, average_precision_1]

            if verbose:
                #unique_values1, unique_values2, unique_values3 = np.unique(mask1),  np.unique(mask2), np.unique(mask3)
                #print(f"Unique values in mask 1: {unique_values1}, mask 2: {unique_values2}, mask 3: {unique_values3}")
                visualize_masks(boundary_true1, boundary_true2, boundary_true3, title=f"Boundaries - {filename}")
            
            boundary_f1_12, boundary_f1_13, boundary_f1_23 = boundary_f1_score(mask1, mask2), boundary_f1_score(mask1, mask3), boundary_f1_score(mask2, mask3)

            if (np.unique(mask1).size == 1 and np.unique(mask1)[0] == 0) and \
               (np.unique(mask2).size == 1 and np.unique(mask2)[0] == 0) and \
               (np.unique(mask3).size == 1 and np.unique(mask3)[0] == 0):
                continue
            
            if verbose:
                #unique_values4, unique_values5, unique_values6 = np.unique(boundary_f1_12), np.unique(boundary_f1_13), np.unique(boundary_f1_23)
                #print(f"Unique values in boundary mask 1: {unique_values4}, mask 2: {unique_values5}, mask 3: {unique_values6}")
                visualize_masks(mask1, mask2, mask3, title=filename)
            
            jaccard12 = jaccard_index(mask1, mask2)
            dice12 = dice_coefficient(mask1, mask2)
            
            jaccard13 = jaccard_index(mask1, mask3)
            dice13 = dice_coefficient(mask1, mask3)
            
            jaccard23 = jaccard_index(mask2, mask3)
            dice23 = dice_coefficient(mask2, mask3)    

            results.append({
                f'filename': filename,
                f'jaccard_{cond_1}_{cond_2}': jaccard12,
                f'dice_{cond_1}_{cond_2}': dice12,
                f'jaccard_{cond_1}_{cond_3}': jaccard13,
                f'dice_{cond_1}_{cond_3}': dice13,
                f'jaccard_{cond_2}_{cond_3}': jaccard23,
                f'dice_{cond_2}_{cond_3}': dice23,
                f'boundary_f1_{cond_1}_{cond_2}': boundary_f1_12,
                f'boundary_f1_{cond_1}_{cond_3}': boundary_f1_13,
                f'boundary_f1_{cond_2}_{cond_3}': boundary_f1_23,
                f'average_precision_{cond_1}_{cond_2}': ap_scores[0],
                f'average_precision_{cond_1}_{cond_3}': ap_scores[1]
            })
        else:
            print(f'Cannot find {path1} or {path2} or {path3}')
    fig = plot_comparison_results(results)
    return results, fig

def compare_cellpose_masks_v1(src, verbose=False):
    from .io import _read_mask
    from .plot import visualize_masks, plot_comparison_results, visualize_cellpose_masks
    from .utils import extract_boundaries, boundary_f1_score, compute_segmentation_ap, jaccard_index

    import os
    import numpy as np
    from skimage.measure import label

    # Collect all subdirectories in src
    dirs = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]

    dirs.sort()  # Optional: sort directories if needed

    # Get common files in all directories
    common_files = set(os.listdir(dirs[0]))
    for d in dirs[1:]:
        common_files.intersection_update(os.listdir(d))
    common_files = list(common_files)

    results = []
    conditions = [os.path.basename(d) for d in dirs]

    for index, filename in enumerate(common_files):
        print(f'Processing image {index+1}/{len(common_files)}', end='\r', flush=True)
        paths = [os.path.join(d, filename) for d in dirs]

        # Check if file exists in all directories
        if not all(os.path.exists(path) for path in paths):
            print(f'Skipping {filename} as it is not present in all directories.')
            continue

        masks = [_read_mask(path) for path in paths]
        boundaries = [extract_boundaries(mask) for mask in masks]

        if verbose:
            visualize_cellpose_masks(masks, titles=conditions, comparison_title=f"Masks Comparison for {filename}")

        # Initialize data structure for results
        file_results = {'filename': filename}

        # Compare each mask with each other
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                condition_i = conditions[i]
                condition_j = conditions[j]
                mask_i = masks[i]
                mask_j = masks[j]

                # Compute metrics
                boundary_f1 = boundary_f1_score(mask_i, mask_j)
                jaccard = jaccard_index(mask_i, mask_j)
                average_precision = compute_segmentation_ap(mask_i, mask_j)

                # Store results
                file_results[f'jaccard_{condition_i}_{condition_j}'] = jaccard
                file_results[f'boundary_f1_{condition_i}_{condition_j}'] = boundary_f1
                file_results[f'average_precision_{condition_i}_{condition_j}'] = average_precision

        results.append(file_results)

    fig = plot_comparison_results(results)
    return results, fig

def compare_mask(args):
    src, filename, dirs, conditions = args
    paths = [os.path.join(d, filename) for d in dirs]

    if not all(os.path.exists(path) for path in paths):
        return None

    from .io import _read_mask  # Import here to avoid issues in multiprocessing
    from .utils import extract_boundaries, boundary_f1_score, compute_segmentation_ap, jaccard_index
    from .plot import plot_comparison_results

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

def compare_cellpose_masks(src, verbose=False, processes=None):
    from .plot import visualize_cellpose_masks, plot_comparison_results
    from .io import _read_mask
    dirs = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    dirs.sort()  # Optional: sort directories if needed
    conditions = [os.path.basename(d) for d in dirs]

    # Get common files in all directories
    common_files = set(os.listdir(dirs[0]))
    for d in dirs[1:]:
        common_files.intersection_update(os.listdir(d))
    common_files = list(common_files)

    # Create a pool of workers
    with Pool(processes=processes) as pool:
        args = [(src, filename, dirs, conditions) for filename in common_files]
        results = pool.map(compare_mask, args)

    # Filter out None results (from skipped files)
    results = [res for res in results if res is not None]

    if verbose:
        for result in results:
            filename = result['filename']
            masks = [_read_mask(os.path.join(d, filename)) for d in dirs]
            visualize_cellpose_masks(masks, titles=conditions, comparison_title=f"Masks Comparison for {filename}")

    fig = plot_comparison_results(results)
    return results, fig


def _calculate_similarity(df, features, col_to_compare, val1, val2):
    """
    Calculate similarity scores of each well to the positive and negative controls using various metrics.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    features (list): List of feature columns to use for similarity calculation.
    col_to_compare (str): Column name to use for comparing groups.
    val1, val2 (str): Values in col_to_compare to create subsets for comparison.

    Returns:
    pandas.DataFrame: DataFrame with similarity scores.
    """
    # Separate positive and negative control wells
    pos_control = df[df[col_to_compare] == val1][features].mean()
    neg_control = df[df[col_to_compare] == val2][features].mean()
    
    # Standardize features for Mahalanobis distance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Regularize the covariance matrix to avoid singularity
    cov_matrix = np.cov(scaled_features, rowvar=False)
    inv_cov_matrix = None
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add a small value to the diagonal elements for regularization
        epsilon = 1e-5
        inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon)
    
    # Calculate similarity scores
    df['similarity_to_pos_euclidean'] = df[features].apply(lambda row: euclidean(row, pos_control), axis=1)
    df['similarity_to_neg_euclidean'] = df[features].apply(lambda row: euclidean(row, neg_control), axis=1)
    df['similarity_to_pos_cosine'] = df[features].apply(lambda row: cosine(row, pos_control), axis=1)
    df['similarity_to_neg_cosine'] = df[features].apply(lambda row: cosine(row, neg_control), axis=1)
    df['similarity_to_pos_mahalanobis'] = df[features].apply(lambda row: mahalanobis(row, pos_control, inv_cov_matrix), axis=1)
    df['similarity_to_neg_mahalanobis'] = df[features].apply(lambda row: mahalanobis(row, neg_control, inv_cov_matrix), axis=1)
    df['similarity_to_pos_manhattan'] = df[features].apply(lambda row: cityblock(row, pos_control), axis=1)
    df['similarity_to_neg_manhattan'] = df[features].apply(lambda row: cityblock(row, neg_control), axis=1)
    df['similarity_to_pos_minkowski'] = df[features].apply(lambda row: minkowski(row, pos_control, p=3), axis=1)
    df['similarity_to_neg_minkowski'] = df[features].apply(lambda row: minkowski(row, neg_control, p=3), axis=1)
    df['similarity_to_pos_chebyshev'] = df[features].apply(lambda row: chebyshev(row, pos_control), axis=1)
    df['similarity_to_neg_chebyshev'] = df[features].apply(lambda row: chebyshev(row, neg_control), axis=1)
    df['similarity_to_pos_hamming'] = df[features].apply(lambda row: hamming(row, pos_control), axis=1)
    df['similarity_to_neg_hamming'] = df[features].apply(lambda row: hamming(row, neg_control), axis=1)
    df['similarity_to_pos_jaccard'] = df[features].apply(lambda row: jaccard(row, pos_control), axis=1)
    df['similarity_to_neg_jaccard'] = df[features].apply(lambda row: jaccard(row, neg_control), axis=1)
    df['similarity_to_pos_braycurtis'] = df[features].apply(lambda row: braycurtis(row, pos_control), axis=1)
    df['similarity_to_neg_braycurtis'] = df[features].apply(lambda row: braycurtis(row, neg_control), axis=1)
    
    return df

def _permutation_importance(df, feature_string='channel_3', col_to_compare='col', pos='c1', neg='c2', exclude=None, n_repeats=10, clean=True, nr_to_plot=30, n_estimators=100, test_size=0.2, random_state=42, model_type='xgboost', n_jobs=-1):
    
    """
    Calculates permutation importance for numerical features in the dataframe,
    comparing groups based on specified column values and uses the model to predict 
    the class for all other rows in the dataframe.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    feature_string (str): String to filter features that contain this substring.
    col_to_compare (str): Column name to use for comparing groups.
    pos, neg (str): Values in col_to_compare to create subsets for comparison.
    exclude (list or str, optional): Columns to exclude from features.
    n_repeats (int): Number of repeats for permutation importance.
    clean (bool): Whether to remove columns with a single value.
    nr_to_plot (int): Number of top features to plot based on permutation importance.
    n_estimators (int): Number of trees in the random forest, gradient boosting, or XGBoost model.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    model_type (str): Type of model to use ('random_forest', 'logistic_regression', 'gradient_boosting', 'xgboost').
    n_jobs (int): Number of jobs to run in parallel for applicable models.

    Returns:
    pandas.DataFrame: The original dataframe with added prediction and data usage columns.
    pandas.DataFrame: DataFrame containing the importances and standard deviations.
    """

    if 'cells_per_well' in df.columns:
        df = df.drop(columns=['cells_per_well'])

    # Subset the dataframe based on specified column values
    df1 = df[df[col_to_compare] == pos].copy()
    df2 = df[df[col_to_compare] == neg].copy()

    # Create target variable
    df1['target'] = 0
    df2['target'] = 1

    # Combine the subsets for analysis
    combined_df = pd.concat([df1, df2])

    # Automatically select numerical features
    features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    features.remove('target')

    if clean:
        combined_df = combined_df.loc[:, combined_df.nunique() > 1]
        features = [feature for feature in features if feature in combined_df.columns]
        
    if feature_string is not None:
        feature_list = ['channel_0', 'channel_1', 'channel_2', 'channel_3']

        # Remove feature_string from the list if it exists
        if feature_string in feature_list:
            feature_list.remove(feature_string)

        features = [feature for feature in features if feature_string in feature]

        # Iterate through the list and remove columns from df
        for feature_ in feature_list:
            features = [feature for feature in features if feature_ not in feature]
            print(f'After removing {feature_} features: {len(features)}')

    if exclude:
        if isinstance(exclude, list):
            features = [feature for feature in features if feature not in exclude]
        else:
            features.remove(exclude)

    X = combined_df[features]
    y = combined_df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Label the data in the original dataframe
    combined_df['data_usage'] = 'train'
    combined_df.loc[X_test.index, 'data_usage'] = 'test'

    # Initialize the model based on model_type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'gradient_boosting':
        model = HistGradientBoostingClassifier(max_iter=n_estimators, random_state=random_state)  # Supports n_jobs internally
    elif model_type == 'xgboost':
        model = XGBClassifier(n_estimators=n_estimators, random_state=random_state, nthread=n_jobs, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)

    # Create a DataFrame for permutation importances
    permutation_df = pd.DataFrame({
        'feature': [features[i] for i in perm_importance.importances_mean.argsort()],
        'importance_mean': perm_importance.importances_mean[perm_importance.importances_mean.argsort()],
        'importance_std': perm_importance.importances_std[perm_importance.importances_mean.argsort()]
    }).tail(nr_to_plot)

    # Plotting
    fig, ax = plt.subplots()
    ax.barh(permutation_df['feature'], permutation_df['importance_mean'], xerr=permutation_df['importance_std'], color="teal", align="center", alpha=0.6)
    ax.set_xlabel('Permutation Importance')
    plt.tight_layout()
    plt.show()
    
    # Feature importance for models that support it
    if model_type in ['random_forest', 'xgboost', 'gradient_boosting']:
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False).head(nr_to_plot)
        
        # Plotting feature importance
        fig, ax = plt.subplots()
        ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color="blue", align="center", alpha=0.6)
        ax.set_xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        feature_importance_df = pd.DataFrame()

    # Predicting the target variable for the test set
    predictions_test = model.predict(X_test)
    combined_df.loc[X_test.index, 'predictions'] = predictions_test

    # Predicting the target variable for the training set
    predictions_train = model.predict(X_train)
    combined_df.loc[X_train.index, 'predictions'] = predictions_train

    # Predicting the target variable for all other rows in the dataframe
    X_all = df[features]
    all_predictions = model.predict(X_all)
    df['predictions'] = all_predictions

    # Combine data usage labels back to the original dataframe
    combined_data_usage = pd.concat([combined_df[['data_usage']], df[['predictions']]], axis=0)
    df = df.join(combined_data_usage, how='left', rsuffix='_model')

    # Calculating and printing the accuracy metrics
    accuracy = accuracy_score(y_test, predictions_test)
    precision = precision_score(y_test, predictions_test)
    recall = recall_score(y_test, predictions_test)
    f1 = f1_score(y_test, predictions_test)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Printing class-specific accuracy metrics
    print("\nClassification Report:")
    print(classification_report(y_test, predictions_test))

    df = _calculate_similarity(df, features, col_to_compare, pos, neg)

    return [df, permutation_df, feature_importance_df, model, X_train, X_test, y_train, y_test]

def _shap_analysis(model, X_train, X_test):
    
    """
    Performs SHAP analysis on the given model and data.

    Args:
    model: The trained model.
    X_train (pandas.DataFrame): Training feature set.
    X_test (pandas.DataFrame): Testing feature set.
    """

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)

def plate_heatmap(src, model_type='xgboost', variable='predictions', grouping='mean', min_max='allq', cmap='viridis', channel_of_interest=3, min_count=25, n_estimators=100, col_to_compare='col', pos='c1', neg='c2', exclude=None, n_repeats=10, clean=True, nr_to_plot=20, verbose=False, n_jobs=-1):
    from .io import _read_and_merge_data
    from .plot import _plot_plates

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    include_multinucleated, include_multiinfected, include_noninfected = True, 2.0, True
    
    df, _ = _read_and_merge_data(db_loc, 
                                 tables,
                                 verbose=verbose,
                                 include_multinucleated=include_multinucleated,
                                 include_multiinfected=include_multiinfected,
                                 include_noninfected=include_noninfected)
        
    if not channel_of_interest is None:
        df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
        feature_string = f'channel_{channel_of_interest}'
    else:
        feature_string = None
    
    output = _permutation_importance(df, feature_string, col_to_compare, pos, neg, exclude, n_repeats, clean, nr_to_plot, n_estimators=n_estimators, random_state=42, model_type=model_type, n_jobs=n_jobs)
    
    _shap_analysis(output[3], output[4], output[5])

    features = output[0].select_dtypes(include=[np.number]).columns.tolist()

    if not variable in features:
        raise ValueError(f"Variable {variable} not found in the dataframe. Please choose one of the following: {features}")
    
    plate_heatmap = _plot_plates(output[0], variable, grouping, min_max, cmap, min_count)
    return [output, plate_heatmap]

def join_measurments_and_annotation(src, tables = ['cell', 'nucleus', 'pathogen','cytoplasm']):
    
    from .io import _read_and_merge_data, _read_db
    
    db_loc = [src+'/measurements/measurements.db']
    loc = src+'/measurements/measurements.db'
    df, _ = _read_and_merge_data(db_loc, 
                                 tables, 
                                 verbose=True, 
                                 include_multinucleated=True, 
                                 include_multiinfected=True, 
                                 include_noninfected=True)
    
    paths_df = _read_db(loc, tables=['png_list'])

    merged_df = pd.merge(df, paths_df[0], on='prcfo', how='left')

    return merged_df

def jitterplot_by_annotation(src, x_column, y_column, plot_title='Jitter Plot', output_path=None, filter_column=None, filter_values=None):
    """
    Reads a CSV file and creates a jitter plot of one column grouped by another column.
    
    Args:
    src (str): Path to the source data.
    x_column (str): Name of the column to be used for the x-axis.
    y_column (str): Name of the column to be used for the y-axis.
    plot_title (str): Title of the plot. Default is 'Jitter Plot'.
    output_path (str): Path to save the plot image. If None, the plot will be displayed. Default is None.
    
    Returns:
    pd.DataFrame: The filtered and balanced DataFrame.
    """
    # Read the CSV file into a DataFrame
    df = join_measurments_and_annotation(src, tables=['cell', 'nucleus', 'pathogen', 'cytoplasm'])

    # Print column names for debugging
    print(f"Generated dataframe with: {df.shape[1]} columns and {df.shape[0]} rows")
    #print("Columns in DataFrame:", df.columns.tolist())

    # Replace NaN values with a specific label in x_column
    df[x_column] = df[x_column].fillna('NaN')

    # Filter the DataFrame if filter_column and filter_values are provided
    if not filter_column is None:
        if isinstance(filter_column, str):
            df = df[df[filter_column].isin(filter_values)]
        if isinstance(filter_column, list):
            for i,val in enumerate(filter_column):
                print(f'hello {len(df)}')
                df = df[df[val].isin(filter_values[i])]

    # Use the correct column names based on your DataFrame
    required_columns = ['plate_x', 'row_x', 'col_x']
    if not all(column in df.columns for column in required_columns):
        raise KeyError(f"DataFrame does not contain the necessary columns: {required_columns}")

    # Filter to retain rows with non-NaN values in x_column and with matching plate, row, col values
    non_nan_df = df[df[x_column] != 'NaN']
    retained_rows = df[df[['plate_x', 'row_x', 'col_x']].apply(tuple, axis=1).isin(non_nan_df[['plate_x', 'row_x', 'col_x']].apply(tuple, axis=1))]

    # Determine the minimum count of examples across all groups in x_column
    min_count = retained_rows[x_column].value_counts().min()
    print(f'Found {min_count} annotated images')

    # Randomly sample min_count examples from each group in x_column
    balanced_df = retained_rows.groupby(x_column).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

    # Create the jitter plot
    plt.figure(figsize=(10, 6))
    jitter_plot = sns.stripplot(data=balanced_df, x=x_column, y=y_column, hue=x_column, jitter=True, palette='viridis', dodge=False)
    plt.title(plot_title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    # Customize the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust the position of the x-axis labels to be centered below the data
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
    
    # Save the plot to a file or display it
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Jitter plot saved to {output_path}")
    else:
        plt.show()

    return balanced_df