import pandas as pd 
import numpy as np
from .core import _permutation_importance, _shap_analysis

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

def plate_heatmap(src, model_type='xgboost', variable='predictions', grouping='mean', min_max='allq', cmap='viridis', channel_of_interest=3, min_count=25, n_estimators=100, col_to_compare='col', pos='c1', neg='c2', exclude=None, n_repeats=10, clean=True, nr_to_plot=20, verbose=False, n_jobs=-1):
    from .io import _read_and_merge_data
    from .plot import _plot_plates

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    include_multinucleated, include_multiinfected, include_noninfected = True, 2.0, True
    
    df = join_measurments_and_annotation(src, tables=['cell', 'nucleus', 'pathogen', 'cytoplasm'])
        
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