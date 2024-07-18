import os

def set_default_plot_merge_settings():
    settings = {}
    settings.setdefault('include_noninfected', True)
    settings.setdefault('include_multiinfected', True)
    settings.setdefault('include_multinucleated', True)
    settings.setdefault('remove_background', False)
    settings.setdefault('filter_min_max', None)
    settings.setdefault('channel_dims', [0,1,2,3])
    settings.setdefault('backgrounds', [100,100,100,100])
    settings.setdefault('cell_mask_dim', 4)
    settings.setdefault('nucleus_mask_dim', 5)
    settings.setdefault('pathogen_mask_dim', 6)
    settings.setdefault('outline_thickness', 3)
    settings.setdefault('outline_color', 'gbr')
    settings.setdefault('overlay_chans', [1,2,3])
    settings.setdefault('overlay', True)
    settings.setdefault('normalization_percentiles', [2,98])
    settings.setdefault('normalize', True)
    settings.setdefault('print_object_number', True)
    settings.setdefault('nr', 1)
    settings.setdefault('figuresize', 50)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('verbose', True)
    return settings

def set_default_settings_preprocess_generate_masks(src, settings={}):
    # Main settings
    settings['src'] = src
    settings.setdefault('preprocess', True)
    settings.setdefault('masks', True)
    settings.setdefault('save', True)
    settings.setdefault('batch_size', 50)
    settings.setdefault('test_mode', False)
    settings.setdefault('test_images', 10)
    settings.setdefault('magnification', 20)
    settings.setdefault('custom_regex', None)
    settings.setdefault('metadata_type', 'cellvoyager')
    settings.setdefault('workers', os.cpu_count()-4)
    settings.setdefault('randomize', True)
    settings.setdefault('verbose', True)
    settings.setdefault('remove_background_cell', False)
    settings.setdefault('remove_background_nucleus', False)
    settings.setdefault('remove_background_pathogen', False)

    # Channel settings
    settings.setdefault('cell_channel', None)
    settings.setdefault('nucleus_channel', None)
    settings.setdefault('pathogen_channel', None)
    settings.setdefault('channels', [0,1,2,3])
    settings.setdefault('pathogen_background', 100)
    settings.setdefault('pathogen_Signal_to_noise', 10)
    settings.setdefault('pathogen_CP_prob', 0)
    settings.setdefault('cell_background', 100)
    settings.setdefault('cell_Signal_to_noise', 10)
    settings.setdefault('cell_CP_prob', 0)
    settings.setdefault('nucleus_background', 100)
    settings.setdefault('nucleus_Signal_to_noise', 10)
    settings.setdefault('nucleus_CP_prob', 0)
    settings.setdefault('nucleus_FT', 100)
    settings.setdefault('cell_FT', 100)
    settings.setdefault('pathogen_FT', 100)
    
    # Plot settings
    settings.setdefault('plot', False)
    settings.setdefault('figuresize', 50)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('normalize', True)
    settings.setdefault('normalize_plots', True)
    settings.setdefault('examples_to_plot', 1)

    # Analasys settings
    settings.setdefault('pathogen_model', None)
    settings.setdefault('merge_pathogens', False)
    settings.setdefault('filter', False)
    settings.setdefault('lower_percentile', 2)

    # Timelapse settings
    settings.setdefault('timelapse', False)
    settings.setdefault('fps', 2)
    settings.setdefault('timelapse_displacement', None)
    settings.setdefault('timelapse_memory', 3)
    settings.setdefault('timelapse_frame_limits', None)
    settings.setdefault('timelapse_remove_transient', False)
    settings.setdefault('timelapse_mode', 'trackpy')
    settings.setdefault('timelapse_objects', 'cells')

    # Misc settings
    settings.setdefault('all_to_mip', False)
    settings.setdefault('pick_slice', False)
    settings.setdefault('skip_mode', '01')
    settings.setdefault('upscale', False)
    settings.setdefault('upscale_factor', 2.0)
    settings.setdefault('adjust_cells', False)
    return settings

def set_default_settings_preprocess_img_data(settings):

    metadata_type = settings.setdefault('metadata_type', 'cellvoyager')
    custom_regex = settings.setdefault('custom_regex', None)
    nr = settings.setdefault('nr', 1)
    plot = settings.setdefault('plot', True)
    batch_size = settings.setdefault('batch_size', 50)
    timelapse = settings.setdefault('timelapse', False)
    lower_percentile = settings.setdefault('lower_percentile', 2)
    randomize = settings.setdefault('randomize', True)
    all_to_mip = settings.setdefault('all_to_mip', False)
    pick_slice = settings.setdefault('pick_slice', False)
    skip_mode = settings.setdefault('skip_mode', False)

    cmap = settings.setdefault('cmap', 'inferno')
    figuresize = settings.setdefault('figuresize', 50)
    normalize = settings.setdefault('normalize', True)
    save_dtype = settings.setdefault('save_dtype', 'uint16')
    
    test_mode = settings.setdefault('test_mode', False)
    test_images = settings.setdefault('test_images', 10)
    random_test = settings.setdefault('random_test', True)

    return settings, metadata_type, custom_regex, nr, plot, batch_size, timelapse, lower_percentile, randomize, all_to_mip, pick_slice, skip_mode, cmap, figuresize, normalize, save_dtype, test_mode, test_images, random_test

def _get_object_settings(object_type, settings):

    from .utils import _get_diam
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

def get_umap_image_settings(settings={}):
    settings.setdefault('src', 'path')
    settings.setdefault('row_limit', 1000)
    settings.setdefault('tables', ['cell', 'cytoplasm', 'nucleus', 'pathogen'])
    settings.setdefault('visualize', 'cell')
    settings.setdefault('image_nr', 16)
    settings.setdefault('dot_size', 50)
    settings.setdefault('n_neighbors', 1000)
    settings.setdefault('min_dist', 0.1)
    settings.setdefault('metric', 'euclidean')
    settings.setdefault('eps', 0.5)
    settings.setdefault('min_samples', 1000)
    settings.setdefault('filter_by', 'channel_0')
    settings.setdefault('img_zoom', 0.5)
    settings.setdefault('plot_by_cluster', True)
    settings.setdefault('plot_cluster_grids', True)
    settings.setdefault('remove_cluster_noise', True)
    settings.setdefault('remove_highly_correlated', True)
    settings.setdefault('log_data', False)
    settings.setdefault('figuresize', 60)
    settings.setdefault('black_background', True)
    settings.setdefault('remove_image_canvas', False)
    settings.setdefault('plot_outlines', True)
    settings.setdefault('plot_points', True)
    settings.setdefault('smooth_lines', True)
    settings.setdefault('clustering', 'dbscan')
    settings.setdefault('exclude', None)
    settings.setdefault('col_to_compare', 'col')
    settings.setdefault('pos', 'c1')
    settings.setdefault('neg', 'c2')
    settings.setdefault('embedding_by_controls', False)
    settings.setdefault('plot_images', True)
    settings.setdefault('reduction_method','umap')
    settings.setdefault('save_figure', False)
    settings.setdefault('n_jobs', -1)
    settings.setdefault('color_by', None)
    settings.setdefault('neg', 'c1')
    settings.setdefault('pos', 'c2')
    settings.setdefault('mix', 'c3')
    settings.setdefault('mix', 'c3')
    settings.setdefault('exclude_conditions', None)
    settings.setdefault('analyze_clusters', False)
    settings.setdefault('resnet_features', False)
    settings.setdefault('verbose',True)
    return settings

def get_measure_crop_settings(settings):

    # Test mode
    settings.setdefault('test_mode', False)
    settings.setdefault('test_nr', 10)

    #measurement settings
    settings.setdefault('save_measurements',True)
    settings.setdefault('radial_dist', True)
    settings.setdefault('calculate_correlation', True)
    settings.setdefault('manders_thresholds', [15,85,95])
    settings.setdefault('homogeneity', True)
    settings.setdefault('homogeneity_distances', [8,16,32])

    # Cropping settings
    settings.setdefault('save_arrays', False)
    settings.setdefault('save_png',True)
    settings.setdefault('use_bounding_box',False)
    settings.setdefault('png_size',[224,224])
    settings.setdefault('png_dims',[0,1,2])
    settings.setdefault('normalize',False)
    settings.setdefault('normalize_by','png')
    settings.setdefault('crop_mode',['cell'])
    settings.setdefault('dialate_pngs', False)
    settings.setdefault('dialate_png_ratios', [0.2])

    # Timelapsed settings
    settings.setdefault('timelapse', False)
    settings.setdefault('timelapse_objects', 'cell')

    # Operational settings
    settings.setdefault('plot',False)
    settings.setdefault('plot_filtration',False)
    settings.setdefault('representative_images', False)
    settings.setdefault('max_workers', os.cpu_count()-2)

    # Object settings
    settings.setdefault('cell_mask_dim',None)
    settings.setdefault('nucleus_mask_dim',None)
    settings.setdefault('pathogen_mask_dim',None)
    settings.setdefault('cytoplasm',False)
    settings.setdefault('include_uninfected',True)
    settings.setdefault('cell_min_size',0)
    settings.setdefault('nucleus_min_size',0)
    settings.setdefault('pathogen_min_size',0)
    settings.setdefault('cytoplasm_min_size',0)
    settings.setdefault('merge_edge_pathogen_cells', True)

    # Miscellaneous settings
    settings.setdefault('experiment', 'exp')
    settings.setdefault('cells', 'HeLa')
    settings.setdefault('cell_loc', None)
    settings.setdefault('pathogens', ['ME49Dku80WT', 'ME49Dku80dgra8:GRA8', 'ME49Dku80dgra8', 'ME49Dku80TKO'])
    settings.setdefault('pathogen_loc', [['c1', 'c2', 'c3', 'c4', 'c5', 'c6'], ['c7', 'c8', 'c9', 'c10', 'c11', 'c12'], ['c13', 'c14', 'c15', 'c16', 'c17', 'c18'], ['c19', 'c20', 'c21', 'c22', 'c23', 'c24']])
    settings.setdefault('treatments', ['BR1', 'BR2', 'BR3'])
    settings.setdefault('treatment_loc', [['c1', 'c2', 'c7', 'c8', 'c13', 'c14', 'c19', 'c20'], ['c3', 'c4', 'c9', 'c10', 'c15', 'c16', 'c21', 'c22'], ['c5', 'c6', 'c11', 'c12', 'c17', 'c18', 'c23', 'c24']])
    settings.setdefault('channel_of_interest', 2)
    settings.setdefault('compartments', ['pathogen', 'cytoplasm'])
    settings.setdefault('measurement', 'mean_intensity')
    settings.setdefault('nr_imgs', 32)
    settings.setdefault('um_per_pixel', 0.1)

    if settings['test_mode']:
        settings['plot'] = True
        settings['plot_filtration'] = True
        test_imgs = settings['test_nr']
        print(f'Test mode enabled with {test_imgs} images, plotting set to True')

    return settings

def set_default_analyze_screen(settings):
    settings.setdefault('model_type','xgboost')
    settings.setdefault('heatmap_feature','predictions')
    settings.setdefault('grouping','mean')
    settings.setdefault('min_max','allq')
    settings.setdefault('cmap','viridis')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('minimum_cell_count',25)
    settings.setdefault('n_estimators',100)
    settings.setdefault('test_size',0.2)
    settings.setdefault('location_column','col')
    settings.setdefault('positive_control','c2')
    settings.setdefault('negative_control','c1')
    settings.setdefault('exclude',None)
    settings.setdefault('n_repeats',10)
    settings.setdefault('top_features',30)
    settings.setdefault('remove_low_variance_features',True)
    settings.setdefault('remove_highly_correlated_features',True)
    settings.setdefault('n_jobs',-1)
    settings.setdefault('verbose',True)
    return settings

def set_default_train_test_model(settings):
    cores = os.cpu_count()-2
    settings.setdefault('train',True)
    settings.setdefault('test',False)
    settings.setdefault('classes',['nc','pc'])
    settings.setdefault('model_type','maxvit_t')
    settings.setdefault('optimizer_type','adamw')
    settings.setdefault('schedule','reduce_lr_on_plateau') #reduce_lr_on_plateau, step_lr
    settings.setdefault('loss_type','focal_loss') # binary_cross_entropy_with_logits
    settings.setdefault('normalize',True)
    settings.setdefault('image_size',224)
    settings.setdefault('batch_size',64)
    settings.setdefault('epochs',100)
    settings.setdefault('val_split',0.1)
    settings.setdefault('train_mode','erm')
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('weight_decay',0.00001)
    settings.setdefault('dropout_rate',0.1)
    settings.setdefault('init_weights',True)
    settings.setdefault('amsgrad',True)
    settings.setdefault('use_checkpoint',True)
    settings.setdefault('gradient_accumulation',True)
    settings.setdefault('gradient_accumulation_steps',4)
    settings.setdefault('intermedeate_save',True)
    settings.setdefault('pin_memory',True)
    settings.setdefault('num_workers',cores)
    settings.setdefault('channels',['r','g','b'])
    settings.setdefault('augment',False)
    settings.setdefault('verbose',False)
    return settings

def get_analyze_recruitment_default_settings(settings):
    settings.setdefault('target','protein')
    settings.setdefault('cell_types',['HeLa'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['pathogen_1', 'pathogen_2'])
    settings.setdefault('pathogen_plate_metadata',[['c1', 'c2', 'c3'],['c4','c5', 'c6']])
    settings.setdefault('treatments',['cm', 'lovastatin'])
    settings.setdefault('treatment_plate_metadata',[['r1', 'r2','r3'], ['r4', 'r5','r6']])
    settings.setdefault('metadata_types',['col', 'col', 'row'])
    settings.setdefault('channel_dims',[0,1,2,3])
    settings.setdefault('cell_chann_dim',3)
    settings.setdefault('cell_mask_dim',4)
    settings.setdefault('nucleus_chann_dim',0)
    settings.setdefault('nucleus_mask_dim',5)
    settings.setdefault('pathogen_chann_dim',2)
    settings.setdefault('pathogen_mask_dim',6)
    settings.setdefault('channel_of_interest',2)
    settings.setdefault('plot',True)
    settings.setdefault('plot_nr',10)
    settings.setdefault('plot_control',True)
    settings.setdefault('figuresize',20)
    settings.setdefault('remove_background',False)
    settings.setdefault('backgrounds',100)
    settings.setdefault('include_noninfected',True)
    settings.setdefault('include_multiinfected',True)
    settings.setdefault('include_multinucleated',True)
    settings.setdefault('cells_per_well',0)
    settings.setdefault('pathogen_size_range',[0,100000])
    settings.setdefault('nucleus_size_range',[0,100000])
    settings.setdefault('cell_size_range',[0,100000])
    settings.setdefault('pathogen_intensity_range',[0,100000])
    settings.setdefault('nucleus_intensity_range',[0,100000])
    settings.setdefault('cell_intensity_range',[0,100000])
    settings.setdefault('target_intensity_min',0)
    return settings

def get_analyze_reads_default_settings(settings):
    settings.setdefault('upstream', 'CTTCTGGTAAATGGGGATGTCAAGTT') 
    settings.setdefault('downstream', 'GTTTAAGAGCTATGCTGGAAACAGCAG') #This is the reverce compliment of the column primer starting from the end #TGCTGTTTAAGAGCTATGCTGGAAACAGCA
    settings.setdefault('barecode_length_1', 8)
    settings.setdefault('barecode_length_2', 7)
    settings.setdefault('chunk_size', 1000000)
    settings.setdefault('test', False)
    return settings

def get_map_barcodes_default_settings(settings):
    settings.setdefault('grna', '/home/carruthers/Documents/grna_barcodes.csv')
    settings.setdefault('barcodes', '/home/carruthers/Documents/SCREEN_BARCODES.csv')
    settings.setdefault('plate_dict', {'EO1': 'plate1', 'EO2': 'plate2', 'EO3': 'plate3', 'EO4': 'plate4', 'EO5': 'plate5', 'EO6': 'plate6', 'EO7': 'plate7', 'EO8': 'plate8'})
    settings.setdefault('test', False)
    settings.setdefault('verbose', True)
    settings.setdefault('pc', 'TGGT1_220950_1')
    settings.setdefault('pc_loc', 'c2')
    settings.setdefault('nc', 'TGGT1_233460_4')
    settings.setdefault('nc_loc', 'c1')
    return settings

def get_train_cellpose_default_settings(settings):
    settings.setdefault('model_name','new_model')
    settings.setdefault('model_type','cyto')
    settings.setdefault('Signal_to_noise',10)
    settings.setdefault('background',200)
    settings.setdefault('remove_background',False)
    settings.setdefault('learning_rate',0.2)
    settings.setdefault('weight_decay',1e-05)
    settings.setdefault('batch_size',8)
    settings.setdefault('n_epochs',10000)
    settings.setdefault('from_scratch',False)
    settings.setdefault('diameter',30)
    settings.setdefault('resize',False)
    settings.setdefault('width_height',[1000,1000])
    settings.setdefault('verbose',True)
    return settings

def get_perform_regression_default_settings(settings):
    settings.setdefault('gene_weights_csv', '/nas_mnt/carruthers/Einar/mitoscreen/sequencing/combined_reads/EO1_combined/EO1_combined_combination_counts.csv')
    settings.setdefault('dependent_variable','predictions')
    settings.setdefault('transform',None)
    settings.setdefault('agg_type','mean')
    settings.setdefault('min_cell_count',25)
    settings.setdefault('regression_type','ols')
    settings.setdefault('remove_row_column_effect',False)
    settings.setdefault('alpha',1)
    settings.setdefault('fraction_threshold',0.1)
    settings.setdefault('nc','c1')
    settings.setdefault('pc','c2')
    settings.setdefault('other','c3')
    settings.setdefault('plate','plate1')
    settings.setdefault('class_1_threshold',None)
    
    if settings['regression_type'] == 'quantile':
        print(f"Using alpha as quantile for quantile regression, alpha: {settings['alpha']}")
        settings['agg_type'] = None
        print(f'agg_type set to None for quantile regression')
    return settings

def get_check_cellpose_models_default_settings(settings):
    settings.setdefault('batch_size', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', True)
    settings.setdefault('normalize', True)
    settings.setdefault('channels', [0,0])
    settings.setdefault('percentiles', None)
    settings.setdefault('circular', False)
    settings.setdefault('invert', False)
    settings.setdefault('plot', True)
    settings.setdefault('diameter', 40)
    settings.setdefault('grayscale', True)
    settings.setdefault('remove_background', False)
    settings.setdefault('background', 100)
    settings.setdefault('Signal_to_noise', 5)
    settings.setdefault('verbose', False)
    settings.setdefault('resize', False)
    settings.setdefault('target_height', None)
    settings.setdefault('target_width', None)
    return settings

def get_identify_masks_finetune_default_settings(settings):
    settings.setdefault('model_name', 'cyto')
    settings.setdefault('custom_model', None)
    settings.setdefault('channels', [0,0])
    settings.setdefault('background', 100)
    settings.setdefault('remove_background', False)
    settings.setdefault('Signal_to_noise', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('diameter', 30)
    settings.setdefault('batch_size', 50)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', False)
    settings.setdefault('verbose', False)
    settings.setdefault('normalize', True)
    settings.setdefault('percentiles', None)
    settings.setdefault('circular', False)
    settings.setdefault('invert', False)
    settings.setdefault('resize', False)
    settings.setdefault('target_height', None)
    settings.setdefault('target_width', None)
    settings.setdefault('rescale', False)
    settings.setdefault('resample', False)
    settings.setdefault('grayscale', True)
    return settings