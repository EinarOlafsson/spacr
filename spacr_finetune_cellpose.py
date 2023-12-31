import os
import gc
import cv2
import numpy as np
import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import time
import torch
import re

print('Torch available:', torch.cuda.is_available())
print('CUDA version:',torch.version.cuda)

from cellpose import models, io
from skimage.exposure import rescale_intensity
from collections import deque
import warnings
import imageio
from matplotlib.patches import Polygon
import matplotlib as mpl
#%matplotlib qt
import gc

def normalize_to_dtype(array, q1=2, q2=98, percentiles=None):
    # Ensure array is at least 3D
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)

    num_channels = array.shape[-1]
    new_stack = np.empty_like(array)

    for channel in range(num_channels):
        img = array[..., channel]
        non_zero_img = img[img > 0]

        # Determine min and max for intensity scaling
        if non_zero_img.size > 0:
            img_min = np.percentile(non_zero_img, q1)
            img_max = np.percentile(non_zero_img, q2)
        else:
            img_min, img_max = (percentiles[channel] if percentiles and channel < len(percentiles)
                                else (img.min(), img.max()))

        # Rescale intensity
        new_stack[..., channel] = rescale_intensity(img, in_range=(img_min, img_max), out_range='dtype')

    # Remove the added dimension for 2D input
    if new_stack.shape[-1] == 1:
        new_stack = np.squeeze(new_stack, axis=-1)

    return new_stack

def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
    paths = []
    for file in os.listdir(src):
        if file.endswith('.tif') or file.endswith('.tiff'):
            path = os.path.join(src, file)
            paths.append(path)
    paths = random.sample(paths, nr)
    for path in paths:
        print(f'Image path:{path}')
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if normalize:
            img = normalize_to_dtype(array=img, q1=q1, q2=q2)
        dim = img.shape
        if len(img.shape) > 2:
            array_nr = img.shape[2]
            fig, axs = plt.subplots(1, array_nr, figsize=(figuresize, figuresize))
            for channel in range(array_nr):
                i = np.take(img, [channel], axis=2)
                axs[channel].imshow(i, cmap=plt.get_cmap(cmap))
                axs[channel].set_title('Channel '+str(channel), size=24)
                axs[channel].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
            ax.imshow(img, cmap=plt.get_cmap(cmap))
            ax.set_title('Channel 0', size=24)
            ax.axis('off')
        fig.tight_layout()
        plt.show()
    return


def print_mask_and_flows(stack, mask, flows):
    # Create subplots: 1 for image, 1 for mask, rest for each flow
    fig, axs = plt.subplots(1,  3, figsize=(40, 5))

    # Plot the original image
    axs[0].imshow(stack[:, :, 0], cmap='gray')
    #axs[0].imshow(stack, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot the mask
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')

    # Plot flow
    axs[2].imshow(flows[0], cmap='jet')
    axs[2].set_title(f'Flows')
    axs[2].axis('off')
    fig.tight_layout()
    plt.show()
    

def identify_masks(paths, dst, model_name, channels, diameter, flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', verbose=False, plot=False, save=False):
    print('========== generating masks ==========')
    print('Torch available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.Cellpose(gpu=True, model_type=model_name, net_avg=True, device=device)

    chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nuclei' else [1,0] if model_name == 'cyto' else [2, 0] 
    
    if verbose == True:
        print(f'Settings: minimum_size: {minimum_size}, maximum_size:{maximum_size}')
        print(f'Cellpose settings: Model: {model_name}, channels: {channels}, cellpose_chans: {chans}, diameter:{diameter}, flow_threshold:{flow_threshold}, cellprob_threshold:{cellprob_threshold}')
        
    time_ls = []
    
    for file_index, path in enumerate(paths):
        stack = cv2.imread(path).astype(np.float32)
        stack = stack[:, :, channels]
        filename = os.path.basename(path)
        start = time.time()
        stack = normalize_to_dtype(stack, q1=2,q2=98)
        
        if stack.max() > 1:
            stack = stack / stack.max()

        mask, flows, _, _ = model.eval(x=stack,
                                        normalize=False,
                                        channels=chans,
                                        channel_axis=3,
                                        diameter=diameter,
                                        flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold,
                                        rescale=None,
                                        resample=True,
                                        net_avg=True,
                                        progress=None)

        stop = time.time()
        duration = (stop - start)
        time_ls.append(duration)
        average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
        print(f'Processing {file_index+1}/{len(paths)} images : Time/image {average_time:.3f} sec', end='\r', flush=True)
        if plot:
            print_mask_and_flows(stack, mask, flows)
        if save:
            output_filename = os.path.join(dst, filename)
            cv2.imwrite(output_filename, mask)
    return

def generate_cp_masks(src, model_name, channels, diameter, regex='.tif', flow_threshold=30, cellprob_threshold=1, figuresize=25, cmap='inferno', verbose=False, plot=False, save=False):
    dst = os.path.join(src,'masks')
    os.makedirs(dst, exist_ok=True)
    paths = []
    
    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        
        if filename.endswith('.tif'):
            if re.search(regex, filename):
                paths.append(path)
    
    identify_masks(paths, dst, model_name, channels, diameter,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, figuresize=figuresize, cmap=cmap, verbose=verbose, plot=plot, save=save)

def train_cellpose(img_src, mask_src, model_name='toxopv', model_type='cyto', nchan=2, channels=[0, 0], learning_rate=0.2, weight_decay=1e-05, batch_size=8, n_epochs=500):
    model_name=model_name+'.CP_model'
    # Load training data
    train_images, train_masks, _, _, _, _ = io.load_train_test_data(img_src, mask_src, mask_filter='')

    # Create a CellposeModel instance
    model = models.CellposeModel(gpu=True, 
                                 model_type=model_type, 
                                 net_avg=True, 
                                 diam_mean=30.0, 
                                 residual_on=True, 
                                 style_on=True, 
                                 concatenation=False, 
                                 nchan=nchan)

    # Specify the save path for the model
    model_save_path = os.path.join(mask_src, 'models', 'cellpose_model')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Train the model
    model.train(train_data=train_images,
                train_labels=train_masks,
                channels=channels,  # Adjust based on your image channels
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                n_epochs=n_epochs,
                save_path=model_save_path,
                model_name=model_name)

    return print(f"Model saved at: {model_save_path}")
