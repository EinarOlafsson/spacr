import os,re, random, cv2, glob, time, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndi
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

from IPython.display import display
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
from skimage.morphology import square, dilation 
from skimage import measure

from ipywidgets import IntSlider, interact
from IPython.display import Image as ipyimage

from .logger import log_function_call


#from .io import _save_figure
#from .timelapse import _save_mask_timelapse_as_gif
#from .utils import normalize_to_dtype, _remove_outside_objects, _remove_multiobject_cells, _find_similar_sized_images, _remove_noninfected

def plot_masks(batch, masks, flows, cmap='inferno', figuresize=20, nr=1, file_type='.npz', print_object_number=True):
    """
    Plot the masks and flows for a given batch of images.

    Args:
        batch (numpy.ndarray): The batch of images.
        masks (list or numpy.ndarray): The masks corresponding to the images.
        flows (list or numpy.ndarray): The flows corresponding to the images.
        cmap (str, optional): The colormap to use for displaying the images. Defaults to 'inferno'.
        figuresize (int, optional): The size of the figure. Defaults to 20.
        nr (int, optional): The maximum number of images to plot. Defaults to 1.
        file_type (str, optional): The file type of the flows. Defaults to '.npz'.
        print_object_number (bool, optional): Whether to print the object number on the mask. Defaults to True.

    Returns:
        None
    """
    if len(batch.shape) == 3:
        batch = np.expand_dims(batch, axis=0)
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(flows, list):
        flows = [flows]
    else:
        flows = flows[0]
    if file_type == 'png':
        flows = [f[0] for f in flows]  # assuming this is what you want to do when file_type is 'png'
    font = figuresize/2
    index = 0
    for image, mask, flow in zip(batch, masks, flows):
        unique_labels = np.unique(mask)
        
        num_objects = len(unique_labels[unique_labels != 0])
        random_colors = np.random.rand(num_objects+1, 4)
        random_colors[:, 3] = 1
        random_colors[0, :] = [0, 0, 0, 1]
        random_cmap = mpl.colors.ListedColormap(random_colors)
        
        if index < nr:
            index += 1
            chans = image.shape[-1]
            fig, ax = plt.subplots(1, image.shape[-1] + 2, figsize=(4 * figuresize, figuresize))
            for v in range(0, image.shape[-1]):
                ax[v].imshow(image[..., v], cmap=cmap) #_imshow
                ax[v].set_title('Image - Channel'+str(v))
            ax[chans].imshow(mask, cmap=random_cmap) #_imshow
            ax[chans].set_title('Mask')
            if print_object_number:
                unique_objects = np.unique(mask)[1:]
                for obj in unique_objects:
                    cy, cx = ndi.center_of_mass(mask == obj)
                    ax[chans].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')
            ax[chans+1].imshow(flow, cmap='viridis') #_imshow
            ax[chans+1].set_title('Flow')
            plt.show()
    return

def _plot_4D_arrays(src, figuresize=10, cmap='inferno', nr_npz=1, nr=1):
    """
    Plot 4D arrays from .npz files.

    Args:
        src (str): The directory path where the .npz files are located.
        figuresize (int, optional): The size of the figure. Defaults to 10.
        cmap (str, optional): The colormap to use for image visualization. Defaults to 'inferno'.
        nr_npz (int, optional): The number of .npz files to plot. Defaults to 1.
        nr (int, optional): The number of images to plot from each .npz file. Defaults to 1.
    """
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    paths = random.sample(paths, min(nr_npz, len(paths)))

    for path in paths:
        with np.load(path) as data:
            stack = data['data']
        num_images = stack.shape[0]
        num_channels = stack.shape[3]

        for i in range(min(nr, num_images)):
            img = stack[i]

            # Create subplots
            if num_channels == 1:
                fig, axs = plt.subplots(1, 1, figsize=(figuresize, figuresize))
                axs = [axs]  # Make axs a list to use axs[c] later
            else:
                fig, axs = plt.subplots(1, num_channels, figsize=(num_channels * figuresize, figuresize))

            for c in range(num_channels):
                axs[c].imshow(img[:, :, c], cmap=cmap) #_imshow
                axs[c].set_title(f'Channel {c}', size=24)
                axs[c].axis('off')

            fig.tight_layout()
            plt.show()
    return

def generate_mask_random_cmap(mask):
    """
    Generate a random colormap based on the unique labels in the given mask.

    Parameters:
    mask (numpy.ndarray): The input mask array.

    Returns:
    matplotlib.colors.ListedColormap: The random colormap.
    """
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap
    
def random_cmap(num_objects=100):
    """
    Generate a random colormap.

    Parameters:
    num_objects (int): The number of objects to generate colors for. Default is 100.

    Returns:
    random_cmap (matplotlib.colors.ListedColormap): A random colormap.
    """
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def _generate_mask_random_cmap(mask):
    """
    Generate a random colormap based on the unique labels in the given mask.

    Parameters:
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

def _get_colours_merged(outline_color):
    """
    Get the merged outline colors based on the specified outline color format.

    Parameters:
    outline_color (str): The outline color format. Can be one of 'rgb', 'bgr', 'gbr', or 'rbg'.

    Returns:
    list: A list of merged outline colors based on the specified format.
    """
    if outline_color == 'rgb':
        outline_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rgb
    elif outline_color == 'bgr':
        outline_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]  # bgr
    elif outline_color == 'gbr':
        outline_colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # gbr
    elif outline_color == 'rbg':
        outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
    else:
        outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
    return outline_colors

def _filter_objects_in_plot(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, mask_dims, filter_min_max, include_multinucleated, include_multiinfected):
    """
    Filters objects in a plot based on various criteria.

    Args:
        stack (numpy.ndarray): The input stack of masks.
        cell_mask_dim (int): The dimension index of the cell mask.
        nucleus_mask_dim (int): The dimension index of the nucleus mask.
        pathogen_mask_dim (int): The dimension index of the pathogen mask.
        mask_dims (list): A list of dimension indices for additional masks.
        filter_min_max (list): A list of minimum and maximum area values for each mask.
        include_multinucleated (bool): Whether to include multinucleated cells.
        include_multiinfected (bool): Whether to include multiinfected cells.

    Returns:
        numpy.ndarray: The filtered stack of masks.
    """
    from .utils import _remove_outside_objects, _remove_multiobject_cells
    
    stack = _remove_outside_objects(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)

    for i, mask_dim in enumerate(mask_dims):
        if not filter_min_max is None:
            min_max = filter_min_max[i]
        else:
            min_max = [0, 100000]

        mask = np.take(stack, mask_dim, axis=2)
        props = measure.regionprops_table(mask, properties=['label', 'area'])
        avg_size_before = np.mean(props['area'])
        total_count_before = len(props['label'])

        if not filter_min_max is None:
            valid_labels = props['label'][np.logical_and(props['area'] > min_max[0], props['area'] < min_max[1])]  
            stack[:, :, mask_dim] = np.isin(mask, valid_labels) * mask  

        props_after = measure.regionprops_table(stack[:, :, mask_dim], properties=['label', 'area']) 
        avg_size_after = np.mean(props_after['area'])
        total_count_after = len(props_after['label'])

        if mask_dim == cell_mask_dim:
            if include_multinucleated is False and nucleus_mask_dim is not None:
                stack = _remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
            if include_multiinfected is False and cell_mask_dim is not None and pathogen_mask_dim is not None:
                stack = _remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=nucleus_mask_dim)
            cell_area_before = avg_size_before
            cell_count_before = total_count_before
            cell_area_after = avg_size_after
            cell_count_after = total_count_after
        if mask_dim == nucleus_mask_dim:
            nucleus_area_before = avg_size_before
            nucleus_count_before = total_count_before
            nucleus_area_after = avg_size_after
            nucleus_count_after = total_count_after
        if mask_dim == pathogen_mask_dim:
            pathogen_area_before = avg_size_before
            pathogen_count_before = total_count_before
            pathogen_area_after = avg_size_after
            pathogen_count_after = total_count_after

    if cell_mask_dim is not None:
        print(f'removed {cell_count_before-cell_count_after} cells, cell size from {cell_area_before} to {cell_area_after}')
    if nucleus_mask_dim is not None:
        print(f'removed {nucleus_count_before-nucleus_count_after} nucleus, nucleus size from {nucleus_area_before} to {nucleus_area_after}')
    if pathogen_mask_dim is not None:
        print(f'removed {pathogen_count_before-pathogen_count_after} pathogens, pathogen size from {pathogen_area_before} to {pathogen_area_after}')

    return stack

def _normalize_and_outline(image, remove_background, backgrounds, normalize, normalization_percentiles, overlay, overlay_chans, mask_dims, outline_colors, outline_thickness):
    """
    Normalize and outline an image.

    Args:
        image (ndarray): The input image.
        remove_background (bool): Flag indicating whether to remove the background.
        backgrounds (list): List of background values for each channel.
        normalize (bool): Flag indicating whether to normalize the image.
        normalization_percentiles (list): List of percentiles for normalization.
        overlay (bool): Flag indicating whether to overlay outlines onto the image.
        overlay_chans (list): List of channel indices to overlay.
        mask_dims (list): List of dimensions to use for masking.
        outline_colors (list): List of colors for the outlines.
        outline_thickness (int): Thickness of the outlines.

    Returns:
        tuple: A tuple containing the overlayed image, the original image, and a list of outlines.
    """
    from .utils import normalize_to_dtype
    
    outlines = []
    if remove_background:
        for chan_index, channel in enumerate(range(image.shape[-1])):
            single_channel = image[:, :, channel]  # Extract the specific channel
            background = backgrounds[chan_index]
            single_channel[single_channel < background] = 0
            image[:, :, channel] = single_channel
    if normalize:
        image = normalize_to_dtype(array=image, q1=normalization_percentiles[0], q2=normalization_percentiles[1])
    rgb_image = np.take(image, overlay_chans, axis=-1)
    rgb_image = rgb_image.astype(float)
    rgb_image -= rgb_image.min()
    rgb_image /= rgb_image.max()
    if overlay:
        overlayed_image = rgb_image.copy()
        for i, mask_dim in enumerate(mask_dims):
            mask = np.take(image, mask_dim, axis=2)
            outline = np.zeros_like(mask)
            # Find the contours of the objects in the mask
            for j in np.unique(mask)[1:]:
                contours = find_contours(mask == j, 0.5)
                for contour in contours:
                    contour = contour.astype(int)
                    outline[contour[:, 0], contour[:, 1]] = j
            # Make the outline thicker
            outline = dilation(outline, square(outline_thickness))
            outlines.append(outline)
            # Overlay the outlines onto the RGB image
            for j in np.unique(outline)[1:]:
                overlayed_image[outline == j] = outline_colors[i % len(outline_colors)]
        return overlayed_image, image, outlines
    else:
        return [], image, []

def _plot_merged_plot(overlay, image, stack, mask_dims, figuresize, overlayed_image, outlines, cmap, outline_colors, print_object_number):
    """
    Plot the merged plot with overlay, image channels, and masks.

    Args:
        overlay (bool): Flag indicating whether to overlay the image with outlines.
        image (ndarray): Input image array.
        stack (ndarray): Stack of masks.
        mask_dims (list): List of mask dimensions.
        figuresize (float): Size of the figure.
        overlayed_image (ndarray): Overlayed image array.
        outlines (list): List of outlines.
        cmap (str): Colormap for the masks.
        outline_colors (list): List of outline colors.
        print_object_number (bool): Flag indicating whether to print object numbers on the masks.

    Returns:
        fig (Figure): The generated matplotlib figure.
    """
    if overlay:
        fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims) + 1, figsize=(4 * figuresize, figuresize))
        ax[0].imshow(overlayed_image) #_imshow
        ax[0].set_title('Overlayed Image')
        ax_index = 1
    else:
        fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims), figsize=(4 * figuresize, figuresize))
        ax_index = 0

    # Normalize and plot each channel with outlines
    for v in range(0, image.shape[-1]):
        channel_image = image[..., v]
        channel_image_normalized = channel_image.astype(float)
        channel_image_normalized -= channel_image_normalized.min()
        channel_image_normalized /= channel_image_normalized.max()
        channel_image_rgb = np.dstack((channel_image_normalized, channel_image_normalized, channel_image_normalized))

        # Apply the outlines onto the RGB image
        for outline, color in zip(outlines, outline_colors):
            for j in np.unique(outline)[1:]:
                channel_image_rgb[outline == j] = mpl.colors.to_rgb(color)

        ax[v + ax_index].imshow(channel_image_rgb)
        ax[v + ax_index].set_title('Image - Channel'+str(v))

    for i, mask_dim in enumerate(mask_dims):
        mask = np.take(stack, mask_dim, axis=2)
        random_cmap = _generate_mask_random_cmap(mask)
        ax[i + image.shape[-1] + ax_index].imshow(mask, cmap=random_cmap)
        ax[i + image.shape[-1] + ax_index].set_title('Mask '+ str(i))
        if print_object_number:
            unique_objects = np.unique(mask)[1:]
            for obj in unique_objects:
                cy, cx = ndi.center_of_mass(mask == obj)
                ax[i + image.shape[-1] + ax_index].text(cx, cy, str(obj), color='white', fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.show()
    return fig

def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
    """
    Plot randomly selected arrays from a given directory.

    Parameters:
    - src (str): The directory path containing the arrays.
    - figuresize (int): The size of the figure (default: 50).
    - cmap (str): The colormap to use for displaying the arrays (default: 'inferno').
    - nr (int): The number of arrays to plot (default: 1).
    - normalize (bool): Whether to normalize the arrays (default: True).
    - q1 (int): The lower percentile for normalization (default: 1).
    - q2 (int): The upper percentile for normalization (default: 99).

    Returns:
    None
    """
    from .utils import normalize_to_dtype
    
    mask_cmap = random_cmap()
    paths = []
    for file in os.listdir(src):
        if file.endswith('.npy'):
            path = os.path.join(src, file)
            paths.append(path)
    paths = random.sample(paths, nr)
    for path in paths:
        print(f'Image path:{path}')
        img = np.load(path)
        if normalize:
            img = normalize_to_dtype(array=img, q1=q1, q2=q2)
        dim = img.shape
        if len(img.shape)>2:
            array_nr = img.shape[2]
            fig, axs = plt.subplots(1, array_nr,figsize=(figuresize,figuresize))
            for channel in range(array_nr):
                i = np.take(img, [channel], axis=2)
                axs[channel].imshow(i, cmap=plt.get_cmap(cmap)) #_imshow
                axs[channel].set_title('Channel '+str(channel),size=24)
                axs[channel].axis('off')
        else:
            fig, ax = plt.subplots(1, 1,figsize=(figuresize,figuresize))
            ax.imshow(img, cmap=plt.get_cmap(cmap)) #_imshow
            ax.set_title('Channel 0',size=24)
            ax.axis('off')
        fig.tight_layout()
        plt.show()
    return

def plot_merged(src, settings):
    """
    Plot the merged images after applying various filters and modifications.

    Args:
        src (ndarray): The source images.
        settings (dict): The settings for the plot.

    Returns:
        None
    """
    from .utils import _remove_noninfected
    
    font = settings['figuresize']/2
    outline_colors = _get_colours_merged(settings['outline_color'])
    index = 0
        
    mask_dims = [settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim']]
    mask_dims = [element for element in mask_dims if element is not None]
    
    if settings['verbose']:
        display(settings)
        
    if settings['pathogen_mask_dim'] is None:
        settings['include_multiinfected'] = True

    for file in os.listdir(src):
        path = os.path.join(src, file)
        stack = np.load(path)
        print(f'Loaded: {path}')
        if not settings['include_noninfected']:
            if settings['pathogen_mask_dim'] is not None and settings['cell_mask_dim'] is not None:
                stack = _remove_noninfected(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'])

        if settings['include_multiinfected'] is not True or settings['include_multinucleated'] is not True or settings['filter_min_max'] is not None:
            stack = _filter_objects_in_plot(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], mask_dims, settings['filter_min_max'], settings['include_multinucleated'], settings['include_multiinfected'])

        #channels_to_keep = np.ones(stack.shape[-1], dtype=bool)
        #channels_to_keep[mask_dims] = False
        #channel_image = stack[:, :, channels_to_keep]
        
        overlayed_image, image, outlines = _normalize_and_outline(stack, settings['remove_background'], settings['backgrounds'], settings['normalize'], settings['normalization_percentiles'], settings['overlay'], settings['overlay_chans'], mask_dims, outline_colors, settings['outline_thickness'])
        
        if index < settings['nr']:
            index += 1
            fig = _plot_merged_plot(settings['overlay'], image, stack, mask_dims, settings['figuresize'], overlayed_image, outlines, settings['cmap'], outline_colors, settings['print_object_number'])
        else:
            return fig

def _plot_images_on_grid(image_files, channel_indices, um_per_pixel, scale_bar_length_um=5, fontsize=8, show_filename=True, channel_names=None, plot=False):
    """
    Plots a grid of images with optional scale bar and channel names.

    Args:
        image_files (list): List of image file paths.
        channel_indices (list): List of channel indices to select from the images.
        um_per_pixel (float): Micrometers per pixel.
        scale_bar_length_um (float, optional): Length of the scale bar in micrometers. Defaults to 5.
        fontsize (int, optional): Font size for the image titles. Defaults to 8.
        show_filename (bool, optional): Whether to show the image file names as titles. Defaults to True.
        channel_names (list, optional): List of channel names. Defaults to None.
        plot (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    print(f'scale bar represents {scale_bar_length_um} um')
    nr_of_images = len(image_files)
    cols = int(np.ceil(np.sqrt(nr_of_images)))
    rows = np.ceil(nr_of_images / cols)
    fig, axes = plt.subplots(int(rows), int(cols), figsize=(20, 20), facecolor='black')
    fig.patch.set_facecolor('black')
    axes = axes.flatten()
    # Calculate the scale bar length in pixels
    scale_bar_length_px = int(scale_bar_length_um / um_per_pixel)  # Convert to pixels

    channel_colors = ['red','green','blue']
    for i, image_file in enumerate(image_files):
        img_array = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # Handle different channel selections
        if channel_indices is not None:
            if len(channel_indices) == 1:  # Single channel (grayscale)
                img_array = img_array[:, :, channel_indices[0]]
                cmap = 'gray'
            elif len(channel_indices) == 2:  # Dual channels
                img_array = np.mean(img_array[:, :, channel_indices], axis=2)
                cmap = 'gray'
            else:  # RGB or more channels
                img_array = img_array[:, :, channel_indices]
                cmap = None
        else:
            cmap = None if img_array.ndim == 3 else 'gray'
        # Normalize based on dtype
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        ax = axes[i]
        ax.imshow(img_array, cmap=cmap)
        ax.axis('off')
        if show_filename:
            ax.set_title(os.path.basename(image_file), color='white', fontsize=fontsize, pad=20)
        # Add scale bar
        ax.plot([10, 10 + scale_bar_length_px], [img_array.shape[0] - 10] * 2, lw=2, color='white')
    # Add channel names at the top if specified
    initial_offset = 0.02  # Starting offset from the left side of the figure
    increment = 0.05  # Fixed increment for each subsequent channel name, adjust based on figure width
    if channel_names:
        current_offset = initial_offset
        for i, channel_name in enumerate(channel_names):
            color = channel_colors[i] if i < len(channel_colors) else 'white'
            fig.text(current_offset, 0.99, channel_name, color=color, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='black', edgecolor='none', pad=3))
            current_offset += increment

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=3)
    if plot:
        plt.show()
    return fig

def _save_scimg_plot(src, nr_imgs=16, channel_indices=[0,1,2], um_per_pixel=0.1, scale_bar_length_um=10, standardize=True, fontsize=8, show_filename=True, channel_names=None, dpi=300, plot=False, i=1, all_folders=1):

    """
    Save and visualize single-cell images.

    Args:
        src (str): The source directory path.
        nr_imgs (int, optional): The number of images to visualize. Defaults to 16.
        channel_indices (list, optional): List of channel indices to visualize. Defaults to [0,1,2].
        um_per_pixel (float, optional): Micrometers per pixel. Defaults to 0.1.
        scale_bar_length_um (float, optional): Length of the scale bar in micrometers. Defaults to 10.
        standardize (bool, optional): Whether to standardize the image sizes. Defaults to True.
        fontsize (int, optional): Font size for the filename. Defaults to 8.
        show_filename (bool, optional): Whether to show the filename on the image. Defaults to True.
        channel_names (list, optional): List of channel names. Defaults to None.
        dpi (int, optional): Dots per inch for the saved image. Defaults to 300.
        plot (bool, optional): Whether to plot the images. Defaults to False.

    Returns:
        None
    """
    from .io import _save_figure
    
    def _visualize_scimgs(src, channel_indices=None, um_per_pixel=0.1, scale_bar_length_um=10, show_filename=True, standardize=True, nr_imgs=None, fontsize=8, channel_names=None, plot=False):
        """
        Visualize single-cell images.

        Args:
            src (str): The source directory path.
            channel_indices (list, optional): List of channel indices to visualize. Defaults to None.
            um_per_pixel (float, optional): Micrometers per pixel. Defaults to 0.1.
            scale_bar_length_um (float, optional): Length of the scale bar in micrometers. Defaults to 10.
            show_filename (bool, optional): Whether to show the filename on the image. Defaults to True.
            standardize (bool, optional): Whether to standardize the image sizes. Defaults to True.
            nr_imgs (int, optional): The number of images to visualize. Defaults to None.
            fontsize (int, optional): Font size for the filename. Defaults to 8.
            channel_names (list, optional): List of channel names. Defaults to None.
            plot (bool, optional): Whether to plot the images. Defaults to False.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plotted images.
        """
        from .utils import _find_similar_sized_images
        def _generate_filelist(src):
            """
            Generate a list of image files in the specified directory.

            Args:
                src (str): The source directory path.

            Returns:
                list: A list of image file paths.

            """
            files = glob.glob(os.path.join(src, '*'))
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'))]
            return image_files

        def _random_sample(file_list, nr_imgs=None):
            """
            Randomly selects a subset of files from the given file list.

            Args:
                file_list (list): A list of file names.
                nr_imgs (int, optional): The number of files to select. If None, all files are selected. Defaults to None.

            Returns:
                list: A list of randomly selected file names.
            """
            if nr_imgs is not None and nr_imgs < len(file_list):
                random.seed(42)
                file_list = random.sample(file_list, nr_imgs)
            return file_list

        image_files = _generate_filelist(src)

        if standardize:
            image_files = _find_similar_sized_images(image_files)

        if nr_imgs is not None:
            image_files = _random_sample(image_files, nr_imgs)

        fig = _plot_images_on_grid(image_files, channel_indices, um_per_pixel, scale_bar_length_um, fontsize, show_filename, channel_names, plot)

        return fig

    fig = _visualize_scimgs(src, channel_indices, um_per_pixel, scale_bar_length_um, show_filename, standardize, nr_imgs, fontsize, channel_names, plot)
    _save_figure(fig, src, text='all_channels')

    for channel in channel_indices:
        channel_indices=[channel]
        fig = _visualize_scimgs(src, channel_indices, um_per_pixel, scale_bar_length_um, show_filename, standardize, nr_imgs, fontsize, channel_names=None, plot=plot)
        _save_figure(fig, src, text=f'channel_{channel}')

    return

def _plot_cropped_arrays(stack, figuresize=20,cmap='inferno'):
    """
    Plot cropped arrays.

    Args:
        stack (ndarray): The array to be plotted.
        figuresize (int, optional): The size of the figure. Defaults to 20.
        cmap (str, optional): The colormap to be used. Defaults to 'inferno'.

    Returns:
        None
    """
    start = time.time()
    dim = stack.shape 
    channel=min(dim)
    if len(stack.shape) == 2:
        f, a = plt.subplots(1, 1,figsize=(figuresize,figuresize))
        a.imshow(stack, cmap=plt.get_cmap(cmap))
        a.set_title('Channel one',size=18)
        a.axis('off')
        f.tight_layout()
        plt.show()
    if len(stack.shape) > 2:
        anr = stack.shape[2]
        f, a = plt.subplots(1, anr,figsize=(figuresize,figuresize))
        for channel in range(anr):
            a[channel].imshow(stack[:,:,channel], cmap=plt.get_cmap(cmap))
            a[channel].set_title('Channel '+str(channel),size=18)
            a[channel].axis('off')
            f.tight_layout()
        plt.show()
    stop = time.time()
    duration = stop - start
    print('plot_cropped_arrays', duration)
    return
    
def _visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df, save, src, name, plot, filenames, object_type, mode='btrack', interactive=False):
    """
    Visualizes and saves a timelapse stack with tracks.

    Args:
        masks (list): List of binary masks representing each frame of the timelapse stack.
        tracks_df (pandas.DataFrame): DataFrame containing track information.
        save (bool): Flag indicating whether to save the timelapse stack.
        src (str): Source file path.
        name (str): Name of the timelapse stack.
        plot (bool): Flag indicating whether to plot the timelapse stack.
        filenames (list): List of filenames corresponding to each frame of the timelapse stack.
        object_type (str): Type of object being tracked.
        mode (str, optional): Tracking mode. Defaults to 'btrack'.
        interactive (bool, optional): Flag indicating whether to display the timelapse stack interactively. Defaults to False.
    """
    
    from .timelapse import _save_mask_timelapse_as_gif
    
    highest_label = max(np.max(mask) for mask in masks)
    # Generate random colors for each label, including the background
    random_colors = np.random.rand(highest_label + 1, 4)
    random_colors[:, 3] = 1  # Full opacity
    random_colors[0] = [0, 0, 0, 1]  # Background color
    cmap = plt.cm.colors.ListedColormap(random_colors)
    # Ensure the normalization range covers all labels
    norm = plt.cm.colors.Normalize(vmin=0, vmax=highest_label)

    # Function to plot a frame and overlay tracks
    def _view_frame_with_tracks(frame=0):
        """
        Display the frame with tracks overlaid.

        Parameters:
        frame (int): The frame number to display.

        Returns:
        None
        """
        fig, ax = plt.subplots(figsize=(50, 50))
        current_mask = masks[frame]
        ax.imshow(current_mask, cmap=cmap, norm=norm)  # Apply both colormap and normalization
        ax.set_title(f'Frame: {frame}')

        # Directly annotate each object with its label number from the mask
        for label_value in np.unique(current_mask):
            if label_value == 0: continue  # Skip background
            y, x = np.mean(np.where(current_mask == label_value), axis=1)
            ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

        # Overlay tracks
        for track in tracks_df['track_id'].unique():
            _track = tracks_df[tracks_df['track_id'] == track]
            ax.plot(_track['x'], _track['y'], '-k', linewidth=1)

        ax.axis('off')
        plt.show()

    if plot:
        if interactive:
            interact(_view_frame_with_tracks, frame=IntSlider(min=0, max=len(masks)-1, step=1, value=0))

    if save:
        # Save as gif
        gif_path = os.path.join(os.path.dirname(src), 'movies', 'gif')
        os.makedirs(gif_path, exist_ok=True)
        save_path_gif = os.path.join(gif_path, f'timelapse_masks_{object_type}_{name}.gif')
        _save_mask_timelapse_as_gif(masks, tracks_df, save_path_gif, cmap, norm, filenames)
        if plot:
            if not interactive:
                _display_gif(save_path_gif)
                
def _display_gif(path):
    """
    Display a GIF image from the given path.

    Parameters:
    path (str): The path to the GIF image file.

    Returns:
    None
    """
    with open(path, 'rb') as file:
        display(ipyimage(file.read()))
        
def _plot_recruitment(df, df_type, channel_of_interest, target, columns=[], figuresize=50):
    """
    Plot recruitment data for different conditions and pathogens.

    Args:
        df (DataFrame): The input DataFrame containing the recruitment data.
        df_type (str): The type of DataFrame (e.g., 'train', 'test').
        channel_of_interest (str): The channel of interest for plotting.
        target (str): The target variable for plotting.
        columns (list, optional): Additional columns to plot. Defaults to an empty list.
        figuresize (int, optional): The size of the figure. Defaults to 50.

    Returns:
        None
    """

    color_list = [(55/255, 155/255, 155/255), 
                  (155/255, 55/255, 155/255), 
                  (55/255, 155/255, 255/255), 
                  (255/255, 55/255, 155/255)]

    sns.set_palette(sns.color_palette(color_list))
    font = figuresize/2
    width=figuresize
    height=figuresize/4

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(width, height))
    sns.barplot(ax=axes[0], data=df, x='condition', y=f'cell_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[0].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[0].set_ylabel(f'cell_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    sns.barplot(ax=axes[1], data=df, x='condition', y=f'nucleus_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[1].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[1].set_ylabel(f'nucleus_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    sns.barplot(ax=axes[2], data=df, x='condition', y=f'cytoplasm_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[2].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[2].set_ylabel(f'cytoplasm_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    sns.barplot(ax=axes[3], data=df, x='condition', y=f'pathogen_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
    axes[3].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[3].set_ylabel(f'pathogen_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    axes[0].legend_.remove()
    axes[1].legend_.remove()
    axes[2].legend_.remove()
    axes[3].legend_.remove()

    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    for i in [0,1,2,3]:
        axes[i].tick_params(axis='both', which='major', labelsize=font)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    columns = columns + ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_periphery_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean']
    columns = columns + [f'pathogen_slope_channel_{channel_of_interest}', f'pathogen_cell_distance_channel_{channel_of_interest}', f'nucleus_cell_distance_channel_{channel_of_interest}']

    width = figuresize*2
    columns_per_row = math.ceil(len(columns) / 2)
    height = (figuresize*2)/columns_per_row

    fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
    axes = axes.flatten()

    print(f'{columns}')

    for i, col in enumerate(columns):

        ax = axes[i]
        sns.barplot(ax=ax, data=df, x='condition', y=f'{col}', hue='pathogen', capsize=.1, ci='sd', dodge=False)
        ax.set_xlabel(f'pathogen {df_type}', fontsize=font)
        ax.set_ylabel(f'{col}', fontsize=int(font*2))
        ax.legend_.remove()
        ax.tick_params(axis='both', which='major', labelsize=font)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if i <= 5:
            ax.set_ylim(1, None)

    for i in range(len(columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
def _plot_controls(df, mask_chans, channel_of_interest, figuresize=5):
    """
    Plot controls for different channels and conditions.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        mask_chans (list): The list of channels to include in the plot.
        channel_of_interest (int): The channel of interest.
        figuresize (int, optional): The size of the figure. Defaults to 5.

    Returns:
        None
    """
    mask_chans.append(channel_of_interest)
    if len(mask_chans) == 4:
        mask_chans = [0,1,2,3]
    if len(mask_chans) == 3:
        mask_chans = [0,1,2]
    if len(mask_chans) == 2:
        mask_chans = [0,1]
    if len(mask_chans) == 1:
        mask_chans = [0]
    controls_cols = []
    for chan in mask_chans:

        controls_cols_c = []
        controls_cols_c.append(f'cell_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'nucleus_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'pathogen_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'cytoplasm_channel_{chan}_mean_intensity')
        controls_cols.append(controls_cols_c)

    unique_conditions = df['condition'].unique().tolist()

    if len(unique_conditions) ==1:
        unique_conditions=unique_conditions+unique_conditions

    fig, axes = plt.subplots(len(unique_conditions), len(mask_chans)+1, figsize=(figuresize*len(mask_chans), figuresize*len(unique_conditions)))

    # Define RGB color tuples (scaled to 0-1 range)
    color_list = [(55/255, 155/255, 155/255), 
                  (155/255, 55/255, 155/255), 
                  (55/255, 155/255, 255/255), 
                  (255/255, 55/255, 155/255)]

    for idx_condition, condition in enumerate(unique_conditions):
        df_temp = df[df['condition'] == condition]
        for idx_channel, control_cols_c in enumerate(controls_cols):
            data = []
            std_dev = []
            for control_col in control_cols_c:
                if control_col in df_temp.columns:
                    mean_intensity = df_temp[control_col].mean()
                    mean_intensity = 0 if np.isnan(mean_intensity) else mean_intensity
                    data.append(mean_intensity)
                    std_dev.append(df_temp[control_col].std())

            current_axis = axes[idx_condition][idx_channel]
            current_axis.bar(["cell", "nucleus", "pathogen", "cytoplasm"], data, yerr=std_dev, 
                             capsize=4, color=color_list)
            current_axis.set_xlabel('Component')
            current_axis.set_ylabel('Mean Intensity')
            current_axis.set_title(f'Condition: {condition} - Channel {idx_channel}')
    plt.tight_layout()
    plt.show()

###################################################
#  Classify
###################################################

def _imshow(img, labels, nrow=20, color='white', fontsize=12):
    """
    Display multiple images in a grid with corresponding labels.

    Args:
        img (list): List of images to display.
        labels (list): List of labels corresponding to each image.
        nrow (int, optional): Number of images per row in the grid. Defaults to 20.
        color (str, optional): Color of the label text. Defaults to 'white'.
        fontsize (int, optional): Font size of the label text. Defaults to 12.
    """
    n_images = len(labels)
    n_col = nrow
    n_row = int(np.ceil(n_images / n_col))
    img_height = img[0].shape[1]
    img_width = img[0].shape[2]
    canvas = np.zeros((img_height * n_row, img_width * n_col, 3))
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < n_images:
                canvas[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = np.transpose(img[idx], (1, 2, 0))        
    plt.figure(figsize=(50, 50))
    plt._imshow(canvas)
    plt.axis("off")
    for i, label in enumerate(labels):
        row = i // n_col
        col = i % n_col
        x = col * img_width + 2
        y = row * img_height + 15
        plt.text(x, y, label, color=color, fontsize=fontsize, fontweight='bold')
    plt.show()
    
def _plot_histograms_and_stats(df):
    conditions = df['condition'].unique()
    
    for condition in conditions:
        subset = df[df['condition'] == condition]
        
        # Calculate the statistics
        mean_pred = subset['pred'].mean()
        over_0_5 = sum(subset['pred'] > 0.5)
        under_0_5 = sum(subset['pred'] <= 0.5)

        # Print the statistics
        print(f"Condition: {condition}")
        print(f"Number of rows: {len(subset)}")
        print(f"Mean of pred: {mean_pred}")
        print(f"Count of pred values over 0.5: {over_0_5}")
        print(f"Count of pred values under 0.5: {under_0_5}")
        print(f"Percent positive: {(over_0_5/(over_0_5+under_0_5))*100}")
        print(f"Percent negative: {(under_0_5/(over_0_5+under_0_5))*100}")
        print('-'*40)
        
        # Plot the histogram
        plt.figure(figsize=(10,6))
        plt.hist(subset['pred'], bins=30, edgecolor='black')
        plt.axvline(mean_pred, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_pred:.2f}")
        plt.title(f'Histogram for pred - Condition: {condition}')
        plt.xlabel('Pred Value')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

def _show_residules(model):

    # Get the residuals
    residuals = model.resid

    # Histogram of residuals
    plt.hist(residuals, bins=30)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.show()

    # QQ plot
    sm.qqplot(residuals, fit=True, line='45')
    plt.title('QQ Plot')
    plt.show()

    # Residuals vs. Fitted values
    plt.scatter(model.fittedvalues, residuals)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values')
    plt.axhline(y=0, color='red')
    plt.show()

    # Shapiro-Wilk test for normality
    W, p_value = stats.shapiro(residuals)
    print(f'Shapiro-Wilk Test W-statistic: {W}, p-value: {p_value}')
    
def _reg_v_plot(df, grouping, variable, plate_number):
    df['-log10(p)'] = -np.log10(df['p'])

    # Create the volcano plot
    plt.figure(figsize=(40, 30))
    sc = plt.scatter(df['effect'], df['-log10(p)'], c=np.sign(df['effect']), cmap='coolwarm')
    plt.title('Volcano Plot', fontsize=12)
    plt.xlabel('Coefficient', fontsize=12)
    plt.ylabel('-log10(P-value)', fontsize=12)

    # Add text for specified points
    for idx, row in df.iterrows():
        if row['p'] < 0.05:# and abs(row['effect']) > 0.1:
            plt.text(row['effect'], -np.log10(row['p']), idx, fontsize=12, ha='center', va='bottom', color='black')

    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--')  # line for p=0.05
    plt.show()
    
def generate_plate_heatmap(df, plate_number, variable, grouping, min_max):
    df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
    df['plate'], df['row'], df['col'] = zip(*df['prc'].str.split('_'))
    
    # Filtering the dataframe based on the plate_number
    df = df[df['plate'] == plate_number].copy()  # Create another copy after filtering
    
    # Ensure proper ordering
    row_order = [f'r{i}' for i in range(1, 17)]
    col_order = [f'c{i}' for i in range(1, 28)]  # Exclude c15 as per your earlier code
    
    df['row'] = pd.Categorical(df['row'], categories=row_order, ordered=True)
    df['col'] = pd.Categorical(df['col'], categories=col_order, ordered=True)
    
    # Explicitly set observed=True to avoid FutureWarning
    grouped = df.groupby(['row', 'col'], observed=True)  
    
    if grouping == 'mean':
        plate = grouped[variable].mean().reset_index()
    elif grouping == 'sum':
        plate = grouped[variable].sum().reset_index()
    elif grouping == 'count':
        plate = grouped[variable].count().reset_index()
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")
        
    plate_map = pd.pivot_table(plate, values=variable, index='row', columns='col').fillna(0)
    
    if min_max == 'all':
        min_max = [plate_map.min().min(), plate_map.max().max()]
    elif min_max == 'allq':
        min_max = np.quantile(plate_map.values, [0.2, 0.98])
    elif min_max == 'plate':
        min_max = [plate_map.min().min(), plate_map.max().max()]
        
    return plate_map, min_max

def _plot_plates(df, variable, grouping, min_max, cmap):
    plates = df['prc'].str.split('_', expand=True)[0].unique()
    n_rows, n_cols = (len(plates) + 3) // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 5 * n_rows))
    ax = ax.flatten()

    for index, plate in enumerate(plates):
        plate_map, min_max_values = generate_plate_heatmap(df, plate, variable, grouping, min_max)
        sns.heatmap(plate_map, cmap=cmap, vmin=0, vmax=2, ax=ax[index])
        ax[index].set_title(plate)
        
    for i in range(len(plates), n_rows * n_cols):
        fig.delaxes(ax[i])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.show()
    return

#from finetune cellpose
#def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
#    paths = []
#    for file in os.listdir(src):
#        if file.endswith('.tif') or file.endswith('.tiff'):
#            path = os.path.join(src, file)
#            paths.append(path)
#    paths = random.sample(paths, nr)
#    for path in paths:
#        print(f'Image path:{path}')
#        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#        if normalize:
#            img = normalize_to_dtype(array=img, q1=q1, q2=q2)
#        dim = img.shape
#        if len(img.shape) > 2:
#            array_nr = img.shape[2]
#            fig, axs = plt.subplots(1, array_nr, figsize=(figuresize, figuresize))
#            for channel in range(array_nr):
#                i = np.take(img, [channel], axis=2)
#                axs[channel].imshow(i, cmap=plt.get_cmap(cmap))
#                axs[channel].set_title('Channel '+str(channel), size=24)
#                axs[channel].axis('off')
#        else:
#            fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
#            ax.imshow(img, cmap=plt.get_cmap(cmap))
#            ax.set_title('Channel 0', size=24)
#            ax.axis('off')
#        fig.tight_layout()
#        plt.show()
#    return

def print_mask_and_flows(stack, mask, flows, overlay=False):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))  # Adjust subplot layout
    
    if stack.shape[-1] == 1:
        stack = np.squeeze(stack)
    
    # Display original image or its first channel
    if stack.ndim == 2:
        axs[0].imshow(stack, cmap='gray')
    elif stack.ndim == 3:
        axs[0].imshow(stack)
    else:
        raise ValueError("Unexpected stack dimensionality.")

    axs[0].set_title('Original Image')
    axs[0].axis('off')
    

    # Overlay mask on original image if overlay is True
    if overlay:
        mask_cmap = generate_mask_random_cmap(mask)  # Generate random colormap for mask
        mask_overlay = np.ma.masked_where(mask == 0, mask)  # Mask background
        outlines = find_boundaries(mask, mode='thick')  # Find mask outlines

        if stack.ndim == 2 or stack.ndim == 3:
            axs[1].imshow(stack, cmap='gray' if stack.ndim == 2 else None)
            axs[1].imshow(mask_overlay, cmap=mask_cmap, alpha=0.5)  # Overlay mask
            axs[1].contour(outlines, colors='r', linewidths=2)  # Add red outlines with thickness 2
    else:
        axs[1].imshow(mask, cmap='gray')
    
    axs[1].set_title('Mask with Overlay' if overlay else 'Mask')
    axs[1].axis('off')

    # Display flow image or its first channel
    if flows and isinstance(flows, list) and flows[0].ndim in [2, 3]:
        flow_image = flows[0]
        if flow_image.ndim == 3:
            flow_image = flow_image[:, :, 0]  # Use first channel for 3D
        axs[2].imshow(flow_image, cmap='jet')
    else:
        raise ValueError("Unexpected flow dimensionality or structure.")
    
    axs[2].set_title('Flows')
    axs[2].axis('off')

    fig.tight_layout()
    plt.show()
    
def plot_resize(images, resized_images, labels, resized_labels):
    # Display an example image and label before and after resizing
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    
    # Check if the image is grayscale; if so, add a colormap and keep dimensions correct
    if images[0].ndim == 2:  # Grayscale image
        ax[0, 0].imshow(images[0], cmap='gray')
    else:  # RGB or RGBA image
        ax[0, 0].imshow(images[0])
    ax[0, 0].set_title('Original Image')

    if resized_images[0].ndim == 2:  # Grayscale image
        ax[0, 1].imshow(resized_images[0], cmap='gray')
    else:  # RGB or RGBA image
        ax[0, 1].imshow(resized_images[0])
    ax[0, 1].set_title('Resized Image')

    # Assuming labels are always grayscale (most common scenario)
    ax[1, 0].imshow(labels[0], cmap='gray')
    ax[1, 0].set_title('Original Label')
    ax[1, 1].imshow(resized_labels[0], cmap='gray')
    ax[1, 1].set_title('Resized Label')
    plt.show()
    
def normalize_and_visualize(image, normalized_image, title=""):
    """Utility function for visualization"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if image.ndim == 3:  # Multi-channel image
        ax[0].imshow(np.mean(image, axis=-1), cmap='gray')  # Display the average over channels for visualization
    else:  # Grayscale image
        ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original " + title)
    ax[0].axis('off')

    if normalized_image.ndim == 3:
        ax[1].imshow(np.mean(normalized_image, axis=-1), cmap='gray')  # Similarly, display the average over channels
    else:
        ax[1].imshow(normalized_image, cmap='gray')
    ax[1].set_title("Normalized " + title)
    ax[1].axis('off')
    
    plt.show()
    
def visualize_masks(mask1, mask2, mask3, title="Masks Comparison"):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax, mask, title in zip(axs, [mask1, mask2, mask3], ['Mask 1', 'Mask 2', 'Mask 3']):
        cmap = generate_mask_random_cmap(mask)
        # If the mask is binary, we can skip normalization
        if np.isin(mask, [0, 1]).all():
            ax.imshow(mask, cmap=cmap)
        else:
            # Normalize the image for displaying purposes
            norm = plt.Normalize(vmin=0, vmax=mask.max())
            ax.imshow(mask, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()
    
def plot_comparison_results(comparison_results):
    df = pd.DataFrame(comparison_results)
    df_melted = pd.melt(df, id_vars=['filename'], var_name='metric', value_name='value')
    df_jaccard = df_melted[df_melted['metric'].str.contains('jaccard')]
    df_dice = df_melted[df_melted['metric'].str.contains('dice')]
    df_boundary_f1 = df_melted[df_melted['metric'].str.contains('boundary_f1')]
    df_ap = df_melted[df_melted['metric'].str.contains('average_precision')]
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    
    # Jaccard Index Plot
    sns.boxplot(data=df_jaccard, x='metric', y='value', ax=axs[0], color='lightgrey')
    sns.stripplot(data=df_jaccard, x='metric', y='value', ax=axs[0], jitter=True, alpha=0.6)
    axs[0].set_title('Jaccard Index by Comparison')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[0].set_xlabel('Comparison')
    axs[0].set_ylabel('Jaccard Index')
    # Dice Coefficient Plot
    sns.boxplot(data=df_dice, x='metric', y='value', ax=axs[1], color='lightgrey')
    sns.stripplot(data=df_dice, x='metric', y='value', ax=axs[1], jitter=True, alpha=0.6)
    axs[1].set_title('Dice Coefficient by Comparison')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[1].set_xlabel('Comparison')
    axs[1].set_ylabel('Dice Coefficient')
    # Border F1 scores
    sns.boxplot(data=df_boundary_f1, x='metric', y='value', ax=axs[2], color='lightgrey')
    sns.stripplot(data=df_boundary_f1, x='metric', y='value', ax=axs[2], jitter=True, alpha=0.6)
    axs[2].set_title('Boundary F1 Score by Comparison')
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[2].set_xlabel('Comparison')
    axs[2].set_ylabel('Boundary F1 Score')
    # AP scores plot
    sns.boxplot(data=df_ap, x='metric', y='value', ax=axs[3], color='lightgrey')
    sns.stripplot(data=df_ap, x='metric', y='value', ax=axs[3], jitter=True, alpha=0.6)
    axs[3].set_title('Average Precision by Comparison')
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[3].set_xlabel('Comparison')
    axs[3].set_ylabel('Average Precision')
    
    plt.tight_layout()
    plt.show()
    return fig
