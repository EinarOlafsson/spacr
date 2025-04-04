import os, random, cv2, glob, math, torch, skimage

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import scipy.ndimage as ndi
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import imageio.v2 as imageio
from IPython.display import display
from skimage.segmentation import find_boundaries
from skimage import measure
from skimage.measure import find_contours, label, regionprops
from skimage.segmentation import mark_boundaries
from skimage.transform import resize as sk_resize
import scikit_posthocs as sp
from scipy.stats import chi2_contingency
import tifffile as tiff

from scipy.stats import normaltest, ttest_ind, mannwhitneyu, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind, mannwhitneyu, levene, wilcoxon, kruskal, normaltest, shapiro
import itertools
import pingouin as pg

from ipywidgets import IntSlider, interact
from IPython.display import Image as ipyimage

import matplotlib.patches as patches
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2

#filter_dict={'cell':[(0,100000), (0, 65000)],'nucleus':[(3000,100000), (1500, 65000)],'pathogen':[(500,100000), (0, 65000)]}
def plot_image_mask_overlay(
    file,
    channels,
    cell_channel,
    nucleus_channel,
    pathogen_channel,
    figuresize=10,
    percentiles=(2, 98),
    thickness=3,
    save_pdf=True,
    mode='outlines',
    export_tiffs=False,
    all_on_all=False,
    all_outlines=False,
    filter_dict=None
):
    """Plot image and mask overlays."""

    def random_color_cmap(n_labels, seed=None):
        """Generates a random color map for a given number of labels."""
        if seed is not None:
            np.random.seed(seed)
        rand_colors = np.random.rand(n_labels, 3)
        rand_colors = np.vstack([[0, 0, 0], rand_colors])  # Ensure background is black
        cmap = ListedColormap(rand_colors)
        return cmap

    def _plot_merged_plot(
        image,
        outlines,
        outline_colors,
        figuresize,
        thickness,
        percentiles,
        mode='outlines',
        all_on_all=False,
        all_outlines=False,
        channels=None,
        cell_channel=None,
        nucleus_channel=None,
        pathogen_channel=None,
        cell_outlines=None,
        nucleus_outlines=None,
        pathogen_outlines=None,
        save_pdf=True
    ):
        """Plot the merged plot with overlay, image channels, and masks."""

        def _generate_colored_mask(mask, cmap):
            """Generate a colored mask using the given colormap."""
            mask_norm = mask / (mask.max() + 1e-5)  # Normalize mask
            colored_mask = cmap(mask_norm)
            colored_mask[..., 3] = np.where(mask > 0, 1, 0)  # Alpha channel
            return colored_mask

        def _overlay_mask(image, mask):
            """Overlay the colored mask onto the original image."""
            combined = np.clip(image * (1 - mask[..., 3:]) + mask[..., :3] * mask[..., 3:], 0, 1)
            return combined

        def _normalize_image(image, percentiles):
            """Normalize the image based on given percentiles."""
            v_min, v_max = np.percentile(image, percentiles)
            image_normalized = np.clip((image - v_min) / (v_max - v_min + 1e-5), 0, 1)
            return image_normalized

        def _generate_contours(mask):
            """Generate contours from the mask using OpenCV."""
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            return contours

        def _apply_contours(image, mask, color, thickness):
            """Apply contours to the image."""
            unique_labels = np.unique(mask)
            for label in unique_labels:
                if label == 0:
                    continue  # Skip background
                label_mask = (mask == label).astype(np.uint8)
                contours = _generate_contours(label_mask)
                cv2.drawContours(
                    image, contours, -1, mpl.colors.to_rgb(color), thickness
                )
            return image

        num_channels = image.shape[-1]
        fig, ax = plt.subplots(1, num_channels + 1, figsize=(4 * figuresize, figuresize))

        # Identify channels without associated outlines
        channels_with_outlines = []
        if cell_channel is not None:
            channels_with_outlines.append(cell_channel)
        if nucleus_channel is not None:
            channels_with_outlines.append(nucleus_channel)
        if pathogen_channel is not None:
            channels_with_outlines.append(pathogen_channel)

        for v in range(num_channels):
            channel_image = image[..., v]
            channel_image_normalized = _normalize_image(channel_image, percentiles)
            channel_image_rgb = np.dstack([channel_image_normalized] * 3)

            current_channel = channels[v]

            if all_on_all:
                # Apply all outlines to all channels
                for outline, color in zip(outlines, outline_colors):
                    if mode == 'outlines':
                        channel_image_rgb = _apply_contours(
                            channel_image_rgb, outline, color, thickness
                        )
                    else:
                        cmap = random_color_cmap(int(outline.max() + 1), random.randint(0, 100))
                        mask = _generate_colored_mask(outline, cmap)
                        channel_image_rgb = _overlay_mask(channel_image_rgb, mask)
            elif current_channel in channels_with_outlines:
                # Apply only the relevant outline to each channel
                outline = None
                color = None

                if current_channel == cell_channel and cell_outlines is not None:
                    outline = cell_outlines
                elif current_channel == nucleus_channel and nucleus_outlines is not None:
                    outline = nucleus_outlines
                elif current_channel == pathogen_channel and pathogen_outlines is not None:
                    outline = pathogen_outlines

                if outline is not None:
                    if mode == 'outlines':
                        # Use magenta color when all_on_all=False
                        channel_image_rgb = _apply_contours(
                            channel_image_rgb, outline, '#FF00FF', thickness
                        )
                    else:
                        cmap = random_color_cmap(int(outline.max() + 1), random.randint(0, 100))
                        mask = _generate_colored_mask(outline, cmap)
                        channel_image_rgb = _overlay_mask(channel_image_rgb, mask)
            else:
                # Channel without associated outlines
                if all_outlines:
                    # Apply all outlines with specified colors
                    for outline, color in zip(outlines, ['blue', 'red', 'green']):
                        if mode == 'outlines':
                            channel_image_rgb = _apply_contours(
                                channel_image_rgb, outline, color, thickness
                            )
                        else:
                            cmap = random_color_cmap(int(outline.max() + 1), random.randint(0, 100))
                            mask = _generate_colored_mask(outline, cmap)
                            channel_image_rgb = _overlay_mask(channel_image_rgb, mask)

            ax[v].imshow(channel_image_rgb)
            ax[v].set_title(f'Image - Channel {current_channel}')

        # Create an image combining all objects filled with colors
        combined_mask = np.zeros_like(outlines[0])
        for outline in outlines:
            combined_mask = np.maximum(combined_mask, outline)

        cmap = random_color_cmap(int(combined_mask.max() + 1), random.randint(0, 100))
        mask = _generate_colored_mask(combined_mask, cmap)
        blank_image = np.zeros((*combined_mask.shape, 3))
        filled_image = _overlay_mask(blank_image, mask)

        ax[-1].imshow(filled_image)
        ax[-1].set_title('Combined Objects Image')

        plt.tight_layout()

        # Save the figure as a PDF
        if save_pdf:
            pdf_dir = os.path.join(
                os.path.dirname(os.path.dirname(file)), 'results', 'overlay'
            )
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(
                pdf_dir, os.path.basename(file).replace('.npy', '.pdf')
            )
            fig.savefig(pdf_path, format='pdf')

        plt.show()
        return fig

    def _save_channels_as_tiff(stack, save_dir, filename):
        """Save each channel in the stack as a grayscale TIFF."""
        os.makedirs(save_dir, exist_ok=True)
        for i in range(stack.shape[-1]):
            channel = stack[..., i]
            tiff_path = os.path.join(save_dir, f"{filename}_channel_{i}.tiff")
            tiff.imwrite(tiff_path, channel.astype(np.uint16), photometric='minisblack')
            print(f"Saved {tiff_path}")

    def _filter_object(mask, intensity_image, min_max_area=(0, 10000000), min_max_intensity=(0, 65000), type_='object'):
        """
        Filter objects in a mask based on their area (size) and mean intensity.

        Args:
            mask (ndarray): The input mask.
            intensity_image (ndarray): The corresponding intensity image.
            min_max_area (tuple): A tuple (min_area, max_area) specifying the minimum and maximum area thresholds.
            min_max_intensity (tuple): A tuple (min_intensity, max_intensity) specifying the minimum and maximum intensity thresholds.

        Returns:
            ndarray: The filtered mask.
        """
        original_dtype = mask.dtype
        mask_int = mask.astype(np.int64)
        intensity_image = intensity_image.astype(np.float64)
        # Compute properties for each labeled object
        unique_labels = np.unique(mask_int)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        num_objects_before = len(unique_labels)

        # Initialize lists to store area and intensity for each object
        areas = []
        mean_intensities = []
        labels_to_keep = []

        for label in unique_labels:
            label_mask = (mask_int == label)
            area = np.sum(label_mask)
            mean_intensity = np.mean(intensity_image[label_mask])

            areas.append(area)
            mean_intensities.append(mean_intensity)

            # Check if the object meets both area and intensity criteria
            if (min_max_area[0] <= area <= min_max_area[1]) and (min_max_intensity[0] <= mean_intensity <= min_max_intensity[1]):
                labels_to_keep.append(label)

        # Convert lists to numpy arrays for easier computation
        areas = np.array(areas)
        mean_intensities = np.array(mean_intensities)
        num_objects_after = len(labels_to_keep)
        # Compute average area and intensity before and after filtering
        avg_area_before = areas.mean() if num_objects_before > 0 else 0
        avg_intensity_before = mean_intensities.mean() if num_objects_before > 0 else 0
        areas_after = areas[np.isin(unique_labels, labels_to_keep)]
        mean_intensities_after = mean_intensities[np.isin(unique_labels, labels_to_keep)]
        avg_area_after = areas_after.mean() if num_objects_after > 0 else 0
        avg_intensity_after = mean_intensities_after.mean() if num_objects_after > 0 else 0
        print(f"Before filtering {type_}: {num_objects_before} objects")
        print(f"Average area {type_}: {avg_area_before:.2f} pixels, Average intensity: {avg_intensity_before:.2f}")
        print(f"After filtering {type_}: {num_objects_after} objects")
        print(f"Average area {type_}: {avg_area_after:.2f} pixels, Average intensity: {avg_intensity_after:.2f}")
        mask_filtered = np.zeros_like(mask_int)
        for label in labels_to_keep:
            mask_filtered[mask_int == label] = label
        mask_filtered = mask_filtered.astype(original_dtype)
        return mask_filtered

    stack = np.load(file)

    if export_tiffs:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(file)),
            'results',
            os.path.splitext(os.path.basename(file))[0],
            'tiff'
        )
        filename = os.path.splitext(os.path.basename(file))[0]
        _save_channels_as_tiff(stack, save_dir, filename)

    # Convert to float for normalization and ensure correct handling of arrays
    if stack.dtype in (np.uint16, np.uint8):
        stack = stack.astype(np.float32)

    image = stack[..., channels]
    outlines = []
    outline_colors = []

    # Define variables to hold individual outlines
    cell_outlines = None
    nucleus_outlines = None
    pathogen_outlines = None

    if pathogen_channel is not None:
        pathogen_mask_dim = -1 
        pathogen_outlines = np.take(stack, pathogen_mask_dim, axis=2)
        if not filter_dict is None:
            pathogen_intensity = np.take(stack, pathogen_channel, axis=2)
            pathogen_outlines = _filter_object(pathogen_outlines, pathogen_intensity, filter_dict['pathogen'][0], filter_dict['pathogen'][1], type_='pathogen')
        
        outlines.append(pathogen_outlines)
        outline_colors.append('green')  

    if nucleus_channel is not None:
        nucleus_mask_dim = -2 if pathogen_channel is not None else -1
        nucleus_outlines = np.take(stack, nucleus_mask_dim, axis=2)
        if not filter_dict is None:
            nucleus_intensity = np.take(stack, nucleus_channel, axis=2)
            nucleus_outlines = _filter_object(nucleus_outlines, nucleus_intensity, filter_dict['nucleus'][0], filter_dict['nucleus'][1], type_='nucleus')
        outlines.append(nucleus_outlines)
        outline_colors.append('blue')  

    if cell_channel is not None:
        if nucleus_channel is not None and pathogen_channel is not None:
            cell_mask_dim = -3
        elif nucleus_channel is not None or pathogen_channel is not None:
            cell_mask_dim = -2
        else:
            cell_mask_dim = -1
        cell_outlines = np.take(stack, cell_mask_dim, axis=2)
        if not filter_dict is None:
            cell_intensity = np.take(stack, cell_channel, axis=2)
            cell_outlines = _filter_object(cell_outlines, cell_intensity, filter_dict['cell'][0], filter_dict['cell'][1], type_='cell')
        outlines.append(cell_outlines)
        outline_colors.append('red')

    fig = _plot_merged_plot(
        image=image,
        outlines=outlines,
        outline_colors=outline_colors,
        figuresize=figuresize,
        thickness=thickness,
        percentiles=percentiles,  # Pass percentiles to the plotting function
        mode=mode,
        all_on_all=all_on_all,
        all_outlines=all_outlines,
        channels=channels,
        cell_channel=cell_channel,
        nucleus_channel=nucleus_channel,
        pathogen_channel=pathogen_channel,
        cell_outlines=cell_outlines,
        nucleus_outlines=nucleus_outlines,
        pathogen_outlines=pathogen_outlines,
        save_pdf=save_pdf
    )

    return fig

def plot_masks(batch, masks, flows, cmap='inferno', figuresize=10, nr=1, file_type='.npz', print_object_number=True):
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

def plot_images_and_arrays(folders, lower_percentile=1, upper_percentile=99, threshold=1000, extensions=['.npy', '.tif', '.tiff', '.png'], overlay=False, max_nr=None, randomize=True):
    
    """
    Plot images and arrays from the given folders.

    Args:
        folders (list): A list of folder paths containing the images and arrays.
        lower_percentile (int, optional): The lower percentile for image normalization. Defaults to 1.
        upper_percentile (int, optional): The upper percentile for image normalization. Defaults to 99.
        threshold (int, optional): The threshold for determining whether to display an image as a mask or normalize it. Defaults to 1000.
        extensions (list, optional): A list of file extensions to consider. Defaults to ['.npy', '.tif', '.tiff', '.png'].
        overlay (bool, optional): If True, overlay the outlines of the objects on the image. Defaults to False.
    """

    def normalize_image(image, lower=1, upper=99):
        p2, p98 = np.percentile(image, (lower, upper))
        return np.clip((image - p2) / (p98 - p2), 0, 1)

    def find_files(folders, extensions=['.npy', '.tif', '.tiff', '.png']):
        file_dict = {}

        for folder in folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        file_name_wo_ext = os.path.splitext(file)[0]
                        file_path = os.path.join(root, file)
                        if file_name_wo_ext not in file_dict:
                            file_dict[file_name_wo_ext] = {}
                        file_dict[file_name_wo_ext][folder] = file_path

        # Filter out files that don't have paths in all folders
        filtered_dict = {k: v for k, v in file_dict.items() if len(v) == len(folders)}
        return filtered_dict

    def plot_from_file_dict(file_dict, threshold=1000, lower_percentile=1, upper_percentile=99, overlay=False, save=False):
        """
        Plot images and arrays from the given file dictionary.

        Args:
            file_dict (dict): A dictionary containing the file paths for each image or array.
            threshold (int, optional): The threshold for determining whether to display an image as a mask or normalize it. Defaults to 1000.
            lower_percentile (int, optional): The lower percentile for image normalization. Defaults to 1.
            upper_percentile (int, optional): The upper percentile for image normalization. Defaults to 99.
            overlay (bool, optional): If True, overlay the outlines of the objects on the image. Defaults to False.
        """

        for filename, folder_paths in file_dict.items():
            image_data = None
            mask_data = None

            for folder, path in folder_paths.items():
                if path.endswith('.npy'):
                    data = np.load(path)
                elif path.endswith('.tif') or path.endswith('.tiff'):
                    data = imageio.imread(path)
                else:
                    continue

                unique_values = np.unique(data)

                if len(unique_values) > threshold:
                    image_data = normalize_image(data, lower_percentile, upper_percentile)
                else:
                    mask_data = data

            if image_data is not None and mask_data is not None:
                fig, axes = plt.subplots(1, 2, figsize=(15, 7))
                
                # Display the mask with random colormap
                cmap = random_cmap(num_objects=len(np.unique(mask_data)))
                axes[0].imshow(mask_data, cmap=cmap)
                axes[0].set_title(f"{filename} - Mask")
                axes[0].axis('off')

                # Display the normalized image
                axes[1].imshow(image_data, cmap='gray')
                if overlay:
                    labeled_mask = label(mask_data)
                    for region in regionprops(labeled_mask):
                        if region.image.shape[0] >= 2 and region.image.shape[1] >= 2:
                            contours = find_contours(region.image, 0.75)
                            for contour in contours:
                                # Adjust contour coordinates relative to the full image
                                contour[:, 0] += region.bbox[0]
                                contour[:, 1] += region.bbox[1]
                                axes[1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='magenta')

                axes[1].set_title(f"{filename} - Normalized Image")
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()

                if save:
                    save_path = os.path.join(folder,f"{filename}.png")
                    plt.savefig(save_path)

    if overlay:
        print(f'Overlay will only work on the first two folders in the list')

    file_dict = find_files(folders, extensions)
    items = list(file_dict.items())
    if randomize:
        random.shuffle(items)
    if isinstance(max_nr, (int, float)):
        items = items[:int(max_nr)]
    file_dict = dict(items)

    plot_from_file_dict(file_dict, threshold, lower_percentile, upper_percentile, overlay, save=False)
    return

def _filter_objects_in_plot(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, mask_dims, filter_min_max, nuclei_limit, pathogen_limit):
    """
    Filters objects in a plot based on various criteria.

    Args:
        stack (numpy.ndarray): The input stack of masks.
        cell_mask_dim (int): The dimension index of the cell mask.
        nucleus_mask_dim (int): The dimension index of the nucleus mask.
        pathogen_mask_dim (int): The dimension index of the pathogen mask.
        mask_dims (list): A list of dimension indices for additional masks.
        filter_min_max (list): A list of minimum and maximum area values for each mask.
        nuclei_limit (bool): Whether to include multinucleated cells.
        pathogen_limit (bool): Whether to include multiinfected cells.

    Returns:
        numpy.ndarray: The filtered stack of masks.
    """
    from .utils import _remove_outside_objects, _remove_multiobject_cells
    
    stack = _remove_outside_objects(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)

    for i, mask_dim in enumerate(mask_dims):
        if not filter_min_max is None:
            min_max = filter_min_max[i]
        else:
            min_max = [0, 100000000]

        mask = np.take(stack, mask_dim, axis=2)
        props = measure.regionprops_table(mask, properties=['label', 'area'])
        #props = measure.regionprops_table(mask, intensity_image=intensity_image, properties=['label', 'area', 'mean_intensity'])
        avg_size_before = np.mean(props['area'])
        total_count_before = len(props['label'])

        if not filter_min_max is None:
            valid_labels = props['label'][np.logical_and(props['area'] > min_max[0], props['area'] < min_max[1])]  
            stack[:, :, mask_dim] = np.isin(mask, valid_labels) * mask  

        props_after = measure.regionprops_table(stack[:, :, mask_dim], properties=['label', 'area']) 
        avg_size_after = np.mean(props_after['area'])
        total_count_after = len(props_after['label'])

        if mask_dim == cell_mask_dim:
            if nuclei_limit is False and nucleus_mask_dim is not None:
                stack = _remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
            if pathogen_limit is False and cell_mask_dim is not None and pathogen_mask_dim is not None:
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

def plot_arrays(src, figuresize=10, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
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
            img = normalize_to_dtype(array=img, p1=q1, p2=q2)
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

def _normalize_and_outline(image, remove_background, normalize, normalization_percentiles, overlay, overlay_chans, mask_dims, outline_colors, outline_thickness):
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
    from .utils import normalize_to_dtype, _outline_and_overlay, _gen_rgb_image

    if remove_background:
        backgrounds = np.percentile(image, 1, axis=(0, 1))
        backgrounds = backgrounds[:, np.newaxis, np.newaxis]
        mask = np.zeros_like(image, dtype=bool)
        for chan_index in range(image.shape[-1]):
            if chan_index not in mask_dims:
                mask[:, :, chan_index] = image[:, :, chan_index] < backgrounds[chan_index]
        image[mask] = 0

    if normalize:
        image = normalize_to_dtype(array=image, p1=normalization_percentiles[0], p2=normalization_percentiles[1])
    else:
        image = normalize_to_dtype(array=image, p1=0, p2=100)

    rgb_image = _gen_rgb_image(image, channels=overlay_chans)

    if overlay:
        overlayed_image, outlines, image = _outline_and_overlay(image, rgb_image, mask_dims, outline_colors, outline_thickness)

        return overlayed_image, image, outlines
    else:
        # Remove mask_dims from image
        channels_to_keep = [i for i in range(image.shape[-1]) if i not in mask_dims]
        image = np.take(image, channels_to_keep, axis=-1)
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

def plot_merged(src, settings):
    """
    Plot the merged images after applying various filters and modifications.

    Args:
        src (path): Path to folder with images.
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
        settings['pathogen_limit'] = True

    for file in os.listdir(src):
        path = os.path.join(src, file)
        stack = np.load(path)
        print(f'Loaded: {path}')
        if settings['pathogen_limit'] > 0:
            if settings['pathogen_mask_dim'] is not None and settings['cell_mask_dim'] is not None:
                stack = _remove_noninfected(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'])

        if settings['pathogen_limit'] is not True or settings['nuclei_limit'] is not True or settings['filter_min_max'] is not None:
            stack = _filter_objects_in_plot(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], mask_dims, settings['filter_min_max'], settings['nuclei_limit'], settings['pathogen_limit'])

        overlayed_image, image, outlines = _normalize_and_outline(image=stack, 
                                                                  remove_background=settings['remove_background'],
                                                                  normalize=settings['normalize'],
                                                                  normalization_percentiles=settings['normalization_percentiles'],
                                                                  overlay=settings['overlay'],
                                                                  overlay_chans=settings['overlay_chans'],
                                                                  mask_dims=mask_dims,
                                                                  outline_colors=outline_colors,
                                                                  outline_thickness=settings['outline_thickness'])
        if index < settings['nr']:
            index += 1
            fig = _plot_merged_plot(overlay=settings['overlay'],
                                    image=image,
                                    stack=stack,
                                    mask_dims=mask_dims,
                                    figuresize=settings['figuresize'],
                                    overlayed_image=overlayed_image,
                                    outlines=outlines,
                                    cmap=settings['cmap'],
                                    outline_colors=outline_colors,
                                    print_object_number=settings['print_object_number'])
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

def _plot_cropped_arrays(stack, filename, figuresize=10, cmap='inferno', threshold=500):
    """
    Plot cropped arrays.

    Args:
        stack (ndarray): The array to be plotted.
        figuresize (int, optional): The size of the figure. Defaults to 20.
        cmap (str, optional): The colormap to be used. Defaults to 'inferno'.
        threshold (int, optional): The threshold for the number of unique intensity values. Defaults to 1000.

    Returns:
        None
    """
    #start = time.time()
    dim = stack.shape
    
    def plot_single_array(array, ax, title, chosen_cmap):
        unique_values = np.unique(array)
        num_unique_values = len(unique_values)
        
        if num_unique_values <= threshold:
            chosen_cmap = _generate_mask_random_cmap(array)
            title = f'{title}, {num_unique_values} (obj.)'
        
        ax.imshow(array, cmap=chosen_cmap)
        ax.set_title(title, size=18)
        ax.axis('off')

    if len(dim) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
        plot_single_array(stack, ax, 'Channel one', plt.get_cmap(cmap))
        fig.tight_layout()
        plt.show()
    elif len(dim) > 2:
        num_channels = dim[2]
        fig, axs = plt.subplots(1, num_channels, figsize=(figuresize, figuresize))
        for channel in range(num_channels):
            plot_single_array(stack[:, :, channel], axs[channel], f'C. {channel}', plt.get_cmap(cmap))
        fig.tight_layout()    
    print(f'{filename}')
    return fig
    
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
    
    from .io import _save_mask_timelapse_as_gif
    
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

def _plot_recruitment_v2(df, df_type, channel_of_interest, columns=[], figuresize=10):
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

    # Create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(width, height))

    # Plot for 'cell_channel' on axes[0]
    plotter_cell = spacrGraph(df,grouping_column='condition', data_column=f'cell_channel_{channel_of_interest}_mean_intensity')
    plotter_cell.create_plot(ax=axes[0])
    axes[0].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[0].set_ylabel(f'cell_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    # Plot for 'nucleus_channel' on axes[1]
    plotter_nucleus = spacrGraph(df,grouping_column='condition', data_column=f'nucleus_channel_{channel_of_interest}_mean_intensity')
    plotter_nucleus.create_plot(ax=axes[1])
    axes[1].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[1].set_ylabel(f'nucleus_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    # Plot for 'cytoplasm_channel' on axes[2]
    plotter_cytoplasm = spacrGraph(df, grouping_column='condition', data_column=f'cytoplasm_channel_{channel_of_interest}_mean_intensity')
    plotter_cytoplasm.create_plot(ax=axes[2])
    axes[2].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[2].set_ylabel(f'cytoplasm_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    # Plot for 'pathogen_channel' on axes[3]
    plotter_pathogen = spacrGraph(df, grouping_column='condition', data_column=f'pathogen_channel_{channel_of_interest}_mean_intensity')
    plotter_pathogen.create_plot(ax=axes[3])
    axes[3].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[3].set_ylabel(f'pathogen_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    #axes[0].legend_.remove()
    #axes[1].legend_.remove()
    #axes[2].legend_.remove()
    #axes[3].legend_.remove()

    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    for i in [0,1,2,3]:
        axes[i].tick_params(axis='both', which='major', labelsize=font)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    columns = columns + ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_periphery_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean']
    #columns = columns + [f'pathogen_slope_channel_{channel_of_interest}', f'pathogen_cell_distance_channel_{channel_of_interest}', f'nucleus_cell_distance_channel_{channel_of_interest}']

    width = figuresize*2
    columns_per_row = math.ceil(len(columns) / 2)
    height = (figuresize*2)/columns_per_row

    fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
    axes = axes.flatten()

    print(f'{columns}')
    for i, col in enumerate(columns):
        ax = axes[i]
        plotter_col = spacrGraph(df, grouping_column='condition', data_column=col)
        plotter_col.create_plot(ax=ax)
        ax.set_xlabel(f'pathogen {df_type}', fontsize=font)
        ax.set_ylabel(f'{col}', fontsize=int(font * 2))
        #ax.legend_.remove()
        ax.tick_params(axis='both', which='major', labelsize=font)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if i <= 5:
            ax.set_ylim(1, None)

    # Turn off any unused axes
    for i in range(len(columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
        
def _plot_recruitment(df, df_type, channel_of_interest, columns=[], figuresize=10):
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

    #axes[0].legend_.remove()
    #axes[1].legend_.remove()
    #axes[2].legend_.remove()
    #axes[3].legend_.remove()
        
    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    for i in [0,1,2,3]:
        axes[i].tick_params(axis='both', which='major', labelsize=font)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    columns = columns + ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_periphery_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean']
    #columns = columns + [f'pathogen_slope_channel_{channel_of_interest}', f'pathogen_cell_distance_channel_{channel_of_interest}', f'nucleus_cell_distance_channel_{channel_of_interest}']

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
    fig = plt.figure(figsize=(50, 50))
    plt.imshow(canvas)
    plt.axis("off")
    for i, label in enumerate(labels):
        row = i // n_col
        col = i % n_col
        x = col * img_width + 2
        y = row * img_height + 15
        plt.text(x, y, label, color=color, fontsize=fontsize, fontweight='bold')
    return fig

def _imshow_gpu(img, labels, nrow=20, color='white', fontsize=12):
    """
    Display multiple images in a grid with corresponding labels.

    Args:
        img (torch.Tensor): A batch of images as a tensor.
        labels (list): List of labels corresponding to each image.
        nrow (int, optional): Number of images per row in the grid. Defaults to 20.
        color (str, optional): Color of the label text. Defaults to 'white'.
        fontsize (int, optional): Font size of the label text. Defaults to 12.
    """
    if img.is_cuda:
        img = img.cpu()  # Move to CPU if the tensor is on GPU

    n_images = len(labels)
    n_col = nrow
    n_row = int(np.ceil(n_images / n_col))

    img_height = img.shape[2]  # Height of the image
    img_width = img.shape[3]   # Width of the image

    # Prepare the canvas on CPU
    canvas = torch.zeros((img_height * n_row, img_width * n_col, 3))

    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < n_images:
                # Place the image on the canvas
                canvas[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = img[idx].permute(1, 2, 0)

    canvas = canvas.numpy()  # Convert to NumPy for plotting

    fig = plt.figure(figsize=(50, 50))
    plt.imshow(canvas)
    plt.axis("off")

    for i, label in enumerate(labels):
        row = i // n_col
        col = i % n_col
        x = col * img_width + 2
        y = row * img_height + 15
        plt.text(x, y, label, color=color, fontsize=fontsize, fontweight='bold')

    return fig
    
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
        plt.figure(figsize=(10,10))
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
    
def generate_plate_heatmap(df, plate_number, variable, grouping, min_max, min_count):

    if not isinstance(min_count, (int, float)):
        min_count = 0

    # Check the number of parts in 'prc'
    num_parts = len(df['prc'].iloc[0].split('_'))
    if num_parts == 4:
        split = df['prc'].str.split('_', expand=True)
        df['rowID'] = split[2]
        df['prc'] = f"{plate_number}" + '_' + split[2] + '_' + split[3]
        
    # Construct 'prc' based on 'plateID', 'rowID', and 'columnID' columns
    #df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)

    if 'column_name' not in df.columns:
        if 'column' in df.columns:
            df['columnID'] = df['column']
        if 'column_name' in df.columns:
            df['columnID'] = df['column_name']
                
    df['plateID'], df['rowID'], df['columnID'] = zip(*df['prc'].str.split('_'))
    
    # Filtering the dataframe based on the plate_number
    df = df[df['plateID'] == plate_number].copy()  # Create another copy after filtering
    
    # Ensure proper ordering
    row_order = [f'r{i}' for i in range(1, 17)]
    col_order = [f'c{i}' for i in range(1, 28)]  # Exclude c15 as per your earlier code
    
    df['rowID'] = pd.Categorical(df['rowID'], categories=row_order, ordered=True)
    df['columnID'] = pd.Categorical(df['columnID'], categories=col_order, ordered=True)
    df['count'] = df.groupby(['rowID', 'columnID'])['rowID'].transform('count')

    if min_count > 0:
        df = df[df['count'] >= min_count]

    # Explicitly set observed=True to avoid FutureWarning
    grouped = df.groupby(['rowID', 'columnID'], observed=True) # Group by row and column
    
    if grouping == 'mean':
        plate = grouped[variable].mean().reset_index()
    elif grouping == 'sum':
        plate = grouped[variable].sum().reset_index()
    elif grouping == 'count':
        variable = 'count'
        plate = grouped[variable].count().reset_index()
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")
        
    plate_map = pd.pivot_table(plate, values=variable, index='rowID', columns='columnID').fillna(0)
    
    if min_max == 'all':
        min_max = [plate_map.min().min(), plate_map.max().max()]
    elif min_max == 'allq':
        min_max = np.quantile(plate_map.values, [0.02, 0.98])
    elif isinstance(min_max, (list, tuple)) and len(min_max) == 2:
        if isinstance(min_max[0], (float)) and isinstance(min_max[1], (float)):
            min_max = np.quantile(plate_map.values, [min_max[0], min_max[1]])
        if isinstance(min_max[0], (int)) and isinstance(min_max[1], (int)): 
            min_max = [min_max[0], min_max[1]]
    return plate_map, min_max

def plot_plates(df, variable, grouping, min_max, cmap, min_count=0, verbose=True, dst=None):
    plates = df['prc'].str.split('_', expand=True)[0].unique()
    n_rows, n_cols = (len(plates) + 3) // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 5 * n_rows))
    ax = ax.flatten()

    for index, plate in enumerate(plates):
        plate_map, min_max_values = generate_plate_heatmap(df, plate, variable, grouping, min_max, min_count)
        sns.heatmap(plate_map, cmap=cmap, vmin=min_max_values[0], vmax=min_max_values[1], ax=ax[index])
        ax[index].set_title(plate)
        
    for i in range(len(plates), n_rows * n_cols):
        fig.delaxes(ax[i])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    if not dst is None:
        for i in range(0,1000):
            filename = os.path.join(dst, f'plate_heatmap_{i}.pdf')
            if os.path.exists(filename):
                continue
            else:
                fig.savefig(filename, format='pdf')
                print(f'Saved heatmap to {filename}')
                break
    if verbose:
        plt.show()
    return fig

def print_mask_and_flows(stack, mask, flows, overlay=True, max_size=1000, thickness=2):
    """
    Display the original image, mask with outlines, and flow images.
    
    Args:
        stack (np.array): Original image or stack.
        mask (np.array): Mask image.
        flows (list): List of flow images.
        overlay (bool): Whether to overlay the mask outlines on the original image.
        max_size (int): Maximum allowed size for any dimension of the images.
        thickness (int): Thickness of the contour outlines.
    """

    def resize_if_needed(image, max_size):
        """Resize image if any dimension exceeds max_size while maintaining aspect ratio."""
        if max(image.shape[:2]) > max_size:
            scale = max_size / max(image.shape[:2])
            new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
            if image.ndim == 3:
                new_shape += (image.shape[2],)
            return sk_resize(image, new_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
        return image

    def generate_contours(mask):
        """Generate contours for each object in the mask using OpenCV."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def apply_contours_on_image(image, mask, color=(255, 0, 0), thickness=2):
        """Draw the contours on the original image."""
        # Ensure the image is in RGB format
        if image.ndim == 2:  # Grayscale to RGB
            image = normalize_to_uint8(image)  # Convert to uint8 if needed
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()

        # Generate and draw contours
        contours = generate_contours(mask)
        cv2.drawContours(image_rgb, contours, -1, color, thickness)

        return image_rgb

    def normalize_to_uint8(image):
        """Normalize and convert image to uint8."""
        image = np.clip(image, 0, 1)  # Ensure values are between 0 and 1
        return (image * 255).astype(np.uint8)  # Convert to uint8
    
    
    # Resize if necessary
    stack = resize_if_needed(stack, max_size)
    mask = resize_if_needed(mask, max_size)
    if flows != None:
        flows = [resize_if_needed(flow, max_size) for flow in flows]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    if stack.shape[-1] == 1:
        stack = np.squeeze(stack)

    # Display original image
    if stack.ndim == 2:
        original_image = stack
    elif stack.ndim == 3:
        original_image = stack[..., 0]  # Use the first channel as the base
    else:
        raise ValueError("Unexpected stack dimensionality.")

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Overlay mask outlines on original image if overlay is True
    if overlay:
        outlined_image = apply_contours_on_image(original_image, mask, color=(255, 0, 0), thickness=thickness)
        axs[1].imshow(outlined_image)
    else:
        axs[1].imshow(mask, cmap='gray')

    axs[1].set_title('Mask with Overlay' if overlay else 'Mask')
    axs[1].axis('off')

    if flows != None:

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
    
def plot_resize_v1(images, resized_images, labels, resized_labels):
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
    

def plot_resize(images, resized_images, labels, resized_labels):
    def prepare_image(img):
        if img.ndim == 2:
            return img, 'gray'
        elif img.ndim == 3:
            if img.shape[-1] == 1:
                return np.squeeze(img, axis=-1), 'gray'
            elif img.shape[-1] == 3:
                return img, None  # RGB
            elif img.shape[-1] == 4:
                return img, None  # RGBA
            else:
                # fallback: average across channels to show as grayscale
                return np.mean(img, axis=-1), 'gray'
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    # Original Image
    img, cmap = prepare_image(images[0])
    ax[0, 0].imshow(img, cmap=cmap)
    ax[0, 0].set_title('Original Image')

    # Resized Image
    img, cmap = prepare_image(resized_images[0])
    ax[0, 1].imshow(img, cmap=cmap)
    ax[0, 1].set_title('Resized Image')

    # Labels (assumed grayscale or single-channel)
    lbl, cmap = prepare_image(labels[0])
    ax[1, 0].imshow(lbl, cmap=cmap)
    ax[1, 0].set_title('Original Label')

    lbl, cmap = prepare_image(resized_labels[0])
    ax[1, 1].imshow(lbl, cmap=cmap)
    ax[1, 1].set_title('Resized Label')

    plt.tight_layout()
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

def visualize_cellpose_masks(masks, titles=None, filename=None, save=False, src=None):
    """
    Visualize multiple masks with optional titles.
    
    Parameters:
        masks (list of np.ndarray): A list of masks to visualize.
        titles (list of str, optional): A list of titles for the masks. If None, default titles will be used.
        comparison_title (str): Title for the entire figure.
    """
    
    comparison_title=f"Masks Comparison for {filename}"
    
    if titles is None:
        titles = [f'Mask {i+1}' for i in range(len(masks))]
    
    # Ensure the length of titles matches the number of masks
    assert len(titles) == len(masks), "Number of titles and masks must match"
    
    num_masks = len(masks)
    fig, axs = plt.subplots(1, num_masks, figsize=(10 * num_masks, 10))  # Adjusting figure size dynamically
    
    for ax, mask, title in zip(axs, masks, titles):
        cmap = generate_mask_random_cmap(mask)
        # Normalize and display the mask
        norm = plt.Normalize(vmin=0, vmax=mask.max())
        ax.imshow(mask, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle(comparison_title)
    plt.show()
    
    if save:
        if src is None:
            src = os.getcwd()
        results_dir = os.path.join(src, 'results')
        os.makedirs(results_dir, exist_ok=True)
        fig_path = os.path.join(results_dir, f'{filename}.pdf')
        fig.savefig(fig_path, format='pdf')
        print(f'Saved figure to {fig_path}')
    return

    
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

def plot_object_outlines(src, objects=['nucleus','cell','pathogen'], channels=[0,1,2], max_nr=10):
    
    for object_, channel in zip(objects, channels):
        folders = [os.path.join(src, 'norm_channel_stack', f'{object_}_mask_stack'),
                   os.path.join(src,f'{channel+1}')]
        print(folders)
        plot_images_and_arrays(folders,
                               lower_percentile=2,
                               upper_percentile=99.5,
                               threshold=1000,
                               extensions=['.npy', '.tif', '.tiff', '.png'],
                               overlay=True,
                               max_nr=10,
                               randomize=True)
                
def volcano_plot(coef_df, filename='volcano_plot.pdf'):
    palette = {
        'pc': 'red',
        'nc': 'green',
        'control': 'blue',
        'other': 'gray'
    }

    # Create the volcano plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=coef_df, 
        x='coefficient', 
        y='-log10(p_value)', 
        hue='condition', 
        palette=palette
    )

    plt.title('Volcano Plot of Coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--')
    plt.legend().remove()
    plt.savefig(filename, format='pdf')
    print(f'Saved Volcano plot: {filename}')
    plt.show()

def plot_histogram(df, column, dst=None):
    # Plot histogram of the dependent variable
    bar_color = (0/255, 155/255, 155/255)
    plt.figure(figsize=(10, 10))
    sns.histplot(df[column], kde=False, color=bar_color, edgecolor=None, alpha=0.6)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    if not dst is None:
        filename = os.path.join(dst, f'{column}_histogram.pdf')
        plt.savefig(filename, format='pdf')
        print(f'Saved histogram to {filename}')

    plt.show()

def plot_lorenz_curves(csv_files, name_column='grna_name', value_column='count', 
                       remove_keys=None, 
                       x_lim=[0.0, 1], y_lim=[0, 1], remove_outliers=False, save=True):
    
    def lorenz_curve(data):
        """Calculate Lorenz curve."""
        sorted_data = np.sort(data)
        cumulative_data = np.cumsum(sorted_data)
        lorenz_curve = cumulative_data / cumulative_data[-1]
        lorenz_curve = np.insert(lorenz_curve, 0, 0)
        return lorenz_curve
    
    def gini_coefficient(data):
        """Calculate Gini coefficient from data."""
        sorted_data = np.sort(data)
        n = len(data)
        cumulative_data = np.cumsum(sorted_data) / np.sum(sorted_data)
        cumulative_data = np.insert(cumulative_data, 0, 0)
        gini = 1 - 2 * np.sum(cumulative_data[:-1] * np.diff(np.linspace(0, 1, n + 1)))
        return gini

    def remove_outliers_by_wells(data, name_col, wells_col):
        """Remove outliers based on 95% confidence interval for well counts."""
        well_counts = data.groupby(name_col).size()
        q1 = well_counts.quantile(0.05)
        q3 = well_counts.quantile(0.95)
        iqr_range = q3 - q1
        lower_bound = q1 - 1.5 * iqr_range
        upper_bound = q3 + 1.5 * iqr_range
        valid_names = well_counts[(well_counts >= lower_bound) & (well_counts <= upper_bound)].index
        return data[data[name_col].isin(valid_names)]
    
    combined_data = []
    gini_values = {}

    plt.figure(figsize=(10, 10))

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        # Remove specified keys
        for remove in remove_keys:
            df = df[df[name_column] != remove]
        
        # Remove outliers
        if remove_outliers:
            df = remove_outliers_by_wells(df, name_column, value_column)
        
        values = df[value_column].values
        combined_data.extend(values)
        
        # Calculate Lorenz curve and Gini coefficient
        lorenz = lorenz_curve(values)
        gini = gini_coefficient(values)
        gini_values[f"plate {idx+1}"] = gini
        
        name = f"plate {idx+1} (Gini: {gini:.4f})"
        plt.plot(np.linspace(0, 1, len(lorenz)), lorenz, label=name)

    # Plot combined Lorenz curve
    combined_lorenz = lorenz_curve(np.array(combined_data))
    combined_gini = gini_coefficient(np.array(combined_data))
    gini_values["Combined"] = combined_gini
    
    plt.plot(np.linspace(0, 1, len(combined_lorenz)), combined_lorenz, label=f"Combined (Gini: {combined_gini:.4f})", linestyle='--', color='black')
    
    if x_lim is not None:
        plt.xlim(x_lim)
    
    if y_lim is not None:
        plt.ylim(y_lim)
        
    plt.title('Lorenz Curves')
    plt.xlabel('Cumulative Share of Individuals')
    plt.ylabel('Cumulative Share of Value')
    plt.legend()
    plt.grid(False)
    
    if save:
        save_path = os.path.join(os.path.dirname(csv_files[0]), 'results')
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'lorenz_curve_with_gini.pdf')
        plt.savefig(save_file_path, format='pdf', bbox_inches='tight')
        print(f"Saved Lorenz Curve: {save_file_path}")
    
    plt.show()

    # Print Gini coefficients
    for plate, gini in gini_values.items():
        print(f"{plate}: Gini Coefficient = {gini:.4f}")

def plot_permutation(permutation_df):
    num_features = len(permutation_df)
    fig_height = max(8, num_features * 0.3)  # Set a minimum height of 8 and adjust height based on number of features
    fig_width = 10  # Width can be fixed or adjusted similarly
    font_size = max(10, 12 - num_features * 0.2)  # Adjust font size dynamically

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.barh(permutation_df['feature'], permutation_df['importance_mean'], xerr=permutation_df['importance_std'], color="teal", align="center", alpha=0.6)
    ax.set_xlabel('Permutation Importance', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance_df):
    num_features = len(feature_importance_df)
    fig_height = max(8, num_features * 0.3)  # Set a minimum height of 8 and adjust height based on number of features
    fig_width = 10  # Width can be fixed or adjusted similarly
    font_size = max(10, 12 - num_features * 0.2)  # Adjust font size dynamically

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color="blue", align="center", alpha=0.6)
    ax.set_xlabel('Feature Importance', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    return fig

def read_and_plot__vision_results(base_dir, y_axis='accuracy', name_split='_time', y_lim=[0.8, 0.9]):
    # List to store data from all CSV files
    data_frames = []

    dst = os.path.join(base_dir, 'result')
    os.mkdir(dst,exists=True)

    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_test_result.csv"):
                file_path = os.path.join(root, file)
                # Extract model information from the file name
                file_name = os.path.basename(file_path)
                model = file_name.split(f'{name_split}')[0]
                
                # Extract epoch information from the file name
                epoch_info = file_name.split('_time')[1]
                base_folder = os.path.dirname(file_path)
                epoch = os.path.basename(base_folder)
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                df['model'] = model
                df['epoch'] = epoch
                
                # Append the data frame to the list
                data_frames.append(df)
    
    # Concatenate all data frames
    if data_frames:
        result_df = pd.concat(data_frames, ignore_index=True)
        
        # Calculate average y_axis per model
        avg_metric = result_df.groupby('model')[y_axis].mean().reset_index()
        avg_metric = avg_metric.sort_values(by=y_axis)
        print(avg_metric)
        
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.bar(avg_metric['model'], avg_metric[y_axis])
        plt.xlabel('Model')
        plt.ylabel(f'{y_axis}')
        plt.title(f'Average {y_axis.capitalize()} per Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.show()
    else:
        print("No CSV files found in the specified directory.")

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

    def join_measurments_and_annotation(src, tables = ['cell', 'nucleus', 'pathogen','cytoplasm']):
        from .io import _read_and_merge_data, _read_db
        db_loc = [src+'/measurements/measurements.db']
        loc = src+'/measurements/measurements.db'
        df, _ = _read_and_merge_data(db_loc, 
                                    tables, 
                                    verbose=True, 
                                    nuclei_limit=True, 
                                    pathogen_limit=True)
        
        paths_df = _read_db(loc, tables=['png_list'])
        merged_df = pd.merge(df, paths_df[0], on='prcfo', how='left')
        return merged_df

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

def create_grouped_plot(df, grouping_column, data_column, graph_type='bar', summary_func='mean', order=None, colors=None, output_dir='./output', save=False, y_lim=None, error_bar_type='std'):
    """
    Create a grouped plot, perform statistical tests, and optionally export the results along with the plot.

    Parameters:
    - df: DataFrame containing the data.
    - grouping_column: Column name for the categorical grouping.
    - data_column: Column name for the data to be grouped and plotted.
    - graph_type: Type of plot ('bar', 'violin', 'jitter', 'box', 'jitter_box').
    - summary_func: Summary function to apply to each group ('mean', 'median', etc.).
    - order: List specifying the order of the groups. If None, groups will be ordered alphabetically.
    - colors: List of colors for each group.
    - output_dir: Directory where the figure and test results will be saved if `save=True`.
    - save: Boolean flag indicating whether to save the plot and results to files.
    - y_lim: Optional y-axis min and max.
    - error_bar_type: Type of error bars to plot, either 'std' for standard deviation or 'sem' for standard error of the mean.

    Outputs:
    - Figure of the plot.
    - DataFrame with full statistical test results, including normality tests.
    """
    
    # Remove NaN rows in grouping_column
    df = df.dropna(subset=[grouping_column])
    
    # Ensure the output directory exists if save is True
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Sorting and ordering
    if order:
        df[grouping_column] = pd.Categorical(df[grouping_column], categories=order, ordered=True)
    else:
        df[grouping_column] = pd.Categorical(df[grouping_column], categories=sorted(df[grouping_column].unique()), ordered=True)
    
    # Get unique groups
    unique_groups = df[grouping_column].unique()
    
    # Initialize test results
    test_results = []

    # Test normality for each group
    grouped_data = [df.loc[df[grouping_column] == group, data_column] for group in unique_groups]
    normal_p_values = [normaltest(data).pvalue for data in grouped_data]
    normal_stats = [normaltest(data).statistic for data in grouped_data]
    is_normal = all(p > 0.05 for p in normal_p_values)

    # Add normality test results to the results_df
    for group, stat, p_value in zip(unique_groups, normal_stats, normal_p_values):
        test_results.append({
            'Comparison': f'Normality test for {group}',
            'Test Statistic': stat,
            'p-value': p_value,
            'Test Name': 'Normality test'
        })

    # Determine statistical test
    if len(unique_groups) == 2:
        if is_normal:
            stat_test = ttest_ind
            test_name = 'T-test'
        else:
            stat_test = mannwhitneyu
            test_name = 'Mann-Whitney U test'
    else:
        if is_normal:
            stat_test = f_oneway
            test_name = 'One-way ANOVA'
        else:
            stat_test = kruskal
            test_name = 'Kruskal-Wallis test'

    # Perform pairwise statistical tests
    comparisons = list(itertools.combinations(unique_groups, 2))
    p_values = []
    test_statistics = []

    for (group1, group2) in comparisons:
        data1 = df[df[grouping_column] == group1][data_column]
        data2 = df[df[grouping_column] == group2][data_column]
        stat, p = stat_test(data1, data2)
        p_values.append(p)
        test_statistics.append(stat)
        test_results.append({'Comparison': f'{group1} vs {group2}', 'Test Statistic': stat, 'p-value': p, 'Test Name': test_name})
    
    # Post-hoc test (Tukey HSD for ANOVA)
    posthoc_p_values = None
    if is_normal and len(unique_groups) > 2:
        tukey_result = pairwise_tukeyhsd(df[data_column], df[grouping_column], alpha=0.05)
        posthoc_p_values = tukey_result.pvalues
        for comparison, p_value in zip(tukey_result._results_table.data[1:], tukey_result.pvalues):
            test_results.append({
                'Comparison': f'{comparison[0]} vs {comparison[1]}',
                'Test Statistic': None,  # Tukey does not provide a test statistic in the same way
                'p-value': p_value,
                'Test Name': 'Tukey HSD Post-hoc'
            })

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    if colors:
        color_palette = colors
    else:
        color_palette = sns.color_palette("husl", len(unique_groups))
    
    # Choose graph type
    if graph_type == 'bar':
        summary_df = df.groupby(grouping_column)[data_column].agg([summary_func, 'std', 'sem'])
        
        # Set error bars based on error_bar_type
        if error_bar_type == 'std':
            error_bars = summary_df['std']
        elif error_bar_type == 'sem':
            error_bars = summary_df['sem']
        else:
            raise ValueError(f"Invalid error_bar_type: {error_bar_type}. Choose either 'std' or 'sem'.")

        sns.barplot(x=grouping_column, y=summary_func, data=summary_df.reset_index(), ci=None, order=order, palette=color_palette)

        # Add error bars (standard deviation or standard error of the mean)
        plt.errorbar(x=np.arange(len(summary_df)), y=summary_df[summary_func], yerr=error_bars, fmt='none', c='black', capsize=5)
    
    elif graph_type == 'violin':
        sns.violinplot(x=grouping_column, y=data_column, data=df, order=order, palette=color_palette)
    elif graph_type == 'jitter':
        sns.stripplot(x=grouping_column, y=data_column, data=df, jitter=True, order=order, palette=color_palette)
    elif graph_type == 'box':
        sns.boxplot(x=grouping_column, y=data_column, data=df, order=order, palette=color_palette)
    elif graph_type == 'jitter_box':
        sns.boxplot(x=grouping_column, y=data_column, data=df, order=order, palette=color_palette)
        sns.stripplot(x=grouping_column, y=data_column, data=df, jitter=True, color='black', alpha=0.5, order=order)

    # Create a DataFrame to summarize the test results
    results_df = pd.DataFrame(test_results)

    # Set y-axis start if provided
    if isinstance(y_lim, list) and len(y_lim) == 2:
        plt.ylim(y_lim)

    # If save is True, save the plot and results as PNG and CSV
    if save:
        # Save the plot as PNG
        plot_path = os.path.join(output_dir, 'grouped_plot.png')
        plt.title(f'{test_name} results for {graph_type} plot')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Save the test results as a CSV file
        results_path = os.path.join(output_dir, 'test_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Test results saved to {results_path}")

    # Show the plot
    plt.show()

    return plt.gcf(), results_df
    
class spacrGraph:
    def __init__(self, df, grouping_column, data_column, graph_type='bar', summary_func='mean', 
                 order=None, colors=None, output_dir='./output', save=False, y_lim=None, log_y=False,
                 log_x=False, error_bar_type='std', remove_outliers=False, theme='pastel', representation='object',
                 paired=False, all_to_all=True, compare_group=None, graph_name=None):
        
        """
        Class for creating grouped plots with optional statistical tests and data preprocessing.
        """

        self.df = df
        self.grouping_column = grouping_column
        self.order = sorted(df[self.grouping_column].unique().tolist())
        self.data_column = data_column if isinstance(data_column, list) else [data_column]
        
        self.graph_type = graph_type
        self.summary_func = summary_func
        self.order = order
        self.colors = colors
        self.output_dir = output_dir
        self.save = save
        self.error_bar_type = error_bar_type
        self.remove_outliers = remove_outliers
        self.theme = theme
        self.representation = representation
        self.paired = paired
        self.all_to_all = all_to_all
        self.compare_group = compare_group
        self.y_lim = y_lim
        self.graph_name = graph_name
        self.log_x = log_x
        self.log_y = log_y

        self.results_df = pd.DataFrame()
        self.sns_palette = None
        self.fig = None

        self.results_name = str(self.graph_name)+'_'+str(self.data_column[0])+'_'+str(self.grouping_column)+'_'+str(self.graph_type)
        
        self._set_theme()
        self.raw_df = self.df.copy()
        self.df = self.preprocess_data()
        
    def _set_theme(self):
        """Set the Seaborn theme and reorder colors if necessary."""
        integer_list = list(range(1, 81))
        color_order = [7,9,4,0,3,6,2] + integer_list
        self.sns_palette = self._set_reordered_theme(self.theme, color_order, 100)

    def _set_reordered_theme(self, theme='deep', order=None, n_colors=100, show_theme=False):
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

    #def preprocess_data(self):
    #    """Preprocess the data: remove NaNs, sort/order the grouping column, and optionally group by 'prc'."""
    #    # Remove NaNs in both the grouping column and each data column
    #    df = self.df.dropna(subset=[self.grouping_column] + self.data_column)
    #    # Group by 'prc' column if representation is 'well'
    #    if self.representation == 'well':
    #        df = df.groupby(['prc', self.grouping_column])[self.data_column].agg(self.summary_func).reset_index()
    #    if self.representation == 'plateID':
    #        df = df.groupby(['plateID', self.grouping_column])[self.data_column].agg(self.summary_func).reset_index()
    #    if self.order:
    #        df[self.grouping_column] = pd.Categorical(df[self.grouping_column], categories=self.order, ordered=True)
    #    else:
    #        df[self.grouping_column] = pd.Categorical(df[self.grouping_column], categories=sorted(df[self.grouping_column].unique()), ordered=True)
    #    return df
  
    def preprocess_data(self):
        """
        Preprocess the data: remove NaNs, optionally ensure 'plateID' column is created,
        then group by either 'prc', 'plateID', or do no grouping at all if representation == 'object'.
        """
        # 1) Remove NaNs in both the grouping column and each data column
        df = self.df.dropna(subset=[self.grouping_column] + self.data_column)

        # 2) Decide how to handle grouping based on 'representation'
        if self.representation == 'object':
            # -- No grouping at all --
            # We do nothing except keep df as-is after removing NaNs
            group_cols = None

        elif self.representation == 'well':
            # Group by ['prc', grouping_column]
            group_cols = ['prc', self.grouping_column]

        elif self.representation == 'plateID':
            # Make sure 'plateID' exists (split from 'prc' if needed)
            if 'plateID' not in df.columns:
                if 'prc' in df.columns:
                    df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)
                else:
                    raise KeyError(
                        "Representation is 'plateID', but no 'plateID' column found. "
                        "Also cannot split from 'prc' because 'prc' column is missing."
                    )
            # If the grouping column IS 'plateID', only group by ['plateID'] once
            if self.grouping_column == 'plateID':
                group_cols = ['plateID']
            else:
                group_cols = ['plateID', self.grouping_column]

        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        # 3) Perform grouping only if group_cols is set
        if group_cols is not None:
            df = df.groupby(group_cols)[self.data_column].agg(self.summary_func).reset_index()

        # 4) Handle ordering if specified (and if the grouping_column still exists)
        if self.order and (self.grouping_column in df.columns):
            df[self.grouping_column] = pd.Categorical(
                df[self.grouping_column],
                categories=self.order,
                ordered=True
            )
        elif (self.grouping_column in df.columns):
            # Default to sorting unique values
            df[self.grouping_column] = pd.Categorical(
                df[self.grouping_column],
                categories=sorted(df[self.grouping_column].unique()),
                ordered=True
            )

        return df
   
    def remove_outliers_from_plot(self):
        """Remove outliers from the plot but keep them in the data."""
        filtered_df = self.df.copy()
        unique_groups = filtered_df[self.grouping_column].unique()
        for group in unique_groups:
            group_data = filtered_df[filtered_df[self.grouping_column] == group][self.data_column]
            q1 = group_data.quantile(0.25)
            q3 = group_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_df = filtered_df.drop(filtered_df[(filtered_df[self.grouping_column] == group) & ((filtered_df[self.data_column] < lower_bound) | (filtered_df[self.data_column] > upper_bound))].index)
        return filtered_df

    def perform_normality_tests(self):
        """Perform normality tests for each group and data column."""
        unique_groups = self.df[self.grouping_column].unique()
        normality_results = []

        for column in self.data_column:
            for group in unique_groups:
                data = self.df.loc[self.df[self.grouping_column] == group, column].dropna()
                n_samples = len(data)

                if n_samples < 3:
                    # Skip test if there aren't enough data points
                    print(f"Skipping normality test for group '{group}' on column '{column}' - Not enough data.")
                    normality_results.append({
                        'Comparison': f'Normality test for {group} on {column}',
                        'Test Statistic': None,
                        'p-value': None,
                        'Test Name': 'Skipped',
                        'Column': column,
                        'n': n_samples
                    })
                    continue

                # Choose the appropriate normality test based on the sample size
                if n_samples >= 8:
                    stat, p_value = normaltest(data)
                    test_name = "D'Agostino-Pearson test"
                else:
                    stat, p_value = shapiro(data)
                    test_name = "Shapiro-Wilk test"

                # Store the result for this group and column
                normality_results.append({
                    'Comparison': f'Normality test for {group} on {column}',
                    'Test Statistic': stat,
                    'p-value': p_value,
                    'Test Name': test_name,
                    'Column': column,
                    'n': n_samples
                })

            # Check if all groups are normally distributed (p > 0.05)
            normal_p_values = [result['p-value'] for result in normality_results if result['Column'] == column and result['p-value'] is not None]
            is_normal = all(p > 0.05 for p in normal_p_values)

        return is_normal, normality_results

    def perform_levene_test(self, unique_groups):
        """Perform Levene's test for equal variance."""
        grouped_data = [self.df.loc[self.df[self.grouping_column] == group, self.data_column] for group in unique_groups]
        stat, p_value = levene(*grouped_data)
        return stat, p_value

    def perform_statistical_tests(self, unique_groups, is_normal):
        """Perform statistical tests separately for each data column."""
        test_results = []
        for column in self.data_column:  # Iterate over each data column
            grouped_data = [self.df.loc[self.df[self.grouping_column] == group, column] for group in unique_groups]
            if len(unique_groups) == 2:  # For two groups: class_0 vs class_1
                if is_normal:
                    if self.paired:
                        stat, p = pg.ttest(grouped_data[0], grouped_data[1], paired=True).iloc[0][['T', 'p-val']]
                        test_name = 'Paired T-test'
                    else:
                        stat, p = ttest_ind(grouped_data[0], grouped_data[1])
                        test_name = 'T-test'
                else:
                    if self.paired:
                        stat, p = pg.wilcoxon(grouped_data[0], grouped_data[1]).iloc[0][['T', 'p-val']]
                        test_name = 'Paired Wilcoxon test'
                    else:
                        stat, p = mannwhitneyu(grouped_data[0], grouped_data[1])
                        test_name = 'Mann-Whitney U test'
            else:
                if is_normal:
                    stat, p = f_oneway(*grouped_data)
                    test_name = 'One-way ANOVA'
                else:
                    stat, p = kruskal(*grouped_data)
                    test_name = 'Kruskal-Wallis test'

            test_results.append({
                'Comparison': f'{unique_groups[0]} vs {unique_groups[1]} ({column})',
                'Test Statistic': stat,
                'p-value': p,
                'Test Name': test_name,
                'Column': column,
                'n_object': len(grouped_data[0]) + len(grouped_data[1]),
                'n_well': len(self.df[self.df[self.grouping_column] == unique_groups[0]]) + 
                          len(self.df[self.df[self.grouping_column] == unique_groups[1]])})

        return test_results
    
    def perform_posthoc_tests(self, is_normal, unique_groups):
        """Perform post-hoc tests for multiple groups based on all_to_all flag."""

        from .sp_stats import choose_p_adjust_method

        posthoc_results = []
        if is_normal and len(unique_groups) > 2 and self.all_to_all:
            tukey_result = pairwise_tukeyhsd(self.df[self.data_column], self.df[self.grouping_column], alpha=0.05)
            posthoc_results = []
            for comparison, p_value in zip(tukey_result._results_table.data[1:], tukey_result.pvalues):
                raw_data1 = self.raw_df[self.raw_df[self.grouping_column] == comparison[0]][self.data_column]
                raw_data2 = self.raw_df[self.raw_df[self.grouping_column] == comparison[1]][self.data_column]

                posthoc_results.append({
                    'Comparison': f'{comparison[0]} vs {comparison[1]}',
                    'Test Statistic': None,  # Tukey does not provide a test statistic
                    'p-value': p_value,
                    'Test Name': 'Tukey HSD Post-hoc',
                    'n_object': len(raw_data1) + len(raw_data2),
                    'n_well': len(self.df[self.df[self.grouping_column] == comparison[0]]) + len(self.df[self.df[self.grouping_column] == comparison[1]])})
            return posthoc_results
        
        elif len(unique_groups) > 2 and self.all_to_all:
            print('performing_dunns')

            # Prepare data for Dunn's test in long format
            long_data = self.df[[self.data_column[0], self.grouping_column]].dropna()

            p_adjust_method = choose_p_adjust_method(num_groups=len(long_data[self.grouping_column].unique()),num_data_points=len(long_data) // len(long_data[self.grouping_column].unique()))

            # Perform Dunn's test with Bonferroni correction
            dunn_result = sp.posthoc_dunn(
                long_data, 
                val_col=self.data_column[0], 
                group_col=self.grouping_column, 
                p_adjust=p_adjust_method
            )

            for group_a, group_b in zip(*np.triu_indices_from(dunn_result, k=1)):
                raw_data1 = self.raw_df[self.raw_df[self.grouping_column] == dunn_result.index[group_a]][self.data_column]
                raw_data2 = self.raw_df[self.raw_df[self.grouping_column] == dunn_result.columns[group_b]][self.data_column]

                posthoc_results.append({
                    'Comparison': f"{dunn_result.index[group_a]} vs {dunn_result.columns[group_b]}",
                    'Test Statistic': None,  # Dunn's test does not return a specific test statistic
                    'p-value': dunn_result.iloc[group_a, group_b],  # Extract the p-value from the matrix
                    'Test Name': "Dunn's Post-hoc",
                    'p_adjust_method': p_adjust_method,
                    'n_object': len(raw_data1) + len(raw_data2),  # Total objects
                    'n_well': len(self.df[self.df[self.grouping_column] == dunn_result.index[group_a]]) +
                            len(self.df[self.grouping_column] == dunn_result.columns[group_b])})

            return posthoc_results

        return posthoc_results
    
    def create_plot(self, ax=None):
        """Create and display the plot based on the chosen graph type."""

        def _generate_tabels(unique_groups):
            """Generate row labels and a symbol table for multi-level grouping."""
            # Create row labels: Include the grouping column and data columns
            row_labels = [self.grouping_column] + self.data_column

            # Initialize table data
            table_data = []

            # Create the grouping row: Alternate each group for every data column
            grouping_row = []
            for _ in self.data_column:
                for group in unique_groups:
                    grouping_row.append(group)
            table_data.append(grouping_row)  # Add the grouping row to the table

            # Create symbol rows for each data column
            for column in self.data_column:
                column_row = []  # Initialize a row for this column
                for data_col in self.data_column:  # Iterate over data columns to align with the structure
                    for group in unique_groups:
                        # Assign '+' if the column matches, otherwise assign '-'
                        if column == data_col:
                            column_row.append('+')
                        else:
                            column_row.append('-')
                table_data.append(column_row)  # Add this row to the table

            # Transpose the table to align with the plot layout
            transposed_table = list(map(list, zip(*table_data)))
            return row_labels, transposed_table


        def _place_symbols(row_labels, transposed_table, x_positions, ax):
            """
            Places symbols and row labels aligned under the bars or jitter points on the graph.
            
            Parameters:
            - row_labels: List of row titles to be displayed along the y-axis.
            - transposed_table: Data to be placed under each bar/jitter as symbols.
            - x_positions: X-axis positions for each group to align the symbols.
            - ax: The matplotlib Axes object where the plot is drawn.
            """
            # Get plot dimensions and adjust for different plot sizes
            y_axis_min = ax.get_ylim()[0]  # Minimum y-axis value (usually 0)
            symbol_start_y = y_axis_min - 0.05 * (ax.get_ylim()[1] - y_axis_min)  # Adjust a bit below the x-axis

            # Calculate spacing for the table rows (adjust as needed)
            y_spacing = 0.04  # Adjust this for better spacing between rows

            # Determine the leftmost x-position for row labels (align with the y-axis)
            label_x_pos = ax.get_xlim()[0] - 0.3  # Adjust offset from the y-axis

            # Place row labels vertically aligned with symbols
            for row_idx, title in enumerate(row_labels):
                y_pos = symbol_start_y - (row_idx * y_spacing)  # Calculate vertical position for each label
                ax.text(label_x_pos, y_pos, title, ha='right', va='center', fontsize=12, fontweight='regular')

            # Place symbols under each bar or jitter point based on x-positions
            for idx, (x_pos, column_data) in enumerate(zip(x_positions, transposed_table)):
                for row_idx, text in enumerate(column_data):
                    y_pos = symbol_start_y - (row_idx * y_spacing)  # Adjust vertical spacing for symbols
                    ax.text(x_pos, y_pos, text, ha='center', va='center', fontsize=12, fontweight='regular')

            # Redraw to apply changes
            ax.figure.canvas.draw()
                    
        def _get_positions(self, ax):
            if self.graph_type in ['bar','jitter_bar']: 
                x_positions = [np.mean(bar.get_paths()[0].vertices[:, 0]) for bar in ax.collections if hasattr(bar, 'get_paths')]

            elif self.graph_type == 'violin':
                x_positions = [np.mean(violin.get_paths()[0].vertices[:, 0]) for violin in ax.collections if hasattr(violin, 'get_paths')]

            elif self.graph_type in ['box', 'jitter_box']:
                x_positions = list(set(line.get_xdata().mean() for line in ax.lines if line.get_linestyle() == '-'))                

            elif self.graph_type == 'jitter': 
                x_positions = [np.mean(collection.get_offsets()[:, 0]) for collection in ax.collections if collection.get_offsets().size > 0]
            
            elif self.graph_type in ['line', 'line_std']:
                x_positions = []
            
            return x_positions
        
        def _draw_comparison_lines(ax, x_positions):
            """Draw comparison lines and annotate significance based on results_df."""
            if self.results_df.empty:
                print("No comparisons available to annotate.")
                return

            y_max = max([bar.get_height() for bar in ax.patches])
            ax.set_ylim(0, y_max * 1.3)

            for idx, row in self.results_df.iterrows():
                group1, group2 = row['Comparison'].split(' vs ')
                p_value = row['p-value']

                # Determine significance marker
                if p_value <= 0.001:
                    signiresults_namecance = '***'
                elif p_value <= 0.01:
                    significance = '**'
                elif p_value <= 0.05:
                    significance = '*'
                else:
                    significance = 'ns'

                # Find the x positions of the compared groups
                x1 = x_positions[unique_groups.tolist().index(group1)]
                x2 = x_positions[unique_groups.tolist().index(group2)]

                # Stagger lines to avoid overlap
                line_y = y_max + (0.1 * y_max) * (idx + 1)

                # Draw the comparison line
                ax.plot([x1, x1, x2, x2], [line_y - 0.02, line_y, line_y, line_y - 0.02], lw=1.5, c='black')

                # Add the significance marker
                ax.text((x1 + x2) / 2, line_y, significance, ha='center', va='bottom', fontsize=12)

        # Optional: Remove outliers for plotting
        if self.remove_outliers:
            self.df = self.remove_outliers_from_plot()

        self.df_melted = pd.melt(self.df, id_vars=[self.grouping_column], value_vars=self.data_column,var_name='Data Column', value_name='Value')
        unique_groups = self.df[self.grouping_column].unique()
        is_normal, normality_results = self.perform_normality_tests()
        levene_stat, levene_p = self.perform_levene_test(unique_groups)
        test_results = self.perform_statistical_tests(unique_groups, is_normal)
        posthoc_results = self.perform_posthoc_tests(is_normal, unique_groups)
        self.results_df = pd.DataFrame(normality_results + test_results + posthoc_results)

        num_groups = len(self.data_column)*len(self.grouping_column)
        num_data_columns = len(self.data_column)
        self.bar_width = 0.4
        spacing_between_groups = self.bar_width/0.5

        self.fig_width = (num_groups * self.bar_width) + (spacing_between_groups * num_groups)
        self.fig_height = self.fig_width/2
        
        if  self.graph_type in ['line','line_std']:
            self.fig_height, self.fig_width = 10, 10 

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(self.fig_height, self.fig_width))
        else:
            self.fig = ax.figure

        if len(self.data_column) == 1:
            self.hue=self.grouping_column
            self.jitter_bar_dodge = False
        else:
            self.hue='Data Column'
            self.jitter_bar_dodge = True
        
        # Handle the different plot types based on `graph_type`
        if self.graph_type == 'bar':
            self._create_bar_plot(ax)
        elif self.graph_type == 'jitter':
            self._create_jitter_plot(ax)
        elif self.graph_type == 'box':
            self._create_box_plot(ax)
        elif self.graph_type == 'violin':
            self._create_violin_plot(ax)
        elif self.graph_type == 'jitter_box':
            self._create_jitter_box_plot(ax)
        elif self.graph_type == 'jitter_bar':
            self._create_jitter_bar_plot(ax)
        elif self.graph_type == 'line':
            self._create_line_graph(ax)
        elif self.graph_type == 'line_std':
            self._create_line_with_std_area(ax)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}") 
        
        if len(self.data_column) == 1:
            num_groups = len(self.df[self.grouping_column].unique())
            self._standerdize_figure_format(ax=ax, num_groups=num_groups, graph_type=self.graph_type)

        # Set y-axis start
        if isinstance(self.y_lim, list):
            if len(self.y_lim) == 2:
                ax.set_ylim(self.y_lim[0], self.y_lim[1])
            elif len(self.y_lim) == 1:
                ax.set_ylim(self.y_lim[0], None)

        sns.despine(ax=ax, top=True, right=True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Data Column') # Move the legend outside the plot
        
        if not self.graph_type in ['line','line_std']:
            ax.set_xlabel('')

        x_positions = _get_positions(self, ax)
        
        if len(self.data_column) == 1 and not self.graph_type in ['line','line_std']:
            ax.legend().remove()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        elif len(self.data_column) > 1 and not self.graph_type in ['line','line_std']:
            ax.set_xticks([])
            ax.tick_params(bottom=False)
            ax.set_xticklabels([])
            legend_ax = self.fig.add_axes([0.1, -0.2, 0.62, 0.2])  # Position the table closer to the graph
            legend_ax.set_axis_off()

            row_labels, table_data = _generate_tabels(unique_groups)
            _place_symbols(row_labels, table_data, x_positions, ax)
            
        #_draw_comparison_lines(ax, x_positions)    
        
        if self.save:
            self._save_results()

        ax.margins(x=0.12)

    def _standerdize_figure_format(self, ax, num_groups, graph_type):
        """
        Adjusts the figure layout (size, bar width, jitter, and spacing) based on the number of groups.

        Parameters:
        - ax: The matplotlib Axes object.
        - num_groups: Number of unique groups.
        - graph_type: The type of graph (e.g., 'bar', 'jitter', 'box', etc.).

        Returns:
        - None. Modifies the figure and Axes in place.
        """
        if graph_type in ['line', 'line_std']:
            print("Skipping layout adjustment for line graphs.")
            return  # Skip layout adjustment for line graphs
        
        correction_factor = 4

        # Set figure size to ensure it remains square with a minimum size
        fig_size = max(6, num_groups * 2)  / correction_factor
        
        if fig_size < 10:
            fig_size = 10
        
        
        ax.figure.set_size_inches(fig_size, fig_size)

        # Configure layout based on the number of groups
        bar_width = min(0.8, 1.5 / num_groups) / correction_factor
        jitter_amount = min(0.1, 0.2 / num_groups) / correction_factor
        jitter_size = max(50 / num_groups, 200)

        # Adjust axis limits to ensure bars are centered with respect to group labels
        ax.set_xlim(-0.5, num_groups - 0.5)

        # Set ticks to match the group labels in your DataFrame
        #group_labels = self.df[self.grouping_column].unique()
        #group_labels = self.order
        #ax.set_xticks(range(len(group_labels)))
        #ax.set_xticklabels(group_labels, rotation=45, ha='right')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Customize elements based on the graph type
        if graph_type == 'bar':
            # Adjust bars' width and position
            for bar in ax.patches:
                bar.set_width(bar_width)
                bar.set_x(bar.get_x() - bar_width / 2)

        elif graph_type in ['jitter', 'jitter_bar', 'jitter_box']:
            # Adjust jitter points' position and size
            for coll in ax.collections:
                offsets = coll.get_offsets()
                offsets[:, 0] += jitter_amount  # Shift jitter points slightly
                coll.set_offsets(offsets)
                coll.set_sizes([jitter_size]  * len(offsets))  # Adjust point size dynamically

        elif graph_type in ['box', 'violin']:
            # Adjust box width for consistent spacing
            for artist in ax.artists:
                artist.set_width(bar_width)

        # Adjust legend and axis labels
        ax.tick_params(axis='x', labelsize=max(10, 15 - num_groups // 2))
        ax.tick_params(axis='y', labelsize=max(10, 15 - num_groups // 2))

        if ax.get_legend():
            ax.get_legend().set_bbox_to_anchor((1.05, 1)) #loc='upper left',borderaxespad=0.
            ax.get_legend().prop.set_size(max(8, 12 - num_groups // 3))

        # Redraw the figure to apply changes
        ax.figure.canvas.draw()
        
    def _create_bar_plot(self, ax):
        """Helper method to create a bar plot with consistent bar thickness and centered error bars."""
        # Flatten DataFrame: Combine grouping column and data column into one group if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = None
    
        summary_df = self.df_melted.groupby([x_axis_column]).agg(mean=('Value', 'mean'),std=('Value', 'std'),sem=('Value', 'sem')).reset_index()
        error_bars = summary_df[self.error_bar_type] if self.error_bar_type in ['std', 'sem'] else None
        sns.barplot(data=self.df_melted, x=x_axis_column, y='Value', hue=self.hue, palette=self.sns_palette, ax=ax, dodge=self.jitter_bar_dodge, ci=None, order=self.order)
        
        # Adjust the bar width manually
        if len(self.data_column) > 1:
            bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
            target_width = self.bar_width * 2
            for bar in bars:
                bar.set_width(target_width)  # Set new width
                # Center the bar on its x-coordinate
                bar.set_x(bar.get_x() - target_width / 2)
            
        # Adjust error bars alignment with bars
        bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
        for bar, (_, row) in zip(bars, summary_df.iterrows()):
            x_bar = bar.get_x() + bar.get_width() / 2
            err = row[self.error_bar_type]
            ax.errorbar(x=x_bar, y=bar.get_height(), yerr=err, fmt='none', c='black', capsize=5, lw=2)
    
        # Set legend and labels
        ax.set_xlabel(self.grouping_column)

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_jitter_plot(self, ax):
        """Helper method to create a jitter plot (strip plot) with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str)  + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None  # Disable hue to avoid two-level grouping
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = None
    
        # Create the jitter plot
        sns.stripplot(data=self.df_melted,x=x_axis_column,y='Value',hue=self.hue, palette=self.sns_palette, dodge=self.jitter_bar_dodge, jitter=self.bar_width, ax=ax, alpha=0.6, size=16, order=self.order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)
       
        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_line_graph(self, ax):
        """Helper method to create a line graph with one line per group based on epochs and accuracy."""
        #display(self.df)
        # Ensure epoch is used on the x-axis and accuracy on the y-axis
        x_axis_column = self.data_column[0]
        y_axis_column = self.data_column[1]

        if self.log_y:
            self.df[y_axis_column] = np.log10(self.df[y_axis_column])
        
        if self.log_x:
            self.df[x_axis_column] = np.log10(self.df[x_axis_column])
        
        # Set hue to the grouping column to get one line per group
        hue = self.grouping_column

        # Check if the required columns exist in the DataFrame
        required_columns = [x_axis_column, y_axis_column, self.grouping_column]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Create the line graph with one line per group
        sns.lineplot(data=self.df,x=x_axis_column,y=y_axis_column,hue=hue,palette=self.sns_palette,ax=ax,marker='o',linewidth=1,markersize=6)

        # Adjust axis labels
        ax.set_xlabel(f"{x_axis_column}")
        ax.set_ylabel(f"{y_axis_column}")

    def _create_line_with_std_area(self, ax):
        """Helper method to create a line graph with shaded area representing standard deviation."""

        x_axis_column = self.data_column[0]
        y_axis_column = self.data_column[1]
        y_axis_column_mean = f"mean_{y_axis_column}"
        y_axis_column_std = f"std_{y_axis_column_mean}"
        
        if self.log_y:
            self.df[y_axis_column] = np.log10(self.df[y_axis_column])
        
        if self.log_x:
            self.df[x_axis_column] = np.log10(self.df[x_axis_column])

        # Pivot the DataFrame to get mean and std for each epoch across plates
        summary_df = self.df.pivot_table(index=x_axis_column,values=y_axis_column,aggfunc=['mean', 'std']).reset_index()
        
        # Flatten MultiIndex columns (result of pivoting)
        summary_df.columns = [x_axis_column, y_axis_column_mean, y_axis_column_std]
            
        # Plot the mean accuracy as a line
        sns.lineplot(data=summary_df,x=x_axis_column,y=y_axis_column_mean,ax=ax,marker='o',linewidth=1,markersize=0,color='blue',label=y_axis_column_mean)


        # Fill the area representing the standard deviation
        ax.fill_between(summary_df[x_axis_column],summary_df[y_axis_column_mean] - summary_df[y_axis_column_std],summary_df[y_axis_column_mean] + summary_df[y_axis_column_std],color='blue',  alpha=0.1 )

        # Adjust axis labels
        ax.set_xlabel(f"{x_axis_column}")
        ax.set_ylabel(f"{y_axis_column}")
        
    def _create_box_plot(self, ax):
        """Helper method to create a box plot with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = None
    
        # Create the box plot
        sns.boxplot(data=self.df_melted,x=x_axis_column,y='Value',hue=self.hue,palette=self.sns_palette,ax=ax, order=self.order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)

        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')
    
    def _create_violin_plot(self, ax):
        """Helper method to create a violin plot with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = None
    
        # Create the violin plot
        sns.violinplot(data=self.df_melted,x=x_axis_column,y='Value', hue=self.hue,palette=self.sns_palette,ax=ax, order=self.order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)
        ax.set_ylabel('Value')
    
        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_jitter_bar_plot(self, ax):
        """Helper method to create a bar plot with consistent bar thickness and centered error bars."""
        # Flatten DataFrame: Combine grouping column and data column into one group if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = None
    
        summary_df = self.df_melted.groupby([x_axis_column]).agg(mean=('Value', 'mean'),std=('Value', 'std'),sem=('Value', 'sem')).reset_index()
        error_bars = summary_df[self.error_bar_type] if self.error_bar_type in ['std', 'sem'] else None
        sns.barplot(data=self.df_melted, x=x_axis_column, y='Value', hue=self.hue, palette=self.sns_palette, ax=ax, dodge=self.jitter_bar_dodge, ci=None, order=self.order)
        sns.stripplot(data=self.df_melted,x=x_axis_column,y='Value',hue=self.hue, palette=self.sns_palette, dodge=self.jitter_bar_dodge, jitter=self.bar_width, ax=ax,alpha=0.6, edgecolor='white',linewidth=1, size=16, order=self.order)
        
        # Adjust the bar width manually
        if len(self.data_column) > 1:
            bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
            target_width = self.bar_width * 2
            for bar in bars:
                bar.set_width(target_width)  # Set new width
                # Center the bar on its x-coordinate
                bar.set_x(bar.get_x() - target_width / 2)
            
        # Adjust error bars alignment with bars
        #bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
        #for bar, (_, row) in zip(bars, summary_df.iterrows()):
        #    x_bar = bar.get_x() + bar.get_width() / 2
        #    err = row[self.error_bar_type]
        #    ax.errorbar(x=x_bar, y=bar.get_height(), yerr=err, fmt='none', c='black', capsize=5, lw=2)
    
        # Set legend and labels
        ax.set_xlabel(self.grouping_column)

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_jitter_box_plot(self, ax):
        """Helper method to create a box plot with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = None
    
        # Create the box plot
        sns.boxplot(data=self.df_melted,x=x_axis_column,y='Value',hue=self.hue,palette=self.sns_palette,ax=ax, order=self.order)
        sns.stripplot(data=self.df_melted,x=x_axis_column,y='Value',hue=self.hue, palette=self.sns_palette, dodge=self.jitter_bar_dodge, jitter=self.bar_width, ax=ax,alpha=0.6, edgecolor='white',linewidth=1, size=12, order=self.order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)

        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _save_results(self):
        """Helper method to save the plot and results."""
        os.makedirs(self.output_dir, exist_ok=True)
        plot_path = os.path.join(self.output_dir, f"{self.results_name}.pdf")
        self.fig.savefig(plot_path, bbox_inches='tight', dpi=600, transparent=True, format='pdf')
        results_path = os.path.join(self.output_dir, f"{self.results_name}.csv")
        self.results_df.to_csv(results_path, index=False)
        print(f"Plot saved to {plot_path}")
        print(f"Test results saved to {results_path}")

    def get_results(self):
        """Return the results dataframe."""
        return self.results_df
    
    def get_figure(self):
        """Return the generated figure."""
        return self.fig

def plot_data_from_db(settings):
    from .io import _read_db, _read_and_merge_data
    from .utils import annotate_conditions, save_settings
    """
    Extracts the specified table from the SQLite database and plots a specified column.

    Args:
        db_path (str): The path to the SQLite database.
        table_names (str): The name of the table to extract.
        data_column (str): The column to plot from the table.

    Returns:
        df (pd.DataFrame): The extracted table as a DataFrame.
    """



    if isinstance(settings['src'], str):
        srcs = [settings['src']]
    elif isinstance(settings['src'], list):
        srcs = settings['src']
    else:
        raise ValueError("src must be a string or a list of strings.")
    
    if isinstance(settings['database'], str):
        settings['database'] = [settings['database'] for _ in range(len(srcs))]
    
    settings['dst'] = os.path.join(srcs[0], 'results')
    
    save_settings(settings, name=f"{settings['graph_name']}_plot_settings_db", show=True)

    dfs = []
    for i, src in enumerate(srcs):

        db_loc = os.path.join(src, 'measurements', settings['database'][i])
        print(f"Database: {db_loc}")
        if settings['table_names'] in ['saliency_image_correlations']:
            print(f"Database table: {settings['table_names']}")
            [df1] = _read_db(db_loc, tables=[settings['table_names']])
        else:
            df1, _ = _read_and_merge_data(locs=[db_loc],
                                    tables = settings['table_names'],
                                    verbose=settings['verbose'],
                                    nuclei_limit=settings['nuclei_limit'],
                                    pathogen_limit=settings['pathogen_limit'])
     
        dft = annotate_conditions(df1, 
                                cells=settings['cell_types'], 
                                cell_loc=settings['cell_plate_metadata'], 
                                pathogens=settings['pathogen_types'],
                                pathogen_loc=settings['pathogen_plate_metadata'],
                                treatments=settings['treatments'], 
                                treatment_loc=settings['treatment_plate_metadata'])
        dfs.append(dft)
        
    df = pd.concat(dfs, axis=0)
    df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
    
    if settings['cell_plate_metadata'] !=  None:
        df = df.dropna(subset='host_cell')

    if settings['pathogen_plate_metadata'] !=  None:
        df = df.dropna(subset='pathogen')

    if settings['treatment_plate_metadata'] !=  None:
        df = df.dropna(subset='treatment')

    df = df.dropna(subset=settings['data_column'])
        
    if settings['grouping_column'] not in df.columns:
        print(f"Grouping column {settings['grouping_column']} not found in DataFrame.")
        print(f'Please use one of the following columns: {df.columns}')
        display(df)
        return None
    
    df = df.dropna(subset=settings['grouping_column'])

    src = srcs[0] 
    dst = os.path.join(src, 'results', settings['graph_name'])
    os.makedirs(dst, exist_ok=True)
    
    spacr_graph = spacrGraph(
        df=df,                                       # Your DataFrame
        grouping_column=settings['grouping_column'], # Column for grouping the data (x-axis)
        data_column=settings['data_column'],         # Column for the data (y-axis)
        graph_type=settings['graph_type'],           # Type of plot ('bar', 'box', 'violin', 'jitter')
        graph_name=settings['graph_name'],           # Name of the plot
        summary_func='mean',                         # Function to summarize data (e.g., 'mean', 'median')
        colors=None,                                 # Custom colors for the plot (optional)
        output_dir=dst,                              # Directory to save the plot and results
        save=settings['save'],                       # Whether to save the plot and results
        y_lim=settings['y_lim'],                     # Starting point for y-axis (optional)
        error_bar_type='std',                        # Type of error bar ('std' or 'sem')
        representation=settings['representation'],
        theme=settings['theme'],                     # Seaborn color palette theme (e.g., 'pastel', 'muted')
    )

    # Create the plot
    spacr_graph.create_plot()

    # Get the figure object if needed
    fig = spacr_graph.get_figure()
    plt.show()

    # Optional: Get the results DataFrame containing statistical test results
    results_df = spacr_graph.get_results()
    return fig, results_df

def plot_data_from_csv(settings):
    from .io import _read_db, _read_and_merge_data
    from .utils import annotate_conditions, save_settings
    """
    Extracts the specified table from the SQLite database and plots a specified column.

    Args:
        db_path (str): The path to the SQLite database.
        table_names (str): The name of the table to extract.
        data_column (str): The column to plot from the table.

    Returns:
        df (pd.DataFrame): The extracted table as a DataFrame.
    """

    if isinstance(settings['src'], str):
        srcs = [settings['src']]
    elif isinstance(settings['src'], list):
        srcs = settings['src']
    else:
        raise ValueError("src must be a string or a list of strings.")
    
    dfs = []
    for i, src in enumerate(srcs):
        dft = pd.read_csv(src)
        if 'plateID' not in dft.columns:
            dft['plateID'] = f"plate{i+1}"
            dft['common'] = 'spacr'
        dfs.append(dft)

    df = pd.concat(dfs, axis=0)
    
    if 'prc' in df.columns:
        # Check if 'plateID', 'rowID', and 'columnID' are all missing from df.columns
        if not all(col in df.columns for col in ['plate', 'rowID', 'columnID']):
            try:
                # Split 'prc' into 'plateID', 'rowID', and 'columnID'
                df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)
            except Exception as e:
                print(f"Could not split the prc column: {e}")

                
    display(df)

    df = df.dropna(subset=settings['data_column'])
    df = df.dropna(subset=settings['grouping_column'])
    src = srcs[0] 
    dst = os.path.join(os.path.dirname(src), 'results', settings['graph_name'])
    os.makedirs(dst, exist_ok=True)
    
    spacr_graph = spacrGraph(
        df=df,                                       # Your DataFrame
        grouping_column=settings['grouping_column'], # Column for grouping the data (x-axis)
        data_column=settings['data_column'],         # Column for the data (y-axis)
        graph_type=settings['graph_type'],           # Type of plot ('bar', 'box', 'violin', 'jitter')
        graph_name=settings['graph_name'],           # Name of the plot
        summary_func='mean',                         # Function to summarize data (e.g., 'mean', 'median')
        colors=None,                                 # Custom colors for the plot (optional)
        output_dir=dst,                              # Directory to save the plot and results
        save=settings['save'],                       # Whether to save the plot and results
        y_lim=settings['y_lim'],                     # Starting point for y-axis (optional)
        log_y=settings['log_y'],                     # Log-transform the y-axis
        log_x=settings['log_x'],                     # Log-transform the x-axis
        error_bar_type='std',                        # Type of error bar ('std' or 'sem')
        representation=settings['representation'],
        theme=settings['theme'],                     # Seaborn color palette theme (e.g., 'pastel', 'muted')
    )

    # Create the plot
    spacr_graph.create_plot()

    # Get the figure object if needed
    fig = spacr_graph.get_figure()
    plt.show()

    # Optional: Get the results DataFrame containing statistical test results
    results_df = spacr_graph.get_results()
    return fig, results_df

def plot_region(settings):

    def _sort_paths_by_basename(paths):
        return sorted(paths, key=lambda path: os.path.basename(path))
    
    def save_figure_as_pdf(fig, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if it doesn't exist
        fig.savefig(path, format='pdf', dpi=600, bbox_inches='tight')
        print(f"Saved {path}")

    from .io import _read_db
    from .utils import correct_paths
    fov_path = os.path.join(settings['src'], 'merged', settings['name'])
    name = os.path.splitext(settings['name'])[0]
    
    db_path = os.path.join(settings['src'], 'measurements', 'measurements.db')
    paths_df = _read_db(db_path, tables=['png_list'])[0]
    paths_df, _ = correct_paths(df=paths_df, base_path=settings['src'], folder='data')
    paths_df = paths_df[paths_df['png_path'].str.contains(name, na=False)]

    activation_mode = f"{settings['activation_mode']}_list"
    activation_db_path = os.path.join(settings['src'], 'measurements', settings['activation_db'])
    activation_paths_df = _read_db(activation_db_path, tables=[activation_mode])[0]
    activation_db = os.path.splitext(settings['activation_db'])[0]
    base_path=os.path.join(settings['src'], 'datasets',activation_db) 
    activation_paths_df, _ = correct_paths(df=activation_paths_df, base_path=base_path, folder=settings['activation_mode'])
    activation_paths_df = activation_paths_df[activation_paths_df['png_path'].str.contains(name, na=False)]

    png_paths = _sort_paths_by_basename(paths_df['png_path'].tolist())
    activation_paths = _sort_paths_by_basename(activation_paths_df['png_path'].tolist())

    
    if activation_paths:
        fig_3 = plot_image_grid(image_paths=activation_paths, percentiles=settings['percentiles'])
    else:
        fig_3 = None
        print(f"Could not find any cropped PNGs")
    if png_paths:
        fig_2 = plot_image_grid(image_paths=png_paths, percentiles=settings['percentiles'])
    else:
        fig_2 = None
        print(f"Could not find any activation maps")
    
    print('fov_path', fov_path)
    fig_1 = plot_image_mask_overlay(file=fov_path,
                                    channels=settings['channels'],
                                    cell_channel=settings['cell_channel'],
                                    nucleus_channel=settings['nucleus_channel'],
                                    pathogen_channel=settings['pathogen_channel'],
                                    figuresize=10,
                                    percentiles=settings['percentiles'],
                                    thickness=3, 
                                    save_pdf=True, 
                                    mode=settings['mode'],
                                    export_tiffs=settings['export_tiffs'])
    
    dst = os.path.join(settings['src'], 'results', name)
    
    if not fig_1 == None:
        save_figure_as_pdf(fig_1, os.path.join(dst, f"{name}_mask_overlay.pdf"))
    if not fig_2 == None:
        save_figure_as_pdf(fig_2, os.path.join(dst, f"{name}_png_grid.pdf"))
    if not fig_3 == None:
        save_figure_as_pdf(fig_3, os.path.join(dst, f"{name}_activation_grid.pdf"))
    
    return fig_1, fig_2, fig_3

def plot_image_grid(image_paths, percentiles):
    """
    Plots a square grid of images from a list of image paths. 
    Unused subplots are filled with black, and padding is minimized.

    Parameters:
    - image_paths: List of paths to images to be displayed.

    Returns:
    - fig: The generated matplotlib figure.
    """

    from PIL import Image
    import matplotlib.pyplot as plt
    import math

    def _normalize_image(image, percentiles=(2, 98)):
        """ Normalize the image to the given percentiles for each channel independently, preserving the input type (either PIL.Image or numpy.ndarray)."""
        
        # Check if the input is a PIL image and convert it to a NumPy array
        is_pil_image = isinstance(image, Image.Image)
        if is_pil_image:
            image = np.array(image)

        # If the image is single-channel, normalize directly
        if image.ndim == 2:
            v_min, v_max = np.percentile(image, percentiles)
            normalized_image = np.clip((image - v_min) / (v_max - v_min), 0, 1)
        else:
            # If multi-channel, normalize each channel independently
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                v_min, v_max = np.percentile(image[..., c], percentiles)
                normalized_image[..., c] = np.clip((image[..., c] - v_min) / (v_max - v_min), 0, 1)

        # If the input was a PIL image, convert the result back to PIL format
        if is_pil_image:
            # Ensure the image is converted back to 8-bit range (0-255) for PIL
            normalized_image = (normalized_image * 255).astype(np.uint8)
            return Image.fromarray(normalized_image)

        return normalized_image

    N = len(image_paths)
    # Calculate the smallest square grid size to fit all images
    grid_size = math.ceil(math.sqrt(N))  

    # Create the square grid of subplots with a black background
    fig, axs = plt.subplots(
        grid_size, grid_size, 
        figsize=(grid_size * 2, grid_size * 2),
        facecolor='black'  # Set figure background to black
    )

    # Flatten axs in case of a 2D array
    axs = axs.flatten()

    for i, img_path in enumerate(image_paths):
        ax = axs[i]

        # Load the image
        img = Image.open(img_path)
        img = _normalize_image(img, percentiles)

        # Display the image
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    # Fill any unused subplots with black
    for j in range(i + 1, len(axs)):
        axs[j].imshow([[0, 0, 0]], cmap='gray')  # Black square
        axs[j].axis('off')  # Hide axes

    # Adjust layout to minimize white space
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    return fig

def overlay_masks_on_images(img_folder, normalize=True, resize=True, save=False, plot=False, thickness=2):
    """
    Load images and masks from folders, overlay mask contours on images, and optionally normalize, resize, and save.

    Args:
        img_folder (str): Path to the folder containing images.
        mask_folder (str): Path to the folder containing masks.
        normalize (bool): If True, normalize images to the 1st and 99th percentiles.
        resize (bool): If True, resize the final overlay to 500x500.
        save (bool): If True, save the final overlay in an 'overlay' folder within the image folder.
        thickness (int): Thickness of the contour lines.
    """

    def normalize_image(image):
        """Normalize the image to the 1st and 99th percentiles."""
        lower, upper = np.percentile(image, [1, 99])
        image = np.clip((image - lower) / (upper - lower), 0, 1)
        return (image * 255).astype(np.uint8)

    
    mask_folder = os.path.join(img_folder,'masks')    
    overlay_folder = os.path.join(img_folder, "overlay")
    if save and not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder)

    # Get common filenames in both image and mask folders
    image_filenames = set(os.listdir(img_folder))
    mask_filenames = set(os.listdir(mask_folder))
    common_filenames = image_filenames.intersection(mask_filenames)

    if not common_filenames:
        print("No matching filenames found in both folders.")
        return

    for filename in common_filenames:
        # Load image and mask
        img_path = os.path.join(img_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        image = tiff.imread(img_path)
        mask = tiff.imread(mask_path)

        # Normalize the image if requested
        if normalize:
            image = normalize_image(image)

        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Resize the mask if it doesn't match the image size
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Generate contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to RGB if grayscale
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
            
        # Draw contours with alpha blending
        overlay = image_rgb.copy()
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness)
        blended = cv2.addWeighted(overlay, 0.7, image_rgb, 0.3, 0)
        
        # Resize the final overlay if requested
        if resize:
            blended = cv2.resize(blended, (1000, 1000), interpolation=cv2.INTER_AREA)

        # Save the overlay if requested
        if save:
            save_path = os.path.join(overlay_folder, filename)
            cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        if plot:
            # Display the result
            plt.figure(figsize=(10, 10))
            plt.imshow(blended)
            plt.title(f"Overlay: {filename}")
            plt.axis('off')
            plt.show()

def graph_importance(settings):
    
    from .settings import set_graph_importance_defaults
    from .utils import save_settings
    
    if not isinstance(settings['csvs'], list):
        settings['csvs'] = settings['csvs']
    
    settings['src'] = os.path.dirname(settings['csvs'][0])
    
    settings = set_graph_importance_defaults(settings)
    save_settings(settings, name='graph_importance')
    
    dfs = []
    for path in settings['csvs']:
        dft = pd.read_csv(path)
        dfs.append(dft)

    df = pd.concat(dfs)
    
    if not all(col in df.columns for col in (settings['grouping_column'], settings['data_column'])):
        print(f"grouping {settings['grouping_column']} and data {settings['data_column']} columns must be in {df.columns.to_list()}")
        return
    
    output_dir = os.path.dirname(settings['csvs'][0])
    
    spacr_graph = spacrGraph(
        df=df,                                     
        grouping_column=settings['grouping_column'],
        data_column=settings['data_column'],   
        graph_type=settings['graph_type'],   
        graph_name=settings['grouping_column'],
        summary_func='mean',                         
        colors=None,                                
        output_dir=output_dir,                              
        save=settings['save'],                       
        y_lim=None,                     
        error_bar_type='std',                       
        representation='object',
        theme='muted',                    
    )

    # Create the plot
    spacr_graph.create_plot()

    # Get the figure object if needed
    fig = spacr_graph.get_figure()
    plt.show()
    
def plot_proportion_stacked_bars(settings, df, group_column, bin_column, prc_column='prc', level='object', cmap='viridis'):
    """
    Generate a stacked bar plot for proportions and perform chi-squared and pairwise tests.
    
    Parameters:
    - settings (dict): Analysis settings.
    - df (DataFrame): Input data.
    - group_column (str): Column indicating the groups.
    - bin_column (str): Column indicating the categories.
    - prc_column (str): Optional; column for additional stratification.
    - level (str): Level of aggregation ('well' or 'object').
    
    Returns:
    - chi2 (float): Chi-squared statistic for the overall test.
    - p (float): p-value for the overall chi-squared test.
    - dof (int): Degrees of freedom for the overall chi-squared test.
    - expected (ndarray): Expected frequencies for the overall chi-squared test.
    - raw_counts (DataFrame): Contingency table of observed counts.
    - fig (Figure): The generated plot.
    - pairwise_results (list): Pairwise test results from `chi_pairwise`.
    """
    
    from .sp_stats import chi_pairwise
    
    # Calculate contingency table for overall chi-squared test
    raw_counts = df.groupby([group_column, bin_column]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(raw_counts)
    print(f"Chi-squared test statistic (raw data): {chi2:.4f}")
    print(f"p-value (raw data): {p:.4e}")

    # Perform pairwise comparisons
    pairwise_results = chi_pairwise(raw_counts, verbose=settings.get('verbose', False))

    # Plot based on level setting
    if level in ['well', 'plateID']:
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
            kind='bar', stacked=True, yerr=std_proportions, capsize=5, colormap=cmap, figsize=(12, 8)
        )
        plt.title('Proportion of Volume Bins by Group (Mean ± SD across wells)')
    else:
        # Object-level plotting without aggregation
        group_counts = df.groupby([group_column, bin_column]).size()
        group_totals = group_counts.groupby(level=0).sum()
        proportions = group_counts / group_totals
        proportion_df = proportions.unstack(fill_value=0)

        ax = proportion_df.plot(kind='bar', stacked=True, colormap=cmap, figsize=(12, 8))
        plt.title('Proportion of Volume Bins by Group')

    plt.xlabel('Group')
    plt.ylabel('Proportion')

    # Update legend with formatted labels, maintaining correct order
    plt.legend(title=f'Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    fig = plt.gcf()
    
    results_df = pd.DataFrame({
        'chi_squared_stat': [chi2],
        'p_value': [p],
        'degrees_of_freedom': [dof]
    })

    return results_df, pairwise_results, fig

def create_venn_diagram(file1, file2, gene_column="gene", filter_coeff=0.1, save=True, save_path=None):
    """
    Reads two CSV files, extracts the `gene` column, and creates a Venn diagram
    to show overlapping and non-overlapping genes.

    Parameters:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        gene_column (str): Name of the column containing gene data (default: "gene").
        filter_coeff (float): Coefficient threshold for filtering genes.
        save (bool): Whether to save the plot.
        save_path (str): Path to save the Venn diagram figure.

    Returns:
        dict: Overlapping and non-overlapping genes.
    """
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Filter based on coefficient
    if filter_coeff is not None:
        df1 = df1[df1['coefficient'] > filter_coeff] if filter_coeff >= 0 else df1[df1['coefficient'] < filter_coeff]
        df2 = df2[df2['coefficient'] > filter_coeff] if filter_coeff >= 0 else df2[df2['coefficient'] < filter_coeff]

    # Extract gene columns and drop NaN values
    genes1 = set(df1[gene_column].dropna())
    genes2 = set(df2[gene_column].dropna())

    # Calculate overlapping and non-overlapping genes
    overlapping_genes = genes1.intersection(genes2)
    unique_to_file1 = genes1.difference(genes2)
    unique_to_file2 = genes2.difference(genes1)

    # Create a Venn diagram
    plt.figure(figsize=(8, 6))
    venn = venn2([genes1, genes2], ('File 1 Genes', 'File 2 Genes'))
    plt.title("Venn Diagram of Overlapping Genes")

    # Save or show the figure
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format='pdf')
        print(f"Venn diagram saved to {save_path}")
    else:
        plt.show()

    # Return the results
    return {
        "overlap": list(overlapping_genes),
        "unique_to_file1": list(unique_to_file1),
        "unique_to_file2": list(unique_to_file2)
    }
