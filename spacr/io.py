import os, re, sqlite3, gc, torch, time, random, shutil, cv2, tarfile, cellpose, glob
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from collections import defaultdict, Counter
from pathlib import Path
from functools import partial
from matplotlib.animation import FuncAnimation
from IPython.display import display
from skimage.util import img_as_uint
from skimage.exposure import rescale_intensity
from skimage import filters
import skimage.measure as measure
from skimage import exposure
import imageio.v2 as imageio2
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import display, clear_output
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


from .logger import log_function_call

def _load_images_and_labels(image_files, label_files, circular=False, invert=False, image_extension="*.tif", label_extension="*.tif"):
    
    from .utils import invert_image, apply_mask
    
    images = []
    labels = []
    
    if not image_files is None:
        image_names = sorted([os.path.basename(f) for f in image_files])
    else:
        image_names = []
        
    if not label_files is None:
        label_names = sorted([os.path.basename(f) for f in label_files])
    else:
        label_names = []

    if not image_files is None and not label_files is None: 
        for img_file, lbl_file in zip(image_files, label_files):
            image = cellpose.io.imread(img_file)
            if invert:
                image = invert_image(image)
            if circular:
                image = apply_mask(image, output_value=0)
            label = cellpose.io.imread(lbl_file)
            if image.max() > 1:
                image = image / image.max()
            images.append(image)
            labels.append(label)
    elif not image_files is None:
        for img_file in image_files:
            image = cellpose.io.imread(img_file)
            if invert:
                image = invert_image(image)
            if circular:
                image = apply_mask(image, output_value=0)
            if image.max() > 1:
                image = image / image.max()
            images.append(image)
    elif not image_files is None:
            for lbl_file in label_files:
                label = cellpose.io.imread(lbl_file)
                if circular:
                    label = apply_mask(label, output_value=0)
            labels.append(label)
            
    if not image_files is None:
        image_dir = os.path.dirname(image_files[0])
    else:
        image_dir = None
        
    if not label_files is None:
        label_dir = os.path.dirname(label_files[0])
    else:
        label_dir = None
    
    # Log the number of loaded images and labels
    print(f'Loaded {len(images)} images and {len(labels)} labels from {image_dir} and {label_dir}')
    if len(labels) > 0 and len(images) > 0:
        print(f'image shape: {images[0].shape}, image type: images[0].shape mask shape: {labels[0].shape}, image type: labels[0].shape')
    return images, labels, image_names, label_names

def _load_normalized_images_and_labels_v1(image_files, label_files, channels=None, percentiles=None,  circular=False, invert=False, visualize=False, remove_background=False, background=0, Signal_to_noise=10):
    
    from .plot import normalize_and_visualize
    from .utils import invert_image, apply_mask

    signal_thresholds = background*Signal_to_noise
    lower_percentile = 2

    images = []
    labels = []
    
    num_channels = 4
    percentiles_1 = [[] for _ in range(num_channels)]
    percentiles_99 = [[] for _ in range(num_channels)]

    image_names = [os.path.basename(f) for f in image_files]
    
    if label_files is not None:
        label_names = [os.path.basename(f) for f in label_files]
        label_dir = os.path.dirname(label_files[0])

    # Load images and check percentiles
    for i,img_file in enumerate(image_files):
        image = cellpose.io.imread(img_file)
        if invert:
            image = invert_image(image)
        if circular:
            image = apply_mask(image, output_value=0)

        # If specific channels are specified, select them
        if channels is not None and image.ndim == 3:
            image = image[..., channels]

        if remove_background:
            image[image < background] = 0
        
        if image.ndim < 3:
            image = np.expand_dims(image, axis=-1)
        
        images.append(image)
        if percentiles is None:
            for c in range(image.shape[-1]):
                p1 = np.percentile(image[..., c], lower_percentile)
                percentiles_1[c].append(p1)
                for percentile in [98, 99, 99.9, 99.99, 99.999]:
                    p = np.percentile(image[..., c], percentile)
                    if p > signal_thresholds:
                        percentiles_99[c].append(p)
                        break
                    
    if not percentiles is None:
        normalized_images = []
        for image in images:
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                low_p = np.percentile(image[..., c], percentiles[0])
                high_p = np.percentile(image[..., c], percentiles[1])
                normalized_image[..., c] = rescale_intensity(image[..., c], in_range=(low_p, high_p), out_range=(0, 1))
            normalized_images.append(normalized_image)
            if visualize:
                normalize_and_visualize(image, normalized_image, title=f"Channel {c+1} Normalized")
            
    if percentiles is None:
        # Calculate average percentiles for normalization
        avg_p1 = [np.mean(p) for p in percentiles_1]
        avg_p99 = [np.mean(p) if len(p) > 0 else np.mean(percentiles_1[i]) for i, p in enumerate(percentiles_99)]

        print(f'Average 1st percentiles: {avg_p1}, Average 99th percentiles: {avg_p99}')

        normalized_images = []
        for image in images:
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                normalized_image[..., c] = rescale_intensity(image[..., c], in_range=(avg_p1[c], avg_p99[c]), out_range=(0, 1))
            normalized_images.append(normalized_image)
            if visualize:
                normalize_and_visualize(image, normalized_image, title=f"Channel {c+1} Normalized")
            
    if not image_files is None:
        image_dir = os.path.dirname(image_files[0])

    else:
        image_dir = None
            
    if label_files is not None:
        for lbl_file in label_files:
            labels.append(cellpose.io.imread(lbl_file))
    else:
        label_names = []
        label_dir = None

    print(f'Loaded and normalized {len(normalized_images)} images and {len(labels)} labels from {image_dir} and {label_dir}')
    
    return normalized_images, labels, image_names, label_names

def _load_normalized_images_and_labels(image_files, label_files, channels=None, percentiles=None,  circular=False, invert=False, visualize=False, remove_background=False, background=0, Signal_to_noise=10, target_height=None, target_width=None):
    
    from .plot import normalize_and_visualize, plot_resize
    from .utils import invert_image, apply_mask
    from skimage.transform import resize as resizescikit

    signal_thresholds = background * Signal_to_noise
    lower_percentile = 2

    images = []
    labels = []
    
    num_channels = 4
    percentiles_1 = [[] for _ in range(num_channels)]
    percentiles_99 = [[] for _ in range(num_channels)]

    image_names = [os.path.basename(f) for f in image_files]
    image_dir = os.path.dirname(image_files[0])
    
    if label_files is not None:
        label_names = [os.path.basename(f) for f in label_files]
        label_dir = os.path.dirname(label_files[0])

    # Load, normalize, and resize images
    for i, img_file in enumerate(image_files):
        image = cellpose.io.imread(img_file)
        if invert:
            image = invert_image(image)
        if circular:
            image = apply_mask(image, output_value=0)

        # If specific channels are specified, select them
        if channels is not None and image.ndim == 3:
            image = image[..., channels]

        if remove_background:
            image[image < background] = 0
        
        if image.ndim < 3:
            image = np.expand_dims(image, axis=-1)
        
        if percentiles is None:
            for c in range(image.shape[-1]):
                p1 = np.percentile(image[..., c], lower_percentile)
                percentiles_1[c].append(p1)
                for percentile in [98, 99, 99.9, 99.99, 99.999]:
                    p = np.percentile(image[..., c], percentile)
                    if p > signal_thresholds:
                        percentiles_99[c].append(p)
                        break
        
        # Resize image
        if target_height is not None and target_width is not None:
            if image.ndim == 2:
                image_shape = (target_height, target_width)
            elif image.ndim == 3:
                image_shape = (target_height, target_width, image.shape[-1])
                
            image = resizescikit(image, image_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
        
        images.append(image)
    
    if percentiles is None:
        # Calculate average percentiles for normalization
        avg_p1 = [np.mean(p) for p in percentiles_1]
        avg_p99 = [np.mean(p) if len(p) > 0 else np.mean(percentiles_1[i]) for i, p in enumerate(percentiles_99)]

        print(f'Average 1st percentiles: {avg_p1}, Average 99th percentiles: {avg_p99}')

        normalized_images = []
        for image in images:
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                normalized_image[..., c] = rescale_intensity(image[..., c], in_range=(avg_p1[c], avg_p99[c]), out_range=(0, 1))
            normalized_images.append(normalized_image)
            if visualize:
                normalize_and_visualize(image, normalized_image, title=f"Channel {c+1} Normalized")
    else:
        normalized_images = []
        for image in images:
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                low_p = np.percentile(image[..., c], percentiles[0])
                high_p = np.percentile(image[..., c], percentiles[1])
                normalized_image[..., c] = rescale_intensity(image[..., c], in_range=(low_p, high_p), out_range=(0, 1))
            normalized_images.append(normalized_image)
            if visualize:
                normalize_and_visualize(image, normalized_image, title=f"Channel {c+1} Normalized")

    if label_files is not None:
        for lbl_file in label_files:
            label = cellpose.io.imread(lbl_file)
            # Resize label
            if target_height is not None and target_width is not None:
                label = resizescikit(label, (target_height, target_width), order=0, preserve_range=True, anti_aliasing=False).astype(label.dtype)
            labels.append(label)
    else:
        label_names = []
        label_dir = None

    print(f'Loaded and normalized {len(normalized_images)} images and {len(labels)} labels from {image_dir} and {label_dir}')

    if visualize and images and labels:
        plot_resize(images, normalized_images, labels, labels)
    
    return normalized_images, labels, image_names, label_names

class CombineLoaders:

    """
    A class that combines multiple data loaders into a single iterator.

    Args:
        train_loaders (list): A list of data loaders.

    Attributes:
        train_loaders (list): A list of data loaders.
        loader_iters (list): A list of iterator objects for each data loader.

    Methods:
        __iter__(): Returns the iterator object itself.
        __next__(): Returns the next batch from one of the data loaders.

    Raises:
        StopIteration: If all data loaders have been exhausted.

    """

    def _init__(self, train_loaders):
        self.train_loaders = train_loaders
        self.loader_iters = [iter(loader) for loader in train_loaders]

    def _iter__(self):
        return self

    def _next__(self):
        while self.loader_iters:
            random.shuffle(self.loader_iters)  # Shuffle the loader_iters list
            for i, loader_iter in enumerate(self.loader_iters):
                try:
                    batch = next(loader_iter)
                    return i, batch
                except StopIteration:
                    self.loader_iters.pop(i)
                    continue
            else:
                break
        raise StopIteration

class CombinedDataset(Dataset):
    """
    A dataset that combines multiple datasets into one.

    Args:
        datasets (list): A list of datasets to be combined.
        shuffle (bool, optional): Whether to shuffle the combined dataset. Defaults to True.
    """

    def _init__(self, datasets, shuffle=True):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)
        self.shuffle = shuffle
        if shuffle:
            self.indices = list(range(self.total_length))
            random.shuffle(self.indices)
        else:
            self.indices = None
    def _getitem__(self, index):
        if self.shuffle:
            index = self.indices[index]
        for dataset, length in zip(self.datasets, self.lengths):
            if index < length:
                return dataset[index]
            index -= length
    def _len__(self):
        return self.total_length
    
class NoClassDataset(Dataset):
    """
    A custom dataset class for handling image data without class labels.
    
    Args:
        data_dir (str): The directory path where the image files are located.
        transform (callable, optional): A function/transform to apply to the image data. Default is None.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        load_to_memory (bool, optional): Whether to load all images into memory. Default is False.
    
    Attributes:
        data_dir (str): The directory path where the image files are located.
        transform (callable): A function/transform to apply to the image data.
        shuffle (bool): Whether to shuffle the dataset.
        load_to_memory (bool): Whether to load all images into memory.
        filenames (list): A list of file paths for the image files.
        images (list): A list of loaded images (if load_to_memory is True).
    """
    
    def __init__(self, data_dir, transform=None, shuffle=True, load_to_memory=False):
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        self.filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if self.shuffle:
            self.shuffle_dataset()
        if self.load_to_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    
    #@lru_cache(maxsize=None)
    def load_image(self, img_path):
        """
        Load an image from the given file path.
        
        Args:
            img_path (str): The file path of the image.
        
        Returns:
            PIL.Image: The loaded image.
        """
        img = Image.open(img_path).convert('RGB')
        return img
    
    def __len__(self):
        """
        Get the total number of images in the dataset.
        
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.filenames)
    
    def shuffle_dataset(self):
        """
        Shuffle the dataset.
        """
        if self.shuffle:
            random.shuffle(self.filenames)
    
    def __getitem__(self, index):
        """
        Get the image and its corresponding filename at the given index.
        
        Args:
            index (int): The index of the image in the dataset.
        
        Returns:
            tuple: A tuple containing the image and its filename.
        """
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = self.load_image(self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        # Return both the image and its filename
        return img, self.filenames[index]

class MyDataset(Dataset):
    """
    A custom dataset class for loading and processing image data.

    Args:
        data_dir (str): The directory path where the image data is stored.
        loader_classes (list): A list of class names for the dataset.
        transform (callable, optional): A function/transform to apply to the image data. Default is None.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        pin_memory (bool, optional): Whether to pin the loaded images to memory. Default is False.
        specific_files (list, optional): A list of specific file paths to include in the dataset. Default is None.
        specific_labels (list, optional): A list of specific labels corresponding to the specific files. Default is None.
    """

    def __init__(self, data_dir, loader_classes, transform=None, shuffle=True, pin_memory=False, specific_files=None, specific_labels=None):
        self.data_dir = data_dir
        self.classes = loader_classes
        self.transform = transform
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.filenames = []
        self.labels = []
        
        if specific_files and specific_labels:
            self.filenames = specific_files
            self.labels = specific_labels
        else:
            for class_name in self.classes:
                class_path = os.path.join(data_dir, class_name)
                class_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                self.filenames.extend(class_files)
                self.labels.extend([self.classes.index(class_name)] * len(class_files))
        
        if self.shuffle:
            self.shuffle_dataset()
            
        if self.pin_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img
    
    def __len__(self):
        return len(self.filenames)
    
    def shuffle_dataset(self):
        combined = list(zip(self.filenames, self.labels))
        random.shuffle(combined)
        self.filenames, self.labels = zip(*combined)
        
    def get_plate(self, filepath):
        filename = os.path.basename(filepath)  # Get just the filename from the full path
        return filename.split('_')[0]
    
    def __getitem__(self, index):
        label = self.labels[index]
        filename = self.filenames[index]
        img = self.load_image(filename)
        if self.transform:
            img = self.transform(img)
        return img, label, filename

class NoClassDataset(Dataset):
    def __init__(self, data_dir, transform=None, shuffle=True, load_to_memory=False):
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        self.filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if self.shuffle:
            self.shuffle_dataset()
        if self.load_to_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def __len__(self):

        return len(self.filenames)

    def shuffle_dataset(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __getitem__(self, index):
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = self.load_image(self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        return img, self.filenames[index]

    
class TarImageDataset(Dataset):
    def _init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform

        # Open the tar file just to build the list of members
        with tarfile.open(self.tar_path, 'r') as f:
            self.members = [m for m in f.getmembers() if m.isfile()]

    def _len__(self):
        return len(self.members)

    def _getitem__(self, idx):
        with tarfile.open(self.tar_path, 'r') as f:
            m = self.members[idx]
            img_file = f.extractfile(m)
            img = Image.open(BytesIO(img_file.read())).convert("RGB")
            
        if self.transform:
            img = self.transform(img)
        
        return img, m.name

#@log_function_call
def _rename_and_organize_image_files(src, regex, batch_size=100, pick_slice=False, skip_mode='01', metadata_type='', img_format='.tif'):
    """
    Convert z-stack images to maximum intensity projection (MIP) images.

    Args:
        src (str): The source directory containing the z-stack images.
        regex (str): The regular expression pattern used to match the filenames of the z-stack images.
        batch_size (int, optional): The number of images to process in each batch. Defaults to 100.
        pick_slice (bool, optional): Whether to pick a specific slice based on the provided skip mode. Defaults to False.
        skip_mode (str, optional): The skip mode used to filter out specific slices. Defaults to '01'.
        metadata_type (str, optional): The type of metadata associated with the images. Defaults to ''.

    Returns:
        None
    """
    
    from .utils import _extract_filename_metadata
    
    regular_expression = re.compile(regex)
    images_by_key = defaultdict(list)
    stack_path = os.path.join(src, 'stack')
    if not os.path.exists(stack_path) or (os.path.isdir(stack_path) and len(os.listdir(stack_path)) == 0):
        all_filenames = [filename for filename in os.listdir(src) if filename.endswith(img_format)]
        print(f'All_files:{len(all_filenames)} in {src}')
        for i in range(0, len(all_filenames), batch_size):
            batch_filenames = all_filenames[i:i+batch_size]
            for filename in batch_filenames:
                images_by_key = _extract_filename_metadata(batch_filenames, src, images_by_key, regular_expression, metadata_type, pick_slice, skip_mode)
            if pick_slice:
                for key in images_by_key:
                    plate, well, field, channel, mode = key
                    max_intensity_slice = max(images_by_key[key], key=lambda x: np.percentile(x, 90))
                    mip_image = Image.fromarray(max_intensity_slice)
                    output_dir = os.path.join(src, channel)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{plate}_{well}_{field}.tif'
                    output_path = os.path.join(output_dir, output_filename)

                    if os.path.exists(output_path):                        
                        print(f'WARNING: A file with the same name already exists at location {output_filename}')
                    else:
                        mip_image.save(output_path)
            else:
                for key, images in images_by_key.items():
                    mip = np.max(np.stack(images), axis=0)
                    mip_image = Image.fromarray(mip)
                    plate, well, field, channel = key[:4]
                    output_dir = os.path.join(src, channel)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{plate}_{well}_{field}.tif'
                    output_path = os.path.join(output_dir, output_filename)

                    if os.path.exists(output_path):                        
                        print(f'WARNING: A file with the same name already exists at location {output_filename}')
                    else:
                        mip_image.save(output_path)

            images_by_key.clear()

        # Move original images to a new directory
        valid_exts = [img_format]
        newpath = os.path.join(src, 'orig')
        os.makedirs(newpath, exist_ok=True)
        for filename in os.listdir(src):
            if os.path.splitext(filename)[1] in valid_exts:
                move = os.path.join(newpath, filename)
                if os.path.exists(move):
                    print(f'WARNING: A file with the same name already exists at location {move}')
                else:
                    shutil.move(os.path.join(src, filename), move)
    return

def _merge_file(chan_dirs, stack_dir, file_name):
    """
    Merge multiple channels into a single stack and save it as a numpy array, using os module for path handling.
    
    Args:
        chan_dirs (list): List of directories containing channel images.
        stack_dir (str): Directory to save the merged stack.
        file_name (str): File name of the channel image.

    Returns:
        None
    """
    # Construct new file path
    file_root, file_ext = os.path.splitext(file_name)
    new_file = os.path.join(stack_dir, file_root + '.npy')
    
    # Check if the new file exists and create the stack directory if it doesn't
    if not os.path.exists(new_file):
        os.makedirs(stack_dir, exist_ok=True)
        channels = []
        for i, chan_dir in enumerate(chan_dirs):
            img_path = os.path.join(chan_dir, file_name)
            img = cv2.imread(img_path, -1)
            if img is None:
                print(f"Warning: Failed to read image {img_path}")
                continue
            chan = np.expand_dims(img, axis=2)
            channels.append(chan)
            del img  # Explicitly delete the reference to the image to free up memory
            if i % 10 == 0:  # Periodically suggest garbage collection
                gc.collect()

        if channels:
            stack = np.concatenate(channels, axis=2)
            np.save(new_file, stack)
        else:
            print(f"No valid channels to merge for file {file_name}")

def _is_dir_empty(dir_path):
    """
    Check if a directory is empty using os module.
    """
    return len(os.listdir(dir_path)) == 0

def _generate_time_lists(file_list):
    """
    Generate sorted lists of filenames grouped by plate, well, and field.

    Args:
        file_list (list): A list of filenames.

    Returns:
        list: A list of sorted file lists, where each file list contains filenames
              belonging to the same plate, well, and field, sorted by timepoint.
    """
    file_dict = defaultdict(list)
    for filename in file_list:
        if filename.endswith('.npy'):
            parts = filename.split('_')
            if len(parts) >= 4:
                plate, well, field = parts[:3]
                try:
                    timepoint = int(parts[3].split('.')[0])
                except ValueError:
                    continue  # Skip file on conversion error
                key = (plate, well, field)
                file_dict[key].append((timepoint, filename))
            else:
                continue  # Skip file if not correctly formatted

    # Sort each list by timepoint, but keep them grouped
    sorted_grouped_filenames = [sorted(files, key=lambda x: x[0]) for files in file_dict.values()]
    # Extract just the filenames from each group
    sorted_file_lists = [[filename for _, filename in group] for group in sorted_grouped_filenames]

    return sorted_file_lists

def _move_to_chan_folder(src, regex, timelapse=False, metadata_type=''):
    
    from .utils import _safe_int_convert, _convert_cq1_well_id
    
    src_path = src
    src = Path(src)
    valid_exts = ['.tif', '.png']

    if not (src / 'stack').exists():
        for file in src.iterdir():
            if file.is_file():
                name, ext = file.stem, file.suffix
                if ext in valid_exts:
                    metadata = re.match(regex, file.name) 
                    try:                  
                        try:
                            plateID = metadata.group('plateID')
                        except:
                            plateID = src.name

                        wellID = metadata.group('wellID')
                        fieldID = metadata.group('fieldID')
                        chanID = metadata.group('chanID')
                        timeID = metadata.group('timeID')

                        if wellID[0].isdigit():
                            wellID = str(_safe_int_convert(wellID))
                        if fieldID[0].isdigit():
                            fieldID = str(_safe_int_convert(fieldID))
                        if chanID[0].isdigit():
                            chanID = str(_safe_int_convert(chanID))
                        if timeID[0].isdigit():
                            timeID = str(_safe_int_convert(timeID))

                        if metadata_type =='cq1':
                            orig_wellID = wellID
                            wellID = _convert_cq1_well_id(wellID)
                            print(f'Converted Well ID: {orig_wellID} to {wellID}')#, end='\r', flush=True)

                        newname = f"{plateID}_{wellID}_{fieldID}_{timeID if timelapse else ''}{ext}"
                        newpath = src / chanID
                        move = newpath / newname
                        if move.exists():
                            print(f'WARNING: A file with the same name already exists at location {move}')
                        else:
                            newpath.mkdir(exist_ok=True)
                            shutil.copy(file, move)
                    except:
                        print(f"Could not extract information from filename {name}{ext} with {regex}")

        # Move original images to a new directory
        valid_exts = ['.tif', '.png']
        newpath = os.path.join(src_path, 'orig')
        os.makedirs(newpath, exist_ok=True)
        for filename in os.listdir(src_path):
            if os.path.splitext(filename)[1] in valid_exts:
                move = os.path.join(newpath, filename)
                if os.path.exists(move):
                    print(f'WARNING: A file with the same name already exists at location {move}')
                else:
                    shutil.move(os.path.join(src, filename), move)
    return

def _merge_channels_v2(src, plot=False):
    from .plot import plot_arrays
    """
    Merge the channels in the given source directory and save the merged files in a 'stack' directory.

    Args:
        src (str): The path to the source directory containing the channel folders.
        plot (bool, optional): Whether to plot the merged arrays. Defaults to False.

    Returns:
        None
    """
    src = Path(src)
    stack_dir = src / 'stack'
    chan_dirs = [d for d in src.iterdir() if d.is_dir() and d.name in ['01', '02', '03', '04', '00', '1', '2', '3', '4','0']]

    chan_dirs.sort(key=lambda x: x.name)
    print(f'List of folders in src: {[d.name for d in chan_dirs]}. Single channel folders.')
    start_time = time.time()

    # First directory and its files
    dir_files = list(chan_dirs[0].iterdir())

    # Create the 'stack' directory if it doesn't exist
    stack_dir.mkdir(exist_ok=True)
    print(f'generated folder with merged arrays: {stack_dir}')

    if _is_dir_empty(stack_dir):
        with Pool(max(cpu_count() // 2, 1)) as pool:
        #with Pool(cpu_count()) as pool:
            merge_func = partial(_merge_file, chan_dirs, stack_dir)
            pool.map(merge_func, dir_files)

    avg_time = (time.time() - start_time) / len(dir_files)
    print(f'Average Time: {avg_time:.3f} sec')

    if plot:
        plot_arrays(src+'/stack')

    return

def _merge_channels(src, plot=False):
    """
    Merge the channels in the given source directory and save the merged files in a 'stack' directory without using multiprocessing.
    """

    from .plot import plot_arrays
    
    stack_dir = os.path.join(src, 'stack')
    allowed_names = ['01', '02', '03', '04', '00', '1', '2', '3', '4', '0']
    
    # List directories that match the allowed names
    chan_dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d in allowed_names]
    chan_dirs.sort()

    print(f'List of folders in src: {chan_dirs}. Single channel folders.')
    start_time = time.time()

    # Assuming chan_dirs[0] is not empty and exists, adjust according to your logic
    first_dir_path = os.path.join(src, chan_dirs[0])
    dir_files = os.listdir(first_dir_path)

    # Create the 'stack' directory if it doesn't exist
    if not os.path.exists(stack_dir):
        os.makedirs(stack_dir, exist_ok=True)
    print(f'Generated folder with merged arrays: {stack_dir}')

    if _is_dir_empty(stack_dir):
        for file_name in dir_files:
            full_file_path = os.path.join(first_dir_path, file_name)
            if os.path.isfile(full_file_path):
                _merge_file([os.path.join(src, d) for d in chan_dirs], stack_dir, file_name)

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / len(dir_files) if dir_files else 0
    print(f'Average Time: {avg_time:.3f} sec, Total Elapsed Time: {elapsed_time:.3f} sec')

    if plot:
        plot_arrays(os.path.join(src, 'stack'))

    return

def _mip_all(src, include_first_chan=True):
    
    """
    Generate maximum intensity projections (MIPs) for each NumPy array file in the specified directory.

    Args:
        src (str): The directory path containing the NumPy array files.
        include_first_chan (bool, optional): Whether to include the first channel of the array in the MIP computation. 
                                                Defaults to True.

    Returns:
        None
    """
    
    from .utils import normalize_to_dtype
    
    #print('========== generating MIPs ==========')
    # Iterate over each file in the specified directory (src).
    for filename in os.listdir(src):
        # Check if the current file is a NumPy array file (with .npy extension).
        if filename.endswith('.npy'):
            # Load the array from the file.
            array = np.load(os.path.join(src, filename))
            # Normalize the array using custom parameters (q1=2, q2=98).
            array = normalize_to_dtype(array, q1=2, q2=98, percentiles=None)

            if array.ndim != 3: # Check if the array is not 3-dimensional.
                # Log a message indicating a zero array will be generated due to unexpected dimensions.
                print(f"Generating zero array for {filename} due to unexpected dimensions: {array.shape}")
                # Create a zero array with the same height and width as the original array, but with a single depth layer.
                zeros_array = np.zeros((array.shape[0], array.shape[1], 1))
                # Concatenate the original array with the zero array along the depth axis.
                concatenated = np.concatenate([array, zeros_array], axis=2)
            else:
                if include_first_chan:
                    # Compute the MIP for the entire array along the third axis.
                    mip = np.max(array, axis=2)
                else:
                    # Compute the MIP excluding the first layer of the array along the depth axis.
                    mip = np.max(array[:, :, 1:], axis=2)
                # Reshape the MIP to make it 3-dimensional.
                mip = mip[:, :, np.newaxis]
                # Concatenate the MIP with the original array.
                concatenated = np.concatenate([array, mip], axis=2)
            # save
            np.save(os.path.join(src, filename), concatenated)
    return

#@log_function_call
def _concatenate_channel(src, channels, randomize=True, timelapse=False, batch_size=100):
    """
    Concatenates channel data from multiple files and saves the concatenated data as numpy arrays.

    Args:
        src (str): The source directory containing the channel data files.
        channels (list): The list of channel indices to be concatenated.
        randomize (bool, optional): Whether to randomize the order of the files. Defaults to True.
        timelapse (bool, optional): Whether the channel data is from a timelapse experiment. Defaults to False.
        batch_size (int, optional): The number of files to be processed in each batch. Defaults to 100.

    Returns:
        str: The directory path where the concatenated channel data is saved.
    """
    channels = [item for item in channels if item is not None]
    paths = []
    index = 0
    channel_stack_loc = os.path.join(os.path.dirname(src), 'channel_stack')
    os.makedirs(channel_stack_loc, exist_ok=True)
    if timelapse:
        try:
            time_stack_path_lists = _generate_time_lists(os.listdir(src))
            for i, time_stack_list in enumerate(time_stack_path_lists):
                stack_region = []
                filenames_region = []
                for idx, file in enumerate(time_stack_list):
                    path = os.path.join(src, file)
                    if idx == 0:
                        parts = file.split('_')
                        name = parts[0]+'_'+parts[1]+'_'+parts[2]
                    array = np.load(path)
                    array = np.take(array, channels, axis=2)
                    stack_region.append(array)
                    filenames_region.append(os.path.basename(path))
                #clear_output(wait=True)
                print(f'Region {i+1}/ {len(time_stack_path_lists)}', end='\r', flush=True)
                stack = np.stack(stack_region)
                save_loc = os.path.join(channel_stack_loc, f'{name}.npz')
                np.savez(save_loc, data=stack, filenames=filenames_region)
                print(save_loc)
                del stack
        except Exception as e:
            print(f"Error processing files, make sure filenames metadata is structured plate_well_field_time.npy")
            print(f"Error: {e}")
    else:
        for file in os.listdir(src):
            if file.endswith('.npy'):
                path = os.path.join(src, file)
                paths.append(path)
        if randomize:
            random.shuffle(paths)
        nr_files = len(paths)
        batch_index = 0  # Added this to name the output files
        stack_ls = []
        filenames_batch = []  # to hold filenames of the current batch
        for i, path in enumerate(paths):
            array = np.load(path)
            array = np.take(array, channels, axis=2)
            stack_ls.append(array)
            filenames_batch.append(os.path.basename(path))  # store the filename
            #clear_output(wait=True)
            print(f'Concatenated: {i+1}/{nr_files} files')
            #print(f'Concatenated: {i+1}/{nr_files} files', end='\r', flush=True)

            if (i+1) % batch_size == 0 or i+1 == nr_files:
                unique_shapes = {arr.shape[:-1] for arr in stack_ls}
                if len(unique_shapes) > 1:
                    max_dims = np.max(np.array(list(unique_shapes)), axis=0)
                    #clear_output(wait=True)
                    print(f'Warning: arrays with multiple shapes found in batch {i+1}. Padding arrays to max X,Y dimentions {max_dims}')
                    #print(f'Warning: arrays with multiple shapes found in batch {i+1}. Padding arrays to max X,Y dimentions {max_dims}', end='\r', flush=True)
                    padded_stack_ls = []
                    for arr in stack_ls:
                        pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_dims, arr.shape[:-1])]
                        pad_width.append((0, 0))
                        padded_arr = np.pad(arr, pad_width)
                        padded_stack_ls.append(padded_arr)
                    stack = np.stack(padded_stack_ls)
                else:
                    stack = np.stack(stack_ls)
                save_loc = os.path.join(channel_stack_loc, f'stack_{batch_index}.npz')
                np.savez(save_loc, data=stack, filenames=filenames_batch)
                batch_index += 1  # increment this after each batch is saved
                del stack  # delete to free memory
                stack_ls = []  # empty the list for the next batch
                filenames_batch = []  # empty the filenames list for the next batch
                padded_stack_ls = []
    print(f'All files concatenated and saved to:{channel_stack_loc}')
    return channel_stack_loc

def concatenate_and_normalize(src, channels, randomize=True, timelapse=False, batch_size=100, backgrounds=[100, 100, 100], remove_backgrounds=[False, False, False], lower_percentile=2, save_dtype=np.float32, signal_to_noise=[5, 5, 5], signal_thresholds=[1000, 1000, 1000]):
    """
    Concatenates and normalizes channel data from multiple files and saves the normalized data.

    Args:
        src (str): The source directory containing the channel data files.
        channels (list): The list of channel indices to be concatenated and normalized.
        randomize (bool, optional): Whether to randomize the order of the files. Defaults to True.
        timelapse (bool, optional): Whether the channel data is from a timelapse experiment. Defaults to False.
        batch_size (int, optional): The number of files to be processed in each batch. Defaults to 100.
        backgrounds (list, optional): Background values for each channel. Defaults to [100, 100, 100].
        remove_backgrounds (list, optional): Whether to remove background values for each channel. Defaults to [False, False, False].
        lower_percentile (int, optional): Lower percentile value for normalization. Defaults to 2.
        save_dtype (numpy.dtype, optional): Data type for saving the normalized stack. Defaults to np.float32.
        signal_to_noise (list, optional): Signal-to-noise ratio thresholds for each channel. Defaults to [5, 5, 5].
        signal_thresholds (list, optional): Signal thresholds for each channel. Defaults to [1000, 1000, 1000].

    Returns:
        str: The directory path where the concatenated and normalized channel data is saved.
    """
    channels = [item for item in channels if item is not None]
    paths = []
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)

    if timelapse:
        try:
            time_stack_path_lists = _generate_time_lists(os.listdir(src))
            for i, time_stack_list in enumerate(time_stack_path_lists):
                stack_region = []
                filenames_region = []
                for idx, file in enumerate(time_stack_list):
                    path = os.path.join(src, file)
                    if idx == 0:
                        parts = file.split('_')
                        name = parts[0] + '_' + parts[1] + '_' + parts[2]
                    array = np.load(path)
                    array = np.take(array, channels, axis=2)
                    stack_region.append(array)
                    filenames_region.append(os.path.basename(path))
                print(f'Region {i + 1}/ {len(time_stack_path_lists)}', end='\r', flush=True)
                stack = np.stack(stack_region)
                normalized_stack = _normalize_stack(stack, backgrounds, remove_backgrounds, lower_percentile, save_dtype, signal_to_noise, signal_thresholds)
                save_loc = os.path.join(output_fldr, f'{name}_norm_timelapse.npz')
                np.savez(save_loc, data=normalized_stack, filenames=filenames_region)
                print(save_loc)
                del stack, normalized_stack
        except Exception as e:
            print(f"Error processing files, make sure filenames metadata is structured plate_well_field_time.npy")
            print(f"Error: {e}")
    else:
        for file in os.listdir(src):
            if file.endswith('.npy'):
                path = os.path.join(src, file)
                paths.append(path)
        if randomize:
            random.shuffle(paths)
        nr_files = len(paths)
        batch_index = 0
        stack_ls = []
        filenames_batch = []

        for i, path in enumerate(paths):
            array = np.load(path)
            array = np.take(array, channels, axis=2)
            stack_ls.append(array)
            filenames_batch.append(os.path.basename(path))
            print(f'Concatenated: {i + 1}/{nr_files} files')

            if (i + 1) % batch_size == 0 or i + 1 == nr_files:
                unique_shapes = {arr.shape[:-1] for arr in stack_ls}
                if len(unique_shapes) > 1:
                    max_dims = np.max(np.array(list(unique_shapes)), axis=0)
                    print(f'Warning: arrays with multiple shapes found in batch {i + 1}. Padding arrays to max X,Y dimensions {max_dims}')
                    padded_stack_ls = []
                    for arr in stack_ls:
                        pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_dims, arr.shape[:-1])]
                        pad_width.append((0, 0))
                        padded_arr = np.pad(arr, pad_width)
                        padded_stack_ls.append(padded_arr)
                    stack = np.stack(padded_stack_ls)
                else:
                    stack = np.stack(stack_ls)

                normalized_stack = _normalize_img_batch(stack, backgrounds, remove_backgrounds, lower_percentile, save_dtype, signal_to_noise, signal_thresholds)

                save_loc = os.path.join(output_fldr, f'stack_{batch_index}_norm.npz')
                np.savez(save_loc, data=normalized_stack, filenames=filenames_batch)
                batch_index += 1
                del stack, normalized_stack
                stack_ls = []
                filenames_batch = []
                padded_stack_ls = []
    print(f'All files concatenated and normalized. Saved to: {output_fldr}')
    return output_fldr

def _normalize_img_batch(stack, backgrounds, remove_backgrounds, lower_percentile, save_dtype, signal_to_noise, signal_thresholds):
    """
    Normalize the stack of images.

    Args:
        stack (numpy.ndarray): The stack of images to normalize.
        backgrounds (list): Background values for each channel.
        remove_backgrounds (list): Whether to remove background values for each channel.
        lower_percentile (int): Lower percentile value for normalization.
        save_dtype (numpy.dtype): Data type for saving the normalized stack.
        signal_to_noise (list): Signal-to-noise ratio thresholds for each channel.
        signal_thresholds (list): Signal thresholds for each channel.

    Returns:
        numpy.ndarray: The normalized stack.
    """
    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    for chan_index, channel in enumerate(range(stack.shape[-1])):
        single_channel = stack[:, :, :, channel]
        background = backgrounds[chan_index]
        signal_threshold = signal_thresholds[chan_index]
        remove_background = remove_backgrounds[chan_index]

        print(f'Processing channel {chan_index}: background={background}, signal_threshold={signal_threshold}, remove_background={remove_background}')

        # Step 3: Remove background if required
        if remove_background:
            single_channel[single_channel < background] = 0

        # Step 4: Calculate global lower percentile for the channel
        non_zero_single_channel = single_channel[single_channel != 0]
        global_lower = np.percentile(non_zero_single_channel, lower_percentile)

        # Step 5: Calculate global upper percentile for the channel
        global_upper = None
        for upper_p in np.linspace(98, 99.5, num=16):
            upper_value = np.percentile(non_zero_single_channel, upper_p)
            if upper_value >= signal_threshold:
                global_upper = upper_value
                break

        if global_upper is None:
            global_upper = np.percentile(non_zero_single_channel, 99.5)  # Fallback in case no upper percentile met the threshold

        print(f'Channel {chan_index}: global_lower={global_lower}, global_upper={global_upper}, Signal-to-noise={global_upper / global_lower}')

        # Step 6: Normalize each array from global_lower to global_upper between 0 and 1
        for array_index in range(single_channel.shape[0]):
            arr_2d = single_channel[array_index, :, :]
            arr_2d_normalized = exposure.rescale_intensity(arr_2d, in_range=(global_lower, global_upper), out_range=(0, 1))
            normalized_stack[array_index, :, :, channel] = arr_2d_normalized

    return normalized_stack.astype(save_dtype)

def _get_lists_for_normalization(settings):
    """
    Get lists for normalization based on the provided settings.

    Args:
        settings (dict): A dictionary containing the settings for normalization.

    Returns:
        tuple: A tuple containing three lists - backgrounds, signal_to_noise, and signal_thresholds.
    """

    # Initialize the lists
    backgrounds = []
    signal_to_noise = []
    signal_thresholds = []
    remove_background = []

    # Iterate through the channels and append the corresponding values if the channel is not None
    for ch in settings['channels']:
        if ch == settings['nucleus_channel']:
            backgrounds.append(settings['nucleus_background'])
            signal_to_noise.append(settings['nucleus_Signal_to_noise'])
            signal_thresholds.append(settings['nucleus_Signal_to_noise']*settings['nucleus_background'])
            remove_background.append(settings['remove_background_nucleus'])
        elif ch == settings['cell_channel']:
            backgrounds.append(settings['cell_background'])
            signal_to_noise.append(settings['cell_Signal_to_noise'])
            signal_thresholds.append(settings['cell_Signal_to_noise']*settings['cell_background'])
            remove_background.append(settings['remove_background_cell'])
        elif ch == settings['pathogen_channel']:
            backgrounds.append(settings['pathogen_background'])
            signal_to_noise.append(settings['pathogen_Signal_to_noise'])
            signal_thresholds.append(settings['pathogen_Signal_to_noise']*settings['pathogen_background'])
            remove_background.append(settings['remove_background_pathogen'])
    return backgrounds, signal_to_noise, signal_thresholds, remove_background

def _normalize_stack(src, backgrounds=[100, 100, 100], remove_backgrounds=[False, False, False], lower_percentile=2, save_dtype=np.float32, signal_to_noise=[5, 5, 5], signal_thresholds=[1000, 1000, 1000]):
    """
    Normalize the stack of images.

    Args:
        src (str): The source directory containing the stack of images.
        backgrounds (list, optional): Background values for each channel. Defaults to [100, 100, 100].
        remove_background (list, optional): Whether to remove background values for each channel. Defaults to [False, False, False].
        lower_percentile (int, optional): Lower percentile value for normalization. Defaults to 2.
        save_dtype (numpy.dtype, optional): Data type for saving the normalized stack. Defaults to np.float32.
        signal_to_noise (list, optional): Signal-to-noise ratio thresholds for each channel. Defaults to [5, 5, 5].
        signal_thresholds (list, optional): Signal thresholds for each channel. Defaults to [1000, 1000, 1000].

    Returns:
        None
    """
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)
    time_ls = []
    
    for file_index, path in enumerate(paths):
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
        
        normalized_stack = np.zeros_like(stack, dtype=np.float32)
        file = os.path.basename(path)
        name, _ = os.path.splitext(file)

        for chan_index, channel in enumerate(range(stack.shape[-1])):
            single_channel = stack[:, :, :, channel]
            background = backgrounds[chan_index]
            signal_threshold = signal_thresholds[chan_index]
            remove_background = remove_backgrounds[chan_index]
            signal_2_noise = signal_to_noise[chan_index]
            print(f'chan_index:{chan_index} background:{background} signal_threshold:{signal_threshold} remove_background:{remove_background} signal_2_noise:{signal_2_noise}')

            if remove_background:
                single_channel[single_channel < background] = 0

            # Calculate the global lower and upper percentiles for non-zero pixels
            non_zero_single_channel = single_channel[single_channel != 0]
            global_lower = np.percentile(non_zero_single_channel, lower_percentile)
            for upper_p in np.linspace(98, 100, num=100).tolist():
                global_upper = np.percentile(non_zero_single_channel, upper_p)
                if global_upper >= signal_threshold:
                    break

            # Normalize the pixels in each image to the global percentiles and then dtype.
            arr_2d_normalized = np.zeros_like(single_channel, dtype=single_channel.dtype)
            signal_to_noise_ratio_ls = []
            for array_index in range(single_channel.shape[0]):
                start = time.time()
                arr_2d = single_channel[array_index, :, :]
                non_zero_arr_2d = arr_2d[arr_2d != 0]
                if non_zero_arr_2d.size > 0:
                    lower, upper = np.percentile(non_zero_arr_2d, (lower_percentile, upper_p))
                    signal_to_noise_ratio = upper / lower
                else:
                    signal_to_noise_ratio = 0
                signal_to_noise_ratio_ls.append(signal_to_noise_ratio)
                average_stnr = np.mean(signal_to_noise_ratio_ls) if len(signal_to_noise_ratio_ls) > 0 else 0

                if signal_to_noise_ratio > signal_2_noise:
                    arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(lower, upper), out_range=(0, 1))
                    arr_2d_normalized[array_index, :, :] = arr_2d_rescaled
                else:
                    arr_2d_normalized[array_index, :, :] = arr_2d
                stop = time.time()
                duration = (stop - start) * single_channel.shape[0]
                time_ls.append(duration)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                print(f'Progress: files {file_index + 1}/{len(paths)}, channels:{chan_index}/{stack.shape[-1] - 1}, arrays:{array_index + 1}/{single_channel.shape[0]}, Signal:{upper:.1f}, noise:{lower:.1f}, Signal-to-noise:{average_stnr:.1f}, Time/channel:{average_time:.2f}sec')

            normalized_stack[:, :, :, channel] = arr_2d_normalized
        
        save_loc = os.path.join(output_fldr, f'{name}_norm_stack.npz')
        np.savez(save_loc, data=normalized_stack.astype(save_dtype), filenames=filenames)
        del normalized_stack, single_channel, arr_2d_normalized, stack, filenames
        gc.collect()
    
    return print(f'Saved stacks: {output_fldr}')

def _normalize_timelapse(src, lower_percentile=2, save_dtype=np.float32):
    """
    Normalize the timelapse data by rescaling the intensity values based on percentiles.

    Args:
        src (str): The source directory containing the timelapse data files.
        lower_percentile (int, optional): The lower percentile used to calculate the intensity range. Defaults to 1.
        save_dtype (numpy.dtype, optional): The data type to save the normalized stack. Defaults to np.float32.
    """
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)

    for file_index, path in enumerate(paths):
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

        normalized_stack = np.zeros_like(stack, dtype=save_dtype)
        file = os.path.basename(path)
        name, _ = os.path.splitext(file)

        for chan_index in range(stack.shape[-1]):
            single_channel = stack[:, :, :, chan_index]

            for array_index in range(single_channel.shape[0]):
                arr_2d = single_channel[array_index]
                # Calculate the 1% and 98% percentiles for this specific image
                q_low = np.percentile(arr_2d[arr_2d != 0], lower_percentile)
                q_high = np.percentile(arr_2d[arr_2d != 0], 98)

                # Rescale intensity based on the calculated percentiles to fill the dtype range
                arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(q_low, q_high), out_range='dtype')
                normalized_stack[array_index, :, :, chan_index] = arr_2d_rescaled

                print(f'Progress: files {file_index+1}/{len(paths)}, channels:{chan_index+1}/{stack.shape[-1]}, arrays:{array_index+1}/{single_channel.shape[0]}', end='\r')

        save_loc = os.path.join(output_fldr, f'{name}_norm_timelapse.npz')
        np.savez(save_loc, data=normalized_stack, filenames=filenames)

        del normalized_stack, stack, filenames
        gc.collect()

    print(f'\nSaved normalized stacks: {output_fldr}')

def _create_movies_from_npy_per_channel(src, fps=10):
    """
    Create movies from numpy files per channel.

    Args:
        src (str): The source directory containing the numpy files.
        fps (int, optional): Frames per second for the output movies. Defaults to 10.
    """
    
    from .timelapse import _npz_to_movie
    
    master_path = os.path.dirname(src)
    save_path = os.path.join(master_path,'movies')
    os.makedirs(save_path, exist_ok=True)
    # Organize files by plate, well, field
    files = [f for f in os.listdir(src) if f.endswith('.npy')]
    organized_files = {}
    for f in files:
        match = re.match(r'(\w+)_(\w+)_(\w+)_(\d+)\.npy', f)
        if match:
            plate, well, field, time = match.groups()
            key = (plate, well, field)
            if key not in organized_files:
                organized_files[key] = []
            organized_files[key].append((int(time), os.path.join(src, f)))
    for key, file_list in organized_files.items():
        plate, well, field = key
        file_list.sort(key=lambda x: x[0])
        arrays = []
        filenames = []
        for f in file_list:
            array = np.load(f[1])
            #if array.dtype != np.uint8:
            #    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
            arrays.append(array)
            filenames.append(os.path.basename(f[1]))
        arrays = np.stack(arrays, axis=0)
    for channel in range(arrays.shape[-1]):
        # Extract the current channel for all time points
        channel_arrays = arrays[..., channel]
        # Flatten the channel data to compute global percentiles
        channel_data_flat = channel_arrays.reshape(-1)
        p1, p99 = np.percentile(channel_data_flat, [1, 99])
        # Normalize and rescale each array in the channel
        normalized_channel_arrays = [(np.clip((arr - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8) for arr in channel_arrays]
        # Convert the list of 2D arrays into a list of 3D arrays with a single channel
        normalized_channel_arrays_3d = [arr[..., np.newaxis] for arr in normalized_channel_arrays]
        # Save as movie for the current channel
        channel_save_path = os.path.join(save_path, f'{plate}_{well}_{field}_channel_{channel}.mp4')
        _npz_to_movie(normalized_channel_arrays_3d, filenames, channel_save_path, fps)

def delete_empty_subdirectories(folder_path):
    """
    Deletes all empty subdirectories in the specified folder.

    Args:
    - folder_path (str): The path to the folder in which to look for empty subdirectories.
    """
    # Check each item in the specified folder
    for dirpath, dirnames, filenames in os.walk(folder_path, topdown=False):
        # os.walk is used with topdown=False to start from the innermost directories and work upwards.
        for dirname in dirnames:
            # Construct the full path to the subdirectory
            full_dir_path = os.path.join(dirpath, dirname)
            # Try to remove the directory and catch any error (like if the directory is not empty)
            try:
                os.rmdir(full_dir_path)
                print(f"Deleted empty directory: {full_dir_path}")
            except OSError as e:
                continue
                # An error occurred, likely because the directory is not empty
                #print(f"Skipping non-empty directory: {full_dir_path}")

#@log_function_call
def preprocess_img_data(settings):
    
    from .plot import plot_arrays, _plot_4D_arrays
    from .utils import _run_test_mode, _get_regex, set_default_settings_preprocess_img_data
    
    """
    Preprocesses image data by converting z-stack images to maximum intensity projection (MIP) images.

    Args:
        src (str): The source directory containing the z-stack images.
        metadata_type (str, optional): The type of metadata associated with the images. Defaults to 'cellvoyager'.
        custom_regex (str, optional): The custom regular expression pattern used to match the filenames of the z-stack images. Defaults to None.
        cmap (str, optional): The colormap used for plotting. Defaults to 'inferno'.
        figuresize (int, optional): The size of the figure for plotting. Defaults to 15.
        normalize (bool, optional): Whether to normalize the images. Defaults to False.
        nr (int, optional): The number of images to preprocess. Defaults to 1.
        plot (bool, optional): Whether to plot the images. Defaults to False.
        mask_channels (list, optional): The channels to use for masking. Defaults to [0, 1, 2].
        batch_size (list, optional): The number of images to process in each batch. Defaults to [100, 100, 100].
        timelapse (bool, optional): Whether the images are from a timelapse experiment. Defaults to False.
        remove_background (bool, optional): Whether to remove the background from the images. Defaults to False.
        backgrounds (int, optional): The number of background images to use for background removal. Defaults to 100.
        lower_percentile (float, optional): The lower percentile used for background removal. Defaults to 1.
        save_dtype (type, optional): The data type used for saving the preprocessed images. Defaults to np.float32.
        randomize (bool, optional): Whether to randomize the order of the images. Defaults to True.
        all_to_mip (bool, optional): Whether to convert all images to MIP. Defaults to False.
        pick_slice (bool, optional): Whether to pick a specific slice based on the provided skip mode. Defaults to False.
        skip_mode (str, optional): The skip mode used to filter out specific slices. Defaults to '01'.
        settings (dict, optional): Additional settings for preprocessing. Defaults to {}.

    Returns:
        None
    """

    src = settings['src']
    valid_ext = ['tif', 'tiff', 'png', 'jpeg']
    files = os.listdir(src)
    extensions = [file.split('.')[-1] for file in files]
    extension_counts = Counter(extensions)
    most_common_extension = extension_counts.most_common(1)[0][0]
    img_format = None
    
    delete_empty_subdirectories(src)

    # Check if the most common extension is one of the specified image formats
    if most_common_extension in valid_ext:
        img_format = f'.{most_common_extension}'
        print(f'Found {extension_counts[most_common_extension]} {most_common_extension} files')
    else:
        print(f'Could not find any {valid_ext} files in {src} only found {extension_counts[0]}')
        if os.path.exists(src+'/stack'):
            print('Found existing stack folder.')
        if os.path.exists(src+'/channel_stack'):
            print('Found existing channel_stack folder.')
        if os.path.exists(src+'/norm_channel_stack'):
            print('Found existing norm_channel_stack folder. Skipping preprocessing')
            return settings, src
        
    mask_channels = [settings['nucleus_channel'], settings['cell_channel'], settings['pathogen_channel']]
    backgrounds = [settings['nucleus_background'], settings['cell_background'], settings['pathogen_background']]

    settings, metadata_type, custom_regex, nr, plot, batch_size, timelapse, lower_percentile, randomize, all_to_mip, pick_slice, skip_mode, cmap, figuresize, normalize, save_dtype, test_mode, test_images, random_test = set_default_settings_preprocess_img_data(settings)

    regex = _get_regex(metadata_type, img_format, custom_regex)

    if test_mode:

        print(f'Running spacr in test mode')
        settings['plot'] = True
        try:
            os.rmdir(os.path.join(src, 'test'))
            print(f"Deleted test directory: {os.path.join(src, 'test')}")
        except OSError as e:
            pass

        src = _run_test_mode(settings['src'], regex, timelapse, test_images, random_test)
        settings['src'] = src

    if img_format == None:
        if not os.path.exists(src+'/stack'):
            _merge_channels(src, plot=False)   
    
    if not os.path.exists(src+'/stack'):
        try:
            if not img_format == None:
                if timelapse:
                    _move_to_chan_folder(src, regex, timelapse, metadata_type)
                else:
                    _rename_and_organize_image_files(src, regex, batch_size, pick_slice, skip_mode, metadata_type, img_format)
                    
                    #Make sure no batches will be of only one image
                    all_imgs = len(src+'/stack')
                    full_batches = all_imgs // batch_size
                    last_batch_size = all_imgs % batch_size
                    
                    # Check if the last batch is of size 1
                    if last_batch_size == 1:
                        # If there's only one batch and its size is 1, it's also an issue
                        if full_batches == 0:
                            raise ValueError("Only one batch of size 1 detected. Adjust the batch size.")
                        # If the last batch is of size 1, merge it with the second last batch
                        elif full_batches > 0:
                            raise ValueError("Last batch of size 1 detected. Adjust the batch size.")
        
                _merge_channels(src, plot=False)

                if timelapse:
                    _create_movies_from_npy_per_channel(src+'/stack', fps=2)

                if plot:
                    print(f'plotting {nr} images from {src}/stack')
                    plot_arrays(src+'/stack', figuresize, cmap, nr=nr, normalize=normalize)
                if all_to_mip:
                    _mip_all(src+'/stack')
                    if plot:
                        print(f'plotting {nr} images from {src}/stack')
                        plot_arrays(src+'/stack', figuresize, cmap, nr=nr, normalize=normalize)
        except Exception as e:
            print(f"Error: {e}")
    
    backgrounds, signal_to_noise, signal_thresholds, remove_backgrounds = _get_lists_for_normalization(settings=settings)
    
    concatenate_and_normalize(src+'/stack',
                              mask_channels,
                              randomize,
                              timelapse,
                              batch_size,
                              backgrounds,
                              remove_backgrounds,
                              lower_percentile,
                              np.float32,
                              signal_to_noise,
                              signal_thresholds)

    if plot:
        _plot_4D_arrays(src+'/norm_channel_stack', nr_npz=1, nr=nr)

    return settings, src
    
def _check_masks(batch, batch_filenames, output_folder):
    """
    Check the masks in a batch and filter out the ones that already exist in the output folder.

    Args:
        batch (list): List of masks.
        batch_filenames (list): List of filenames corresponding to the masks.
        output_folder (str): Path to the output folder.

    Returns:
        tuple: A tuple containing the filtered batch (numpy array) and the filtered filenames (list).
    """
    # Create a mask for filenames that are already present in the output folder
    existing_files_mask = [not os.path.isfile(os.path.join(output_folder, filename)) for filename in batch_filenames]

    # Use the mask to filter the batch and batch_filenames
    filtered_batch = [b for b, exists in zip(batch, existing_files_mask) if exists]
    filtered_filenames = [f for f, exists in zip(batch_filenames, existing_files_mask) if exists]

    return np.array(filtered_batch), filtered_filenames

def _get_avg_object_size(masks):
    """
    Calculate the average size of objects in a list of masks.

    Parameters:
    masks (list): A list of masks representing objects.

    Returns:
    float: The average size of objects in the masks. Returns 0 if no objects are found.
    """
    object_areas = []
    for mask in masks:
        # Check if the mask is a 2D or 3D array and is not empty
        if mask.ndim in [2, 3] and np.any(mask):
            properties = measure.regionprops(mask)
            object_areas += [prop.area for prop in properties]
        else:
            if not np.any(mask):
                print(f"Mask is empty. ")
            if not mask.ndim in [2, 3]:
                print(f"Mask is not in the correct format. dim: {mask.ndim}")
            continue

    if object_areas:
        return sum(object_areas) / len(object_areas)
    else:
        return 0  # Return 0 if no objects are found
    
def _save_figure(fig, src, text, dpi=300, i=1, all_folders=1):
    """
    Save a figure to a specified location.

    Parameters:
    fig (matplotlib.figure.Figure): The figure to be saved.
    src (str): The source file path.
    text (str): The text to be included in the figure name.
    dpi (int, optional): The resolution of the saved figure. Defaults to 300.
    """
    save_folder = os.path.dirname(src)
    obj_type = os.path.basename(src)
    name = os.path.basename(save_folder)
    save_folder = os.path.join(save_folder, 'figure')
    os.makedirs(save_folder, exist_ok=True)
    fig_name = f'{obj_type}_{name}_{text}.pdf'        
    save_location = os.path.join(save_folder, fig_name)
    fig.savefig(save_location, bbox_inches='tight', dpi=dpi)
    clear_output(wait=True)
    print(f'Progress: {i}/{all_folders}, Saved single cell figure: {os.path.basename(save_location)}')
    #print(f'Progress: {i}/{all_folders}, Saved single cell figure: {os.path.basename(save_location)}', end='\r', flush=True)
    # Close and delete the figure to free up memory
    plt.close(fig)
    del fig
    gc.collect()
    
def _read_and_join_tables(db_path, table_names=['cell', 'cytoplasm', 'nucleus', 'pathogen', 'parasite', 'png_list']):
    """
    Reads and joins tables from a SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.
        table_names (list, optional): The names of the tables to read and join. Defaults to ['cell', 'cytoplasm', 'nucleus', 'pathogen', 'parasite', 'png_list'].

    Returns:
        pandas.DataFrame: The joined DataFrame containing the data from the specified tables, or None if an error occurs.
    """
    conn = sqlite3.connect(db_path)
    dataframes = {}
    for table_name in table_names:
        try:
            dataframes[table_name] = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
            print(f"Table {table_name} not found in the database.")
            print(e)
    conn.close()
    if 'png_list' in dataframes:
        png_list_df = dataframes['png_list'][['cell_id', 'png_path', 'plate', 'row', 'col']].copy()
        png_list_df['cell_id'] = png_list_df['cell_id'].str[1:].astype(int)
        png_list_df.rename(columns={'cell_id': 'object_label'}, inplace=True)
        if 'cell' in dataframes:
            join_cols = ['object_label', 'plate', 'row', 'col']
            dataframes['cell'] = pd.merge(dataframes['cell'], png_list_df, on=join_cols, how='left')
        else:
            print("Cell table not found. Cannot join with png_list.")
            return None
    for entity in ['nucleus', 'pathogen', 'parasite']:
        if entity in dataframes:
            numeric_cols = dataframes[entity].select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = dataframes[entity].select_dtypes(exclude=[np.number]).columns.tolist()
            agg_dict = {col: 'mean' for col in numeric_cols}
            agg_dict.update({col: 'first' for col in non_numeric_cols if col not in ['cell_id', 'prcf']})
            grouping_cols = ['cell_id', 'prcf']
            agg_df = dataframes[entity].groupby(grouping_cols).agg(agg_dict)
            agg_df['count_' + entity] = dataframes[entity].groupby(grouping_cols).size()
            dataframes[entity] = agg_df
    joined_df = None
    if 'cell' in dataframes:
        joined_df = dataframes['cell']
        if 'cytoplasm' in dataframes:
            joined_df = pd.merge(joined_df, dataframes['cytoplasm'], on=['object_label', 'prcf'], how='left', suffixes=('', '_cytoplasm'))
        for entity in ['nucleus', 'pathogen']:
            if entity in dataframes:
                joined_df = pd.merge(joined_df, dataframes[entity], left_on=['object_label', 'prcf'], right_index=True, how='left', suffixes=('', f'_{entity}'))
    else:
        print("Cell table not found. Cannot proceed with joining.")
        return None
    return joined_df
    
def _save_settings_to_db(settings):
    """
    Save the settings dictionary to a SQLite database.

    Args:
        settings (dict): A dictionary containing the settings.

    Returns:
        None
    """
    # Convert the settings dictionary into a DataFrame
    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
    # Convert all values in the 'setting_value' column to strings
    settings_df['setting_value'] = settings_df['setting_value'].apply(str)
    display(settings_df)
    # Determine the directory path
    src = os.path.dirname(settings['input_folder'])
    directory = f'{src}/measurements'
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Database connection and saving the settings DataFrame
    conn = sqlite3.connect(f'{directory}/measurements.db', timeout=5)
    settings_df.to_sql('settings', conn, if_exists='replace', index=False)  # Replace the table if it already exists
    conn.close()

def _save_mask_timelapse_as_gif(masks, tracks_df, path, cmap, norm, filenames):
    """
    Save a timelapse animation of masks as a GIF.

    Parameters:
    - masks (list): List of mask frames.
    - tracks_df (pandas.DataFrame): DataFrame containing track information.
    - path (str): Path to save the GIF file.
    - cmap (str or matplotlib.colors.Colormap): Colormap for displaying the masks.
    - norm (matplotlib.colors.Normalize): Normalization for the colormap.
    - filenames (list): List of filenames corresponding to each mask frame.

    Returns:
    None
    """
    # Set the face color for the figure to black
    fig, ax = plt.subplots(figsize=(50, 50), facecolor='black')
    ax.set_facecolor('black')  # Set the axes background color to black
    ax.axis('off')  # Turn off the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # Adjust the subplot edges

    filename_text_obj = None  # Initialize a variable to keep track of the text object

    def _update(frame):
        """
        Update the frame of the animation.

        Parameters:
        - frame (int): The frame number to update.

        Returns:
        None
        """
        nonlocal filename_text_obj  # Reference the nonlocal variable to update it
        if filename_text_obj is not None:
            filename_text_obj.remove()  # Remove the previous text object if it exists

        ax.clear()  # Clear the axis to draw the new frame
        ax.axis('off')  # Ensure axis is still off after clearing
        current_mask = masks[frame]
        ax.imshow(current_mask, cmap=cmap, norm=norm)
        ax.set_title(f'Frame: {frame}', fontsize=24, color='white')

        # Add the filename as text on the figure
        filename_text = filenames[frame]  # Get the filename corresponding to the current frame
        filename_text_obj = fig.text(0.5, 0.01, filename_text, ha='center', va='center', fontsize=20, color='white')  # Adjust text position, size, and color as needed

        # Annotate each object with its label number from the mask
        for label_value in np.unique(current_mask):
            if label_value == 0: continue  # Skip background
            y, x = np.mean(np.where(current_mask == label_value), axis=1)
            ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

        # Overlay tracks
        if tracks_df is not None:
            for track in tracks_df['track_id'].unique():
                _track = tracks_df[tracks_df['track_id'] == track]
                ax.plot(_track['x'], _track['y'], '-w', linewidth=1)

    anim = FuncAnimation(fig, _update, frames=len(masks), blit=False)
    anim.save(path, writer='pillow', fps=2, dpi=80)  # Adjust DPI for size/quality
    plt.close(fig)
    print(f'Saved timelapse to {path}')

def _save_object_counts_to_database(arrays, object_type, file_names, db_path, added_string):
    """
    Save the counts of unique objects in masks to a SQLite database.

    Args:
        arrays (List[np.ndarray]): List of masks.
        object_type (str): Type of object.
        file_names (List[str]): List of file names corresponding to the masks.
        db_path (str): Path to the SQLite database.
        added_string (str): Additional string to append to the count type.

    Returns:
        None
    """
    def _count_objects(mask):
        """Count unique objects in a mask, assuming 0 is the background."""
        unique, counts = np.unique(mask, return_counts=True)
        # Assuming 0 is the background label, remove it from the count
        if unique[0] == 0:
            return len(unique) - 1
        return len(unique)

    records = []
    for mask, file_name in zip(arrays, file_names):
        object_count = _count_objects(mask)
        count_type = f"{object_type}{added_string}"

        # Append a tuple of (file_name, count_type, object_count) to the records list
        records.append((file_name, count_type, object_count))

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS object_counts (
        file_name TEXT,
        count_type TEXT,
        object_count INTEGER,
        PRIMARY KEY (file_name, count_type)
    )
    ''')

    # Batch insert or update the object counts
    cursor.executemany('''
    INSERT INTO object_counts (file_name, count_type, object_count)
    VALUES (?, ?, ?)
    ON CONFLICT(file_name, count_type) DO UPDATE SET
    object_count = excluded.object_count
    ''', records)

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

def _create_database(db_path):
    """
    Creates a SQLite database at the specified path.

    Args:
        db_path (str): The path where the database should be created.

    Returns:
        None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close() 
    
def _load_and_concatenate_arrays(src, channels, cell_chann_dim, nucleus_chann_dim, pathogen_chann_dim):
    """
    Load and concatenate arrays from multiple folders.

    Args:
        src (str): The source directory containing the arrays.
        channels (list): List of channel indices to select from the arrays.
        cell_chann_dim (int): Dimension of the cell channel.
        nucleus_chann_dim (int): Dimension of the nucleus channel.
        pathogen_chann_dim (int): Dimension of the pathogen channel.

    Returns:
        None
    """
    folder_paths = [os.path.join(src+'/stack')]

    if cell_chann_dim is not None or os.path.exists(os.path.join(src, 'norm_channel_stack', 'cell_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'norm_channel_stack','cell_mask_stack')]
    if nucleus_chann_dim is not None or os.path.exists(os.path.join(src, 'norm_channel_stack', 'nucleus_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'norm_channel_stack','nucleus_mask_stack')]
    if pathogen_chann_dim is not None or os.path.exists(os.path.join(src, 'norm_channel_stack', 'pathogen_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'norm_channel_stack','pathogen_mask_stack')]

    output_folder = src+'/merged'
    reference_folder = folder_paths[0]
    os.makedirs(output_folder, exist_ok=True)

    count=0
    all_imgs = len(os.listdir(reference_folder))

    # Iterate through each file in the reference folder
    for filename in os.listdir(reference_folder):
        stack_ls = []
        if filename.endswith('.npy'):
            count += 1

            # Check if this file exists in all the other specified folders
            exists_in_all_folders = all(os.path.isfile(os.path.join(folder, filename)) for folder in folder_paths)

            if exists_in_all_folders:
                # Load and potentially modify the array from the reference folder
                ref_array_path = os.path.join(reference_folder, filename)
                concatenated_array = np.load(ref_array_path)

                if channels is not None:
                    concatenated_array = np.take(concatenated_array, channels, axis=2)

                # Add the array from the reference folder to 'stack_ls'
                stack_ls.append(concatenated_array)

                # For each of the other folders, load the array and add it to 'stack_ls'
                for folder in folder_paths[1:]:
                    array_path = os.path.join(folder, filename)
                    array = np.load(array_path)
                    if array.ndim == 2:
                        array = np.expand_dims(array, axis=-1)  # Add an extra dimension if the array is 2D
                    stack_ls.append(array)

            if len(stack_ls) > 0:
                stack_ls = [np.expand_dims(arr, axis=-1) if arr.ndim == 2 else arr for arr in stack_ls]
                unique_shapes = {arr.shape[:-1] for arr in stack_ls}
                if len(unique_shapes) > 1:
                    #max_dims = np.max(np.array(list(unique_shapes)), axis=0)
                    # Determine the maximum length of tuples in unique_shapes
                    max_tuple_length = max(len(shape) for shape in unique_shapes)
                    # Pad shorter tuples with zeros to make them all the same length
                    padded_shapes = [shape + (0,) * (max_tuple_length - len(shape)) for shape in unique_shapes]
                    # Now create a NumPy array and find the maximum dimensions
                    max_dims = np.max(np.array(padded_shapes), axis=0)
                    #clear_output(wait=True)
                    print(f'Warning: arrays with multiple shapes found. Padding arrays to max X,Y dimentions {max_dims}')
                    #print(f'Warning: arrays with multiple shapes found. Padding arrays to max X,Y dimentions {max_dims}', end='\r', flush=True)
                    padded_stack_ls = []
                    for arr in stack_ls:
                        pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_dims, arr.shape[:-1])]
                        pad_width.append((0, 0))
                        padded_arr = np.pad(arr, pad_width)
                        padded_stack_ls.append(padded_arr)
                    # Concatenate the padded arrays along the channel dimension (last dimension)
                    stack = np.concatenate(padded_stack_ls, axis=-1)

                else:
                    stack = np.concatenate(stack_ls, axis=-1)

                if stack.shape[-1] > concatenated_array.shape[-1]:
                    output_path = os.path.join(output_folder, filename)
                    np.save(output_path, stack)
        
        #clear_output(wait=True)
        print(f'Files merged: {count}/{all_imgs}')
        #print(f'Files merged: {count}/{all_imgs}', end='\r', flush=True)
    return
    
def _read_db(db_loc, tables):
    """
    Read data from a SQLite database.

    Parameters:
    - db_loc (str): The location of the SQLite database file.
    - tables (list): A list of table names to read from.

    Returns:
    - dfs (list): A list of pandas DataFrames, each containing the data from a table.
    """
    conn = sqlite3.connect(db_loc)
    dfs = []
    for table in tables:
        query = f'SELECT * FROM {table}'
        df = pd.read_sql_query(query, conn)
        dfs.append(df)
    conn.close()
    return dfs
    
def _read_and_merge_data(locs, tables, verbose=False, include_multinucleated=False, include_multiinfected=False, include_noninfected=False):
    """
    Read and merge data from SQLite databases and perform data preprocessing.

    Parameters:
    - locs (list): A list of file paths to the SQLite database files.
    - tables (list): A list of table names to read from the databases.
    - verbose (bool): Whether to print verbose output. Default is False.
    - include_multinucleated (bool): Whether to include multinucleated cells. Default is False.
    - include_multiinfected (bool): Whether to include cells with multiple infections. Default is False.
    - include_noninfected (bool): Whether to include non-infected cells. Default is False.

    Returns:
    - merged_df (pandas.DataFrame): The merged and preprocessed dataframe.
    - obj_df_ls (list): A list of pandas DataFrames, each containing the data for a specific object type.
    """

    from .utils import _split_data
    
    #Extract plate DataFrames
    all_dfs = []
    for loc in locs:
        db_dfs = _read_db(loc, tables)
        all_dfs.append(db_dfs)
    
    #Extract Tables from DataFrames and concatinate rows
    for i, dfs in enumerate(all_dfs):
        if 'cell' in tables:
            cell = dfs[0]
            print(f'plate: {i+1} cells:{len(cell)}')

        if 'nucleus' in tables:
            nucleus = dfs[1]
            print(f'plate: {i+1} nucleus:{len(nucleus)} ')

        if 'pathogen' in tables:
            pathogen = dfs[2]
            
            print(f'plate: {i+1} pathogens:{len(pathogen)}')
        if 'cytoplasm' in tables:
            if not 'pathogen' in tables:
                cytoplasm = dfs[2]
            else:
                cytoplasm = dfs[3]
            print(f'plate: {i+1} cytoplasms: {len(cytoplasm)}')

        if i > 0:
            if 'cell' in tables:
                cells = pd.concat([cells, cell], axis = 0)
            if 'nucleus' in tables:
                nucleus = pd.concat([nucleus, nucleus], axis = 0)
            if 'pathogen' in tables:
                pathogens = pd.concat([pathogens, pathogen], axis = 0)
            if 'cytoplasm' in tables:
                cytoplasms = pd.concat([cytoplasms, cytoplasm], axis = 0)
        else:
            if 'cell' in tables:
                cells = cell.copy()
            if 'nucleus' in tables:
                nucleus = nucleus.copy()
            if 'pathogen' in tables:
                pathogens = pathogen.copy()
            if 'cytoplasm' in tables:
                cytoplasms = cytoplasm.copy()
    
    #Add an o in front of all object and cell lables to convert them to strings
    if 'cell' in tables:
        cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cells = cells.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cells_g_df, metadata = _split_data(cells, 'prcfo', 'object_label')
        print(f'cells: {len(cells)}')
        print(f'cells grouped: {len(cells_g_df)}')
    if 'cytoplasm' in tables:
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cytoplasms_g_df, _ = _split_data(cytoplasms, 'prcfo', 'object_label')
        merged_df = cells_g_df.merge(cytoplasms_g_df, left_index=True, right_index=True)
        print(f'cytoplasms: {len(cytoplasms)}')
        print(f'cytoplasms grouped: {len(cytoplasms_g_df)}')
    if 'nucleus' in tables:
        nucleus = nucleus.dropna(subset=['cell_id'])
        nucleus = nucleus.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nucleus = nucleus.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nucleus = nucleus.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        nucleus['nucleus_prcfo_count'] = nucleus.groupby('prcfo')['prcfo'].transform('count')
        if include_multinucleated == False:
            #nucleus = nucleus[~nucleus['prcfo'].duplicated()]
            nucleus = nucleus[nucleus['nucleus_prcfo_count']==1]
        nucleus_g_df, _ = _split_data(nucleus, 'prcfo', 'cell_id')
        print(f'nucleus: {len(nucleus)}')
        print(f'nucleus grouped: {len(nucleus_g_df)}')
        if 'cytoplasm' in tables:
            merged_df = merged_df.merge(nucleus_g_df, left_index=True, right_index=True)
        else:
            merged_df = cells_g_df.merge(nucleus_g_df, left_index=True, right_index=True)
    if 'pathogen' in tables:
        pathogens = pathogens.dropna(subset=['cell_id'])
        pathogens = pathogens.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        pathogens = pathogens.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        pathogens = pathogens.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        pathogens['pathogen_prcfo_count'] = pathogens.groupby('prcfo')['prcfo'].transform('count')
        if include_noninfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']>=1]
        if include_multiinfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']<=1]
        pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
        print(f'pathogens: {len(pathogens)}')
        print(f'pathogens grouped: {len(pathogens_g_df)}')
        merged_df = merged_df.merge(pathogens_g_df, left_index=True, right_index=True)
    
    #Add prc column (plate row column)
    metadata = metadata.assign(prc = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col'])

    #Count cells per well
    cells_well = pd.DataFrame(metadata.groupby('prc')['object_label'].nunique())

    cells_well.reset_index(inplace=True)
    cells_well.rename(columns={'object_label': 'cells_per_well'}, inplace=True)
    metadata = pd.merge(metadata, cells_well, on='prc', how='inner', suffixes=('', '_drop_col'))
    object_label_cols = [col for col in metadata.columns if '_drop_col' in col]
    metadata.drop(columns=object_label_cols, inplace=True)

    #Add prcfo column (plate row column field object)
    metadata = metadata.assign(prcfo = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col']+ '_' +x['field']+ '_' +x['object_label'])
    metadata.set_index('prcfo', inplace=True)

    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    
    merged_df = merged_df.dropna(axis=1)
    
    print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')
    
    obj_df_ls = []
    if 'cell' in tables:
        obj_df_ls.append(cells)
    if 'cytoplasm' in tables:
        obj_df_ls.append(cytoplasms)
    if 'nucleus' in tables:
        obj_df_ls.append(nucleus)
    if 'pathogen' in tables:
        obj_df_ls.append(pathogens)
        
    return merged_df, obj_df_ls
    
def _results_to_csv(src, df, df_well):
    """
    Save the given dataframes as CSV files in the specified directory.

    Args:
        src (str): The directory path where the CSV files will be saved.
        df (pandas.DataFrame): The dataframe containing cell data.
        df_well (pandas.DataFrame): The dataframe containing well data.

    Returns:
        tuple: A tuple containing the cell dataframe and well dataframe.
    """
    cells = df
    wells = df_well
    results_loc = src+'/results'
    wells_loc = results_loc+'/wells.csv'
    cells_loc = results_loc+'/cells.csv'
    os.makedirs(results_loc, exist_ok=True)
    wells.to_csv(wells_loc, index=True, header=True)
    cells.to_csv(cells_loc, index=True, header=True)
    return cells, wells
    
###################################################
#  Classify
###################################################
    
def _save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94]):
    """
    Save the model based on certain conditions during training.

    Args:
        model (torch.nn.Module): The trained model to be saved.
        model_type (str): The type of the model.
        results_df (pandas.DataFrame): The dataframe containing the training results.
        dst (str): The destination directory to save the model.
        epoch (int): The current epoch number.
        epochs (int): The total number of epochs.
        intermedeate_save (list, optional): List of accuracy thresholds to trigger intermediate model saves. 
                                            Defaults to [0.99, 0.98, 0.95, 0.94].
    """
    
    if epoch % 100 == 0:
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}.pth')
        
    if epoch == epochs:
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}.pth')
    
    if results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[0] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[0]:
        percentile = str(intermedeate_save[0]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')

    elif results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[1] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[1]:
        percentile = str(intermedeate_save[1]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')
    
    elif results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[2] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[2]:
        percentile = str(intermedeate_save[2]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')
    
    elif results_df['neg_accuracy'].dropna().mean() >= intermedeate_save[3] and results_df['pos_accuracy'].dropna().mean() >= intermedeate_save[3]:
        percentile = str(intermedeate_save[3]*100)
        print(f'\rfound: {percentile}% accurate model', end='\r', flush=True)
        torch.save(model, f'{dst}/{model_type}_epoch_{str(epoch)}_acc_{str(percentile)}.pth')

def _save_progress(dst, results_df, train_metrics_df):
    """
    Save the progress of the classification model.

    Parameters:
    dst (str): The destination directory to save the progress.
    results_df (pandas.DataFrame): The DataFrame containing accuracy, loss, and PRAUC.
    train_metrics_df (pandas.DataFrame): The DataFrame containing training metrics.

    Returns:
    None
    """
    # Save accuracy, loss, PRAUC
    os.makedirs(dst, exist_ok=True)
    results_path = os.path.join(dst, 'acc_loss_prauc.csv')
    if not os.path.exists(results_path):
        results_df.to_csv(results_path, index=True, header=True, mode='w')
    else:
        results_df.to_csv(results_path, index=True, header=False, mode='a')
    training_metrics_path = os.path.join(dst, 'training_metrics.csv')
    if not os.path.exists(training_metrics_path):
        train_metrics_df.to_csv(training_metrics_path, index=True, header=True, mode='w')
    else:
        train_metrics_df.to_csv(training_metrics_path, index=True, header=False, mode='a')
    return

def _save_settings(settings, src):
    """
    Save the settings dictionary to a CSV file.

    Parameters:
    - settings (dict): A dictionary containing the settings.
    - src (str): The source directory where the settings file will be saved.

    Returns:
    None
    """
    dst = os.path.join(src,'model')
    settings_loc =  os.path.join(dst,'settings.csv')
    os.makedirs(dst, exist_ok=True)
    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
    display(settings_df)
    settings_df.to_csv(settings_loc, index=False)
    return
    
    
def _copy_missclassified(df):
    misclassified = df[df['true_label'] != df['predicted_label']]
    for _, row in misclassified.iterrows():
        original_path = row['filename']
        filename = os.path.basename(original_path)
        dest_folder = os.path.dirname(os.path.dirname(original_path))
        if "pc" in original_path:
            new_path = os.path.join(dest_folder, "missclassified/pc", filename)
        else:
            new_path = os.path.join(dest_folder, "missclassified/nc", filename)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy(original_path, new_path)
    print(f"Copied {len(misclassified)} misclassified images.")
    return
    
def _read_db(db_loc, tables):
    conn = sqlite3.connect(db_loc) # Create a connection to the database
    dfs = []
    for table in tables:
        query = f'SELECT * FROM {table}' # Write a SQL query to get the data from the database
        df = pd.read_sql_query(query, conn) # Use the read_sql_query function to get the data and save it as a DataFrame
        dfs.append(df)
    conn.close() # Close the connection
    return dfs
    
def _read_and_merge_data(locs, tables, verbose=False, include_multinucleated=False, include_multiinfected=False, include_noninfected=False):
    
    from .utils import _split_data
    
    #Extract plate DataFrames
    all_dfs = []
    for loc in locs:
        db_dfs = _read_db(loc, tables)
        all_dfs.append(db_dfs)
    
    #Extract Tables from DataFrames and concatinate rows
    for i, dfs in enumerate(all_dfs):
        if 'cell' in tables:
            cell = dfs[0]
            if verbose:
                print(f'plate: {i+1} cells:{len(cell)}')
	# see pathogens logic, copy logic to other tables #here
        if 'nucleus' in tables:
            nucleus = dfs[1]
            if verbose:
                print(f'plate: {i+1} nucleus:{len(nucleus)} ')

        if 'pathogen' in tables:
            if len(tables) == 1:
                pathogen = dfs[0]
                print(len(pathogen))
            else:
                pathogen = dfs[2]
            if verbose:
                print(f'plate: {i+1} pathogens:{len(pathogen)}')
        
        if 'cytoplasm' in tables:
            if not 'pathogen' in tables:
                cytoplasm = dfs[2]
            else:
                cytoplasm = dfs[3]
            if verbose:
                print(f'plate: {i+1} cytoplasms: {len(cytoplasm)}')

        if i > 0:
            if 'cell' in tables:
                cells = pd.concat([cells, cell], axis = 0)
            if 'nucleus' in tables:
                nucleus = pd.concat([nucleus, nucleus], axis = 0)
            if 'pathogen' in tables:
                pathogens = pd.concat([pathogens, pathogen], axis = 0)
            if 'cytoplasm' in tables:
                cytoplasms = pd.concat([cytoplasms, cytoplasm], axis = 0)
        else:
            if 'cell' in tables:
                cells = cell.copy()
            if 'nucleus' in tables:
                nucleus = nucleus.copy()
            if 'pathogen' in tables:
                pathogens = pathogen.copy()
            if 'cytoplasm' in tables:
                cytoplasms = cytoplasm.copy()
    
    #Add an o in front of all object and cell lables to convert them to strings
    if 'cell' in tables:
        cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cells = cells.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cells_g_df, metadata = _split_data(cells, 'prcfo', 'object_label')
        merged_df = cells_g_df.copy()
        if verbose:
            print(f'cells: {len(cells)}')
            print(f'cells grouped: {len(cells_g_df)}')
		
    if 'cytoplasm' in tables:
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cytoplasms_g_df, _ = _split_data(cytoplasms, 'prcfo', 'object_label')
        merged_df = cells_g_df.merge(cytoplasms_g_df, left_index=True, right_index=True)
        if verbose:
            print(f'cytoplasms: {len(cytoplasms)}')
            print(f'cytoplasms grouped: {len(cytoplasms_g_df)}')
		
    if 'nucleus' in tables:
        if not 'cell' in tables:
            cells_g_df = pd.DataFrame()
        nucleus = nucleus.dropna(subset=['cell_id'])
        nucleus = nucleus.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nucleus = nucleus.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nucleus = nucleus.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        nucleus['nucleus_prcfo_count'] = nucleus.groupby('prcfo')['prcfo'].transform('count')
        if include_multinucleated == False:
            nucleus = nucleus[nucleus['nucleus_prcfo_count']==1]
        nucleus_g_df, _ = _split_data(nucleus, 'prcfo', 'cell_id')
        if verbose:
            print(f'nucleus: {len(nucleus)}')
            print(f'nucleus grouped: {len(nucleus_g_df)}')
        if 'cytoplasm' in tables:
            merged_df = merged_df.merge(nucleus_g_df, left_index=True, right_index=True)
        else:
            merged_df = cells_g_df.merge(nucleus_g_df, left_index=True, right_index=True)
		
    if 'pathogen' in tables:
        if not 'cell' in tables:
            cells_g_df = pd.DataFrame()
            merged_df = []
        try:
            pathogens = pathogens.dropna(subset=['cell_id'])

        except:
            pathogens['cell_id'] = pathogens['object_label']
            pathogens = pathogens.dropna(subset=['cell_id'])
		
        pathogens = pathogens.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        pathogens = pathogens.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        pathogens = pathogens.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        pathogens['pathogen_prcfo_count'] = pathogens.groupby('prcfo')['prcfo'].transform('count')
        if include_noninfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']>=1]
        if isinstance(include_multiinfected, bool):
            if include_multiinfected == False:
                pathogens = pathogens[pathogens['pathogen_prcfo_count']<=1]
        if isinstance(include_multiinfected, float):
            pathogens = pathogens[pathogens['pathogen_prcfo_count']<=include_multiinfected]
        if not 'cell' in tables:
            pathogens_g_df, metadata = _split_data(pathogens, 'prcfo', 'cell_id')
        else:
            pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
        if verbose:
            print(f'pathogens: {len(pathogens)}')
            print(f'pathogens grouped: {len(pathogens_g_df)}')
        if len(merged_df) == 0:
            merged_df = pathogens_g_df
        else:
            merged_df = merged_df.merge(pathogens_g_df, left_index=True, right_index=True)
        
    #Add prc column (plate row column)
    metadata = metadata.assign(prc = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col'])

    #Count cells per well
    cells_well = pd.DataFrame(metadata.groupby('prc')['object_label'].nunique())

    cells_well.reset_index(inplace=True)
    cells_well.rename(columns={'object_label': 'cells_per_well'}, inplace=True)
    metadata = pd.merge(metadata, cells_well, on='prc', how='inner', suffixes=('', '_drop_col'))
    object_label_cols = [col for col in metadata.columns if '_drop_col' in col]
    metadata.drop(columns=object_label_cols, inplace=True)

    #Add prcfo column (plate row column field object)
    metadata = metadata.assign(prcfo = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col']+ '_' +x['field']+ '_' +x['object_label'])
    metadata.set_index('prcfo', inplace=True)

    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    
    merged_df = merged_df.dropna(axis=1)
    if verbose:
        print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')
    
    obj_df_ls = []
    if 'cell' in tables:
        obj_df_ls.append(cells)
    if 'cytoplasm' in tables:
        obj_df_ls.append(cytoplasms)
    if 'nucleus' in tables:
        obj_df_ls.append(nucleus)
    if 'pathogen' in tables:
        obj_df_ls.append(pathogens)
        
    return merged_df, obj_df_ls  
    
def _read_mask(mask_path):
    mask = imageio2.imread(mask_path)
    if mask.dtype != np.uint16:
        mask = img_as_uint(mask)
    return mask


def convert_numpy_to_tiff(folder_path, limit=None):
    """
    Converts all numpy files in a folder to TIFF format and saves them in a subdirectory 'tiff'.
    
    Args:
    folder_path (str): The path to the folder containing numpy files.
    """
    # Create the subdirectory 'tiff' within the specified folder if it doesn't already exist
    tiff_subdir = os.path.join(folder_path, 'tiff')
    os.makedirs(tiff_subdir, exist_ok=True)

    files = os.listdir(folder_path)

    npy_files = [f for f in files if f.endswith('.npy')]
    
    # Iterate over all files in the folder
    for i, filename in enumerate(files):
        if limit is not None and i >= limit:
            break
        if not filename.endswith('.npy'):
            continue

        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Load the numpy file
        numpy_array = np.load(file_path)
        
        # Construct the output TIFF file path
        tiff_filename = os.path.splitext(filename)[0] + '.tif'
        tiff_file_path = os.path.join(tiff_subdir, tiff_filename)
        
        # Save the numpy array as a TIFF file
        tifffile.imwrite(tiff_file_path, numpy_array)
        
        print(f"Converted {filename} to {tiff_filename} and saved in 'tiff' subdirectory.")
    return
    
def generate_cellpose_train_test(src, test_split=0.1):
    mask_src = os.path.join(src, 'masks')
    img_paths = glob.glob(os.path.join(src, '*.tif'))
    img_filenames = [os.path.basename(file) for file in img_paths]
    img_filenames = [file for file in img_filenames if os.path.exists(os.path.join(mask_src, file))]
    print(f'Found {len(img_filenames)} images with masks')
    
    random.shuffle(img_filenames)
    split_index = int(len(img_filenames) * test_split)
    train_files = img_filenames[split_index:]
    test_files = img_filenames[:split_index]
    list_of_lists = [test_files, train_files]
    print(f'Split dataset into Train {len(train_files)} and Test {len(test_files)} files')
    
    train_dir = os.path.join(os.path.dirname(src), 'train')
    train_dir_masks = os.path.join(train_dir, 'masks')
    test_dir = os.path.join(os.path.dirname(src), 'test')
    test_dir_masks = os.path.join(test_dir, 'masks')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(train_dir_masks, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_dir_masks, exist_ok=True)
    
    for i, ls in enumerate(list_of_lists):
        if i == 0:
            dst = test_dir
            dst_mask = test_dir_masks
            _type = 'Test'
        else:
            dst = train_dir
            dst_mask = train_dir_masks
            _type = 'Train'
            
        for idx, filename in enumerate(ls):
            img_path = os.path.join(src, filename)
            mask_path = os.path.join(mask_src, filename)
            new_img_path = os.path.join(dst, filename)
            new_mask_path = os.path.join(dst_mask, filename)            
            shutil.copy(img_path, new_img_path)
            shutil.copy(mask_path, new_mask_path)
            print(f'Copied {idx+1}/{len(ls)} images to {_type} set', end='\r', flush=True)
            