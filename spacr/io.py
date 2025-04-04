import os, re, sqlite3, gc, torch, time, random, shutil, cv2, tarfile, cellpose, glob, queue, tifffile, czifile, atexit, datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from collections import defaultdict, Counter
from pathlib import Path
from functools import partial
from matplotlib.animation import FuncAnimation
from IPython.display import display
from skimage.util import img_as_uint
from skimage.exposure import rescale_intensity
import skimage.measure as measure
from skimage import exposure
import imageio.v2 as imageio2
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import display
from multiprocessing import Pool, cpu_count, Process, Queue, Value, Lock
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import seaborn as sns 
from nd2reader import ND2Reader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import readlif

def process_non_tif_non_2D_images(folder):
    """Processes all images in the folder and splits them into grayscale channels, preserving bit depth."""
    
    # Helper function to save grayscale images
    def save_grayscale_images(image, base_name, folder, dtype, channel=None, z=None, t=None):
        """Save grayscale images with appropriate suffix based on channel, z, and t, preserving bit depth."""
        suffix = ""
        if channel is not None:
            suffix += f"_C{channel}"
        if z is not None:
            suffix += f"_Z{z}"
        if t is not None:
            suffix += f"_T{t}"

        output_filename = os.path.join(folder, f"{base_name}{suffix}.tif")
        tifffile.imwrite(output_filename, image.astype(dtype))

    # Function to handle splitting of multi-dimensional images into grayscale channels
    def split_channels(image, folder, base_name, dtype):
        """Splits the image into channels and handles 3D, 4D, and 5D image cases."""
        if image.ndim == 2:
            # Grayscale image, already processed separately
            return
        
        elif image.ndim == 3:
            # 3D image: (height, width, channels)
            for c in range(image.shape[2]):
                save_grayscale_images(image[..., c], base_name, folder, dtype, channel=c+1)
        
        elif image.ndim == 4:
            # 4D image: (height, width, channels, Z-dimension)
            for z in range(image.shape[3]):
                for c in range(image.shape[2]):
                    save_grayscale_images(image[..., c, z], base_name, folder, dtype, channel=c+1, z=z+1)
        
        elif image.ndim == 5:
            # 5D image: (height, width, channels, Z-dimension, Time)
            for t in range(image.shape[4]):
                for z in range(image.shape[3]):
                    for c in range(image.shape[2]):
                        save_grayscale_images(image[..., c, z, t], base_name, folder, dtype, channel=c+1, z=z+1, t=t+1)

    # Function to load images in various formats
    def load_image(file_path):
        """Loads image from various formats and returns it as a numpy array along with its dtype."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.tif', '.tiff']:
            image = tifffile.imread(file_path)
            return image, image.dtype
        
        elif ext in ['.png', '.jpg', '.jpeg']:
            image = Image.open(file_path)
            return np.array(image), image.mode
        
        elif ext == '.czi':
            with czifile.CziFile(file_path) as czi:
                image = czi.asarray()
                return image, image.dtype
        
        elif ext == '.nd2':
            with ND2Reader(file_path) as nd2:
                image = np.array(nd2)
                return image, image.dtype
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # Function to check if an image is grayscale and save it as a TIFF if it isn't already
    def convert_grayscale_to_tiff(image, filename, folder, dtype):
        """Convert grayscale images that are not in TIFF format to TIFF, preserving bit depth."""
        base_name = os.path.splitext(filename)[0]
        output_filename = os.path.join(folder, f"{base_name}.tif")
        tifffile.imwrite(output_filename, image.astype(dtype))
        print(f"Converted grayscale image {filename} to TIFF with bit depth {dtype}.")

    # Supported formats
    supported_formats = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.czi', '.nd2']
    
    # Loop through all files in the folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        ext = os.path.splitext(file_path)[1].lower()

        if ext in supported_formats:
            print(f"Processing {filename}")
            try:
                # Load the image and its dtype
                image, dtype = load_image(file_path)
                
                # If the image is grayscale (2D), convert it to TIFF if it's not already in TIFF format
                if image.ndim == 2:
                    if ext not in ['.tif', '.tiff']:
                        convert_grayscale_to_tiff(image, filename, folder, dtype)
                    else:
                        print(f"Image {filename} is already grayscale and in TIFF format, skipping.")
                    continue
                
                # Otherwise, split channels and save images
                base_name = os.path.splitext(filename)[0]
                split_channels(image, folder, base_name, dtype)
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def _load_images_and_labels(image_files, label_files, invert=False):
    
    from .utils import invert_image
    
    images = []
    labels = []

    image_names = sorted([os.path.basename(f) for f in image_files]) if image_files else []
    label_names = sorted([os.path.basename(f) for f in label_files]) if label_files else []

    if image_files and label_files:
        for img_file, lbl_file in zip(image_files, label_files):
            image = cellpose.io.imread(img_file)
            if image is None:
                print(f"WARNING: Could not load image: {img_file}")
                continue
            if invert:
                image = invert_image(image)
            if image.max() > 1:
                image = image / image.max()

            label = cellpose.io.imread(lbl_file)
            if label is None:
                print(f"WARNING: Could not load label: {lbl_file}")
                continue

            images.append(image)
            labels.append(label)

    elif image_files:
        for img_file in image_files:
            image = cellpose.io.imread(img_file)
            if image is None:
                print(f"WARNING: Could not load image: {img_file}")
                continue
            if invert:
                image = invert_image(image)
            if image.max() > 1:
                image = image / image.max()
            images.append(image)

    elif label_files:
        for lbl_file in label_files:
            label = cellpose.io.imread(lbl_file)
            if label is None:
                print(f"WARNING: Could not load label: {lbl_file}")
                continue
            labels.append(label)

    image_dir = os.path.dirname(image_files[0]) if image_files else None
    label_dir = os.path.dirname(label_files[0]) if label_files else None

    print(f'Loaded {len(images)} images and {len(labels)} labels from {image_dir} and {label_dir}')
    if images and labels:
        print(f'image shape: {images[0].shape}, image type: {images[0].dtype}; '
              f'label shape: {labels[0].shape}, label type: {labels[0].dtype}')

    return images, labels, image_names, label_names

def _load_normalized_images_and_labels(image_files, label_files, channels=None, percentiles=None,  
                                       invert=False, visualize=False, remove_background=False, 
                                       background=0, Signal_to_noise=10, target_height=None, target_width=None):
    
    from .plot import normalize_and_visualize, plot_resize
    from .utils import invert_image, apply_mask
    from skimage.transform import resize as resizescikit

    # Ensure percentiles are valid
    if isinstance(percentiles, list) and len(percentiles) == 2:
        try:
            percentiles = [int(percentiles[0]), int(percentiles[1])]
        except ValueError:
            percentiles = None
    else:
        percentiles = None

    signal_thresholds = float(background) * float(Signal_to_noise)
    lower_percentile = 2

    images, labels, orig_dims = [], [], []
    num_channels = 4
    percentiles_1 = [[] for _ in range(num_channels)]
    percentiles_99 = [[] for _ in range(num_channels)]

    image_names = [os.path.basename(f) for f in image_files]
    image_dir = os.path.dirname(image_files[0])

    if label_files is not None:
        label_names = [os.path.basename(f) for f in label_files]
        label_dir = os.path.dirname(label_files[0])
    else:
        label_names, label_dir = [], None

    # Load, normalize, and resize images
    for i, img_file in enumerate(image_files):
        image = cellpose.io.imread(img_file)
        orig_dims.append((image.shape[0], image.shape[1]))

        if invert:
            image = invert_image(image)

        # Select specific channels if needed
        if channels is not None and image.ndim == 3:
            image = image[..., channels]

        if remove_background:
            image = np.where(image < background, 0, image)

        if image.ndim < 3:
            image = np.expand_dims(image, axis=-1)

        # Calculate percentiles if not provided
        if percentiles is None:
            for c in range(image.shape[-1]):
                p1 = np.percentile(image[..., c], lower_percentile)
                percentiles_1[c].append(p1)

                # Ensure `signal_thresholds` and `p` are floats for comparison
                for percentile in [98, 99, 99.9, 99.99, 99.999]:
                    p = np.percentile(image[..., c], percentile)
                    if float(p) > signal_thresholds:
                        percentiles_99[c].append(p)
                        break

        # Resize image if required
        if target_height and target_width:
            image_shape = (target_height, target_width) if image.ndim == 2 else (target_height, target_width, image.shape[-1])
            image = resizescikit(image, image_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)

        images.append(image)

    # Calculate average percentiles if needed
    if percentiles is None:
        avg_p1 = [np.mean(p) for p in percentiles_1]
        avg_p99 = [np.mean(p) if p else avg_p1[i] for i, p in enumerate(percentiles_99)]

        print(f'Average 1st percentiles: {avg_p1}, Average 99th percentiles: {avg_p99}')

        normalized_images = [
            np.stack([rescale_intensity(img[..., c], in_range=(avg_p1[c], avg_p99[c]), out_range=(0, 1))
                      for c in range(img.shape[-1])], axis=-1) for img in images
        ]

    else:
        normalized_images = [
            np.stack([rescale_intensity(img[..., c], 
                                        in_range=(np.percentile(img[..., c], percentiles[0]),
                                                  np.percentile(img[..., c], percentiles[1])), 
                                        out_range=(0, 1)) for c in range(img.shape[-1])], axis=-1) 
            for img in images
        ]

    # Load and resize labels if provided
    if label_files is not None:
        labels = [resizescikit(cellpose.io.imread(lbl_file), 
                               (target_height, target_width) if target_height and target_width else orig_dims[i], 
                               order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                  for i, lbl_file in enumerate(label_files)]

    print(f'Loaded and normalized {len(normalized_images)} images and {len(labels)} labels from {image_dir} and {label_dir}')

    if visualize and images and labels:
        plot_resize(images, normalized_images, labels, labels)

    return normalized_images, labels, image_names, label_names, orig_dims

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

    def __init__(self, train_loaders):
        self.train_loaders = train_loaders
        self.loader_iters = [iter(loader) for loader in train_loaders]

    def __iter__(self):
        return self

    def __next__(self):
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

    def __init__(self, datasets, shuffle=True):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)
        self.shuffle = shuffle
        if shuffle:
            self.indices = list(range(self.total_length))
            random.shuffle(self.indices)
        else:
            self.indices = None
    def __getitem__(self, index):
        if self.shuffle:
            index = self.indices[index]
        for dataset, length in zip(self.datasets, self.lengths):
            if index < length:
                return dataset[index]
            index -= length
    def __len__(self):
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

class spacrDataset(Dataset):
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
            # Use multiprocessing to load images in parallel
            with Pool(processes=cpu_count()) as pool:
                self.images = pool.map(self.load_image, self.filenames)
        else:
            self.images = None

    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)  # Handle image orientation
        return img

    def __len__(self):
        return len(self.filenames)

    def shuffle_dataset(self):
        combined = list(zip(self.filenames, self.labels))
        random.shuffle(combined)
        self.filenames, self.labels = zip(*combined)

    def get_plate(self, filepath):
        filename = os.path.basename(filepath)
        return filename.split('_')[0]

    def __getitem__(self, index):
        if self.pin_memory:
            img = self.images[index]
        else:
            img = self.load_image(self.filenames[index])
        label = self.labels[index]
        filename = self.filenames[index]
        if self.transform:
            img = self.transform(img)
        return img, label, filename
    
class spacrDataLoader(DataLoader):
    def __init__(self, *args, preload_batches=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.preload_batches = preload_batches
        self.batch_queue = Queue(maxsize=preload_batches)
        self.process = None
        self.current_batch_index = 0
        self._stop_event = False
        self.pin_memory = kwargs.get('pin_memory', False)
        atexit.register(self.cleanup)

    def _preload_next_batches(self):
        try:
            for _ in range(self.preload_batches):
                if self._stop_event:
                    break
                batch = next(self._iterator)
                if self.pin_memory:
                    batch = self._pin_memory_batch(batch)
                self.batch_queue.put(batch)
        except StopIteration:
            pass

    def _start_preloading(self):
        if self.process is None or not self.process.is_alive():
            self._iterator = iter(super().__iter__())
            if not self.pin_memory:
                self.process = Process(target=self._preload_next_batches)
                self.process.start()
            else:
                self._preload_next_batches()  # Directly load if pin_memory is True

    def _pin_memory_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.pin_memory() if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.pin_memory()
        else:
            return batch

    def __iter__(self):
        self._start_preloading()
        return self

    def __next__(self):
        if self.process and not self.process.is_alive() and self.batch_queue.empty():
            raise StopIteration

        try:
            if self.pin_memory:
                next_batch = self.batch_queue.get(timeout=60)
            else:
                next_batch = self.batch_queue.get(timeout=60)
            self.current_batch_index += 1

            # Start preloading the next batches
            if self.batch_queue.qsize() < self.preload_batches:
                self._start_preloading()

            return next_batch
        except queue.Empty:
            raise StopIteration

    def cleanup(self):
        self._stop_event = True
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def __del__(self):
        self.cleanup()

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
    def __init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform

        # Open the tar file just to build the list of members
        with tarfile.open(self.tar_path, 'r') as f:
            self.members = [m for m in f.getmembers() if m.isfile()]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        with tarfile.open(self.tar_path, 'r') as f:
            m = self.members[idx]
            img_file = f.extractfile(m)
            img = Image.open(BytesIO(img_file.read())).convert("RGB")
            
        if self.transform:
            img = self.transform(img)
        
        return img, m.name
    
def load_images_from_paths(images_by_key):
    images_dict = {}

    for key, paths in images_by_key.items():
        images_dict[key] = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    images_dict[key].append(np.array(img))
            except Exception as e:
                print(f"Error loading image from {path}: {e}")
    
    return images_dict

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
    
    if isinstance(img_format, str):
        img_format = [img_format]
    
    from .utils import _extract_filename_metadata, print_progress
    
    regular_expression = re.compile(regex)
    stack_path = os.path.join(src, 'stack')
    files_processed = 0
    if not os.path.exists(stack_path) or (os.path.isdir(stack_path) and len(os.listdir(stack_path)) == 0):
        all_filenames = [filename for filename in os.listdir(src) if any(filename.endswith(ext) for ext in img_format)]
        print(f'All files: {len(all_filenames)} in {src}')
        time_ls = []
        image_paths_by_key = _extract_filename_metadata(all_filenames, src, regular_expression, metadata_type, pick_slice, skip_mode)
        # Convert dictionary keys to a list for batching
        batching_keys = list(image_paths_by_key.keys())
        print(f'All unique FOV: {len(image_paths_by_key)} in {src}')
        for idx in range(0, len(image_paths_by_key), batch_size):
            start = time.time()

            # Select batch keys and create a subset of the dictionary for this batch
            batch_keys = batching_keys[idx:idx+batch_size]
            batch_images_by_key = {key: image_paths_by_key[key] for key in batch_keys}
            images_by_key = load_images_from_paths(batch_images_by_key)

            if pick_slice:
                for i, key in enumerate(images_by_key):
                    plate, well, field, channel, mode = key
                    max_intensity_slice = max(images_by_key[key], key=lambda x: np.percentile(x, 90))
                    mip_image = Image.fromarray(max_intensity_slice)
                    output_dir = os.path.join(src, channel)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{plate}_{well}_{field}.tif'
                    output_path = os.path.join(output_dir, output_filename)
                    files_processed += 1
                    stop = time.time()
                    duration = stop - start
                    time_ls.append(duration)
                    files_to_process = len(all_filenames)
                    print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=batch_size, operation_type='Preprocessing filenames')

                    if not os.path.exists(output_path):
                        mip_image.save(output_path)
                    else:
                        print(f'WARNING: A file with the same name already exists at location {output_filename}')
            else:
                for i, (key, images) in enumerate(images_by_key.items()):
                    plate, well, field, channel = key[:4]
                    output_dir = os.path.join(src, channel)
                    mip = np.max(np.stack(images), axis=0)
                    mip_image = Image.fromarray(mip)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{plate}_{well}_{field}.tif'
                    output_path = os.path.join(output_dir, output_filename)
                    files_processed += 1
                    stop = time.time()
                    duration = stop - start
                    time_ls.append(duration)
                    files_to_process = len(all_filenames)
                    print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=batch_size, operation_type='Preprocessing filenames')

                    if not os.path.exists(output_path):
                        mip_image.save(output_path)
                    else:
                        print(f'WARNING: A file with the same name already exists at location {output_filename}')

            images_by_key.clear()

        # Move original images to a new directory
        newpath = os.path.join(src, 'orig')
        os.makedirs(newpath, exist_ok=True)
        for filename in os.listdir(src):
            #print(f"{filename}: {os.path.splitext(filename)[1]}")
            if os.path.splitext(filename)[1] in img_format:
                move = os.path.join(newpath, filename)
                if os.path.exists(move):
                    print(f'WARNING: A file with the same name already exists at location {move}')
                else:
                    shutil.move(os.path.join(src, filename), move)
    files_processed = 0
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

def _merge_channels(src, plot=False):
    """
    Merge the channels in the given source directory and save the merged files in a 'stack' directory without using multiprocessing.
    """

    from .plot import plot_arrays
    from .utils import print_progress
    
    stack_dir = os.path.join(src, 'stack')
    #allowed_names = ['01', '02', '03', '04', '00', '1', '2', '3', '4', '0']
    
    string_list = [str(i) for i in range(101)]+[f"{i:02d}" for i in range(10)]
    allowed_names = sorted(string_list, key=lambda x: int(x))
    
    # List directories that match the allowed names
    chan_dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d in allowed_names]
    chan_dirs.sort()
    
    num_matching_folders = len(chan_dirs)

    print(f'List of folders in src: {chan_dirs}. Single channel folders.')
    
    # Assuming chan_dirs[0] is not empty and exists, adjust according to your logic
    first_dir_path = os.path.join(src, chan_dirs[0])
    dir_files = os.listdir(first_dir_path)

    # Create the 'stack' directory if it doesn't exist
    if not os.path.exists(stack_dir):
        os.makedirs(stack_dir, exist_ok=True)
    print(f'Generated folder with merged arrays: {stack_dir}')

    if _is_dir_empty(stack_dir):
        time_ls = []
        files_to_process = len(dir_files)
        for i, file_name in enumerate(dir_files):
            start_time = time.time()
            full_file_path = os.path.join(first_dir_path, file_name)
            if os.path.isfile(full_file_path):
                _merge_file([os.path.join(src, d) for d in chan_dirs], stack_dir, file_name)
            stop_time = time.time()
            duration = stop_time - start_time
            time_ls.append(duration)
            files_processed = i + 1
            print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type='Merging channels into npy stacks')

    if plot:
        plot_arrays(os.path.join(src, 'stack'))

    return num_matching_folders

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

    #print('========== generating MIPs ==========')
    # Iterate over each file in the specified directory (src).
    for filename in os.listdir(src):
        # Check if the current file is a NumPy array file (with .npy extension).
        if filename.endswith('.npy'):
            # Load the array from the file.
            array = np.load(os.path.join(src, filename))
            # Normalize the array 
            #array = normalize_to_dtype(array, q1=0, q2=99, percentiles=None)

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
    from .utils import print_progress
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
    time_ls = []
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

                stop = time.time()
                duration = stop - start
                time_ls.append(duration)
                files_processed = i+1
                files_to_process = time_stack_path_lists
                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=batch_size, operation_type="Concatinating")
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
        filenames_batch = []
        for i, path in enumerate(paths):
            start = time.time()
            array = np.load(path)
            array = np.take(array, channels, axis=2)
            stack_ls.append(array)
            filenames_batch.append(os.path.basename(path))  # store the filename
            stop = time.time()
            duration = stop - start
            time_ls.append(duration)
            files_processed = i+1
            files_to_process = nr_files
            print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=batch_size, operation_type="Concatinating")
            if (i+1) % batch_size == 0 or i+1 == nr_files:
                unique_shapes = {arr.shape[:-1] for arr in stack_ls}
                if len(unique_shapes) > 1:
                    max_dims = np.max(np.array(list(unique_shapes)), axis=0)
                    print(f'Warning: arrays with multiple shapes found in batch {i+1}. Padding arrays to max X,Y dimentions {max_dims}')
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

def _normalize_img_batch(stack, channels, save_dtype, settings):
    
    from .utils import print_progress
    """
    Normalize the stack of images.

    Args:
        stack (numpy.ndarray): The stack of images to normalize.
        lower_percentile (int): Lower percentile value for normalization.
        save_dtype (numpy.dtype): Data type for saving the normalized stack.
        settings (dict): keword arguments

    Returns:
        numpy.ndarray: The normalized stack.
    """

    normalized_stack = np.zeros_like(stack, dtype=np.float32)
    
    #for channel in range(stack.shape[-1]):
    time_ls = []
    for i, channel in enumerate(channels):
        start = time.time()
        if channel == settings['nucleus_channel']:
            background = settings['nucleus_background']
            signal_threshold = settings['nucleus_Signal_to_noise']*settings['nucleus_background']
            remove_background = settings['remove_background_nucleus']

        if channel == settings['cell_channel']:
            background = settings['cell_background']
            signal_threshold = settings['cell_Signal_to_noise']*settings['cell_background']
            remove_background = settings['remove_background_cell']

        if channel == settings['pathogen_channel']:
            background = settings['pathogen_background']
            signal_threshold = settings['pathogen_Signal_to_noise']*settings['pathogen_background']
            remove_background = settings['remove_background_pathogen']

        single_channel = stack[:, :, :, channel]

        print(f'Processing channel {channel}: background={background}, signal_threshold={signal_threshold}, remove_background={remove_background}')

        # Step 3: Remove background if required
        if remove_background:
            single_channel[single_channel < background] = 0

        # Step 4: Calculate global lower percentile for the channel
        non_zero_single_channel = single_channel[single_channel != 0]
        global_lower = np.percentile(non_zero_single_channel, settings['lower_percentile'])

        # Step 5: Calculate global upper percentile for the channel
        global_upper = None
        for upper_p in np.linspace(98, 99.5, num=16):
            upper_value = np.percentile(non_zero_single_channel, upper_p)
            if upper_value >= signal_threshold:
                global_upper = upper_value
                break

        if global_upper is None:
            global_upper = np.percentile(non_zero_single_channel, 99.5)  # Fallback in case no upper percentile met the threshold

        print(f'Channel {channel}: global_lower={global_lower}, global_upper={global_upper}, Signal-to-noise={global_upper / global_lower}')

        # Step 6: Normalize each array from global_lower to global_upper between 0 and 1
        for array_index in range(single_channel.shape[0]):
            arr_2d = single_channel[array_index, :, :]
            arr_2d_normalized = exposure.rescale_intensity(arr_2d, in_range=(global_lower, global_upper), out_range=(0, 1))
            normalized_stack[array_index, :, :, channel] = arr_2d_normalized

        stop = time.time()
        duration = stop - start
        time_ls.append(duration)
        files_processed = i+1
        files_to_process = len(channels)
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f"Normalizing")

    return normalized_stack.astype(save_dtype)

def concatenate_and_normalize(src, channels, save_dtype=np.float32, settings={}):
    from .utils import print_progress

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
    time_ls = []
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)

    if settings['timelapse']:
        try:
            time_stack_path_lists = _generate_time_lists(os.listdir(src))
            for i, time_stack_list in enumerate(time_stack_path_lists):
                start = time.time()
                stack_region = []
                filenames_region = []
                for idx, file in enumerate(time_stack_list):
                    path = os.path.join(src, file)
                    if idx == 0:
                        parts = file.split('_')
                        name = parts[0] + '_' + parts[1] + '_' + parts[2]
                    array = np.load(path)
                    stack_region.append(array)
                    filenames_region.append(os.path.basename(path))
                stop = time.time()
                duration = stop - start
                time_ls.append(duration)
                files_processed = i+1
                files_to_process = len(time_stack_path_lists)
                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Concatinating")
                stack = np.stack(stack_region)

                normalized_stack = _normalize_img_batch(stack=stack,
                                                        channels=channels, 
                                                        save_dtype=save_dtype,
                                                        settings=settings)
                
                normalized_stack = normalized_stack[..., channels]
                
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
        if settings['randomize']:
            random.shuffle(paths)
        nr_files = len(paths)
        batch_index = 0
        stack_ls = []
        filenames_batch = []
        time_ls = []
        files_processed = 0
        for i, path in enumerate(paths):
            start = time.time()
            array = np.load(path)
            stack_ls.append(array)
            filenames_batch.append(os.path.basename(path))
            stop = time.time()
            duration = stop - start
            time_ls.append(duration)
            files_processed += 1
            files_to_process = nr_files
            print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Concatinating")

            if (i + 1) % settings['batch_size'] == 0 or i + 1 == nr_files:
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
                
                normalized_stack = _normalize_img_batch(stack=stack,
                                                        channels=channels,
                                                        save_dtype=save_dtype,
                                                        settings=settings)
                
                normalized_stack = normalized_stack[..., channels]

                save_loc = os.path.join(output_fldr, f'stack_{batch_index}_norm.npz')
                np.savez(save_loc, data=normalized_stack, filenames=filenames_batch)
                batch_index += 1
                del stack, normalized_stack
                stack_ls = []
                filenames_batch = []
                padded_stack_ls = []

    print(f'All files concatenated and normalized. Saved to: {output_fldr}')
    return output_fldr

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
    # for ch in settings['channels']:
    for ch in [settings['nucleus_channel'], settings['cell_channel'], settings['pathogen_channel']]:
        if not ch is None:
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
            time_ls = []
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
                print(f'channels:{chan_index}/{stack.shape[-1] - 1}, arrays:{array_index + 1}/{single_channel.shape[0]}, Signal:{upper:.1f}, noise:{lower:.1f}, Signal-to-noise:{average_stnr:.1f}, Time/channel:{average_time:.2f}sec')

                #stop = time.time()
                #duration = stop - start
                #time_ls.append(duration)
                #files_processed = file_index + 1
                #files_to_process = len(paths)
                #print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Normalizing")
                
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
            time_ls = []
            for array_index in range(single_channel.shape[0]):
                start = time.time()
                arr_2d = single_channel[array_index]
                # Calculate the 1% and 98% percentiles for this specific image
                q_low = np.percentile(arr_2d[arr_2d != 0], lower_percentile)
                q_high = np.percentile(arr_2d[arr_2d != 0], 98)

                # Rescale intensity based on the calculated percentiles to fill the dtype range
                arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(q_low, q_high), out_range='dtype')
                normalized_stack[array_index, :, :, chan_index] = arr_2d_rescaled

                print(f'channels:{chan_index+1}/{stack.shape[-1]}, arrays:{array_index+1}/{single_channel.shape[0]}', end='\r')

                #stop = time.time()
                #duration = stop - start
                #time_ls.append(duration)
                #files_processed = file_index+1
                #files_to_process = len(paths)
                #print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Normalizing")

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
    
    from .plot import plot_arrays
    from .utils import _run_test_mode, _get_regex
    from .settings import set_default_settings_preprocess_img_data
    
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
        
        if os.path.exists(os.path.join(src,'stack')):
            print('Found existing stack folder.')
        if os.path.exists(os.path.join(src,'channel_stack')):
            print('Found existing channel_stack folder.')
        if os.path.exists(os.path.join(src,'norm_channel_stack')):
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
            print(f"Error deleting test directory: {e}")
            print(f"Delete manually before running test mode")
            pass

        src = _run_test_mode(settings['src'], regex, timelapse, test_images, random_test)
        settings['src'] = src
    
    stack_path = os.path.join(src, 'stack')
    if img_format == None:
        if not os.path.exists(stack_path):
            _merge_channels(src, plot=False)   
   
    if not os.path.exists(stack_path):
        try:
            if not img_format == None:
                if timelapse:
                    _move_to_chan_folder(src, regex, timelapse, metadata_type)
                else:
                    img_format = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.nd2', '.czi', '.lif']
                    _rename_and_organize_image_files(src, regex, batch_size, pick_slice, skip_mode, metadata_type, img_format)
                    
                    #Make sure no batches will be of only one image
                    all_imgs = len(stack_path)
                    full_batches = all_imgs // batch_size
                    last_batch_size = all_imgs % batch_size
                    
                    # Check if the last batch is of size 1
                    if last_batch_size == 1:
                        # If there's only one batch and its size is 1, it's also an issue
                        if full_batches == 0:
                            raise ValueError("Only one batch of size 1 detected. Adjust the batch size.")
                        # If the last batch is of size 1, merge it with the second last batch
                        elif full_batches > 0:
                            print(f"all images: {all_imgs},  full batch: {full_batches}, last batch: {last_batch_size}")
                            raise ValueError("Last batch of size 1 detected. Adjust the batch size.")
        
                nr_channel_folders = _merge_channels(src, plot=False)
                
                if len(settings['channels']) != nr_channel_folders:
                    print(f"Number of channels does not match number of channel folders. channels: {settings['channels']} channel folders: {nr_channel_folders}")
                    new_channels = list(range(nr_channel_folders))
                    print(f"Changing channels from {settings['channels']} to {new_channels}")
                    settings['channels'] = new_channels

                if timelapse:
                    _create_movies_from_npy_per_channel(stack_path, fps=2)

                if plot:
                    print(f'plotting {nr} images from {src}/stack')
                    plot_arrays(stack_path, figuresize, cmap, nr=nr, normalize=normalize)

                if all_to_mip:
                    _mip_all(stack_path)
                    if plot:
                        print(f'plotting {nr} images from {src}/stack')
                        plot_arrays(stack_path, figuresize, cmap, nr=nr, normalize=normalize)
        except Exception as e:
            print(f"Error: {e}")
    
    concatenate_and_normalize(src=stack_path,
                              channels=mask_channels,
                              save_dtype=np.float32,
                              settings=settings)

    #if plot:
    #    _plot_4D_arrays(src+'/norm_channel_stack', nr_npz=1, nr=nr)

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
    from .utils import print_progress
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

    files_processed = i
    files_to_process = all_folders
    print_progress(files_processed, files_to_process, n_jobs=1, time_ls=None, batch_size=None, operation_type="Saving Figures")
    print(f'Saved single cell figure: {os.path.basename(save_location)}')
    plt.close(fig)
    del fig
    gc.collect()
    
def _read_and_join_tables(db_path, table_names=['cell', 'cytoplasm', 'nucleus', 'pathogen', 'png_list']):
    """
    Reads and joins tables from a SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.
        table_names (list, optional): The names of the tables to read and join. Defaults to ['cell', 'cytoplasm', 'nucleus', 'pathogen', 'png_list'].

    Returns:
        pandas.DataFrame: The joined DataFrame containing the data from the specified tables, or None if an error occurs.
    """
    from .utils import rename_columns_in_db
    rename_columns_in_db(db_path)
    
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
        png_list_df = dataframes['png_list'][['cell_id', 'png_path', 'plateID', 'rowID', 'columnID', 'fieldID']].copy()
        png_list_df['cell_id'] = png_list_df['cell_id'].str[1:].astype(int)
        png_list_df.rename(columns={'cell_id': 'object_label'}, inplace=True)
        if 'cell' in dataframes:
            join_cols = ['object_label', 'plateID', 'rowID', 'columnID','fieldID']
            dataframes['cell'] = pd.merge(dataframes['cell'], png_list_df, on=join_cols, how='left')
        else:
            print("Cell table not found in database tables.")
            return png_list_df
    for entity in ['nucleus', 'pathogen']:
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
    src = os.path.dirname(settings['src'])
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

    from .utils import print_progress

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
    time_ls = []
    # Iterate through each file in the reference folder
    for idx, filename in enumerate(os.listdir(reference_folder)):
        start = time.time()
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
                    #array = np.load(array_path)
                    array = np.load(array_path, allow_pickle=True)
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
        
        stop = time.time()
        duration = stop - start
        time_ls.append(duration)
        files_processed = idx+1
        files_to_process = all_imgs
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Merging Arrays")

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
    from .utils import rename_columns_in_db, correct_metadata
    
    rename_columns_in_db(db_loc)
    conn = sqlite3.connect(db_loc)
    dfs = []
    
    for table in tables:
        query = f'SELECT * FROM {table}'
        df = pd.read_sql_query(query, conn)
        df = correct_metadata(df)
        dfs.append(df)
        
    conn.close()
    return dfs
    
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

def read_plot_model_stats(train_file_path, val_file_path ,save=False):
    
    def _plot_and_save(train_df, val_df, column='accuracy', save=False, path=None, dpi=600):
        
        pdf_path = os.path.join(path, f'{column}.pdf')

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

        # Plotting
        sns.lineplot(ax=axes[0], x='epoch', y=column, data=train_df, marker='o', color='red')
        sns.lineplot(ax=axes[1], x='epoch', y=column, data=val_df, marker='o', color='blue')

        # Set titles and labels
        axes[0].set_title(f'Train {column} vs. Epoch', fontsize=20)
        axes[0].set_xlabel('Epoch', fontsize=16)
        axes[0].set_ylabel(column, fontsize=16)
        axes[0].tick_params(axis='both', which='major', labelsize=12)

        axes[1].set_title(f'Validation {column} vs. Epoch', fontsize=20)
        axes[1].set_xlabel('Epoch', fontsize=16)
        axes[1].tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        if save:
            plt.savefig(pdf_path, format='pdf', dpi=dpi)
        else:
            plt.show()

    # Read the CSVs into DataFrames
    train_df = pd.read_csv(train_file_path, index_col=0)
    val_df = pd.read_csv(val_file_path, index_col=0)

    # Get the folder path for saving plots
    fldr_1 = os.path.dirname(train_file_path)
    
    if save:
        # Setting the style
        sns.set(style="whitegrid")
    
    # Plot and save the results
    _plot_and_save(train_df, val_df, column='accuracy', save=save, path=fldr_1)
    _plot_and_save(train_df, val_df, column='neg_accuracy', save=save, path=fldr_1)
    _plot_and_save(train_df, val_df, column='pos_accuracy', save=save, path=fldr_1)
    _plot_and_save(train_df, val_df, column='loss', save=save, path=fldr_1)
    _plot_and_save(train_df, val_df, column='prauc', save=save, path=fldr_1)
    _plot_and_save(train_df, val_df, column='optimal_threshold', save=save, path=fldr_1)
    
def _save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94], channels=['r','g','b']):
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
        channels (list, optional): List of channels used. Defaults to ['r', 'g', 'b'].
    """

    channels_str = ''.join(channels)
    
    def save_model_at_threshold(threshold, epoch, suffix=""):
        percentile = str(threshold * 100)
        print(f'Found: {percentile}% accurate model')
        model_path = f'{dst}/{model_type}_epoch_{str(epoch)}{suffix}_acc_{percentile}_channels_{channels_str}.pth'
        torch.save(model, model_path)
        return model_path

    if epoch % 100 == 0 or epoch == epochs:
        model_path = f'{dst}/{model_type}_epoch_{str(epoch)}_channels_{channels_str}.pth'
        torch.save(model, model_path)
        return model_path

    for threshold in intermedeate_save:
        if results_df['neg_accuracy'] >= threshold and results_df['pos_accuracy'] >= threshold:
            print(f"Nc class accuracy: {results_df['neg_accuracy']} Pc class Accuracy: {results_df['pos_accuracy']}")
            model_path = save_model_at_threshold(threshold, epoch)
            break
        else:
            model_path = None
    
    return model_path

def _save_progress(dst, train_df, validation_df):
    """
    Save the progress of the classification model.

    Parameters:
    dst (str): The destination directory to save the progress.
    train_df (pandas.DataFrame): The DataFrame containing training stats.
    validation_df (pandas.DataFrame): The DataFrame containing validation stats (if available).

    Returns:
    None
    """

    def _save_df_to_csv(file_path, df):
        """
        Save the given DataFrame to the specified CSV file, either creating a new file or appending to an existing one.

        Parameters:
        file_path (str): The file path where the CSV will be saved.
        df (pandas.DataFrame): The DataFrame to save.
        """
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                df.to_csv(f, index=True, header=True)
                f.flush()  # Ensure data is written to the file system
        else:
            with open(file_path, 'a') as f:
                df.to_csv(f, index=True, header=False)
                f.flush()
                
    # Save accuracy, loss, PRAUC
    os.makedirs(dst, exist_ok=True)
    results_path_train = os.path.join(dst, 'train.csv')
    results_path_validation = os.path.join(dst, 'validation.csv')

    # Save training data
    _save_df_to_csv(results_path_train, train_df)

    # Save validation data if available
    if validation_df is not None:
        _save_df_to_csv(results_path_validation, validation_df)

        # Call read_plot_model_stats after ensuring the files are saved
        read_plot_model_stats(results_path_train, results_path_validation, save=True)

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
    
    from .utils import rename_columns_in_db, correct_metadata
    
    rename_columns_in_db(db_loc)
    conn = sqlite3.connect(db_loc) # Create a connection to the database
    dfs = []
    for table in tables:
        query = f'SELECT * FROM {table}' # Write a SQL query to get the data from the database
        df = pd.read_sql_query(query, conn) # Use the read_sql_query function to get the data and save it as a DataFrame
        df = correct_metadata(df)
        dfs.append(df)
    conn.close() # Close the connection
    return dfs

def _read_and_merge_data(locs, tables, verbose=False, nuclei_limit=10, pathogen_limit=10, change_plate=False):

    from .utils import _split_data

    # Initialize an empty dictionary to store DataFrames by table name
    data_dict = {table: [] for table in tables}

    # Extract plate DataFrames
    for idx, loc in enumerate(locs):
        db_dfs = _read_db(loc, tables)
        if change_plate:
            db_dfs['plateID'] = f'plate{idx+1}'
            db_dfs['prc'] = db_dfs['plateID'].astype(str) + '_' + db_dfs['rowID'].astype(str) + '_' + db_dfs['columnID'].astype(str)
        for table, df in zip(tables, db_dfs):
            data_dict[table].append(df)

    # Concatenate rows across locations for each table
    for table, dfs in data_dict.items():
        if dfs:
            data_dict[table] = pd.concat(dfs, axis=0)
            
        if verbose:
            print(f"{table}: {len(data_dict[table])}")

    # Initialize merged DataFrame with 'cells' if available
    merged_df = pd.DataFrame()

    # Process each table
    if 'cell' in data_dict:
        cells = data_dict['cell'].copy()
        cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cells = cells.assign(prcfo=lambda x: x['prcf'] + '_' + x['object_label'])
        cells_g_df, metadata = _split_data(cells, 'prcfo', 'object_label')
        merged_df = cells_g_df.copy()
        if verbose:
            print(f'cells: {len(cells)}, cells grouped: {len(cells_g_df)}')

    if 'cytoplasm' in data_dict:
        cytoplasms = data_dict['cytoplasm'].copy()
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo=lambda x: x['prcf'] + '_' + x['object_label'])
        
        if not 'cell' in data_dict:
            merged_df, metadata = _split_data(cytoplasms, 'prcfo', 'object_label')
            
            if verbose:
                print(f'nucleus: {len(cytoplasms)}, cytoplasms grouped: {len(merged_df)}')
            
        else:
            cytoplasms_g_df, _ = _split_data(cytoplasms, 'prcfo', 'object_label')
            merged_df = merged_df.merge(cytoplasms_g_df, left_index=True, right_index=True)
            
            if verbose:
                print(f'cytoplasms: {len(cytoplasms)}, cytoplasms grouped: {len(cytoplasms_g_df)}')

    if 'nucleus' in data_dict:
        nucleus = data_dict['nucleus'].copy()
        nucleus = nucleus.dropna(subset=['cell_id'])
        nucleus = nucleus.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nucleus = nucleus.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nucleus = nucleus.assign(prcfo=lambda x: x['prcf'] + '_' + x['cell_id'])
        nucleus['nucleus_prcfo_count'] = nucleus.groupby('prcfo')['prcfo'].transform('count')
        if not nuclei_limit:
            nucleus = nucleus[nucleus['nucleus_prcfo_count'] == 1]
            
        if all(key not in data_dict for key in ['cell', 'cytoplasm']):
            merged_df, metadata = _split_data(nucleus, 'prcfo', 'cell_id')
            
            if verbose:
                print(f'nucleus: {len(nucleus)}, nucleus grouped: {len(merged_df)}')
            
        else:
            nucleus_g_df, _ = _split_data(nucleus, 'prcfo', 'cell_id')
            merged_df = merged_df.merge(nucleus_g_df, left_index=True, right_index=True)
            
            if verbose:
                print(f'nucleus: {len(nucleus)}, nucleus grouped: {len(nucleus_g_df)}')

    if 'pathogen' in data_dict:
        pathogens = data_dict['pathogen'].copy()
        pathogens = pathogens.dropna(subset=['cell_id'])
        pathogens = pathogens.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        pathogens = pathogens.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        pathogens = pathogens.assign(prcfo=lambda x: x['prcf'] + '_' + x['cell_id'])
        pathogens['pathogen_prcfo_count'] = pathogens.groupby('prcfo')['prcfo'].transform('count')

        if isinstance(pathogen_limit, bool) and not pathogen_limit:
            pathogens = pathogens[pathogens['pathogen_prcfo_count'] <= 1]
        elif isinstance(pathogen_limit, (float, int)):
            pathogens = pathogens[pathogens['pathogen_prcfo_count'] <= int(pathogen_limit)]

        if all(key not in data_dict for key in ['cell', 'cytoplasm', 'nucleus']):
            merged_df, metadata = _split_data(pathogens, 'prcfo', 'cell_id')
            
            if verbose:
                print(f'pathogens: {len(pathogens)}, pathogens grouped: {len(merged_df)}')
            
        else:
            pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
            merged_df = merged_df.merge(pathogens_g_df, left_index=True, right_index=True)
        
            if verbose:
                print(f'pathogens: {len(pathogens)}, pathogens grouped: {len(pathogens_g_df)}')
            
    if 'png_list' in data_dict:
        png_list = data_dict['png_list'].copy()
        png_list_g_df_numeric, png_list_g_df_non_numeric = _split_data(png_list, 'prcfo', 'cell_id')
        png_list_g_df_non_numeric.drop(columns=['plateID','rowID','columnID','fieldID','file_name','cell_id', 'prcf'], inplace=True)
        if verbose:
            print(f'png_list: {len(png_list)}, png_list grouped: {len(png_list_g_df_numeric)}')
            print(f"Added png_list columns: {png_list_g_df_numeric.columns}, {png_list_g_df_non_numeric.columns}")
        merged_df = merged_df.merge(png_list_g_df_numeric, left_index=True, right_index=True)
        merged_df = merged_df.merge(png_list_g_df_non_numeric, left_index=True, right_index=True)
    
    # Add prc (plate row column) and prcfo (plate row column field object) columns
    metadata = metadata.assign(prc=lambda x: x['plateID'] + '_' + x['rowID'] + '_' + x['columnID'])
    cells_well = metadata.groupby('prc')['object_label'].nunique().reset_index(name='cells_per_well')
    metadata = metadata.merge(cells_well, on='prc')
    metadata = metadata.assign(prcfo=lambda x: x['plateID'] + '_' + x['rowID'] + '_' + x['columnID'] + '_' + x['fieldID'] + '_' + x['object_label'])
    metadata.set_index('prcfo', inplace=True)
    
    # Merge metadata with final merged DataFrame
    #merged_df = metadata.merge(merged_df, left_index=True, right_index=True).dropna(axis=1)
    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    merged_df.drop(columns=['label_list_morphology', 'label_list_intensity'], errors='ignore', inplace=True)
    
    if verbose:
        print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')
    
    # Prepare object DataFrames for output
    obj_df_ls = [data_dict[table] for table in ['cell', 'cytoplasm', 'nucleus', 'pathogen'] if table in data_dict]
    
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
            print(f'Copied {idx+1}/{len(ls)} images to {_type} set')#, end='\r', flush=True)

def parse_gz_files(folder_path):
    """
    Parses the .fastq.gz files in the specified folder path and returns a dictionary
    containing the sample names and their corresponding file paths.

    Args:
        folder_path (str): The path to the folder containing the .fastq.gz files.

    Returns:
        dict: A dictionary where the keys are the sample names and the values are
        dictionaries containing the file paths for the 'R1' and 'R2' read directions.
    """
    files = os.listdir(folder_path)
    gz_files = [f for f in files if f.endswith('.fastq.gz')]

    samples_dict = {}
    for gz_file in gz_files:
        parts = gz_file.split('_')
        sample_name = parts[0]
        read_direction = parts[1]

        if sample_name not in samples_dict:
            samples_dict[sample_name] = {}

        if read_direction == "R1":
            samples_dict[sample_name]['R1'] = os.path.join(folder_path, gz_file)
        elif read_direction == "R2":
            samples_dict[sample_name]['R2'] = os.path.join(folder_path, gz_file)
    return samples_dict

def generate_dataset(settings={}):
    
    from .utils import initiate_counter, add_images_to_tar, save_settings, generate_path_list_from_db, correct_paths
    from .settings import set_generate_dataset_defaults

    settings = set_generate_dataset_defaults(settings)
    save_settings(settings, 'generate_dataset', show=True)

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]
    if isinstance(settings['src'], list):
        all_paths = []
        for i, src in enumerate(settings['src']):
            db_path = os.path.join(src, 'measurements', 'measurements.db')
            if i == 0:
                dst = os.path.join(src, 'datasets')
            paths = generate_path_list_from_db(db_path, file_metadata=settings['file_metadata'])
            correct_paths(paths, src)
            all_paths.extend(paths)
        if isinstance(settings['sample'], int):
            selected_paths = random.sample(all_paths, settings['sample'])
            print(f"Random selection of {len(selected_paths)} paths")
        elif isinstance(settings['sample'], list):
            sample = settings['sample'][i]
            selected_paths = random.sample(all_paths, settings['sample'])
            print(f"Random selection of {len(selected_paths)} paths")
        else:
            selected_paths = all_paths
            random.shuffle(selected_paths)
            print(f"All paths: {len(selected_paths)} paths")

    total_images = len(selected_paths)
    print(f"Found {total_images} images")

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

    temp_tar_files = [os.path.join(temp_dir, f"temp_{i}.tar") for i in range(num_procs)]

    print(f"Generating temporary tar files in {dst}")

    # Initialize shared counter and lock
    counter = Value('i', 0)
    lock = Lock()

    with Pool(processes=num_procs, initializer=initiate_counter, initargs=(counter, lock)) as pool:
        pool.starmap(add_images_to_tar, [(paths_chunks[i], temp_tar_files[i], total_images) for i in range(num_procs)])

    # Combine the temporary tar files into a final tar
    date_name = datetime.date.today().strftime('%y%m%d')
    if len(settings['src']) > 1:
        date_name = f"{date_name}_combined"
    #if not settings['file_metadata'] is None:
    #    tar_name = f"{date_name}_{settings['experiment']}_{settings['file_metadata']}.tar"
    #else:
    tar_name = f"{date_name}_{settings['experiment']}.tar"
    tar_name = os.path.join(dst, tar_name)
    if os.path.exists(tar_name):
        number = random.randint(1, 100)
        tar_name_2 = f"{date_name}_{settings['experiment']}_{settings['file_metadata']}_{number}.tar"
        print(f"Warning: {os.path.basename(tar_name)} exists, saving as {os.path.basename(tar_name_2)} ")
        tar_name = os.path.join(dst, tar_name_2)

    print(f"Merging temporary files")

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

    return tar_name

def generate_loaders(src, mode='train', image_size=224, batch_size=32, classes=['nc','pc'], n_jobs=None, validation_split=0.0, pin_memory=False, normalize=False, channels=[1, 2, 3], augment=False, verbose=False):
    
    """
    Generate data loaders for training and validation/test datasets.

    Parameters:
    - src (str): The source directory containing the data.
    - mode (str): The mode of operation. Options are 'train' or 'test'.
    - image_size (int): The size of the input images.
    - batch_size (int): The batch size for the data loaders.
    - classes (list): The list of classes to consider.
    - n_jobs (int): The number of worker threads for data loading.
    - validation_split (float): The fraction of data to use for validation.
    - pin_memory (bool): Whether to pin memory for faster data transfer.
    - normalize (bool): Whether to normalize the input images.
    - verbose (bool): Whether to print additional information and show images.
    - channels (list): The list of channels to retain. Options are [1, 2, 3] for all channels, [1, 2] for blue and green, etc.

    Returns:
    - train_loaders (list): List of data loaders for training datasets.
    - val_loaders (list): List of data loaders for validation datasets.
    """

    from .utils import SelectChannels, augment_dataset

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
        
    train_loaders = []
    val_loaders = []

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
        print('Loading Train and validation datasets')
    elif mode == 'test':
        data_dir = os.path.join(src, 'test')
        val_loaders = []
        validation_split = 0.0
        shuffle = True
        print('Loading test dataset')
    else:
        print(f'mode:{mode} is not valid, use mode = train or test')
        return
    
    class_1_path = os.path.join(data_dir, classes[0])
    class_2_path = os.path.join(data_dir, classes[1])
    if not os.path.exists(class_1_path) or not os.path.exists(class_2_path):
        print(f'One or more classes not found in {data_dir}')
        print (f'Possible class names are {os.listdir(data_dir)}')

    data = spacrDataset(data_dir, classes, transform=transform, shuffle=shuffle, pin_memory=pin_memory)
    num_workers = n_jobs if n_jobs is not None else 0
    
    if validation_split > 0:
        train_size = int((1 - validation_split) * len(data))
        val_size = len(data) - train_size
        if not augment:
            print(f'Train data:{train_size}, Validation data:{val_size}')
        train_dataset, val_dataset = random_split(data, [train_size, val_size])

        if augment:

            print(f'Data before augmentation: Train: {len(train_dataset)}, Validataion:{len(val_dataset)}')
            train_dataset = augment_dataset(train_dataset, is_grayscale=(len(channels) == 1))
            print(f'Data after augmentation: Train: {len(train_dataset)}')
            
        print(f'Generating Dataloader with {n_jobs} workers')
        train_loaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=pin_memory, persistent_workers=True)
        val_loaders = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=pin_memory, persistent_workers=True)
    else:
        train_loaders = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=pin_memory, persistent_workers=True)

    #dataset (Dataset) – dataset from which to load the data.
    #batch_size (int, optional) – how many samples per batch to load (default: 1).
    #shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    #sampler (Sampler or Iterable, optional) – defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified.
    #batch_sampler (Sampler or Iterable, optional) – like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
    #num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    #collate_fn (Callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
    #pin_memory (bool, optional) – If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.
    #drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
    #timeout (numeric, optional) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: 0)
    #worker_init_fn (Callable, optional) – If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)
    #multiprocessing_context (str or multiprocessing.context.BaseContext, optional) – If None, the default multiprocessing context of your operating system will be used. (default: None)
    #generator (torch.Generator, optional) – If not None, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers. (default: None)
    #prefetch_factor (int, optional, keyword-only arg) – Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. (default value depends on the set value for num_workers. If value of num_workers=0 default is None. Otherwise, if value of num_workers > 0 default is 2).
    #persistent_workers (bool, optional) – If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: False)
    #pin_memory_device (str, optional) – the device to pin_memory to if pin_memory is True.

    #images, labels, filenames = next(iter(train_loaders))
    #images = images.cpu()
    #label_strings = [str(label.item()) for label in labels]
    #train_fig = _imshow_gpu(images, label_strings, nrow=20, fontsize=12)
    #if verbose:
    #    plt.show()
    
    train_fig = None

    return train_loaders, val_loaders, train_fig

def generate_training_dataset(settings):
    
    # Function to filter png_list_df by prcfo present in df without merging
    def filter_png_list(db_path, settings, tables = ['cell', 'nucleus', 'pathogen', 'cytoplasm']):
        df, _ = _read_and_merge_data(locs=[db_path],
                                     tables=tables,
                                     verbose=False,
                                     nuclei_limit=settings['nuclei_limit'],
                                     pathogen_limit=settings['pathogen_limit'])
                
        [png_list_df] = _read_db(db_loc=db_path, tables=['png_list'])
        filtered_png_list_df = png_list_df[png_list_df['prcfo'].isin(df.index)]
        return filtered_png_list_df

    # Function to get the smallest class size based on the dataset mode
    def get_smallest_class_size(df, settings, dataset_mode):
        if dataset_mode == 'metadata':
            sizes = [len(df[df['condition'] == c]) for c in settings['class_metadata']]
            #sizes = [len(df[df['condition'].isin(class_list)]) for class_list in settings['class_metadata']]
            print(f'Class sizes: {sizes}')
        elif dataset_mode == 'annotation':
            sizes = [len(class_paths) for class_paths in df]
        size = min(sizes)
        print(f'Using the smallest class size: {size}')
        return size
    
    # Measurement-based selection logic
    def measurement_based_selection(settings, db_path, tables = ['cell', 'nucleus', 'pathogen', 'cytoplasm']):
        class_paths_ls = []
        df, _ = _read_and_merge_data(locs=[db_path],
                                     tables=tables,
                                     verbose=False,
                                     nuclei_limit=settings['nuclei_limit'],
                                     pathogen_limit=settings['pathogen_limit'])

        print('length df 1', len(df))
        df = annotate_conditions(df, cells=['HeLa'], pathogens=['pathogen'], treatments=settings['classes'],
                                 treatment_loc=settings['class_metadata'])#, types=settings['metadata_type_by'])
        print('length df 2', len(df))
        
        png_list_df = filter_png_list(db_path, settings, tables=settings['tables'])

        if settings['custom_measurement']:
            if isinstance(settings['custom_measurement'], list):
                if len(settings['custom_measurement']) == 2:
                    df['recruitment'] = df[f"{settings['custom_measurement'][0]}"] / df[f"{settings['custom_measurement'][1]}"]
                else:
                    df['recruitment'] = df[f"{settings['custom_measurement'][0]}"]
            else:
                print("custom_measurement should be a list.")
                return

        else:
            df['recruitment'] = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"] / df[f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]

        q25 = df['recruitment'].quantile(0.25)
        q75 = df['recruitment'].quantile(0.75)
        df_lower = df[df['recruitment'] <= q25]
        df_upper = df[df['recruitment'] >= q75]

        class_paths_lower = get_paths_from_db(df=df_lower, png_df=png_list_df, image_type=settings['png_type'])
        class_paths_lower = random.sample(class_paths_lower['png_path'].tolist(), settings['size'])
        class_paths_ls.append(class_paths_lower)

        class_paths_upper = get_paths_from_db(df=df_upper, png_df=png_list_df, image_type=settings['png_type'])
        class_paths_upper = random.sample(class_paths_upper['png_path'].tolist(), settings['size'])
        class_paths_ls.append(class_paths_upper)

        return class_paths_ls

    # Metadata-based selection logic
    def metadata_based_selection(db_path, settings):
        class_paths_ls = []
        df = filter_png_list(db_path, settings, tables=settings['tables'])
                
        df = annotate_conditions(df,
                                 cells=None,
                                 cell_loc=None,
                                 pathogens=settings['metadata_item_1_name'],
                                 pathogen_loc=settings['metadata_item_1_value'],
                                 treatments=settings['metadata_item_2_name'],
                                 treatment_loc=settings['metadata_item_2_value'])
        
        #if settings['metadata_type_by'] == 'condition':
        df = df.dropna(subset=['condition'])
            
        display(df)

        size = get_smallest_class_size(df, settings, 'metadata')
        
        for class_ in settings['class_metadata']:
            class_temp_df = df[df['condition'] == class_]
            #class_temp_df = df[df['condition'].isin(class_)]
            print(f'Found {len(class_temp_df)} images for class {class_}')
            class_paths_temp = class_temp_df['png_path'].tolist()

            # Ensure to sample `size` number of images (smallest class size)
            if len(class_paths_temp) > size:
                class_paths_temp = random.sample(class_paths_temp, size)

            class_paths_ls.append(class_paths_temp)

        return class_paths_ls

    # Annotation-based selection logic
    def annotation_based_selection(db_path, dst, settings):
        class_paths_ls = training_dataset_from_annotation(db_path, dst, settings['annotation_column'], annotated_classes=settings['annotated_classes'])

        return class_paths_ls
    
    # Metadata-Annotation-based selection logic
    def metadata_annotation_based_selection(db_path, dst, settings):
        class_paths_ls = training_dataset_from_annotation_metadata(db_path, dst, settings['annotation_column'], annotated_classes=settings['annotated_classes'], metadata_type_by=settings['metadata_type_by'], class_metadata=settings['class_metadata'])

        return class_paths_ls
    
    from .io import _read_and_merge_data, _read_db
    from .utils import get_paths_from_db, annotate_conditions, save_settings
    from .settings import set_generate_training_dataset_defaults
    
    settings = set_generate_training_dataset_defaults(settings)

    if 'nucleus' not in settings['tables']:
        settings['nuclei_limit'] = False
        
    if 'pathogen' not in settings['tables']:
        settings['pathogen_limit'] = 0
       
    # Set default settings and save
    save_settings(settings, 'cv_dataset', show=True)

    class_path_list = None

    if isinstance(settings['src'], str):
        src = [settings['src']]
        settings['src'] = src

    for i, src in enumerate(settings['src']):
        db_path = os.path.join(src, 'measurements', 'measurements.db')
        
        if len(settings['src']) > 1 and i == 0:
            dst = os.path.join(src, 'datasets', 'training_all')
        elif len(settings['src']) == 1:
            dst = os.path.join(src, 'datasets', 'training')

        # Create a new directory for training data if necessary
        if os.path.exists(dst):
            for i in range(1, 100000):
                dst = dst + f'_{i}'
                if not os.path.exists(dst):
                    print(f'Creating new directory for training: {dst}')
                    break

        # Select dataset based on dataset mode
        if settings['dataset_mode'] == 'annotation':
            class_paths_ls = annotation_based_selection(db_path, dst, settings)

        elif settings['dataset_mode'] == 'metadata':
            class_paths_ls = metadata_based_selection(db_path, settings)

        elif settings['dataset_mode'] == 'measurement':
            class_paths_ls = measurement_based_selection(settings, db_path, tables=settings['tables'])
            
        elif settings['dataset_mode'] == 'metadata_annotation':
            class_paths_ls = metadata_annotation_based_selection(db_path, dst, settings)
            
        else:
            print(f"Invalid dataset mode: {settings['dataset_mode']}")
            print(f"Valid options are: 'annotation', 'metadata', 'measurement', 'metadata_annotation'")
            return
        
        if class_path_list is None:
            class_path_list = [[] for _ in range(len(class_paths_ls))]

        # Extend each list in class_path_list with the corresponding list from class_paths_ls
        for idx in range(len(class_paths_ls)):
            class_path_list[idx].extend(class_paths_ls[idx])

    # Generate and return training and testing directories
    print('class_path_list',len(class_path_list))
    train_class_dir, test_class_dir = generate_dataset_from_lists(dst, class_data=class_path_list, classes=settings['class_metadata'], test_split=settings['test_split'])

    return train_class_dir, test_class_dir

def training_dataset_from_annotation(db_path, dst, annotation_column='test', annotated_classes=(1, 2)):
    all_paths = []

    # Connect to the database and retrieve the image paths and annotations
    print(f'Reading DataBase: {db_path}')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Retrieve all paths and annotations from the database
        query = f"SELECT png_path, {annotation_column} FROM png_list"
        cursor.execute(query)
        
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for row in rows:
                all_paths.append(row)

    print('Total paths retrieved:', len(all_paths))
    
    # Filter paths based on annotated_classes
    class_paths = []
    for class_ in annotated_classes:
        class_paths_temp = [path for path, annotation in all_paths if annotation == class_]
        class_paths.append(class_paths_temp)
        print(f'Found {len(class_paths_temp)} images in class {class_}')
        
    # If only one class is provided, create an alternative list by sampling paths from all_paths that are not in the annotated class
    if len(annotated_classes) == 1:
        target_class = annotated_classes[0]
        count_target_class = len(class_paths[0])
        print(f'Annotated class: {target_class} with {count_target_class} images')
        
        # Filter all_paths to exclude paths that belong to the target class
        alt_class_paths = [path for path, annotation in all_paths if annotation != target_class]
        print('Alternative paths available:', len(alt_class_paths))
        
        # Sample the same number of images for both classes
        balanced_count = min(count_target_class, len(alt_class_paths))
        print(f'Sampling {balanced_count} images for each class')

        # Resample target class to match the smaller size
        sampled_target_class_paths = random.sample(class_paths[0], balanced_count)
        sampled_alt_class_paths = random.sample(alt_class_paths, balanced_count)
        
        # Update class paths
        class_paths[0] = sampled_target_class_paths
        class_paths.append(sampled_alt_class_paths)

    print(f'Generated a list of lists from annotation of {len(class_paths)} classes')
    for i, ls in enumerate(class_paths):
        print(f'Class {i}: {len(ls)} images')
        
    return class_paths

def training_dataset_from_annotation_metadata(db_path, dst, annotation_column='test', annotated_classes=(1, 2), metadata_type_by='columnID', class_metadata=['c1','c2']):
    all_paths = []

    # Connect to the database and retrieve the image paths and annotations
    print(f'Reading DataBase: {db_path}')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Retrieve all paths and annotations from the database
        query = f"SELECT png_path, {annotation_column}, row_name, column_name FROM png_list"
        cursor.execute(query)
        
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for row in rows:
                all_paths.append(row)

    print('Total paths retrieved:', len(all_paths))
    
    # Filter all_paths by metadata_type_by and class_metadata
    filtered_paths = []
    metadata_index = {'rowID': 2, 'columnID': 3}.get(metadata_type_by, None)
    if metadata_index is None:
        raise ValueError(f"Invalid metadata_type_by value: {metadata_type_by}. Must be 'rowID' or 'columnID'. {class_metadata} must be a list formatted as ['c1', 'c2'] or ['r1', 'r2']")

    for row in all_paths:
        if row[metadata_index] in class_metadata:
            filtered_paths.append(row)

    print('Total filtered paths:', len(filtered_paths))
    #all_paths = filtered_paths
    all_paths = [(row[0], row[1]) for row in filtered_paths]
    
    # Filter paths based on annotated_classes
    class_paths = []
    for class_ in annotated_classes:
        class_paths_temp = [path for path, annotation in all_paths if annotation == class_]
        class_paths.append(class_paths_temp)
        print(f'Found {len(class_paths_temp)} images in class {class_}')
        
    # If only one class is provided, create an alternative list by sampling paths from all_paths that are not in the annotated class
    if len(annotated_classes) == 1:
        target_class = annotated_classes[0]
        count_target_class = len(class_paths[0])
        print(f'Annotated class: {target_class} with {count_target_class} images')
        
        # Filter all_paths to exclude paths that belong to the target class
        alt_class_paths = [path for path, annotation in all_paths if annotation != target_class]
        print('Alternative paths available:', len(alt_class_paths))
        
        # Sample the same number of images for both classes
        balanced_count = min(count_target_class, len(alt_class_paths))
        print(f'Sampling {balanced_count} images for each class')

        # Resample target class to match the smaller size
        sampled_target_class_paths = random.sample(class_paths[0], balanced_count)
        sampled_alt_class_paths = random.sample(alt_class_paths, balanced_count)
        
        # Update class paths
        class_paths[0] = sampled_target_class_paths
        class_paths.append(sampled_alt_class_paths)

    print(f'Generated a list of lists from annotation of {len(class_paths)} classes')
    for i, ls in enumerate(class_paths):
        print(f'Class {i}: {len(ls)} images')
        
    return class_paths

def generate_dataset_from_lists(dst, class_data, classes, test_split=0.1):
    from .utils import print_progress
    # Make sure that the length of class_data matches the length of classes
    if len(class_data) != len(classes):
        raise ValueError("class_data and classes must have the same length.")

    total_files = sum(len(data) for data in class_data)
    processed_files = 0
    time_ls = []
    
    for cls, data in zip(classes, class_data):
        # Create directories
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
                
        # Split the data
        print('data',len(data), test_split)
        train_data, test_data = train_test_split(data, test_size=test_split, shuffle=True, random_state=42)
        
        # Copy train files
        for path in train_data:
            start = time.time()
            shutil.copy(path, os.path.join(train_class_dir, os.path.basename(path)))
            duration = time.time() - start
            time_ls.append(duration)
            print_progress(processed_files, total_files, n_jobs=1, time_ls=None, batch_size=None, operation_type="Copying files for Train dataset")
            processed_files += 1

        # Copy test files
        for path in test_data:
            start = time.time()
            shutil.copy(path, os.path.join(test_class_dir, os.path.basename(path)))
            duration = time.time() - start
            time_ls.append(duration)
            print_progress(processed_files, total_files, n_jobs=1, time_ls=None, batch_size=None, operation_type="Copying files for Test dataset")
            processed_files += 1

    # Print summary
    for cls in classes:
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        print(f'Train class {cls}: {len(os.listdir(train_class_dir))}, Test class {cls}: {len(os.listdir(test_class_dir))}')

    return os.path.join(dst, 'train'), os.path.join(dst, 'test')

def convert_separate_files_to_yokogawa(folder, regex):
    
    ROWS = "ABCDEFGHIJKLMNOP"
    COLS = [f"{i:02d}" for i in range(1, 25)]
    WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

    def _get_next_well(used_wells):
        plate = 1
        for well in WELLS:
            well_name = f"plate{plate}_{well}"
            if well_name not in used_wells:
                return well_name
            if well == "P24":
                plate += 1
        return f"plate{plate}_A01"

    pattern = re.compile(regex, re.I)

    files_by_region = {}
    rename_log = []
    csv_path = os.path.join(folder, "rename_log.csv")
    used_wells = set()
    region_to_well = {}

    # Group files by (plateID, wellID, fieldID, timeID, chanID)
    for file in os.listdir(folder):
        match = pattern.match(file)
        if not match:
            print(f"Skipping {file}: does not match regex.")
            continue

        meta = match.groupdict()

        # Mandatory metadata
        if 'wellID' not in meta or meta['wellID'] is None:
            print(f"Skipping {file}: missing mandatory wellID.")
            continue
        wellID = meta['wellID']

        # Optional metadata with defaults
        plateID = meta.get('plateID', '1') or '1'
        fieldID = meta.get('fieldID', '1') or '1'
        timeID = int(meta.get('timeID', 1) or 1)
        chanID = int(meta.get('chanID', 1) or 1)
        sliceID = meta.get('sliceID')
        sliceID = int(sliceID) if sliceID is not None else None

        region_key = (plateID, wellID, fieldID, timeID, chanID)

        files_by_region.setdefault(region_key, []).append((file, sliceID))

    # Assign wells and process files per region
    for region, file_list in files_by_region.items():
        if region[:3] not in region_to_well:
            next_well = _get_next_well(used_wells)
            region_to_well[region[:3]] = next_well
            used_wells.add(next_well)

        assigned_well = region_to_well[region[:3]]
        plateID, wellID, fieldID, timeID, chanID = region

        # Check if multiple slices exist and are meaningful
        slice_ids = [sid for _, sid in file_list if sid is not None]
        unique_slices = set(slice_ids)

        images = []
        for filename, _ in sorted(file_list, key=lambda x: x[1] or 1):
            img = tifffile.imread(os.path.join(folder, filename))
            images.append(img)

        # Perform MIP only if multiple unique slices are present
        if len(unique_slices) > 1:
            img_to_save = np.max(np.stack(images), axis=0)
        else:
            img_to_save = images[0]

        dtype = img_to_save.dtype

        new_filename = f"{assigned_well}_T{timeID:04d}F{int(fieldID):03d}L01C{chanID:02d}.tif"
        new_filepath = os.path.join(folder, new_filename)
        tifffile.imwrite(new_filepath, img_to_save.astype(dtype))

        # Log original filenames involved in MIP or single file rename
        original_files = ";".join(f[0] for f in file_list)
        rename_log.append({"Original File(s)": original_files, "Renamed TIFF": new_filename})

    pd.DataFrame(rename_log).to_csv(csv_path, index=False)
    print(f"Processing complete. Files saved in {folder} and rename log saved as {csv_path}.")
    
def convert_to_yokogawa(folder):
    """
    Detects file type in the folder and converts them
    to Yokogawa-style naming with Maximum Intensity Projection (MIP).
    """
    
    #def _get_next_well(used_wells):
    #    """
    #    Determines the next available well position in a 384-well format.
    #    Iterates wells, and after P24, switches to plate2.
    #    """
    #    plate = 1
    #    for well in WELLS:
    #        well_name = f"plate{plate}_{well}"
    #        if well_name not in used_wells:
    #            used_wells.add(well_name)
    #            return well_name
    #        if well == "P24":  
    #            plate += 1  
    #    raise ValueError("All wells exhausted.")
    
    def _get_next_well(used_wells):
        """
        Determines the next available well position across multiple 384-well plates.
        """
        ROWS = "ABCDEFGHIJKLMNOP"
        COLS = [f"{i:02d}" for i in range(1, 25)]
        WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

        plate = 1
        while True:
            for well in WELLS:
                well_name = f"plate{plate}_{well}"
                if well_name not in used_wells:
                    used_wells.add(well_name)
                    return well_name
            plate += 1  # All wells exhausted in current plate, increment to next plate


    # Define 384-well plate format
    ROWS = "ABCDEFGHIJKLMNOP"
    COLS = [f"{i:02d}" for i in range(1, 25)]
    WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

    filenames = []
    rename_log = []
    csv_path = os.path.join(folder, "rename_log.csv")
    used_wells = set()

    # **Dictionary to store well assignments per original file**
    file_to_well = {}

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        ext = file.lower().split('.')[-1]

        # **Assign a well only once per original file**
        if file not in file_to_well:
            file_to_well[file] = _get_next_well(used_wells)
            #used_wells.add(file_to_well[file])  # Mark it as used

        well = file_to_well[file]  # Use the same well for all channels/times

        ### **Process Nikon ND2 Files**
        if ext == 'nd2':
            try:
                nd2 = ND2Reader(path)
                metadata = nd2.metadata

                timepoints = list(range(len(metadata.get("frames", [0])))) or [0]
                fields = list(range(len(metadata.get("fields_of_view", [0])))) or [0]
                z_levels = list(metadata.get("z_levels", range(1))) if metadata.get("z_levels") else [0]
                channels = metadata.get("channels", [])

                for t_idx in timepoints:
                    for f_idx in fields:
                        for c_idx, channel in enumerate(channels):
                            try:
                                mip_image = np.max.reduce([
                                    nd2.get_frame_2D(t=t_idx, v=f_idx, z=z_idx, c=c_idx) 
                                    for z_idx in z_levels
                                ], axis=0)

                                dtype = mip_image.dtype
                                filename = f"{well}_T{t_idx+1:04d}F{f_idx+1:03d}L01C{c_idx+1:02d}.tif"
                                filepath = os.path.join(folder, filename)

                                tifffile.imwrite(filepath, mip_image.astype(dtype))
                                rename_log.append({"Original File": file, "Renamed TIFF": filename})

                            except IndexError:
                                print(f"Warning: ND2 file {file} has an incomplete data structure. Skipping.")

            except Exception as e:
                print(f"Error processing ND2 file {file}: {e}")

        ### **Process Zeiss CZI Files**
        elif ext == 'czi':
            with czifile.CziFile(path) as czi:
                img_data = czi.asarray()  # Read the full image array

                # Remove singleton dimensions (if any)
                img_data = np.squeeze(img_data)

                # Get the actual shape of the data
                shape = img_data.shape
                num_dims = len(shape)

                # Default values if dimensions are missing
                timepoints = 1
                z_levels = 1
                channels = 1

                # Determine dimension mapping dynamically
                if num_dims == 2:  # (Y, X) → Single 2D image
                    y_dim, x_dim = shape
                    img_data = img_data.reshape(1, 1, 1, y_dim, x_dim)  # Add missing dimensions
                elif num_dims == 3:  # (C, Y, X) or (Z, Y, X)
                    if shape[0] <= 4:  # Likely (C, Y, X)
                        channels, y_dim, x_dim = shape
                        img_data = img_data.reshape(1, 1, channels, y_dim, x_dim)  # Add missing dimensions
                    else:  # Likely (Z, Y, X)
                        z_levels, y_dim, x_dim = shape
                        img_data = img_data.reshape(1, z_levels, 1, y_dim, x_dim)  # Add missing dimensions
                elif num_dims == 4:  # Could be (T, C, Y, X) or (T, Z, Y, X) or (Z, C, Y, X)
                    if shape[1] <= 4:  # Assume (T, C, Y, X)
                        timepoints, channels, y_dim, x_dim = shape
                        img_data = img_data.reshape(timepoints, 1, channels, y_dim, x_dim)  # Add missing Z
                    else:  # Assume (T, Z, Y, X) or (Z, C, Y, X)
                        timepoints, z_levels, y_dim, x_dim = shape
                        img_data = img_data.reshape(timepoints, z_levels, 1, y_dim, x_dim)  # Add missing C
                elif num_dims == 5:  # Standard (T, Z, C, Y, X)
                    timepoints, z_levels, channels, y_dim, x_dim = shape
                else:
                    raise ValueError(f"Unexpected CZI shape: {shape}. Unable to process.")

                # Iterate over detected timepoints, channels, and perform MIP over Z
                for t_idx in range(timepoints):
                    for c_idx in range(channels):
                        # Extract Z-stack or single image
                        if z_levels > 1:
                            z_stack = img_data[t_idx, :, c_idx, :, :]  # MIP over Z
                            mip_image = np.max(z_stack, axis=0)
                        else:
                            mip_image = img_data[t_idx, 0, c_idx, :, :]  # No Z, take directly

                        # Ensure correct dtype
                        dtype = mip_image.dtype

                        # Generate Yokogawa-style filename
                        filename = f"{well}_T{t_idx+1:04d}F001L01C{c_idx+1:02d}.tif"
                        filepath = os.path.join(folder, filename)

                        # Save the extracted image
                        tifffile.imwrite(filepath, mip_image.astype(dtype))

                        rename_log.append({"Original File": file, "Renamed TIFF": filename})

        ### **Process Leica LIF Files**
        elif ext == 'lif':
            try:
                lif_file = readlif.Reader(path)

                for image_idx, image in enumerate(lif_file.getIterImage()):
                    timepoints = range(getattr(image.dims, 't', 1))
                    z_levels = range(getattr(image.dims, 'z', 1))
                    channels = range(getattr(image.dims, 'c', 1))

                    for t_idx in timepoints:
                        for c_idx in channels:
                            z_stack = []
                            for z_idx in z_levels:
                                try:
                                    frame = image.getFrame(z=z_idx, t=t_idx, c=c_idx)
                                    z_stack.append(frame)
                                except IndexError:
                                    print(f"Missing frame: T{t_idx}, Z{z_idx}, C{c_idx} in {file}, skipping frame.")
                            
                            if z_stack:
                                mip_image = np.max(np.stack(z_stack), axis=0)
                                dtype = mip_image.dtype
                                filename = f"{well}_T{t_idx+1:04d}F{image_idx+1:03d}L01C{c_idx+1:02d}.tif"
                                filepath = os.path.join(folder, filename)

                                tifffile.imwrite(filepath, mip_image.astype(dtype))
                                rename_log.append({"Original File": file, "Renamed TIFF": filename})

            except Exception as e:
                print(f"Error processing LIF file {file}: {e}")

        ### **Process Standard Image Files (TIFF, PNG, JPEG, BMP)**
        elif ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'] and not file.startswith("plate"):
            try:
                with tifffile.TiffFile(path) as tif:
                    images = tif.asarray()
                    ndim = images.ndim

                    # Defaults
                    t_dim = z_dim = c_dim = 1

                    # Determine dimensions more explicitly
                    if ndim == 2:
                        mip_image = images
                        filename = f"{well}_T0001F001L01C01.tif"
                        tifffile.imwrite(os.path.join(folder, filename), mip_image)
                        rename_log.append({"Original File": file, "Renamed TIFF": filename})
                        continue

                    elif ndim == 3:
                        if images.shape[0] <= 4:  # Likely channels
                            c_dim = images.shape[0]
                            for c in range(c_dim):
                                mip_image = images[c, :, :]
                                filename = f"{well}_T0001F001L01C{c+1:02d}.tif"
                                tifffile.imwrite(os.path.join(folder, filename), mip_image)
                                rename_log.append({"Original File": file, "Renamed TIFF": filename})
                        else:  # Z-stack
                            mip_image = np.max(images, axis=0)
                            filename = f"{well}_T0001F001L01C01.tif"
                            tifffile.imwrite(os.path.join(folder, filename), mip_image)
                            rename_log.append({"Original File": file, "Renamed TIFF": filename})

                    elif ndim == 4:
                        t_dim, z_dim, y_dim, x_dim = images.shape
                        for t in range(t_dim):
                            mip_image = np.max(images[t, :, :, :], axis=0)
                            filename = f"{well}_T{t+1:04d}F001L01C01.tif"
                            tifffile.imwrite(os.path.join(folder, filename), mip_image)
                            rename_log.append({"Original File": file, "Renamed TIFF": filename})

                    else:
                        raise ValueError(f"Unsupported TIFF dimensions: {images.shape}")

            except Exception as e:
                print(f"Error processing standard image file {file}: {e}")


    # Save rename log as CSV
    pd.DataFrame(rename_log).to_csv(csv_path, index=False)
    print(f"Processing complete. Files saved in {folder} and rename log saved as {csv_path}.")
    
def apply_augmentation(image, method):
    if method == 'rotate90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif method == 'rotate180':
        return cv2.rotate(image, cv2.ROTATE_180)
    elif method == 'rotate270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif method == 'flip_h':
        return cv2.flip(image, 1)
    elif method == 'flip_v':
        return cv2.flip(image, 0)
    return image

def process_instruction(entry):
    img = tifffile.imread(entry["src_img"])
    msk = tifffile.imread(entry["src_msk"])
    if entry["augment"]:
        img = apply_augmentation(img, entry["augment"])
        msk = apply_augmentation(msk, entry["augment"])
    tifffile.imwrite(entry["dst_img"], img)
    tifffile.imwrite(entry["dst_msk"], msk)
    return 1

def prepare_cellpose_dataset(input_root, augment_data=False, train_fraction=0.8, n_jobs=None):
    
    from .utils import print_progress

    time_ls = []
    input_root = os.path.abspath(input_root)
    output_root = os.path.join(input_root, "cellpose_dataset")

    def get_augmentations():
        return ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v']

    def find_image_mask_pairs(dataset_path):
        mask_dir = os.path.join(dataset_path, "masks")
        pairs = []
        for fname in os.listdir(dataset_path):
            if fname.lower().endswith((".tif", ".tiff")):
                img_path = os.path.join(dataset_path, fname)
                msk_path = os.path.join(mask_dir, fname)
                if os.path.isfile(msk_path):
                    pairs.append((img_path, msk_path))
        return pairs

    def prepare_output_folders(base):
        for subset in ["train", "test"]:
            os.makedirs(os.path.join(base, subset, "images"), exist_ok=True)
            os.makedirs(os.path.join(base, subset, "masks"), exist_ok=True)

    print("Scanning datasets...")
    datasets = []
    for subdir in os.listdir(input_root):
        dataset_path = os.path.join(input_root, subdir)
        if os.path.isdir(dataset_path) and os.path.isdir(os.path.join(dataset_path, "masks")):
            pairs = find_image_mask_pairs(dataset_path)
            if pairs:
                datasets.append(pairs)
                print(f"  Found {len(pairs)} images in {dataset_path}")

    if not datasets:
        raise ValueError("No valid datasets with images and masks found.")

    prepare_output_folders(output_root)

    min_size = min(len(pairs) for pairs in datasets)
    target_size = min_size if not augment_data else max(len(pairs) for pairs in datasets)

    print("\nPreparing instruction list...")
    instructions = []
    global_index = 0

    for pairs in datasets:
        dataset_len = len(pairs)

        # --- Step 1: Sample or augment ---
        sampled_pairs = []
        if dataset_len >= target_size:
            sampled_pairs = random.sample(pairs, target_size)
        else:
            sampled_pairs = pairs.copy()
            if augment_data:
                needed = target_size - dataset_len
                aug_methods = get_augmentations()
                full_loops = needed // len(aug_methods)
                extra = needed % len(aug_methods)

                for _ in range(full_loops):
                    for (img_path, msk_path), aug in zip(pairs, aug_methods * (dataset_len // len(aug_methods))):
                        sampled_pairs.append((img_path, msk_path, aug))
                if extra > 0:
                    subset = random.sample(pairs * ((extra // len(aug_methods)) + 1), extra)
                    for (img_path, msk_path), aug in zip(subset, aug_methods[:extra]):
                        sampled_pairs.append((img_path, msk_path, aug))

        # Add "no augmentation" tag to original files
        augmented_sampled = [
            (tup[0], tup[1], None) if len(tup) == 2 else tup
            for tup in sampled_pairs
        ]

        # --- Step 2: Split into train/test ---
        random.shuffle(augmented_sampled)
        split_idx = int(train_fraction * len(augmented_sampled))
        split_sets = {
            "train": augmented_sampled[:split_idx],
            "test": augmented_sampled[split_idx:]
        }

        for subset, items in split_sets.items():
            for img_path, msk_path, aug in items:
                dst_img = os.path.join(output_root, subset, "images", f"{global_index:05d}.tif")
                dst_msk = os.path.join(output_root, subset, "masks", f"{global_index:05d}.tif")
                instructions.append({
                    "src_img": img_path,
                    "src_msk": msk_path,
                    "dst_img": dst_img,
                    "dst_msk": dst_msk,
                    "augment": aug
                })
                global_index += 1

    print(f"Total files to process: {len(instructions)}")

    # --- Step 3: Process with multiprocessing ---
    print("Processing images with multiprocessing...")
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    else:
        n_jobs = int(n_jobs)
        
    with Pool(n_jobs) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_instruction, instructions), 1):
            print_progress(i, len(instructions), n_jobs=n_jobs, time_ls=time_ls, batch_size=None, operation_type="cellpose dataset")

    print(f"Done. Dataset saved to: {output_root}")

