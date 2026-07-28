import os, re, json, sqlite3, gc, torch, time, random, shutil, cv2, tarfile, cellpose, glob, queue, threading, tifffile, czifile, atexit, datetime, readlif, tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from collections import defaultdict, Counter
from pathlib import Path
from matplotlib.animation import FuncAnimation
try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass
from skimage.util import img_as_uint
from skimage.exposure import rescale_intensity
import skimage.measure as measure
from skimage import exposure
import imageio.v2 as imageio2
import matplotlib.pyplot as plt
from io import BytesIO
try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass
from multiprocessing import Pool, cpu_count, Process, Queue, Value, Lock
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import seaborn as sns 
from nd2reader import ND2Reader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pylibCZIrw import czi as pyczi

# Fail-loud accounting. Every per-file skip below is recorded on a RunLedger
# so a batch that lost 40 of 384 files says so at the end and stamps the
# artifact it produced, instead of writing a silently-short result.
from .errors import RunLedger, ConfigurationError, raise_if_strict

# One definition of what a well is called. spacr.convert imports nothing
# heavier than spacr.schema, so this costs nothing here, and it is the reason
# the two Yokogawa converters below can name a well on a 1536-well plate:
# they used to carry three hand-written copies of "ABCDEFGHIJKLMNOP" and
# range(1, 25), which stop at P24.
from . import convert as _cv

def process_non_tif_non_2D_images(folder):
    """Split multi-dimensional or non-TIFF images in ``folder`` into per-channel TIFFs.

    Grayscale non-TIFF images are converted to TIFF in place. Multi-
    dimensional images (3D/4D/5D) are split into one grayscale TIFF per
    ``(channel, Z, T)`` combination. Bit depth is preserved.

    A file that cannot be read is recorded on a
    :class:`spacr.errors.RunLedger` and skipped, so one corrupt image
    does not abort the folder — but the ledger prints a loud summary of
    everything that was skipped once the folder is done.

    :param folder: Directory containing the input images.
    :returns: the :class:`spacr.errors.RunLedger` for the conversion, so
        callers can check ``ledger.is_complete`` before trusting the
        folder's contents.
    """

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
            # Return a numpy dtype like every sibling branch. Returning PIL's
            # mode string here fed image.astype('RGB') -> TypeError (swallowed,
            # so multi-channel PNG/JPEG were silently dropped), and astype('L')
            # -> uint64, inflating 8-bit greyscale 8x despite the
            # "bit depth is preserved" contract.
            image = np.array(Image.open(file_path))
            return image, image.dtype
        
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
    ledger = RunLedger('process_non_tif_non_2D_images')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        ext = os.path.splitext(file_path)[1].lower()

        if ext in supported_formats:
            print(f"Processing {filename}")
            with ledger.item(filename, stage='split_channels',
                             echo=f"Error processing {filename}"):
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

    # Last thing on screen, so a partial conversion cannot scroll past.
    ledger.finalize()
    return ledger

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
    """Round-robin iterator over multiple DataLoaders.

    Yields ``(loader_index, batch)`` pairs, drawing from a random loader
    each step and dropping loaders once exhausted.

    :param train_loaders: DataLoaders to combine.
    :raises StopIteration: when every wrapped loader is exhausted.
    """

    def __init__(self, train_loaders):
        """Store loaders and initialise per-loader iterators."""
        self.train_loaders = train_loaders
        # Carry the ORIGINAL loader index alongside each iterator: the list is
        # shuffled and pruned as loaders empty, so a positional index would not
        # identify which loader a batch came from.
        self.loader_iters = [(i, iter(loader))
                             for i, loader in enumerate(train_loaders)]

    def __iter__(self):
        """Return self — this object is its own iterator."""
        return self

    def __next__(self):
        """Return ``(loader_index, batch)`` from a randomly-chosen live loader."""
        while self.loader_iters:
            random.shuffle(self.loader_iters)
            for pos, (idx, loader_iter) in enumerate(self.loader_iters):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    # Exhausted: it sits at a position before `pos` and is
                    # dropped below. Do NOT pop here — mutating the list while
                    # enumerating it skips the entry that shifts into the freed
                    # slot, which silently discarded still-live loaders and
                    # truncated the combined stream.
                    continue
                if pos:
                    self.loader_iters = self.loader_iters[pos:]
                return idx, batch
            # Every remaining loader raised StopIteration on this pass.
            self.loader_iters = []
        raise StopIteration

class CombinedDataset(Dataset):
    """Concatenation of multiple ``Dataset`` objects behind a single index space.

    :param datasets: Datasets to concatenate; their samples must be
        index-compatible.
    :param shuffle: If True, index lookups are permuted once at
        construction time. Default ``True``.
    """

    def __init__(self, datasets, shuffle=True):
        """Precompute per-dataset lengths and optionally shuffle indices."""
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
        """Return the sample at ``index`` from the appropriate sub-dataset."""
        if self.shuffle:
            index = self.indices[index]
        for dataset, length in zip(self.datasets, self.lengths):
            if index < length:
                return dataset[index]
            index -= length
    def __len__(self):
        """Return the total number of samples across all sub-datasets."""
        return self.total_length
    
class NoClassDataset(Dataset):
    """Flat directory of unlabelled images returned alongside their file paths.

    :param data_dir: Directory containing image files.
    :param transform: Optional callable applied to each PIL image. If
        ``None``, images are converted with ``ToTensor``.
    :param shuffle: If True, shuffle filename list at construction.
        Default ``True``.
    :param load_to_memory: If True, decode all images once and hold them
        in RAM. Default ``False``.
    """

    def __init__(self, data_dir, transform=None, shuffle=True, load_to_memory=False):
        """Enumerate files in ``data_dir`` and optionally preload them."""
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        # Hidden files are not images. A crop folder carries a
        # `.spacr_crop_format.json` sidecar (spacr.crops), and a folder that
        # has been near a Mac or Windows carries .DS_Store / Thumbs.db; every
        # one of them used to be handed to Image.open as a sample.
        self.filenames = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f)) and not f.startswith('.')
        ]
        if self.shuffle:
            self.shuffle_dataset()
        if self.load_to_memory:
            self.images = [self.load_image(f) for f in self.filenames]
    
    def load_image(self, img_path):
        """Return the image at ``img_path`` decoded as RGB.

        :param img_path: Path to the image file.
        :returns: PIL ``Image`` in RGB mode.
        """
        img = Image.open(img_path).convert('RGB')
        return img

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.filenames)

    def shuffle_dataset(self):
        """Shuffle the internal filename list in place."""
        if self.shuffle:
            random.shuffle(self.filenames)

    def __getitem__(self, index):
        """Return ``(image_tensor, filename)`` for the given index.

        :param index: Position within the dataset.
        :returns: ``(tensor, path)`` where ``tensor`` is the transformed
            image and ``path`` is the source filename.
        """
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = self.load_image(self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        return img, self.filenames[index]


class spacrDataset(Dataset):
    """Image classification dataset that reads class subfolders under ``data_dir``.

    :param data_dir: Root directory containing one subdirectory per class.
    :param loader_classes: Ordered list of class names — the index in
        this list becomes the integer label.
    :param transform: Optional callable applied to each PIL image.
    :param shuffle: If True, shuffle files+labels at construction.
    :param pin_memory: If True, eagerly load every image into RAM via a
        multiprocessing pool.
    :param specific_files: Optional explicit list of image paths. If
        supplied together with ``specific_labels``, directory scanning
        is skipped.
    :param specific_labels: Labels paired with ``specific_files``.
    """

    def __init__(self, data_dir, loader_classes, transform=None, shuffle=True, pin_memory=False, specific_files=None, specific_labels=None):
        """Build the filename/label lists and optionally preload images."""
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
                # A class folder that was never created is a real condition,
                # not a crash: generate_training_dataset only makes a folder
                # for a class it actually selected rows for. Skip it so the
                # empty-dataset guard below can report every class at once,
                # rather than dying on os.listdir of the first missing one.
                if not os.path.isdir(class_path):
                    continue
                # Hidden files are not samples: a class folder written by
                # generate_dataset_from_lists carries the crop-format sidecar
                # `.spacr_crop_format.json`, and any folder that has been near
                # a Mac carries .DS_Store. Both used to reach Image.open.
                class_files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                               if os.path.isfile(os.path.join(class_path, f))
                               and not f.startswith('.')]
                self.filenames.extend(class_files)
                self.labels.extend([self.classes.index(class_name)] * len(class_files))
        
        # An empty dataset must say so HERE, where the directory and the class
        # names are still in hand. Left to run on, shuffle_dataset() does
        # `zip(*[])` and dies with "not enough values to unpack (expected 2,
        # got 0)" -- which tells the user nothing about which folder was empty
        # or which classes were looked for. This is reachable whenever
        # generate_training_dataset selects no rows: a class_metadata value
        # that matches nothing, an annotation column with no positives, or a
        # filter that removed everything.
        if not self.filenames:
            looked = []
            for class_name in self.classes:
                cp = os.path.join(data_dir, class_name)
                if not os.path.isdir(cp):
                    looked.append(f"  {class_name}: NO SUCH FOLDER ({cp})")
                else:
                    n = len([f for f in os.listdir(cp) if not f.startswith('.')])
                    looked.append(f"  {class_name}: {n} file(s) in {cp}")
            raise ValueError(
                "The training dataset is empty -- no images were found for any "
                "class, so there is nothing to train on.\n"
                f"Looked under {data_dir} for classes {list(self.classes)}:\n"
                + "\n".join(looked) +
                "\n\nThis usually means the dataset-generation step selected no "
                "rows. Check that class_metadata values actually occur in the "
                "column named by metadata_type_by, that the annotation column "
                "holds the classes in annotated_classes, and that png_type "
                "matches the crops that exist.")

        if self.shuffle:
            self.shuffle_dataset()

        if self.pin_memory:
            # Use multiprocessing to load images in parallel
            with Pool(processes=cpu_count()) as pool:
                self.images = pool.map(self.load_image, self.filenames)
        else:
            self.images = None

    def load_image(self, img_path):
        """Return the image at ``img_path`` decoded as RGB with EXIF orientation applied."""
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)  # Handle image orientation
        return img

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def shuffle_dataset(self):
        """Jointly shuffle ``filenames`` and ``labels`` in place."""
        combined = list(zip(self.filenames, self.labels))
        random.shuffle(combined)
        self.filenames, self.labels = zip(*combined)

    def get_plate(self, filepath):
        """Return the plate identifier parsed from a filename (leading token before ``_``).

        :param filepath: Image path.
        :returns: Plate ID string.
        """
        filename = os.path.basename(filepath)
        return filename.split('_')[0]

    def __getitem__(self, index):
        """Return ``(image, label, filename)`` for the given index."""
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
    """DataLoader that pre-fetches batches into a queue on a background thread.

    Wraps ``torch.utils.data.DataLoader`` and runs a daemon thread that
    stays one or more batches ahead of consumption to hide I/O latency.
    End-of-stream is signalled with a sentinel, so the full batch stream
    is always delivered.

    :param preload_batches: Number of batches to keep queued ahead.
        Default ``1``.
    """

    def __init__(self, *args, preload_batches=1, **kwargs):
        """Initialise the underlying DataLoader and the preload queue."""
        super().__init__(*args, **kwargs)
        self.preload_batches = preload_batches
        # NOTE: the preloader used to run in a multiprocessing.Process writing
        # into a multiprocessing.Queue, and __next__ stopped as soon as
        # `not process.is_alive() and queue.empty()`. Both are unreliable: an
        # mp.Queue can still have data in flight after the child exits, and the
        # child only ever advanced its OWN copy of the iterator. The loader
        # therefore silently yielded a truncated stream (0 of 4 batches in
        # testing). A daemon THREAD shares the iterator, needs no pickling, and
        # a sentinel gives an unambiguous end-of-stream signal.
        self.batch_queue = queue.Queue(maxsize=max(1, preload_batches))
        self.thread = None
        self.current_batch_index = 0
        self._stop_event = False
        self._sentinel = object()
        self._error = None
        self.pin_memory = kwargs.get('pin_memory', False)
        atexit.register(self.cleanup)

    def _preload_next_batches(self, q, iterator):
        """Feed every batch of ``iterator`` into ``q``, then a sentinel
        marking end-of-stream.

        ``q`` and ``iterator`` are passed in rather than read off ``self`` so
        a producer started by an earlier ``__iter__`` can never write into a
        queue created by a later one (that duplicated the whole stream).
        """
        try:
            for batch in iterator:
                if self._stop_event:
                    break
                if self.pin_memory:
                    batch = self._pin_memory_batch(batch)
                q.put(batch)
        except Exception as e:
            # Hand the failure to the consumer instead of swallowing it: a
            # collate/decode error used to look identical to "dataset is
            # empty", which silently trains a model on no data.
            self._error = e
        finally:
            q.put(self._sentinel)

    def _pin_memory_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.pin_memory() if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.pin_memory()
        else:
            return batch

    def __iter__(self):
        """Start a fresh pass over the data and return self.

        Safe to call more than once (``list(iter(dl))`` calls it twice): any
        in-flight producer is stopped first, so the stream is never doubled.
        """
        self.cleanup()
        self._stop_event = False
        self._error = None
        self.current_batch_index = 0
        q = queue.Queue(maxsize=max(1, self.preload_batches))
        self.batch_queue = q
        iterator = iter(super().__iter__())
        self._iterator = iterator
        self.thread = threading.Thread(
            target=self._preload_next_batches, args=(q, iterator), daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        """Return the next queued batch, or raise ``StopIteration`` at the
        sentinel the preloader pushes when the stream is exhausted."""
        try:
            next_batch = self.batch_queue.get(timeout=60)
        except queue.Empty:
            raise StopIteration
        if next_batch is self._sentinel:
            if self._error is not None:
                err, self._error = self._error, None
                raise err
            raise StopIteration
        self.current_batch_index += 1
        return next_batch

    def cleanup(self):
        """Signal the preloader to stop and join the background thread."""
        self._stop_event = True
        thread = getattr(self, 'thread', None)
        if thread is not None and thread.is_alive():
            # Drain so a full queue can't block the producer's final put.
            try:
                while True:
                    self.batch_queue.get_nowait()
            except Exception:
                pass
            thread.join(timeout=5)

    def __del__(self):
        """Ensure background resources are released on garbage collection."""
        self.cleanup()

class TarImageDataset(Dataset):
    """Image dataset backed by a tar archive, decoded on demand.

    A tar written by :func:`generate_dataset` from on-demand crops carries a
    ``.spacr_crop_format.json`` member -- the same marker
    :mod:`spacr.crops` writes into a crop folder, travelling with the bytes it
    describes. It is **not** an image, so it is excluded from the sample list
    and surfaced as :attr:`crop_format` instead; an archive without one
    reports None, which is every tar written before this existed.

    The pixels are handed over exactly as they are stored. A legacy archive is
    NOT silently un-reversed here: a model's weights are tied to the channel
    order it was trained on, so correcting the order at inference time would
    quietly invalidate every model trained before the fix. :attr:`crop_format`
    is what lets a caller notice.

    :param tar_path: Path to the tar archive.
    :param transform: Optional callable applied to each PIL image.
    """

    def __init__(self, tar_path, transform=None):
        """Enumerate archive members without extracting."""
        self.tar_path = tar_path
        self.transform = transform
        self.crop_format = None

        # Open the tar file just to build the list of members
        from . import crops
        with tarfile.open(self.tar_path, 'r') as f:
            self.members = []
            for m in f.getmembers():
                if not m.isfile():
                    continue
                if os.path.basename(m.name) == crops.CROP_FORMAT_SIDECAR:
                    try:
                        payload = json.loads(f.extractfile(m).read().decode('utf-8'))
                        self.crop_format = int(payload.get('spacr_crop_format'))
                    except Exception:
                        self.crop_format = None
                    continue
                self.members.append(m)

    def __len__(self):
        """Return the number of image members in the archive."""
        return len(self.members)

    def __getitem__(self, idx):
        """Return ``(image, member_name)`` extracted from the tar at ``idx``."""
        with tarfile.open(self.tar_path, 'r') as f:
            m = self.members[idx]
            img_file = f.extractfile(m)
            img = Image.open(BytesIO(img_file.read())).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, m.name

def load_images_from_paths(images_by_key):
    """Load images grouped by key into NumPy arrays.

    :param images_by_key: Mapping of key -> list of image paths.
    :returns: Mapping of the same keys -> list of ``ndarray`` images.
        Paths that fail to load are skipped, recorded on a
        :class:`spacr.errors.RunLedger` and reported in a loud summary,
        so a short list is never mistaken for a complete one.
    """
    images_dict = {}
    ledger = RunLedger('load_images_from_paths')

    for key, paths in images_by_key.items():
        images_dict[key] = []
        for path in paths:
            with ledger.item(path, stage='load',
                             echo=f"Error loading image from {path}"):
                with Image.open(path) as img:
                    images_dict[key].append(np.array(img))

    ledger.finalize()
    return images_dict

#@log_function_call 
def _rename_and_organize_image_files(src, regex, batch_size=100, metadata_type='', img_format='.tif', timelapse=False, save_original_images=True):
    """
    Convert z-stack images to maximum intensity projection (MIP) images and
    write the merged multi-channel ``stack/`` arrays **directly** — without
    ever creating the intermediate per-channel sub-folders.

    Instead of MIP-ing each channel to a ``src/<channel>/`` folder and then
    re-reading those folders to merge them (which duplicated the pixel data on
    disk), this builds an in-memory dict ``{fov_filename: {channel: mip}}`` and
    concatenates the channels of each FOV into one ``stack/<fov>.npy``. The
    merge order and MIP maths are identical to the old folder+``_merge_file``
    path, so the produced stacks are byte-for-byte the same.

    Args:
        src (str): The source directory containing the z-stack images.
        regex (str): The regular expression pattern used to match the filenames of the z-stack images.
        batch_size (int, optional): The number of images to process in each batch. Defaults to 100.
        metadata_type (str, optional): The type of metadata associated with the images. Defaults to ''.
        save_original_images (bool, optional): When True (default) the raw input
            images are moved aside into ``src/orig/`` for safekeeping. When
            False they are deleted after the stack is written, so the pixel data
            lives only in ``stack/`` (no duplication). Defaults to True.

    Returns:
        int: the number of distinct channels found (0 when nothing was processed).
    """

    if isinstance(img_format, str):
        img_format = [img_format]

    from .utils import _extract_filename_metadata, print_progress

    regular_expression = re.compile(regex)
    stack_path = os.path.join(src, 'stack')
    files_processed = 0
    channels_seen = set()
    if not os.path.exists(stack_path) or (os.path.isdir(stack_path) and len(os.listdir(stack_path)) == 0):
        all_filenames = [filename for filename in os.listdir(src) if any(filename.endswith(ext) for ext in img_format)]
        print(f'All files: {len(all_filenames)} in {src}')
        all_filenames = [f for f in all_filenames if not f.startswith('.')] #Exclude hidden files
        time_ls = []
        image_paths_by_key = _extract_filename_metadata(all_filenames, src, regular_expression, metadata_type)
        # Convert dictionary keys to a list for batching
        batching_keys = list(image_paths_by_key.keys())
        print(f'All unique FOV: {len(image_paths_by_key)} in {src}')

        # fov_channels[output_filename][channel] = MIP array. We collect every
        # channel's MIP into this dict and only write the concatenated stack
        # once a FOV has been fully assembled (below).
        fov_channels = {}
        for idx in range(0, len(image_paths_by_key), batch_size):
            start = time.time()

            # Select batch keys and create a subset of the dictionary for this batch
            batch_keys = batching_keys[idx:idx+batch_size]
            batch_images_by_key = {key: image_paths_by_key[key] for key in batch_keys}
            images_by_key = load_images_from_paths(batch_images_by_key)

            # Process each batch of images
            for i, (key, images) in enumerate(images_by_key.items()):

                plate, well, field, channel, timeID, sliceID = key

                # load_images_from_paths deliberately skips unreadable files, so
                # this list can be empty. np.stack([]) below used to raise and
                # abort the whole ingest before stack/ was written, discarding
                # every healthy FOV in the plate because of one corrupt raw.
                if not images:
                    print(f"Warning: no readable images for {key}, skipping")
                    files_processed += 1
                    continue

                if timelapse:
                    output_filename = f'{plate}_{well}_{field}.tif'
                else:
                    output_filename = f'{plate}_{well}_{field}_{timeID}.tif'

                mip = np.max(np.stack(images), axis=0)
                channels_seen.add(channel)
                # Combine, do not overwrite. The grouping key built in
                # utils._extract_filename_metadata includes sliceID, so with
                # cellvoyager/cq1 metadata every z-plane arrives as its OWN key
                # and this assignment let each plane replace the last: a
                # 21-plane stack silently became one arbitrarily chosen plane,
                # decided by os.listdir order, with no warning and no log line.
                # (Under metadata_type='auto' the regex has no sliceID group,
                # so every plane is already in `images` and this is a no-op.)
                _chans = fov_channels.setdefault(output_filename, {})
                _prev = _chans.get(channel)
                _chans[channel] = mip if _prev is None else np.maximum(_prev, mip)

                files_processed += 1
                stop = time.time()
                duration = stop - start
                time_ls.append(duration)
                files_to_process = len(all_filenames)
                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=batch_size, operation_type='Preprocessing filenames')

            images_by_key.clear()

        # Assemble each FOV's channels into a single stacked .npy, using the same
        # sorted-channel order the old folder-based _merge_channels used.
        os.makedirs(stack_path, exist_ok=True)
        sorted_channels = sorted(channels_seen)
        for output_filename, chan_mips in fov_channels.items():
            file_root = os.path.splitext(output_filename)[0]
            new_file = os.path.join(stack_path, file_root + '.npy')
            if os.path.exists(new_file):
                print(f'WARNING: A file with the same name already exists at location {new_file}')
                continue
            planes = []
            for channel in sorted_channels:
                mip = chan_mips.get(channel)
                if mip is None:
                    print(f"Warning: FOV {output_filename} is missing channel {channel}")
                    continue
                planes.append(np.expand_dims(mip, axis=2))
            if planes:
                np.save(new_file, np.concatenate(planes, axis=2))
            else:
                print(f"No valid channels to merge for file {output_filename}")
        fov_channels.clear()

        # Handle the raw input images: keep a backup copy under orig/ only when
        # requested, otherwise delete them (the pixels now live in stack/).
        if save_original_images:
            newpath = os.path.join(src, 'orig')
            os.makedirs(newpath, exist_ok=True)
            for filename in os.listdir(src):
                if os.path.splitext(filename)[1] in img_format:
                    move = os.path.join(newpath, filename)
                    if os.path.exists(move):
                        print(f'WARNING: A file with the same name already exists at location {move}')
                    else:
                        shutil.move(os.path.join(src, filename), move)
        else:
            for filename in os.listdir(src):
                if os.path.splitext(filename)[1] in img_format:
                    try:
                        os.remove(os.path.join(src, filename))
                    except OSError as e:
                        print(f"Warning: could not delete original image {filename}: {e}")
    files_processed = 0
    return len(channels_seen)

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
    
    from .utils import _int_or_token, _convert_cq1_well_id

    src_path = src
    src = Path(src)
    valid_exts = ['.tif', '.png']

    ledger = RunLedger('_move_to_chan_folder')
    if not (src / 'stack').exists():
        for file in src.iterdir():
            if file.is_file():
                name, ext = file.stem, file.suffix
                if ext in valid_exts:
                    metadata = re.match(regex, file.name)
                    with ledger.item(
                            file.name, stage='parse_filename',
                            echo=(f"Could not extract information from filename "
                                  f"{name}{ext} with {regex}")):
                        try:
                            plateID = metadata.group('plateID')
                        except Exception:
                            plateID = src.name

                        wellID = metadata.group('wellID')
                        fieldID = metadata.group('fieldID')
                        chanID = metadata.group('chanID')
                        timeID = metadata.group('timeID')

                        # Undo zero padding, but keep a token that holds no
                        # integer rather than turning it into '0' — see
                        # utils._int_or_token.
                        if wellID[0].isdigit():
                            wellID = _int_or_token(wellID)
                        if fieldID[0].isdigit():
                            fieldID = _int_or_token(fieldID)
                        if chanID[0].isdigit():
                            chanID = _int_or_token(chanID)
                        if timeID[0].isdigit():
                            timeID = _int_or_token(timeID)

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
    # Files whose metadata could not be parsed never reach a channel folder;
    # without this the plate silently continues with fewer fields.
    # Returns None, as it always has — callers treat this as a side-effecting
    # sorter and one existing caller asserts on the None.
    ledger.finalize()
    return

def _merge_channels(src, plot=False):
    """
    Merge the channels in the given source directory and save the merged files in a 'stack' directory without using multiprocessing.
    """

    from .plot import plot_arrays
    from .utils import print_progress
    
    stack_dir = os.path.join(src, 'stack')
    print(f'generated stack dir at {stack_dir}')
    
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
                # A 2-D array has no depth axis to concatenate onto; promote it to
                # (H, W, 1) first. Previously np.concatenate(..., axis=2) on the raw
                # 2-D array raised AxisError.
                if array.ndim == 2:
                    array = array[:, :, np.newaxis]
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
                # `start` used to be bound only in the non-timelapse branch, so
                # this branch raised UnboundLocalError on its first group and
                # the except below reported it as a filename-metadata problem
                # while silently writing nothing. Time per group, to match the
                # group-based files_processed/files_to_process below.
                start = time.time()
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
                # A count, not the list-of-lists: print_progress normalises a
                # list via len(set(...)), which raises on unhashable lists.
                files_to_process = len(time_stack_path_lists)
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

    # Channel indices may arrive as strings (e.g. from a settings CSV);
    # coerce so ``stack[:, :, :, channel]`` indexing works.
    channels = [int(c) for c in channels]

    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    #for channel in range(stack.shape[-1]):
    time_ls = []
    for i, channel in enumerate(channels):
        start = time.time()
        # Default normalisation params for any channel that isn't one
        # of the recognised object channels (e.g. an organelle channel,
        # or an intensity-only channel measured but not segmented).
        # Without these defaults a channel matching NONE of the object
        # types below raised UnboundLocalError: 'background'.
        background = settings.get('background', 100)
        signal_threshold = settings.get('Signal_to_noise', 10) * background
        remove_background = settings.get('remove_background', False)

        if settings.get('nucleus_channel') is not None and channel == settings['nucleus_channel']:
            background = settings['nucleus_background']
            signal_threshold = settings['nucleus_Signal_to_noise']*settings['nucleus_background']
            remove_background = settings['remove_background_nucleus']

        if settings.get('cell_channel') is not None and channel == settings['cell_channel']:
            background = settings['cell_background']
            signal_threshold = settings['cell_Signal_to_noise']*settings['cell_background']
            remove_background = settings['remove_background_cell']

        if settings.get('pathogen_channel') is not None and channel == settings['pathogen_channel']:
            background = settings['pathogen_background']
            signal_threshold = settings['pathogen_Signal_to_noise']*settings['pathogen_background']
            remove_background = settings['remove_background_pathogen']

        # Organelle channel — use organelle-specific settings when
        # present, otherwise the generic defaults above.
        if settings.get('organelle_channel') is not None and channel == settings['organelle_channel']:
            background = settings.get('organelle_background', background)
            signal_threshold = settings.get(
                'organelle_Signal_to_noise',
                settings.get('Signal_to_noise', 10)) * background
            remove_background = settings.get(
                'remove_background_organelle', remove_background)

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

def concatenate_and_normalize(src, channels, save_dtype=np.float32, settings=None):
    """Concatenate per-file channel arrays and normalise them into a single stack.

    :param src: Directory containing per-FOV ``.npy`` channel arrays.
    :param channels: Channel indices to keep in the output stack.
    :param save_dtype: NumPy dtype for the saved normalised arrays.
        Default ``np.float32``.
    :param settings: Preprocessing settings dict. **Required** — it must
        contain the background, signal-to-noise, randomize, timelapse,
        batch_size and plotting keys used elsewhere in preprocessing. The
        ``None`` in the signature is kept only so the argument can still be
        passed positionally; omitting it is an error.
    :returns: Path to the directory where normalised arrays were saved.
    :raises ValueError: if ``settings`` is not supplied.
    """
    # `settings = {}` used to be substituted here, but the very next reads are
    # settings['timelapse'] / ['randomize'] / ['batch_size'], so the empty dict
    # could only ever produce a cryptic KeyError from deep inside the function
    # (after masks/ had already been created). Say what is actually wrong.
    if settings is None:
        raise ValueError(
            "concatenate_and_normalize requires a settings dict (it reads "
            "'timelapse', 'randomize', 'batch_size', 'lower_percentile' and the "
            "per-channel background / Signal_to_noise keys); pass the dict "
            "returned by settings.set_default_settings_preprocess_img_data.")
    from .utils import print_progress
    from .plot import plot_arrays

    # Coerce channel indices to int up-front so both the per-batch
    # normalisation and the ``normalized_stack[..., channels]`` slice work
    # even when channels came through as strings ('0', '1', ...).
    # Drop Nones first: an unused object channel is passed as None, and
    # coercing before the (later) None filter made int(None) raise TypeError.
    channels = [int(c) for c in channels if c is not None]

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
    
    print(f"Generating concatenated and normalized channel data for channels: {channels}")

    paths = []
    time_ls = []
    output_fldr = os.path.join(os.path.dirname(src), 'masks')
    os.makedirs(output_fldr, exist_ok=True)
    # Every FOV that fails to load is dropped from the normalised stacks.
    # Nothing downstream can tell, so account for it here.
    ledger = RunLedger('concatenate_and_normalize')

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
                np.savez_compressed(save_loc, data=normalized_stack, filenames=filenames_region)
                
                # Only plot when the user asked for it: an interactive
                # matplotlib backend makes plt.show() block, which would hang
                # the whole pipeline in a script/terminal run.
                if i == 0 and settings.get('plot'):
                    plot_arrays(save_loc, settings['figuresize'], settings['cmap'], nr=settings['nr'], normalize=False)
                
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
            # An unreadable file must skip only its own accumulation. The old
            # `continue` also jumped past the batch-flush check below, so a bad
            # file in the final position discarded every good image already
            # collected in that batch (and elsewhere merged two batches into
            # one, silently changing the per-batch normalisation grouping).
            with ledger.item(path, stage='load_npy',
                             echo=f"Error loading file {path}"):
                array = np.load(path)
                stack_ls.append(array)
                filenames_batch.append(os.path.basename(path))
                stop = time.time()
                duration = stop - start
                time_ls.append(duration)
                files_processed += 1
                files_to_process = nr_files
                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Concatinating")

            # `stack_ls and` guards the case where every file in a batch failed:
            # np.stack([]) would raise.
            if stack_ls and ((i + 1) % settings['batch_size'] == 0 or i + 1 == nr_files):
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
                # Lossless-compressed so the on-disk normalised batch is much
                # smaller (np.load reads it transparently); it's deleted with
                # masks/ after merged/ is built unless keep_intermediate is set.
                np.savez_compressed(save_loc, data=normalized_stack, filenames=filenames_batch)
                # Gated on settings['plot'] — see the timelapse branch above:
                # an interactive backend blocks the pipeline on plt.show().
                if batch_index == 0 and settings.get('plot'):
                    print(f"plotting: {save_loc}")
                    plot_arrays(save_loc, settings['figuresize'], settings['cmap'], nr=settings['nr'], normalize=False)
                
                batch_index += 1
                del stack, normalized_stack
                stack_ls = []
                filenames_batch = []
                padded_stack_ls = []

    print(f'All files concatenated and normalized. Saved to: {output_fldr}')
    # Emitted last so a partially-loaded stack cannot scroll off the top of
    # a 400-line progress log. No stamp: output_fldr is masks/, which the
    # segmentation step globs, and a stray sidecar there is not worth the risk.
    ledger.finalize()
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

def _normalize_stack(src, backgrounds=None, remove_backgrounds=None, lower_percentile=2, save_dtype=np.float32, signal_to_noise=None, signal_thresholds=None):
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
    if backgrounds is None:
        backgrounds = [100, 100, 100]
    if remove_backgrounds is None:
        remove_backgrounds = [False, False, False]
    if signal_to_noise is None:
        signal_to_noise = [5, 5, 5]
    if signal_thresholds is None:
        signal_thresholds = [1000, 1000, 1000]
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    output_fldr = os.path.join(os.path.dirname(src), 'masks')
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
            # Seeded because the per-frame progress print below formats these
            # unconditionally while they are only assigned for frames that have
            # non-zero pixels: a blank FIRST frame used to abort the whole run
            # with UnboundLocalError, and a later blank frame reported the
            # previous frame's percentiles.
            lower = upper = 0.0
            for array_index in range(single_channel.shape[0]):
                start = time.time()
                arr_2d = single_channel[array_index, :, :]
                non_zero_arr_2d = arr_2d[arr_2d != 0]
                if non_zero_arr_2d.size > 0:
                    lower, upper = np.percentile(non_zero_arr_2d, (lower_percentile, upper_p))
                    signal_to_noise_ratio = upper / lower
                else:
                    lower, upper = 0.0, 0.0
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
    output_fldr = os.path.join(os.path.dirname(src), 'masks')
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
        if not arrays:
            continue
        arrays = np.stack(arrays, axis=0)
        # NOTE: this loop must stay INSIDE the per-(plate, well, field) loop.
        # When it was dedented, `arrays` was unbound if no filename matched
        # the regex (UnboundLocalError) and only the LAST field ever got a
        # movie — every other field was silently dropped.
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
    """Recursively delete every empty subdirectory under ``folder_path``.

    :param folder_path: Root directory to scan.
    :returns: None
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

def preprocess_img_data(settings):
    """Convert raw microscopy images into normalized, channel-merged ``.npy`` stacks ready for mask generation.

    Usually invoked internally by
    :func:`spacr.core.preprocess_generate_masks`, but callable directly
    when you only want the preprocessing half. Converts z-stacks to MIPs,
    renames files into the Yokogawa/spacr layout, merges per-channel
    folders into stacked ``.npy`` arrays with optional background
    subtraction and percentile normalization, and (in ``test_mode``)
    emits example plots.

    :param settings: Preprocessing settings dict, canonicalized via
        :func:`spacr.settings.set_default_settings_preprocess_img_data`.
        Key entries:

        - ``src`` — folder of raw images (``.tif/.nd2/.czi/.lif`` etc.).
        - ``metadata_type`` — ``'cellvoyager'`` / ``'auto'``; drives
          filename regex.
        - ``custom_regex`` — override the built-in regex.
        - ``cell_channel``, ``nucleus_channel``, ``pathogen_channel``,
          ``organelle_channel``, ``channels`` — channel selection.
        - ``all_to_mip`` — max-project z-stacks before saving.
        - ``remove_background_cell`` / ``_nucleus`` / ``_pathogen`` and
          the ``*_background`` cutoffs.
        - ``normalize``, ``lower_percentile``, ``save_dtype``.
        - ``batch_size``, ``randomize``, ``test_mode``, ``test_images``,
          ``plot``, ``cmap``, ``figuresize``.

    :returns: Tuple ``(settings, src)`` — ``settings`` with defaults
        applied and ``src`` pointing at the folder containing the
        generated ``stack/`` / ``channel_stack/`` outputs (the
        downstream mask stage reads from here).

    Example:
        .. code-block:: python

            from spacr.io import preprocess_img_data
            settings = {
                'src': '/data/plate01',
                'metadata_type': 'cellvoyager',
                'cell_channel': 0, 'nucleus_channel': 1, 'pathogen_channel': 2,
                'channels': [0, 1, 2, 3], 'normalize': True,
            }
            settings, src = preprocess_img_data(settings)

    See Also:
        :func:`spacr.core.preprocess_generate_masks` — full pipeline
        wrapper that calls this then generates masks.
    """
    src = settings['src']
    
    if len(os.listdir(src)) < 100:
        delete_empty_subdirectories(src)
    
    files = os.listdir(src)
    valid_ext = ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp', 'nd2', 'czi', 'lif']
    extensions = [file.split('.')[-1].lower() for file in files]
    # Filter only valid extensions
    valid_extensions = [ext for ext in extensions if ext in valid_ext]
    # Determine most common valid extension
    img_format = None
    if valid_extensions:
        extension_counts = Counter(valid_extensions)
        most_common_extension = Counter(valid_extensions).most_common(1)[0][0]
        img_format = most_common_extension
    
        print(f"Found {extension_counts[most_common_extension]} {most_common_extension} files")
    
    else:
        print(f"Could not find any {valid_ext} files in {src}")
        print(f"{files} in {src}")
        print(f"Please check the folder and try again")
        
        if os.path.exists(os.path.join(src,'stack')):
            print('Found existing stack folder.')
        if os.path.exists(os.path.join(src,'channel_stack')):
            print('Found existing channel_stack folder.')
        if os.path.exists(os.path.join(src,'masks')):
            print('Found existing masks folder. Skipping preprocessing')
            return settings, src

    #mask_channels = [settings['nucleus_channel'], settings['cell_channel'], settings['pathogen_channel'], settings['organelle_channel']]
    
    mask_channels_raw = [settings.get('nucleus_channel'), settings.get('cell_channel'), settings.get('pathogen_channel'), settings.get('organelle_channel')]
    
    # Deduplicate while tracking positions. Coerce to int: channel indices
    # loaded from a settings CSV (or passed from the GUI) can arrive as
    # strings like '0', which then blow up array indexing downstream
    # (stack[:, :, :, '0'] -> IndexError).
    seen = {}
    mask_channels = []
    for ch in mask_channels_raw:
        if ch is None:
            continue
        try:
            ch = int(ch)
        except (TypeError, ValueError):
            continue
        if ch not in seen:
            seen[ch] = len(mask_channels)
            mask_channels.append(ch)
    
    from .settings import set_default_settings_preprocess_img_data
    from .utils import _get_regex, _run_test_mode
    from .plot import plot_arrays   # used below; import here so the plot step
                                    # doesn't raise NameError under try/except
    settings = set_default_settings_preprocess_img_data(settings)

    regex = _get_regex(settings['metadata_type'], img_format, settings['custom_regex'])
    
    if settings['test_mode']:
        print(f"Running spacr in test mode")
        settings['plot'] = True
        if os.path.exists(os.path.join(src,'test')):
            try:
                os.rmdir(os.path.join(src, 'test'))
                print(f"Deleted test directory: {os.path.join(src, 'test')}")
            except OSError as e:
                print(f"Error deleting test directory: {e}")
                print(f"Delete manually before running test mode")
                pass

        src = _run_test_mode(settings['src'], regex, settings['timelapse'], settings['test_images'], settings['random_test'])
        settings['src'] = src
    
    stack_path = os.path.join(src, 'stack')
    if img_format == None:
        if not os.path.exists(stack_path):
            _merge_channels(src, plot=False)   
   
    if not os.path.exists(stack_path):
        try:
            if not img_format == None:
                img_format = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.nd2', '.czi', '.lif']
                # Builds the stack/ arrays directly from an in-memory channel dict
                # (no per-channel sub-folders) and returns the channel count.
                nr_channel_folders = _rename_and_organize_image_files(
                    src, regex, settings['batch_size'], settings['metadata_type'], img_format,
                    timelapse=settings['timelapse'],
                    save_original_images=settings.get('save_original_images', True))

                #Make sure no batches will be of only one image
                # This counted len(stack_path) — the number of CHARACTERS in the
                # path string, which always ends in 'stack' — so the check fired
                # (or stayed silent) purely because of how long src happened to
                # be. Count the .npy stacks that concatenate_and_normalize will
                # actually batch over instead.
                all_imgs = len([f for f in os.listdir(stack_path) if f.endswith('.npy')]) if os.path.isdir(stack_path) else 0
                batch_size = int(settings.get('batch_size') or 0)
                full_batches = all_imgs // batch_size if batch_size else 0
                last_batch_size = all_imgs % batch_size if batch_size else 0

                # Report, don't raise: the stack is already written by this
                # point so aborting cannot fix the batching, it only skipped the
                # channel-count fix-up, the movies, the plot and the MIP below —
                # silently corrupting the output of an otherwise fine run.
                if last_batch_size == 1:
                    if full_batches == 0:
                        print(f"Warning: Only one batch of size 1 detected (all images: {all_imgs}). Adjust the batch size.")
                    else:
                        print(f"all images: {all_imgs},  full batch: {full_batches}, last batch: {last_batch_size}")
                        print("Warning: Last batch of size 1 detected. Adjust the batch size.")

                if len(settings['channels']) != nr_channel_folders:
                    print(f"Number of channels does not match number of channel folders. channels: {settings['channels']} channel folders: {nr_channel_folders}")
                    new_channels = list(range(nr_channel_folders))
                    print(f"Changing channels from {settings['channels']} to {new_channels}")
                    settings['channels'] = new_channels

                if settings['timelapse']:
                    _create_movies_from_npy_per_channel(stack_path, fps=settings['fps'])

                if settings['plot']:
                    print(f"plotting {settings['nr']} images from {src}/stack")
                    plot_arrays(stack_path, settings['figuresize'], settings['cmap'], nr=settings['nr'], normalize=settings['normalize'])

                if settings['all_to_mip']:
                    _mip_all(stack_path)
                    if settings['plot']:
                        print(f"plotting {settings['nr']} images from {src}/stack")
                        plot_arrays(stack_path, settings['figuresize'], settings['cmap'], nr=settings['nr'], normalize=settings['normalize'])
        
        except Exception as e:
            print(f"Error: {e}")
    
    concatenate_and_normalize(src=stack_path,
                              channels=mask_channels,
                              save_dtype=np.float32,
                              settings=settings)
        
    for key in ['nucleus_channel', 'cell_channel', 'pathogen_channel', 'organelle_channel']:
        ch = settings.get(key)
        if ch is None:
            continue
        # `seen` is keyed on int(ch) (see the dedup loop above), so looking the
        # raw value up meant a string channel index ('0') never matched and no
        # cellpose_* key was written — leaving the objects to be segmented on
        # the wrong plane, silently. Uncoercible values are dropped as before.
        try:
            ch = int(ch)
        except (TypeError, ValueError):
            continue
        if ch in seen:
            settings[f"cellpose_{key}"] = seen[ch]
            
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
    Calculate:
    - average number of objects per image
    - average object size over all objects

    Parameters:
    masks (list): A list of 2D or 3D masks with labeled objects.

    Returns:
    tuple:
        avg_num_objects_per_image (float)
        avg_object_size (float)
    """
    per_image_counts = []
    all_areas = []

    for idx, mask in enumerate(masks):
        if mask.ndim in [2, 3] and np.any(mask):
            props = measure.regionprops(mask)
            areas = [prop.area for prop in props]
            per_image_counts.append(len(areas))
            all_areas.extend(areas)
        else:
            per_image_counts.append(0)
            if not np.any(mask):
                print(f"Warning: Mask {idx} is empty.")
            elif mask.ndim not in [2, 3]:
                print(f"Warning: Mask {idx} has invalid dimension: {mask.ndim}")

    # Average number of objects per image
    if per_image_counts:
        avg_num_objects_per_image = sum(per_image_counts) / len(per_image_counts)
    else:
        avg_num_objects_per_image = 0

    # Average object size over all objects
    if all_areas:
        avg_object_size = sum(all_areas) / len(all_areas)
    else:
        avg_object_size = 0

    return avg_num_objects_per_image, avg_object_size
    
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
    
class TimelapseKeyMismatch(ValueError):
    """One side of the ``png_list`` join carries a timepoint and the other does not.

    A timepoint column is written only by a timelapse run — by
    :func:`spacr.utils.filepaths_to_database` onto ``png_list`` and by
    :func:`spacr.utils._merge_and_save_to_database` onto every object table —
    so a database where one of the two has it and the other does not was
    written by two runs that disagreed about whether the experiment was a
    timelapse. Joining them without the timepoint silently multiplies every
    object by the number of frames, which is precisely the failure this
    exception exists to stop.
    """


class JoinFanOut(ValueError):
    """A left join returned more rows than the frame it started from.

    The object tables carry one row per object per field per timepoint and
    ``png_list`` carries one crop per the same key, so the join is many-to-one
    and the row count cannot grow. If it did, the join key does not identify a
    ``png_list`` row uniquely and every downstream measurement is duplicated.
    """


class CropModeMismatch(ValueError):
    """``png_list`` holds no crops of the object this join is anchored on.

    ``measure_crop`` writes one object-id column per ``crop_mode`` --
    ``cell_id`` for ``crop_mode=['cell']``, ``nucleus_id`` for ``['nucleus']``
    and so on; the mapping is :data:`spacr.utils.PNG_OBJECT_ID_COLUMNS`.
    :func:`_read_and_join_tables` anchors on the ``cell`` table, so it needs
    ``cell_id``. A database measured only with nucleus crops does not have that
    column, and used to fail with ``KeyError: "['cell_id'] not in index"`` --
    which names neither the table, nor the column's absence, nor the setting
    that caused it.

    The cell a nucleus crop belongs to *is* in the crop's file name --
    :func:`spacr.utils._generate_names` writes
    ``<field>_<cell>_<nucleus>.png`` -- but it is not stored:
    :func:`spacr.utils.filepaths_to_database` keeps only the last token.
    Recovering it would mean a second file-name parser beside
    :mod:`spacr.schema`, so this is a refusal rather than a guess.
    """


def _report_fan_out(left, merged, join_cols, left_name='cell',
                    right_name='png_list'):
    """Raise :class:`JoinFanOut` if ``merged`` grew, naming the offending keys.

    The invariant checked is ``len(merged) == len(left)``, not
    ``len(merged) <= len(png_list)``. For a LEFT join those are different
    statements and only the first one is true of a healthy database: crops are
    routinely a strict subset of the measured objects — ``save_png`` can be off
    for some fields, a crop can fail to write, and ``png_list`` is appended per
    field so an interrupted run leaves fewer crops than cells. In all of those
    cases ``len(merged) == len(cell) > len(png_list)`` and nothing is wrong.
    What cannot happen is the join *growing* the left frame, and that is the
    exact signature of the timelapse bug (12 cell rows in, 36 out).
    """
    if len(merged) <= len(left):
        return
    raise JoinFanOut(
        f"Joining {left_name} to {right_name} on {list(join_cols)} turned "
        f"{len(left)} {left_name} rows into {len(merged)}: {right_name} holds "
        f"more than one row per {list(join_cols)}, so every measurement in the "
        f"result is duplicated. This usually means the crop step ran twice and "
        f"appended a second set of rows to {right_name}; de-duplicate that "
        f"table before reading."
    )


def _read_and_join_tables(db_path, table_names=None):
    """
    Reads and joins tables from a SQLite database.

    ``png_list`` is joined to the object tables on plate / row / column / field
    **and on the timepoint when both sides carry one**. Without the timepoint
    the join is many-to-many on a timelapse database: every frame's crop
    matches every frame's object row, so N objects x T frames came back as
    N x T x T rows with the wrong PNG attached to most of them.

    Either spelling of the timepoint is accepted on read (``timeID`` is
    canonical, ``time_id`` is what ``png_list`` was written with before the two
    were unified). :func:`spacr.utils.rename_columns_in_db` runs above and
    repairs an old database in place, but a database opened read-only, or one
    carrying both spellings, still reads correctly here.

    **The object id is migrated, not assumed.** ``png_list.cell_id`` is text
    (``'o5'``) and every object table's key is an integer, so the two are
    reconciled through :func:`spacr.utils.object_label_from_png_id` -- one
    implementation, shared with anything else that has to cross that boundary.
    The migration this replaces was ``.str[1:].astype(int)``, which died on
    four values the real writers produce every day: ``'omulti'`` and
    ``'onone'`` (a crop overlapping several cells, or none), ``'error'`` (an
    unparseable crop name) and ``NULL`` (any row belonging to a *different*
    crop mode, in a database measured with more than one). Those rows are now
    dropped from the ``png_list`` side and counted out loud: the object keeps
    its measurements and simply has no crop path, which is the same state as a
    crop that was never written.

    Args:
        db_path (str): The path to the SQLite database file.
        table_names (list, optional): The names of the tables to read and join. Defaults to ['cell', 'cytoplasm', 'nucleus', 'pathogen', 'png_list'].

    Returns:
        pandas.DataFrame: The joined DataFrame containing the data from the specified tables, or None if an error occurs.

    Raises:
        TimelapseKeyMismatch: when exactly one of ``png_list`` and ``cell``
            carries a timepoint column.
        JoinFanOut: when the join returns more rows than the object table had.
        CropModeMismatch: when ``png_list`` carries no ``cell_id`` column, i.e.
            it holds crops of some other object.
    """
    if table_names is None:
        table_names = ['cell', 'cytoplasm', 'nucleus', 'pathogen', 'png_list']
    from .utils import (PNG_CROP_MODE_BY_ID_COLUMN, PNG_OBJECT_ID_COLUMNS,
                        TIME_COLUMN_ALIASES, object_label_from_png_id,
                        rename_columns_in_db, _time_column)
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
        png_raw = dataframes['png_list']
        id_column = PNG_OBJECT_ID_COLUMNS['cell']          # 'cell_id'
        if id_column not in png_raw.columns:
            present = [c for c in png_raw.columns
                       if c in PNG_CROP_MODE_BY_ID_COLUMN]
            modes = sorted(PNG_CROP_MODE_BY_ID_COLUMN[c] for c in present)
            raise CropModeMismatch(
                f"png_list in {db_path} has no {id_column!r} column, so its "
                f"crops cannot be attached to the cell table this join is "
                f"anchored on. It holds "
                + (f"{', '.join(modes)} crops ({', '.join(present)})"
                   if present else "no object-id column at all")
                + f". Re-run the Measure module with 'cell' in crop_mode to "
                  f"write cell crops alongside the ones already there."
            )
        png_cols = [id_column, 'png_path', 'plateID', 'rowID', 'columnID',
                    'fieldID']
        png_time = _time_column(png_raw.columns)
        if png_time is not None:
            png_cols = png_cols + [png_time]
        png_list_df = png_raw[png_cols].copy()

        labels = object_label_from_png_id(png_list_df[id_column])
        usable = labels.notna()
        if not usable.all():
            # Two different reasons, reported separately because they call for
            # two different actions: NULL means "this row is another crop
            # mode's" and is expected in a multi-mode database, while a token
            # that is not a number means the crop's own name could not be read.
            raw = png_list_df[id_column]
            other_mode = int((raw.isna() & ~usable).sum())
            unreadable = raw[~usable & raw.notna()]
            if other_mode:
                print(f"png_list: {other_mode} of {len(png_list_df)} rows have "
                      f"no {id_column} and belong to another crop mode; they "
                      f"are not cell crops and take no part in this join.")
            if len(unreadable):
                sample = ', '.join(repr(v) for v in unreadable.unique()[:4])
                print(f"png_list: {len(unreadable)} row(s) carry a "
                      f"{id_column} that is not an object number ({sample}"
                      f"{' ...' if unreadable.nunique() > 4 else ''}); those "
                      f"crops cannot be matched to an object and are skipped. "
                      f"'omulti'/'onone' mean the crop overlapped several "
                      f"cells or none, 'error' means its file name could not "
                      f"be parsed.")
            png_list_df = png_list_df.loc[usable].copy()
            labels = labels.loc[usable]
        png_list_df[id_column] = labels.astype('int64')
        png_list_df.rename(columns={id_column: 'object_label'}, inplace=True)
        if 'cell' in dataframes:
            join_cols = ['object_label', 'plateID', 'rowID', 'columnID','fieldID']
            cell_time = _time_column(dataframes['cell'].columns)
            if png_time is not None and cell_time is not None:
                if png_time != cell_time:
                    # The two tables spell one concept two ways. Align the copy
                    # of png_list, never the object table: the object table's
                    # column survives into the result and renaming it there
                    # would change the schema the caller gets back.
                    png_list_df = png_list_df.rename(columns={png_time: cell_time})
                join_cols = join_cols + [cell_time]
            elif png_time is not None or cell_time is not None:
                raise TimelapseKeyMismatch(
                    f"png_list and cell disagree about the timepoint: png_list "
                    f"has {png_time!r} and cell has {cell_time!r} (of "
                    f"{list(TIME_COLUMN_ALIASES)}). One of the two was written "
                    f"by a non-timelapse run, so there is no timepoint to join "
                    f"on and joining without it would match every frame's crop "
                    f"to every frame's object. Re-run the missing step with the "
                    f"same 'timelapse' setting."
                )
            merged = pd.merge(dataframes['cell'], png_list_df, on=join_cols, how='left')
            _report_fan_out(dataframes['cell'], merged, join_cols)
            dataframes['cell'] = merged
        else:
            print("Cell table not found in database tables.")
            return png_list_df
    for entity in ['nucleus', 'pathogen']:
        if entity in dataframes:
            if 'cell_id' not in dataframes[entity].columns:
                # A child table measured with cell_mask_dim=None has no parent
                # link at all -- _merge_and_save_to_database drops 'cell_id'
                # from its key columns in exactly that case, deliberately. The
                # roll-up onto the cell is then not merely empty, it is
                # undefined, and this used to be a bare KeyError('cell_id')
                # naming neither the table nor the setting behind it.
                print(f"{entity} was measured without a cell mask, so its rows "
                      f"carry no cell_id and cannot be rolled up onto the cell "
                      f"table; {entity} features are left out of the join. "
                      f"Re-run Measure with cell_mask_dim set to link them.")
                del dataframes[entity]
                continue
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
    
#: Table holding the settings of the run that wrote the database **last**.
#: Two columns, ``setting_key`` / ``setting_value``, one row per setting.
SETTINGS_TABLE = 'settings'

#: Append-only companion to :data:`SETTINGS_TABLE`: every stage that has ever
#: written settings into this database, oldest first.
SETTINGS_HISTORY_TABLE = 'settings_history'

#: Columns of :data:`SETTINGS_HISTORY_TABLE`. ``setting_key`` /
#: ``setting_value`` come last and are spelled identically to the ``settings``
#: table, so ``SELECT setting_key, setting_value FROM settings_history`` reads
#: exactly like the table it archives.
SETTINGS_HISTORY_COLUMNS = ('run_id', 'stage', 'stamped_utc', 'setting_key',
                            'setting_value')


def _settings_history_rows(conn):
    """Read :data:`SETTINGS_HISTORY_TABLE`, or ``[]`` when there is none."""
    try:
        return conn.execute(
            f'SELECT {", ".join(SETTINGS_HISTORY_COLUMNS)} '
            f'FROM "{SETTINGS_HISTORY_TABLE}" ORDER BY rowid').fetchall()
    except sqlite3.Error:
        return []


def read_settings_history(db_path):
    """Every settings snapshot ever written into ``db_path``, oldest first.

    :param db_path: path to a ``measurements.db``.
    :returns: list of ``{'run_id', 'stage', 'stamped_utc', 'settings'}``, one
        entry per recorded run, oldest first. A database that predates the
        history table returns ``[]``.

    Example:
        .. code-block:: python

            from spacr.io import read_settings_history
            for run in read_settings_history('.../measurements/measurements.db'):
                print(run['stamped_utc'], run['stage'],
                      run['settings'].get('crop_mode'))
    """
    if not os.path.isfile(str(db_path)):
        return []
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        rows = _settings_history_rows(conn)
    finally:
        conn.close()
    runs = []
    index = {}
    for run_id, stage, stamped, key, value in rows:
        marker = (run_id, stage, stamped)
        if marker not in index:
            index[marker] = {'run_id': run_id, 'stage': stage,
                             'stamped_utc': stamped, 'settings': {}}
            runs.append(index[marker])
        index[marker]['settings'][key] = value
    return runs


def _save_settings_to_db(settings, stage=None):
    """Record this run's settings in the database it is about to write.

    Two tables, because two different questions are being asked:

    * ``settings`` — what the **most recent** run was configured with.
      Replaced, which is what :func:`spacr.resume.read_recorded_settings`
      reads and compares against before a resume; it has to be exactly one
      run's settings or that comparison means nothing.
    * ``settings_history`` — **every** run that has ever written settings
      here, appended, each tagged with a ``run_id``, a stage name and a UTC
      timestamp.

    The second is the repair. ``settings`` alone is replace-only, so a database
    written by more than one stage — or measured twice, once for cell crops and
    once for pathogen crops, both appending to the same ``png_list`` — kept the
    last stage's settings only, and every row the earlier ones wrote was left
    with no record of how it was produced. Worse, this call happens *before*
    any field is measured, so a run that recorded its settings and then died
    replaced the settings of the run that actually produced the rows on disk.
    Measured on two saves with different ``crop_mode``/``channels``: 1 of 2
    stages recoverable before, 2 of 2 after.

    A ``settings`` table written before this history existed is copied into the
    history the first time this runs, so a database already on disk keeps its
    one snapshot instead of losing it to the next run.

    :param settings: settings dict; must contain ``src``.
    :param stage: name of the pipeline stage, e.g. ``'measure_crop'``. When
        None it is taken from ``settings['stage']`` or ``settings['module']``
        if either is set, and recorded as ``'unknown'`` otherwise — ``run_id``
        and ``stamped_utc`` still keep the runs apart.
    :returns: None.
    """
    import uuid

    from .errors import _utcnow

    if stage is None:
        for key in ('stage', 'module'):
            candidate = settings.get(key)
            if isinstance(candidate, str) and candidate.strip():
                stage = candidate.strip()
                break
    stage = stage or 'unknown'
    run_id = uuid.uuid4().hex
    stamped = _utcnow()

    # Convert the settings dictionary into a DataFrame
    settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
    # Convert all values in the 'setting_value' column to strings
    settings_df['setting_value'] = settings_df['setting_value'].apply(str)
    # (No display here — save_settings already renders the settings table via
    # pretty_print_settings; displaying again produced the double print.)
    # Determine the directory path
    src = os.path.dirname(settings['src'])
    directory = f'{src}/measurements'
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Database connection and saving the settings DataFrame
    conn = sqlite3.connect(f'{directory}/measurements.db', timeout=5)
    try:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{SETTINGS_HISTORY_TABLE}" ('
            'run_id TEXT, stage TEXT, stamped_utc TEXT, '
            'setting_key TEXT, setting_value TEXT)')
        insert = (f'INSERT INTO "{SETTINGS_HISTORY_TABLE}" '
                  f'({", ".join(SETTINGS_HISTORY_COLUMNS)}) VALUES (?,?,?,?,?)')
        # Migrate what is already on disk before it is replaced. A database
        # written before the history table existed carries exactly one
        # snapshot, in `settings`; without this it would be the one run that
        # still gets forgotten.
        if not _settings_history_rows(conn):
            try:
                previous = conn.execute(
                    f'SELECT setting_key, setting_value '
                    f'FROM "{SETTINGS_TABLE}"').fetchall()
            except sqlite3.Error:
                previous = []
            if previous:
                conn.executemany(insert, [('', 'before-history', '', key, value)
                                          for key, value in previous])
        conn.executemany(insert, [(run_id, stage, stamped, key, value)
                                  for key, value
                                  in zip(settings_df['setting_key'],
                                         settings_df['setting_value'])])
        settings_df.to_sql(SETTINGS_TABLE, conn, if_exists='replace', index=False)  # Replace the table if it already exists
        conn.commit()
    finally:
        # Closed on every path: an open connection holds the lock, and this
        # runs immediately before measure_crop's workers start writing.
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
    
def save_object_mask(output_folder, filename, mask, compression='lzw'):
    """Save an integer label mask as a lossless, compressed TIFF.

    Masks are saved as TIFF (not .npy) so they're readable by ImageJ/other
    tools, with lossless compression (default LZW). Object labels are NEVER
    altered — the array is written verbatim as uint16, exactly as recorded in
    the measurements database.

    :param output_folder: destination folder (e.g. ``masks/cell_mask_stack``).
    :param filename: reference filename (the stack basename; extension ignored).
    :param mask: 2-D integer label array.
    :param compression: lossless codec — ``'lzw'`` | ``'zlib'`` | ``'none'``.
    :returns: the path written.
    """
    import tifffile
    base = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(output_folder, base + '.tif')
    comp = None if str(compression).lower() in ('none', '', 'no', 'false') else str(compression).lower()
    tifffile.imwrite(out_path, np.asarray(mask).astype(np.uint16),
                     compression=comp)
    return out_path


def _mask_variant_path(folder, ref_filename):
    """Return the path to ``ref_filename``'s array in ``folder``, preferring a
    compressed ``.tif`` mask, then legacy ``.npy``, then the exact name."""
    base = os.path.splitext(ref_filename)[0]
    for cand in (base + '.tif', base + '.tiff', base + '.npy',
                 ref_filename):
        p = os.path.join(folder, cand)
        if os.path.isfile(p):
            return p
    return None


def _save_array_atomic(output_path, array):
    """Write ``array`` to ``output_path`` as ``.npy`` atomically.

    ``np.save(path, arr)`` writes straight onto the destination, so a
    process killed part-way through — full disk, OOM killer, Ctrl-C —
    leaves a *short* file at the final name. It still starts with the
    ``.npy`` magic and still parses as a header, so anything that decides
    "this field is done because the file is there" will happily accept it
    and measure whatever bytes happened to land. That is the failure mode
    that turns a resume into silently corrupt output.

    Writing to a sibling temporary file and then ``os.replace``-ing it
    into position makes the destination atomic within the filesystem: it
    is either the previous content or the complete new array, never a
    prefix of it. The temp file is removed if anything goes wrong.

    :param output_path: final ``.npy`` path.
    :param array: array to write.
    :returns: ``output_path``.
    """
    directory = os.path.dirname(output_path) or '.'
    os.makedirs(directory, exist_ok=True)
    # Same directory as the destination: os.replace is only atomic within
    # one filesystem, and /tmp is routinely a different one.
    fd, tmp_path = tempfile.mkstemp(prefix='.spacr_tmp_', suffix='.npy',
                                    dir=directory)
    os.close(fd)
    try:
        # allow_pickle stays at numpy's default (False) — these are plain
        # numeric stacks and a pickled payload here would be a bug.
        with open(tmp_path, 'wb') as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return output_path


def _load_array_any(path):
    """Load a ``.tif``/``.tiff`` (via tifffile) or ``.npy`` array."""
    if path.endswith(('.tif', '.tiff')):
        import tifffile
        return tifffile.imread(path)
    return np.load(path, allow_pickle=True)


def _load_and_concatenate_arrays(src, channels, cell_chann_dim, nucleus_chann_dim, pathogen_chann_dim, organelle_chann_dim, resume=False):
    """
    Load and concatenate arrays from multiple folders.

    Every merged stack is written **atomically** — to a temporary file in
    the destination folder, then ``os.replace``\\ d into place. The plain
    ``np.save`` this used to do wrote straight onto the destination, so a
    run killed mid-write (full disk, OOM, Ctrl-C) left a short file that
    still looked like a valid ``.npy`` to anything that only checked
    whether it existed. ``os.replace`` is atomic within a filesystem, so
    ``merged/<field>.npy`` is now either absent or complete.

    Args:
        src (str): The source directory containing the arrays.
        channels (list): List of channel indices to select from the arrays.
        cell_chann_dim (int): Dimension of the cell channel.
        nucleus_chann_dim (int): Dimension of the nucleus channel.
        pathogen_chann_dim (int): Dimension of the pathogen channel.
        organelle_chann_dim (int or None): Dimension of the organelle channel. If None, organelle masks are included only if the folder exists.
        resume (bool): Opt-in checkpointing. When True, fields whose merged
            stack is already present **and verified complete** are skipped, so
            a run that died at field 900 of 1000 does not redo the first 900.
            Verification is deliberately not ``os.path.exists``: files left
            behind by older, non-atomic versions of this function can be
            truncated, and those are re-merged rather than trusted. Default
            False, which redoes every field exactly as before.

    Returns:
        None
    """
    from .utils import print_progress
    from .resume import completed_fields_in_merged, format_resume, plan_resume

    folder_paths = [os.path.join(src+'/stack')]

    if cell_chann_dim is not None or os.path.exists(os.path.join(src, 'masks', 'cell_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'masks','cell_mask_stack')]
    
    if nucleus_chann_dim is not None or os.path.exists(os.path.join(src, 'masks', 'nucleus_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'masks','nucleus_mask_stack')]
    
    if pathogen_chann_dim is not None or os.path.exists(os.path.join(src, 'masks', 'pathogen_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'masks','pathogen_mask_stack')]
    
    if organelle_chann_dim is not None or os.path.exists(os.path.join(src, 'masks', 'organelle_mask_stack')):
        folder_paths = folder_paths + [os.path.join(src, 'masks','organelle_mask_stack')]

    output_folder = src+'/merged'
    reference_folder = folder_paths[0]
    os.makedirs(output_folder, exist_ok=True)

    count=0
    reference_files = os.listdir(reference_folder)
    all_imgs = len(reference_files)
    time_ls = []

    # Opt-in resume: skip fields whose merged stack is already there AND
    # verified complete. Reported before any work starts, so a resume that
    # rejects three truncated leftovers says so rather than quietly
    # re-merging them.
    already_done = set()
    if resume:
        candidates = [os.path.splitext(f)[0] for f in reference_files
                      if f.endswith('.npy')]
        rejected = {}
        already_done = completed_fields_in_merged(
            output_folder, reasons=rejected, fields=candidates)
        print(format_resume(plan_resume(candidates, already_done,
                                        reasons=rejected, enabled=True,
                                        src=output_folder)))

    # Iterate through each file in the reference folder
    for idx, filename in enumerate(reference_files):
        start = time.time()
        stack_ls = []
        # `and not already done` rather than a `continue`, so a skipped field
        # still advances the progress bar instead of the counter jumping.
        if filename.endswith('.npy') and os.path.splitext(filename)[0] not in already_done:
            count += 1

            # Check if this file exists in all the other specified folders.
            # Masks may be .tif (new, compressed) or legacy .npy — resolve both.
            exists_in_all_folders = all(
                _mask_variant_path(folder, filename) is not None
                for folder in folder_paths)

            if exists_in_all_folders:
                # Load and potentially modify the array from the reference folder
                ref_array_path = os.path.join(reference_folder, filename)
                concatenated_array = np.load(ref_array_path)

                if channels is not None:
                    concatenated_array = np.take(concatenated_array, channels, axis=2)

                # Add the array from the reference folder to 'stack_ls'
                stack_ls.append(concatenated_array)

                # For each of the other folders, load the mask (tif or npy).
                for folder in folder_paths[1:]:
                    array_path = _mask_variant_path(folder, filename)
                    array = _load_array_any(array_path)
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
                    _save_array_atomic(output_path, stack)
        
        stop = time.time()
        duration = stop - start
        time_ls.append(duration)
        files_processed = idx+1
        files_to_process = all_imgs
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Merging Arrays")

    return
        
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
    """Plot training vs. validation curves from a saved model's per-epoch CSVs.

    :param train_file_path: Path to the training stats CSV.
    :param val_file_path: Path to the validation stats CSV.
    :param save: If True, write PDFs next to the training CSV instead of
        showing them. Default ``False``.
    :returns: None
    """

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

def _save_model(model, model_type, results_dict, dst, epoch, epochs,
                intermedeate_save=None,
                channels=None,
                # FIX: accept an optional validation dict for checkpoint decisions
                # WHY: the original used train_dict, so checkpoints reflected memorization
                #      not generalization — val metrics are the correct signal
                val_dict=None):
    """
    Save the model based on certain conditions during training.

    Args:
        model (torch.nn.Module): The trained model to be saved.
        model_type (str): The type of the model.
        results_df (pandas.DataFrame): The dataframe containing the validation results.
        dst (str): The destination directory to save the model.
        epoch (int): The current epoch number.
        epochs (int): The total number of epochs.
        intermedeate_save (list, optional): List of accuracy thresholds to trigger intermediate model saves. 
                                            Defaults to [0.99, 0.98, 0.95, 0.94].
        channels (list, optional): List of channels used. Defaults to ['r', 'g', 'b'].
    """
    
    if intermedeate_save is None:
        intermedeate_save = [0.99, 0.98, 0.95, 0.94]
    if channels is None:
        channels = ['r', 'g', 'b']
    channels_str = ''.join(channels)

    def save_model_at_threshold(threshold, epoch, suffix=""):
        """Persist the current model to disk when validation accuracy crosses ``threshold``."""
        percentile = str(threshold * 100)
        print(f'Found: {percentile}% accurate model')
        model_path = f'{dst}/{model_type}_epoch_{str(epoch)}{suffix}_acc_{percentile}_channels_{channels_str}.pth'
        torch.save(model, model_path)
        return model_path

    if epoch % 100 == 0 or epoch == epochs:
        model_path = f'{dst}/{model_type}_epoch_{str(epoch)}_channels_{channels_str}.pth'
        torch.save(model, model_path)
        return model_path

    # FIX: use val_dict if available, otherwise fall back to train results_dict
    # WHY: checkpointing on training accuracy lets the model overfit past the
    #      val plateau — you save a model that memorized the train set, not the
    #      one that generalizes best
    check_dict = val_dict if val_dict is not None else results_dict

    for threshold in intermedeate_save:
        # FIX: use the generic 'accuracy' key instead of 'neg_accuracy'/'pos_accuracy'
        # WHY: the original checked results_df['neg_accuracy'] and results_df['pos_accuracy']
        #      — these keys only exist for binary classification with specific class naming.
        #      For multiclass (>2 classes), these keys don't exist, so the intermediate
        #      checkpoint NEVER fires.  You only ever saved at epoch 100/200/etc or the
        #      final epoch.  Using the generic 'accuracy' key works for any number of classes.
        acc = check_dict.get('accuracy', 0.0)
        if acc >= threshold:
            print(f"Accuracy: {acc:.4f}")
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
    import gc
    import sqlite3
    import pandas as pd

    from .utils import rename_columns_in_db, correct_metadata

    def _quote_identifier(name):
        """Safely quote SQLite identifiers (e.g., table names)."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid table name: {name!r}")
        return '"' + name.replace('"', '""') + '"'

    rename_columns_in_db(db_loc)

    dfs = []
    chunksize = 100_000  # internal safety setting; adjust if needed

    with sqlite3.connect(db_loc) as conn:
        # Optional but useful: fail early if a table name is wrong
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        
        print('existing_tables:', existing_tables)
        print('tables:', tables)

        for table in tables:
            if table not in existing_tables:
                raise ValueError(f"Table not found in database: {table}")

            quoted_table = _quote_identifier(table)
            query = f"SELECT * FROM {quoted_table}"

            # Read in chunks to reduce peak memory during SQL -> pandas conversion
            chunks = []
            for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
                chunks.append(chunk)

            if len(chunks) == 0:
                # Empty table: preserve columns
                df = pd.read_sql_query(f"SELECT * FROM {quoted_table} LIMIT 0", conn)
            elif len(chunks) == 1:
                df = chunks[0]
            else:
                df = pd.concat(chunks, ignore_index=True)

            del chunks
            gc.collect()

            df = correct_metadata(df)
            dfs.append(df)

            # Drop local reference before next loop iteration
            del df
            gc.collect()

    return dfs

def _read_and_merge_data(locs, tables, verbose=False, nuclei_limit=10, pathogen_limit=10, change_plate=False):

    from .utils import MEASUREMENT_STAMP_COLUMNS, _split_data

    pathogen_counts = None
    metadata_key = 'object_label'
    shared_metadata_columns = set(MEASUREMENT_STAMP_COLUMNS)

    def _merge_grouped(left, right):
        """Merge grouped tables while keeping only one copy of shared acquisition metadata."""
        if left.empty:
            return right.copy()
        if right.empty:
            return left.copy()

        shared = [col for col in shared_metadata_columns if col in left.columns and col in right.columns]

        for col in shared:
            common_idx = left.index.intersection(right.index)
            if len(common_idx):
                a = left.loc[common_idx, col]
                b = right.loc[common_idx, col]
                mismatch = a.notna() & b.notna() & a.ne(b)
                if mismatch.any():
                    print(f"Warning: {int(mismatch.sum())} mismatched values for shared metadata column {col!r}; keeping the first.")

        right = right.drop(columns=shared)
        return left.merge(right, left_index=True, right_index=True)

    data_dict = {table: [] for table in tables}

    for idx, loc in enumerate(locs):
        db_dfs = _read_db(loc, tables)

        if change_plate:
            for df in db_dfs:
                df['plateID'] = f'plate{idx+1}'
                df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)

        for table, df in zip(tables, db_dfs):
            data_dict[table].append(df)

    for table, dfs in data_dict.items():
        if dfs:
            data_dict[table] = pd.concat(dfs, axis=0)
        if verbose:
            print(f"{table}: {len(data_dict[table])}")

    merged_df = pd.DataFrame()

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

        if 'cell' not in data_dict:
            merged_df, metadata = _split_data(cytoplasms, 'prcfo', 'object_label')

            if verbose:
                print(f'cytoplasms: {len(cytoplasms)}, cytoplasms grouped: {len(merged_df)}')

        else:
            cytoplasms_g_df, _ = _split_data(cytoplasms, 'prcfo', 'object_label')
            merged_df = _merge_grouped(merged_df, cytoplasms_g_df)

            if verbose:
                print(f'cytoplasms: {len(cytoplasms)}, cytoplasms grouped: {len(cytoplasms_g_df)}')

    if 'nucleus' in data_dict:
        nucleus = data_dict['nucleus'].copy()
        nucleus = nucleus.dropna(subset=['cell_id'])
        nucleus = nucleus.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nucleus = nucleus.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nucleus = nucleus.assign(prcfo=lambda x: x['prcf'] + '_' + x['cell_id'])
        nucleus['nucleus_prcfo_count'] = nucleus.groupby('prcfo')['prcfo'].transform('count')

        if nuclei_limit is not None:
            if nuclei_limit is True:
                nucleus = nucleus[nucleus['nucleus_prcfo_count'] == 1]
            elif isinstance(nuclei_limit, (float, int)):
                nucleus = nucleus[nucleus['nucleus_prcfo_count'] <= int(nuclei_limit)]

        if all(key not in data_dict for key in ['cell', 'cytoplasm']):
            merged_df, metadata = _split_data(nucleus, 'prcfo', 'cell_id')
            metadata_key = 'cell_id'

            if verbose:
                print(f'nucleus: {len(nucleus)}, nucleus grouped: {len(merged_df)}')

        else:
            nucleus_g_df, _ = _split_data(nucleus, 'prcfo', 'cell_id')
            merged_df = _merge_grouped(merged_df, nucleus_g_df)

            if verbose:
                print(f'nucleus: {len(nucleus)}, nucleus grouped: {len(nucleus_g_df)}')

    if 'pathogen' in data_dict:
        pathogens = data_dict['pathogen'].copy()
        pathogens = pathogens.dropna(subset=['cell_id'])
        pathogens = pathogens.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        pathogens = pathogens.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        pathogens = pathogens.assign(prcfo=lambda x: x['prcf'] + '_' + x['cell_id'])
        pathogens['pathogen_prcfo_count'] = pathogens.groupby('prcfo')['prcfo'].transform('count')

        if pathogen_limit is not None:
            if pathogen_limit is True:
                pathogens = pathogens[pathogens['pathogen_prcfo_count'] <= 1]
            elif isinstance(pathogen_limit, (float, int)):
                pathogens = pathogens[pathogens['pathogen_prcfo_count'] <= int(pathogen_limit)]

        if all(key not in data_dict for key in ['cell', 'cytoplasm', 'nucleus']):
            merged_df, metadata = _split_data(pathogens, 'prcfo', 'cell_id')
            metadata_key = 'cell_id'

            if verbose:
                print(f'pathogens: {len(pathogens)}, pathogens grouped: {len(merged_df)}')

        else:
            pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
            merged_df = _merge_grouped(merged_df, pathogens_g_df)

            if verbose:
                print(f'pathogens: {len(pathogens)}, pathogens grouped: {len(pathogens_g_df)}')

        pathogen_counts = pathogens.groupby('prcfo')['prcfo'].size().rename('pathogen_prcfo_count')

    if 'png_list' in data_dict:
        from .utils import PNG_CROP_MODE_BY_ID_COLUMN, PNG_OBJECT_ID_COLUMNS, object_label_from_png_id

        png_list = data_dict['png_list'].copy()
        id_column = PNG_OBJECT_ID_COLUMNS['cell']

        if id_column not in png_list.columns:
            present = [c for c in png_list.columns if c in PNG_CROP_MODE_BY_ID_COLUMN]
            modes = sorted(PNG_CROP_MODE_BY_ID_COLUMN[c] for c in present)
            raise CropModeMismatch(
                f"png_list has no {id_column!r} column, so its crops cannot be keyed onto the objects being merged. It holds "
                + (f"{', '.join(modes)} crops ({', '.join(present)})" if present else "no object-id column at all")
                + ". Re-run the Measure module with 'cell' in crop_mode."
            )

        keep = object_label_from_png_id(png_list[id_column]).notna()

        if not keep.all():
            print(
                f"png_list: {int((~keep).sum())} of {len(png_list)} rows are not usable cell crops "
                f"(another crop mode, or an object id that is not a number); they take no part in the merge."
            )
            png_list = png_list.loc[keep].copy()

        png_list_g_df_numeric, png_list_g_df_non_numeric = _split_data(png_list, 'prcfo', id_column)
        png_list_g_df_non_numeric.drop(
            columns=['plateID', 'rowID', 'columnID', 'fieldID', 'file_name', 'cell_id', 'prcf'],
            inplace=True,
            errors='ignore',
        )

        if verbose:
            print(f'png_list: {len(png_list)}, png_list grouped: {len(png_list_g_df_numeric)}')
            print(f"Added png_list columns: {png_list_g_df_numeric.columns}, {png_list_g_df_non_numeric.columns}")

        merged_df = _merge_grouped(merged_df, png_list_g_df_numeric)
        merged_df = _merge_grouped(merged_df, png_list_g_df_non_numeric)

    metadata = metadata.assign(prc=lambda x: x['plateID'] + '_' + x['rowID'] + '_' + x['columnID'])
    cells_well = metadata.groupby('prc')[metadata_key].nunique().reset_index(name='cells_per_well')
    metadata = metadata.merge(cells_well, on='prc')

    if 'prcf' in metadata.columns:
        metadata = metadata.assign(prcfo=lambda x: x['prcf'] + '_' + x[metadata_key])
    else:
        metadata = metadata.assign(
            prcfo=lambda x: x['plateID'] + '_' + x['rowID'] + '_' + x['columnID'] + '_' + x['fieldID'] + '_' + x[metadata_key]
        )

    metadata.set_index('prcfo', inplace=True)

    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    merged_df.drop(columns=['label_list_morphology', 'label_list_intensity'], errors='ignore', inplace=True)

    if pathogen_counts is not None:
        merged_df['pathogen_prcfo_count'] = merged_df.index.to_series().map(pathogen_counts).fillna(0).astype('Int64')

    if verbose:
        print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')

    obj_df_ls = [data_dict[table] for table in ['cell', 'cytoplasm', 'nucleus', 'pathogen'] if table in data_dict]

    return merged_df, obj_df_ls

def _read_and_merge_data_v1(locs, tables, verbose=False, nuclei_limit=10, pathogen_limit=10, change_plate=False):

    from .utils import _split_data

    # keep final integer counts per prcfo for pathogens
    pathogen_counts = None

    # Column of `metadata` that the grouping key (prcfo) was built from. The
    # child-only branches key on the PARENT cell ('cell_id'), so rebuilding
    # metadata's prcfo from 'object_label' further down made the final
    # metadata/data merge match nothing and silently return zero rows.
    metadata_key = 'object_label'

    # Initialize an empty dictionary to store DataFrames by table name
    data_dict = {table: [] for table in tables}

    # Extract plate DataFrames
    for idx, loc in enumerate(locs):
        db_dfs = _read_db(loc, tables)
        if change_plate:
            # _read_db returns a LIST of DataFrames (one per table) — it was
            # string-subscripted here, so change_plate=True always raised
            # TypeError and the feature never worked. Relabel each frame.
            for df in db_dfs:
                df['plateID'] = f'plate{idx+1}'
                df['prc'] = (
                    df['plateID'].astype(str)
                    + '_' + df['rowID'].astype(str)
                    + '_' + df['columnID'].astype(str)
                )
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

        if 'cell' not in data_dict:
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

        if nuclei_limit is not None:
            if nuclei_limit is True:
                nucleus = nucleus[nucleus['nucleus_prcfo_count'] == 1]
            elif isinstance(nuclei_limit, (float, int)):
                nucleus = nucleus[nucleus['nucleus_prcfo_count'] <= int(nuclei_limit)]

        if all(key not in data_dict for key in ['cell', 'cytoplasm']):
            merged_df, metadata = _split_data(nucleus, 'prcfo', 'cell_id')
            metadata_key = 'cell_id'

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

        if pathogen_limit is not None:
            if pathogen_limit is True:
                pathogens = pathogens[pathogens['pathogen_prcfo_count'] <= 1]
            elif isinstance(pathogen_limit, (float, int)):
                pathogens = pathogens[pathogens['pathogen_prcfo_count'] <= int(pathogen_limit)]

        if all(key not in data_dict for key in ['cell', 'cytoplasm', 'nucleus']):
            merged_df, metadata = _split_data(pathogens, 'prcfo', 'cell_id')
            metadata_key = 'cell_id'

            if verbose:
                print(f'pathogens: {len(pathogens)}, pathogens grouped: {len(merged_df)}')

        else:
            pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
            merged_df = merged_df.merge(pathogens_g_df, left_index=True, right_index=True)

            if verbose:
                print(f'pathogens: {len(pathogens)}, pathogens grouped: {len(pathogens_g_df)}')

        # ---- NEW: true integer counts per prcfo after pathogen_limit filter ----
        pathogen_counts = (
            pathogens.groupby('prcfo')['prcfo']
            .size()
            .rename('pathogen_prcfo_count')
        )
        # -----------------------------------------------------------------------

    if 'png_list' in data_dict:
        from .utils import (PNG_CROP_MODE_BY_ID_COLUMN, PNG_OBJECT_ID_COLUMNS,
                            object_label_from_png_id)

        png_list = data_dict['png_list'].copy()
        id_column = PNG_OBJECT_ID_COLUMNS['cell']          # 'cell_id'
        if id_column not in png_list.columns:
            # Same contract as _read_and_join_tables, same refusal. This used
            # to be a bare KeyError('cell_id') raised inside _split_data.
            present = [c for c in png_list.columns
                       if c in PNG_CROP_MODE_BY_ID_COLUMN]
            modes = sorted(PNG_CROP_MODE_BY_ID_COLUMN[c] for c in present)
            raise CropModeMismatch(
                f"png_list has no {id_column!r} column, so its crops cannot be "
                f"keyed onto the objects being merged. It holds "
                + (f"{', '.join(modes)} crops ({', '.join(present)})"
                   if present else "no object-id column at all")
                + ". Re-run the Measure module with 'cell' in crop_mode.")
        # Rows of another crop mode carry NULL here. _split_data rebuilds
        # prcfo as prcf + '_' + cell_id, so every one of them collapsed onto
        # one '<field>_None' key per field, was aggregated together, and then
        # missed the merge. The right answer came out for the wrong reason --
        # it depended on str(None) not colliding with a real object id -- and
        # the rows silently averaged each other on the way. Drop them on
        # purpose instead.
        keep = object_label_from_png_id(png_list[id_column]).notna()
        if not keep.all():
            print(f"png_list: {int((~keep).sum())} of {len(png_list)} rows are "
                  f"not usable cell crops (another crop mode, or an object id "
                  f"that is not a number); they take no part in the merge.")
            png_list = png_list.loc[keep].copy()
        png_list_g_df_numeric, png_list_g_df_non_numeric = _split_data(png_list, 'prcfo', id_column)
        png_list_g_df_non_numeric.drop(
            columns=['plateID', 'rowID', 'columnID', 'fieldID', 'file_name', 'cell_id', 'prcf'],
            inplace=True,
            errors='ignore',
        )
        if verbose:
            print(f'png_list: {len(png_list)}, png_list grouped: {len(png_list_g_df_numeric)}')
            print(f"Added png_list columns: {png_list_g_df_numeric.columns}, {png_list_g_df_non_numeric.columns}")
        merged_df = merged_df.merge(png_list_g_df_numeric, left_index=True, right_index=True)
        merged_df = merged_df.merge(png_list_g_df_non_numeric, left_index=True, right_index=True)

    # Add prc (plate row column) and prcfo (plate row column field object) columns
    metadata = metadata.assign(
        prc=lambda x: x['plateID'] + '_' + x['rowID'] + '_' + x['columnID']
    )
    # metadata_key, not a hard-coded 'object_label': for a child-only merge the
    # cells per well are the distinct PARENT cells, and prcfo must be rebuilt
    # from the same column the data was grouped on or the merge below is empty.
    cells_well = metadata.groupby('prc')[metadata_key].nunique().reset_index(name='cells_per_well')
    metadata = metadata.merge(cells_well, on='prc')

    if 'prcf' in metadata.columns:
        metadata = metadata.assign(prcfo=lambda x: x['prcf'] + '_' + x[metadata_key])
    else:
        metadata = metadata.assign(
            prcfo=lambda x: (
                x['plateID'] + '_' + x['rowID'] + '_' + x['columnID'] + '_' + x['fieldID'] + '_' + x[metadata_key]
            )
        )
    metadata.set_index('prcfo', inplace=True)

    # Merge metadata with final merged DataFrame
    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    merged_df.drop(columns=['label_list_morphology', 'label_list_intensity'], errors='ignore', inplace=True)

    # ---- NEW: overwrite pathogen_prcfo_count with true integer counts ---------
    if pathogen_counts is not None:
        merged_df['pathogen_prcfo_count'] = (
            merged_df.index.to_series()
            .map(pathogen_counts)
            .fillna(0)
            .astype('Int64')
        )
    # ---------------------------------------------------------------------------

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
    """Convert every ``.npy`` array in ``folder_path`` to a TIFF under ``folder_path/tiff``.

    :param folder_path: Folder containing the ``.npy`` files.
    :param limit: If set, stop after processing this many files.
    :returns: None
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
    """Split image/mask pairs in ``src`` into ``train`` and ``test`` sibling folders.

    Only images that have a corresponding mask in ``src/masks`` are
    considered.

    :param src: Folder containing images and a ``masks`` subfolder.
    :param test_split: Fraction of pairs to route into the test set.
        Default ``0.1``.
    :returns: None
    """
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
    """Group ``.fastq.gz`` files in ``folder_path`` by sample name and read direction.

    :param folder_path: Directory containing gzipped FASTQ files named
        ``<sample>_R1_...`` / ``<sample>_R2_...``.
    :returns: Mapping ``{sample_name: {"R1": path, "R2": path}}``.
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


# ===========================================================================
# On-demand crops
#
# :mod:`spacr.crops` cuts a single object straight out of ``merged/*.npy`` --
# the array already holds both the intensity planes and the integer label-mask
# planes, so the crop the PNG folder holds can be reproduced on demand,
# pixel for pixel, without the folder existing. Until now only the Qt Annotate
# screen was wired to it; the image UMAP and the Classify dataset builders
# still required a pre-generated folder, which costs disk, has to be
# regenerated whenever a crop setting changes, and goes stale silently.
#
# Everything below is the seam those consumers use. It is deliberately
# **additive**: ``crop_source='png'`` (and ``'auto'`` on any project that has
# a crop folder) behaves exactly as before, byte for byte.
#
# Two rules hold everywhere in this section:
#
#   1. A crop is only ever produced by :mod:`spacr.crops`. On-demand crops go
#      through ``CropSource.get`` -> ``crops.png_view``; crops read back off
#      disk go through ``crops.read_crop_png``. Neither path re-implements the
#      channel handling, and neither goes around the format versioning added
#      in 341f446.
#   2. Any folder of crop PNGs spaCR *writes* is stamped with the crop-format
#      sidecar before it is filled -- with the current (RGB) format when the
#      crops were cut here, and with the SOURCE folder's format when they were
#      byte-copied out of one. An unmarked folder means legacy, so leaving a
#      freshly written folder unmarked is the one mistake that silently
#      reverses everything downstream.
# ===========================================================================

#: Object types that can be cut on demand.
CROP_OBJECT_TYPES = ('cell', 'nucleus', 'pathogen', 'cytoplasm', 'organelle')

#: ``png_list`` column holding the object id (``'o<N>'``) for each crop mode,
#: as written by :func:`spacr.utils.filepaths_to_database`.
PNG_LIST_ID_COLUMNS = {
    'cell': 'cell_id', 'nucleus': 'nucleus_id', 'pathogen': 'pathogen_id',
    'cytoplasm': 'cytoplasm_id', 'organelle': 'organelle_id',
}

#: Column name used to carry a per-row crop handle through the frames in this
#: module without colliding with a measurement column.
CROP_REF_COLUMN = '_spacr_crop_ref'


def crop_object_type(png_type, default='cell'):
    """Return the object type named by a ``png_type`` / ``file_metadata`` string.

    ``'cell_png'`` -> ``'cell'``, ``'…/nucleus_png/…'`` -> ``'nucleus'``. A
    string that names no object type (a plate prefix, say) falls back to
    ``default``, because that filter is about *which rows*, not *which mask*.

    :param png_type: the setting value, or any path/substring containing it.
    :param default: object type to assume when nothing is named.
    :returns: one of :data:`CROP_OBJECT_TYPES`.
    """
    text = str(png_type or '').lower()
    for obj in CROP_OBJECT_TYPES:
        if f'{obj}_png' in text:
            return obj
    return default


#: ``measure_crop`` settings that shape a crop, and that a later run may
#: legitimately override when cutting one on demand.
CROP_SHAPE_KEYS = ('png_dims', 'png_size', 'normalize', 'normalize_by',
                   'crop_mode', 'use_bounding_box', 'dialate_pngs',
                   'dialate_png_ratios', 'cell_mask_dim', 'nucleus_mask_dim',
                   'pathogen_mask_dim', 'organelle_mask_dim')


def _crop_shape_overrides(settings):
    """Return the crop-shaping settings that may override the saved snapshot.

    Only the keys that mean the same thing to a crop as they do to
    ``measure_crop``. ``normalize`` is the trap and the reason this function
    exists: ``measure_crop`` writes a ``[p1, p2]`` percentile pair, and
    ``train_test_model`` / ``deep_spacr`` write a **bool** meaning "normalise
    the tensor". Forwarding the bool would replace the ``[1, 99]`` stretch the
    PNG folder was written with by a full 0-100 one and change every pixel,
    silently, on the one path whose entire purpose is to be pixel-identical
    to that folder.
    """
    out = {}
    for key in CROP_SHAPE_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if key == 'normalize' and not (
                value is False
                or (isinstance(value, (list, tuple)) and len(value) == 2)):
            continue
        out[key] = value
    return out


def open_crop_source(settings, src=None, object_type=None, verbose=True):
    """Return the :class:`spacr.crops.CropSource` a run should read crops from.

    Thin, non-raising wrapper over :func:`spacr.crops.resolve_crop_source`:
    it reads ``settings['crop_source']`` (``'auto'`` | ``'png'`` |
    ``'merged'``), prints which source was chosen and why, and returns None
    when neither is available -- so a caller can fall back to whatever it did
    before instead of failing on a project that predates ``merged/``.

    The run's own crop-shaping settings are forwarded (see
    :func:`_crop_shape_overrides`), which is what makes "cut fresh at the
    current crop settings" true rather than a slogan:
    ``resolve_crop_source`` starts from the ``measure_crop`` snapshot in
    ``measurements.db`` and lets those override it, so a run that asks for
    96 px crops gets 96 px crops out of ``merged/`` even though the folder on
    disk holds 48 px ones.

    :param settings: settings dict (or a source path) holding ``crop_source``.
    :param src: the experiment root; defaults to ``settings['src']`` (its
        first entry when that is a list).
    :param object_type: default object type for a merged source.
    :param verbose: print the chosen source.
    :returns: a :class:`spacr.crops.CropSource`, or None.
    """
    from . import crops

    if isinstance(settings, dict):
        request = _crop_shape_overrides(settings)
        choice = settings.get('crop_source') or 'auto'
        if src is None:
            src = settings.get('src')
    else:
        request = {}
        choice = 'auto'
        if src is None:
            src = settings
    if isinstance(src, (list, tuple)):
        src = src[0] if len(src) else None
    if not src:
        return None
    request['src'] = src
    request['crop_source'] = choice
    try:
        source = crops.resolve_crop_source(request, object_type=object_type)
    except crops.CropError as exc:
        if verbose:
            print(f"crop_source={choice!r}: {exc}")
        return None
    if verbose:
        print(f"Crop source: {source.describe()}")
    return source


class LazyCropPNG:
    """A PNG-shaped byte stream that is only produced when something opens it.

    ``spacr.utils.plot_umap_images`` and ``spacr.utils.plot_clusters_grid``
    reach for every thumbnail with ``PIL.Image.open(image_paths[i])``.
    ``Image.open`` accepts a path *or* any seekable binary stream, so an
    instance of this class can sit in that list exactly where a path string
    used to and the plotting code needs no change -- which is the point: the
    PNG list and the on-demand list have to be interchangeable, or the two
    sources are not really alternatives.

    Nothing is read until something opens the object, so building one per row
    of a large screen costs a small dict and only the handful of thumbnails
    actually drawn ever touch ``merged/``.

    The bytes are always a **current-format (RGB) crop PNG**: the array comes
    from ``CropSource.get``, which is ``crops.png_view`` for the merged source
    and ``crops.read_crop_png`` for the PNG one, and both of those return the
    corrected order. A legacy folder is therefore corrected on the way through
    here, the same way the Annotate screen corrects it.

    :param source: the :class:`spacr.crops.CropSource` to cut/read with.
    :param row: the row mapping identifying the object.
    :param name: the crop's file name, for messages and tar members.
    """

    __slots__ = ('source', 'row', 'name', '_buf')

    def __init__(self, source, row, name=''):
        """Record the source and row; produce nothing yet."""
        self.source = source
        self.row = row
        self.name = name
        self._buf = None

    # -- production --------------------------------------------------------
    def array(self):
        """Return the crop as an ``(H, W, 3)`` uint8 RGB array."""
        return self.source.get(self.row)

    def png_bytes(self):
        """Return the crop encoded as a current-format (RGB) PNG."""
        return self._stream().getvalue()

    def _raw_bytes(self):
        """Return the bytes already on disk, for a PNG source. None otherwise."""
        resolve = getattr(self.source, 'resolve', None)
        if resolve is None:
            return None
        try:
            with open(resolve(self.row), 'rb') as handle:
                return handle.read()
        except Exception:
            return None

    def _stream(self):
        """Materialise (once) and return the BytesIO holding the PNG."""
        if self._buf is None:
            buf = BytesIO()
            try:
                Image.fromarray(self.array()).save(buf, format='PNG')
            except Exception:
                # A crop PNG that spacr.crops cannot decode still has bytes on
                # disk. Hand those over rather than losing the thumbnail: this
                # is a display path, and a file that is merely unusual should
                # not take the whole figure down.
                raw = self._raw_bytes()
                if raw is None:
                    raise
                buf = BytesIO(raw)
            buf.seek(0)
            self._buf = buf
        return self._buf

    # -- the file protocol PIL needs ---------------------------------------
    def read(self, size=-1):
        """Read up to ``size`` bytes of the PNG."""
        return self._stream().read(size)

    def seek(self, offset, whence=0):
        """Seek within the PNG."""
        return self._stream().seek(offset, whence)

    def tell(self):
        """Return the current offset."""
        return self._stream().tell()

    def readable(self):
        """Return True -- the stream is readable."""
        return True

    def seekable(self):
        """Return True -- the stream is seekable."""
        return True

    def writable(self):
        """Return False -- the stream is read-only."""
        return False

    def close(self):
        """Drop the materialised bytes; a later read produces them again."""
        self._buf = None

    @property
    def closed(self):
        """Return False -- this object is never permanently closed."""
        return False

    def __enter__(self):
        """Return self, so the object can be used as a context manager."""
        return self

    def __exit__(self, *exc):
        """Release the materialised bytes."""
        self.close()
        return False

    def __repr__(self):
        """Return a short description naming the crop."""
        kind = getattr(self.source, 'kind', '?')
        return f"<LazyCropPNG {self.name or '?'} from {kind}>"


def _object_id_int(value):
    """Return the integer in a ``png_list`` object id (``'o12'`` -> ``12``).

    ``'omulti'`` / ``'onone'`` -- a crop that overlaps several objects or none
    -- have no single label to cut, and come back as None.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return None if np.isnan(value) else int(value)
    text = str(value).strip()
    if text[:1] in ('o', 'O'):
        text = text[1:]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _merged_field_paths(db_path, object_type='cell'):
    """Return ``{(plateID, rowID, columnID, fieldID): (path_name, file_name)}``.

    Read off a measurement table, which is where
    :func:`spacr.utils._merge_and_save_to_database` records the merged array
    each object came from. ``png_list`` records neither, so this is the join
    that lets a ``png_list`` row be cut on demand.

    The requested object's own table is preferred and the other object tables
    are tried in turn, because every one of them names the same field.
    """
    out = {}
    if not os.path.isfile(db_path):
        return out
    order = [object_type] + [t for t in ('cell', 'cytoplasm', 'nucleus',
                                         'pathogen', 'organelle')
                             if t != object_type]
    conn = sqlite3.connect(db_path)
    try:
        for table in order:
            try:
                rows = conn.execute(
                    f'SELECT DISTINCT plateID, rowID, columnID, fieldID, '
                    f'path_name, file_name FROM "{table}"').fetchall()
            except sqlite3.Error:
                continue
            for plate, row, col, field, path_name, file_name in rows:
                out.setdefault((plate, row, col, field), (path_name, file_name))
            if out:
                break
    finally:
        conn.close()
    return out


def crop_png_name(file_name, object_type, object_label, cell_id=None):
    """Return the file name :func:`spacr.utils._generate_names` gives this crop.

    The name matters downstream: :func:`_png_group_id` parses the plate / well
    / field out of it for group-aware cross-validation, and
    :func:`spacr.utils.process_vision_results` parses the ``prcfo`` that
    :func:`spacr.deep_spacr.merge_predictions_into_db` merges on. A crop cut
    on demand has to carry the same name as the one the PNG folder would have
    held or those two stop lining up.

    :param file_name: the merged array's stem (``plate1_A01_1``).
    :param object_type: which crop mode this is.
    :param object_label: the object's integer label.
    :param cell_id: the parent cell label, for nucleus/pathogen crops.
    :returns: the crop's file name, ending in ``.png``.
    """
    stem = os.path.splitext(os.path.basename(str(file_name)))[0]
    label = int(object_label)
    if object_type in ('nucleus', 'pathogen'):
        parent = _object_id_int(cell_id)
        parent_str = 'none' if not parent else str(parent)
        return f"{stem}_{parent_str}_{label}.png"
    return f"{stem}_{label}.png"


def crop_rows_from_png_list(db_path, png_df, object_type='cell', verbose=True):
    """Give ``png_list`` rows the keys a crop has to be cut from ``merged/``.

    ``png_list`` records where a crop was *written* and which object it came
    from (``<object>_id``), but not which merged array produced it. This joins
    the object table on plate/row/column/field to recover ``path_name``, and
    turns ``'o12'`` into ``12``.

    Rows whose object id is ``'omulti'`` / ``'onone'`` (a crop overlapping
    several objects or none) cannot be cut from a single label and are
    dropped, with a count, rather than silently producing the wrong object.

    :param db_path: the ``measurements.db`` ``png_df`` came from.
    :param png_df: the ``png_list`` frame.
    :param object_type: which crop mode the rows describe.
    :param verbose: report dropped rows.
    :returns: a copy of ``png_df`` with ``path_name``, ``object_label`` and
        ``object_type`` columns, minus the rows that cannot be cut.
    """
    df = png_df.copy()
    id_col = PNG_LIST_ID_COLUMNS.get(object_type, 'cell_id')
    if id_col not in df.columns:
        # A png_list written for one crop mode carries only that mode's id
        # column; fall back to whichever object column it does have.
        for candidate in PNG_LIST_ID_COLUMNS.values():
            if candidate in df.columns:
                id_col = candidate
                break
    if id_col in df.columns:
        labels = df[id_col].map(_object_id_int)
    elif 'object_label' in df.columns:
        # Not a png_list at all: a frame that already came off the object
        # table (crop_rows_from_object_table) carries the integer label
        # directly. Looking up a column that is not there would drop every row.
        labels = df['object_label'].map(_object_id_int)
    else:
        labels = pd.Series([None] * len(df), index=df.index)

    key_cols = ['plateID', 'rowID', 'columnID', 'fieldID']
    if 'path_name' in df.columns and df['path_name'].notna().any():
        pass                    # the frame already names its merged array
    elif all(c in df.columns for c in key_cols):
        # png_list records where a crop was written, never which merged array
        # produced it; the object table is the only place that link exists.
        fields = _merged_field_paths(db_path, object_type)
        keys = list(zip(*(df[c] for c in key_cols)))
        df['path_name'] = [fields.get(k, (None, None))[0] for k in keys]
    else:
        df['path_name'] = None
    df['object_label'] = labels
    df['object_type'] = object_type

    usable = df['object_label'].notna() & df['path_name'].notna()
    dropped = int((~usable).sum())
    if dropped and verbose:
        print(f"crop_rows_from_png_list: {dropped} of {len(df)} png_list rows "
              f"cannot be cut from merged/ (no single object label, or no "
              f"matching row in the '{object_type}' table); they are skipped.")
    return df[usable].copy()


def crop_rows_from_object_table(db_path, object_type='cell', verbose=True):
    """Return one crop row per object, straight off the measurement table.

    This is the path for a project that never wrote a PNG folder at all, so
    there is no ``png_list`` to start from: ``object_label``, ``path_name``
    and the well keys are already on every measurement row.

    :param db_path: the ``measurements.db``.
    :param object_type: which object table to read.
    :param verbose: print what was found.
    :returns: a DataFrame with ``path_name``, ``object_label``, the well keys,
        ``png_name`` and ``png_path`` (the path the crop *would* have had).
    """
    if not os.path.isfile(db_path):
        return pd.DataFrame()
    select = ('object_label, plateID, rowID, columnID, fieldID, prcf, '
              'file_name, path_name')
    conn = sqlite3.connect(db_path)
    try:
        try:
            df = pd.read_sql(f'SELECT {select} FROM "{object_type}"', conn)
        except Exception:
            try:
                df = pd.read_sql(
                    f'SELECT object_label, plateID, rowID, columnID, fieldID, '
                    f'file_name, path_name FROM "{object_type}"', conn)
            except Exception:
                if verbose:
                    print(f"crop_rows_from_object_table: no '{object_type}' "
                          f"table in {db_path}")
                return pd.DataFrame()
        parents = {}
        if object_type in ('nucleus', 'pathogen'):
            try:
                link = pd.read_sql(
                    f'SELECT object_label, prcf, cell_id FROM "{object_type}"',
                    conn)
                parents = {(r.prcf, r.object_label): r.cell_id
                           for r in link.itertuples()}
            except Exception:
                parents = {}
    finally:
        conn.close()
    if df.empty:
        return df
    df['object_type'] = object_type
    df['png_name'] = [
        crop_png_name(row.file_name, object_type, row.object_label,
                      parents.get((getattr(row, 'prcf', None), row.object_label)))
        for row in df.itertuples()
    ]
    df['png_path'] = [
        os.path.join(str(plate) + '_' + str(well), f'{object_type}_png', name)
        for plate, well, name in zip(df['plateID'], df['rowID'], df['png_name'])
    ]
    if verbose:
        print(f"crop_rows_from_object_table: {len(df)} '{object_type}' objects "
              f"in {db_path}")
    return df


def crop_refs_for_rows(source, df, object_type='cell', name_column=None):
    """Return one :class:`LazyCropPNG` per row of ``df``.

    :param source: the crop source to cut/read with.
    :param df: rows carrying whatever the source needs (``png_path`` for the
        PNG source, ``path_name`` + ``object_label`` for the merged one).
    :param object_type: object type stamped onto each row.
    :param name_column: column holding the crop's file name; defaults to
        the basename of ``png_path``.
    :returns: list of :class:`LazyCropPNG`.
    """
    n = len(df)

    def _col(name):
        # Columns are pulled out as plain lists rather than walked with
        # itertuples(): the joined UMAP frame carries a couple of hundred
        # columns, and building a namedtuple per row of it costs more than
        # the crops themselves do -- and itertuples silently renames any
        # column whose name is not a valid identifier.
        if name and name in df.columns:
            return df[name].tolist()
        return [None] * n

    def _missing(value):
        return value is None or (isinstance(value, float) and np.isnan(value))

    png_paths = _col('png_path')
    path_names = _col('path_name')
    labels = _col('object_label')
    names = _col(name_column)

    refs = []
    for i in range(n):
        entry = {'object_type': object_type}
        if not _missing(png_paths[i]):
            entry['png_path'] = png_paths[i]
        if not _missing(path_names[i]):
            entry['path_name'] = path_names[i]
        if not _missing(labels[i]):
            entry['object_label'] = int(labels[i])
        if not _missing(names[i]):
            name = str(names[i])
        elif not _missing(png_paths[i]):
            name = os.path.basename(str(png_paths[i]))
        else:
            name = ''
        refs.append(LazyCropPNG(source, entry, name=name))
    return refs


def mark_crop_output_folder(folder, fmt=None, source_folder=None,
                            db_path=None, **extra):
    """Stamp a folder spaCR has just filled with crop PNGs.

    Called *before* the folder is filled, exactly as
    :func:`spacr.crops.stamp_crop_folder` is on the measure path, so an
    interrupted run leaves a marked folder holding fewer crops rather than an
    unmarked folder of corrected ones -- the one state that is silently
    misread.

    ``fmt=None`` inherits the format from ``source_folder``. That is what
    keeps a byte-for-byte copy honestly labelled: copying legacy crops into a
    training folder produces legacy crops, and marking that folder as current
    would reverse every channel name attached to the model trained on it.

    :param folder: the folder about to be filled.
    :param fmt: the format to record; None inherits from ``source_folder``.
    :param source_folder: the folder the crops are being copied from.
    :param db_path: ``measurements.db`` consulted when ``source_folder``
        carries no sidecar.
    :param extra: extra keys recorded in the sidecar.
    :returns: the sidecar path, or None when it could not be written.
    """
    from . import crops

    if fmt is None:
        if source_folder:
            try:
                fmt = crops.crop_folder_format(source_folder, db_path=db_path)
            except Exception:
                fmt = crops.CROP_FORMAT_LEGACY_BGR
        else:
            fmt = crops.CROP_FORMAT_CURRENT
    try:
        return crops.write_crop_folder_marker(folder, fmt=int(fmt), **extra)
    except Exception as exc:
        # Loud, never silent: the consequence of a missing marker is that the
        # crops read back reversed. But failing a whole training run over a
        # 300-byte sidecar helps nobody.
        print(f"Warning: could not stamp the crop format on {folder}: {exc}")
        return None


def generate_dataset(settings=None):
    """Pack per-object PNGs referenced by one or more ``measurements.db`` files into a single tar for inference or upload.

    Selects PNG paths (via the ``png_list`` table plus optional
    ``file_metadata`` filter) from each source's measurements database,
    optionally random-subsamples, then bundles the images in parallel
    into a dated tar under the first source's ``datasets/`` folder.
    Use this to produce the ``tar_path`` consumed by
    :func:`spacr.deep_spacr.deep_spacr` / ``apply_model_to_tar``.

    ``crop_source`` chooses where the images come from. ``'png'`` (and
    ``'auto'`` wherever a crop folder exists) is the behaviour above,
    unchanged: the files are byte-copied into the tar. ``'merged'`` (and
    ``'auto'`` on a project with no crop folder) cuts every crop out of
    ``merged/*.npy`` through :mod:`spacr.crops` instead, so the tar can be
    built with no PNG folder on disk at all, and is rebuilt at the *current*
    crop settings rather than whatever the folder was generated with. The
    members are named exactly as the PNG folder would have named them, so
    everything that parses a crop file name downstream -- fold grouping,
    ``prcfo``, the prediction merge -- keeps working either way.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.set_generate_dataset_defaults`. Key
        entries:

        - ``src`` (str or list of str) — folder(s) containing
          ``measurements/measurements.db`` and the PNG crops.
        - ``file_metadata`` — filter/join key applied against
          ``png_list``.
        - ``sample`` — ``int`` or ``[int]`` cap on selected PNGs
          (random subsample); omit for all.
        - ``experiment`` — string suffix used in the tar filename.
        - ``crop_source`` — ``'auto'`` | ``'png'`` | ``'merged'``.

    :returns: Absolute path to the created ``…/datasets/<date>_<
        experiment>.tar``.
    :raises RuntimeError: if ``src`` is not a string / list of strings,
        no images are selected, no image could be written, or the
        destination folder cannot be resolved.

    Example:
        .. code-block:: python

            from spacr.io import generate_dataset
            tar_path = generate_dataset({
                'src': ['/data/plate01', '/data/plate02'],
                'experiment': 'screen_v1',
                'sample': 100000,
            })

    See Also:
        :func:`training_dataset_from_annotation` — build a labeled
        ``train/`` / ``test/`` tree instead of a flat tar.
        :func:`spacr.deep_spacr.deep_spacr` — consumes the tar via
        ``apply_model_to_tar``.
    """
    if settings is None:
        settings = {}
    import os, tarfile, shutil, random, datetime
    from multiprocessing import Pool, Value, Lock, cpu_count

    from .utils import (
        initiate_counter, add_images_to_tar, save_settings,
        generate_path_list_from_db, correct_paths
    )
    from .settings import set_generate_dataset_defaults

    settings = set_generate_dataset_defaults(settings)
    save_settings(settings, 'generate_dataset', show=True)

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    object_type = crop_object_type(
        settings.get('file_metadata') or settings.get('png_type'))

    if isinstance(settings['src'], list):
        all_paths = []
        n_on_demand = 0
        dst = None
        for i, src in enumerate(settings['src']):
            db_path = os.path.join(src, 'measurements', 'measurements.db')
            if i == 0:
                dst = os.path.join(src, 'datasets')
            source = open_crop_source(settings, src, object_type=object_type)
            if source is not None and getattr(source, 'kind', 'png') == 'merged':
                refs = _dataset_crop_refs(db_path, source, settings, object_type)
                n_on_demand += len(refs)
                all_paths.extend(refs)
                continue
            paths = generate_path_list_from_db(db_path, file_metadata=settings['file_metadata'])
            # generate_path_list_from_db returns None when the query fails
            # (a database with no png_list, say). correct_paths then died on
            # an unbound local three frames away instead of saying so.
            if not paths:
                print(f"No png_list rows selected from {db_path}.")
                continue
            paths = correct_paths(paths, src)  # <- capture corrected paths
            all_paths.extend(paths)

        # --- sampling (guard against k > N) ---
        if isinstance(settings['sample'], int) and settings['sample']:
            k = min(int(settings['sample']), len(all_paths))
            selected_paths = random.sample(all_paths, k) if k else []
            print(f"Random selection of {len(selected_paths)} paths")
        elif isinstance(settings['sample'], list) and settings['sample']:
            k = min(int(settings['sample'][0]), len(all_paths))
            selected_paths = random.sample(all_paths, k) if k else []
            print(f"Random selection of {len(selected_paths)} paths")
        else:
            selected_paths = list(all_paths)
            random.shuffle(selected_paths)
            print(f"All paths: {len(selected_paths)} paths")
    else:
        raise RuntimeError("settings['src'] must be a string or list of strings.")

    total_images = len(selected_paths)
    print(f"Found {total_images} images")
    if total_images == 0:
        raise RuntimeError("No images selected; nothing to tar.")

    # ensure destination exists
    if dst is None:
        raise RuntimeError("Destination folder (dst) was not set.")
    os.makedirs(dst, exist_ok=True)

    # Combine the temporary tar files into a final tar
    date_name = datetime.date.today().strftime('%y%m%d')
    if len(settings['src']) > 1:
        date_name = f"{date_name}_combined"

    tar_name = f"{date_name}_{settings['experiment']}.tar"
    tar_name = os.path.join(dst, tar_name)
    if os.path.exists(tar_name):
        number = random.randint(1, 100)
        tar_name_2 = f"{date_name}_{settings['experiment']}_{settings['file_metadata']}_{number}.tar"
        print(f"Warning: {os.path.basename(tar_name)} exists, saving as {os.path.basename(tar_name_2)} ")
        tar_name = os.path.join(dst, tar_name_2)

    if n_on_demand:
        # On-demand crops are cut here, in this process: a CropSource holds
        # memory-mapped merged arrays and a per-field label index, and neither
        # survives a fork usefully -- every worker would re-open and re-index
        # every field it touched. The reads are the cost either way, so the
        # pool buys nothing and the bookkeeping is simpler without it.
        written, skipped = _write_crop_tar(selected_paths, tar_name, settings)
        if written == 0:
            raise RuntimeError(
                f"No image could be written to {tar_name}: all "
                f"{total_images} selected crops failed. Check that "
                f"merged/*.npy is where measurements.db says it is.")
        if skipped:
            print(f"Warning: {skipped} of {total_images} crops could not be "
                  f"produced and are NOT in the tar.")
        print(f"\nSaved {written} images to {tar_name}")
        return tar_name

    # Create a temp folder in dst
    temp_dir = os.path.join(dst, "temp_tars")
    os.makedirs(temp_dir, exist_ok=True)

    # Chunking the data
    # cap workers by total images so we don't spawn useless pools
    num_procs = max(1, min(max(2, cpu_count() - 2), total_images))
    chunk_size = total_images // num_procs
    remainder = total_images % num_procs

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
        pool.starmap(
            add_images_to_tar,
            [(paths_chunks[i], temp_tar_files[i], total_images) for i in range(num_procs)]
        )

    print(f"Merging temporary files")

    written = 0
    with tarfile.open(tar_name, 'w') as final_tar:
        for temp_tar_path in temp_tar_files:
            with tarfile.open(temp_tar_path, 'r') as temp_tar:
                for member in temp_tar.getmembers():
                    if member.isfile():
                        file_obj = temp_tar.extractfile(member)
                        final_tar.addfile(member, file_obj)
                        written += 1
            os.remove(temp_tar_path)

    # Delete the temp folder
    shutil.rmtree(temp_dir)
    # `written`, not `total_images`: add_images_to_tar swallows a missing file
    # with a print, so a tar built against a crop folder that has been deleted
    # or moved used to be announced as "Saved 48 images" while holding none,
    # and the run only failed later, inside inference, on an empty dataset.
    if written == 0:
        raise RuntimeError(
            f"No image could be written to {tar_name}: none of the "
            f"{total_images} selected PNG paths exist on disk. The crop "
            f"folder has been deleted or moved -- set crop_source='merged' "
            f"to cut the crops out of merged/*.npy instead.")
    if written < total_images:
        print(f"Warning: {total_images - written} of {total_images} selected "
              f"PNGs were missing and are NOT in the tar.")
    print(f"\nSaved {written} images to {tar_name}")

    return tar_name


def _dataset_crop_refs(db_path, source, settings, object_type, verbose=True):
    """Return the on-demand crops one source folder contributes to a dataset tar.

    Prefers ``png_list`` when there is one, so the tar holds exactly the crops
    the PNG path would have held, under exactly the same names and the same
    ``file_metadata`` filter. Falls back to the object measurement table for a
    project that never wrote a PNG folder at all.

    :param db_path: the source's ``measurements/measurements.db``.
    :param source: the merged :class:`spacr.crops.CropSource`.
    :param settings: the ``generate_dataset`` settings.
    :param object_type: which crop mode to cut.
    :param verbose: print what was selected.
    :returns: list of :class:`LazyCropPNG`.
    """
    file_metadata = settings.get('file_metadata')
    png_df = None
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        try:
            png_df = pd.read_sql('SELECT * FROM png_list', conn)
        except Exception:
            png_df = None
        finally:
            conn.close()

    def _filter(frame, column):
        if not file_metadata or column not in frame.columns:
            return frame
        terms = file_metadata if isinstance(file_metadata, (list, tuple)) else [file_metadata]
        text = frame[column].astype(str)
        mask = np.zeros(len(frame), dtype=bool)
        for term in terms:
            mask |= text.str.contains(str(term), regex=False, na=False).to_numpy()
        return frame[mask]

    if png_df is not None and len(png_df):
        png_df = _filter(png_df, 'png_path')
        rows = crop_rows_from_png_list(db_path, png_df, object_type,
                                       verbose=verbose)
        return crop_refs_for_rows(source, rows, object_type)

    rows = crop_rows_from_object_table(db_path, object_type, verbose=verbose)
    if len(rows):
        rows = _filter(rows, 'png_path')
    return crop_refs_for_rows(source, rows, object_type,
                              name_column='png_name')


def _write_crop_tar(items, tar_name, settings=None):
    """Write ``items`` into ``tar_name``, cutting on-demand crops as it goes.

    ``items`` may mix plain PNG paths (byte-copied, exactly as the parallel
    path does) and :class:`LazyCropPNG` handles (cut out of ``merged/*.npy``
    and stored as current-format RGB PNGs).

    The archive also carries a ``.spacr_crop_format.json`` member, the same
    marker :mod:`spacr.crops` writes into a crop folder, so "which channel
    order is this tar in?" is answerable from the tar alone.
    :class:`TarImageDataset` skips it and reports it as ``crop_format``.

    :param items: paths and/or :class:`LazyCropPNG` handles.
    :param tar_name: destination archive.
    :param settings: optional settings, recorded in the marker.
    :returns: ``(written, skipped)`` counts.
    """
    from . import crops
    from .utils import print_progress

    total = len(items)
    written = 0
    skipped = 0
    used = set()
    with tarfile.open(tar_name, 'w') as tar:
        marker = json.dumps({
            'spacr_crop_format': crops.CROP_FORMAT_CURRENT,
            'channel_order': 'rgb',
            'narrowing': 'high-byte',
            'note': ('Cut on demand from merged/*.npy by spacr.io.'
                     'generate_dataset; png_dims[0] is each member\'s red '
                     'channel.'),
            'png_dims': list((settings or {}).get('png_dims') or []),
        }, indent=2, sort_keys=True).encode('utf-8')
        info = tarfile.TarInfo(crops.CROP_FORMAT_SIDECAR)
        info.size = len(marker)
        tar.addfile(info, BytesIO(marker))

        for i, item in enumerate(items):
            try:
                if isinstance(item, LazyCropPNG):
                    payload = item.png_bytes()
                    name = item.name or f"crop_{i}.png"
                else:
                    name = os.path.basename(str(item))
                    with open(str(item), 'rb') as handle:
                        payload = handle.read()
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"Could not read crop {item!r}: {exc}")
                continue
            # Two crops sharing a basename would overwrite each other inside
            # the archive, which is exactly the collision spacr.predictions
            # documents. Make the second one distinct instead of losing it.
            if name in used:
                stem, ext = os.path.splitext(name)
                name = f"{stem}__{i}{ext}"
            used.add(name)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, BytesIO(payload))
            written += 1
            if written % 100 == 0 or written == total:
                print_progress(written, total, n_jobs=1, time_ls=None,
                               batch_size=None,
                               operation_type="generating .tar dataset")
    return written, skipped

# ---------------------------------------------------------------------------
# Class-imbalance handling and cross-validation splitting
#
# Both live here because both change *how the training data is split and
# weighted* — the one place that decision is made is generate_loaders.
#
# Two rules are load-bearing and are enforced by tests:
#   1. A WeightedRandomSampler is only ever attached to the TRAIN loader.
#      Resampling validation or test data changes the class prior the metrics
#      are measured against, so a "balanced" accuracy would no longer describe
#      the real screen.
#   2. Cross-validation folds are group-aware by default. Crops taken from the
#      same well (or field, or plate) share illumination, focus, seeding
#      density and edge effects; splitting them across folds lets the model
#      recognise the well rather than the phenotype and inflates every score.
# ---------------------------------------------------------------------------

#: Accepted values for the ``class_balance`` setting.
CLASS_BALANCE_MODES = ('none', 'weighted_sampler', 'sqrt_weighted_sampler', 'weighted_loss')

#: Accepted values for the ``cv_group_by`` setting.
CV_GROUP_LEVELS = ('none', 'field', 'well', 'plate')

#: max/min class-count ratio at or above which the data is called skewed.
IMBALANCE_RATIO_WARN = 1.5

#: max/min class-count ratio at or above which the skew is called severe.
IMBALANCE_RATIO_SEVERE = 10.0


def _png_group_id(path, level):
    """Return the plate / well / field group id encoded in a spacr crop filename.

    spacr object crops are named ``<plate>_<well>_<field>_..._<object>.png``
    (see ``spacr.utils._generate_names``), so the grouping key is a prefix of
    the underscore-separated basename.

    :param path: image path or bare filename.
    :param level: ``'plate'``, ``'well'`` or ``'field'``.
    :returns: group id string, or None when the name has too few parts to
        carry the requested level.
    :raises ValueError: if ``level`` is not a supported grouping level.
    """
    if level not in ('plate', 'well', 'field'):
        raise ValueError(
            f"group level {level!r} is not one of {('plate', 'well', 'field')}")
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    parts = stem.split('_')
    n_needed = {'plate': 1, 'well': 2, 'field': 3}[level]
    if len(parts) < n_needed or any(p == '' for p in parts[:n_needed]):
        return None
    return '_'.join(parts[:n_needed])


def dataset_labels(dataset):
    """Return the integer class label of every sample in ``dataset``.

    Handles the three shapes that flow through the training path: a
    ``spacrDataset`` (labels are already a list), a ``torch.utils.data.Subset``
    of one (produced by ``random_split`` and by the fold splitter), and the
    plain list of ``(image, label, filename)`` tuples that ``augment_dataset``
    returns. Only the last shape has to be walked, and it holds tensors in
    memory already, so nothing here decodes an image.

    :param dataset: dataset, Subset, or sequence of ``(img, label, name)``.
    :returns: list of int labels, positionally aligned with the dataset.
    """
    if isinstance(dataset, Subset):
        parent = dataset_labels(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    labels = getattr(dataset, 'labels', None)
    if labels is not None:
        return [int(v) for v in labels]
    return [int(item[1]) for item in dataset]


def dataset_filenames(dataset):
    """Return the source filename of every sample in ``dataset``.

    Mirrors :func:`dataset_labels` so group ids can be derived without
    touching pixels.

    :param dataset: dataset, Subset, or sequence of ``(img, label, name)``.
    :returns: list of filename strings, positionally aligned with the dataset.
    """
    if isinstance(dataset, Subset):
        parent = dataset_filenames(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    names = getattr(dataset, 'filenames', None)
    if names is not None:
        return [str(v) for v in names]
    return [str(item[2]) for item in dataset]


def summarize_class_imbalance(labels, classes=None):
    """Measure the class skew of a label vector.

    :param labels: iterable of integer class labels.
    :param classes: ordered class names; index i names label i. Defaults to
        ``['class_0', ...]`` sized to the largest label seen.
    :returns: dict with ``counts``, ``fractions``, ``imbalance_ratio``
        (majority/minority, ``inf`` when a class is empty), ``minority``,
        ``majority``, ``empty_classes``, ``skewed`` and ``severe``.
    """
    labels = [int(v) for v in labels]
    if classes is None:
        n_classes = (max(labels) + 1) if labels else 0
        classes = [f'class_{i}' for i in range(n_classes)]
    classes = list(classes)
    counts = [0] * len(classes)
    unknown = 0
    for v in labels:
        if 0 <= v < len(counts):
            counts[v] += 1
        else:
            unknown += 1

    total = sum(counts)
    fractions = [(c / total) if total else 0.0 for c in counts]
    hi = max(counts) if counts else 0
    lo = min(counts) if counts else 0
    if lo > 0:
        ratio = hi / lo
    elif hi > 0:
        ratio = float('inf')
    else:
        ratio = 1.0

    empty = [classes[i] for i, c in enumerate(counts) if c == 0]
    return {
        'classes': classes,
        'counts': counts,
        'fractions': fractions,
        'n': total,
        'unknown_labels': unknown,
        'imbalance_ratio': ratio,
        'majority': classes[counts.index(hi)] if counts else None,
        'minority': classes[counts.index(lo)] if counts else None,
        'minority_fraction': (lo / total) if total else 0.0,
        'empty_classes': empty,
        'skewed': bool(counts) and ratio >= IMBALANCE_RATIO_WARN,
        'severe': bool(counts) and ratio >= IMBALANCE_RATIO_SEVERE,
    }


def class_sampling_weights(counts, mode):
    """Per-class sampling weight for a ``WeightedRandomSampler``.

    ``'weighted_sampler'`` uses ``1/n_c``, which makes every class equally
    likely to be drawn. ``'sqrt_weighted_sampler'`` uses ``1/sqrt(n_c)``, a
    partial correction that moves the realised frequencies toward balance
    without oversampling a tiny class so hard that the model memorises its
    handful of crops.

    :param counts: per-class sample counts.
    :param mode: ``'weighted_sampler'`` or ``'sqrt_weighted_sampler'``.
    :returns: list of per-class weights, scaled so they sum to 1.
    :raises ValueError: if ``mode`` does not describe a sampler.
    """
    if mode == 'weighted_sampler':
        power = 1.0
    elif mode == 'sqrt_weighted_sampler':
        power = 0.5
    else:
        raise ValueError(
            f"class_balance mode {mode!r} does not build a sampler; "
            f"expected 'weighted_sampler' or 'sqrt_weighted_sampler'")
    raw = [(1.0 / (float(c) ** power)) if c > 0 else 0.0 for c in counts]
    total = sum(raw)
    return [w / total for w in raw] if total > 0 else raw


def expected_sampled_fractions(counts, mode):
    """Class frequencies the loader is expected to realise under ``mode``.

    This is what makes the effect visible before a single epoch runs: the
    report prints the observed fractions next to these.

    :param counts: per-class sample counts.
    :param mode: any value of ``CLASS_BALANCE_MODES``.
    :returns: list of expected per-class draw probabilities.
    """
    total = sum(counts)
    if mode not in ('weighted_sampler', 'sqrt_weighted_sampler'):
        return [(c / total) if total else 0.0 for c in counts]
    per_class = class_sampling_weights(counts, mode)
    # every sample of class c carries weight per_class[c]; class c therefore
    # attracts n_c * per_class[c] of the total probability mass.
    mass = [n * w for n, w in zip(counts, per_class)]
    s = sum(mass)
    return [m / s for m in mass] if s > 0 else mass


def make_class_balance_sampler(labels, mode, num_samples=None, generator=None):
    """Build the ``WeightedRandomSampler`` for a class-balance mode.

    :param labels: integer labels of the split being sampled.
    :param mode: any value of ``CLASS_BALANCE_MODES``; the non-sampler modes
        return ``(None, None)``.
    :param num_samples: draws per epoch. Defaults to ``len(labels)`` so the
        epoch keeps its usual length.
    :param generator: optional ``torch.Generator`` for reproducible draws.
    :returns: ``(sampler, per_sample_weights)``, or ``(None, None)``.
    :raises ValueError: if ``mode`` is not a recognised class-balance mode.
    """
    if mode not in CLASS_BALANCE_MODES:
        raise ValueError(
            f"class_balance {mode!r} is not one of {CLASS_BALANCE_MODES}")
    if mode in ('none', 'weighted_loss'):
        return None, None
    labels = [int(v) for v in labels]
    if not labels:
        return None, None
    n_classes = max(labels) + 1
    counts = [0] * n_classes
    for v in labels:
        counts[v] += 1
    per_class = class_sampling_weights(counts, mode)
    weights = torch.as_tensor([per_class[v] for v in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples) if num_samples is not None else len(labels),
        replacement=True,
        generator=generator,
    )
    return sampler, weights


def format_class_balance_report(summary, class_balance='none', split_name='train'):
    """Render the human-readable skew report printed on every training run.

    :param summary: dict from :func:`summarize_class_imbalance`.
    :param class_balance: the mode that was requested.
    :param split_name: which split is being described, e.g. ``'train'``.
    :returns: multi-line report string.
    """
    counts = summary['counts']
    ratio = summary['imbalance_ratio']
    ratio_txt = 'inf' if ratio == float('inf') else f"{ratio:.2f}"
    lines = [f"--- Class balance ({split_name}, n={summary['n']}) ---"]
    # Only the train split is ever resampled, so only it can show moved
    # frequencies; showing them for validation/test would be a lie.
    effective = class_balance if split_name == 'train' else 'none'
    expected = expected_sampled_fractions(counts, effective)
    for name, count, frac, exp in zip(summary['classes'], counts,
                                      summary['fractions'], expected):
        line = f"  {name}: {count} ({frac * 100:.1f}%)"
        if abs(exp - frac) > 1e-9:
            line += f" -> sampled at ~{exp * 100:.1f}%"
        lines.append(line)
    lines.append(f"  imbalance ratio (majority/minority): {ratio_txt}"
                 f"  [majority={summary['majority']}, minority={summary['minority']}]")
    if summary['empty_classes']:
        lines.append(f"  WARNING: classes with no {split_name} samples: "
                     f"{summary['empty_classes']}")
    lines.append(f"  action: {summary['action']}")
    if summary.get('recommendation'):
        lines.append(f"  recommendation: {summary['recommendation']}")
    return '\n'.join(lines)


def report_class_balance(labels, classes=None, class_balance='none',
                         split_name='train', verbose=True):
    """Measure class skew, decide what was done about it, and say so out loud.

    A silent auto-fix is worse than none: the printed report always names the
    per-class counts, the imbalance ratio and the concrete action taken, and
    when ``class_balance='none'`` on skewed data it names the modes that would
    have helped instead of quietly doing nothing.

    :param labels: integer labels of the split.
    :param classes: ordered class names.
    :param class_balance: requested mode, one of ``CLASS_BALANCE_MODES``.
    :param split_name: split being described (``'train'``, ``'validation'``, ``'test'``).
    :param verbose: print the report. The dict is returned either way.
    :returns: the summary dict, extended with ``mode``, ``action``,
        ``recommendation`` and ``report``.
    :raises ValueError: if ``class_balance`` is not a recognised mode.
    """
    if class_balance not in CLASS_BALANCE_MODES:
        raise ValueError(
            f"class_balance {class_balance!r} is not one of {CLASS_BALANCE_MODES}")

    summary = summarize_class_imbalance(labels, classes=classes)
    summary['mode'] = class_balance
    summary['split'] = split_name
    recommendation = ''

    if split_name != 'train':
        summary['action'] = (f"none - {split_name} data is never resampled or "
                             f"reweighted, so its metrics keep the real class prior")
    elif class_balance == 'weighted_sampler':
        summary['action'] = ("WeightedRandomSampler on the train loader only "
                             "(per-class weight 1/n, draws ~uniform across classes)")
    elif class_balance == 'sqrt_weighted_sampler':
        summary['action'] = ("WeightedRandomSampler on the train loader only "
                             "(per-class weight 1/sqrt(n), partial correction)")
    elif class_balance == 'weighted_loss':
        summary['action'] = ("loss reweighting - loss_type switched to "
                             "'ce_weighted' (inverse-frequency class weights); "
                             "sampling is unchanged")
    elif summary['severe']:
        summary['action'] = 'none (no rebalancing applied)'
        recommendation = (
            "severe skew - set class_balance='weighted_sampler' to draw classes "
            "uniformly, or 'weighted_loss' to reweight cross-entropy instead; "
            "'sqrt_weighted_sampler' is the safer choice when the minority class "
            "is small enough to be memorised")
    elif summary['skewed']:
        summary['action'] = 'none (no rebalancing applied)'
        recommendation = (
            "the data is skewed - consider class_balance='sqrt_weighted_sampler' "
            "or 'weighted_loss'; accuracy will flatter the majority class as it is")
    else:
        summary['action'] = 'none needed (classes are within 1.5x of each other)'

    if summary['empty_classes'] and not recommendation:
        recommendation = (f"classes {summary['empty_classes']} have no {split_name} "
                          f"samples and cannot be learned or scored")
    summary['recommendation'] = recommendation
    summary['report'] = format_class_balance_report(summary, class_balance, split_name)
    if verbose:
        print(summary['report'])
    return summary


def make_cv_folds(labels, n_splits, groups=None, seed=0):
    """Split indices into ``n_splits`` class-stratified, optionally grouped folds.

    Every index lands in exactly one validation fold, so the k folds partition
    the dataset. With ``groups`` supplied, a whole group is assigned to a
    single fold — crops from the same well never straddle the train/val line —
    and groups are placed greedily into whichever fold currently leaves the
    per-class proportions most even, which is how stratification survives
    grouping.

    :param labels: integer labels, one per sample.
    :param n_splits: number of folds, must be >= 2.
    :param groups: optional group id per sample (same length as ``labels``).
    :param seed: seed for the deterministic shuffle.
    :returns: list of ``(train_idx, val_idx)`` numpy integer arrays.
    :raises ValueError: if ``n_splits`` < 2, if ``groups`` is the wrong
        length, or if there are fewer samples/groups than folds.
    """
    labels = np.asarray([int(v) for v in labels])
    n = len(labels)
    k = int(n_splits)
    if k < 2:
        raise ValueError(f"n_splits must be >= 2 for k-fold, got {n_splits!r}")
    if n < k:
        raise ValueError(f"cannot build {k} folds from {n} samples")
    if groups is not None and len(groups) != n:
        raise ValueError(
            f"groups has {len(groups)} entries but there are {n} samples")

    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1 if n else 0
    fold_of = np.empty(n, dtype=int)

    if groups is None:
        # Plain stratified k-fold: deal each class round-robin into the folds,
        # starting at a per-class offset so fold 0 does not collect the
        # remainder of every class.
        for c in range(n_classes):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                continue
            rng.shuffle(idx)
            offset = int(rng.integers(k))
            fold_of[idx] = (np.arange(idx.size) + offset) % k
    else:
        groups = np.asarray([str(g) for g in groups])
        uniq = np.unique(groups)
        if uniq.size < k:
            raise ValueError(
                f"cannot build {k} group-aware folds from {uniq.size} distinct "
                f"group(s); lower cross_validation_folds or group at a finer "
                f"level (e.g. cv_group_by='field')")
        # Per-group class histogram, then greedy assignment largest-first.
        hist = {g: np.zeros(n_classes, dtype=float) for g in uniq}
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        for g in uniq:
            for c in labels[members[g]]:
                hist[g][c] += 1.0
        order = sorted(uniq, key=lambda g: (-hist[g].sum(), str(g)))

        class_totals = np.bincount(labels, minlength=n_classes).astype(float)
        class_totals[class_totals == 0] = 1.0
        fold_class = np.zeros((k, n_classes), dtype=float)
        fold_size = np.zeros(k, dtype=float)
        for g in order:
            best_f, best_cost = None, None
            for f in range(k):
                fold_class[f] += hist[g]
                # Spread of each class across folds, as a fraction of that
                # class's total; lower is a more even stratification.
                cost = float(np.mean(np.std(fold_class / class_totals, axis=0)))
                fold_class[f] -= hist[g]
                key = (cost, fold_size[f], f)
                if best_cost is None or key < best_cost:
                    best_cost, best_f = key, f
            fold_class[best_f] += hist[g]
            fold_size[best_f] += hist[g].sum()
            fold_of[members[g]] = best_f

    all_idx = np.arange(n)
    folds = []
    for f in range(k):
        val_idx = all_idx[fold_of == f]
        train_idx = all_idx[fold_of != f]
        folds.append((train_idx, val_idx))
    return folds


def summarize_cv_folds(labels, folds, classes=None, groups=None):
    """Tabulate fold sizes and per-class validation counts.

    :param labels: integer labels, one per sample.
    :param folds: list of ``(train_idx, val_idx)`` from :func:`make_cv_folds`.
    :param classes: ordered class names.
    :param groups: optional group id per sample; adds a distinct-group column.
    :returns: DataFrame with one row per fold.
    """
    labels = np.asarray([int(v) for v in labels])
    if classes is None:
        classes = [f'class_{i}' for i in range(int(labels.max()) + 1 if labels.size else 0)]
    rows = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        y_val = labels[val_idx]
        row = {'fold': i, 'n_train': len(train_idx), 'n_val': len(val_idx)}
        missing = []
        for c, name in enumerate(classes):
            cnt = int(np.sum(y_val == c))
            row[f'val_{name}'] = cnt
            if cnt == 0:
                missing.append(name)
        if groups is not None:
            g = np.asarray([str(x) for x in groups])
            row['val_groups'] = int(np.unique(g[val_idx]).size)
        row['val_classes_missing'] = ','.join(missing)
        rows.append(row)
    return pd.DataFrame(rows)


def report_cv_folds(labels, folds, classes=None, groups=None, group_by='none',
                    verbose=True):
    """Print the fold table and every warning the split earned.

    Two failure modes are called out rather than allowed to surface later as
    mysterious metrics: a class too rare to reach every fold's validation set
    (its recall is undefined there), and ungrouped folds on object crops
    (which leak well identity between train and validation).

    :param labels: integer labels, one per sample.
    :param folds: list of ``(train_idx, val_idx)``.
    :param classes: ordered class names.
    :param groups: optional group id per sample.
    :param group_by: the grouping level that produced ``groups``.
    :param verbose: print the table and warnings.
    :returns: ``(fold_table, warnings)``.
    """
    table = summarize_cv_folds(labels, folds, classes=classes, groups=groups)
    warnings_out = []

    missing = table[table['val_classes_missing'] != '']
    for _, row in missing.iterrows():
        warnings_out.append(
            f"fold {int(row['fold'])}: no validation samples for class(es) "
            f"{row['val_classes_missing']} - per-class scores for those classes "
            f"are undefined in this fold and are dropped from the fold spread")
    if (table['n_val'] == 0).any():
        bad = table.loc[table['n_val'] == 0, 'fold'].tolist()
        warnings_out.append(f"fold(s) {bad} have an empty validation set")
    if groups is None or group_by == 'none':
        warnings_out.append(
            "folds are NOT group-aware: crops from the same well or field can "
            "land on both sides of a fold, which leaks and inflates every "
            "score - set cv_group_by to 'field', 'well' or 'plate' for object crops")

    if verbose:
        print(f"--- Cross-validation folds (k={len(folds)}, "
              f"grouping={group_by}) ---")
        print(table.to_string(index=False))
        for w in warnings_out:
            print(f"  WARNING: {w}")
    return table, warnings_out


def _resolve_channel_indices(channels, verbose=False):
    """Map ``['r','g','b']``-style channel names to tensor channel indices."""
    if channels is None:
        channels = ['r', 'g', 'b']
    chans = []
    if 'r' in channels: chans.append(1)
    if 'g' in channels: chans.append(2)
    if 'b' in channels: chans.append(3)
    if verbose:
        print(f'Training a network on channels: {chans}')
        print(f'Channel 1: Red, Channel 2: Green, Channel 3: Blue')
    return chans


def _classification_data_dir(src, mode, classes):
    """Return ``src/<mode>`` after checking it and every class subfolder exists.

    :param src: dataset root holding ``train/`` and ``test/``.
    :param mode: ``'train'`` or ``'test'``.
    :param classes: ordered class-folder names.
    :returns: validated path to the split folder.
    :raises FileNotFoundError: if the split folder or any class folder is absent.
    """
    data_dir = os.path.join(src, mode)

    # Clear, actionable error when the train/test split hasn't been generated
    # yet — the most common Train-CV mistake is pointing src at the plate folder
    # before the annotated crops were split into train/ and test/.
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No '{mode}/' folder found at: {data_dir}\n"
            f"The classifier trains on a '{src}/train' and '{src}/test' split of\n"
            f"annotated crops, which doesn't exist yet. Generate it first with the\n"
            f"'Generate Training Data' step (spacr.io.generate_training_dataset),\n"
            f"then point the Classify (CV) 'src' at the folder that contains\n"
            f"train/ and test/ (each with class subfolders, e.g. 1/ and 2/)."
        )

    # FIX: raise an error instead of just printing when class folders are missing
    # WHY: the original printed a warning but continued execution, silently
    #      training on a broken/incomplete dataset — this masks data problems
    #      that look like model performance problems
    missing = [c for c in classes if not os.path.isdir(os.path.join(data_dir, c))]
    if missing:
        available = sorted([d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))])
        raise FileNotFoundError(
            f"Class folders missing in {data_dir}:\n"
            f"  Missing:   {missing}\n"
            f"  Available: {available}"
        )
    return data_dir


def _classification_transform(image_size, channel_idx, normalize):
    """Compose the resize / channel-select / normalise transform for crops."""
    from .utils import SelectChannels

    # FIX: match normalization mean/std tuple length to the actual number of
    #      selected channels, not a hardcoded 3
    # WHY: if you select only 1 channel (e.g. channels=['g']), the original
    #      Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) will crash or
    #      silently produce wrong values because the tensor has 1 channel but
    #      normalize expects 3
    n_ch = len(channel_idx)
    norm_transforms = (
        [transforms.Normalize(mean=(0.5,) * n_ch, std=(0.5,) * n_ch)]
        if normalize else []
    )
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(size=(image_size, image_size)),
        SelectChannels(channel_idx),
        *norm_transforms,  # FIX: uses channel-count-aware normalization
    ])


def _cv_group_ids(filenames, group_by, verbose=True):
    """Derive per-sample group ids at the requested plate/well/field level.

    Filenames that do not carry the level fall back to their own basename, so
    they simply behave as an ungrouped sample rather than being silently
    lumped together with unrelated crops. The count of such files is reported,
    because it is exactly the number of crops whose independence is unproven.

    :param filenames: image paths.
    :param group_by: one of ``CV_GROUP_LEVELS``.
    :param verbose: print the grouping summary.
    :returns: ``(group_ids, n_unparsed)`` — ``(None, 0)`` when ``group_by='none'``.
    :raises ValueError: if ``group_by`` is not a supported level.
    """
    if group_by not in CV_GROUP_LEVELS:
        raise ValueError(f"cv_group_by {group_by!r} is not one of {CV_GROUP_LEVELS}")
    if group_by == 'none':
        return None, 0
    ids, unparsed = [], 0
    for p in filenames:
        gid = _png_group_id(p, group_by)
        if gid is None:
            unparsed += 1
            gid = os.path.splitext(os.path.basename(str(p)))[0]
        ids.append(gid)
    if verbose:
        print(f"Grouping folds by {group_by}: {len(set(ids))} distinct "
              f"{group_by}(s) across {len(ids)} crops")
        if unparsed:
            print(f"  WARNING: {unparsed} filename(s) do not encode a "
                  f"{group_by} (expected <plate>_<well>_<field>_..._<object>.png); "
                  f"each is treated as its own group, so their independence is "
                  f"not enforced")
    return ids, unparsed


def generate_cv_loaders(src, n_splits, mode='train', image_size=224, batch_size=32,
                        classes=None, n_jobs=None, pin_memory=False, normalize=False,
                        channels=None, augment=False, verbose=False,
                        group_by='well', class_balance='none', seed=0):
    """Build one ``(train_loader, val_loader)`` pair per cross-validation fold.

    The dataset under ``src/<mode>`` is read once and then re-split k ways, so
    every crop is used for validation exactly once. Folds are class-stratified
    and, by default, grouped by well so that crops from the same well stay on
    one side of the split. Class balancing is applied to the fold's train
    loader only.

    :param src: dataset root containing ``train``/``test`` subfolders.
    :param n_splits: number of folds, must be >= 2.
    :param mode: which split to fold — normally ``'train'``.
    :param image_size: square resize target in pixels.
    :param batch_size: loader batch size.
    :param classes: ordered class names matching the subfolder names.
    :param n_jobs: DataLoader worker count.
    :param pin_memory: if True, pin batches to page-locked memory.
    :param normalize: if True, apply per-channel normalisation.
    :param channels: subset of RGB channels to keep.
    :param augment: if True, 8-fold augment each fold's train split.
    :param verbose: log configuration to stdout.
    :param group_by: fold grouping level, one of ``CV_GROUP_LEVELS``.
    :param class_balance: one of ``CLASS_BALANCE_MODES``, train loaders only.
    :param seed: seed for the deterministic fold assignment.
    :returns: ``(fold_loaders, info)`` where ``fold_loaders`` is a list of
        ``(train_loader, val_loader)`` and ``info`` holds ``fold_table``,
        ``warnings``, ``imbalance`` and ``groups``.
    :raises ValueError: if ``n_splits`` < 2 or a setting value is unknown.
    """
    from .utils import augment_dataset

    if int(n_splits) < 2:
        raise ValueError(
            f"cross_validation_folds must be >= 2 to build folds, got {n_splits!r}; "
            f"0 or 1 means the single train/validation split")
    if classes is None:
        classes = ['nc', 'pc']

    channel_idx = _resolve_channel_indices(channels, verbose=verbose)
    data_dir = _classification_data_dir(src, mode, classes)
    transform = _classification_transform(image_size, channel_idx, normalize)
    data = spacrDataset(data_dir, classes, transform=transform,
                        shuffle=True, pin_memory=pin_memory)

    labels = dataset_labels(data)
    filenames = dataset_filenames(data)
    groups, _ = _cv_group_ids(filenames, group_by, verbose=True)

    folds = make_cv_folds(labels, int(n_splits), groups=groups, seed=seed)
    fold_table, fold_warnings = report_cv_folds(
        labels, folds, classes=classes, groups=groups, group_by=group_by,
        verbose=True)

    imbalance = report_class_balance(labels, classes=classes,
                                     class_balance=class_balance,
                                     split_name='train', verbose=True)

    num_workers = max(n_jobs, 4) if n_jobs is not None else 0
    use_persistent = num_workers > 0

    fold_loaders = []
    for train_idx, val_idx in folds:
        train_dataset = Subset(data, list(train_idx))
        val_dataset = Subset(data, list(val_idx))
        if augment:
            train_dataset = augment_dataset(
                train_dataset, is_grayscale=(len(channel_idx) == 1))

        sampler, _ = make_class_balance_sampler(
            dataset_labels(train_dataset), class_balance)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=use_persistent)
        # The validation loader is never sampled and never shuffled: its job is
        # to measure the model against the real class prior.
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=use_persistent)
        fold_loaders.append((train_loader, val_loader))

    info = {
        'fold_table': fold_table,
        'warnings': fold_warnings,
        'imbalance': imbalance,
        'groups': groups,
        'group_by': group_by,
        'labels': labels,
        'folds': folds,
        'classes': list(classes),
    }
    return fold_loaders, info


def generate_loaders(src, mode='train', image_size=224, batch_size=32,
                     classes=None, n_jobs=None, validation_split=0.0,
                     pin_memory=False, normalize=False, channels=None,
                     augment=False, verbose=False, class_balance='none'):
    """Build ``spacrDataLoader`` objects for training, validation, or testing.

    Reads class subfolders under ``src/<mode>``, applies the requested
    transforms (channel selection, optional normalisation, optional
    augmentation) and returns loaders sized to ``batch_size``.

    :param src: Root folder containing ``train``/``test`` subfolders.
    :param mode: Which split to load — ``'train'`` or ``'test'``.
    :param image_size: Square resize target in pixels. Default ``224``.
    :param batch_size: Loader batch size. Default ``32``.
    :param classes: Ordered class names. Default ``['nc', 'pc']``.
    :param n_jobs: DataLoader worker count. Default: derived from CPU count.
    :param validation_split: Fraction of the train split to hold out.
    :param pin_memory: If True, pin batches to page-locked memory.
    :param normalize: If True, apply per-channel normalisation.
    :param channels: Subset of RGB channels to keep, e.g. ``['r', 'g']``.
    :param augment: If True, apply the training augmentation pipeline.
    :param verbose: If True, log configuration to stdout.
    :param class_balance: One of ``CLASS_BALANCE_MODES``. ``'none'`` (default)
        leaves sampling untouched; the sampler modes attach a
        ``WeightedRandomSampler`` to the TRAIN loader only. The skew is
        reported either way.
    :returns: For ``mode='train'``, a tuple of loaders and a plot handle;
        for ``mode='test'``, the test loader (plus optional metadata).
    :raises ValueError: if ``class_balance`` is not a recognised mode.
    """

    if classes is None:
        classes = ['nc', 'pc']
    from .utils import augment_dataset

    if class_balance not in CLASS_BALANCE_MODES:
        raise ValueError(
            f"class_balance {class_balance!r} is not one of {CLASS_BALANCE_MODES}")

    channels = _resolve_channel_indices(channels, verbose=verbose)

    if mode == 'train':
        print('Loading Train and validation datasets')
    elif mode == 'test':
        validation_split = 0.0
        print('Loading test dataset')
    else:
        print(f'mode:{mode} is not valid, use mode = train or test')
        return

    data_dir = _classification_data_dir(src, mode, classes)
    transform = _classification_transform(image_size, channels, normalize)

    data = spacrDataset(data_dir, classes, transform=transform,
                        shuffle=True, pin_memory=pin_memory)

    #num_workers = n_jobs if n_jobs is not None else 0
    num_workers = max(n_jobs, 4) if n_jobs is not None else 0
    use_persistent = num_workers > 0

    if validation_split > 0 and mode == 'train':
        train_size = int((1 - validation_split) * len(data))
        val_size = len(data) - train_size
        if not augment:
            print(f'Train data:{train_size}, Validation data:{val_size}')
        train_dataset, val_dataset = random_split(data, [train_size, val_size])

        if augment:
            print(f'Data before augmentation: Train: {len(train_dataset)}, Validation:{len(val_dataset)}')
            train_dataset = augment_dataset(train_dataset, is_grayscale=(len(channels) == 1))
            print(f'Data after augmentation: Train: {len(train_dataset)}')

        # Skew is measured on the labels the model will actually see, after the
        # split and after augmentation, and reported on every run.
        report_class_balance(dataset_labels(train_dataset), classes=classes,
                             class_balance=class_balance, split_name='train')
        report_class_balance(dataset_labels(val_dataset), classes=classes,
                             class_balance='none', split_name='validation')

        # A sampler and shuffle=True are mutually exclusive in DataLoader; the
        # sampler already draws in random order.
        sampler, _ = make_class_balance_sampler(
            dataset_labels(train_dataset), class_balance)

        print(f'Generating Dataloader with {num_workers} workers')
        train_loaders = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=(sampler is None),
                                   sampler=sampler,
                                   num_workers=num_workers,  # FIX: was hardcoded to 1
                                   pin_memory=pin_memory,
                                   persistent_workers=use_persistent)

        # FIX: don't shuffle the validation DataLoader
        # WHY: shuffling validation data wastes time and has zero benefit —
        #      evaluation metrics are computed over the entire set regardless of order
        # The validation loader also never receives the sampler: resampling it
        # would change the class prior the reported metrics are measured
        # against, so a "balanced" accuracy would stop describing the screen.
        val_loaders = DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False,  # FIX: was True
                                 num_workers=num_workers,  # FIX: was hardcoded to 1
                                 pin_memory=pin_memory,
                                 persistent_workers=use_persistent)
        train_fig = None
        return train_loaders, val_loaders, train_fig

    else:
        split_name = 'train' if mode == 'train' else 'test'
        # Held-out test data is reported but never resampled.
        effective_balance = class_balance if split_name == 'train' else 'none'
        report_class_balance(dataset_labels(data), classes=classes,
                             class_balance=effective_balance,
                             split_name=split_name)
        sampler = None
        if split_name == 'train':
            sampler, _ = make_class_balance_sampler(dataset_labels(data),
                                                    class_balance)
        train_loaders = DataLoader(data, batch_size=batch_size,
                                   shuffle=(sampler is None),
                                   sampler=sampler,
                                   num_workers=num_workers,  # FIX: was hardcoded to 1
                                   pin_memory=pin_memory,
                                   persistent_workers=use_persistent)
        val_loaders = []
        train_fig = None
        return train_loaders, val_loaders, train_fig

def generate_training_dataset(settings):
    """
    Build a balanced training/testing dataset from one of:
      - metadata rules (exact matches or compound 'where' rules)
      - annotation columns (each <col>_<value> is a standalone class)
      - measurement rules (numeric ranges/bins; supports multiple conditions per class)

    New behavior (annotation mode):
      - If a column has only one annotated value (e.g., only '1's), we add a
        '<column>_random' class using unannotated rows for that column (same size as positives).
      - Optional: persist that random selection into DB as a new INT column named '<column>_random' with 1's.

    ``crop_source`` chooses where the pixels come from. ``'png'`` (and
    ``'auto'`` wherever a crop folder exists) copies the pre-generated PNGs,
    unchanged. ``'merged'`` (and ``'auto'`` with no crop folder) cuts each
    selected crop out of ``merged/*.npy`` through :mod:`spacr.crops` instead:
    the labels still come from ``png_list``, but the pixels are cut fresh at
    the current crop settings, so the training set costs no standing disk and
    cannot be built out of a folder that has gone stale. A project with no
    ``png_list`` at all falls back to the object measurement table, which
    still carries everything the metadata rules select on.

    Required helpers:
      - _read_and_merge_data, _read_db (from .io)
      - generate_dataset_from_lists(dst, class_data, classes, test_split)
      - save_settings (from .utils)
    """
    import os, random, operator, sqlite3
    import numpy as np

    from .io import _read_and_merge_data, _read_db
    from .utils import save_settings
    from .settings import set_generate_training_dataset_defaults

    # --- defaults & toggles --------------------------------------------------
    settings = set_generate_training_dataset_defaults(settings)
    balance_to_smallest = bool(settings.get('balance_to_smallest', True))
    png_type = settings.get('png_type', 'cell_png')
    tables = settings.get('tables') or ['cell', 'nucleus', 'pathogen', 'cytoplasm']
    write_rand_col = bool(settings.get('write_random_annotation_column', False))

    # Limits for merge helper
    if 'nucleus' not in tables:
        settings['nuclei_limit'] = False
    if 'pathogen' not in tables:
        settings['pathogen_limit'] = 0

    save_settings(settings, 'cv_dataset', show=True)

    # Normalize src to list
    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    # --- helpers -------------------------------------------------------------
    def _ensure_unique_dir(dst_base):
        dst = dst_base
        if os.path.exists(dst):
            base = dst
            for j in range(1, 100000):
                try_dst = f"{base}_{j}"
                if not os.path.exists(try_dst):
                    print(f'Creating new directory for training: {try_dst}')
                    dst = try_dst
                    break
        return dst

    def _load_png_table(db_path, object_type='cell'):
        # read only png_list (we don't force-meet with measurements; keep it permissive)
        try:
            [png_df] = _read_db(db_loc=db_path, tables=['png_list'])
            png_df = png_df.copy()
        except Exception:
            png_df = pd.DataFrame()
        if len(png_df):
            return png_df
        # No png_list means no PNG folder was ever written. The objects are
        # still in the measurement table, with the same well metadata the
        # metadata rules select on, so fall back to those rather than
        # returning "0 classes" for a project that has everything it needs.
        print(f"No 'png_list' rows in {db_path}; falling back to the "
              f"'{object_type}' measurement table for the crop list.")
        return crop_rows_from_object_table(db_path, object_type)

    def _class_items(frame):
        """Return the per-row crop entries a class list is built from.

        On-demand handles when the merged source is in play, PNG paths
        otherwise -- generate_dataset_from_lists takes either.
        """
        if CROP_REF_COLUMN in frame.columns:
            return [ref for ref in frame[CROP_REF_COLUMN].tolist()
                    if ref is not None]
        return frame['png_path'].dropna().tolist()

    def _fix_path_under_src(src_root, p):
        """Make sure png_path lives under the current src root (portable absolute fix)."""
        if not isinstance(p, str) or p.strip() == "":
            return None
        # already under root?
        if os.path.isabs(p) and p.startswith(src_root):
            return p if os.path.exists(p) else p  # keep as-is; existence checked later when copying
        # try CV folder pattern split and rebuild
        parts = p.split('/data/')
        if len(parts) > 1:
            return os.path.join(src_root, 'data', parts[1])
        # fallback: join relative to src_root
        if not os.path.isabs(p):
            return os.path.join(src_root, p.lstrip('/'))
        return p

    def _apply_where(df, where):
        """where: list of {'column','op','value'} AND-combined."""
        if not where:
            return df
        OPS = {
            '==': operator.eq,  '!=': operator.ne,
            '<': operator.lt,   '<=': operator.le,
            '>': operator.gt,   '>=': operator.ge,
            'in': lambda a,b: a.isin(b) if hasattr(a, 'isin') else False,
            'notin': lambda a,b: ~a.isin(b) if hasattr(a, 'isin') else False,
        }
        mask = np.ones(len(df), dtype=bool)
        for cond in where:
            col, op, val = cond['column'], cond['op'], cond.get('value', None)
            if col not in df.columns or op not in OPS:
                mask &= False
                continue
            series = df[col]
            if op in ('in','notin'):
                vals = val if isinstance(val, (list,tuple,set)) else [val]
                mask &= OPS[op](series, vals)
            else:
                mask &= OPS[op](series, val)
        return df[mask]

    def _balance_lists(list_of_lists):
        if not list_of_lists:
            return list_of_lists
        if not balance_to_smallest:
            return list_of_lists
        sizes = [len(x) for x in list_of_lists]
        size = min(sizes) if sizes else 0
        print(f"Class sizes: {sizes} -> balancing to {size}")
        out = []
        for paths in list_of_lists:
            if len(paths) > size:
                out.append(random.sample(paths, size))
            else:
                out.append(paths)
        return out

    def _annotation_classes_from_columns(png_df, ann_cols, ann_vals_filter=None, db_path=None):
        """
        Build classes per (column,value). If a column only has one annotated value in {1,2},
        also create '<column>_random' from unannotated rows (same count as positives).
        Optionally persist '<column>_random' as a new INT column with 1's for sampled rows.

        Returns (names, lists) aligned.
        """
        names, lists = [], []
        if not ann_cols:
            return names, lists

        # Work with numeric-ish annotations 1/2; accept strings that can be cast to int.
        df = png_df.copy()
        # We only care about png_path, the crop handle and the annotation cols
        keep_cols = ['png_path'] + (
            [CROP_REF_COLUMN] if CROP_REF_COLUMN in df.columns else []
        ) + [c for c in ann_cols if c in df.columns]
        df = df[keep_cols]

        # For lookups by path when writing back random labels
        df_idx_by_path = {p: i for i, p in enumerate(df['png_path'])}

        for col in ann_cols:
            if col not in df.columns:
                print(f"Warning: annotation column '{col}' not in png_list; skipping.")
                continue

            # Identify annotated values present (castable to int)
            col_series = df[col].dropna()
            try:
                vals = sorted(set(col_series.astype(int).tolist()))
            except Exception:
                # Non-numeric labels -> keep as-is
                vals = sorted(set(col_series.tolist()))

            # Optional filter: {col: [allowed_values]}
            if ann_vals_filter and col in ann_vals_filter:
                allow = set(ann_vals_filter[col])
                vals = [v for v in vals if v in allow]

            # Collect classes for each observed value
            distinct_vals = []
            for v in vals:
                cls_name = f"{col}_{v}"
                sel = _class_items(df[df[col] == v])
                distinct_vals.append((v, sel))
                names.append(cls_name)
                lists.append(sel)

            # If only one annotated value (typical 1-only column), create <col>_random
            if len(distinct_vals) == 1:
                v, pos_paths = distinct_vals[0]
                pos_n = len(pos_paths)

                # Unannotated = rows where column is NULL/NaN
                unann_paths = _class_items(df[df[col].isna()])
                if not unann_paths:
                    print(f"Column '{col}': no unannotated rows available for <{col}_random>; skipping random class.")
                    continue

                if pos_n == 0:
                    print(f"Column '{col}': only one value present but it has 0 rows; skipping random class.")
                    continue

                # Sample negatives
                if len(unann_paths) >= pos_n:
                    rand_paths = random.sample(unann_paths, pos_n)
                else:
                    # Not enough; sample all unannotated (and we’ll balance later anyway)
                    rand_paths = unann_paths

                names.append(f"{col}_random")
                lists.append(rand_paths)

                # Optionally persist a new column in DB and mark sampled as 1
                if write_rand_col and db_path:
                    rand_col = f"{col}_random"
                    qcol = rand_col.replace('"', '""')
                    with sqlite3.connect(db_path, timeout=30) as conn:
                        cur = conn.cursor()
                        cur.execute('PRAGMA table_info("png_list")')
                        existing = {r[1] for r in cur.fetchall()}
                        if rand_col not in existing:
                            cur.execute(f'ALTER TABLE "png_list" ADD COLUMN "{qcol}" INTEGER')
                            conn.commit()

                        # write 1 for sampled paths; NULL elsewhere (default).
                        # An on-demand handle carries the png_path its row
                        # named, so the column is written the same way
                        # whichever source produced the pixels.
                        for p in rand_paths:
                            png_path = (p.row.get('png_path')
                                        if isinstance(p, LazyCropPNG) else p)
                            if not png_path:
                                continue
                            cur.execute(
                                f'UPDATE "png_list" SET "{qcol}" = 1 WHERE png_path = ?',
                                (png_path,)
                            )
                        conn.commit()

        return names, lists

    # --- main assembly across sources ---------------------------------------
    class_path_list = None
    class_names = None
    dst_final = None  # last destination
    crop_db_path = None  # last measurements.db, for the crop-format lookup

    for i, src in enumerate(settings['src']):
        db_path = os.path.join(src, 'measurements', 'measurements.db')

        if len(settings['src']) > 1 and i == 0:
            dst = os.path.join(src, 'datasets', 'training_all')
        else:
            dst = os.path.join(src, 'datasets', 'training')
        dst = _ensure_unique_dir(dst)
        dst_final = dst

        object_type = crop_object_type(png_type)
        png_df = _load_png_table(db_path, object_type)

        # Fix/normalize paths under this src
        fixed_paths = [ _fix_path_under_src(src, p) for p in png_df['png_path'] ]
        png_df['png_path'] = fixed_paths

        # Filter by image type if requested
        if png_type:
            png_df = png_df[png_df['png_path'].astype(str).str.contains(png_type, na=False)]

        # Where the pixels come from. 'png' (and 'auto' with a crop folder
        # present) leaves every list below holding plain paths, which
        # generate_dataset_from_lists copies exactly as it always has. Only
        # the merged source replaces them with on-demand handles.
        source = open_crop_source(settings, src, object_type=object_type)
        if source is not None and getattr(source, 'kind', 'png') == 'merged':
            rows = crop_rows_from_png_list(db_path, png_df, object_type)
            refs = crop_refs_for_rows(source, rows, object_type)
            rows = rows.copy()
            rows[CROP_REF_COLUMN] = refs
            png_df = rows
        crop_db_path = db_path if os.path.isfile(db_path) else None

        mode = str(settings['dataset_mode']).lower()
        this_names, this_lists = [], []

        if mode == 'metadata':
            rules = settings.get('metadata_rules')
            if rules:
                if all('name' in r for r in rules):
                    for r in rules:
                        where = r.get('where')
                        col, op, val = r.get('column'), r.get('op'), r.get('value')
                        if where is None and col is not None and op is not None:
                            where = [{'column': col, 'op': op, 'value': val}]
                        df_sel = _apply_where(png_df, where)
                        this_names.append(r['name'])
                        this_lists.append(_class_items(df_sel))
                else:
                    for r in rules:
                        col, op, val = r['column'], r['op'], r['value']
                        df_sel = _apply_where(png_df, [{'column': col, 'op': op, 'value': val}])
                        name = r.get('name', f"{col}{op}{val}")
                        this_names.append(name)
                        this_lists.append(_class_items(df_sel))
            else:
                class_meta = settings.get('class_metadata') or []
                if isinstance(class_meta, str):
                    # A settings CSV stores the repr, and a caller that hands
                    # the string straight through used to be iterated one
                    # CHARACTER at a time -- "[['c1'], ['c2']]" became
                    # seventeen classes named '[', '[', "'", 'c', ... The Qt
                    # panel now collects a real list; this covers the CSV and
                    # CLI paths that do not go through it.
                    import ast as _ast
                    try:
                        parsed = _ast.literal_eval(class_meta.strip())
                    except (ValueError, SyntaxError):
                        parsed = [p.strip() for p in class_meta.split(',') if p.strip()]
                    class_meta = parsed if isinstance(parsed, (list, tuple)) else [parsed]
                # The column the class_metadata values are matched against is
                # the one the user named in 'metadata_type_by'. It used to be
                # hard-coded to 'condition' -- a column no spaCR writer puts
                # in png_list unless annotate_conditions has been run -- so a
                # run configured with metadata_type_by='columnID' selected on
                # a column it was never pointed at, and the guard below
                # printed "got 0 classes" and then indexed the missing column
                # anyway, turning a diagnosable misconfiguration into a bare
                # KeyError several frames down.
                meta_col = settings.get('metadata_type_by') or 'condition'
                meta_col = str(meta_col).strip() or 'condition'
                if meta_col not in png_df.columns:
                    raise ValueError(
                        f"metadata mode: column '{meta_col}' is not in png_list, "
                        f"so no class can be selected. Present columns: "
                        f"{sorted(map(str, png_df.columns))}. Set "
                        f"'metadata_type_by' to one of those (usually 'columnID' "
                        f"or 'rowID'), or switch 'dataset_mode' to "
                        f"'annotation'/'measurement'."
                    )
                # Compare as text: png_list holds 'c1'/'r1' strings but a
                # fallback to the object table can hand back a numeric column,
                # and class_metadata is whatever the settings CSV parsed to.
                meta_values = png_df[meta_col].astype(str)
                for cm in class_meta:
                    # One class per entry. An entry may be a single value
                    # ('c1') or a group of values (['c1','c2']) that share one
                    # label -- the list-of-lists form the GUI defaults to and
                    # every settings CSV on disk already carries. It used to be
                    # str()'d whole, so ['c1'] was matched as the literal text
                    # "['c1']" and selected nothing.
                    if isinstance(cm, (list, tuple, set)):
                        wanted = [str(v) for v in cm]
                    else:
                        wanted = [str(cm)]
                    name = wanted[0] if len(wanted) == 1 else '_'.join(wanted)
                    sel = png_df[meta_values.isin(wanted)]
                    this_names.append(name)
                    this_lists.append(_class_items(sel))

        elif mode == 'annotation':
            ann_cols = settings.get('annotation_columns')
            if not ann_cols:
                # backward compatibility
                ann_cols = [settings.get('annotation_column')]
            ann_cols = [c for c in (ann_cols or []) if c]
            ann_vals = settings.get('annotation_values')  # optional dict {col:[values]}

            this_names, this_lists = _annotation_classes_from_columns(
                png_df, ann_cols, ann_vals_filter=ann_vals, db_path=db_path
            )

        elif mode == 'measurement':
            m_rules = settings.get('measurement_rules') or []
            for r in m_rules:
                name = r['name']
                where = r.get('where', [])
                df_sel = _apply_where(png_df, where)
                this_names.append(name)
                this_lists.append(_class_items(df_sel))

        else:
            print(f"Invalid dataset_mode: {settings['dataset_mode']}. Use 'metadata'|'annotation'|'measurement'.")
            return None, None

        # Initialize global collectors (keep class order of first source)
        if class_path_list is None:
            class_path_list = [[] for _ in range(len(this_lists))]
            class_names = this_names[:]

        # Warn on mismatch; align by index
        if this_names != class_names:
            print("Warning: class name/order mismatch across sources; aligning by index. "
                  "Make sure your rules are identical for all 'src' roots.")
        for idx in range(min(len(class_path_list), len(this_lists))):
            class_path_list[idx].extend(this_lists[idx])

    # Nothing to do?
    if not class_path_list or sum(len(x) for x in class_path_list) == 0:
        print("No class data assembled; aborting.")
        return None, None

    # Balance to smallest (optional)
    class_path_list = _balance_lists(class_path_list)

    # Write out
    from .io import generate_dataset_from_lists
    final_names = class_names or [f"class_{i}" for i in range(len(class_path_list))]
    print(f"class_path_list: {len(class_path_list)} classes")

    train_class_dir, test_class_dir = generate_dataset_from_lists(
        dst_final,
        class_data=class_path_list,
        classes=final_names,
        test_split=settings['test_split'],
        db_path=crop_db_path,
    )

    # expose actual disk classes for downstream training
    settings['classes'] = final_names
    settings['nr_classes'] = len(final_names)
    
    try:
        save_settings(settings, 'cv_dataset', show=False)
    except Exception:
        pass

    return train_class_dir, test_class_dir

def training_dataset_from_annotation(db_path, dst, annotation_column='test', annotated_classes=(1, 2)):
    """Group per-object PNG paths by manual annotation values so they can be turned into a CNN training set.

    Reads the ``png_list`` table of a spacr ``measurements.db``, buckets
    PNG paths by the value found in ``annotation_column`` (typically
    filled by the spacr annotation GUI), and, when only one class has
    been annotated, samples an equal-sized "other" class from
    unannotated rows. The returned list-of-lists is consumed by
    :func:`generate_dataset_from_lists` to lay out
    ``train/<class>/*.png`` / ``test/<class>/*.png``.

    :param db_path: SQLite ``measurements.db`` containing a
        ``png_list`` table with ``png_path`` plus ``annotation_column``.
    :param dst: Output root (currently unused; kept for API symmetry
        with sister builders).
    :param annotation_column: Column in ``png_list`` holding class
        labels. Default ``'test'``.
    :param annotated_classes: Class values to pull from
        ``annotation_column``. When length is 1, an equal-sized "other"
        class is sampled from rows whose annotation != that value.
    :returns: List of lists — one list of PNG paths per output class,
        in the same order as ``annotated_classes``.

    Example:
        .. code-block:: python

            from spacr.io import training_dataset_from_annotation, generate_dataset_from_lists
            class_data = training_dataset_from_annotation(
                '/data/plate01/measurements/measurements.db',
                dst='/data/plate01/dataset',
                annotation_column='test', annotated_classes=(1, 2),
            )
            generate_dataset_from_lists('/data/plate01/dataset', class_data, classes=['neg','pos'])

    See Also:
        :func:`training_dataset_from_annotation_metadata` — same, but
        first restricts rows by plate row/column metadata.
        :func:`generate_dataset_from_lists` — turns the returned lists
        into a ``train/`` / ``test/`` folder tree.
    """
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

def training_dataset_from_annotation_metadata(db_path, dst, annotation_column='test', annotated_classes=(1, 2), metadata_type_by='columnID', class_metadata=None):
    """Same as :func:`training_dataset_from_annotation` but pre-filtered by plate metadata.

    Restricts source rows to those whose ``rowID`` or ``columnID`` is in
    ``class_metadata`` before grouping by annotation value.

    :param db_path: SQLite database with a ``png_list`` table.
    :param dst: Output root (unused; kept for API symmetry).
    :param annotation_column: Column holding class labels.
    :param annotated_classes: Class values to pull.
    :param metadata_type_by: Which metadata column to filter on —
        ``'rowID'`` or ``'columnID'``.
    :param class_metadata: Allowed values for ``metadata_type_by``.
        Default ``['c1', 'c2']``.
    :returns: List of lists — one list of PNG paths per output class.
    :raises ValueError: if ``metadata_type_by`` is not ``'rowID'`` or
        ``'columnID'``.
    """
    if class_metadata is None:
        class_metadata = ['c1','c2']
    all_paths = []

    # Connect to the database and retrieve the image paths and annotations
    print(f'Reading DataBase: {db_path}')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Retrieve all paths and annotations from the database
        query = f"SELECT png_path, {annotation_column}, rowID, columnID FROM png_list"
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

def _crop_format_of_items(items, db_path=None):
    """Return the crop format the items share, or None when they disagree.

    Copied PNGs keep whatever format the folder they came from was in, so the
    destination has to be stamped with *that*, not with the current one --
    marking a folder of legacy crops as RGB reverses every channel name
    attached to a model trained on it. Crops cut on demand are always current.
    """
    from . import crops

    formats = set()
    folders = set()
    for item in items:
        if isinstance(item, LazyCropPNG):
            formats.add(crops.CROP_FORMAT_CURRENT)
        else:
            folders.add(os.path.dirname(os.path.abspath(str(item))))
    for folder in folders:
        try:
            formats.add(crops.crop_folder_format(folder, db_path=db_path))
        except Exception:
            formats.add(crops.CROP_FORMAT_LEGACY_BGR)
    if len(formats) == 1:
        return formats.pop()
    return None


def _write_class_item(item, dst_dir):
    """Put one crop into ``dst_dir``: copy a path, cut a :class:`LazyCropPNG`."""
    if isinstance(item, LazyCropPNG):
        out = os.path.join(dst_dir, item.name or 'crop.png')
        with open(out, 'wb') as handle:
            handle.write(item.png_bytes())
        return out
    out = os.path.join(dst_dir, os.path.basename(str(item)))
    shutil.copy(str(item), out)
    return out


def generate_dataset_from_lists(dst, class_data, classes, test_split=0.1,
                                db_path=None):
    """Put the crops listed per class into ``dst/train/<class>`` and ``dst/test/<class>``.

    An entry may be a **path**, which is copied byte for byte exactly as
    before, or a :class:`LazyCropPNG`, which is cut out of ``merged/*.npy``
    through :mod:`spacr.crops` and written as a current-format (RGB) crop PNG.
    The two are interchangeable, so a training set can be built with no crop
    folder on disk at all.

    Each destination class folder is stamped with the crop-format sidecar
    *before* it is filled: with the current format when the crops were cut
    here, with the source folder's format when they were copied out of one,
    and not at all (loudly) when one class mixes the two. Leaving a folder of
    crops unmarked is what makes it legacy by default, which is the one
    outcome that silently reverses the channels a model is trained on.

    :param dst: Output root; ``train`` and ``test`` subfolders are created.
    :param class_data: Sequence of per-class lists of paths and/or
        :class:`LazyCropPNG` handles.
    :param classes: Class names paired positionally with ``class_data``.
    :param test_split: Fraction of each class routed to ``test/``.
        Default ``0.1``.
    :param db_path: optional ``measurements.db`` consulted for the crop format
        of a source folder that carries no sidecar.
    :returns: ``(train_dir, test_dir)`` tuple of the top-level split paths.
    :raises ValueError: if ``len(class_data) != len(classes)``.
    """
    from .utils import print_progress
    # Make sure that the length of class_data matches the length of classes
    if len(class_data) != len(classes):
        raise ValueError("class_data and classes must have the same length.")

    total_files = sum(len(data) for data in class_data)
    processed_files = 0
    time_ls = []
    failed = 0

    # Stamp BEFORE the first crop lands, for the reason
    # spacr.crops.stamp_crop_folder gives: a run killed part-way through
    # leaves a marked tree holding fewer crops, never an unmarked tree of
    # corrected ones. The marker goes at the dataset root and describes the
    # whole split -- not inside train/<class>/, because the class folders are
    # enumerated as "the classes" and as "the samples", and a sidecar there
    # would be counted as one of each.
    every_item = [item for data in class_data for item in data]
    fmt = _crop_format_of_items(every_item, db_path=db_path)
    if every_item and fmt is None:
        print(f"Warning: this dataset mixes crops of more than one format, so "
              f"{dst} is left unmarked. Migrate the legacy folders first: "
              f"python -m spacr.crops <root>")
    elif every_item:
        os.makedirs(dst, exist_ok=True)
        mark_crop_output_folder(dst, fmt=fmt, classes=list(map(str, classes)),
                                split='train/test')

    for cls, data in zip(classes, class_data):
        # Create directories
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Split the data
        print('data',len(data), test_split)
        if not data:
            # sklearn answers an empty class with "With n_samples=0,
            # test_size=0.25 ... the resulting train set will be empty", which
            # names the splitter's parameters rather than the rule that
            # selected nothing. Say which class, keep the folder so the class
            # list still matches the tree, and let the summary below flag it.
            print(f"Class {cls!r} selected no crops; its folders are empty.")
            continue
        train_data, test_data = train_test_split(data, test_size=test_split, shuffle=True, random_state=42)

        # Write train files
        for item in train_data:
            start = time.time()
            try:
                _write_class_item(item, train_class_dir)
            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(f"Could not add {item!r} to {train_class_dir}: {exc}")
            duration = time.time() - start
            time_ls.append(duration)
            print_progress(processed_files, total_files, n_jobs=1, time_ls=None, batch_size=None, operation_type="Copying files for Train dataset")
            processed_files += 1

        # Write test files
        for item in test_data:
            start = time.time()
            try:
                _write_class_item(item, test_class_dir)
            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(f"Could not add {item!r} to {test_class_dir}: {exc}")
            duration = time.time() - start
            time_ls.append(duration)
            print_progress(processed_files, total_files, n_jobs=1, time_ls=None, batch_size=None, operation_type="Copying files for Test dataset")
            processed_files += 1

    # Print summary. The sidecar is not a crop, so it is not counted.
    empty = []
    for cls in classes:
        train_class_dir = os.path.join(dst, f'train/{cls}')
        test_class_dir = os.path.join(dst, f'test/{cls}')
        n_train = len([f for f in os.listdir(train_class_dir) if not f.startswith('.')])
        n_test = len([f for f in os.listdir(test_class_dir) if not f.startswith('.')])
        print(f'Train class {cls}: {n_train}, Test class {cls}: {n_test}')
        if n_train == 0:
            empty.append(cls)

    if failed:
        # A crop that cannot be written used to take the whole run down with a
        # bare FileNotFoundError from shutil.copy, naming one file and not the
        # scale of the problem. Say how many, and finish the split -- unless
        # nothing landed at all, which is not a partial result but a broken
        # input, and training on it would just be training on nothing.
        print(f"Warning: {failed} of {total_files} crops could not be written "
              f"into {dst}.")
        if failed == total_files:
            raise RuntimeError(
                f"No crop could be written into {dst}: all {total_files} "
                f"selected crops failed. If the PNG crop folder has been "
                f"deleted or moved, set crop_source='merged' to cut the crops "
                f"out of merged/*.npy instead.")
    if empty:
        print(f"Warning: class(es) {', '.join(map(str, empty))} have no "
              f"training images; the model cannot learn them.")

    return os.path.join(dst, 'train'), os.path.join(dst, 'test')

def _next_synthetic_yokogawa_well(used_wells, n_wells=384):
    """Return the next free ``plate<N>_<well>`` id, and claim it.

    Fills one plate before starting the next, and **never returns an id
    that is already in** ``used_wells``. The version this replaces fell out
    of its ``for`` loop and returned ``f"plate{plate}_A01"`` unconditionally,
    so the 386th caller got ``plate2_A01`` a second time and its TIFF
    overwrote the 385th's — 386 inputs, 385 outputs, nothing said.

    :param used_wells: set of ids already handed out; mutated in place.
    :param n_wells: plate format to fill, a key of
        :data:`spacr.schema.PLATE_FORMATS`.
    :returns: the claimed ``plate<N>_<well>`` id.
    """
    sequence = _cv.well_sequence(n_wells)
    plate = 1
    while True:
        for well in sequence:
            name = f"plate{plate}_{well}"
            if name not in used_wells:
                used_wells.add(name)
                return name
        plate += 1


def convert_separate_files_to_yokogawa(folder, regex):
    """Rename per-slice TIFFs in ``folder`` into the Yokogawa CV filename convention.

    Files are grouped by ``(plateID, wellID, fieldID, timeID, chanID)``
    parsed from the regex. Groups with multiple Z-slices are max-
    projected before saving, and the mapping is logged to
    ``rename_log.csv``.

    Well naming, in full:

    1. A ``wellID`` that **is** a well address keeps it —
       :func:`spacr.convert.normalise_well` reads ``a1``, ``A-01``, ``Q01``
       (row 17) and ``AA13`` (row 27) alike. Every one of those used to be
       thrown away and replaced with the next free synthetic id, so a real
       1536-plate came out relabelled ``A01, A02, …`` with only
       ``rename_log.csv`` to say what had happened.
    2. Anything else — ``1``, ``well_left``, a positional number — is
       handed a synthetic id, in ``_natural_key`` order so the same folder
       always converts the same way. It used to follow ``os.listdir`` order,
       which is the filesystem's business and not reproducible.
    3. Each distinct source ``plateID`` gets its own ``plate<N>`` token, so
       well ``A01`` of two source plates stays two wells.

    :param folder: Folder containing the source TIFFs.
    :param regex: Pattern with named groups ``wellID`` (required) plus
        optional ``plateID``, ``fieldID``, ``timeID``, ``chanID``,
        ``sliceID``.
    :returns: None
    """
    pattern = re.compile(regex, re.I)

    files_by_region = {}
    rename_log = []
    csv_path = os.path.join(folder, "rename_log.csv")
    used_wells = set()
    region_to_well = {}

    # Group files by (plateID, wellID, fieldID, timeID, chanID)
    for file in sorted(os.listdir(folder)):
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

    # -- well assignment, before a single file is written ------------------
    # A well is a well, not a field: keyed on (plateID, wellID) so the two
    # fields of one well do not become two wells, which is what keying on
    # (plateID, wellID, fieldID) did.
    source_wells = sorted({region[:2] for region in files_by_region},
                          key=lambda pair: (_cv.natural_key(pair[0]),
                                            _cv.natural_key(pair[1])))
    plate_tokens = {plate_key: f'plate{index}' for index, plate_key in enumerate(
        sorted({plate_key for plate_key, _ in source_wells},
               key=_cv.natural_key), start=1)}

    # Pass 1: every source well that is a real address keeps it. Sized to the
    # plate the addresses actually need — a folder holding AA13 is a 1536.
    canonical_wells = {}
    for plate_key, well_key in source_wells:
        canonical = _cv.normalise_well(well_key)
        if canonical is not None:
            canonical_wells[(plate_key, well_key)] = canonical
    n_wells = _cv.plate_format_for_names(0, sorted(set(canonical_wells.values())))

    for key, canonical in canonical_wells.items():
        name = f'{plate_tokens[key[0]]}_{canonical}'
        if name in used_wells:
            continue        # two source names for one address; pass 2 splits them
        region_to_well[key] = name
        used_wells.add(name)

    # Pass 2: the rest, deterministically, skipping everything pass 1 claimed.
    for key in source_wells:
        if key in region_to_well:
            continue
        region_to_well[key] = _next_synthetic_yokogawa_well(used_wells, n_wells)
        print(f"Well {key[1]!r} is not a plate address; converted as "
              f"{region_to_well[key]} (see {os.path.basename(csv_path)}).")

    # Process files per region
    for region, file_list in files_by_region.items():
        assigned_well = region_to_well[region[:2]]
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
    """Convert every image in ``folder`` to Yokogawa-style naming with a MIP.

    ND2, CZI, LIF and plain TIFF/PNG/JPEG inputs are detected by
    extension, max-projected over Z, and written out as
    ``plate<N>_<well>_T####F###L01C##.tif``. A ``rename_log.csv``
    records the original-to-new mapping.

    A file that cannot be read is skipped so the rest of the folder
    still converts — but the skip is recorded on a
    :class:`spacr.errors.RunLedger`, printed as a loud summary at the
    end, and **stamped into a sibling ``rename_log.run_status.json``**.
    That sidecar is what lets a later reader (or
    :func:`spacr.errors.run_is_complete`) tell that the converted
    folder is missing inputs, instead of quietly analysing a subset.

    :param folder: Directory of raw images, converted in place.
    :returns: the :class:`spacr.errors.RunLedger` for the conversion.
    """

    def _get_next_well(used_wells):
        """Return the next free well, filling one plate before the next.

        The well ids come from :func:`spacr.convert.well_sequence`, which
        builds them out of :data:`spacr.schema.PLATE_FORMATS` — one
        definition instead of the three copies of ``"ABCDEFGHIJKLMNOP"`` and
        ``range(1, 25)`` this module used to carry.

        The plate format stays 384 here: unlike
        :func:`convert_separate_files_to_yokogawa`, the inputs carry no well
        names at all, so nothing in them can ask for a bigger plate and the
        addresses are synthetic either way.
        """
        return _next_synthetic_yokogawa_well(used_wells, 384)

    filenames = []
    rename_log = []
    csv_path = os.path.join(folder, "rename_log.csv")
    used_wells = set()
    ledger = RunLedger('convert_to_yokogawa')

    # **Dictionary to store well assignments per original file**
    file_to_well = {}

    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        ext = file.lower().split('.')[-1]

        # **Assign a well only once per original file**
        if file not in file_to_well:
            file_to_well[file] = _get_next_well(used_wells)
            #used_wells.add(file_to_well[file])  # Mark it as used

        well = file_to_well[file]  # Use the same well for all channels/times

        ### **Process Nikon ND2 Files**
        if ext == 'nd2':
            with ledger.item(file, stage='nd2',
                             echo=f"Error processing ND2 file {file}"):
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
                                # np.max is a dispatcher, not a ufunc, so
                                # np.max.reduce raised AttributeError before a
                                # single frame was read: every ND2 silently
                                # produced no TIFF (and the IndexError handler
                                # below was dead code). np.maximum is the ufunc.
                                mip_image = np.maximum.reduce([
                                    nd2.get_frame_2D(t=t_idx, v=f_idx, z=z_idx, c=c_idx)
                                    for z_idx in z_levels
                                ], axis=0)

                                dtype = mip_image.dtype
                                filename = f"{well}_T{t_idx+1:04d}F{f_idx+1:03d}L01C{c_idx+1:02d}.tif"
                                filepath = os.path.join(folder, filename)

                                tifffile.imwrite(filepath, mip_image.astype(dtype))
                                rename_log.append({"Original File": file, 
                                                   "Renamed TIFF": filename,
                                                   "ext": ext,
                                                   "time": t_idx,
                                                   "field": f_idx,
                                                   "channel": channel,
                                                   "z": z_levels})

                            except IndexError as frame_err:
                                # A dropped frame silently shrinks the FOV set —
                                # record it as its own item so the summary shows
                                # how much of the ND2 never made it to disk.
                                ledger.record_failure(
                                    f"{file}:T{t_idx}F{f_idx}C{c_idx}",
                                    stage='nd2_frame', exc=frame_err)
                                print(f"Warning: ND2 file {file} has an incomplete data structure. Skipping.")

        elif ext == 'czi':
            with ledger.item(file, stage='czi',
                             echo=f"Error processing CZI file {file}"):
                # Open the CZI in streaming mode
                with pyczi.open_czi(path) as czidoc:

                    # 1) Global dimension ranges
                    bbox    = czidoc.total_bounding_box
                    _, tlen = bbox.get('T', (0,1))
                    _, clen = bbox.get('C', (0,1))
                    _, zlen = bbox.get('Z', (0,1))

                    # 2) Scene → list of scene indices
                    scenes_bb = czidoc.scenes_bounding_rectangle
                    scenes    = sorted(scenes_bb.keys()) if scenes_bb else [None]

                    # 3) Output folder (same as .czi)
                    folder = os.path.dirname(path)

                    # 4) Loop scene × time × channel × Z
                    for scene in scenes:
                        # *** assign a unique well for this scene ***
                        scene_well = _get_next_well(used_wells)

                        # Field index = scene+1 (or 1 if no scene)
                        F_idx = scene + 1 if scene is not None else 1
                        # Scene index for “A”
                        A_idx = scene + 1 if scene is not None else 1

                        for t in range(tlen):
                            for c in range(clen):
                                for z in range(zlen):
                                    # Read exactly one 2D plane
                                    arr = czidoc.read(
                                        plane={'T': t, 'C': c, 'Z': z},
                                        scene=scene
                                    )
                                    plane = np.squeeze(arr)

                                    # Build Yokogawa‐style filename:
                                    fn = (
                                        f"{scene_well}_"
                                        f"T{t+1:04d}"
                                        f"F{F_idx:03d}"
                                        f"L01"
                                        f"A{A_idx:02d}"
                                        f"Z{z+1:02d}"
                                        f"C{c+1:02d}.tif"
                                    )
                                    outpath = os.path.join(folder, fn)

                                    # Write with lossless compression
                                    tifffile.imwrite(
                                        outpath,
                                        plane.astype(plane.dtype),
                                        compression='zlib'
                                    )

                                    # Log it
                                    rename_log.append({
                                        "Original File": file,
                                        "Renamed TIFF": fn,
                                        "ext": ext,
                                        "scene": scene,
                                        "time": t,
                                        "slice": z,
                                        "field": F_idx,
                                        "channel": c,
                                        "well": scene_well
                                    })

        ### **Process Leica LIF Files**
        elif ext == 'lif':
            with ledger.item(file, stage='lif',
                             echo=f"Error processing LIF file {file}"):
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
                                except IndexError as frame_err:
                                    ledger.record_failure(
                                        f"{file}:T{t_idx}Z{z_idx}C{c_idx}",
                                        stage='lif_frame', exc=frame_err)
                                    print(f"Missing frame: T{t_idx}, Z{z_idx}, C{c_idx} in {file}, skipping frame.")

                            if z_stack:
                                mip_image = np.max(np.stack(z_stack), axis=0)
                                dtype = mip_image.dtype
                                filename = f"{well}_T{t_idx+1:04d}F{image_idx+1:03d}L01C{c_idx+1:02d}.tif"
                                filepath = os.path.join(folder, filename)

                                tifffile.imwrite(filepath, mip_image.astype(dtype))
                                rename_log.append({"Original File": file, "Renamed TIFF": filename})

        ### **Process Standard Image Files (TIFF, PNG, JPEG, BMP)**
        elif ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'] and not file.startswith("plate"):
            with ledger.item(file, stage='tiff',
                             echo=f"Error processing standard image file {file}"):
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
                        # The two leading axes are t and z, in an order the
                        # SHAPE cannot reveal. This used to assume TZYX
                        # unconditionally, so a genuine (Z, T, Y, X) file had
                        # every z-plane written out as a "timepoint" and every
                        # projection taken over TIME rather than over z - wrong
                        # data under a confident filename, with nothing saying
                        # so. tifffile records the real order; ask it, and when
                        # the file does not declare one, say which way it was
                        # read and what that means if it is wrong.
                        try:
                            axes = (tif.series[0].axes or '').upper()
                        except Exception:
                            axes = ''
                        t_axis = 0
                        if axes[:2] == 'ZT':
                            t_axis = 1
                        elif axes[:2] != 'TZ':
                            print(f"WARNING: {file} is 4-D but declares axes "
                                  f"{axes or '(none)'}; reading it as "
                                  f"(T, Z, Y, X). If it is really (Z, T, Y, X), "
                                  f"every timepoint written below is a z-plane "
                                  f"and every projection is over time.")
                        t_dim = images.shape[t_axis]
                        z_dim = images.shape[1 - t_axis]
                        for t in range(t_dim):
                            plane_stack = images[t] if t_axis == 0 else images[:, t]
                            mip_image = np.max(plane_stack, axis=0)
                            filename = f"{well}_T{t+1:04d}F001L01C01.tif"
                            tifffile.imwrite(os.path.join(folder, filename), mip_image)
                            rename_log.append({"Original File": file, "Renamed TIFF": filename})

                    else:
                        raise ValueError(f"Unsupported TIFF dimensions: {images.shape}")

    # Save rename log as CSV
    pd.DataFrame(rename_log).to_csv(csv_path, index=False)
    print(f"Processing complete. Files saved in {folder} and rename log saved as {csv_path}.")
    # Stamp the artifact, then say so last. rename_log.csv on its own cannot
    # tell you that three of the ten inputs never converted; the sidecar can.
    ledger.finalize(artifact=csv_path)
    return ledger


def apply_augmentation(image, method):
    """Return ``image`` transformed by a named geometric augmentation.

    :param image: NumPy image array.
    :param method: One of ``'rotate90'``, ``'rotate180'``, ``'rotate270'``,
        ``'flip_h'``, ``'flip_v'``; any other value returns the input
        unchanged.
    :returns: Augmented image array.
    """
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
    """Copy one image/mask pair described by ``entry``, applying an optional augmentation.

    :param entry: Dict with keys ``src_img``, ``src_msk``, ``dst_img``,
        ``dst_msk`` and ``augment`` (augmentation name or falsy).
    :returns: ``1`` on success — used for progress counting.
    """
    img = tifffile.imread(entry["src_img"])
    msk = tifffile.imread(entry["src_msk"])
    if entry["augment"]:
        img = apply_augmentation(img, entry["augment"])
        msk = apply_augmentation(msk, entry["augment"])
    tifffile.imwrite(entry["dst_img"], img)
    tifffile.imwrite(entry["dst_msk"], msk)
    return 1

def prepare_cellpose_dataset(input_root, augment_data=False, train_fraction=0.8, n_jobs=None):
    """Aggregate image/mask pairs from sibling dataset folders into a Cellpose training split.

    Discovers ``<input_root>/*/masks`` layouts, balances datasets to a
    common size (with augmentations if requested) and copies the
    selected pairs into ``<input_root>/cellpose_dataset/train`` and
    ``.../test``.

    :param input_root: Directory containing one subfolder per dataset.
    :param augment_data: If True, expand under-sized datasets by
        applying geometric augmentations. Default ``False``.
    :param train_fraction: Fraction of pairs routed to the train split.
        Default ``0.8``.
    :param n_jobs: Worker count for parallel copies. Default: CPU count.
    :returns: None
    :raises ValueError: if no valid ``<subdir>/masks`` datasets are found.
    """
    from .utils import print_progress

    time_ls = []
    input_root = os.path.abspath(input_root)
    output_root = os.path.join(input_root, "cellpose_dataset")

    def get_augmentations():
        """Return the list of augmentation names used to expand datasets."""
        return ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v']

    def find_image_mask_pairs(dataset_path):
        """Return ``(image_path, mask_path)`` pairs found under ``dataset_path``."""
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
        """Create ``train/{images,masks}`` and ``test/{images,masks}`` under ``base``."""
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
