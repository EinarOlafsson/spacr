spacr.io
========

.. py:module:: spacr.io






Module Contents
---------------

.. py:function:: process_non_tif_non_2D_images(folder)

   Process and standardize image files in a folder by converting or splitting them into grayscale TIFFs.

   This function supports various image formats (PNG, JPEG, CZI, ND2, TIFF) and ensures all output images
   are grayscale TIFF files saved with consistent naming based on dimensions (channel, z-plane, timepoint).

   For 2D grayscale images in non-TIFF formats, it converts them to TIFF.
   For 3D, 4D, or 5D images, it splits them into individual grayscale channels and saves them with suffixes
   (_C#, _Z#, _T#) to indicate channel, z-stack, and time point respectively.

   :param folder: Path to the folder containing image files to be processed.
   :type folder: str

   Supported file extensions:
       - .tif, .tiff
       - .png
       - .jpg, .jpeg
       - .czi
       - .nd2

   Output:
       - Saves standardized grayscale TIFF images in the same folder with descriptive filenames.
       - Prints a log message for each file processed or skipped.


.. py:class:: CombineLoaders(train_loaders)

   A class that combines multiple PyTorch data loaders into a single iterable.

   This class allows iteration over a mixed sequence of batches from several
   data loaders, yielding a tuple with the loader index and the corresponding batch.
   Once a loader is exhausted, it is removed from the iteration pool.

   :param train_loaders: A list of PyTorch DataLoader objects.
   :type train_loaders: list

   :raises StopIteration: When all data loaders are exhausted.

   Initialize the CombineLoaders instance.

   Converts each data loader into an iterator for independent traversal.

   :param train_loaders: List of torch.utils.data.DataLoader instances.
   :type train_loaders: list


   .. py:attribute:: train_loaders


   .. py:attribute:: loader_iters


   .. py:method:: __iter__()

      Return the iterator object (self).

      :returns: The iterator object itself.
      :rtype: CombineLoaders



   .. py:method:: __next__()

      Return the next batch from the available data loaders.

      Data loaders are shuffled at each step to randomize the batch source.
      If a data loader is exhausted, it is removed from the pool.

      :returns:

                A tuple (i, batch) where i is the index of the originating loader,
                       and batch is the next batch of data from that loader.
      :rtype: tuple

      :raises StopIteration: When all loaders have been exhausted.



.. py:class:: CombinedDataset(datasets, shuffle=True)

   Bases: :py:obj:`torch.utils.data.Dataset`


   A dataset that combines multiple datasets into one seamless dataset.

   This class supports optional shuffling across datasets and presents
   a unified indexing interface for training or evaluation.

   :param datasets: A list of PyTorch Dataset objects to combine.
   :type datasets: list
   :param shuffle: Whether to shuffle the indices for data access. Defaults to True.
   :type shuffle: bool, optional

   Initialize the CombinedDataset.

   Computes lengths of each dataset and optionally shuffles the access indices.

   :param datasets: A list of datasets to be combined.
   :type datasets: list
   :param shuffle: Whether to shuffle the combined dataset. Defaults to True.
   :type shuffle: bool, optional


   .. py:attribute:: datasets


   .. py:attribute:: lengths


   .. py:attribute:: total_length


   .. py:attribute:: shuffle
      :value: True



   .. py:method:: __getitem__(index)

      Retrieve an item from the combined dataset.

      The method accounts for shuffling and maps the index to the appropriate dataset.

      :param index: Index of the item in the combined dataset.
      :type index: int

      :returns: The item retrieved from the corresponding sub-dataset.
      :rtype: Any



   .. py:method:: __len__()

      Return the total length of the combined dataset.

      :returns: Total number of items across all datasets.
      :rtype: int



.. py:class:: NoClassDataset(data_dir, transform=None, shuffle=True, load_to_memory=False)

   Bases: :py:obj:`torch.utils.data.Dataset`


   A custom dataset class for handling image data without class labels.

   :param data_dir: The directory path where the image files are located.
   :type data_dir: str
   :param transform: A function/transform to apply to the image data. Default is None.
   :type transform: callable, optional
   :param shuffle: Whether to shuffle the dataset. Default is True.
   :type shuffle: bool, optional
   :param load_to_memory: Whether to load all images into memory. Default is False.
   :type load_to_memory: bool, optional


   .. py:attribute:: data_dir


   .. py:attribute:: transform
      :value: None



   .. py:attribute:: shuffle
      :value: True



   .. py:attribute:: load_to_memory
      :value: False



   .. py:attribute:: filenames


   .. py:method:: load_image(img_path)

      Load an image from the given file path.

      :param img_path: The file path of the image.
      :type img_path: str

      :returns: The loaded image.
      :rtype: PIL.Image



   .. py:method:: __len__()

      Get the total number of images in the dataset.

      :returns: The number of images in the dataset.
      :rtype: int



   .. py:method:: shuffle_dataset()

      Shuffle the dataset.



   .. py:method:: __getitem__(index)

      Get the image and its corresponding filename at the given index.

      :param index: The index of the image in the dataset.
      :type index: int

      :returns: A tuple containing the image and its filename.
      :rtype: tuple



.. py:class:: spacrDataset(data_dir, loader_classes, transform=None, shuffle=True, pin_memory=False, specific_files=None, specific_labels=None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Custom PyTorch Dataset for loading labeled image data organized by class folders or from specified file lists.

   This dataset supports loading images either from directory structures organized by class or from explicit
   file and label lists. It supports optional preloading of all images into memory for faster access.

   :param data_dir: Root directory containing subfolders for each class.
   :type data_dir: str
   :param loader_classes: List of class names corresponding to subfolder names in `data_dir`.
   :type loader_classes: list[str]
   :param transform: Transform to apply to images (e.g., torchvision transforms).
   :type transform: callable, optional
   :param shuffle: Whether to shuffle the dataset. Default is True.
   :type shuffle: bool
   :param pin_memory: If True, pre-load all images into memory using multiprocessing. Default is False.
   :type pin_memory: bool
   :param specific_files: Specific image file paths to load instead of scanning `data_dir`.
   :type specific_files: list[str], optional
   :param specific_labels: Corresponding labels for `specific_files`.
   :type specific_labels: list[int], optional

   Initialize the spacrDataset.

   Constructs the dataset either by scanning the data directory or using provided file paths and labels.
   Optionally shuffles and preloads images into memory.

   :param data_dir: Directory containing class subfolders.
   :type data_dir: str
   :param loader_classes: List of class names.
   :type loader_classes: list
   :param transform: Transform function to apply to images.
   :type transform: callable, optional
   :param shuffle: Whether to shuffle the dataset. Default is True.
   :type shuffle: bool
   :param pin_memory: Whether to preload images into memory. Default is False.
   :type pin_memory: bool
   :param specific_files: List of file paths to use directly.
   :type specific_files: list[str], optional
   :param specific_labels: List of labels corresponding to specific files.
   :type specific_labels: list[int], optional


   .. py:attribute:: data_dir


   .. py:attribute:: classes


   .. py:attribute:: transform
      :value: None



   .. py:attribute:: shuffle
      :value: True



   .. py:attribute:: pin_memory
      :value: False



   .. py:attribute:: filenames
      :value: []



   .. py:attribute:: labels
      :value: []



   .. py:method:: load_image(img_path)

      Load and return a single image with orientation correction.

      :param img_path: Path to the image file.
      :type img_path: str

      :returns: Loaded RGB image.
      :rtype: PIL.Image



   .. py:method:: __len__()

      Return the number of samples in the dataset.

      :returns: Total number of images.
      :rtype: int



   .. py:method:: shuffle_dataset()

      Shuffle the dataset filenames and labels in unison.



   .. py:method:: get_plate(filepath)

      Extract the plate identifier from the filename.

      :param filepath: Full path to the file.
      :type filepath: str

      :returns: Plate ID extracted from the filename.
      :rtype: str



   .. py:method:: __getitem__(index)

      Retrieve an image, its label, and the filename.

      :param index: Index of the image to retrieve.
      :type index: int

      :returns: (image, label, filename)
      :rtype: tuple



.. py:class:: spacrDataLoader(*args, preload_batches=1, **kwargs)

   Bases: :py:obj:`torch.utils.data.DataLoader`


   Custom DataLoader with background batch preloading support using multiprocessing.

   This class extends `torch.utils.data.DataLoader` and adds asynchronous background
   preloading of a specified number of batches using a separate process or in-place loading
   if `pin_memory=True`.

   :param \*args: Arguments passed to the base DataLoader.
   :param preload_batches: Number of batches to preload in a background process. Default is 1.
   :type preload_batches: int
   :param \*\*kwargs: Keyword arguments passed to the base DataLoader. Supports all standard DataLoader arguments.

   Initialize the spacrDataLoader.

   Sets up the queue and multiprocessing process for background preloading of batches.

   :param \*args: Arguments passed to torch.utils.data.DataLoader.
   :param preload_batches: Number of batches to preload. Default is 1.
   :type preload_batches: int
   :param \*\*kwargs: Keyword arguments passed to the base DataLoader.


   .. py:attribute:: preload_batches
      :value: 1



   .. py:attribute:: batch_queue


   .. py:attribute:: process
      :value: None



   .. py:attribute:: current_batch_index
      :value: 0



   .. py:attribute:: pin_memory


   .. py:method:: __iter__()

      Return the iterator and initiate background preloading.

      :returns: self



   .. py:method:: __next__()

      Return the next batch from the queue.

      If the queue is empty and the process has exited, raises StopIteration.

      :returns: The next batch of data.



   .. py:method:: cleanup()

      Cleanup method to terminate background preloading processes.

      Ensures graceful shutdown of worker processes at exit.



   .. py:method:: __del__()

      Destructor to ensure cleanup is called when the object is deleted.



.. py:class:: TarImageDataset(tar_path, transform=None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   A PyTorch Dataset for loading images directly from a .tar archive without extraction.

   This is useful for large datasets stored as compressed tar archives, enabling on-the-fly
   access to individual image files without unpacking the archive to disk.

   :param tar_path: Path to the .tar archive containing image files.
   :type tar_path: str
   :param transform: Optional transform to be applied on a sample.
   :type transform: callable, optional

   .. attribute:: members

      List of image members in the tar archive.

      :type: List[TarInfo]

   Initialize the dataset and index image members from the tar archive.

   :param tar_path: Path to the .tar file.
   :type tar_path: str
   :param transform: Transform function to apply to each image.
   :type transform: callable, optional


   .. py:attribute:: tar_path


   .. py:attribute:: transform
      :value: None



   .. py:method:: __len__()

      Return the number of image files in the archive.

      :returns: Number of image files.
      :rtype: int



   .. py:method:: __getitem__(idx)

      Retrieve an image by index directly from the tar archive.

      :param idx: Index of the image to retrieve.
      :type idx: int

      :returns: (PIL.Image.Image or transformed image, str) where the string is the file name.
      :rtype: tuple



.. py:function:: load_images_from_paths(images_by_key)

   Load images from a dictionary mapping keys to lists of image file paths.

   Each key in the input dictionary corresponds to a list of file paths. The function
   loads each image as a NumPy array and returns a new dictionary with the same keys,
   where each value is a list of loaded images.

   :param images_by_key: A dictionary where each key maps to a list of image file paths (str).
   :type images_by_key: dict

   :returns: A dictionary where each key maps to a list of NumPy arrays representing the loaded images.
   :rtype: dict

   .. rubric:: Notes

   - Images are loaded using PIL and converted to NumPy arrays.
   - Any image that fails to load will be skipped, and an error message will be printed.


.. py:function:: concatenate_and_normalize(src, channels, save_dtype=np.float32, settings={})

   Concatenates and normalizes channel data from multiple files and saves the normalized data.

   :param src: The source directory containing the channel data files.
   :type src: str
   :param channels: The list of channel indices to be concatenated and normalized.
   :type channels: list
   :param randomize: Whether to randomize the order of the files. Defaults to True.
   :type randomize: bool, optional
   :param timelapse: Whether the channel data is from a timelapse experiment. Defaults to False.
   :type timelapse: bool, optional
   :param batch_size: The number of files to be processed in each batch. Defaults to 100.
   :type batch_size: int, optional
   :param backgrounds: Background values for each channel. Defaults to [100, 100, 100].
   :type backgrounds: list, optional
   :param remove_backgrounds: Whether to remove background values for each channel. Defaults to [False, False, False].
   :type remove_backgrounds: list, optional
   :param lower_percentile: Lower percentile value for normalization. Defaults to 2.
   :type lower_percentile: int, optional
   :param save_dtype: Data type for saving the normalized stack. Defaults to np.float32.
   :type save_dtype: numpy.dtype, optional
   :param signal_to_noise: Signal-to-noise ratio thresholds for each channel. Defaults to [5, 5, 5].
   :type signal_to_noise: list, optional
   :param signal_thresholds: Signal thresholds for each channel. Defaults to [1000, 1000, 1000].
   :type signal_thresholds: list, optional

   :returns: The directory path where the concatenated and normalized channel data is saved.
   :rtype: str


.. py:function:: delete_empty_subdirectories(folder_path)

   Deletes all empty subdirectories in the specified folder.

   Args:
   - folder_path (str): The path to the folder in which to look for empty subdirectories.


.. py:function:: preprocess_img_data(settings)

   Preprocesses image data by converting z-stack images to maximum intensity projection (MIP) images.

   :param src: The source directory containing the z-stack images.
   :type src: str
   :param metadata_type: The type of metadata associated with the images. Defaults to 'cellvoyager'.
   :type metadata_type: str, optional
   :param custom_regex: The custom regular expression pattern used to match the filenames of the z-stack images. Defaults to None.
   :type custom_regex: str, optional
   :param cmap: The colormap used for plotting. Defaults to 'inferno'.
   :type cmap: str, optional
   :param figuresize: The size of the figure for plotting. Defaults to 15.
   :type figuresize: int, optional
   :param normalize: Whether to normalize the images. Defaults to False.
   :type normalize: bool, optional
   :param nr: The number of images to preprocess. Defaults to 1.
   :type nr: int, optional
   :param plot: Whether to plot the images. Defaults to False.
   :type plot: bool, optional
   :param mask_channels: The channels to use for masking. Defaults to [0, 1, 2].
   :type mask_channels: list, optional
   :param batch_size: The number of images to process in each batch. Defaults to [100, 100, 100].
   :type batch_size: list, optional
   :param timelapse: Whether the images are from a timelapse experiment. Defaults to False.
   :type timelapse: bool, optional
   :param remove_background: Whether to remove the background from the images. Defaults to False.
   :type remove_background: bool, optional
   :param backgrounds: The number of background images to use for background removal. Defaults to 100.
   :type backgrounds: int, optional
   :param lower_percentile: The lower percentile used for background removal. Defaults to 1.
   :type lower_percentile: float, optional
   :param save_dtype: The data type used for saving the preprocessed images. Defaults to np.float32.
   :type save_dtype: type, optional
   :param randomize: Whether to randomize the order of the images. Defaults to True.
   :type randomize: bool, optional
   :param all_to_mip: Whether to convert all images to MIP. Defaults to False.
   :type all_to_mip: bool, optional
   :param settings: Additional settings for preprocessing. Defaults to {}.
   :type settings: dict, optional

   :returns: None


.. py:function:: read_plot_model_stats(train_file_path, val_file_path, save=False)

.. py:function:: convert_numpy_to_tiff(folder_path, limit=None)

   Converts all .npy files in a folder to .tiff images and saves them in a 'tiff' subdirectory.

   This function searches for `.npy` files in the specified folder, loads each as a NumPy array,
   and writes it as a `.tiff` image using `tifffile.imwrite`. The resulting images are saved in
   a `tiff` subdirectory within the input folder. Optionally, processing can be limited to a
   specific number of files.

   :param folder_path: The path to the directory containing `.npy` files to be converted.
   :type folder_path: str
   :param limit: Maximum number of `.npy` files to convert. If None (default), all `.npy` files are converted.
   :type limit: int, optional

   :returns: The function saves the converted TIFF files to disk and prints status messages.
   :rtype: None


.. py:function:: generate_cellpose_train_test(src, test_split=0.1)

   Splits a directory of TIFF images and corresponding Cellpose masks into training and test sets.

   This function searches the `src` directory for TIFF images and ensures that corresponding
   masks exist in the `src/masks/` folder. It then shuffles and splits the dataset into
   training and test sets based on the specified `test_split` ratio. The resulting subsets
   are copied into `train/` and `test/` folders (with `masks/` subfolders) located
   in the parent directory of `src`.

   :param src: Path to the directory containing TIFF images and a subdirectory `masks/` with corresponding mask files.
   :type src: str
   :param test_split: Proportion of the dataset to be used for testing (default is 0.1, i.e., 10%).
   :type test_split: float, optional

   :returns: Files are copied to disk and progress messages are printed.
   :rtype: None


.. py:function:: parse_gz_files(folder_path)

   Parses the .fastq.gz files in the specified folder path and returns a dictionary
   containing the sample names and their corresponding file paths.

   :param folder_path: The path to the folder containing the .fastq.gz files.
   :type folder_path: str

   :returns: A dictionary where the keys are the sample names and the values are
             dictionaries containing the file paths for the 'R1' and 'R2' read directions.
   :rtype: dict


.. py:function:: generate_dataset(settings={})

   Generates a tar archive containing a dataset of images collected from one or more database sources.

   This function selects image paths from one or more SQLite databases, optionally samples from them,
   and writes the images into a tar archive using multiprocessing to parallelize the process. Temporary tar
   files are created and merged into a final tar file. The function also logs and saves dataset settings.

   :param settings: Dictionary of user-defined settings. The following keys are used:

                    - 'src' (str or list of str): Path(s) to the source folder(s) containing the database(s).
                    - 'experiment' (str): Name of the experiment, used to name the output tar.
                    - 'sample' (int or list, optional): If int, randomly sample that many images; if list, sample per src index.
                    - 'file_metadata' (str, optional): Metadata column name used to filter/select files.
   :type settings: dict, optional

   :returns: Path to the final tar archive containing the dataset.
   :rtype: str


.. py:function:: generate_loaders(src, mode='train', image_size=224, batch_size=32, classes=['nc', 'pc'], n_jobs=None, validation_split=0.0, pin_memory=False, normalize=False, channels=[1, 2, 3], augment=False, verbose=False)

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


.. py:function:: generate_training_dataset(settings)

   Generate a training dataset from a SQLite database using measurement-based or annotation/metadata-based selection.

   Depending on the `settings`, this function selects images corresponding to high and low phenotypes (e.g., recruitment)
   or based on metadata or manual annotation. Selected image paths are grouped by class and returned for further use
   (e.g., saving to folders or training models).

   :param settings: Configuration dictionary with the following required keys:

                    - 'class_metadata' (list of str): Metadata conditions to define classes (e.g., treatment names).
                    - 'channel_of_interest' (int): Channel index used for computing recruitment scores.
                    - 'png_type' (str): Type of PNG to retrieve ('raw', 'outline', etc.).
                    - 'size' (int): Number of images to sample per class.
                    - 'nuclei_limit' (int or None): Minimum nucleus size for filtering (used in _read_and_merge_data).
                    - 'pathogen_limit' (int or None): Minimum pathogen size for filtering.
                    - 'custom_measurement' (list of str or None): If provided, defines custom numerator and denominator columns.
                    - 'classes' (list of str): Treatments to annotate using `annotate_conditions`.
                    - 'metadata_type_by' (str): Column in the DB to use for metadata classification ('columnID' or 'rowID').
                    - 'tables' (list of str): Tables to extract from database, e.g., ['cell', 'nucleus'].
                    - 'dataset_mode' (str): Either 'annotation' or 'metadata'. Controls how class sizes are determined.
   :type settings: dict

   :returns: A list where each sublist contains paths to PNGs belonging to one class (e.g., low vs high recruitment).
   :rtype: list of list of str


.. py:function:: training_dataset_from_annotation(db_path, dst, annotation_column='test', annotated_classes=(1, 2))

   Extracts image paths from a database and groups them into class-based lists based on annotation values.

   This function reads from a SQLite database (`png_list` table), extracts image paths and corresponding
   class annotations, and groups them by the specified `annotated_classes`. If only one class is provided,
   it automatically generates a second class by sampling the remaining entries not in the target class
   to create a balanced binary dataset.

   :param db_path: Path to the SQLite database file containing the `png_list` table.
   :type db_path: str
   :param dst: Output path (currently unused in the function, included for compatibility with caller).
   :type dst: str
   :param annotation_column: Column name in the `png_list` table that contains class annotations.
   :type annotation_column: str, default='test'
   :param annotated_classes: Class labels to extract from the annotation column.
   :type annotated_classes: tuple of int, default=(1, 2)

   :returns: **class_paths** -- A list where each sublist contains the file paths for images belonging to one class.
             The number of sublists equals the number of unique classes returned.
   :rtype: list of list of str

   .. rubric:: Notes

   - If only one annotated class is provided, the function creates a balanced second class
     from non-annotated images.
   - This function does not copy or move any files — it only collects and returns path lists.
   - All path and annotation data is assumed to be stored in the `png_list` table of the SQLite DB.


.. py:function:: training_dataset_from_annotation_metadata(db_path, dst, annotation_column='test', annotated_classes=(1, 2), metadata_type_by='columnID', class_metadata=['c1', 'c2'])

   Extracts annotated image paths from a database, filtered by metadata location (row/column).

   This function reads image paths and annotations from a SQLite database (`png_list` table), filters them
   by metadata (either `row_name` or `column_name`), and organizes them into class-specific lists based on
   annotation values. If only one class is specified, the function samples a balanced second class from
   remaining entries.

   :param db_path: Path to the SQLite database containing the `png_list` table.
   :type db_path: str
   :param dst: Output directory (unused in this function but required for compatibility).
   :type dst: str
   :param annotation_column: The column name in `png_list` storing the annotation labels.
   :type annotation_column: str, default='test'
   :param annotated_classes: Annotation values to be used for splitting data into separate class groups.
   :type annotated_classes: tuple of int, default=(1, 2)
   :param metadata_type_by: Which metadata field to filter by — either 'rowID' (uses `row_name`) or 'columnID' (uses `column_name`).
   :type metadata_type_by: str, {'rowID', 'columnID'}, default='columnID'
   :param class_metadata: The metadata values to include (e.g., specific row or column identifiers to filter on).
   :type class_metadata: list of str, default=['c1', 'c2']

   :returns: **class_paths** -- A list where each sublist contains paths to images in one class.
   :rtype: list of list of str

   :raises ValueError: If `metadata_type_by` is not one of 'rowID' or 'columnID'.

   .. rubric:: Notes

   - If only one class is specified in `annotated_classes`, a second class is constructed by sampling
     from non-target annotations in the filtered set to ensure balanced class representation.
   - This function assumes that `png_path`, `annotation_column`, `row_name`, and `column_name` exist
     in the `png_list` table.


.. py:function:: generate_dataset_from_lists(dst, class_data, classes, test_split=0.1)

   Generates a train/test image dataset directory structure from class-wise path lists.

   This function creates `train` and `test` subdirectories under the given destination directory (`dst`)
   and copies the image files into class-specific folders after performing a train-test split.

   :param dst: Destination directory where the dataset will be created. Subdirectories for each class will be made under `train/` and `test/`.
   :type dst: str
   :param class_data: A list where each sublist contains paths to image files belonging to a specific class.
   :type class_data: list of list of str
   :param classes: Class names corresponding to the order of `class_data`.
   :type classes: list of str
   :param test_split: Proportion of data to be used for the test set. The remainder is used for training.
   :type test_split: float, default=0.1

   :returns: * **train_dir** (*str*) -- Path to the top-level training directory.
             * **test_dir** (*str*) -- Path to the top-level test directory.

   :raises ValueError: If the number of class labels does not match the number of class data lists.

   .. rubric:: Notes

   The train/test split is deterministic (random_state=42).
   File copying is timed and progress is reported via `print_progress`.


.. py:function:: convert_separate_files_to_yokogawa(folder, regex)

   Converts image files from a folder into Yokogawa-style naming format with optional MIP across Z-slices.

   This function parses filenames using a provided regex, extracts metadata such as well ID, channel, field,
   timepoint, and slice, and renames the images to the Yokogawa convention:
   `plateX_WELL_TttttFfffL01Ccc.tif`. If multiple Z-slices exist, it computes a maximum intensity projection (MIP).

   :param folder: Path to the folder containing input TIFF images.
   :type folder: str
   :param regex: Regular expression with named capture groups:
                 - 'plateID' (optional)
                 - 'wellID' (required)
                 - 'fieldID' (optional)
                 - 'timeID' (optional)
                 - 'chanID' (optional)
                 - 'sliceID' (optional)
   :type regex: str

   :returns: Saves renamed TIFF files and a CSV log (`rename_log.csv`) in the same folder.
   :rtype: None

   .. rubric:: Notes

   - Automatically assigns new well names (`plateX_WELL`) if missing or non-standard.
   - Groups images by region (plate, well, field, time, channel) and performs MIP if multiple slices are present.
   - Skips files that do not match the regex or are missing required metadata.


.. py:function:: convert_to_yokogawa(folder)

   Converts microscopy image files in a folder to Yokogawa-style TIFF filenames.

   This function processes raw microscopy images in various formats (ND2, CZI, LIF, TIFF, PNG, JPEG, BMP)
   and converts them into a standardized Yokogawa naming scheme using maximum intensity projections (MIPs).
   Each image is assigned a unique well location (e.g., plate1_A01) across one or more 384-well plates.
   The output files are saved in the same directory with renamed filenames. A CSV log is generated
   to track the mapping between original files and the renamed TIFFs.

   :param folder: Path to the directory containing the input microscopy files.
   :type folder: str
   :param Supported Formats:
   :param -----------------:
   :param - `.nd2`:
   :type - `.nd2`: Nikon ND2 format (processed using ND2Reader)
   :param - `.czi`:
   :type - `.czi`: Zeiss CZI format (processed using pyczi)
   :param - `.lif`:
   :type - `.lif`: Leica LIF format (processed using readlif)
   :param - `.tif`:
   :type - `.tif`: Image files (processed using tifffile)
   :param `.tiff`:
   :type `.tiff`: Image files (processed using tifffile)
   :param `.png`:
   :type `.png`: Image files (processed using tifffile)
   :param `.jpg`:
   :type `.jpg`: Image files (processed using tifffile)
   :param `.jpeg`:
   :type `.jpeg`: Image files (processed using tifffile)
   :param `.bmp`:
   :type `.bmp`: Image files (processed using tifffile)
   :param Behavior:
   :param --------:
   :param - Computes maximum intensity projections across Z-stacks.:
   :param - Generates Yokogawa-style filenames:
   :type - Generates Yokogawa-style filenames: `plateX_<WELL>_T####F###L01C##.tif`
   :param - Handles timepoints:
   :param Z-stacks:
   :param channels:
   :param fields:
   :param and scenes depending on format.:
   :param - Avoids reusing well positions across multiple files and scenes.:
   :param - Skips malformed or incomplete image structures.:
   :param - Logs all renamed output files to `rename_log.csv` in the same folder.:
   :param Output:
   :param ------:
   :param - Converted TIFF images saved in the input folder with Yokogawa-style filenames.:
   :param - A CSV log `rename_log.csv` containing columns: 'Original File', 'Renamed TIFF', 'ext', 'time', 'field', 'channel', 'z', 'scene', 'slice', 'well'

   .. rubric:: Notes

   - Requires `ND2Reader`, `pyczi`, `readlif`, `tifffile`, and `pandas`.
   - Handles multi-dimensional images (2D, 3D, 4D).
   - Images with unsupported dimensions or structure are skipped with warnings.

   .. rubric:: Example

   >>> convert_to_yokogawa("/path/to/raw_images")
   Processing complete. Files saved in /path/to/raw_images and rename log saved as rename_log.csv.


.. py:function:: apply_augmentation(image, method)

   Applies the specified augmentation method to the given image.

   :param image: The input image to be augmented.
   :type image: numpy.ndarray
   :param method: The augmentation method to apply. Supported methods are:
                  - 'rotate90': Rotates the image 90 degrees clockwise.
                  - 'rotate180': Rotates the image 180 degrees.
                  - 'rotate270': Rotates the image 90 degrees counterclockwise.
                  - 'flip_h': Flips the image horizontally.
                  - 'flip_v': Flips the image vertically.
   :type method: str

   :returns: The augmented image. If the method is not recognized,
             the original image is returned unchanged.
   :rtype: numpy.ndarray


.. py:function:: process_instruction(entry)

   Processes a single image/mask entry by reading, optionally augmenting, and saving both image and mask.

   :param entry:
                 A dictionary with the following keys:
                     - 'src_img' (str): Path to the source image file.
                     - 'src_msk' (str): Path to the source mask file.
                     - 'dst_img' (str): Path to save the processed image.
                     - 'dst_msk' (str): Path to save the processed mask.
                     - 'augment' (str or None): Augmentation identifier to apply (e.g., 'rotate90', 'flip', or None).
   :type entry: dict

   :returns: Returns 1 upon successful completion.
   :rtype: int


.. py:function:: prepare_cellpose_dataset(input_root, augment_data=False, train_fraction=0.8, n_jobs=None)

   Prepare a training and testing dataset for Cellpose from multiple subdirectories containing TIFF images and corresponding masks.

   This function scans all subfolders in `input_root` that contain a "masks/" directory, finds image-mask pairs,
   and splits them into train/test sets. Optionally, it augments data using rotation and flipping to balance dataset sizes
   across all datasets. All output is saved in a standardized format to a new "cellpose_dataset/" folder inside `input_root`.

   :param input_root: Path to the folder containing one or more datasets. Each dataset should have a 'masks/' subfolder with mask files
                      matching the TIFF filenames.
   :type input_root: str
   :param augment_data: If True, perform data augmentation (rotation/flipping) to increase or equalize the number of samples per dataset.
                        Default is False.
   :type augment_data: bool, optional
   :param train_fraction: Fraction of data to use for training. The rest will go to testing. Default is 0.8 (i.e., 80% train, 20% test).
   :type train_fraction: float, optional
   :param n_jobs: Number of parallel worker processes. If None, uses all available CPUs minus one.
   :type n_jobs: int or None, optional

   :returns: All output TIFFs are saved under `input_root/cellpose_dataset/train/` and `.../test/` folders with consistent naming.
             A progress bar is printed to track the status of preprocessing.
   :rtype: None


