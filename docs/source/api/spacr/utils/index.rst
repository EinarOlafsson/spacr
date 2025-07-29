spacr.utils
===========

.. py:module:: spacr.utils








Module Contents
---------------

.. py:data:: spacr_path

.. py:function:: filepaths_to_database(img_paths, settings, source_folder, crop_mode)

   Insert image paths and metadata into a SQLite database.

   :param img_paths: Image file paths to insert.
   :type img_paths: list of str
   :param settings: Configuration dictionary. Must contain a 'timelapse' key (bool).
   :type settings: dict
   :param source_folder: Folder containing the SQLite database.
   :type source_folder: str
   :param crop_mode: One of 'cell', 'nucleus', 'pathogen', or 'cytoplasm'.
   :type crop_mode: str

   :raises sqlite3.OperationalError: On database connection or write errors.


.. py:function:: activation_maps_to_database(img_paths, source_folder, settings)

   Store image paths and metadata in a SQLite database.

   :param img_paths: Paths to image files.
   :type img_paths: list of str
   :param source_folder: Folder where the database is stored.
   :type source_folder: str
   :param settings: Must include 'dataset' and 'cam_type' keys.
   :type settings: dict

   :raises sqlite3.OperationalError: On database errors.


.. py:function:: activation_correlations_to_database(df, img_paths, source_folder, settings)

   Save activation correlation data to a SQLite database.

   :param df: DataFrame with correlation data. Must include 'file_name'.
   :type df: pd.DataFrame
   :param img_paths: List of image file paths.
   :type img_paths: list
   :param source_folder: Folder where the database is stored.
   :type source_folder: str
   :param settings: Must include 'dataset' and 'cam_type' keys.
   :type settings: dict

   :raises sqlite3.OperationalError: On database errors.


.. py:function:: calculate_activation_correlations(inputs, activation_maps, file_names, manders_thresholds=[15, 50, 75])

   Calculates Pearson and Manders correlations between input image channels and activation map channels.

   :param inputs: A batch of input images, Tensor of shape (batch_size, channels, height, width)
   :param activation_maps: A batch of activation maps, Tensor of shape (batch_size, channels, height, width)
   :param file_names: List of file names corresponding to each image in the batch.
   :param manders_thresholds: List of intensity percentiles to calculate Manders correlation.

   :returns:

             A DataFrame with columns for pairwise correlations (Pearson and Manders)
                              between input channels and activation map channels.
   :rtype: df_correlations


.. py:function:: load_settings(csv_file_path, show=False, setting_key='setting_key', setting_value='setting_value')

   Convert a CSV file with 'settings_key' and 'settings_value' columns into a dictionary.
   Handles special cases where values are lists, tuples, booleans, None, integers, floats, and nested dictionaries.

   :param csv_file_path: The path to the CSV file.
   :type csv_file_path: str
   :param show: Whether to display the dataframe (for debugging).
   :type show: bool
   :param setting_key: The name of the column that contains the setting keys.
   :type setting_key: str
   :param setting_value: The name of the column that contains the setting values.
   :type setting_value: str

   :returns: A dictionary where 'settings_key' are the keys and 'settings_value' are the values.
   :rtype: dict


.. py:function:: save_settings(settings, name='settings', show=False)

   Save a dictionary of settings to a CSV file.

   This function takes a dictionary of settings, processes it, and saves it
   as a CSV file in a specified directory. It also provides an option to
   display the settings as a DataFrame.

   :param settings: A dictionary containing the settings to be saved.
   :type settings: dict
   :param name: The base name for the output CSV file. Defaults to 'settings'.
   :type name: str, optional
   :param show: If True, displays the settings as a DataFrame. Defaults to False.
   :type show: bool, optional

   :raises KeyError: If the 'src' key is not present in the settings dictionary.

   .. rubric:: Notes

   - If the 'src' key in the settings dictionary is a list, the first element
     is used as the source directory, and the file name is appended with '_list'.
   - If the 'test_mode' key exists in the settings dictionary, it is set to False.
   - If the 'plot' key exists in the settings dictionary, it is set to False.
   - A directory named 'settings' is created inside the source directory if it does not exist.
   - The settings are saved as a CSV file in the 'settings' directory.
   - The file path where the settings are saved is printed.

   .. rubric:: Example

   >>> settings = {
   ...     'src': '/path/to/source',
   ...     'test_mode': True,
   ...     'plot': True,
   ...     'param1': 42
   ... }
   >>> save_settings(settings, name='experiment1', show=True)


.. py:function:: print_progress(files_processed, files_to_process, n_jobs, time_ls=None, batch_size=None, operation_type='')

   Prints the progress of a file processing operation along with estimated time information.

   :param files_processed: The number of files processed so far. If a list is provided,
                           its unique length will be used.
   :type files_processed: int or list
   :param files_to_process: The total number of files to process. If a list is provided,
                            its unique length will be used.
   :type files_to_process: int or list
   :param n_jobs: The number of parallel jobs being used for processing.
   :type n_jobs: int
   :param time_ls: A list of time durations for processing batches or files. Used to
                   calculate average time and estimated time left. Defaults to None.
   :type time_ls: list, optional
   :param batch_size: The size of each batch being processed. If a list is
                      provided, its length will be used. Defaults to None.
   :type batch_size: int or list, optional
   :param operation_type: A string describing the type of operation being performed.
                          Defaults to an empty string.
   :type operation_type: str, optional

   :returns: This function prints the progress and time information to the console.
   :rtype: None

   .. rubric:: Notes

   - If `time_ls` is provided, the function calculates the average time per batch or file and
     estimates the time remaining for the operation.
   - If `batch_size` is provided, the function calculates the average time per image within
     a batch.
   - Handles cases where inputs are lists or non-integer types by converting them to integers
     or calculating their lengths.


.. py:function:: reset_mp()

   Resets the multiprocessing start method based on the operating system.
   On Windows, the start method is set to 'spawn' if it is not already set.
   On Linux and macOS (Darwin), the start method is set to 'fork' if it is not already set.
   This function ensures compatibility with the multiprocessing module by
   enforcing the appropriate start method for the current platform.
   .. note::

      - The function uses `get_start_method` to retrieve the current start method.
      - The `set_start_method` function is called with `force=True` to override
        the existing start method if necessary.

   :raises ValueError: If an invalid start method is encountered or if the start
       method cannot be set for some reason.


.. py:function:: is_multiprocessing_process(process)

   Check if the given process is a multiprocessing process.

   This function examines the command-line arguments of the provided process
   to determine if it is associated with Python's multiprocessing module.

   :param process: A psutil Process object representing the process to check.
   :type process: psutil.Process

   :returns: True if the process is a multiprocessing process, False otherwise.
   :rtype: bool

   :raises psutil.NoSuchProcess: If the process no longer exists.
   :raises psutil.AccessDenied: If access to the process information is denied.
   :raises psutil.ZombieProcess: If the process is a zombie process.


.. py:function:: close_file_descriptors()

   Closes all open file descriptors starting from 3 up to the soft limit
   of the maximum number of file descriptors allowed for the process.

   This function is useful for cleaning up resources by ensuring that
   no unnecessary file descriptors remain open. It skips standard input,
   output, and error (file descriptors 0, 1, and 2).

   Exceptions during the closing of file descriptors are caught and ignored.

   .. note::

      The function uses the `resource` module to retrieve the soft limit
      for the maximum number of file descriptors.

   :raises OSError: If an error occurs while closing a file descriptor, it is
   :raises caught and ignored.:


.. py:function:: close_multiprocessing_processes()

   Close all multiprocessing processes and clean up associated resources.
   This function iterates through all running processes and terminates any
   that are identified as multiprocessing processes, excluding the current
   process. It waits for up to 5 seconds for each process to terminate and
   logs the termination status. Additionally, it handles exceptions that may
   occur during the termination process, such as access denial or the process
   no longer existing.
   After terminating the processes, it ensures that any open file descriptors
   are properly closed.
   .. note::

      This function relies on the `psutil` library to inspect and manage
      processes, and assumes the existence of helper functions:
      - `is_multiprocessing_process(proc)`: Determines if a process is a
        multiprocessing process.
      - `close_file_descriptors()`: Closes any open file descriptors.


.. py:function:: check_mask_folder(src, mask_fldr)

.. py:function:: smooth_hull_lines(cluster_data)

   Smooths the convex hull of a set of 2D points using spline interpolation.

   :param cluster_data: A 2D array of shape (n_points, 2) representing
                        the coordinates of the points in the cluster.
   :type cluster_data: numpy.ndarray

   :returns:

             Two 1D numpy arrays representing the x and y coordinates of the smoothed
                    convex hull, respectively. Each array contains 100 points.
   :rtype: tuple


.. py:function:: mask_object_count(mask)

   Counts the number of objects in a given mask.

   Args:
   - mask: numpy.ndarray. The mask containing object labels.

   Returns:
   - int. The number of objects in the mask.


.. py:function:: is_list_of_lists(var)

.. py:function:: normalize_to_dtype(array, p1=2, p2=98, percentile_list=None, new_dtype=None)

   Normalize each image in the stack to its own percentiles.

   Args:
   - array: numpy array
   The input stack to be normalized.
   - p1: int, optional
   The lower percentile value for normalization. Default is 2.
   - p2: int, optional
   The upper percentile value for normalization. Default is 98.
   - percentile_list: list, optional
   A list of pre-calculated percentiles for each image in the stack. Default is None.

   Returns:
   - new_stack: numpy array
   The normalized stack with the same shape as the input stack.


.. py:function:: annotate_conditions(df, cells=None, cell_loc=None, pathogens=None, pathogen_loc=None, treatments=None, treatment_loc=None)

   Annotates conditions in a DataFrame based on specified criteria and combines them into a 'condition' column.
   NaN is used for missing values, and they are excluded from the 'condition' column.

   :param df: The DataFrame to annotate.
   :type df: pandas.DataFrame
   :param cells: Host cell types. Defaults to None.
   :type cells: list/str, optional
   :param cell_loc: Values for each host cell type. Defaults to None.
   :type cell_loc: list of lists, optional
   :param pathogens: Pathogens. Defaults to None.
   :type pathogens: list/str, optional
   :param pathogen_loc: Values for each pathogen. Defaults to None.
   :type pathogen_loc: list of lists, optional
   :param treatments: Treatments. Defaults to None.
   :type treatments: list/str, optional
   :param treatment_loc: Values for each treatment. Defaults to None.
   :type treatment_loc: list of lists, optional

   :returns: Annotated DataFrame with a combined 'condition' column.
   :rtype: pandas.DataFrame


.. py:class:: Cache(max_size)

   A class representing a cache with a maximum size.

   :param max_size: The maximum size of the cache.
   :type max_size: int


   .. py:attribute:: cache


   .. py:attribute:: max_size


   .. py:method:: get(key)


   .. py:method:: put(key, value)


.. py:class:: ScaledDotProductAttention_v1(d_k)

   Bases: :py:obj:`torch.nn.Module`


   Scaled Dot-Product Attention module.

   :param d_k: The dimension of the key and query vectors.
   :type d_k: int

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:attribute:: d_k


   .. py:method:: forward(Q, K, V)

      Performs the forward pass of the attention mechanism.

      :param Q: The query tensor of shape (batch_size, seq_len_q, d_k).
      :type Q: torch.Tensor
      :param K: The key tensor of shape (batch_size, seq_len_k, d_k).
      :type K: torch.Tensor
      :param V: The value tensor of shape (batch_size, seq_len_v, d_k).
      :type V: torch.Tensor

      :returns: The output tensor of shape (batch_size, seq_len_q, d_k).
      :rtype: torch.Tensor



.. py:class:: SelfAttention_v1(in_channels, d_k)

   Bases: :py:obj:`torch.nn.Module`


   Self-Attention module that applies scaled dot-product attention mechanism.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param d_k: Dimensionality of the key and query vectors.
   :type d_k: int

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:attribute:: W_q


   .. py:attribute:: W_k


   .. py:attribute:: W_v


   .. py:attribute:: attention


   .. py:method:: forward(x)

      Forward pass of the SelfAttention module.

      :param x: Input tensor of shape (batch_size, in_channels).
      :type x: torch.Tensor

      :returns: Output tensor of shape (batch_size, d_k).
      :rtype: torch.Tensor



.. py:class:: ScaledDotProductAttention(d_k)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   Initializes the ScaledDotProductAttention module.

   :param d_k: The dimension of the key and query vectors.
   :type d_k: int


   .. py:attribute:: d_k


   .. py:method:: forward(Q, K, V)

      Performs the forward pass of the ScaledDotProductAttention module.

      :param Q: The query tensor.
      :type Q: torch.Tensor
      :param K: The key tensor.
      :type K: torch.Tensor
      :param V: The value tensor.
      :type V: torch.Tensor

      :returns: The output tensor.
      :rtype: torch.Tensor



.. py:class:: SelfAttention(in_channels, d_k)

   Bases: :py:obj:`torch.nn.Module`


   Self-Attention module that applies scaled dot-product attention mechanism.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param d_k: Dimensionality of the key and query vectors.
   :type d_k: int

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:attribute:: W_q


   .. py:attribute:: W_k


   .. py:attribute:: W_v


   .. py:attribute:: attention


   .. py:method:: forward(x)

      Forward pass of the SelfAttention module.

      :param x: Input tensor of shape (batch_size, in_channels).
      :type x: torch.Tensor

      :returns: Output tensor after applying self-attention mechanism.
      :rtype: torch.Tensor



.. py:class:: EarlyFusion(in_channels)

   Bases: :py:obj:`torch.nn.Module`


   Early Fusion module for image classification.

   :param in_channels: Number of input channels.
   :type in_channels: int

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:attribute:: conv1


   .. py:method:: forward(x)

      Forward pass of the Early Fusion module.

      :param x: Input tensor of shape (batch_size, in_channels, height, width).
      :type x: torch.Tensor

      :returns: Output tensor of shape (batch_size, 64, height, width).
      :rtype: torch.Tensor



.. py:class:: SpatialAttention(kernel_size=7)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   Initializes the SpatialAttention module.

   :param kernel_size: The size of the convolutional kernel. Default is 7.
   :type kernel_size: int


   .. py:attribute:: conv1


   .. py:attribute:: sigmoid


   .. py:method:: forward(x)

      Performs forward pass of the SpatialAttention module.

      :param x: The input tensor.
      :type x: torch.Tensor

      :returns: The output tensor after applying spatial attention.
      :rtype: torch.Tensor



.. py:class:: MultiScaleBlockWithAttention(in_channels, out_channels)

   Bases: :py:obj:`torch.nn.Module`


   A PyTorch module implementing a multi-scale convolutional block with spatial attention.

   This module applies a dilated convolution followed by a spatial attention mechanism
   to enhance input feature maps.

   Initialize the MultiScaleBlockWithAttention module.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param out_channels: Number of output channels after convolution.
   :type out_channels: int


   .. py:attribute:: dilated_conv1


   .. py:attribute:: spatial_attention


   .. py:method:: custom_forward(x)

      Apply dilated convolution followed by spatial attention.

      :param x: Input tensor of shape (N, C, H, W).
      :type x: torch.Tensor

      :returns: Output tensor after attention-enhanced feature transformation.
      :rtype: torch.Tensor



   .. py:method:: forward(x)

      Standard forward pass that delegates to `custom_forward`.

      :param x: Input tensor.
      :type x: torch.Tensor

      :returns: Output tensor.
      :rtype: torch.Tensor



.. py:class:: CustomCellClassifier(num_classes, pathogen_channel, use_attention, use_checkpoint, dropout_rate)

   Bases: :py:obj:`torch.nn.Module`


   A custom neural network module for single-cell classification using early fusion and multi-scale attention.

   This architecture supports optional gradient checkpointing for reduced memory usage during training.

   Initialize the classifier with early fusion and attention blocks.

   :param num_classes: Number of classification categories.
   :type num_classes: int
   :param pathogen_channel: Unused; reserved for future feature fusion.
   :type pathogen_channel: int
   :param use_attention: Unused; reserved for attention gating.
   :type use_attention: bool
   :param use_checkpoint: Whether to use checkpointing to reduce memory.
   :type use_checkpoint: bool
   :param dropout_rate: Unused; reserved for future regularization.
   :type dropout_rate: float


   .. py:attribute:: early_fusion


   .. py:attribute:: multi_scale_block_1


   .. py:attribute:: fc1


   .. py:attribute:: use_checkpoint


   .. py:method:: custom_forward(x)

      Perform a standard forward pass without gradient checkpointing.

      :param x: Input tensor of shape (N, C, H, W).
      :type x: torch.Tensor

      :returns: Logits for each class.
      :rtype: torch.Tensor



   .. py:method:: forward(x)

      Forward pass with optional gradient checkpointing.

      :param x: Input tensor.
      :type x: torch.Tensor

      :returns: Output logits.
      :rtype: torch.Tensor



.. py:class:: TorchModel(model_name='resnet50', pretrained=True, dropout_rate=None, use_checkpoint=False)

   Bases: :py:obj:`torch.nn.Module`


   A PyTorch wrapper for pretrained torchvision models with a custom SPACR classifier head.

   This class supports custom dropout insertion and optional gradient checkpointing
   for memory efficiency during training.

   Initialize the TorchModel with optional dropout and checkpointing.

   :param model_name: The model architecture to load from torchvision.models.
   :type model_name: str
   :param pretrained: Whether to initialize with pretrained weights.
   :type pretrained: bool
   :param dropout_rate: Dropout rate for the classifier head.
   :type dropout_rate: float or None
   :param use_checkpoint: Whether to enable gradient checkpointing.
   :type use_checkpoint: bool


   .. py:attribute:: model_name
      :value: 'resnet50'



   .. py:attribute:: use_checkpoint
      :value: False



   .. py:attribute:: base_model


   .. py:attribute:: num_ftrs


   .. py:method:: apply_dropout_rate(model, dropout_rate)

      Recursively set dropout probability for all nn.Dropout layers in the model.

      :param model: The model to modify.
      :type model: nn.Module
      :param dropout_rate: New dropout probability.
      :type dropout_rate: float



   .. py:method:: init_base_model(pretrained)

      Load the base model from torchvision.models.

      :param pretrained: Whether to load pretrained weights.
      :type pretrained: bool

      :returns: The base feature extractor.
      :rtype: nn.Module



   .. py:method:: get_weight_choice()

      Get the default weights enum for the selected model.

      :returns: Default weights or None.
      :rtype: Optional[torchvision.models.WeightsEnum]



   .. py:method:: get_num_ftrs()

      Determine output feature dimensionality from the base model.

      :returns: Feature vector size.
      :rtype: int



   .. py:method:: init_spacr_classifier(dropout_rate)

      Create the final classification layer and optional dropout.

      :param dropout_rate: Dropout probability. If None, dropout is skipped.
      :type dropout_rate: float or None



   .. py:method:: forward(x)

      Forward pass through base model, optional dropout, and final classifier.

      :param x: Input image tensor of shape (N, 3, H, W).
      :type x: torch.Tensor

      :returns: Output logits of shape (N,).
      :rtype: torch.Tensor



.. py:class:: FocalLossWithLogits(alpha=1, gamma=2)

   Bases: :py:obj:`torch.nn.Module`


   Focal Loss with logits for binary classification.

   This loss function is especially useful for addressing class imbalance by focusing more
   on hard-to-classify examples.

   :param alpha: Balancing factor for positive/negative examples. Default is 1.
   :type alpha: float
   :param gamma: Focusing parameter that down-weights easy examples. Default is 2.
   :type gamma: float

   Initialize the focal loss.

   :param alpha: Balancing factor for positive/negative examples. Default is 1.
   :type alpha: float
   :param gamma: Focusing parameter to down-weight well-classified examples. Default is 2.
   :type gamma: float


   .. py:attribute:: alpha
      :value: 1



   .. py:attribute:: gamma
      :value: 2



   .. py:method:: forward(logits, target)

      Compute the focal loss between logits and targets.

      :param logits: Predicted unnormalized scores (logits).
      :type logits: torch.Tensor
      :param target: Ground truth binary labels (same shape as logits).
      :type target: torch.Tensor

      :returns: Scalar focal loss value.
      :rtype: torch.Tensor



.. py:class:: ResNet(resnet_type='resnet50', dropout_rate=None, use_checkpoint=False, init_weights='imagenet')

   Bases: :py:obj:`torch.nn.Module`


   A wrapper around torchvision ResNet models with optional dropout, checkpointing,
   and a custom classifier head.

   Supported ResNet variants: resnet18, resnet34, resnet50, resnet101, resnet152.

   Initialize the ResNet model wrapper.

   :param resnet_type: Which ResNet variant to use. Options: 'resnet18', 'resnet34', etc.
   :type resnet_type: str
   :param dropout_rate: Dropout rate to apply before the final layer.
   :type dropout_rate: float or None
   :param use_checkpoint: Whether to enable gradient checkpointing.
   :type use_checkpoint: bool
   :param init_weights: Either 'imagenet' to load pretrained weights or 'none'.
   :type init_weights: str


   .. py:method:: initialize_base(base_model_dict, dropout_rate, use_checkpoint, init_weights)

      Initialize the base model and classifier layers.

      :param base_model_dict: Contains model constructor and weight enum.
      :type base_model_dict: dict
      :param dropout_rate: Dropout rate to use.
      :type dropout_rate: float or None
      :param use_checkpoint: Whether to use gradient checkpointing.
      :type use_checkpoint: bool
      :param init_weights: Weight initialization mode.
      :type init_weights: str



   .. py:method:: forward(x)

      Forward pass through ResNet and classification layers.

      :param x: Input tensor of shape (N, 3, H, W).
      :type x: torch.Tensor

      :returns: Logits of shape (N,).
      :rtype: torch.Tensor



.. py:function:: split_my_dataset(dataset, split_ratio=0.1)

   Splits a dataset into training and validation subsets.

   :param dataset: The dataset to be split.
   :type dataset: torch.utils.data.Dataset
   :param split_ratio: The ratio of validation samples to total samples. Defaults to 0.1.
   :type split_ratio: float, optional

   :returns: A tuple containing the training dataset and validation dataset.
   :rtype: tuple


.. py:function:: classification_metrics(all_labels, prediction_pos_probs, loss, epoch)

   Calculate classification metrics for binary classification.

   Args:
   - all_labels (list): List of true labels.
   - prediction_pos_probs (list): List of predicted positive probabilities.
   - loader_name (str): Name of the data loader.
   - loss (float): Loss value.
   - epoch (int): Epoch number.

   Returns:
   - data_df (DataFrame): DataFrame containing the calculated metrics.


.. py:function:: compute_irm_penalty(losses, dummy_w, device)

   Computes the Invariant Risk Minimization (IRM) penalty.

   :param losses: A list of losses.
   :type losses: list
   :param dummy_w: A dummy weight tensor.
   :type dummy_w: torch.Tensor
   :param device: The device to perform computations on.
   :type device: torch.device

   :returns: The computed IRM penalty.
   :rtype: float


.. py:function:: choose_model(model_type, device, init_weights=True, dropout_rate=0, use_checkpoint=False, channels=3, height=224, width=224, chan_dict=None, num_classes=2, verbose=False)

   Choose a model for classification.

   :param model_type: The type of model to choose. Can be one of the pre-defined TorchVision models or 'custom' for a custom model.
   :type model_type: str
   :param device: The device to use for model inference.
   :type device: str
   :param init_weights: Whether to initialize the model with pre-trained weights. Defaults to True.
   :type init_weights: bool, optional
   :param dropout_rate: The dropout rate to use in the model. Defaults to 0.
   :type dropout_rate: float, optional
   :param use_checkpoint: Whether to use checkpointing during model training. Defaults to False.
   :type use_checkpoint: bool, optional
   :param channels: The number of input channels for the model. Defaults to 3.
   :type channels: int, optional
   :param height: The height of the input images for the model. Defaults to 224.
   :type height: int, optional
   :param width: The width of the input images for the model. Defaults to 224.
   :type width: int, optional
   :param chan_dict: A dictionary containing channel information for custom models. Defaults to None.
   :type chan_dict: dict, optional
   :param num_classes: The number of output classes for the model. Defaults to 2.
   :type num_classes: int, optional

   :returns: The chosen model.
   :rtype: torch.nn.Module


.. py:function:: calculate_loss(output, target, loss_type='binary_cross_entropy_with_logits')

   Calculates the loss between the model output and the target based on the specified loss type.

   :param output: The predicted output from the model.
   :type output: Tensor
   :param target: The ground truth target values.
   :type target: Tensor
   :param loss_type: The type of loss function to use.
                     Supported values are:
                     - 'binary_cross_entropy_with_logits': Uses binary cross-entropy loss with logits.
                     - 'focal_loss': Uses focal loss with logits. Defaults to 'binary_cross_entropy_with_logits'.
   :type loss_type: str, optional

   :returns: The computed loss value.
   :rtype: Tensor

   :raises ValueError: If an unsupported loss_type is provided.


.. py:function:: pick_best_model(src)

   Selects the best model file from a given directory based on accuracy and epoch.
   This function scans the specified directory for files with a `.pth` extension,
   extracts accuracy and epoch information from their filenames using a predefined
   pattern, and selects the file with the highest accuracy. If multiple files have
   the same accuracy, the one with the highest epoch is selected.
   :param src: The path to the directory containing the model files.
   :type src: str

   :returns: The full path to the best model file based on accuracy and epoch.
   :rtype: str

   .. rubric:: Notes

   - The filenames are expected to follow the pattern `_epoch_<epoch>_acc_<accuracy>.pth`,
     where `<epoch>` is an integer and `<accuracy>` is a float.
   - If no files match the pattern, the function may raise an IndexError when
     attempting to access the first element of the sorted list.


.. py:function:: get_paths_from_db(df, png_df, image_type='cell_png')

   Filters and retrieves paths from a DataFrame based on specified criteria.

   :param df: A DataFrame whose index contains the objects of interest.
   :type df: pd.DataFrame
   :param png_df: A DataFrame containing a 'png_path' column and a 'prcfo' column.
   :type png_df: pd.DataFrame
   :param image_type: A string to filter the 'png_path' column. Defaults to 'cell_png'.
   :type image_type: str, optional

   :returns:

             A filtered DataFrame containing rows from `png_df` where the 'png_path'
                           column contains the `image_type` string and the 'prcfo' column matches
                           the index of `df`.
   :rtype: pd.DataFrame


.. py:function:: save_file_lists(dst, data_set, ls)

   Saves a list of file paths or data entries to a CSV file.

   :param dst: The destination directory where the CSV file will be saved.
   :type dst: str
   :param data_set: The name of the dataset, which will also be used as the column name in the CSV file and the filename.
   :type data_set: str
   :param ls: A list of file paths or data entries to be saved.
   :type ls: list

   :returns: None


.. py:function:: augment_single_image(args)

   Augment a single image by applying various transformations and saving the results.

   This function reads an image from the specified file path, applies a series of
   transformations (original, rotations, and flips), and saves the transformed images
   to the destination directory with appropriate filenames.

   :param args: A tuple containing:
                img_path (str): The file path to the input image.
                dst (str): The destination directory where the augmented images will be saved.
   :type args: tuple

   .. rubric:: Notes

   The following transformations are applied to the input image:
   - Original image (no transformation)
   - 90-degree clockwise rotation
   - 180-degree rotation
   - 270-degree clockwise rotation
   - Horizontal flip
   - Vertical flip

   Side Effects:
       Saves the augmented images to the specified destination directory.
       Filenames indicate the type of transformation applied.


.. py:function:: augment_images(file_paths, dst)

   Augments a list of images and saves the augmented images to the specified destination directory.

   :param file_paths: A list of file paths to the images to be augmented.
   :type file_paths: list of str
   :param dst: The destination directory where the augmented images will be saved.
               If the directory does not exist, it will be created.
   :type dst: str

   :returns: None


.. py:function:: augment_classes(dst, nc, pc, generate=True, move=True)

.. py:function:: annotate_predictions(csv_loc)

   Reads a CSV file containing image metadata, processes the data to extract
   additional information, and assigns a condition label to each row based on
   specific rules.
   :param csv_loc: The file path to the CSV file containing the metadata.
   :type csv_loc: str

   :returns:

             A DataFrame with the following additional columns:
                 - 'filename': Extracted filename from the 'path' column.
                 - 'plateID': Plate ID extracted from the filename.
                 - 'well': Well information extracted from the filename.
                 - 'fieldID': Field ID extracted from the filename.
                 - 'object': Object ID extracted from the filename (with '.png' removed).
                 - 'cond': Assigned condition label ('screen', 'pc', 'nc', or '') based on rules.
   :rtype: pandas.DataFrame


.. py:function:: initiate_counter(counter_, lock_)

   Initializes global variables `counter` and `lock` with the provided arguments.

   This function sets the global variables `counter` and `lock` to the values
   passed as `counter_` and `lock_`, respectively. It is typically used to
   share a counter and a lock object across multiple threads or processes.

   :param counter_: The counter object to be assigned to the global `counter`.
   :type counter_: Any
   :param lock_: The lock object to be assigned to the global `lock`.
   :type lock_: Any


.. py:function:: add_images_to_tar(paths_chunk, tar_path, total_images)

   Adds a chunk of image files to a tar archive.

   :param paths_chunk: A list of file paths to the images to be added to the tar archive.
   :type paths_chunk: list of str
   :param tar_path: The path where the tar archive will be created or overwritten.
   :type tar_path: str
   :param total_images: The total number of images being processed, used for progress tracking.
   :type total_images: int

   Behavior:
       - Opens a tar archive at the specified `tar_path` in write mode.
       - Iterates through the provided `paths_chunk` and adds each image to the tar archive.
       - Tracks progress using a shared counter and prints progress updates every 10 images.
       - Handles missing files gracefully by printing a warning message if a file is not found.

   .. rubric:: Notes

   - This function assumes the existence of a global `lock` object for thread-safe counter updates.
   - The `counter` object is expected to be a shared multiprocessing.Value or similar.
   - The `print_progress` function is used to display progress updates.

   Exceptions:
       - Prints a warning message if a file in `paths_chunk` is not found, but continues processing other files.

   .. rubric:: Example

   add_images_to_tar(['/path/to/image1.jpg', '/path/to/image2.jpg'], '/path/to/archive.tar', 100)


.. py:function:: generate_fraction_map(df, gene_column, min_frequency=0.0)

   Generates a fraction map from a given DataFrame and writes it to a CSV file.

   This function calculates the fraction of counts for each gene and well,
   organizes the data into a pivot table-like structure, and filters out
   columns based on a minimum frequency threshold. The resulting DataFrame
   is saved as a CSV file.

   :param df: Input DataFrame containing the data. It must include
              the columns 'count', 'well_read_sum', 'prc', and the specified
              `gene_column`.
   :type df: pd.DataFrame
   :param gene_column: The name of the column in `df` that contains gene
                       identifiers.
   :type gene_column: str
   :param min_frequency: The minimum frequency threshold for
                         filtering columns. Columns with a maximum value below this
                         threshold are dropped. Defaults to 0.0.
   :type min_frequency: float, optional

   :returns: A DataFrame containing the fraction map, with wells as
             rows and genes as columns. Cells contain the fraction values, and
             missing values are filled with 0.0.
   :rtype: pd.DataFrame


.. py:function:: fishers_odds(df, threshold=0.5, phenotyp_col='mean_pred')

   Perform Fisher's exact test to evaluate the association between mutants and a binned phenotype score.

   :param df: DataFrame containing binary mutant indicators and a phenotype score column.
   :type df: pandas.DataFrame
   :param threshold: Threshold to bin the phenotype score. Defaults to 0.5.
   :type threshold: float, optional
   :param phenotyp_col: Name of the column containing phenotype scores. Defaults to 'mean_pred'.
   :type phenotyp_col: str, optional

   :returns: DataFrame with columns 'Mutant', 'OddsRatio', 'PValue', and 'AdjustedPValue'.
   :rtype: pandas.DataFrame


.. py:function:: model_metrics(model)

   Calculate and display additional metrics and generate diagnostic plots for a given model.

   Args:
   -----------
   model : statsmodels.regression.linear_model.RegressionResultsWrapper
       A fitted regression model object from the statsmodels library.

   Metrics Calculated:
   --------------------
   - Root Mean Squared Error (RMSE): Measures the standard deviation of residuals.
   - Mean Absolute Error (MAE): Measures the average magnitude of residuals.
   - Durbin-Watson: Tests for the presence of autocorrelation in residuals.

   Diagnostic Plots:
   ------------------
   1. Residuals vs. Fitted: Scatter plot to check for non-linearity or unequal error variance.
   2. Histogram of Residuals: Distribution of residuals to check for normality.
   3. QQ Plot: Quantile-Quantile plot to assess if residuals follow a normal distribution.
   4. Scale-Location: Scatter plot of standardized residuals to check for homoscedasticity.

   Notes:
   ------
   - This function uses matplotlib and seaborn for plotting.
   - Ensure that the input model is fitted and contains the necessary attributes like `resid`,
     `fittedvalues`, and `mse_resid`.

   Returns:
   --------
   None


.. py:function:: check_multicollinearity(x)

   Checks multicollinearity of the predictors by computing the Variance Inflation Factor (VIF).

   :param x: A DataFrame containing the predictor variables.
   :type x: pd.DataFrame

   :returns:

             A DataFrame with two columns:
                 - 'Variable': The names of the predictor variables.
                 - 'VIF': The Variance Inflation Factor for each predictor variable.
                   A VIF value greater than 10 indicates high multicollinearity.
   :rtype: pd.DataFrame


.. py:function:: lasso_reg(merged_df, alpha_value=0.01, reg_type='lasso')

   Perform Lasso or Ridge regression on the input DataFrame.

   :param merged_df: DataFrame with columns 'gene', 'grna', 'plateID', 'rowID', 'columnID', and 'pred'.
   :type merged_df: pandas.DataFrame
   :param alpha_value: Regularization strength. Defaults to 0.01.
   :type alpha_value: float, optional
   :param reg_type: Type of regression to perform, either 'lasso' or 'ridge'. Defaults to 'lasso'.
   :type reg_type: str, optional

   :returns: DataFrame with 'Feature' and 'Coefficient' columns.
   :rtype: pandas.DataFrame


.. py:function:: MLR(merged_df, refine_model)

   Perform multiple linear regression (MLR) and extract interaction coefficients.

   :param merged_df: Input DataFrame with data for regression analysis.
   :type merged_df: pd.DataFrame
   :param refine_model: Whether to remove outliers before refitting the model.
   :type refine_model: bool

   :returns:

             Contains:
                 - max_effects (dict): Maximum interaction effect size per gene.
                 - max_effects_pvalues (dict): Corresponding p-values.
                 - model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted regression model.
                 - df (pd.DataFrame): DataFrame with sorted interaction effects and p-values.
   :rtype: tuple


.. py:function:: get_files_from_dir(dir_path, file_extension='*')

   Retrieves a list of files from the specified directory that match the given file extension.

   :param dir_path: The path to the directory from which to retrieve files.
   :type dir_path: str
   :param file_extension: The file extension to filter files by. Defaults to "*"
                          (matches all files).
   :type file_extension: str, optional

   :returns: A list of file paths matching the specified file extension in the given directory.
   :rtype: list


.. py:function:: create_circular_mask(h, w, center=None, radius=None)

   Creates a circular mask for a 2D array with the specified dimensions.

   :param h: The height of the 2D array.
   :type h: int
   :param w: The width of the 2D array.
   :type w: int
   :param center: The (x, y) coordinates of the circle's center.
                  Defaults to the center of the array.
   :type center: tuple, optional
   :param radius: The radius of the circle. Defaults to the smallest
                  distance from the center to the array's edges.
   :type radius: int, optional

   :returns:

             A boolean 2D array where `True` represents the pixels
                            inside the circle and `False` represents the pixels outside.
   :rtype: numpy.ndarray


.. py:function:: apply_mask(image, output_value=0)

   Apply a circular mask to an image, setting pixels outside the mask to a specified value.

   :param image: Input image (2D grayscale or 3D RGB array).
   :type image: np.ndarray
   :param output_value: Value for pixels outside the mask. Defaults to 0.
   :type output_value: int, optional

   :returns: Image with circular mask applied.
   :rtype: np.ndarray


.. py:function:: invert_image(image)

   Inverts the pixel values of an image.

   The function calculates the inverted image by subtracting each pixel value
   from the maximum possible value for the image's data type. For example,
   for an image with dtype `uint8`, the maximum value is 255.

   :param image: The input image to be inverted. The image should
   :type image: numpy.ndarray
   :param be a NumPy array with a valid integer data type:
   :type be a NumPy array with a valid integer data type: e.g., uint8, uint16

   :returns: The inverted image, with the same shape and dtype as the input.
   :rtype: numpy.ndarray


.. py:function:: resize_images_and_labels(images, labels, target_height, target_width, show_example=True)

   Resize images and labels to the specified target dimensions.

   :param images: List of 2D or 3D numpy arrays representing input images.
                  If None, only labels will be resized.
   :type images: list or None
   :param labels: List of 2D numpy arrays representing label masks.
                  If None, only images will be resized.
   :type labels: list or None
   :param target_height: Desired height of output arrays.
   :type target_height: int
   :param target_width: Desired width of output arrays.
   :type target_width: int
   :param show_example: Whether to display an example of original vs resized output.
                        Defaults to True.
   :type show_example: bool, optional

   :returns:

             A tuple (resized_images, resized_labels), where:
                 - resized_images (list): List of resized image arrays. Empty if `images` is None.
                 - resized_labels (list): List of resized label arrays. Empty if `labels` is None.
   :rtype: tuple

   :raises ValueError: If both `images` and `labels` are None.

   .. rubric:: Notes

   - Uses `resizescikit` for resizing.
   - Applies anti-aliasing when resizing images.
   - Uses nearest-neighbor interpolation (`order=0`) for labels to preserve class values.
   - Visualization of the resizing process is shown using `plot_resize` if `show_example` is True.


.. py:function:: resize_labels_back(labels, orig_dims)

   Resize a list of label arrays back to their original dimensions.

   :param labels: A list of label arrays to be resized.
   :type labels: list of numpy.ndarray
   :param orig_dims: A list of tuples where each tuple contains
                     two integers representing the original dimensions (width, height)
                     of the corresponding label array.
   :type orig_dims: list of tuple

   :returns: A list of resized label arrays with dimensions
             matching the corresponding tuples in `orig_dims`.
   :rtype: list of numpy.ndarray

   :raises ValueError: If the length of `labels` and `orig_dims` do not match.
   :raises ValueError: If any element in `orig_dims` is not a tuple of two integers.

   .. rubric:: Notes

   - The resizing operation uses nearest-neighbor interpolation (order=0).
   - The `preserve_range` parameter ensures that the data range of the
     input is preserved during resizing.
   - Anti-aliasing is disabled for this operation.


.. py:function:: calculate_iou(mask1, mask2)

   Calculate the Intersection over Union (IoU) between two binary masks.

   The IoU is a measure of the overlap between two binary masks, defined as the
   ratio of the intersection area to the union area of the masks.

   :param mask1: The first binary mask. Must be a 2D array.
   :type mask1: numpy.ndarray
   :param mask2: The second binary mask. Must be a 2D array.
   :type mask2: numpy.ndarray

   :returns: The IoU value, ranging from 0 to 1. Returns 0 if the union of the
             masks is empty.
   :rtype: float


.. py:function:: match_masks(true_masks, pred_masks, iou_threshold)

   Matches predicted masks to ground truth masks based on Intersection over Union (IoU) threshold.

   :param true_masks: A list of ground truth masks.
   :type true_masks: list
   :param pred_masks: A list of predicted masks.
   :type pred_masks: list
   :param iou_threshold: The IoU threshold for determining a match between a true mask and a predicted mask.
   :type iou_threshold: float

   :returns: A list of tuples where each tuple contains a matched pair of (true_mask, pred_mask).
   :rtype: list


.. py:function:: compute_average_precision(matches, num_true_masks, num_pred_masks)

   Computes the precision and recall based on the provided matches, number of true masks,
   and number of predicted masks.

   :param matches: A list of matched predictions to ground truth masks.
   :type matches: list
   :param num_true_masks: The total number of ground truth masks.
   :type num_true_masks: int
   :param num_pred_masks: The total number of predicted masks.
   :type num_pred_masks: int

   :returns:

             A tuple containing:
                 - precision (float): The precision value, calculated as TP / (TP + FP).
                 - recall (float): The recall value, calculated as TP / (TP + FN).
   :rtype: tuple


.. py:function:: pad_to_same_shape(mask1, mask2)

   Pads two 2D arrays (masks) to the same shape by adding zero-padding to the
   right and bottom of each array as needed.
   :param mask1: The first 2D array to be padded.
   :type mask1: numpy.ndarray
   :param mask2: The second 2D array to be padded.
   :type mask2: numpy.ndarray

   :returns:

             A tuple containing two 2D numpy arrays:
                 - padded_mask1 (numpy.ndarray): The first array padded to match the shape of the larger array.
                 - padded_mask2 (numpy.ndarray): The second array padded to match the shape of the larger array.
   :rtype: tuple

   .. rubric:: Notes

   - The padding is applied with constant values of 0.
   - The function assumes that both inputs are 2D arrays.


.. py:function:: compute_ap_over_iou_thresholds(true_masks, pred_masks, iou_thresholds)

   Compute the Average Precision (AP) over a range of Intersection over Union (IoU) thresholds.

   This function calculates the precision-recall pairs for each IoU threshold, validates that
   precision and recall values are within the range [0, 1], and computes the Average Precision
   by integrating the precision-recall curve using the trapezoidal rule.

   :param true_masks: Ground truth masks.
   :type true_masks: list or array-like
   :param pred_masks: Predicted masks.
   :type pred_masks: list or array-like
   :param iou_thresholds: A list of IoU thresholds to evaluate.
   :type iou_thresholds: list or array-like

   :returns: The computed Average Precision (AP) over the specified IoU thresholds.
   :rtype: float

   :raises ValueError: If precision or recall values are out of the valid range [0, 1].


.. py:function:: compute_segmentation_ap(true_masks, pred_masks, iou_thresholds=np.linspace(0.5, 0.95, 10))

   Compute the Average Precision (AP) for segmentation masks over a range of IoU thresholds.

   This function calculates the AP by comparing the ground truth masks (`true_masks`) with the
   predicted masks (`pred_masks`) across multiple Intersection over Union (IoU) thresholds.

   :param true_masks: A binary array representing the ground truth segmentation masks.
                      Each connected component is treated as a separate object.
   :type true_masks: ndarray
   :param pred_masks: A binary array representing the predicted segmentation masks.
                      Each connected component is treated as a separate object.
   :type pred_masks: ndarray
   :param iou_thresholds: A 1D array of IoU thresholds to evaluate AP over.
                          Defaults to `np.linspace(0.5, 0.95, 10)`.
   :type iou_thresholds: ndarray, optional

   :returns: The computed Average Precision (AP) over the specified IoU thresholds.
   :rtype: float


.. py:function:: jaccard_index(mask1, mask2)

   Computes the Jaccard Index (Intersection over Union) between two binary masks.

   The Jaccard Index is a measure of similarity between two sets, defined as the size
   of the intersection divided by the size of the union of the sets.

   :param mask1: A binary mask (e.g., a 2D array of boolean or 0/1 values).
   :type mask1: numpy.ndarray
   :param mask2: Another binary mask of the same shape as `mask1`.
   :type mask2: numpy.ndarray

   :returns:

             The Jaccard Index, a value between 0 and 1, where 1 indicates perfect overlap
                    and 0 indicates no overlap.
   :rtype: float


.. py:function:: dice_coefficient(mask1, mask2)

   Compute the Dice coefficient, a measure of overlap between two binary masks.

   The Dice coefficient is defined as:
       ``2 * |A ∩ B| / (|A| + |B|)``

   where A and B are binary masks, ``|A ∩ B|`` is the number of overlapping non-zero elements,
   and ``|A| + |B|`` is the total number of non-zero elements in both masks.

   :param mask1: First binary mask.
   :type mask1: numpy.ndarray
   :param mask2: Second binary mask.
   :type mask2: numpy.ndarray

   :returns:

             Dice coefficient between 0.0 (no overlap) and 1.0 (perfect overlap).
                    Returns 1.0 if both masks are empty.
   :rtype: float


.. py:function:: extract_boundaries(mask, dilation_radius=1)

   Extracts the boundaries of a binary mask by applying morphological dilation
   and erosion operations and computing their difference.

   :param mask: A 2D array representing the input mask. Non-zero
                values are considered part of the mask.
   :type mask: numpy.ndarray
   :param dilation_radius: The radius of the structuring element
                           used for dilation and erosion.
                           Defaults to 1.
   :type dilation_radius: int, optional

   :returns:

             A binary array of the same shape as the input mask,
                            where the boundary pixels are marked as 1 and all
                            other pixels are 0.
   :rtype: numpy.ndarray


.. py:function:: boundary_f1_score(mask_true, mask_pred, dilation_radius=1)

   Calculate the boundary F1 score between two binary masks.

   The boundary F1 score evaluates the alignment of object boundaries between
   predicted and ground truth masks. It computes the harmonic mean of boundary
   precision and recall after dilating the boundaries.

   :param mask_true: Ground truth binary mask. Non-zero values are considered foreground.
   :type mask_true: np.ndarray
   :param mask_pred: Predicted binary mask. Non-zero values are considered foreground.
   :type mask_pred: np.ndarray
   :param dilation_radius: Radius used for boundary dilation to allow tolerance in matching. Defaults to 1.
   :type dilation_radius: int, optional

   :returns: Boundary F1 score between 0 (no boundary match) and 1 (perfect boundary alignment).
   :rtype: float


.. py:function:: merge_touching_objects(mask, threshold=0.25)

   Merges touching objects in a binary mask based on the percentage of their shared boundary.

   :param mask: Binary mask representing objects.
   :type mask: ndarray
   :param threshold: Threshold value for merging objects. Defaults to 0.25.
   :type threshold: float, optional

   :returns: Merged mask.
   :rtype: ndarray


.. py:function:: remove_intensity_objects(image, mask, intensity_threshold, mode)

   Removes objects from the mask based on their mean intensity in the original image.

   :param image: The original image.
   :type image: ndarray
   :param mask: The mask containing labeled objects.
   :type mask: ndarray
   :param intensity_threshold: The threshold value for mean intensity.
   :type intensity_threshold: float
   :param mode: The mode for intensity comparison. Can be 'low' or 'high'.
   :type mode: str

   :returns: The updated mask with objects removed.
   :rtype: ndarray


.. py:class:: SelectChannels(channels)

   Selectively retain specific color channels in a 3-channel image tensor.

   This transformation zeros out unselected channels based on a list of 1-based
   indices corresponding to RGB channels.

   :param channels: List of 1-based indices for channels to retain.
                    Valid values are:
                    - 1: Red
                    - 2: Green
                    - 3: Blue
   :type channels: list of int

   .. rubric:: Example

   >>> select_channels = SelectChannels([1, 3])
   >>> modified_img = select_channels(img)

   .. note:: The input image must be a PyTorch tensor with shape (3, H, W).

   Initialize the SelectChannels instance.

   :param channels: Channels (1-based) to retain.
   :type channels: list of int


   .. py:attribute:: channels


   .. py:method:: __call__(img)

      Apply the channel selection to the input image tensor.

      :param img: A 3-channel image tensor of shape (3, H, W).
      :type img: torch.Tensor

      :returns: A tensor where unselected channels are set to zero.
      :rtype: torch.Tensor



.. py:function:: preprocess_image_v1(image_path, image_size=224, channels=[1, 2, 3], normalize=True)

   Preprocess an image for input into a machine learning model.

   :param image_path: The file path to the image to be processed.
   :type image_path: str
   :param image_size: The size to which the image will be center-cropped. Defaults to 224.
   :type image_size: int, optional
   :param channels: A list of channel indices to select from the image. Defaults to [1, 2, 3].
   :type channels: list, optional
   :param normalize: Whether to normalize the image using mean and standard deviation. Defaults to True.
   :type normalize: bool, optional

   :returns:

             A tuple containing:
                 - image (PIL.Image.Image): The original image loaded as a PIL Image object.
                 - input_tensor (torch.Tensor): The preprocessed image as a PyTorch tensor with an added batch dimension.
   :rtype: tuple


.. py:class:: SaliencyMapGenerator(model)

   SaliencyMapGenerator is a utility class for generating saliency maps and visualizing model predictions
   for PyTorch models in binary classification tasks.

   Initialize the SaliencyMapGenerator.


   .. py:attribute:: model


   .. py:method:: compute_saliency_maps(X, y)

      Compute saliency maps for the given inputs and target labels.

      :param X: Input tensor with requires_grad enabled.
      :type X: torch.Tensor
      :param y: Ground truth labels for the input samples.
      :type y: torch.Tensor

      :returns: The computed saliency maps, same shape as input.
      :rtype: torch.Tensor



   .. py:method:: compute_saliency_and_predictions(X)

      Compute saliency maps and predictions for the given input batch.

      :param X: Input tensor with requires_grad enabled.
      :type X: torch.Tensor

      :returns:     - torch.Tensor: Saliency maps for the input.
                    - torch.Tensor: Predicted class labels.
      :rtype: tuple



   .. py:method:: plot_activation_grid(X, saliency, predictions, overlay=True, normalize=False)

      Plot a grid of input images with overlaid saliency maps and predicted labels.

      :param X: Input tensor of images (N, C, H, W).
      :type X: torch.Tensor
      :param saliency: Corresponding saliency maps (N, C, H, W).
      :type saliency: torch.Tensor
      :param predictions: Predicted class labels (N,).
      :type predictions: torch.Tensor
      :param overlay: Whether to overlay saliency maps on input images. Default is True.
      :type overlay: bool
      :param normalize: Whether to normalize input images by percentiles. Default is False.
      :type normalize: bool

      :returns: A matplotlib figure object showing the grid.
      :rtype: matplotlib.figure.Figure



   .. py:method:: percentile_normalize(img, lower_percentile=2, upper_percentile=98)

      Normalize an image's intensity per channel using percentile clipping.

      :param img: Image of shape (H, W, C) to be normalized.
      :type img: np.ndarray
      :param lower_percentile: Lower percentile for intensity clipping. Default is 2.
      :type lower_percentile: int
      :param upper_percentile: Upper percentile for intensity clipping. Default is 98.
      :type upper_percentile: int

      :returns: Percentile-normalized image.
      :rtype: np.ndarray



.. py:class:: GradCAMGenerator(model, target_layer, cam_type='gradcam')

   GradCAMGenerator generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations
   for CNN-based PyTorch models, supporting binary classification tasks.

   Initialize the GradCAMGenerator and register hooks.

   :param model: A trained PyTorch model.
   :type model: torch.nn.Module
   :param target_layer: The name of the layer to compute Grad-CAM on.
   :type target_layer: str
   :param cam_type: Type of CAM method (default is 'gradcam').
   :type cam_type: str


   .. py:attribute:: model


   .. py:attribute:: target_layer


   .. py:attribute:: cam_type
      :value: 'gradcam'



   .. py:attribute:: gradients
      :value: None



   .. py:attribute:: activations
      :value: None



   .. py:attribute:: target_layer_module


   .. py:method:: hook_layers()

      Register forward and backward hooks to capture activations and gradients
      from the specified target layer during inference and backpropagation.



   .. py:method:: get_layer(model, target_layer)

      Recursively retrieve a layer object from a nested model.

      :param model: The model containing the layer.
      :type model: torch.nn.Module
      :param target_layer: Dot-separated string path to the desired layer.
      :type target_layer: str

      :returns: The resolved layer module.
      :rtype: torch.nn.Module



   .. py:method:: compute_gradcam_maps(X, y)

      Compute Grad-CAM heatmaps for an input batch and target labels.

      :param X: Input tensor of shape (N, C, H, W).
      :type X: torch.Tensor
      :param y: Target labels (0 or 1) for each sample.
      :type y: torch.Tensor

      :returns: Grad-CAM heatmaps normalized to [0, 1] for each input.
      :rtype: np.ndarray



   .. py:method:: compute_gradcam_and_predictions(X)

      Compute Grad-CAM heatmaps and class predictions for a batch.

      :param X: Input tensor of shape (N, C, H, W).
      :type X: torch.Tensor

      :returns:     - Grad-CAM heatmaps for each sample.
                    - Predicted class labels (0 or 1).
      :rtype: Tuple[torch.Tensor, torch.Tensor]



   .. py:method:: plot_activation_grid(X, gradcam, predictions, overlay=True, normalize=False)

      Plot a grid of input images overlaid with Grad-CAM heatmaps.

      :param X: Input image batch (N, C, H, W).
      :type X: torch.Tensor
      :param gradcam: Grad-CAM heatmaps (N, H, W).
      :type gradcam: torch.Tensor
      :param predictions: Predicted class labels.
      :type predictions: torch.Tensor
      :param overlay: Whether to overlay Grad-CAM on input images.
      :type overlay: bool
      :param normalize: Whether to normalize image intensities by percentiles.
      :type normalize: bool

      :returns: The generated grid figure.
      :rtype: matplotlib.figure.Figure



   .. py:method:: percentile_normalize(img, lower_percentile=2, upper_percentile=98)

      Normalize each channel of the input image to the specified percentiles.

      :param img: Image array (H, W, C).
      :type img: np.ndarray
      :param lower_percentile: Lower clipping percentile.
      :type lower_percentile: int
      :param upper_percentile: Upper clipping percentile.
      :type upper_percentile: int

      :returns: Percentile-normalized image.
      :rtype: np.ndarray



.. py:function:: preprocess_image(image_path, normalize=True, image_size=224, channels=[1, 2, 3])

   Preprocess an image for input into a machine learning model.

   :param image_path: Path to the input image file.
   :type image_path: str
   :param normalize: If True, apply ImageNet normalization
                     (mean and std). Defaults to True.
   :type normalize: bool, optional
   :param image_size: Target size (height and width) for resizing.
                      Defaults to 224.
   :type image_size: int, optional
   :param channels: 1-based channel indices to retain (e.g., [1, 2, 3]
                    for RGB). Defaults to [1, 2, 3].
   :type channels: list of int, optional

   :returns:     - PIL.Image.Image: The original image.
                 - torch.Tensor: The processed image tensor suitable for model input.
   :rtype: tuple


.. py:function:: class_visualization(target_y, model_path, dtype, img_size=224, channels=[0, 1, 2], l2_reg=0.001, learning_rate=25, num_iterations=100, blur_every=10, max_jitter=16, show_every=25, class_names=['nc', 'pc'])

.. py:function:: get_submodules(model, prefix='')

   Recursively retrieves the names of all submodules in a given model.

   :param model: The model whose submodules are to be retrieved.
   :type model: torch.nn.Module
   :param prefix: A prefix to prepend to the names of the submodules.
                  Defaults to an empty string.
   :type prefix: str, optional

   :returns: A list of strings representing the full names of all submodules
             in the model, including nested submodules.
   :rtype: list of str


.. py:class:: GradCAM(model, target_layers=None, use_cuda=True)

   Compute Grad-CAM (Gradient-weighted Class Activation Mapping) for a given model and target layer(s).

   Initialize the GradCAM object.

   :param model: The model for which Grad-CAM will be computed.
   :type model: nn.Module
   :param target_layers: Names of layers to register hooks on.
   :type target_layers: list of str
   :param use_cuda: Whether to use CUDA (GPU) for computation.
   :type use_cuda: bool


   .. py:attribute:: model


   .. py:attribute:: target_layers
      :value: None



   .. py:attribute:: cuda
      :value: True



   .. py:method:: forward(input)

      Run a forward pass through the model.

      :param input: Input tensor.
      :type input: torch.Tensor

      :returns: Model output.
      :rtype: torch.Tensor



   .. py:method:: __call__(x, index=None)

      Compute the Grad-CAM heatmap for an input image.

      :param x: Input tensor of shape (1, C, H, W).
      :type x: torch.Tensor
      :param index: Class index to compute gradients for. If None, uses the predicted class.
      :type index: int or None

      :returns: Normalized Grad-CAM heatmap of shape (H, W).
      :rtype: numpy.ndarray



.. py:function:: show_cam_on_image(img, mask)

   Overlay a heatmap generated from a mask onto an image.

   This function applies a color map to the mask, combines it with the input
   image, and normalizes the result to create a visually interpretable
   representation of the mask overlaid on the image.

   :param img: The input image as a NumPy array with pixel values
               normalized between 0 and 1.
   :type img: numpy.ndarray
   :param mask: The mask to overlay on the image, with values
                normalized between 0 and 1.
   :type mask: numpy.ndarray

   :returns: The resulting image with the heatmap overlay, as a
             NumPy array with pixel values in the range [0, 255].
   :rtype: numpy.ndarray


.. py:function:: recommend_target_layers(model)

   Identifies and recommends target layers in a given model for further processing.

   This function iterates through all the modules in the provided model and collects
   the names of all 2D convolutional layers (`torch.nn.Conv2d`). It then recommends
   the last convolutional layer as the primary target layer.

   :param model: The neural network model to analyze.
   :type model: torch.nn.Module

   :returns:     - list: A list containing the name of the recommended target layer (last Conv2d layer).
                 - list: A list of all Conv2d layer names found in the model.
   :rtype: tuple

   :raises ValueError: If no convolutional layers (`torch.nn.Conv2d`) are found in the model.


.. py:class:: IntegratedGradients(model)

   Compute Integrated Gradients for model interpretability.

   This class implements the Integrated Gradients method to attribute the prediction
   of a neural network to its input features. It approximates the integral of gradients
   along a straight path from a baseline to the input.

   Initialize the IntegratedGradients instance.

   :param model: A trained PyTorch model.
   :type model: torch.nn.Module


   .. py:attribute:: model


   .. py:method:: generate_integrated_gradients(input_tensor, target_label_idx, baseline=None, num_steps=50)

      Compute the integrated gradients for a given input and target class.

      :param input_tensor: The input tensor of shape (1, C, H, W) or similar.
      :type input_tensor: torch.Tensor
      :param target_label_idx: Index of the target class for which gradients are computed.
      :type target_label_idx: int
      :param baseline: Baseline tensor with the same shape as input. Defaults to zeros.
      :type baseline: torch.Tensor, optional
      :param num_steps: Number of steps in the Riemann approximation of the integral. Defaults to 50.
      :type num_steps: int, optional

      :returns: Integrated gradients as a NumPy array with the same shape as `input_tensor`.
      :rtype: np.ndarray



.. py:function:: get_db_paths(src)

   Generate a list of database file paths based on the given source(s).

   This function takes a single source path or a list of source paths and
   constructs the corresponding paths to the 'measurements.db' file located
   in the 'measurements' subdirectory of each source.

   :param src: A single source path as a string or a list
               of source paths.
   :type src: str or list of str

   :returns: A list of file paths pointing to 'measurements/measurements.db'
             for each source in the input.
   :rtype: list of str


.. py:function:: get_sequencing_paths(src)

   Generate a list of file paths pointing to sequencing data CSV files.

   This function takes a single source path or a list of source paths and
   constructs the full file paths to the 'sequencing/sequencing_data.csv'
   file located within each source directory.

   :param src: A single source directory path as a string
               or a list of source directory paths.
   :type src: str or list of str

   :returns: A list of full file paths to the 'sequencing_data.csv'
             files for each source directory.
   :rtype: list of str


.. py:function:: load_image_paths(c, visualize)

   Loads image paths from a database table and optionally filters them based on a visualization keyword.

   :param c: A database cursor object used to execute SQL queries.
   :type c: sqlite3.Cursor
   :param visualize: A keyword to filter image paths. If provided, only rows where the 'png_path' column
                     contains the keyword followed by '_png' will be included. If None or empty, no filtering
                     is applied.
   :type visualize: str

   :returns:

             A DataFrame containing the image paths and other associated data from the 'png_list' table.
                               The DataFrame is indexed by the 'prcfo' column.
   :rtype: pandas.DataFrame


.. py:function:: merge_dataframes(df, image_paths_df, verbose)

   Merges two pandas DataFrames on their indices and optionally displays the result.

   :param df: The main DataFrame to be merged. It must have a column named 'prcfo',
              which will be set as the index before merging.
   :type df: pandas.DataFrame
   :param image_paths_df: The DataFrame containing image paths to be merged with `df`.
   :type image_paths_df: pandas.DataFrame
   :param verbose: If True, the resulting merged DataFrame will be displayed.
   :type verbose: bool

   :returns: The merged DataFrame with the indices aligned.
   :rtype: pandas.DataFrame


.. py:function:: remove_highly_correlated_columns_v1(df, threshold)

   Removes columns from a DataFrame that are highly correlated with other columns.

   This function calculates the correlation matrix of the given DataFrame, identifies
   columns with a correlation higher than the specified threshold, and removes them
   to reduce multicollinearity.

   :param df: The input DataFrame containing the data.
   :type df: pandas.DataFrame
   :param threshold: The correlation threshold above which columns are considered
                     highly correlated and will be removed.
   :type threshold: float

   :returns: A DataFrame with highly correlated columns removed.
   :rtype: pandas.DataFrame

   .. rubric:: Example

   >>> import pandas as pd
   >>> import numpy as np
   >>> data = {'A': [1, 2, 3], 'B': [2, 4, 6], 'C': [7, 8, 9]}
   >>> df = pd.DataFrame(data)
   >>> remove_highly_correlated_columns_v1(df, threshold=0.9)
      A  C


.. py:function:: filter_columns(df, filter_by)

   Filters the columns of a DataFrame based on a specified criterion.

   :param df: The input DataFrame whose columns are to be filtered.
   :type df: pandas.DataFrame
   :param filter_by: The criterion for filtering columns. If 'morphology',
                     columns containing 'channel' in their names are excluded.
                     Otherwise, only columns containing the specified string
                     are included.
   :type filter_by: str

   :returns: A DataFrame containing only the filtered columns.
   :rtype: pandas.DataFrame


.. py:function:: reduction_and_clustering(numeric_data, n_neighbors, min_dist, metric, eps, min_samples, clustering, reduction_method='umap', verbose=False, embedding=None, n_jobs=-1, mode='fit', model=False)

   Perform dimensionality reduction and clustering on the given data.

   Args:
   numeric_data (np.ndarray): Numeric data for embedding and clustering.
   n_neighbors (int or float): Number of neighbors for UMAP or perplexity for t-SNE.
   min_dist (float): Minimum distance for UMAP.
   metric (str): Metric for UMAP and DBSCAN.
   eps (float): Epsilon for DBSCAN.
   min_samples (int): Minimum samples for DBSCAN or number of clusters for KMeans.
   clustering (str): Clustering method ('DBSCAN' or 'KMeans').
   reduction_method (str): Dimensionality reduction method ('UMAP' or 'tSNE').
   verbose (bool): Whether to print verbose output.
   embedding (np.ndarray, optional): Precomputed embedding. Default is None.
   return_model (bool): Whether to return the reducer model. Default is False.

   Returns:
   tuple: embedding, labels (and optionally the reducer model)


.. py:function:: remove_noise(embedding, labels)

   Removes noise from the given embedding and labels by filtering out elements
   where the corresponding label is -1.

   :param embedding: The embedding array, where each row corresponds
                     to a data point.
   :type embedding: numpy.ndarray
   :param labels: The array of labels corresponding to the embedding,
                  where a label of -1 indicates noise.
   :type labels: numpy.ndarray

   :returns:

             A tuple containing:
                 - numpy.ndarray: The filtered embedding array with noise removed.
                 - numpy.ndarray: The filtered labels array with noise removed.
   :rtype: tuple


.. py:function:: plot_embedding(embedding, image_paths, labels, image_nr, img_zoom, colors, plot_by_cluster, plot_outlines, plot_points, plot_images, smooth_lines, black_background, figuresize, dot_size, remove_image_canvas, verbose)

   Plots a 2D embedding with optional images, clusters, and customization options.

   :param embedding: A 2D array of shape (n_samples, 2) representing the embedding coordinates.
   :type embedding: np.ndarray
   :param image_paths: A list of file paths to images corresponding to the data points, or None if no images are used.
   :type image_paths: list or None
   :param labels: An array of cluster labels for each data point.
   :type labels: np.ndarray
   :param image_nr: The number of images to display on the plot.
   :type image_nr: int
   :param img_zoom: The zoom factor for the displayed images.
   :type img_zoom: float
   :param colors: A list of colors to use for clusters, or None to use default colors.
   :type colors: list or None
   :param plot_by_cluster: Whether to plot images grouped by cluster.
   :type plot_by_cluster: bool
   :param plot_outlines: Whether to draw outlines around clusters.
   :type plot_outlines: bool
   :param plot_points: Whether to plot individual data points.
   :type plot_points: bool
   :param plot_images: Whether to overlay images on the embedding.
   :type plot_images: bool
   :param smooth_lines: Whether to draw smooth lines between cluster centers.
   :type smooth_lines: bool
   :param black_background: Whether to use a black background for the plot.
   :type black_background: bool
   :param figuresize: The size of the figure in inches (width, height).
   :type figuresize: tuple
   :param dot_size: The size of the dots representing data points.
   :type dot_size: float
   :param remove_image_canvas: Whether to remove the canvas around the images.
   :type remove_image_canvas: bool
   :param verbose: Whether to print verbose output during the plotting process.
   :type verbose: bool

   :returns: The generated plot as a Matplotlib figure object.
   :rtype: matplotlib.figure.Figure


.. py:function:: generate_colors(num_clusters, black_background)

   Generate a set of RGBA colors for visualization purposes.

   This function generates a list of random RGBA colors, appends specific predefined colors,
   and optionally includes a black background color.

   :param num_clusters: The number of clusters for which colors need to be generated.
                        Additional random colors will be generated beyond the predefined ones.
   :type num_clusters: int
   :param black_background: If True, a black background color ([0, 0, 0, 1]) will be included
                            at the beginning of the color list.
   :type black_background: bool

   :returns:

             A 2D array of shape (num_colors, 4), where each row represents an RGBA color.
                            The first dimension corresponds to the total number of colors generated.
   :rtype: numpy.ndarray


.. py:function:: assign_colors(unique_labels, random_colors)

   Assigns colors to unique labels and creates a mapping from labels to color indices.

   :param unique_labels: A collection of unique labels for which colors need to be assigned.
   :type unique_labels: list or iterable
   :param random_colors: An array or list of RGB color values, where each color is represented
                         as a triplet of integers in the range [0, 255].
   :type random_colors: numpy.ndarray or list

   :returns:

             A tuple containing:
                 - colors (list of tuple): A list of RGB color tuples in the original [0, 255] range.
                 - label_to_color_index (dict): A dictionary mapping each unique label to its corresponding color index.
   :rtype: tuple


.. py:function:: setup_plot(figuresize, black_background)

   Sets up a matplotlib plot with specified figure size and background color.

   :param figuresize: The size of the figure in inches (used for both width and height).
   :type figuresize: float
   :param black_background: If True, sets the plot to have a black background with white text and labels.
                            If False, sets the plot to have a white background with black text and labels.
   :type black_background: bool

   :returns: A tuple containing the figure (`matplotlib.figure.Figure`) and axes (`matplotlib.axes._axes.Axes`) objects.
   :rtype: tuple


.. py:function:: plot_clusters(ax, embedding, labels, colors, cluster_centers, plot_outlines, plot_points, smooth_lines, figuresize=10, dot_size=50, verbose=False)

   Plots clusters on a 2D embedding using matplotlib.

   :param ax: The matplotlib Axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param embedding: A 2D array of shape (n_samples, 2) representing the embedding coordinates.
   :type embedding: numpy.ndarray
   :param labels: An array of cluster labels for each point in the embedding.
   :type labels: numpy.ndarray
   :param colors: A list of colors corresponding to each cluster.
   :type colors: list
   :param cluster_centers: A 2D array of shape (n_clusters, 2) representing the coordinates of cluster centers.
   :type cluster_centers: numpy.ndarray
   :param plot_outlines: Whether to plot the outlines of clusters using convex hulls or smoothed lines.
   :type plot_outlines: bool
   :param plot_points: Whether to plot individual points in the clusters.
   :type plot_points: bool
   :param smooth_lines: Whether to use smoothed lines for cluster outlines instead of convex hulls.
   :type smooth_lines: bool
   :param figuresize: The size of the figure. Defaults to 10.
   :type figuresize: int, optional
   :param dot_size: The size of the points in the scatter plot. Defaults to 50.
   :type dot_size: int, optional
   :param verbose: Whether to print additional information for debugging. Defaults to False.
   :type verbose: bool, optional

   :returns: None

   .. rubric:: Notes

   - This function assumes that the embedding is 2D.
   - Cluster labels should be integers, with -1 typically representing noise.
   - The function uses matplotlib for plotting and assumes that the required libraries (e.g., numpy, matplotlib) are imported.


.. py:function:: plot_umap_images(ax, image_paths, embedding, labels, image_nr, img_zoom, colors, plot_by_cluster, remove_image_canvas, verbose)

   Plots UMAP embeddings with associated images on a given matplotlib axis.

   :param ax: The matplotlib axis on which to plot the images.
   :type ax: matplotlib.axes.Axes
   :param image_paths: List of file paths to the images to be plotted.
   :type image_paths: list of str
   :param embedding: 2D array of UMAP embeddings with shape (n_samples, 2).
   :type embedding: numpy.ndarray
   :param labels: Array of cluster labels for each embedding point.
   :type labels: numpy.ndarray
   :param image_nr: Number of images to plot.
   :type image_nr: int
   :param img_zoom: Zoom factor for the images.
   :type img_zoom: float
   :param colors: List of colors for each cluster.
   :type colors: list
   :param plot_by_cluster: If True, plot images grouped by cluster; otherwise, plot randomly sampled images.
   :type plot_by_cluster: bool
   :param remove_image_canvas: If True, remove the image canvas (background) when plotting.
   :type remove_image_canvas: bool
   :param verbose: If True, print additional information during execution.
   :type verbose: bool

   :returns: None


.. py:function:: plot_images_by_cluster(ax, image_paths, embedding, labels, image_nr, img_zoom, colors, cluster_indices, remove_image_canvas, verbose)

   Plots images on a given axis based on their cluster assignments and embeddings.

   :param ax: The matplotlib axis on which to plot the images.
   :type ax: matplotlib.axes.Axes
   :param image_paths: List of file paths to the images to be plotted.
   :type image_paths: list of str
   :param embedding: 2D array of shape (n_samples, 2) containing the x and y coordinates for each image.
   :type embedding: array-like
   :param labels: Array of cluster labels for each image. -1 indicates noise or unclustered points.
   :type labels: array-like
   :param image_nr: Maximum number of images to display per cluster.
   :type image_nr: int
   :param img_zoom: Zoom factor for the displayed images.
   :type img_zoom: float
   :param colors: List of colors corresponding to each cluster.
   :type colors: list of str
   :param cluster_indices: Dictionary mapping cluster labels to lists of indices of images in each cluster.
   :type cluster_indices: dict
   :param remove_image_canvas: If True, removes the canvas (border) around the plotted images.
   :type remove_image_canvas: bool
   :param verbose: If True, prints additional information during execution.
   :type verbose: bool

   :returns: None


.. py:function:: plot_image(ax, x, y, img, img_zoom, remove_image_canvas=True)

   Plots an image on a given matplotlib axis at specified coordinates.

   :param ax: The axis on which to plot the image.
   :type ax: matplotlib.axes.Axes
   :param x: The x-coordinate where the image will be placed.
   :type x: float
   :param y: The y-coordinate where the image will be placed.
   :type y: float
   :param img: The image data to be plotted.
   :type img: numpy.ndarray or array-like
   :param img_zoom: The zoom factor for the image.
   :type img_zoom: float
   :param remove_image_canvas: If True, removes the canvas
                               (e.g., padding or borders) from the image before plotting.
                               Defaults to True.
   :type remove_image_canvas: bool, optional

   :returns: None


.. py:function:: remove_canvas(img)

   Converts an image to a normalized RGBA format by adding an alpha channel.

   This function processes images in either grayscale ('L', 'I') or RGB ('RGB') mode.
   For grayscale images, the pixel values are normalized, and an alpha channel is
   created based on non-zero pixel values. For RGB images, the pixel values are
   normalized to the range [0, 1], and an alpha channel is created based on the
   presence of non-zero pixel values across all channels.

   :param img: The input image to process. Must be in 'L', 'I', or 'RGB' mode.
   :type img: PIL.Image.Image

   :returns: A 4-channel RGBA image as a NumPy array, where the first three
             channels represent the normalized RGB values, and the fourth channel represents
             the alpha channel.
   :rtype: numpy.ndarray

   :raises ValueError: If the input image mode is not 'L', 'I', or 'RGB'.


.. py:function:: plot_clusters_grid(embedding, labels, image_nr, image_paths, colors, figuresize, black_background, verbose)

   Plot a grid of images for each cluster based on the given labels and embeddings.

   :param embedding: Embedding of data points for visualization.
   :type embedding: np.ndarray
   :param labels: Cluster labels for each data point. A value of -1 indicates noise or outliers.
   :type labels: np.ndarray
   :param image_nr: Maximum number of images to display per cluster.
   :type image_nr: int
   :param image_paths: File paths to images corresponding to the data points.
   :type image_paths: list of str
   :param colors: List of colors for each cluster.
   :type colors: list of str
   :param figuresize: Size of the figure (width, height).
   :type figuresize: tuple
   :param black_background: Whether to use a black background.
   :type black_background: bool
   :param verbose: Whether to print progress information.
   :type verbose: bool

   :returns: The generated figure, or None if no valid clusters are found.
   :rtype: matplotlib.figure.Figure or None

   .. rubric:: Notes

   - Clusters larger than `image_nr` are randomly subsampled.
   - If all labels are -1, the function returns None.
   - Relies on an external `plot_grid` function for grid rendering.


.. py:function:: plot_grid(cluster_images, colors, figuresize, black_background, verbose)

   Plot a grid of images grouped by cluster with optional background and labels.

   :param cluster_images: Dictionary mapping cluster labels to lists of images.
   :type cluster_images: dict
   :param colors: List of RGB tuples specifying colors for each cluster.
   :type colors: list
   :param figuresize: Base figure size; actual size scales with the number of clusters.
   :type figuresize: float
   :param black_background: If True, use a black background; otherwise, use white.
   :type black_background: bool
   :param verbose: If True, print cluster labels and index info during plotting.
   :type verbose: bool

   :returns: The generated figure containing the image grid.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Notes

   - Grid size is dynamically adjusted per cluster.
   - Cluster labels are shown alongside image grids using corresponding colors.
   - A maximum figure size limit prevents overly large plots.


.. py:function:: generate_path_list_from_db(db_path, file_metadata)

   Generate a list of file paths from a SQLite database using optional metadata filters.

   :param db_path: Path to the SQLite database.
   :type db_path: str
   :param file_metadata: Filter criteria for file paths.
                         - str: Only include paths containing the string.
                         - list of str: Include paths containing any of the strings.
                         - None or empty: Include all paths.
   :type file_metadata: str | list[str] | None

   :returns: List of matching file paths, or None if an error occurs.
   :rtype: list[str] or None

   :raises sqlite3.Error: If a database operation fails.
   :raises Exception: For any other unexpected error.

   .. rubric:: Notes

   - Paths are fetched from the 'png_list' table using the 'png_path' column.
   - Results are retrieved in batches of 1000 rows for efficiency.


.. py:function:: correct_paths(df, base_path, folder='data')

   Adjust file paths to include the specified base directory and folder.

   :param df: Input containing file paths.
              - If a DataFrame, it must have a 'png_path' column.
              - If a list, it should contain file path strings.
   :type df: pandas.DataFrame or list
   :param base_path: Base directory to prepend if not already present.
   :type base_path: str
   :param folder: Folder name to insert into paths (default: 'data').
   :type folder: str, optional

   :returns:     - If input is a DataFrame: (updated DataFrame, list of adjusted paths).
                 - If input is a list: list of adjusted paths.
   :rtype: tuple or list

   :raises ValueError: If the DataFrame does not contain a 'png_path' column.

   .. rubric:: Notes

   Paths already containing the base path are not modified.


.. py:function:: delete_folder(folder_path)

   Deletes a folder and all of its contents, including subdirectories and files.

   :param folder_path: The path to the folder to be deleted.
   :type folder_path: str

   Behavior:
       - If the specified folder exists and is a directory, it recursively deletes all files
         and subdirectories within it, and then removes the folder itself.
       - If the folder does not exist or is not a directory, a message is printed indicating this.

   Prints:
       - A confirmation message if the folder is successfully deleted.
       - An error message if the folder does not exist or is not a directory.

   .. rubric:: Example

   delete_folder('/path/to/folder')


.. py:function:: measure_test_mode(settings)

   Adjusts the source folder in the settings dictionary for test mode.

   If `test_mode` is enabled in the `settings` dictionary, this function:
   - Checks if the current source folder (`settings['src']`) is not already set to 'test'.
   - Selects a random subset of files from the source folder based on `settings['test_nr']`.
   - Copies the selected files into a new 'test/merged' directory.
   - Updates the `settings['src']` to point to the new 'test/merged' directory.
   - Prints a message indicating the change in the source folder.

   If the source folder is already set to 'test', it simply prints a message confirming the test mode.

   :param settings: A dictionary containing configuration settings.
                    Expected keys:
                    - 'test_mode' (bool): Whether test mode is enabled.
                    - 'src' (str): Path to the source folder.
                    - 'test_nr' (int): Number of files to select for test mode.
   :type settings: dict

   :returns: The updated settings dictionary with the modified source folder if test mode is enabled.
   :rtype: dict


.. py:function:: preprocess_data(df, filter_by, remove_highly_correlated, log_data, exclude, column_list=False)

   Preprocesses the given dataframe by applying filtering, removing highly correlated columns,
   applying log transformation, filling NaN values, and scaling the numeric data.

   Args:
   df (pandas.DataFrame): The input dataframe.
   filter_by (str or None): The channel of interest to filter the dataframe by.
   remove_highly_correlated (bool or float): Whether to remove highly correlated columns.
   If a float is provided, it represents the correlation threshold.
   log_data (bool): Whether to apply log transformation to the numeric data.
   exclude (list or None): List of features to exclude from the filtering process.
   verbose (bool): Whether to print verbose output during preprocessing.

   Returns:
   numpy.ndarray: The preprocessed numeric data.

   Raises:
   ValueError: If no numeric columns are available after filtering.



.. py:function:: remove_low_variance_columns(df, threshold=0.01, verbose=False)

   Removes columns from the dataframe that have low variance.

   Args:
   df (pandas.DataFrame): The DataFrame containing the data.
   threshold (float): The variance threshold below which columns will be removed.

   Returns:
   pandas.DataFrame: The DataFrame with low variance columns removed.


.. py:function:: remove_highly_correlated_columns(df, threshold=0.95, verbose=False)

   Removes columns from the dataframe that are highly correlated with one another.

   Args:
   df (pandas.DataFrame): The DataFrame containing the data.
   threshold (float): The correlation threshold above which columns will be removed.

   Returns:
   pandas.DataFrame: The DataFrame with highly correlated columns removed.


.. py:function:: filter_dataframe_features(df, channel_of_interest, exclude=None, remove_low_variance_features=True, remove_highly_correlated_features=True, verbose=False)

   Filter the dataframe `df` based on the specified `channel_of_interest` and `exclude` parameters.

   Args:
   - df (pandas.DataFrame): The input dataframe to be filtered.
   - channel_of_interest (str, int, list, None): The channel(s) of interest to filter the dataframe. If None, no filtering is applied. If 'morphology', only morphology features are included.If an integer, only the specified channel is included. If a list, only the specified channels are included.If a string, only the specified channel is included.
   - exclude (str, list, None): The feature(s) to exclude from the filtered dataframe. If None, no features are excluded. If a string, the specified feature is excluded.If a list, the specified features are excluded.

   Returns:
   - filtered_df (pandas.DataFrame): The filtered dataframe based on the specified parameters.
   - features (list): The list of selected features after filtering.



.. py:function:: check_overlap(current_position, other_positions, threshold)

   Checks if the current position overlaps with any of the other positions
   within a specified threshold distance.

   :param current_position: The current position as a list, tuple, or array of coordinates.
   :type current_position: iterable
   :param other_positions: A collection of positions to compare against,
                           where each position is a list, tuple, or array of coordinates.
   :type other_positions: iterable
   :param threshold: The distance threshold below which two positions are considered overlapping.
   :type threshold: float

   :returns:

             True if the current position overlaps with any of the other positions,
                   False otherwise.
   :rtype: bool


.. py:function:: find_non_overlapping_position(x, y, image_positions, threshold, max_attempts=100)

   Finds a new position near the given coordinates (x, y) that does not overlap
   with any of the positions in the provided image_positions list, based on a
   specified threshold.

   :param x: The x-coordinate of the initial position.
   :type x: float
   :param y: The y-coordinate of the initial position.
   :type y: float
   :param image_positions: A list of (x, y) tuples representing
                           existing positions to avoid overlapping with.
   :type image_positions: list of tuples
   :param threshold: The minimum distance required to avoid overlap.
   :type threshold: float
   :param max_attempts: The maximum number of attempts to find a
                        non-overlapping position. Defaults to 100.
   :type max_attempts: int, optional

   :returns: A tuple (new_x, new_y) representing the new non-overlapping
             position. If no suitable position is found within the maximum attempts,
             the original position (x, y) is returned.
   :rtype: tuple


.. py:function:: search_reduction_and_clustering(numeric_data, n_neighbors, min_dist, metric, eps, min_samples, clustering, reduction_method, verbose, reduction_param=None, embedding=None, n_jobs=-1)

   Perform dimensionality reduction and clustering on the given data.

   Args:
   numeric_data (np.array): Numeric data to process.
   n_neighbors (int): Number of neighbors for UMAP or perplexity for tSNE.
   min_dist (float): Minimum distance for UMAP.
   metric (str): Metric for UMAP, tSNE, and DBSCAN.
   eps (float): Epsilon for DBSCAN clustering.
   min_samples (int): Minimum samples for DBSCAN or number of clusters for KMeans.
   clustering (str): Clustering method ('DBSCAN' or 'KMeans').
   reduction_method (str): Dimensionality reduction method ('UMAP' or 'tSNE').
   verbose (bool): Whether to print verbose output.
   reduction_param (dict): Additional parameters for the reduction method.
   embedding (np.array): Precomputed embedding (optional).
   n_jobs (int): Number of parallel jobs to run.

   Returns:
   embedding (np.array): Embedding of the data.
   labels (np.array): Cluster labels.


.. py:function:: load_image(image_path)

   Load and transform an image into a normalized tensor.

   Applies resizing to 224x224, converts to a tensor, and normalizes using
   ImageNet mean and standard deviation.

   :param image_path: Path to the input image.
   :type image_path: str

   :returns: Transformed image tensor with shape (1, 3, 224, 224).
   :rtype: torch.Tensor


.. py:function:: extract_features(image_paths, resnet=resnet50)

   Extracts features from a list of image paths using a pre-trained ResNet model.

   :param image_paths: A list of file paths to the images from which features are to be extracted.
   :type image_paths: list of str
   :param resnet: A ResNet model class to use for feature extraction.
                  Defaults to torchvision.models.resnet50.
   :type resnet: torchvision.models, optional

   :returns: A 2D array where each row corresponds to the extracted features of an image.
   :rtype: numpy.ndarray


.. py:function:: check_normality(series)

   Test whether a given data series follows a normal distribution.

   This function uses the D'Agostino and Pearson's test to check the null
   hypothesis that the data comes from a normal distribution. If the p-value
   is less than the significance level (alpha), the null hypothesis is rejected.

   :param series: The data series to test for normality.
   :type series: array-like

   :returns:

             True if the data follows a normal distribution (p >= alpha),
                   False otherwise.
   :rtype: bool


.. py:function:: random_forest_feature_importance(all_df, cluster_col='cluster')

   Computes feature importance using a Random Forest Classifier.

   This function takes a DataFrame, selects numeric features, and computes
   the importance of each feature in predicting the specified cluster column
   using a Random Forest Classifier. The results are returned as a sorted
   DataFrame of feature importances.

   :param all_df: The input DataFrame containing the data.
   :type all_df: pd.DataFrame
   :param cluster_col: The name of the column representing the
                       target variable (cluster). Defaults to 'cluster'.
   :type cluster_col: str, optional

   :returns: A DataFrame containing the features and their corresponding
             importance scores, sorted in descending order of importance.
   :rtype: pd.DataFrame

   .. rubric:: Notes

   - The function assumes that the target column (`cluster_col`) is numeric.
   - Standard scaling is applied to the numeric features before fitting the model.
   - The Random Forest Classifier is initialized with 100 estimators and a
     random state of 42 for reproducibility.


.. py:function:: perform_statistical_tests(all_df, cluster_col='cluster')

   Perform ANOVA and Kruskal-Wallis tests on numeric features grouped by clusters.

   This function evaluates whether numeric features differ significantly across groups
   defined by the `cluster_col`.

   :param all_df: DataFrame containing numeric features and cluster assignments.
   :type all_df: pd.DataFrame
   :param cluster_col: Name of the column indicating cluster/group labels. Defaults to 'cluster'.
   :type cluster_col: str, optional

   :returns:

                 - anova_df (pd.DataFrame): ANOVA test results with columns
                   ['Feature', 'ANOVA_Statistic', 'ANOVA_pValue'].
                 - kruskal_df (pd.DataFrame): Kruskal-Wallis test results with columns
                   ['Feature', 'Kruskal_Statistic', 'Kruskal_pValue'].
   :rtype: tuple

   .. rubric:: Notes

   - Normality of each feature is assessed using `check_normality`.
   - ANOVA is used for normally distributed features.
   - Kruskal-Wallis is used for non-normal features.
   - Assumes `check_normality`, `scipy.stats.f_oneway`, and `scipy.stats.kruskal` are available.


.. py:function:: combine_results(rf_df, anova_df, kruskal_df)

   Combines results from multiple DataFrames into a single DataFrame.

   This function merges three DataFrames (`rf_df`, `anova_df`, and `kruskal_df`)
   on the 'Feature' column using a left join. The resulting DataFrame contains
   all features from `rf_df` and their corresponding data from `anova_df` and
   `kruskal_df` where available.

   :param rf_df: A DataFrame containing features and their associated
                 data from a random forest analysis.
   :type rf_df: pd.DataFrame
   :param anova_df: A DataFrame containing features and their associated
                    data from an ANOVA analysis.
   :type anova_df: pd.DataFrame
   :param kruskal_df: A DataFrame containing features and their associated
                      data from a Kruskal-Wallis analysis.
   :type kruskal_df: pd.DataFrame

   :returns:

             A combined DataFrame with features and their associated data
                           from all three input DataFrames.
   :rtype: pd.DataFrame


.. py:function:: cluster_feature_analysis(all_df, cluster_col='cluster')

   Perform feature analysis for clustering by combining results from
   random forest feature importance and statistical tests.

   This function calculates feature importance using a random forest model,
   performs statistical tests (ANOVA and Kruskal-Wallis) to assess the
   significance of features, and combines the results into a single DataFrame.

   :param all_df: The input DataFrame containing features and cluster labels.
   :type all_df: pd.DataFrame
   :param cluster_col: The name of the column representing cluster labels.
                       Defaults to 'cluster'.
   :type cluster_col: str, optional

   :returns:

             A DataFrame combining the results of random forest feature
                           importance and statistical tests for feature analysis.
   :rtype: pd.DataFrame


.. py:function:: process_mask_file_adjust_cell(file_name, parasite_folder, cell_folder, nuclei_folder, overlap_threshold, perimeter_threshold)

   Processes and adjusts a cell mask file based on parasite overlap and perimeter thresholds.

   This function loads parasite, cell, and nuclei mask files, merges cells based on parasite overlap
   and perimeter thresholds, and saves the updated cell mask back to the file system.

   :param file_name: The name of the mask file to process.
   :type file_name: str
   :param parasite_folder: The directory containing parasite mask files.
   :type parasite_folder: str
   :param cell_folder: The directory containing cell mask files.
   :type cell_folder: str
   :param nuclei_folder: The directory containing nuclei mask files.
   :type nuclei_folder: str
   :param overlap_threshold: The threshold for parasite overlap to merge cells.
   :type overlap_threshold: float
   :param perimeter_threshold: The threshold for cell perimeter to merge cells.
   :type perimeter_threshold: float

   :returns: The time taken to process the mask file, in seconds.
   :rtype: float

   :raises ValueError: If the corresponding cell or nuclei mask file for the given file_name is not found.


.. py:function:: adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, overlap_threshold=5, perimeter_threshold=30, n_jobs=None)

   Adjusts cell masks based on parasite, cell, and nuclei data files.
   This function processes `.npy` files from the specified folders to adjust cell masks
   by considering overlap and perimeter thresholds. It uses multiprocessing to parallelize
   the processing of files.
   :param parasite_folder: Path to the folder containing parasite `.npy` files.
   :type parasite_folder: str
   :param cell_folder: Path to the folder containing cell `.npy` files.
   :type cell_folder: str
   :param nuclei_folder: Path to the folder containing nuclei `.npy` files.
   :type nuclei_folder: str
   :param overlap_threshold: Threshold for overlap adjustment. Defaults to 5.
   :type overlap_threshold: int, optional
   :param perimeter_threshold: Threshold for perimeter adjustment. Defaults to 30.
   :type perimeter_threshold: int, optional
   :param n_jobs: Number of parallel jobs to run. Defaults to the number of CPU cores minus 2.
   :type n_jobs: int, optional

   :raises ValueError: If the number of files in the parasite, cell, and nuclei folders do not match.

   .. rubric:: Notes

   - The function assumes that the files in the folders are named in a way that allows
     them to be sorted and matched correctly.
   - Progress is printed to the console during processing.

   :returns: None


.. py:function:: adjust_cell_masks_v1(parasite_folder, cell_folder, nuclei_folder, overlap_threshold=5, perimeter_threshold=30)

   Process all npy files in the given folders. Merge and relabel cells in cell masks
   based on parasite overlap and cell perimeter sharing conditions.

   :param parasite_folder: Path to the folder containing parasite masks.
   :type parasite_folder: str
   :param cell_folder: Path to the folder containing cell masks.
   :type cell_folder: str
   :param nuclei_folder: Path to the folder containing nuclei masks.
   :type nuclei_folder: str
   :param overlap_threshold: The percentage threshold for merging cells based on parasite overlap.
   :type overlap_threshold: float
   :param perimeter_threshold: The percentage threshold for merging cells based on shared perimeter.
   :type perimeter_threshold: float


.. py:function:: process_masks(mask_folder, image_folder, channel, batch_size=50, n_clusters=2, plot=False)

   Processes mask files by measuring object properties, clustering objects, and removing objects
   not belonging to the largest cluster.
   :param mask_folder: Path to the folder containing mask files (.npy format).
   :type mask_folder: str
   :param image_folder: Path to the folder containing corresponding image files (.npy format).
   :type image_folder: str
   :param channel: The channel index to extract from the image files.
   :type channel: int
   :param batch_size: Number of files to process in each batch. Defaults to 50.
   :type batch_size: int, optional
   :param n_clusters: Number of clusters for KMeans clustering. Defaults to 2.
   :type n_clusters: int, optional
   :param plot: Whether to plot the clustering results using PCA. Defaults to False.
   :type plot: bool, optional

   :returns: The function modifies the mask files in place by removing objects not in the largest cluster.
   :rtype: None

   .. rubric:: Notes

   - The mask files are expected to be in .npy format and contain labeled regions.
   - The image files are expected to be in .npy format and have the same names as the mask files.
   - The function assumes that the mask and image files are sorted in the same order.
   - The clustering is performed on accumulated object properties across all files.
   - The largest cluster is determined based on the number of objects in each cluster.


.. py:function:: merge_regression_res_with_metadata(results_file, metadata_file, name='_metadata')

   Merge regression results with metadata using gene identifiers.

   Reads regression results and metadata from two CSV files, extracts gene identifiers,
   merges the data on gene names, and saves the merged DataFrame.

   :param results_file: Path to the regression results CSV. Must contain a 'feature' column.
   :type results_file: str
   :param metadata_file: Path to the metadata CSV. Must contain a 'Gene ID' column.
   :type metadata_file: str
   :param name: Suffix for the output file name. Defaults to '_metadata'.
   :type name: str, optional

   :returns: Merged DataFrame with regression results and metadata.
   :rtype: pandas.DataFrame

   .. rubric:: Notes

   - Extracts gene names from 'feature' (format: '[gene]') and 'Gene ID' (format: 'prefix_gene').
   - Gene extraction failures result in NaNs in the merge but are not dropped.
   - Output CSV is saved next to the input `results_file`, with the suffix appended.


.. py:function:: process_vision_results(df, threshold=0.5)

   Process vision results by extracting metadata from file paths and thresholding predictions.

   :param df: DataFrame with vision results. Must include 'path' and 'pred' columns.
   :type df: pd.DataFrame
   :param threshold: Threshold for classifying predictions. Defaults to 0.5.
   :type threshold: float, optional

   :returns:

             Modified DataFrame with added columns:
                 - 'plateID', 'rowID', 'columnID', 'fieldID', 'object': extracted from 'path'
                 - 'prc': combination of 'plateID', 'rowID', and 'columnID'
                 - 'cv_predictions': binary classification based on `threshold`
   :rtype: pd.DataFrame


.. py:function:: get_ml_results_paths(src, model_type='xgboost', channel_of_interest=1)

   Generate file paths for machine learning result outputs based on model type and feature selection.

   :param src: Base directory where the results folder structure will be created.
   :type src: str
   :param model_type: Type of ML model used (e.g., 'xgboost', 'random_forest'). Defaults to 'xgboost'.
   :type model_type: str, optional
   :param channel_of_interest: Feature set specification:
                               - int: Single channel (e.g., 1)
                               - list[int]: Multiple channels (e.g., [1, 2, 3])
                               - 'morphology': Use only morphology features
                               - None: Use all features
                               Defaults to 1.
   :type channel_of_interest: int | list[int] | str | None, optional

   :returns:

             Tuple of 10 file paths:
                 - data_path: CSV with predictions or main results
                 - permutation_path: CSV with permutation importances
                 - feature_importance_path: CSV with model feature importances
                 - model_metricks_path: CSV with metrics from trained model
                 - permutation_fig_path: PDF plot of permutation importances
                 - feature_importance_fig_path: PDF plot of feature importances
                 - shap_fig_path: PDF SHAP summary plot
                 - plate_heatmap_path: PDF visualization of plate layout
                 - settings_csv: CSV with ML settings and parameters
                 - ml_features: CSV with extracted feature data used for training
   :rtype: tuple[str, ...]

   :raises ValueError: If `channel_of_interest` is not an int, list, None, or 'morphology'.

   .. rubric:: Example

   >>> get_ml_results_paths('/home/user/data', model_type='random_forest', channel_of_interest=[1, 2])
   (
       '/home/user/data/results/random_forest/channels_1_2/results.csv',
       '/home/user/data/results/random_forest/channels_1_2/permutation.csv',
       '/home/user/data/results/random_forest/channels_1_2/feature_importance.csv',
       '/home/user/data/results/random_forest/channels_1_2/random_forest_model.csv',
       '/home/user/data/results/random_forest/channels_1_2/permutation.pdf',
       '/home/user/data/results/random_forest/channels_1_2/feature_importance.pdf',
       '/home/user/data/results/random_forest/channels_1_2/shap.pdf',
       '/home/user/data/results/random_forest/channels_1_2/plate_heatmap.pdf',
       '/home/user/data/results/random_forest/channels_1_2/ml_settings.csv',
       '/home/user/data/results/random_forest/channels_1_2/ml_features.csv'
   )


.. py:function:: augment_image(image)

   Perform data augmentation by rotating and reflecting the image.

   Args:
   - image (PIL Image or numpy array): The input image.

   Returns:
   - augmented_images (list): A list of augmented images.


.. py:function:: augment_dataset(dataset, is_grayscale=False)

   Perform data augmentation on the entire dataset by rotating and reflecting the images.

   Args:
   - dataset (list of tuples): The input dataset, each entry is a tuple (image, label, filename).
   - is_grayscale (bool): Flag indicating if the images are grayscale.

   Returns:
   - augmented_dataset (list of tuples): A dataset with augmented (image, label, filename) tuples.


.. py:function:: convert_and_relabel_masks(folder_path)

   Converts all int64 npy masks in a folder to uint16 with relabeling to ensure all labels are retained.

   Args:
   - folder_path (str): The path to the folder containing int64 npy mask files.

   Returns:
   - None


.. py:function:: correct_masks(src)

   Corrects and processes mask files located in the specified source directory.

   This function performs the following steps:
   1. Constructs the file path for the cell mask stack within the 'masks' subdirectory of the source.
   2. Converts and relabels the masks using the `convert_and_relabel_masks` function.
   3. Loads and concatenates arrays from the source directory using the `_load_and_concatenate_arrays` function.

   :param src: The path to the source directory containing the mask files.
   :type src: str

   :returns: None


.. py:function:: count_reads_in_fastq(fastq_file)

   Counts the number of reads in a FASTQ file.

   A FASTQ file contains sequencing reads, where each read is represented
   by four lines: a header, the sequence, a separator, and the quality scores.
   This function calculates the total number of reads by dividing the total
   number of lines in the file by 4.

   :param fastq_file: Path to the FASTQ file, which can be gzip-compressed.
   :type fastq_file: str

   :returns: The number of reads in the FASTQ file.
   :rtype: int


.. py:function:: get_cuda_version()

   Retrieves the installed CUDA version by invoking the `nvcc --version` command.

   :returns:

             The CUDA version as a string with dots removed (e.g., '110' for version 11.0),
                  or None if the `nvcc` command is not found or an error occurs.
   :rtype: str


.. py:function:: all_elements_match(list1, list2)

   Check if all elements in the first list are present in the second list.

   :param list1: The first list containing elements to check.
   :type list1: list
   :param list2: The second list to check against.
   :type list2: list

   :returns: True if all elements in list1 are found in list2, False otherwise.
   :rtype: bool


.. py:function:: prepare_batch_for_segmentation(batch)

   Prepare a batch of images for segmentation by ensuring correct data type and normalization.

   :param batch: Batch of images with shape (N, H, W, C), where:
                 - N: number of images
                 - H, W: height and width
                 - C: number of channels (e.g., 1 for grayscale, 3 for RGB)
   :type batch: np.ndarray

   :returns: Batch with dtype `float32`, normalized to [0, 1] if needed.
   :rtype: np.ndarray

   .. rubric:: Notes

   - Converts to `float32` if not already.
   - Each image is divided by its own maximum pixel value if that value > 1.


.. py:function:: check_index(df, elements=5, split_char='_')

   Checks the indices of a DataFrame to ensure they can be split into a specified number of parts.

   :param df: The DataFrame whose indices are to be checked.
   :type df: pandas.DataFrame
   :param elements: The expected number of parts after splitting an index. Defaults to 5.
   :type elements: int, optional
   :param split_char: The character used to split the index. Defaults to '_'.
   :type split_char: str, optional

   :raises ValueError: If any index cannot be split into the specified number of parts,
       a ValueError is raised listing the problematic indices.

   .. rubric:: Example

   >>> import pandas as pd
   >>> data = {'col1': [1, 2, 3]}
   >>> df = pd.DataFrame(data, index=['a_b_c_d_e', 'f_g_h_i', 'j_k_l_m_n'])
   >>> check_index(df)
   ValueError: Found 1 problematic indices that do not split into 5 parts.


.. py:function:: map_condition(col_value, neg='c1', pos='c2', mix='c3')

   Maps a given column value to a specific condition label.

   :param col_value: The value to be mapped.
   :type col_value: str
   :param neg: The value representing the 'neg' condition. Defaults to 'c1'.
   :type neg: str, optional
   :param pos: The value representing the 'pos' condition. Defaults to 'c2'.
   :type pos: str, optional
   :param mix: The value representing the 'mix' condition. Defaults to 'c3'.
   :type mix: str, optional

   :returns:

             A string representing the mapped condition:
                  - 'neg' if col_value matches the neg parameter.
                  - 'pos' if col_value matches the pos parameter.
                  - 'mix' if col_value matches the mix parameter.
                  - 'screen' if col_value does not match any of the above.
   :rtype: str


.. py:function:: download_models(repo_id='einarolafsson/models', retries=5, delay=5)

   Downloads all model files from Hugging Face and stores them in the `resources/models` directory
   within the installed `spacr` package.

   :param repo_id: The repository ID on Hugging Face (default is 'einarolafsson/models').
   :type repo_id: str
   :param retries: Number of retry attempts in case of failure.
   :type retries: int
   :param delay: Delay in seconds between retries.
   :type delay: int

   :returns: The local path to the downloaded models.
   :rtype: str


.. py:function:: generate_cytoplasm_mask(nucleus_mask, cell_mask)

   Generates a cytoplasm mask from nucleus and cell masks.

   Args:
   - nucleus_mask (np.array): Binary or segmented mask of the nucleus (non-zero values represent nucleus).
   - cell_mask (np.array): Binary or segmented mask of the whole cell (non-zero values represent cell).

   Returns:
   - cytoplasm_mask (np.array): Mask for the cytoplasm (1 for cytoplasm, 0 for nucleus and pathogens).


.. py:function:: add_column_to_database(settings)

   Adds a new column to the database table by matching on a common column from the DataFrame.
   If the column already exists in the database, it adds the column with a suffix.
   NaN values will remain as NULL in the database.

   :param settings: A dictionary containing the following keys:
                    csv_path (str): Path to the CSV file with the data to be added.
                    db_path (str): Path to the SQLite database (or connection string for other databases).
                    table_name (str): The name of the table in the database.
                    update_column (str): The name of the new column in the DataFrame to add to the database.
                    match_column (str): The common column used to match rows.
   :type settings: dict

   :returns: None


.. py:function:: fill_holes_in_mask(mask)

   Fill holes in each object in the mask while keeping objects separated.

   :param mask: A labeled mask where each object has a unique integer value.
   :type mask: np.ndarray

   :returns: A mask with holes filled and original labels preserved.
   :rtype: np.ndarray


.. py:function:: correct_metadata_column_names(df)

   Standardize column names in a metadata DataFrame.

   This function renames commonly used but inconsistent metadata columns to a standardized format.

   Renaming rules:
       - 'plate_name' → 'plateID'
       - 'column_name' or 'col' → 'columnID'
       - 'row_name' → 'rowID'
       - 'grna_name' → 'grna'
       - If 'plate_row' exists, it is split into 'plateID' and 'rowID' using '_' as a delimiter.

   :param df: Input DataFrame with metadata columns.
   :type df: pd.DataFrame

   :returns: DataFrame with standardized column names.
   :rtype: pd.DataFrame


.. py:function:: control_filelist(folder, mode='columnID', values=['01', '02'])

.. py:function:: rename_columns_in_db(db_path)

   Renames specific columns in all user tables of a SQLite database based on a predefined mapping.

   This function connects to the SQLite database at the given path, retrieves all user-defined tables,
   and renames columns in those tables according to the `rename_map` dictionary. If a column with the
   old name exists and the new name does not already exist in the same table, the column is renamed.

   :param db_path: The file path to the SQLite database.
   :type db_path: str

   Behavior:
       - Retrieves all user-defined tables in the database.
       - For each table, checks the column names against the `rename_map`.
       - Renames columns as specified in the `rename_map` if conditions are met.
       - Commits the changes to the database.

   .. rubric:: Notes

   - The `rename_map` dictionary defines the mapping of old column names to new column names.
   - If a column with the new name already exists in a table, the old column will not be renamed.
   - The function uses SQLite's `ALTER TABLE ... RENAME COLUMN` syntax, which requires SQLite version 3.25.0 or higher.

   .. rubric:: Example

   rename_columns_in_db("/path/to/database.db")

   :raises sqlite3.OperationalError: If there are issues executing SQL commands, such as unsupported SQLite versions.


.. py:function:: group_feature_class(df, feature_groups=['cell', 'cytoplasm', 'nucleus', 'pathogen'], name='compartment')

   Classify and group features by category, then compute summed importance for each group.

   This function adds a new column to the DataFrame to classify features based on regex matching
   against the given `feature_groups`. It then computes the total importance per group.

   :param df: Input DataFrame with at least the columns `'feature'` and `'importance'`.
   :type df: pd.DataFrame
   :param feature_groups: List of feature group identifiers (used in regex matching).
                          Defaults to ['cell', 'cytoplasm', 'nucleus', 'pathogen'].
   :type feature_groups: list of str, optional
   :param name: Name of the new column used to store group classifications.
                If set to `'channel'`, missing values in that column will be filled with `'morphology'`.
                Defaults to 'compartment'.
   :type name: str, optional

   :returns:

             Modified DataFrame including:
                 - A new column with group classifications (based on `name` argument).
                 - A summary with summed importance for each group and a row for total importance.
   :rtype: pd.DataFrame

   .. rubric:: Notes

   - A feature matching multiple groups will have their labels joined with a hyphen (e.g., "cell-nucleus").
   - Classification is done via regex search on the `'feature'` column.


.. py:function:: delete_intermedeate_files(settings)

   Delete intermediate files and directories specified in the settings dictionary.

   This function removes a predefined set of subdirectories under the path given by
   `settings['src']`. If deletion fails, an error message is printed.

   :param settings: Must include the key `'src'`, the base directory containing
                    intermediate files and subdirectories.
   :type settings: dict

   Behavior:
       - Verifies that `'src'` and its `orig/` subdirectory exist.
       - Deletes the following under `src` if they exist:
           * 'stack'
           * 'masks'
           * directories '1' through '10'
       - Compares lengths of `merged_stack` and `path_stack` to decide on deletion.
       - Prints success or error messages for each directory.

   .. rubric:: Notes

   - If `src` or `orig/` is missing, the function exits early.
   - Any deletion exceptions are caught and reported; no exceptions are raised.

   .. rubric:: Example

   >>> settings = {'src': '/path/to/source'}
   >>> delete_intermedeate_files(settings)


.. py:function:: filter_and_save_csv(input_csv, output_csv, column_name, upper_threshold, lower_threshold)

   Reads a CSV into a DataFrame, filters rows based on a column for values > upper_threshold and < lower_threshold,
   and saves the filtered DataFrame to a new CSV file.

   :param input_csv: Path to the input CSV file.
   :type input_csv: str
   :param output_csv: Path to save the filtered CSV file.
   :type output_csv: str
   :param column_name: Column name to apply the filters on.
   :type column_name: str
   :param upper_threshold: Upper threshold for filtering (values greater than this are retained).
   :type upper_threshold: float
   :param lower_threshold: Lower threshold for filtering (values less than this are retained).
   :type lower_threshold: float

   :returns: None


.. py:function:: extract_tar_bz2_files(folder_path)

   Extracts all .tar.bz2 files in the given folder into subfolders with the same name as the tar file.

   :param folder_path: Path to the folder containing .tar.bz2 files.
   :type folder_path: str


.. py:function:: calculate_shortest_distance(df, object1, object2)

   Calculate the shortest edge-to-edge distance between two objects (e.g., pathogen and nucleus).

   Args:
   - df: Pandas DataFrame containing measurements
   - object1: String, name of the first object (e.g., "pathogen")
   - object2: String, name of the second object (e.g., "nucleus")

   Returns:
   - df: Pandas DataFrame with a new column for shortest edge-to-edge distance.


.. py:function:: format_path_for_system(path)

   Takes a file path and reformats it to be compatible with the current operating system.

   :param path: The file path to be formatted.
   :type path: str

   :returns: The formatted path for the current operating system.
   :rtype: str


.. py:function:: normalize_src_path(src)

   Ensures that the 'src' value is properly formatted as either a list of strings or a single string.

   :param src: The input source path(s).
   :type src: str or list

   :returns:

             A correctly formatted list if the input was a list (or string representation of a list),
                          otherwise a single string.
   :rtype: list or str


.. py:function:: generate_image_path_map(root_folder, valid_extensions=('tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp', 'czi', 'nd2', 'lif'))

   Recursively scans a folder and its subfolders for images, then creates a mapping of:
   {original_image_path: new_image_path}, where the new path includes all subfolder names.

   :param root_folder: The root directory to scan for images.
   :type root_folder: str
   :param valid_extensions: Tuple of valid image file extensions.
   :type valid_extensions: tuple

   :returns: A dictionary mapping original image paths to their new paths.
   :rtype: dict


.. py:function:: copy_images_to_consolidated(image_path_map, root_folder)

   Copies images from their original locations to a 'consolidated' folder,
   renaming them according to the generated dictionary.

   :param image_path_map: Dictionary mapping {original_path: new_path}.
   :type image_path_map: dict
   :param root_folder: The root directory where the 'consolidated' folder will be created.
   :type root_folder: str


.. py:function:: correct_metadata(df)

   Corrects and standardizes the metadata column names in a DataFrame.
   This function checks for specific column names in the input DataFrame and
   renames or maps them to standardized names for consistency. The following
   transformations are applied:
   - If 'object_name' exists, it is mapped to 'objectID'.
   - If 'field_name' exists, it is mapped to 'fieldID'.
   - If 'plate' or 'plate_name' exists, they are mapped to 'plateID'.
   - If 'row' or 'row_name' exists, they are renamed to 'rowID'.
   - If 'col', 'column', or 'column_name' exists, they are renamed to 'columnID'.
   - If 'field' or 'field_name' exists, they are renamed to 'fieldID'.
   :param df: The input DataFrame containing metadata columns.
   :type df: pandas.DataFrame

   :returns: The DataFrame with standardized metadata column names.
   :rtype: pandas.DataFrame


.. py:function:: remove_outliers_by_group(df, group_col, value_col, method='iqr', threshold=1.5)

   Removes outliers from `value_col` within each group defined by `group_col`.

   :param df: The input DataFrame.
   :type df: pd.DataFrame
   :param group_col: Column name to group by.
   :type group_col: str
   :param value_col: Column containing values to check for outliers.
   :type value_col: str
   :param method: 'iqr' or 'zscore'.
   :type method: str
   :param threshold: Threshold multiplier for IQR (default 1.5) or z-score.
   :type threshold: float

   :returns: A DataFrame with outliers removed.
   :rtype: pd.DataFrame


