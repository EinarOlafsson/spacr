spacr.deep_spacr
================

.. py:module:: spacr.deep_spacr






Module Contents
---------------

.. py:function:: apply_model(src, model_path, image_size=224, batch_size=64, normalize=True, n_jobs=10)

   Apply a trained binary classification model to a folder of images.

   Loads a PyTorch model and applies it to images in the specified folder using batch inference.
   Supports optional normalization and GPU acceleration. Outputs prediction probabilities and
   saves results as a CSV file alongside the model.

   :param src: Path to a folder containing input images (e.g., PNG, JPG).
   :type src: str
   :param model_path: Path to a trained PyTorch model file (.pt or .pth).
   :type model_path: str
   :param image_size: Size to center-crop input images to. Default is 224.
   :type image_size: int, optional
   :param batch_size: Number of images to process per batch. Default is 64.
   :type batch_size: int, optional
   :param normalize: If True, normalize images to [-1, 1] using ImageNet-style transform. Default is True.
   :type normalize: bool, optional
   :param n_jobs: Number of subprocesses to use for data loading. Default is 10.
   :type n_jobs: int, optional

   :returns:

             A DataFrame with two columns:
                 - "path": Filenames of processed images.
                 - "pred": Model output probabilities (sigmoid of logits).
   :rtype: pandas.DataFrame

   Saves:
       A CSV file named like <model_path><YYMMDD>_<ext>_test_result.csv, containing the prediction results.

   .. rubric:: Notes

   - Uses GPU if available, otherwise runs on CPU.
   - Assumes model outputs raw logits for binary classification (sigmoid is applied).
   - The input folder must contain only images readable by `PIL.Image.open`.


.. py:function:: apply_model_to_tar(settings={})

   Apply a trained model to images stored inside a tar archive.

   Loads a model and applies it to images within a `.tar` archive using batch inference. Results are
   filtered by a probability threshold and saved to a CSV. Supports GPU acceleration and normalization.

   :param settings: Dictionary with the following keys:
                    - tar_path (str): Path to the tar archive with input images.
                    - model_path (str): Path to the trained PyTorch model (.pt/.pth).
                    - image_size (int): Center crop size for input images. Default is 224.
                    - batch_size (int): Batch size for DataLoader. Default is 64.
                    - normalize (bool): Apply normalization to [-1, 1]. Default is True.
                    - n_jobs (int): Number of workers for data loading. Default is system CPU count - 4.
                    - verbose (bool): If True, print progress and model details.
                    - score_threshold (float): Probability threshold for positive classification (used in result filtering).
   :type settings: dict

   :returns:

             DataFrame with:
                 - "path": Filenames inside the tar archive.
                 - "pred": Model prediction scores (sigmoid output).
   :rtype: pandas.DataFrame

   Saves:
       A CSV file with prediction results to the same directory as the tar file.


.. py:function:: evaluate_model_performance(model, loader, epoch, loss_type)

   Evaluates the performance of a model on a given data loader.

   :param model: The model to evaluate.
   :type model: torch.nn.Module
   :param loader: The data loader to evaluate the model on.
   :type loader: torch.utils.data.DataLoader
   :param loader_name: The name of the data loader.
   :type loader_name: str
   :param epoch: The current epoch number.
   :type epoch: int
   :param loss_type: The type of loss function to use.
   :type loss_type: str

   :returns: The classification metrics data as a DataFrame.
             prediction_pos_probs (list): The positive class probabilities for each prediction.
             all_labels (list): The true labels for each prediction.
   :rtype: data_df (pandas.DataFrame)


.. py:function:: test_model_core(model, loader, loader_name, epoch, loss_type)

   Evaluate a trained model on a test DataLoader and return performance metrics and predictions.

   This function evaluates a binary classification model using a specified loss function, computes
   classification metrics, and logs predictions, targets, and file-level results.

   :param model: The trained PyTorch model to evaluate.
   :type model: torch.nn.Module
   :param loader: DataLoader providing test data and labels.
   :type loader: torch.utils.data.DataLoader
   :param loader_name: Identifier name for the loader (used for logging/debugging).
   :type loader_name: str
   :param epoch: Current epoch number (used for metric tracking).
   :type epoch: int
   :param loss_type: Type of loss function to use for reporting (e.g., 'bce', 'focal').
   :type loss_type: str

   :returns:     - data_df (pd.DataFrame): DataFrame containing classification metrics for the test set.
                 - prediction_pos_probs (list): List of predicted probabilities for the positive class.
                 - all_labels (list): Ground truth binary labels.
                 - results_df (pd.DataFrame): Per-sample results, including filename, true label, predicted label,
                   and probability for class 1.
   :rtype: tuple


.. py:function:: test_model_performance(loaders, model, loader_name_list, epoch, loss_type)

   Test the performance of a model on given data loaders.

   :param loaders: List of data loaders.
   :type loaders: list
   :param model: The model to be tested.
   :param loader_name_list: List of names for the data loaders.
   :type loader_name_list: list
   :param epoch: The current epoch.
   :type epoch: int
   :param loss_type: The type of loss function.

   :returns: A tuple containing the test results and the results dataframe.
   :rtype: tuple


.. py:function:: train_test_model(settings)

.. py:function:: train_model(dst, model_type, train_loaders, epochs=100, learning_rate=0.0001, weight_decay=0.05, amsgrad=False, optimizer_type='adamw', use_checkpoint=False, dropout_rate=0, n_jobs=20, val_loaders=None, test_loaders=None, init_weights='imagenet', intermedeate_save=None, chan_dict=None, schedule=None, loss_type='binary_cross_entropy_with_logits', gradient_accumulation=False, gradient_accumulation_steps=4, channels=['r', 'g', 'b'], verbose=False)

   Trains a model using the specified parameters.

   :param dst: The destination path to save the model and results.
   :type dst: str
   :param model_type: The type of model to train.
   :type model_type: str
   :param train_loaders: A list of training data loaders.
   :type train_loaders: list
   :param epochs: The number of training epochs. Defaults to 100.
   :type epochs: int, optional
   :param learning_rate: The learning rate for the optimizer. Defaults to 0.0001.
   :type learning_rate: float, optional
   :param weight_decay: The weight decay for the optimizer. Defaults to 0.05.
   :type weight_decay: float, optional
   :param amsgrad: Whether to use AMSGrad for the optimizer. Defaults to False.
   :type amsgrad: bool, optional
   :param optimizer_type: The type of optimizer to use. Defaults to 'adamw'.
   :type optimizer_type: str, optional
   :param use_checkpoint: Whether to use checkpointing during training. Defaults to False.
   :type use_checkpoint: bool, optional
   :param dropout_rate: The dropout rate for the model. Defaults to 0.
   :type dropout_rate: float, optional
   :param n_jobs: The number of n_jobs for data loading. Defaults to 20.
   :type n_jobs: int, optional
   :param val_loaders: A list of validation data loaders. Defaults to None.
   :type val_loaders: list, optional
   :param test_loaders: A list of test data loaders. Defaults to None.
   :type test_loaders: list, optional
   :param init_weights: The initialization weights for the model. Defaults to 'imagenet'.
   :type init_weights: str, optional
   :param intermedeate_save: The intermediate save thresholds. Defaults to None.
   :type intermedeate_save: list, optional
   :param chan_dict: The channel dictionary. Defaults to None.
   :type chan_dict: dict, optional
   :param schedule: The learning rate schedule. Defaults to None.
   :type schedule: str, optional
   :param loss_type: The loss function type. Defaults to 'binary_cross_entropy_with_logits'.
   :type loss_type: str, optional
   :param gradient_accumulation: Whether to use gradient accumulation. Defaults to False.
   :type gradient_accumulation: bool, optional
   :param gradient_accumulation_steps: The number of steps for gradient accumulation. Defaults to 4.
   :type gradient_accumulation_steps: int, optional

   :returns: None


.. py:function:: generate_activation_map(settings)

   Generate activation maps (Grad-CAM or saliency) from a trained model applied to a dataset stored in a tar archive.

   This function loads a model, computes class activation maps or saliency maps for each input image, and saves the
   results as images. Optionally, it plots batch-wise grids of maps and stores correlation results and image metadata
   into an SQL database.

   :param settings: Dictionary of parameters controlling activation map generation. Key fields include:

                    Required paths:
                        - dataset (str): Path to the `.tar` archive containing images.
                        - model_path (str): Path to the trained PyTorch model (.pt or .pth).

                    Model and method:
                        - model_type (str): Model architecture used (e.g., 'maxvit').
                        - cam_type (str): One of ['gradcam', 'gradcam_pp', 'saliency_image', 'saliency_channel'].
                        - target_layer (str or None): Name of the target layer for Grad-CAM (optional, required for Grad-CAM variants).

                    Input transforms:
                        - image_size (int): Size to center-crop images to (e.g., 224).
                        - normalize_input (bool): Whether to normalize images to [-1, 1] range.
                        - channels (list): Channel indices to select from input data (e.g., [0,1,2]).

                    Inference:
                        - batch_size (int): Number of images per inference batch.
                        - shuffle (bool): Whether to shuffle image order in DataLoader.
                        - n_jobs (int): Number of parallel DataLoader workers (default is CPU count - 4).

                    Output control:
                        - save (bool): If True, saves individual activation maps to disk.
                        - plot (bool): If True, generates and saves batch-wise PDF grid plots.
                        - overlay (bool): If True, overlays activation maps on input images.
                        - correlation (bool): If True, computes activation correlation features (e.g., Manders').

                    Correlation-specific:
                        - manders_thresholds (list or float): Threshold(s) for calculating Manders' coefficients.
   :type settings: dict

   :returns:     - PNG or JPEG activation maps organized by predicted class and well.
                 - PDF files with batch-wise overlay plots if `plot=True`.
                 - Activation image metadata and correlations saved to SQL database if `save=True`.
   :rtype: None. The following outputs are saved


.. py:function:: visualize_classes(model, dtype, class_names, **kwargs)

   Visualize synthetic input images that maximize class activation.

   :param model: The trained classification model.
   :type model: torch.nn.Module
   :param dtype: Data type or domain tag used for visualization.
   :type dtype: str
   :param class_names: List of class names (length 2 assumed for binary classification).
   :type class_names: list
   :param \*\*kwargs: Additional keyword arguments passed to `class_visualization()`.

   :returns: None. Displays matplotlib plots of class visualizations.


.. py:function:: visualize_integrated_gradients(src, model_path, target_label_idx=0, image_size=224, channels=[1, 2, 3], normalize=True, save_integrated_grads=False, save_dir='integrated_grads')

   Visualize integrated gradients for PNG images in a directory.

   :param src: Directory containing `.png` images.
   :type src: str
   :param model_path: Path to the trained PyTorch model.
   :type model_path: str
   :param target_label_idx: Index of the target class label.
   :type target_label_idx: int
   :param image_size: Image size after preprocessing (center crop).
   :type image_size: int
   :param channels: List of channels to extract (1-indexed).
   :type channels: list
   :param normalize: Whether to normalize image input to [-1, 1].
   :type normalize: bool
   :param save_integrated_grads: Whether to save integrated gradient maps.
   :type save_integrated_grads: bool
   :param save_dir: Directory to save integrated gradient outputs.
   :type save_dir: str

   :returns: None. Displays overlays and optionally saves saliency maps.


.. py:class:: SmoothGrad(model, n_samples=50, stdev_spread=0.15)

   Compute SmoothGrad saliency maps from a trained model.

   :param model: Trained classification model.
   :type model: torch.nn.Module
   :param n_samples: Number of noise samples to average.
   :type n_samples: int
   :param stdev_spread: Standard deviation of noise relative to input range.
   :type stdev_spread: float


   .. py:attribute:: model


   .. py:attribute:: n_samples
      :value: 50



   .. py:attribute:: stdev_spread
      :value: 0.15



   .. py:method:: compute_smooth_grad(input_tensor, target_class)


.. py:function:: visualize_smooth_grad(src, model_path, target_label_idx, image_size=224, channels=[1, 2, 3], normalize=True, save_smooth_grad=False, save_dir='smooth_grad')

   Visualize SmoothGrad maps for PNG images in a folder.

   :param src: Path to directory containing `.png` images.
   :type src: str
   :param model_path: Path to trained PyTorch model file.
   :type model_path: str
   :param target_label_idx: Index of the class to explain.
   :type target_label_idx: int
   :param image_size: Size for center cropping during preprocessing.
   :type image_size: int
   :param channels: Channel indices to extract from images.
   :type channels: list
   :param normalize: Whether to normalize inputs to [-1, 1].
   :type normalize: bool
   :param save_smooth_grad: If True, saves saliency maps to disk.
   :type save_smooth_grad: bool
   :param save_dir: Folder where smooth grad maps are saved.
   :type save_dir: str

   :returns: None. Displays overlay figures and optionally saves maps to disk.


.. py:function:: deep_spacr(settings={})

   Run deep learning-based classification workflow on microscopy data using SpaCr.

   This function handles dataset generation, model training, and inference using a trained model on tar-archived image datasets.
   Settings are filled using `deep_spacr_defaults`.

   :param settings: Dictionary of settings with the following keys:

                    General:
                        - src (str): Path to the input dataset.
                        - dataset (str): Path to a dataset archive.
                        - dataset_mode (str): Dataset generation mode. Typically 'metadata'.
                        - file_type (str): Type of input files (e.g., 'cell_png').
                        - file_metadata (str or None): Path to file-level metadata, if available.
                        - sample (int or None): Limit to N random samples for development/testing.
                        - experiment (str): Experiment name prefix. Default is 'exp.'.

                    Annotation and class mapping:
                        - annotation_column (str): Metadata column containing class annotations.
                        - annotated_classes (list): List of class IDs used for training (e.g., [1, 2]).
                        - classes (list): Class labels (e.g., ['nc', 'pc']).
                        - class_metadata (list of lists): Mapping of classes to metadata terms (e.g., [['c1'], ['c2']]).
                        - metadata_type_by (str): How to interpret metadata structure. Typically 'columnID'.

                    Image processing:
                        - channel_of_interest (int): Channel index to use for classification.
                        - png_type (str): Type of image format (e.g., 'cell_png').
                        - image_size (int): Input size (e.g., 224 for 224x224 crop).
                        - train_channels (list): Channels to use for training (e.g., ['r', 'g', 'b']).
                        - normalize (bool): Whether to normalize input images. Default is True.
                        - augment (bool): Whether to apply data augmentation.

                    Model and training:
                        - model_type (str): Model architecture (e.g., 'maxvit_t').
                        - optimizer_type (str): Optimizer (e.g., 'adamw').
                        - schedule (str): Learning rate scheduler ('reduce_lr_on_plateau' or 'step_lr').
                        - loss_type (str): Loss function ('focal_loss' or 'binary_cross_entropy_with_logits').
                        - dropout_rate (float): Dropout probability.
                        - init_weights (bool): Initialize model with pretrained weights.
                        - amsgrad (bool): Use AMSGrad variant of AdamW optimizer.
                        - use_checkpoint (bool): Enable checkpointing.
                        - intermedeate_save (bool): Save intermediate models during training.

                    Training control:
                        - train (bool): Enable training phase.
                        - test (bool): Enable evaluation on test set.
                        - train_DL_model (bool): Enable deep learning model training.
                        - generate_training_dataset (bool): Enable generation of train/test splits.
                        - test_split (float): Proportion of data used for testing.
                        - val_split (float): Fraction of training set used for validation.
                        - epochs (int): Number of training epochs.
                        - batch_size (int): Batch size for training and inference.
                        - learning_rate (float): Learning rate.
                        - weight_decay (float): L2 regularization strength.
                        - gradient_accumulation (bool): Accumulate gradients over multiple steps.
                        - gradient_accumulation_steps (int): Number of steps per gradient update.

                    Inference:
                        - apply_model_to_dataset (bool): Run prediction on tar dataset.
                        - tar_path (str): Path to tar file for inference input.
                        - model_path (str): Path to trained model file.
                        - score_threshold (float): Probability threshold for binary classification.

                    Execution:
                        - n_jobs (int): Number of parallel workers.
                        - pin_memory (bool): Whether to use pinned memory in DataLoader.
                        - verbose (bool): Print training and evaluation progress.
   :type settings: dict

   :returns: None. All outputs (trained models, predictions, settings) are saved to disk.


.. py:function:: model_knowledge_transfer(teacher_paths, student_save_path, data_loader, device='cpu', student_model_name='maxvit_t', pretrained=True, dropout_rate=None, use_checkpoint=False, alpha=0.5, temperature=2.0, lr=0.0001, epochs=10)

   Perform knowledge distillation from one or more teacher models to a student model.

   :param teacher_paths: Paths to pretrained teacher model files (.pth).
   :type teacher_paths: list of str
   :param student_save_path: Output path for the saved student model.
   :type student_save_path: str
   :param data_loader: DataLoader for training data.
   :type data_loader: torch.utils.data.DataLoader
   :param device: Device to use ('cpu' or 'cuda').
   :type device: str
   :param student_model_name: Name of the student model architecture (e.g., 'maxvit_t').
   :type student_model_name: str
   :param pretrained: Whether to initialize the student model with ImageNet weights.
   :type pretrained: bool
   :param dropout_rate: Dropout rate for the student model.
   :type dropout_rate: float or None
   :param use_checkpoint: Whether to use gradient checkpointing.
   :type use_checkpoint: bool
   :param alpha: Weighting factor between cross-entropy and distillation loss.
   :type alpha: float
   :param temperature: Temperature scaling for soft targets.
   :type temperature: float
   :param lr: Learning rate for optimizer.
   :type lr: float
   :param epochs: Number of training epochs.
   :type epochs: int

   :returns: The trained student model after knowledge distillation.
   :rtype: TorchModel


.. py:function:: model_fusion(model_paths, save_path, device='cpu', model_name='maxvit_t', pretrained=True, dropout_rate=None, use_checkpoint=False, aggregator='mean')

   Fuse multiple trained models by combining their parameters using a specified aggregation method.

   :param model_paths: List of paths to model checkpoints to be fused.
   :type model_paths: list of str
   :param save_path: Path where the fused model will be saved.
   :type save_path: str
   :param device: Device to use ('cpu' or 'cuda').
   :type device: str
   :param model_name: Model architecture to use when initializing.
   :type model_name: str
   :param pretrained: Whether to initialize with pretrained weights.
   :type pretrained: bool
   :param dropout_rate: Dropout rate to apply to the model.
   :type dropout_rate: float or None
   :param use_checkpoint: Whether to use gradient checkpointing.
   :type use_checkpoint: bool
   :param aggregator: Aggregation strategy to combine weights. One of {'mean', 'geomean', 'median', 'sum', 'max', 'min'}.
   :type aggregator: str

   :returns: The fused model with combined weights.
   :rtype: TorchModel


.. py:function:: annotate_filter_vision(settings)

   Annotate and filter a CSV file with experimental metadata and optionally remove training samples.

   :param settings: Configuration dictionary with keys:
                    - 'src' (str or list): Paths to CSV annotation files.
                    - 'cells' (dict): Mapping of cell types to annotation labels.
                    - 'cell_loc' (str): Column name for cell type annotations.
                    - 'pathogens' (dict): Mapping of pathogens to annotation labels.
                    - 'pathogen_loc' (str): Column name for pathogen annotations.
                    - 'treatments' (dict): Mapping of treatments to annotation labels.
                    - 'treatment_loc' (str): Column name for treatment annotations.
                    - 'filter_column' (str or None): Column to filter on.
                    - 'upper_threshold' (float): Upper bound for filtering.
                    - 'lower_threshold' (float): Lower bound for filtering.
                    - 'remove_train' (bool): If True, removes rows present in training folders.
   :type settings: dict

   :returns: None. Saves filtered and annotated CSVs to disk.


