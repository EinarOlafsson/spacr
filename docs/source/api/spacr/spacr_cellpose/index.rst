spacr.spacr_cellpose
====================

.. py:module:: spacr.spacr_cellpose




Module Contents
---------------

.. py:function:: parse_cellpose4_output(output)

   General parser for Cellpose eval output.
   Handles:
   - batched format (list of 4 arrays)
   - per-image list of flows
   :returns: masks, flows0, flows1, flows2, flows3


.. py:function:: identify_masks_finetune(settings)

   Generate Cellpose segmentation masks for a batch of images using a pretrained or custom model.

   This function loads a set of images from the provided source directory, applies optional
   preprocessing (normalization, resizing), and uses a Cellpose model to generate segmentation masks.
   Masks are optionally visualized and saved to disk. The model, channels, and other parameters are
   defined in the `settings` dictionary.

   :param settings: Dictionary containing configuration parameters. Must include:
                    - 'src' (str): Source folder with `.tif` images.
                    - 'model_name' (str): Name of Cellpose model to use (e.g., 'cyto2', 'nucleus').
                    - 'custom_model' (str or None): Path to custom model file (.pt), if used.
                    - 'channels' (list): List of image channel indices to use for segmentation.
                    - 'grayscale' (bool): Whether input images are single-channel.
                    - 'diameter' (float): Estimated diameter of objects in pixels.
                    - 'flow_threshold' (float): Threshold for mask acceptance based on flow prediction.
                    - 'CP_prob' (float): Cell probability threshold for segmentation.
                    - 'rescale' (float): Rescaling factor.
                    - 'resample' (bool): Whether to resample the image during preprocessing.
                    - 'normalize' (bool): Whether to normalize pixel intensities.
                    - 'percentiles' (list): Lower and upper percentiles for normalization.
                    - 'invert' (bool): Whether to invert image intensities.
                    - 'remove_background' (bool): Whether to subtract background value.
                    - 'background' (list): Background pixel intensity values to subtract per channel.
                    - 'Signal_to_noise' (float): Threshold for signal-to-noise filtering.
                    - 'resize' (bool): Whether to resize to fixed target dimensions.
                    - 'target_height' (int): Height for resizing.
                    - 'target_width' (int): Width for resizing.
                    - 'batch_size' (int): Number of images to process per batch.
                    - 'fill_in' (bool): Whether to fill holes in masks.
                    - 'save' (bool): Whether to save the masks to disk.
                    - 'verbose' (bool): Whether to print detailed progress and visualization output.
   :type settings: dict

   :returns: None. Masks are optionally saved to the 'masks' subdirectory in the source folder.


.. py:function:: generate_masks_from_imgs(src, model, model_name, batch_size, diameter, cellprob_threshold, flow_threshold, grayscale, save, normalize, channels, percentiles, invert, plot, resize, target_height, target_width, remove_background, background, Signal_to_noise, verbose)

   Apply a Cellpose model to a batch of images and generate segmentation masks.

   :param src: Directory containing input .tif images.
   :type src: str
   :param model: Initialized Cellpose model.
   :type model: CellposeModel
   :param model_name: Model identifier (e.g., 'cyto2', 'nucleus').
   :type model_name: str
   :param batch_size: Number of images processed in each batch.
   :type batch_size: int
   :param diameter: Estimated object diameter in pixels.
   :type diameter: float
   :param cellprob_threshold: Cell probability threshold.
   :type cellprob_threshold: float
   :param flow_threshold: Flow threshold for mask acceptance.
   :type flow_threshold: float
   :param grayscale: If True, treat images as single-channel.
   :type grayscale: bool
   :param save: Whether to save output masks.
   :type save: bool
   :param normalize: Whether to normalize input images.
   :type normalize: bool
   :param channels: Channels to use for processing (e.g., [0, 1]).
   :type channels: list
   :param percentiles: Percentiles for normalization (e.g., [2, 99]).
   :type percentiles: list
   :param invert: If True, invert image intensity.
   :type invert: bool
   :param plot: If True, display masks and flows.
   :type plot: bool
   :param resize: Whether to resize images to fixed target dimensions.
   :type resize: bool
   :param target_height: Height after resizing.
   :type target_height: int
   :param target_width: Width after resizing.
   :type target_width: int
   :param remove_background: Whether to subtract background intensity.
   :type remove_background: bool
   :param background: Background intensity values for subtraction.
   :type background: list
   :param Signal_to_noise: Minimum SNR threshold.
   :type Signal_to_noise: float
   :param verbose: If True, print detailed status messages.
   :type verbose: bool

   :returns: None. Saves masks to disk if `save=True`.


.. py:function:: check_cellpose_models(settings)

   Evaluate multiple pretrained Cellpose models ('cyto', 'nuclei', 'cyto2', 'cyto3')
   on a given dataset using standardized settings.

   :param settings: Dictionary of parameters controlling input source, model parameters,
                    image preprocessing, and save/visualization options.
   :type settings: dict

   :returns: None. Runs `generate_masks_from_imgs()` for each model and displays results.


.. py:function:: save_results_and_figure(src, fig, results)

   Save a results DataFrame and associated figure to disk.

   :param src: Path to the source directory where the 'results' subfolder will be created.
   :type src: str
   :param fig: The figure object to be saved as a PDF.
   :type fig: matplotlib.figure.Figure
   :param results: Results to be saved. If not a DataFrame,
                   it will be converted to one.
   :type results: pd.DataFrame or dict or list

   :returns: None. Writes results to 'results.csv' and the figure to 'model_comparison_plot.pdf'.


.. py:function:: compare_mask(args)

   Compare segmentation masks across different directories for a given filename
   using multiple evaluation metrics.

   :param args: A tuple containing:
                - src (str): Not used directly, reserved for future use.
                - filename (str): Name of the mask file to compare across directories.
                - dirs (list of str): List of directory paths where mask files are located.
                - conditions (list of str): Labels corresponding to each directory for result naming.
   :type args: tuple

   :returns:

             A dictionary containing comparison metrics (Jaccard index, boundary F1 score,
                           and average precision) for all pairwise combinations of masks.
                           Returns None if any mask file is missing.
   :rtype: dict or None


.. py:function:: compare_cellpose_masks(src, verbose=False, processes=None, save=True)

   Compare Cellpose segmentation masks across multiple model output folders.

   :param src: Path to the parent directory containing subdirectories for each model condition.
   :type src: str
   :param verbose: If True, visualize each mask comparison using matplotlib.
   :type verbose: bool
   :param processes: Number of parallel processes to use. If None, uses os.cpu_count().
   :type processes: int or None
   :param save: Whether to save the visualization outputs and results to disk.
   :type save: bool

   :returns: None. Results are printed, plotted, and optionally saved to disk.


