spacr.gui_utils
===============

.. py:module:: spacr.gui_utils






Module Contents
---------------

.. py:function:: initialize_cuda()

   Initializes CUDA for the main process if a compatible GPU is available.

   This function checks if CUDA is available on the system. If it is, it allocates
   a small tensor on the GPU to ensure that CUDA is properly initialized. A message
   is printed to indicate whether CUDA was successfully initialized or if it is not
   available.

   .. note::

      This function is intended to be used in environments where CUDA-enabled GPUs
      are present and PyTorch is installed.

   Prints:
       - "CUDA initialized in the main process." if CUDA is available and initialized.
       - "CUDA is not available." if no compatible GPU is detected.


.. py:function:: set_high_priority(process)

   Sets the priority of a given process to high.

   On Windows systems, the process priority is set to HIGH_PRIORITY_CLASS.
   On Unix-like systems, the process priority is adjusted to a higher level
   by setting its niceness value to -10.

   :param process: The process whose priority is to be adjusted.
   :type process: psutil.Process

   :raises psutil.AccessDenied: If the current user does not have permission to change
       the priority of the process.
   :raises psutil.NoSuchProcess: If the specified process does not exist.
   :raises Exception: For any other errors encountered during the operation.

   .. rubric:: Notes

   - This function requires the `psutil` library to interact with system processes.
   - Adjusting process priority may require elevated privileges depending on the
     operating system and user permissions.


.. py:function:: set_cpu_affinity(process)

   Set the CPU affinity for a given process to use all available CPUs.

   This function modifies the CPU affinity of the specified process, allowing
   it to run on all CPUs available on the system.

   :param process: A psutil.Process object representing the process
                   whose CPU affinity is to be set.
   :type process: psutil.Process

   :raises psutil.NoSuchProcess: If the process does not exist.
   :raises psutil.AccessDenied: If the process cannot be accessed due to insufficient permissions.
   :raises psutil.ZombieProcess: If the process is a zombie process.


.. py:function:: proceed_with_app(root, app_name, app_func)

   Prepares the application window to load a new app by clearing the current
   content frame and initializing the specified app.

   :param root: The root window or parent container that
                contains the content frame.
   :type root: tk.Tk or tk.Toplevel
   :param app_name: The name of the application to be loaded (not used in
                    the current implementation but could be useful for logging or
                    debugging purposes).
   :type app_name: str
   :param app_func: A function that initializes the new application
                    within the content frame. It should accept the content frame as
                    its only argument.
   :type app_func: callable

   Behavior:
       - Destroys all widgets in the `content_frame` attribute of `root`
         (if it exists).
       - Calls `app_func` with `root.content_frame` to initialize the new
         application.

   .. note::

      Ensure that `root` has an attribute `content_frame` that is a valid
      tkinter container (e.g., a `tk.Frame`) before calling this function.


.. py:function:: load_app(root, app_name, app_func)

   Load a new application into the GUI framework.

   This function handles the transition between applications in the GUI by
   clearing the current canvas, canceling scheduled tasks, and invoking
   exit functionality for specific applications if necessary.

   :param root: The root object of the GUI, which contains the canvas,
                after_tasks, and other application state.
   :param app_name: The name of the application to load.
   :type app_name: str
   :param app_func: The function to initialize the new application.
   :type app_func: callable

   Behavior:
       - Clears the current canvas if it exists.
       - Cancels all scheduled `after` tasks associated with the root object.
       - If the current application has an exit function and the new app is
         not "Annotate" or "make_masks", the exit function is invoked before
         proceeding to the new application.
       - Proceeds to load the new application using the provided `app_func`.

   .. note::

      The `proceed_with_app` function is used internally to finalize the
      transition to the new application.


.. py:function:: parse_list(value)

   Parses a string representation of a list and returns the parsed list.

   :param value: The string representation of the list.
   :type value: str

   :returns: The parsed list, which can contain integers, floats, or strings.
   :rtype: list

   :raises ValueError: If the input value is not a valid list format or contains mixed types or unsupported types.


.. py:function:: create_input_field(frame, label_text, row, var_type='entry', options=None, default_value=None)

   Create an input field in the specified frame.

   :param frame: The frame in which the input field will be created.
   :type frame: tk.Frame
   :param label_text: The text to be displayed as the label for the input field.
   :type label_text: str
   :param row: The row in which the input field will be placed.
   :type row: int
   :param var_type: The type of input field to create. Defaults to 'entry'.
   :type var_type: str, optional
   :param options: The list of options for a combo box input field. Defaults to None.
   :type options: list, optional
   :param default_value: The default value for the input field. Defaults to None.
   :type default_value: str, optional

   :returns: A tuple containing the label, input widget, variable, and custom frame.
   :rtype: tuple

   :raises Exception: If an error occurs while creating the input field.


.. py:function:: process_stdout_stderr(q)

   Redirects the standard output (stdout) and standard error (stderr) streams
   to a queue for processing.

   This function replaces the default `sys.stdout` and `sys.stderr` with
   instances of `WriteToQueue`, which write all output to the provided queue.

   :param q: A queue object where the redirected output will be stored.
   :type q: queue.Queue


.. py:class:: WriteToQueue(q)

   Bases: :py:obj:`io.TextIOBase`


   A file-like object that redirects writes to a queue.

   :param q: The queue to write output to.
   :type q: queue.Queue

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: q


   .. py:method:: write(msg)

      Write string to stream.

      :param msg: The string message to write.
      :type msg: str
      :returns: Number of characters written.
      :rtype: int



   .. py:method:: flush()

      Flush write buffers, if applicable.

      This is a no-op in this implementation.



.. py:function:: cancel_after_tasks(frame)

   Cancels all scheduled 'after' tasks associated with a given frame.

   This function checks if the provided frame object has an attribute
   named 'after_tasks', which is expected to be a list of task IDs
   scheduled using the `after` method (e.g., in a Tkinter application).
   If such tasks exist, it cancels each of them using the `after_cancel`
   method and then clears the list.

   :param frame: An object (typically a Tkinter widget) that may have an
                 'after_tasks' attribute containing scheduled task IDs.

   :raises AttributeError: If the frame does not have the required methods
       (`after_cancel` or `after_tasks` attribute).


.. py:function:: load_next_app(root)

   Loads the next application by invoking the function stored in the `next_app_func`
   attribute of the provided `root` object. If the current root window has been
   destroyed, a new root window is initialized before invoking the next application.

   :param root: The current root window object, which contains the attributes
                `next_app_func` (a callable for the next application) and
                `next_app_args` (a tuple of arguments to pass to the callable).
   :type root: tk.Tk

   :raises tk.TclError: If the root window does not exist and needs to be reinitialized.


.. py:function:: convert_settings_dict_for_gui(settings)

   Convert a dictionary of settings into a format suitable for GUI rendering.

   Each key in the input dictionary is mapped to a tuple of the form:
   (input_type, options, default_value), where:

   - input_type (str): The type of GUI element. One of:
       * 'combo' for dropdown menus
       * 'check' for checkboxes
       * 'entry' for entry fields
   - options (list or None): A list of selectable options for 'combo' types, or None for other types.
   - default_value: The current or default value to be displayed in the GUI.

   Special keys are mapped to pre-defined configurations with known option sets
   (e.g., 'metadata_type', 'channels', 'model_type').

   :param settings: Dictionary where keys are setting names and values are their current values.
   :type settings: dict

   :return: Dictionary mapping setting names to tuples for GUI rendering.
   :rtype: dict


.. py:function:: spacrFigShow(fig_queue=None)

   Displays the current matplotlib figure or adds it to a queue.

   This function retrieves the current matplotlib figure using `plt.gcf()`.
   If a `fig_queue` is provided, the figure is added to the queue.
   Otherwise, the figure is displayed using the `show()` method.
   After the figure is either queued or displayed, it is closed using `plt.close()`.

   :param fig_queue: A queue to store the figure.
                     If None, the figure is displayed instead.
   :type fig_queue: queue.Queue, optional

   :returns: None


.. py:function:: function_gui_wrapper(function=None, settings={}, q=None, fig_queue=None, imports=1)

   Wraps the run_multiple_simulations function to integrate with GUI processes.

   Args:
   - settings: dict, The settings for the run_multiple_simulations function.
   - q: multiprocessing.Queue, Queue for logging messages to the GUI.
   - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.


.. py:function:: run_function_gui(settings_type, settings, q, fig_queue, stop_requested)

   Executes a specified processing function in the GUI context based on `settings_type`.

   This function selects and runs one of the core `spaCR` processing functions
   (e.g., segmentation, measurement, classification, barcode mapping) based on the
   provided `settings_type` string. It wraps the execution with a logging mechanism
   to redirect stdout/stderr to the GUI console and handles exceptions cleanly.

   :param settings_type: A string indicating which processing function to execute. Supported values include:
                         'mask', 'measure', 'classify', 'train_cellpose', 'ml_analyze', 'cellpose_masks',
                         'cellpose_all', 'map_barcodes', 'regression', 'recruitment', 'analyze_plaques', 'convert'.
   :type settings_type: str
   :param settings: A dictionary of parameters required by the selected function.
   :type settings: dict
   :param q: Queue for redirecting standard output and errors to the GUI console.
   :type q: multiprocessing.Queue
   :param fig_queue: Queue used to transfer figures (e.g., plots) from the worker process to the GUI.
   :type fig_queue: multiprocessing.Queue
   :param stop_requested: A shared value to signal whether execution has completed or was interrupted.
   :type stop_requested: multiprocessing.Value

   :raises ValueError: If an invalid `settings_type` is provided.

   .. rubric:: Notes

   - Redirects stdout/stderr to the GUI using `process_stdout_stderr`.
   - Catches and reports any exceptions to the GUI queue.
   - Sets `stop_requested.value = 1` when the task finishes (whether successful or not).


.. py:function:: hide_all_settings(vars_dict, categories)

   Function to initially hide all settings in the GUI.

   Args:
   - categories: dict, The categories of settings with their corresponding settings.
   - vars_dict: dict, The dictionary containing the settings and their corresponding widgets.


.. py:function:: setup_frame(parent_frame)

   Set up the main GUI layout within the given parent frame.

   This function initializes a dark-themed, resizable GUI layout using `PanedWindow`
   containers. It organizes the layout into left-hand settings, central vertical content,
   and bottom horizontal panels. It also sets initial sash positions and layout weights.

   :param parent_frame: The parent Tkinter frame to populate with the GUI layout.
   :type parent_frame: tk.Frame

   :returns: A tuple containing:
             - parent_frame (tk.Frame): The modified parent frame with the layout initialized.
             - vertical_container (tk.PanedWindow): Top container in the right-hand area for main content.
             - horizontal_container (tk.PanedWindow): Bottom container for additional widgets.
             - settings_container (tk.PanedWindow): Left-hand container for GUI settings.
   :rtype: tuple

   .. rubric:: Notes

   - Uses `set_dark_style` and `set_element_size` from `gui_elements` to theme and size widgets.
   - Dynamically positions the sash between the left and right panes to 25% of the screen width.


.. py:function:: download_hug_dataset(q, vars_dict)

   Downloads a dataset and settings files from the Hugging Face Hub and updates the provided variables dictionary.

   :param q: A queue object used for logging messages during the download process.
   :type q: queue.Queue
   :param vars_dict: A dictionary containing variables to be updated. If 'src' is present in the dictionary,
                     the third element of 'src' will be updated with the downloaded dataset path.
   :type vars_dict: dict

   The function performs the following steps:
       1. Downloads a dataset from the Hugging Face Hub using the specified repository ID and subfolder.
       2. Updates the 'src' variable in `vars_dict` with the local path of the downloaded dataset, if applicable.
       3. Logs the dataset download status to the provided queue.
       4. Downloads settings files from another repository on the Hugging Face Hub.
       5. Logs the settings download status to the provided queue.

   .. rubric:: Notes

   - The dataset is downloaded to a local directory under the user's home directory named "datasets".
   - Any exceptions during the download process are caught and logged to the queue.

   :raises None: All exceptions are handled internally and logged to the queue.


.. py:function:: download_dataset(q, repo_id, subfolder, local_dir=None, retries=5, delay=5)

   Downloads a dataset or settings files from Hugging Face and returns the local path.

   :param repo_id: The repository ID (e.g., 'einarolafsson/toxo_mito' or 'einarolafsson/spacr_settings').
   :type repo_id: str
   :param subfolder: The subfolder path within the repository (e.g., 'plate1' or the settings subfolder).
   :type subfolder: str
   :param local_dir: The local directory where the files will be saved. Defaults to the user's home directory.
   :type local_dir: str
   :param retries: Number of retry attempts in case of failure.
   :type retries: int
   :param delay: Delay in seconds between retries.
   :type delay: int

   :returns: The local path to the downloaded files.
   :rtype: str


.. py:function:: ensure_after_tasks(frame)

.. py:function:: display_gif_in_plot_frame(gif_path, parent_frame)

   Display and zoom a GIF to fill the entire parent_frame, maintaining aspect ratio, with lazy resizing and caching.


.. py:function:: display_media_in_plot_frame(media_path, parent_frame)

   Display a media file (MP4, AVI, or GIF) in a Tkinter frame, playing it on repeat
   while fully filling the frame and maintaining the aspect ratio.

   :param media_path: The file path to the media file (MP4, AVI, or GIF).
   :type media_path: str
   :param parent_frame: The Tkinter frame where the media will be displayed.
   :type parent_frame: tk.Frame

   Behavior:
       - For MP4 and AVI files:
           - Uses OpenCV to read and play the video.
           - Resizes and crops the video to fully fill the parent frame while maintaining aspect ratio.
           - Plays the video on repeat.
       - For GIF files:
           - Uses PIL to read and play the GIF.
           - Resizes and crops the GIF to fully fill the parent frame while maintaining aspect ratio.
           - Plays the GIF on repeat.

   :raises ValueError: If the file format is not supported (only MP4, AVI, and GIF are supported).

   .. rubric:: Notes

   - The function clears any existing widgets in the parent frame before displaying the media.
   - The parent frame is configured to expand and adapt to the media's aspect ratio.


.. py:function:: print_widget_structure(widget, indent=0)

.. py:function:: get_screen_dimensions()

.. py:function:: convert_to_number(value)

   Converts a string value to an integer if possible, otherwise converts to a float.

   :param value: The string representation of the number.
   :type value: str

   :returns: The converted number.
   :rtype: int or float


