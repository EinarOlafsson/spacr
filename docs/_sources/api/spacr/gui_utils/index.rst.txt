spacr.gui_utils
===============

.. py:module:: spacr.gui_utils








Module Contents
---------------

.. py:function:: initialize_cuda()

   Initializes CUDA in the main process by performing a simple GPU operation.


.. py:function:: set_high_priority(process)

.. py:function:: set_cpu_affinity(process)

.. py:function:: proceed_with_app(root, app_name, app_func)

.. py:function:: load_app(root, app_name, app_func)

.. py:function:: parse_list(value)

   Parses a string representation of a list and returns the parsed list.

   Args:
       value (str): The string representation of the list.

   Returns:
       list: The parsed list, which can contain integers, floats, or strings.

   Raises:
       ValueError: If the input value is not a valid list format or contains mixed types or unsupported types.


.. py:function:: create_input_field(frame, label_text, row, var_type='entry', options=None, default_value=None)

   Create an input field in the specified frame.

   Args:
       frame (tk.Frame): The frame in which the input field will be created.
       label_text (str): The text to be displayed as the label for the input field.
       row (int): The row in which the input field will be placed.
       var_type (str, optional): The type of input field to create. Defaults to 'entry'.
       options (list, optional): The list of options for a combo box input field. Defaults to None.
       default_value (str, optional): The default value for the input field. Defaults to None.

   Returns:
       tuple: A tuple containing the label, input widget, variable, and custom frame.

   Raises:
       Exception: If an error occurs while creating the input field.



.. py:function:: process_stdout_stderr(q)

   Redirect stdout and stderr to the queue q.


.. py:class:: WriteToQueue(q)

   Bases: :py:obj:`io.TextIOBase`


   A custom file-like class that writes any output to a given queue.
   This can be used to redirect stdout and stderr.


   .. py:attribute:: q


   .. py:method:: write(msg)

      Write string to stream.
      Returns the number of characters written (which is always equal to
      the length of the string).



   .. py:method:: flush()

      Flush write buffers, if applicable.

      This is not implemented for read-only and non-blocking streams.



.. py:function:: cancel_after_tasks(frame)

.. py:function:: annotate(settings)

.. py:function:: generate_annotate_fields(frame)

.. py:function:: run_annotate_app(vars_dict, parent_frame)

.. py:data:: global_image_refs
   :value: []


.. py:function:: annotate_app(parent_frame, settings)

.. py:function:: load_next_app(root)

.. py:function:: annotate_with_image_refs(settings, root, shutdown_callback)

.. py:function:: convert_settings_dict_for_gui(settings)

.. py:function:: spacrFigShow(fig_queue=None)

   Replacement for plt.show() that queues figures instead of displaying them.


.. py:function:: function_gui_wrapper(function=None, settings={}, q=None, fig_queue=None, imports=1)

   Wraps the run_multiple_simulations function to integrate with GUI processes.

   Parameters:
   - settings: dict, The settings for the run_multiple_simulations function.
   - q: multiprocessing.Queue, Queue for logging messages to the GUI.
   - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.


.. py:function:: run_function_gui(settings_type, settings, q, fig_queue, stop_requested)

.. py:function:: hide_all_settings(vars_dict, categories)

   Function to initially hide all settings in the GUI.

   Parameters:
   - categories: dict, The categories of settings with their corresponding settings.
   - vars_dict: dict, The dictionary containing the settings and their corresponding widgets.


.. py:function:: setup_frame(parent_frame)

.. py:function:: download_hug_dataset(q, vars_dict)

.. py:function:: download_dataset(q, repo_id, subfolder, local_dir=None, retries=5, delay=5)

   Downloads a dataset or settings files from Hugging Face and returns the local path.

   Args:
       repo_id (str): The repository ID (e.g., 'einarolafsson/toxo_mito' or 'einarolafsson/spacr_settings').
       subfolder (str): The subfolder path within the repository (e.g., 'plate1' or the settings subfolder).
       local_dir (str): The local directory where the files will be saved. Defaults to the user's home directory.
       retries (int): Number of retry attempts in case of failure.
       delay (int): Delay in seconds between retries.

   Returns:
       str: The local path to the downloaded files.


.. py:function:: ensure_after_tasks(frame)

.. py:function:: display_gif_in_plot_frame(gif_path, parent_frame)

   Display and zoom a GIF to fill the entire parent_frame, maintaining aspect ratio, with lazy resizing and caching.


.. py:function:: display_media_in_plot_frame(media_path, parent_frame)

   Display an MP4, AVI, or GIF and play it on repeat in the parent_frame, fully filling the frame while maintaining aspect ratio.


.. py:function:: print_widget_structure(widget, indent=0)

   Recursively print the widget structure.


.. py:function:: get_screen_dimensions()

.. py:function:: convert_to_number(value)

   Converts a string value to an integer if possible, otherwise converts to a float.

   Args:
       value (str): The string representation of the number.

   Returns:
       int or float: The converted number.


