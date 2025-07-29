spacr.gui_core
==============

.. py:module:: spacr.gui_core






Module Contents
---------------

.. py:data:: q
   :value: None


.. py:data:: console_output
   :value: None


.. py:data:: parent_frame
   :value: None


.. py:data:: vars_dict
   :value: None


.. py:data:: canvas
   :value: None


.. py:data:: canvas_widget
   :value: None


.. py:data:: scrollable_frame
   :value: None


.. py:data:: progress_label
   :value: None


.. py:data:: fig_queue
   :value: None


.. py:data:: figures
   :value: None


.. py:data:: figure_index
   :value: None


.. py:data:: progress_bar
   :value: None


.. py:data:: usage_bars
   :value: None


.. py:data:: index_control
   :value: None


.. py:data:: thread_control

.. py:function:: toggle_settings(button_scrollable_frame)

   Initializes and displays a dropdown menu to toggle visibility of categorized GUI settings.

   :param button_scrollable_frame: A scrollable frame where the dropdown menu will be placed.
   :type button_scrollable_frame: tk.Widget

   :raises ValueError: If `vars_dict` is not initialized.

   Behavior:
       - Imports setting categories and hides all setting widgets initially.
       - Adds a dropdown menu that lets users toggle the visibility of settings belonging to each category.
       - When a category is selected, it toggles visibility for all settings in that category.
       - Updates the dropdown appearance based on which categories are currently active.

   Globals:
       vars_dict (dict): A dictionary mapping setting keys to (label, widget, _, frame) tuples.
           Must be initialized before calling this function.


.. py:function:: display_figure(fig)

   Displays a matplotlib figure within a Tkinter GUI canvas, enabling interactive features such as:
   - Zooming with mouse scroll
   - Navigation via left/right click
   - Context menu for saving or modifying the figure
   - Dynamic annotation visibility based on zoom level

   :param fig: The figure to display in the GUI.
   :type fig: matplotlib.figure.Figure

   Global Variables:
       canvas (FigureCanvasTkAgg): The current canvas used for displaying the figure.
       canvas_widget (tk.Widget): The widget associated with the canvas.

   Behavior:
       - Destroys any existing canvas before rendering the new figure.
       - Initializes zooming around cursor position using scroll wheel.
       - Binds left/right clicks to navigate between figures (via show_previous_figure / show_next_figure).
       - Adds a right-click context menu to save as PNG/PDF, modify figure, or reset zoom.
       - Preserves original axis limits for reset functionality.
       - Dynamically toggles text label visibility depending on zoom.
       - Applies consistent dark theme styling for background and context menu.

   .. rubric:: Notes

   - Assumes `show_previous_figure()`, `show_next_figure()`, `set_dark_style()`,
     `save_figure_as_format()`, `modify_figure()`, and `process_fig_queue()` are defined elsewhere.
   - Should be called after global `canvas` and `canvas_widget` have been initialized.

   :raises RuntimeError: If canvas_widget is not properly initialized prior to calling this function.


.. py:function:: clear_unused_figures()

   Clears figures from memory that are not within ±20 of the current figure index to reduce memory usage.

   Globals:
       figures (collections.deque): A deque of currently stored matplotlib figures.
       figure_index (int): Index of the currently displayed figure.

   Behavior:
       - Retains only figures within 20 indices before and after the current figure_index.
       - Updates the figure_index to remain within valid bounds of the updated figures deque.


.. py:function:: show_previous_figure()

   Displays the previous figure in the global figures list, if available.

   Globals:
       figure_index (int): Current index of the displayed figure.
       figures (list): List of matplotlib figures.
       fig_queue (queue.Queue): Queue of figures to be displayed later (unused here).
       index_control (tk.IntVar or custom object): UI controller for setting and tracking current index.

   Behavior:
       - Decrements figure_index.
       - Standardizes and displays the previous figure.
       - Updates index_control to reflect the new figure index.


.. py:function:: show_next_figure()

   Displays the next figure in the figures list or loads one from the queue if at the end.

   Globals:
       figure_index (int): Current index of the displayed figure.
       figures (list): List of existing matplotlib figures.
       fig_queue (queue.Queue): Queue of figures to display when user navigates forward.
       index_control (tk.IntVar or custom object): UI control to sync figure index display.

   Behavior:
       - If not at the end of figures list, increments index and displays the next figure.
       - If at the end and the figure queue is not empty, loads and displays the next figure from the queue.
       - Updates index_control to reflect the current position.


.. py:function:: process_fig_queue()

   Processes the figure queue and updates the GUI with new figures as they arrive.

   Globals:
       canvas (FigureCanvasTkAgg): The current canvas displaying the figure.
       fig_queue (queue.Queue): Queue containing matplotlib figures to be displayed.
       canvas_widget (tk.Widget): The widget holding the canvas.
       parent_frame (tk.Frame): Parent frame that manages periodic GUI updates.
       uppdate_frequency (int): Delay in milliseconds between re-checking the queue.
       figures (collections.deque): A deque holding all currently active figures.
       figure_index (int): Index of the currently displayed figure.
       index_control (tk.IntVar or similar): Control object for syncing figure index in GUI.

   Behavior:
       - Retrieves and displays figures from fig_queue.
       - Standardizes figure appearance using `standardize_figure`.
       - Maintains a fixed-size deque (max 100) by discarding oldest figures.
       - If no figure is currently displayed, it displays the first one.
       - Sets slider/index control to match the latest figure.
       - Reschedules itself using `after()` to poll for new figures continuously.


.. py:function:: update_figure(value)

   Updates the currently displayed figure based on slider value.

   :param value: Index of the figure to display.
   :type value: str or int

   Globals:
       figure_index (int): Index of the current figure.
       figures (deque): Deque of matplotlib figures.
       index_control (spacrSlider): Slider control for index selection.

   Effects:
       - Updates the canvas with the selected figure.
       - Applies standard formatting to the figure.
       - Sets the slider and index state accordingly.


.. py:function:: setup_plot_section(vertical_container, settings_type)

   Initializes the figure display section and associated slider for navigating figures.

   :param vertical_container: Parent container.
   :type vertical_container: tk.PanedWindow or tk.Frame
   :param settings_type: Mode to configure specific behaviors (e.g. 'map_barcodes').
   :type settings_type: str

   :returns: (canvas, canvas_widget)
   :rtype: tuple

   Globals:
       canvas, canvas_widget: For figure rendering.
       figures (deque): Storage of figures.
       figure_index (int): Current index.
       index_control (spacrSlider): Slider widget for navigating figure list.

   Behavior:
       - Displays initial blank figure.
       - Adds slider for figure navigation.
       - Applies dark style.
       - Optionally shows media for 'map_barcodes'.


.. py:function:: set_globals(thread_control_var, q_var, console_output_var, parent_frame_var, vars_dict_var, canvas_var, canvas_widget_var, scrollable_frame_var, fig_queue_var, progress_bar_var, usage_bars_var)

   Assigns external references to global variables for use throughout the GUI.

   :param \*_var: Various widget and control object references.

   Globals Set:
       thread_control, q, console_output, parent_frame, vars_dict,
       canvas, canvas_widget, scrollable_frame, fig_queue,
       progress_bar, usage_bars


.. py:function:: import_settings(settings_type='mask')

   Imports a settings CSV and applies it to the GUI panel for a specific mode.

   :param settings_type: Type of settings to load ('mask', 'measure', 'classify', etc.)
   :type settings_type: str

   Globals:
       vars_dict: Dictionary of GUI widget references.
       scrollable_frame: Frame that holds the settings widgets.

   Behavior:
       - Reads key-value settings from CSV.
       - Applies them to the current settings panel.
       - Updates `vars_dict` accordingly.


.. py:function:: setup_settings_panel(vertical_container, settings_type='mask')

   Creates the scrollable settings panel for a given analysis type.

   :param vertical_container: Parent container.
   :type vertical_container: tk.PanedWindow or tk.Frame
   :param settings_type: One of the predefined modes such as 'mask', 'classify', etc.
   :type settings_type: str

   :returns: (scrollable_frame, vars_dict)
   :rtype: tuple

   Globals:
       vars_dict: Dict mapping setting names to widget metadata.
       scrollable_frame: Frame containing all input widgets.

   Behavior:
       - Initializes and populates settings UI.
       - Loads defaults depending on `settings_type`.
       - Applies theme and layout configuration.


.. py:function:: setup_console(vertical_container)

   Sets up a scrollable console output section with hover effect and dark theme styling.

   :param vertical_container: Parent container to attach the console.
   :type vertical_container: tk.PanedWindow or tk.Frame

   :returns:     console_output (tk.Text): The text widget for console output.
                 console_frame (tk.Frame): The frame containing the console.
   :rtype: tuple

   Globals Set:
       console_output: Reference to the console's text widget.


.. py:function:: setup_button_section(horizontal_container, settings_type='mask', run=True, abort=True, download=True, import_btn=True)

   Creates the button section with control buttons (run, abort, download, import) and a progress bar.

   :param horizontal_container: Container to add the button section.
   :type horizontal_container: tk.PanedWindow or tk.Frame
   :param settings_type: Workflow type to adjust button behavior.
   :type settings_type: str
   :param run: Whether to include a "Run" button.
   :type run: bool
   :param abort: Whether to include an "Abort" button.
   :type abort: bool
   :param download: Whether to include a "Download" button.
   :type download: bool
   :param import_btn: Whether to include a "Settings Import" button.
   :type import_btn: bool

   :returns:     button_scrollable_frame (spacrFrame): The frame holding the buttons.
                 btn_col (int): Final column index after placing buttons.
   :rtype: tuple

   Globals Set:
       run_button, abort_button, download_dataset_button, import_button, progress_bar
       button_frame, button_scrollable_frame, thread_control, q, fig_queue, vars_dict


.. py:function:: setup_usage_panel(horizontal_container, btn_col, uppdate_frequency)

   Creates the system usage panel displaying RAM, VRAM, GPU, and CPU core usage as progress bars.

   :param horizontal_container: Container for the usage panel.
   :type horizontal_container: tk.PanedWindow or tk.Frame
   :param btn_col: Starting column index from button section (not used directly here).
   :type btn_col: int
   :param uppdate_frequency: Update interval for usage stats in milliseconds.
   :type uppdate_frequency: int

   :returns:     usage_scrollable_frame (spacrFrame): Frame containing the usage bars.
                 usage_bars (list): List of spacrProgressBar widgets for RAM, VRAM, GPU, and CPU cores.
                 usg_col (int): Column index used for placement in the parent container.
   :rtype: tuple

   Globals Set:
       usage_bars: List of progress bars used to update usage statistics.


.. py:function:: initiate_abort()

   Terminates the currently running threaded process if any, and resets the control flags.

   Globals:
       thread_control (dict): Holds the active thread and abort flag.
       q (queue.Queue): Message queue for status updates.
       parent_frame: Not used directly, but kept for potential extension.


.. py:function:: check_src_folders_files(settings, settings_type, q)

   Verifies that source folder paths exist and contain the necessary subfolders and/or image files
   depending on the selected processing `settings_type`.

   :param settings: A dictionary containing at least the key `"src"` with either a string path or a list of paths.
   :type settings: dict
   :param settings_type: Type of operation. Supported values include:
                         - 'mask': Requires image-containing subfolders or raw image files.
                         - 'measure': Requires 'merged' subfolder containing `.npy` files.
                         (Other types like 'regression', 'umap', 'classify' are commented out.)
   :type settings_type: str
   :param q: A thread-safe queue to which error messages are pushed.
   :type q: queue.Queue

   :returns: `True` if validation fails (request should stop), `False` if all required paths and files exist.
   :rtype: bool


.. py:function:: start_process(q=None, fig_queue=None, settings_type='mask')

   Starts a multiprocessing task based on the provided settings type and GUI settings.

   :param q: Queue for logging or progress updates.
   :type q: multiprocessing.Queue, optional
   :param fig_queue: Queue for passing figures or plot data.
   :type fig_queue: multiprocessing.Queue, optional
   :param settings_type: Type of processing task to run (e.g., 'mask', 'measure').
   :type settings_type: str

   Globals Used:
       thread_control (dict): Controls the running thread and abort flags.
       vars_dict (dict): Holds current GUI settings.
       parent_frame (tk.Frame): Main parent GUI frame.


.. py:function:: process_console_queue()

   Periodically reads from the message queue and updates the console widget with formatted messages.
   Supports special formatting for errors, warnings, and progress updates.

   Globals Used:
       q (Queue): Queue for status and log messages.
       console_output (tk.Text): The console widget.
       parent_frame (tk.Frame): The main GUI frame containing the console.
       progress_bar (spacrProgressBar): Progress bar updated with progress information.
       uppdate_frequency (int): Refresh rate in milliseconds.


.. py:function:: main_thread_update_function(root, q, fig_queue, canvas_widget)

   Periodically triggers updates on the GUI's canvas or figures from a queue.

   :param root: Root GUI window.
   :type root: tk.Tk
   :param q: Message queue (currently unused inside function).
   :type q: Queue
   :param fig_queue: Queue for figure data (currently unused inside function).
   :type fig_queue: Queue
   :param canvas_widget: Canvas widget to refresh.
   :type canvas_widget: tk.Widget

   Globals Used:
       uppdate_frequency (int): Interval for update scheduling.


.. py:function:: cleanup_previous_instance()

   Cleans up any remnants of the previous application run, including GUI widgets,
   background tasks, queues, threads, and canvas elements.

   Globals Modified:
       parent_frame, usage_bars, figures, figure_index, thread_control,
       canvas, q, fig_queue


.. py:function:: initiate_root(parent, settings_type='mask')

   Initializes and builds the GUI based on the selected settings type.

   :param parent: Root window for the GUI.
   :type parent: tk.Tk or tk.Toplevel
   :param settings_type: Type of settings to configure (e.g., 'mask', 'measure').
   :type settings_type: str

   :returns:     parent_frame (tk.Frame): The root GUI container.
                 vars_dict (dict): Dictionary of user-configurable variables.
   :rtype: tuple

   Globals Set:
       q, fig_queue, thread_control, parent_frame, scrollable_frame,
       button_frame, vars_dict, canvas, canvas_widget, button_scrollable_frame,
       progress_bar, uppdate_frequency, figures, figure_index, index_control, usage_bars


