import os, io, sys, ast, ctypes, ast, sqlite3, requests, time, traceback, torch, cv2
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from huggingface_hub import list_repo_files
import psutil
from PIL import Image, ImageTk
from screeninfo import get_monitors

from .gui_elements import spacrEntry, spacrCheck, spacrCombo

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

def initialize_cuda():
    """
    Initializes CUDA for the main process if a compatible GPU is available.

    This function checks if CUDA is available on the system. If it is, it allocates
    a small tensor on the GPU to ensure that CUDA is properly initialized. A message
    is printed to indicate whether CUDA was successfully initialized or if it is not
    available.

    Note:
        This function is intended to be used in environments where CUDA-enabled GPUs
        are present and PyTorch is installed.

    Prints:
        - "CUDA initialized in the main process." if CUDA is available and initialized.
        - "CUDA is not available." if no compatible GPU is detected.
    """
    if torch.cuda.is_available():
        # Allocate a small tensor on the GPU
        _ = torch.tensor([0.0], device='cuda')
        print("CUDA initialized in the main process.")
    else:
        print("CUDA is not available.")

def set_high_priority(process):
    """
    Sets the priority of a given process to high.

    On Windows systems, the process priority is set to HIGH_PRIORITY_CLASS.
    On Unix-like systems, the process priority is adjusted to a higher level
    by setting its niceness value to -10.

    Args:
        process (psutil.Process): The process whose priority is to be adjusted.

    Raises:
        psutil.AccessDenied: If the current user does not have permission to change
                             the priority of the process.
        psutil.NoSuchProcess: If the specified process does not exist.
        Exception: For any other errors encountered during the operation.

    Notes:
        - This function requires the `psutil` library to interact with system processes.
        - Adjusting process priority may require elevated privileges depending on the
          operating system and user permissions.
    """
    try:
        p = psutil.Process(process.pid)
        if os.name == 'nt':  # Windows
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:  # Unix-like systems
            p.nice(-10)  # Adjusted priority level
        print(f"Successfully set high priority for process: {process.pid}")
    except psutil.AccessDenied as e:
        print(f"Access denied when trying to set high priority for process {process.pid}: {e}")
    except psutil.NoSuchProcess as e:
        print(f"No such process {process.pid}: {e}")
    except Exception as e:
        print(f"Failed to set high priority for process {process.pid}: {e}")

def set_cpu_affinity(process):
    """
    Set the CPU affinity for a given process to use all available CPUs.

    This function modifies the CPU affinity of the specified process, allowing
    it to run on all CPUs available on the system.

    Args:
        process (psutil.Process): A psutil.Process object representing the process
                                  whose CPU affinity is to be set.

    Raises:
        psutil.NoSuchProcess: If the process does not exist.
        psutil.AccessDenied: If the process cannot be accessed due to insufficient permissions.
        psutil.ZombieProcess: If the process is a zombie process.
    """
    p = psutil.Process(process.pid)
    p.cpu_affinity(list(range(os.cpu_count())))
    
def proceed_with_app(root, app_name, app_func):
    """
    Prepares the application window to load a new app by clearing the current 
    content frame and initializing the specified app.

    Args:
        root (tk.Tk or tk.Toplevel): The root window or parent container that 
            contains the content frame.
        app_name (str): The name of the application to be loaded (not used in 
            the current implementation but could be useful for logging or 
            debugging purposes).
        app_func (callable): A function that initializes the new application 
            within the content frame. It should accept the content frame as 
            its only argument.

    Behavior:
        - Destroys all widgets in the `content_frame` attribute of `root` 
          (if it exists).
        - Calls `app_func` with `root.content_frame` to initialize the new 
          application.

    Note:
        Ensure that `root` has an attribute `content_frame` that is a valid 
        tkinter container (e.g., a `tk.Frame`) before calling this function.
    """
    # Clear the current content frame
    if hasattr(root, 'content_frame'):
        for widget in root.content_frame.winfo_children():
            try:
                widget.destroy()
            except tk.TclError as e:
                print(f"Error destroying widget: {e}")

    # Initialize the new app in the content frame
    app_func(root.content_frame)

def load_app(root, app_name, app_func):
    """
    Load a new application into the GUI framework.

    This function handles the transition between applications in the GUI by 
    clearing the current canvas, canceling scheduled tasks, and invoking 
    exit functionality for specific applications if necessary.

    Args:
        root: The root object of the GUI, which contains the canvas, 
              after_tasks, and other application state.
        app_name (str): The name of the application to load.
        app_func (callable): The function to initialize the new application.

    Behavior:
        - Clears the current canvas if it exists.
        - Cancels all scheduled `after` tasks associated with the root object.
        - If the current application has an exit function and the new app is 
          not "Annotate" or "make_masks", the exit function is invoked before 
          proceeding to the new application.
        - Proceeds to load the new application using the provided `app_func`.

    Note:
        The `proceed_with_app` function is used internally to finalize the 
        transition to the new application.
    """
    # Clear the canvas if it exists
    if root.canvas is not None:
        root.clear_frame(root.canvas)

    # Cancel all scheduled after tasks
    if hasattr(root, 'after_tasks'):
        for task in root.after_tasks:
            root.after_cancel(task)
    root.after_tasks = []

    # Exit functionality only for the annotation and make_masks apps
    if app_name not in ["Annotate", "make_masks"] and hasattr(root, 'current_app_exit_func'):
        root.next_app_func = proceed_with_app
        root.next_app_args = (app_name, app_func)
        root.current_app_exit_func()
    else:
        proceed_with_app(root, app_name, app_func)

def parse_list(value):
    """
    Parses a string representation of a list and returns the parsed list.

    Args:
        value (str): The string representation of the list.

    Returns:
        list: The parsed list, which can contain integers, floats, or strings.

    Raises:
        ValueError: If the input value is not a valid list format or contains mixed types or unsupported types.
    """
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            # Check if all elements are homogeneous (either all int, float, or str)
            if all(isinstance(item, (int, float, str)) for item in parsed_value):
                return parsed_value
            else:
                raise ValueError("List contains mixed types or unsupported types")
        elif isinstance(parsed_value, tuple):
            # Convert tuple to list if it’s a single-element tuple
            return list(parsed_value) if len(parsed_value) > 1 else [parsed_value[0]]
        else:
            raise ValueError(f"Expected a list but got {type(parsed_value).__name__}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid format for list: {value}. Error: {e}")

def create_input_field(frame, label_text, row, var_type='entry', options=None, default_value=None):
    """
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

    """
    from .gui_elements import set_dark_style, set_element_size
    
    label_column = 0
    widget_column = 0  # Both label and widget will be in the same column

    style_out = set_dark_style(ttk.Style())
    font_loader = style_out['font_loader']
    font_size = style_out['font_size']
    size_dict = set_element_size()
    size_dict['settings_width'] = size_dict['settings_width'] - int(size_dict['settings_width']*0.1)

    # Replace underscores with spaces and capitalize the first letter

    label_text = label_text.replace('_', ' ').capitalize()

    # Configure the column widths
    frame.grid_columnconfigure(label_column, weight=1)  # Allow the column to expand

    # Create a custom frame with a translucent background and rounded edges
    custom_frame = tk.Frame(frame, bg=style_out['bg_color'], bd=2, relief='solid', width=size_dict['settings_width'])
    custom_frame.grid(column=label_column, row=row, sticky=tk.EW, padx=(5, 5), pady=5)

    # Apply styles to custom frame
    custom_frame.update_idletasks()
    custom_frame.config(highlightbackground=style_out['bg_color'], highlightthickness=1, bd=2)

    # Create and configure the label
    label = tk.Label(custom_frame, text=label_text, bg=style_out['bg_color'], fg=style_out['fg_color'], font=font_loader.get_font(size=font_size), anchor='e', justify='right')
    label.grid(column=label_column, row=0, sticky=tk.W, padx=(5, 2), pady=5)  # Place the label in the first row

    # Create and configure the input widget based on var_type
    try:
        if var_type == 'entry':
            var = tk.StringVar(value=default_value)
            entry = spacrEntry(custom_frame, textvariable=var, outline=False, width=size_dict['settings_width'])
            entry.grid(column=widget_column, row=1, sticky=tk.W, padx=(2, 5), pady=5)  # Place the entry in the second row
            return (label, entry, var, custom_frame)  # Return both the label and the entry, and the variable
        elif var_type == 'check':
            var = tk.BooleanVar(value=default_value)  # Set default value (True/False)
            check = spacrCheck(custom_frame, text="", variable=var)
            check.grid(column=widget_column, row=1, sticky=tk.W, padx=(2, 5), pady=5)  # Place the checkbutton in the second row
            return (label, check, var, custom_frame)  # Return both the label and the checkbutton, and the variable
        elif var_type == 'combo':
            var = tk.StringVar(value=default_value)  # Set default value
            combo = spacrCombo(custom_frame, textvariable=var, values=options, width=size_dict['settings_width'])  # Apply TCombobox style
            combo.grid(column=widget_column, row=1, sticky=tk.W, padx=(2, 5), pady=5)  # Place the combobox in the second row
            if default_value:
                combo.set(default_value)
            return (label, combo, var, custom_frame)  # Return both the label and the combobox, and the variable
        else:
            var = None  # Placeholder in case of an undefined var_type
            return (label, None, var, custom_frame)
    except Exception as e:
        traceback.print_exc()
        print(f"Error creating input field: {e}")
        print(f"Wrong type for {label_text} Expected {var_type}")

def process_stdout_stderr(q):
    """
    Redirects the standard output (stdout) and standard error (stderr) streams
    to a queue for processing.

    This function replaces the default `sys.stdout` and `sys.stderr` with
    instances of `WriteToQueue`, which write all output to the provided queue.

    :param q: A queue object where the redirected output will be stored.
    :type q: queue.Queue
    """
    sys.stdout = WriteToQueue(q)
    sys.stderr = WriteToQueue(q)

class WriteToQueue(io.TextIOBase):
    """
    A file-like object that redirects writes to a queue.

    :param q: The queue to write output to.
    :type q: queue.Queue
    """
    def __init__(self, q):
        self.q = q
    def write(self, msg):
        """
        Write string to stream.

        :param msg: The string message to write.
        :type msg: str
        :returns: Number of characters written.
        :rtype: int
        """
        if msg.strip():  # Avoid empty messages
            self.q.put(msg)
    def flush(self):
        """
        Flush write buffers, if applicable.

        This is a no-op in this implementation.
        """
        pass

def cancel_after_tasks(frame):
    """
    Cancels all scheduled 'after' tasks associated with a given frame.

    This function checks if the provided frame object has an attribute 
    named 'after_tasks', which is expected to be a list of task IDs 
    scheduled using the `after` method (e.g., in a Tkinter application). 
    If such tasks exist, it cancels each of them using the `after_cancel` 
    method and then clears the list.

    Args:
        frame: An object (typically a Tkinter widget) that may have an 
               'after_tasks' attribute containing scheduled task IDs.

    Raises:
        AttributeError: If the frame does not have the required methods 
                        (`after_cancel` or `after_tasks` attribute).
    """
    if hasattr(frame, 'after_tasks'):
        for task in frame.after_tasks:
            frame.after_cancel(task)
        frame.after_tasks.clear()

def load_next_app(root):
    """
    Loads the next application by invoking the function stored in the `next_app_func` 
    attribute of the provided `root` object. If the current root window has been 
    destroyed, a new root window is initialized before invoking the next application.

    Args:
        root (tk.Tk): The current root window object, which contains the attributes 
                      `next_app_func` (a callable for the next application) and 
                      `next_app_args` (a tuple of arguments to pass to the callable).

    Raises:
        tk.TclError: If the root window does not exist and needs to be reinitialized.
    """
    # Get the next app function and arguments
    next_app_func = root.next_app_func
    next_app_args = root.next_app_args

    if next_app_func:
        try:
            if not root.winfo_exists():
                raise tk.TclError
            next_app_func(root, *next_app_args)
        except tk.TclError:
            # Reinitialize root if it has been destroyed
            new_root = tk.Tk()
            width = new_root.winfo_screenwidth()
            height = new_root.winfo_screenheight()
            new_root.geometry(f"{width}x{height}")
            new_root.title("SpaCr Application")
            next_app_func(new_root, *next_app_args)

def convert_settings_dict_for_gui(settings):
    """
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
    """

    from torchvision import models as torch_models
    torchvision_models = [name for name, obj in torch_models.__dict__.items() if callable(obj)]
    chans = ['0', '1', '2', '3', '4', '5', '6', '7', '8', None]
    chan_list = ['[0,1,2,3,4,5,6,7,8]','[0,1,2,3,4,5,6,7]','[0,1,2,3,4,5,6]','[0,1,2,3,4,5]','[0,1,2,3,4]','[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]', '[0,0]']
    chans_v2 = [0, 1, 2, 3, None]
    chans_v3 = list(range(0, 21, 1)) + [None]
    chans_v4 = [0, 1, 2, 3, None]
    variables = {}
    special_cases = {
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'auto', 'custom'], 'cellvoyager'),
        'channels': ('combo', chan_list, '[0,1,2,3]'),
        'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
        'channel_dims': ('combo', chan_list, '[0,1,2,3]'),
        'dataset_mode': ('combo', ['annotation', 'metadata', 'recruitment'], 'metadata'),
        'cov_type': ('combo', ['HC0', 'HC1', 'HC2', 'HC3', None], None),
        'crop_mode': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']"], "['cell']"),
        'timelapse_mode': ('combo', ['trackpy', 'iou', 'btrack'], 'trackpy'),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
        'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
        'model_name': ('combo', ['cyto', 'cyto_2', 'cyto_3', 'nuclei'], 'cyto'),
        'regression_type': ('combo', ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge'], 'ols'),
        'timelapse_objects': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']", None], None),
        'model_type': ('combo', torchvision_models, 'resnet50'),
        'optimizer_type': ('combo', ['adamw', 'adam'], 'adamw'),
        'schedule': ('combo', ['reduce_lr_on_plateau', 'step_lr'], 'reduce_lr_on_plateau'),
        'loss_type': ('combo', ['focal_loss', 'binary_cross_entropy_with_logits'], 'focal_loss'),
        'normalize_by': ('combo', ['fov', 'png'], 'png'),
        'agg_type': ('combo', ['mean', 'median'], 'mean'),
        'grouping': ('combo', ['mean', 'median'], 'mean'),
        'min_max': ('combo', ['allq', 'all'], 'allq'),
        'transform': ('combo', ['log', 'sqrt', 'square', None], None)
    }

    for key, value in settings.items():
        if key in special_cases:
            variables[key] = special_cases[key]
        elif isinstance(value, bool):
            variables[key] = ('check', None, value)
        elif isinstance(value, int) or isinstance(value, float):
            variables[key] = ('entry', None, value)
        elif isinstance(value, str):
            variables[key] = ('entry', None, value)
        elif value is None:
            variables[key] = ('entry', None, value)
        elif isinstance(value, list):
            variables[key] = ('entry', None, str(value))
        else:
            variables[key] = ('entry', None, str(value))
    
    return variables

def spacrFigShow(fig_queue=None):
    """
    Displays the current matplotlib figure or adds it to a queue.

    This function retrieves the current matplotlib figure using `plt.gcf()`. 
    If a `fig_queue` is provided, the figure is added to the queue. 
    Otherwise, the figure is displayed using the `show()` method. 
    After the figure is either queued or displayed, it is closed using `plt.close()`.

    Args:
        fig_queue (queue.Queue, optional): A queue to store the figure. 
                                           If None, the figure is displayed instead.

    Returns:
        None
    """
    fig = plt.gcf()
    if fig_queue:
        fig_queue.put(fig)
    else:
        fig.show()
    plt.close(fig)

def function_gui_wrapper(function=None, settings={}, q=None, fig_queue=None, imports=1):

    """
    Wraps the run_multiple_simulations function to integrate with GUI processes.
    
    Args:
    - settings: dict, The settings for the run_multiple_simulations function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        if imports == 1:
            function(settings=settings)
        elif imports == 2:
            function(src=settings['src'], settings=settings)
    except Exception as e:
        # Send the error message to the GUI via the queue
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage) 
        traceback.print_exc()
    finally:
        # Restore the original plt.show function
        plt.show = original_show
        
def run_function_gui(settings_type, settings, q, fig_queue, stop_requested):
    """
    Executes a specified processing function in the GUI context based on `settings_type`.

    This function selects and runs one of the core `spaCR` processing functions 
    (e.g., segmentation, measurement, classification, barcode mapping) based on the 
    provided `settings_type` string. It wraps the execution with a logging mechanism 
    to redirect stdout/stderr to the GUI console and handles exceptions cleanly.

    Args
    ----------
    settings_type : str
        A string indicating which processing function to execute. Supported values include:
        'mask', 'measure', 'classify', 'train_cellpose', 'ml_analyze', 'cellpose_masks',
        'cellpose_all', 'map_barcodes', 'regression', 'recruitment', 'analyze_plaques', 'convert'.

    settings : dict
        A dictionary of parameters required by the selected function.

    q : multiprocessing.Queue
        Queue for redirecting standard output and errors to the GUI console.

    fig_queue : multiprocessing.Queue
        Queue used to transfer figures (e.g., plots) from the worker process to the GUI.

    stop_requested : multiprocessing.Value
        A shared value to signal whether execution has completed or was interrupted.

    Raises
    ------
    ValueError
        If an invalid `settings_type` is provided.

    Notes
    -----
    - Redirects stdout/stderr to the GUI using `process_stdout_stderr`.
    - Catches and reports any exceptions to the GUI queue.
    - Sets `stop_requested.value = 1` when the task finishes (whether successful or not).
    """
    from .core import preprocess_generate_masks
    from .spacr_cellpose import identify_masks_finetune, check_cellpose_models
    from .submodules import analyze_recruitment
    from .ml import generate_ml_scores, perform_regression
    from .submodules import train_cellpose, analyze_plaques
    from .io import process_non_tif_non_2D_images
    from .measure import measure_crop
    from .deep_spacr import deep_spacr
    from .sequencing import generate_barecode_mapping
    
    process_stdout_stderr(q)
    
    print(f'run_function_gui settings_type: {settings_type}')
    
    if settings_type == 'mask':
        function = preprocess_generate_masks
        imports = 1
    elif settings_type == 'measure':
        function = measure_crop
        imports = 1
    elif settings_type == 'classify':
        function = deep_spacr
        imports = 1
    elif settings_type == 'train_cellpose':
        function = train_cellpose
        imports = 1
    elif settings_type == 'ml_analyze':
        function = generate_ml_scores
        imports = 1
    elif settings_type == 'cellpose_masks':
        function = identify_masks_finetune
        imports = 1
    elif settings_type == 'cellpose_all':
        function = check_cellpose_models
        imports = 1
    elif settings_type == 'map_barcodes':
        function = generate_barecode_mapping
        imports = 1
    elif settings_type == 'regression':
        function = perform_regression
        imports = 2
    elif settings_type == 'recruitment':
        function = analyze_recruitment
        imports = 1
    elif settings_type == 'analyze_plaques':
        function = analyze_plaques
        imports = 1
    elif settings_type == 'convert':
        function = process_non_tif_non_2D_images
        imports = 1
    else:
        raise ValueError(f"Error: Invalid settings type: {settings_type}")
    try:
        function_gui_wrapper(function, settings, q, fig_queue, imports)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def hide_all_settings(vars_dict, categories):
    """
    Function to initially hide all settings in the GUI.

    Args:
    - categories: dict, The categories of settings with their corresponding settings.
    - vars_dict: dict, The dictionary containing the settings and their corresponding widgets.
    """

    if categories is None:
        from .settings import categories

    for category, settings in categories.items():
        if any(setting in vars_dict for setting in settings):
            vars_dict[category] = (None, None, tk.IntVar(value=0), None)
            
            # Initially hide all settings
            for setting in settings:
                if setting in vars_dict:
                    label, widget, _, frame = vars_dict[setting]
                    label.grid_remove()
                    widget.grid_remove()
                    frame.grid_remove()
    return vars_dict

def setup_frame(parent_frame):
    """
    Set up the main GUI layout within the given parent frame.

    This function initializes a dark-themed, resizable GUI layout using `PanedWindow` 
    containers. It organizes the layout into left-hand settings, central vertical content, 
    and bottom horizontal panels. It also sets initial sash positions and layout weights.

    Args
    ----------
    parent_frame : tk.Frame
        The parent Tkinter frame to populate with the GUI layout.

    Returns
    -------
    tuple
        A tuple containing:
        - parent_frame (tk.Frame): The modified parent frame with the layout initialized.
        - vertical_container (tk.PanedWindow): Top container in the right-hand area for main content.
        - horizontal_container (tk.PanedWindow): Bottom container for additional widgets.
        - settings_container (tk.PanedWindow): Left-hand container for GUI settings.
    
    Notes
    -----
    - Uses `set_dark_style` and `set_element_size` from `gui_elements` to theme and size widgets.
    - Dynamically positions the sash between the left and right panes to 25% of the screen width.
    """
    from .gui_elements import set_dark_style, set_element_size

    style = ttk.Style(parent_frame)
    size_dict = set_element_size()
    style_out = set_dark_style(style)

    # Configure the main layout using PanedWindow
    main_paned = tk.PanedWindow(parent_frame, orient=tk.HORIZONTAL, bg=style_out['bg_color'], bd=0, relief='flat')
    main_paned.grid(row=0, column=0, sticky="nsew")

    # Allow the main_paned to expand and fill the window
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)

    # Create the settings container on the left
    settings_container = tk.PanedWindow(main_paned, orient=tk.VERTICAL, width=size_dict['settings_width'], bg=style_out['bg_color'], bd=0, relief='flat')
    main_paned.add(settings_container, minsize=100)  # Allow resizing with a minimum size

    # Create a right container frame to hold vertical and horizontal containers
    right_frame = tk.Frame(main_paned, bg=style_out['bg_color'], bd=0, highlightthickness=0, relief='flat')
    main_paned.add(right_frame, stretch="always")

    # Configure the right_frame grid layout
    right_frame.grid_rowconfigure(0, weight=1)  # Vertical container expands
    right_frame.grid_rowconfigure(1, weight=0)  # Horizontal container at bottom
    right_frame.grid_columnconfigure(0, weight=1)

    # Inside right_frame, add vertical_container at the top
    vertical_container = tk.PanedWindow(right_frame, orient=tk.VERTICAL, bg=style_out['bg_color'], bd=0, relief='flat')
    vertical_container.grid(row=0, column=0, sticky="nsew")

    # Add horizontal_container aligned with the bottom of settings_container
    horizontal_container = tk.PanedWindow(right_frame, orient=tk.HORIZONTAL, height=size_dict['panel_height'], bg=style_out['bg_color'], bd=0, relief='flat')
    horizontal_container.grid(row=1, column=0, sticky="ew")

    # Example content for settings_container
    tk.Label(settings_container, text="Settings Container", bg=style_out['bg_color']).pack(fill=tk.BOTH, expand=True)

    set_dark_style(style, parent_frame, [settings_container, vertical_container, horizontal_container, main_paned])
    
    # Set initial sash position for main_paned (left/right split)
    parent_frame.update_idletasks()
    screen_width = parent_frame.winfo_screenwidth()
    target_width = int(screen_width / 4)
    main_paned.sash_place(0, target_width, 0)

    return parent_frame, vertical_container, horizontal_container, settings_container

def download_hug_dataset(q, vars_dict):
    """
    Downloads a dataset and settings files from the Hugging Face Hub and updates the provided variables dictionary.

    Args:
        q (queue.Queue): A queue object used for logging messages during the download process.
        vars_dict (dict): A dictionary containing variables to be updated. If 'src' is present in the dictionary,
                          the third element of 'src' will be updated with the downloaded dataset path.

    The function performs the following steps:
        1. Downloads a dataset from the Hugging Face Hub using the specified repository ID and subfolder.
        2. Updates the 'src' variable in `vars_dict` with the local path of the downloaded dataset, if applicable.
        3. Logs the dataset download status to the provided queue.
        4. Downloads settings files from another repository on the Hugging Face Hub.
        5. Logs the settings download status to the provided queue.

    Notes:
        - The dataset is downloaded to a local directory under the user's home directory named "datasets".
        - Any exceptions during the download process are caught and logged to the queue.

    Raises:
        None: All exceptions are handled internally and logged to the queue.
    """
    dataset_repo_id = "einarolafsson/toxo_mito"
    settings_repo_id = "einarolafsson/spacr_settings"
    dataset_subfolder = "plate1"
    local_dir = os.path.join(os.path.expanduser("~"), "datasets")

    # Download the dataset
    try:
        dataset_path = download_dataset(q, dataset_repo_id, dataset_subfolder, local_dir)
        if 'src' in vars_dict:
            vars_dict['src'][2].set(dataset_path)
            q.put(f"Set source path to: {vars_dict['src'][2].get()}\n")
        q.put(f"Dataset downloaded to: {dataset_path}\n")
    except Exception as e:
        q.put(f"Failed to download dataset: {e}\n")

    # Download the settings files
    try:
        settings_path = download_dataset(q, settings_repo_id, "", local_dir)
        q.put(f"Settings downloaded to: {settings_path}\n")
    except Exception as e:
        q.put(f"Failed to download settings: {e}\n")

def download_dataset(q, repo_id, subfolder, local_dir=None, retries=5, delay=5):
    """
    Downloads a dataset or settings files from Hugging Face and returns the local path.

    Args:
        repo_id (str): The repository ID (e.g., 'einarolafsson/toxo_mito' or 'einarolafsson/spacr_settings').
        subfolder (str): The subfolder path within the repository (e.g., 'plate1' or the settings subfolder).
        local_dir (str): The local directory where the files will be saved. Defaults to the user's home directory.
        retries (int): Number of retry attempts in case of failure.
        delay (int): Delay in seconds between retries.

    Returns:
        str: The local path to the downloaded files.
    """
    if local_dir is None:
        local_dir = os.path.join(os.path.expanduser("~"), "datasets")

    local_subfolder_dir = os.path.join(local_dir, subfolder if subfolder else "settings")
    if not os.path.exists(local_subfolder_dir):
        os.makedirs(local_subfolder_dir)
    elif len(os.listdir(local_subfolder_dir)) > 0:
        q.put(f"Files already downloaded to: {local_subfolder_dir}")
        return local_subfolder_dir

    attempt = 0
    while attempt < retries:
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            subfolder_files = [file for file in files if file.startswith(subfolder) or (subfolder == "" and file.endswith('.csv'))]

            for file_name in subfolder_files:
                for download_attempt in range(retries):
                    try:
                        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}?download=true"
                        response = requests.get(url, stream=True)
                        response.raise_for_status()

                        local_file_path = os.path.join(local_subfolder_dir, os.path.basename(file_name))
                        with open(local_file_path, 'wb') as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        q.put(f"Downloaded file: {file_name}")
                        break
                    except (requests.HTTPError, requests.Timeout) as e:
                        q.put(f"Error downloading {file_name}: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                else:
                    raise Exception(f"Failed to download {file_name} after multiple attempts.")

            return local_subfolder_dir

        except (requests.HTTPError, requests.Timeout) as e:
            q.put(f"Error downloading files: {e}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)

    raise Exception("Failed to download files after multiple attempts.")

def ensure_after_tasks(frame):
    if not hasattr(frame, 'after_tasks'):
        frame.after_tasks = []

def display_gif_in_plot_frame(gif_path, parent_frame):
    """Display and zoom a GIF to fill the entire parent_frame, maintaining aspect ratio, with lazy resizing and caching."""
    # Clear parent_frame if it contains any previous widgets
    for widget in parent_frame.winfo_children():
        widget.destroy()

    # Load the GIF
    gif = Image.open(gif_path)

    # Get the aspect ratio of the GIF
    gif_width, gif_height = gif.size
    gif_aspect_ratio = gif_width / gif_height

    # Create a label to display the GIF and configure it to fill the parent_frame
    label = tk.Label(parent_frame, bg="black")
    label.grid(row=0, column=0, sticky="nsew")  # Expands in all directions (north, south, east, west)

    # Configure parent_frame to stretch the label to fill available space
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)

    # Cache for storing resized frames (lazily filled)
    resized_frames_cache = {}

    # Store last frame size and aspect ratio
    last_frame_width = 0
    last_frame_height = 0

    def resize_and_crop_frame(frame_idx, frame_width, frame_height):
        """Resize and crop the current frame of the GIF to fit the parent_frame while maintaining the aspect ratio."""
        # If the frame is already cached at the current size, return it
        if (frame_idx, frame_width, frame_height) in resized_frames_cache:
            return resized_frames_cache[(frame_idx, frame_width, frame_height)]

        # Calculate the scaling factor to zoom in on the GIF
        scale_factor = max(frame_width / gif_width, frame_height / gif_height)

        # Calculate new dimensions while maintaining the aspect ratio
        new_width = int(gif_width * scale_factor)
        new_height = int(gif_height * scale_factor)

        # Resize the GIF to fit the frame using NEAREST for faster resizing
        gif.seek(frame_idx)
        resized_gif = gif.copy().resize((new_width, new_height), Image.Resampling.NEAREST if scale_factor > 2 else Image.Resampling.LANCZOS)

        # Calculate the cropping box to center the resized GIF in the frame
        crop_left = (new_width - frame_width) // 2
        crop_top = (new_height - frame_height) // 2
        crop_right = crop_left + frame_width
        crop_bottom = crop_top + frame_height

        # Crop the resized GIF to exactly fit the frame
        cropped_gif = resized_gif.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Convert the cropped frame to a Tkinter-compatible format
        frame_image = ImageTk.PhotoImage(cropped_gif)

        # Cache the resized frame
        resized_frames_cache[(frame_idx, frame_width, frame_height)] = frame_image

        return frame_image

    def update_frame(frame_idx):
        """Update the GIF frame using lazy resizing and caching."""
        # Get the current size of the parent_frame
        frame_width = parent_frame.winfo_width()
        frame_height = parent_frame.winfo_height()

        # Only resize if the frame size has changed
        nonlocal last_frame_width, last_frame_height
        if frame_width != last_frame_width or frame_height != last_frame_height:
            last_frame_width, last_frame_height = frame_width, frame_height

        # Get the resized and cropped frame image
        frame_image = resize_and_crop_frame(frame_idx, frame_width, frame_height)
        label.config(image=frame_image)
        label.image = frame_image  # Keep a reference to avoid garbage collection

        # Move to the next frame, or loop back to the beginning
        next_frame_idx = (frame_idx + 1) % gif.n_frames
        parent_frame.after(gif.info['duration'], update_frame, next_frame_idx)

    # Start the GIF animation from frame 0
    update_frame(0)
    
def display_media_in_plot_frame(media_path, parent_frame):
    """
    Display a media file (MP4, AVI, or GIF) in a Tkinter frame, playing it on repeat 
    while fully filling the frame and maintaining the aspect ratio.

    Args:
        media_path (str): The file path to the media file (MP4, AVI, or GIF).
        parent_frame (tk.Frame): The Tkinter frame where the media will be displayed.

    Behavior:
        - For MP4 and AVI files:
            - Uses OpenCV to read and play the video.
            - Resizes and crops the video to fully fill the parent frame while maintaining aspect ratio.
            - Plays the video on repeat.
        - For GIF files:
            - Uses PIL to read and play the GIF.
            - Resizes and crops the GIF to fully fill the parent frame while maintaining aspect ratio.
            - Plays the GIF on repeat.

    Raises:
        ValueError: If the file format is not supported (only MP4, AVI, and GIF are supported).

    Notes:
        - The function clears any existing widgets in the parent frame before displaying the media.
        - The parent frame is configured to expand and adapt to the media's aspect ratio.
    """
    """Display an MP4, AVI, or GIF and play it on repeat in the parent_frame, fully filling the frame while maintaining aspect ratio."""
    # Clear parent_frame if it contains any previous widgets
    for widget in parent_frame.winfo_children():
        widget.destroy()

    # Check file extension to decide between video (mp4/avi) or gif
    file_extension = os.path.splitext(media_path)[1].lower()

    if file_extension in ['.mp4', '.avi']:
        # Handle video formats (mp4, avi) using OpenCV
        video = cv2.VideoCapture(media_path)

        # Create a label to display the video
        label = tk.Label(parent_frame, bg="black")
        label.grid(row=0, column=0, sticky="nsew")

        # Configure the parent_frame to expand
        parent_frame.grid_rowconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

        def update_frame():
            """Update function for playing video."""
            ret, frame = video.read()
            if ret:
                # Get the frame dimensions
                frame_height, frame_width, _ = frame.shape

                # Get parent frame dimensions
                parent_width = parent_frame.winfo_width()
                parent_height = parent_frame.winfo_height()

                # Ensure dimensions are greater than 0
                if parent_width > 0 and parent_height > 0:
                    # Calculate the aspect ratio of the media
                    frame_aspect_ratio = frame_width / frame_height
                    parent_aspect_ratio = parent_width / parent_height

                    # Determine whether to scale based on width or height to cover the parent frame
                    if parent_aspect_ratio > frame_aspect_ratio:
                        # The parent frame is wider than the video aspect ratio
                        # Fit to width, crop height
                        new_width = parent_width
                        new_height = int(parent_width / frame_aspect_ratio)
                    else:
                        # The parent frame is taller than the video aspect ratio
                        # Fit to height, crop width
                        new_width = int(parent_height * frame_aspect_ratio)
                        new_height = parent_height

                    # Resize the frame to the new dimensions (cover the parent frame)
                    resized_frame = cv2.resize(frame, (new_width, new_height))

                    # Crop the frame to fit exactly within the parent frame
                    x_offset = (new_width - parent_width) // 2
                    y_offset = (new_height - parent_height) // 2
                    cropped_frame = resized_frame[y_offset:y_offset + parent_height, x_offset:x_offset + parent_width]

                    # Convert the frame to RGB (OpenCV uses BGR by default)
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                    # Convert the frame to a Tkinter-compatible format
                    frame_image = ImageTk.PhotoImage(Image.fromarray(cropped_frame))

                    # Update the label with the new frame
                    label.config(image=frame_image)
                    label.image = frame_image  # Keep a reference to avoid garbage collection

                # Call update_frame again after a delay to match the video's frame rate
                parent_frame.after(int(1000 / video.get(cv2.CAP_PROP_FPS)), update_frame)
            else:
                # Restart the video if it reaches the end
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                update_frame()

        # Start the video playback
        update_frame()

    elif file_extension == '.gif':
        # Handle GIF format using PIL
        gif = Image.open(media_path)

        # Create a label to display the GIF
        label = tk.Label(parent_frame, bg="black")
        label.grid(row=0, column=0, sticky="nsew")

        # Configure the parent_frame to expand
        parent_frame.grid_rowconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

        def update_gif_frame(frame_idx):
            """Update function for playing GIF."""
            try:
                gif.seek(frame_idx)  # Move to the next frame

                # Get the frame dimensions
                gif_width, gif_height = gif.size

                # Get parent frame dimensions
                parent_width = parent_frame.winfo_width()
                parent_height = parent_frame.winfo_height()

                # Ensure dimensions are greater than 0
                if parent_width > 0 and parent_height > 0:
                    # Calculate the aspect ratio of the GIF
                    gif_aspect_ratio = gif_width / gif_height
                    parent_aspect_ratio = parent_width / parent_height

                    # Determine whether to scale based on width or height to cover the parent frame
                    if parent_aspect_ratio > gif_aspect_ratio:
                        # Fit to width, crop height
                        new_width = parent_width
                        new_height = int(parent_width / gif_aspect_ratio)
                    else:
                        # Fit to height, crop width
                        new_width = int(parent_height * gif_aspect_ratio)
                        new_height = parent_height

                    # Resize the GIF frame to cover the parent frame
                    resized_gif = gif.copy().resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Crop the resized GIF to fit the exact parent frame dimensions
                    x_offset = (new_width - parent_width) // 2
                    y_offset = (new_height - parent_height) // 2
                    cropped_gif = resized_gif.crop((x_offset, y_offset, x_offset + parent_width, y_offset + parent_height))

                    # Convert the frame to a Tkinter-compatible format
                    frame_image = ImageTk.PhotoImage(cropped_gif)

                    # Update the label with the new frame
                    label.config(image=frame_image)
                    label.image = frame_image  # Keep a reference to avoid garbage collection
                    frame_idx += 1
            except EOFError:
                frame_idx = 0  # Restart the GIF if at the end

            # Schedule the next frame update
            parent_frame.after(gif.info['duration'], update_gif_frame, frame_idx)

        # Start the GIF animation from frame 0
        update_gif_frame(0)

    else:
        raise ValueError("Unsupported file format. Only .mp4, .avi, and .gif are supported.")

def print_widget_structure(widget, indent=0):    
    # Print the widget's name and class
    print(" " * indent + f"{widget}: {widget.winfo_class()}")
    
    # Recursively print all child widgets
    for child_name, child_widget in widget.children.items():
        print_widget_structure(child_widget, indent + 2)

def get_screen_dimensions():
    monitor = get_monitors()[0]  # Get the primary monitor
    screen_width = monitor.width
    screen_height = monitor.height
    return screen_width, screen_height

def convert_to_number(value):
    
    """
    Converts a string value to an integer if possible, otherwise converts to a float.

    Args:
        value (str): The string representation of the number.

    Returns:
        int or float: The converted number.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Unable to convert '{value}' to an integer or float.")