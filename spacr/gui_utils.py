import os, io, sys, ast, ctypes, ast, sqlite3, requests, time, traceback, torch
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from huggingface_hub import list_repo_files
import psutil

from .gui_elements import AnnotateApp, spacrEntry, spacrCheck, spacrCombo

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

def initialize_cuda():
    """
    Initializes CUDA in the main process by performing a simple GPU operation.
    """
    if torch.cuda.is_available():
        # Allocate a small tensor on the GPU
        _ = torch.tensor([0.0], device='cuda')
        print("CUDA initialized in the main process.")
    else:
        print("CUDA is not available.")

def set_high_priority(process):
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
    p = psutil.Process(process.pid)
    p.cpu_affinity(list(range(os.cpu_count())))
    
def proceed_with_app(root, app_name, app_func):
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
        list: The parsed list.

    Raises:
        ValueError: If the input value is not a valid list format or contains mixed types or unsupported types.
    """
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            # Check if the list elements are homogeneous (all int or all str)
            if all(isinstance(item, int) for item in parsed_value):
                return parsed_value
            elif all(isinstance(item, str) for item in parsed_value):
                return parsed_value
            else:
                raise ValueError("List contains mixed types or unsupported types")
        else:
            raise ValueError(f"Expected a list but got {type(parsed_value).__name__}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid format for list: {value}. Error: {e}")

# Usage example in your create_input_field function
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
    Redirect stdout and stderr to the queue q.
    """
    sys.stdout = WriteToQueue(q)
    sys.stderr = WriteToQueue(q)

class WriteToQueue(io.TextIOBase):
    """
    A custom file-like class that writes any output to a given queue.
    This can be used to redirect stdout and stderr.
    """
    def __init__(self, q):
        self.q = q
    def write(self, msg):
        if msg.strip():  # Avoid empty messages
            self.q.put(msg)
    def flush(self):
        pass

def cancel_after_tasks(frame):
    if hasattr(frame, 'after_tasks'):
        for task in frame.after_tasks:
            frame.after_cancel(task)
        frame.after_tasks.clear()

def annotate(settings):
    from .settings import set_annotate_default_settings
    settings = set_annotate_default_settings(settings)
    src  = settings['src']

    db = os.path.join(src, 'measurements/measurements.db')
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    if settings['annotation_column'] not in [col[1] for col in cols]:
        c.execute(f"ALTER TABLE png_list ADD COLUMN {settings['annotation_column']} integer")
    conn.commit()
    conn.close()

    root = tk.Tk()
    root.geometry(settings['geom'])
    app = AnnotateApp(root, db, src, image_type=settings['image_type'], channels=settings['channels'], image_size=settings['img_size'], grid_rows=settings['rows'], grid_cols=settings['columns'], annotation_column=settings['annotation_column'], normalize=settings['normalize'], percentiles=settings['percentiles'], measurement=settings['measurement'], threshold=settings['threshold'])
    next_button = tk.Button(root, text="Next", command=app.next_page)
    next_button.grid(row=app.grid_rows, column=app.grid_cols - 1)
    back_button = tk.Button(root, text="Back", command=app.previous_page)
    back_button.grid(row=app.grid_rows, column=app.grid_cols - 2)
    exit_button = tk.Button(root, text="Exit", command=app.shutdown)
    exit_button.grid(row=app.grid_rows, column=app.grid_cols - 3)
    
    app.load_images()
    root.mainloop()

def generate_annotate_fields(frame):
    from .settings import set_annotate_default_settings
    from .gui_elements import set_dark_style

    style_out = set_dark_style(ttk.Style())
    font_loader = style_out['font_loader']
    font_size = style_out['font_size'] - 2

    vars_dict = {}
    settings = set_annotate_default_settings(settings={})
    
    for setting in settings:
        vars_dict[setting] = {
            'entry': ttk.Entry(frame),
            'value': settings[setting]
        }

    # Arrange input fields and labels
    for row, (name, data) in enumerate(vars_dict.items()):
        tk.Label(frame, text=f"{name.replace('_', ' ').capitalize()}:", bg=style_out['bg_color'], fg=style_out['fg_color'], font=font_loader.get_font(size=font_size)).grid(row=row, column=0)
        #ttk.Label(frame, text=f"{name.replace('_', ' ').capitalize()}:", background="black", foreground="white").grid(row=row, column=0)
        if isinstance(data['value'], list):
            # Convert lists to comma-separated strings
            data['entry'].insert(0, ','.join(map(str, data['value'])))
        else:
            data['entry'].insert(0, data['value'])
        data['entry'].grid(row=row, column=1)
    
    return vars_dict

def run_annotate_app(vars_dict, parent_frame):
    settings = {key: data['entry'].get() for key, data in vars_dict.items()}
    settings['channels'] = settings['channels'].split(',')
    settings['img_size'] = list(map(int, settings['img_size'].split(',')))  # Convert string to list of integers
    settings['percentiles'] = list(map(int, settings['percentiles'].split(',')))  # Convert string to list of integers
    settings['normalize'] = settings['normalize'].lower() == 'true'
    settings['rows'] = int(settings['rows'])
    settings['columns'] = int(settings['columns'])
    settings['measurement'] = settings['measurement'].split(',')
    settings['threshold'] = None if settings['threshold'].lower() == 'none' else int(settings['threshold'])

    # Clear previous content instead of destroying the root
    if hasattr(parent_frame, 'winfo_children'):
        for widget in parent_frame.winfo_children():
            widget.destroy()

    # Start the annotate application in the same root window
    annotate_app(parent_frame, settings)

# Global list to keep references to PhotoImage objects
global_image_refs = []

def annotate_app(parent_frame, settings):
    global global_image_refs
    global_image_refs.clear()
    root = parent_frame.winfo_toplevel()
    annotate_with_image_refs(settings, root, lambda: load_next_app(root))

def load_next_app(root):
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

def annotate_with_image_refs(settings, root, shutdown_callback):
    #from .gui_utils import proceed_with_app
    from .gui import gui_app
    from .settings import set_annotate_default_settings

    settings = set_annotate_default_settings(settings)
    src = settings['src']

    db = os.path.join(src, 'measurements/measurements.db')
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    if settings['annotation_column'] not in [col[1] for col in cols]:
        c.execute(f"ALTER TABLE png_list ADD COLUMN {settings['annotation_column']} integer")
    conn.commit()
    conn.close()

    app = AnnotateApp(root, db, src, image_type=settings['image_type'], channels=settings['channels'], image_size=settings['img_size'], grid_rows=settings['rows'], grid_cols=settings['columns'], annotation_column=settings['annotation_column'], normalize=settings['normalize'], percentiles=settings['percentiles'], measurement=settings['measurement'], threshold=settings['threshold'])

    # Set the canvas background to black
    root.configure(bg='black')

    next_button = tk.Button(root, text="Next", command=app.next_page, background='black', foreground='white')
    next_button.grid(row=app.grid_rows, column=app.grid_cols - 1)
    back_button = tk.Button(root, text="Back", command=app.previous_page, background='black', foreground='white')
    back_button.grid(row=app.grid_rows, column=app.grid_cols - 2)
    exit_button = tk.Button(root, text="Exit", command=lambda: [app.shutdown(), shutdown_callback()], background='black', foreground='white')
    exit_button.grid(row=app.grid_rows, column=app.grid_cols - 3)

    #app.load_images()

    # Store the shutdown function and next app details in the root
    root.current_app_exit_func = lambda: [app.shutdown(), shutdown_callback()]
    root.next_app_func = proceed_with_app
    root.next_app_args = ("Main App", gui_app)

    # Call load_images after setting up the root window
    app.load_images()

def annotate_with_image_refs(settings, root, shutdown_callback):
    from .settings import set_annotate_default_settings

    settings = set_annotate_default_settings(settings)
    src = settings['src']

    db = os.path.join(src, 'measurements/measurements.db')
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    if settings['annotation_column'] not in [col[1] for col in cols]:
        c.execute(f"ALTER TABLE png_list ADD COLUMN {settings['annotation_column']} integer")
    conn.commit()
    conn.close()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")

    app = AnnotateApp(root, db, src, image_type=settings['image_type'], channels=settings['channels'], image_size=settings['img_size'], annotation_column=settings['annotation_column'], normalize=settings['normalize'], percentiles=settings['percentiles'], measurement=settings['measurement'], threshold=settings['threshold'])

    # Set the canvas background to black
    root.configure(bg='black')

    # Store the shutdown function and next app details in the root
    root.current_app_exit_func = lambda: [app.shutdown(), shutdown_callback()]

    # Call load_images after setting up the root window
    app.load_images()

def convert_settings_dict_for_gui(settings):
    from torchvision import models as torch_models
    torchvision_models = [name for name, obj in torch_models.__dict__.items() if callable(obj)]
    chans = ['0', '1', '2', '3', '4', '5', '6', '7', '8', None]
    chans_v2 = [0, 1, 2, 3, None]
    variables = {}
    special_cases = {
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'nikon', 'zeis', 'custom'], 'cellvoyager'),
        'channels': ('combo', ['[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]'], '[0,1,2,3]'),
        'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
        'channel_dims': ('combo', ['[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]'], '[0,1,2,3]'),
        'dataset_mode': ('combo', ['annotation', 'metadata', 'recruitment'], 'metadata'),
        'cell_mask_dim': ('combo', chans, None),
        'cell_chann_dim': ('combo', chans, None),
        'nucleus_mask_dim': ('combo', chans, None),
        'nucleus_chann_dim': ('combo', chans, None),
        'pathogen_mask_dim': ('combo', chans, None),
        'pathogen_chann_dim': ('combo', chans, None),
        'crop_mode': ('combo', ['cell', 'nucleus', 'pathogen', '[cell, nucleus, pathogen]', '[cell,nucleus, pathogen]'], ['cell']),
        'magnification': ('combo', [20, 40, 60], 20),
        'nucleus_channel': ('combo', chans_v2, None),
        'cell_channel': ('combo', chans_v2, None),
        'channel_of_interest': ('combo', chans_v2, None),
        'pathogen_channel': ('combo', chans_v2, None),
        'timelapse_mode': ('combo', ['trackpy', 'btrack'], 'trackpy'),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
        'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
        'model_name': ('combo', ['cyto', 'cyto_2', 'cyto_3', 'nuclei'], 'cyto'),
        'regression_type': ('combo', ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge'], 'ols'),
        'timelapse_objects': ('combo', ['cell', 'nucleus', 'pathogen', 'cytoplasm', None], None),
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
    Replacement for plt.show() that queues figures instead of displaying them.
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
    
    Parameters:
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
    
    from .gui_utils import process_stdout_stderr
    from .core import generate_image_umap, preprocess_generate_masks, generate_ml_scores, identify_masks_finetune, check_cellpose_models, analyze_recruitment, train_cellpose, compare_cellpose_masks, analyze_plaques, generate_dataset, apply_model_to_tar
    from .io import generate_cellpose_train_test
    from .measure import measure_crop
    from .sim import run_multiple_simulations
    from .deep_spacr import deep_spacr
    from .sequencing import generate_barecode_mapping, perform_regression
    process_stdout_stderr(q)

    print(f'run_function_gui settings_type: {settings_type}') 
    
    if settings_type == 'mask':
        function = preprocess_generate_masks
        imports = 2
    elif settings_type == 'measure':
        function = measure_crop
        imports = 1
    elif settings_type == 'simulation':
        function = run_multiple_simulations
        imports = 1
    elif settings_type == 'classify':
        function = deep_spacr
        imports = 1
    elif settings_type == 'train_cellpose':
        function = train_cellpose
        imports = 1
    elif settings_type == 'ml_analyze':
        function = generate_ml_scores
        imports = 2
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
    elif settings_type == 'umap':
        function = generate_image_umap
        imports = 1
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")
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

    Parameters:
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
    from .gui_elements import set_dark_style, set_element_size

    style = ttk.Style(parent_frame)
    size_dict = set_element_size()
    style_out = set_dark_style(style)

    settings_container = tk.PanedWindow(parent_frame, orient=tk.VERTICAL, width=size_dict['settings_width'], bg=style_out['bg_color'])
    vertical_container = tk.PanedWindow(parent_frame, orient=tk.VERTICAL, width=size_dict['panel_width'], bg=style_out['bg_color'])
    horizontal_container = tk.PanedWindow(parent_frame, orient=tk.HORIZONTAL, height=size_dict['panel_height'], width=size_dict['panel_width'], bg=style_out['bg_color'])

    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_rowconfigure(1, weight=0)
    parent_frame.grid_columnconfigure(0, weight=0)
    parent_frame.grid_columnconfigure(1, weight=1)

    settings_container.grid(row=0, column=0, rowspan=2, sticky="nsew")
    vertical_container.grid(row=0, column=1, sticky="nsew")
    horizontal_container.grid(row=1, column=1, sticky="ew")

    # Ensure settings_container maintains its width
    settings_container.grid_propagate(False)
    settings_container.update_idletasks()

    tk.Label(settings_container, text="Settings Container", bg=style_out['bg_color']).grid(row=0, column=0, sticky="ew")

    set_dark_style(style, parent_frame, [settings_container, vertical_container, horizontal_container])
    
    #size = style_out['font_size'] - 2
    #set_default_font(parent_frame, font_name=style_out['font_family'], size=size)

    return parent_frame, vertical_container, horizontal_container, settings_container

def download_hug_dataset(q, vars_dict):
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