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
    
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    
    db_path = os.path.join(settings['src'], 'measurements/measurements.db')

    app = AnnotateApp(root,
                      db_path=db_path,
                      src=settings['src'],
                      image_type=settings['image_type'],
                      channels=settings['channels'],
                      image_size=settings['img_size'],
                      annotation_column=settings['annotation_column'],
                      normalize=settings['normalize'],
                      percentiles=settings['percentiles'],
                      measurement=settings['measurement'],
                      threshold=settings['threshold'],
                      normalize_channels=settings['normalize_channels'])
    
    #next_button = tk.Button(root, text="Next", command=app.next_page)
    #next_button.grid(row=app.grid_rows, column=app.grid_cols - 1)
    #back_button = tk.Button(root, text="Back", command=app.previous_page)
    #back_button.grid(row=app.grid_rows, column=app.grid_cols - 2)
    #exit_button = tk.Button(root, text="Exit", command=app.shutdown)
    #exit_button.grid(row=app.grid_rows, column=app.grid_cols - 3)
    
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
    settings['normalize_channels'] = settings['channels'].split(',')
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

    app = AnnotateApp(root, db, src, image_type=settings['image_type'], channels=settings['channels'], image_size=settings['img_size'], annotation_column=settings['annotation_column'], normalize=settings['normalize'], percentiles=settings['percentiles'], measurement=settings['measurement'], threshold=settings['threshold'], normalize_channels=settings['normalize_channels'])

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
        #'cell_mask_dim': ('combo', chans_v3, None),
        #'cell_chann_dim': ('combo', chans_v3, None),
        #'nucleus_mask_dim': ('combo', chans_v3, None),
        #'nucleus_chann_dim': ('combo', chans_v3, None),
        #'pathogen_mask_dim': ('combo', chans_v3, None),
        #'pathogen_chann_dim': ('combo', chans_v3, None),
        'crop_mode': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']"], "['cell']"),
         #'magnification': ('combo', [20, 40, 60], 20),
        #'nucleus_channel': ('combo', chans_v3, None),
        #'cell_channel': ('combo', chans_v3, None),
        #'channel_of_interest': ('combo', chans_v3, None),
        #'pathogen_channel': ('combo', chans_v3, None),
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
    
    from .core import generate_image_umap, preprocess_generate_masks
    from .cellpose import identify_masks_finetune, check_cellpose_models, compare_cellpose_masks
    from .submodules import analyze_recruitment
    from .ml import generate_ml_scores, perform_regression
    from .submodules import train_cellpose, analyze_plaques
    from .io import process_non_tif_non_2D_images, generate_cellpose_train_test, generate_dataset
    from .measure import measure_crop
    from .sim import run_multiple_simulations
    from .deep_spacr import deep_spacr, apply_model_to_tar
    from .sequencing import generate_barecode_mapping
    
    process_stdout_stderr(q)
    
    print(f'run_function_gui settings_type: {settings_type}')
    
    if settings_type == 'mask':
        function = preprocess_generate_masks
        imports = 1
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
    elif settings_type == 'umap':
        function = generate_image_umap
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
    """Recursively print the widget structure."""
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