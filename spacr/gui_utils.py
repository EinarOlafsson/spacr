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

def attach_dependency_listeners(vars_dict, categories, category_dependencies, category_group_dependencies):
    """Wire up show/hide dependencies between boolean settings and categories.

    Registers trace callbacks so toggling a boolean widget hides or shows every
    dependent category, supporting both 1:1 dependencies and any-of group
    dependencies. Initial visibility is applied immediately.

    :param vars_dict: mapping ``key -> (label, widget, var, frame)``.
    :param categories: mapping of category name to the list of settings it owns.
    :param category_dependencies: mapping ``bool_key -> [categories to toggle]``.
    :param category_group_dependencies: mapping ``category -> [bool_keys that any-enable it]``.
    :returns: None.
    """

    def _get_entry(setting):
        entry = vars_dict.get(setting)
        if entry is None:
            return None
        if not isinstance(entry, (tuple, list)):
            return None
        if len(entry) < 4:
            return None
        if any(item is None for item in entry[:4]):
            return None
        return entry

    def _set_category_visibility(category_name, visible):
        if category_name not in categories:
            return

        for setting in categories[category_name]:
            entry = _get_entry(setting)
            if entry is None:
                continue

            label, widget, _, frame = entry

            if visible:
                label.grid()
                widget.grid()
                frame.grid()
            else:
                label.grid_remove()
                widget.grid_remove()
                frame.grid_remove()

    def _is_truthy(tk_var):
        val = tk_var.get()
        if isinstance(val, bool):
            return val
        return str(val).lower() in ('1', 'true')

    # --- Simple 1:1 dependencies ---
    def _make_simple_callback(bool_key):
        def _on_change(*args):
            entry = _get_entry(bool_key)
            if entry is None:
                return

            _, _, tk_var, _ = entry
            is_on = _is_truthy(tk_var)

            for cat_name in category_dependencies.get(bool_key, []):
                _set_category_visibility(cat_name, is_on)

        return _on_change

    for bool_key in category_dependencies:
        entry = _get_entry(bool_key)
        if entry is None:
            continue

        cb = _make_simple_callback(bool_key)
        cb()  # set initial state
        entry[2].trace_add('write', cb)

    # --- Group (any-of) dependencies ---
    def _make_group_callback(cat_name, bool_keys):
        def _on_change(*args):
            visible = any(
                _is_truthy(entry[2])
                for k in bool_keys
                for entry in [_get_entry(k)]
                if entry is not None
            )
            _set_category_visibility(cat_name, visible)

        return _on_change

    for cat_name, bool_keys in category_group_dependencies.items():
        cb = _make_group_callback(cat_name, bool_keys)
        cb()  # set initial state

        for k in bool_keys:
            entry = _get_entry(k)
            if entry is None:
                continue
            entry[2].trace_add('write', cb)


def initialize_cuda():
    """Initialize CUDA in the main process by performing a trivial GPU op.

    :returns: None.
    """
    if torch.cuda.is_available():
        # Allocate a small tensor on the GPU
        _ = torch.tensor([0.0], device='cuda')
        print("CUDA initialized in the main process.")
    else:
        print("CUDA is not available.")

def set_high_priority(process):
    """Raise the OS scheduling priority of a subprocess.

    Uses ``HIGH_PRIORITY_CLASS`` on Windows and ``nice(-10)`` on Unix-like systems.
    Failures are logged but never raised.

    :param process: a ``multiprocessing.Process`` (or object exposing ``.pid``).
    :returns: None.
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
    """Pin a subprocess to all available CPU cores on Linux.

    No-op on non-Linux platforms.

    :param process: a ``multiprocessing.Process`` (or object exposing ``.pid``).
    :returns: None.
    """
    import platform
    if platform.system() == 'Linux':
        p = psutil.Process(process.pid)
        p.cpu_affinity(list(range(os.cpu_count())))
    
def proceed_with_app(root, app_name, app_func):
    """Replace ``root.content_frame`` contents with a new app.

    :param root: the Tk root that owns ``content_frame``.
    :param app_name: display name of the app (currently unused, kept for logging/hooks).
    :param app_func: callable invoked with ``root.content_frame`` to build the new app.
    :returns: None.
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
    """Tear down the current spacr app and load another in its place.

    Cancels pending ``after`` tasks and defers the swap to the current app's
    exit hook when one is registered (annotation/make_masks apps swap
    immediately since they own the root themselves).

    :param root: the Tk root.
    :param app_name: name of the app to load.
    :param app_func: callable invoked with ``root.content_frame`` to build it.
    :returns: None.
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
    """Parse a string literal into a homogeneous list of scalars.

    Accepts Python-list or tuple literals and rejects mixed-type contents.
    Single-element tuples are returned as one-element lists.

    :param value: string representation of a list or tuple.
    :returns: parsed list containing only ints, floats, or strings.
    :raises ValueError: if the string is not a valid literal or contains
        mixed / unsupported types.
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
    """Create a labeled settings input widget on ``frame`` at ``row``.

    Supports entry, checkbox and combo variants and coerces ``default_value``
    to the widget's expected type; unrecognised ``var_type`` returns a bare
    label with no widget.

    :param frame: parent frame that hosts the row.
    :param label_text: raw settings key; underscores are replaced with spaces
        and the first letter capitalised for display.
    :param row: grid row inside ``frame`` to occupy.
    :param var_type: one of ``'entry'``, ``'check'``, ``'combo'``.
    :param options: list of choices used when ``var_type='combo'``.
    :param default_value: initial value; falls back to a type-appropriate default.
    :returns: tuple ``(label, widget, tk_var, container_frame)``.
    """
    from .gui_elements import set_dark_style, set_element_size
    
    style_out = set_dark_style(ttk.Style())
    font_loader = style_out['font_loader']
    font_size = style_out['font_size']
    size_dict = set_element_size()
    size_dict['settings_width'] = size_dict['settings_width'] - int(size_dict['settings_width'] * 0.1)

    label_text = label_text.replace('_', ' ').capitalize()

    frame.grid_columnconfigure(0, weight=1)

    custom_frame = tk.Frame(frame, bg=style_out['bg_color'], bd=0, relief='flat')
    custom_frame.grid(column=0, row=row, sticky=tk.EW, padx=(5, 5), pady=5)
    custom_frame.grid_columnconfigure(0, weight=1)

    label = tk.Label(custom_frame, text=label_text, bg=style_out['bg_color'], fg=style_out['fg_color'],
                     font=font_loader.get_font(size=font_size), anchor='w', justify='left')
    label.grid(column=0, row=0, sticky=tk.W, padx=(5, 2), pady=5)

    try:
        if var_type == 'entry':
            if default_value is None:
                default_value = ''
            else:
                default_value = str(default_value)
            var = tk.StringVar(value=default_value)
            entry = spacrEntry(custom_frame, textvariable=var, outline=False, width=size_dict['settings_width'])
            entry.grid(column=0, row=1, sticky=tk.EW, padx=(2, 5), pady=5)
            return (label, entry, var, custom_frame)

        elif var_type == 'check':
            if isinstance(default_value, str):
                default_value = default_value.lower() in ('true', '1', 'yes')
            elif default_value is None:
                default_value = False
            else:
                default_value = bool(default_value)
            var = tk.BooleanVar(value=default_value)
            check = spacrCheck(custom_frame, text="", variable=var)
            check.grid(column=0, row=1, sticky=tk.W, padx=(2, 5), pady=5)
            return (label, check, var, custom_frame)

        elif var_type == 'combo':
            if default_value is None or default_value == '':
                default_value = options[0] if options else ''
            else:
                default_str = str(default_value).replace(' ', '')
                options_stripped = [str(o).replace(' ', '') for o in options] if options else []
                if options and default_str not in options_stripped:
                    print(f"Warning: '{label_text}' value '{default_value}' not in options {options}, using first option.")
                    default_value = options[0] if options else ''
            default_value = str(default_value)
            var = tk.StringVar(value=default_value)
            combo = spacrCombo(custom_frame, textvariable=var, values=options, width=size_dict['settings_width'])
            combo.grid(column=0, row=1, sticky=tk.EW, padx=(2, 5), pady=5)
            combo.set(default_value)
            return (label, combo, var, custom_frame)
        else:
            var = None
            return (label, None, var, custom_frame)
    except Exception as e:
        print(f"Error creating input field: {e}")
        print(f"Wrong type for {label_text} Expected {var_type}")

def process_stdout_stderr(q):
    """Redirect ``sys.stdout`` and ``sys.stderr`` writes into a queue.

    :param q: queue receiving each written message.
    :returns: None.
    """
    sys.stdout = WriteToQueue(q)
    sys.stderr = WriteToQueue(q)

class WriteToQueue(io.TextIOBase):
    """File-like sink that forwards writes into a queue.

    Used to reroute ``stdout``/``stderr`` into the GUI console.

    :param q: queue receiving each non-empty write.
    """
    def __init__(self, q):
        """Store the target queue."""
        self.q = q
    def write(self, msg):
        """Forward a non-empty message to the queue."""
        if msg.strip():  # Avoid empty messages
            self.q.put(msg)
    def flush(self):
        """No-op required by the file-like interface."""
        pass

def cancel_after_tasks(frame):
    """Cancel every scheduled Tk ``after`` task tracked on ``frame``.

    :param frame: Tk widget with an ``after_tasks`` attribute.
    :returns: None.
    """
    if hasattr(frame, 'after_tasks'):
        for task in frame.after_tasks:
            frame.after_cancel(task)
        frame.after_tasks.clear()

def annotate(settings):
    """Launch the standalone annotation UI on a measurements database.

    Ensures the requested annotation column exists in the ``png_list`` table,
    then opens ``AnnotateApp`` in its own Tk root and blocks on the mainloop.

    :param settings: annotation settings dict (see ``set_annotate_default_settings``).
    :returns: None.
    """
    from .settings import set_annotate_default_settings
    settings = set_annotate_default_settings(settings)
    src  = settings['src']

    db = os.path.join(src, 'measurements/measurements.db')
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    
    if settings['annotation_column'] not in [col[1] for col in cols]:
        
        try:
            c.execute(f"ALTER TABLE png_list ADD COLUMN {settings['annotation_column']} integer")
        except sqlite3.OperationalError:
            pass  # column already exists
        
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
                      threshold_direction=settings['threshold_direction'],
                      normalize_channels=settings['normalize_channels'])
    
    app.load_images()
    root.mainloop()

def generate_annotate_fields(frame):
    """Build labelled entry widgets for the annotation-settings defaults.

    :param frame: parent Tk frame that hosts the field grid.
    :returns: mapping ``key -> {'entry': ttk.Entry, 'value': default}``.
    """
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
        tk.Label(
            frame,
            text=f"{name.replace('_', ' ').capitalize()}:",
            bg=style_out['bg_color'],
            fg=style_out['fg_color'],
            font=font_loader.get_font(size=font_size)
        ).grid(row=row, column=0)

        value = data['value']
        if isinstance(value, list):
            string_value = ','.join(map(str, value))
        elif isinstance(value, (int, float, bool)):
            string_value = str(value)
        elif value is None:
            string_value = ''
        else:
            string_value = value

        data['entry'].insert(0, string_value)
        data['entry'].grid(row=row, column=1)

    return vars_dict

def run_annotate_app(vars_dict, parent_frame):
    """Collect the annotation-fields values, coerce types, and start the annotator.

    Clears ``parent_frame`` of existing widgets before launching the app.

    :param vars_dict: widget map produced by :func:`generate_annotate_fields`.
    :param parent_frame: Tk frame that hosts the annotation UI.
    :returns: None.
    """
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
    """Start the annotation app inside an existing GUI frame.

    :param parent_frame: Tk frame whose toplevel hosts the annotator.
    :param settings: annotation settings dict.
    :returns: None.
    """
    global global_image_refs
    global_image_refs.clear()
    root = parent_frame.winfo_toplevel()
    annotate_with_image_refs(settings, root, lambda: load_next_app(root))

def load_next_app(root):
    """Invoke the queued next-app callback, reinitialising root if it was destroyed.

    :param root: current Tk root; expected to hold ``next_app_func`` and ``next_app_args``.
    :returns: None.
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

def annotate_with_image_refs(settings, root, shutdown_callback):
    """Start ``AnnotateApp`` inside an existing root with a shutdown chain.

    Ensures the annotation column exists in the ``png_list`` table, sizes the
    root to the full screen, and registers an exit hook that runs
    ``shutdown_callback`` after the app closes.

    :param settings: annotation settings dict.
    :param root: existing Tk root to reuse.
    :param shutdown_callback: callable invoked after the annotator shuts down.
    :returns: None.
    """
    from .settings import set_annotate_default_settings

    settings = set_annotate_default_settings(settings)
    src = settings['src']

    db = os.path.join(src, 'measurements/measurements.db')
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('PRAGMA table_info(png_list)')
    cols = c.fetchall()
    if settings['annotation_column'] not in [col[1] for col in cols]:
        try:
            c.execute(f"ALTER TABLE png_list ADD COLUMN {settings['annotation_column']} integer")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")

    app = AnnotateApp(root, db, src, image_type=settings['image_type'], channels=settings['channels'], image_size=settings['img_size'], annotation_column=settings['annotation_column'], percentiles=settings['percentiles'], measurement=settings['measurement'], threshold=settings['threshold'], threshold_direction=settings['threshold_direction'], normalize_channels=settings['normalize_channels'], outline=settings['outline'], outline_threshold_factor=settings['outline_threshold_factor'], outline_sigma=settings['outline_sigma'])

    # Set the canvas background to black
    root.configure(bg='black')

    # Store the shutdown function and next app details in the root
    root.current_app_exit_func = lambda: [app.shutdown(), shutdown_callback()]

    # Call load_images after setting up the root window
    app.load_images()


# Curated torchvision classification models for the `model_type` combo. Kept
# static so opening a settings screen never triggers a slow `import torchvision`
# (see convert_settings_dict_for_gui). The pipeline validates/instantiates the
# real model by name at train time.
_TORCHVISION_MODELS_CURATED = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'densenet121', 'densenet169', 'densenet201',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
    'maxvit_t', 'regnet_y_400mf', 'regnet_y_1_6gf', 'regnet_y_8gf',
    'squeezenet1_0', 'squeezenet1_1', 'alexnet', 'googlenet', 'inception_v3',
]


def _torchvision_model_names():
    """Return model names for the combo WITHOUT importing torchvision. If
    torchvision is already loaded (e.g. after a training run) use its full zoo;
    otherwise fall back to the curated static list."""
    import sys
    mods = sys.modules.get("torchvision.models")
    if mods is not None:
        try:
            names = [n for n, o in mods.__dict__.items()
                     if callable(o) and not n.startswith("_")]
            if names:
                return sorted(set(names) | set(_TORCHVISION_MODELS_CURATED))
        except Exception:
            pass
    return list(_TORCHVISION_MODELS_CURATED)


def convert_settings_dict_for_gui(settings):
    """Convert a plain settings dict into the GUI variable spec.

    Maps each key to a ``(widget_type, options, default_value)`` triple, using
    combo boxes for keys with known enumerated options and inferring
    check/entry widgets otherwise.

    :param settings: mapping of setting names to default values.
    :returns: mapping ``key -> (var_type, options, default_value)`` ready for
        :func:`create_input_field`.
    """
    # NOTE: we deliberately do NOT `import torchvision` here. Enumerating the
    # torchvision model zoo pulls in torch + torchvision, a ~5 s import that
    # made every FIRST module open sluggish. The classify pipeline still
    # instantiates the real torchvision model by name at train time — the GUI
    # combo just needs a list of valid names, so we use a curated static list
    # (if torchvision happens to be imported already we extend it with the full
    # zoo, for free).
    torchvision_models = _torchvision_model_names()
    chan_list = ['[0,1,2,3,4,5,6,7,8]','[0,1,2,3,4,5,6,7]','[0,1,2,3,4,5,6]','[0,1,2,3,4,5]','[0,1,2,3,4]','[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]', '[0,0]']
    
    variables = {}
    special_cases = {
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'auto', 'custom'], 'cellvoyager'),
        'channels': ('combo', chan_list, '[0,1,2,3]'),
        'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
        'channel_dims': ('combo', chan_list, '[0,1,2,3]'),
        'dataset_mode': ('combo', ['annotation', 'metadata', 'recruitment'], 'metadata'),
        'cov_type': ('combo', ['HC0', 'HC1', 'HC2', 'HC3', None], None),
        'crop_mode': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['organelle']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['cell', 'organelle']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen', 'organelle']"], "['cell']"),
        'timelapse_mode': ('combo', ['trackpy', 'iou', 'btrack'], 'trackpy'),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
        'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
        'model_name': ('combo', ['cyto', 'cyto_2', 'cyto_3', 'nuclei'], 'cyto'),
        'regression_type': ('combo', ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge'], 'ols'),
        'timelapse_objects': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['organelle']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['cell', 'organelle']", "['nucleus', 'pathogen']", "['nucleus', 'organelle']", "['cell', 'nucleus', 'pathogen']", "['cell', 'nucleus', 'organelle']", "['cell', 'nucleus', 'pathogen', 'organelle']"], "['cell']"),
        'model_type': ('combo', torchvision_models, 'resnet50'),
        'model_type_ml': ('combo', ['xgboost', 'random_forest', 'logistic_regression', 'gradient_boosting'], 'xgboost'),
        'optimizer_type': ('combo', ['adamw', 'adam'], 'adamw'),
        'schedule': ('combo', ['cosine','reduce_lr_on_plateau', 'step_lr'], 'cosine'),
        'loss_type': ('combo', ['focal_loss', 'binary_cross_entropy_with_logits'], 'focal_loss'),
        'normalize_by': ('combo', ['fov', 'png'], 'png'),
        'agg_type': ('combo', ['mean', 'median'], 'mean'),
        'grouping': ('combo', ['mean', 'median'], 'mean'),
        'min_max': ('combo', ['allq', 'all'], 'allq'),
        'transform': ('combo', ['log', 'sqrt', 'square', None], None),
        'organelle_morphology': ('combo', ['spots', 'network', 'irregular', 'ring'], 'spots'),
        'organelle_method': ('combo', ['otsu', 'adaptive', 'log', 'dog', 'ridge', 'hysteresis', 'cellpose', 'unet'], 'otsu'),
        'organelle_model_name': ('combo', ['cyto', 'cyto2', 'cyto3', 'nuclei'], 'cyto3'),
        'organelle_ridge_filter': ('combo', ['frangi', 'sato', 'meijering'], 'frangi'),
        'organelle_network_threshold': ('combo', ['otsu', 'adaptive'], 'otsu'),
        'organelle_ring_fill_method': ('combo', ['flood', 'convex'], 'flood'),
        'summarize_organelles_by': ('combo', ["['cell']","['nucleus']","['pathogen']","['cytoplasm']","['cell', 'nucleus']","['cell', 'pathogen']","['cell', 'cytoplasm']","['cell', 'nucleus', 'pathogen']","['cell', 'nucleus', 'pathogen', 'cytoplasm']",None], None)
        
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
    """Route matplotlib figures into a queue instead of displaying them.

    Drop-in replacement for ``plt.show()`` used while spacr runs inside the GUI
    process; falls back to ``fig.show()`` when no queue is provided.

    :param fig_queue: queue that receives the current figure, or None.
    :returns: None.
    """
    fig = plt.gcf()
    if fig_queue:
        fig_queue.put(fig)
    else:
        fig.show()
    plt.close(fig)

def function_gui_wrapper(function=None, settings=None, q=None, fig_queue=None, imports=1):
    """Run a spacr worker function with GUI-safe stdout, error and figure routing.

    Temporarily replaces ``plt.show`` with :func:`spacrFigShow` so any figures
    are shipped to ``fig_queue`` instead of blocking, and forwards exception
    text to ``q``.

    :param function: worker callable to invoke.
    :param settings: settings dict passed to ``function``.
    :param q: queue for log/error messages sent to the GUI.
    :param fig_queue: queue for matplotlib figures produced during the run.
    :param imports: 1 to call ``function(settings=...)``; 2 to call
        ``function(src=settings['src'], settings=...)``.
    :returns: None.
    """

    # Temporarily override plt.show
    if settings is None:
        settings = {}
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
    """Dispatch a spacr module by ``settings_type`` and run it in the worker.

    Redirects stdout/stderr into ``q``, invokes the mapped module via
    :func:`function_gui_wrapper`, and sets ``stop_requested`` on completion so
    the GUI can reap the process.

    :param settings_type: identifier that selects the target spacr function.
    :param settings: settings dict passed through to the worker.
    :param q: queue for log/error messages.
    :param fig_queue: queue for matplotlib figures.
    :param stop_requested: shared ``multiprocessing.Value('i')`` flipped to 1 on exit.
    :returns: None.
    :raises ValueError: if ``settings_type`` is not a recognised module.
    """
    from .core import generate_image_umap, preprocess_generate_masks
    from .spacr_cellpose import identify_masks_finetune, check_cellpose_models, compare_cellpose_masks
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

def hide_all_settings(vars_dict, categories=None):
    """Hide every widget that belongs to any known category.

    Used to collapse all optional-category settings until their triggering
    boolean is toggled on.

    :param vars_dict: mapping ``key -> (label, widget, var, frame)``.
    :param categories: category-to-settings map; if None, ``vars_dict`` is returned unchanged.
    :returns: the (mutated) ``vars_dict``.
    """
    if categories is None:
        return vars_dict
    for cat_name, settings in categories.items():
        for setting in settings:
            if setting in vars_dict and vars_dict[setting] is not None:
                label, widget, _, frame = vars_dict[setting]
                label.grid_remove()
                widget.grid_remove()
                frame.grid_remove()
    return vars_dict


def setup_frame(parent_frame):
    """Build the settings/plot/console panel layout inside ``parent_frame``.

    Creates the horizontal-split PanedWindow, a vertical container for figures
    and a horizontal container for buttons, and applies the dark theme.

    :param parent_frame: Tk frame that will host the layout.
    :returns: tuple ``(parent_frame, vertical_container, horizontal_container, settings_container)``.
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
    """Download the demo dataset and settings pack from Hugging Face.

    Also updates ``vars_dict['src']`` with the downloaded dataset path so the
    settings panel points at it. Progress and errors are reported through ``q``.

    :param q: queue used for status/error messages.
    :param vars_dict: settings widget map; the ``'src'`` entry is updated if present.
    :returns: None.
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
    """Download a Hugging Face dataset subfolder (or CSVs) to a local directory.

    Skips the download if the target directory already contains files, and
    retries transient HTTP errors per-file and per-listing.

    :param q: queue used for progress/error messages.
    :param repo_id: HF dataset repo id (e.g. ``'einarolafsson/toxo_mito'``).
    :param subfolder: subfolder within the repo; empty string downloads top-level CSVs.
    :param local_dir: destination directory; defaults to ``~/datasets``.
    :param retries: number of retry attempts for both listing and each file.
    :param delay: delay in seconds between retries.
    :returns: path to the local directory containing the downloaded files.
    :raises Exception: if downloads fail after all retry attempts.
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
    """Ensure ``frame.after_tasks`` exists so scheduled callbacks can be tracked.

    :param frame: Tk widget to annotate.
    :returns: None.
    """
    if not hasattr(frame, 'after_tasks'):
        frame.after_tasks = []

def display_gif_in_plot_frame(gif_path, parent_frame):
    """Loop a GIF in ``parent_frame``, cover-cropped and cached per frame size.

    :param gif_path: filesystem path to the GIF.
    :param parent_frame: Tk frame that hosts the animation.
    :returns: None.
    """
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
    """Loop an MP4/AVI/GIF in ``parent_frame``, cover-cropped to fill it.

    :param media_path: path to the media file; extension picks the decoder.
    :param parent_frame: Tk frame that hosts the playback.
    :returns: None.
    :raises ValueError: for unsupported file extensions.
    """
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
    """Print the Tk widget tree rooted at ``widget`` for debugging.

    :param widget: root widget to descend from.
    :param indent: current indent depth (spaces) used by recursive calls.
    :returns: None.
    """
    # Print the widget's name and class
    print(" " * indent + f"{widget}: {widget.winfo_class()}")
    
    # Recursively print all child widgets
    for child_name, child_widget in widget.children.items():
        print_widget_structure(child_widget, indent + 2)

def get_screen_dimensions():
    """Return the pixel dimensions of the primary monitor.

    :returns: tuple ``(screen_width, screen_height)`` in pixels.
    """
    monitor = get_monitors()[0]  # Get the primary monitor
    screen_width = monitor.width
    screen_height = monitor.height
    return screen_width, screen_height

def convert_to_number(value):
    """Convert a string to ``int`` when possible, otherwise to ``float``.

    :param value: string representation of a number.
    :returns: parsed number as ``int`` (preferred) or ``float``.
    :raises ValueError: if the string is neither.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Unable to convert '{value}' to an integer or float.")