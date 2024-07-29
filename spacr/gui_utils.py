import os, io, sys, ast, ctypes, re, ast, sqlite3
import tkinter as tk
from tkinter import ttk

from . gui_core import initiate_root
from .gui_elements import spacrLabel, spacrCheckbutton, AnnotateApp

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

def set_default_font(root, font_name="Helvetica", size=12):
    default_font = (font_name, size)
    root.option_add("*Font", default_font)
    root.option_add("*TButton.Font", default_font)
    root.option_add("*TLabel.Font", default_font)
    root.option_add("*TEntry.Font", default_font)

def proceed_with_app_v1(root, app_name, app_func):
    from .gui import gui_app

    # Clear the current content frame
    if hasattr(root, 'content_frame'):
        for widget in root.content_frame.winfo_children():
            try:
                widget.destroy()
            except tk.TclError as e:
                print(f"Error destroying widget: {e}")
    else:
        root.content_frame = tk.Frame(root)
        root.content_frame.grid(row=1, column=0, sticky="nsew")
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

    # Initialize the new app in the content frame
    if app_name == "Mask":
        initiate_root(root.content_frame, 'mask')
    elif app_name == "Measure":
        initiate_root(root.content_frame, 'measure')
    elif app_name == "Classify":
        initiate_root(root.content_frame, 'classify')
    elif app_name == "Sequencing":
        initiate_root(root.content_frame, 'sequencing')
    elif app_name == "Umap":
        initiate_root(root.content_frame, 'umap')
    elif app_name == "Annotate":
        initiate_root(root.content_frame, 'annotate')
    elif app_name == "Make Masks":
        initiate_root(root.content_frame, 'make_masks')
    else:
        raise ValueError(f"Invalid app name: {app_name}")
    
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
    # Cancel all scheduled after tasks
    if hasattr(root, 'after_tasks'):
        for task in root.after_tasks:
            root.after_cancel(task)
    root.after_tasks = []

    # Exit functionality only for the annotation and make_masks apps
    if app_name not in ["Annotate", "make_masks"] and hasattr(root, 'current_app_exit_func'):
        root.next_app_func = proceed_with_app
        root.next_app_args = (app_name, app_func)  # Ensure correct arguments
        root.current_app_exit_func()
    else:
        proceed_with_app(root, app_name, app_func)

def parse_list_v1(value):
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise ValueError("Invalid format for list")
    
def parse_list(value):
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
        else:
            raise ValueError(f"Expected a list but got {type(parsed_value).__name__}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid format for list: {value}. Error: {e}")
    
def create_input_field(frame, label_text, row, var_type='entry', options=None, default_value=None):
    label_column = 0
    widget_column = 1

    # Configure the column widths
    frame.grid_columnconfigure(label_column, weight=0)  # Allow the label column to expand
    frame.grid_columnconfigure(widget_column, weight=1)  # Allow the widget column to expand

    # Right-align the label text and the label itself
    label = spacrLabel(frame, text=label_text, background="black", foreground="white", anchor='e', justify='right')
    label.grid(column=label_column, row=row, sticky=tk.E, padx=(5, 2), pady=5)  # Align label to the right

    if var_type == 'entry':
        var = tk.StringVar(value=default_value)  # Set default value
        entry = ttk.Entry(frame, textvariable=var, style='TEntry')  # Apply TEntry style for entries
        entry.grid(column=widget_column, row=row, sticky=tk.W, padx=(2, 5), pady=5)  # Align widget to the left
        return (label, entry, var)  # Return both the label and the entry, and the variable
    elif var_type == 'check':
        var = tk.BooleanVar(value=default_value)  # Set default value (True/False)
        check = spacrCheckbutton(frame, text="", variable=var, style='TCheckbutton')
        check.grid(column=widget_column, row=row, sticky=tk.W, padx=(2, 5), pady=5)  # Align widget to the left
        return (label, check, var)  # Return both the label and the checkbutton, and the variable
    elif var_type == 'combo':
        var = tk.StringVar(value=default_value)  # Set default value
        combo = ttk.Combobox(frame, textvariable=var, values=options, style='TCombobox')  # Apply TCombobox style
        combo.grid(column=widget_column, row=row, sticky=tk.W, padx=(2, 5), pady=5)  # Align widget to the left
        if default_value:
            combo.set(default_value)
        return (label, combo, var)  # Return both the label and the combobox, and the variable
    else:
        var = None  # Placeholder in case of an undefined var_type
        return (label, None, var)

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
        self.q.put(msg)

    def flush(self):
        pass

def cancel_after_tasks(frame):
    if hasattr(frame, 'after_tasks'):
        for task in frame.after_tasks:
            frame.after_cancel(task)
        frame.after_tasks.clear()

def main_thread_update_function(root, q, fig_queue, canvas_widget, progress_label):
    try:
        ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        while not q.empty():
            message = q.get_nowait()
            clean_message = ansi_escape_pattern.sub('', message)
            if clean_message.startswith("Progress"):
                progress_label.config(text=clean_message)
            if clean_message.startswith("\rProgress"):
                progress_label.config(text=clean_message)
            elif clean_message.startswith("Successfully"):
                progress_label.config(text=clean_message)
            elif clean_message.startswith("Processing"):
                progress_label.config(text=clean_message)
            elif clean_message.startswith("scale"):
                pass
            elif clean_message.startswith("plot_cropped_arrays"):
                pass
            elif clean_message == "" or clean_message == "\r" or clean_message.strip() == "":
                pass
            else:
                print(clean_message)
    except Exception as e:
        print(f"Error updating GUI canvas: {e}")
    finally:
        root.after(100, lambda: main_thread_update_function(root, q, fig_queue, canvas_widget, progress_label))

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
    vars_dict = {}
    settings = set_annotate_default_settings(settings={})
    
    for setting in settings:
        vars_dict[setting] = {
            'entry': ttk.Entry(frame),
            'value': settings[setting]
        }

    # Arrange input fields and labels
    for row, (name, data) in enumerate(vars_dict.items()):
        ttk.Label(frame, text=f"{name.replace('_', ' ').capitalize()}:",
                  background="black", foreground="white").grid(row=row, column=0)
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

