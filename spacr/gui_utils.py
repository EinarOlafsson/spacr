import io, sys, ast, ctypes, re, csv, ast
import tkinter as tk
from tkinter import ttk

from . gui_core import initiate_root
from .gui_elements import spacrLabel, spacrCheckbutton, set_dark_style

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

def proceed_with_app(root, app_name, app_func):
    from .app_annotate import gui_annotate
    from .app_make_masks import gui_make_masks
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
        gui_annotate()
    elif app_name == "Make Masks":
        gui_make_masks()
    else:
        raise ValueError(f"Invalid app name: {app_name}")

def load_app(root, app_name, app_func):
    # Cancel all scheduled after tasks
    if hasattr(root, 'after_tasks'):
        for task in root.after_tasks:
            root.after_cancel(task)
    root.after_tasks = []

    # Exit functionality only for the annotation app
    if app_name != "Annotate" and hasattr(root, 'current_app_exit_func'):
        root.next_app_func = proceed_with_app
        root.next_app_args = (app_name, app_func)  # Ensure correct arguments
        root.current_app_exit_func()
    else:
        proceed_with_app(root, app_name, app_func)

def parse_list(value):
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise ValueError("Invalid format for list")
    
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
