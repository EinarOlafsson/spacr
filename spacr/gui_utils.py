import os, spacr, traceback, io, sys, ast, ctypes, matplotlib, re, csv, requests, ast
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog
from tkinter import Checkbutton
from tkinter import font as tkFont
from multiprocessing import Process, Value, Queue, Manager, set_start_method
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import requests
from huggingface_hub import list_repo_files, hf_hub_download

from .logger import log_function_call
from .settings import set_default_train_test_model, get_measure_crop_settings, set_default_settings_preprocess_generate_masks, get_analyze_reads_default_settings, set_default_umap_image_settings

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

# Define global variables
q = None
console_output = None
parent_frame = None
vars_dict = None
canvas = None
canvas_widget = None
scrollable_frame = None
progress_label = None
fig_queue = None

thread_control = {"run_thread": None, "stop_requested": False}

class spacrCheckbutton(ttk.Checkbutton):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.configure(text=self.text, variable=self.variable, command=self.command, style='Spacr.TCheckbutton')

class spacrFrame(ttk.Frame):
    def __init__(self, container, width=None, *args, bg='black', **kwargs):
        super().__init__(container, *args, **kwargs)
        self.configure(style='TFrame')
        if width is None:
            screen_width = self.winfo_screenwidth()
            width = screen_width // 4
        canvas = tk.Canvas(self, bg=bg, width=width)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas, style='TFrame')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        
        for child in self.scrollable_frame.winfo_children():
            child.configure(bg='black')

class spacrLabel(tk.Frame):
    def __init__(self, parent, text="", font=None, style=None, align="right", **kwargs):
        label_kwargs = {k: v for k, v in kwargs.items() if k in ['foreground', 'background', 'font', 'anchor', 'justify', 'wraplength']}
        for key in label_kwargs.keys():
            kwargs.pop(key)
        super().__init__(parent, **kwargs)
        self.text = text
        self.kwargs = label_kwargs
        self.align = align
        screen_height = self.winfo_screenheight()
        label_height = screen_height // 50
        label_width = label_height * 10
        self.canvas = tk.Canvas(self, width=label_width, height=label_height, highlightthickness=0, bg=self.kwargs.get("background", "black"))
        self.canvas.grid(row=0, column=0, sticky="ew")

        self.font_style = font if font else tkFont.Font(family=self.kwargs.get("font_family", "Helvetica"), size=self.kwargs.get("font_size", 12), weight=tkFont.NORMAL)
        self.style = style

        if self.align == "center":
            anchor_value = tk.CENTER
            text_anchor = 'center'
        else:  # default to right alignment
            anchor_value = tk.E
            text_anchor = 'e'

        if self.style:
            ttk_style = ttk.Style()
            ttk_style.configure(self.style, **label_kwargs)
            self.label_text = ttk.Label(self.canvas, text=self.text, style=self.style, anchor=text_anchor, justify=text_anchor)
            self.label_text.pack(fill=tk.BOTH, expand=True)
        else:
            self.label_text = self.canvas.create_text(label_width // 2 if self.align == "center" else label_width - 5, 
                                                      label_height // 2, text=self.text, fill=self.kwargs.get("foreground", "white"), 
                                                      font=self.font_style, anchor=anchor_value, justify=tk.RIGHT)

    def set_text(self, text):
        if self.style:
            self.label_text.config(text=text)
        else:
            self.canvas.itemconfig(self.label_text, text=text)

class spacrButton(tk.Frame):
    def __init__(self, parent, text="", command=None, font=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.command = command
        screen_height = self.winfo_screenheight()
        button_height = screen_height // 50
        button_width = button_height * 3

        # Increase the canvas size to accommodate the button and the rim
        self.canvas = tk.Canvas(self, width=button_width + 4, height=button_height + 4, highlightthickness=0, bg="black")
        self.canvas.grid(row=0, column=0)

        self.button_bg = self.create_rounded_rectangle(2, 2, button_width + 2, button_height + 2, radius=20, fill="#000000", outline="#ffffff")

        self.font_style = font if font else tkFont.Font(family="Helvetica", size=12, weight=tkFont.NORMAL)
        self.button_text = self.canvas.create_text((button_width + 4) // 2, (button_height + 4) // 2, text=self.text, fill="white", font=self.font_style)

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

    def on_enter(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill="#008080")  # Teal color

    def on_leave(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill="#000000")  # Black color

    def on_click(self, event=None):
        if self.command:
            self.command()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)


class spacrSwitch(ttk.Frame):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.canvas = tk.Canvas(self, width=40, height=20, highlightthickness=0, bd=0, bg="black")
        self.canvas.grid(row=0, column=1, padx=(10, 0))
        self.switch_bg = self.create_rounded_rectangle(2, 2, 38, 18, radius=9, outline="", fill="#fff")
        self.switch = self.canvas.create_oval(4, 4, 16, 16, outline="", fill="#800080")  # Purple initially
        self.label = spacrLabel(self, text=self.text, background="black", foreground="white")
        self.label.grid(row=0, column=0, padx=(0, 10))
        self.bind("<Button-1>", self.toggle)
        self.canvas.bind("<Button-1>", self.toggle)
        self.label.bind("<Button-1>", self.toggle)
        self.update_switch()

    def toggle(self, event=None):
        self.variable.set(not self.variable.get())
        self.animate_switch()
        if self.command:
            self.command()

    def update_switch(self):
        if self.variable.get():
            self.canvas.itemconfig(self.switch, fill="#008080")  # Teal
            self.canvas.coords(self.switch, 24, 4, 36, 16)  # Move switch to the right
        else:
            self.canvas.itemconfig(self.switch, fill="#800080")  # Purple
            self.canvas.coords(self.switch, 4, 4, 16, 16)  # Move switch to the left

    def animate_switch(self):
        if self.variable.get():
            start_x, end_x = 4, 24
            final_color = "#008080"  # Teal
        else:
            start_x, end_x = 24, 4
            final_color = "#800080"  # Purple

        self.animate_movement(start_x, end_x, final_color)

    def animate_movement(self, start_x, end_x, final_color):
        step = 1 if start_x < end_x else -1
        for i in range(start_x, end_x, step):
            self.canvas.coords(self.switch, i, 4, i + 12, 16)
            self.canvas.update()
            self.after(10)  # Small delay for smooth animation
        self.canvas.itemconfig(self.switch, fill=final_color)

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)
        self.update_switch()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=9, **kwargs):  # Smaller radius for smaller switch
        points = [x1 + radius, y1,
                  x1 + radius, y1,
                  x2 - radius, y1,
                  x2 - radius, y1,
                  x2, y1,
                  x2, y1 + radius,
                  x2, y1 + radius,
                  x2, y2 - radius,
                  x2, y2 - radius,
                  x2, y2,
                  x2 - radius, y2,
                  x2 - radius, y2,
                  x1 + radius, y2,
                  x1 + radius, y2,
                  x1, y2,
                  x1, y2 - radius,
                  x1, y2 - radius,
                  x1, y1 + radius,
                  x1, y1 + radius,
                  x1, y1]

        return self.canvas.create_polygon(points, **kwargs, smooth=True)

class spacrToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x = event.x_root + 20
        y = event.y_root + 10
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        self.tooltip_window.configure(bg='black')
        label = tk.Label(self.tooltip_window, text=self.text, background="#333333", foreground="white", relief='flat', borderwidth=0)
        label.grid(row=0, column=0, padx=5, pady=5)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

def initiate_abort():
    global thread_control
    if thread_control.get("stop_requested") is not None:
        thread_control["stop_requested"].value = 1

    if thread_control.get("run_thread") is not None:
        thread_control["run_thread"].join(timeout=5)
        if thread_control["run_thread"].is_alive():
            thread_control["run_thread"].terminate()
        thread_control["run_thread"] = None

def start_process(q, fig_queue, settings_type='mask'):
    global thread_control, vars_dict
    from .settings import check_settings

    settings = check_settings(vars_dict)
    if thread_control.get("run_thread") is not None:
        initiate_abort()
    stop_requested = Value('i', 0)  # multiprocessing shared value for inter-process communication
    thread_control["stop_requested"] = stop_requested
    if settings_type == 'mask':
        thread_control["run_thread"] = Process(target=run_mask_gui, args=(settings, q, fig_queue, stop_requested))
    elif settings_type == 'measure':
        thread_control["run_thread"] = Process(target=run_measure_gui, args=(settings, q, fig_queue, stop_requested))
    elif settings_type == 'classify':
        thread_control["run_thread"] = Process(target=run_classify_gui, args=(settings, q, fig_queue, stop_requested))
    elif settings_type == 'sequencing':
        thread_control["run_thread"] = Process(target=run_sequencing_gui, args=(settings, q, fig_queue, stop_requested))
    elif settings_type == 'umap':
        thread_control["run_thread"] = Process(target=run_umap_gui, args=(settings, q, fig_queue, stop_requested))
    thread_control["run_thread"].start()

def import_settings(settings_type='mask'):
    global vars_dict, scrollable_frame
    from .settings import generate_fields
    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    csv_settings = read_settings_from_csv(csv_file_path)
    if settings_type == 'mask':
        settings = set_default_settings_preprocess_generate_masks(src='path', settings={})
    elif settings_type == 'measure':
        settings = get_measure_crop_settings(settings={})
    elif settings_type == 'classify':
        settings = set_default_train_test_model(settings={})
    elif settings_type == 'sequencing':
        settings = get_analyze_reads_default_settings(settings={})
    elif settings_type == 'umap':
        settings = set_default_umap_image_settings(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")
    
    variables = convert_settings_dict_for_gui(settings)
    new_settings = update_settings_from_csv(variables, csv_settings)
    vars_dict = generate_fields(new_settings, scrollable_frame)

def set_dark_style(style):
    font_style = tkFont.Font(family="Helvetica", size=24)
    
    # Entry
    style.configure('TEntry', padding='5 5 5 5', borderwidth=1, relief='solid', fieldbackground='black', foreground='#ffffff', font=font_style)
    
    # Combobox
    style.configure('TCombobox', fieldbackground='black', background='black', foreground='#ffffff', font=font_style)
    style.map('TCombobox', fieldbackground=[('readonly', 'black')], foreground=[('readonly', '#ffffff')])
    
    # Custom Button
    style.configure('Custom.TButton', background='black', foreground='white', bordercolor='white', focusthickness=3, focuscolor='white', font=('Helvetica', 12))
    style.map('Custom.TButton', background=[('active', 'teal'), ('!active', 'black')], foreground=[('active', 'white'), ('!active', 'white')], bordercolor=[('active', 'white'), ('!active', 'white')])
    
    # Custom Label
    style.configure('Custom.TLabel', padding='5 5 5 5', borderwidth=1, relief='flat', background='black', foreground='#ffffff', font=font_style)
    
    # Checkbutton
    style.configure('Spacr.TCheckbutton', background='black', foreground='#ffffff', indicatoron=False, relief='flat', font="15")
    style.map('Spacr.TCheckbutton', background=[('selected', 'black'), ('active', 'black')], foreground=[('selected', '#ffffff'), ('active', '#ffffff')])

    # General Label
    style.configure('TLabel', background='black', foreground='#ffffff', font=font_style)
    
    # Frame
    style.configure('TFrame', background='black')
    
    # PanedWindow
    style.configure('TPanedwindow', background='black')
    
    # Notebook
    style.configure('TNotebook', background='black', tabmargins=[2, 5, 2, 0])
    style.configure('TNotebook.Tab', background='black', foreground='#ffffff', padding=[5, 5], font=font_style)
    style.map('TNotebook.Tab', background=[('selected', '#555555'), ('active', '#555555')], foreground=[('selected', '#ffffff'), ('active', '#ffffff')])
    
    # Button (regular)
    style.configure('TButton', background='black', foreground='#ffffff', padding='5 5 5 5', font=font_style)
    style.map('TButton', background=[('active', '#555555'), ('disabled', '#333333')])
    
    # Scrollbar
    style.configure('Vertical.TScrollbar', background='black', troughcolor='black', bordercolor='black')
    style.configure('Horizontal.TScrollbar', background='black', troughcolor='black', bordercolor='black')
    
    # LabelFrame
    style.configure('Custom.TLabelFrame', font=('Helvetica', 10, 'bold'), background='black', foreground='white', relief='solid', borderwidth=1)
    style.configure('Custom.TLabelFrame.Label', background='black', foreground='white', font=('Helvetica', 10, 'bold'))

def set_default_font(root, font_name="Helvetica", size=12):
    default_font = (font_name, size)
    root.option_add("*Font", default_font)
    root.option_add("*TButton.Font", default_font)
    root.option_add("*TLabel.Font", default_font)
    root.option_add("*TEntry.Font", default_font)

def create_menu_bar(root):
    from .app_annotate import initiate_annotation_app_root
    from .app_make_masks import initiate_mask_app_root

    gui_apps = {
        "Mask": 'mask',
        "Measure": 'measure',
        "Annotate": initiate_annotation_app_root,
        "Make Masks": initiate_mask_app_root,
        "Classify": 'classify',
        "Sequencing": 'sequencing',
        "Umap": 'umap'
    }

    def load_app_wrapper(app_name, app_func):
        load_app(root, app_name, app_func)

    # Create the menu bar
    menu_bar = tk.Menu(root, bg="#008080", fg="white")
    # Create a "SpaCr Applications" menu
    app_menu = tk.Menu(menu_bar, tearoff=0, bg="#008080", fg="white")
    menu_bar.add_cascade(label="SpaCr Applications", menu=app_menu)
    # Add options to the "SpaCr Applications" menu
    for app_name, app_func in gui_apps.items():
        app_menu.add_command(label=app_name, command=lambda app_name=app_name, app_func=app_func: load_app_wrapper(app_name, app_func))
    # Add a separator and an exit option
    app_menu.add_separator()
    app_menu.add_command(label="Exit", command=root.quit)
    # Configure the menu for the root window
    root.config(menu=menu_bar)

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

def read_settings_from_csv(csv_file_path):
    settings = {}
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['Key']
            value = row['Value']
            settings[key] = value
    return settings

def update_settings_from_csv(variables, csv_settings):
    new_settings = variables.copy()  # Start with a copy of the original settings
    for key, value in csv_settings.items():
        if key in new_settings:
            # Get the variable type and options from the original settings
            var_type, options, _ = new_settings[key]
            # Update the default value with the CSV value, keeping the type and options unchanged
            new_settings[key] = (var_type, options, value)
    return new_settings

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

def clear_canvas(canvas):
    # Clear each plot (axes) in the figure
    for ax in canvas.figure.get_axes():
        ax.clear()

    # Redraw the now empty canvas without changing its size
    canvas.draw_idle()

def my_show():
    """
    Replacement for plt.show() that queues figures instead of displaying them.
    """
    fig = plt.gcf()
    fig_queue.put(fig)
    plt.close(fig)
    
def measure_crop_wrapper(settings, q, fig_queue):
    """
    Wraps the measure_crop function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the measure_crop function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = my_show

    try:
        print('start')
        spacr.measure.measure_crop(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show  
        
def preprocess_generate_masks_wrapper(settings, q, fig_queue):
    """
    Wraps the measure_crop function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the measure_crop function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = my_show

    try:
        spacr.core.preprocess_generate_masks(src=settings['src'], settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show

def sequencing_wrapper(settings, q, fig_queue):

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = my_show

    try:
        spacr.sequencing.analyze_reads(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show

def umap_wrapper(settings, q, fig_queue):

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = my_show

    try:
        spacr.core.generate_image_umap(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show

def train_test_model_wrapper(settings, q, fig_queue):
    """
    Wraps the measure_crop function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the measure_crop function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    def my_show():
        """
        Replacement for plt.show() that queues figures instead of displaying them.
        """
        fig = plt.gcf()
        fig_queue.put(fig)  # Queue the figure for GUI display
        plt.close(fig)  # Prevent the figure from being shown by plt.show()

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = my_show

    try:
        spacr.core.train_test_model(settings['src'], settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)  # Send the error message to the GUI via the queue
        traceback.print_exc()
    finally:
        plt.show = original_show  # Restore the original plt.show function
        
        
def run_multiple_simulations_wrapper(settings, q, fig_queue):
    """
    Wraps the run_multiple_simulations function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the run_multiple_simulations function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    def my_show():
        """
        Replacement for plt.show() that queues figures instead of displaying them.
        """
        fig = plt.gcf()
        fig_queue.put(fig)  # Queue the figure for GUI display
        plt.close(fig)  # Prevent the figure from being shown by plt.show()

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = my_show

    try:
        spacr.sim.run_multiple_simulations(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)  # Send the error message to the GUI via the queue
        traceback.print_exc()
    finally:
        plt.show = original_show  # Restore the original plt.show function

def convert_settings_dict_for_gui(settings):
    variables = {}
    special_cases = {
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'nikon', 'zeis', 'custom'], 'cellvoyager'),
        'channels': ('combo', ['[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]'], '[0,1,2,3]'),
        'cell_mask_dim': ('combo', ['0', '1', '2', '3', '4', '5', '6', '7', '8', None], None),
        'nucleus_mask_dim': ('combo', ['0', '1', '2', '3', '4', '5', '6', '7', '8', None], None),
        'pathogen_mask_dim': ('combo', ['0', '1', '2', '3', '4', '5', '6', '7', '8', None], None),
        #'crop_mode': ('combo', ['cell', 'nucleus', 'pathogen', '[cell, nucleus, pathogen]', '[cell,nucleus, pathogen]'], ['cell']),
        'magnification': ('combo', [20, 40, 60], 20),
        'nucleus_channel': ('combo', [0, 1, 2, 3, None], None),
        'cell_channel': ('combo', [0, 1, 2, 3, None], None),
        'pathogen_channel': ('combo', [0, 1, 2, 3, None], None),
        'timelapse_mode': ('combo', ['trackpy', 'btrack'], 'trackpy'),
        'timelapse_objects': ('combo', ['cell', 'nucleus', 'pathogen', 'cytoplasm', None], None),
        'model_type': ('combo', ['resnet50', 'other_model'], 'resnet50'),
        'optimizer_type': ('combo', ['adamw', 'adam'], 'adamw'),
        'schedule': ('combo', ['reduce_lr_on_plateau', 'step_lr'], 'reduce_lr_on_plateau'),
        'loss_type': ('combo', ['focal_loss', 'binary_cross_entropy_with_logits'], 'focal_loss'),
        'normalize_by': ('combo', ['fov', 'png'], 'png'),
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

def setup_frame(parent_frame):
    style = ttk.Style(parent_frame)
    set_dark_style(style)
    set_default_font(parent_frame, font_name="Helvetica", size=8)
    parent_frame.configure(bg='black')
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)
    vertical_container = tk.PanedWindow(parent_frame, orient=tk.VERTICAL, bg='black')
    vertical_container.grid(row=0, column=0, sticky=tk.NSEW)
    horizontal_container = tk.PanedWindow(vertical_container, orient=tk.HORIZONTAL, bg='black')
    vertical_container.add(horizontal_container, stretch="always")
    horizontal_container.grid_columnconfigure(0, weight=1)
    horizontal_container.grid_columnconfigure(1, weight=1)
    settings_frame = tk.Frame(horizontal_container, bg='black')
    settings_frame.grid_rowconfigure(0, weight=0)
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)
    horizontal_container.add(settings_frame, stretch="always", sticky="nsew")
    return parent_frame, vertical_container, horizontal_container

def setup_settings_panel(vertical_container, settings_type='mask', frame_height=500, frame_width=1000):
    global vars_dict, scrollable_frame
    from .settings import set_default_settings_preprocess_generate_masks, get_measure_crop_settings, set_default_train_test_model, get_analyze_reads_default_settings, set_default_umap_image_settings, generate_fields

    print("Setting up settings panel")
    
    # Create settings frame
    settings_frame = tk.Frame(vertical_container, bg='black', height=frame_height, width=frame_width)
    vertical_container.add(settings_frame, stretch="always")

    # Add settings label
    settings_label = spacrLabel(settings_frame, text="Settings", background="black", foreground="white", anchor='center', justify='center', align="center")
    settings_label.grid(row=0, column=0, pady=10, padx=10)

    # Create a spacrFrame inside the settings_frame
    scrollable_frame = spacrFrame(settings_frame, bg='black', width=frame_width)
    scrollable_frame.grid(row=1, column=0, sticky="nsew")

    # Configure the weights for resizing
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)

    # Load settings based on type
    if settings_type == 'mask':
        settings = set_default_settings_preprocess_generate_masks(src='path', settings={})
    elif settings_type == 'measure':
        settings = get_measure_crop_settings(settings={})
    elif settings_type == 'classify':
        settings = set_default_train_test_model(settings={})
    elif settings_type == 'sequencing':
        settings = get_analyze_reads_default_settings(settings={})
    elif settings_type == 'umap':
        settings = set_default_umap_image_settings(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")

    # Generate fields for settings
    variables = convert_settings_dict_for_gui(settings)
    vars_dict = generate_fields(variables, scrollable_frame)
    print("Settings panel setup complete")
    return scrollable_frame, vars_dict

def setup_plot_section(vertical_container):
    global canvas, canvas_widget
    plot_frame = tk.PanedWindow(vertical_container, orient=tk.VERTICAL)
    vertical_container.add(plot_frame, stretch="always")
    figure = Figure(figsize=(30, 4), dpi=100, facecolor='black')
    plot = figure.add_subplot(111)
    plot.plot([], [])  # This creates an empty plot.
    plot.axis('off')
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().configure(cursor='arrow', background='black', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    plot_frame.add(canvas_widget, stretch="always")
    canvas.draw()
    canvas.figure = figure
    return canvas, canvas_widget

def setup_console(vertical_container):
    global console_output
    print("Setting up console frame")
    console_frame = tk.Frame(vertical_container, bg='black')
    vertical_container.add(console_frame, stretch="always")
    console_label = spacrLabel(console_frame, text="Console", background="black", foreground="white", anchor='center', justify='center', align="center")
    console_label.grid(row=0, column=0, pady=10, padx=10)
    console_output = scrolledtext.ScrolledText(console_frame, height=10, bg='black', fg='white', insertbackground='white')
    console_output.grid(row=1, column=0, sticky="nsew")
    console_frame.grid_rowconfigure(1, weight=1)
    console_frame.grid_columnconfigure(0, weight=1)
    print("Console setup complete")
    return console_output

def setup_progress_frame(vertical_container):
    global progress_output
    progress_frame = tk.Frame(vertical_container, bg='black')
    vertical_container.add(progress_frame, stretch="always")
    label_frame = tk.Frame(progress_frame, bg='black')
    label_frame.grid(row=0, column=0, sticky="ew", pady=(5, 0), padx=10)
    progress_label = spacrLabel(label_frame, text="Processing: 0%", background="black", foreground="white", font=('Helvetica', 12), anchor='w', justify='left', align="left")
    progress_label.grid(row=0, column=0, sticky="w")
    progress_output = scrolledtext.ScrolledText(progress_frame, height=10, bg='black', fg='white', insertbackground='white')
    progress_output.grid(row=1, column=0, sticky="nsew")
    progress_frame.grid_rowconfigure(1, weight=1)
    progress_frame.grid_columnconfigure(0, weight=1)
    print("Progress frame setup complete")
    return progress_output

def download_hug_dataset():
    global vars_dict, q
    repo_id = "einarolafsson/toxo_mito"
    subfolder = "plate1"
    local_dir = os.path.join(os.path.expanduser("~"), "datasets")  # Set to the home directory
    try:
        local_path = download_dataset(repo_id, subfolder, local_dir)
        if 'src' in vars_dict:
            vars_dict['src'][2].set(local_path)  # Assuming vars_dict['src'] is a tuple and the 3rd element is a StringVar
            q.put(f"Set source path to: {vars_dict['src'][2].get()}\n")
        q.put(f"Dataset downloaded to: {local_path}\n")
    except Exception as e:
        q.put(f"Failed to download dataset: {e}\n")

def download_dataset(repo_id, subfolder, local_dir=None, retries=5, delay=5):
    global q
    """
    Downloads a dataset from Hugging Face and returns the local path.

    Args:
        repo_id (str): The repository ID (e.g., 'einarolafsson/toxo_mito').
        subfolder (str): The subfolder path within the repository (e.g., 'plate1').
        local_dir (str): The local directory where the dataset will be saved. Defaults to the user's home directory.
        retries (int): Number of retry attempts in case of failure.
        delay (int): Delay in seconds between retries.

    Returns:
        str: The local path to the downloaded dataset.
    """
    if local_dir is None:
        local_dir = os.path.join(os.path.expanduser("~"), "datasets")
    
    local_subfolder_dir = os.path.join(local_dir, subfolder)
    if not os.path.exists(local_subfolder_dir):
        os.makedirs(local_subfolder_dir)
    elif len(os.listdir(local_subfolder_dir)) == 40:
        q.put(f"Dataset already downloaded to: {local_subfolder_dir}")
        return local_subfolder_dir

    attempt = 0
    while attempt < retries:
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            subfolder_files = [file for file in files if file.startswith(subfolder)]

            for file_name in subfolder_files:
                for attempt in range(retries):
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
            q.put(f"Error downloading dataset: {e}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)
    
    raise Exception("Failed to download dataset after multiple attempts.")

def setup_button_section(horizontal_container, settings_type='mask', settings_row=5, run=True, abort=True, download=True, import_btn=True):
    global button_frame, run_button, abort_button, download_dataset_button, import_button, q, fig_queue, vars_dict

    button_frame = tk.Frame(horizontal_container, bg='black')
    horizontal_container.add(button_frame, stretch="always", sticky="nsew")
    button_frame.grid_rowconfigure(0, weight=0)
    button_frame.grid_rowconfigure(1, weight=1)
    button_frame.grid_columnconfigure(0, weight=1)

    categories_label = spacrLabel(button_frame, text="Categories", background="black", foreground="white", font=('Helvetica', 12), anchor='center', justify='center', align="center")  # Increase font size
    categories_label.grid(row=0, column=0, pady=10, padx=10)

    button_scrollable_frame = spacrFrame(button_frame, bg='black')
    button_scrollable_frame.grid(row=1, column=0, sticky="nsew")

    btn_col = 0
    btn_row = 1
    
    if run:
        run_button = spacrButton(button_scrollable_frame.scrollable_frame, text="Run", command=lambda: start_process(q, fig_queue, settings_type), font=('Helvetica', 12))
        run_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        btn_row += 1

    if abort and settings_type in ['mask', 'measure', 'classify', 'sequencing', 'umap']:
        abort_button = spacrButton(button_scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort, font=('Helvetica', 12))
        abort_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        btn_row += 1

    if download and settings_type in ['mask']:
        download_dataset_button = spacrButton(button_scrollable_frame.scrollable_frame, text="Download", command=download_hug_dataset, font=('Helvetica', 12))
        download_dataset_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        btn_row += 1

    if import_btn:
        import_button = spacrButton(button_scrollable_frame.scrollable_frame, text="Import", command=lambda: import_settings(settings_row, settings_type), font=('Helvetica', 12))
        import_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')

    # Call toggle_settings after vars_dict is initialized
    if vars_dict is not None:
        toggle_settings(button_scrollable_frame)
    return button_scrollable_frame

def toggle_test_mode():
    global vars_dict, test_mode_button
    current_state = vars_dict['test_mode'][2].get()
    new_state = not current_state
    vars_dict['test_mode'][2].set(new_state)
    if new_state:
        test_mode_button.config(bg="blue")
    else:
        test_mode_button.config(bg="gray")

def toggle_settings(button_scrollable_frame):
    global vars_dict
    from .settings import categories

    if vars_dict is None:
        raise ValueError("vars_dict is not initialized.")

    def toggle_category(settings, var):
        for setting in settings:
            if setting in vars_dict:
                label, widget, _ = vars_dict[setting]
                if var.get() == 0:
                    label.grid_remove()
                    widget.grid_remove()
                else:
                    label.grid()
                    widget.grid()

    row = 1
    col = 2 
    category_idx = 0

    for category, settings in categories.items():
        if any(setting in vars_dict for setting in settings):
            category_var = tk.IntVar(value=0)
            vars_dict[category] = (None, None, category_var)
            toggle = spacrCheckbutton(
                button_scrollable_frame.scrollable_frame, 
                text=category, 
                variable=category_var, 
                command=lambda cat=settings, var=category_var: toggle_category(cat, var)
            )
            # Directly set the style
            style = ttk.Style()
            font_style = tkFont.Font(family="Helvetica", size=12, weight="bold")
            style.configure('Spacr.TCheckbutton', font=font_style, background='black', foreground='#ffffff', indicatoron=False, relief='flat')
            style.map('Spacr.TCheckbutton', background=[('selected', 'black'), ('active', 'black')], foreground=[('selected', '#ffffff'), ('active', '#ffffff')])
            toggle.configure(style='Spacr.TCheckbutton')
            toggle.grid(row=row, column=col, sticky="w", pady=2, padx=2)
            col += 1
            category_idx += 1

            if category_idx % 4 == 0:  
                row += 1
                col = 2

    for settings in categories.values():
        for setting in settings:
            if setting in vars_dict:
                label, widget, _ = vars_dict[setting]
                label.grid_remove()
                widget.grid_remove()

def process_fig_queue():
    global canvas, fig_queue, canvas_widget, parent_frame
    try:
        while not fig_queue.empty():
            clear_canvas(canvas)
            fig = fig_queue.get_nowait()
            for ax in fig.get_axes():
                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.xaxis.set_visible(False)  # Hide the x-axis
                ax.yaxis.set_visible(False)  # Hide the y-axis
            fig.tight_layout()
            fig.set_facecolor('black')
            canvas.figure = fig
            fig_width, fig_height = canvas_widget.winfo_width(), canvas_widget.winfo_height()
            fig.set_size_inches(fig_width / fig.dpi, fig_height / fig.dpi, forward=True)
            canvas.draw_idle()
    except Exception as e:
        traceback.print_exc()
    finally:
        after_id = canvas_widget.after(100, process_fig_queue)
        parent_frame.after_tasks.append(after_id)

def process_console_queue():
    global q, console_output, parent_frame
    while not q.empty():
        message = q.get_nowait()
        console_output.insert(tk.END, message)
        console_output.see(tk.END)
    after_id = console_output.after(100, process_console_queue)
    parent_frame.after_tasks.append(after_id)

def run_mask_gui(settings, q, fig_queue, stop_requested):
    process_stdout_stderr(q)
    try:
        preprocess_generate_masks_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_sequencing_gui(settings, q, fig_queue, stop_requested):
    process_stdout_stderr(q)
    try:
        sequencing_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_umap_gui(settings, q, fig_queue, stop_requested):
    process_stdout_stderr(q)
    try:
        umap_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_measure_gui(settings, q, fig_queue, stop_requested):
    process_stdout_stderr(q)
    try:
        settings['input_folder'] = settings['src']
        measure_crop_wrapper(settings=settings, q=q, fig_queue=fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_classify_gui(settings, q, fig_queue, stop_requested):
    process_stdout_stderr(q)
    try:
        train_test_model_wrapper(settings['src'], settings)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def set_globals(q_var, console_output_var, parent_frame_var, vars_dict_var, canvas_var, canvas_widget_var, scrollable_frame_var, progress_label_var, fig_queue_var):
    global q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, progress_label, fig_queue
    q = q_var
    console_output = console_output_var
    parent_frame = parent_frame_var
    vars_dict = vars_dict_var
    canvas = canvas_var
    canvas_widget = canvas_widget_var
    scrollable_frame = scrollable_frame_var
    progress_label = progress_label_var
    fig_queue = fig_queue_var

def initiate_root(parent, settings_type='mask'):
    global q, fig_queue, parent_frame, scrollable_frame, button_frame, vars_dict, canvas, canvas_widget, progress_label, button_scrollable_frame
    print("Initializing root with settings_type:", settings_type)
    parent_frame = parent

    if not hasattr(parent_frame, 'after_tasks'):
        parent_frame.after_tasks = []

    for widget in parent_frame.winfo_children():
        if widget.winfo_exists():
            try:
                widget.destroy()
            except tk.TclError as e:
                print(f"Error destroying widget: {e}")

    q = Queue()
    fig_queue = Queue()
    parent_frame, vertical_container, horizontal_container = setup_frame(parent_frame)
    scrollable_frame, vars_dict = setup_settings_panel(horizontal_container, settings_type)  # Adjust height and width as needed
    button_scrollable_frame = setup_button_section(horizontal_container, settings_type)
    canvas, canvas_widget = setup_plot_section(vertical_container)
    console_output = setup_console(vertical_container)

    if settings_type in ['mask', 'measure', 'classify', 'sequencing']:
        progress_output = setup_progress_frame(vertical_container)
    else:
        progress_output = None

    set_globals(q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, progress_label, fig_queue)
    process_console_queue()
    process_fig_queue()
    after_id = parent_frame.after(100, lambda: main_thread_update_function(parent_frame, q, fig_queue, canvas_widget, progress_label))
    parent_frame.after_tasks.append(after_id)
    print("Root initialization complete")
    return parent_frame, vars_dict

def cancel_after_tasks(frame):
    if hasattr(frame, 'after_tasks'):
        for task in frame.after_tasks:
            frame.after_cancel(task)
        frame.after_tasks.clear()

def start_gui_app(settings_type='mask'):
    global q, fig_queue, parent_frame, scrollable_frame, vars_dict, canvas, canvas_widget, progress_label
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry(f"{width}x{height}")
    root.title(f"SpaCr: {settings_type.capitalize()}")
    root.content_frame = tk.Frame(root)
    print("Starting GUI app with settings_type:", settings_type)
    initiate_root(root.content_frame, settings_type)
    create_menu_bar(root)
    root.mainloop()



