import os, spacr, inspect, traceback, io, sys, ast, ctypes, matplotlib, re, csv, requests, ast
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
from tkinter import filedialog
from tkinter import Checkbutton
from tkinter import font as tkFont
from torchvision import models

from multiprocessing import Process, Value, Queue, Manager, set_start_method
import multiprocessing as mp
#mp.set_start_method('spawn', force=True)


from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import requests
from requests.exceptions import HTTPError, Timeout
from huggingface_hub import list_repo_files, hf_hub_download

from .logger import log_function_call
from .settings import set_default_train_test_model, get_measure_crop_settings, set_default_settings_preprocess_generate_masks

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

# Set multiprocessing start method to 'spawn'
set_start_method('spawn', force=True)

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

def set_dark_style(style):
    font_style = tkFont.Font(family="Helvetica", size=10)
    style.configure('TEntry', padding='5 5 5 5', borderwidth=1, relief='solid', fieldbackground='black', foreground='#ffffff', font=font_style)  # Entry
    style.configure('TCombobox', fieldbackground='black', background='black', foreground='#ffffff', font=font_style)  # Combobox
    style.configure('Custom.TButton', background='black', foreground='white', bordercolor='white', focusthickness=3, focuscolor='white', font=('Helvetica', 12))
    style.map('Custom.TButton',
              background=[('active', 'teal'), ('!active', 'black')],
              foreground=[('active', 'white'), ('!active', 'white')],
              bordercolor=[('active', 'white'), ('!active', 'white')])
    style.configure('Custom.TLabel', padding='5 5 5 5', borderwidth=1, relief='flat', background='black', foreground='#ffffff', font=font_style)  # Custom Label
    style.configure('TCheckbutton', background='black', foreground='#ffffff', indicatoron=False, relief='flat', font=font_style)  # Checkbutton
    style.map('TCheckbutton', background=[('selected', '#555555'), ('active', '#555555')])
    style.configure('TLabel', background='black', foreground='#ffffff', font=font_style)  # Label
    style.configure('TFrame', background='black')  # Frame
    style.configure('TPanedwindow', background='black')  # PanedWindow
    style.configure('TNotebook', background='black', tabmargins=[2, 5, 2, 0])  # Notebook
    style.configure('TNotebook.Tab', background='black', foreground='#ffffff', padding=[5, 5], font=font_style)
    style.map('TNotebook.Tab', background=[('selected', '#555555'), ('active', '#555555')])
    style.configure('TButton', background='black', foreground='#ffffff', padding='5 5 5 5', font=font_style)  # Button (regular)
    style.map('TButton', background=[('active', '#555555'), ('disabled', '#333333')])
    style.configure('Vertical.TScrollbar', background='black', troughcolor='black', bordercolor='black')  # Scrollbar
    style.configure('Horizontal.TScrollbar', background='black', troughcolor='black', bordercolor='black')  # Scrollbar

    # Define custom LabelFrame style
    style.configure('Custom.TLabelFrame', font=('Helvetica', 10, 'bold'), background='black', foreground='white', relief='solid', borderwidth=1)
    style.configure('Custom.TLabelFrame.Label', background='black', foreground='white')  # Style for the Label inside LabelFrame
    style.configure('Custom.TLabelFrame.Label', font=('Helvetica', 10, 'bold'))

def set_default_font(root, font_name="Helvetica", size=12):
    default_font = (font_name, size)
    root.option_add("*Font", default_font)
    root.option_add("*TButton.Font", default_font)
    root.option_add("*TLabel.Font", default_font)
    root.option_add("*TEntry.Font", default_font)

class ScrollableFrame(ttk.Frame):
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
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for child in self.scrollable_frame.winfo_children():
            child.configure(bg='black')

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        try:
            if self.text_widget.winfo_exists():
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
        except tk.TclError:
            pass  # Handle or log the error as needed

    def flush(self):
        pass

class CustomButton(tk.Frame):
    def __init__(self, parent, text="", command=None, font=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.command = command

        # Detect screen height and calculate button dimensions
        screen_height = self.winfo_screenheight()
        button_height = screen_height // 50
        button_width = button_height * 3

        self.canvas = tk.Canvas(self, width=button_width, height=button_height, highlightthickness=0, bg="black")
        self.canvas.grid(row=0, column=0)

        self.button_bg = self.create_rounded_rectangle(0, 0, button_width, button_height, radius=20, fill="#800080")

        # Use the passed font or default to Helvetica if not provided
        self.font_style = font if font else tkFont.Font(family="Helvetica", size=12, weight=tkFont.NORMAL)
        self.button_text = self.canvas.create_text(button_width // 2, button_height // 2, text=self.text, fill="white", font=self.font_style)

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

    def on_enter(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill="#993399")

    def on_leave(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill="#800080")

    def on_click(self, event=None):
        if self.command:
            self.command()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
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

class ToggleSwitch(ttk.Frame):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        
        self.canvas = tk.Canvas(self, width=40, height=20, highlightthickness=0, bd=0, bg="black")
        self.canvas.grid(row=0, column=1, padx=(10, 0))
        
        # Background rounded rectangle with smaller dimensions and no outline
        self.switch_bg = self.create_rounded_rectangle(2, 2, 38, 18, radius=9, outline="", fill="#fff")

        # Switch ball with no outline
        self.switch = self.canvas.create_oval(4, 4, 16, 16, outline="", fill="#800080")  # Purple initially
        
        self.label = ttk.Label(self, text=self.text, background="black", foreground="white")
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

class ToolTip:
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
        label = tk.Label(self.tooltip_window, text=self.text, background="yellow", relief='solid', borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

def create_menu_bar(root):
    from .app_annotate import initiate_annotation_app_root
    from .app_make_masks import initiate_mask_app_root

    gui_apps = {
        "Mask": 'mask',
        "Measure": 'measure',
        "Annotate": initiate_annotation_app_root,
        "Make Masks": initiate_mask_app_root,
        "Classify": 'classify'
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
    

def check_settings(vars_dict):
    global q
    settings = {}
    # Define the expected types for each key, including None where applicable
    expected_types = {
        "src": str,
        "metadata_type": str,
        "custom_regex": (str, type(None)),
        "experiment": str,
        "channels": list,
        "magnification": int,
        "nucleus_channel": (int, type(None)),
        "nucleus_background": int,
        "nucleus_Signal_to_noise": float,
        "nucleus_CP_prob": float,
        "nucleus_FT": float,
        "cell_channel": (int, type(None)),
        "cell_background": (int, float),
        "cell_Signal_to_noise": (int, float),
        "cell_CP_prob": (int, float),
        "cell_FT": (int, float),
        "pathogen_channel": (int, type(None)),
        "pathogen_background": (int, float),
        "pathogen_Signal_to_noise": (int, float),
        "pathogen_CP_prob": (int, float),
        "pathogen_FT": (int, float),
        "preprocess": bool,
        "masks": bool,
        "examples_to_plot": int,
        "randomize": bool,
        "batch_size": int,
        "timelapse": bool,
        "timelapse_displacement": int,
        "timelapse_memory": int,
        "timelapse_frame_limits": list,  # This can be a list of lists
        "timelapse_remove_transient": bool,
        "timelapse_mode": str,
        "timelapse_objects": list,
        "fps": int,
        "remove_background": bool,
        "lower_percentile": (int, float),
        "merge_pathogens": bool,
        "normalize_plots": bool,
        "all_to_mip": bool,
        "pick_slice": bool,
        "skip_mode": str,
        "save": bool,
        "plot": bool,
        "workers": int,
        "verbose": bool,
        "input_folder": str,
        "cell_mask_dim": int,
        "cell_min_size": int,
        "cytoplasm_min_size": int,
        "nucleus_mask_dim": int,
        "nucleus_min_size": int,
        "pathogen_mask_dim": int,
        "pathogen_min_size": int,
        "save_png": bool,
        "crop_mode": list,
        "use_bounding_box": bool,
        "png_size": list,  # This can be a list of lists
        "normalize": bool,
        "png_dims": list,
        "normalize_by": str,
        "save_measurements": bool,
        "representative_images": bool,
        "plot_filtration": bool,
        "include_uninfected": bool,
        "dialate_pngs": bool,
        "dialate_png_ratios": list,
        "max_workers": int,
        "cells": list,
        "cell_loc": list,
        "pathogens": list,
        "pathogen_loc": (list, list),  # This can be a list of lists
        "treatments": list,
        "treatment_loc": (list, list),  # This can be a list of lists
        "channel_of_interest": int,
        "compartments": list,
        "measurement": str,
        "nr_imgs": int,
        "um_per_pixel": (int, float),
        # Additional settings based on provided defaults
        "include_noninfected": bool,
        "include_multiinfected": bool,
        "include_multinucleated": bool,
        "filter_min_max": (list, type(None)),
        "channel_dims": list,
        "backgrounds": list,
        "outline_thickness": int,
        "outline_color": str,
        "overlay_chans": list,
        "overlay": bool,
        "normalization_percentiles": list,
        "print_object_number": bool,
        "nr": int,
        "figuresize": int,
        "cmap": str,
        "test_mode": bool,
        "test_images": int,
        "remove_background_cell": bool,
        "remove_background_nucleus": bool,
        "remove_background_pathogen": bool,
        "pathogen_model": (str, type(None)),
        "filter": bool,
        "upscale": bool,
        "upscale_factor": float,
        "adjust_cells": bool,
        "row_limit": int,
        "tables": list,
        "visualize": str,
        "image_nr": int,
        "dot_size": int,
        "n_neighbors": int,
        "min_dist": float,
        "metric": str,
        "eps": float,
        "min_samples": int,
        "filter_by": str,
        "img_zoom": float,
        "plot_by_cluster": bool,
        "plot_cluster_grids": bool,
        "remove_cluster_noise": bool,
        "remove_highly_correlated": bool,
        "log_data": bool,
        "black_background": bool,
        "remove_image_canvas": bool,
        "plot_outlines": bool,
        "plot_points": bool,
        "smooth_lines": bool,
        "clustering": str,
        "exclude": (str, type(None)),
        "col_to_compare": str,
        "pos": str,
        "neg": str,
        "embedding_by_controls": bool,
        "plot_images": bool,
        "reduction_method": str,
        "save_figure": bool,
        "color_by": (str, type(None)),
        "analyze_clusters": bool,
        "resnet_features": bool,
        "test_nr": int,
        "radial_dist": bool,
        "calculate_correlation": bool,
        "manders_thresholds": list,
        "homogeneity": bool,
        "homogeneity_distances": list,
        "save_arrays": bool,
        "cytoplasm": bool,
        "merge_edge_pathogen_cells": bool,
        "cells_per_well": int,
        "pathogen_size_range": list,
        "nucleus_size_range": list,
        "cell_size_range": list,
        "pathogen_intensity_range": list,
        "nucleus_intensity_range": list,
        "cell_intensity_range": list,
        "target_intensity_min": int,
        "model_type": str,
        "heatmap_feature": str,
        "grouping": str,
        "min_max": str,
        "minimum_cell_count": int,
        "n_estimators": int,
        "test_size": float,
        "location_column": str,
        "positive_control": str,
        "negative_control": str,
        "n_repeats": int,
        "top_features": int,
        "remove_low_variance_features": bool,
        "n_jobs": int,
        "classes": list,
        "schedule": str,
        "loss_type": str,
        "image_size": int,
        "epochs": int,
        "val_split": float,
        "train_mode": str,
        "learning_rate": float,
        "weight_decay": float,
        "dropout_rate": float,
        "init_weights": bool,
        "amsgrad": bool,
        "use_checkpoint": bool,
        "gradient_accumulation": bool,
        "gradient_accumulation_steps": int,
        "intermedeate_save": bool,
        "pin_memory": bool,
        "num_workers": int,
        "augment": bool,
        "target": str,
        "cell_types": list,
        "cell_plate_metadata": (list, type(None)),
        "pathogen_types": list,
        "pathogen_plate_metadata": (list, list),  # This can be a list of lists
        "treatment_plate_metadata": (list, list),  # This can be a list of lists
        "metadata_types": list,
        "cell_chann_dim": int,
        "nucleus_chann_dim": int,
        "pathogen_chann_dim": int,
        "plot_nr": int,
        "plot_control": bool,
        "remove_background": bool,
        "target": str,
        "upstream": str,
        "downstream": str,
        "barecode_length_1": int,
        "barecode_length_2": int,
        "chunk_size": int,
        "grna": str,
        "barcodes": str,
        "plate_dict": dict,
        "pc": str,
        "pc_loc": str,
        "nc": str,
        "nc_loc": str,
        "dependent_variable": str,
        "transform": (str, type(None)),
        "agg_type": str,
        "min_cell_count": int,
        "regression_type": str,
        "remove_row_column_effect": bool,
        "alpha": float,
        "fraction_threshold": float,
        "class_1_threshold": (float, type(None)),
        "batch_size": int,
        "CP_prob": float,
        "flow_threshold": float,
        "percentiles": (list, type(None)),
        "circular": bool,
        "invert": bool,
        "diameter": int,
        "grayscale": bool,
        "resize": bool,
        "target_height": (int, type(None)),
        "target_width": (int, type(None)),
        "rescale": bool,
        "resample": bool,
        "model_name": str,
        "Signal_to_noise": int,
        "learning_rate": float,
        "weight_decay": float,
        "batch_size": int,
        "n_epochs": int,
        "from_scratch": bool,
        "width_height": list,
        "resize": bool,
        "gene_weights_csv": str,
        "fraction_threshold": float,
    }

    for key, (label, widget, var) in vars_dict.items():
        if key not in expected_types:
            if key not in ['General', 'Nucleus', 'Cell', 'Pathogen', 'Timelapse', 'Plot', 'Advanced', 'Test', 'Object Image', 'Annotate Data', 'Miscilanious', 'Test', 'Measurements']:
                
                q.put(f"Key {key} not found in expected types.")
                continue

        value = var.get()
        expected_type = expected_types.get(key, str)

        try:
            if key in ["png_size", "pathogen_plate_metadata", "treatment_plate_metadata"]:
                parsed_value = ast.literal_eval(value) if value else None
                if isinstance(parsed_value, list):
                    if all(isinstance(i, list) for i in parsed_value) or all(not isinstance(i, list) for i in parsed_value):
                        settings[key] = parsed_value
                    else:
                        raise ValueError("Invalid format: Mixed list and list of lists")
                else:
                    raise ValueError("Invalid format for list or list of lists")
            elif expected_type == list:
                settings[key] = parse_list(value) if value else None
            elif expected_type == bool:
                settings[key] = value if isinstance(value, bool) else value.lower() in ['true', '1', 't', 'y', 'yes']
            elif expected_type == (int, type(None)):
                settings[key] = int(value) if value else None
            elif expected_type == (float, type(None)):
                settings[key] = float(value) if value else None
            elif expected_type == (int, float):
                settings[key] = float(value) if '.' in value else int(value)
            elif expected_type == (str, type(None)):
                settings[key] = str(value) if value else None
            elif isinstance(expected_type, tuple):
                for typ in expected_type:
                    try:
                        settings[key] = typ(value) if value else None
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError
            else:
                settings[key] = expected_type(value) if value else None
        except (ValueError, SyntaxError):
            expected_type_name = ' or '.join([t.__name__ for t in expected_type]) if isinstance(expected_type, tuple) else expected_type.__name__
            q.put(f"Error: Invalid format for {key}. Expected type: {expected_type_name}.")
            return

    return settings

def create_input_field(frame, label_text, row, var_type='entry', options=None, default_value=None):
    label = ttk.Label(frame, text=label_text, style='Custom.TLabel')  # Apply Custom.TLabel style for labels
    label.grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
    
    if var_type == 'entry':
        var = tk.StringVar(value=default_value)  # Set default value
        entry = ttk.Entry(frame, textvariable=var, style='TEntry')  # Apply TEntry style for entries
        entry.grid(column=1, row=row, sticky=tk.EW, padx=5)
        return (label, entry, var)  # Return both the label and the entry, and the variable
    elif var_type == 'check':
        var = tk.BooleanVar(value=default_value)  # Set default value (True/False)
        check = Checkbutton(frame, text="", variable=var)
        check.grid(column=1, row=row, sticky=tk.W, padx=5)
        return (label, check, var)  # Return both the label and the checkbutton, and the variable
    elif var_type == 'combo':
        var = tk.StringVar(value=default_value)  # Set default value
        combo = ttk.Combobox(frame, textvariable=var, values=options, style='TCombobox')  # Apply TCombobox style
        combo.grid(column=1, row=row, sticky=tk.EW, padx=5)
        if default_value:
            combo.set(default_value)
        return (label, combo, var)  # Return both the label and the combobox, and the variable
    else:
        var = None  # Placeholder in case of an undefined var_type
        return (label, None, var)
    
def generate_fields(variables, scrollable_frame, row = 5):
    vars_dict = {}
    tooltips = {
        "src": "Path to the folder containing the images.",
        "metadata_type": "Type of metadata to expect in the images. This will determine how the images are processed. If 'custom' is selected, you can provide a custom regex pattern to extract metadata from the image names.",
        "custom_regex": "Custom regex pattern to extract metadata from the image names. This will only be used if 'custom' is selected for 'metadata_type'.",
        "experiment": "Name of the experiment. This will be used to name the output files.",
        "channels": "List of channels to use for the analysis. The first channel is 0, the second is 1, and so on. For example, [0,1,2] will use channels 0, 1, and 2.",
        "magnification": "At what magnification the images were taken. This will be used to determine the size of the objects in the images.",
        "nucleus_channel": "The channel to use for the nucleus. If None, the nucleus will not be segmented.",
        "nucleus_background": "The background intensity for the nucleus channel. This will be used to remove background noise.",
        "nucleus_Signal_to_noise": "The signal-to-noise ratio for the nucleus channel. This will be used to determine the range of intensities to normalize images to for nucleus segmentation.",
        "nucleus_CP_prob": "The cellpose probability threshold for the nucleus channel. This will be used to segment the nucleus.",
        "nucleus_FT": "The flow threshold for nucleus objects. This will be used in nuclues segmentation.",
        "cell_channel": "The channel to use for the cell. If None, the cell will not be segmented.",
        "cell_background": "The background intensity for the cell channel. This will be used to remove background noise.",
        "cell_Signal_to_noise": "The signal-to-noise ratio for the cell channel. This will be used to determine the range of intensities to normalize images to for cell segmentation.",
        "cell_CP_prob": "The cellpose probability threshold for the cell channel. This will be used in cell segmentation.",
        "cell_FT": "The flow threshold for cell objects. This will be used to segment the cells.",
        "pathogen_channel": "The channel to use for the pathogen. If None, the pathogen will not be segmented.",
        "pathogen_background": "The background intensity for the pathogen channel. This will be used to remove background noise.",
        "pathogen_Signal_to_noise": "The signal-to-noise ratio for the pathogen channel. This will be used to determine the range of intensities to normalize images to for pathogen segmentation.",
        "pathogen_CP_prob": "The cellpose probability threshold for the pathogen channel. This will be used to segment the pathogen.",
        "pathogen_FT": "The flow threshold for pathogen objects. This will be used in pathogen segmentation.",
        "preprocess": "Whether to preprocess the images before segmentation. This includes background removal and normalization. Set to False only if this step has already been done.",
        "masks": "Whether to generate masks for the segmented objects. If True, masks will be generated for the nucleus, cell, and pathogen.",
        "examples_to_plot": "The number of images to plot for each segmented object. This will be used to visually inspect the segmentation results and normalization.",
        "randomize": "Whether to randomize the order of the images before processing. Recommended to avoid bias in the segmentation.",
        "batch_size": "The batch size to use for processing the images. This will determine how many images are processed at once. Images are normalized and segmented in batches. Lower if application runs out of RAM or VRAM.",
        "timelapse": "Whether to process the images as a timelapse.",
        "timelapse_displacement": "The displacement between frames in the timelapse. This will be used to align the frames before processing.",
        "timelapse_memory": "The number of frames to in tandem objects must be present in to be considered the same object in the timelapse.",
        "timelapse_frame_limits": "The frame limits to use for the timelapse. This will determine which frames are processed. For example, [5,20] will process frames 5 to 20.",
        "timelapse_remove_transient": "Whether to remove transient objects in the timelapse. Transient objects are present in fewer than all frames.",
        "timelapse_mode": "The mode to use for processing the timelapse. 'trackpy' uses the trackpy library for tracking objects, while 'btrack' uses the btrack library.",
        "timelapse_objects": "The objects to track in the timelapse (cell, nucleus or pathogen). This will determine which objects are tracked over time. If None, all objects will be tracked.",
        "fps": "Frames per second of the automatically generated timelapse movies.",
        "remove_background": "Whether to remove background noise from the images. This will help improve the quality of the segmentation.",
        "lower_percentile": "The lower quantile to use for normalizing the images. This will be used to determine the range of intensities to normalize images to.",
        "merge_pathogens": "Whether to merge pathogen objects that share more than 75% of their perimeter.",
        "normalize_plots": "Whether to normalize the plots.",
        "all_to_mip": "Whether to convert all images to maximum intensity projections before processing.",
        "pick_slice": "Whether to pick a single slice from the z-stack images. If False, the maximum intensity projection will be used.",
        "skip_mode": "The mode to use for skipping images. This will determine how to handle images that cannot be processed.",
        "save": "Whether to save the results to disk.",
        "merge_edge_pathogen_cells": "Whether to merge cells that share pathogen objects.",
        "plot": "Whether to plot the results.",
        "workers": "The number of workers to use for processing the images. This will determine how many images are processed in parallel. Increase to speed up processing.",
        "verbose": "Whether to print verbose output during processing.",
        "input_folder": "Path to the folder containing the images.",
        "cell_mask_dim": "The dimension of the array the cell mask is saved in.",
        "cell_min_size": "The minimum size of cell objects in pixels^2.",
        "cytoplasm": "Whether to segment the cytoplasm (Cell - Nucleus + Pathogen).",
        "cytoplasm_min_size": "The minimum size of cytoplasm objects in pixels^2.",
        "nucleus_mask_dim": "The dimension of the array the nucleus mask is saved in.",
        "nucleus_min_size": "The minimum size of nucleus objects in pixels^2.",
        "pathogen_mask_dim": "The dimension of the array the pathogen mask is saved in.",
        "pathogen_min_size": "The minimum size of pathogen objects in pixels^2.",
        "save_png": "Whether to save the segmented objects as PNG images.",
        "crop_mode": "The mode to use for cropping the images. This will determine which objects are cropped from the images (cell, nucleus, pathogen, cytoplasm).",
        "use_bounding_box": "Whether to use the bounding box of the objects for cropping. If False, only the object itself will be cropped.",
        "png_size": "The size of the PNG images to save. This will determine the size of the saved images.",
        "normalize": "The percentiles to use for normalizing the images. This will be used to determine the range of intensities to normalize images to. If None, no normalization is done.",
        "png_dims": "The dimensions of the PNG images to save. This will determine the dimensions of the saved images. Maximum of 3 dimensions e.g. [1,2,3].",
        "normalize_by": "Whether to normalize the images by field of view (fov) or by PNG image (png).",
        "save_measurements": "Whether to save the measurements to disk.",
        "representative_images": "Whether to save representative images of the segmented objects (Not working yet).",
        "plot_filtration": "Whether to plot the filtration steps.",
        "include_uninfected": "Whether to include uninfected cells in the analysis.",
        "dialate_pngs": "Whether to dilate the PNG images before saving.",
        "dialate_png_ratios": "The ratios to use for dilating the PNG images. This will determine the amount of dilation applied to the images before cropping.",
        "max_workers": "The number of workers to use for processing the images. This will determine how many images are processed in parallel. Increase to speed up processing.",
        "cells": "The cell types to include in the analysis.",
        "cell_loc": "The locations of the cell types in the images.",
        "pathogens": "The pathogen types to include in the analysis.",
        "pathogen_loc": "The locations of the pathogen types in the images.",
        "treatments": "The treatments to include in the analysis.",
        "treatment_loc": "The locations of the treatments in the images.",
        "channel_of_interest": "The channel of interest to use for the analysis.",
        "compartments": "The compartments to measure in the images.",
        "measurement": "The measurement to use for the analysis.",
        "nr_imgs": "The number of images to plot.",
        "um_per_pixel": "The micrometers per pixel for the images."
    }

    for key, (var_type, options, default_value) in variables.items():
        label, widget, var = create_input_field(scrollable_frame.scrollable_frame, key, row, var_type, options, default_value)
        vars_dict[key] = (label, widget, var)  # Store the label, widget, and variable
        
        # Add tooltip to the label if it exists in the tooltips dictionary
        if key in tooltips:
            ToolTip(label, tooltips[key])
        
        row += 1
    return vars_dict

class TextRedirector(object):
    def __init__(self, widget, queue):
        self.widget = widget
        self.queue = queue

    def write(self, str):
        self.queue.put(str)

    def flush(self):
        pass

def create_dark_mode(root, style, console_output):
    dark_bg = 'black'
    light_text = 'white'
    dark_text = 'black'
    input_bg = '#555555'  # Slightly lighter background for input fields
    
    # Configure ttkcompartments('TFrame', background=dark_bg)
    style.configure('TLabel', background=dark_bg, foreground=light_text)
    style.configure('TEntry', fieldbackground=input_bg, foreground=dark_text, background=dark_bg)
    style.configure('TButton', background=dark_bg, foreground=dark_text)
    style.map('TButton', background=[('active', dark_bg)], foreground=[('active', dark_text)])
    style.configure('Dark.TCheckbutton', background=dark_bg, foreground=dark_text)
    style.map('Dark.TCheckbutton', background=[('active', dark_bg)], foreground=[('active', dark_text)])
    style.configure('TCombobox', fieldbackground=input_bg, foreground=dark_text, background=dark_bg, selectbackground=input_bg, selectforeground=dark_text)
    style.map('TCombobox', fieldbackground=[('readonly', input_bg)], selectbackground=[('readonly', input_bg)], foreground=[('readonly', dark_text)])
    
    if console_output != None:
        console_output.config(bg=dark_bg, fg=light_text, insertbackground=light_text) #, font=("Helvetica", 12)
    root.configure(bg=dark_bg)

##@log_function_call   
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
    
def measure_crop_wrapper(settings, q, fig_queue):
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
        print('start')
        spacr.measure.measure_crop(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)  # Send the error message to the GUI via the queue
        traceback.print_exc()
    finally:
        plt.show = original_show  # Restore the original plt.show function
        
#@log_function_call
def preprocess_generate_masks_wrapper(settings, q, fig_queue):
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
        spacr.core.preprocess_generate_masks(src=settings['src'], settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)  # Send the error message to the GUI via the queue
        traceback.print_exc()
    finally:
        plt.show = original_show  # Restore the original plt.show function

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

def setup_settings_panel(vertical_container, settings_type='mask', settings_row=0):
    global vars_dict, scrollable_frame
    from .settings import set_default_settings_preprocess_generate_masks, get_measure_crop_settings, set_default_train_test_model
    
    settings_frame = tk.Frame(vertical_container, bg='black')
    vertical_container.add(settings_frame, stretch="always")
    settings_label = ttk.Label(settings_frame, text="Settings", style="Custom.TLabel", background="black", foreground="white")
    settings_label.grid(row=settings_row, column=0, pady=10, padx=10)
    settings_row += 1
    scrollable_frame = ScrollableFrame(settings_frame, bg='black')
    scrollable_frame.grid(row=settings_row, column=0, sticky="nsew")
    settings_frame.grid_rowconfigure(settings_row, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)

    if settings_type == 'mask':
        settings = set_default_settings_preprocess_generate_masks(src='path', settings={})
    elif settings_type == 'measure':
        settings = get_measure_crop_settings(settings={})
    elif settings_type == 'classify':
        settings = set_default_train_test_model(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")

    variables = convert_settings_dict_for_gui(settings)
    vars_dict = generate_fields(variables, scrollable_frame, settings_row)
    toggle_settings()
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

def setup_button_section(scrollable_frame, settings_type='mask', btn_row=0, settings_row=5, run=True, abort=True, download=True, import_btn=True, progress=True):
    global run_button, abort_button, download_dataset_button, import_button, progress_label, q, fig_queue, var_dict
    if run:
        run_button = ttk.Button(scrollable_frame.scrollable_frame, text="Run", command=lambda: start_process(q, fig_queue, settings_type), style='Custom.TButton')
        run_button.grid(row=btn_row, column=0, pady=5, padx=5)
    if abort:
        abort_button = ttk.Button(scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort, style='Custom.TButton')
        abort_button.grid(row=btn_row, column=1, pady=5, padx=5)
    btn_row += 1
    if download:
        download_dataset_button = ttk.Button(scrollable_frame.scrollable_frame, text="Download", command=download_hug_dataset, style='Custom.TButton')
        download_dataset_button.grid(row=btn_row, column=0, pady=5, padx=5)
    if import_btn:
        import_button = ttk.Button(scrollable_frame.scrollable_frame, text="Import", command=lambda: import_settings(settings_row, settings_type), style='Custom.TButton')
        import_button.grid(row=btn_row, column=1, pady=5, padx=5)
    btn_row += 1
    if progress:
        progress_label = ttk.Label(scrollable_frame.scrollable_frame, text="Processing: 0%", background="black", foreground="white")
        progress_label.grid(row=btn_row, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=10)
    return progress_label

def setup_console(vertical_container):
    global console_output
    print("Setting up console frame")
    console_frame = tk.Frame(vertical_container, bg='black')
    vertical_container.add(console_frame, stretch="always")
    console_label = ttk.Label(console_frame, text="Console", background="black", foreground="white")
    console_label.grid(row=0, column=0, pady=10, padx=10)
    console_output = scrolledtext.ScrolledText(console_frame, height=10, bg='black', fg='white', insertbackground='white')
    console_output.grid(row=1, column=0, sticky="nsew")
    console_frame.grid_rowconfigure(1, weight=1)
    console_frame.grid_columnconfigure(0, weight=1)
    # Temporarily avoid redirecting stdout and stderr
    # sys.stdout = StdoutRedirector(console_output)
    # sys.stderr = StdoutRedirector(console_output)
    print("Console setup complete")
    return console_output

def toggle_test_mode():
    global vars_dict, test_mode_button
    current_state = vars_dict['test_mode'][2].get()
    new_state = not current_state
    vars_dict['test_mode'][2].set(new_state)
    if new_state:
        test_mode_button.config(bg="blue")
    else:
        test_mode_button.config(bg="gray")

def toggle_settings():
    global vars_dict, scrollable_frame

    if vars_dict is None:
        raise ValueError("vars_dict is not initialized.")

    categories = {
        "General": ["src", "input_folder","metadata_type", "custom_regex", "experiment", "channels", "magnification"],
        "Nucleus": ["nucleus_channel", "nucleus_background", "nucleus_Signal_to_noise", "nucleus_CP_prob", "nucleus_FT", "remove_background_nucleus", "nucleus_min_size", "nucleus_mask_dim", "nucleus_loc"],
        "Cell": ["cell_channel", "cell_background", "cell_Signal_to_noise", "cell_CP_prob", "cell_FT", "remove_background_cell", "cell_min_size", "cell_mask_dim", "cytoplasm", "cytoplasm_min_size","include_uninfected", "merge_edge_pathogen_cells", "adjust_cells"],
        "Pathogen": ["pathogen_channel", "pathogen_background", "pathogen_Signal_to_noise", "pathogen_CP_prob", "pathogen_FT", "pathogen_model", "remove_background_pathogen", "pathogen_min_size", "pathogen_mask_dim",],
        "Timelapse": ["timelapse", "fps", "timelapse_displacement", "timelapse_memory", "timelapse_frame_limits", "timelapse_remove_transient", "timelapse_mode", "timelapse_objects","compartments"],
        "Plot": ["plot_filtration","examples_to_plot", "normalize_plots", "normalize", "cmap", "figuresize", "plot", ],
        "Object Image": ["save_png", "dialate_pngs", "dialate_png_ratios", "png_size", "png_dims", "save_arrays", "normalize_by", "dialate_png_ratios", "crop_mode", "dialate_pngs", "normalize", "use_bounding_box"],
        "Annotate Data":["treatment_loc","cells", "cell_loc", "pathogens", "pathogen_loc", "channel_of_interest", "measurement","treatments", "representative_images", "um_per_pixel","nr_imgs"],
        "Measurements": ["homogeneity","homogeneity_distances", "radial_dist", "calculate_correlation", "manders_thresholds", "save_measurements"],
        "Advanced": ["preprocess", "remove_background", "normalize", "lower_percentile", "merge_pathogens", "batch_size", "filter","save", "masks", "verbose", "randomize", "max_workers", "workers"],
        "Miscilanious": ["all_to_mip","pick_slice","skip_mode","upscale","upscale_factor"],
        "Test": ["test_mode", "test_images", "random_test", "test_nr"]
    }

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

    # Create toggles for each category
    row = 0
    col = 3
    for i, (category, settings) in enumerate(categories.items()):
        if any(setting in vars_dict for setting in settings):
            category_var = tk.IntVar(value=0)
            vars_dict[category] = (None, None, category_var)
            toggle = Checkbutton(
                scrollable_frame.scrollable_frame, 
                text=category, 
                variable=category_var, 
                command=lambda cat=settings, var=category_var: toggle_category(cat, var),
                background="black", 
                foreground="white", 
                selectcolor="gray",
                anchor="w",  # Align the checkbox to the left
                justify="left"  # Align the text to the right
            )
            toggle.grid(row=row, column=col, pady=2, padx=2, sticky="w")  # Sticky west to align to left
            if i % 2 == 0:
                row += 1
            else:
                col += 1
            if row == 2:
                row = 0
            
    # Hide all settings initially
    for settings in categories.values():
        for setting in settings:
            if setting in vars_dict:
                label, widget, _ = vars_dict[setting]
                label.grid_remove()
                widget.grid_remove()

def initiate_abort():
    global thread_control
    if thread_control.get("stop_requested") is not None:
        thread_control["stop_requested"].value = 1

    if thread_control.get("run_thread") is not None:
        thread_control["run_thread"].join(timeout=5)
        if thread_control["run_thread"].is_alive():
            thread_control["run_thread"].terminate()
        thread_control["run_thread"] = None

def run_mask_gui(settings, q, fig_queue, stop_requested):
    process_stdout_stderr(q)
    try:
        preprocess_generate_masks_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def start_process(q, fig_queue, settings_type='mask'):
    global thread_control, vars_dict
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
    thread_control["run_thread"].start()

def import_settings(settings_row, settings_type='mask'):
    global vars_dict, scrollable_frame
    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    csv_settings = read_settings_from_csv(csv_file_path)
    if settings_type == 'mask':
        settings = set_default_settings_preprocess_generate_masks(src='path', settings={})
    elif settings_type == 'measure':
        settings = get_measure_crop_settings(settings={})
    elif settings_type == 'classify':
        settings = set_default_train_test_model(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")
    
    variables = convert_settings_dict_for_gui(settings)
    new_settings = update_settings_from_csv(variables, csv_settings)
    #print(new_settings)
    vars_dict = generate_fields(new_settings, scrollable_frame, settings_row)

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

def setup_frame(parent_frame):
    style = ttk.Style(parent_frame)
    set_dark_style(style)
    set_default_font(parent_frame, font_name="Helvetica", size=8)
    parent_frame.configure(bg='black')
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)
    vertical_container = tk.PanedWindow(parent_frame, orient=tk.HORIZONTAL, bg='black')
    vertical_container.grid(row=0, column=0, sticky=tk.NSEW)
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)
    return parent_frame, vertical_container

#@log_function_call
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
    global q, fig_queue, parent_frame, scrollable_frame, vars_dict, canvas, canvas_widget, progress_label
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
    settings_row = 5
    parent_frame, vertical_container = setup_frame(parent_frame)
    scrollable_frame, vars_dict = setup_settings_panel(vertical_container, settings_type, settings_row)
    progress_label = setup_button_section(scrollable_frame, settings_type, btn_row=0, settings_row=settings_row)
    canvas, canvas_widget = setup_plot_section(vertical_container)
    console_output = setup_console(vertical_container)
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