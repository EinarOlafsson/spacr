import os, traceback, ctypes, matplotlib, requests, csv
matplotlib.use('Agg')
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Process, Value, Queue
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import set_start_method
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import requests
from huggingface_hub import list_repo_files

from .settings import set_default_train_test_model, get_measure_crop_settings, set_default_settings_preprocess_generate_masks, get_analyze_reads_default_settings, set_default_umap_image_settings
from .gui_elements import create_menu_bar, spacrButton, spacrLabel, spacrFrame, spacrDropdownMenu ,set_dark_style, set_default_font
from . gui_run import run_mask_gui, run_measure_gui, run_classify_gui, run_sequencing_gui, run_umap_gui

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

def initiate_abort():
    global thread_control
    if isinstance(thread_control.get("stop_requested"), Synchronized):
        thread_control["stop_requested"].value = 1
    if thread_control.get("run_thread") is not None:
        thread_control["run_thread"].terminate()
        thread_control["run_thread"].join()
        thread_control["run_thread"] = None

def start_process_v1(q, fig_queue, settings_type='mask'):
    global thread_control, vars_dict
    from .settings import check_settings, expected_types

    settings = check_settings(vars_dict, expected_types, q)
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

def start_process(q=None, fig_queue=None, settings_type='mask'):
    global thread_control, vars_dict
    from .settings import check_settings, expected_types

    if q is None:
        q = Queue()
    if fig_queue is None:
        fig_queue = Queue()

    try:
        settings = check_settings(vars_dict, expected_types, q)
    except ValueError as e:
        q.put(f"Error: {e}")
        return

    if thread_control.get("run_thread") is not None:
        initiate_abort()
    
    stop_requested = Value('i', 0)  # multiprocessing shared value for inter-process communication
    thread_control["stop_requested"] = stop_requested

    process_args = (settings, q, fig_queue, stop_requested)

    if settings_type == 'mask':
        thread_control["run_thread"] = Process(target=run_mask_gui, args=process_args)
    elif settings_type == 'measure':
        thread_control["run_thread"] = Process(target=run_measure_gui, args=process_args)
    elif settings_type == 'classify':
        thread_control["run_thread"] = Process(target=run_classify_gui, args=process_args)
    elif settings_type == 'sequencing':
        thread_control["run_thread"] = Process(target=run_sequencing_gui, args=process_args)
    elif settings_type == 'umap':
        thread_control["run_thread"] = Process(target=run_umap_gui, args=process_args)
    else:
        q.put(f"Error: Unknown settings type '{settings_type}'")
        return

    thread_control["run_thread"].start()

def import_settings(settings_type='mask'):
    global vars_dict, scrollable_frame, button_scrollable_frame
    from .settings import generate_fields

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

    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not csv_file_path:  # If no file is selected, return early
        return
    
    #vars_dict = hide_all_settings(vars_dict, categories=None)
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
    vars_dict = hide_all_settings(vars_dict, categories=None)

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

def setup_settings_panel(vertical_container, settings_type='mask', window_dimensions=[500, 1000]):
    global vars_dict, scrollable_frame
    from .settings import set_default_settings_preprocess_generate_masks, get_measure_crop_settings, set_default_train_test_model, get_analyze_reads_default_settings, set_default_umap_image_settings, generate_fields, descriptions

    width = (window_dimensions[0])//6
    height = window_dimensions[1]

    # Settings Frame
    settings_frame = tk.Frame(vertical_container, bg='black', height=height, width=width)
    vertical_container.add(settings_frame, stretch="always")
    settings_label = spacrLabel(settings_frame, text="Settings", background="black", foreground="white", anchor='center', justify='center', align="center")
    settings_label.grid(row=0, column=0, pady=10, padx=10)
    scrollable_frame = spacrFrame(settings_frame, bg='black', width=width)
    scrollable_frame.grid(row=1, column=0, sticky="nsew")
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)

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
    dataset_repo_id = "einarolafsson/toxo_mito"
    settings_repo_id = "einarolafsson/spacr_settings"
    dataset_subfolder = "plate1"
    local_dir = os.path.join(os.path.expanduser("~"), "datasets")  # Set to the home directory

    # Download the dataset
    try:
        dataset_path = download_dataset(dataset_repo_id, dataset_subfolder, local_dir)
        if 'src' in vars_dict:
            vars_dict['src'][2].set(dataset_path)
            q.put(f"Set source path to: {vars_dict['src'][2].get()}\n")
        q.put(f"Dataset downloaded to: {dataset_path}\n")
    except Exception as e:
        q.put(f"Failed to download dataset: {e}\n")

    # Download the settings files
    try:
        settings_path = download_dataset(settings_repo_id, "", local_dir)
        q.put(f"Settings downloaded to: {settings_path}\n")
    except Exception as e:
        q.put(f"Failed to download settings: {e}\n")

def download_dataset(repo_id, subfolder, local_dir=None, retries=5, delay=5):
    global q
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

def setup_button_section(horizontal_container, settings_type='mask',  window_dimensions=[500, 1000], run=True, abort=True, download=True, import_btn=True):
    global button_frame, button_scrollable_frame, run_button, abort_button, download_dataset_button, import_button, q, fig_queue, vars_dict
    from .settings import descriptions

    width = (window_dimensions[0])//8
    height = window_dimensions[1]

    button_frame = tk.Frame(horizontal_container, bg='black', height=height, width=width)
    horizontal_container.add(button_frame, stretch="always", sticky="nsew")
    button_frame.grid_rowconfigure(0, weight=0)
    button_frame.grid_rowconfigure(1, weight=1)
    button_frame.grid_columnconfigure(0, weight=1)

    categories_label = spacrLabel(button_frame, text="Categories", background="black", foreground="white", font=('Helvetica', 12), anchor='center', justify='center', align="center")
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
        import_button = spacrButton(button_scrollable_frame.scrollable_frame, text="Import", command=lambda: import_settings(settings_type), font=('Helvetica', 12))
        import_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')

    # Call toggle_settings after vars_dict is initialized
    if vars_dict is not None:
        toggle_settings(button_scrollable_frame)

    # Description frame
    description_frame = tk.Frame(horizontal_container, bg='black', height=height, width=width)
    horizontal_container.add(description_frame, stretch="always", sticky="nsew")
    description_label = tk.Label(description_frame, text="Module Description", bg='black', fg='white', anchor='nw', justify='left', wraplength=width//2-100)
    description_label.pack(pady=10, padx=10)
    description_text = descriptions.get(settings_type, "No description available for this module.")
    description_label.config(text=description_text)

    return button_scrollable_frame

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
            vars_dict[category] = (None, None, tk.IntVar(value=0))
            
            # Initially hide all settings
            for setting in settings:
                if setting in vars_dict:
                    label, widget, _ = vars_dict[setting]
                    label.grid_remove()
                    widget.grid_remove()
    return vars_dict

def toggle_settings(button_scrollable_frame):
    global vars_dict
    from .settings import categories

    if vars_dict is None:
        raise ValueError("vars_dict is not initialized.")

    active_categories = set()

    def toggle_category(settings):
        for setting in settings:
            if setting in vars_dict:
                label, widget, _ = vars_dict[setting]
                if widget.grid_info():
                    label.grid_remove()
                    widget.grid_remove()
                else:
                    label.grid()
                    widget.grid()

    def on_category_select(selected_category):
        if selected_category == "Select Category":
            return
        #print(f"Selected category: {selected_category}")
        if selected_category in categories:
            toggle_category(categories[selected_category])
            if selected_category in active_categories:
                active_categories.remove(selected_category)
            else:
                active_categories.add(selected_category)
        category_dropdown.update_styles(active_categories)
        category_var.set("Select Category")  # Reset dropdown text to "Select Category"

    category_var = tk.StringVar()
    non_empty_categories = [category for category, settings in categories.items() if any(setting in vars_dict for setting in settings)]
    category_dropdown = spacrDropdownMenu(button_scrollable_frame.scrollable_frame, category_var, non_empty_categories, command=on_category_select)
    category_dropdown.grid(row=1, column=3, sticky="ew", pady=2, padx=2)
    vars_dict = hide_all_settings(vars_dict, categories)

def process_fig_queue():
    global canvas, fig_queue, canvas_widget, parent_frame

    def clear_canvas(canvas):
        for ax in canvas.figure.get_axes():
            ax.clear()
        canvas.draw_idle()

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

def initiate_root(parent, settings_type='mask'):
    global q, fig_queue, parent_frame, scrollable_frame, button_frame, vars_dict, canvas, canvas_widget, progress_label, progress_output, button_scrollable_frame
    from .gui_utils import main_thread_update_function
    from .gui import gui_app
    set_start_method('spawn', force=True)
    print("Initializing root with settings_type:", settings_type)

    parent_frame = parent
    parent_frame.update_idletasks()
    frame_width = int(parent_frame.winfo_width())
    frame_height = int(parent_frame.winfo_height())
    print(frame_width, frame_height)
    dims = [frame_width, frame_height]

    if not hasattr(parent_frame, 'after_tasks'):
        parent_frame.after_tasks = []

    # Clear previous content instead of destroying the root
    for widget in parent_frame.winfo_children():
        try:
            widget.destroy()
        except tk.TclError as e:
            print(f"Error destroying widget: {e}")

    q = Queue()
    fig_queue = Queue()
    parent_frame, vertical_container, horizontal_container = setup_frame(parent_frame)

    if settings_type == 'annotate':
        from .app_annotate import initiate_annotation_app
        initiate_annotation_app(horizontal_container)
    elif settings_type == 'make_masks':
        from .app_make_masks import initiate_make_mask_app
        initiate_make_mask_app(horizontal_container)
    else:
        scrollable_frame, vars_dict = setup_settings_panel(horizontal_container, settings_type, window_dimensions=dims)
        button_scrollable_frame = setup_button_section(horizontal_container, settings_type, window_dimensions=dims)
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
