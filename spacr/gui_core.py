import os, traceback, ctypes, matplotlib, requests, csv, matplotlib, time, requests, re
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Process, Value, Queue, set_start_method
from multiprocessing.sharedctypes import Synchronized
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from huggingface_hub import list_repo_files
import numpy as np

import psutil, gpustat
import GPUtil
from threading import Thread
from time import sleep


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .settings import set_default_train_test_model, get_measure_crop_settings, set_default_settings_preprocess_generate_masks, get_analyze_reads_default_settings, set_default_umap_image_settings
from .gui_elements import spacrProgressBar, spacrButton, spacrLabel, spacrFrame, spacrDropdownMenu ,set_dark_style, set_default_font

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
    from .core import preprocess_generate_masks, generate_ml_scores, identify_masks_finetune, check_cellpose_models, analyze_recruitment, train_cellpose, compare_cellpose_masks, analyze_plaques, generate_dataset, apply_model_to_tar
    from .io import generate_cellpose_train_test
    from .measure import measure_crop
    from .sim import run_multiple_simulations
    from .deep_spacr import train_test_model
    from .sequencing import analyze_reads, map_barcodes_folder, perform_regression
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
    elif settings_type == 'sequencing':
        function = analyze_reads
        imports = 1
    elif settings_type == 'classify':
        function = train_test_model
        imports = 2
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
        function = map_barcodes_folder
        imports = 2
    elif settings_type == 'regression':
        function = perform_regression
        imports = 2
    elif settings_type == 'recruitment':
        function = analyze_recruitment
        imports = 2
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")
    try:
        function_gui_wrapper(function, settings, q, fig_queue, imports)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

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
    
    stop_requested = Value('i', 0)
    thread_control["stop_requested"] = stop_requested

    process_args = (settings_type, settings, q, fig_queue, stop_requested)
    if settings_type in ['mask','measure','simulation','sequencing','classify','cellpose_dataset','train_cellpose','ml_analyze','cellpose_masks','cellpose_all','map_barcodes','regression','recruitment','plaques','cellpose_compare','vision_scores','vision_dataset']:
        thread_control["run_thread"] = Process(target=run_function_gui, args=process_args)
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
    from torchvision import models as torch_models
    torchvision_models = [name for name, obj in torch_models.__dict__.items() if callable(obj)]
    chans = ['0', '1', '2', '3', '4', '5', '6', '7', '8', None]
    chans_v2 = [0, 1, 2, 3, None]
    variables = {}
    special_cases = {
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'nikon', 'zeis', 'custom'], 'cellvoyager'),
        'channels': ('combo', ['[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]'], '[0,1,2,3]'),
        'channel_dims': ('combo', ['[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]'], '[0,1,2,3]'),
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

def setup_settings_panel(vertical_container, settings_type='mask', window_dimensions=[500, 1000]):
    global vars_dict, scrollable_frame
    from .settings import get_identify_masks_finetune_default_settings, set_default_analyze_screen, set_default_settings_preprocess_generate_masks, get_measure_crop_settings, set_default_train_test_model, get_analyze_reads_default_settings, set_default_umap_image_settings, generate_fields, get_perform_regression_default_settings, get_train_cellpose_default_settings, get_map_barcodes_default_settings, get_analyze_recruitment_default_settings, get_check_cellpose_models_default_settings

    width = (window_dimensions[0]) // 6
    height = window_dimensions[1]

    settings_frame = tk.Frame(vertical_container)
    vertical_container.add(settings_frame, stretch="always")
    settings_label = spacrLabel(settings_frame, text="Settings", anchor='center', justify='center', align="center")
    settings_label.grid(row=0, column=0, pady=10, padx=10)
    scrollable_frame = spacrFrame(settings_frame)
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
    elif settings_type == 'train_cellpose':
        settings = get_train_cellpose_default_settings(settings={})
    elif settings_type == 'ml_analyze':
        settings = set_default_analyze_screen(settings={})
    elif settings_type == 'cellpose_masks':
        settings = get_identify_masks_finetune_default_settings(settings={})
    elif settings_type == 'cellpose_all':
        settings = get_check_cellpose_models_default_settings(settings={})
    elif settings_type == 'map_barcodes':
        settings = get_map_barcodes_default_settings(settings={})
    elif settings_type == 'regression':
        settings = get_perform_regression_default_settings(settings={})
    elif settings_type == 'recruitment':
        settings = get_analyze_recruitment_default_settings(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")

    variables = convert_settings_dict_for_gui(settings)
    vars_dict = generate_fields(variables, scrollable_frame)
    
    containers = [settings_frame]
    widgets = [settings_label, scrollable_frame]

    style = ttk.Style(vertical_container)
    _ = set_dark_style(style, containers=containers, widgets=widgets)

    print("Settings panel setup complete")
    return scrollable_frame, vars_dict

def setup_plot_section(vertical_container):
    global canvas, canvas_widget
    plot_frame = tk.PanedWindow(vertical_container, orient=tk.VERTICAL)
    vertical_container.add(plot_frame, stretch="always")
    figure = Figure(figsize=(30, 4), dpi=100)
    plot = figure.add_subplot(111)
    plot.plot([], [])  # This creates an empty plot.
    plot.axis('off')
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().configure(cursor='arrow', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    plot_frame.add(canvas_widget, stretch="always")
    canvas.draw()
    canvas.figure = figure
    style_out = set_dark_style(ttk.Style())

    figure.patch.set_facecolor(style_out['bg_color'])
    plot.set_facecolor(style_out['bg_color'])
    containers = [plot_frame]
    widgets = [canvas_widget]
    style = ttk.Style(vertical_container)
    _ = set_dark_style(style, containers=containers, widgets=widgets)
    return canvas, canvas_widget

def setup_console(vertical_container):
    global console_output
    console_frame = tk.Frame(vertical_container)
    vertical_container.add(console_frame, stretch="always")
    console_label = spacrLabel(console_frame, text="Console", anchor='center', justify='center', align="center")
    console_label.grid(row=0, column=0, pady=10, padx=10)
    console_output = scrolledtext.ScrolledText(console_frame, height=10)
    console_output.grid(row=1, column=0, sticky="nsew")
    console_frame.grid_rowconfigure(1, weight=1)
    console_frame.grid_columnconfigure(0, weight=1)
    containers = [console_frame]
    widgets = [console_label, console_output]
    style = ttk.Style(vertical_container)
    _ = set_dark_style(style, containers=containers, widgets=widgets)
    return console_output

def setup_progress_frame(vertical_container):
    global progress_output
    progress_frame = tk.Frame(vertical_container)
    vertical_container.add(progress_frame, stretch="always")
    label_frame = tk.Frame(progress_frame)
    label_frame.grid(row=0, column=0, sticky="ew", pady=(5, 0), padx=10)
    progress_label = spacrLabel(label_frame, text="Processing: 0%", font=('Helvetica', 12), anchor='w', justify='left', align="left")
    progress_label.grid(row=0, column=0, sticky="w")
    progress_output = scrolledtext.ScrolledText(progress_frame, height=10)
    progress_output.grid(row=1, column=0, sticky="nsew")
    progress_frame.grid_rowconfigure(1, weight=1)
    progress_frame.grid_columnconfigure(0, weight=1)
    containers = [progress_frame, label_frame]
    widgets = [progress_label, progress_output]
    style = ttk.Style(vertical_container)
    _ = set_dark_style(style, containers=containers, widgets=widgets)
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



def setup_button_section(horizontal_container, settings_type='mask', run=True, abort=True, download=True, import_btn=True):
    global button_frame, button_scrollable_frame, run_button, abort_button, download_dataset_button, import_button, q, fig_queue, vars_dict, progress_bar
    from .gui_utils import set_element_size
    size_dict = set_element_size(horizontal_container)
    button_frame = tk.Frame(horizontal_container)
    horizontal_container.add(button_frame, stretch="always", sticky="nsew")
    button_frame.grid_rowconfigure(0, weight=0)
    button_frame.grid_rowconfigure(1, weight=1)
    button_frame.grid_columnconfigure(0, weight=1)

    categories_label = spacrLabel(button_frame, text="Categories", anchor='center', justify='center', align="center")
    categories_label.grid(row=0, column=0, pady=10, padx=10)
    button_scrollable_frame = spacrFrame(button_frame)
    button_scrollable_frame.grid(row=1, column=0, sticky="nsew")

    widgets = [categories_label, button_scrollable_frame.scrollable_frame]

    btn_col = 0
    btn_row = 3

    if run:
        print(f'settings_type: {settings_type}')
        run_button = spacrButton(button_scrollable_frame.scrollable_frame, text="run", command=lambda: start_process(q, fig_queue, settings_type), show_text=False, size=size_dict['btn_size'])
        run_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(run_button)
        btn_row += 1

    if abort and settings_type in ['mask', 'measure', 'classify', 'sequencing', 'umap']:
        abort_button = spacrButton(button_scrollable_frame.scrollable_frame, text="abort", command=initiate_abort, show_text=False, size=size_dict['btn_size'])
        abort_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(abort_button)
        btn_row += 1

    if download and settings_type in ['mask']:
        download_dataset_button = spacrButton(button_scrollable_frame.scrollable_frame, text="download", command=download_hug_dataset, show_text=False, size=size_dict['btn_size'])
        download_dataset_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(download_dataset_button)
        btn_row += 1

    if import_btn:
        import_button = spacrButton(button_scrollable_frame.scrollable_frame, text="settings", command=lambda: import_settings(settings_type),show_text=False, size=size_dict['btn_size'])
        import_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(import_button)
        btn_row += 1

    # Add the progress bar under the settings category menu
    progress_bar = spacrProgressBar(button_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate')
    progress_bar.grid(row=0, column=0, columnspan=2, pady=5, padx=5, sticky='ew')
    widgets.append(progress_bar)

    if vars_dict is not None:
        toggle_settings(button_scrollable_frame)

    style = ttk.Style(horizontal_container)
    _ = set_dark_style(style, containers=[button_frame], widgets=widgets)

    return button_scrollable_frame

def setup_help_section(horizontal_container, settings_type='mask'):
    from .settings import descriptions

    description_frame = tk.Frame(horizontal_container)
    horizontal_container.add(description_frame, stretch="always", sticky="nsew")
    description_frame.grid_columnconfigure(0, weight=1)
    description_frame.grid_rowconfigure(1, weight=1)  # Ensure the text widget row is expandable

    description_label = spacrLabel(description_frame, text=f"{settings_type} Module", anchor='center', justify='center', align="center")
    description_label.grid(row=0, column=0, pady=10, padx=10, sticky='ew')

    # Set background color directly
    style_out = set_dark_style(ttk.Style())
    bg_color = style_out['bg_color']
    fg_color = style_out['fg_color']

    description_text_widget = tk.Text(description_frame, wrap="word", bg=bg_color, fg=fg_color)
    description_text_widget.grid(row=1, column=0, sticky="nsew")

    description_text = descriptions.get(settings_type, "No description available for this module.")
    description_text_widget.insert("1.0", description_text)
    description_text_widget.config(state="disabled")  # Make the text widget read-only

    def update_wraplength(event):
        new_width = event.width - 20  # Adjust as needed
        description_text_widget.config(width=new_width)

    description_text_widget.bind('<Configure>', update_wraplength)

    style = ttk.Style(horizontal_container)
    _ = set_dark_style(style, containers=[description_frame], widgets=[description_label, description_text_widget])

    return description_frame

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
        if selected_category in categories:
            toggle_category(categories[selected_category])
            if selected_category in active_categories:
                active_categories.remove(selected_category)
            else:
                active_categories.add(selected_category)
        category_dropdown.update_styles(active_categories)
        category_var.set("Select Category")

    category_var = tk.StringVar()
    non_empty_categories = [category for category, settings in categories.items() if any(setting in vars_dict for setting in settings)]
    category_dropdown = spacrDropdownMenu(button_scrollable_frame.scrollable_frame, category_var, non_empty_categories, command=on_category_select)
    category_dropdown.grid(row=7, column=0, sticky="ew", pady=2, padx=2)
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
    global q, console_output, parent_frame, progress_bar

    # Initialize function attribute if it doesn't exist
    if not hasattr(process_console_queue, "completed_tasks"):
        process_console_queue.completed_tasks = []

    ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    
    while not q.empty():
        message = q.get_nowait()
        clean_message = ansi_escape_pattern.sub('', message)
        console_output.insert(tk.END, clean_message + "\n")
        console_output.see(tk.END)
        
        # Check if the message contains progress information
        if clean_message.startswith("Progress"):
            try:
                # Extract the progress information
                match = re.search(r'(\d+)/(\d+)', clean_message)
                if match:
                    current_progress = int(match.group(1))
                    total_progress = int(match.group(2))

                    # Add the task to the completed set
                    process_console_queue.completed_tasks.append(current_progress)
                    
                    # Calculate the unique progress count
                    unique_progress_count = len(np.unique(process_console_queue.completed_tasks))
                    
                    # Update the progress bar
                    if progress_bar:
                        progress_bar['maximum'] = total_progress
                        progress_bar['value'] = unique_progress_count

                    # Extract and update additional information
                    operation_match = re.search(r'operation_type: ([\w\s]+)', clean_message)
                    if operation_match:
                        progress_bar.operation_type = operation_match.group(1)

                    time_image_match = re.search(r'Time/image: ([\d.]+) sec', clean_message)
                    if time_image_match:
                        progress_bar.time_image = float(time_image_match.group(1))

                    time_batch_match = re.search(r'Time/batch: ([\d.]+) sec', clean_message)
                    if time_batch_match:
                        progress_bar.time_batch = float(time_batch_match.group(1))

                    time_left_match = re.search(r'Time_left: ([\d.]+) min', clean_message)
                    if time_left_match:
                        progress_bar.time_left = float(time_left_match.group(1))

                    # Update the progress label
                    if progress_bar.progress_label:
                        progress_bar.update_label()
                        
                    # Clear completed tasks when progress is complete
                    if unique_progress_count >= total_progress:
                        process_console_queue.completed_tasks.clear()
            except Exception as e:
                print(f"Error parsing progress message: {e}")
    
    after_id = console_output.after(100, process_console_queue)
    parent_frame.after_tasks.append(after_id)

def set_globals(q_var, console_output_var, parent_frame_var, vars_dict_var, canvas_var, canvas_widget_var, scrollable_frame_var, fig_queue_var, progress_bar_var, usage_bars_var):
    global q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, fig_queue, progress_bar, usage_bars
    q = q_var
    console_output = console_output_var
    parent_frame = parent_frame_var
    vars_dict = vars_dict_var
    canvas = canvas_var
    canvas_widget = canvas_widget_var
    scrollable_frame = scrollable_frame_var
    fig_queue = fig_queue_var
    progress_bar = progress_bar_var
    usage_bars = usage_bars_var

def create_containers(parent_frame):
    vertical_container = tk.PanedWindow(parent_frame, orient=tk.VERTICAL)
    horizontal_container = tk.PanedWindow(vertical_container, orient=tk.HORIZONTAL)
    settings_frame = tk.Frame(horizontal_container)
    return vertical_container, horizontal_container, settings_frame

def setup_frame(parent_frame):
    style = ttk.Style(parent_frame)
    vertical_container, horizontal_container, settings_frame = create_containers(parent_frame)
    containers = [vertical_container, horizontal_container, settings_frame]
    
    set_dark_style(style, parent_frame, containers)
    set_default_font(parent_frame, font_name="Helvetica", size=8)
    
    vertical_container.grid(row=0, column=0, sticky=tk.NSEW)
    vertical_container.add(horizontal_container, stretch="always")
    horizontal_container.grid_columnconfigure(0, weight=1)
    horizontal_container.grid_columnconfigure(1, weight=1)
    settings_frame.grid_rowconfigure(0, weight=0)
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)
    horizontal_container.add(settings_frame, stretch="always", sticky="nsew")
    
    return parent_frame, vertical_container, horizontal_container

def setup_usage_panel(horizontal_container):
    global usage_bars
    from .gui_utils import set_element_size

    def update_usage(ram_bar, vram_bar, gpu_bar, usage_bars, parent_frame):
        # Update RAM usage
        ram_usage = psutil.virtual_memory().percent
        ram_bar['value'] = ram_usage

        # Update GPU and VRAM usage
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            vram_usage = gpu.memoryUtil * 100
            gpu_usage = gpu.load * 100
            vram_bar['value'] = vram_usage
            gpu_bar['value'] = gpu_usage

        # Update CPU usage for each core
        cpu_percentages = psutil.cpu_percent(percpu=True)
        for bar, usage in zip(usage_bars[3:], cpu_percentages):
            bar['value'] = usage

        # Schedule the function to run again after 1000 ms (1 second)
        parent_frame.after(1000, update_usage, ram_bar, vram_bar, gpu_bar, usage_bars, parent_frame)

    size_dict = set_element_size(horizontal_container)
    print(size_dict)

    usage_frame = tk.Frame(horizontal_container)
    horizontal_container.add(usage_frame, stretch="always", sticky="nsew")
    usage_frame.grid_rowconfigure(0, weight=0)
    usage_frame.grid_rowconfigure(1, weight=1)
    usage_frame.grid_columnconfigure(0, weight=1)
    usage_frame.grid_columnconfigure(1, weight=1)

    usage_label = spacrLabel(usage_frame, text="Hardware Stats", anchor='center', justify='center', align="center")
    usage_label.grid(row=0, column=0, pady=10, padx=10, columnspan=2)
    
    usage_scrollable_frame = spacrFrame(usage_frame)
    usage_scrollable_frame.grid(row=1, column=0, sticky="nsew", columnspan=2)
    widgets = [usage_label, usage_scrollable_frame.scrollable_frame]
    usage_bars = []
    max_elements_per_column = 12
    row = 0
    col = 0

    # Initialize RAM, VRAM, and GPU bars as None
    ram_bar, vram_bar, gpu_bar = None, None, None

    # Try adding RAM bar
    try:
        ram_info = psutil.virtual_memory()
        ram_label_text = f"RAM"
        label = ttk.Label(usage_scrollable_frame.scrollable_frame, text=ram_label_text, anchor='w')
        label.grid(row=row, column=2 * col, pady=5, padx=5, sticky='w')
        ram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
        ram_bar.grid(row=row, column=2 * col + 1, pady=5, padx=5, sticky='ew')
        widgets.append(label)
        widgets.append(ram_bar)
        usage_bars.append(ram_bar)
        row += 1
    except Exception as e:
        print(f"Could not add RAM usage bar: {e}")

    # Try adding VRAM and GPU usage bars
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            vram_label_text = f"VRAM"
            label = ttk.Label(usage_scrollable_frame.scrollable_frame, text=vram_label_text, anchor='w')
            label.grid(row=row, column=2 * col, pady=5, padx=5, sticky='w')
            vram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
            vram_bar.grid(row=row, column=2 * col + 1, pady=5, padx=5, sticky='ew')
            widgets.append(label)
            widgets.append(vram_bar)
            usage_bars.append(vram_bar)
            row += 1

            gpu_label_text = f"GPU"
            label = ttk.Label(usage_scrollable_frame.scrollable_frame, text=gpu_label_text, anchor='w')
            label.grid(row=row, column=2 * col, pady=5, padx=5, sticky='w')
            gpu_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
            gpu_bar.grid(row=row, column=2 * col + 1, pady=5, padx=5, sticky='ew')
            widgets.append(label)
            widgets.append(gpu_bar)
            usage_bars.append(gpu_bar)
            row += 1
    except Exception as e:
        print(f"Could not add VRAM or GPU usage bars: {e}")

    # Add CPU core usage bars
    try:
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        for core in range(cpu_cores):
            if row > 0 and row % max_elements_per_column == 0:
                col += 1
                row = 0
            label = ttk.Label(usage_scrollable_frame.scrollable_frame, text=f"Core {core+1}", anchor='w')
            label.grid(row=row, column=2 * col, pady=2, padx=5, sticky='w')
            bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
            bar.grid(row=row, column=2 * col + 1, pady=2, padx=5, sticky='ew')
            widgets.append(label)
            widgets.append(bar)
            usage_bars.append(bar)
            row += 1
    except Exception as e:
        print(f"Could not add CPU core usage bars: {e}")

    # Adding the text box for hardware information
    #hardware_frame = tk.Frame(horizontal_container)
    #horizontal_container.add(hardware_frame, stretch="always", sticky="nsew")
    #hardware_frame.grid_columnconfigure(0, weight=1)

    #hardware_info = tk.Text(hardware_frame, height=1, wrap='none', bg='black', fg='white', bd=0)
    #hardware_info.grid(row=0, column=0, pady=10, padx=5, sticky='ew')

    #hardware_text = ""
    #try:
    #    ram_info = psutil.virtual_memory()
    #    hardware_text += f"RAM: {ram_info.total / (1024 ** 3):.1f} GB  "
    #except Exception as e:
    #    hardware_text += f"RAM: Could not retrieve ({e})  "

    #try:
    #    gpus = GPUtil.getGPUs()
    #    if gpus:
    #        gpu = gpus[0]
    #        hardware_text += f"VRAM: {gpu.memoryTotal / 1024:.1f} GB  "
    #        hardware_text += f"GPU: {gpu.name}  "
    #except Exception as e:
    #    hardware_text += f"VRAM and GPU: Could not retrieve ({e})  "

    #try:
    #    if cpu_freq:
    #        hardware_text += f"CPU Max Clock Speed: {cpu_freq.max / 1000:.0f} GHz"
    #except Exception as e:
    #    hardware_text += f"CPU Max Clock Speed: Could not retrieve ({e})"

    #hardware_info.insert(tk.END, hardware_text)
    #hardware_info.configure(state='disabled')
    #widgets.append(hardware_info)

    style = ttk.Style(horizontal_container)
    _ = set_dark_style(style, containers=[usage_frame], widgets=widgets) # hardware_frame

    if ram_bar is None:
        ram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
    if vram_bar is None:
        vram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
    if gpu_bar is None:
        gpu_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)

    update_usage(ram_bar, vram_bar, gpu_bar, usage_bars, usage_frame)
    return usage_scrollable_frame, usage_bars

def initiate_root(parent, settings_type='mask'):
    global q, fig_queue, parent_frame, scrollable_frame, button_frame, vars_dict, canvas, canvas_widget, button_scrollable_frame, progress_bar
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
        button_scrollable_frame = setup_button_section(horizontal_container, settings_type)

        _, usage_bars = setup_usage_panel(horizontal_container)
        _ = setup_help_section(horizontal_container, settings_type)

        canvas, canvas_widget = setup_plot_section(vertical_container)
        console_output = setup_console(vertical_container)

        set_globals(q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, fig_queue, progress_bar, usage_bars)
        process_console_queue()
        process_fig_queue()
        after_id = parent_frame.after(100, lambda: main_thread_update_function(parent_frame, q, fig_queue, canvas_widget))
        parent_frame.after_tasks.append(after_id)

    print("Root initialization complete")
    return parent_frame, vars_dict

