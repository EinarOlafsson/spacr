import os, traceback, ctypes, requests, csv, time, requests, re
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Process, Value, Queue, set_start_method
from multiprocessing.sharedctypes import Synchronized
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import psutil
import GPUtil

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .settings import set_default_train_test_model, get_measure_crop_settings, set_default_settings_preprocess_generate_masks, get_analyze_reads_default_settings, set_default_umap_image_settings
from .gui_elements import spacrProgressBar, spacrButton, spacrLabel, spacrFrame, spacrDropdownMenu ,set_dark_style

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

def start_process(q=None, fig_queue=None, settings_type='mask'):
    global thread_control, vars_dict
    from .settings import check_settings, expected_types
    from .gui_utils import run_function_gui

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

def toggle_settings(button_scrollable_frame):
    global vars_dict
    from .settings import categories
    from .gui_utils import hide_all_settings
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
    category_dropdown.grid(row=0, column=0, sticky="ew", pady=2, padx=2)
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
    
    usage_scrollable_frame = spacrFrame(usage_frame, scrollbar=False)
    usage_scrollable_frame.grid(row=1, column=0, sticky="nsew", columnspan=2)
    widgets = [usage_label, usage_scrollable_frame.scrollable_frame]
    usage_bars = []
    max_elements_per_column = 3
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

def import_settings(settings_type='mask'):
    from .gui_utils import convert_settings_dict_for_gui, hide_all_settings
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

def setup_settings_panel(vertical_container, settings_type='mask', window_dimensions=[500, 1000]):
    global vars_dict, scrollable_frame
    from .settings import get_identify_masks_finetune_default_settings, set_default_analyze_screen, set_default_settings_preprocess_generate_masks, get_measure_crop_settings, set_default_train_test_model, get_analyze_reads_default_settings, set_default_umap_image_settings, generate_fields, get_perform_regression_default_settings, get_train_cellpose_default_settings, get_map_barcodes_default_settings, get_analyze_recruitment_default_settings, get_check_cellpose_models_default_settings
    from .gui_utils import convert_settings_dict_for_gui

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

def setup_button_section(horizontal_container, settings_type='mask', run=True, abort=True, download=True, import_btn=True):
    global button_frame, button_scrollable_frame, run_button, abort_button, download_dataset_button, import_button, q, fig_queue, vars_dict, progress_bar
    from .gui_utils import set_element_size, download_hug_dataset
    from .settings import categories

    size_dict = set_element_size(horizontal_container)
    button_frame = tk.Frame(horizontal_container)
    horizontal_container.add(button_frame, stretch="always", sticky="nsew")
    button_frame.grid_rowconfigure(0, weight=0)
    button_frame.grid_rowconfigure(1, weight=1)
    button_frame.grid_columnconfigure(0, weight=1)

    categories_label = spacrLabel(button_frame, text="Categories", anchor='center', justify='center', align="center")
    categories_label.grid(row=0, column=0, pady=10, padx=10)
    button_scrollable_frame = spacrFrame(button_frame, scrollbar=False)
    button_scrollable_frame.grid(row=1, column=0, sticky="nsew")
    widgets = [categories_label, button_scrollable_frame.scrollable_frame]

    btn_col = 3
    btn_row = 0

    if run:
        print(f'settings_type: {settings_type}')
        run_button = spacrButton(button_scrollable_frame.scrollable_frame, text="run", command=lambda: start_process(q, fig_queue, settings_type), show_text=False, size=size_dict['btn_size'], animation=False)
        run_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(run_button)
        btn_col += 1

    if abort and settings_type in ['mask', 'measure', 'classify', 'sequencing', 'umap']:
        abort_button = spacrButton(button_scrollable_frame.scrollable_frame, text="abort", command=initiate_abort, show_text=False, size=size_dict['btn_size'], animation=False)
        abort_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(abort_button)
        btn_col += 1

    if download and settings_type in ['mask']:
        download_dataset_button = spacrButton(button_scrollable_frame.scrollable_frame, text="download", command=download_hug_dataset, show_text=False, size=size_dict['btn_size'], animation=False)
        download_dataset_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(download_dataset_button)
        btn_col += 1

    if import_btn:
        import_button = spacrButton(button_scrollable_frame.scrollable_frame, text="settings", command=lambda: import_settings(settings_type),show_text=False, size=size_dict['btn_size'], animation=False)
        import_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(import_button)
        btn_row += 1

    # Add the progress bar under the settings category menu
    progress_bar = spacrProgressBar(button_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate')
    progress_bar.grid(row=btn_row, column=0, columnspan=3, pady=5, padx=5, sticky='ew')
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

def initiate_root(parent, settings_type='mask'):
    global q, fig_queue, parent_frame, scrollable_frame, button_frame, vars_dict, canvas, canvas_widget, button_scrollable_frame, progress_bar
    from .gui_utils import main_thread_update_function, setup_frame
    from .gui import gui_app
    set_start_method('spawn', force=True)
    print("Initializing root with settings_type:", settings_type)

    parent_frame = parent
    parent_frame.update_idletasks()
    frame_width = int(parent_frame.winfo_width())
    frame_height = int(parent_frame.winfo_height())
    print(f"Frame dimensions: {frame_width}x{frame_height}")
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
    parent_frame, vertical_container, horizontal_container, settings_container = setup_frame(parent_frame)

    # Debugging: Add a label to the settings container to ensure visibility
    label = tk.Label(settings_container, text="Settings Container", bg="yellow")
    label.pack(fill=tk.BOTH, expand=True)
    settings_container.update_idletasks()
    print(f"Settings container size: {settings_container.winfo_width()}x{settings_container.winfo_height()}")

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


