import traceback, ctypes, csv, re
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Process, Value, Queue, set_start_method
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import psutil
import GPUtil
from collections import deque
import tracemalloc
from tkinter import Menu
import io

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .gui_elements import spacrProgressBar, spacrButton, spacrLabel, spacrFrame, spacrDropdownMenu , set_dark_style

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
figures = None
figure_index = None
progress_bar = None
usage_bars = None
fig_memory_limit = None
figure_current_memory_usage = None

thread_control = {"run_thread": None, "stop_requested": False}

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
                label, widget, _, frame = vars_dict[setting]
                if widget.grid_info():
                    label.grid_remove()
                    widget.grid_remove()
                    frame.grid_remove()
                else:
                    label.grid()
                    widget.grid()
                    frame.grid()

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
    category_dropdown.grid(row=0, column=4, sticky="ew", pady=2, padx=2)
    vars_dict = hide_all_settings(vars_dict, categories)

def process_fig_queue():
    global canvas, fig_queue, canvas_widget, parent_frame, uppdate_frequency, figures, figure_index

    from .gui_elements import standardize_figure
    try:
        while not fig_queue.empty():
            fig = fig_queue.get_nowait()

            if fig is None:
                print("Warning: Retrieved a None figure from fig_queue.")
                continue  # Skip processing if the figure is None

            # Standardize the figure appearance before adding it to the list
            standardize_figure(fig)

            figures.append(fig)
            if figure_index == len(figures) - 2:
                figure_index += 1
                display_figure(fig)
    except Exception as e:
        traceback.print_exc()
    finally:
        after_id = canvas_widget.after(uppdate_frequency, process_fig_queue)
        parent_frame.after_tasks.append(after_id)

def display_figure(fig):
    global canvas, canvas_widget

    from .gui_elements import modify_figure_properties, save_figure_as_format, modify_figure

    # Apply the dark style to the context menu
    style_out = set_dark_style(ttk.Style())
    bg_color = style_out['bg_color']
    fg_color = style_out['fg_color']

    # Initialize the scale factor for zooming
    scale_factor = 1.0

    # Save the original x and y limits of the first axis (assuming all axes have the same limits)
    original_xlim = [ax.get_xlim() for ax in fig.get_axes()]
    original_ylim = [ax.get_ylim() for ax in fig.get_axes()]

    # Clear previous canvas content
    if canvas:
        canvas.get_tk_widget().destroy()

    # Create a new canvas for the figure
    new_canvas = FigureCanvasTkAgg(fig, master=canvas_widget.master)
    new_canvas.draw()
    new_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # Update the global canvas and canvas_widget references
    canvas = new_canvas
    canvas_widget = new_canvas.get_tk_widget()
    canvas_widget.configure(bg=bg_color)

    # Create the context menu
    context_menu = tk.Menu(canvas_widget, tearoff=0, bg=bg_color, fg=fg_color)
    context_menu.add_command(label="Save Figure as PDF", command=lambda: save_figure_as_format(fig, 'pdf'))
    context_menu.add_command(label="Save Figure as PNG", command=lambda: save_figure_as_format(fig, 'png'))
    context_menu.add_command(label="Modify Figure", command=lambda: modify_figure(fig))
    context_menu.add_command(label="Reset Zoom", command=lambda: reset_zoom(fig))  # Add Reset Zoom option

    def reset_zoom(fig):
        global scale_factor
        scale_factor = 1.0  # Reset the scale factor

        for i, ax in enumerate(fig.get_axes()):
            ax.set_xlim(original_xlim[i])
            ax.set_ylim(original_ylim[i])
        fig.canvas.draw_idle()

    def on_right_click(event):
        context_menu.post(event.x_root, event.y_root)

    def on_hover(event):
        widget_width = event.widget.winfo_width()
        x_position = event.x

        if x_position < widget_width / 2:
            canvas_widget.config(cursor="hand2")
        else:
            canvas_widget.config(cursor="hand2")

    def on_leave(event):
        canvas_widget.config(cursor="arrow")

    def flash_feedback(side):
        flash = tk.Toplevel(canvas_widget.master)
        flash.overrideredirect(True)
        flash_width = int(canvas_widget.winfo_width() / 2)
        flash_height = canvas_widget.winfo_height()
        flash.configure(bg='white')
        flash.attributes('-alpha', 0.9)

        if side == "left":
            flash.geometry(f"{flash_width}x{flash_height}+{canvas_widget.winfo_rootx()}+{canvas_widget.winfo_rooty()}")
        else:
            flash.geometry(f"{flash_width}x{flash_height}+{canvas_widget.winfo_rootx() + flash_width}+{canvas_widget.winfo_rooty()}")

        flash.lift()

        # Ensure the flash covers the correct area only
        flash.update_idletasks()
        flash.after(100, flash.destroy)

    def on_click(event):
        widget_width = event.widget.winfo_width()
        x_position = event.x

        if x_position < widget_width / 2:
            #flash_feedback("left")
            show_previous_figure()
        else:
            #flash_feedback("right")
            show_next_figure()

    def zoom(event):
        nonlocal scale_factor

        zoom_speed = 0.1  # Adjust the zoom speed for smoother experience

        # Adjust zoom factor based on the operating system and mouse event
        if event.num == 4 or event.delta > 0:  # Scroll up
            scale_factor *= (1 + zoom_speed)
        elif event.num == 5 or event.delta < 0:  # Scroll down
            scale_factor /= (1 + zoom_speed)

        # Get mouse position relative to the figure
        x_mouse, y_mouse = event.x, event.y
        x_ratio = x_mouse / canvas_widget.winfo_width()
        y_ratio = y_mouse / canvas_widget.winfo_height()

        for ax in fig.get_axes():
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Calculate the new limits
            x_center = xlim[0] + x_ratio * (xlim[1] - xlim[0])
            y_center = ylim[0] + (1 - y_ratio) * (ylim[1] - ylim[0])

            x_range = (xlim[1] - xlim[0]) * scale_factor
            y_range = (ylim[1] - ylim[0]) * scale_factor

            ax.set_xlim([x_center - x_range * x_ratio, x_center + x_range * (1 - x_ratio)])
            ax.set_ylim([y_center - y_range * (1 - y_ratio), y_center + y_range * y_ratio])

        # Redraw the figure
        fig.canvas.draw_idle()

    # Bind events for hover, click interactions, and zoom
    canvas_widget.bind("<Motion>", on_hover)
    canvas_widget.bind("<Leave>", on_leave)
    canvas_widget.bind("<Button-1>", on_click)
    canvas_widget.bind("<Button-3>", on_right_click)

    # Bind mouse wheel for zooming (cross-platform)
    canvas_widget.bind("<MouseWheel>", zoom)  # Windows
    canvas_widget.bind("<Button-4>", zoom)  # Linux/macOS Scroll Up
    canvas_widget.bind("<Button-5>", zoom)  # Linux/macOS Scroll Down

def clear_unused_figures():
    global figures, figure_index

    lower_bound = max(0, figure_index - 20)
    upper_bound = min(len(figures), figure_index + 20)
    # Clear figures outside of the +/- 20 range
    figures = deque([fig for i, fig in enumerate(figures) if lower_bound <= i <= upper_bound])
    # Update the figure index after clearing
    figure_index = min(max(figure_index, 0), len(figures) - 1)

def show_previous_figure():
    global figure_index, figures, fig_queue
    if figure_index is not None and figure_index > 0:
        figure_index -= 1
        display_figure(figures[figure_index])
        clear_unused_figures()

def show_next_figure():
    global figure_index, figures, fig_queue
    if figure_index is not None and figure_index < len(figures) - 1:
        figure_index += 1
        display_figure(figures[figure_index])
        clear_unused_figures()
    elif figure_index == len(figures) - 1 and not fig_queue.empty():
        fig = fig_queue.get_nowait()
        figures.append(fig)
        figure_index += 1
        display_figure(fig)

def set_globals(thread_control_var, q_var, console_output_var, parent_frame_var, vars_dict_var, canvas_var, canvas_widget_var, scrollable_frame_var, fig_queue_var, figures_var, figure_index_var, progress_bar_var, usage_bars_var, fig_memory_limit_var, figure_current_memory_usage_var):
    global thread_control, q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, fig_queue, figures, figure_index, progress_bar, usage_bars, fig_memory_limit, figure_current_memory_usage
    thread_control = thread_control_var
    q = q_var
    console_output = console_output_var
    parent_frame = parent_frame_var
    vars_dict = vars_dict_var
    canvas = canvas_var
    canvas_widget = canvas_widget_var
    scrollable_frame = scrollable_frame_var
    fig_queue = fig_queue_var
    figures = figures_var
    figure_index = figure_index_var
    progress_bar = progress_bar_var
    usage_bars = usage_bars_var
    fig_memory_limit = fig_memory_limit_var
    figure_current_memory_usage = figure_current_memory_usage_var

def import_settings(settings_type='mask'):
    from .gui_utils import convert_settings_dict_for_gui, hide_all_settings
    global vars_dict, scrollable_frame, button_scrollable_frame
    from .settings import generate_fields, set_default_settings_preprocess_generate_masks, get_measure_crop_settings, set_default_train_test_model, set_default_generate_barecode_mapping, set_default_umap_image_settings, get_analyze_recruitment_default_settings

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
        settings = set_default_generate_barecode_mapping(settings={})
    elif settings_type == 'umap':
        settings = set_default_umap_image_settings(settings={})
    elif settings_type == 'recruitment':
        settings = get_analyze_recruitment_default_settings(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")
    
    variables = convert_settings_dict_for_gui(settings)
    new_settings = update_settings_from_csv(variables, csv_settings)
    vars_dict = generate_fields(new_settings, scrollable_frame)
    vars_dict = hide_all_settings(vars_dict, categories=None)

def setup_settings_panel(vertical_container, settings_type='mask'):
    global vars_dict, scrollable_frame
    from .settings import get_identify_masks_finetune_default_settings, set_default_analyze_screen, set_default_settings_preprocess_generate_masks, get_measure_crop_settings, deep_spacr_defaults, set_default_generate_barecode_mapping, set_default_umap_image_settings, generate_fields, get_perform_regression_default_settings, get_train_cellpose_default_settings, get_map_barcodes_default_settings, get_analyze_recruitment_default_settings, get_check_cellpose_models_default_settings
    from .gui_utils import convert_settings_dict_for_gui
    from .gui_elements import set_element_size

    size_dict = set_element_size()
    settings_width = size_dict['settings_width']

    # Create a PanedWindow for the settings panel
    settings_paned_window = tk.PanedWindow(vertical_container, orient=tk.HORIZONTAL, width=size_dict['settings_width'])
    vertical_container.add(settings_paned_window, stretch="always")

    settings_frame = tk.Frame(settings_paned_window, width=settings_width)
    settings_frame.pack_propagate(False)  # Prevent the frame from resizing based on its children

    settings_paned_window.add(settings_frame)

    scrollable_frame = spacrFrame(settings_frame)
    scrollable_frame.grid(row=1, column=0, sticky="nsew")
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)

    if settings_type == 'mask':
        settings = set_default_settings_preprocess_generate_masks(src='path', settings={})
    elif settings_type == 'measure':
        settings = get_measure_crop_settings(settings={})
    elif settings_type == 'classify':
        settings = deep_spacr_defaults(settings={})
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
        settings = set_default_generate_barecode_mapping(settings={})
    elif settings_type == 'regression':
        settings = get_perform_regression_default_settings(settings={})
    elif settings_type == 'recruitment':
        settings = get_analyze_recruitment_default_settings(settings={})
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")

    variables = convert_settings_dict_for_gui(settings)
    vars_dict = generate_fields(variables, scrollable_frame)
    
    containers = [settings_frame]
    widgets = [scrollable_frame]

    style = ttk.Style(vertical_container)
    _ = set_dark_style(style, containers=containers, widgets=widgets)

    print("Settings panel setup complete")
    return scrollable_frame, vars_dict

def setup_plot_section(vertical_container):
    global canvas, canvas_widget, figures, figure_index

    from .gui_elements import set_element_size

    # Initialize deque for storing figures and the current index
    figures = deque()
    figure_index = -1

    # Create a frame for the plot section
    plot_frame = tk.Frame(vertical_container)
    vertical_container.add(plot_frame, stretch="always")

    # Set up the plot
    figure = Figure(figsize=(30, 4), dpi=100)
    plot = figure.add_subplot(111)
    plot.plot([], [])
    plot.axis('off')
    
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().configure(cursor='arrow', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew")

    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    canvas.draw()
    canvas.figure = figure  # Ensure that the figure is linked to the canvas
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
    from .gui_elements import set_dark_style

    # Apply dark style and get style output
    style = ttk.Style()
    style_out = set_dark_style(style)

    # Create a frame for the console section
    console_frame = tk.Frame(vertical_container, bg=style_out['bg_color'])
    vertical_container.add(console_frame, stretch="always")

    # Create a thicker frame at the top for the hover effect
    top_border = tk.Frame(console_frame, height=5, bg=style_out['bg_color'])
    top_border.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    # Create the scrollable frame (which is a Text widget) with white text
    family = style_out['font_family']
    font_size = style_out['font_size']
    font_loader = style_out['font_loader']
    console_output = tk.Text(console_frame, bg=style_out['bg_color'], fg=style_out['fg_color'], font=font_loader.get_font(size=font_size), bd=0, highlightthickness=0)
    
    console_output.grid(row=1, column=0, sticky="nsew")  # Use grid for console_output

    # Configure the grid to allow expansion
    console_frame.grid_rowconfigure(1, weight=1)
    console_frame.grid_columnconfigure(0, weight=1)

    def on_enter(event):
        top_border.config(bg=style_out['active_color'])

    def on_leave(event):
        top_border.config(bg=style_out['bg_color'])

    console_output.bind("<Enter>", on_enter)
    console_output.bind("<Leave>", on_leave)

    return console_output, console_frame

def setup_progress_frame(vertical_container):
    global progress_output
    style_out = set_dark_style(ttk.Style())
    font_loader = style_out['font_loader']
    font_size = style_out['font_size']
    progress_frame = tk.Frame(vertical_container)
    vertical_container.add(progress_frame, stretch="always")
    label_frame = tk.Frame(progress_frame)
    label_frame.grid(row=0, column=0, sticky="ew", pady=(5, 0), padx=10)
    progress_label = spacrLabel(label_frame, text="Processing: 0%", font=font_loader.get_font(size=font_size), anchor='w', justify='left', align="left")
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
    global thread_control, parent_frame, button_frame, button_scrollable_frame, run_button, abort_button, download_dataset_button, import_button, q, fig_queue, vars_dict, progress_bar
    from .gui_utils import download_hug_dataset
    from .gui_elements import set_element_size

    size_dict = set_element_size()
    button_section_height = size_dict['panel_height']
    button_frame = tk.Frame(horizontal_container, height=button_section_height)
    
    horizontal_container.add(button_frame, stretch="always", sticky="nsew")
    button_scrollable_frame = spacrFrame(button_frame, scrollbar=False)
    button_scrollable_frame.grid(row=1, column=0, sticky="nsew")
    widgets = [button_scrollable_frame.scrollable_frame]

    btn_col = 0
    btn_row = 0

    if run:
        run_button = spacrButton(button_scrollable_frame.scrollable_frame, text="run", command=lambda: start_process(q, fig_queue, settings_type), show_text=False, size=size_dict['btn_size'], animation=False)
        run_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(run_button)
        btn_col += 1

    if abort and settings_type in ['mask', 'measure', 'classify', 'sequencing', 'umap', 'map_barcodes']:
        abort_button = spacrButton(button_scrollable_frame.scrollable_frame, text="abort", command=lambda: initiate_abort(), show_text=False, size=size_dict['btn_size'], animation=False)
        abort_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(abort_button)
        btn_col += 1

    if download and settings_type in ['mask']:
        download_dataset_button = spacrButton(button_scrollable_frame.scrollable_frame, text="download", command=lambda: download_hug_dataset(q, vars_dict), show_text=False, size=size_dict['btn_size'], animation=False)
        download_dataset_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(download_dataset_button)
        btn_col += 1

    if import_btn:
        import_button = spacrButton(button_scrollable_frame.scrollable_frame, text="settings", command=lambda: import_settings(settings_type), show_text=False, size=size_dict['btn_size'], animation=False)
        import_button.grid(row=btn_row, column=btn_col, pady=5, padx=5, sticky='ew')
        widgets.append(import_button)
        btn_row += 1

    # Add the batch progress bar
    progress_bar = spacrProgressBar(button_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate')
    progress_bar.grid(row=btn_row, column=0, columnspan=7, pady=5, padx=5, sticky='ew')
    progress_bar.set_label_position()  # Set the label position after grid placement
    widgets.append(progress_bar)

    if vars_dict is not None:
        toggle_settings(button_scrollable_frame)

    style = ttk.Style(horizontal_container)
    _ = set_dark_style(style, containers=[button_frame], widgets=widgets)

    return button_scrollable_frame, btn_col

def setup_usage_panel(horizontal_container, btn_col, uppdate_frequency):
    global usage_bars
    from .gui_elements import set_dark_style, set_element_size

    usg_col = 1

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
        parent_frame.after(uppdate_frequency, update_usage, ram_bar, vram_bar, gpu_bar, usage_bars, parent_frame)

    size_dict = set_element_size()
    usage_panel_height = size_dict['panel_height']
    usage_frame = tk.Frame(horizontal_container, height=usage_panel_height)
    horizontal_container.add(usage_frame)

    usage_frame.grid_rowconfigure(0, weight=0)
    usage_frame.grid_rowconfigure(1, weight=1)
    usage_frame.grid_columnconfigure(0, weight=1)
    usage_frame.grid_columnconfigure(1, weight=1)

    usage_scrollable_frame = spacrFrame(usage_frame, scrollbar=False)
    usage_scrollable_frame.grid(row=1, column=0, sticky="nsew", columnspan=2)
    widgets = [usage_scrollable_frame.scrollable_frame]
    
    usage_bars = []
    max_elements_per_column = 6
    row = 0
    col = 0

    # Initialize RAM, VRAM, and GPU bars as None
    ram_bar, vram_bar, gpu_bar = None, None, None
    
    # Configure the style for the label
    style = ttk.Style()
    style_out = set_dark_style(style)
    font_loader = style_out['font_loader']
    font_size = style_out['font_size'] - 2
    style.configure("usage.TLabel", font=font_loader.get_font(size=font_size), foreground=style_out['fg_color'])

    # Try adding RAM bar
    try:
        ram_info = psutil.virtual_memory()
        ram_label_text = f"RAM"
        label = tk.Label(usage_scrollable_frame.scrollable_frame,text=ram_label_text,anchor='w',font=font_loader.get_font(size=font_size),bg=style_out['bg_color'],fg=style_out['fg_color'])
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
            label = tk.Label(usage_scrollable_frame.scrollable_frame,text=vram_label_text,anchor='w',font=font_loader.get_font(size=font_size),bg=style_out['bg_color'],fg=style_out['fg_color'])
            label.grid(row=row, column=2 * col, pady=5, padx=5, sticky='w')
            vram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
            vram_bar.grid(row=row, column=2 * col + 1, pady=5, padx=5, sticky='ew')
            widgets.append(label)
            widgets.append(vram_bar)
            usage_bars.append(vram_bar)
            row += 1

            gpu_label_text = f"GPU"
            label = tk.Label(usage_scrollable_frame.scrollable_frame,text=gpu_label_text,anchor='w',font=font_loader.get_font(size=font_size),bg=style_out['bg_color'],fg=style_out['fg_color'])
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

            label = tk.Label(usage_scrollable_frame.scrollable_frame,text=f"C{core+1}",anchor='w',font=font_loader.get_font(size=font_size),bg=style_out['bg_color'],fg=style_out['fg_color'])
            label.grid(row=row, column=2 * col, pady=2, padx=5, sticky='w')
            bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
            bar.grid(row=row, column=2 * col + 1, pady=2, padx=5, sticky='ew')
            widgets.append(label)
            widgets.append(bar)
            usage_bars.append(bar)
            row += 1
    except Exception as e:
        print(f"Could not add CPU core usage bars: {e}")

    style = ttk.Style(horizontal_container)
    _ = set_dark_style(style, containers=[usage_frame], widgets=widgets)

    if ram_bar is None:
        ram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
    if vram_bar is None:
        vram_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
    if gpu_bar is None:
        gpu_bar = spacrProgressBar(usage_scrollable_frame.scrollable_frame, orient='horizontal', mode='determinate', length=size_dict['bar_size'], label=False)
    
    update_usage(ram_bar, vram_bar, gpu_bar, usage_bars, usage_frame)    
    return usage_scrollable_frame, usage_bars, usg_col

def initiate_abort():
    global thread_control, q, parent_frame
    if thread_control.get("run_thread") is not None:
        try:
            q.put("Aborting processes...")
            thread_control.get("run_thread").terminate()
            thread_control["run_thread"] = None
            q.put("Processes aborted.")
        except Exception as e:
            q.put(f"Error aborting process: {e}")

    thread_control = {"run_thread": None, "stop_requested": False}

def start_process(q=None, fig_queue=None, settings_type='mask'):
    global thread_control, vars_dict, parent_frame
    from .settings import check_settings, expected_types
    from .gui_utils import run_function_gui, set_high_priority, set_cpu_affinity, initialize_cuda

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

    # Initialize CUDA in the main process
    initialize_cuda()

    process_args = (settings_type, settings, q, fig_queue, stop_requested)
    if settings_type in ['mask', 'umap', 'measure', 'simulation', 'sequencing', 
                         'classify', 'cellpose_dataset', 'train_cellpose', 'ml_analyze', 'cellpose_masks', 'cellpose_all', 'map_barcodes', 
                         'regression', 'recruitment', 'plaques', 'cellpose_compare', 'vision_scores', 'vision_dataset']:

        # Start the process
        process = Process(target=run_function_gui, args=process_args)
        process.start()

        # Set high priority for the process
        #set_high_priority(process)

        # Set CPU affinity if necessary
        set_cpu_affinity(process)

        # Store the process in thread_control for future reference
        thread_control["run_thread"] = process
    else:
        q.put(f"Error: Unknown settings type '{settings_type}'")
        return

def process_console_queue():
    global q, console_output, parent_frame, progress_bar, process_console_queue

    # Initialize function attribute if it doesn't exist
    if not hasattr(process_console_queue, "completed_tasks"):
        process_console_queue.completed_tasks = []
    if not hasattr(process_console_queue, "current_maximum"):
        process_console_queue.current_maximum = None

    ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    while not q.empty():
        message = q.get_nowait()
        clean_message = ansi_escape_pattern.sub('', message)
        #console_output.insert(tk.END, clean_message + "\n")
        #console_output.see(tk.END)
        
        # Check if the message contains progress information
        if clean_message.startswith("Progress:"):
            try:
                # Extract the progress information
                match = re.search(r'Progress: (\d+)/(\d+), operation_type: ([\w\s]*),(.*)', clean_message)

                if match:
                    current_progress = int(match.group(1))
                    total_progress = int(match.group(2))
                    operation_type = match.group(3).strip()
                    additional_info = match.group(4).strip()  # Capture everything after operation_type
                    
                    # Check if the maximum value has changed
                    if process_console_queue.current_maximum != total_progress:
                        process_console_queue.current_maximum = total_progress
                        process_console_queue.completed_tasks = []

                    # Add the task to the completed set
                    process_console_queue.completed_tasks.append(current_progress)
                    
                    # Calculate the unique progress count
                    unique_progress_count = len(np.unique(process_console_queue.completed_tasks))

                    # Update the progress bar
                    if progress_bar:
                        progress_bar['maximum'] = total_progress
                        progress_bar['value'] = unique_progress_count

                    # Store operation type and additional info
                    if operation_type:
                        progress_bar.operation_type = operation_type
                        progress_bar.additional_info = additional_info

                    # Update the progress label
                    if progress_bar.progress_label:
                        progress_bar.update_label()

                    # Clear completed tasks when progress is complete
                    if unique_progress_count >= total_progress:
                        process_console_queue.completed_tasks.clear()

            except Exception as e:
                print(f"Error parsing progress message: {e}")
        else:
            # Only insert messages that do not start with "Progress:"
            console_output.insert(tk.END, clean_message + "\n")
            console_output.see(tk.END)
        
    after_id = console_output.after(uppdate_frequency, process_console_queue)
    parent_frame.after_tasks.append(after_id)

def main_thread_update_function(root, q, fig_queue, canvas_widget):
    global uppdate_frequency
    try:
        while not q.empty():
            message = q.get_nowait()
    except Exception as e:
        print(f"Error updating GUI canvas: {e}")
    finally:
        root.after(uppdate_frequency, lambda: main_thread_update_function(root, q, fig_queue, canvas_widget))

def initiate_root(parent, settings_type='mask'):
    """
    Initializes the root window and sets up the GUI components based on the specified settings type.

    Args:
        parent (tkinter.Tk or tkinter.Toplevel): The parent window for the GUI.
        settings_type (str, optional): The type of settings to be displayed in the GUI. Defaults to 'mask'.

    Returns:
        tuple: A tuple containing the parent frame and the dictionary of variables used in the GUI.
    """
    
    global q, fig_queue, thread_control, parent_frame, scrollable_frame, button_frame, vars_dict, canvas, canvas_widget, button_scrollable_frame, progress_bar, uppdate_frequency, figures, figure_index, fig_memory_limit, figure_current_memory_usage
    
    from .gui_utils import setup_frame
    from .settings import descriptions

    uppdate_frequency = 100

    # Start tracemalloc and initialize global variables
    tracemalloc.start()

    set_start_method('spawn', force=True)
    #set_start_method('forkserver', force=True)
    print("Initializing root with settings_type:", settings_type)

    # Initialize global variables
    figures = deque()
    figure_index = -1
    fig_memory_limit = 200 * 1024 * 1024  # 200 MB limit
    figure_current_memory_usage = 0

    parent_frame = parent

    if not isinstance(parent_frame, (tk.Tk, tk.Toplevel)):
        parent_window = parent_frame.winfo_toplevel()
    else:
        parent_window = parent_frame

    parent_window.update_idletasks()

    if not hasattr(parent_window, 'after_tasks'):
        parent_window.after_tasks = []

    q = Queue()
    fig_queue = Queue()
    parent_frame, vertical_container, horizontal_container, settings_container = setup_frame(parent_frame)

    if settings_type == 'annotate':
        from .app_annotate import initiate_annotation_app
        initiate_annotation_app(horizontal_container)
    elif settings_type == 'make_masks':
        from .app_make_masks import initiate_make_mask_app
        initiate_make_mask_app(horizontal_container)
    else:
        scrollable_frame, vars_dict = setup_settings_panel(settings_container, settings_type)
        print('setup_settings_panel')
        canvas, canvas_widget = setup_plot_section(vertical_container)
        console_output, _ = setup_console(vertical_container)
        button_scrollable_frame, btn_col = setup_button_section(horizontal_container, settings_type)
        _, usage_bars, btn_col = setup_usage_panel(horizontal_container, btn_col, uppdate_frequency)

        set_globals(thread_control, q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, fig_queue, figures, figure_index, progress_bar, usage_bars, fig_memory_limit, figure_current_memory_usage)
        description_text = descriptions.get(settings_type, "No description available for this module.")
        
        q.put(f"Console")
        q.put(f" ")
        q.put(description_text)
        
        process_console_queue()
        process_fig_queue()
        after_id = parent_window.after(uppdate_frequency, lambda: main_thread_update_function(parent_window, q, fig_queue, canvas_widget))
        parent_window.after_tasks.append(after_id)

    print("Root initialization complete")
    return parent_frame, vars_dict