import os, traceback, ctypes, csv, re, platform
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Process, Value, Queue, set_start_method
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import psutil
import GPUtil
from collections import deque
import tracemalloc

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .gui_elements import spacrProgressBar, spacrButton, spacrFrame, spacrDropdownMenu , spacrSlider, set_dark_style

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
index_control = None

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

def display_figure(fig):
    global canvas, canvas_widget

    from .gui_elements import save_figure_as_format, modify_figure

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

        # Determine the zoom direction based on the scroll event
        if event.num == 4 or event.delta > 0:  # Scroll up (zoom in)
            scale_factor /= (1 + zoom_speed)  # Divide to zoom in
        elif event.num == 5 or event.delta < 0:  # Scroll down (zoom out)
            scale_factor *= (1 + zoom_speed)  # Multiply to zoom out

        # Adjust the axes limits based on the new scale factor
        for ax in canvas.figure.get_axes():
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2

            x_range = (xlim[1] - xlim[0]) * scale_factor
            y_range = (ylim[1] - ylim[0]) * scale_factor

            # Set the new limits
            ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
            ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])

        # Redraw the figure efficiently
        canvas.draw_idle()


    # Bind events for hover, click interactions, and zoom
    canvas_widget.bind("<Motion>", on_hover)
    canvas_widget.bind("<Leave>", on_leave)
    canvas_widget.bind("<Button-1>", on_click)
    canvas_widget.bind("<Button-3>", on_right_click)


    # Detect the operating system and bind the appropriate mouse wheel events
    current_os = platform.system()

    if current_os == "Windows":
        canvas_widget.bind("<MouseWheel>", zoom)  # Windows
    elif current_os == "Darwin":  # macOS
        canvas_widget.bind("<MouseWheel>", zoom)
        canvas_widget.bind("<Button-4>", zoom)  # Scroll up
        canvas_widget.bind("<Button-5>", zoom)  # Scroll down
    elif current_os == "Linux":
        canvas_widget.bind("<Button-4>", zoom)  # Linux Scroll up
        canvas_widget.bind("<Button-5>", zoom)  # Linux Scroll down

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
                       
def process_fig_queue():
    global canvas, fig_queue, canvas_widget, parent_frame, uppdate_frequency, figures, figure_index, index_control

    from .gui_elements import standardize_figure
    try:
        while not fig_queue.empty():
            fig = fig_queue.get_nowait()

            if fig is None:
                print("Warning: Retrieved a None figure from fig_queue.")
                continue  # Skip processing if the figure is None

            # Standardize the figure appearance before adding it to the list
            fig = standardize_figure(fig)

            figures.append(fig)

            # Update the slider range and set the value to the latest figure index
            index_control.set_to(len(figures) - 1)

            if figure_index == -1:
                figure_index += 1
                display_figure(figures[figure_index])
                index_control.set(figure_index)
            
    except Exception as e:
        traceback.print_exc()
    finally:
        after_id = canvas_widget.after(uppdate_frequency, process_fig_queue)
        parent_frame.after_tasks.append(after_id)

def update_figure(value):
    global figure_index, figures

    # Convert the value to an integer
    index = int(value)
    
    # Check if the index is valid
    if 0 <= index < len(figures):
        figure_index = index
        display_figure(figures[figure_index])

    # Update the index control widget's range and value
    index_control.set_to(len(figures) - 1)
    index_control.set(figure_index)

def setup_plot_section(vertical_container, settings_type):
    global canvas, canvas_widget, figures, figure_index, index_control
    from .gui_utils import display_media_in_plot_frame

    style_out = set_dark_style(ttk.Style())
    bg = style_out['bg_color']
    fg = style_out['fg_color']

    # Initialize deque for storing figures and the current index
    figures = deque()

    # Create a frame for the plot section
    plot_frame = tk.Frame(vertical_container)
    plot_frame.configure(bg=bg)
    vertical_container.add(plot_frame, stretch="always")

    # Clear the plot_frame (optional, to handle cases where it may already have content)
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Create a figure and plot
    figure = Figure(figsize=(30, 4), dpi=100)
    plot = figure.add_subplot(111)
    plot.plot([], [])
    plot.axis('off')

    if settings_type == 'map_barcodes':
        # Load and display GIF
        current_dir = os.path.dirname(__file__)
        resources_path = os.path.join(current_dir, 'resources', 'icons')
        gif_path = os.path.join(resources_path, 'dna_matrix.mp4')

        display_media_in_plot_frame(gif_path, plot_frame)
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.get_tk_widget().configure(cursor='arrow', highlightthickness=0)
        canvas_widget = canvas.get_tk_widget()
        return canvas, canvas_widget
    
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().configure(cursor='arrow', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew")
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)
    canvas.draw()
    canvas.figure = figure
    figure.patch.set_facecolor(bg)
    plot.set_facecolor(bg)
    containers = [plot_frame]

    # Create slider
    control_frame = tk.Frame(plot_frame, height=15*2,  bg=bg)
    control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    control_frame.grid_propagate(False) 

    # Pass the update_figure function as the command to spacrSlider
    index_control = spacrSlider(control_frame, from_=0, to=0, value=0, thickness=2, knob_radius=10, position="center", show_index=True, command=update_figure)
    index_control.grid(row=0, column=0, sticky="ew")
    control_frame.grid_columnconfigure(0, weight=1)

    widgets = [canvas_widget, index_control]
    style = ttk.Style(vertical_container)
    _ = set_dark_style(style, containers=containers, widgets=widgets)

    return canvas, canvas_widget

def set_globals(thread_control_var, q_var, console_output_var, parent_frame_var, vars_dict_var, canvas_var, canvas_widget_var, scrollable_frame_var, fig_queue_var, figures_var, figure_index_var, index_control_var, progress_bar_var, usage_bars_var):
    global thread_control, q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, fig_queue, figures, figure_index, progress_bar, usage_bars, index_control
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
    index_control = index_control_var

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
    elif settings_type == 'analyze_plaques':
        settings = {}
    elif settings_type == 'convert':
        settings = {}
    else:
        raise ValueError(f"Invalid settings type: {settings_type}")
    
    variables = convert_settings_dict_for_gui(settings)
    new_settings = update_settings_from_csv(variables, csv_settings)
    vars_dict = generate_fields(new_settings, scrollable_frame)
    vars_dict = hide_all_settings(vars_dict, categories=None)

def setup_settings_panel(vertical_container, settings_type='mask'):
    global vars_dict, scrollable_frame
    from .settings import get_identify_masks_finetune_default_settings, set_default_analyze_screen, set_default_settings_preprocess_generate_masks, get_measure_crop_settings, deep_spacr_defaults, set_default_generate_barecode_mapping, set_default_umap_image_settings
    from .settings import get_map_barcodes_default_settings, get_analyze_recruitment_default_settings, get_check_cellpose_models_default_settings, generate_fields, get_perform_regression_default_settings, get_train_cellpose_default_settings
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
    elif settings_type == 'analyze_plaques':
        settings = {'src':'path to images'}
    elif settings_type == 'convert':
        settings = {'src':'path to images'}
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

    #def on_enter_key(event):
    #    user_input = console_output.get("1.0", "end-1c").strip()  # Get the user input from the console
    #    if user_input:
    #        # Print the user input with the (user) tag
    #        console_output.insert("end", f"\n(user): {user_input}\n")
    #        
    #        # Get the AI response from the chatbot
    #        response = chatbot.ask_question(user_input)
    #        
    #        # Print the AI response with the (ai) tag
    #        console_output.insert("end", f"(ai): {response}\n")
    #        
    #        console_output.see("end")  # Scroll to the end
    #        #console_output.delete("1.0", "end")  # Clear the input field
    #    return "break"  # Prevent the default behavior of inserting a new line

    console_output.bind("<Enter>", on_enter)
    console_output.bind("<Leave>", on_leave)

    #console_output.bind("<Return>", on_enter_key)

    return console_output, console_frame

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
    max_elements_per_column = 5
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
    from .gui_utils import run_function_gui, set_cpu_affinity, initialize_cuda, display_gif_in_plot_frame, print_widget_structure

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
    if settings_type in ['mask', 'umap', 'measure', 'simulation', 'sequencing', 'classify', 'analyze_plaques', 
                         'cellpose_dataset', 'train_cellpose', 'ml_analyze', 'cellpose_masks', 'cellpose_all', 
                         'map_barcodes', 'regression', 'recruitment', 'cellpose_compare', 'vision_scores', 
                         'vision_dataset', 'convert']:

        # Start the process
        process = Process(target=run_function_gui, args=process_args)
        process.start()

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
    
    global q, fig_queue, thread_control, parent_frame, scrollable_frame, button_frame, vars_dict, canvas, canvas_widget, button_scrollable_frame, progress_bar, uppdate_frequency, figures, figure_index, index_control, usage_bars
    
    from .gui_utils import setup_frame, get_screen_dimensions
    from .settings import descriptions
    #from .openai import Chatbot

    uppdate_frequency = 500
    num_cores = os.cpu_count()

    #chatbot = Chatbot(api_key="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")

    # Start tracemalloc and initialize global variables
    tracemalloc.start()

    set_start_method('spawn', force=True)
    #set_start_method('forkserver', force=True)
    print("Initializing root with settings_type:", settings_type)

    # Initialize global variables
    figures = deque()
    figure_index = -1

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
        canvas, canvas_widget = setup_plot_section(vertical_container, settings_type)
        console_output, _ = setup_console(vertical_container) #, chatbot)
        button_scrollable_frame, btn_col = setup_button_section(horizontal_container, settings_type)
        
        if num_cores > 12:
            _, usage_bars, btn_col = setup_usage_panel(horizontal_container, btn_col, uppdate_frequency)
        else:
            usage_bars = []

        set_globals(thread_control, q, console_output, parent_frame, vars_dict, canvas, canvas_widget, scrollable_frame, fig_queue, figures, figure_index, index_control, progress_bar, usage_bars)
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


