#import customtkinter as ctk

import sys, ctypes, matplotlib
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('Agg')
from tkinter import filedialog
from multiprocessing import Process, Queue, Value
import traceback

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .logger import log_function_call
from .gui_utils import ScrollableFrame, StdoutRedirector, ToggleSwitch, CustomButton, ToolTip
from .gui_utils import clear_canvas, main_thread_update_function, set_dark_style, generate_fields, process_stdout_stderr, set_default_font, style_text_boxes
from .gui_utils import mask_variables, check_mask_gui_settings, preprocess_generate_masks_wrapper, read_settings_from_csv, update_settings_from_csv, create_menu_bar

thread_control = {"run_thread": None, "stop_requested": False}

def toggle_test_mode():
    global vars_dict
    current_state = vars_dict['test_mode'][2].get()
    new_state = not current_state
    vars_dict['test_mode'][2].set(new_state)
    if new_state:
        test_mode_button.config(bg="blue")
    else:
        test_mode_button.config(bg="gray")

def toggle_advanced_settings():
    global vars_dict

    timelapse_settings = ['timelapse', 'timelapse_memory', 'timelapse_remove_transient', 'timelapse_mode', 'timelapse_objects', 'timelapse_displacement', 'timelapse_frame_limits', 'fps']
    misc_settings = ['examples_to_plot', 'all_to_mip', 'pick_slice', 'skip_mode']
    opperational_settings = ['preprocess', 'masks', 'randomize', 'batch_size', 'custom_regex', 'merge', 'normalize_plots', 'workers', 'plot', 'remove_background', 'lower_quantile']

    advanced_settings = timelapse_settings+misc_settings+opperational_settings

    # Toggle visibility of advanced settings
    for setting in advanced_settings:
        label, widget, var = vars_dict[setting]
        if advanced_var.get() is False:
            label.grid_remove()  # Hide the label
            widget.grid_remove()  # Hide the widget
        else:
            label.grid()  # Show the label
            widget.grid()  # Show the widget

#@log_function_call
def initiate_abort():
    global thread_control
    if thread_control.get("stop_requested") is not None:
        thread_control["stop_requested"].value = 1

    if thread_control.get("run_thread") is not None:
        thread_control["run_thread"].join(timeout=5)
        if thread_control["run_thread"].is_alive():
            thread_control["run_thread"].terminate()
        thread_control["run_thread"] = None

#@log_function_call
def run_mask_gui(q, fig_queue, stop_requested):
    global vars_dict
    process_stdout_stderr(q)
    try:
        settings = check_mask_gui_settings(vars_dict)
        preprocess_generate_masks_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

#@log_function_call
def start_process(q, fig_queue):
    global thread_control
    if thread_control.get("run_thread") is not None:
        initiate_abort()

    stop_requested = Value('i', 0)  # multiprocessing shared value for inter-process communication
    thread_control["stop_requested"] = stop_requested
    thread_control["run_thread"] = Process(target=run_mask_gui, args=(q, fig_queue, stop_requested))
    thread_control["run_thread"].start()

def import_settings(scrollable_frame):
    global vars_dict

    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    csv_settings = read_settings_from_csv(csv_file_path)
    variables = mask_variables()
    new_settings = update_settings_from_csv(variables, csv_settings)
    vars_dict = generate_fields(new_settings, scrollable_frame)

#@log_function_call
def initiate_mask_root(parent_frame, width, height):
    global vars_dict, q, canvas, fig_queue, canvas_widget, thread_control, advanced_var, scrollable_frame

    style = ttk.Style(parent_frame)
    set_dark_style(style)
    style_text_boxes(style)
    set_default_font(parent_frame, font_name="Arial", size=8)
    parent_frame.configure(bg='black')
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)
    
    fig_queue = Queue()

    def _process_fig_queue():
        global canvas
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
            canvas_widget.after(100, _process_fig_queue)

    def _process_console_queue():
        while not q.empty():
            message = q.get_nowait()
            console_output.insert(tk.END, message)
            console_output.see(tk.END)
        console_output.after(100, _process_console_queue)

    vertical_container = tk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
    vertical_container.grid(row=0, column=0, sticky=tk.NSEW)
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)

    # Settings Section
    settings_frame = tk.Frame(vertical_container, bg='black')
    vertical_container.add(settings_frame, stretch="always")
    settings_label = ttk.Label(settings_frame, text="Settings", style="Custom.TLabel")
    settings_label.grid(row=0, column=0, pady=10, padx=10)
    scrollable_frame = ScrollableFrame(settings_frame, width=600)
    scrollable_frame.grid(row=1, column=0, sticky="nsew")
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)
    
    # Create advanced settings checkbox
    advanced_var = tk.BooleanVar(value=False)
    advanced_Toggle = ToggleSwitch(scrollable_frame.scrollable_frame, text="Advanced Settings", variable=advanced_var, command=toggle_advanced_settings)
    advanced_Toggle.grid(row=48, column=0, pady=10, padx=10)
    variables = mask_variables()
    vars_dict = generate_fields(variables, scrollable_frame)
    toggle_advanced_settings()
    vars_dict['Test mode'] = (None, None, tk.BooleanVar(value=False))
    
    # Button section
    test_mode_button = CustomButton(scrollable_frame.scrollable_frame, text="Test Mode", command=toggle_test_mode)
    test_mode_button.grid(row=47, column=1, pady=10, padx=10)
    import_btn = CustomButton(scrollable_frame.scrollable_frame, text="Import", command=lambda: import_settings(scrollable_frame))
    import_btn.grid(row=47, column=0, pady=10, padx=10)
    run_button = CustomButton(scrollable_frame.scrollable_frame, text="Run", command=lambda: start_process(q, fig_queue))
    run_button.grid(row=45, column=0, pady=10, padx=10)
    abort_button = CustomButton(scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort)
    abort_button.grid(row=45, column=1, pady=10, padx=10)
    progress_label = ttk.Label(scrollable_frame.scrollable_frame, text="Processing: 0%", background="black", foreground="white") # Create progress field
    progress_label.grid(row=50, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=10)

    # Plot Canvas Section
    plot_frame = tk.PanedWindow(vertical_container, orient=tk.VERTICAL) # Horizontal container for Matplotlib figure and the vertical pane (for settings and console)
    vertical_container.add(plot_frame, stretch="always")
    figure = Figure(figsize=(30, 4), dpi=100, facecolor='black') # Matplotlib figure setup
    plot = figure.add_subplot(111)
    plot.plot([], [])  # This creates an empty plot.
    plot.axis('off')
    canvas = FigureCanvasTkAgg(figure, master=plot_frame) # Embedd Matplotlib figure in Tkinter window
    canvas.get_tk_widget().configure(cursor='arrow', background='black', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    plot_frame.add(canvas_widget, stretch="always")
    canvas.draw()
    canvas.figure = figure

    # Console Section
    console_frame = tk.Frame(vertical_container, bg='black')
    vertical_container.add(console_frame, stretch="always")
    console_label = ttk.Label(console_frame, text="Console", background="black", foreground="white")
    console_label.grid(row=0, column=0, pady=10, padx=10)
    console_output = scrolledtext.ScrolledText(console_frame, height=10, bg='black', fg='white', insertbackground='white')
    console_output.grid(row=1, column=0, sticky="nsew")
    console_frame.grid_rowconfigure(1, weight=1)
    console_frame.grid_columnconfigure(0, weight=1)

    q = Queue()
    sys.stdout = StdoutRedirector(console_output)
    sys.stderr = StdoutRedirector(console_output)

    _process_console_queue()
    _process_fig_queue()
    
    parent_frame.after(100, lambda: main_thread_update_function(parent_frame, q, fig_queue, canvas_widget, progress_label))

    return parent_frame, vars_dict

def gui_mask():
    root = tk.Tk()
    root.geometry("1000x800")
    root.title("SpaCer: generate masks")
    initiate_mask_root(root, 1000, 800)
    create_menu_bar(root)
    root.mainloop()

if __name__ == "__main__":
    gui_mask()