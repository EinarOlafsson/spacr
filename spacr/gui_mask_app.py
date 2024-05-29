import sys, ctypes, matplotlib
import tkinter as tk
from tkinter import ttk, scrolledtext
from ttkthemes import ThemedTk
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
from .gui_utils import ScrollableFrame, StdoutRedirector, clear_canvas, main_thread_update_function, create_dark_mode, set_dark_style, generate_fields, process_stdout_stderr, set_default_font, style_text_boxes
from .gui_utils import mask_variables, check_mask_gui_settings, preprocess_generate_masks_wrapper, read_settings_from_csv, update_settings_from_csv

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
    advanced_settings = ['preprocess', 'masks', 'examples_to_plot', 'randomize', 'batch_size', 'timelapse', 'timelapse_memory', 'timelapse_remove_transient', 'timelapse_mode', 'timelapse_objects', 'fps', 'remove_background', 'lower_quantile', 'merge', 'normalize_plots', 'all_to_mip', 'pick_slice', 'skip_mode', 'workers', 'plot']
    
    # Toggle visibility of advanced settings
    for setting in advanced_settings:
        label, widget, var = vars_dict[setting]
        if advanced_var.get() is False:
            label.grid_remove()  # Hide the label
            widget.grid_remove()  # Hide the widget
        else:
            label.grid()  # Show the label
            widget.grid()  # Show the widget

@log_function_call
def initiate_abort():
    global thread_control
    if thread_control.get("stop_requested") is not None:
        thread_control["stop_requested"].value = 1

    if thread_control.get("run_thread") is not None:
        thread_control["run_thread"].join(timeout=5)
        if thread_control["run_thread"].is_alive():
            thread_control["run_thread"].terminate()
        thread_control["run_thread"] = None

@log_function_call
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

@log_function_call
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

@log_function_call
def initiate_mask_root(width, height):
    global root, vars_dict, q, canvas, fig_queue, canvas_widget, thread_control, advanced_widgets, advanced_var, scrollable_frame

    theme = 'breeze'

    if theme in ['clam']:
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use(theme)
        set_dark_style(style)
    elif theme in ['breeze']:
        root = ThemedTk(theme="breeze")
        style = ttk.Style(root)
        set_dark_style(style)

    style_text_boxes(style)
    set_default_font(root, font_name="Arial", size=8)
    root.attributes('-fullscreen', True)
    root.geometry(f"{width}x{height}")
    root.title("SpaCer: generate masks")
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
                fig.set_facecolor('#333333')
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

    vertical_container = tk.PanedWindow(root, orient=tk.HORIZONTAL)
    vertical_container.grid(row=0, column=0, sticky=tk.NSEW)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    scrollable_frame = ScrollableFrame(vertical_container, bg='#333333')
    vertical_container.add(scrollable_frame, stretch="always")

    advanced_var = tk.BooleanVar(value=False)
    advanced_checkbox = ttk.Checkbutton(scrollable_frame.scrollable_frame, text="Advanced Settings", variable=advanced_var, command=toggle_advanced_settings)
    advanced_checkbox.grid(row=46, column=1, pady=10, padx=10)

    variables = mask_variables()
    vars_dict = generate_fields(variables, scrollable_frame)
    toggle_advanced_settings()

    vars_dict['Test mode'] = (None, None, tk.BooleanVar(value=False))

    # Create test_mode button
    test_mode_button = tk.Button(scrollable_frame.scrollable_frame, text="Test Mode", command=toggle_test_mode, bg="gray")
    test_mode_button.grid(row=47, column=1, pady=10, padx=10)

    import_btn = tk.Button(scrollable_frame.scrollable_frame, text="Import Settings", command=lambda: import_settings(scrollable_frame))
    import_btn.grid(row=47, column=0, pady=10, padx=10)

    horizontal_container = tk.PanedWindow(vertical_container, orient=tk.VERTICAL)
    vertical_container.add(horizontal_container, stretch="always")

    figure = Figure(figsize=(30, 4), dpi=100, facecolor='#333333')
    plot = figure.add_subplot(111)
    plot.plot([], [])  # This creates an empty plot.
    plot.axis('off')

    canvas = FigureCanvasTkAgg(figure, master=horizontal_container)
    canvas.get_tk_widget().configure(cursor='arrow', background='#333333', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    horizontal_container.add(canvas_widget, stretch="always")
    canvas.draw()
    canvas.figure = figure

    console_output = scrolledtext.ScrolledText(vertical_container, height=10)
    vertical_container.add(console_output, stretch="always")

    q = Queue()
    sys.stdout = StdoutRedirector(console_output)
    sys.stderr = StdoutRedirector(console_output)

    run_button = ttk.Button(scrollable_frame.scrollable_frame, text="Run",command=lambda: start_process(q, fig_queue))
    run_button.grid(row=45, column=0, pady=10, padx=10)

    abort_button = ttk.Button(scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort)
    abort_button.grid(row=45, column=1, pady=10, padx=10)

    progress_label = ttk.Label(scrollable_frame.scrollable_frame, text="Processing: 0%", background="#333333", foreground="white")
    progress_label.grid(row=50, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=10)

    _process_console_queue()
    _process_fig_queue()
    create_dark_mode(root, style, console_output)

    root.after(100, lambda: main_thread_update_function(root, q, fig_queue, canvas_widget, progress_label))

    return root, vars_dict

def gui_mask():
    global vars_dict, root
    root, vars_dict = initiate_mask_root(1000, 1500)
    root.mainloop()

if __name__ == "__main__":
    gui_mask()
