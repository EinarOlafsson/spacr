import sys, ctypes, matplotlib
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
matplotlib.use('Agg')
from tkinter import filedialog
from multiprocessing import Process, Queue, Value
import traceback

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

from .logger import log_function_call
from .gui_utils import ScrollableFrame, StdoutRedirector, CustomButton, set_dark_style, set_default_font, generate_fields, process_stdout_stderr, clear_canvas, main_thread_update_function
from .gui_utils import classify_variables, check_classify_gui_settings, train_test_model_wrapper, read_settings_from_csv, update_settings_from_csv, style_text_boxes, create_menu_bar

thread_control = {"run_thread": None, "stop_requested": False}

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
def run_classify_gui(q, fig_queue, stop_requested):
    global vars_dict
    process_stdout_stderr(q)
    try:
        settings = check_classify_gui_settings(vars_dict)
        for key in settings:
            value = settings[key]
            print(key, value, type(value))
        train_test_model_wrapper(settings['src'], settings)
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
    thread_control["run_thread"] = Process(target=run_classify_gui, args=(q, fig_queue, stop_requested))
    thread_control["run_thread"].start()
    
def import_settings(scrollable_frame):
    global vars_dict

    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    csv_settings = read_settings_from_csv(csv_file_path)
    variables = classify_variables()
    new_settings = update_settings_from_csv(variables, csv_settings)
    vars_dict = generate_fields(new_settings, scrollable_frame)

#@log_function_call
def initiate_classify_root(parent_frame):#, width, height):
    global vars_dict, q, canvas, fig_queue, canvas_widget, thread_control
    
    style = ttk.Style(parent_frame)
    set_dark_style(style)
    style_text_boxes(style)
    set_default_font(parent_frame, font_name="Arial", size=8)

    parent_frame.configure(bg='#333333')
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

    vertical_container = tk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
    vertical_container.grid(row=0, column=0, sticky=tk.NSEW)
    parent_frame.grid_rowconfigure(0, weight=1)
    parent_frame.grid_columnconfigure(0, weight=1)

    # Settings Section
    settings_frame = tk.Frame(vertical_container, bg='#333333')
    vertical_container.add(settings_frame, stretch="always")
    settings_label = ttk.Label(settings_frame, text="Settings", background="#333333", foreground="white")
    settings_label.grid(row=0, column=0, pady=10, padx=10)
    scrollable_frame = ScrollableFrame(settings_frame, width=500)
    scrollable_frame.grid(row=1, column=0, sticky="nsew")
    settings_frame.grid_rowconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(0, weight=1)

    # Setup for user input fields (variables)
    variables = classify_variables()
    vars_dict = generate_fields(variables, scrollable_frame)

    # Button section
    import_btn = CustomButton(scrollable_frame.scrollable_frame, text="Import Settings", command=lambda: import_settings(scrollable_frame))
    import_btn.grid(row=47, column=0, pady=20, padx=20)
    run_button = CustomButton(scrollable_frame.scrollable_frame, text="Run", command=lambda: start_process(q, fig_queue))
    run_button.grid(row=45, column=0, pady=20, padx=20)
    abort_button = CustomButton(scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort)
    abort_button.grid(row=45, column=1, pady=20, padx=20)
    progress_label = ttk.Label(scrollable_frame.scrollable_frame, text="Processing: 0%", background="black", foreground="white") # Create progress field
    progress_label.grid(row=50, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=10)

    # Plot Canvas Section
    plot_frame = tk.PanedWindow(vertical_container, orient=tk.VERTICAL)
    vertical_container.add(plot_frame, stretch="always")
    figure = Figure(figsize=(30, 4), dpi=100, facecolor='#333333')
    plot = figure.add_subplot(111)
    plot.plot([], [])
    plot.axis('off')
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().configure(cursor='arrow', background='#333333', highlightthickness=0)
    canvas_widget = canvas.get_tk_widget()
    plot_frame.add(canvas_widget, stretch="always")
    canvas.draw()
    canvas.figure = figure

    # Console Section
    console_frame = tk.Frame(vertical_container, bg='#333333')
    vertical_container.add(console_frame, stretch="always")
    console_label = ttk.Label(console_frame, text="Console", background="#333333", foreground="white")
    console_label.grid(row=0, column=0, pady=10, padx=10)
    console_output = scrolledtext.ScrolledText(console_frame, height=10, bg='#333333', fg='white', insertbackground='white')
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

def gui_classify():
    root = tk.Tk()
    root.geometry("1000x800")
    root.title("SpaCer: generate masks")
    initiate_classify_root(root, 1000, 800)
    create_menu_bar(root)
    root.mainloop()

if __name__ == "__main__":
    gui_classify()
