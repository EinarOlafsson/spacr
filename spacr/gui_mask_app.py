import spacr, sys, ctypes, csv, matplotlib
import tkinter as tk
from tkinter import ttk, scrolledtext
from ttkthemes import ThemedTk
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
from .gui_utils import ScrollableFrame, StdoutRedirector, safe_literal_eval, clear_canvas, main_thread_update_function, create_dark_mode, set_dark_style, set_default_font, generate_fields, process_stdout_stderr
from .gui_utils import mask_variables, check_mask_gui_settings, add_mask_gui_defaults, preprocess_generate_masks_wrapper

thread_control = {"run_thread": None, "stop_requested": False}

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
        settings = add_mask_gui_defaults(settings)
        #for key in settings:
        #    value = settings[key]
        #    print(key, value, type(value))
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
    
    if not csv_file_path:
        return
    
    imported_variables = {}

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['Key']
            value = row['Value']
            # Evaluate the value safely using safe_literal_eval
            imported_variables[key] = safe_literal_eval(value)

    # Track changed variables and apply the imported ones, printing changes as we go
    for key, var in vars_dict.items():
        if key in imported_variables and var.get() != imported_variables[key]:
            print(f"Updating '{key}' from '{var.get()}' to '{imported_variables[key]}'")
            var.set(imported_variables[key])
    
@log_function_call
def initiate_mask_root(width, height):
    global root, vars_dict, q, canvas, fig_queue, canvas_widget, thread_control
        
    theme = 'breeze'
    
    if theme in ['clam']:
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use(theme) #plastik, clearlooks, elegance, default was clam #alt, breeze, arc
        set_dark_style(style)
    elif theme in ['breeze']:
        root = ThemedTk(theme="breeze")
        style = ttk.Style(root)
        set_dark_style(style)
        
    set_default_font(root, font_name="Arial", size=10)
    #root.state('zoomed')  # For Windows to maximize the window
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
                #set_fig_text_properties(fig, font_size=8)
                for ax in fig.get_axes():
                    ax.set_xticks([])  # Remove x-axis ticks
                    ax.set_yticks([])  # Remove y-axis ticks
                    ax.xaxis.set_visible(False)  # Hide the x-axis
                    ax.yaxis.set_visible(False)  # Hide the y-axis
                    #ax.title.set_fontsize(14) 
                #disable_interactivity(fig)
                fig.tight_layout()
                fig.set_facecolor('#333333')
                canvas.figure = fig
                fig_width, fig_height = canvas_widget.winfo_width(), canvas_widget.winfo_height()
                fig.set_size_inches(fig_width / fig.dpi, fig_height / fig.dpi, forward=True)
                canvas.draw_idle() 
        except Exception as e:
            traceback.print_exc()
            #pass
        finally:
            canvas_widget.after(100, _process_fig_queue)
            
    # Process queue for console output
    def _process_console_queue():
        while not q.empty():
            message = q.get_nowait()
            console_output.insert(tk.END, message)
            console_output.see(tk.END)
        console_output.after(100, _process_console_queue)
        
    # Vertical container for settings and console
    vertical_container = tk.PanedWindow(root, orient=tk.HORIZONTAL) #VERTICAL
    vertical_container.pack(fill=tk.BOTH, expand=True)

    # Scrollable Frame for user settings
    scrollable_frame = ScrollableFrame(vertical_container, bg='#333333')
    vertical_container.add(scrollable_frame, stretch="always")

    # Setup for user input fields (variables)
    variables = mask_variables()
    vars_dict = generate_fields(variables, scrollable_frame)
    
    # Horizontal container for Matplotlib figure and the vertical pane (for settings and console)
    horizontal_container = tk.PanedWindow(vertical_container, orient=tk.VERTICAL) #HORIZONTAL
    vertical_container.add(horizontal_container, stretch="always")

    # Matplotlib figure setup
    figure = Figure(figsize=(30, 4), dpi=100, facecolor='#333333')
    plot = figure.add_subplot(111)
    plot.plot([], [])  # This creates an empty plot.
    plot.axis('off')

    # Embedding the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(figure, master=horizontal_container)
    canvas.get_tk_widget().configure(cursor='arrow', background='#333333', highlightthickness=0)
    #canvas.get_tk_widget().configure(cursor='arrow')
    canvas_widget = canvas.get_tk_widget()
    horizontal_container.add(canvas_widget, stretch="always")
    canvas.draw()
    canvas.figure = figure

    # Console output setup below the settings
    console_output = scrolledtext.ScrolledText(vertical_container, height=10)
    vertical_container.add(console_output, stretch="always")

    # Queue and redirection setup for updating console output safely
    q = Queue()
    sys.stdout = StdoutRedirector(console_output)
    sys.stderr = StdoutRedirector(console_output)
    
    # This is your GUI setup where you create the Run button
    run_button = ttk.Button(scrollable_frame.scrollable_frame, text="Run",command=lambda: start_process(q, fig_queue))
    run_button.grid(row=40, column=0, pady=10)
    
    abort_button = ttk.Button(scrollable_frame.scrollable_frame, text="Abort", command=initiate_abort)
    abort_button.grid(row=40, column=1, pady=10)
    
    progress_label = ttk.Label(scrollable_frame.scrollable_frame, text="Processing: 0%", background="#333333", foreground="white")
    progress_label.grid(row=41, column=0, columnspan=2, sticky="ew", pady=(5, 0))
    
    # Create the Import Settings button
    import_btn = tk.Button(root, text="Import Settings", command=lambda: import_settings(scrollable_frame))
    import_btn.pack(pady=20)
    
    _process_console_queue()
    _process_fig_queue()
    create_dark_mode(root, style, console_output)
    
    root.after(100, lambda: main_thread_update_function(root, q, fig_queue, canvas_widget, progress_label))
    
    return root, vars_dict

#def gui_mask():
#    global vars_dict, root
#    root, vars_dict = initiate_mask_root(1000, 1500)
#    root.mainloop()
    
def gui_mask():
    from .cli import get_arg_parser
    from .version import version_str

    args = get_arg_parser().parse_args()
    
    if args.version:
        print(version_str)
        return

    if args.headless:
        settings = {}
        spacr.core.preprocess_generate_masks(settings['src'], settings=settings, advanced_settings={})
        return

    global vars_dict, root
    root, vars_dict = initiate_mask_root(1000, 1500)
    root.mainloop()

if __name__ == "__main__":
    gui_mask()