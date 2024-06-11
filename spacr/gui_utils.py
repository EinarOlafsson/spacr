import os, spacr, inspect, traceback, io, sys, ast, ctypes, matplotlib, re, csv, requests
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
from torchvision import models
#from ttf_opensans import opensans

from tkinter import font as tkFont


from .logger import log_function_call

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except AttributeError:
    pass

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


def load_app(root, app_name, app_func):
    # Destroy the current window
    root.destroy()
    # Create a new window for the app
    app_window = tk.Tk()
    app_window.title(f"SpaCr - {app_name}")
    app_window.geometry("1200x800")
    #app_window.attributes('-fullscreen', True)
    app_window.configure(bg="black")
    create_menu_bar(app_window)  # Add menu to the new window
    app_func(app_window, app_window.winfo_width(), app_window.winfo_height())

def create_menu_bar(root):

    from .gui_mask_app import initiate_mask_root
    from .gui_measure_app import initiate_measure_root
    from .annotate_app import initiate_annotation_app_root
    from .mask_app import initiate_mask_app_root
    from .gui_classify_app import initiate_classify_root

    gui_apps = {
        "Mask": initiate_mask_root,
        "Measure": initiate_measure_root,
        "Annotate": initiate_annotation_app_root,
        "Make Masks": initiate_mask_app_root,
        "Classify": initiate_classify_root
    }
    # Create the menu bar
    menu_bar = tk.Menu(root, bg="#008080", fg="white")
    # Create a "SpaCr Applications" menu
    app_menu = tk.Menu(menu_bar, tearoff=0, bg="#008080", fg="white")
    menu_bar.add_cascade(label="SpaCr Applications", menu=app_menu)
    # Add options to the "SpaCr Applications" menu
    for app_name, app_func in gui_apps.items():
        app_menu.add_command(label=app_name, command=lambda app_name=app_name, app_func=app_func: load_app(root, app_name, app_func))
    # Add a separator and an exit option
    app_menu.add_separator()
    app_menu.add_command(label="Exit", command=root.quit)
    # Configure the menu for the root window
    root.config(menu=menu_bar)

class CustomButton(tk.Frame):
    def __init__(self, parent, text="", command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.command = command

        self.canvas = tk.Canvas(self, width=150, height=50, highlightthickness=0, bg="black")
        self.canvas.grid(row=0, column=0)

        self.button_bg = self.create_rounded_rectangle(0, 0, 150, 50, radius=20, fill="#800080")

        # Create a Tkinter font object using OpenSans
        self.font_style = tkFont.Font(family="Arial", size=12, weight=tkFont.NORMAL)
        self.button_text = self.canvas.create_text(75, 25, text=self.text, fill="white", font=self.font_style)

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
    
def set_default_font(root, font_name="Arial", size=12):
    default_font = (font_name, size)
    root.option_add("*Font", default_font)
    root.option_add("*TButton.Font", default_font)
    root.option_add("*TLabel.Font", default_font)
    root.option_add("*TEntry.Font", default_font)

def check_and_download_font():
    font_name = "Arial"
    font_dir = "fonts"
    font_path = os.path.join(font_dir, "OpenSans-Regular.ttf")

    # Check if the font is already available
    available_fonts = list(tkFont.families())
    if font_name not in available_fonts:
        print(f"Font '{font_name}' not found. Downloading...")
        if not os.path.exists(font_dir):
            os.makedirs(font_dir)

        if not os.path.exists(font_path):
            url = "https://github.com/google/fonts/blob/main/apache/opensans/OpenSans-Regular.ttf?raw=true"
            response = requests.get(url)
            with open(font_path, "wb") as f:
                f.write(response.content)

        # Load the font
        try:
            tkFont.nametofont("TkDefaultFont").configure(family=font_name, size=10)
            tkFont.nametofont("TkTextFont").configure(family=font_name, size=10)
            tkFont.nametofont("TkHeadingFont").configure(family=font_name, size=12)
        except tk.TclError:
            tkFont.nametofont("TkDefaultFont").configure(family="Arial", size=10)
            tkFont.nametofont("TkTextFont").configure(family="Arial", size=10)
            tkFont.nametofont("TkHeadingFont").configure(family="Arial", size=12)
    else:
        tkFont.nametofont("TkDefaultFont").configure(family=font_name, size=10)
        tkFont.nametofont("TkTextFont").configure(family=font_name, size=10)
        tkFont.nametofont("TkHeadingFont").configure(family=font_name, size=12)

def style_text_boxes(style):
    check_and_download_font()
    font_style = tkFont.Font(family="Arial", size=10)  # Define the Arial font
    style.configure('TEntry', padding='5 5 5 5', borderwidth=1, relief='solid', fieldbackground='#000000', foreground='#ffffff', font=font_style)
    style.configure('TCombobox', fieldbackground='#000000', background='#000000', foreground='#ffffff', font=font_style)
    style.configure('Custom.TButton', padding='10 10 10 10', borderwidth=1, relief='solid', background='#008080', foreground='#ffffff', font=font_style)
    style.map('Custom.TButton',
              background=[('active', '#66b2b2'), ('disabled', '#004d4d'), ('!disabled', '#008080')],
              foreground=[('active', '#ffffff'), ('disabled', '#888888')])
    style.configure('Custom.TLabel', padding='5 5 5 5', borderwidth=1, relief='flat', background='#000000', foreground='#ffffff', font=font_style)
    style.configure('TCheckbutton', background='#333333', foreground='#ffffff', indicatoron=False, relief='flat', font=font_style)
    style.map('TCheckbutton', background=[('selected', '#555555'), ('active', '#555555')])



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

def safe_literal_eval(value):
    try:
        # First, try to evaluate as a literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If it fails, return the value as it is (a string)
        return value

def disable_interactivity(fig):
    if hasattr(fig.canvas, 'toolbar'):
        fig.canvas.toolbar.pack_forget()

    event_handlers = fig.canvas.callbacks.callbacks
    for event, handlers in list(event_handlers.items()):
        for handler_id in list(handlers.keys()):
            fig.canvas.mpl_disconnect(handler_id)

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, bg='black', **kwargs):
        super().__init__(container, *args, **kwargs)
        self.configure(style='TFrame')  # Ensure this uses the styled frame from dark mode
        
        canvas = tk.Canvas(self, bg=bg)  # Set canvas background to match dark mode
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas, style='TFrame')  # Ensure it uses the styled frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

def check_mask_gui_settings(vars_dict):
    settings = {}
    for key, var in vars_dict.items():
        value = var.get()
        
        # Handle conversion for specific types if necessary
        if key in ['channels', 'timelapse_frame_limits']:  # Assuming these should be lists
            try:
                # Convert string representation of a list into an actual list
                settings[key] = eval(value)
            except:
                messagebox.showerror("Error", f"Invalid format for {key}. Please enter a valid list.")
                return
        elif key in ['nucleus_channel', 'cell_channel', 'pathogen_channel', 'examples_to_plot', 'batch_size', 'timelapse_memory', 'workers', 'fps', 'magnification']:  # Assuming these should be integers
            try:
                settings[key] = int(value) if value else None
            except ValueError:
                messagebox.showerror("Error", f"Invalid number for {key}.")
                return
        elif key in ['nucleus_background', 'cell_background', 'pathogen_background', 'nucleus_Signal_to_noise', 'cell_Signal_to_noise', 'pathogen_Signal_to_noise', 'nucleus_CP_prob', 'cell_CP_prob', 'pathogen_CP_prob', 'lower_quantile']:  # Assuming these should be floats
            try:
                settings[key] = float(value) if value else None
            except ValueError:
                messagebox.showerror("Error", f"Invalid number for {key}.")
                return
        else:
            settings[key] = value
    return settings

def check_measure_gui_settings(vars_dict):
    settings = {}
    for key, var in vars_dict.items():
        value = var.get()  # Retrieves the string representation for entries or the actual value for checkboxes and combos.

        try:
            if key in ['channels', 'png_dims']:
                settings[key] = [int(chan) for chan in ast.literal_eval(value)] if value else []
                
            elif key in ['cell_loc', 'pathogen_loc', 'treatment_loc']:
                # Convert to a list of lists of strings, ensuring all structures are lists.
                settings[key] = [list(map(str, sublist)) for sublist in ast.literal_eval(value)] if value else []

            elif key == 'dialate_png_ratios':
                settings[key] = [float(num) for num in ast.literal_eval(value)] if value else []

            elif key == 'normalize':
                settings[key] = [int(num) for num in ast.literal_eval(value)] if value else []

            # Directly assign string values for these specific keys
            elif key in ['normalize_by', 'experiment', 'measurement', 'input_folder']:
                settings[key] = value

            elif key == 'png_size':
                settings[key] = [list(map(int, dim)) for dim in ast.literal_eval(value)] if value else []
            
            # Ensure these are lists of strings, converting from tuples if necessary
            elif key in ['timelapse_objects', 'crop_mode', 'cells', 'pathogens', 'treatments']:
                eval_value = ast.literal_eval(value) if value else []
                settings[key] = list(map(str, eval_value)) if isinstance(eval_value, (list, tuple)) else [str(eval_value)]

            # Handling for single non-string values (int, float, bool)
            elif key in ['cell_mask_dim', 'cell_min_size', 'nucleus_mask_dim', 'nucleus_min_size', 'pathogen_mask_dim', 'pathogen_min_size', 'cytoplasm_min_size', 'max_workers', 'channel_of_interest', 'nr_imgs']:
                settings[key] = int(value) if value else None

            elif key == 'um_per_pixel':
                settings[key] = float(value) if value else None

            # Handling boolean values based on checkboxes
            elif key in ['save_png', 'use_bounding_box', 'save_measurements', 'plot', 'plot_filtration', 'include_uninfected', 'dialate_pngs', 'timelapse', 'representative_images']:
                settings[key] = var.get()

        except SyntaxError as e:
            print(f"Syntax error processing {key}: {str(e)}")
            #messagebox.showerror("Error", f"Syntax error processing {key}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            #messagebox.showerror("Error", f"Error processing {key}: {str(e)}")
            return None

    return settings

def check_classify_gui_settings(vars_dict):
    settings = {}
    for key, var in vars_dict.items():
        value = var.get()  # This retrieves the string representation for entries or the actual value for checkboxes and combos

        try:
            if key in ['src', 'measurement']:
                # Directly assign string values
                settings[key] = str(value)
            elif key in ['cell_mask_dim', 'image_size', 'batch_size', 'epochs', 'gradient_accumulation_steps', 'num_workers']:
                # Convert to integer
                settings[key] = int(value)
            elif key in ['val_split', 'learning_rate', 'weight_decay', 'dropout_rate']:
                # Convert to float
                settings[key] = float(value)
            elif key == 'classes':
                # Evaluate as list
                settings[key] = ast.literal_eval(value)

            elif key in ['model_type','optimizer_type','schedule','loss_type','train_mode']:
                settings[key] = value

            elif key in ['gradient_accumulation','normalize','save','plot', 'init_weights','amsgrad','use_checkpoint','intermedeate_save','pin_memory', 'num_workers','verbose']:
                settings[key] = bool(value)

        except SyntaxError as e:
            messagebox.showerror("Error", f"Syntax error processing {key}: {str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Error processing {key}: {str(e)}")
            return None

    return settings

def check_sim_gui_settings(vars_dict):
    settings = {}
    for key, var in vars_dict.items():
        value = var.get()  # This retrieves the string representation for entries or the actual value for checkboxes and combos

        try:
            if key in ['src', 'name', 'variable']:
                # Directly assign string values
                settings[key] = str(value)
            
            elif key in ['nr_plates', 'number_of_genes','number_of_active_genes','avg_genes_per_well','avg_cells_per_well','avg_reads_per_gene']:
                #generate list of integers from list
                ls = [int(num) for num in ast.literal_eval(value)]
                if len(ls) == 3 and ls[2] > 0:
                    list_of_integers = list(range(ls[0], ls[1], ls[2]))
                    list_of_integers = [num + 1 if num == 0 else num for num in list_of_integers]
                else:
                    list_of_integers = [ls[0]]
                settings[key] = list_of_integers
                
            elif key in ['sequencing_error','well_ineq_coeff','gene_ineq_coeff', 'positive_mean']:
                #generate list of floats from list
                ls = [float(num) for num in ast.literal_eval(value)]
                if len(ls) == 3 and ls[2] > 0:
                    list_of_floats = np.linspace(ls[0], ls[1], ls[2])
                    list_of_floats.tolist()
                    list_of_floats = [x if x != 0.0 else x + 0.01 for x in list_of_floats]
                else:
                    list_of_floats = [ls[0]]
                settings[key] = list_of_floats

            elif key in ['plot', 'random_seed']:
                # Evaluate as bool
                settings[key] = bool(value)
                
            elif key in ['number_of_control_genes', 'replicates', 'max_workers']:
                # Convert to integer
                settings[key] = int(value)
                
        except SyntaxError as e:
            messagebox.showerror("Error", f"Syntax error processing {key}: {str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Error processing {key}: {str(e)}")
            return None

    return settings

def sim_variables():
    variables = {
        'name':('entry', None, 'plates_2_4_8'),
        'variable':('entry', None, 'all'),
        'src':('entry', None, '/home/olafsson/Desktop/simulations'),
        'number_of_control_genes':('entry', None, 30),
        'replicates':('entry', None, 4),
        'max_workers':('entry', None, 1),
        'plot':('check', None, True),
        'random_seed':('check', None, True),
        'nr_plates': ('entry', None, '[8,8,0]'),# '[2,2,8]'
        'number_of_genes': ('entry', None, '[100, 100, 0]'), #[1384, 1384, 0]
        'number_of_active_genes': ('entry', None, '[10,10,0]'),
        'avg_genes_per_well': ('entry', None, '[2, 10, 2]'),
        'avg_cells_per_well': ('entry', None, '[100, 100, 0]'),
        'positive_mean': ('entry', None, '[0.8, 0.8, 0]'),
        'avg_reads_per_gene': ('entry', None, '[1000,1000, 0]'),
        'sequencing_error': ('entry', None, '[0.01, 0.01, 0]'),
        'well_ineq_coeff': ('entry', None, '[0.3,0.3,0]'),
        'gene_ineq_coeff': ('entry', None, '[0.8,0.8,0]'),
    }
    return variables

def add_measure_gui_defaults(settings):
    settings['compartments'] = ['pathogen', 'cytoplasm']
    return settings

def measure_variables():
    variables = {
        'input_folder':('entry', None, '/mnt/data/CellVoyager/40x/einar/mitotrackerHeLaToxoDsRed_20240224_123156/test_gui/merged'),
        'channels': ('combo', ['[0,1,2,3]','[0,1,2]','[0,1]','[0]'], '[0,1,2,3]'),
        'cell_mask_dim':('entry', None, 4),
        'cell_min_size':('entry', None, 0),
        'cytoplasm_min_size':('entry', None, 0),
        'nucleus_mask_dim':('entry', None, 5),
        'nucleus_min_size':('entry', None, 0),
        'pathogen_mask_dim':('entry', None, 6),
        'pathogen_min_size':('entry', None, 0),
        'save_png':('check', None, True),
        'crop_mode':('entry', None, '["cell"]'),
        'use_bounding_box':('check', None, True),
        'png_size': ('entry', None, '[[224,224]]'), 
        'normalize':('entry', None, '[2,98]'),
        'png_dims':('entry', None, '[1,2,3]'),
        'normalize_by':('combo', ['fov', 'png'], 'png'),
        'save_measurements':('check', None, True),
        'representative_images':('check', None, True),
        'plot':('check', None, True),
        'plot_filtration':('check', None, True),
        'include_uninfected':('check', None, True),
        'dialate_pngs':('check', None, False),
        'dialate_png_ratios':('entry', None, '[0.2]'),
        'timelapse':('check', None, False),
        'timelapse_objects':('combo', ['["cell"]', '["nucleus"]', '["pathogen"]', '["cytoplasm"]'], '["cell"]'),
        'max_workers':('entry', None, 30),
        'experiment':('entry', None, 'experiment name'),
        'cells':('entry', None, ['HeLa']),
        'cell_loc': ('entry', None, '[["c1","c2"], ["c3","c4"]]'),
        'pathogens':('entry', None, '["wt","mutant"]'),
        'pathogen_loc': ('entry', None, '[["c1","c2"], ["c3","c4"]]'),
        'treatments': ('entry', None, '["cm","lovastatin_20uM"]'),
        'treatment_loc': ('entry', None, '[["c1","c2"], ["c3","c4"]]'),
        'channel_of_interest':('entry', None, 3),
        'compartments':('entry', None, '["pathogen","cytoplasm"]'),
        'measurement':('entry', None, 'mean_intensity'),
        'nr_imgs':('entry', None, 32),
        'um_per_pixel':('entry', None, 0.1)
    }
    return variables

def classify_variables():
    
    def get_torchvision_models():
        # Fetch all public callable attributes from torchvision.models that are functions
        model_names = [name for name, obj in inspect.getmembers(models) 
                    if inspect.isfunction(obj) and not name.startswith("__")]
        return model_names
    
    model_names = get_torchvision_models()
    variables = {
        'src':('entry', None, '/mnt/data/CellVoyager/40x/einar/mitotrackerHeLaToxoDsRed_20240224_123156/test_gui/merged'),
        'cell_mask_dim':('entry', None, 4),
        'classes':('entry', None, '["nc","pc"]'),
        'measurement':('entry', None, 'mean_intensity'),
        'model_type': ('combo', model_names, 'resnet50'),
        'optimizer_type': ('combo', ['adamw','adam'], 'adamw'),
        'schedule': ('combo', ['reduce_lr_on_plateau','step_lr'], 'reduce_lr_on_plateau'),
        'loss_type': ('combo', ['focal_loss', 'binary_cross_entropy_with_logits'], 'focal_loss'),
        'image_size': ('entry', None, 224),
        'batch_size': ('entry', None, 12),
        'epochs': ('entry', None, 2),
        'val_split': ('entry', None, 0.1),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'learning_rate': ('entry', None, 0.0001),
        'weight_decay': ('entry', None, 0.00001),
        'dropout_rate': ('entry', None, 0.1),
        'gradient_accumulation': ('check', None, True),
        'gradient_accumulation_steps': ('entry', None, 4),
        'normalize': ('check', None, True),
        'save': ('check', None, True), 
        'plot': ('check', None, True),
        'init_weights': ('check', None, True),
        'amsgrad': ('check', None, True),
        'use_checkpoint': ('check', None, True),
        'intermedeate_save': ('check', None, True),
        'pin_memory': ('check', None, True),
        'num_workers': ('entry', None, 30),
        'verbose': ('check', None, True),
    }
    return variables
    
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
        check = ToggleSwitch(frame, text=label_text, variable=var)  # Use ToggleSwitch class
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

def mask_variables():
    variables = {
        'src': ('entry', None, '/mnt/data/CellVoyager/40x/einar/mitotrackerHeLaToxoDsRed_20240224_123156/test_gui'),
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'nikon', 'zeis', 'custom'], 'cellvoyager'),
        'custom_regex': ('entry', None, None),
        'experiment': ('entry', None, 'exp'),
        'channels': ('combo', ['[0,1,2,3]','[0,1,2]','[0,1]','[0]'], '[0,1,2,3]'),
        'magnification': ('combo', [20, 40, 60,], 40),
        'nucleus_channel': ('combo', [0,1,2,3, None], 0),
        'nucleus_background': ('entry', None, 100),
        'nucleus_Signal_to_noise': ('entry', None, 5),
        'nucleus_CP_prob': ('entry', None, 0),
        'cell_channel': ('combo', [0,1,2,3, None], 3),
        'cell_background': ('entry', None, 100),
        'cell_Signal_to_noise': ('entry', None, 5),
        'cell_CP_prob': ('entry', None, 0),
        'pathogen_channel': ('combo', [0,1,2,3, None], 2),
        'pathogen_background': ('entry', None, 100),
        'pathogen_Signal_to_noise': ('entry', None, 3),
        'pathogen_CP_prob': ('entry', None, 0),
        'preprocess': ('check', None, True),
        'masks': ('check', None, True),
        'examples_to_plot': ('entry', None, 1),
        'randomize': ('check', None, True),
        'batch_size': ('entry', None, 50),
        'timelapse': ('check', None, False),
        'timelapse_displacement': ('entry', None, None),
        'timelapse_memory': ('entry', None, 0),
        'timelapse_frame_limits': ('entry', None, '[0,1000]'),
        'timelapse_remove_transient': ('check', None, True),
        'timelapse_mode': ('combo',  ['trackpy', 'btrack'], 'trackpy'),
        'timelapse_objects': ('combo', ['cell','nucleus','pathogen','cytoplasm', None], None),
        'fps': ('entry', None, 2),
        'remove_background': ('check', None, True),
        'lower_quantile': ('entry', None, 0.01),
        'merge': ('check', None, False),
        'normalize_plots': ('check', None, True),
        'all_to_mip': ('check', None, False),
        'pick_slice': ('check', None, False),
        'skip_mode': ('entry', None, None),
        'save': ('check', None, True),
        'plot': ('check', None, True),
        'workers': ('entry', None, 30),
        'verbose': ('check', None, True),
    }
    return variables

def add_mask_gui_defaults(settings):
    settings['remove_background'] = True
    settings['fps'] = 2
    settings['all_to_mip'] = False
    settings['pick_slice'] = False
    settings['merge'] = False
    settings['skip_mode'] = ''
    settings['verbose'] = False
    settings['normalize_plots'] = True
    settings['randomize'] = True
    settings['preprocess'] = True
    settings['masks'] = True
    settings['examples_to_plot'] = 1
    return settings

def generate_fields(variables, scrollable_frame):
    vars_dict = {}
    row = 0
    tooltips = {
        "src": "Path to the folder containing the images.",
        "metadata_type": "Type of metadata to expect in the images. This will determine how the images are processed. If 'custom' is selected, you can provide a custom regex pattern to extract metadata from the image names",
        "custom_regex": "Custom regex pattern to extract metadata from the image names. This will only be used if 'custom' is selected for 'metadata_type'.",
        "experiment": "Name of the experiment. This will be used to name the output files.",
        "channels": "List of channels to use for the analysis. The first channel is 0, the second is 1, and so on. For example, [0,1,2] will use channels 0, 1, and 2.",
        "magnification": "At what magnification the images were taken. This will be used to determine the size of the objects in the images.",
        "nucleus_channel": "The channel to use for the nucleus. If None, the nucleus will not be segmented.",
        "nucleus_background": "The background intensity for the nucleus channel. This will be used to remove background noise.",
        "nucleus_Signal_to_noise": "The signal-to-noise ratio for the nucleus channel. This will be used to determine the range of intensities to normalize images to for nucleus segmentation.",
        "nucleus_CP_prob": "The cellpose probability threshold for the nucleus channel. This will be used to segment the nucleus.",
        "cell_channel": "The channel to use for the cell. If None, the cell will not be segmented.",
        "cell_background": "The background intensity for the cell channel. This will be used to remove background noise.",
        "cell_Signal_to_noise": "The signal-to-noise ratio for the cell channel. This will be used to determine the range of intensities to normalize images to for cell segmentation.",
        "cell_CP_prob": "The cellpose probability threshold for the cell channel. This will be used to segment the cell.",
        "pathogen_channel": "The channel to use for the pathogen. If None, the pathogen will not be segmented.",
        "pathogen_background": "The background intensity for the pathogen channel. This will be used to remove background noise.",
        "pathogen_Signal_to_noise": "The signal-to-noise ratio for the pathogen channel. This will be used to determine the range of intensities to normalize images to for pathogen segmentation.",
        "pathogen_CP_prob": "The cellpose probability threshold for the pathogen channel. This will be used to segment the pathogen.",
        "preprocess": "Whether to preprocess the images before segmentation. This includes background removal and normalization. Set to False only if this step has already been done.",
        "masks": "Whether to generate masks for the segmented objects. If True, masks will be generated for the nucleus, cell, and pathogen.",
        "examples_to_plot": "The number of images to plot for each segmented object. This will be used to visually inspect the segmentation results and normalization .",
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
        "lower_quantile": "The lower quantile to use for normalizing the images. This will be used to determine the range of intensities to normalize images to.",
        "merge_pathogens": "Whether to merge pathogen objects that share more than 75% of their perimiter.",
        "normalize_plots": "Whether to normalize the plots.",
        "all_to_mip": "Whether to convert all images to maximum intensity projections before processing.",
        "pick_slice": "Whether to pick a single slice from the z-stack images. If False, the maximum intensity projection will be used.",
        "skip_mode": "The mode to use for skipping images. This will determine how to handle images that cannot be processed.",
        "save": "Whether to save the results to disk.",
        "plot": "Whether to plot the results.",
        "workers": "The number of workers to use for processing the images. This will determine how many images are processed in parallel. Increase to speed up processing.",
        "verbose": "Whether to print verbose output during processing.",
        "input_folder": "Path to the folder containing the images.",
        "cell_mask_dim": "The dimension of the array the cell mask is saved in.",
        "cell_min_size": "The minimum size of cell objects in pixels2.",
        "cytoplasm_min_size": "The minimum size of cytoplasm objects in pixels2.",
        "nucleus_mask_dim": "The dimension of the array the nucleus mask is saved in.",
        "nucleus_min_size": "The minimum size of nucleus objects in pixels2.",
        "pathogen_mask_dim": "The dimension of the array the pathogen mask is saved in.",
        "pathogen_min_size": "The minimum size of pathogen objects in pixels2.",
        "save_png": "Whether to save the segmented objects as PNG images.",
        "crop_mode": "The mode to use for cropping the images. This will determine which objects are cropped from the images (cell, nucleus, pathogen, cytoplasm).",
        "use_bounding_box": "Whether to use the bounding box of the objects for cropping. If False, only the object itself will be cropped.",
        "png_size": "The size of the PNG images to save. This will determine the size of the saved images.",
        "normalize": "The percentiles to use for normalizing the images. This will be used to determine the range of intensities to normalize images to., if None, no normalization is done.",
        "png_dims": "The dimensions of the PNG images to save. This will determine the dimensions of the saved images. Maximum of 3 dimensions e.g. [1,2,3].",
        "normalize_by": "Whether to normalize the images by field of view (fov) or by PNG image (png).",
        "save_measurements": "Whether to save the measurements to disk.",
        "representative_images": "Whether to save representative images of the segmented objects (Not working yet).",
        "plot": "Whether to plot results.",
        "plot_filtration": "Whether to plot the filtration steps.",
        "include_uninfected": "Whether to include uninfected cells in the analysis.",
        "dialate_pngs": "Whether to dialate the PNG images before saving.",
        "dialate_png_ratios": "The ratios to use for dialating the PNG images. This will determine the amount of dialation applied to the images before cropping.",
        "timelapse_objects": "The objects to track in the timelapse (cell, nucleus or pathogen). This will determine which objects are tracked over time. If None, all objects will be tracked.",
        "max_workers": "The number of workers to use for processing the images. This will determine how many images are processed in parallel. Increase to speed up processing.",
        "cells: ": "The cell types to include in the analysis.",
        "cell_loc": "The locations of the cell types in the images.",
        "pathogens": "The pathogen types to include in the analysis.",
        "pathogen_loc": "The locations of the pathogen types in the images.",
        "treatments": "The treatments to include in the analysis.",
        "treatment_loc": "The locations of the treatments in the images.",
        "channel_of_interest": "The channel of interest to use for the analysis.",
        "compartments": "The compartments to measure in the images.",
        "measurement": "The measurement to use for the analysis.",
        "nr_imgs": "The number of images to plot.",
        "um_per_pixel": "The micrometers per pixel for the images.",
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
        console_output.config(bg=dark_bg, fg=light_text, insertbackground=light_text) #, font=("Arial", 12)
    root.configure(bg=dark_bg)
    
def set_dark_style(style):
    style.configure('TFrame', background='black')
    style.configure('TLabel', background='black', foreground='white')
    style.configure('TEntry', background='black', foreground='white')
    style.configure('TCheckbutton', background='black', foreground='white')

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
        spacr.core.preprocess_generate_masks(settings['src'], settings=settings)
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