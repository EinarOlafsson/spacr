import tkinter as tk
from tkinter import ttk
from multiprocessing import set_start_method
from .gui_elements import spacrButton, create_menu_bar, set_dark_style
from .gui_core import initiate_root
from screeninfo import get_monitors

class MainApp(tk.Tk):
    def __init__(self, default_app=None):
        super().__init__()

        # Initialize the window
        self.geometry("100x100")
        self.update_idletasks()

        # Get the current window position
        self.update_idletasks()
        x = self.winfo_x()
        y = self.winfo_y()

        # Find the monitor where the window is located
        for monitor in get_monitors():
            if monitor.x <= x < monitor.x + monitor.width and monitor.y <= y < monitor.y + monitor.height:
                width = monitor.width
                height = monitor.height
                break
        else:
            monitor = get_monitors()[0]
            width = monitor.width
            height = monitor.height

        # Set the window size to the dimensions of the monitor where it is located
        self.geometry(f"{width}x{height}")
        self.title("SpaCr GUI Collection")
        self.configure(bg='#333333')

        style = ttk.Style()
        self.color_settings = set_dark_style(style, parent_frame=self)
        self.main_buttons = {}
        self.additional_buttons = {}

        self.main_gui_apps = {
            "Mask": (lambda frame: initiate_root(self, 'mask'), "Generate cellpose masks for cells, nuclei and pathogen images."),
            "Measure": (lambda frame: initiate_root(self, 'measure'), "Measure single object intensity and morphological feature. Crop and save single object image"),
            "Annotate": (lambda frame: initiate_root(self, 'annotate'), "Annotation single object images on a grid. Annotations are saved to database."),
            "Make Masks": (lambda frame: initiate_root(self, 'make_masks'), "Adjust pre-existing Cellpose models to your specific dataset for improved performance"),
            "Classify": (lambda frame: initiate_root(self, 'classify'), "Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images."),
        }

        self.additional_gui_apps = {
            "Convert": (lambda frame: initiate_root(self, 'convert'), "Convert images to Grayscale TIFs."),
            "Umap": (lambda frame: initiate_root(self, 'umap'), "Generate UMAP embeddings with datapoints represented as images."),
            "Train Cellpose": (lambda frame: initiate_root(self, 'train_cellpose'), "Train custom Cellpose models."),
            "ML Analyze": (lambda frame: initiate_root(self, 'ml_analyze'), "Machine learning analysis of data."),
            "Cellpose Masks": (lambda frame: initiate_root(self, 'cellpose_masks'), "Generate Cellpose masks."),
            "Cellpose All": (lambda frame: initiate_root(self, 'cellpose_all'), "Run Cellpose on all images."),
            "Map Barcodes": (lambda frame: initiate_root(self, 'map_barcodes'), "Map barcodes to data."),
            "Regression": (lambda frame: initiate_root(self, 'regression'), "Perform regression analysis."),
            "Recruitment": (lambda frame: initiate_root(self, 'recruitment'), "Analyze recruitment data."),
            "Activation": (lambda frame: initiate_root(self, 'activation'), "Generate activation maps of computer vision models and measure channel-activation correlation."),
            "Plaque": (lambda frame: initiate_root(self, 'analyze_plaques'), "Analyze plaque data.")
        }

        self.selected_app = tk.StringVar()
        self.create_widgets()

        if default_app in self.main_gui_apps:
            self.load_app(default_app, self.main_gui_apps[default_app][0])
        elif default_app in self.additional_gui_apps:
            self.load_app(default_app, self.additional_gui_apps[default_app][0])

    def create_widgets(self):
        create_menu_bar(self)

        # Use a grid layout for centering
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.content_frame = tk.Frame(self)
        self.content_frame.grid(row=0, column=0, sticky="nsew")
        
        # Center the content frame within the window
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.inner_frame = tk.Frame(self.content_frame)
        self.inner_frame.grid(row=0, column=0)
        
        set_dark_style(ttk.Style(), containers=[self.content_frame, self.inner_frame])

        self.create_startup_screen()

    def create_startup_screen(self):
        self.clear_frame(self.inner_frame)

        main_buttons_frame = tk.Frame(self.inner_frame)
        main_buttons_frame.pack(pady=10)
        set_dark_style(ttk.Style(), containers=[main_buttons_frame])

        additional_buttons_frame = tk.Frame(self.inner_frame)
        additional_buttons_frame.pack(pady=10)
        set_dark_style(ttk.Style(), containers=[additional_buttons_frame])

        description_frame = tk.Frame(self.inner_frame, height=70)
        description_frame.pack(fill=tk.X, pady=10)
        description_frame.pack_propagate(False)
        set_dark_style(ttk.Style(), containers=[description_frame])

        self.description_label = tk.Label(description_frame, text="", wraplength=800, justify="center", font=('Helvetica', 12), fg=self.color_settings['fg_color'], bg=self.color_settings['bg_color'])
        self.description_label.pack(fill=tk.BOTH, pady=10)

        logo_button = spacrButton(main_buttons_frame, text="SpaCr", command=lambda: self.load_app("logo_spacr", initiate_root), icon_name="logo_spacr", size=100, show_text=False)
        logo_button.grid(row=0, column=0, padx=5, pady=5)
        self.main_buttons[logo_button] = "SpaCr provides a flexible toolset to extract single-cell images and measurements from high-content cell painting experiments, train deep-learning models to classify cellular/subcellular phenotypes, simulate, and analyze pooled CRISPR-Cas9 imaging screens."

        for i, (app_name, app_data) in enumerate(self.main_gui_apps.items()):
            app_func, app_desc = app_data
            button = spacrButton(main_buttons_frame, text=app_name, command=lambda app_name=app_name, app_func=app_func: self.load_app(app_name, app_func), icon_name=app_name.lower(), size=100, show_text=False)
            button.grid(row=0, column=i + 1, padx=5, pady=5)
            self.main_buttons[button] = app_desc

        for i, (app_name, app_data) in enumerate(self.additional_gui_apps.items()):
            app_func, app_desc = app_data
            button = spacrButton(additional_buttons_frame, text=app_name, command=lambda app_name=app_name, app_func=app_func: self.load_app(app_name, app_func), icon_name=app_name.lower(), size=75, show_text=False)
            button.grid(row=0, column=i, padx=5, pady=5)
            self.additional_buttons[button] = app_desc

        self.update_description()

    def update_description(self):
        for button, desc in {**self.main_buttons, **self.additional_buttons}.items():
            if button.canvas.itemcget(button.button_bg, "fill") == self.color_settings['active_color']:
                self.show_description(desc)
                return
        self.clear_description()

    def show_description(self, description):
        if self.description_label.winfo_exists():
            self.description_label.config(text=description)
            self.description_label.update_idletasks()

    def clear_description(self):
        if self.description_label.winfo_exists():
            self.description_label.config(text="")
            self.description_label.update_idletasks()

    def load_app(self, app_name, app_func):
        self.clear_frame(self.inner_frame)

        app_frame = tk.Frame(self.inner_frame)
        app_frame.pack(fill=tk.BOTH, expand=True)
        set_dark_style(ttk.Style(), containers=[app_frame])
        app_func(app_frame)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

def gui_app():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    gui_app()
