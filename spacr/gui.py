import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import os
from multiprocessing import set_start_method
from .gui_elements import spacrButton, create_menu_bar, set_dark_style
from .gui_core import initiate_root

class MainApp(tk.Tk):
    def __init__(self, default_app=None):
        super().__init__()
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry(f"{width}x{height}")
        self.title("SpaCr GUI Collection")
        self.configure(bg='#333333')  # Set window background to dark gray

        # Initialize style and apply dark style to the main window
        style = ttk.Style()
        self.color_settings = set_dark_style(style, parent_frame=self)
        self.main_buttons = {}  # Initialize main_buttons dictionary here
        self.additional_buttons = {}  # Initialize additional_buttons dictionary here

        self.main_gui_apps = {
            "Mask": (lambda frame: initiate_root(frame, 'mask'), "Generate cellpose masks for cells, nuclei and pathogen images."),
            "Measure": (lambda frame: initiate_root(frame, 'measure'), "Measure single object intensity and morphological feature. Crop and save single object image"),
            "Annotate": (lambda frame: initiate_root(frame, 'annotate'), "Annotation single object images on a grid. Annotations are saved to database."),
            "Make Masks": (lambda frame: initiate_root(frame, 'make_masks'), "Adjust pre-existing Cellpose models to your specific dataset for improved performance"),
            "Classify": (lambda frame: initiate_root(frame, 'classify'), "Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images."),
        }

        self.additional_gui_apps = {
            "Sequencing": (lambda frame: initiate_root(frame, 'sequencing'), "Analyze sequencing data."),
            "Umap": (lambda frame: initiate_root(frame, 'umap'), "Generate UMAP embeddings with datapoints represented as images."),
            "Train Cellpose": (lambda frame: initiate_root(frame, 'train_cellpose'), "Train custom Cellpose models."),
            "ML Analyze": (lambda frame: initiate_root(frame, 'ml_analyze'), "Machine learning analysis of data."),
            "Cellpose Masks": (lambda frame: initiate_root(frame, 'cellpose_masks'), "Generate Cellpose masks."),
            "Cellpose All": (lambda frame: initiate_root(frame, 'cellpose_all'), "Run Cellpose on all images."),
            "Map Barcodes": (lambda frame: initiate_root(frame, 'map_barcodes'), "Map barcodes to data."),
            "Regression": (lambda frame: initiate_root(frame, 'regression'), "Perform regression analysis."),
            "Recruitment": (lambda frame: initiate_root(frame, 'recruitment'), "Analyze recruitment data.")
        }

        self.selected_app = tk.StringVar()
        self.create_widgets()

        if default_app in self.main_gui_apps:
            self.load_app(default_app, self.main_gui_apps[default_app][0])
        elif default_app in self.additional_gui_apps:
            self.load_app(default_app, self.additional_gui_apps[default_app][0])

    def create_widgets(self):
        # Create the menu bar
        create_menu_bar(self)

        # Create a canvas to hold the selected app and other elements
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a frame inside the canvas to hold the main content
        self.content_frame = tk.Frame(self.canvas)
        self.content_frame.grid(row=0, column=0, sticky="nsew")

        # Center the content frame within the canvas
        self.canvas.create_window((self.winfo_screenwidth() // 2, self.winfo_screenheight() // 2), window=self.content_frame, anchor="center")

        # Apply dark style to canvas and content_frame
        set_dark_style(ttk.Style(), containers=[self.canvas, self.content_frame])

        # Create startup screen with buttons for each main GUI app and drop-down for additional apps
        self.create_startup_screen()

    def create_startup_screen(self):
        self.clear_frame(self.content_frame)

        # Create frames for the grids
        main_buttons_frame = tk.Frame(self.content_frame)
        main_buttons_frame.pack(pady=10)
        set_dark_style(ttk.Style(), containers=[main_buttons_frame])

        additional_buttons_frame = tk.Frame(self.content_frame)
        additional_buttons_frame.pack(pady=10)
        set_dark_style(ttk.Style(), containers=[additional_buttons_frame])

        # Create a frame for the description below the icon grids
        description_frame = tk.Frame(self.content_frame, height=70)
        description_frame.pack(fill=tk.X, pady=10)
        description_frame.pack_propagate(False)  # Prevent the frame from resizing based on its content
        set_dark_style(ttk.Style(), containers=[description_frame])

        # Use a Label widget to display descriptions
        self.description_label = tk.Label(description_frame, text="", wraplength=800, justify="center", font=('Helvetica', 12), fg=self.color_settings['fg_color'], bg=self.color_settings['bg_color'])
        self.description_label.pack(fill=tk.BOTH, pady=10)

        # Load the logo image and place it in the main apps row
        logo_button = spacrButton(main_buttons_frame, text="SpaCr", command=lambda: self.load_app("logo_spacr", initiate_root), icon_name="logo_spacr", size=100, show_text=False)
        logo_button.grid(row=0, column=0, padx=5, pady=5)
        self.main_buttons[logo_button] = "SpaCr provides a flexible toolset to extract single-cell images and measurements from high-content cell painting experiments, train deep-learning models to classify cellular/subcellular phenotypes, simulate, and analyze pooled CRISPR-Cas9 imaging screens.."

        # Create icon buttons for the main apps
        for i, (app_name, app_data) in enumerate(self.main_gui_apps.items()):
            app_func, app_desc = app_data
            button = spacrButton(main_buttons_frame, text=app_name, command=lambda app_name=app_name, app_func=app_func: self.load_app(app_name, app_func), icon_name=app_name.lower(), size=100, show_text=False)
            button.grid(row=0, column=i + 1, padx=5, pady=5)
            self.main_buttons[button] = app_desc

        # Create icon buttons for the additional apps
        for i, (app_name, app_data) in enumerate(self.additional_gui_apps.items()):
            app_func, app_desc = app_data
            button = spacrButton(additional_buttons_frame, text=app_name, command=lambda app_name=app_name, app_func=app_func: self.load_app(app_name, app_func), icon_name=app_name.lower(), size=75, show_text=False)
            button.grid(row=0, column=i, padx=5, pady=5)
            self.additional_buttons[button] = app_desc

        # Update description initially
        self.update_description()
        #

    def update_description(self):
        # Check all buttons and update description if any has the active color
        for button, desc in {**self.main_buttons, **self.additional_buttons}.items():
            if button.canvas.itemcget(button.button_bg, "fill") == self.color_settings['active_color']:
                self.show_description(desc)
                return
        self.clear_description()

    def show_description(self, description):
        if self.description_label.winfo_exists():
            self.description_label.config(text=description)
            self.description_label.update_idletasks()  # Ensure the label updates immediately

    def clear_description(self):
        if self.description_label.winfo_exists():
            self.description_label.config(text="")
            self.description_label.update_idletasks()  # Ensure the label updates immediately

    def load_app(self, app_name, app_func):
        # Clear the current content frame
        self.clear_frame(self.canvas)

        # Initialize the selected app
        app_frame = tk.Frame(self.canvas)
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
