import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os, requests
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
        self.configure(bg="black")
        style = ttk.Style()
        set_dark_style(style)

        self.gui_apps = {
            "Mask": (lambda frame: initiate_root(frame, 'mask'), "Generate cellpose masks for cells, nuclei and pathogen images."),
            "Measure": (lambda frame: initiate_root(frame, 'measure'), "Measure single object intensity and morphological feature. Crop and save single object image"),
            "Annotate": (lambda frame: initiate_root(frame, 'annotate'), "Annotation single object images on a grid. Annotations are saved to database."),
            "Make Masks": (lambda frame: initiate_root(frame, 'make_masks'),"Adjust pre-existing Cellpose models to your specific dataset for improved performance"),
            "Classify": (lambda frame: initiate_root(frame, 'classify'), "Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images."),
            "Sequencing": (lambda frame: initiate_root(frame, 'sequencing'), "Analyze sequensing data."),
            "Umap": (lambda frame: initiate_root(frame, 'umap'), "Generate UMAP embedings with datapoints represented as images.")
        }

        self.selected_app = tk.StringVar()
        self.create_widgets()

        if default_app in self.gui_apps:
            self.load_app(default_app, self.gui_apps[default_app][0])

    def create_widgets(self):
        # Create the menu bar
        create_menu_bar(self)

        # Create a canvas to hold the selected app and other elements
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a frame inside the canvas to hold the main content
        self.content_frame = tk.Frame(self.canvas, bg="black")
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Create startup screen with buttons for each GUI app
        self.create_startup_screen()

    def create_startup_screen(self):
        self.clear_frame(self.content_frame)

        # Create a frame for the logo and description
        logo_frame = tk.Frame(self.content_frame, bg="black")
        logo_frame.pack(pady=20, expand=True)

        # Load the logo image
        if not self.load_logo(logo_frame):
            tk.Label(logo_frame, text="Logo not found", bg="black", fg="white", font=('Helvetica', 24)).pack(padx=10, pady=10)

        # Add SpaCr text below the logo with padding for sharper text
        tk.Label(logo_frame, text="SpaCr", bg="black", fg="#008080", font=('Helvetica', 24)).pack(padx=10, pady=10)

        # Create a frame for the buttons and descriptions
        buttons_frame = tk.Frame(self.content_frame, bg="black")
        buttons_frame.pack(pady=10, expand=True, padx=10)

        for i, (app_name, app_data) in enumerate(self.gui_apps.items()):
            app_func, app_desc = app_data

            # Create custom button with text
            button = spacrButton(buttons_frame, text=app_name, command=lambda app_name=app_name, app_func=app_func: self.load_app(app_name, app_func), font=('Helvetica', 12))
            button.grid(row=i, column=0, pady=10, padx=10, sticky="w")

            description_label = tk.Label(buttons_frame, text=app_desc, bg="black", fg="white", wraplength=800, justify="left", font=('Helvetica', 12))
            description_label.grid(row=i, column=1, pady=10, padx=10, sticky="w")

        # Ensure buttons have a fixed width
        buttons_frame.grid_columnconfigure(0, minsize=150)
        # Ensure descriptions expand as needed
        buttons_frame.grid_columnconfigure(1, weight=1)

    def load_logo(self, frame):
        def download_image(url, save_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image from {url}: {e}")
                return False

        try:
            img_path = os.path.join(os.path.dirname(__file__), 'logo_spacr.png')
            logo_image = Image.open(img_path)
        except (FileNotFoundError, Image.UnidentifiedImageError):
            print(f"File {img_path} not found or is not a valid image. Attempting to download from GitHub.")
            if download_image('https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/logo_spacr.png', img_path):
                try:
                    print(f"Downloaded file size: {os.path.getsize(img_path)} bytes")
                    logo_image = Image.open(img_path)
                except Image.UnidentifiedImageError as e:
                    print(f"Downloaded file is not a valid image: {e}")
                    return False
            else:
                return False
        except Exception as e:
            print(f"An error occurred while loading the logo: {e}")
            return False
        try:
            screen_height = frame.winfo_screenheight()
            new_height = int(screen_height // 4)
            logo_image = logo_image.resize((new_height, new_height), Image.Resampling.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(frame, image=logo_photo, bg="black")
            logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            logo_label.pack()
            return True
        except Exception as e:
            print(f"An error occurred while processing the logo image: {e}")
            return False

    def load_app(self, app_name, app_func):
        # Clear the current content frame
        self.clear_frame(self.content_frame)

        # Initialize the selected app
        app_frame = tk.Frame(self.content_frame, bg="black")
        app_frame.pack(fill=tk.BOTH, expand=True)
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
