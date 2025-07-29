import tkinter as tk
from tkinter import ttk
from multiprocessing import set_start_method
from .gui_elements import spacrButton, create_menu_bar, set_dark_style
from .gui_core import initiate_root
from screeninfo import get_monitors
import webbrowser

class MainApp(tk.Tk):
    """
    The main graphical user interface (GUI) window for launching various SpaCr modules.

    This class creates a full-screen, themed Tkinter application with buttons to launch
    sub-GUIs for image mask generation, measurement, classification, and advanced analysis
    tools like regression, barcode mapping, and model activation visualization.
    """
    def __init__(self, default_app=None):
        """
        Initialize the SpaCr main application window.

        Sets up the full-screen GUI, applies dark theme styling, registers available
        sub-applications, and optionally launches a specific sub-application.

        Args:
            default_app (str, optional): If specified, directly launches the corresponding
                application view. Should match a key in `main_gui_apps` or `additional_gui_apps`.
        """
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
                self.width = width
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
            "Classify": (lambda frame: initiate_root(self, 'classify'), "Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images."),
        }

        self.additional_gui_apps = {
            "ML Analyze": (lambda frame: initiate_root(self, 'ml_analyze'), "Machine learning analysis of data."),
            "Map Barcodes": (lambda frame: initiate_root(self, 'map_barcodes'), "Map barcodes to data."),
            "Regression": (lambda frame: initiate_root(self, 'regression'), "Perform regression analysis."),
            "Recruitment": (lambda frame: initiate_root(self, 'recruitment'), "Analyze recruitment data."),
            "Activation": (lambda frame: initiate_root(self, 'activation'), "Generate activation maps of computer vision models and measure channel-activation correlation."),
        }

        self.selected_app = tk.StringVar()
        self.create_widgets()

        if default_app in self.main_gui_apps:
            self.load_app(default_app, self.main_gui_apps[default_app][0])
        elif default_app in self.additional_gui_apps:
            self.load_app(default_app, self.additional_gui_apps[default_app][0])

    def create_widgets(self):
        """
        Create and place all GUI widgets and layout frames.

        This includes the content and inner frames, sets dark style themes,
        and triggers the startup screen.
        """
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
        
    def _update_wraplength(self, event):
        """
        Update the wrap length of the description label based on current frame width.

        Args:
            event (tk.Event): The event triggering the resize (usually <Configure>).
        """
        if self.description_label.winfo_exists():
            # Use the actual width of the inner_frame as a proxy for full width
            available_width = self.inner_frame.winfo_width()
            if available_width > 0:
                self.description_label.config(wraplength=int(available_width * 0.9))  # or 0.9

    def create_startup_screen(self):
        """
        Create the initial screen with buttons to launch different SpaCr modules.

        This function generates main and additional app buttons with tooltips and
        handles layout and styling.
        """
        self.clear_frame(self.inner_frame)

        main_buttons_frame = tk.Frame(self.inner_frame)
        main_buttons_frame.pack(pady=10)
        set_dark_style(ttk.Style(), containers=[main_buttons_frame])

        additional_buttons_frame = tk.Frame(self.inner_frame)
        additional_buttons_frame.pack(pady=10)
        set_dark_style(ttk.Style(), containers=[additional_buttons_frame])
        
        description_frame = tk.Frame(self.inner_frame)
        description_frame.pack(fill=tk.X, pady=10)
        description_frame.columnconfigure(0, weight=1)
        
        set_dark_style(ttk.Style(), containers=[description_frame])
        style_out = set_dark_style(ttk.Style())
        font_loader = style_out['font_loader']
        font_size = style_out['font_size']
        
        self.description_label = tk.Label( description_frame, text="", wraplength=int(self.width * 0.9), justify="center", font=font_loader.get_font(size=font_size), fg=self.color_settings['fg_color'], bg=self.color_settings['bg_color'])
        
        # Pack it without expanding
        self.description_label.pack(pady=10)

        # Force character width and center it
        self.description_label.configure(width=int(self.width * 0.5 // 7))
        self.description_label.pack_configure(anchor='center')
        
        #logo_button = spacrButton(main_buttons_frame, text="SpaCR", command=lambda: self.load_app("logo_spacr", initiate_root), icon_name="logo_spacr", size=100, show_text=False)
        logo_button = spacrButton(main_buttons_frame,text="SpaCR",command=lambda: webbrowser.open_new("https://einarolafsson.github.io/spacr/tutorial/"),icon_name="logo_spacr",size=100,show_text=False)
        
        logo_button.grid(row=0, column=0, padx=5, pady=5)
        self.main_buttons[logo_button] = "spaCR provides a flexible toolset to extract single-cell images and measurements from high-content cell painting experiments, train deep-learning models to classify cellular/subcellular phenotypes, simulate, and analyze pooled CRISPR-Cas9 imaging screens. Click to open the spaCR tutorial in your browser."

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
        self.inner_frame.bind("<Configure>", self._update_wraplength)

    def update_description(self):
        """
        Update the description label based on the currently active (highlighted) button.

        If no button is active, the label is cleared.
        """
        for button, desc in {**self.main_buttons, **self.additional_buttons}.items():
            if button.canvas.itemcget(button.button_bg, "fill") == self.color_settings['active_color']:
                self.show_description(desc)
                return
        self.clear_description()

    def show_description(self, description):
        """
        Show a specific description text in the description label.

        Args:
            description (str): The text to be displayed.
        """
        if self.description_label.winfo_exists():
            self.description_label.config(text=description)
            self.description_label.update_idletasks()

    def clear_description(self):
        """
        Clear the content of the description label.
        """
        if self.description_label.winfo_exists():
            self.description_label.config(text="")
            self.description_label.update_idletasks()

    def load_app(self, app_name, app_func):
        """
        Load and display the selected application GUI in a new frame.

        Args:
            app_name (str): The name of the app being loaded.
            app_func (Callable): The function that initializes the app.
        """
        self.clear_frame(self.inner_frame)

        app_frame = tk.Frame(self.inner_frame)
        app_frame.pack(fill=tk.BOTH, expand=True)
        set_dark_style(ttk.Style(), containers=[app_frame])
        app_func(app_frame)

    def clear_frame(self, frame):
        """
        Remove all widgets from a given Tkinter frame.

        Args:
            frame (tk.Frame): The frame to clear.
        """
        for widget in frame.winfo_children():
            widget.destroy()

def gui_app():
    """
    Launch the main SpaCr GUI.

    This function initializes and runs the `MainApp` class with no default app specified,
    allowing users to select functionality from the startup screen.
    """
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    gui_app()
