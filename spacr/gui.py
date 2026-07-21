import tkinter as tk
from tkinter import ttk
from multiprocessing import set_start_method
from .gui_elements import spacrButton, spacrCard, create_menu_bar, set_dark_style
from .gui_core import initiate_root
from screeninfo import get_monitors
import webbrowser

class MainApp(tk.Tk):
    def __init__(self, default_app=None):
        super().__init__()

        # Enable font smoothing on all platforms
        self._setup_font_rendering()

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
            "Annotate": (lambda frame: initiate_root(self, 'annotate'), "Annotation single object images on a grid. Annotations are saved to database."),
            "Make Masks": (lambda frame: initiate_root(self, 'make_masks'), "Adjust pre-existing Cellpose models to your specific dataset for improved performance"),
            "Classify": (lambda frame: initiate_root(self, 'classify'), "Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images."),
        }

        self.additional_gui_apps = {
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
            
    def _setup_font_rendering(self):
        import platform
        system = platform.system()
        
        if system == 'Linux':
            # Enable subpixel anti-aliasing
            self.option_add('*Font', 'OpenSans 12')
            self.tk.call('tk', 'scaling', self.winfo_fpixels('1i') / 72.0)
            try:
                self.option_add('*TkDefaultFont', 'OpenSans 12')
                self.tk.eval("""
                    option add *font {OpenSans 12}
                    font configure TkDefaultFont -family OpenSans -size 12
                    font configure TkTextFont -family OpenSans -size 12
                    font configure TkMenuFont -family OpenSans -size 12
                    font configure TkFixedFont -family monospace -size 11
                """)
            except Exception:
                pass
        elif system == 'Darwin':
            try:
                self.tk.eval("""
                    font configure TkDefaultFont -family {Open Sans} -size 13
                    font configure TkTextFont -family {Open Sans} -size 13
                    font configure TkMenuFont -family {Open Sans} -size 13
                """)
            except Exception:
                pass
        elif system == 'Windows':
            try:
                self.tk.eval("""
                    font configure TkDefaultFont -family {Open Sans} -size 10
                    font configure TkTextFont -family {Open Sans} -size 10
                    font configure TkMenuFont -family {Open Sans} -size 10
                """)
            except Exception:
                pass

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
        
    def _update_wraplength(self, event):
        if self.description_label.winfo_exists():
            # Use the actual width of the inner_frame as a proxy for full width
            available_width = self.inner_frame.winfo_width()
            if available_width > 0:
                self.description_label.config(wraplength=int(available_width * 0.9))  # or 0.9

    def create_startup_screen(self):
        self.clear_frame(self.inner_frame)

        # Pull the shared palette + typography scale so the startup screen
        # matches the rest of the modernised GUI.
        style_out = set_dark_style(ttk.Style())
        bg = style_out['bg_color']
        fg = style_out['fg_color']
        muted = style_out.get('muted_color', fg)
        spacing = style_out.get('spacing', {'sm': 8, 'md': 12, 'lg': 16, 'xl': 24})
        font_sizes = style_out.get('font_sizes', {'small': 11, 'body': 12,
                                                    'header': 14, 'title': 18})
        font_loader = style_out.get('font_loader')

        def _font(size_key, weight="normal"):
            size = font_sizes.get(size_key, style_out['font_size'])
            if font_loader:
                return font_loader.get_font(size=size)
            return (style_out['font_family'], size, weight)

        # --- Title block: "SpaCR" + subtitle ----------------------------
        title_frame = tk.Frame(self.inner_frame, bg=bg)
        title_frame.pack(pady=(spacing['xl'], spacing['md']))
        tk.Label(
            title_frame, text="SpaCR", font=_font('title'),
            bg=bg, fg=fg, cursor="hand2",
        ).pack()
        subtitle = tk.Label(
            title_frame,
            text="Spatial single-cell analysis for microscopy",
            font=_font('small'), bg=bg, fg=muted,
        )
        subtitle.pack(pady=(spacing['xs'], 0))

        # --- Core applications card -------------------------------------
        main_card = spacrCard(self.inner_frame, title="Core applications",
                              padding='md')
        main_card.pack(pady=(spacing['md'], spacing['sm']),
                       padx=spacing['lg'], anchor='center')

        # Logo button (opens the tutorial) sits leftmost in the core row.
        logo_button = spacrButton(
            main_card.body, text="SpaCR",
            command=lambda: webbrowser.open_new(
                "https://einarolafsson.github.io/spacr/tutorial/"),
            icon_name="logo_spacr", size=90, show_text=False,
        )
        logo_button.grid(row=0, column=0, padx=spacing['sm'], pady=spacing['sm'])
        self.main_buttons[logo_button] = (
            "spaCR: spatial single-cell analysis tools for microscopy data. "
            "Click to open the tutorial. (under construction)"
        )

        for i, (app_name, app_data) in enumerate(self.main_gui_apps.items()):
            app_func, app_desc = app_data
            button = spacrButton(
                main_card.body, text=app_name,
                command=lambda app_name=app_name, app_func=app_func:
                    self.load_app(app_name, app_func),
                icon_name=app_name.lower(), size=90, show_text=False,
            )
            button.grid(row=0, column=i + 1,
                        padx=spacing['sm'], pady=spacing['sm'])
            self.main_buttons[button] = app_desc

        # --- Additional tools card --------------------------------------
        extra_card = spacrCard(self.inner_frame, title="Additional tools",
                               padding='md')
        extra_card.pack(pady=(spacing['sm'], spacing['md']),
                        padx=spacing['lg'], anchor='center')

        for i, (app_name, app_data) in enumerate(self.additional_gui_apps.items()):
            app_func, app_desc = app_data
            button = spacrButton(
                extra_card.body, text=app_name,
                command=lambda app_name=app_name, app_func=app_func:
                    self.load_app(app_name, app_func),
                icon_name=app_name.lower(), size=66, show_text=False,
            )
            button.grid(row=0, column=i,
                        padx=spacing['sm'], pady=spacing['sm'])
            self.additional_buttons[button] = app_desc

        # --- Description panel (bottom, muted text) ---------------------
        description_frame = tk.Frame(self.inner_frame, bg=bg)
        description_frame.pack(fill=tk.X, pady=(spacing['sm'], spacing['xl']),
                                padx=spacing['xl'])
        description_frame.columnconfigure(0, weight=1)

        self.description_label = tk.Label(
            description_frame, text="",
            wraplength=int(self.width * 0.7),
            justify="center", font=_font('body'),
            fg=muted, bg=bg,
        )
        self.description_label.pack(pady=spacing['sm'])
        self.description_label.configure(width=int(self.width * 0.5 // 7))
        self.description_label.pack_configure(anchor='center')

        self.update_description()
        self.inner_frame.bind("<Configure>", self._update_wraplength)

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

#def gui_app():
#    app = MainApp()
#    app.mainloop()

def gui_app():
    from multiprocessing import freeze_support
    freeze_support()
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    gui_app()
