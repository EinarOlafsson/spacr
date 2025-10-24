import os, threading, time, sqlite3, webbrowser, pyautogui, random, cv2
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog
from tkinter import font
from queue import Queue
from tkinter import Label, Frame, Button
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, HTML
import imageio.v2 as imageio
from collections import deque
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity
from skimage.draw import polygon, line
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte
from scipy.ndimage import binary_fill_holes, label, gaussian_filter
from tkinter import ttk, scrolledtext
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


fig = None

def restart_gui_app(root):
    """
    Restarts the GUI application by destroying the current instance
    and launching a fresh one.
    """
    try:
        # Destroy the current root window
        root.destroy()
        
        # Import and launch a new instance of the application
        from .gui import gui_app
        new_root = tk.Tk()  # Create a fresh Tkinter root instance
        gui_app()
    except Exception as e:
        print(f"Error restarting GUI application: {e}")

def create_menu_bar(root):
    from .gui import initiate_root
    gui_apps = {
        "Mask": lambda: initiate_root(root, settings_type='mask'),
        "Measure": lambda: initiate_root(root, settings_type='measure'),
        "Annotate (Beta)": lambda: initiate_root(root, settings_type='annotate'),
        "Make Masks": lambda: initiate_root(root, settings_type='make_masks'),
        "Classify": lambda: initiate_root(root, settings_type='classify'),
        "Umap": lambda: initiate_root(root, settings_type='umap'),
        "Train Cellpose": lambda: initiate_root(root, settings_type='train_cellpose'),
        "ML Analyze": lambda: initiate_root(root, settings_type='ml_analyze'),
        "Cellpose Masks": lambda: initiate_root(root, settings_type='cellpose_masks'),
        "Cellpose All": lambda: initiate_root(root, settings_type='cellpose_all'),
        "Map Barcodes": lambda: initiate_root(root, settings_type='map_barcodes'),
        "Regression": lambda: initiate_root(root, settings_type='regression'),
        "Activation": lambda: initiate_root(root, settings_type='activation'),
        "Recruitment (Beta)": lambda: initiate_root(root, settings_type='recruitment')
    }

    # Create the menu bar
    menu_bar = tk.Menu(root, bg="#008080", fg="white")

    # Create a "SpaCr Applications" menu
    app_menu = tk.Menu(menu_bar, tearoff=0, bg="#008080", fg="white")
    menu_bar.add_cascade(label="SpaCr Applications", menu=app_menu)

    # Add options to the "SpaCr Applications" menu
    for app_name, app_func in gui_apps.items():
        app_menu.add_command(
            label=app_name,
            command=app_func
        )

    # Add a separator and an exit option
    app_menu.add_separator()
    #app_menu.add_command(label="Home",command=lambda: restart_gui_app(root))
    app_menu.add_command(label="Help", command=lambda: webbrowser.open("https://einarolafsson.github.io/spacr/index.html"))
    app_menu.add_command(label="Exit", command=root.quit)

    # Configure the menu for the root window
    root.config(menu=menu_bar)

def set_element_size():

    screen_width, screen_height = pyautogui.size()
    screen_area = screen_width * screen_height
    
    # Calculate sizes based on screen dimensions
    btn_size = int((screen_area * 0.002) ** 0.5)  # Button size as a fraction of screen area
    bar_size = screen_height // 20  # Bar size based on screen height
    settings_width = screen_width // 4  # Settings panel width as a fraction of screen width
    panel_width = screen_width - settings_width  # Panel width as a fraction of screen width
    panel_height = screen_height // 6  # Panel height as a fraction of screen height
    
    size_dict = {
        'btn_size': btn_size,
        'bar_size': bar_size,
        'settings_width': settings_width,
        'panel_width': panel_width,
        'panel_height': panel_height
    }
    return size_dict
    
def set_dark_style(style, parent_frame=None, containers=None, widgets=None, font_family="OpenSans", font_size=12, bg_color='black', fg_color='white', active_color='blue', inactive_color='dark_gray'):

    if active_color == 'teal':
        active_color = '#008080'
    if inactive_color == 'dark_gray':
        inactive_color = '#2B2B2B' # '#333333' #'#050505'
    if bg_color == 'black':
        bg_color = '#000000'
    if fg_color == 'white':
        fg_color = '#ffffff'
    if active_color == 'blue':
        active_color = '#007BFF'

    padding = '5 5 5 5'
    font_style = tkFont.Font(family=font_family, size=font_size)

    if font_family == 'OpenSans':
        font_loader = spacrFont(font_name='OpenSans', font_style='Regular', font_size=12)
    else:
        font_loader = None
        
    style.theme_use('clam')
    
    style.configure('TEntry', padding=padding)
    style.configure('TCombobox', padding=padding)
    style.configure('Spacr.TEntry', padding=padding)
    style.configure('TEntry', padding=padding)
    style.configure('Spacr.TEntry', padding=padding)
    style.configure('Custom.TLabel', padding=padding)
    style.configure('TButton', padding=padding)
    style.configure('TFrame', background=bg_color)
    style.configure('TPanedwindow', background=bg_color)
    if font_loader:
        style.configure('TLabel', background=bg_color, foreground=fg_color, font=font_loader.get_font(size=font_size))
    else:
        style.configure('TLabel', background=bg_color, foreground=fg_color, font=(font_family, font_size))

    if parent_frame:
        parent_frame.configure(bg=bg_color)
        parent_frame.grid_rowconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

    if containers:
        for container in containers:
            if isinstance(container, ttk.Frame):
                container_style = ttk.Style()
                container_style.configure(f'{container.winfo_class()}.TFrame', background=bg_color)
                container.configure(style=f'{container.winfo_class()}.TFrame')
            else:
                container.configure(bg=bg_color)

    if widgets:
        for widget in widgets:
            if isinstance(widget, (tk.Label, tk.Button, tk.Frame, ttk.LabelFrame, tk.Canvas)):
                widget.configure(bg=bg_color)
            if isinstance(widget, (tk.Label, tk.Button)):
                if font_loader:
                    widget.configure(fg=fg_color, font=font_loader.get_font(size=font_size))
                else:
                    widget.configure(fg=fg_color, font=(font_family, font_size))
            if isinstance(widget, scrolledtext.ScrolledText):
                widget.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            if isinstance(widget, tk.OptionMenu):
                if font_loader:
                    widget.configure(bg=bg_color, fg=fg_color, font=font_loader.get_font(size=font_size))
                else:
                    widget.configure(bg=bg_color, fg=fg_color, font=(font_family, font_size))
                menu = widget['menu']
                if font_loader:
                    menu.configure(bg=bg_color, fg=fg_color, font=font_loader.get_font(size=font_size))
                else:
                    menu.configure(bg=bg_color, fg=fg_color, font=(font_family, font_size))

    return {'font_loader':font_loader, 'font_family': font_family, 'font_size': font_size, 'bg_color': bg_color, 'fg_color': fg_color, 'active_color': active_color, 'inactive_color': inactive_color}

class spacrFont:
    def __init__(self, font_name, font_style, font_size=12):
        """
        Initializes the FontLoader class.

        Parameters:
        - font_name: str, the name of the font (e.g., 'OpenSans').
        - font_style: str, the style of the font (e.g., 'Regular', 'Bold').
        - font_size: int, the size of the font (default: 12).
        """
        self.font_name = font_name
        self.font_style = font_style
        self.font_size = font_size

        # Determine the path based on the font name and style
        self.font_path = self.get_font_path(font_name, font_style)

        # Register the font with Tkinter
        self.load_font()

    def get_font_path(self, font_name, font_style):
        """
        Returns the font path based on the font name and style.

        Parameters:
        - font_name: str, the name of the font.
        - font_style: str, the style of the font.

        Returns:
        - str, the path to the font file.
        """
        base_dir = os.path.dirname(__file__)
        
        if font_name == 'OpenSans':
            if font_style == 'Regular':
                return os.path.join(base_dir, 'resources/font/open_sans/static/OpenSans-Regular.ttf')
            elif font_style == 'Bold':
                return os.path.join(base_dir, 'resources/font/open_sans/static/OpenSans-Bold.ttf')
            elif font_style == 'Italic':
                return os.path.join(base_dir, 'resources/font/open_sans/static/OpenSans-Italic.ttf')
            # Add more styles as needed
        # Add more fonts as needed
        
        raise ValueError(f"Font '{font_name}' with style '{font_style}' not found.")

    def load_font(self):
        """
        Loads the font into Tkinter.
        """
        try:
            font.Font(family=self.font_name, size=self.font_size)
        except tk.TclError:
            # Load the font manually if it's not already loaded
            self.tk_font = font.Font(
                name=self.font_name,
                file=self.font_path,
                size=self.font_size
            )

    def get_font(self, size=None):
        """
        Returns the font in the specified size.

        Parameters:
        - size: int, the size of the font (optional).

        Returns:
        - tkFont.Font object.
        """
        if size is None:
            size = self.font_size
        return font.Font(family=self.font_name, size=size)

class spacrContainer(tk.Frame):
    def __init__(self, parent, orient=tk.VERTICAL, bg=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.orient = orient
        self.bg = bg if bg else 'lightgrey'
        self.sash_thickness = 10

        self.panes = []
        self.sashes = []
        self.bind("<Configure>", self.on_configure)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def add(self, widget, stretch='always'):
        print(f"Adding widget: {widget} with stretch: {stretch}")
        pane = tk.Frame(self, bg=self.bg)
        pane.grid_propagate(False)
        widget.grid(in_=pane, sticky="nsew")  # Use grid for the widget within the pane
        self.panes.append((pane, widget))

        if len(self.panes) > 1:
            self.create_sash()

        self.reposition_panes()

    def create_sash(self):
        sash = tk.Frame(self, bg=self.bg, cursor='sb_v_double_arrow' if self.orient == tk.VERTICAL else 'sb_h_double_arrow', height=self.sash_thickness, width=self.sash_thickness)
        sash.bind("<Enter>", self.on_enter_sash)
        sash.bind("<Leave>", self.on_leave_sash)
        sash.bind("<ButtonPress-1>", self.start_resize)
        self.sashes.append(sash)

    def reposition_panes(self):
        if not self.panes:
            return

        total_size = self.winfo_height() if self.orient == tk.VERTICAL else self.winfo_width()
        pane_size = total_size // len(self.panes)

        print(f"Total size: {total_size}, Pane size: {pane_size}, Number of panes: {len(self.panes)}")

        for i, (pane, widget) in enumerate(self.panes):
            if self.orient == tk.VERTICAL:
                pane.grid(row=i * 2, column=0, sticky="nsew", pady=(0, self.sash_thickness if i < len(self.panes) - 1 else 0))
            else:
                pane.grid(row=0, column=i * 2, sticky="nsew", padx=(0, self.sash_thickness if i < len(self.panes) - 1 else 0))

        for i, sash in enumerate(self.sashes):
            if self.orient == tk.VERTICAL:
                sash.grid(row=(i * 2) + 1, column=0, sticky="ew")
            else:
                sash.grid(row=0, column=(i * 2) + 1, sticky="ns")

    def on_configure(self, event):
        print(f"Configuring container: {self}")
        self.reposition_panes()

    def on_enter_sash(self, event):
        event.widget.config(bg='blue')

    def on_leave_sash(self, event):
        event.widget.config(bg=self.bg)

    def start_resize(self, event):
        sash = event.widget
        self.start_pos = event.y_root if self.orient == tk.VERTICAL else event.x_root
        self.start_size = sash.winfo_y() if self.orient == tk.VERTICAL else sash.winfo_x()
        sash.bind("<B1-Motion>", self.perform_resize)

    def perform_resize(self, event):
        sash = event.widget
        delta = (event.y_root - self.start_pos) if self.orient == tk.VERTICAL else (event.x_root - self.start_pos)
        new_size = self.start_size + delta

        for i, (pane, widget) in enumerate(self.panes):
            if self.orient == tk.VERTICAL:
                new_row = max(0, new_size // self.sash_thickness)
                if pane.winfo_y() >= new_size:
                    pane.grid_configure(row=new_row)
                elif pane.winfo_y() < new_size and i > 0:
                    previous_row = max(0, (new_size - pane.winfo_height()) // self.sash_thickness)
                    self.panes[i - 1][0].grid_configure(row=previous_row)
            else:
                new_col = max(0, new_size // self.sash_thickness)
                if pane.winfo_x() >= new_size:
                    pane.grid_configure(column=new_col)
                elif pane.winfo_x() < new_size and i > 0:
                    previous_col = max(0, (new_size - pane.winfo_width()) // self.sash_thickness)
                    self.panes[i - 1][0].grid_configure(column=previous_col)

        self.reposition_panes()

class spacrEntry(tk.Frame):
    def __init__(self, parent, textvariable=None, outline=False, width=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Set dark style
        style_out = set_dark_style(ttk.Style())
        self.bg_color = style_out['inactive_color']
        self.active_color = style_out['active_color']
        self.fg_color = style_out['fg_color']
        self.outline = outline
        self.font_family = style_out['font_family']
        self.font_size = style_out['font_size']
        self.font_loader = style_out['font_loader']
        
        # Set the background color of the frame
        self.configure(bg=style_out['bg_color'])

        # Create a canvas for the rounded rectangle background
        if width is None:
            self.canvas_width = 220  # Adjusted for padding
        else:
            self.canvas_width = width
        self.canvas_height = 40   # Adjusted for padding
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bd=0, highlightthickness=0, relief='ridge', bg=style_out['bg_color'])
        self.canvas.pack()
        
        # Create the entry widget
        if self.font_loader:
            self.entry = tk.Entry(self, textvariable=textvariable, bd=0, highlightthickness=0, fg=self.fg_color, font=self.font_loader.get_font(size=self.font_size), bg=self.bg_color)
        else:
            self.entry = tk.Entry(self, textvariable=textvariable, bd=0, highlightthickness=0, fg=self.fg_color, font=(self.font_family, self.font_size), bg=self.bg_color)
        self.entry.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=self.canvas_width - 30, height=20)  # Centered positioning
        
        # Bind events to change the background color on focus
        self.entry.bind("<FocusIn>", self.on_focus_in)
        self.entry.bind("<FocusOut>", self.on_focus_out)
        
        self.draw_rounded_rectangle(self.bg_color)

    def draw_rounded_rectangle(self, color):
        radius = 15  # Increased radius for more rounded corners
        x0, y0 = 10, 5
        x1, y1 = self.canvas_width - 10, self.canvas_height - 5
        self.canvas.delete("all")
        self.canvas.create_arc((x0, y0, x0 + radius, y0 + radius), start=90, extent=90, fill=color, outline=color)
        self.canvas.create_arc((x1 - radius, y0, x1, y0 + radius), start=0, extent=90, fill=color, outline=color)
        self.canvas.create_arc((x0, y1 - radius, x0 + radius, y1), start=180, extent=90, fill=color, outline=color)
        self.canvas.create_arc((x1 - radius, y1 - radius, x1, y1), start=270, extent=90, fill=color, outline=color)
        self.canvas.create_rectangle((x0 + radius / 2, y0, x1 - radius / 2, y1), fill=color, outline=color)
        self.canvas.create_rectangle((x0, y0 + radius / 2, x1, y1 - radius / 2), fill=color, outline=color)
    
    def on_focus_in(self, event):
        self.draw_rounded_rectangle(self.active_color)
        self.entry.config(bg=self.active_color)
    
    def on_focus_out(self, event):
        self.draw_rounded_rectangle(self.bg_color)
        self.entry.config(bg=self.bg_color)

class spacrCheck(tk.Frame):
    def __init__(self, parent, text="", variable=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        style_out = set_dark_style(ttk.Style())
        self.bg_color = style_out['bg_color']
        self.active_color = style_out['active_color']
        self.fg_color = style_out['fg_color']
        self.inactive_color = style_out['inactive_color']
        self.variable = variable

        self.configure(bg=self.bg_color)

        # Create a canvas for the rounded square background
        self.canvas_width = 20
        self.canvas_height = 20
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bd=0, highlightthickness=0, relief='ridge', bg=self.bg_color)
        self.canvas.pack()

        # Draw the initial rounded square based on the variable's value
        self.draw_rounded_square(self.active_color if self.variable.get() else self.inactive_color)

        # Bind variable changes to update the checkbox
        self.variable.trace_add('write', self.update_check)

        # Bind click event to toggle the variable
        self.canvas.bind("<Button-1>", self.toggle_variable)

    def draw_rounded_square(self, color):
        radius = 5  # Adjust the radius for more rounded corners
        x0, y0 = 2, 2
        x1, y1 = 18, 18
        self.canvas.delete("all")
        self.canvas.create_arc((x0, y0, x0 + radius, y0 + radius), start=90, extent=90, fill=color, outline=self.fg_color)
        self.canvas.create_arc((x1 - radius, y0, x1, y0 + radius), start=0, extent=90, fill=color, outline=self.fg_color)
        self.canvas.create_arc((x0, y1 - radius, x0 + radius, y1), start=180, extent=90, fill=color, outline=self.fg_color)
        self.canvas.create_arc((x1 - radius, y1 - radius, x1, y1), start=270, extent=90, fill=color, outline=self.fg_color)
        self.canvas.create_rectangle((x0 + radius / 2, y0, x1 - radius / 2, y1), fill=color, outline=color)
        self.canvas.create_rectangle((x0, y0 + radius / 2, x1, y1 - radius / 2), fill=color, outline=color)
        self.canvas.create_line(x0 + radius / 2, y0, x1 - radius / 2, y0, fill=self.fg_color)
        self.canvas.create_line(x0 + radius / 2, y1, x1 - radius / 2, y1, fill=self.fg_color)
        self.canvas.create_line(x0, y0 + radius / 2, x0, y1 - radius / 2, fill=self.fg_color)
        self.canvas.create_line(x1, y0 + radius / 2, x1, y1 - radius / 2, fill=self.fg_color)

    def update_check(self, *args):
        self.draw_rounded_square(self.active_color if self.variable.get() else self.inactive_color)

    def toggle_variable(self, event):
        self.variable.set(not self.variable.get())

class spacrCombo(tk.Frame):
    def __init__(self, parent, textvariable=None, values=None, width=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Set dark style
        style_out = set_dark_style(ttk.Style())
        self.bg_color = style_out['bg_color']
        self.active_color = style_out['active_color']
        self.fg_color = style_out['fg_color']
        self.inactive_color = style_out['inactive_color']
        self.font_family = style_out['font_family']
        self.font_size = style_out['font_size']
        self.font_loader = style_out['font_loader']

        self.values = values or []

        # Create a canvas for the rounded rectangle background
        self.canvas_width = width if width is not None else 220  # Adjusted for padding
        self.canvas_height = 40   # Adjusted for padding
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bd=0, highlightthickness=0, relief='ridge', bg=self.bg_color)
        self.canvas.pack()
        
        self.var = textvariable if textvariable else tk.StringVar()
        self.selected_value = self.var.get()
        
        # Create the label to display the selected value
        if self.font_loader:
            self.label = tk.Label(self, text=self.selected_value, bg=self.inactive_color, fg=self.fg_color, font=self.font_loader.get_font(size=self.font_size))
        else:
            self.label = tk.Label(self, text=self.selected_value, bg=self.inactive_color, fg=self.fg_color, font=(self.font_family, self.font_size))
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Bind events to open the dropdown menu
        self.canvas.bind("<Button-1>", self.on_click)
        self.label.bind("<Button-1>", self.on_click)
        
        self.draw_rounded_rectangle(self.inactive_color)

        self.dropdown_menu = None

    def draw_rounded_rectangle(self, color):
        radius = 15  # Increased radius for more rounded corners
        x0, y0 = 10, 5
        x1, y1 = self.canvas_width - 10, self.canvas_height - 5
        self.canvas.delete("all")
        self.canvas.create_arc((x0, y0, x0 + radius, y0 + radius), start=90, extent=90, fill=color, outline=color)
        self.canvas.create_arc((x1 - radius, y0, x1, y0 + radius), start=0, extent=90, fill=color, outline=color)
        self.canvas.create_arc((x0, y1 - radius, x0 + radius, y1), start=180, extent=90, fill=color, outline=color)
        self.canvas.create_arc((x1 - radius, y1 - radius, x1, y1), start=270, extent=90, fill=color, outline=color)
        self.canvas.create_rectangle((x0 + radius / 2, y0, x1 - radius / 2, y1), fill=color, outline=color)
        self.canvas.create_rectangle((x0, y0 + radius / 2, x1, y1 - radius / 2), fill=color, outline=color)
        self.label.config(bg=color)  # Update label background to match rectangle color

    def on_click(self, event):
        if self.dropdown_menu is None:
            self.open_dropdown()
        else:
            self.close_dropdown()

    def open_dropdown(self):
        self.draw_rounded_rectangle(self.active_color)
        
        self.dropdown_menu = tk.Toplevel(self)
        self.dropdown_menu.wm_overrideredirect(True)
        
        x, y, width, height = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        self.dropdown_menu.geometry(f"{width}x{len(self.values) * 30}+{x}+{y + height}")
        
        for index, value in enumerate(self.values):
            display_text = value if value is not None else 'None'
            if self.font_loader:
                item = tk.Label(self.dropdown_menu, text=display_text, bg=self.inactive_color, fg=self.fg_color, font=self.font_loader.get_font(size=self.font_size), anchor='w')
            else:
                item = tk.Label(self.dropdown_menu, text=display_text, bg=self.inactive_color, fg=self.fg_color, font=(self.font_family, self.font_size), anchor='w')
            item.pack(fill='both')
            item.bind("<Button-1>", lambda e, v=value: self.on_select(v))
            item.bind("<Enter>", lambda e, w=item: w.config(bg=self.active_color))
            item.bind("<Leave>", lambda e, w=item: w.config(bg=self.inactive_color))

    def close_dropdown(self):
        self.draw_rounded_rectangle(self.inactive_color)
        
        if self.dropdown_menu:
            self.dropdown_menu.destroy()
            self.dropdown_menu = None

    def on_select(self, value):
        display_text = value if value is not None else 'None'
        self.var.set(value)
        self.label.config(text=display_text)
        self.selected_value = value
        self.close_dropdown()

    def set(self, value):
        display_text = value if value is not None else 'None'
        self.var.set(value)
        self.label.config(text=display_text)
        self.selected_value = value

class spacrDropdownMenu(tk.Frame):

    def __init__(self, parent, variable, options, command=None, font=None, size=50, **kwargs):
        super().__init__(parent, **kwargs)
        self.variable = variable
        self.options = options
        self.command = command
        self.text = "Settings"
        self.size = size

        # Apply dark style and get color settings
        style_out = set_dark_style(ttk.Style())
        self.font_size = style_out['font_size']
        self.font_loader = style_out['font_loader']

        # Button size configuration
        self.button_width = int(size * 3)
        self.canvas_width = self.button_width + 4
        self.canvas_height = self.size + 4

        # Create the canvas and rounded button
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, highlightthickness=0, bg=style_out['bg_color'])
        self.canvas.grid(row=0, column=0)

        # Apply dark style and get color settings
        color_settings = set_dark_style(ttk.Style(), containers=[self], widgets=[self.canvas])
        self.inactive_color = color_settings['inactive_color']
        self.active_color = color_settings['active_color']
        self.fg_color = color_settings['fg_color']
        self.bg_color = style_out['bg_color']

        # Create the button with rounded edges
        self.button_bg = self.create_rounded_rectangle(2, 2, self.button_width + 2, self.size + 2, radius=20, fill=self.inactive_color, outline=self.inactive_color)

        # Create and place the label on the button
        if self.font_loader:
            self.font_style = self.font_loader.get_font(size=self.font_size)
        else:
            self.font_style = font if font else ("Arial", 12)

        self.button_text = self.canvas.create_text(self.button_width // 2, self.size // 2 + 2, text=self.text, fill=self.fg_color, font=self.font_style, anchor="center")

        # Bind events for button behavior
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

        # Create a popup menu with the desired background color
        self.menu = tk.Menu(self, tearoff=0, bg=self.bg_color, fg=self.fg_color)
        for option in self.options:
            self.menu.add_command(label=option, command=lambda opt=option: self.on_select(opt))

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    def on_enter(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill=self.active_color)

    def on_leave(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill=self.inactive_color)

    def on_click(self, event=None):
        self.post_menu()

    def post_menu(self):
        x, y, width, height = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        self.menu.post(x, y + height)

    def on_select(self, option):
        if self.command:
            self.command(option)

    def update_styles(self, active_categories=None):
        style_out = set_dark_style(ttk.Style(), widgets=[self.menu])

        if active_categories is not None:
            for idx in range(self.menu.index("end") + 1):
                option = self.menu.entrycget(idx, "label")
                if option in active_categories:
                    self.menu.entryconfig(idx, background=style_out['active_color'], foreground=style_out['fg_color'])
                else:
                    self.menu.entryconfig(idx, background=style_out['bg_color'], foreground=style_out['fg_color'])

class spacrCheckbutton(ttk.Checkbutton):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.configure(text=self.text, variable=self.variable, command=self.command, style='Spacr.TCheckbutton')
        style = ttk.Style()
        _ = set_dark_style(style, widgets=[self])

class spacrProgressBar(ttk.Progressbar):
    def __init__(self, parent, label=True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Get the style colors
        style_out = set_dark_style(ttk.Style())

        self.fg_color = style_out['fg_color']
        self.bg_color = style_out['bg_color']
        self.active_color = style_out['active_color']
        self.inactive_color = style_out['inactive_color']
        self.font_size = style_out['font_size']
        self.font_loader = style_out['font_loader']

        # Configure the style for the progress bar
        self.style = ttk.Style()
        
        # Remove any borders and ensure the active color fills the entire space
        self.style.configure(
            "spacr.Horizontal.TProgressbar",
            troughcolor=self.inactive_color, # Set the trough to bg color
            background=self.active_color,    # Active part is the active color
            borderwidth=0,                   # Remove border width
            pbarrelief="flat",               # Flat relief for the progress bar
            troughrelief="flat",             # Flat relief for the trough
            thickness=20,                    # Set the thickness of the progress bar
            darkcolor=self.active_color,     # Ensure darkcolor matches the active color
            lightcolor=self.active_color,    # Ensure lightcolor matches the active color
            bordercolor=self.bg_color        # Set the border color to the background color to hide it
        )

        self.configure(style="spacr.Horizontal.TProgressbar")

        # Set initial value to 0
        self['value'] = 0

        # Track whether to show the progress label
        self.label = label

        # Create the progress label with text wrapping
        if self.label:
            self.progress_label = tk.Label(
                parent,
                text="Processing: 0/0",
                anchor='w',
                justify='left',
                bg=self.inactive_color,
                fg=self.fg_color,
                wraplength=300,
                font=self.font_loader.get_font(size=self.font_size)
            )
            self.progress_label.grid_forget()

        # Initialize attributes for time and operation
        self.operation_type = None
        self.additional_info = None

    def set_label_position(self):
        if self.label and self.progress_label:
            row_info = self.grid_info().get('rowID', 0)
            col_info = self.grid_info().get('columnID', 0)
            col_span = self.grid_info().get('columnspan', 1)
            self.progress_label.grid(row=row_info + 1, column=col_info, columnspan=col_span, pady=5, padx=5, sticky='ew')

    def update_label(self):
        if self.label and self.progress_label:
            # Start with the base progress information
            label_text = f"Processing: {self['value']}/{self['maximum']}"
            
            # Include the operation type if it exists
            if self.operation_type:
                label_text += f", {self.operation_type}"
            
            # Handle additional info without adding newlines
            if hasattr(self, 'additional_info') and self.additional_info:
                # Join all additional info items with a space and ensure they're on the same line
                items = self.additional_info.split(", ")
                formatted_additional_info = " ".join(items)

                # Append the additional info to the label_text, ensuring it's all in one line
                label_text += f" {formatted_additional_info.strip()}"

            # Update the progress label
            self.progress_label.config(text=label_text)

class spacrSlider(tk.Frame):
    def __init__(self, master=None, length=None, thickness=2, knob_radius=10, position="center", from_=0, to=100, value=None, show_index=False, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.specified_length = length  # Store the specified length, if any
        self.knob_radius = knob_radius
        self.thickness = thickness
        self.knob_position = knob_radius  # Start at the beginning of the slider
        self.slider_line = None
        self.knob = None
        self.position = position.lower()  # Store the position option
        self.offset = 0  # Initialize offset
        self.from_ = from_  # Minimum value of the slider
        self.to = to  # Maximum value of the slider
        self.value = value if value is not None else from_  # Initial value of the slider
        self.show_index = show_index  # Whether to show the index Entry widget
        self.command = command  # Callback function to handle value changes

        # Initialize the style and colors
        style_out = set_dark_style(ttk.Style())
        self.fg_color = style_out['fg_color']
        self.bg_color = style_out['bg_color']
        self.active_color = style_out['active_color']
        self.inactive_color = style_out['inactive_color']

        # Configure the frame's background color
        self.configure(bg=self.bg_color)

        # Create a frame for the slider and entry if needed
        self.grid_columnconfigure(1, weight=1)

        # Entry widget for showing and editing index, if enabled
        if self.show_index:
            self.index_var = tk.StringVar(value=str(int(self.value)))
            self.index_entry = tk.Entry(self, textvariable=self.index_var, width=5, bg=self.bg_color, fg=self.fg_color, insertbackground=self.fg_color)
            self.index_entry.grid(row=0, column=0, padx=5)
            # Bind the entry to update the slider on change
            self.index_entry.bind("<Return>", self.update_slider_from_entry)

        # Create the slider canvas
        self.canvas = tk.Canvas(self, height=knob_radius * 2, bg=self.bg_color, highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="ew")

        # Set initial length to specified length or default value
        self.length = self.specified_length if self.specified_length is not None else self.canvas.winfo_reqwidth()

        # Calculate initial knob position based on the initial value
        self.knob_position = self.value_to_position(self.value)

        # Bind resize event to dynamically adjust the slider length if no length is specified
        self.canvas.bind("<Configure>", self.resize_slider)

        # Draw the slider components
        self.draw_slider(inactive=True)

        # Bind mouse events to the knob and slider
        self.canvas.bind("<B1-Motion>", self.move_knob)
        self.canvas.bind("<Button-1>", self.activate_knob)  # Activate knob on click
        self.canvas.bind("<ButtonRelease-1>", self.release_knob)  # Trigger command on release

    def resize_slider(self, event):
        if self.specified_length is not None:
            self.length = self.specified_length
        else:
            self.length = int(event.width * 0.9)  # 90% of the container width
        
        # Calculate the horizontal offset based on the position
        if self.position == "center":
            self.offset = (event.width - self.length) // 2
        elif self.position == "right":
            self.offset = event.width - self.length
        else:  # position is "left"
            self.offset = 0

        # Update the knob position after resizing
        self.knob_position = self.value_to_position(self.value)
        self.draw_slider(inactive=True)

    def value_to_position(self, value):
        if self.to == self.from_:
            return self.knob_radius
        relative_value = (value - self.from_) / (self.to - self.from_)
        return self.knob_radius + relative_value * (self.length - 2 * self.knob_radius)

    def position_to_value(self, position):
        if self.to == self.from_:
            return self.from_
        relative_position = (position - self.knob_radius) / (self.length - 2 * self.knob_radius)
        return self.from_ + relative_position * (self.to - self.from_)

    def draw_slider(self, inactive=False):
        self.canvas.delete("all")

        self.slider_line = self.canvas.create_line(
            self.offset + self.knob_radius, 
            self.knob_radius, 
            self.offset + self.length - self.knob_radius, 
            self.knob_radius, 
            fill=self.fg_color, 
            width=self.thickness
        )

        knob_color = self.inactive_color if inactive else self.active_color
        self.knob = self.canvas.create_oval(
            self.offset + self.knob_position - self.knob_radius, 
            self.knob_radius - self.knob_radius, 
            self.offset + self.knob_position + self.knob_radius, 
            self.knob_radius + self.knob_radius, 
            fill=knob_color, 
            outline=""
        )

    def move_knob(self, event):
        new_position = min(max(event.x - self.offset, self.knob_radius), self.length - self.knob_radius)
        self.knob_position = new_position
        self.value = self.position_to_value(self.knob_position)
        self.canvas.coords(
            self.knob, 
            self.offset + self.knob_position - self.knob_radius, 
            self.knob_radius - self.knob_radius, 
            self.offset + self.knob_position + self.knob_radius, 
            self.knob_radius + self.knob_radius
        )
        if self.show_index:
            self.index_var.set(str(int(self.value)))

    def activate_knob(self, event):
        self.draw_slider(inactive=False)
        self.move_knob(event)

    def release_knob(self, event):
        self.draw_slider(inactive=True)
        if self.command:
            self.command(self.value)  # Call the command with the final value when the knob is released

    def set_to(self, new_to):
        self.to = new_to
        self.knob_position = self.value_to_position(self.value)
        self.draw_slider(inactive=False)

    def get(self):
        return self.value

    def set(self, value):
        """Set the slider's value and update the knob position."""
        self.value = max(self.from_, min(value, self.to))  # Ensure the value is within bounds
        self.knob_position = self.value_to_position(self.value)
        self.draw_slider(inactive=False)
        if self.show_index:
            self.index_var.set(str(int(self.value)))

    def jump_to_click(self, event):
        self.activate_knob(event)

    def update_slider_from_entry(self, event):
        """Update the slider's value from the entry."""
        try:
            index = int(self.index_var.get())
            self.set(index)
            if self.command:
                self.command(self.value)
        except ValueError:
            pass

def spacrScrollbarStyle(style, inactive_color, active_color):
    # Check if custom elements already exist to avoid duplication
    if not style.element_names().count('custom.Vertical.Scrollbar.trough'):
        style.element_create('custom.Vertical.Scrollbar.trough', 'from', 'clam')
    if not style.element_names().count('custom.Vertical.Scrollbar.thumb'):
        style.element_create('custom.Vertical.Scrollbar.thumb', 'from', 'clam')

    style.layout('Custom.Vertical.TScrollbar',
                 [('Vertical.Scrollbar.trough', {'children': [('Vertical.Scrollbar.thumb', {'expand': '1', 'sticky': 'nswe'})], 'sticky': 'ns'})])

    style.configure('Custom.Vertical.TScrollbar',
                    background=inactive_color,
                    troughcolor=inactive_color,
                    bordercolor=inactive_color,
                    lightcolor=inactive_color,
                    darkcolor=inactive_color)

    style.map('Custom.Vertical.TScrollbar',
              background=[('!active', inactive_color), ('active', active_color)],
              troughcolor=[('!active', inactive_color), ('active', inactive_color)],
              bordercolor=[('!active', inactive_color), ('active', inactive_color)],
              lightcolor=[('!active', inactive_color), ('active', active_color)],
              darkcolor=[('!active', inactive_color), ('active', active_color)])

class spacrFrame(ttk.Frame):
    def __init__(self, container, width=None, *args, bg='black', radius=20, scrollbar=True, textbox=False, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.configure(style='TFrame')
        if width is None:
            screen_width = self.winfo_screenwidth()
            width = screen_width // 4

        # Create the canvas
        canvas = tk.Canvas(self, bg=bg, width=width, highlightthickness=0)
        self.rounded_rectangle(canvas, 0, 0, width, self.winfo_screenheight(), radius, fill=bg)

        # Define scrollbar styles
        style_out = set_dark_style(ttk.Style())
        self.inactive_color = style_out['inactive_color']
        self.active_color = style_out['active_color']
        self.fg_color = style_out['fg_color']  # Foreground color for text

        # Set custom scrollbar style
        style = ttk.Style()
        spacrScrollbarStyle(style, self.inactive_color, self.active_color)

        # Create scrollbar with custom style if scrollbar option is True
        if scrollbar:
            scrollbar_widget = ttk.Scrollbar(self, orient="vertical", command=canvas.yview, style='Custom.Vertical.TScrollbar')
        
        if textbox:
            self.scrollable_frame = tk.Text(canvas, bg=bg, fg=self.fg_color, wrap=tk.WORD)
        else:
            self.scrollable_frame = ttk.Frame(canvas, style='TFrame')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        if scrollbar:
            canvas.configure(yscrollcommand=scrollbar_widget.set)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        if scrollbar:
            scrollbar_widget.grid(row=0, column=1, sticky="ns")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        if scrollbar:
            self.grid_columnconfigure(1, weight=0)
        
        _ = set_dark_style(style, containers=[self], widgets=[canvas, self.scrollable_frame])
        if scrollbar:
            _ = set_dark_style(style, widgets=[scrollbar_widget])

    def rounded_rectangle(self, canvas, x1, y1, x2, y2, radius=20, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return canvas.create_polygon(points, **kwargs, smooth=True)

class spacrLabel(tk.Frame):
    def __init__(self, parent, text="", font=None, style=None, align="right", height=None, **kwargs):
        valid_kwargs = {k: v for k, v in kwargs.items() if k not in ['foreground', 'background', 'font', 'anchor', 'justify', 'wraplength']}
        super().__init__(parent, **valid_kwargs)
        
        self.text = text
        self.align = align

        if height is None:
            screen_height = self.winfo_screenheight()
            label_height = screen_height // 50
            label_width = label_height * 10
        else:
            label_height = height
            label_width = label_height * 10

        self.style_out = set_dark_style(ttk.Style())
        self.font_style = self.style_out['font_family']
        self.font_size = self.style_out['font_size']
        self.font_family = self.style_out['font_family']
        self.font_loader = self.style_out['font_loader']

        self.canvas = tk.Canvas(self, width=label_width, height=label_height, highlightthickness=0, bg=self.style_out['bg_color'])
        self.canvas.grid(row=0, column=0, sticky="ew")
        if self.style_out['font_family'] != 'OpenSans':
            self.font_style = font if font else tkFont.Font(family=self.style_out['font_family'], size=self.style_out['font_size'], weight=tkFont.NORMAL)
        self.style = style

        if self.align == "center":
            anchor_value = tk.CENTER
            text_anchor = 'center'
        else:  # default to right alignment
            anchor_value = tk.E
            text_anchor = 'e'

        if self.style:
            ttk_style = ttk.Style()
            if self.font_loader:
                ttk_style.configure(self.style, font=self.font_loader.get_font(size=self.font_size), background=self.style_out['bg_color'], foreground=self.style_out['fg_color'])
            else:
                ttk_style.configure(self.style, font=self.font_style, background=self.style_out['bg_color'], foreground=self.style_out['fg_color'])
            self.label_text = ttk.Label(self.canvas, text=self.text, style=self.style, anchor=text_anchor)
            self.label_text.pack(fill=tk.BOTH, expand=True)
        else:
            if self.font_loader:
                self.label_text = self.canvas.create_text(label_width // 2 if self.align == "center" else label_width - 5, 
                                                          label_height // 2, text=self.text, fill=self.style_out['fg_color'], 
                                                          font=self.font_loader.get_font(size=self.font_size), anchor=anchor_value, justify=tk.RIGHT)
            else:
                self.label_text = self.canvas.create_text(label_width // 2 if self.align == "center" else label_width - 5, 
                                                        label_height // 2, text=self.text, fill=self.style_out['fg_color'], 
                                                        font=self.font_style, anchor=anchor_value, justify=tk.RIGHT)
        
        _ = set_dark_style(ttk.Style(), containers=[self], widgets=[self.canvas])

    def set_text(self, text):
        if self.style:
            self.label_text.config(text=text)
        else:
            self.canvas.itemconfig(self.label_text, text=text)

class spacrButton(tk.Frame):
    def __init__(self, parent, text="", command=None, font=None, icon_name=None, size=50, show_text=True, outline=False, animation=True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.text = text.capitalize()  # Capitalize only the first letter of the text
        self.command = command
        self.icon_name = icon_name if icon_name else text.lower()
        self.size = size
        self.show_text = show_text
        self.outline = outline
        self.animation = animation  # Add animation attribute

        style_out = set_dark_style(ttk.Style())
        self.font_size = style_out['font_size']
        self.font_loader = style_out['font_loader']

        if self.show_text:
            self.button_width = int(size * 3)
        else:
            self.button_width = self.size  # Make the button width equal to the size if show_text is False

        # Create the canvas first
        self.canvas = tk.Canvas(self, width=self.button_width + 4, height=self.size + 4, highlightthickness=0, bg=style_out['bg_color'])
        self.canvas.grid(row=0, column=0)

        # Apply dark style and get color settings
        color_settings = set_dark_style(ttk.Style(), containers=[self], widgets=[self.canvas])

        self.inactive_color = color_settings['inactive_color']

        if self.outline:
            self.button_bg = self.create_rounded_rectangle(2, 2, self.button_width + 2, self.size + 2, radius=20, fill=self.inactive_color, outline=color_settings['fg_color'])
        else:
            self.button_bg = self.create_rounded_rectangle(2, 2, self.button_width + 2, self.size + 2, radius=20, fill=self.inactive_color, outline=self.inactive_color)
        
        self.load_icon()
        if self.font_loader:
            self.font_style = self.font_loader.get_font(size=self.font_size)
        else:
            self.font_style = font if font else ("Arial", 12)
        
        if self.show_text:
            self.button_text = self.canvas.create_text(self.size + 10, self.size // 2 + 2, text=self.text, fill=color_settings['fg_color'], font=self.font_style, anchor="w")  # Align text to the left of the specified point

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

        self.bg_color = self.inactive_color
        self.active_color = color_settings['active_color']
        self.fg_color = color_settings['fg_color']
        self.is_zoomed_in = False  # Track zoom state for smooth transitions

    def load_icon(self):
        icon_path = self.get_icon_path(self.icon_name)
        try:
            icon_image = Image.open(icon_path)
        except (FileNotFoundError, Image.UnidentifiedImageError):
            try:
                icon_path = icon_path.replace(' ', '_')
                icon_image = Image.open(icon_path)
            except (FileNotFoundError, Image.UnidentifiedImageError):
                icon_image = Image.open(self.get_icon_path("default"))
                print(f'Icon not found: {icon_path}. Using default icon instead.')

        initial_size = int(self.size * 0.65)  # 65% of button size initially
        self.original_icon_image = icon_image.resize((initial_size, initial_size), Image.Resampling.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(self.original_icon_image)

        self.button_icon = self.canvas.create_image(self.size // 2 + 2, self.size // 2 + 2, image=self.icon_photo)
        self.canvas.image = self.icon_photo  # Keep a reference to avoid garbage collection

    def get_icon_path(self, icon_name):
        icon_dir = os.path.join(os.path.dirname(__file__), 'resources', 'icons')
        return os.path.join(icon_dir, f"{icon_name}.png")

    def on_enter(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill=self.active_color)
        self.update_description(event)
        if self.animation and not self.is_zoomed_in:
            self.animate_zoom(0.85)  # Zoom in the icon to 85% of button size

    def on_leave(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill=self.inactive_color)
        self.clear_description(event)
        if self.animation and self.is_zoomed_in:
            self.animate_zoom(0.65)  # Reset the icon size to 65% of button size

    def on_click(self, event=None):
        if self.command:
            self.command()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)
    
    def update_description(self, event):
        parent = self.master
        while parent:
            if hasattr(parent, 'show_description'):
                parent.show_description(parent.main_buttons.get(self, parent.additional_buttons.get(self, "No description available.")))
                return
            parent = parent.master

    def clear_description(self, event):
        parent = self.master
        while parent:
            if hasattr(parent, 'clear_description'):
                parent.clear_description()
                return
            parent = parent.master

    def animate_zoom(self, target_scale, steps=10, delay=10):
        current_scale = 0.85 if self.is_zoomed_in else 0.65
        step_scale = (target_scale - current_scale) / steps
        self._animate_step(current_scale, step_scale, steps, delay)

    def _animate_step(self, current_scale, step_scale, steps, delay):
        if steps > 0:
            new_scale = current_scale + step_scale
            self.zoom_icon(new_scale)
            self.after(delay, self._animate_step, new_scale, step_scale, steps - 1, delay)
        else:
            self.is_zoomed_in = not self.is_zoomed_in

    def zoom_icon(self, scale_factor):
        # Resize the original icon image
        new_size = int(self.size * scale_factor)
        resized_icon = self.original_icon_image.resize((new_size, new_size), Image.Resampling.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(resized_icon)

        # Update the icon on the canvas
        self.canvas.itemconfig(self.button_icon, image=self.icon_photo)
        self.canvas.image = self.icon_photo

class spacrSwitch(ttk.Frame):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.canvas = tk.Canvas(self, width=40, height=20, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=1, padx=(10, 0))
        self.switch_bg = self.create_rounded_rectangle(2, 2, 38, 18, radius=9, outline="", fill="#fff")
        self.switch = self.canvas.create_oval(4, 4, 16, 16, outline="", fill="#800080")
        self.label = spacrLabel(self, text=self.text)
        self.label.grid(row=0, column=0, padx=(0, 10))
        self.bind("<Button-1>", self.toggle)
        self.canvas.bind("<Button-1>", self.toggle)
        self.label.bind("<Button-1>", self.toggle)
        self.update_switch()

        style = ttk.Style()
        _ = set_dark_style(style, containers=[self], widgets=[self.canvas, self.label])

    def toggle(self, event=None):
        self.variable.set(not self.variable.get())
        self.animate_switch()
        if self.command:
            self.command()

    def update_switch(self):
        if self.variable.get():
            self.canvas.itemconfig(self.switch, fill="#008080")
            self.canvas.coords(self.switch, 24, 4, 36, 16)
        else:
            self.canvas.itemconfig(self.switch, fill="#800080")
            self.canvas.coords(self.switch, 4, 4, 16, 16)

    def animate_switch(self):
        if self.variable.get():
            start_x, end_x = 4, 24
            final_color = "#008080"
        else:
            start_x, end_x = 24, 4
            final_color = "#800080"

        self.animate_movement(start_x, end_x, final_color)

    def animate_movement(self, start_x, end_x, final_color):
        step = 1 if start_x < end_x else -1
        for i in range(start_x, end_x, step):
            self.canvas.coords(self.switch, i, 4, i + 12, 16)
            self.canvas.update()
            self.after(10)
        self.canvas.itemconfig(self.switch, fill=final_color)

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)
        self.update_switch()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=9, **kwargs):
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

class spacrToolTip:
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
        label = tk.Label(self.tooltip_window, text=self.text, relief='flat', borderwidth=0)
        label.grid(row=0, column=0, padx=5, pady=5)

        style = ttk.Style()
        _ = set_dark_style(style, containers=[self.tooltip_window], widgets=[label])

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

class ModifyMaskApp:
    def __init__(self, root, folder_path, scale_factor):
        self.root = root
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.image_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.masks_folder = os.path.join(folder_path, 'masks')
        self.current_image_index = 0
        self.initialize_flags()
        self.canvas_width = self.root.winfo_screenheight() -100
        self.canvas_height = self.root.winfo_screenheight() -100
        self.root.configure(bg='black')
        self.setup_navigation_toolbar()
        self.setup_mode_toolbar()
        self.setup_function_toolbar()
        self.setup_zoom_toolbar()
        self.setup_canvas()
        self.load_first_image()
    
    ####################################################################################################
    # Helper functions#
    ####################################################################################################
    
    def update_display(self):
        if self.zoom_active:
            self.display_zoomed_image()
        else:
            self.display_image()
    
    def update_original_mask_from_zoom(self):
        y0, y1, x0, x1 = self.zoom_y0, self.zoom_y1, self.zoom_x0, self.zoom_x1
        zoomed_mask_resized = resize(self.zoom_mask, (y1 - y0, x1 - x0), order=0, preserve_range=True).astype(np.uint8)
        self.mask[y0:y1, x0:x1] = zoomed_mask_resized
        
    def update_original_mask(self, zoomed_mask, x0, x1, y0, y1):
        actual_mask_region = self.mask[y0:y1, x0:x1]
        target_shape = actual_mask_region.shape
        resized_mask = resize(zoomed_mask, target_shape, order=0, preserve_range=True).astype(np.uint8)
        if resized_mask.shape != actual_mask_region.shape:
            raise ValueError(f"Shape mismatch: resized_mask {resized_mask.shape}, actual_mask_region {actual_mask_region.shape}")
        self.mask[y0:y1, x0:x1] = np.maximum(actual_mask_region, resized_mask)
        self.mask = self.mask.copy()
        self.mask[y0:y1, x0:x1] = np.maximum(self.mask[y0:y1, x0:x1], resized_mask)
        self.mask = self.mask.copy()

    def get_scaling_factors(self, img_width, img_height, canvas_width, canvas_height):
        x_scale = img_width / canvas_width
        y_scale = img_height / canvas_height
        return x_scale, y_scale
    
    def canvas_to_image(self, x_canvas, y_canvas):
        x_scale, y_scale = self.get_scaling_factors(
            self.image.shape[1], self.image.shape[0],
            self.canvas_width, self.canvas_height
        )
        x_image = int(x_canvas * x_scale)
        y_image = int(y_canvas * y_scale)
        return x_image, y_image

    def apply_zoom_on_enter(self, event):
        if self.zoom_active and self.zoom_rectangle_start is not None:
            self.set_zoom_rectangle_end(event)
        
    def normalize_image(self, image, lower_quantile, upper_quantile):
        lower_bound = np.percentile(image, lower_quantile)
        upper_bound = np.percentile(image, upper_quantile)
        normalized = np.clip(image, lower_bound, upper_bound)
        normalized = (normalized - lower_bound) / (upper_bound - lower_bound)
        max_value = np.iinfo(image.dtype).max
        normalized = (normalized * max_value).astype(image.dtype)
        return normalized
    
    def resize_arrays(self, img, mask):
        original_dtype = img.dtype
        scaled_height = int(img.shape[0] * self.scale_factor)
        scaled_width = int(img.shape[1] * self.scale_factor)
        scaled_img = resize(img, (scaled_height, scaled_width), anti_aliasing=True, preserve_range=True)
        scaled_mask = resize(mask, (scaled_height, scaled_width), order=0, anti_aliasing=False, preserve_range=True)
        stretched_img = resize(scaled_img, (self.canvas_height, self.canvas_width), anti_aliasing=True, preserve_range=True)
        stretched_mask = resize(scaled_mask, (self.canvas_height, self.canvas_width), order=0, anti_aliasing=False, preserve_range=True)
        return stretched_img.astype(original_dtype), stretched_mask.astype(original_dtype)
    
    ####################################################################################################
    #Initiate canvas elements#
    ####################################################################################################
    
    def load_first_image(self):
        self.image, self.mask = self.load_image_and_mask(self.current_image_index)
        self.original_size = self.image.shape
        self.image, self.mask = self.resize_arrays(self.image, self.mask)
        self.display_image()

    def setup_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()
        self.canvas.bind("<Motion>", self.update_mouse_info)

    def initialize_flags(self):
        self.zoom_rectangle_start = None
        self.zoom_rectangle_end = None
        self.zoom_rectangle_id = None
        self.zoom_x0 = None
        self.zoom_y0 = None
        self.zoom_x1 = None
        self.zoom_y1 = None
        self.zoom_mask = None
        self.zoom_image = None
        self.zoom_image_orig = None
        self.zoom_scale = 1
        self.drawing = False
        self.zoom_active = False
        self.magic_wand_active = False
        self.brush_active = False
        self.dividing_line_active = False
        self.dividing_line_coords = []
        self.current_dividing_line = None
        self.lower_quantile = tk.StringVar(value="1.0")
        self.upper_quantile = tk.StringVar(value="99.9")
        self.magic_wand_tolerance = tk.StringVar(value="1000")

    def update_mouse_info(self, event):
        x, y = event.x, event.y
        intensity = "N/A"
        mask_value = "N/A"
        pixel_count = "N/A"  
        if self.zoom_active:
            if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                intensity = self.zoom_image_orig[y, x] if self.zoom_image_orig is not None else "N/A"
                mask_value = self.zoom_mask[y, x] if self.zoom_mask is not None else "N/A"
        else:
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                intensity = self.image[y, x]
                mask_value = self.mask[y, x]
        if mask_value != "N/A" and mask_value != 0:
            pixel_count = np.sum(self.mask == mask_value)
        self.intensity_label.config(text=f"Intensity: {intensity}")
        self.mask_value_label.config(text=f"Mask: {mask_value}, Area: {pixel_count}")
        self.mask_value_label.config(text=f"Mask: {mask_value}")
        if mask_value != "N/A" and mask_value != 0:
            self.pixel_count_label.config(text=f"Area: {pixel_count}")
        else:
            self.pixel_count_label.config(text="Area: N/A")
    
    def setup_navigation_toolbar(self):
        navigation_toolbar = tk.Frame(self.root, bg='black')
        navigation_toolbar.pack(side='top', fill='x')
        prev_btn = tk.Button(navigation_toolbar, text="Previous", command=self.previous_image, bg='black', fg='white')
        prev_btn.pack(side='left')
        next_btn = tk.Button(navigation_toolbar, text="Next", command=self.next_image, bg='black', fg='white')
        next_btn.pack(side='left')
        save_btn = tk.Button(navigation_toolbar, text="Save", command=self.save_mask, bg='black', fg='white')
        save_btn.pack(side='left')
        self.intensity_label = tk.Label(navigation_toolbar, text="Image: N/A", bg='black', fg='white')
        self.intensity_label.pack(side='right')
        self.mask_value_label = tk.Label(navigation_toolbar, text="Mask: N/A", bg='black', fg='white')
        self.mask_value_label.pack(side='right')
        self.pixel_count_label = tk.Label(navigation_toolbar, text="Area: N/A", bg='black', fg='white')
        self.pixel_count_label.pack(side='right')

    def setup_mode_toolbar(self):
        self.mode_toolbar = tk.Frame(self.root, bg='black')
        self.mode_toolbar.pack(side='top', fill='x')
        self.draw_btn = tk.Button(self.mode_toolbar, text="Draw", command=self.toggle_draw_mode, bg='black', fg='white')
        self.draw_btn.pack(side='left')
        self.magic_wand_btn = tk.Button(self.mode_toolbar, text="Magic Wand", command=self.toggle_magic_wand_mode, bg='black', fg='white')
        self.magic_wand_btn.pack(side='left')
        tk.Label(self.mode_toolbar, text="Tolerance:", bg='black', fg='white').pack(side='left')
        self.tolerance_entry = tk.Entry(self.mode_toolbar, textvariable=self.magic_wand_tolerance, bg='black', fg='white')
        self.tolerance_entry.pack(side='left')
        tk.Label(self.mode_toolbar, text="Max Pixels:", bg='black', fg='white').pack(side='left')
        self.max_pixels_entry = tk.Entry(self.mode_toolbar, bg='black', fg='white')
        self.max_pixels_entry.insert(0, "1000")
        self.max_pixels_entry.pack(side='left')
        self.erase_btn = tk.Button(self.mode_toolbar, text="Erase", command=self.toggle_erase_mode, bg='black', fg='white')
        self.erase_btn.pack(side='left')
        self.brush_btn = tk.Button(self.mode_toolbar, text="Brush", command=self.toggle_brush_mode, bg='black', fg='white')
        self.brush_btn.pack(side='left')
        self.brush_size_entry = tk.Entry(self.mode_toolbar, bg='black', fg='white')
        self.brush_size_entry.insert(0, "10")
        self.brush_size_entry.pack(side='left')
        tk.Label(self.mode_toolbar, text="Brush Size:", bg='black', fg='white').pack(side='left')
        self.dividing_line_btn = tk.Button(self.mode_toolbar, text="Dividing Line", command=self.toggle_dividing_line_mode, bg='black', fg='white')
        self.dividing_line_btn.pack(side='left')

    def setup_function_toolbar(self):
        self.function_toolbar = tk.Frame(self.root, bg='black')
        self.function_toolbar.pack(side='top', fill='x')
        self.fill_btn = tk.Button(self.function_toolbar, text="Fill", command=self.fill_objects, bg='black', fg='white')
        self.fill_btn.pack(side='left')
        self.relabel_btn = tk.Button(self.function_toolbar, text="Relabel", command=self.relabel_objects, bg='black', fg='white')
        self.relabel_btn.pack(side='left')
        self.clear_btn = tk.Button(self.function_toolbar, text="Clear", command=self.clear_objects, bg='black', fg='white')
        self.clear_btn.pack(side='left')
        self.invert_btn = tk.Button(self.function_toolbar, text="Invert", command=self.invert_mask, bg='black', fg='white')
        self.invert_btn.pack(side='left')
        remove_small_btn = tk.Button(self.function_toolbar, text="Remove Small", command=self.remove_small_objects, bg='black', fg='white')
        remove_small_btn.pack(side='left')
        tk.Label(self.function_toolbar, text="Min Area:", bg='black', fg='white').pack(side='left')
        self.min_area_entry = tk.Entry(self.function_toolbar, bg='black', fg='white')
        self.min_area_entry.insert(0, "100")  # Default minimum area
        self.min_area_entry.pack(side='left')

    def setup_zoom_toolbar(self):
        self.zoom_toolbar = tk.Frame(self.root, bg='black')
        self.zoom_toolbar.pack(side='top', fill='x')
        self.zoom_btn = tk.Button(self.zoom_toolbar, text="Zoom", command=self.toggle_zoom_mode, bg='black', fg='white')
        self.zoom_btn.pack(side='left')
        self.normalize_btn = tk.Button(self.zoom_toolbar, text="Apply Normalization", command=self.apply_normalization, bg='black', fg='white')
        self.normalize_btn.pack(side='left')
        tk.Label(self.zoom_toolbar, text="Lower Percentile:", bg='black', fg='white').pack(side='left')
        self.lower_entry = tk.Entry(self.zoom_toolbar, textvariable=self.lower_quantile, bg='black', fg='white')
        self.lower_entry.pack(side='left')
        
        tk.Label(self.zoom_toolbar, text="Upper Percentile:", bg='black', fg='white').pack(side='left')
        self.upper_entry = tk.Entry(self.zoom_toolbar, textvariable=self.upper_quantile, bg='black', fg='white')
        self.upper_entry.pack(side='left')
    
    def load_image_and_mask(self, index):
        # Load the image
        image_path = os.path.join(self.folder_path, self.image_filenames[index])
        image = imageio.imread(image_path)
        print(f"Original Image shape: {image.shape}, dtype: {image.dtype}")

        # Handle multi-channel or transparency issues
        if image.ndim == 3:
            if image.shape[2] == 4:  # If the image has an alpha channel (RGBA)
                image = image[..., :3]  # Remove the alpha channel

            # Convert RGB to grayscale using weighted average
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            print(f"Converted to grayscale: {image.shape}")

        # Ensure the shape is (height, width) without extra channel
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=-1)

        if image.dtype != np.uint16:
            # Scale the image to fit the 16-bit range (0–65535)
            image = (image / image.max() * 65535).astype(np.uint16)
            # eventually remove this images should not have to be 16 bit look into downstream function (non 16bit images are jsut black)

        # Load the corresponding mask
        mask_path = os.path.join(self.masks_folder, self.image_filenames[index])
        if os.path.exists(mask_path):
            print(f'Loading mask: {mask_path} for image: {image_path}')
            mask = imageio.imread(mask_path)

            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = (mask / mask.max() * 255).astype(np.uint8)
        else:
            # Create a new mask with the same size as the image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            print(f'Loaded new mask for image: {image_path}')

        return image, mask
    
    ####################################################################################################
    # Image Display functions#
    ####################################################################################################
    def display_image(self):
        if self.zoom_rectangle_id is not None:
            self.canvas.delete(self.zoom_rectangle_id)
            self.zoom_rectangle_id = None
        lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() else 1.0
        upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() else 99.9
        normalized = self.normalize_image(self.image, lower_quantile, upper_quantile)
        combined = self.overlay_mask_on_image(normalized, self.mask)
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def display_zoomed_image(self):
        if self.zoom_rectangle_start and self.zoom_rectangle_end:
            # Convert canvas coordinates to image coordinates
            x0, y0 = self.canvas_to_image(*self.zoom_rectangle_start)
            x1, y1 = self.canvas_to_image(*self.zoom_rectangle_end)
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            self.zoom_x0 = x0
            self.zoom_y0 = y0
            self.zoom_x1 = x1
            self.zoom_y1 = y1
            # Normalize the entire image
            lower_quantile = float(self.lower_quantile.get()) if self.lower_quantile.get() else 1.0
            upper_quantile = float(self.upper_quantile.get()) if self.upper_quantile.get() else 99.9
            normalized_image = self.normalize_image(self.image, lower_quantile, upper_quantile)
            # Extract the zoomed portion of the normalized image and mask
            self.zoom_image = normalized_image[y0:y1, x0:x1]
            self.zoom_image_orig = self.image[y0:y1, x0:x1]
            self.zoom_mask = self.mask[y0:y1, x0:x1]
            original_mask_area = self.mask.shape[0] * self.mask.shape[1]
            zoom_mask_area = self.zoom_mask.shape[0] * self.zoom_mask.shape[1]
            if original_mask_area > 0:
                self.zoom_scale = original_mask_area/zoom_mask_area
            # Resize the zoomed image and mask to fit the canvas
            canvas_height = self.canvas.winfo_height()
            canvas_width = self.canvas.winfo_width()
            
            if self.zoom_image.size > 0 and canvas_height > 0 and canvas_width > 0:
                self.zoom_image = resize(self.zoom_image, (canvas_height, canvas_width), preserve_range=True).astype(self.zoom_image.dtype)
                self.zoom_image_orig = resize(self.zoom_image_orig, (canvas_height, canvas_width), preserve_range=True).astype(self.zoom_image_orig.dtype)
                #self.zoom_mask = resize(self.zoom_mask, (canvas_height, canvas_width), preserve_range=True).astype(np.uint8)
                self.zoom_mask = resize(self.zoom_mask, (canvas_height, canvas_width), order=0, preserve_range=True).astype(np.uint8)
                combined = self.overlay_mask_on_image(self.zoom_image, self.zoom_mask)
                self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(combined))
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def overlay_mask_on_image(self, image, mask, alpha=0.5):
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = mask.astype(np.int32)
        max_label = np.max(mask)
        np.random.seed(0)
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # background color
        colored_mask = colors[mask]
        image_8bit = (image / 256).astype(np.uint8)
        # Blend the mask and the image with transparency
        combined_image = np.where(mask[..., None] > 0, 
                                  np.clip(image_8bit * (1 - alpha) + colored_mask * alpha, 0, 255), 
                                  image_8bit)
        # Convert the final image back to uint8
        combined_image = combined_image.astype(np.uint8)
        return combined_image
    
    ####################################################################################################
    # Navigation functions#
    ####################################################################################################
    
    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.initialize_flags()            
            self.image, self.mask = self.load_image_and_mask(self.current_image_index)
            self.original_size = self.image.shape
            self.image, self.mask = self.resize_arrays(self.image, self.mask)
            self.display_image()

    def next_image(self):
        if self.current_image_index < len(self.image_filenames) - 1:
            self.current_image_index += 1
            self.initialize_flags()            
            self.image, self.mask = self.load_image_and_mask(self.current_image_index)
            self.original_size = self.image.shape
            self.image, self.mask = self.resize_arrays(self.image, self.mask)
            self.display_image()
            
    def save_mask(self):
        if self.current_image_index < len(self.image_filenames):
            original_size = self.original_size
            if self.mask.shape != original_size:
                resized_mask = resize(self.mask, original_size, order=0, preserve_range=True).astype(np.uint16)
            else:
                resized_mask = self.mask
            resized_mask, _ = label(resized_mask > 0)
            save_folder = os.path.join(self.folder_path, 'masks')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            image_filename = os.path.splitext(self.image_filenames[self.current_image_index])[0] + '.tif'
            save_path = os.path.join(save_folder, image_filename)

            print(f"Saving mask to: {save_path}")  # Debug print
            imageio.imwrite(save_path, resized_mask)

    ####################################################################################################
    # Zoom Functions #
    ####################################################################################################
    def set_zoom_rectangle_start(self, event):
        if self.zoom_active:
            self.zoom_rectangle_start = (event.x, event.y)
    
    def set_zoom_rectangle_end(self, event):
        if self.zoom_active:
            self.zoom_rectangle_end = (event.x, event.y)
            if self.zoom_rectangle_id is not None:
                self.canvas.delete(self.zoom_rectangle_id)
                self.zoom_rectangle_id = None
            self.display_zoomed_image()  
            self.canvas.unbind("<Motion>")  
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")  
            self.canvas.bind("<Motion>", self.update_mouse_info)
            
    def update_zoom_box(self, event):
        if self.zoom_active and self.zoom_rectangle_start is not None:
            if self.zoom_rectangle_id is not None:
                self.canvas.delete(self.zoom_rectangle_id)
            # Assuming event.x and event.y are already in image coordinates
            self.zoom_rectangle_end = (event.x, event.y)
            x0, y0 = self.zoom_rectangle_start
            x1, y1 = self.zoom_rectangle_end
            self.zoom_rectangle_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)
    
    ####################################################################################################
    # Mode activation#
    ####################################################################################################
    
    def toggle_zoom_mode(self):
        if not self.zoom_active:
            self.brush_btn.config(text="Brush")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-3>")
            self.zoom_active = True
            self.drawing = False
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_active = False
            self.dividing_line_active = False
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.magic_wand_btn.config(text="Magic Wand")
            self.zoom_btn.config(text="Zoom ON")
            self.dividing_line_btn.config(text="Dividing Line")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Button-1>", self.set_zoom_rectangle_start)
            self.canvas.bind("<Button-3>", self.set_zoom_rectangle_end)
            self.canvas.bind("<Motion>", self.update_zoom_box)
        else:
            self.zoom_active = False
            self.zoom_btn.config(text="Zoom")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Motion>")
            self.zoom_rectangle_start = self.zoom_rectangle_end = None
            self.zoom_rectangle_id = None
            self.display_image()
            self.canvas.bind("<Motion>", self.update_mouse_info)
            self.zoom_rectangle_start = None
            self.zoom_rectangle_end = None
            self.zoom_rectangle_id = None
            self.zoom_x0 = None
            self.zoom_y0 = None
            self.zoom_x1 = None
            self.zoom_y1 = None
            self.zoom_mask = None
            self.zoom_image = None
            self.zoom_image_orig = None

    def toggle_brush_mode(self):
        self.brush_active = not self.brush_active
        if self.brush_active:
            self.drawing = False
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_btn.config(text="Brush ON")
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.magic_wand_btn.config(text="Magic Wand")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<B1-Motion>", self.apply_brush)  # Left click and drag to apply brush
            self.canvas.bind("<B3-Motion>", self.erase_brush)  # Right click and drag to erase with brush
            self.canvas.bind("<ButtonRelease-1>", self.apply_brush_release)  # Left button release
            self.canvas.bind("<ButtonRelease-3>", self.erase_brush_release)  # Right button release
        else:
            self.brush_active = False
            self.brush_btn.config(text="Brush")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-3>")

    def image_to_canvas(self, x_image, y_image):
        x_scale, y_scale = self.get_scaling_factors(
            self.image.shape[1], self.image.shape[0],
            self.canvas_width, self.canvas_height
        )
        x_canvas = int(x_image / x_scale)
        y_canvas = int(y_image / y_scale)
        return x_canvas, y_canvas

    def toggle_dividing_line_mode(self):
        self.dividing_line_active = not self.dividing_line_active
        if self.dividing_line_active:
            self.drawing = False
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.magic_wand_btn.config(text="Magic Wand")
            self.brush_btn.config(text="Brush")
            self.dividing_line_btn.config(text="Dividing Line ON")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Button-1>", self.start_dividing_line)
            self.canvas.bind("<ButtonRelease-1>", self.finish_dividing_line)
            self.canvas.bind("<Motion>", self.update_dividing_line_preview)
        else:
            print("Dividing Line Mode: OFF")
            self.dividing_line_active = False
            self.dividing_line_btn.config(text="Dividing Line")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<Motion>")
            self.display_image()

    def start_dividing_line(self, event):
        if self.dividing_line_active:
            self.dividing_line_coords = [(event.x, event.y)]
            self.current_dividing_line = self.canvas.create_line(event.x, event.y, event.x, event.y, fill="red", width=2)

    def finish_dividing_line(self, event):
        if self.dividing_line_active:
            self.dividing_line_coords.append((event.x, event.y))
            if self.zoom_active:
                self.dividing_line_coords = [self.canvas_to_image(x, y) for x, y in self.dividing_line_coords]
            self.apply_dividing_line()
            self.canvas.delete(self.current_dividing_line)
            self.current_dividing_line = None

    def update_dividing_line_preview(self, event):
        if self.dividing_line_active and self.dividing_line_coords:
            x, y = event.x, event.y
            if self.zoom_active:
                x, y = self.canvas_to_image(x, y)
            self.dividing_line_coords.append((x, y))
            canvas_coords = [(self.image_to_canvas(*pt) if self.zoom_active else pt) for pt in self.dividing_line_coords]
            flat_canvas_coords = [coord for pt in canvas_coords for coord in pt]
            self.canvas.coords(self.current_dividing_line, *flat_canvas_coords)

    def apply_dividing_line(self):
        if self.dividing_line_coords:
            coords = self.dividing_line_coords
            if self.zoom_active:
                coords = [self.canvas_to_image(x, y) for x, y in coords]

            rr, cc = [], []
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                line_rr, line_cc = line(y0, x0, y1, x1)
                rr.extend(line_rr)
                cc.extend(line_cc)
            rr, cc = np.array(rr), np.array(cc)

            mask_copy = self.mask.copy()

            if self.zoom_active:
                # Update the zoomed mask
                self.zoom_mask[rr, cc] = 0
                # Reflect changes to the original mask
                y0, y1, x0, x1 = self.zoom_y0, self.zoom_y1, self.zoom_x0, self.zoom_x1
                zoomed_mask_resized_back = resize(self.zoom_mask, (y1 - y0, x1 - x0), order=0, preserve_range=True).astype(np.uint8)
                self.mask[y0:y1, x0:x1] = zoomed_mask_resized_back
            else:
                # Directly update the original mask
                mask_copy[rr, cc] = 0
                self.mask = mask_copy

            labeled_mask, num_labels = label(self.mask > 0)
            self.mask = labeled_mask
            self.update_display()

            self.dividing_line_coords = []
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<Motion>")
            self.dividing_line_active = False
            self.dividing_line_btn.config(text="Dividing Line")

    def toggle_draw_mode(self):
        self.drawing = not self.drawing
        if self.drawing:
            self.brush_btn.config(text="Brush")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-3>")
            self.magic_wand_active = False
            self.erase_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw ON")
            self.magic_wand_btn.config(text="Magic Wand")
            self.erase_btn.config(text="Erase")
            self.draw_coordinates = []
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.finish_drawing)
        else:
            self.drawing = False
            self.draw_btn.config(text="Draw")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            
    def toggle_magic_wand_mode(self):
        self.magic_wand_active = not self.magic_wand_active
        if self.magic_wand_active:
            self.brush_btn.config(text="Brush")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-3>")
            self.drawing = False
            self.erase_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw")
            self.erase_btn.config(text="Erase")
            self.magic_wand_btn.config(text="Magic Wand ON")
            self.canvas.bind("<Button-1>", self.use_magic_wand)
            self.canvas.bind("<Button-3>", self.use_magic_wand)
        else:
            self.magic_wand_btn.config(text="Magic Wand")
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            
    def toggle_erase_mode(self):
        self.erase_active = not self.erase_active
        if self.erase_active:
            self.brush_btn.config(text="Brush")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<ButtonRelease-3>")
            self.erase_btn.config(text="Erase ON")
            self.canvas.bind("<Button-1>", self.erase_object)
            self.drawing = False
            self.magic_wand_active = False
            self.brush_active = False
            self.draw_btn.config(text="Draw")
            self.magic_wand_btn.config(text="Magic Wand")
        else:
            self.erase_active = False
            self.erase_btn.config(text="Erase")
            self.canvas.unbind("<Button-1>")
    
    ####################################################################################################
    # Mode functions#
    ####################################################################################################
    
    def apply_brush_release(self, event):
        if hasattr(self, 'brush_path'):
            for x, y, brush_size in self.brush_path:
                img_x, img_y = (x, y) if self.zoom_active else self.canvas_to_image(x, y)
                x0 = max(img_x - brush_size // 2, 0)
                y0 = max(img_y - brush_size // 2, 0)
                x1 = min(img_x + brush_size // 2, self.zoom_mask.shape[1] if self.zoom_active else self.mask.shape[1])
                y1 = min(img_y + brush_size // 2, self.zoom_mask.shape[0] if self.zoom_active else self.mask.shape[0])
                if self.zoom_active:
                    self.zoom_mask[y0:y1, x0:x1] = 255
                    self.update_original_mask_from_zoom()
                else:
                    self.mask[y0:y1, x0:x1] = 255
            del self.brush_path
            self.canvas.delete("temp_line")
            self.update_display()

    def erase_brush_release(self, event):
        if hasattr(self, 'erase_path'):
            for x, y, brush_size in self.erase_path:
                img_x, img_y = (x, y) if self.zoom_active else self.canvas_to_image(x, y)
                x0 = max(img_x - brush_size // 2, 0)
                y0 = max(img_y - brush_size // 2, 0)
                x1 = min(img_x + brush_size // 2, self.zoom_mask.shape[1] if self.zoom_active else self.mask.shape[1])
                y1 = min(img_y + brush_size // 2, self.zoom_mask.shape[0] if self.zoom_active else self.mask.shape[0])
                if self.zoom_active:
                    self.zoom_mask[y0:y1, x0:x1] = 0                    
                    self.update_original_mask_from_zoom()
                else:
                    self.mask[y0:y1, x0:x1] = 0
            del self.erase_path
            self.canvas.delete("temp_line")
            self.update_display()
        
    def apply_brush(self, event):
        brush_size = int(self.brush_size_entry.get())
        x, y = event.x, event.y
        if not hasattr(self, 'brush_path'):
            self.brush_path = []
            self.last_brush_coord = (x, y)
        if self.last_brush_coord:
            last_x, last_y = self.last_brush_coord
            rr, cc = line(last_y, last_x, y, x)
            for ry, rx in zip(rr, cc):
                self.brush_path.append((rx, ry, brush_size))

        self.canvas.create_line(self.last_brush_coord[0], self.last_brush_coord[1], x, y, width=brush_size, fill="blue", tag="temp_line")
        self.last_brush_coord = (x, y)

    def erase_brush(self, event):
        brush_size = int(self.brush_size_entry.get())
        x, y = event.x, event.y
        if not hasattr(self, 'erase_path'):
            self.erase_path = []
            self.last_erase_coord = (x, y)
        if self.last_erase_coord:
            last_x, last_y = self.last_erase_coord
            rr, cc = line(last_y, last_x, y, x)
            for ry, rx in zip(rr, cc):
                self.erase_path.append((rx, ry, brush_size))

        self.canvas.create_line(self.last_erase_coord[0], self.last_erase_coord[1], x, y, width=brush_size, fill="white", tag="temp_line")
        self.last_erase_coord = (x, y)

    def erase_object(self, event):
        x, y = event.x, event.y
        if self.zoom_active:
            canvas_x, canvas_y = x, y
            zoomed_x = int(canvas_x * (self.zoom_image.shape[1] / self.canvas_width))
            zoomed_y = int(canvas_y * (self.zoom_image.shape[0] / self.canvas_height))
            orig_x = int(zoomed_x * ((self.zoom_x1 - self.zoom_x0) / self.canvas_width) + self.zoom_x0)
            orig_y = int(zoomed_y * ((self.zoom_y1 - self.zoom_y0) / self.canvas_height) + self.zoom_y0)
            if orig_x < 0 or orig_y < 0 or orig_x >= self.image.shape[1] or orig_y >= self.image.shape[0]:
                print("Point is out of bounds in the original image.")
                return
        else:
            orig_x, orig_y = x, y
        label_to_remove = self.mask[orig_y, orig_x]
        if label_to_remove > 0:
            self.mask[self.mask == label_to_remove] = 0
        self.update_display()
            
    def use_magic_wand(self, event):
        x, y = event.x, event.y
        tolerance = int(self.magic_wand_tolerance.get())
        maximum = int(self.max_pixels_entry.get())
        action = 'add' if event.num == 1 else 'erase'
        if self.zoom_active:
            self.magic_wand_zoomed((x, y), tolerance, action)
        else:
            self.magic_wand_normal((x, y), tolerance, action)
    
    def apply_magic_wand(self, image, mask, seed_point, tolerance, maximum, action='add'):
        x, y = seed_point
        initial_value = image[y, x].astype(np.float32)
        visited = np.zeros_like(image, dtype=bool)
        queue = deque([(x, y)])
        added_pixels = 0

        while queue and added_pixels < maximum:
            cx, cy = queue.popleft()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            current_value = image[cy, cx].astype(np.float32)

            if np.linalg.norm(abs(current_value - initial_value)) <= tolerance:
                if mask[cy, cx] == 0:
                    added_pixels += 1
                mask[cy, cx] = 255 if action == 'add' else 0

                if added_pixels >= maximum:
                    break

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                        queue.append((nx, ny))
        return mask

    def magic_wand_normal(self, seed_point, tolerance, action):
        try:
            maximum = int(self.max_pixels_entry.get())
        except ValueError:
            print("Invalid maximum value; using default of 1000")
            maximum = 1000 
        self.mask = self.apply_magic_wand(self.image, self.mask, seed_point, tolerance, maximum, action)
        self.display_image()
        
    def magic_wand_zoomed(self, seed_point, tolerance, action):
        if self.zoom_image_orig is None or self.zoom_mask is None:
            print("Zoomed image or mask not initialized")
            return
        try:
            maximum = int(self.max_pixels_entry.get())
            maximum = maximum * self.zoom_scale
        except ValueError:
            print("Invalid maximum value; using default of 1000")
            maximum = 1000
            
        canvas_x, canvas_y = seed_point
        if canvas_x < 0 or canvas_y < 0 or canvas_x >= self.zoom_image_orig.shape[1] or canvas_y >= self.zoom_image_orig.shape[0]:
            print("Selected point is out of bounds in the zoomed image.")
            return
        
        self.zoom_mask = self.apply_magic_wand(self.zoom_image_orig, self.zoom_mask, (canvas_x, canvas_y), tolerance, maximum, action)
        y0, y1, x0, x1 = self.zoom_y0, self.zoom_y1, self.zoom_x0, self.zoom_x1
        zoomed_mask_resized_back = resize(self.zoom_mask, (y1 - y0, x1 - x0), order=0, preserve_range=True).astype(np.uint8)
        if action == 'erase':
            self.mask[y0:y1, x0:x1] = np.where(zoomed_mask_resized_back == 0, 0, self.mask[y0:y1, x0:x1])
        else:
            self.mask[y0:y1, x0:x1] = np.where(zoomed_mask_resized_back > 0, zoomed_mask_resized_back, self.mask[y0:y1, x0:x1])
        self.update_display()
                
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            if self.draw_coordinates:
                last_x, last_y = self.draw_coordinates[-1]
                self.current_line = self.canvas.create_line(last_x, last_y, x, y, fill="yellow", width=3)
            self.draw_coordinates.append((x, y))
            
    def draw_on_zoomed_mask(self, draw_coordinates):
        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        zoomed_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        rr, cc = polygon(np.array(draw_coordinates)[:, 1], np.array(draw_coordinates)[:, 0], shape=zoomed_mask.shape)
        zoomed_mask[rr, cc] = 255
        return zoomed_mask
            
    def finish_drawing(self, event):
        if len(self.draw_coordinates) > 2:
            self.draw_coordinates.append(self.draw_coordinates[0])
            if self.zoom_active:
                x0, x1, y0, y1 = self.zoom_x0, self.zoom_x1, self.zoom_y0, self.zoom_y1
                zoomed_mask = self.draw_on_zoomed_mask(self.draw_coordinates)
                self.update_original_mask(zoomed_mask, x0, x1, y0, y1)
            else:
                rr, cc = polygon(np.array(self.draw_coordinates)[:, 1], np.array(self.draw_coordinates)[:, 0], shape=self.mask.shape)
                self.mask[rr, cc] = np.maximum(self.mask[rr, cc], 255)
                self.mask = self.mask.copy()
            self.canvas.delete(self.current_line)
            self.draw_coordinates.clear()
        self.update_display()
            
    def finish_drawing_if_active(self, event):
        if self.drawing and len(self.draw_coordinates) > 2:
            self.finish_drawing(event)

    ####################################################################################################
    # Single function butons#
    ####################################################################################################
            
    def apply_normalization(self):
        self.lower_quantile.set(self.lower_entry.get())
        self.upper_quantile.set(self.upper_entry.get())
        self.update_display()

    def fill_objects(self):
        binary_mask = self.mask > 0
        filled_mask = binary_fill_holes(binary_mask)
        self.mask = filled_mask.astype(np.uint8) * 255
        labeled_mask, _ = label(filled_mask)
        self.mask = labeled_mask
        self.update_display()

    def relabel_objects(self):
        mask = self.mask
        labeled_mask, num_labels = label(mask > 0)
        self.mask = labeled_mask
        self.update_display()
        
    def clear_objects(self):
        self.mask = np.zeros_like(self.mask)
        self.update_display()

    def invert_mask(self):
        self.mask = np.where(self.mask > 0, 0, 1)
        self.relabel_objects()
        self.update_display()
        
    def remove_small_objects(self):
        try:
            min_area = int(self.min_area_entry.get())
        except ValueError:
            print("Invalid minimum area value; using default of 100")
            min_area = 100

        labeled_mask, num_labels = label(self.mask > 0)
        for i in range(1, num_labels + 1):  # Skip background
            if np.sum(labeled_mask == i) < min_area:
                self.mask[labeled_mask == i] = 0  # Remove small objects
        self.update_display()

class AnnotateApp:
    def __init__(self, root, db_path, src, image_type=None, channels=None, image_size=200, annotation_column='annotate', percentiles=(1, 99), measurement=None, threshold=None, normalize_channels=None, outline=None, outline_threshold_factor=1, outline_sigma=1, edge_thickness=1, edge_transparency=100, edge_image=False, object_size=(0,0)):
        self.root = root
        self.db_path = db_path
        self.src = src
        self.index = 0
        
        #self.update_queue.put(self.SENTINEL)
        self.SENTINEL = object()        # unique sentinel for shutdown
        #self.update_queue = Queue()     # create the queue
        
        if isinstance(image_size, list):
            self.image_size = (int(image_size[0]), int(image_size[0]))
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            raise ValueError("Invalid image size")
        
        self.orig_annotation_columns = annotation_column
        self.annotation_column = annotation_column
        self._ensure_annotation_column()
        self.image_type = image_type
        self.channels = channels
        self.percentiles = percentiles
        self.images = {}
        self.pending_updates = {}
        self.labels = []
        self.adjusted_to_original_paths = {}
        self.terminate = False
        self.update_queue = Queue()
        self.measurement = measurement
        self.threshold = threshold
        self.normalize_channels = normalize_channels
        self.outline = outline
        self.outline_threshold_factor = outline_threshold_factor
        self.outline_sigma = outline_sigma
        self.edge_thickness = edge_thickness
        self.edge_transparency = edge_transparency
        self.edge_image = edge_image
        self.object_size = tuple(object_size) if object_size else (0, 0)
        
        style_out = set_dark_style(ttk.Style())
        self.font_loader = style_out['font_loader']
        self.font_size = style_out['font_size']
        self.bg_color = style_out['bg_color']
        self.fg_color = style_out['fg_color']
        self.active_color = style_out['active_color']
        self.inactive_color = style_out['inactive_color']
        
        # --- save-status UI & state ---
        self._spinner_frames = ["⠋","⠙","⠸","⠴","⠦","⠇"]  # simple TTY spinner
        self._spinner_idx = 0
        self.worker_busy = False        # set by the worker thread only
        self._last_save_ts = None       # time of last successful commit

        if self.font_loader:
            self.font_style = self.font_loader.get_font(size=self.font_size)
        else:
            self.font_style = ("Arial", 12)
        
        self.root.configure(bg=style_out['inactive_color'])

        self.filtered_paths_annotations = []
        self.prefilter_paths_annotations()
                
        self.db_update_thread = threading.Thread(target=self.update_database_worker)
        self.db_update_thread.start()

        # Set the initial window size and make it fit the screen size
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        self.root.update_idletasks()

        # grid at top
        self.grid_frame = Frame(root, bg=self.root.cget('bg'))
        self.grid_frame.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

        # status (left) + buttons (right) on the same bottom row
        self.status_label = Label(root, text="", font=self.font_style, bg=self.bg_color, fg=self.fg_color)
        self.status_label.grid(row=2, column=0, padx=10, pady=8, sticky="w")
        
        # begin polling the status 6–10 times/sec
        self._poll_save_status()        # schedules itself via .after()

        self.button_frame = Frame(root, bg=self.root.cget('bg'))
        self.button_frame.grid(row=2, column=1, padx=10, pady=8, sticky="e")  # or "ew"

        # buttons
        self.next_button = Button(self.button_frame, text="Next", command=self.next_page, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.previous_button = Button(self.button_frame, text="Back", command=self.previous_page, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.exit_button = Button(self.button_frame, text="Exit", command=self.shutdown, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.train_button = Button(self.button_frame, text="Train & Classify (beta)", command=self.train_and_classify, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.settings_button = Button(self.button_frame, text="Settings", command=self.open_settings_window, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.clear_button = Button(self.button_frame,text="Clear annotation",command=self.clear_current_annotation,bg=self.bg_color, fg=self.fg_color,highlightbackground=self.fg_color,highlightcolor=self.fg_color,highlightthickness=1)
        self.count_button = Button(self.button_frame, text="Count classes", command=self.show_class_counts, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.dl_train_button = Button(self.button_frame,text="Train (Beta)", command=self.open_deep_spacr_window,bg=self.bg_color,fg=self.fg_color,highlightbackground=self.fg_color,highlightcolor=self.fg_color,highlightthickness=1)
        #self.umap_button = Button(self.button_frame, text="Image UMAP / HParam (Beta)", command=self.open_umap_window, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        
        # pack (right to left)
        self.next_button.pack(side="right", padx=5)
        self.previous_button.pack(side="right", padx=5)
        self.exit_button.pack(side="right", padx=5)
        self.train_button.pack(side="right", padx=5)
        self.settings_button.pack(side="right", padx=5)
        self.clear_button.pack(side="right", padx=5)
        self.count_button.pack(side="right", padx=5)
        self.dl_train_button.pack(side="right", padx=5)
        #self.umap_button.pack(side="right", padx=5)

        # compute grid size (after buttons exist with real height)
        self.button_frame.update_idletasks()
        needed = self.button_frame.winfo_reqwidth()
        self.root.grid_columnconfigure(1, minsize=needed + 10, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.root.update_idletasks()
        self.calculate_grid_dimensions()

        for i in range(self.grid_rows * self.grid_cols):
            label = Label(self.grid_frame, bg=self.root.cget('bg'))
            label.grid(row=i // self.grid_cols, column=i % self.grid_cols, padx=2, pady=2, sticky="nsew")
            self.labels.append(label)
        
        # column/row weights
        self.root.grid_rowconfigure(0, weight=1)   # grid grows
        self.root.grid_rowconfigure(2, weight=0)   # bottom fixed

        for row in range(self.grid_rows):
            self.grid_frame.grid_rowconfigure(row, weight=1)
        for col in range(self.grid_cols):
            self.grid_frame.grid_columnconfigure(col, weight=1)
            
    def _int_to_color(self, k, s=0.65, v=0.95):
        """
        Deterministically map any non-negative integer k -> hex color using
        the golden-ratio conjugate to distribute hues around the color wheel.
        s,v control saturation and value (brightness).
        """
        import colorsys

        # Golden ratio conjugate (~0.618...) spreads hues evenly
        phi = 0.618033988749895
        # Wrap k around the unit interval
        h = (k * phi) % 1.0

        r, g, b = colorsys.hsv_to_rgb(h, float(s), float(v))
        return "#{:02x}{:02x}{:02x}".format(int(r * 255 + 0.5),
                                            int(g * 255 + 0.5),
                                            int(b * 255 + 0.5))

    def _label_to_color(self, val):
        """
        Public helper: for an integer label return a hex color.
        - None/0/invalid -> None (no border).
        - 1 -> blue, 2 -> red.
        - 3+ -> infinite distinct colors, deterministic.
        Caches results so colors stay stable across the session.
        """
        # Lazy-init cache on the instance
        if not hasattr(self, "_class_color_cache"):
            self._class_color_cache = {}

        try:
            if val is None:
                return None
            iv = int(val)
            if iv <= 0:
                return None
        except Exception:
            return None

        if iv in self._class_color_cache:
            return self._class_color_cache[iv]

        # Fixed starters
        if iv == 1:
            color = "#1f77b4"  # blue
        elif iv == 2:
            color = "#d62728"  # red
        else:
            # Map 3 -> k=0, 4 -> k=1, ... so early classes are well separated
            k = iv - 3
            color = self._int_to_color(k)

        self._class_color_cache[iv] = color
        return color
            
    def _embed_figure_in(self, parent, fig):
        # Clear parent
        for w in parent.winfo_children():
            try: w.destroy()
            except Exception: pass
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            lab = tk.Label(parent, text=f"Matplotlib not available: {e}", bg=self.bg_color, fg="red")
            lab.pack()
            return None
        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        canvas.draw()
        return canvas
    
    def open_umap_window(self):
        import tkinter as tk
        from tkinter import ttk, messagebox
        import threading
        import ast

        win = tk.Toplevel(self.root)
        win.title("Image UMAP & Hyperparameter Search")
        win.configure(bg=self.bg_color)
        win.geometry("1200x800")

        outer = tk.Frame(win, bg=self.bg_color)
        outer.pack(fill=tk.BOTH, expand=True)

        # Left: settings
        left = tk.Frame(outer, bg=self.bg_color)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right: live plot
        right = tk.Frame(outer, bg=self.bg_color)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10), pady=10)

        # --- Settings widgets (a practical subset; add more if you like) ---
        def _row(lbl, widget):
            r = tk.Frame(left, bg=self.bg_color)
            tk.Label(r, text=lbl, bg=self.bg_color, fg=self.fg_color, font=self.font_style).pack(side=tk.TOP, anchor="w")
            widget.pack(in_=r, fill=tk.X, expand=True)
            r.pack(fill=tk.X, pady=6)

        src_entry = tk.Entry(left)
        src_entry.insert(0, self.src)

        tables_entry = tk.Entry(left)
        tables_entry.insert(0, "cell,cytoplasm,nucleus,pathogen")  # sensible default

        row_limit_entry = tk.Entry(left)
        row_limit_entry.insert(0, "")  # blank = no limit

        # UMAP params
        n_neighbors_entry = tk.Entry(left); n_neighbors_entry.insert(0, "15")
        min_dist_entry    = tk.Entry(left); min_dist_entry.insert(0, "0.1")
        metric_entry      = tk.Entry(left); metric_entry.insert(0, "euclidean")

        # Clustering params
        clustering_cbx = ttk.Combobox(left, state="readonly", values=["dbscan","kmeans"])
        clustering_cbx.set("dbscan")

        eps_entry         = tk.Entry(left); eps_entry.insert(0, "0.5")     # DBSCAN
        min_samples_entry = tk.Entry(left); min_samples_entry.insert(0, "5")
        kmeans_k_entry    = tk.Entry(left); kmeans_k_entry.insert(0, "8")  # KMeans

        color_by_entry    = tk.Entry(left); color_by_entry.insert(0, "")   # e.g. columnID or cond
        dot_size_entry    = tk.Entry(left); dot_size_entry.insert(0, "6")
        fig_size_entry    = tk.Entry(left); fig_size_entry.insert(0, "10") # inches

        img_nr_entry      = tk.Entry(left); img_nr_entry.insert(0, "200")  # images in overlay (if plotting images)
        plot_images_var   = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="plot_images (heavy)", variable=plot_images_var,
                    bg=self.bg_color, fg=self.fg_color, selectcolor=self.bg_color).pack(anchor="w", pady=(4,0))

        # Hyperparam grids (comma-separated lists interpreted as Python literals)
        # UMAP grid: list of dicts like {"n_neighbors": 10, "min_dist": 0.1}
        red_grid_entry = tk.Entry(left)
        red_grid_entry.insert(0, """[{"n_neighbors":10,"min_dist":0.05},{"n_neighbors":15,"min_dist":0.1},{"n_neighbors":30,"min_dist":0.3}]""")

        # DBSCAN grid: list of dicts like {"eps": 0.5, "min_samples":5}
        dbscan_grid_entry = tk.Entry(left)
        dbscan_grid_entry.insert(0, """[{"eps":0.3,"min_samples":5},{"eps":0.5,"min_samples":5},{"eps":0.7,"min_samples":3}]""")

        # KMeans grid: list of dicts like {"n_clusters": 6}
        kmeans_grid_entry = tk.Entry(left)
        kmeans_grid_entry.insert(0, """[{"n_clusters":6},{"n_clusters":8},{"n_clusters":10}]""")

        # pack rows
        _row("src", src_entry)
        _row("tables (csv)", tables_entry)
        _row("row_limit (blank = all)", row_limit_entry)
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        _row("UMAP n_neighbors", n_neighbors_entry)
        _row("UMAP min_dist",    min_dist_entry)
        _row("metric",           metric_entry)
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        _row("clustering",       clustering_cbx)
        _row("DBSCAN eps",       eps_entry)
        _row("DBSCAN min_samples", min_samples_entry)
        _row("KMeans n_clusters", kmeans_k_entry)
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        _row("color_by (optional)", color_by_entry)
        _row("dot_size",         dot_size_entry)
        _row("figsize (inches)", fig_size_entry)
        _row("image_nr (if plotting images)", img_nr_entry)
        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        _row("UMAP grid (JSON list of dicts)", red_grid_entry)
        _row("DBSCAN grid (JSON list of dicts)", dbscan_grid_entry)
        _row("KMeans grid (JSON list of dicts)", kmeans_grid_entry)

        # Status + buttons
        status = tk.Label(left, text="", bg=self.bg_color, fg=self.fg_color, font=self.font_style)
        status.pack(fill=tk.X, pady=(8,4))

        btn_row = tk.Frame(left, bg=self.bg_color)
        btn_row.pack(fill=tk.X, pady=(0,6))
        run_umap_btn = ttk.Button(btn_row, text="Run UMAP")
        run_grid_btn = ttk.Button(btn_row, text="Run Hyperparam Search")
        run_umap_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        run_grid_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))

        # ----- runners -----
        def _collect_common_settings():
            # parse helpers
            def _csv_list(s):
                s = (s or "").strip()
                if not s: return []
                return [p.strip() for p in s.split(",") if p.strip()]
            def _int_or_none(s):
                s = (s or "").strip()
                return None if s == "" else int(float(s))
            def _float(s, default):
                try: return float(str(s).strip())
                except Exception: return default

            tables = _csv_list(tables_entry.get())
            row_limit = _int_or_none(row_limit_entry.get())

            settings = {
                "src": src_entry.get().strip(),
                "tables": tables if tables else ["cell","cytoplasm","nucleus","pathogen"],
                "row_limit": row_limit,
                "reduction_method": "umap",
                "n_neighbors": _int_or_none(n_neighbors_entry.get()) or 15,
                "min_dist": _float(min_dist_entry.get(), 0.1),
                "metric": metric_entry.get().strip() or "euclidean",
                "clustering": clustering_cbx.get().strip().upper(),  # DBSCAN or KMEANS
                "eps": _float(eps_entry.get(), 0.5),
                "min_samples": _int_or_none(min_samples_entry.get()) or 5,
                "image_nr": _int_or_none(img_nr_entry.get()) or 200,
                "dot_size": _int_or_none(dot_size_entry.get()) or 6,
                "figuresize": _float(fig_size_entry.get(), 10.0),
                "plot_images": bool(plot_images_var.get()),
                "color_by": (color_by_entry.get().strip() or None),
                # defaults you already support in set_default_umap_image_settings:
                "verbose": True,
                "black_background": False,
                "remove_image_canvas": False,
                "plot_outlines": False,
                "plot_points": True,
                "smooth_lines": False,
                "embedding_by_controls": False,
                "exclude": [],
                "save_figure": False,
                "plot_cluster_grids": False,
                "analyze_clusters": False,
                "n_jobs": max(1, (os.cpu_count() or 8) - 2),
                # clustering-specific extra:
                "kmeans_k": _int_or_none(kmeans_k_entry.get()) or 8,
            }
            return settings

        def _run_umap():
            settings = _collect_common_settings()
            def worker():
                try:
                    status.config(text="Running UMAP…")
                    # Call your function; ask it to return a Figure (see tweak below)
                    from spacr.core import generate_image_umap as _gen
                    fig = _gen(settings=settings, return_fig=True)
                    status.config(text="Done.")
                    self._embed_figure_in(right, fig)
                except Exception as e:
                    status.config(text=f"Error: {e}")
            threading.Thread(target=worker, daemon=True).start()

        def _run_grid():
            # parse JSON-ish lists safely
            def _parse_list(s):
                txt = (s or "").strip()
                if not txt: return []
                try:
                    return ast.literal_eval(txt)
                except Exception:
                    return []
            settings = _collect_common_settings()
            red_grid    = _parse_list(red_grid_entry.get())
            dbscan_grid = _parse_list(dbscan_grid_entry.get())
            kmeans_grid = _parse_list(kmeans_grid_entry.get())

            def worker():
                try:
                    status.config(text="Running hyperparameter search…")
                    from spacr.core import reducer_hyperparameter_search as _search
                    fig = _search(
                        settings=settings,
                        reduction_params=red_grid or [{"n_neighbors":15,"min_dist":0.1}],
                        dbscan_params=dbscan_grid or [{"eps":0.5,"min_samples":5}],
                        kmeans_params=kmeans_grid or [{"n_clusters":settings["kmeans_k"]}],
                        show=False, return_fig=True
                    )
                    status.config(text="Done.")
                    self._embed_figure_in(right, fig)
                except Exception as e:
                    status.config(text=f"Error: {e}")
            threading.Thread(target=worker, daemon=True).start()

        run_umap_btn.configure(command=_run_umap)
        run_grid_btn.configure(command=_run_grid)

            
    def _poll_save_status(self):
        """
        Main-thread UI poller: reads thread-safe flags and queue length
        to show a 'Saving…' spinner and counts. Never called from worker.
        """
        try:
            qlen = self.update_queue.qsize()
        except NotImplementedError:
            qlen = 0  # some platforms don't implement qsize reliably

        saving = self.worker_busy or qlen > 0 or bool(self.pending_updates)

        if saving:
            self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
            spin = self._spinner_frames[self._spinner_idx]
            msg = f"{spin} Saving…  queue={qlen}"
        else:
            if self._last_save_ts:
                msg = f"✓ All changes saved"
            else:
                msg = ""  # nothing saved yet, keep bar clean

        # Update the status label in the main thread
        self.status_label.config(text=msg)

        # poll ~8 times per second
        self.root.after(125, self._poll_save_status)
            
    def open_settings_window(self):
        from .gui_utils import generate_annotate_fields, convert_to_number

        # Create settings window
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Modify Annotation Settings")

        style_out = set_dark_style(ttk.Style())
        settings_window.configure(bg=style_out['bg_color'])
        
        settings_frame = tk.Frame(settings_window, bg=style_out['bg_color'])
        settings_frame.pack(fill=tk.BOTH, expand=True)

        # Generate fields with current settings pre-filled
        vars_dict = generate_annotate_fields(settings_frame)

        # Pre-fill the current settings into vars_dict
        current_settings = {
            'image_type': self.image_type or '',
            'channels': ','.join(self.channels) if self.channels else '',
            'img_size': f"{self.image_size[0]},{self.image_size[1]}",
            'annotation_column': self.annotation_column or '',
            'percentiles': ','.join(map(str, self.percentiles)),
            'measurement': ','.join(self.measurement) if self.measurement else '',
            'threshold': str(self.threshold) if self.threshold is not None else '',
            'normalize_channels': ','.join([s for s in (self.normalize_channels or []) if isinstance(s, str) and s.strip()]),
            'outline': ','.join(self.outline) if self.outline else '',
            'outline_threshold_factor': str(self.outline_threshold_factor) if hasattr(self, 'outline_threshold_factor') else '1.0',
            'outline_sigma': str(self.outline_sigma) if hasattr(self, 'outline_sigma') else '1.0',
            'edge_thickness': str(self.edge_thickness) if hasattr(self, 'edge_thickness') else '1',
            'edge_transparency': str(getattr(self, 'edge_transparency', 0.0)),
            'edge_image': str(getattr(self, 'edge_image', False)),
            'object_size': (f"{getattr(self, 'object_size', (0,0))[0]},{getattr(self, 'object_size', (0,0))[1]}"),
            'src': self.src,
            'db_path': self.db_path,
        }
        
        for key, data in vars_dict.items():
            if key in current_settings:
                data['entry'].delete(0, tk.END)
                data['entry'].insert(0, current_settings[key])
                
        def apply_new_settings():
            settings = {key: data['entry'].get() for key, data in vars_dict.items()}

            # --- channels (comma string -> list or None) ---
            settings['channels'] = (
                [s.strip().lower() for s in (settings.get('channels') or '').split(',') if s.strip()]
                or None
            )

            # --- image size & percentiles ---
            settings['img_size'] = list(map(int, settings['img_size'].split(',')))
            settings['percentiles'] = (
                list(map(convert_to_number, settings['percentiles'].split(',')))
                if settings['percentiles'] else [1, 99]
            )

            # --- normalize_channels: empty => [] (so update_settings will process it) ---
            raw_nc = settings.get('normalize_channels')
            if raw_nc is None or raw_nc.strip() == '':
                settings['normalize_channels'] = []      
            else:
                nc = [s.strip().lower() for s in raw_nc.split(',') if s.strip()]
                nc = [s for s in nc if s in {'r','g','b'}]
                settings['normalize_channels'] = nc

            raw_ol = settings.get('outline')
            if raw_ol is None or raw_ol.strip() == '':
                settings['outline'] = []
            else:
                ol = [s.strip().lower() for s in raw_ol.split(',') if s.strip()]
                ol = [s for s in ol if s in {'r','g','b'}]
                settings['outline'] = ol

            # --- numeric fields ---
            
            # --- object_size: "(min,max)" where 0 disables that bound ---
            raw_os = (settings.get('object_size') or '').strip()
            def _parse_object_size(s):
                if not s:
                    return (0, 0)
                # accept "min,max", "min , max", "min", "min," etc.
                s = s.replace(';', ',')
                parts = [p.strip() for p in s.split(',') if p.strip() != '']
                nums = []
                for p in parts[:2]:
                    try:
                        # allow floats in UI, cast to int; clamp negatives to 0
                        nums.append(max(0, int(float(p))))
                    except Exception:
                        nums.append(0)
                while len(nums) < 2:
                    nums.append(0)
                mn, mx = nums[0], nums[1]
                # if both set and out of order, swap
                if mn and mx and mn > mx:
                    mn, mx = mx, mn
                return (mn, mx)

            settings['object_size'] = _parse_object_size(raw_os)
            
            settings['outline_threshold_factor'] = (
                float(settings['outline_threshold_factor'].replace(',', '.'))
                if settings['outline_threshold_factor'] else 1.0)
            settings['outline_sigma'] = (
                float(settings['outline_sigma'].replace(',', '.'))
                if settings['outline_sigma'] else 1.0)
            settings['edge_thickness'] = (
                float(settings['edge_thickness'].replace(',', '.'))
                if settings['edge_thickness'] else 1)
            et = settings.get('edge_transparency')
            if et is None or et == '':
                settings['edge_transparency'] = 0.0
            else:
                try:
                    settings['edge_transparency'] = float(str(et).replace(',', '.'))
                except Exception:
                    settings['edge_transparency'] = 0.0
                settings['edge_transparency'] = max(0.0, min(100.0, settings['edge_transparency']))

            # --- edge_image (string -> bool) ---
            ei_raw = str(settings.get('edge_image', 'true')).strip().lower()
            settings['edge_image'] = ei_raw in ('1', 'true', 't', 'yes', 'y')

            # --- measurement / threshold ---
            try:
                settings['measurement'] = (
                    [s.strip() for s in settings['measurement'].split(',') if s.strip()]
                    if settings['measurement'] else None
                )
                settings['threshold'] = (
                    None if str(settings['threshold']).strip().lower() == 'none'
                    else int(settings['threshold'])
                )
            except Exception:
                settings['measurement'] = None
                settings['threshold'] = None

            # --- cleanup: drop empties inside lists; only top-level '' -> None ---
            for k, v in list(settings.items()):
                if isinstance(v, list):
                    settings[k] = [x for x in v if x not in (None, '')]
                elif v == '':
                    settings[k] = None

            # db path & apply
            self.db_path = os.path.join(settings.get('src'), 'measurements', 'measurements.db')

            self.update_settings(**{
                'image_type': settings.get('image_type'),
                'channels': settings.get('channels'),
                'image_size': settings.get('img_size'),
                'annotation_column': settings.get('annotation_column'),
                'percentiles': settings.get('percentiles'),
                'measurement': settings.get('measurement'),
                'threshold': settings.get('threshold'),
                'normalize_channels': settings.get('normalize_channels'),   # None => no normalization
                'outline': settings.get('outline'),                         # None => no outlines
                'outline_threshold_factor': settings.get('outline_threshold_factor'),
                'outline_sigma': settings.get('outline_sigma'),
                'edge_thickness': settings.get('edge_thickness'),
                'edge_transparency': settings.get('edge_transparency'),
                'edge_image': settings.get('edge_image'),
                'object_size': settings.get('object_size'),
                'src': settings.get('src'),
                'db_path': self.db_path
            })

            settings_window.destroy()

        apply_button = spacrButton(settings_window, text="Apply Settings", command=apply_new_settings,show_text=False, icon_name="annotate")
        apply_button.pack(pady=10)
        
    def _ensure_annotation_column(self):
        import sqlite3
        if not getattr(self, "annotation_column", None):
            return

        col = (self.annotation_column or "").replace('"', '""')
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute('PRAGMA table_info("png_list")')
            cols = {row[1] for row in cur.fetchall()}
            if self.annotation_column not in cols:
                # NULL allowed; values will be 1/2 per your app
                cur.execute(f'ALTER TABLE "png_list" ADD COLUMN "{col}" INTEGER')
                # commit occurs automatically on exiting the context if no exception
        
    def update_settings(self, **kwargs):
        import threading

        allowed_attributes = {
            'image_type', 'channels', 'image_size', 'annotation_column', 'src', 'db_path',
            'percentiles', 'measurement', 'threshold', 'normalize_channels',
            'outline', 'outline_threshold_factor', 'outline_sigma',
            'edge_thickness', 'edge_transparency', 'edge_image', 'object_size'
        }

        old_db  = getattr(self, 'db_path', None)
        old_src = getattr(self, 'src', None)

        updated = False

        for attr, value in kwargs.items():
            if attr in allowed_attributes and value is not None:

                if attr == 'normalize_channels':
                    if isinstance(value, (list, tuple)):
                        value = [str(s).strip().lower() for s in value if s is not None and str(s).strip()]
                        value = [s for s in value if s in {'r','g','b'}]
                        value = value or None
                    elif isinstance(value, str):
                        parts = [s.strip().lower() for s in value.split(',') if s.strip()]
                        parts = [s for s in parts if s in {'r','g','b'}]
                        value = parts or None
                    else:
                        value = None

                elif attr == 'outline':
                    if isinstance(value, (list, tuple)):
                        value = [str(s).strip().lower() for s in value if s is not None and str(s).strip()]
                    elif isinstance(value, str):
                        value = [s.strip().lower() for s in value.split(',') if s.strip()]
                    else:
                        value = []
                    value = [s for s in value if s in {'r','g','b'}]
                    value = value or None

                elif attr == 'outline_threshold_factor':
                    value = float(value)
                elif attr == 'outline_sigma':
                    value = float(value)

                # **CHANGED: keep fractional thickness**
                elif attr == 'edge_thickness':
                    value = float(value)

                elif attr == 'edge_transparency':
                    try:
                        value = float(value)
                    except Exception:
                        value = 0.0
                    value = max(0.0, min(100.0, value))
                elif attr == 'edge_image':
                    value = bool(value)
                    
                elif attr == 'object_size':
                    # normalize to a 2-tuple of non-negative ints; (0,0) means no bounds
                    v = value
                    if v in (None, '', []):
                        v = (0, 0)
                    elif isinstance(v, str):
                        # reuse the same parsing logic as above, inline:
                        s = v.replace(';', ',')
                        parts = [p.strip() for p in s.split(',') if p.strip() != '']
                        a = []
                        for p in parts[:2]:
                            try:
                                a.append(max(0, int(float(p))))
                            except Exception:
                                a.append(0)
                        while len(a) < 2:
                            a.append(0)
                        mn, mx = a
                    elif isinstance(v, (list, tuple)):
                        mn = max(0, int(v[0])) if len(v) > 0 else 0
                        mx = max(0, int(v[1])) if len(v) > 1 else 0
                    else:
                        mn, mx = (0, 0)
                    if mn and mx and mn > mx:
                        mn, mx = mx, mn
                    value = (mn, mx)

                setattr(self, attr, value)
                updated = True

        if ('annotation_column' in kwargs and kwargs['annotation_column']) or ('db_path' in kwargs and kwargs['db_path']):
            self._ensure_annotation_column()

        if 'image_size' in kwargs:
            if isinstance(self.image_size, list):
                self.image_size = (int(self.image_size[0]), int(self.image_size[0]))
            elif isinstance(self.image_size, int):
                self.image_size = (self.image_size, self.image_size)
            elif isinstance(self.image_size, tuple) and len(self.image_size) == 2:
                self.image_size = tuple(map(int, self.image_size))
            else:
                raise ValueError("Invalid image size")

            self.calculate_grid_dimensions()
            self.recreate_image_grid()

        if self.src != old_src:
            self.adjusted_to_original_paths.clear()
            self.index = 0

        if self.db_path != old_db:
            if self.pending_updates:
                self.update_queue.put(self.pending_updates.copy())
                self.pending_updates.clear()
            self.update_queue.put(self.SENTINEL)
            self.update_queue.join()
            try:
                if getattr(self, 'db_update_thread', None):
                    self.db_update_thread.join()
            except Exception:
                pass
            self.terminate = False
            self.worker_busy = False
            self._last_save_ts = None
            self.db_update_thread = threading.Thread(target=self.update_database_worker, daemon=True)
            self.db_update_thread.start()

        if updated:
            current_index = self.index
            self.prefilter_paths_annotations()
            max_index = len(self.filtered_paths_annotations) - 1
            self.index = min(current_index, max(0, max(len(self.filtered_paths_annotations) - self.grid_rows * self.grid_cols, 0)))
            self.load_images()

    def recreate_image_grid(self):
        # Remove current labels
        for label in self.labels:
            label.destroy()
        self.labels.clear()

        # Recreate the labels grid with updated dimensions
        for i in range(self.grid_rows * self.grid_cols):
            label = Label(self.grid_frame, bg=self.root.cget('bg'))
            label.grid(row=i // self.grid_cols, column=i % self.grid_cols, padx=2, pady=2, sticky="nsew")
            self.labels.append(label)

        # Reconfigure grid weights
        for row in range(self.grid_rows):
            self.grid_frame.grid_rowconfigure(row, weight=1)
        for col in range(self.grid_cols):
            self.grid_frame.grid_columnconfigure(col, weight=1)
            
    def update_display(self):
        self.prefilter_paths_annotations()
        self.load_images()
            
    def swich_back_annotation_column(self):
        self.annotation_column = self.orig_annotation_columns
        self._ensure_annotation_column()
        self.prefilter_paths_annotations()
        self.update_display()
            
    def calculate_grid_dimensions(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        status_h  = self.status_label.winfo_height()
        buttons_h = self.button_frame.winfo_height()
        bottom_h  = max(status_h, buttons_h) + 8  # same row => max
        self.grid_cols = max(1, w // (self.image_size[0] + 4))
        self.grid_rows = max(1, (h - bottom_h) // (self.image_size[1] + 4))

    def prefilter_paths_annotations(self):
        from .io import _read_and_join_tables, _read_db
        from .utils import is_list_of_lists
        
        self._ensure_annotation_column()

        if self.measurement and self.threshold is not None:
            df = _read_and_join_tables(self.db_path)
            png_list_df = _read_db(self.db_path, tables=['png_list'])[0]
            png_list_df = png_list_df.set_index('prcfo')
            df = df.merge(png_list_df, left_index=True, right_index=True)
            df[self.annotation_column] = None
            before = len(df)

            if isinstance(self.threshold, int):
                if isinstance(self.measurement, list):
                    mes = self.measurement[0]
                if isinstance(self.measurement, str):
                    mes = self.measurement
                df = df[df[f'{mes}'] == self.threshold]

            if is_list_of_lists(self.measurement):
                if isinstance(self.threshold, list) or is_list_of_lists(self.threshold):
                    if len(self.measurement) == len(self.threshold):
                        for idx, var in enumerate(self.measurement):
                            df = df[df[var[idx]] > self.threshold[idx]]
                        after = len(df)
                    elif len(self.measurement) == len(self.threshold) * 2:
                        th_idx = 0
                        for idx, var in enumerate(self.measurement):
                            if idx % 2 != 0:
                                th_idx += 1
                                thd = self.threshold
                                if isinstance(thd, list):
                                    thd = thd[0]
                                df[f'threshold_measurement_{idx}'] = df[self.measurement[idx]] / df[self.measurement[idx + 1]]
                                print(f"mean threshold_measurement_{idx}: {np.mean(df['threshold_measurement'])}")
                                print(f"median threshold measurement: {np.median(df[self.measurement])}")
                                df = df[df[f'threshold_measurement_{idx}'] > thd]
                        after = len(df)

            elif isinstance(self.measurement, list):
                df['threshold_measurement'] = df[self.measurement[0]] / df[self.measurement[1]]
                print(f"mean threshold measurement: {np.mean(df['threshold_measurement'])}")
                print(f"median threshold measurement: {np.median(df[self.measurement])}")
                df = df[df['threshold_measurement'] > self.threshold]
                after = len(df)
                self.measurement = 'threshold_measurement'
                print(f'Removed: {before-after} rows, retained {after}')

            else:
                print(f"mean threshold measurement: {np.mean(df[self.measurement])}")
                print(f"median threshold measurement: {np.median(df[self.measurement])}")
                before = len(df)
                if isinstance(self.threshold, str):
                    if self.threshold == 'q1':
                        self.threshold = df[self.measurement].quantile(0.1)
                    if self.threshold == 'q2':
                        self.threshold = df[self.measurement].quantile(0.2)
                    if self.threshold == 'q3':
                        self.threshold = df[self.measurement].quantile(0.3)
                    if self.threshold == 'q4':
                        self.threshold = df[self.measurement].quantile(0.4)
                    if self.threshold == 'q5':
                        self.threshold = df[self.measurement].quantile(0.5)
                    if self.threshold == 'q6':
                        self.threshold = df[self.measurement].quantile(0.6)
                    if self.threshold == 'q7':
                        self.threshold = df[self.measurement].quantile(0.7)
                    if self.threshold == 'q8':
                        self.threshold = df[self.measurement].quantile(0.8)
                    if self.threshold == 'q9':
                        self.threshold = df[self.measurement].quantile(0.9)
                print(f"threshold: {self.threshold}")

                df = df[df[self.measurement] > self.threshold]
                after = len(df)
                print(f'Removed: {before-after} rows, retained {after}')

            df = df.dropna(subset=['png_path'])
            if self.image_type:
                before = len(df)
                if isinstance(self.image_type, list):
                    for tpe in self.image_type:
                        print(f"Looking for {tpe}")
                        df = df[df['png_path'].str.contains(tpe)]
                        print(f"Found {len(df)} entries for {tpe}")
                else:
                    df = df[df['png_path'].str.contains(self.image_type)]
                after = len(df)
                print(f'image_type: Removed: {before-after} rows, retained {after}')

            self.filtered_paths_annotations = df[['png_path', self.annotation_column]].values.tolist()

        else:
            # simple SELECT branch -> use context manager
            col = (self.annotation_column or "").replace('"', '""')
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                c = conn.cursor()
                if self.image_type:
                    c.execute(
                        f'SELECT png_path, "{col}" FROM "png_list" WHERE png_path LIKE ?',
                        (f"%{self.image_type}%",)
                    )
                else:
                    c.execute(f'SELECT png_path, "{col}" FROM "png_list"')
                self.filtered_paths_annotations = c.fetchall()

    def load_images_v1(self):
        for label in self.labels:
            label.config(image='')

        self.images = {}
        paths_annotations = self.filtered_paths_annotations[self.index:self.index + self.grid_rows * self.grid_cols]

        adjusted_paths = []
        for path, annotation in paths_annotations:
            if not path.startswith(self.src):
                parts = path.split('/data/')
                if len(parts) > 1:
                    new_path = os.path.join(self.src, 'data', parts[1])
                    self.adjusted_to_original_paths[new_path] = path
                    adjusted_paths.append((new_path, annotation))
                else:
                    adjusted_paths.append((path, annotation))
            else:
                adjusted_paths.append((path, annotation))

        with ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(self.load_single_image, adjusted_paths))

        for i, (img, annotation) in enumerate(loaded_images):
            if annotation:
                border_color = self.active_color if annotation == 1 else 'red'
                img = self.add_colored_border(img, border_width=5, border_color=border_color)

            photo = ImageTk.PhotoImage(img)
            label = self.labels[i]
            self.images[label] = photo
            label.config(image=photo)

            path = adjusted_paths[i][0]
            label.bind('<Button-1>', self.get_on_image_click(path, label, img))
            label.bind('<Button-3>', self.get_on_image_click(path, label, img))

        self.root.update()
        
    def load_images(self):
        for label in self.labels:
            label.config(image='')

        self.images = {}
        paths_annotations = self.filtered_paths_annotations[self.index:self.index + self.grid_rows * self.grid_cols]

        adjusted_paths = []
        for path, annotation in paths_annotations:
            if not path.startswith(self.src):
                parts = path.split('/data/')
                if len(parts) > 1:
                    new_path = os.path.join(self.src, 'data', parts[1])
                    self.adjusted_to_original_paths[new_path] = path
                    adjusted_paths.append((new_path, annotation))
                else:
                    adjusted_paths.append((path, annotation))
            else:
                adjusted_paths.append((path, annotation))

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(self.load_single_image, adjusted_paths))

        for i, (img, annotation) in enumerate(loaded_images):
            # NEW: infinite palette
            border_color = self._label_to_color(annotation)
            if border_color:
                img = self.add_colored_border(img, border_width=5, border_color=border_color)

            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img)
            label = self.labels[i]
            self.images[label] = photo
            label.config(image=photo)

            path = adjusted_paths[i][0]
            label.bind('<Button-1>', self.get_on_image_click(path, label, img))
            label.bind('<Button-3>', self.get_on_image_click(path, label, img))

        self.root.update()
        
    def show_class_counts(self):
        import tkinter as tk
        from tkinter import ttk, messagebox

        if not self.annotation_column:
            messagebox.showerror("Error", "No annotation column is set.")
            return
        self._ensure_annotation_column()

        col = (self.annotation_column or "").replace('"', '""')
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                f'SELECT "{col}" AS cls, COUNT(*) '
                f'FROM "png_list" '
                f'WHERE "{col}" IS NOT NULL '
                f'GROUP BY "{col}" '
                f'ORDER BY 1'
            )
            rows = cur.fetchall()

        win = tk.Toplevel(self.root)
        win.title("Class counts (all)")
        win.configure(bg=self.root.cget('bg'))

        frame = tk.Frame(win, bg=self.root.cget('bg'))
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree = ttk.Treeview(frame, columns=("cls","count","color"), show="headings", height=10)
        for cid, text, width in (("cls","Class",80), ("count","Count",100), ("color","Color",120)):
            tree.heading(cid, text=text)
            tree.column(cid, anchor="center", width=width)
        tree.pack(fill=tk.BOTH, expand=True)

        # Insert rows with hex color in the last column
        for cls, cnt in rows:
            try:
                cls_int = int(cls)
            except Exception:
                continue
            hexcol = self._label_to_color(cls_int) or ""
            tree.insert('', 'end', values=(cls_int, int(cnt), hexcol))

        # Simple legend of colored squares
        legend = tk.Frame(win, bg=self.root.cget('bg'))
        legend.pack(fill=tk.X, padx=10, pady=8)
        for cls, _ in rows[:20]:  # avoid over-long legends
            try:
                c = int(cls)
            except Exception:
                continue
            hx = self._label_to_color(c) or "#888888"
            sw = tk.Canvas(legend, width=18, height=18, highlightthickness=0, bg=self.root.cget('bg'))
            sw.create_rectangle(2, 2, 16, 16, outline=hx, fill=hx)
            tk.Label(legend, text=str(c), bg=self.root.cget('bg'), fg=self.fg_color).pack(side="left", padx=(2,8))
            sw.pack(side="left")
        
    def show_class_counts_v1(self):
        import sqlite3
        import tkinter as tk
        from tkinter import ttk, messagebox

        # Make sure the column exists
        if not self.annotation_column:
            messagebox.showerror("Error", "No annotation column is set.")
            return
        self._ensure_annotation_column()

        # Count classes 1 and 2
        col = (self.annotation_column or "").replace('"', '""')
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(
                f'SELECT "{col}" AS cls, COUNT(*) '
                f'FROM "png_list" '
                f'WHERE "{col}" IN (1, 2) '
                f'GROUP BY "{col}"'
            )
            rows = cur.fetchall()

        counts = {1: 0, 2: 0}
        for cls, cnt in rows:
            if cls in (1, 2):
                counts[int(cls)] = int(cnt)

        # Build popup window
        win = tk.Toplevel(self.root)
        win.title("Class counts")
        win.configure(bg=self.root.cget('bg'))

        # Table using Treeview
        frame = tk.Frame(win, bg=self.root.cget('bg'))
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree = ttk.Treeview(frame, columns=(self.annotation_column,), show='headings', height=2)
        tree['columns'] = ('class_name', self.annotation_column)
        tree['show'] = 'headings'
        tree.heading('class_name', text='Class')
        tree.column('class_name', width=120, anchor='w')
        tree.heading(self.annotation_column, text=self.annotation_column)
        tree.column(self.annotation_column, anchor='center', width=120)

        # Insert rows
        tree.insert('', 'end', values=('class 1', counts[1]))
        tree.insert('', 'end', values=('class 2', counts[2]))
        tree.pack(fill=tk.BOTH, expand=True)

        # Close button
        btn = ttk.Button(win, text="Close", command=win.destroy)
        btn.pack(pady=8)

    def load_single_image(self, path_annotation_tuple):
        path, annotation = path_annotation_tuple
        if not os.path.exists(path):
            blank = Image.new('RGB', self.image_size, color=(30, 30, 30))
            print(f"Could not find image: {path}")
            return blank, annotation

        img = Image.open(path)

        # Normalize (optionally) – returns RGB ndarray-equivalent in a PIL Image
        img = self.normalize_image(img, self.percentiles, self.normalize_channels)
        #img = self.normalize_image(img, self.normalize, self.percentiles, self.normalize_channels)
        img = img.convert('RGB')

        # Keep a copy BEFORE filtering – used for edge generation and for edge_image=True blending
        full_img = img

        # Apply channel filter for the visible base
        img = self.filter_channels(img)

        if self.outline:
            img = self.outline_image(
                base_img=img,
                full_img=full_img,
                edge_sigma=self.outline_sigma,
                edge_thickness=self.edge_thickness,
                fill_holes=True,
                object_size=getattr(self, "object_size", (0, 0))
            )

        img = img.resize(self.image_size)
        return img, annotation
    
    @staticmethod
    def fill_holes(mask, min_size=0):
        """
        Fill holes inside True regions of a binary mask.

        Args:
            mask (ndarray[bool]): Binary mask where True denotes foreground.
            min_size (int): Minimum hole area to fill (in pixels).
                - <= 0 : fill ALL internal holes.
                -  > 0 : fill only holes smaller than min_size; reopen larger ones.

        Returns:
            ndarray[bool]: Hole-filled mask.
        """
        import numpy as np
        from scipy.ndimage import binary_fill_holes, label

        m = mask.astype(bool)
        filled = binary_fill_holes(m)

        if min_size <= 0:
            return filled

        # Pixels that were holes and got filled
        filled_holes = filled & ~m

        # Reopen (unfill) holes whose area >= min_size
        lbl, n = label(filled_holes)
        if n == 0:
            return filled

        reopen = np.zeros_like(m, dtype=bool)
        for i in range(1, n + 1):
            if (lbl == i).sum() >= int(min_size):
                reopen |= (lbl == i)

        return filled & ~reopen
    
    @staticmethod
    def _filter_objects_by_area(mask, min_size=0, max_size=0):
        import numpy as np
        from scipy.ndimage import label
        m = mask.astype(bool)
        if not m.any():
            return m
        lbl, n = label(m)
        if n == 0:
            return m
        counts = np.bincount(lbl.ravel())
        lo = int(min_size) if int(min_size) > 0 else 0
        hi = int(max_size) if int(max_size) > 0 else np.iinfo(np.int64).max
        keep = np.zeros_like(counts, dtype=bool)
        for i in range(1, len(counts)):
            area = counts[i]
            if lo <= area <= hi:
                keep[i] = True
        return keep[lbl]
        
    def outline_image(self, base_img, full_img, edge_sigma=1, edge_thickness=1, fill_holes=True, object_size=(0, 0)):
        """
        Anti-aliased outlines with sub-pixel thickness that never get dimmer as they get thinner.
        Uses peak normalization so outline brightness is thickness-invariant (then scaled only by edge_transparency).

        Args:
            base_img (PIL.Image): already filtered (visible base)
            full_img (PIL.Image): normalized RGB before filtering (edge detection / underlay)
            edge_sigma (float): Gaussian smoothing before thresholding
            edge_thickness (float): outline thickness in output pixels (supports < 1, e.g. 0.01)
            fill_holes (bool): fill internal holes in foreground masks before boundary extraction
            object_size (tuple[int,int]): (min_px, max_px) connected-component area filter.
                                        0 disables that bound.
        """
        import numpy as np
        from PIL import Image
        from scipy.ndimage import gaussian_filter, binary_closing
        from skimage.filters import threshold_otsu
        from skimage.segmentation import find_boundaries

        base_arr = np.asarray(base_img).copy()
        full_arr = np.asarray(full_img)
        if base_arr.ndim != 3 or base_arr.shape[2] != 3:
            return base_img

        out_img = base_arr
        channel_map = {'r': 0, 'g': 1, 'b': 2}
        factor = float(getattr(self, 'outline_threshold_factor', 1.0))

        # global opacity 0..1 (100 => fully bright)
        transp = float(getattr(self, 'edge_transparency', 0.0))
        opacity_global = max(0.0, min(1.0, transp / 100.0))

        outline_channels = [ch for ch in (self.outline or []) if ch in channel_map]
        show_underlay = bool(getattr(self, 'edge_image', True))

        if not show_underlay and outline_channels:
            for ch in outline_channels:
                out_img[:, :, channel_map[ch]] = 0

        if opacity_global == 0.0 or not outline_channels:
            from PIL import Image as _Image
            return _Image.fromarray(out_img)

        # Supersampling factor (AA quality; does NOT widen geometry)
        SS = 8
        H, W = out_img.shape[:2]
        upW, upH = W * SS, H * SS

        # unpack object_size bounds
        try:
            min_px, max_px = object_size if object_size is not None else (0, 0)
        except Exception:
            min_px, max_px = (0, 0)

        for ch in outline_channels:
            idx = channel_map[ch]

            if show_underlay:
                out_img[:, :, idx] = full_arr[:, :, idx]

            # Smooth & threshold (original grid)
            ch_sm = gaussian_filter(full_arr[:, :, idx].astype(np.float32), sigma=float(edge_sigma))
            try:
                otsu = threshold_otsu(ch_sm)
            except Exception:
                otsu = np.percentile(ch_sm, 50.0)
            thr = float(min(255.0, max(0.0, otsu * factor)))
            fg_mask = (ch_sm > thr)

            # Bridge tiny gaps + fill internal holes
            fg_mask = binary_closing(fg_mask, structure=np.ones((3, 3), dtype=bool))
            if fill_holes:
                fg_mask = self.fill_holes(fg_mask, min_size=0)

            # Area filtering (keep only sizes within [min_px, max_px], with 0 => no bound)
            if (min_px and min_px > 0) or (max_px and max_px > 0):
                fg_mask = self._filter_objects_by_area(fg_mask, min_size=min_px, max_size=max_px)

            # 1-px boundary (original grid)
            edge = find_boundaries(fg_mask, mode='inner').astype(np.uint8)

            # Supersample WITHOUT widening: keep a crisp hi-res line
            edge_img = Image.fromarray((edge * 255).astype(np.uint8), mode='L')
            edge_hi = edge_img.resize((upW, upH), resample=Image.NEAREST)
            edge_hi_arr = np.asarray(edge_hi, dtype=np.float32) / 255.0  # {0,1} in hi-res

            # Thickness mapping (output px -> hi-res px); only dilate if >= 1 px
            desired = max(0.0, float(edge_thickness))
            hi_radius = desired * SS
            if hi_radius >= 1.0:
                from skimage.morphology import dilation, disk
                r_int = int(np.floor(hi_radius))
                if r_int >= 1:
                    thick = dilation(edge_hi_arr > 0.5, disk(r_int)).astype(np.float32)
                    edge_hi_arr = np.maximum(edge_hi_arr, thick)

            # Downsample → anti-aliased coverage (0..1)
            alpha_lo = Image.fromarray((edge_hi_arr * 255).astype(np.uint8), mode='L') \
                            .resize((W, H), resample=Image.LANCZOS)
            alpha = np.asarray(alpha_lo, dtype=np.float32) / 255.0

            # NEVER-DIM: normalize to unit peak
            peak = float(alpha.max())
            if peak > 0:
                alpha = alpha / peak

            # Apply global opacity
            alpha = np.clip(alpha * opacity_global, 0.0, 1.0)

            # Alpha blend onto the channel
            orig = out_img[:, :, idx].astype(np.float32)
            blended = alpha * 255.0 + (1.0 - alpha) * orig
            out_img[:, :, idx] = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(out_img)

    @staticmethod
    def normalize_image(img, percentiles=(1, 99), normalize_channels=None):
        """
        If normalize_channels is None or [], do nothing.
        Otherwise normalize only those channels (r/g/b).
        """
        img_array = np.array(img)
        img_array = np.clip(img_array, 0, 255)

        if not normalize_channels:  # None or []
            return Image.fromarray(img_array.astype('uint8'))

        if img_array.ndim == 2:
            p2, p98 = np.percentile(img_array, percentiles)
            out = rescale_intensity(img_array, in_range=(p2, p98), out_range=(0, 255))
            return Image.fromarray(np.clip(out, 0, 255).astype('uint8'))

        channel_map = {'r': 0, 'g': 1, 'b': 2}
        out = img_array.astype(np.float32).copy()
        for ch in normalize_channels:
            idx = channel_map.get(str(ch).lower())
            if idx is None: 
                continue
            p2, p98 = np.percentile(out[:, :, idx], percentiles)
            out[:, :, idx] = rescale_intensity(out[:, :, idx], in_range=(p2, p98), out_range=(0, 255))
        return Image.fromarray(np.clip(out, 0, 255).astype('uint8'))

    def add_colored_border(self, img, border_width, border_color):
        top_border = Image.new('RGB', (img.width, border_width), color=border_color)
        bottom_border = Image.new('RGB', (img.width, border_width), color=border_color)
        left_border = Image.new('RGB', (border_width, img.height), color=border_color)
        right_border = Image.new('RGB', (border_width, img.height), color=border_color)

        bordered_img = Image.new('RGB', (img.width + 2 * border_width, img.height + 2 * border_width), color=self.fg_color)
        bordered_img.paste(top_border, (border_width, 0))
        bordered_img.paste(bottom_border, (border_width, img.height + border_width))
        bordered_img.paste(left_border, (0, border_width))
        bordered_img.paste(right_border, (img.width + border_width, border_width))
        bordered_img.paste(img, (border_width, border_width))

        return bordered_img
    
    def filter_channels(self, img):
        r, g, b = img.split()

        if self.channels:
            # normalize and sanitize input like ['R', ' g ', None] -> {'r','g'}
            chset = {str(c).strip().lower() for c in self.channels if c is not None and str(c).strip()}

            if 'r' not in chset:
                r = r.point(lambda _: 0)
            if 'g' not in chset:
                g = g.point(lambda _: 0)
            if 'b' not in chset:
                b = b.point(lambda _: 0)

        # always return RGB; never collapse to grayscale
        return Image.merge("RGB", (r, g, b))

    def get_on_image_click_v1(self, path, label, img):
        def on_image_click(event):
            new_annotation = 1 if event.num == 1 else (2 if event.num == 3 else None)
            
            original_path = self.adjusted_to_original_paths.get(path, path)
            
            if original_path in self.pending_updates and self.pending_updates[original_path] == new_annotation:
                self.pending_updates[original_path] = None
                new_annotation = None
            else:
                self.pending_updates[original_path] = new_annotation
            
            print(f"Image {os.path.split(path)[1]} annotated: {new_annotation}")
            
            img_ = img.crop((5, 5, img.width-5, img.height-5))
            border_fill = self.active_color if new_annotation == 1 else ('red' if new_annotation == 2 else None)
            img_ = ImageOps.expand(img_, border=5, fill=border_fill) if border_fill else img_

            photo = ImageTk.PhotoImage(img_)
            self.images[label] = photo
            label.config(image=photo)
            self.root.update()

        return on_image_click
    
    def get_on_image_click(self, path, label, img):
        from PIL import ImageTk, ImageOps
        import os

        def on_image_click(event):
            new_annotation = 1 if event.num == 1 else (2 if event.num == 3 else None)

            original_path = self.adjusted_to_original_paths.get(path, path)

            if original_path in self.pending_updates and self.pending_updates[original_path] == new_annotation:
                self.pending_updates[original_path] = None
                new_annotation = None
            else:
                self.pending_updates[original_path] = new_annotation

            print(f"Image {os.path.split(path)[1]} annotated: {new_annotation}")

            # Remove existing 5px border then reapply with new color (if any)
            img_ = img.crop((5, 5, img.width - 5, img.height - 5))
            border_fill = self._label_to_color(new_annotation)

            if border_fill:
                img_ = ImageOps.expand(img_, border=5, fill=border_fill)

            photo = ImageTk.PhotoImage(img_)
            self.images[label] = photo
            label.config(image=photo)
            self.root.update()

        return on_image_click

    @staticmethod
    def update_html(text):
        display(HTML(f"""
        <script>
        document.getElementById('unique_id').innerHTML = '{text}';
        </script>
        """))
        
    def clear_current_annotation(self):
        import sqlite3, queue
        from tkinter import messagebox

        # Confirm
        if not messagebox.askyesno(
            "Confirm",
            f'This will clear all annotations in "{self.annotation_column}".'
        ):
            return  # cancel

        # Ensure column exists
        self._ensure_annotation_column()

        # Null the entire column (context manager => fast close/unlock)
        col = (self.annotation_column or "").replace('"', '""')
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute(f'UPDATE "png_list" SET "{col}" = NULL')

        # Clear any pending updates and drain the queue
        self.pending_updates.clear()
        try:
            while True:
                self.update_queue.get_nowait()
        except queue.Empty:
            pass

        # Refresh UI (no borders now)
        self.prefilter_paths_annotations()
        self.load_images()
        
    def update_database_worker(self):
        import sqlite3, queue, time

        # generous busy-timeout so short locks don't blow up under load
        conn = sqlite3.connect(self.db_path, timeout=30)
        cur = conn.cursor()
        try:
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                conn.commit()
            except Exception:
                pass

            while True:
                try:
                    item = self.update_queue.get(timeout=0.1)
                except queue.Empty:
                    # allow graceful exit after shutdown signal
                    if self.terminate:
                        break
                    continue

                # --- graceful shutdown path ---
                if item is self.SENTINEL:
                    # mark the SENTINEL as done so update_queue.join() can finish
                    self.update_queue.task_done()
                    break

                # --- normal batch update ---
                pending_updates = item  # dict: {png_path: annotation or None}
                if not pending_updates:
                    self.update_queue.task_done()
                    continue

                self.worker_busy = True
                col = (self.annotation_column or "").replace('"', '""')
                to_null = [p for p, v in pending_updates.items() if v is None]
                to_set  = [(int(v), p) for p, v in pending_updates.items() if v is not None]

                try:
                    if to_null:
                        cur.executemany(
                            f'UPDATE "png_list" SET "{col}" = NULL WHERE png_path = ?',
                            [(p,) for p in to_null]
                        )
                    if to_set:
                        cur.executemany(
                            f'UPDATE "png_list" SET "{col}" = ? WHERE png_path = ?',
                            to_set
                        )
                    conn.commit()
                finally:
                    self.worker_busy = False
                    self._last_save_ts = time.time()
                    self.update_queue.task_done()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()
    
    def shutdown(self):
        # push any pending UI updates first
        if self.pending_updates:
            self.update_queue.put(self.pending_updates.copy())
            self.pending_updates.clear()

        # signal termination and sentinel
        self.terminate = True
        self.update_queue.put(self.SENTINEL)

        # wait for ALL tasks (including the sentinel) to be marked done
        self.update_queue.join()

        # now the worker has exited; join without timeout
        try:
            self.db_update_thread.join()
        except Exception:
            pass

        # close UI
        try:
            self.root.quit()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

        print("Quit application")
        
    def next_page(self):
        if self.pending_updates:
            # show saving right away until worker picks it up
            self.worker_busy = True
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index += self.grid_rows * self.grid_cols
        self.prefilter_paths_annotations()
        self.load_images()

    def previous_page(self):
        if self.pending_updates:
            self.worker_busy = True
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index = max(0, self.index - self.grid_rows * self.grid_cols)
        self.prefilter_paths_annotations()
        self.load_images()

    def update_gui_text(self, text):
        self.status_label.config(text=text)
        self.root.update()
        
    def train_and_classify(self):
        """
        1) Merge data from the relevant DB tables (including png_list).
        2) Collect manual annotations from png_list.<annotation_column> => 'manual_annotation'.
        - 1 => class=1, 2 => class=0 (for training).
        3) If only one class is present, randomly sample unannotated images as the other class.
        4) Train an XGBoost model.
        5) Classify *all* rows -> fill XGboost_score (prob of class=1) & XGboost_annotation (1 or 2 if high confidence).
        6) Write those columns back to sqlite.
        7) Refresh the UI.
        """
        import sqlite3
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from xgboost import XGBClassifier

        # Optionally, update your GUI status label
        self.update_gui_text("Merging data...")

        from .io import _read_and_merge_data

        # (1) Merge data
        merged_df, obj_df_ls = _read_and_merge_data(
            locs=[self.db_path],
            tables=['cell', 'cytoplasm', 'nucleus', 'pathogen', 'png_list'],
            verbose=False
        )

        # (2) Load manual annotations from the DB (with context manager)
        self._ensure_annotation_column()
        colq = (self.annotation_column or "").replace('"', '""')
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            c = conn.cursor()
            c.execute(
                f'SELECT png_path, "{colq}" FROM "png_list" '
                f'WHERE "{colq}" IS NOT NULL'
            )
            annotated_rows = c.fetchall()

        annot_dict = dict(annotated_rows)
        merged_df['manual_annotation'] = merged_df['png_path'].map(annot_dict)

        # Subset with manual labels
        annotated_df = merged_df.dropna(subset=['manual_annotation']).copy()
        annotated_df['manual_annotation'] = annotated_df['manual_annotation'].replace({2: 0}).astype(int)

        # (3) Handle single-class scenario
        class_counts = annotated_df['manual_annotation'].value_counts()
        if len(class_counts) == 1:
            single_class = class_counts.index[0]  # 0 or 1
            needed = class_counts.iloc[0]
            other_class = 1 if single_class == 0 else 0

            unannotated_df_all = merged_df[merged_df['manual_annotation'].isna()].copy()
            if len(unannotated_df_all) == 0:
                print("No unannotated rows to sample for the other class. Cannot proceed.")
                self.update_gui_text("Not enough data to train (no second class).")
                return

            sample_size = min(needed, len(unannotated_df_all))
            artificially_labeled = unannotated_df_all.sample(n=sample_size, replace=False).copy()
            artificially_labeled['manual_annotation'] = other_class

            annotated_df = pd.concat([annotated_df, artificially_labeled], ignore_index=True)
            print(f"Only one class was present => randomly labeled {sample_size} unannotated rows as {other_class}.")

        if len(annotated_df) < 2:
            print("Not enough annotated data to train (need at least 2).")
            self.update_gui_text("Not enough data to train.")
            return

        # (4) Train XGBoost
        self.update_gui_text("Training XGBoost model...")

        # Identify numeric columns
        ignore_cols = {'png_path', 'manual_annotation'}
        feature_cols = [
            col for col in annotated_df.columns
            if col not in ignore_cols
            and (annotated_df[col].dtype == float or annotated_df[col].dtype == int)
        ]

        X_data = annotated_df[feature_cols].fillna(0).values
        y_data = annotated_df['manual_annotation'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.1, random_state=42
        )
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print("=== Classification Report ===")
        print(classification_report(y_test, preds))
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test, preds))

        # (5) Classify ALL rows
        all_df = merged_df.copy()
        X_all = all_df[feature_cols].fillna(0).values
        probs_all = model.predict_proba(X_all)[:, 1]
        all_df['XGboost_score'] = probs_all

        def get_annotation_from_prob(prob):
            if prob > 0.9:
                return 1
            elif prob < 0.1:
                return 0
            return None

        xgb_anno_col = [get_annotation_from_prob(p) for p in probs_all]
        xgb_anno_col = [2 if x == 0 else x for x in xgb_anno_col]  # convert 0->2
        all_df['XGboost_annotation'] = xgb_anno_col

        # (6) Write results back (context manager + WAL tuning)
        self.update_gui_text("Updating the database with XGBoost predictions...")
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            c = conn.cursor()
            try:
                c.execute("ALTER TABLE png_list ADD COLUMN XGboost_annotation INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                c.execute("ALTER TABLE png_list ADD COLUMN XGboost_score FLOAT")
            except sqlite3.OperationalError:
                pass

            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")

            for _, row in all_df.iterrows():
                score_val = float(row['XGboost_score'])
                anno_val  = row['XGboost_annotation']
                the_path  = row['png_path']
                if pd.isna(the_path):
                    continue

                if pd.isna(anno_val):
                    c.execute("""
                        UPDATE png_list
                        SET XGboost_annotation = NULL,
                            XGboost_score       = ?
                        WHERE png_path = ?
                    """, (score_val, the_path))
                else:
                    c.execute("""
                        UPDATE png_list
                        SET XGboost_annotation = ?,
                            XGboost_score       = ?
                        WHERE png_path = ?
                    """, (int(anno_val), score_val, the_path))

        # switch to the new column and (optionally) refresh the view
        self.annotation_column = 'XGboost_annotation'
        
    def _get_png_list_columns(self):
        """Return all columns from png_list; caller can decide which are 'annotation'."""
        import sqlite3
        cols = []
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute('PRAGMA table_info("png_list")')
            for _, name, coltype, *_ in cur.fetchall():
                cols.append((name, (coltype or "").upper()))
        return cols

    def _parse_field_value(self, key, raw):
        """
        Convert string from Entry to the right Python type using defaults as a hint.
        Handles bools, ints, floats, lists (comma-separated), and passthrough strings.
        """
        if raw is None:
            return None
        s = str(raw).strip()
        if s == "":
            return None
        low = s.lower()

        # booleans
        if low in ("true", "t", "1", "yes", "y", "on"):
            return True
        if low in ("false", "f", "0", "no", "n", "off"):
            return False

        # numbers
        try:
            if "." in s or "e" in low:
                return float(s)
            return int(s)
        except Exception:
            pass

        # lists (comma-separated) for known list keys
        listy = {
            "classes", "annotated_classes", "class_metadata", "train_channels",
            "tables", "file_metadata"
        }
        if key in listy or ("," in s):
            # split, trim, coerce numbers if possible
            out = []
            for token in s.split(","):
                token = token.strip()
                if token == "":
                    continue
                try:
                    if "." in token or "e" in token.lower():
                        out.append(float(token))
                    else:
                        out.append(int(token))
                except Exception:
                    out.append(token)
            return out

        return s
    
    @staticmethod
    def convert_settings_dict_for_gui(settings):
        """
        Decide widget type per setting:
        - 'check'  => Checkbutton (bools)
        - 'combo'  => readonly Combobox (predefined choices)
        - 'entry'  => Entry (free text / numbers / lists)
        Returns: {key: (kind, options, initial)}
        """
        try:
            from torchvision import models as torch_models
            torchvision_models = sorted({name for name, obj in torch_models.__dict__.items() if callable(obj)})
        except Exception:
            torchvision_models = ['resnet18', 'resnet34', 'resnet50', 'densenet121', 'mobilenet_v2']

        chan_list = [
            '[0,1,2,3,4,5,6,7,8]',
            '[0,1,2,3,4,5,6,7]',
            '[0,1,2,3,4,5,6]',
            '[0,1,2,3,4,5]',
            '[0,1,2,3,4]',
            '[0,1,2,3]',
            '[0,1,2]',
            '[0,1]',
            '[0]',
            '[0,0]'
        ]

        variables = {}
        special_cases = {
            'metadata_type': ('combo', ['cellvoyager', 'cq1', 'auto', 'custom'], 'cellvoyager'),
            'channels': ('combo', chan_list, '[0,1,2,3]'),
            'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
            'channel_dims': ('combo', chan_list, '[0,1,2,3]'),

            # CHANGED: include 'measurement' instead of 'recruitment'
            'dataset_mode': ('combo', ['annotation', 'metadata', 'measurement'], 'metadata'),

            'cov_type': ('combo', ['HC0', 'HC1', 'HC2', 'HC3', None], None),
            'crop_mode': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']"], "['cell']"),
            'timelapse_mode': ('combo', ['trackpy', 'iou', 'btrack'], 'trackpy'),
            'train_mode': ('combo', ['erm', 'irm'], 'erm'),
            'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
            'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
            'model_name': ('combo', ['cyto', 'cyto_2', 'cyto_3', 'nuclei'], 'cyto'),
            'regression_type': ('combo', ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge'], 'ols'),
            'timelapse_objects': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']", None], None),
            'model_type': ('combo', torchvision_models, 'resnet50'),
            'optimizer_type': ('combo', ['adamw', 'adam'], 'adamw'),
            'schedule': ('combo', ['reduce_lr_on_plateau', 'step_lr'], 'reduce_lr_on_plateau'),
            'loss_type': ('combo', ['focal_loss', 'binary_cross_entropy_with_logits'], 'focal_loss'),
            'normalize_by': ('combo', ['fov', 'png'], 'png'),
            'agg_type': ('combo', ['mean', 'median'], 'mean'),
            'grouping': ('combo', ['mean', 'median'], 'mean'),
            'min_max': ('combo', ['allq', 'all'], 'allq'),
            'transform': ('combo', ['log', 'sqrt', 'square', None], None)
        }

        for key, value in settings.items():
            if key in special_cases:
                variables[key] = special_cases[key]
            elif isinstance(value, bool):
                variables[key] = ('check', None, value)
            elif isinstance(value, (int, float)):
                variables[key] = ('entry', None, value)
            elif isinstance(value, list):
                variables[key] = ('entry', None, str(value))
            else:  # str / None / other
                variables[key] = ('entry', None, "" if value is None else value)
                
        return variables

    def open_deep_spacr_window_v1(self):
        import tkinter as tk
        from tkinter import ttk, messagebox
        import sqlite3, threading, ast, os

        from spacr.settings import deep_spacr_defaults

        # --- defaults & base keys
        defaults = deep_spacr_defaults({})
        defaults['src'] = self.src or defaults.get('src')
        defaults['annotation_column'] = self.annotation_column or defaults.get('annotation_column')

        # SHOWN ONLY IN SIMPLE MODE
        minimal_keys = ["src", "dataset_mode", "model_type", "apply_model_to_dataset"]

        # --- window
        win = tk.Toplevel(self.root)
        win.title("Deep SPACR – Training Settings")
        style_out = set_dark_style(ttk.Style())
        win.configure(bg=style_out['bg_color'])
        win.geometry("980x620")

        outer = tk.Frame(win, bg=style_out['bg_color'])
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left column
        left_col = tk.Frame(outer, bg=style_out['bg_color'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))

        topbar = tk.Frame(left_col, bg=style_out['bg_color'])
        topbar.pack(fill=tk.X, pady=(0,6))
        advanced_var = tk.BooleanVar(value=False)
        adv_chk = tk.Checkbutton(
            topbar, text="Advanced settings", variable=advanced_var,
            bg=style_out['bg_color'], fg=self.fg_color, selectcolor=style_out['bg_color'],
            command=lambda: rebuild_form(advanced_var.get())
        )
        adv_chk.pack(anchor="w")

        form_frame = tk.LabelFrame(left_col, text="Model / Training Settings",
                                bg=style_out['bg_color'], fg=self.fg_color)
        form_frame.pack(fill=tk.BOTH, expand=True)

        # --- Right column: DB columns
        right = tk.LabelFrame(outer, text="Use DB Annotation Columns",
                            bg=style_out['bg_color'], fg=self.fg_color)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        use_db_var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(
            right, text="Use selected DB columns as classes", variable=use_db_var,
            bg=style_out['bg_color'], fg=self.fg_color, selectcolor=style_out['bg_color']
        )
        chk.pack(anchor="w", padx=6, pady=(6,2))

        lb = tk.Listbox(right, selectmode=tk.EXTENDED, height=18)
        lb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # populate listbox from sqlite schema (png_list)
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cur = conn.cursor()
                cur.execute('PRAGMA table_info("png_list")')
                cols = cur.fetchall()
                for _, name, coltype, *_ in cols:
                    nm = str(name)
                    if nm.lower() in ('png_path', 'prcfo'):
                        continue
                    if (coltype or '').upper().startswith('INT') or nm not in ('png_path',):
                        lb.insert(tk.END, nm)
        except Exception:
            pass

        # --- Buttons
        btns = tk.Frame(win, bg=style_out['bg_color'])
        btns.pack(fill=tk.X, padx=10, pady=(0,10))

        # --- spec & form state
        widget_specs = self.convert_settings_dict_for_gui(defaults)
        entries, var_bools = {}, {}

        def add_row(parent, row, key, spec, init_override=None):
            kind, options, initial = spec
            if init_override is not None:
                initial = init_override

            lbl = tk.Label(parent, text=key, bg=style_out['bg_color'], fg=self.fg_color)
            lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)

            if kind == 'check':
                v = tk.BooleanVar(value=bool(initial))
                w = tk.Checkbutton(parent, variable=v, bg=style_out['bg_color'],
                                fg=self.fg_color, selectcolor=style_out['bg_color'])
                w.grid(row=row, column=1, sticky="w", padx=4, pady=3)
                var_bools[key] = v
            elif kind == 'combo':
                w = ttk.Combobox(parent, values=[str(x) for x in (options or [])], state="readonly")
                init_val = initial if initial is not None else defaults.get(key, "")
                if init_val is None:
                    init_val = ""
                w.set(str(init_val))
                w.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
                parent.grid_columnconfigure(1, weight=1)
            else:
                w = tk.Entry(parent)
                w.insert(0, "" if initial is None else str(initial))
                w.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
                parent.grid_columnconfigure(1, weight=1)

            entries[key] = w

        def read_current_values():
            current = {}
            for k, w in entries.items():
                spec = widget_specs.get(k, ('entry', None, None))
                kind, _, _ = spec
                if kind == 'check':
                    current[k] = bool(var_bools[k].get())
                else:
                    current[k] = w.get()
            return current

        def rebuild_form(show_all):
            snapshot = read_current_values()

            for child in form_frame.winfo_children():
                child.destroy()
            entries.clear()
            var_bools.clear()

            keys_to_show = sorted(widget_specs.keys()) if show_all else [k for k in minimal_keys if k in widget_specs]

            r = 0
            for k in keys_to_show:
                spec = widget_specs[k]
                init_override = snapshot.get(k, None)
                add_row(form_frame, r, k, spec, init_override=init_override)
                r += 1

        # initial build (simple)
        rebuild_form(show_all=False)

        def _parse_gui_value(key, widget, spec):
            kind, options, _ = spec
            if kind == 'check':
                return bool(var_bools[key].get())

            raw = widget.get()
            s = raw.strip()
            if s == "":
                return None
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
            try:
                if any(ch in s for ch in ('.', 'e', 'E')):
                    return float(s)
                return int(s)
            except Exception:
                l = s.lower()
                if l in ('true', 'false'):
                    return l == 'true'
                return s

        def on_run():
            # collect form values
            settings = {}
            for k, widget in entries.items():
                spec = widget_specs.get(k, ('entry', None, None))
                val = _parse_gui_value(k, widget, spec)
                settings[k] = defaults.get(k) if val is None else val

            # ensure essentials
            if not settings.get("src"):
                settings["src"] = self.src
            if not settings.get("dataset"):
                settings["dataset"] = self.src

            # === CHANGED BLOCK: build explicit metadata_rules from selected columns & values ===
            if use_db_var.get():
                sel_cols = [lb.get(i) for i in lb.curselection()]
                if not sel_cols:
                    messagebox.showwarning("No columns selected", "Select at least one annotation column or uncheck the DB option.")
                    return

                rules = []
                try:
                    with sqlite3.connect(self.db_path, timeout=30) as conn:
                        cur = conn.cursor()
                        for col in sel_cols:
                            qcol = col.replace('"', '""')
                            cur.execute(f'SELECT DISTINCT "{qcol}" FROM "png_list" WHERE "{qcol}" IN (1,2,3,4,5,6,7,8,9)')
                            vals = sorted(int(r[0]) for r in cur.fetchall() if r[0] is not None)
                            for v in vals:
                                rules.append({
                                    "name": f"{col}_{v}",        # folder/class name
                                    "where": [{"column": col, "op": "==", "value": v}],
                                })
                except Exception as e:
                    messagebox.showerror("DB error", f"Failed to read classes from DB:\n{e}")
                    return

                if not rules:
                    messagebox.showwarning("No labels found", "No labels (1/2/…) found in the selected columns.")
                    return

                # configure settings for metadata mode using rules
                settings["dataset_mode"] = "metadata"
                settings["metadata_rules"] = rules

            win.destroy()

            def _worker():
                try:
                    self.update_gui_text("Deep SPACR: preparing…")
                    from spacr.deep_spacr import deep_spacr
                    deep_spacr(settings)
                    self.update_gui_text("Deep SPACR: done.")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.update_gui_text(f"Deep SPACR error: {e}")

            threading.Thread(target=_worker, daemon=True).start()

        run_btn = ttk.Button(btns, text="Run", command=on_run)
        cancel_btn = ttk.Button(btns, text="Cancel", command=win.destroy)
        run_btn.pack(side=tk.RIGHT, padx=5)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
    def build_multi_annotation(self, source_columns, target_column="multi_annot"):
        """
        Consolidate multiple {1,2,NULL} columns into a single integer code column.
        Unique per combination. If all inputs NULL -> store NULL.
        Sets self.annotation_column to target_column and refreshes UI.

        Encoding:
        per col: NULL->0, 1->1, 2->2
        code = 1 + sum( digit_i * (3^i) ), i = 0..N-1
        (all digits 0) => store NULL instead of 1
        """
        import sqlite3

        if not source_columns or not isinstance(source_columns, (list, tuple)):
            raise ValueError("build_multi_annotation: provide a non-empty list of source columns")

        # precompute multipliers 3^i in Python (SQLite lacks POWER())
        multipliers = [1]
        for _ in range(1, len(source_columns)):
            multipliers.append(multipliers[-1] * 3)

        # safe identifiers
        src_q = [f'"{c.replace(chr(34), chr(34)*2)}"' for c in source_columns]
        tgt_q = f'"{target_column.replace(chr(34), chr(34)*2)}"'

        # CASE to map each source to {0,1,2}
        digits = [f"(CASE {c} WHEN 1 THEN 1 WHEN 2 THEN 2 ELSE 0 END)" for c in src_q]

        # sum_i digit_i * 3^i
        weighted_sum = " + ".join(f"{digits[i]} * {multipliers[i]}" for i in range(len(digits))) or "0"

        # final value: NULL if all zero; else 1 + sum
        final_expr = f"CASE WHEN ({weighted_sum}) = 0 THEN NULL ELSE (1 + {weighted_sum}) END"

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()

            # ensure all source columns exist (as INTEGER, NULL) so SQL won't fail
            cur.execute('PRAGMA table_info("png_list")')
            have = {row[1] for row in cur.fetchall()}
            for col in source_columns:
                if col not in have:
                    cq = col.replace('"','""')
                    cur.execute(f'ALTER TABLE "png_list" ADD COLUMN "{cq}" INTEGER')

            # ensure target column exists
            if target_column not in have:
                tq = target_column.replace('"','""')
                cur.execute(f'ALTER TABLE "png_list" ADD COLUMN "{tq}" INTEGER')

            # compute in-place
            cur.execute(f'UPDATE "png_list" SET {tgt_q} = {final_expr};')
            conn.commit()

        # make it the working annotation column and refresh view
        self.annotation_column = target_column
        self._ensure_annotation_column()
        self.prefilter_paths_annotations()
        self.load_images()
        
    def ensure_multi_annot_from_selection(self, source_columns, target_column="class_column", force_rebuild=True):
        """
        If one column selected -> use it directly (no consolidation).
        If >=2 selected -> build a consolidated column. If target_column already exists,
        auto-bump to target_column_1, target_column_2, ... and use that actual name everywhere.

        Returns the effective annotation column name.
        """
        import sqlite3

        if not source_columns or not isinstance(source_columns, (list, tuple)):
            raise ValueError("ensure_multi_annot_from_selection: provide a non-empty list of source columns")

        # Single column => just use it directly
        if len(source_columns) == 1:
            self.annotation_column = source_columns[0]
            self._ensure_annotation_column()
            self.prefilter_paths_annotations()
            self.load_images()
            return self.annotation_column

        # Multi-column consolidation: pick a free target name (auto-bump)
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cur = conn.cursor()
            cur.execute('PRAGMA table_info("png_list")')
            existing = {row[1] for row in cur.fetchall()}

        base = (str(target_column).strip() or "class_column")
        effective = base
        suffix = 1
        while effective in existing:
            effective = f"{base}_{suffix}"
            suffix += 1

        # Build / refresh the consolidated column under 'effective'
        # (build_multi_annotation will set self.annotation_column and refresh UI)
        if force_rebuild or self.annotation_column != effective:
            self.build_multi_annotation(source_columns, target_column=effective)
        else:
            self.build_multi_annotation(source_columns, target_column=effective)

        # Ensure local state reflects the chosen column name
        self.annotation_column = effective
        self._ensure_annotation_column()
        self.prefilter_paths_annotations()
        self.load_images()
        return effective

    def open_deep_spacr_window(self):
        import tkinter as tk
        from tkinter import ttk, messagebox
        import sqlite3, threading, ast, json, os

        from spacr.settings import deep_spacr_defaults

        # ---- defaults ---------------------------------------------------------
        defaults = deep_spacr_defaults({})
        defaults['src'] = self.src or defaults.get('src')
        defaults['dataset'] = defaults.get('dataset', defaults['src'])
        defaults['annotation_column'] = self.annotation_column or defaults.get('annotation_column')

        # keep your app-wide style usage
        style_out = set_dark_style(ttk.Style())
        bg = self.bg_color
        fg = self.fg_color
        font = self.font_style

        # ---- window -----------------------------------------------------------
        win = tk.Toplevel(self.root)
        win.title("Deep SPACR — Train (Beta)")
        win.configure(bg=bg)
        win.geometry("1120x760")

        outer = tk.Frame(win, bg=bg)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- header master toggles (govern tab enablement) --------------------
        header = tk.Frame(outer, bg=bg)
        header.pack(fill=tk.X, pady=(0,8))

        gen_var   = tk.BooleanVar(value=bool(defaults.get('generate_training_dataset', True)))
        train_var = tk.BooleanVar(value=bool(defaults.get('train_DL_model', True)))
        apply_var = tk.BooleanVar(value=bool(defaults.get('apply_model_to_dataset', True)))

        def _chk(label, var):
            return tk.Checkbutton(
                header, text=label, variable=var,
                bg=bg, fg=fg, selectcolor=bg, font=font, activebackground=bg, activeforeground=fg
            )

        _chk("Generate training dataset", gen_var).pack(side=tk.LEFT, padx=(0,12))
        _chk("Train",                        train_var).pack(side=tk.LEFT, padx=(0,12))
        _chk("Apply model to dataset",       apply_var).pack(side=tk.LEFT, padx=(0,12))

        # ---- notebook ---------------------------------------------------------
        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True)

        # ---- helpers ----------------------------------------------------------
        def _label(parent, text):
            return tk.Label(parent, text=text, bg=bg, fg=fg, anchor='w', font=font)

        def _row(parent, r, label_text, widget):
            _label(parent, label_text).grid(row=r, column=0, sticky="w", padx=6, pady=4)
            widget.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
            parent.grid_columnconfigure(1, weight=1)

        def _parse_list_literal(s, fallback=None):
            if s is None or str(s).strip() == "":
                return fallback
            try:
                return ast.literal_eval(str(s))
            except Exception:
                return fallback

        def _parse_csv_list(s, fallback=None):
            if s is None or str(s).strip() == "":
                return fallback
            parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
            return parts if parts else fallback

        def _set_disabled_state(frame, disabled=True):
            # toggle "viability": enable/disable inputs inside the frame
            state = tk.DISABLED if disabled else tk.NORMAL
            for child in frame.winfo_children():
                try:
                    child.configure(state=state)
                except Exception:
                    pass

        # ======================================================================
        # TAB 1: Generate training dataset
        # ======================================================================
        tab_gen = tk.Frame(nb, bg=bg)
        nb.add(tab_gen, text="Generate training dataset")

        gen_split = tk.PanedWindow(tab_gen, orient=tk.HORIZONTAL, sashwidth=6, bg=bg)
        gen_split.pack(fill=tk.BOTH, expand=True)

        gen_form  = tk.Frame(gen_split, bg=bg)
        gen_right = tk.Frame(gen_split, bg=bg)
        gen_split.add(gen_form)
        gen_split.add(gen_right)

        # --- Left column (general) --------------------------------------------
        r = 0
        dataset_mode_cbx = ttk.Combobox(gen_form, values=['annotation','metadata','measurement'], state='readonly')
        dataset_mode_cbx.set(defaults.get('dataset_mode', 'annotation'))
        _row(gen_form, r, "dataset_mode", dataset_mode_cbx); r += 1

        size_sp = ttk.Spinbox(gen_form, from_=16, to=4096, increment=16)
        size_sp.set(int(defaults.get('size', 224)))
        _row(gen_form, r, "size (cropped PNG side)", size_sp); r += 1

        img_size_sp = ttk.Spinbox(gen_form, from_=16, to=4096, increment=16)
        img_size_sp.set(int(defaults.get('image_size', 224)))
        _row(gen_form, r, "image_size (model input)", img_size_sp); r += 1

        test_split_sp = ttk.Spinbox(gen_form, from_=0.0, to=0.9, increment=0.01)
        test_split_sp.set(float(defaults.get('test_split', 0.1)))
        _row(gen_form, r, "test_split", test_split_sp); r += 1

        sample_sp = ttk.Spinbox(gen_form, from_=0, to=10**9, increment=1)
        sample_val = defaults.get('sample', None)
        sample_sp.delete(0, tk.END)
        sample_sp.insert(0, "" if sample_val in (None, "") else str(sample_val))
        _row(gen_form, r, "sample (rows, optional)", sample_sp); r += 1

        # FILE TYPE: free text (any string)
        file_type_entry = tk.Entry(gen_form)
        file_type_entry.insert(0, str(defaults.get('file_type', defaults.get('png_type','cell_png'))))
        _row(gen_form, r, "file_type / png_type", file_type_entry); r += 1

        tables_entry = tk.Entry(gen_form)
        tables_entry.insert(0, "" if defaults.get('tables') in (None, []) else ",".join(defaults.get('tables')))
        _row(gen_form, r, "tables (csv)", tables_entry); r += 1

        file_metadata_entry = tk.Entry(gen_form)
        if defaults.get('file_metadata') not in (None, []):
            file_metadata_entry.insert(0, ",".join(defaults['file_metadata']) if isinstance(defaults['file_metadata'], list) else str(defaults['file_metadata']))
        _row(gen_form, r, "file_metadata (csv)", file_metadata_entry); r += 1

        metadata_type_by_cbx = ttk.Combobox(gen_form, state='readonly', values=['columnID','something_else'])
        metadata_type_by_cbx.set(defaults.get('metadata_type_by','columnID'))
        _row(gen_form, r, "metadata_type_by", metadata_type_by_cbx); r += 1

        class_metadata_entry = tk.Entry(gen_form)
        class_metadata_entry.insert(0, str(defaults.get('class_metadata', [['c1'],['c2']])))
        _row(gen_form, r, "class_metadata (list-of-lists)", class_metadata_entry); r += 1

        classes_entry = tk.Entry(gen_form)
        classes_entry.insert(0, str(defaults.get('classes', ['nc','pc'])))
        _row(gen_form, r, "classes (list)", classes_entry); r += 1

        annotated_classes_entry = tk.Entry(gen_form)
        annotated_classes_entry.insert(0, str(defaults.get('annotated_classes', [1,2])))
        _row(gen_form, r, "annotated_classes (list)", annotated_classes_entry); r += 1

        ch_interest_sp = ttk.Spinbox(gen_form, from_=1, to=5, increment=1)
        ch_interest_sp.set(int(defaults.get('channel_of_interest', 3)))
        _row(gen_form, r, "channel_of_interest", ch_interest_sp); r += 1

        custom_measurement_entry = tk.Entry(gen_form)
        if defaults.get('custom_measurement'):
            custom_measurement_entry.insert(0, str(defaults['custom_measurement']))
        _row(gen_form, r, "custom_measurement (optional)", custom_measurement_entry); r += 1

        balance_var = tk.BooleanVar(value=bool(defaults.get('balance_to_smallest', True)))
        balance_chk = tk.Checkbutton(gen_form, text="Balance classes to smallest",
                                    variable=balance_var, bg=bg, fg=fg, selectcolor=bg, font=font,
                                    activebackground=bg, activeforeground=fg)
        _row(gen_form, r, "", balance_chk); r += 1

        # --- Right column: three MODE-SPECIFIC panels -------------------------
        # 1) Annotation panel
        ann_frame = tk.LabelFrame(gen_right, text="Annotation columns", bg=bg, fg=fg, font=font, labelanchor='n')
        ann_inner = tk.Frame(ann_frame, bg=bg)
        _label(ann_inner, "Use DB Annotation Columns").grid(row=0, column=0, sticky="w", padx=6, pady=(8,2))
        use_db_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ann_inner, text="Use selected DB columns as classes",
                    variable=use_db_var, bg=bg, fg=fg, selectcolor=bg, font=font,
                    activebackground=bg, activeforeground=fg).grid(row=1, column=0, sticky="w", padx=6, pady=(0,6))
        lb = tk.Listbox(ann_inner, selectmode=tk.EXTENDED, height=10,
                        bg=self.inactive_color, fg=fg, highlightbackground=fg, selectbackground=self.active_color)
        lb.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0,8))
        ann_inner.grid_columnconfigure(0, weight=1)
        ann_inner.grid_rowconfigure(2, weight=1)

        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cur = conn.cursor()
                cur.execute('PRAGMA table_info("png_list")')
                for _, name, coltype, *_ in cur.fetchall():
                    nm = str(name)
                    if nm.lower() in ('png_path', 'prcfo'):
                        continue
                    if (coltype or '').upper().startswith('INT') or nm not in ('png_path',):
                        lb.insert(tk.END, nm)
        except Exception:
            pass

        ann_inner.pack(fill=tk.BOTH, expand=True)

        # 2) Metadata panel
        meta_grp = tk.LabelFrame(gen_right, text="Metadata rules (JSON)", bg=bg, fg=fg, font=font, labelanchor='n')
        meta_inner = tk.Frame(meta_grp, bg=bg)
        meta_rules_entry = tk.Entry(meta_inner)
        meta_rules_entry.pack(fill=tk.X, padx=6, pady=(6,4))
        ex = tk.Frame(meta_inner, bg=bg)
        tk.Label(
            ex,
            text=("Example:\n"
                "[\n"
                "  {\"name\":\"test_1\",     \"where\":[{\"column\":\"test\",\"op\":\"==\",\"value\":1}]},\n"
                "  {\"name\":\"test_2\",     \"where\":[{\"column\":\"test\",\"op\":\"==\",\"value\":2}]},\n"
                "  {\"name\":\"parasite_1\", \"where\":[{\"column\":\"parasite\",\"op\":\"==\",\"value\":1}]}\n"
                "]"),
            justify='left', anchor='w', bg=bg, fg=fg
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        def _insert_meta_example():
            meta_rules_entry.delete(0, tk.END)
            meta_rules_entry.insert(
                0,
                '[{"name":"test_1","where":[{"column":"test","op":"==","value":1}]},'
                ' {"name":"test_2","where":[{"column":"test","op":"==","value":2}]},'
                ' {"name":"parasite_1","where":[{"column":"parasite","op":"==","value":1}]}]'
            )
        ttk.Button(ex, text="Insert example", command=_insert_meta_example).pack(side=tk.RIGHT, padx=6)
        ex.pack(fill=tk.X, padx=6, pady=(0,6))
        meta_inner.pack(fill=tk.BOTH, expand=True)

        # 3) Measurement panel
        meas_grp = tk.LabelFrame(gen_right, text="Measurement selection", bg=bg, fg=fg, font=font, labelanchor='n')
        meas_inner = tk.Frame(meas_grp, bg=bg)
        _label(meas_inner, "measurement (csv: columns)").grid(row=0, column=0, sticky="w", padx=6, pady=(8,2))
        meas_cols_entry = tk.Entry(meas_inner)
        meas_cols_entry.insert(0, "" if defaults.get('measurement') in (None, []) else (
            ",".join(defaults['measurement']) if isinstance(defaults['measurement'], list) else str(defaults['measurement'])
        ))
        meas_cols_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=(8,2))

        _label(meas_inner, "threshold (float or q1..q9)").grid(row=1, column=0, sticky="w", padx=6, pady=(4,2))
        threshold_entry = tk.Entry(meas_inner)
        threshold_entry.insert(0, str(defaults.get('threshold', 'q8')))
        threshold_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(4,2))
        tk.Label(meas_inner, text="Examples: 0.42  or  q7", bg=bg, fg=fg, font=font)\
            .grid(row=2, column=1, sticky="w", padx=6, pady=(0,6))
        meas_inner.grid_columnconfigure(1, weight=1)
        meas_inner.pack(fill=tk.BOTH, expand=True)

        # start with correct panel visible & viable
        def _toggle_gen_right(*_):
            mode = dataset_mode_cbx.get().strip().lower()

            # hide all
            for w in (ann_frame, meta_grp, meas_grp):
                w.pack_forget()
                _set_disabled_state(w, disabled=True)

            # show + enable chosen
            if mode == 'annotation':
                ann_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,8))
                _set_disabled_state(ann_frame, disabled=False)
            elif mode == 'metadata':
                meta_grp.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,8))
                _set_disabled_state(meta_grp, disabled=False)
            elif mode == 'measurement':
                meas_grp.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,8))
                _set_disabled_state(meas_grp, disabled=False)

        dataset_mode_cbx.bind("<<ComboboxSelected>>", _toggle_gen_right)
        _toggle_gen_right()

        # ======================================================================
        # TAB 2: Train
        # ======================================================================
        tab_train = tk.Frame(nb, bg=bg)
        nb.add(tab_train, text="Train")

        tr_basic = tk.LabelFrame(tab_train, text="Basic", bg=bg, fg=fg)
        tr_basic.pack(fill=tk.X, padx=8, pady=(8,6))
        rr = 0

        try:
            import torchvision
            model_names = sorted({n for n, o in getattr(torchvision.models, '__dict__', {}).items() if callable(o)})
        except Exception:
            model_names = ['resnet18','resnet34','resnet50','densenet121','mobilenet_v2']

        model_cbx = ttk.Combobox(tr_basic, state='readonly', values=model_names)
        model_cbx.set(defaults.get('model_type', 'resnet50'))
        _row(tr_basic, rr, "model_type", model_cbx); rr += 1

        epochs_sp = ttk.Spinbox(tr_basic, from_=1, to=2000, increment=1)
        epochs_sp.set(int(defaults.get('epochs', 100)))
        _row(tr_basic, rr, "epochs", epochs_sp); rr += 1

        bs_sp = ttk.Spinbox(tr_basic, from_=1, to=4096, increment=1)
        bs_sp.set(int(defaults.get('batch_size', 64)))
        _row(tr_basic, rr, "batch_size", bs_sp); rr += 1

        lr_sp = ttk.Spinbox(tr_basic, from_=1e-6, to=1e-1, increment=1e-6)
        lr_sp.set(float(defaults.get('learning_rate', 1e-3)))
        _row(tr_basic, rr, "learning_rate", lr_sp); rr += 1

        val_split_sp = ttk.Spinbox(tr_basic, from_=0.0, to=0.9, increment=0.01)
        val_split_sp.set(float(defaults.get('val_split', 0.1)))
        _row(tr_basic, rr, "val_split", val_split_sp); rr += 1

        loss_cbx = ttk.Combobox(tr_basic, state='readonly',
                                values=['focal_loss', 'binary_cross_entropy_with_logits', 'auto'])
        loss_cbx.set(defaults.get('loss_type', 'focal_loss'))
        _row(tr_basic, rr, "loss_type", loss_cbx); rr += 1

        train_channels_cbx = ttk.Combobox(tr_basic, state='readonly',
                                        values=["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"])
        tdef = defaults.get('train_channels', ['r','g','b'])
        train_channels_cbx.set(str(tdef if isinstance(tdef, list) else "['r','g','b']"))
        _row(tr_basic, rr, "train_channels", train_channels_cbx); rr += 1

        do_train_var = tk.BooleanVar(value=bool(defaults.get('train', True)))
        do_test_var  = tk.BooleanVar(value=bool(defaults.get('test', False)))
        _row(tr_basic, rr, "", tk.Checkbutton(tr_basic, text="train (legacy flag)",
                                            variable=do_train_var, bg=bg, fg=fg, selectcolor=bg, font=font)); rr += 1
        _row(tr_basic, rr, "", tk.Checkbutton(tr_basic, text="test after training (legacy flag)",
                                            variable=do_test_var, bg=bg, fg=fg, selectcolor=bg, font=font)); rr += 1

        adv = tk.LabelFrame(tab_train, text="Advanced", bg=bg, fg=fg)
        adv.pack(fill=tk.X, padx=8, pady=(0,8))
        ra = 0

        opt_cbx = ttk.Combobox(adv, state='readonly', values=['adamw','adagrad','adam'])
        opt_cbx.set(defaults.get('optimizer_type', 'adamw'))
        _row(adv, ra, "optimizer_type", opt_cbx); ra += 1

        sched_cbx = ttk.Combobox(adv, state='readonly', values=['reduce_lr_on_plateau','step_lr'])
        sched_cbx.set(defaults.get('schedule', 'reduce_lr_on_plateau'))
        _row(adv, ra, "schedule", sched_cbx); ra += 1

        wd_sp = ttk.Spinbox(adv, from_=0.0, to=1.0, increment=1e-6)
        wd_sp.set(float(defaults.get('weight_decay', 1e-5)))
        _row(adv, ra, "weight_decay", wd_sp); ra += 1

        dr_sp = ttk.Spinbox(adv, from_=0.0, to=0.9, increment=0.01)
        dr_sp.set(float(defaults.get('dropout_rate', 0.1)))
        _row(adv, ra, "dropout_rate", dr_sp); ra += 1

        init_w_var = tk.BooleanVar(value=bool(defaults.get('init_weights', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="init_weights",
                                        variable=init_w_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        use_ckpt_var = tk.BooleanVar(value=bool(defaults.get('use_checkpoint', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="use_checkpoint (activation checkpointing)",
                                        variable=use_ckpt_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        amsgrad_var = tk.BooleanVar(value=bool(defaults.get('amsgrad', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="AMSGrad",
                                        variable=amsgrad_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        intermed_var = tk.BooleanVar(value=bool(defaults.get('intermedeate_save', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="intermedeate_save",
                                        variable=intermed_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        jobs_sp = ttk.Spinbox(adv, from_=0, to=max(1, os.cpu_count() or 64), increment=1)
        jobs_sp.set(int(defaults.get('n_jobs', max(1, (os.cpu_count() or 8)-4))))
        _row(adv, ra, "n_jobs (DataLoader workers)", jobs_sp); ra += 1

        pin_var = tk.BooleanVar(value=bool(defaults.get('pin_memory', False)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="pin_memory",
                                        variable=pin_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        ga_sp = ttk.Spinbox(adv, from_=1, to=64, increment=1)
        ga_sp.set(int(defaults.get('gradient_accumulation_steps', 4)))
        _row(adv, ra, "gradient_accumulation_steps", ga_sp); ra += 1

        grad_acc_var = tk.BooleanVar(value=bool(defaults.get('gradient_accumulation', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="gradient_accumulation",
                                        variable=grad_acc_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        augment_var = tk.BooleanVar(value=bool(defaults.get('augment', False)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="augment",
                                        variable=augment_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        normalize_var = tk.BooleanVar(value=bool(defaults.get('normalize', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="normalize",
                                        variable=normalize_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        verbose_var = tk.BooleanVar(value=bool(defaults.get('verbose', True)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="verbose",
                                        variable=verbose_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1

        custom_model_var = tk.BooleanVar(value=bool(defaults.get('custom_model', False)))
        _row(adv, ra, "", tk.Checkbutton(adv, text="custom_model",
                                        variable=custom_model_var, bg=bg, fg=fg, selectcolor=bg, font=font)); ra += 1
        custom_model_entry = tk.Entry(adv)
        custom_model_entry.insert(0, str(defaults.get('custom_model_path','path')))
        _row(adv, ra, "custom_model_path", custom_model_entry); ra += 1

        # ======================================================================
        # TAB 3: Apply model to dataset
        # ======================================================================
        tab_apply = tk.Frame(nb, bg=bg)
        nb.add(tab_apply, text="Apply model")

        apply_frame = tk.LabelFrame(tab_apply, text="Inference", bg=bg, fg=fg)
        apply_frame.pack(fill=tk.X, padx=8, pady=8)

        rr2 = 0
        score_sp = ttk.Spinbox(apply_frame, from_=0.0, to=1.0, increment=0.01)
        score_sp.set(float(defaults.get('score_threshold', 0.5)))
        _row(apply_frame, rr2, "score_threshold", score_sp); rr2 += 1

        dataset_entry = tk.Entry(apply_frame)
        dataset_entry.insert(0, str(defaults.get('dataset', defaults['src'])))
        _row(apply_frame, rr2, "dataset (apply on this path)", dataset_entry); rr2 += 1

        model_path_entry = tk.Entry(apply_frame)
        model_path_entry.insert(0, str(defaults.get('model_path','path')))
        _row(apply_frame, rr2, "model_path (optional override)", model_path_entry); rr2 += 1

        # ---- enable/disable tabs based on header toggles ----------------------
        def _apply_tab_state(*_):
            nb.tab(0, state='normal' if gen_var.get() else 'disabled')
            nb.tab(1, state='normal' if train_var.get() else 'disabled')
            nb.tab(2, state='normal' if apply_var.get() else 'disabled')

        for var in (gen_var, train_var, apply_var):
            var.trace_add("write", _apply_tab_state)
        _apply_tab_state()

        # ---- bottom buttons ---------------------------------------------------
        btns = tk.Frame(win, bg=bg)
        btns.pack(fill=tk.X, padx=10, pady=(0,10))
        run_btn = ttk.Button(btns, text="Run")
        cancel_btn = ttk.Button(btns, text="Cancel", command=win.destroy)
        run_btn.pack(side=tk.RIGHT, padx=5)
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        # ---- run handler ------------------------------------------------------
        def on_run():
            settings = dict(defaults)  # copy

            # Master toggles
            settings['generate_training_dataset'] = bool(gen_var.get())
            settings['train_DL_model'] = bool(train_var.get())
            settings['apply_model_to_dataset'] = bool(apply_var.get())

            # GENERATE / DATASET (shared)
            mode = dataset_mode_cbx.get().strip()
            settings['dataset_mode'] = mode
            settings['size'] = int(float(size_sp.get()))
            settings['image_size'] = int(float(img_size_sp.get()))
            settings['test_split'] = float(test_split_sp.get())
            settings['sample'] = None if str(sample_sp.get()).strip() == "" else int(float(sample_sp.get()))
            ft = file_type_entry.get().strip()
            settings['file_type'] = ft
            settings['png_type'] = ft
            settings['tables'] = _parse_csv_list(tables_entry.get(), None)
            settings['file_metadata'] = _parse_csv_list(file_metadata_entry.get(), None)
            settings['metadata_type_by'] = metadata_type_by_cbx.get().strip()
            settings['class_metadata'] = _parse_list_literal(class_metadata_entry.get(), defaults.get('class_metadata'))
            settings['classes'] = _parse_list_literal(classes_entry.get(), defaults.get('classes'))
            settings['annotated_classes'] = _parse_list_literal(annotated_classes_entry.get(), defaults.get('annotated_classes'))
            settings['channel_of_interest'] = int(float(ch_interest_sp.get()))
            cm = custom_measurement_entry.get().strip()
            settings['custom_measurement'] = (cm if cm != "" else None)
            settings['balance_to_smallest'] = bool(balance_var.get())

            # MODE-SPECIFIC
            if mode == 'annotation':
                settings['use_db_columns'] = bool(use_db_var.get())
                if settings['use_db_columns']:
                    sel_cols = [lb.get(i) for i in lb.curselection()]
                    if not sel_cols:
                        messagebox.showwarning("No DB columns selected", "Select at least one annotation column or uncheck the DB option.")
                        return

                    # Build/choose effective consolidated column name.
                    # Base name is "class_column"; if it exists, you'll get class_column_1, _2, ...
                    base_name = "class_column" if len(sel_cols) > 1 else sel_cols[0]
                    effective_col = self.ensure_multi_annot_from_selection(
                        sel_cols, target_column=base_name, force_rebuild=True
                    )
                    settings['annotation_column'] = effective_col
                else:
                    settings['annotation_column'] = self.annotation_column

                # Remove non-annotation keys
                settings.pop('metadata_rules', None)
                settings.pop('measurement', None)
                settings.pop('threshold', None)

            elif mode == 'metadata':
                raw = meta_rules_entry.get().strip()
                rules = None
                if raw:
                    try:
                        rules = json.loads(raw)
                    except Exception:
                        rules = _parse_list_literal(raw, None)
                if not rules:
                    messagebox.showwarning("Metadata rules", "Provide valid JSON rules or click 'Insert example'.")
                    return
                settings['metadata_rules'] = rules
                settings.pop('measurement', None)
                settings.pop('threshold', None)
                settings.pop('annotation_column', None)
                settings.pop('db_annotation_columns', None)
                settings.pop('use_db_columns', None)

            elif mode == 'measurement':
                meas_cols = _parse_csv_list(meas_cols_entry.get(), None)
                if not meas_cols:
                    messagebox.showwarning("Measurement", "Provide at least one measurement column (csv).")
                    return
                settings['measurement'] = meas_cols if len(meas_cols) > 1 else meas_cols[0]
                th_raw = threshold_entry.get().strip()
                if th_raw == "":
                    messagebox.showwarning("Measurement", "Provide a threshold (number) or a quantile code q1..q9.")
                    return
                try:
                    settings['threshold'] = float(th_raw)
                except Exception:
                    settings['threshold'] = th_raw  # e.g. "q8"
                settings.pop('metadata_rules', None)
                settings.pop('annotation_column', None)
                settings.pop('db_annotation_columns', None)
                settings.pop('use_db_columns', None)

            # TRAIN
            settings['model_type'] = model_cbx.get().strip()
            settings['epochs'] = int(float(epochs_sp.get()))
            settings['batch_size'] = int(float(bs_sp.get()))
            settings['learning_rate'] = float(lr_sp.get())
            settings['val_split'] = float(val_split_sp.get())
            settings['loss_type'] = loss_cbx.get().strip()
            settings['train_channels'] = _parse_list_literal(train_channels_cbx.get(), ['r','g','b'])

            settings['train'] = bool(do_train_var.get())   # legacy flag
            settings['test']  = bool(do_test_var.get())    # legacy flag

            settings['optimizer_type'] = opt_cbx.get().strip()
            settings['schedule'] = sched_cbx.get().strip()
            settings['weight_decay'] = float(wd_sp.get())
            settings['dropout_rate'] = float(dr_sp.get())
            settings['init_weights'] = bool(init_w_var.get())
            settings['use_checkpoint'] = bool(use_ckpt_var.get())
            settings['amsgrad'] = bool(amsgrad_var.get())
            settings['intermedeate_save'] = bool(intermed_var.get())
            settings['n_jobs'] = int(float(jobs_sp.get()))
            settings['pin_memory'] = bool(pin_var.get())
            settings['gradient_accumulation_steps'] = int(float(ga_sp.get()))
            settings['gradient_accumulation'] = bool(grad_acc_var.get())
            settings['augment'] = bool(augment_var.get())
            settings['normalize'] = bool(normalize_var.get())
            settings['verbose'] = bool(verbose_var.get())
            settings['custom_model'] = bool(custom_model_var.get())
            settings['custom_model_path'] = custom_model_entry.get().strip() or settings.get('custom_model_path')

            # APPLY
            settings['score_threshold'] = float(score_sp.get())
            settings['dataset'] = dataset_entry.get().strip() or self.src
            mp = model_path_entry.get().strip()
            if mp:
                settings['model_path'] = mp

            # Essentials
            settings['src'] = self.src

            win.destroy()

            def _worker():
                try:
                    self.update_gui_text("Deep SPACR: preparing…")
                    from spacr.deep_spacr import deep_spacr
                    deep_spacr(settings)
                    self.update_gui_text("Deep SPACR: done.")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.update_gui_text(f"Deep SPACR error: {e}")

            threading.Thread(target=_worker, daemon=True).start()

        run_btn.configure(command=on_run)

def standardize_figure(fig):
    from .gui_elements import set_dark_style
    from matplotlib.font_manager import FontProperties

    style_out = set_dark_style(ttk.Style())
    bg_color = style_out['bg_color']
    fg_color = style_out['fg_color']
    font_size = style_out['font_size']
    font_loader = style_out['font_loader']

    # Get the custom font path from the font loader
    font_path = font_loader.font_path
    font_prop = FontProperties(fname=font_path, size=font_size)

    """
    Standardizes the appearance of the figure:
    - Font size: from style
    - Font color: from style
    - Font family: custom OpenSans from font_loader
    - Removes top and right spines
    - Figure and subplot background: from style
    - Line width: 1
    - Line color: from style
    """
    

    for ax in fig.get_axes():
        # Set font properties for title and labels
        ax.title.set_fontsize(font_size)
        ax.title.set_color(fg_color)
        ax.title.set_fontproperties(font_prop)

        ax.xaxis.label.set_fontsize(font_size)
        ax.xaxis.label.set_color(fg_color)
        ax.xaxis.label.set_fontproperties(font_prop)

        ax.yaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_color(fg_color)
        ax.yaxis.label.set_fontproperties(font_prop)

        # Set font properties for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(font_size)
            label.set_color(fg_color)
            label.set_fontproperties(font_prop)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # Set spine line width and color
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_edgecolor(fg_color)

        # Set line width and color
        for line in ax.get_lines():
            line.set_linewidth(1)
            line.set_color(fg_color)

        # Set subplot background color
        ax.set_facecolor(bg_color)

        # Adjust the grid if needed
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Set figure background color
    fig.patch.set_facecolor(bg_color)

    fig.canvas.draw_idle()

    return fig

def modify_figure_properties(fig, scale_x=None, scale_y=None, line_width=None, font_size=None, x_lim=None, y_lim=None, grid=False, legend=None, title=None, x_label_rotation=None, remove_axes=False, bg_color=None, text_color=None, line_color=None):
    """
    Modifies the properties of the figure, including scaling, line widths, font sizes, axis limits, x-axis label rotation, background color, text color, line color, and other common options.

    Parameters:
    - fig: The Matplotlib figure object to modify.
    - scale_x: Scaling factor for the width of subplots (optional).
    - scale_y: Scaling factor for the height of subplots (optional).
    - line_width: Desired line width for all lines (optional).
    - font_size: Desired font size for all text (optional).
    - x_lim: Tuple specifying the x-axis limits (min, max) (optional).
    - y_lim: Tuple specifying the y-axis limits (min, max) (optional).
    - grid: Boolean to add grid lines to the plot (optional).
    - legend: Boolean to show/hide the legend (optional).
    - title: String to set as the title of the plot (optional).
    - x_label_rotation: Angle to rotate the x-axis labels (optional).
    - remove_axes: Boolean to remove or show the axes labels (optional).
    - bg_color: Color for the figure and subplot background (optional).
    - text_color: Color for all text in the figure (optional).
    - line_color: Color for all lines in the figure (optional).
    """
    if fig is None:
        print("Error: The figure provided is None.")
        return

    for ax in fig.get_axes():
        # Rescale subplots if scaling factors are provided
        if scale_x is not None or scale_y is not None:
            bbox = ax.get_position()
            width = bbox.width * (scale_x if scale_x else 1)
            height = bbox.height * (scale_y if scale_y else 1)
            new_bbox = [bbox.x0, bbox.y0, width, height]
            ax.set_position(new_bbox)

        # Set axis limits if provided
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        # Set grid visibility only
        ax.grid(grid)

        # Adjust line width and color if specified
        if line_width is not None or line_color is not None:
            for line in ax.get_lines():
                if line_width is not None:
                    line.set_linewidth(line_width)
                if line_color is not None:
                    line.set_color(line_color)
            for spine in ax.spines.values():  # Modify width and color of spines (e.g., scale bars)
                if line_width is not None:
                    spine.set_linewidth(line_width)
                if line_color is not None:
                    spine.set_edgecolor(line_color)
            ax.tick_params(width=line_width, colors=text_color if text_color else 'black')

        # Adjust font size if specified
        if font_size is not None:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(font_size)
            ax.title.set_fontsize(font_size)
            ax.xaxis.label.set_fontsize(font_size)
            ax.yaxis.label.set_fontsize(font_size)
            if ax.legend_:
                for text in ax.legend_.get_texts():
                    text.set_fontsize(font_size)

        # Rotate x-axis labels if rotation is specified
        if x_label_rotation is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(x_label_rotation)
                if 0 <= x_label_rotation <= 90:
                    label.set_ha('center')

        # Toggle axes labels visibility without affecting the grid or spines
        if remove_axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)

        # Set text color if specified
        if text_color:
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color(text_color)

        # Set background color for subplots if specified
        if bg_color:
            ax.set_facecolor(bg_color)

    # Set figure background color if specified
    if bg_color:
        fig.patch.set_facecolor(bg_color)

    fig.canvas.draw_idle()

def save_figure_as_format(fig, file_format):
    file_path = filedialog.asksaveasfilename(defaultextension=f".{file_format}", filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")])
    if file_path:
        try:
            fig.savefig(file_path, format=file_format)
            print(f"Figure saved as {file_format.upper()} at {file_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

def modify_figure(fig):
    from .gui_core import display_figure
    def apply_modifications():
        try:
            # Only apply changes if the fields are filled
            scale_x = float(scale_x_var.get()) if scale_x_var.get() else None
            scale_y = float(scale_y_var.get()) if scale_y_var.get() else None
            line_width = float(line_width_var.get()) if line_width_var.get() else None
            font_size = int(font_size_var.get()) if font_size_var.get() else None
            x_lim = eval(x_lim_var.get()) if x_lim_var.get() else None
            y_lim = eval(y_lim_var.get()) if y_lim_var.get() else None
            title = title_var.get() if title_var.get() else None
            bg_color = bg_color_var.get() if bg_color_var.get() else None
            text_color = text_color_var.get() if text_color_var.get() else None
            line_color = line_color_var.get() if line_color_var.get() else None
            x_label_rotation = int(x_label_rotation_var.get()) if x_label_rotation_var.get() else None

            modify_figure_properties(
                fig,
                scale_x=scale_x,
                scale_y=scale_y,
                line_width=line_width,
                font_size=font_size,
                x_lim=x_lim,
                y_lim=y_lim,
                grid=grid_var.get(),
                legend=legend_var.get(),
                title=title,
                x_label_rotation=x_label_rotation,
                remove_axes=remove_axes_var.get(),
                bg_color=bg_color,
                text_color=text_color,
                line_color=line_color
            )
            display_figure(fig)
        except ValueError:
            print("Invalid input; please enter numeric values.")

    def toggle_spleens():
        for ax in fig.get_axes():
            if spleens_var.get():
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['top'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
            else:
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
            display_figure(fig)

    # Create a new window for user input
    modify_window = tk.Toplevel()
    modify_window.title("Modify Figure Properties")

    # Apply dark style to the popup window
    style = ttk.Style()
    style.configure("TCheckbutton", background="#2E2E2E", foreground="white", selectcolor="blue")

    modify_window.configure(bg="#2E2E2E")

    # Create and style the input fields
    scale_x_var = tk.StringVar()
    scale_y_var = tk.StringVar()
    line_width_var = tk.StringVar()
    font_size_var = tk.StringVar()
    x_lim_var = tk.StringVar()
    y_lim_var = tk.StringVar()
    title_var = tk.StringVar()
    bg_color_var = tk.StringVar()
    text_color_var = tk.StringVar()
    line_color_var = tk.StringVar()
    x_label_rotation_var = tk.StringVar()
    remove_axes_var = tk.BooleanVar()
    grid_var = tk.BooleanVar()
    legend_var = tk.BooleanVar()
    spleens_var = tk.BooleanVar()

    options = [
        ("Rescale X:", scale_x_var),
        ("Rescale Y:", scale_y_var),
        ("Line Width:", line_width_var),
        ("Font Size:", font_size_var),
        ("X Axis Limits (tuple):", x_lim_var),
        ("Y Axis Limits (tuple):", y_lim_var),
        ("Title:", title_var),
        ("X Label Rotation (degrees):", x_label_rotation_var),
        ("Background Color:", bg_color_var),
        ("Text Color:", text_color_var),
        ("Line Color:", line_color_var)
    ]

    for i, (label_text, var) in enumerate(options):
        tk.Label(modify_window, text=label_text, bg="#2E2E2E", fg="white").grid(row=i, column=0, padx=10, pady=5)
        tk.Entry(modify_window, textvariable=var, bg="#2E2E2E", fg="white").grid(row=i, column=1, padx=10, pady=5)

    checkboxes = [
        ("Grid", grid_var),
        ("Legend", legend_var),
        ("Spleens", spleens_var),
        ("Remove Axes", remove_axes_var)
    ]

    for i, (label_text, var) in enumerate(checkboxes, start=len(options)):
        ttk.Checkbutton(modify_window, text=label_text, variable=var, style="TCheckbutton").grid(row=i, column=0, padx=10, pady=5, columnspan=2, sticky='w')

    spleens_var.trace_add("write", lambda *args: toggle_spleens())

    # Apply button
    apply_button = tk.Button(modify_window, text="Apply", command=apply_modifications, bg="#2E2E2E", fg="white")
    apply_button.grid(row=len(options) + len(checkboxes), column=0, columnspan=2, pady=10)

def generate_dna_matrix(output_path='dna_matrix.gif', canvas_width=1500, canvas_height=1000, duration=30, fps=20, base_size=20, transition_frames=30, font_type='arial.ttf', enhance=[1.1, 1.5, 1.2, 1.5], lowercase_prob=0.3):
    """
    Generate a DNA matrix animation and save it as GIF, MP4, or AVI using OpenCV for videos.
    """

    def save_output(frames, output_path, fps, output_format):
        """Save the animation based on output format."""
        if output_format in ['.mp4', '.avi']:
            images = [np.array(img.convert('RGB')) for img in frames]
            fourcc = cv2.VideoWriter_fourcc(*('mp4v' if output_format == '.mp4' else 'XVID'))
            out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
            for img in images:
                out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            out.release()
        elif output_format == '.gif':
            frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)

    def draw_base(draw, col_idx, base_position, base, font, alpha=255, fill_color=None):
        """Draws a DNA base at the specified position."""
        draw.text((col_idx * base_size, base_position * base_size), base, fill=(*fill_color, alpha), font=font)

    # Setup variables
    num_frames = duration * fps
    num_columns = canvas_width // base_size
    bases = ['A', 'T', 'C', 'G']
    active_color = (155, 55, 155)
    color = (255, 255, 255)
    base_colors = {'A': color, 'T': color, 'C': color, 'G': color}

    _, output_format = os.path.splitext(output_path)
    
    # Initialize font
    try:
        font = ImageFont.truetype(font_type, base_size)
    except IOError:
        font = ImageFont.load_default()

    # DNA string and positions
    string_lengths = [random.randint(10, 100) for _ in range(num_columns)]
    visible_bases = [0] * num_columns
    base_positions = [random.randint(-canvas_height // base_size, 0) for _ in range(num_columns)]
    column_strings = [[''] * 100 for _ in range(num_columns)]
    random_white_sequences = [None] * num_columns

    frames = []
    end_frame_start = int(num_frames * 0.8)

    for frame_idx in range(num_frames):
        img = Image.new('RGBA', (canvas_width, canvas_height), color=(0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        for col_idx in range(num_columns):
            if base_positions[col_idx] >= canvas_height // base_size and frame_idx < end_frame_start:
                string_lengths[col_idx] = random.randint(10, 100)
                base_positions[col_idx] = -string_lengths[col_idx]
                visible_bases[col_idx] = 0
                # Randomly choose whether to make each base lowercase
                column_strings[col_idx] = [
                    random.choice([base.lower(), base]) if random.random() < lowercase_prob else base
                    for base in [random.choice(bases) for _ in range(string_lengths[col_idx])]
                ]
                if string_lengths[col_idx] > 8:
                    random_start = random.randint(0, string_lengths[col_idx] - 8)
                    random_white_sequences[col_idx] = range(random_start, random_start + 8)

            last_10_percent_start = max(0, int(string_lengths[col_idx] * 0.9))
            
            for row_idx in range(min(visible_bases[col_idx], string_lengths[col_idx])):
                base_position = base_positions[col_idx] + row_idx
                if 0 <= base_position * base_size < canvas_height:
                    base = column_strings[col_idx][row_idx]
                    if base:
                        if row_idx == visible_bases[col_idx] - 1:
                            draw_base(draw, col_idx, base_position, base, font, fill_color=active_color)
                        elif row_idx >= last_10_percent_start:
                            alpha = 255 - int(((row_idx - last_10_percent_start) / (string_lengths[col_idx] - last_10_percent_start)) * 127)
                            draw_base(draw, col_idx, base_position, base, font, alpha=alpha, fill_color=base_colors[base.upper()])
                        elif random_white_sequences[col_idx] and row_idx in random_white_sequences[col_idx]:
                            draw_base(draw, col_idx, base_position, base, font, fill_color=active_color)
                        else:
                            draw_base(draw, col_idx, base_position, base, font, fill_color=base_colors[base.upper()])

            if visible_bases[col_idx] < string_lengths[col_idx]:
                visible_bases[col_idx] += 1
            base_positions[col_idx] += 2

        # Convert the image to numpy array to check unique pixel values
        img_array = np.array(img)
        if len(np.unique(img_array)) > 2:  # Only append frames with more than two unique pixel values (avoid black frames)
            # Enhance contrast and saturation
            if enhance:
                img = ImageEnhance.Brightness(img).enhance(enhance[0])   # Slightly increase brightness
                img = ImageEnhance.Sharpness(img).enhance(enhance[1])    # Sharpen the image
                img = ImageEnhance.Contrast(img).enhance(enhance[2])     # Enhance contrast
                img = ImageEnhance.Color(img).enhance(enhance[3])        # Boost color saturation 

            frames.append(img)

    for i in range(transition_frames):
        alpha = i / float(transition_frames)
        transition_frame = Image.blend(frames[-1], frames[0], alpha)
        frames.append(transition_frame)

    save_output(frames, output_path, fps, output_format)