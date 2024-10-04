import os, threading, time, sqlite3, webbrowser, pyautogui, random, cv2
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog
from tkinter import font
from queue import Queue
from tkinter import Label, Frame, Button
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from skimage.exposure import rescale_intensity
from IPython.display import display, HTML
import imageio.v2 as imageio
from collections import deque
from skimage.draw import polygon, line
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes, label
from tkinter import ttk, scrolledtext 

fig = None

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
            troughcolor=self.inactive_color,       # Set the trough to bg color
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
            row_info = self.grid_info().get('row', 0)
            col_info = self.grid_info().get('column', 0)
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
        image_path = os.path.join(self.folder_path, self.image_filenames[index])
        image = imageio.imread(image_path)        
        mask_path = os.path.join(self.masks_folder, self.image_filenames[index])
        if os.path.exists(mask_path):
            print(f'loading mask:{mask_path} for image: {image_path}')
            mask = imageio.imread(mask_path)
            if mask.dtype != np.uint8:
                mask = (mask / np.max(mask) * 255).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            print(f'loaded new mask for image: {image_path}')
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
    def __init__(self, root, db_path, src, image_type=None, channels=None, image_size=200, annotation_column='annotate', normalize=False, percentiles=(1, 99), measurement=None, threshold=None):
        self.root = root
        self.db_path = db_path
        self.src = src
        self.index = 0

        if isinstance(image_size, list):
            self.image_size = (int(image_size[0]), int(image_size[0]))
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            raise ValueError("Invalid image size")

        self.annotation_column = annotation_column
        self.image_type = image_type
        self.channels = channels
        self.normalize = normalize
        self.percentiles = percentiles
        self.images = {}
        self.pending_updates = {}
        self.labels = []
        self.adjusted_to_original_paths = {}
        self.terminate = False
        self.update_queue = Queue()
        self.measurement = measurement
        self.threshold = threshold

        style_out = set_dark_style(ttk.Style())
        self.font_loader = style_out['font_loader']
        self.font_size = style_out['font_size']
        self.bg_color = style_out['bg_color']
        self.fg_color = style_out['fg_color']
        self.active_color = style_out['active_color']
        self.inactive_color = style_out['inactive_color']


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

        # Create the status label
        self.status_label = Label(root, text="", font=self.font_style, bg=self.root.cget('bg'))
        self.status_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        # Place the buttons at the bottom right
        self.button_frame = Frame(root, bg=self.root.cget('bg'))
        self.button_frame.grid(row=2, column=1, padx=10, pady=10, sticky="se")

        self.next_button = Button(self.button_frame, text="Next", command=self.next_page, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.next_button.pack(side="right", padx=5)

        self.previous_button = Button(self.button_frame, text="Back", command=self.previous_page, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.previous_button.pack(side="right", padx=5)

        self.exit_button = Button(self.button_frame, text="Exit", command=self.shutdown, bg=self.bg_color, fg=self.fg_color, highlightbackground=self.fg_color, highlightcolor=self.fg_color, highlightthickness=1)
        self.exit_button.pack(side="right", padx=5)

        # Calculate grid rows and columns based on the root window size and image size
        self.calculate_grid_dimensions()

        # Create a frame to hold the image grid
        self.grid_frame = Frame(root, bg=self.root.cget('bg'))
        self.grid_frame.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

        for i in range(self.grid_rows * self.grid_cols):
            label = Label(self.grid_frame, bg=self.root.cget('bg'))
            label.grid(row=i // self.grid_cols, column=i % self.grid_cols, padx=2, pady=2, sticky="nsew")
            self.labels.append(label)

        # Make the grid frame resize with the window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        for row in range(self.grid_rows):
            self.grid_frame.grid_rowconfigure(row, weight=1)
        for col in range(self.grid_cols):
            self.grid_frame.grid_columnconfigure(col, weight=1)

    def calculate_grid_dimensions(self):
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        self.grid_cols = window_width // (self.image_size[0] + 4)
        self.grid_rows = (window_height - self.button_frame.winfo_height() - 4) // (self.image_size[1] + 4)

        # Update to make sure grid_rows and grid_cols are at least 1
        self.grid_cols = max(1, self.grid_cols)
        self.grid_rows = max(1, self.grid_rows)

    def prefilter_paths_annotations(self):
        from .io import _read_and_join_tables
        from .utils import is_list_of_lists

        if self.measurement and self.threshold is not None:
            df = _read_and_join_tables(self.db_path)
            df[self.annotation_column] = None
            before = len(df)

            if is_list_of_lists(self.measurement):
                if isinstance(self.threshold, list) or is_list_of_lists(self.threshold):
                    if len(self.measurement) == len(self.threshold):
                        for idx, var in enumerate(self.measurement):
                            df = df[df[var[idx]] > self.threshold[idx]]
                        after = len(df)
                    elif len(self.measurement) == len(self.threshold)*2:
                        th_idx = 0
                        for idx, var in enumerate(self.measurement):
                            if idx % 2 != 0:
                                th_idx += 1
                                thd = self.threshold
                                if isinstance(thd, list):
                                    thd = thd[0]
                                df[f'threshold_measurement_{idx}'] = df[self.measurement[idx]]/df[self.measurement[idx+1]]
                                print(f"mean threshold_measurement_{idx}: {np.mean(df['threshold_measurement'])}")
                                print(f"median threshold measurement: {np.median(df[self.measurement])}")
                                df = df[df[f'threshold_measurement_{idx}'] > thd]
                        after = len(df)
            
            elif isinstance(self.measurement, list):
                df['threshold_measurement'] = df[self.measurement[0]]/df[self.measurement[1]]
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
                        df = df[df['png_path'].str.contains(tpe)]
                else:
                    df = df[df['png_path'].str.contains(self.image_type)]
                after = len(df)
                print(f'image_type: Removed: {before-after} rows, retained {after}')

            self.filtered_paths_annotations = df[['png_path', self.annotation_column]].values.tolist()
        else:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            if self.image_type:
                c.execute(f"SELECT png_path, {self.annotation_column} FROM png_list WHERE png_path LIKE ?", (f"%{self.image_type}%",))
            else:
                c.execute(f"SELECT png_path, {self.annotation_column} FROM png_list")
            self.filtered_paths_annotations = c.fetchall()
            conn.close()

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

    def load_single_image(self, path_annotation_tuple):
        path, annotation = path_annotation_tuple
        img = Image.open(path)
        img = self.normalize_image(img, self.normalize, self.percentiles)
        img = img.convert('RGB')
        img = self.filter_channels(img)
        img = img.resize(self.image_size)
        return img, annotation

    @staticmethod
    def normalize_image(img, normalize=False, percentiles=(1, 99)):
        img_array = np.array(img)

        if normalize:
            if img_array.ndim == 2:  # Grayscale image
                p2, p98 = np.percentile(img_array, percentiles)
                img_array = rescale_intensity(img_array, in_range=(p2, p98), out_range=(0, 255))
            else:  # Color image or multi-channel image
                for channel in range(img_array.shape[2]):
                    p2, p98 = np.percentile(img_array[:, :, channel], percentiles)
                    img_array[:, :, channel] = rescale_intensity(img_array[:, :, channel], in_range=(p2, p98), out_range=(0, 255))

        img_array = np.clip(img_array, 0, 255).astype('uint8')

        return Image.fromarray(img_array)
    
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
            if 'r' not in self.channels:
                r = r.point(lambda _: 0)
            if 'g' not in self.channels:
                g = g.point(lambda _: 0)
            if 'b' not in self.channels:
                b = b.point(lambda _: 0)

            if len(self.channels) == 1:
                channel_img = r if 'r' in self.channels else (g if 'g' in self.channels else b)
                return ImageOps.grayscale(channel_img)

        return Image.merge("RGB", (r, g, b))

    def get_on_image_click(self, path, label, img):
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

    @staticmethod
    def update_html(text):
        display(HTML(f"""
        <script>
        document.getElementById('unique_id').innerHTML = '{text}';
        </script>
        """))

    def update_database_worker(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        display(HTML("<div id='unique_id'>Initial Text</div>"))

        while True:
            if self.terminate:
                conn.close()
                break

            if not self.update_queue.empty():
                AnnotateApp.update_html("Do not exit, Updating database...")
                self.status_label.config(text='Do not exit, Updating database...')

                pending_updates = self.update_queue.get()
                for path, new_annotation in pending_updates.items():
                    if new_annotation is None:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = NULL WHERE png_path = ?', (path,))
                    else:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = ? WHERE png_path = ?', (new_annotation, path))
                conn.commit()

                AnnotateApp.update_html('')
                self.status_label.config(text='')
                self.root.update()
            time.sleep(0.1)

    def update_gui_text(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def next_page(self):
        if self.pending_updates:
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index += self.grid_rows * self.grid_cols
        self.prefilter_paths_annotations()  # Re-fetch annotations from the database
        self.load_images()

    def previous_page(self):
        if self.pending_updates:
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index -= self.grid_rows * self.grid_cols
        if self.index < 0:
            self.index = 0
        self.prefilter_paths_annotations()  # Re-fetch annotations from the database
        self.load_images()

    def shutdown(self):
        self.terminate = True
        self.update_queue.put(self.pending_updates.copy())
        if not self.pending_updates:
            self.pending_updates.clear()
            self.db_update_thread.join()
            self.root.quit()
            self.root.destroy()
            print(f'Quit application')
        else:
            print('Waiting for pending updates to finish before quitting')

def create_menu_bar(root):
    from .gui import initiate_root
    gui_apps = {
        "Mask": lambda: initiate_root(root, settings_type='mask'),
        "Measure": lambda: initiate_root(root, settings_type='measure'),
        "Annotate": lambda: initiate_root(root, settings_type='annotate'),
        "Make Masks": lambda: initiate_root(root, settings_type='make_masks'),
        "Classify": lambda: initiate_root(root, settings_type='classify'),
        "Umap": lambda: initiate_root(root, settings_type='umap'),
        "Train Cellpose": lambda: initiate_root(root, settings_type='train_cellpose'),
        "ML Analyze": lambda: initiate_root(root, settings_type='ml_analyze'),
        "Cellpose Masks": lambda: initiate_root(root, settings_type='cellpose_masks'),
        "Cellpose All": lambda: initiate_root(root, settings_type='cellpose_all'),
        "Map Barcodes": lambda: initiate_root(root, settings_type='map_barcodes'),
        "Regression": lambda: initiate_root(root, settings_type='regression'),
        "Recruitment": lambda: initiate_root(root, settings_type='recruitment')
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
    app_menu.add_command(label="Help", command=lambda: webbrowser.open("https://spacr.readthedocs.io/en/latest/?badge=latest"))
    app_menu.add_command(label="Exit", command=root.quit)

    # Configure the menu for the root window
    root.config(menu=menu_bar)

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