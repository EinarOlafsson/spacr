import os, threading, time, sqlite3
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from queue import Queue
from tkinter import Label
import numpy as np
from PIL import Image, ImageOps, ImageTk
from concurrent.futures import ThreadPoolExecutor
from skimage.exposure import rescale_intensity
from IPython.display import display, HTML
import imageio.v2 as imageio
from collections import deque
from skimage.draw import polygon, line
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes, label
from tkinter import ttk, scrolledtext
import platform

def set_dark_style(style, parent_frame=None, containers=None, widgets=None, font_family="Arial", font_size=12, bg_color='black', fg_color='white', active_color='blue', inactive_color='dark_gray'):

    if active_color == 'teal':
        active_color = '#008080'
    if inactive_color == 'dark_gray':
        inactive_color = '#050505'
    if bg_color == 'black':
        bg_color = '#000000'
    if fg_color == 'white':
        fg_color = '#ffffff'
    if active_color == 'blue':
        active_color = '#007BFF'

    style.theme_use('clam')
    font_style = tkFont.Font(family=font_family, size=font_size)
    style.configure('TEntry', padding='5 5 5 5', borderwidth=1, relief='solid', fieldbackground=bg_color, foreground=fg_color, font=font_style)
    style.configure('TCombobox', fieldbackground=bg_color, background=bg_color, foreground=fg_color, selectbackground=bg_color, selectforeground=fg_color, font=font_style)
    style.map('TCombobox', fieldbackground=[('readonly', bg_color)], foreground=[('readonly', fg_color)], selectbackground=[('readonly', bg_color)], selectforeground=[('readonly', fg_color)])
    style.configure('Custom.TButton', background=bg_color, foreground=fg_color, bordercolor=fg_color, focusthickness=3, focuscolor=fg_color, font=(font_family, font_size))
    style.map('Custom.TButton', background=[('active', active_color), ('!active', bg_color)], foreground=[('active', fg_color), ('!active', fg_color)], bordercolor=[('active', fg_color), ('!active', fg_color)])
    style.configure('Custom.TLabel', padding='5 5 5 5', borderwidth=1, relief='flat', background=bg_color, foreground=fg_color, font=font_style)
    style.configure('Spacr.TCheckbutton', background=bg_color, foreground=fg_color, indicatoron=False, relief='flat', font="15")
    style.map('Spacr.TCheckbutton', background=[('selected', bg_color), ('active', bg_color)], foreground=[('selected', fg_color), ('active', fg_color)])
    style.configure('TLabel', background=bg_color, foreground=fg_color, font=font_style)
    style.configure('TFrame', background=bg_color)
    style.configure('TPanedwindow', background=bg_color)
    style.configure('TNotebook', background=bg_color, tabmargins=[2, 5, 2, 0])
    style.configure('TNotebook.Tab', background=bg_color, foreground=fg_color, padding=[5, 5], font=font_style)
    style.map('TNotebook.Tab', background=[('selected', active_color), ('active', active_color)], foreground=[('selected', fg_color), ('active', fg_color)])
    style.configure('TButton', background=bg_color, foreground=fg_color, padding='5 5 5 5', font=font_style)
    style.map('TButton', background=[('active', active_color), ('disabled', inactive_color)])
    style.configure('Vertical.TScrollbar', background=bg_color, troughcolor=bg_color, bordercolor=bg_color)
    style.configure('Horizontal.TScrollbar', background=bg_color, troughcolor=bg_color, bordercolor=bg_color)
    style.configure('Custom.TLabelFrame', font=(font_family, font_size, 'bold'), background=bg_color, foreground='white', relief='solid', borderwidth=1)
    style.configure('Custom.TLabelFrame.Label', background=bg_color, foreground='white', font=(font_family, font_size, 'bold'))

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
                widget.configure(fg=fg_color, font=(font_family, font_size))
            if isinstance(widget, scrolledtext.ScrolledText):
                widget.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            if isinstance(widget, tk.OptionMenu):
                widget.configure(bg=bg_color, fg=fg_color, font=(font_family, font_size))
                menu = widget['menu']
                menu.configure(bg=bg_color, fg=fg_color, font=(font_family, font_size))

    return {'font_family': font_family, 'font_size': font_size, 'bg_color': bg_color, 'fg_color': fg_color, 'active_color': active_color, 'inactive_color': inactive_color}

def set_default_font(root, font_name="Arial", size=12):
    default_font = (font_name, size)
    root.option_add("*Font", default_font)
    root.option_add("*TButton.Font", default_font)
    root.option_add("*TLabel.Font", default_font)
    root.option_add("*TEntry.Font", default_font)

class spacrDropdownMenu(tk.OptionMenu):
    def __init__(self, parent, variable, options, command=None, **kwargs):
        self.variable = variable
        self.variable.set("Select Category")
        super().__init__(parent, self.variable, *options, command=command, **kwargs)
        self.update_styles()

    def update_styles(self, active_categories=None):
        style = ttk.Style()
        style_out = set_dark_style(style, widgets=[self])
        self.menu = self['menu']
        style_out = set_dark_style(style, widgets=[self.menu])

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

class spacrFrame(ttk.Frame):
    def __init__(self, container, width=None, *args, bg='black', **kwargs):
        super().__init__(container, *args, **kwargs)
        self.configure(style='TFrame')
        if width is None:
            screen_width = self.winfo_screenwidth()
            width = screen_width // 4
        canvas = tk.Canvas(self, bg=bg, width=width)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas, style='TFrame')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        
        style = ttk.Style()
        _ = set_dark_style(style, containers=[self], widgets=[canvas, scrollbar, self.scrollable_frame])

class spacrLabel(tk.Frame):
    def __init__(self, parent, text="", font=None, style=None, align="right", **kwargs):
        valid_kwargs = {k: v for k, v in kwargs.items() if k not in ['foreground', 'background', 'font', 'anchor', 'justify', 'wraplength']}
        super().__init__(parent, **valid_kwargs)
        
        self.text = text
        self.align = align
        screen_height = self.winfo_screenheight()
        label_height = screen_height // 50
        label_width = label_height * 10
        style_out = set_dark_style(ttk.Style())

        self.canvas = tk.Canvas(self, width=label_width, height=label_height, highlightthickness=0, bg=style_out['bg_color'])
        self.canvas.grid(row=0, column=0, sticky="ew")

        self.font_style = font if font else tkFont.Font(family=style_out['font_family'], size=style_out['font_size'], weight=tkFont.NORMAL)
        self.style = style

        if self.align == "center":
            anchor_value = tk.CENTER
            text_anchor = 'center'
        else:  # default to right alignment
            anchor_value = tk.E
            text_anchor = 'e'

        if self.style:
            ttk_style = ttk.Style()
            ttk_style.configure(self.style, font=self.font_style, background=style_out['bg_color'], foreground=style_out['fg_color'])
            self.label_text = ttk.Label(self.canvas, text=self.text, style=self.style, anchor=text_anchor)
            self.label_text.pack(fill=tk.BOTH, expand=True)
        else:
            self.label_text = self.canvas.create_text(label_width // 2 if self.align == "center" else label_width - 5, 
                                                      label_height // 2, text=self.text, fill=style_out['fg_color'], 
                                                      font=self.font_style, anchor=anchor_value, justify=tk.RIGHT)
        
        _ = set_dark_style(ttk.Style(), containers=[self], widgets=[self.canvas])

    def set_text(self, text):
        if self.style:
            self.label_text.config(text=text)
        else:
            self.canvas.itemconfig(self.label_text, text=text)

class spacrButton(tk.Frame):
    def __init__(self, parent, text="", command=None, font=None, icon_name=None, size=50, show_text=True, outline=True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.text = text.capitalize()  # Capitalize only the first letter of the text
        self.command = command
        self.icon_name = icon_name if icon_name else text.lower()
        self.size = size
        self.show_text = show_text
        self.outline = outline

        style_out = set_dark_style(ttk.Style())

        if self.show_text:
            self.button_width = int(size * 3)
        else:
            self.button_width = self.size  # Make the button width equal to the size if show_text is False

        # Create the canvas first
        self.canvas = tk.Canvas(self, width=self.button_width + 4, height=self.size + 4, highlightthickness=0, bg=style_out['bg_color'])
        self.canvas.grid(row=0, column=0)

        # Apply dark style and get color settings
        color_settings = set_dark_style(ttk.Style(), containers=[self], widgets=[self.canvas])

        if self.outline:
            self.button_bg = self.create_rounded_rectangle(2, 2, self.button_width + 2, self.size + 2, radius=20, fill=color_settings['bg_color'], outline=color_settings['fg_color'])
        else:
            self.button_bg = self.create_rounded_rectangle(2, 2, self.button_width + 2, self.size + 2, radius=20, fill=color_settings['bg_color'], outline=color_settings['bg_color'])
        
        self.load_icon()
        self.font_style = font if font else ("Arial", 12)  # Default font if not provided
        
        if self.show_text:
            self.button_text = self.canvas.create_text(self.size + 10, self.size // 2 + 2, text=self.text, fill=color_settings['fg_color'], font=self.font_style, anchor="w")  # Align text to the left of the specified point

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

        self.bg_color = color_settings['bg_color']
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

        initial_size = int(self.size * 0.9)  # Make the initial size slightly smaller
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
        if not self.is_zoomed_in:
            self.animate_zoom(1.1)  # Zoom in the icon by 10%

    def on_leave(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill=self.bg_color)
        self.clear_description(event)
        if self.is_zoomed_in:
            self.animate_zoom(1.0)  # Reset the icon size

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
        current_scale = 1.1 if self.is_zoomed_in else 1.0
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
        new_size = int(self.size * 0.9 * scale_factor)
        resized_icon = self.original_icon_image.resize((new_size, new_size), Image.Resampling.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(resized_icon)

        # Update the icon on the canvas
        self.canvas.itemconfig(self.button_icon, image=self.icon_photo)
        self.canvas.image = self.icon_photo  # Keep a reference to avoid garbage collection

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
    def __init__(self, root, db_path, src, image_type=None, channels=None, grid_rows=None, grid_cols=None, image_size=(200, 200), annotation_column='annotate', normalize=False, percentiles=(1,99), measurement=None, threshold=None):
        self.root = root
        self.db_path = db_path
        self.src = src
        self.index = 0
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.image_size = image_size
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
        self.status_label = Label(self.root, text="", font=("Arial", 12))
        self.status_label.grid(row=self.grid_rows + 1, column=0, columnspan=self.grid_cols)
        self.measurement = measurement
        self.threshold = threshold

        self.filtered_paths_annotations = []
        self.prefilter_paths_annotations()

        self.db_update_thread = threading.Thread(target=self.update_database_worker)
        self.db_update_thread.start()

        for i in range(grid_rows * grid_cols):
            label = Label(root)
            label.grid(row=i // grid_cols, column=i % grid_cols)
            self.labels.append(label)

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
                border_color = 'teal' if annotation == 1 else 'red'
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

        bordered_img = Image.new('RGB', (img.width + 2 * border_width, img.height + 2 * border_width), color='white')
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
            border_fill = 'teal' if new_annotation == 1 else ('red' if new_annotation == 2 else None)
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
        self.load_images()

    def previous_page(self):
        if self.pending_updates:
            self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.index -= self.grid_rows * self.grid_cols
        if self.index < 0:
            self.index = 0
        self.load_images()

    def shutdown(self):
        self.terminate = True
        self.update_queue.put(self.pending_updates.copy())
        self.pending_updates.clear()
        self.db_update_thread.join()
        self.root.quit()
        self.root.destroy()
        print(f'Quit application')

def create_menu_bar(root):
    from .gui import initiate_root
    gui_apps = {
        "Mask": (lambda frame: initiate_root(frame, 'mask'), "Generate cellpose masks for cells, nuclei and pathogen images."),
        "Measure": (lambda frame: initiate_root(frame, 'measure'), "Measure single object intensity and morphological feature. Crop and save single object image"),
        "Annotate": (lambda frame: initiate_root(frame, 'annotate'), "Annotation single object images on a grid. Annotations are saved to database."),
        "Make Masks": (lambda frame: initiate_root(frame, 'make_masks'), "Adjust pre-existing Cellpose models to your specific dataset for improved performance"),
        "Classify": (lambda frame: initiate_root(frame, 'classify'), "Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images."),
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

    def load_app_wrapper(app_name, app_func):
        root.load_app(app_name, app_func)

    # Create the menu bar
    menu_bar = tk.Menu(root, bg="#008080", fg="white")

    # Create a "SpaCr Applications" menu
    app_menu = tk.Menu(menu_bar, tearoff=0, bg="#008080", fg="white")
    menu_bar.add_cascade(label="SpaCr Applications", menu=app_menu)

    # Add options to the "SpaCr Applications" menu
    for app_name, app_data in gui_apps.items():
        app_func, app_desc = app_data
        app_menu.add_command(label=app_name, command=lambda app_name=app_name, app_func=app_func: load_app_wrapper(app_name, app_func))

    # Add a separator and an exit option
    app_menu.add_separator()
    app_menu.add_command(label="Exit", command=root.quit)

    # Configure the menu for the root window
    root.config(menu=menu_bar)
