import os, webbrowser, pyautogui, random, cv2
from tkinter import ttk, scrolledtext
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog
from tkinter import font
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageEnhance

fig = None

def restart_gui_app(root):
    """
    Restarts the SpaCr GUI application by destroying the current root window 
    and launching a fresh instance.

    Args:
        root (tk.Tk): The current Tkinter root window to be destroyed.
    
    Note:
        The new instance is launched by importing and invoking `gui_app()`.
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
    """
    Creates a top-level menu bar for the SpaCr GUI containing shortcuts to all
    major application modules and help resources.

    Args:
        root (tk.Tk): The root window where the menu bar will be attached.

    Adds:
        - A 'SpaCr Applications' menu with links to:
          'Mask', 'Measure', 'Classify', 'ML Analyze', 'Map Barcodes',
          'Regression', 'Activation', and 'Recruitment'.
        - A Help option linking to the online documentation.
        - An Exit option to quit the application.
    """
    from .gui import initiate_root
    gui_apps = {
        "Mask": lambda: initiate_root(root, settings_type='mask'),
        "Measure": lambda: initiate_root(root, settings_type='measure'),
        "Classify": lambda: initiate_root(root, settings_type='classify'),
        "ML Analyze": lambda: initiate_root(root, settings_type='ml_analyze'),
        "Map Barcodes": lambda: initiate_root(root, settings_type='map_barcodes'),
        "Regression": lambda: initiate_root(root, settings_type='regression'),
        "Activation": lambda: initiate_root(root, settings_type='activation'),
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
    #app_menu.add_command(label="Home",command=lambda: restart_gui_app(root))
    app_menu.add_command(label="Help", command=lambda: webbrowser.open("https://einarolafsson.github.io/spacr/index.html"))
    app_menu.add_command(label="Exit", command=root.quit)

    # Configure the menu for the root window
    root.config(menu=menu_bar)

def set_element_size():
    """
    Calculates and returns standardized UI element dimensions 
    based on the current screen size.

    Returns:
        dict: A dictionary with element dimensions including:
            - 'btn_size' (int): Size of buttons.
            - 'bar_size' (int): Height of progress bars.
            - 'settings_width' (int): Width of the settings panel.
            - 'panel_width' (int): Width of the plotting panel.
            - 'panel_height' (int): Height of the bottom control panel.
    """
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
    """
    Applies a dark theme to the SpaCr GUI using the provided styling options.

    Args:
        style (ttk.Style): The ttk style instance to configure.
        parent_frame (tk.Widget, optional): The top-level container to apply styles to.
        containers (list, optional): Additional containers (ttk.Frame or tk.Frame) to style.
        widgets (list, optional): List of individual widgets to apply colors and fonts.
        font_family (str): Font family for all labels and buttons.
        font_size (int): Font size for all text elements.
        bg_color (str): Background color.
        fg_color (str): Foreground/text color.
        active_color (str): Highlight or selected color.
        inactive_color (str): Secondary background color.

    Returns:
        dict: Style parameters used, including resolved font and color values.
    """
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

        Args:
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

        Args:
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

        Args:
        - size: int, the size of the font (optional).

        Returns:
        - tkFont.Font object.
        """
        if size is None:
            size = self.font_size
        return font.Font(family=self.font_name, size=size)

class spacrContainer(tk.Frame):
    """
    A custom container widget that manages multiple resizable panes arranged 
    either vertically or horizontally, separated by draggable sashes.

    Args:
        parent (tk.Widget): The parent widget.
        orient (str): Orientation of the layout ('tk.VERTICAL' or 'tk.HORIZONTAL'). Default is vertical.
        bg (str): Background color of the container and sashes. Defaults to 'lightgrey'.
    """
    def __init__(self, parent, orient=tk.VERTICAL, bg=None, *args, **kwargs):
        """
        Initialize the spacrContainer with the specified orientation and background color.

        Args:
            parent (tk.Widget): Parent widget.
            orient (str): Layout orientation (tk.VERTICAL or tk.HORIZONTAL).
            bg (str, optional): Background color. Defaults to 'lightgrey'.
        """
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
        """
        Add a new widget as a pane to the container.

        Args:
            widget (tk.Widget): Widget to add.
            stretch (str): Stretch policy (currently unused).
        """
        print(f"Adding widget: {widget} with stretch: {stretch}")
        pane = tk.Frame(self, bg=self.bg)
        pane.grid_propagate(False)
        widget.grid(in_=pane, sticky="nsew")  # Use grid for the widget within the pane
        self.panes.append((pane, widget))

        if len(self.panes) > 1:
            self.create_sash()

        self.reposition_panes()

    def create_sash(self):
        """
        Create a draggable sash between panes.
        """
        sash = tk.Frame(self, bg=self.bg, cursor='sb_v_double_arrow' if self.orient == tk.VERTICAL else 'sb_h_double_arrow', height=self.sash_thickness, width=self.sash_thickness)
        sash.bind("<Enter>", self.on_enter_sash)
        sash.bind("<Leave>", self.on_leave_sash)
        sash.bind("<ButtonPress-1>", self.start_resize)
        self.sashes.append(sash)

    def reposition_panes(self):
        """
        Reposition panes and sashes within the container based on orientation.
        """
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
        """
        Event handler triggered on container resize.

        Args:
            event (tk.Event): Tkinter event object.
        """
        print(f"Configuring container: {self}")
        self.reposition_panes()

    def on_enter_sash(self, event):
        """
        Change sash color on mouse enter.

        Args:
            event (tk.Event): Tkinter event object.
        """
        event.widget.config(bg='blue')

    def on_leave_sash(self, event):
        """
        Reset sash color on mouse leave.

        Args:
            event (tk.Event): Tkinter event object.
        """
        event.widget.config(bg=self.bg)

    def start_resize(self, event):
        """
        Initiate resizing behavior when mouse press begins on a sash.

        Args:
            event (tk.Event): Tkinter event object.
        """
        sash = event.widget
        self.start_pos = event.y_root if self.orient == tk.VERTICAL else event.x_root
        self.start_size = sash.winfo_y() if self.orient == tk.VERTICAL else sash.winfo_x()
        sash.bind("<B1-Motion>", self.perform_resize)

    def perform_resize(self, event):
        """
        Adjust pane sizes during mouse drag on a sash.

        Args:
            event (tk.Event): Tkinter event object.
        """
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
    """
    A custom Tkinter entry widget with rounded corners, dark theme styling, and active/inactive color handling.

    Args:
        parent (tk.Widget): Parent widget.
        textvariable (tk.StringVar, optional): Tkinter textvariable to bind to the entry.
        outline (bool, optional): Whether to show an outline. Currently unused. Defaults to False.
        width (int, optional): Width of the entry widget. Defaults to 220 if not provided.
        \*args, \*\*kwargs: Additional arguments passed to the parent Frame.
    """
    def __init__(self, parent, textvariable=None, outline=False, width=None, *args, **kwargs):
        """
        Initialize the custom entry widget with dark theme and rounded styling.
        """
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
        """
        Draws a rounded rectangle with the given color as background.
        
        Args:
            color (str): Fill color for the rounded rectangle.
        """
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
        """
        Event handler for focus in. Changes the background to the active color.
        """
        self.draw_rounded_rectangle(self.active_color)
        self.entry.config(bg=self.active_color)
    
    def on_focus_out(self, event):
        """
        Event handler for focus out. Reverts the background to the inactive color.
        """
        self.draw_rounded_rectangle(self.bg_color)
        self.entry.config(bg=self.bg_color)

class spacrCheck(tk.Frame):
    """
    A custom checkbox widget with rounded square appearance and dark style.

    Args:
        parent (tk.Widget): Parent widget.
        text (str, optional): Label text (currently unused).
        variable (tk.BooleanVar): Tkinter variable to bind the checkbox state.
        \*args, \*\*kwargs: Additional arguments passed to the parent Frame.
    """
    def __init__(self, parent, text="", variable=None, *args, **kwargs):
        """
        Initializes the custom checkbox widget and binds visual updates to variable state.
        """
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
        """
        Draws a rounded square with border and fill.

        Args:
            color (str): The fill color based on the current checkbox state.
        """
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
        """
        Redraws the checkbox based on the current value of the associated variable.
        """
        self.draw_rounded_square(self.active_color if self.variable.get() else self.inactive_color)

    def toggle_variable(self, event):
        """
        Toggles the value of the associated variable when the checkbox is clicked.

        Args:
            event (tk.Event): The mouse click event.
        """
        self.variable.set(not self.variable.get())

class spacrCombo(tk.Frame):
    """
    A custom styled combo box widget with rounded edges and dropdown functionality.

    This widget mimics a modern dropdown menu with dark-themed styling, allowing
    users to select from a list of values in a visually appealing interface.
    
    Args:
        parent (tk.Widget): Parent widget.
        textvariable (tk.StringVar, optional): Variable linked to the combo box selection.
        values (list, optional): List of selectable values. Defaults to empty list.
        width (int, optional): Width of the combo box in pixels. Defaults to 220.
    """
    def __init__(self, parent, textvariable=None, values=None, width=None, *args, **kwargs):
        """
        Initialize the combo box UI and style settings.

        Args:
            parent (tk.Widget): The parent widget.
            textvariable (tk.StringVar, optional): A Tkinter StringVar linked to the selected value.
            values (list, optional): List of values to populate the dropdown.
            width (int, optional): Width of the combo box. Defaults to 220 pixels.
        """
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
        """
        Draw a rounded rectangle on the canvas with the specified background color.

        Args:
            color (str): The fill color for the rounded rectangle.
        """
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
        """
        Handle click event on the combo box to toggle the dropdown menu.

        Args:
            event (tk.Event): The mouse click event.
        """
        if self.dropdown_menu is None:
            self.open_dropdown()
        else:
            self.close_dropdown()

    def open_dropdown(self):
        """
        Display the dropdown menu with available values.
        """
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
        """
        Close the dropdown menu if it is open.
        """
        self.draw_rounded_rectangle(self.inactive_color)
        
        if self.dropdown_menu:
            self.dropdown_menu.destroy()
            self.dropdown_menu = None

    def on_select(self, value):
        """
        Update the displayed label and internal variable when a value is selected.

        Args:
            value (str): The selected value from the dropdown.
        """
        display_text = value if value is not None else 'None'
        self.var.set(value)
        self.label.config(text=display_text)
        self.selected_value = value
        self.close_dropdown()

    def set(self, value):
        """
        Programmatically set the combo box selection to the specified value.

        Args:
            value (str): The value to set in the combo box.
        """
        display_text = value if value is not None else 'None'
        self.var.set(value)
        self.label.config(text=display_text)
        self.selected_value = value

class spacrDropdownMenu(tk.Frame):
    """
    A custom dark-themed dropdown menu widget with rounded edges and hover interaction.

    This widget displays a labeled button that reveals a menu of selectable options
    when clicked. It supports external callback functions, styling updates, and dynamic
    highlighting of active menu items.

    Args:
        parent (tk.Widget): Parent widget in which the dropdown menu is placed.
        variable (tk.StringVar): A Tkinter variable to store the selected option.
        options (list): A list of option labels to populate the dropdown menu.
        command (callable, optional): A function to call when an option is selected.
        font (tuple or tk.Font, optional): Font used for the button label.
        size (int, optional): Height of the button in pixels. Defaults to 50.
        **kwargs: Additional keyword arguments passed to the `tk.Frame` base class.
    """
    def __init__(self, parent, variable, options, command=None, font=None, size=50, **kwargs):
        """
        Initialize the spacrDropdownMenu with a canvas-based button and popup menu.

        Sets up the button appearance, binds mouse interaction events,
        and constructs the dropdown menu with the given options.

        Args:
            parent (tk.Widget): Parent container.
            variable (tk.StringVar): Variable that stores the selected option.
            options (list): List of strings representing the dropdown options.
            command (callable, optional): Callback function when an option is selected.
            font (tk.Font or tuple, optional): Font used for the button text.
            size (int, optional): Button height in pixels. Defaults to 50.
            **kwargs: Additional keyword arguments for the Frame.
        """
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
        """
        Draw a rounded rectangle polygon on the internal canvas.

        Args:
            x1 (int): Top-left x coordinate.
            y1 (int): Top-left y coordinate.
            x2 (int): Bottom-right x coordinate.
            y2 (int): Bottom-right y coordinate.
            radius (int, optional): Radius of the corners. Defaults to 20.
            **kwargs: Canvas polygon configuration options (fill, outline, etc.).

        Returns:
            int: Canvas item ID of the created polygon.
        """
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
        """
        Handle mouse enter event by updating the button's background color.

        Args:
            event (tk.Event, optional): The event object. Defaults to None.
        """
        self.canvas.itemconfig(self.button_bg, fill=self.active_color)

    def on_leave(self, event=None):
        """
        Handle mouse leave event by resetting the button's background color.

        Args:
            event (tk.Event, optional): The event object. Defaults to None.
        """
        self.canvas.itemconfig(self.button_bg, fill=self.inactive_color)

    def on_click(self, event=None):
        """
        Handle button click event to display the dropdown menu.

        Args:
            event (tk.Event, optional): The event object. Defaults to None.
        """
        self.post_menu()

    def post_menu(self):
        """
        Display the dropdown menu below the button using screen coordinates.
        """
        x, y, width, height = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        self.menu.post(x, y + height)

    def on_select(self, option):
        """
        Callback when an option is selected from the dropdown menu.

        Args:
            option (str): The selected option label.
        """
        if self.command:
            self.command(option)

    def update_styles(self, active_categories=None):
        """
        Update the styles of the dropdown menu entries based on active categories.

        Args:
            active_categories (list or None): List of option labels to highlight
                with the active color. If None, all entries are styled as inactive.
        """
        style_out = set_dark_style(ttk.Style(), widgets=[self.menu])

        if active_categories is not None:
            for idx in range(self.menu.index("end") + 1):
                option = self.menu.entrycget(idx, "label")
                if option in active_categories:
                    self.menu.entryconfig(idx, background=style_out['active_color'], foreground=style_out['fg_color'])
                else:
                    self.menu.entryconfig(idx, background=style_out['bg_color'], foreground=style_out['fg_color'])

class spacrCheckbutton(ttk.Checkbutton):
    """
    A dark-themed styled Checkbutton widget for use in the SpaCr GUI.

    This class wraps a `ttk.Checkbutton` with a custom style and binds it to a
    `BooleanVar`, allowing it to integrate seamlessly into dark-themed interfaces.

    Args:
        parent (tk.Widget): The parent widget in which to place the checkbutton.
        text (str, optional): The label text displayed next to the checkbutton.
        variable (tk.BooleanVar, optional): Variable linked to the checkbutton's state.
            If None, a new `BooleanVar` is created.
        command (callable, optional): A callback function to execute when the checkbutton is toggled.
        *args: Additional positional arguments passed to `ttk.Checkbutton`.
        **kwargs: Additional keyword arguments passed to `ttk.Checkbutton`.
    """
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.configure(text=self.text, variable=self.variable, command=self.command, style='Spacr.TCheckbutton')
        style = ttk.Style()
        _ = set_dark_style(style, widgets=[self])

class spacrProgressBar(ttk.Progressbar):
    """
    A dark-themed progress bar widget with optional progress label display.

    This class extends `ttk.Progressbar` and applies a dark visual theme consistent
    with SpaCr GUI styling. It also provides an optional label that displays real-time
    progress, operation type, and additional information.

    Args:
        parent (tk.Widget): The parent widget in which to place the progress bar.
        label (bool, optional): Whether to show a label below the progress bar. Defaults to True.
        *args: Additional positional arguments passed to `ttk.Progressbar`.
        **kwargs: Additional keyword arguments passed to `ttk.Progressbar`.
    """
    def __init__(self, parent, label=True, *args, **kwargs):
        """
        Initialize the progress bar and optional label with dark theme styling.

        Sets the initial value to 0, applies custom style attributes, and creates
        a label for displaying progress information.

        Args:
            parent (tk.Widget): Parent container for the widget.
            label (bool, optional): Whether to show a label for progress updates. Defaults to True.
        """
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
        """
        Place the progress label one row below the progress bar in the grid layout.

        Should be called after the progress bar has been placed with `.grid(...)`.
        """
        if self.label and self.progress_label:
            row_info = self.grid_info().get('rowID', 0)
            col_info = self.grid_info().get('columnID', 0)
            col_span = self.grid_info().get('columnspan', 1)
            self.progress_label.grid(row=row_info + 1, column=col_info, columnspan=col_span, pady=5, padx=5, sticky='ew')

    def update_label(self):
        """
        Update the progress label with current progress, operation type, and extra info.

        Constructs a single-line status message with:
        - Current progress value
        - Operation type (if set)
        - Additional info (if set)
        """
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
    """
    A custom slider widget with dark-themed styling, optional numeric entry, and mouse interaction.

    This slider is designed for GUI applications where numeric control is needed,
    supporting dynamic resizing, labeled value entry, and a callback on release.

    Args:
        master (tk.Widget): Parent widget.
        length (int, optional): Fixed pixel length of the slider. If None, adapts to canvas width.
        thickness (int, optional): Thickness of the slider bar in pixels. Defaults to 2.
        knob_radius (int, optional): Radius of the slider knob in pixels. Defaults to 10.
        position (str, optional): Alignment of slider within the frame. One of "left", "center", "right".
        from_ (float): Minimum slider value.
        to (float): Maximum slider value.
        value (float, optional): Initial value. Defaults to `from_`.
        show_index (bool, optional): Whether to show an entry for numeric value. Defaults to False.
        command (Callable, optional): Function to call with the final value upon knob release.
        **kwargs: Additional options passed to `tk.Frame`.
    """
    def __init__(self, master=None, length=None, thickness=2, knob_radius=10, position="center", from_=0, to=100, value=None, show_index=False, command=None, **kwargs):
        """
        Initialize a custom dark-themed slider widget.

        This slider supports mouse interaction, optional direct numeric input via an Entry,
        and dynamically adapts its layout based on container resizing unless a fixed `length` is specified.

        Args:
            master (tk.Widget, optional): Parent widget.
            length (int, optional): Fixed length of the slider in pixels. If None, dynamically resizes with the container.
            thickness (int, optional): Thickness of the slider bar in pixels. Default is 2.
            knob_radius (int, optional): Radius of the draggable knob in pixels. Default is 10.
            position (str, optional): Alignment of the slider within the frame. One of {"left", "center", "right"}. Default is "center".
            from_ (float): Minimum value of the slider.
            to (float): Maximum value of the slider.
            value (float, optional): Initial value of the slider. Defaults to `from_` if not specified.
            show_index (bool, optional): If True, displays a text entry box to manually input the slider value. Default is False.
            command (Callable, optional): Optional function to be called with the final value when the knob is released.
            **kwargs: Additional keyword arguments passed to the `tk.Frame` initializer.
        """
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
        """
        Recalculate slider dimensions and redraw upon resizing.
        
        Args:
            event (tk.Event): Resize event.
        """
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
        """
        Convert a numerical slider value to a pixel position on the canvas.

        Args:
            value (float): The numerical value to convert.

        Returns:
            float: Corresponding position in pixels on the slider.
        """
        if self.to == self.from_:
            return self.knob_radius
        relative_value = (value - self.from_) / (self.to - self.from_)
        return self.knob_radius + relative_value * (self.length - 2 * self.knob_radius)

    def position_to_value(self, position):
        """
        Convert a pixel position on the slider to a numerical value.

        Args:
            position (float): Pixel position on the slider.

        Returns:
            float: Corresponding numerical slider value.
        """ 
        if self.to == self.from_:
            return self.from_
        relative_position = (position - self.knob_radius) / (self.length - 2 * self.knob_radius)
        return self.from_ + relative_position * (self.to - self.from_)

    def draw_slider(self, inactive=False):
        """
        Draw the slider bar and knob on the canvas.

        Args:
            inactive (bool): If True, draw knob in inactive color. Otherwise, use active color.
        """
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
        """
        Move the knob in response to mouse drag, updating internal value and position.

        Args:
            event (tk.Event): Motion event.
        """
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
        """
        Highlight knob and respond to click by positioning knob at mouse location.

        Args:
            event (tk.Event): Click event.
        """
        self.draw_slider(inactive=False)
        self.move_knob(event)

    def release_knob(self, event):
        """
        Finalize knob movement and call the `command` callback with final value.

        Args:
            event (tk.Event): Mouse release event.
        """
        self.draw_slider(inactive=True)
        if self.command:
            self.command(self.value)  # Call the command with the final value when the knob is released

    def set_to(self, new_to):
        """
        Change the maximum value (`to`) of the slider.

        Args:
            new_to (float): New upper bound of the slider.
        """
        self.to = new_to
        self.knob_position = self.value_to_position(self.value)
        self.draw_slider(inactive=False)

    def get(self):
        """
        Get the current slider value.

        Returns:
            float: Current value of the slider.
        """ 
        return self.value

    def set(self, value):
        """
        Set the slider to a specific value and redraw.

        Args:
            value (float): New value to set (clipped to bounds).
        """
        self.value = max(self.from_, min(value, self.to))  # Ensure the value is within bounds
        self.knob_position = self.value_to_position(self.value)
        self.draw_slider(inactive=False)
        if self.show_index:
            self.index_var.set(str(int(self.value)))

    def jump_to_click(self, event):
        """
        Move the knob directly to the mouse click position.

        Args:
            event (tk.Event): Click event.
        """
        self.activate_knob(event)

    def update_slider_from_entry(self, event):
        """
        Update the slider from a value entered manually in the index entry.

        Args:
            event (tk.Event): Return key press event.
        """
        try:
            index = int(self.index_var.get())
            self.set(index)
            if self.command:
                self.command(self.value)
        except ValueError:
            pass

def spacrScrollbarStyle(style, inactive_color, active_color):
    """
    Applies a custom vertical scrollbar style using the given colors.

    This function defines a new ttk scrollbar style named 'Custom.Vertical.TScrollbar'.
    It reuses the base elements from the 'clam' theme and sets the colors for active 
    and inactive states accordingly. If the required elements do not exist, it creates 
    them from the base theme.

    Args:
        style (ttk.Style): The ttk Style object to configure.
        inactive_color (str): Hex or color name for the scrollbar in its default (inactive) state.
        active_color (str): Hex or color name for the scrollbar when hovered or active.
    """
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
    """
    A styled frame with optional rounded background, vertical scrollbar, and embedded content area (text or widgets).
    
    This frame supports both scrollable `ttk.Frame` containers and scrollable `tk.Text` areas, with a dark custom theme.

    Attributes:
        scrollable_frame (Union[ttk.Frame, tk.Text]): The inner content widget.
    """
    def __init__(self, container, width=None, *args, bg='black', radius=20, scrollbar=True, textbox=False, **kwargs):
        """
        Initialize the spacrFrame.

        Args:
            container (tk.Widget): The parent container for this frame.
            width (int, optional): Width of the frame in pixels. Defaults to 1/4 of screen width if None.
            *args: Additional positional arguments for ttk.Frame.
            bg (str): Background color of the frame. Defaults to 'black'.
            radius (int): Radius of the rounded rectangle background. Defaults to 20.
            scrollbar (bool): Whether to include a vertical scrollbar. Defaults to True.
            textbox (bool): If True, use a scrollable `tk.Text` widget. Otherwise, use a `ttk.Frame`. Defaults to False.
            **kwargs: Additional keyword arguments for ttk.Frame.
        """
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
        """
        Draw a rounded rectangle on a canvas.

        Args:
            canvas (tk.Canvas): The canvas to draw on.
            x1 (int): Left coordinate.
            y1 (int): Top coordinate.
            x2 (int): Right coordinate.
            y2 (int): Bottom coordinate.
            radius (int): Radius of the rounded corners. Defaults to 20.
            **kwargs: Options passed to the canvas `create_polygon` method.

        Returns:
            int: ID of the created polygon.
        """
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
    """
    A custom label widget with optional dark styling, alignment options, and support for both `ttk.Label` and `Canvas`-rendered text.
    
    The label adapts to screen size or a given height, and can display text either centered or right-aligned.
    """
    def __init__(self, parent, text="", font=None, style=None, align="right", height=None, **kwargs):
        """
        Initialize the spacrLabel widget.

        Args:
            parent (tk.Widget): The parent widget.
            text (str): The text to display on the label. Defaults to "".
            font (tkFont.Font, optional): A custom font to use if not using the default style. Defaults to None.
            style (str, optional): A ttk style name to apply to the label. If set, uses a `ttk.Label` instead of `Canvas` text.
            align (str): Text alignment, either "right" or "center". Defaults to "right".
            height (int, optional): Height of the label. If None, scales based on screen height.
            **kwargs: Additional keyword arguments for the outer frame (excluding font/background/anchor-specific ones).
        """
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
        """
        Update the label text.

        Args:
            text (str): The new text to display.
        """
        if self.style:
            self.label_text.config(text=text)
        else:
            self.canvas.itemconfig(self.label_text, text=text)

class spacrButton(tk.Frame):
    """
    A custom animated button widget with icon and optional text, styled with dark mode and zoom animation on hover.

    The button is rendered using a Canvas to support rounded corners and icon embedding. Optional description
    display is supported via parent methods `show_description` and `clear_description`.
    """
    def __init__(self, parent, text="", command=None, font=None, icon_name=None, size=50, show_text=True, outline=False, animation=True, *args, **kwargs):
        """
        Initialize the spacrButton.

        Args:
            parent (tk.Widget): The parent container.
            text (str): Button text to display.
            command (callable, optional): Function to call when button is clicked.
            font (tkFont.Font or tuple, optional): Font to use if font loader is unavailable.
            icon_name (str, optional): Name of icon (without extension) to load from resources/icons.
            size (int): Button height (and icon size reference). Defaults to 50.
            show_text (bool): Whether to show text next to the icon. Defaults to True.
            outline (bool): Whether to draw a border around the button. Defaults to False.
            animation (bool): Whether to animate the icon on hover. Defaults to True.
            *args: Additional positional arguments for the Frame.
            **kwargs: Additional keyword arguments for the Frame.
        """
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
        """
        Load and resize the icon from the icon folder.

        Falls back to 'default.png' if the specified icon cannot be found.
        """
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
        """
        Get the full path to the icon file.

        Args:
            icon_name (str): Icon name without extension.

        Returns:
            str: Full path to the icon file.
        """
        icon_dir = os.path.join(os.path.dirname(__file__), 'resources', 'icons')
        return os.path.join(icon_dir, f"{icon_name}.png")

    def on_enter(self, event=None):
        """
        Handle mouse hover enter event.

        Changes button color, shows description, and animates zoom-in if enabled.
        """
        self.canvas.itemconfig(self.button_bg, fill=self.active_color)
        self.update_description(event)
        if self.animation and not self.is_zoomed_in:
            self.animate_zoom(0.85)  # Zoom in the icon to 85% of button size

    def on_leave(self, event=None):
        """
        Handle mouse hover leave event.

        Resets button color, clears description, and animates zoom-out if enabled.
        """
        self.canvas.itemconfig(self.button_bg, fill=self.inactive_color)
        self.clear_description(event)
        if self.animation and self.is_zoomed_in:
            self.animate_zoom(0.65)  # Reset the icon size to 65% of button size

    def on_click(self, event=None):
        """
        Trigger the button's command callback when clicked.
        """
        if self.command:
            self.command()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
        """
        Create a rounded rectangle on the canvas.

        Args:
            x1, y1, x2, y2 (int): Coordinates of the rectangle.
            radius (int): Radius of the corners.
            **kwargs: Passed to `create_polygon`.

        Returns:
            int: Canvas item ID of the rounded rectangle.
        """
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
        """
        Call parent container’s `show_description()` if available,
        passing the description based on `main_buttons` or `additional_buttons` maps.
        """
        parent = self.master
        while parent:
            if hasattr(parent, 'show_description'):
                parent.show_description(parent.main_buttons.get(self, parent.additional_buttons.get(self, "No description available.")))
                return
            parent = parent.master

    def clear_description(self, event):
        """
        Call parent container’s `clear_description()` if available.
        """
        parent = self.master
        while parent:
            if hasattr(parent, 'clear_description'):
                parent.clear_description()
                return
            parent = parent.master

    def animate_zoom(self, target_scale, steps=10, delay=10):
        """
        Animate zoom effect by resizing icon incrementally.

        Args:
            target_scale (float): Final scale factor relative to base icon size.
            steps (int): Number of animation steps. Defaults to 10.
            delay (int): Delay between steps in milliseconds. Defaults to 10.
        """
        current_scale = 0.85 if self.is_zoomed_in else 0.65
        step_scale = (target_scale - current_scale) / steps
        self._animate_step(current_scale, step_scale, steps, delay)

    def _animate_step(self, current_scale, step_scale, steps, delay):
        """
        Helper method to perform recursive icon zoom animation.

        Args:
            current_scale (float): Current zoom scale.
            step_scale (float): Incremental change per step.
            steps (int): Steps remaining.
            delay (int): Delay per step in ms.
        """
        if steps > 0:
            new_scale = current_scale + step_scale
            self.zoom_icon(new_scale)
            self.after(delay, self._animate_step, new_scale, step_scale, steps - 1, delay)
        else:
            self.is_zoomed_in = not self.is_zoomed_in

    def zoom_icon(self, scale_factor):
        """
        Resize and update the icon image on the canvas.

        Args:
            scale_factor (float): Scaling factor relative to base icon size.
        """
        # Resize the original icon image
        new_size = int(self.size * scale_factor)
        resized_icon = self.original_icon_image.resize((new_size, new_size), Image.Resampling.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(resized_icon)

        # Update the icon on the canvas
        self.canvas.itemconfig(self.button_icon, image=self.icon_photo)
        self.canvas.image = self.icon_photo

class spacrSwitch(ttk.Frame):
    """
    A custom toggle switch widget with animated transitions and label, styled using the spacr dark theme.

    This switch mimics a physical toggle with animated motion of the switch knob and changes in color.
    """
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        """
        Initialize the spacrSwitch widget.

        Args:
            parent (tk.Widget): Parent container.
            text (str): Label displayed to the left of the switch.
            variable (tk.BooleanVar, optional): Tkinter BooleanVar linked to the switch state.
            command (callable, optional): Function to call when the switch is toggled.
            *args: Additional positional arguments for the Frame.
            **kwargs: Additional keyword arguments for the Frame.
        """
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
        """
        Toggle the state of the switch.

        Updates the linked variable, animates the movement, and calls the `command` callback if defined.
        """
        self.variable.set(not self.variable.get())
        self.animate_switch()
        if self.command:
            self.command()

    def update_switch(self):
        """
        Immediately update the switch position and color based on the current value of the variable.
        """
        if self.variable.get():
            self.canvas.itemconfig(self.switch, fill="#008080")
            self.canvas.coords(self.switch, 24, 4, 36, 16)
        else:
            self.canvas.itemconfig(self.switch, fill="#800080")
            self.canvas.coords(self.switch, 4, 4, 16, 16)

    def animate_switch(self):
        """
        Trigger an animated transition of the switch knob between on and off states.
        """
        if self.variable.get():
            start_x, end_x = 4, 24
            final_color = "#008080"
        else:
            start_x, end_x = 24, 4
            final_color = "#800080"

        self.animate_movement(start_x, end_x, final_color)

    def animate_movement(self, start_x, end_x, final_color):
        """
        Animate the horizontal movement of the switch knob.

        Args:
            start_x (int): Starting x-coordinate of the knob.
            end_x (int): Ending x-coordinate of the knob.
            final_color (str): Fill color of the knob at the end of the animation.
        """
        step = 1 if start_x < end_x else -1
        for i in range(start_x, end_x, step):
            self.canvas.coords(self.switch, i, 4, i + 12, 16)
            self.canvas.update()
            self.after(10)
        self.canvas.itemconfig(self.switch, fill=final_color)

    def get(self):
        """
        Get the current Boolean value of the switch.

        Returns:
            bool: True if switch is on, False otherwise.
        """
        return self.variable.get()

    def set(self, value):
        """
        Set the switch to a given Boolean value.

        Args:
            value (bool): New state for the switch.
        """
        self.variable.set(value)
        self.update_switch()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=9, **kwargs):
        """
        Draw a rounded rectangle polygon on the canvas.

        Args:
            x1, y1, x2, y2 (int): Coordinates of the rectangle bounds.
            radius (int): Radius of corner curvature.
            **kwargs: Options passed to `create_polygon`.

        Returns:
            int: ID of the created polygon item on the canvas.
        """
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
    """
    A simple tooltip widget for displaying hover text in a Tkinter application using spacr dark styling.
    """
    def __init__(self, widget, text):
        """
        Initialize the tooltip for a given widget.

        Args:
            widget (tk.Widget): The widget to attach the tooltip to.
            text (str): The text to display in the tooltip.
        """
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        """
        Display the tooltip near the cursor when mouse enters the widget.
        """
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
        """
        Hide and destroy the tooltip when mouse leaves the widget.
        """
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

def standardize_figure(fig):
    """
    Apply standardized appearance settings to a matplotlib figure using spaCR GUI style preferences.

    This includes:
    - Setting font size and font family based on spaCR's theme
    - Setting text and spine colors to match spaCR foreground color
    - Applying OpenSans font via `font_loader`
    - Removing top and right spines
    - Setting line and spine widths
    - Adjusting background and grid colors

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure to standardize.

    Returns:
        matplotlib.figure.Figure: The updated figure with standardized style.
    """
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

    Args:
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
    """
    Opens a file dialog to save a matplotlib figure in the specified format.

    Prompts the user to choose a save location and filename using a file dialog.
    The figure is saved using the provided format if a valid path is selected.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        file_format (str): The file format to save as (e.g., 'png', 'pdf', 'svg').

    Returns:
        None
    """     
    file_path = filedialog.asksaveasfilename(defaultextension=f".{file_format}", filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")])
    if file_path:
        try:
            fig.savefig(file_path, format=file_format)
            print(f"Figure saved as {file_format.upper()} at {file_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

def modify_figure(fig):
    """
    Opens a GUI window for interactively modifying various properties of a matplotlib figure.

    This function allows users to:
    - Rescale the X and Y axes
    - Change line width and font size
    - Set axis limits and title
    - Customize colors (background, text, line)
    - Rotate x-axis labels
    - Enable/disable grid, legend, and axes
    - Toggle spine visibility ("spleens")

    Once modifications are entered and applied, the figure is updated and re-rendered via `display_figure`.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure to modify.

    Returns:
        None
    """
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
    Generates an animated matrix-style DNA sequence visual and saves it as a GIF or video.

    The animation simulates vertical streams of random DNA bases ('A', 'T', 'C', 'G') cascading down the screen.
    Each column has independently scrolling strings of bases. The latest base in each stream is highlighted,
    and fading effects, coloring, and random lowercase transformations are applied for visual flair.

    Args:
        output_path (str): Path to the output file (should end in .gif, .mp4, or .avi).
        canvas_width (int): Width of the canvas in pixels.
        canvas_height (int): Height of the canvas in pixels.
        duration (int): Duration of the animation in seconds.
        fps (int): Frames per second of the animation.
        base_size (int): Font size (in pixels) for the bases.
        transition_frames (int): Number of frames for the looping transition.
        font_type (str): Path to a .ttf font to use. Defaults to Arial.
        enhance (list): List of four enhancement multipliers for [brightness, sharpness, contrast, saturation].
        lowercase_prob (float): Probability that a given base will be drawn in lowercase.

    Returns:
        None
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