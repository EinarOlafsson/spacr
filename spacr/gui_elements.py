import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

class spacrDropdownMenu(tk.OptionMenu):
    def __init__(self, parent, variable, options, command=None, **kwargs):
        self.variable = variable
        self.variable.set("Select Category")
        super().__init__(parent, self.variable, *options, command=command, **kwargs)
        self.config(bg='black', fg='white', font=('Helvetica', 12), indicatoron=0)
        self.menu = self['menu']
        self.menu.config(bg='black', fg='white', font=('Helvetica', 12))

    def update_styles(self, active_categories):
        menu = self['menu']
        for idx, option in enumerate(self['menu'].entrycget(idx, "label") for idx in range(self['menu'].index("end")+1)):
            if option in active_categories:
                menu.entryconfig(idx, background='teal', foreground='white')
            else:
                menu.entryconfig(idx, background='black', foreground='white')

class spacrCheckbutton(ttk.Checkbutton):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.configure(text=self.text, variable=self.variable, command=self.command, style='Spacr.TCheckbutton')

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
        
        for child in self.scrollable_frame.winfo_children():
            child.configure(bg='black')

class spacrLabel(tk.Frame):
    def __init__(self, parent, text="", font=None, style=None, align="right", **kwargs):
        label_kwargs = {k: v for k, v in kwargs.items() if k in ['foreground', 'background', 'font', 'anchor', 'justify', 'wraplength']}
        for key in label_kwargs.keys():
            kwargs.pop(key)
        super().__init__(parent, **kwargs)
        self.text = text
        self.kwargs = label_kwargs
        self.align = align
        screen_height = self.winfo_screenheight()
        label_height = screen_height // 50
        label_width = label_height * 10
        self.canvas = tk.Canvas(self, width=label_width, height=label_height, highlightthickness=0, bg=self.kwargs.get("background", "black"))
        self.canvas.grid(row=0, column=0, sticky="ew")

        self.font_style = font if font else tkFont.Font(family=self.kwargs.get("font_family", "Helvetica"), size=self.kwargs.get("font_size", 12), weight=tkFont.NORMAL)
        self.style = style

        if self.align == "center":
            anchor_value = tk.CENTER
            text_anchor = 'center'
        else:  # default to right alignment
            anchor_value = tk.E
            text_anchor = 'e'

        if self.style:
            ttk_style = ttk.Style()
            ttk_style.configure(self.style, **label_kwargs)
            self.label_text = ttk.Label(self.canvas, text=self.text, style=self.style, anchor=text_anchor, justify=text_anchor)
            self.label_text.pack(fill=tk.BOTH, expand=True)
        else:
            self.label_text = self.canvas.create_text(label_width // 2 if self.align == "center" else label_width - 5, 
                                                      label_height // 2, text=self.text, fill=self.kwargs.get("foreground", "white"), 
                                                      font=self.font_style, anchor=anchor_value, justify=tk.RIGHT)

    def set_text(self, text):
        if self.style:
            self.label_text.config(text=text)
        else:
            self.canvas.itemconfig(self.label_text, text=text)

class spacrButton(tk.Frame):
    def __init__(self, parent, text="", command=None, font=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.command = command
        screen_height = self.winfo_screenheight()
        button_height = screen_height // 50
        button_width = button_height * 3

        # Increase the canvas size to accommodate the button and the rim
        self.canvas = tk.Canvas(self, width=button_width + 4, height=button_height + 4, highlightthickness=0, bg="black")
        self.canvas.grid(row=0, column=0)

        self.button_bg = self.create_rounded_rectangle(2, 2, button_width + 2, button_height + 2, radius=20, fill="#000000", outline="#ffffff")

        self.font_style = font if font else tkFont.Font(family="Helvetica", size=12, weight=tkFont.NORMAL)
        self.button_text = self.canvas.create_text((button_width + 4) // 2, (button_height + 4) // 2, text=self.text, fill="white", font=self.font_style)

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

    def on_enter(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill="#008080")  # Teal color

    def on_leave(self, event=None):
        self.canvas.itemconfig(self.button_bg, fill="#000000")  # Black color

    def on_click(self, event=None):
        if self.command:
            self.command()

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=20, **kwargs):
        points = [
            x1 + radius, y1,
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
            x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

class spacrSwitch(ttk.Frame):
    def __init__(self, parent, text="", variable=None, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.text = text
        self.variable = variable if variable else tk.BooleanVar()
        self.command = command
        self.canvas = tk.Canvas(self, width=40, height=20, highlightthickness=0, bd=0, bg="black")
        self.canvas.grid(row=0, column=1, padx=(10, 0))
        self.switch_bg = self.create_rounded_rectangle(2, 2, 38, 18, radius=9, outline="", fill="#fff")
        self.switch = self.canvas.create_oval(4, 4, 16, 16, outline="", fill="#800080")  # Purple initially
        self.label = spacrLabel(self, text=self.text, background="black", foreground="white")
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
        self.tooltip_window.configure(bg='black')
        label = tk.Label(self.tooltip_window, text=self.text, background="#333333", foreground="white", relief='flat', borderwidth=0)
        label.grid(row=0, column=0, padx=5, pady=5)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

def create_menu_bar(root):
    from .app_annotate import initiate_annotation_app_root
    from .app_make_masks import initiate_mask_app_root
    from .gui_utils import load_app

    gui_apps = {
        "Mask": 'mask',
        "Measure": 'measure',
        "Annotate": initiate_annotation_app_root,
        "Make Masks": initiate_mask_app_root,
        "Classify": 'classify',
        "Sequencing": 'sequencing',
        "Umap": 'umap'
    }

    def load_app_wrapper(app_name, app_func):
        load_app(root, app_name, app_func)

    # Create the menu bar
    menu_bar = tk.Menu(root, bg="#008080", fg="white")
    # Create a "SpaCr Applications" menu
    app_menu = tk.Menu(menu_bar, tearoff=0, bg="#008080", fg="white")
    menu_bar.add_cascade(label="SpaCr Applications", menu=app_menu)
    # Add options to the "SpaCr Applications" menu
    for app_name, app_func in gui_apps.items():
        app_menu.add_command(label=app_name, command=lambda app_name=app_name, app_func=app_func: load_app_wrapper(app_name, app_func))
    # Add a separator and an exit option
    app_menu.add_separator()
    app_menu.add_command(label="Exit", command=root.quit)
    # Configure the menu for the root window
    root.config(menu=menu_bar)

def set_dark_style(style):
    font_style = tkFont.Font(family="Helvetica", size=24)
    style.configure('TEntry', padding='5 5 5 5', borderwidth=1, relief='solid', fieldbackground='black', foreground='#ffffff', font=font_style)
    style.configure('TCombobox', fieldbackground='black', background='black', foreground='#ffffff', font=font_style)
    style.map('TCombobox', fieldbackground=[('readonly', 'black')], foreground=[('readonly', '#ffffff')])
    style.configure('Custom.TButton', background='black', foreground='white', bordercolor='white', focusthickness=3, focuscolor='white', font=('Helvetica', 12))
    style.map('Custom.TButton', background=[('active', 'teal'), ('!active', 'black')], foreground=[('active', 'white'), ('!active', 'white')], bordercolor=[('active', 'white'), ('!active', 'white')])
    style.configure('Custom.TLabel', padding='5 5 5 5', borderwidth=1, relief='flat', background='black', foreground='#ffffff', font=font_style)
    style.configure('Spacr.TCheckbutton', background='black', foreground='#ffffff', indicatoron=False, relief='flat', font="15")
    style.map('Spacr.TCheckbutton', background=[('selected', 'black'), ('active', 'black')], foreground=[('selected', '#ffffff'), ('active', '#ffffff')])
    style.configure('TLabel', background='black', foreground='#ffffff', font=font_style)
    style.configure('TFrame', background='black')
    style.configure('TPanedwindow', background='black')
    style.configure('TNotebook', background='black', tabmargins=[2, 5, 2, 0])
    style.configure('TNotebook.Tab', background='black', foreground='#ffffff', padding=[5, 5], font=font_style)
    style.map('TNotebook.Tab', background=[('selected', '#555555'), ('active', '#555555')], foreground=[('selected', '#ffffff'), ('active', '#ffffff')])
    style.configure('TButton', background='black', foreground='#ffffff', padding='5 5 5 5', font=font_style)
    style.map('TButton', background=[('active', '#555555'), ('disabled', '#333333')])
    style.configure('Vertical.TScrollbar', background='black', troughcolor='black', bordercolor='black')
    style.configure('Horizontal.TScrollbar', background='black', troughcolor='black', bordercolor='black')
    style.configure('Custom.TLabelFrame', font=('Helvetica', 10, 'bold'), background='black', foreground='white', relief='solid', borderwidth=1)
    style.configure('Custom.TLabelFrame.Label', background='black', foreground='white', font=('Helvetica', 10, 'bold'))

def set_default_font(root, font_name="Helvetica", size=12):
    default_font = (font_name, size)
    root.option_add("*Font", default_font)
    root.option_add("*TButton.Font", default_font)
    root.option_add("*TLabel.Font", default_font)
    root.option_add("*TEntry.Font", default_font)