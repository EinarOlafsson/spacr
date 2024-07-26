import os, threading, time, sqlite3
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from queue import Queue
from tkinter import Label
import numpy as np
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageTk
from skimage.exposure import rescale_intensity
from IPython.display import display, HTML


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
        #screen_height = self.winfo_screenheight()
        button_height = 50 #screen_height // 50
        button_width = 140 #button_height * 3

        #print(button_height, button_width)

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

class ImageApp:
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
                ImageApp.update_html("Do not exit, Updating database...")
                self.status_label.config(text='Do not exit, Updating database...')

                pending_updates = self.update_queue.get()
                for path, new_annotation in pending_updates.items():
                    if new_annotation is None:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = NULL WHERE png_path = ?', (path,))
                    else:
                        c.execute(f'UPDATE png_list SET {self.annotation_column} = ? WHERE png_path = ?', (new_annotation, path))
                conn.commit()

                ImageApp.update_html('')
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
    style.configure('TCombobox', fieldbackground='black', background='black', foreground='#ffffff', selectbackground='black', selectforeground='#ffffff', font=font_style)
    style.map('TCombobox', fieldbackground=[('readonly', 'black')], foreground=[('readonly', '#ffffff')], selectbackground=[('readonly', 'black')], selectforeground=[('readonly', '#ffffff')])
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