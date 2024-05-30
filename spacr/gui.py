import tkinter as tk
from tkinter import ttk

# Import your GUI apps
from spacr.gui_mask_app import initiate_mask_root
from spacr.gui_measure_app import initiate_measure_root
from spacr.annotate_app import initiate_annotation_app_root
from spacr.mask_app import initiate_mask_app_root
from spacr.gui_classify_app import initiate_classify_root

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SpaCr GUI Collection")
        self.geometry("1000x800")
        self.configure(bg="#333333")

        self.gui_apps = {
            "Mask": initiate_mask_root,
            "Measure": initiate_measure_root,
            "Annotate": initiate_annotation_app_root,
            "Make Masks": initiate_mask_app_root,
            "Classify": initiate_classify_root
        }

        self.selected_app = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        menu_frame = ttk.Frame(self)
        menu_frame.pack(fill=tk.X, pady=10)

        ttk.Label(menu_frame, text="Select GUI App: ", background="#333333", foreground="white").pack(side=tk.LEFT, padx=(10, 0))
        self.app_selector = ttk.Combobox(menu_frame, textvariable=self.selected_app, values=list(self.gui_apps.keys()), state="readonly")
        self.app_selector.pack(side=tk.LEFT, padx=(0, 10))
        self.app_selector.bind("<<ComboboxSelected>>", self.load_app)

        self.app_frame = ttk.Frame(self)
        self.app_frame.pack(fill=tk.BOTH, expand=True)

    def load_app(self, event):
        selected_app_name = self.selected_app.get()
        selected_app_func = self.gui_apps[selected_app_name]
        self.clear_frame(self.app_frame)

        app_root, _ = selected_app_func(self.winfo_width(), self.winfo_height())
        app_root.pack(fill=tk.BOTH, expand=True)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

def gui_app():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    gui_app()