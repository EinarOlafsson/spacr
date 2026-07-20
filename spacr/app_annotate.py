import tkinter as tk
from tkinter import ttk
from .gui import MainApp
from .gui_elements import set_dark_style, spacrButton

def convert_to_number(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Unable to convert '{value}' to an integer or float.")

def initiate_annotation_app(parent_frame):
    """Bootstrap: pick the experiment src directory, construct AnnotateApp,
    then hand off to its own settings window for everything else.
    """
    import os
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from .gui_elements import set_dark_style, AnnotateApp

    # 1. Pick src (the only thing we need before instantiation)
    src = filedialog.askdirectory(parent=parent_frame, title="Select experiment directory")
    if not src or not os.path.isdir(src):
        return  # user cancelled

    db_path = os.path.join(src, 'measurements', 'measurements.db')
    if not os.path.isfile(db_path):
        if not messagebox.askyesno(
            "Database not found",
            f"No file at {db_path}. Continue anyway?",
            parent=parent_frame,
        ):
            return

    # 2. Create the AnnotateApp window with defaults; user refines later.
    root = tk.Toplevel(parent_frame)
    root.title("Annotate")
    style_out = set_dark_style(ttk.Style())
    root.configure(bg=style_out['bg_color'])

    app = AnnotateApp(root=root, db_path=db_path, src=src)

    # 3. Hand off to the single source-of-truth settings UI.
    app.open_settings_window()

    return app


def start_annotate_app():
    app = MainApp(default_app="Annotate")
    app.mainloop()

if __name__ == "__main__":
    start_annotate_app()