import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont

from .gui_elements import spacrFrame, spacrButton, set_dark_style, create_menu_bar, set_default_font

def initiate_annotation_app_root(parent_frame):
    from .gui_utils import generate_annotate_fields, run_annotate_app

    style = ttk.Style(parent_frame)
    set_dark_style(style)
    set_default_font(parent_frame, font_name="Arial", size=8)
    parent_frame.configure(bg='black')
    container = tk.PanedWindow(parent_frame, orient=tk.HORIZONTAL, bg='black')
    container.pack(fill=tk.BOTH, expand=True)
    scrollable_frame = spacrFrame(container, bg='black')
    container.add(scrollable_frame, stretch="always")
    vars_dict = generate_annotate_fields(scrollable_frame)
    run_button = spacrButton(
        scrollable_frame.scrollable_frame, 
        text="Run", 
        command=lambda: run_annotate_app(vars_dict, parent_frame),
        font=tkFont.Font(family="Arial", size=12, weight=tkFont.NORMAL)
    )
    run_button.grid(row=12, column=0, columnspan=2, pady=10, padx=10)
    return parent_frame

def gui_annotate():
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry(f"{width}x{height}")
    root.title("Annotate Application")
    
    # Clear previous content if any
    if hasattr(root, 'content_frame'):
        for widget in root.content_frame.winfo_children():
            widget.destroy()
        root.content_frame.grid_forget()
    else:
        root.content_frame = tk.Frame(root)
        root.content_frame.grid(row=1, column=0, sticky="nsew")
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
    
    initiate_annotation_app_root(root.content_frame)
    create_menu_bar(root)
    root.mainloop()

if __name__ == "__main__":
    gui_annotate()