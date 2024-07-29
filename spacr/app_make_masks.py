import tkinter as tk
from tkinter import ttk
from .gui import MainApp

def initiate_make_mask_app(parent_frame):
    from .gui_elements import ModifyMaskApp, set_dark_style
    settings_window = tk.Toplevel(parent_frame)
    settings_window.title("Make Masks Settings")
    style_out = set_dark_style(ttk.Style())
    settings_window.configure(bg=style_out['bg_color'])
    settings_frame = tk.Frame(settings_window, bg=style_out['bg_color'])
    settings_frame.pack(fill=tk.BOTH, expand=True)
    
    vars_dict = {
        'folder_path': ttk.Entry(settings_frame),
        'scale_factor': ttk.Entry(settings_frame)
    }
    row = 0
    for name, entry in vars_dict.items():
        ttk.Label(settings_frame, text=f"{name.replace('_', ' ').capitalize()}:",
                  background=style_out['bg_color'], foreground=style_out['fg_color']).grid(row=row, column=0)
        entry.grid(row=row, column=1)
        row += 1

    # Function to be called when "Run" button is clicked
    def start_make_mask_app():
        folder_path = vars_dict['folder_path'].get()
        try:
            scale_factor = float(vars_dict['scale_factor'].get())
        except ValueError:
            scale_factor = None
        folder_path = folder_path if folder_path != '' else None
        settings_window.destroy()
        ModifyMaskApp(parent_frame, folder_path, scale_factor)
    
    run_button = tk.Button(settings_window, text="Start Make Masks", command=start_make_mask_app, bg=style_out['bg_color'], fg=style_out['fg_color'])
    run_button.pack(pady=10)

def start_make_mask_app():
    app = MainApp(default_app="Make Masks")
    app.mainloop()

if __name__ == "__main__":
    start_make_mask_app()