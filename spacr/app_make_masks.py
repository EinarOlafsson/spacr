import tkinter as tk
from tkinter import ttk
from .gui import MainApp

def initiate_make_mask_app(parent_frame):
    from .gui_elements import modify_masks
    # Set up the settings window
    settings_window = tk.Toplevel(parent_frame)
    settings_window.title("Make Masks Settings")
    settings_window.configure(bg='black')  # Set the background color to black
    
    # Use the existing function to create the settings UI
    settings_frame = tk.Frame(settings_window, bg='black')  # Set the background color to black
    settings_frame.pack(fill=tk.BOTH, expand=True)
    
    vars_dict = {
        'folder_path': ttk.Entry(settings_frame),
        'scale_factor': ttk.Entry(settings_frame)
    }

    # Arrange input fields and labels
    row = 0
    for name, entry in vars_dict.items():
        ttk.Label(settings_frame, text=f"{name.replace('_', ' ').capitalize()}:",
                  background="black", foreground="white").grid(row=row, column=0)
        entry.grid(row=row, column=1)
        row += 1

    # Function to be called when "Run" button is clicked
    def start_make_mask_app():
        folder_path = vars_dict['folder_path'].get()
        try:
            scale_factor = float(vars_dict['scale_factor'].get())
        except ValueError:
            scale_factor = None  # Handle invalid input gracefully

        # Convert empty strings to None
        folder_path = folder_path if folder_path != '' else None

        settings_window.destroy()
        modify_masks(parent_frame, folder_path, scale_factor)
    
    run_button = tk.Button(settings_window, text="Start Make Masks", command=start_make_mask_app, bg='black', fg='white')
    run_button.pack(pady=10)

def start_make_mask_app():
    app = MainApp(default_app="Make Masks")
    app.mainloop()

if __name__ == "__main__":
    start_make_mask_app()