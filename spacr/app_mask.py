from .gui import MainApp

def start_mask_app():
    """Launch the main spacr GUI with the Mask tab preselected."""
    app = MainApp(default_app="Mask")
    app.mainloop()

if __name__ == "__main__":
    start_mask_app()