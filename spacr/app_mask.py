from .gui import MainApp

def start_mask_app():
    """
    Launch the spaCR GUI with the Mask application preloaded.

    This function initializes the main GUI window with "Mask" selected as the default active module.
    It is intended for users who want to directly start the application in mask generation mode.

    Typical use case:
        Called from the command line or another script to launch the mask generation workflow of spaCR.
    """
    app = MainApp(default_app="Mask")
    app.mainloop()

if __name__ == "__main__":
    start_mask_app()