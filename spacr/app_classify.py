from .gui import MainApp

def start_classify_app():
    """
    Launch the spaCR GUI with the Classify application preloaded.

    This function initializes the main GUI window with "Classify" set as the default active application.
    It then starts the Tkinter main event loop to display the interface.

    Typical use case:
        Called from the command line or another script to directly launch the Classify module of spaCR.

    """
    app = MainApp(default_app="Classify")
    app.mainloop()

if __name__ == "__main__":
    start_classify_app()