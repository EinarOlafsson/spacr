from .gui import MainApp

def start_seq_app():
    """
    Launch the spaCR GUI with the Measure application preloaded.

    This function initializes the main GUI window with "Measure" selected as the default active module.
    It is used to directly open the application in object measurement mode.

    Typical use case:
        Called from the command line or another script to start spaCR in measurement mode.
    """
    app = MainApp(default_app="Sequencing")
    app.mainloop()

if __name__ == "__main__":
    start_seq_app()