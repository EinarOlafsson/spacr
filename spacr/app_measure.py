from .gui import MainApp

def start_measure_app():
    """Launch the main spacr GUI with the Measure tab preselected."""
    app = MainApp(default_app="Measure")
    app.mainloop()

if __name__ == "__main__":
    start_measure_app()