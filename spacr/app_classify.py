from .gui import MainApp

def start_classify_app():
    """Launch the main spacr GUI with the Classify tab preselected."""
    app = MainApp(default_app="Classify")
    app.mainloop()

if __name__ == "__main__":
    start_classify_app()