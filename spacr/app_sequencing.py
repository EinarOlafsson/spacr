from .gui import MainApp

def start_seq_app():
    """Launch the main spacr GUI with the Sequencing tab preselected."""
    app = MainApp(default_app="Sequencing")
    app.mainloop()

if __name__ == "__main__":
    start_seq_app()