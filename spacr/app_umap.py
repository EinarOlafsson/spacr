from .gui import MainApp

def start_umap_app():
    """Launch the main spacr GUI with the Umap tab preselected."""
    app = MainApp(default_app="Umap")
    app.mainloop()

if __name__ == "__main__":
    start_umap_app()