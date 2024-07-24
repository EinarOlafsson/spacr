from .gui import MainApp

def start_umap_app():
    app = MainApp(default_app="Umap")
    app.mainloop()

if __name__ == "__main__":
    start_umap_app()