from .gui import MainApp

def start_mask_app():
    app = MainApp(default_app="Mask")
    app.mainloop()

if __name__ == "__main__":
    start_mask_app()