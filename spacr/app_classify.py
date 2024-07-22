from .gui import MainApp

def start_classify_app():
    app = MainApp(default_app="Classify")
    app.mainloop()

if __name__ == "__main__":
    start_classify_app()