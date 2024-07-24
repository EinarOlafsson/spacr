from .gui import MainApp

def start_seq_app():
    app = MainApp(default_app="Sequencing")
    app.mainloop()

if __name__ == "__main__":
    start_seq_app()