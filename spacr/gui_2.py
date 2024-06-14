import customtkinter as ctk
from PIL import Image, ImageTk
import os
import requests

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SpaCr GUI Collection")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue")
        
        # Set scaling factor for high DPI displays; use a floating-point value.
        self.tk.call('tk', 'scaling', 1.5)
        
        self.create_widgets()

    def create_widgets(self):
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        logo_frame = ctk.CTkFrame(self.content_frame)
        logo_frame.pack(pady=20, expand=True)

        if not self.load_logo(logo_frame):
            ctk.CTkLabel(logo_frame, text="Logo not found", text_color="white", font=('Helvetica', 24)).pack(padx=10, pady=10)

        ctk.CTkLabel(logo_frame, text="SpaCr", text_color="#00BFFF", font=('Helvetica', 36, "bold")).pack(padx=10, pady=10)

        button = ctk.CTkButton(
            self.content_frame,
            text="Mask",
            command=self.load_mask_app,
            width=250,
            height=60,
            corner_radius=20,
            fg_color="#1E90FF",
            hover_color="#4682B4",
            text_color="white",
            font=("Helvetica", 18, "bold")
        )
        button.pack(pady=20)

    def load_logo(self, frame):
        def download_image(url, save_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image from {url}: {e}")
                return False

        try:
            img_path = os.path.join(os.path.dirname(__file__), 'logo_spacr.png')
            logo_image = Image.open(img_path)
        except (FileNotFoundError, Image.UnidentifiedImageError):
            if download_image('https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/logo_spacr.png', img_path):
                try:
                    logo_image = Image.open(img_path)
                except Image.UnidentifiedImageError as e:
                    return False
            else:
                return False
        except Exception as e:
            return False

        try:
            logo_image = logo_image.resize((200, 200), Image.Resampling.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ctk.CTkLabel(frame, image=logo_photo)
            logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            logo_label.pack()
            return True
        except Exception as e:
            return False

    def load_mask_app(self):
        print("Mask app loaded.")  # Placeholder for mask app loading function

def gui_app():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    gui_app()
