from transformers import ViTForImageClassification
import torch
from PIL import Image, ImageTk
from torchvision import transforms
import cv2
import time
from tkinter import *

# -------------------------------
# Load ViT model architecture
# -------------------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=7  # adjust this to your training setup
)

# Load your trained weights (state_dict)
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# -------------------------------
# Mood labels
# -------------------------------
MOODS = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Disgust", "Fear"]

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# -------------------------------
# MoodCam Application
# -------------------------------
class MoodCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodCam (ViT Model)")
        self.root.geometry("700x580")
        self.root.configure(bg="#ececec")

        self.start_page = Frame(root, bg="#ececec")
        self.camera_page = Frame(root, bg="#ececec")

        for frame in (self.start_page, self.camera_page):
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        Label(
            self.start_page,
            text="Welcome to MoodCam!",
            font=("Arial", 24, "bold"),
            bg="#ececec"
        ).pack(pady=100)

        Button(
            self.start_page,
            text="Start",
            font=("Arial", 16),
            bg="#4caf50",
            fg="white",
            command=self.show_camera_page
        ).pack()

        Label(
            self.start_page,
            text="Detect your mood live using AI!",
            font=("Arial", 14),
            bg="#ececec",
            fg="#555"
        ).pack(pady=20)

        self.start_page.tkraise()

        # Camera page
        self.canvas = Canvas(self.camera_page, width=640, height=480)
        self.canvas.pack(pady=10)

        self.mood_label = Label(
            self.camera_page,
            text="Your mood will appear here",
            font=("Arial", 18, "bold"),
            bg="#ececec"
        )
        self.mood_label.pack(pady=10)

        Button(
            self.camera_page,
            text="Back",
            font=("Arial", 14),
            bg="#f44336",
            fg="white",
            command=self.show_start_page
        ).pack(pady=5)

        self.video_capture = cv2.VideoCapture(0)
        self.last_analysis_time = 0
        self.analysis_interval = 0.5

    def show_start_page(self):
        self.start_page.tkraise()

    def show_camera_page(self):
        self.camera_page.tkraise()
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=photo, anchor=NW)
            self.canvas.image = photo

            current_time = time.time()
            if current_time - self.last_analysis_time >= self.analysis_interval:
                self.analyze_frame(frame)
                self.last_analysis_time = current_time

        self.root.after(15, self.update_frame)

    def analyze_frame(self, frame):
        # Convert OpenCV frame to PIL and preprocess
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            mood = MOODS[pred_idx]

        self.mood_label.config(text=f"Mood: {mood}")

    def close(self):
        self.video_capture.release()
        self.root.destroy()


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    root = Tk()
    app = MoodCamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
