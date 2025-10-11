from tkinter import *
from PIL import Image, ImageTk
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 1. Define the Model Architecture
# ==================================================
class MoodModel(nn.Module):
    def __init__(self):
        super(MoodModel, self).__init__()
        # Simple CNN for grayscale 48x48 images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 mood classes

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # (48 → 24)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # (24 → 12)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==================================================
# 2. Load Model and Labels
# ==================================================
MOODS = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Disgust", "Fear"]

model = MoodModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()


# ==================================================
# 3. Main Application
# ==================================================
class MoodCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodCam")
        self.root.geometry("700x580")
        self.root.resizable(False, False)

        # Define frames (pages)
        self.start_page = Frame(root, bg="#ececec")
        self.camera_page = Frame(root, bg="#ececec")

        for frame in (self.start_page, self.camera_page):
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        # ---------------- START PAGE ----------------
        Label(
            self.start_page,
            text="Welcome to MoodCam!",
            font=("Arial", 24, "bold"),
            bg="#ececec",
            fg="#333"
        ).pack(pady=100)

        Button(
            self.start_page,
            text="Start",
            font=("Arial", 16, "bold"),
            bg="#4caf50",
            fg="white",
            padx=20,
            pady=10,
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

        # ---------------- CAMERA PAGE ----------------
        self.canvas = Canvas(self.camera_page, width=640, height=480)
        self.canvas.pack(pady=10)

        self.mood_label = Label(
            self.camera_page,
            text="Your mood will appear here",
            font=("Arial", 18, "bold"),
            bg="#ececec",
            fg="#333"
        )
        self.mood_label.pack(pady=10)

        Button(
            self.camera_page,
            text="Back",
            font=("Arial", 14),
            bg="#f44336",
            fg="white",
            padx=10,
            command=self.show_start_page
        ).pack(pady=5)

        # ---------------- VIDEO CAPTURE ----------------
        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None

        # Timing for frame analysis
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # seconds

    # ==================================================
    # PAGE CONTROL
    # ==================================================
    def show_start_page(self):
        self.start_page.tkraise()

    def show_camera_page(self):
        self.camera_page.tkraise()
        self.update_frame()

    # ==================================================
    # CAMERA LOOP
    # ==================================================
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=self.current_image)

            self.canvas.create_image(0, 0, image=photo, anchor=NW)
            self.canvas.image = photo

            # Analyze every 0.5 seconds
            current_time = time.time()
            if current_time - self.last_analysis_time >= self.analysis_interval:
                self.analyze_frame(frame)
                self.last_analysis_time = current_time

        self.root.after(15, self.update_frame)

    # ==================================================
    # FRAME ANALYSIS
    # ==================================================
    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            outputs = model(tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            mood = MOODS[predicted_idx]

        self.mood_label.config(text=f"Mood: {mood}")

    # ==================================================
    # CLEAN EXIT
    # ==================================================
    def close(self):
        self.video_capture.release()
        self.root.destroy()


# ==================================================
# 4. Run the Application
# ==================================================
if __name__ == "__main__":
    root = Tk()
    app = MoodCamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
