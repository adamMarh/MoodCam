import threading
import time
import cv2
import torch
import tkinter as tk
from PIL import Image as PilImage, ImageTk
from torchvision import transforms
from transformers import ViTForImageClassification

# -------------------------------
# Device setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# -------------------------------
# Load ViT model
# -------------------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=7
)
state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# -------------------------------
# Mood labels
# -------------------------------
MOODS = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Disgust", "Fear"]

# -------------------------------
# Preprocessing
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
        self.root.title("MoodCam (Smooth ViT)")
        self.root.geometry("700x580")
        self.root.configure(bg="#ececec")

        # ====== Pages ======
        self.start_page = tk.Frame(root, bg="#ececec")
        self.camera_page = tk.Frame(root, bg="#ececec")

        for frame in (self.start_page, self.camera_page):
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        # ====== Start Page ======
        tk.Label(
            self.start_page,
            text="Welcome to MoodCam!",
            font=("Arial", 24, "bold"),
            bg="#ececec"
        ).pack(pady=100)

        tk.Button(
            self.start_page,
            text="Start",
            font=("Arial", 16, "bold"),
            bg="#4caf50",
            fg="white",
            padx=20,
            pady=10,
            command=self.show_camera_page
        ).pack()

        tk.Label(
            self.start_page,
            text="Detect your mood live using AI!",
            font=("Arial", 14),
            bg="#ececec",
            fg="#555"
        ).pack(pady=20)

        self.start_page.tkraise()

        # ====== Camera Page ======
        self.canvas = tk.Canvas(self.camera_page, width=640, height=480)
        self.canvas.pack(pady=10)

        self.mood_label = tk.Label(
            self.camera_page,
            text="Your mood will appear here",
            font=("Arial", 18, "bold"),
            bg="#ececec"
        )
        self.mood_label.pack(pady=10)

        tk.Button(
            self.camera_page,
            text="Back",
            font=("Arial", 14),
            bg="#f44336",
            fg="white",
            command=self.show_start_page
        ).pack(pady=5)

        # ====== Camera Setup ======
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.running = False
        self.prediction_running = False
        self.last_prediction = "Analyzing..."
        self.analysis_interval = 1.0  # seconds

        # Start background prediction thread
        self.prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
        self.prediction_thread.start()

    # ============================================================
    # Page Control
    # ============================================================
    def show_start_page(self):
        self.running = False
        self.start_page.tkraise()

    def show_camera_page(self):
        self.running = True
        self.camera_page.tkraise()
        self.update_video()

    # ============================================================
    # Video Feed (Smooth UI)
    # ============================================================
    def update_video(self):
        if not self.running:
            return
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror view
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb

            img = PilImage.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

            # Update mood label (from last prediction)
            self.mood_label.config(text=f"Mood: {self.last_prediction}")

        # Schedule next frame
        self.root.after(15, self.update_video)

    # ============================================================
    # Background Prediction Loop (Non-blocking)
    # ============================================================
    def prediction_loop(self):
        while True:
            if self.running and self.current_frame is not None and not self.prediction_running:
                self.prediction_running = True
                threading.Thread(target=self.analyze_frame, daemon=True).start()
            time.sleep(self.analysis_interval)

    def analyze_frame(self):
        try:
            frame = self.current_frame
            img = PilImage.fromarray(frame)
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                pred_idx = torch.argmax(outputs.logits, dim=1).item()
                self.last_prediction = MOODS[pred_idx]
        except Exception as e:
            print("Prediction error:", e)
        finally:
            self.prediction_running = False

    # ============================================================
    # Clean Exit
    # ============================================================
    def close(self):
        self.running = False
        self.video_capture.release()
        self.root.destroy()


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MoodCamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
