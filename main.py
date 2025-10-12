import threading
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from PIL import Image as PilImage, ImageTk
from torchvision import transforms
from PIL import Image
import sys
import os

# -------------------------------
# CONFIG - adjust paths if needed
# -------------------------------
MODEL_PATH = "model.pth"
START_BG_PATH = "assets/start_bg.png"
CAMERA_BG_PATH = "assets/camera_bg.png"

PRED_INTERVALS = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# CLASS NAMES
CLASS_NAMES = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

# MOOD DISPLAY NAMES
DISPLAY_NAMES = {
    'sad': "Sad",
    'disgust': "Disgust",
    'angry': "Angry",
    'neutral': "Neutral",
    'fear': "Fear",
    'surprise': "Surprised",
    'happy': "Happy"
}

MOOD_EMOJIS = {
    "Happy": "😄\n",
    "Sad": "😢\n",
    "Angry": "😡\n",
    "Surprised": "😲\n",
    "Neutral": "😐\n",
    "Disgust": "🤢\n",
    "Fear": "😨\n",
}

MOOD_COLORS = {
    "Happy": {"fg": "#2E7D32"},
    "Sad": {"fg": "#0D47A1"},
    "Angry": {"fg": "#B71C1C"},
    "Surprised": {"fg": "#F57F17"},
    "Neutral": {"fg": "#424242"},
    "Disgust": {"fg": "#33691E"},
    "Fear": {"fg": "#4A148C"},
}

# -------------------------------
# Face detector
# -------------------------------
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# Preprocessing pipelines
# -------------------------------
gray48_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Model loader (tries ViT then fallback)
# -------------------------------
def load_model(model_path):
    vit_ok = False
    model = None
    try:
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(CLASS_NAMES)
        )
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        vit_ok = True
        print("[Loader] Loaded HuggingFace ViT (strict=False).")
        return model, vit_ok
    except Exception as e:
        print(f"[Loader] ViT load failed or not appropriate: {e}. Falling back...")

    try:
        obj = torch.load(model_path, map_location=DEVICE)
        if hasattr(obj, "state_dict"):
            # loaded full model object
            model = obj.to(DEVICE).eval()
            print("[Loader] Loaded saved nn.Module directly.")
            return model, vit_ok
        else:
            import torch.nn as nn
            class TinyFER(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                    )
                    self.fc = nn.Linear(128, len(CLASS_NAMES))
                def forward(self, x):
                    x = self.net(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            model = TinyFER()
            model.load_state_dict(obj, strict=False)
            model = model.to(DEVICE).eval()
            print("[Loader] Loaded state_dict into TinyFER fallback.")
            return model, vit_ok
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_path}: {e}")

# -------------------------------
# Face detection + preprocess helpers
# -------------------------------
def detect_largest_face_bgr(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    return (x, y, w, h)

def preprocess_face_for_gray48(face_bgr):
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_48 = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(face_48)
    tens = gray48_transform(pil).unsqueeze(0).to(DEVICE)
    return tens

def preprocess_face_for_vit(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tens = vit_transform(pil).unsqueeze(0).to(DEVICE)
    return tens

def infer_logits(model, inp, vit_mode):
    with torch.no_grad():
        if vit_mode:
            out = model(inp)
            logits = out.logits
        else:
            logits = model(inp)
        return logits

# -------------------------------
# GUI + App (merging interface + inference)
# -------------------------------
class MoodCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodCam")
        self.root.geometry("700x580")
        self.root.resizable(False, False)

        # Load model
        print("[Info] Loading model...")
        self.model, self.vit_mode = load_model(MODEL_PATH)
        print(f"[Info] vit_mode={self.vit_mode}, device={DEVICE}")

        # Pages
        self.start_page = tk.Frame(root, width=700, height=580)
        self.camera_page = tk.Frame(root, width=700, height=580)
        for frame in (self.start_page, self.camera_page):
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.setup_start_page()
        self.setup_camera_page()

        # Camera / prediction variables
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
        self.current_frame = None
        self.running = False
        self.prediction_running = False
        self.last_prediction_label = "Detecting…"
        self.last_prediction_conf = 0.0
        self.analysis_interval = PRED_INTERVALS

        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
        self.prediction_thread.start()

        self.start_page.tkraise()

        # Bind escape to go back
        self.root.bind("<Escape>", lambda e: self.show_start_page())

    # ---------- Start page ----------
    def setup_start_page(self):
        self.start_canvas = tk.Canvas(self.start_page, width=700, height=580, highlightthickness=0)
        self.start_canvas.pack(fill="both", expand=True)
        # background image if available
        if os.path.exists(START_BG_PATH):
            try:
                bg = PilImage.open(START_BG_PATH).convert("RGBA").resize((700, 580), PilImage.LANCZOS)
                self.start_bg_img = ImageTk.PhotoImage(bg)
                self.start_canvas.create_image(0, 0, anchor=tk.NW, image=self.start_bg_img)
            except Exception as e:
                print("Warning: failed to load start bg:", e)
                self.start_canvas.create_rectangle(0,0,700,580, fill="#ffd2c7", outline="")
        else:
            self.start_canvas.create_rectangle(0,0,700,580, fill="#ffd2c7", outline="")


        start_btn = tk.Button(self.start_page, text="Start", font=("Segoe UI", 22, "bold"),
                              bg="#f6b98a", fg="#34204d", bd=0, activebackground="#f6b98a",
                              command=self.show_camera_page)
        self.start_canvas.create_window(500, 363, window=start_btn, anchor="center", width=214, height=94)

        self.start_canvas.create_text(500, 460, text="Detect your mood live using AI", font=("Segoe UI", 12),
                                      fill="#6b6b6b", anchor="center")

    # ---------- Camera page ----------
    def setup_camera_page(self):
        self.cam_canvas = tk.Canvas(self.camera_page, width=700, height=580, highlightthickness=0)
        self.cam_canvas.pack(fill="both", expand=True)
        if os.path.exists(CAMERA_BG_PATH):
            try:
                bg = PilImage.open(CAMERA_BG_PATH).convert("RGBA").resize((700, 580), PilImage.LANCZOS)
                self.camera_bg_img = ImageTk.PhotoImage(bg)
                self.cam_canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_bg_img)
            except Exception as e:
                print("Warning: failed to load camera bg:", e)
                self.cam_canvas.create_rectangle(0,0,700,580, fill="#f7f7f7", outline="")
        else:
            self.cam_canvas.create_rectangle(0,0,700,580, fill="#f7f7f7", outline="")

        # camera preview coordinates and size (as in your interface script)
        self.screen_x, self.screen_y = 130, 210
        self.screen_w, self.screen_h = 390, 250

        # Video canvas (we'll draw zoomed face here)
        self.video_canvas = tk.Canvas(self.camera_page, width=self.screen_w, height=self.screen_h, highlightthickness=0, bd=0)
        self.video_canvas.place(x=self.screen_x, y=self.screen_y)

        # Mood label centered near top of camera area (or bottom center per preference)
        self.mood_label = tk.Label(self.camera_page, text="Analyzing...", font=("Segoe UI", 18, "bold"),
                                   bg="#ffffff", fg="#333333")
        # place above the video area
        self.mood_label.place(x=320, y=160, anchor="center")

        # Back button at bottom center (keeps ability to return)
        back_btn = tk.Button(self.camera_page, text="Back", font=("Segoe UI", 18, "bold"),
                             bg="#f6b98a", fg="#34204d", bd=0, activebackground="#f6b98a",
                             command=self.show_start_page)
        self.cam_canvas.create_window(350, 530, window=back_btn, anchor="center", width=180, height=72)

    # ---------- Page control ----------
    def show_start_page(self):
        self.running = False
        self.start_page.tkraise()

    def show_camera_page(self):
        self.running = True
        self.camera_page.tkraise()
        self.update_video()

    # ---------- Video update (smooth UI) ----------
    def update_video(self):
        if not self.running:
            return
        ret, frame = self.video_capture.read()
        if not ret:
            self.root.after(50, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()

        # detect largest face to create zoomed view
        rect = detect_largest_face_bgr(frame)
        if rect is not None:
            x, y, w, h = rect
            pad = int(0.18 * max(w, h))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
            roi = frame[y0:y1, x0:x1]
            # visualize rectangle on an overlay of full frame? We choose to show zoomed face in the preview
            display_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            # no face: show a resized full-frame
            display_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize display to preview area
        pil = PilImage.fromarray(display_img).resize((self.screen_w, self.screen_h), PilImage.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=pil)
        # Clear previous and draw
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.image = imgtk

        # Update mood label with last known prediction
        label_text = f"{MOOD_EMOJIS.get(self.last_prediction_label, '🧠')}  {self.last_prediction_label} ({self.last_prediction_conf*100:.1f}%)" if self.last_prediction_label != "Detecting…" else "Analyzing..."
        self.mood_label.config(text=label_text)
        # color
        color_fg = MOOD_COLORS.get(self.last_prediction_label, {"fg": "#333333"})["fg"]
        self.mood_label.config(fg=color_fg)

        # schedule next frame
        self.root.after(15, self.update_video)

    # ---------- Background prediction loop ----------
    def prediction_loop(self):
        last_pred_time = 0.0
        while True:
            if self.running and self.current_frame is not None and not self.prediction_running:
                # detect face again on current_frame to choose crop
                self.prediction_running = True
                try:
                    frame = self.current_frame.copy()
                    rect = detect_largest_face_bgr(frame)
                    if rect is not None:
                        x, y, w, h = rect
                        pad = int(0.18 * max(w, h))
                        x0 = max(0, x - pad); y0 = max(0, y - pad)
                        x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
                        roi = frame[y0:y1, x0:x1]
                    else:
                        roi = None

                    now = time.time()
                    if roi is not None and now - last_pred_time >= self.analysis_interval:
                        # preprocess according to model type
                        if self.vit_mode:
                            inp = preprocess_face_for_vit(roi)
                        else:
                            inp = preprocess_face_for_gray48(roi)
                        logits = infer_logits(self.model, inp, self.vit_mode)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                        idx = int(np.argmax(probs))
                        class_key = CLASS_NAMES[idx]
                        display_name = DISPLAY_NAMES.get(class_key, class_key)
                        self.last_prediction_label = display_name
                        self.last_prediction_conf = float(probs[idx])
                        last_pred_time = now
                    elif roi is None:
                        # no face: optional fallback
                        self.last_prediction_label = "No face"
                        self.last_prediction_conf = 0.0
                except Exception as e:
                    print("Prediction thread error:", e)
                finally:
                    self.prediction_running = False
            time.sleep(self.analysis_interval / 2.0)

    # ---------- Clean exit ----------
    def close(self):
        self.running = False
        try:
            if self.video_capture is not None:
                self.video_capture.release()
        except Exception:
            pass
        self.root.destroy()


# -------------------------------
# Run application
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MoodCamApp(root)
    # start prediction background loop thread
    pred_thread = threading.Thread(target=app.prediction_loop, daemon=True)
    pred_thread.start()
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
