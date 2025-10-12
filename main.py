import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# =========================
# Config (edit if needed)
# =========================
MODEL_PATH     = "model.pth"
PRED_INTERVALS = 0.5   # seconds between predictions
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Class order EXACTLY as requested
CLASS_NAMES = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

# OpenCV face detector (frontal face)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# Two preprocessing pipelines:
#   A) grayscale 48x48 (1ch)  -> for CNNs trained on FER-style images
#   B) ViT 224x224 (3ch RGB)  -> if your state_dict fits ViT
# -------------------------------
gray48_transform = transforms.Compose([
    transforms.ToTensor(),                         # HxWxC -> CxHxW in [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])    # match your note: mean=0.5 std=0.5
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                         # [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Model loader that adapts to your .pth
# -------------------------------
def load_model(model_path):
    """
    Try to load as HuggingFace ViT first (common for the Kaggle base),
    otherwise assume a plain torch nn.Module that expects 1x48x48.
    """
    vit_ok, model = False, None
    try:
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(CLASS_NAMES)
        )
        state_dict = torch.load(model_path, map_location=DEVICE)
        # Accept partial keys to be tolerant to head names, etc.
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE).eval()
        vit_ok = True
        print("[Loader] Loaded ViTForImageClassification with strict=False.")
        return model, vit_ok
    except Exception as e:
        print(f"[Loader] ViT load failed ({e}). Falling back to raw torch model...")

    # Fallback: raw torch model saved via torch.save(model.state_dict()) or torch.save(model)
    try:
        obj = torch.load(model_path, map_location=DEVICE)
        if hasattr(obj, "state_dict"):   # a full nn.Module saved
            model = obj.to(DEVICE).eval()
            print("[Loader] Loaded raw nn.Module from checkpoint.")
        else:
            # If it's a state_dict, you must have the architecture code available.
            # Minimal generic head for 1x48x48 -> 7 classes (best effort).
            # NOTE: Replace this with your exact net if you saved only state_dict.
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
                    x = self.net(x)       # [B,128,1,1]
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            model = TinyFER()
            model.load_state_dict(obj, strict=False)
            model = model.to(DEVICE).eval()
            print("[Loader] Loaded state_dict into TinyFER fallback. You should replace with your exact arch.")
        return model, vit_ok
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_path}: {e}")

# -------------------------------
# Face utilities
# -------------------------------
def detect_largest_face_bgr(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # pick largest by area
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    return (x, y, w, h)

def preprocess_face_for_gray48(face_bgr):
    # to gray 48x48 in [0,1], normalized mean=0.5 std=0.5
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_48 = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(face_48)  # single channel image
    tens = gray48_transform(pil).unsqueeze(0).to(DEVICE)  # [1,1,48,48]
    return tens

def preprocess_face_for_vit(face_bgr):
    # ViT expects 3ch RGB 224x224 normalized
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tens = vit_transform(pil).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    return tens

# -------------------------------
# Inference helpers
# -------------------------------
def infer_logits(model, inp, vit_mode):
    with torch.no_grad():
        if vit_mode:
            # HF ViT returns an object with .logits
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(inp)
                logits = out.logits
        else:
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(inp)
        return logits

# -------------------------------
# Main
# -------------------------------
def main():
    print(f"[Info] Using device: {DEVICE}")
    model, vit_mode = load_model(MODEL_PATH)
    last_pred_time = 0.0
    last_label = "…"
    last_conf  = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("[Info] Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for natural webcam feel
        frame = cv2.flip(frame, 1)

        # Detect largest face
        roi = None
        rect = detect_largest_face_bgr(frame)
        if rect is not None:
            x, y, w, h = rect
            # slightly enlarge box for context
            pad = int(0.15 * max(w, h))
            x0 = max(x - pad, 0); y0 = max(y - pad, 0)
            x1 = min(x + w + pad, frame.shape[1]); y1 = min(y + h + pad, frame.shape[0])
            roi = frame[y0:y1, x0:x1]

            # draw rectangle
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Throttle predictions to every 500 ms
        now = time.time()
        if roi is not None and now - last_pred_time >= PRED_INTERVALS:
            # Choose preprocessing path
            if vit_mode:
                inp = preprocess_face_for_vit(roi)
            else:
                inp = preprocess_face_for_gray48(roi)

            logits = infer_logits(model, inp, vit_mode)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            last_label = CLASS_NAMES[idx]
            last_conf  = float(probs[idx])
            last_pred_time = now

        # Put label
        text = f"{last_label} ({last_conf*100:.1f}%)" if last_label != "…" else "Detecting…"
        # Draw above face or top-left
        org = (10, 30)
        if rect is not None:
            org = (rect[0], max(0, rect[1]-10))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2, cv2.LINE_AA)

        cv2.imshow("MoodCam — Face Emotion (500ms refresh)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
