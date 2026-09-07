# MoodCam

MoodCam is a desktop computer-vision prototype that reads a webcam stream, isolates the largest detected face, and predicts one of seven facial-expression classes: happy, sad, angry, disgust, fear, surprise, or neutral.

## Features

- OpenCV Haar-cascade face detection
- ViT inference with a lightweight CNN/state-dict fallback
- CPU/GPU device selection through PyTorch
- Normalized 48×48 grayscale or 224×224 RGB preprocessing
- Background prediction thread with a Tkinter interface
- Confidence and mood visualization with themed assets

## Requirements

Python 3.10+ is recommended. Install the model/runtime dependencies used by the source, including `torch`, `torchvision`, `transformers`, `opencv-python`, `Pillow`, and `numpy`. Place a compatible `model.pth` at the project root.

## Run

```bash
python main.py
```

A webcam must be available as camera index `0`. Press Escape to return to the start screen.

## Notes

Mood classification is experimental and should not be treated as a psychological or medical assessment. Model quality depends on the training data, lighting, camera angle, and calibration.
