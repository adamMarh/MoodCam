# MoodCam

MoodCam is a small desktop computer-vision app that uses a webcam to estimate the facial expression of the largest detected face. It displays one of seven classes:

`sad`, `disgust`, `angry`, `neutral`, `fear`, `surprise`, or `happy`.

The app is a prototype for experimenting with live face detection and image-classification inference. It is not a psychological or medical assessment.

## How it works

1. OpenCV reads frames from webcam device `0`.
2. An OpenCV Haar cascade finds faces and selects the largest one.
3. The face crop is sent to the model roughly every 0.5 seconds.
4. Tkinter shows the live preview, predicted expression, and confidence.

At startup, MoodCam first tries to load the checkpoint as a fine-tuned Hugging Face ViT model. If that fails, it tries the grayscale `TinyFER` CNN fallback defined in `main.py`.

## Requirements

- Python 3.10 or newer
- A working webcam available as camera index `0`
- A compatible `model.pth` checkpoint in the repository root
- Tkinter, which is included with many Python installations but may need to be installed separately on Linux

The Python packages used by the app are:

- `torch`
- `torchvision`
- `transformers`
- `opencv-python`
- `Pillow`
- `numpy`

## Installation

Create and activate a virtual environment from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision transformers opencv-python Pillow numpy
```

For Ubuntu/Debian, install Tkinter if `import tkinter` fails:

```bash
sudo apt install python3-tk
```

PyTorch installation can vary by operating system and CUDA version. If you want GPU inference, use the installation command recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/). CPU inference works without CUDA.

## Model checkpoint

The repository does not include `model.pth`. Add a compatible checkpoint at:

```text
MoodCam/
	model.pth
	main.py
	assets/
```

The checkpoint must produce the seven classes in this order:

```text
sad, disgust, angry, neutral, fear, surprise, happy
```

For a ViT checkpoint, the model should be compatible with `google/vit-base-patch16-224-in21k`; the first run may download that base model from Hugging Face. For the fallback path, `model.pth` must be a PyTorch state dictionary compatible with the `TinyFER` architecture in `main.py`, or a saved PyTorch module.

## Run

With the virtual environment activated and `model.pth` in place:

```bash
python main.py
```

Click **Start** to open the camera view. Press **Escape** or click **Back** to return to the start screen. Close the window to stop the webcam.

## Optional: build an executable

The repository includes a PyInstaller specification. Install PyInstaller and build it with:

```bash
python -m pip install pyinstaller
pyinstaller main.spec
```

Before distributing the generated executable, make sure `model.pth` and the `assets/` directory are included beside it. The current spec is a starting point and may need additional `datas` or hidden imports depending on the model and PyTorch/Transformers versions.

## Troubleshooting

- **`Could not load model from model.pth`**: confirm the file exists in the repository root and matches one of the supported checkpoint formats.
- **The window reports that it cannot open the webcam**: check camera permissions, close other camera applications, or update `cv2.VideoCapture(0)` in `main.py` if the camera uses another index.
- **Tkinter import errors**: install the platform-specific Tk package, such as `python3-tk` on Ubuntu/Debian.
- **No face is detected**: improve lighting, face the camera, and move closer. The app uses a frontal-face Haar cascade.
- **Unexpected predictions**: results depend on the checkpoint's training data, lighting, camera angle, and calibration.

## Project files

```text
main.py         Application, camera loop, face detection, and inference
main.spec       PyInstaller build configuration
assets/         Start and camera screen background images
model.pth       Local model checkpoint; not included in the repository
```
