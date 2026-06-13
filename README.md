# Egg Detector

Detects eggs in a chicken coop. Pulls a snapshot from a Raspberry Pi over SSH,
runs a trained YOLO model to find eggs, and saves an annotated image with a count.

A trained model (`models/best.pt`) is included, so it works out of the box once
configured.

```
Raspberry Pi (camera)  ──scp──▶  detect.py  ──YOLO──▶  snapshots/annotated_*.jpg
```

## Setup

```bash
# 1. Create a virtualenv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Create your config from the template
cp config.example.json config.json

# 3. Set up an SSH key for the Pi (skip if you already have one)
ssh-keygen -t ed25519 -f ~/.ssh/pi_key -N ""
ssh-copy-id -i ~/.ssh/pi_key.pub pi@<pi-ip>

# 4. Edit config.json — set the Pi host/user and the image path on the Pi
$EDITOR config.json
```

`config.json` holds your Pi credentials and is gitignored — keep it out of
version control. See `config.example.json` for the expected shape.

## Usage

```bash
# Pull the latest snapshot from the Pi and detect
python detect.py

# Test against a local image (no Pi needed)
python detect.py --image path/to/test.jpg

# Print per-detection details (label, confidence, box)
python detect.py --image path/to/test.jpg --debug
```

Annotated images are written to `snapshots/annotated_*.jpg` with bounding boxes
and an egg count. Eggs are drawn in green; any other detected class (e.g.
`chicken`) is drawn in orange but not counted. The `snapshots/` directory is
trimmed to the most recent `max_snapshots` images.

## How it works

`detect.py` runs three phases:

1. **Acquire** — `scp` the latest snapshot from the Pi (or use `--image`).
2. **Detect** — run the YOLO model from `config.json`'s `model_path`.
3. **Output** — save the annotated image and log the count.

## Training your own model

The included model was trained on coop images annotated in
[Label Studio](https://labelstud.io/). To retrain on your own data:

```bash
# 1. Export annotations from Label Studio (YOLO format) into label-studio-project/
#    and put the source images in old-snaps/.

# 2. Convert the export into a YOLO dataset (80/20 train/val split)
python prepare_dataset.py

# 3. Train
yolo train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640

# 4. Point config.json at the new weights
#    "model_path": "runs/detect/train/weights/best.pt"
#    (or copy them to models/best.pt)
```

Training data (`old-snaps/`, `dataset/`, `label-studio-project/`) and training
runs (`runs/`) are gitignored — they're large and specific to your setup.

> **Note:** the model's class list is `["Airplane", "Car", "chicken", "egg"]`.
> `Airplane`/`Car` are leftover defaults from the annotation template that were
> never removed. They're baked into the trained weights, so the class list is
> kept as-is — only the `egg` class is counted.

## Optional: OpenCV heuristic detector

`detect_opencv.py` is a standalone, no-training detector that finds eggs with
HSV color filtering plus Hough circle detection. It predates the YOLO model and
isn't wired into `detect.py`, but it's kept for experimentation or environments
without trained weights. It expects a `detection` block in the config, e.g.:

```json
"detection": {
  "min_area": 500,
  "max_area": 8000,
  "min_circularity": 0.6,
  "blur_kernel": 5,
  "canny_low": 30,
  "canny_high": 100,
  "color_ranges": [
    { "lower": [0, 0, 180], "upper": [180, 40, 255] }
  ],
  "use_rois": false,
  "rois": [{ "x": 100, "y": 150, "width": 300, "height": 200 }]
}
```

Lower `min_circularity` to catch more eggs; raise it to cut false positives.
Set `use_rois: true` with rectangles covering just the nest box to ignore
background clutter.

## Docker

```bash
docker build -t egg-detector .
docker run --rm \
  -v "$PWD/snapshots:/app/snapshots" \
  -v "$HOME/.ssh:/root/.ssh:ro" \
  egg-detector
```

The container runs detection every `INTERVAL` seconds (default 900). The image
bakes in `config.example.json` as `config.json` — mount your own `config.json`
over it (`-v "$PWD/config.json:/app/config.json:ro"`) to point at your Pi.

## Requirements

Python 3.12, plus the packages in `requirements.txt` (opencv-python, numpy,
ultralytics; label-studio is only needed for annotation, not at runtime).
