#!/usr/bin/env python3
"""
Egg detector - pull image from Pi, detect eggs via YOLO, log results.

Usage:
  python detect.py                        # pull from Pi and detect
  python detect.py --image path/to/img    # use local image (skip SCP)
  python detect.py --image img --debug    # print per-detection details
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json

import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Image pull
# ---------------------------------------------------------------------------

def pull_image(cfg) -> Path | None:
    pi      = cfg["pi"]
    ssh_key = Path(cfg["ssh_key"]).expanduser()
    remote  = f"{pi['user']}@{pi['host']}:{pi['image_path']}"
    dest    = Path(cfg["snapshot_dir"]) / f"coop_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pull] {remote} → {dest}")
    result = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-i", str(ssh_key), "-P", str(pi.get("port", 22)), remote, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[pull] FAILED: {result.stderr.strip()}", file=sys.stderr)
        return None

    print(f"[pull] OK ({dest.stat().st_size // 1024}KB)")
    return dest

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

def detect_yolo(image_path: Path, model_path: str, debug: bool) -> tuple[int, "np.ndarray"]:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    model   = YOLO(model_path)
    results = model(str(image_path), verbose=False)
    result  = results[0]
    img     = cv2.imread(str(image_path))

    detections = []
    for box in result.boxes:
        cls        = int(box.cls)
        label      = model.names[cls]
        confidence = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((label, confidence, x1, y1, x2, y2))
        if debug:
            print(f"  [{label}] conf={confidence:.2f}  box=({x1},{y1})-({x2},{y2})")

    # Count only eggs; other classes (e.g. "chicken") are drawn but not counted
    eggs     = [(c, x1, y1, x2, y2) for label, c, x1, y1, x2, y2 in detections if label == "egg"]
    non_eggs = [(l, c, x1, y1, x2, y2) for l, c, x1, y1, x2, y2 in detections if l != "egg"]
    count    = len(eggs)

    annotated = img.copy()
    for conf, x1, y1, x2, y2 in eggs:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"egg {conf:.0%}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for label, conf, x1, y1, x2, y2 in non_eggs:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    label_text = f"{count} egg{'s' if count != 1 else ''} detected  {datetime.now():%H:%M:%S}  [YOLO]"
    cv2.putText(annotated, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if count else (0, 0, 255), 2)

    return count, annotated

# ---------------------------------------------------------------------------
# Snapshot cleanup
# ---------------------------------------------------------------------------

def cleanup_snapshots(snapshot_dir: Path, max_count: int):
    snapshots = sorted(snapshot_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    to_delete = snapshots[max_count:]
    if to_delete:
        for f in to_delete:
            f.unlink()
        print(f"[cleanup] Removed {len(to_delete)} old snapshots")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect eggs in coop image")
    parser.add_argument("--image", help="Use a local image instead of pulling from Pi")
    parser.add_argument("--debug", action="store_true", help="Print per-detection details")
    args = parser.parse_args()

    cfg = load_config()
    snapshot_dir = Path(cfg["snapshot_dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Get image
    if args.image:
        image_path = Path(args.image)
        print(f"[info] Using local image: {image_path}")
    else:
        image_path = pull_image(cfg)
        if image_path is None:
            sys.exit(1)

    # Detect with YOLO
    model_path = cfg.get("model_path")
    if not model_path or not Path(model_path).exists():
        print(f"[error] YOLO model not found: {model_path}", file=sys.stderr)
        print("[error] Train a model first: yolo train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640", file=sys.stderr)
        sys.exit(1)

    print(f"[detect] Using YOLO model: {model_path}")
    count, annotated = detect_yolo(image_path, model_path, args.debug)

    # Save annotated image
    annotated_path = snapshot_dir / f"annotated_{image_path.stem}.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    cleanup_snapshots(snapshot_dir, cfg.get("max_snapshots", 10))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if count > 0:
        print(f"[{ts}] 🥚 {count} egg{'s' if count != 1 else ''} detected → {annotated_path}")
    else:
        print(f"[{ts}] No eggs detected → {annotated_path}")

if __name__ == "__main__":
    main()
