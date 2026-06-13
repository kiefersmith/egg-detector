#!/usr/bin/env python3
"""
OpenCV heuristic egg detection — standalone module.

Previously part of detect.py; extracted when YOLO became the sole detection
method. Kept available for experimentation or environments without a trained
YOLO model.

Usage:
    from detect_opencv import detect_opencv
    count, annotated = detect_opencv(image_path, cfg, debug=False)
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def apply_roi(img: np.ndarray, rois: list) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for r in rois:
        x, y, w, h = r["x"], r["y"], r["width"], r["height"]
        mask[y:y+h, x:x+w] = 255
    return cv2.bitwise_and(img, img, mask=mask)


def in_any_roi(cx, cy, rois):
    for r in rois:
        if r["x"] <= cx <= r["x"] + r["width"] and r["y"] <= cy <= r["y"] + r["height"]:
            return True
    return False


def detect_opencv(image_path: Path, cfg: dict, debug: bool) -> tuple[int, np.ndarray]:
    det      = cfg["detection"]
    img      = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    use_rois = det.get("use_rois") and det.get("rois")
    working  = apply_roi(img, det["rois"]) if use_rois else img.copy()

    # Preprocessing
    gray     = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (det["blur_kernel"], det["blur_kernel"]), 0)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Method 1: color detection
    hsv        = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros(gray.shape, dtype=np.uint8)
    for cr in det["color_ranges"]:
        lo         = np.array(cr["lower"], dtype=np.uint8)
        hi         = np.array(cr["upper"], dtype=np.uint8)
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lo, hi))

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (det["min_area"] <= area <= det["max_area"]):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < det["min_circularity"]:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if debug:
            print(f"  [color] center=({int(cx)},{int(cy)}) r={int(radius)} area={int(area)} circ={circularity:.2f}")
        color_detections.append((int(cx), int(cy), int(radius)))

    # Method 2: Hough circles
    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=det["canny_high"], param2=det["canny_low"],
        minRadius=int(np.sqrt(det["min_area"] / np.pi)),
        maxRadius=int(np.sqrt(det["max_area"] / np.pi)),
    )
    hough_detections = []
    if circles is not None:
        for x, y, r in circles[0]:
            if debug:
                print(f"  [hough] center=({int(x)},{int(y)}) r={int(r)}")
            hough_detections.append((int(x), int(y), int(r)))

    # Filter: discard anything whose center is outside all ROIs
    if use_rois:
        color_detections = [(cx, cy, r) for cx, cy, r in color_detections if in_any_roi(cx, cy, det["rois"])]
        hough_detections = [(cx, cy, r) for cx, cy, r in hough_detections if in_any_roi(cx, cy, det["rois"])]

    # Merge: deduplicate by proximity
    all_detections = list(color_detections)
    for hx, hy, hr in hough_detections:
        too_close = any(
            np.hypot(hx - cx, hy - cy) < (hr + cr) * 0.6
            for cx, cy, cr in all_detections
        )
        if not too_close:
            all_detections.append((hx, hy, hr))

    # Annotate
    annotated = img.copy()
    for i, (cx, cy, r) in enumerate(all_detections, 1):
        cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(annotated, str(i), (cx - 10, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if use_rois:
        for roi in det["rois"]:
            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 165, 0), 2)

    count      = len(all_detections)
    label_text = f"{count} egg{'s' if count != 1 else ''} detected  {datetime.now():%H:%M:%S}  [OpenCV]"
    cv2.putText(annotated, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if count else (0, 0, 255), 2)

    return count, annotated
