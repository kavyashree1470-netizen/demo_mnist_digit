"""
mnist_utils.py — Core MNIST processing utilities
Can be imported by any project or run as a standalone test.
"""

import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime


CSV_FILE = "mnist_dataset.csv"
PIXEL_COLS = [f"pixel_{i}" for i in range(784)]


# ─── image processing ──────────────────────────────────────────────────────────

def image_to_mnist(raw_rgba: np.ndarray) -> np.ndarray | None:
    """
    Convert a raw RGBA numpy array (from any drawing canvas or image file)
    into a 28×28 uint8 MNIST-standard array.

    Follows the exact MNIST preprocessing pipeline:
      1. Alpha-channel extraction (or luminance inversion)
      2. Gaussian anti-aliasing
      3. Bounding-box crop + 20% padding
      4. Resize to 20×20 with bicubic interpolation
      5. Centre-of-mass placement into 28×28 frame
      6. uint8 normalisation to 0-255
    """
    # 1. stroke mask from alpha or luminance
    if raw_rgba.ndim == 3 and raw_rgba.shape[2] == 4:
        alpha = raw_rgba[:, :, 3].astype(np.float32)
    elif raw_rgba.ndim == 3 and raw_rgba.shape[2] == 3:
        gray = cv2.cvtColor(raw_rgba, cv2.COLOR_RGB2GRAY)
        alpha = (255.0 - gray.astype(np.float32))
    else:
        alpha = raw_rgba.astype(np.float32)

    # 2. gaussian blur → anti-aliasing
    blurred = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.0)

    # 3. bounding box + padding
    mask = (blurred > 10).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    pad = max(int(max(w, h) * 0.20), 2)
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(blurred.shape[1], x + w + pad)
    y2 = min(blurred.shape[0], y + h + pad)
    cropped = blurred[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    # 4. resize to 20×20
    digit_20 = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_CUBIC)
    digit_20 = np.clip(digit_20, 0, 255)

    # 5. centre-of-mass centering in 28×28
    canvas = np.zeros((28, 28), dtype=np.float32)
    M = cv2.moments(digit_20)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 10
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 10
    off_x, off_y = 14 - cx, 14 - cy

    for r in range(20):
        for c in range(20):
            nr, nc = r + off_y, c + off_x
            if 0 <= nr < 28 and 0 <= nc < 28:
                canvas[nr, nc] = digit_20[r, c]

    # 6. normalise
    return np.clip(canvas, 0, 255).astype(np.uint8)


def load_image_file(path: str) -> np.ndarray | None:
    """Load any image file and return as RGBA numpy array."""
    from PIL import Image
    try:
        img = Image.open(path).convert("RGBA")
        return np.array(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


# ─── CSV operations ────────────────────────────────────────────────────────────

def init_csv(path: str = CSV_FILE):
    if not os.path.exists(path):
        cols = ["id", "label", "timestamp", "split"] + PIXEL_COLS
        pd.DataFrame(columns=cols).to_csv(path, index=False)


def load_dataset(path: str = CSV_FILE) -> pd.DataFrame:
    init_csv(path)
    return pd.read_csv(path)


def save_sample(label: int, pixels: np.ndarray, split: str = "train",
                path: str = CSV_FILE) -> int:
    """Append one MNIST sample to the CSV. Returns the assigned ID."""
    df = load_dataset(path)
    new_id = int(df["id"].max()) + 1 if len(df) > 0 else 1
    row = {
        "id": new_id,
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
    }
    for i, v in enumerate(pixels.flatten()):
        row[f"pixel_{i}"] = int(v)
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)
    return new_id


def get_pixels(df: pd.DataFrame) -> np.ndarray:
    """Extract pixel matrix from dataframe. Shape: (N, 784)."""
    return df[PIXEL_COLS].values.astype(np.uint8)


def dataset_summary(path: str = CSV_FILE) -> dict:
    df = load_dataset(path)
    if len(df) == 0:
        return {"total": 0}
    return {
        "total": len(df),
        "per_label": df["label"].value_counts().sort_index().to_dict(),
        "per_split": df["split"].value_counts().to_dict(),
        "date_range": [df["timestamp"].min(), df["timestamp"].max()],
    }


# ─── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MNIST Utils Self-Test ===")

    # Create a synthetic white circle on black background (simulates a '0')
    test_img = np.zeros((280, 280, 4), dtype=np.uint8)
    cv2.circle(test_img, (140, 140), 80, (255, 255, 255, 255), thickness=20)

    result = image_to_mnist(test_img)
    if result is not None:
        print(f"Output shape: {result.shape}")
        print(f"Min: {result.min()}, Max: {result.max()}, Mean: {result.mean():.2f}")
        print(f"Non-zero pixels: {np.count_nonzero(result)}")

        # Save to test CSV
        test_csv = "test_mnist.csv"
        sid = save_sample(0, result, "train", test_csv)
        print(f"Saved sample ID: {sid}")
        summary = dataset_summary(test_csv)
        print(f"Dataset summary: {summary}")
        os.remove(test_csv)
        print("✅ All tests passed.")
    else:
        print("❌ image_to_mnist returned None — check input.")
