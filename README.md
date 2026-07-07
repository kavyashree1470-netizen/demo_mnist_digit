# ✏️ MNIST Digit Collector

A **Streamlit** app for collecting handwritten digits at MNIST standards.

## Features

| Feature | Details |
|---|---|
| Canvas drawing | Smooth HTML5 canvas with thick anti-aliased pen |
| Image upload | PNG / JPG support |
| MNIST pipeline | 28×28 · Grayscale · Anti-aliased · Centre-of-mass centred |
| Storage | Appends to `mnist_dataset.csv` with full 784-pixel rows |
| CSV columns | `id, label, timestamp, split, pixel_0 … pixel_783` |
| Visualisation | Live 28×28 pixel grid + stats |
| Download | Per-sample PNG + full CSV |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## CSV Format

```
id, label, timestamp,        split, pixel_0, pixel_1, ..., pixel_783
1,  7,     2024-06-14 10:23, train, 0,       0,       ..., 255
```

Each row = one 28×28 digit sample (784 pixel values, 0-255).

## MNIST Processing Pipeline

```
Raw drawing (280×280 RGBA)
  ↓ alpha channel extraction
  ↓ Gaussian blur σ=1.0  [anti-aliasing]
  ↓ bounding-box crop + 20% padding
  ↓ bicubic resize → 20×20
  ↓ centre-of-mass centering → 28×28 canvas
  ↓ uint8 normalisation [0-255]
= MNIST-compatible 28×28 array
```

## Using the Utilities Standalone

```python
from mnist_utils import image_to_mnist, save_sample, load_dataset
import numpy as np
from PIL import Image

img = np.array(Image.open("my_digit.png").convert("RGBA"))
mnist_arr = image_to_mnist(img)   # → (28, 28) uint8
save_sample(label=5, pixels=mnist_arr, split="train")
df = load_dataset()               # full CSV as DataFrame
```

## Project Structure

```
mnist_collector/
├── app.py           # Streamlit application (main entry point)
├── mnist_utils.py   # Core processing + CSV utilities
├── requirements.txt # Python dependencies
├── README.md        # This file
└── mnist_dataset.csv  (auto-created on first save)
```
