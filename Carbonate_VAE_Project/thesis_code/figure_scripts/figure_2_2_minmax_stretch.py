"""
figure_2_2_minmax_stretch.py
=============================
Figure 2.2 — Dynamic Min-Max Contrast Stretching for Carbonate Rock CT Images

Reads 16-bit low-contrast TIF slices, performs Min-Max scaling to 8-bit [0,255],
and renders a 2×2 academic figure: Original | Stretched | Histogram(Original) | Histogram(Stretched).

Output: figure_2_2.png (DPI=300, Times New Roman, publication-ready)
"""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ===== Paths =====
from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figure_2_2.png")

# ===== Configuration =====
# Specify 1–3 TIF files: use the first one for the figure
# To show 3 images side-by-side, uncomment the other paths
TIF_FILES = [
    os.path.join(DATA_DIR, "1633-1663-1203-50um0000.tif"),
    # os.path.join(DATA_DIR, "1633-1663-1203-50um0050.tif"),
    # os.path.join(DATA_DIR, "1633-1663-1203-50um0100.tif"),
]


def load_tif_u16(filepath):
    """
    Load a 16-bit TIF using PIL, return raw uint16 numpy array.
    """
    img = np.array(Image.open(filepath), dtype=np.uint16)
    return img


def minmax_stretch_u8(img_u16):
    """
    Dynamic Min-Max scaling: (img - min) / (max - min) → [0, 1] → ×255 → uint8.

    Args:
        img_u16: np.ndarray, uint16

    Returns:
        img_u8: np.ndarray, uint8, 0–255
    """
    img_f = img_u16.astype(np.float64)
    vmin = img_f.min()
    vmax = img_f.max()
    if vmax == vmin:
        return np.zeros_like(img_u16, dtype=np.uint8)
    stretched_f = (img_f - vmin) / (vmax - vmin) * 255.0
    stretched_f = np.clip(stretched_f, 0.0, 255.0)
    return stretched_f.round().astype(np.uint8)


def main():
    # ===== Font & style: Times New Roman, academic =====
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 11
    rcParams['mathtext.fontset'] = 'stix'

    for i, fp in enumerate(TIF_FILES):
        if not os.path.exists(fp):
            print(f"[SKIP] File not found: {fp}")
            continue

        basename = os.path.splitext(os.path.basename(fp))[0]
        print(f"[INFO] Processing: {basename}")

        # ── Load raw 16-bit ──
        raw_u16 = load_tif_u16(fp)
        print(f"       dtype={raw_u16.dtype}, shape={raw_u16.shape}, "
              f"min={raw_u16.min()}, max={raw_u16.max()}, mean={raw_u16.mean():.1f}")

        # ── Min-Max stretch to 8-bit ──
        stretched_u8 = minmax_stretch_u8(raw_u16)
        print(f"       Stretched → uint8, min={stretched_u8.min()}, max={stretched_u8.max()}")

        # ── Flatten for histogram ──
        raw_flat = raw_u16.ravel()
        stretched_flat = stretched_u8.ravel()

        # ===== Plot: 2×2 subplot =====
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (A) Top-Left: Original image
        ax0 = axes[0, 0]
        ax0.imshow(raw_u16, cmap='gray', vmin=raw_u16.min(), vmax=raw_u16.max())
        ax0.set_title("(a) Original 16-bit CT Slice", fontsize=13, fontweight='bold')
        ax0.set_xlabel("X (pixel)")
        ax0.set_ylabel("Y (pixel)")
        ax0.tick_params(labelsize=9)

        # (B) Top-Right: Stretched image
        ax1 = axes[0, 1]
        ax1.imshow(stretched_u8, cmap='gray', vmin=0, vmax=255)
        ax1.set_title("(b) Min-Max Stretched 8-bit", fontsize=13, fontweight='bold')
        ax1.set_xlabel("X (pixel)")
        ax1.set_ylabel("Y (pixel)")
        ax1.tick_params(labelsize=9)

        # (C) Bottom-Left: Histogram of original
        ax2 = axes[1, 0]
        ax2.hist(raw_flat, bins=256, color='steelblue', edgecolor='white', alpha=0.85,
                 range=(raw_u16.min(), raw_u16.max()))
        ax2.axvline(raw_u16.min(), color='red', linestyle='--', linewidth=1.2,
                     label=f'Min = {raw_u16.min()}')
        ax2.axvline(raw_u16.max(), color='darkorange', linestyle='--', linewidth=1.2,
                     label=f'Max = {raw_u16.max()}')
        ax2.set_title("(c) Histogram — Original (16-bit)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Pixel Intensity (16-bit)")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=9, loc='upper right')
        ax2.tick_params(labelsize=9)

        # (D) Bottom-Right: Histogram of stretched
        ax3 = axes[1, 1]
        ax3.hist(stretched_flat, bins=256, color='firebrick', edgecolor='white', alpha=0.85,
                 range=(0, 255))
        ax3.axvline(0, color='darkblue', linestyle='--', linewidth=1.2,
                     label='Min = 0')
        ax3.axvline(255, color='darkgreen', linestyle='--', linewidth=1.2,
                     label='Max = 255')
        ax3.set_title("(d) Histogram — Stretched (8-bit)", fontsize=13, fontweight='bold')
        ax3.set_xlabel("Pixel Intensity (8-bit)")
        ax3.set_ylabel("Frequency")
        ax3.legend(fontsize=9, loc='upper right')
        ax3.tick_params(labelsize=9)

        # ===== Global title & layout =====
        fig.suptitle(
            f"Figure 2.2 — Dynamic Min-Max Contrast Stretching\n"
            f"{basename}  |  Original range: [{raw_u16.min()}, {raw_u16.max()}]  →  8-bit [0, 255]",
            fontsize=15, fontweight='bold', y=1.01
        )
        fig.tight_layout()

        # ===== Save =====
        # If multiple files, append index to filename
        if len(TIF_FILES) > 1:
            out = OUTPUT_PATH.replace(".png", f"_{i+1}.png")
        else:
            out = OUTPUT_PATH
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[DONE] Saved to: {out}\n")


if __name__ == "__main__":
    main()
