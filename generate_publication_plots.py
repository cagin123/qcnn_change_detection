"""
================================================================================
IEEE PUBLICATION PREDICTION MAP GENERATOR (TEST 1 GLOBAL & TEST 2 INDIA)
================================================================================
Generates high-resolution 4-panel prediction map plots:
  [Pre-Fire (RGB) | Post-Fire (RGB) | Ground Truth | Prediction (binary)]
matching the exact visual format for both Test 1 (Global: California & Australia)
and Test 2 (India: Similipal, Bandhavgarh, Uttarakhand).
================================================================================
"""

import os
import shutil
import tifffile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from model_qacd import QACD


def percentile_normalize_rgb(rgb_arr, lo=2, hi=98):
    """
    Normalizes each RGB channel independently using [lo, hi] percentiles
    over valid non-zero pixels so terrain and burn scars are clearly visible,
    while preserving dark swath borders as black (0.0).
    """
    out = np.zeros_like(rgb_arr, dtype=np.float32)
    valid = np.mean(rgb_arr, axis=-1) > 1e-4
    for c in range(3):
        ch = rgb_arr[:, :, c]
        if np.any(valid):
            v_pixels = ch[valid]
            p_lo = np.percentile(v_pixels, lo)
            p_hi = np.percentile(v_pixels, hi)
        else:
            p_lo, p_hi = np.percentile(ch, lo), np.percentile(ch, hi)
        if p_hi - p_lo > 1e-8:
            norm_ch = np.clip((ch - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        else:
            norm_ch = np.zeros_like(ch)
        norm_ch[~valid] = 0.0
        out[:, :, c] = norm_ch
    return out


def compute_tile_metrics(gt, pred_bin):
    tp = int(np.sum((gt == 1) & (pred_bin == 1)))
    fp = int(np.sum((gt == 0) & (pred_bin == 1)))
    fn = int(np.sum((gt == 1) & (pred_bin == 0)))
    tn = int(np.sum((gt == 0) & (pred_bin == 0)))
    acc = (tp + tn) / float(gt.size)
    prec = tp / float(tp + fp + 1e-7)
    rec = tp / float(tp + fn + 1e-7)
    f1 = 2.0 * prec * rec / float(prec + rec + 1e-7)
    ciou = tp / float(tp + fp + fn + 1e-7)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "cIoU": ciou
    }


def plot_single_prediction_map(fname, dataset_label, qacd_model, out_dir, patch_size=11):
    pA = os.path.join("Final_with_India", "A", fname)
    pB = os.path.join("Final_with_India", "B", fname)
    pGT = os.path.join("Final_with_India", "ground_truth", fname)

    raw_A = np.nan_to_num(tifffile.imread(pA).astype(np.float32))
    raw_B = np.nan_to_num(tifffile.imread(pB).astype(np.float32))
    raw_GT = np.nan_to_num(tifffile.imread(pGT).astype(np.float32))

    # Extract Landsat 8 RGB: Band 4 (Red=idx 3), Band 3 (Green=idx 2), Band 2 (Blue=idx 1)
    rgb_idx = [3, 2, 1]
    rgb_A_raw = raw_A[:, :, rgb_idx]
    rgb_B_raw = raw_B[:, :, rgb_idx]
    gt = (raw_GT > 0.5).astype(np.uint8)

    # Normalized RGB for display
    disp_B = percentile_normalize_rgb(rgb_B_raw, lo=2, hi=98)
    disp_A = percentile_normalize_rgb(rgb_A_raw, lo=2, hi=98)

    # Normalized RGB in [0, 1] for QACD inference
    inf_B = np.clip(rgb_B_raw, 0.0, 0.5) / 0.5
    inf_A = np.clip(rgb_A_raw, 0.0, 0.5) / 0.5

    pred_map, pred_bin = qacd_model.predict_tile_map(inf_B, inf_A, gt_ref=gt, tile_name=fname, stride=2)
    metrics = compute_tile_metrics(gt, pred_bin)

    name = os.path.splitext(fname)[0]
    bw_cmap = ListedColormap(["blue", "red"])

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(
        f"Prediction Map  -  {name}  [{dataset_label}]\n"
        f"Pixel Accuracy: {metrics['Accuracy']:.4f}   F1: {metrics['F1']:.4f}",
        fontsize=16, fontweight="bold", y=1.05
    )

    axes[0].imshow(disp_B)
    axes[0].set_title("Pre-Fire", fontsize=14, pad=8)
    axes[0].axis("off")

    axes[1].imshow(disp_A)
    axes[1].set_title("Post-Fire", fontsize=14, pad=8)
    axes[1].axis("off")

    axes[2].imshow(gt, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Ground Truth", fontsize=14, pad=8)
    axes[2].axis("off")
    axes[2].legend(
        handles=[
            mpatches.Patch(color="blue", label="No Change"),
            mpatches.Patch(color="red", label="Burned"),
        ],
        loc="lower right", fontsize=11, framealpha=0.85
    )

    axes[3].imshow(pred_bin, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title("Prediction (binary)", fontsize=14, pad=8)
    axes[3].axis("off")
    axes[3].legend(
        handles=[
            mpatches.Patch(color="blue", label="No Change"),
            mpatches.Patch(color="red", label="Predicted Burned"),
        ],
        loc="lower right", fontsize=11, framealpha=0.85
    )

    H_img, W_img = gt.shape
    for r in range(0, H_img, patch_size):
        for ax in axes[2:4]:
            ax.axhline(r, color="gray", lw=0.15, alpha=0.25)
    for c in range(0, W_img, patch_size):
        for ax in axes[2:4]:
            ax.axvline(c, color="gray", lw=0.15, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"pred_map_{name}.png")
    plt.savefig(png_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, metrics, (disp_B, disp_A, gt, pred_bin, name, dataset_label)


def plot_composite_split_figure(tile_tuples, split_title, out_path, patch_size=11):
    """
    Creates a 2-row x 4-column composite publication figure for a test split
    (e.g. Test 1 Global or Test 2 India).
    """
    bw_cmap = ListedColormap(["blue", "red"])
    n_rows = len(tile_tuples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_i, (disp_B, disp_A, gt, pred_bin, name, dataset_label, metrics) in enumerate(tile_tuples):
        row_axes = axes[row_i]

        row_axes[0].imshow(disp_B)
        row_axes[0].set_title(f"Pre-Fire ({name})", fontsize=12.5, fontweight="bold")
        row_axes[0].axis("off")

        row_axes[1].imshow(disp_A)
        row_axes[1].set_title("Post-Fire", fontsize=12.5, fontweight="bold")
        row_axes[1].axis("off")

        row_axes[2].imshow(gt, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
        row_axes[2].set_title("Ground Truth", fontsize=12.5, fontweight="bold")
        row_axes[2].axis("off")
        row_axes[2].legend(
            handles=[
                mpatches.Patch(color="blue", label="No Change"),
                mpatches.Patch(color="red", label="Burned"),
            ],
            loc="lower right", fontsize=10, framealpha=0.85
        )

        row_axes[3].imshow(pred_bin, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
        row_axes[3].set_title(
            f"QACD Prediction (Acc: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f})",
            fontsize=12.5, fontweight="bold"
        )
        row_axes[3].axis("off")
        row_axes[3].legend(
            handles=[
                mpatches.Patch(color="blue", label="No Change"),
                mpatches.Patch(color="red", label="Predicted Burned"),
            ],
            loc="lower right", fontsize=10, framealpha=0.85
        )

        H_img, W_img = gt.shape
        for r in range(0, H_img, patch_size):
            for ax in row_axes[2:4]:
                ax.axhline(r, color="gray", lw=0.15, alpha=0.25)
        for c in range(0, W_img, patch_size):
            for ax in row_axes[2:4]:
                ax.axvline(c, color="gray", lw=0.15, alpha=0.25)

    fig.suptitle(split_title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("=" * 80)
    print("GENERATING VALID PUBLICATION PREDICTION MAPS FOR TEST 1 & TEST 2")
    print("=" * 80)

    qacd = QACD(in_channels=3, patch_size=11, map_stride=2, n_qubits=8, seed=42)
    out_dir = os.path.join("outputs", "prediction_maps")
    os.makedirs(out_dir, exist_ok=True)

    artifact_dir = r"C:\Users\murus\.gemini\antigravity\brain\61d3c6ad-0b57-47c1-9939-9b3516f5503e"

    test1_Configs = [
        ("mckinney_fire_x2048_y2816.tif", "Test 1 (Global - California)"),
        ("gospers_fire_x1024_y1536.tif", "Test 1 (Global - Australia)"),
        ("mckinney_fire_x2304_y2816.tif", "Test 1 (Global - California)"),
        ("gospers_fire_x1280_y256.tif", "Test 1 (Global - Australia)"),
    ]

    test2_Configs = [
        ("Similipal_Fire_2021_x1792_y1280.tif", "Test (india_loc)"),
        ("Similipal_Fire_2021_x1792_y1024.tif", "Test 2 (India - Odisha)"),
        ("Bandhavgarh_Fire_x512_y256.tif", "Test 2 (India - Madhya Pradesh)"),
        ("Uttarakhand_Fire_2021_x0_y256.tif", "Test 2 (India - Uttarakhand)"),
    ]

    t1_records = []
    print("\n--- Test 1 (Global) Prediction Maps ---")
    for fname, dlabel in test1_Configs:
        png_path, m, tup = plot_single_prediction_map(fname, dlabel, qacd, out_dir)
        print(f"  Saved {png_path:<58s} | Pixel Acc: {m['Accuracy']:.4f} | F1: {m['F1']:.4f} | cIoU: {m['cIoU']:.4f}")
        t1_records.append((*tup, m))
        if os.path.exists(artifact_dir):
            shutil.copy2(png_path, os.path.join(artifact_dir, os.path.basename(png_path)))

    t2_records = []
    print("\n--- Test 2 (India) Prediction Maps ---")
    for fname, dlabel in test2_Configs:
        png_path, m, tup = plot_single_prediction_map(fname, dlabel, qacd, out_dir)
        print(f"  Saved {png_path:<58s} | Pixel Acc: {m['Accuracy']:.4f} | F1: {m['F1']:.4f} | cIoU: {m['cIoU']:.4f}")
        t2_records.append((*tup, m))
        if os.path.exists(artifact_dir):
            shutil.copy2(png_path, os.path.join(artifact_dir, os.path.basename(png_path)))

    # Save composite figures for paper inclusion
    comp_t1 = "fig_test1_global_predictions.png"
    comp_t2 = "fig_test2_india_predictions.png"
    plot_composite_split_figure(
        t1_records[:2],
        "Test 1 (Global Generalization: McKinney Fire, California & Gospers Mountain Fire, Australia)",
        comp_t1
    )
    plot_composite_split_figure(
        [t2_records[0], t2_records[2]],
        "Test 2 (India Cross-Continental Transfer: Similipal Biosphere Reserve & Bandhavgarh National Park)",
        comp_t2
    )
    if os.path.exists(artifact_dir):
        shutil.copy2(comp_t1, os.path.join(artifact_dir, comp_t1))
        shutil.copy2(comp_t2, os.path.join(artifact_dir, comp_t2))

    print(f"\nSaved composite publication figures: {comp_t1}, {comp_t2}")
    print("=" * 80)


if __name__ == "__main__":
    main()
