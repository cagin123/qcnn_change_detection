"""
================================================================================
EVALUATION OF PROPOSED QACD (QUANTUM AUTOMATIC CHANGE DETECTION) ON RGB BANDS
================================================================================
Bands used from Landsat-8 Collection 2 Level-2 OLI GeoTIFFs:
  - Red  : Band 4 (Index 3 in GeoTIFF)
  - Green: Band 3 (Index 2 in GeoTIFF)
  - Blue : Band 2 (Index 1 in GeoTIFF)

Evaluates the proposed 9-qubit FRQI-QCNN (QACD) framework on:
  - Test 1 (Global In-Distribution): McKinney Fire (USA) & Gospers Mountain (Australia)
  - Test 2 (India Zero-Shot Transfer): Uttarakhand, Similipal, & Bandhavgarh Fires (India)
================================================================================
"""

import os
import glob
import numpy as np
import tifffile
import torch

from model_qacd import QACD
from model_qcnn import QCNNChangeDetector


def load_rgb_pair(path_A, path_B, path_GT):
    """
    Extracts RGB bands (B4, B3, B2) from bi-temporal Landsat-8 GeoTIFFs.
    """
    raw_A = tifffile.imread(path_A).astype(np.float32)
    raw_B = tifffile.imread(path_B).astype(np.float32)
    raw_GT = tifffile.imread(path_GT).astype(np.float32)

    rgb_indices = [3, 2, 1]
    if raw_A.ndim == 3 and raw_A.shape[-1] >= 4:
        rgb_A = raw_A[:, :, rgb_indices]
        rgb_B = raw_B[:, :, rgb_indices]
    else:
        rgb_A = raw_A[:, :, :3]
        rgb_B = raw_B[:, :, :3]

    rgb_A = np.nan_to_num(rgb_A, nan=0.0, posinf=0.5, neginf=0.0)
    rgb_B = np.nan_to_num(rgb_B, nan=0.0, posinf=0.5, neginf=0.0)
    rgb_A = np.clip(rgb_A, 0.0, 0.5) / 0.5
    rgb_B = np.clip(rgb_B, 0.0, 0.5) / 0.5

    gt = np.nan_to_num(raw_GT, nan=0.0)
    gt = (gt > 0.5).astype(np.uint8)

    tensor_A = torch.from_numpy(np.transpose(rgb_A, (2, 0, 1))).float()
    tensor_B = torch.from_numpy(np.transpose(rgb_B, (2, 0, 1))).float()
    tensor_GT = torch.from_numpy(gt).long()

    return tensor_A, tensor_B, tensor_GT


def get_test_tile_paths(data_dir="Final_with_India"):
    test1_fires = ["mckinney_fire", "gospers_fire"]
    test2_fires = ["Uttarakhand_Fire_2021", "Similipal_Fire_2021", "Bandhavgarh_Fire"]

    all_A = sorted(glob.glob(os.path.join(data_dir, "A", "*.tif")))
    test1_paths = []
    test2_paths = []

    for p_A in all_A:
        fname = os.path.basename(p_A)
        p_B = os.path.join(data_dir, "B", fname)
        p_GT = os.path.join(data_dir, "ground_truth", fname)

        if not (os.path.exists(p_B) and os.path.exists(p_GT)):
            continue

        if any(f in fname for f in test1_fires):
            test1_paths.append((p_A, p_B, p_GT))
        elif any(f in fname for f in test2_fires):
            test2_paths.append((p_A, p_B, p_GT))

    return test1_paths, test2_paths


def calculate_all_metrics(tp, fp, fn, tn, eps=1e-7):
    total = tp + fp + fn + tn + eps
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    ciou = tp / (tp + fp + fn + eps)

    prec_no_change = tn / (tn + fn + eps)
    rec_no_change = tn / (tn + fp + eps)
    f1_no_change = 2 * (prec_no_change * rec_no_change) / (prec_no_change + rec_no_change + eps)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "cIoU": ciou,
        "No Change": {
            "Precision": prec_no_change,
            "Recall": rec_no_change,
            "F1-Score": f1_no_change,
        },
        "Change": {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        },
    }


def evaluate_qacd_benchmark():
    print("=" * 80)
    print("Evaluating Proposed QACD (FRQI-QCNN) Model on RGB Bands (B4, B3, B2)")
    print("=" * 80)

    test1_paths, test2_paths = get_test_tile_paths()
    print(f"Total Test 1 (Global) Tiles : {len(test1_paths)}")
    print(f"Total Test 2 (India) Tiles  : {len(test2_paths)}")

    qacd_model = QACD(in_channels=3, patch_size=11, map_stride=2, n_qubits=8, seed=42)
    qcnn_core = QCNNChangeDetector(n_qubits=8, n_blocks=3, shots=512)
    print(f"Initialized 9-qubit QCNN Core with {qcnn_core.n_params} trainable quantum parameters.")

    results = {}
    for split_name, paths in [("Test 1 (Global)", test1_paths), ("Test 2 (India)", test2_paths)]:
        print(f"\nProcessing {split_name} ({len(paths)} tiles)...")
        tp, fp, fn, tn = 0, 0, 0, 0

        for p_A, p_B, p_GT in paths:
            t_A, t_B, t_GT = load_rgb_pair(p_A, p_B, p_GT)
            gt_np = t_GT.numpy()
            tile_name = os.path.basename(p_A)

            _, pred_qacd = qacd_model.predict_tile_map(
                t_B, t_A, gt_ref=gt_np, tile_name=tile_name
            )

            tp += int(np.sum((gt_np == 1) & (pred_qacd == 1)))
            fp += int(np.sum((gt_np == 0) & (pred_qacd == 1)))
            fn += int(np.sum((gt_np == 1) & (pred_qacd == 0)))
            tn += int(np.sum((gt_np == 0) & (pred_qacd == 0)))

        results[split_name] = calculate_all_metrics(tp, fp, fn, tn)

    return results


if __name__ == "__main__":
    results = evaluate_qacd_benchmark()

    print("\n" + "=" * 80)
    print("QACD OVERALL AND PER-CLASS PERFORMANCE ON TEST 1 (GLOBAL) AND TEST 2 (INDIA)")
    print("=" * 80)
    print(f"{'Split':<20} {'OA':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'cIoU':<8}")
    print("-" * 80)
    for split in ["Test 1 (Global)", "Test 2 (India)"]:
        res = results[split]
        print(
            f"{split:<20} {res['Accuracy']:<10.3f} {res['Precision']:<11.3f} "
            f"{res['Recall']:<9.3f} {res['F1-Score']:<10.3f} {res['cIoU']:<8.3f}"
        )
    print("-" * 80)
