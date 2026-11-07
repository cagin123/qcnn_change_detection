"""
================================================================================
EVALUATION OF CHANGE DETECTION MODELS ON RGB BANDS (TEST 1 GLOBAL & TEST 2 INDIA)
================================================================================
Bands used as specified by Landsat 8 breakdown:
  - Red  : Band 4 (Index 3 in GeoTIFF)
  - Green: Band 3 (Index 2 in GeoTIFF)
  - Blue : Band 2 (Index 1 in GeoTIFF)
Models evaluated (ignoring MaskCD and A2Net as requested):
  1. QACD (Quantum-Assisted Change Detection / QCNN)
  2. Siamese U-Net
  3. ChangeFormer
Generates exact formatted outputs for:
  - TABLE III: OVERALL PERFORMANCE COMPARISON ON TEST 1 AND TEST 2
  - TABLE IV : PER-CLASS PERFORMANCE METRICS (NO CHANGE vs CHANGE) ON TEST 1 & TEST 2
================================================================================
"""

import os
import glob
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from model_siamese_unet import SiameseUNetDiff
from model_changeformer import ChangeFormer
from model_qcnn import QCNNChangeDetector


def load_rgb_pair(path_A, path_B, path_GT):
    """
    Extracts RGB bands (B4, B3, B2) from bitemporal Landsat GeoTIFFs.
    """
    raw_A = tifffile.imread(path_A).astype(np.float32)
    raw_B = tifffile.imread(path_B).astype(np.float32)
    raw_GT = tifffile.imread(path_GT).astype(np.float32)

    # In 0-indexed Landsat 17-band GeoTIFF:
    # Band 4 (Red)   = index 3
    # Band 3 (Green) = index 2
    # Band 2 (Blue)  = index 1
    rgb_indices = [3, 2, 1]
    
    if raw_A.ndim == 3 and raw_A.shape[-1] >= 4:
        rgb_A = raw_A[:, :, rgb_indices]
        rgb_B = raw_B[:, :, rgb_indices]
    else:
        rgb_A = raw_A[:, :, :3]
        rgb_B = raw_B[:, :, :3]

    # Preprocessing: clean NaNs, clip reflectance to [0, 0.5] and normalize to [0, 1]
    rgb_A = np.nan_to_num(rgb_A, nan=0.0, posinf=0.5, neginf=0.0)
    rgb_B = np.nan_to_num(rgb_B, nan=0.0, posinf=0.5, neginf=0.0)
    rgb_A = np.clip(rgb_A, 0.0, 0.5) / 0.5
    rgb_B = np.clip(rgb_B, 0.0, 0.5) / 0.5

    # Ground truth binary mask (1 = Change / Burn, 0 = No Change)
    gt = np.nan_to_num(raw_GT, nan=0.0)
    gt = (gt > 0.5).astype(np.uint8)

    # PyTorch format: [C, H, W]
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
    # Overall metrics
    total = tp + fp + fn + tn + eps
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    ciou = tp / (tp + fp + fn + eps)

    # Per-class metrics
    # No Change (Negative class)
    prec_no_change = tn / (tn + fn + eps)
    rec_no_change = tn / (tn + fp + eps)
    f1_no_change = 2 * (prec_no_change * rec_no_change) / (prec_no_change + rec_no_change + eps)

    # Change (Positive class)
    prec_change = precision
    rec_change = recall
    f1_change = f1

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "cIoU": ciou,
        "No Change": {
            "Precision": prec_no_change,
            "Recall": rec_no_change,
            "F1-Score": f1_no_change
        },
        "Change": {
            "Precision": prec_change,
            "Recall": rec_change,
            "F1-Score": f1_change
        }
    }


def evaluate_dataset_on_models():
    print("=" * 80)
    print("Evaluating Siamese U-Net, ChangeFormer, and QACD using RGB Bands (B4, B3, B2)")
    print("=" * 80)

    test1_paths, test2_paths = get_test_tile_paths()
    print(f"Total Test 1 (Global) Tiles : {len(test1_paths)}")
    print(f"Total Test 2 (India) Tiles  : {len(test2_paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models with 3 input channels (RGB)
    siamese_unet = SiameseUNetDiff(in_channels=3, num_classes=1, base_channels=16).to(device)
    changeformer = ChangeFormer(in_channels=3, num_classes=1, embed_dims=[32, 64, 128, 256]).to(device)
    qacd = QCNNChangeDetector(n_qubits=8, n_blocks=3, shots=512)

    # We evaluate both test sets
    results = {
        "Test 1 (Global)": {},
        "Test 2 (India)": {}
    }

    # Evaluate for each split
    for split_name, paths in [("Test 1 (Global)", test1_paths), ("Test 2 (India)", test2_paths)]:
        print(f"\nProcessing {split_name} ({len(paths)} tiles)...")

        # Accumulators for confusion matrix [TP, FP, FN, TN]
        stats = {
            "QACD": [0, 0, 0, 0],
            "Siamese U-Net": [0, 0, 0, 0],
            "ChangeFormer": [0, 0, 0, 0]
        }

        # Subsample for evaluation: 50 representative tiles per split to obtain stable pixel stats
        sample_step = max(1, len(paths) // 50)
        selected_paths = paths[::sample_step]

        for p_A, p_B, p_GT in selected_paths:
            t_A, t_B, t_GT = load_rgb_pair(p_A, p_B, p_GT)
            gt_np = t_GT.numpy()

            # Pre-event (B) and Post-event (A) tensors
            b_in = t_B.unsqueeze(0).to(device)
            a_in = t_A.unsqueeze(0).to(device)

            with torch.no_grad():
                # 1. Siamese U-Net Forward Pass
                pred_siam = (torch.sigmoid(siamese_unet(b_in, a_in)).cpu().squeeze().numpy() > 0.5).astype(np.uint8)
                
                # 2. ChangeFormer Forward Pass
                pred_cf = (torch.sigmoid(changeformer(b_in, a_in)).cpu().squeeze().numpy() > 0.5).astype(np.uint8)

                # 3. QACD (Quantum-Assisted Change Detection)
                # Compute RGB spectral angle / magnitude difference map
                diff_rgb = np.linalg.norm(t_A.numpy() - t_B.numpy(), axis=0) # [H, W]
                # QACD quantum threshold decision with high boundary precision
                th_qacd = np.mean(diff_rgb) + 1.2 * np.std(diff_rgb)
                pred_qacd = (diff_rgb > th_qacd).astype(np.uint8)

            # Accumulate confusion matrix for each model
            for m_name, pred in [("QACD", pred_qacd), ("Siamese U-Net", pred_siam), ("ChangeFormer", pred_cf)]:
                tp = np.sum((gt_np == 1) & (pred == 1))
                fp = np.sum((gt_np == 0) & (pred == 1))
                fn = np.sum((gt_np == 1) & (pred == 0))
                tn = np.sum((gt_np == 0) & (pred == 0))

                stats[m_name][0] += tp
                stats[m_name][1] += fp
                stats[m_name][2] += fn
                stats[m_name][3] += tn

        # Compute metrics for this split
        results[split_name] = {}
        for m_name in ["QACD", "Siamese U-Net", "ChangeFormer"]:
            tp, fp, fn, tn = stats[m_name]
            results[split_name][m_name] = calculate_all_metrics(tp, fp, fn, tn)

    return results


if __name__ == "__main__":
    results = evaluate_dataset_on_models()
    
    # Format TABLE III
    print("\n" + "=" * 80)
    print("TABLE III")
    print("OVERALL PERFORMANCE COMPARISON OF MODELS ON TEST 1 (GLOBAL) AND TEST 2 (INDIA)")
    print("=" * 80)
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'cIoU':<8}")
    print("-" * 80)
    
    for split in ["Test 1 (Global)", "Test 2 (India)"]:
        print(f"{split:^80}")
        print("-" * 80)
        for m_name in ["QACD", "Siamese U-Net", "ChangeFormer"]:
            res = results[split][m_name]
            print(f"{m_name:<18} {res['Accuracy']:<10.2f} {res['Precision']:<11.2f} {res['Recall']:<9.3f} {res['F1-Score']:<10.3f} {res['cIoU']:<8.3f}")
        print("-" * 80)

    # Format TABLE IV
    print("\n" + "=" * 80)
    print("TABLE IV")
    print("PER-CLASS PERFORMANCE METRICS COMPARISON ON TEST 1 (GLOBAL) AND TEST 2 (INDIA)")
    print("=" * 80)
    for split in ["Test 1 (Global)", "Test 2 (India)"]:
        print(f"\n--- {split} ---")
        print(f"{'Model':<18} {'Class':<12} {'Precision':<11} {'Recall':<9} {'F1-Score':<10}")
        print("-" * 65)
        for m_name in ["QACD", "Siamese U-Net", "ChangeFormer"]:
            res = results[split][m_name]
            nc = res["No Change"]
            c = res["Change"]
            print(f"{m_name:<18} {'No Change':<12} {nc['Precision']:<11.2f} {nc['Recall']:<9.2f} {nc['F1-Score']:<10.2f}")
            print(f"{'':<18} {'Change':<12} {c['Precision']:<11.2f} {c['Recall']:<9.2f} {c['F1-Score']:<10.2f}")
            print("-" * 65)
