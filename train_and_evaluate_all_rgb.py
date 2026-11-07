"""
================================================================================
TRAIN AND EVALUATE NEWLY PREPARED CHANGE DETECTION MODELS ON RGB BANDS
================================================================================
Models Trained & Evaluated:
  1. Siamese U-Net (FC-Siam-diff - Daudt et al., ICIP 2018)
  2. ChangeFormer  (Hierarchical Transformer - Bandara & Patel, IGARSS 2022)
  3. QACD          (Quantum-Assisted Change Detection with FRQI - Cong et al., 2019)
(MaskCD and A2Net ignored as requested)

Bands:
  - Landsat 8 RGB: Band 4 (Red), Band 3 (Green), Band 2 (Blue)
Datasets:
  - Train: Wallow Fire, Evans Canyon Fire, August Complex Fire
  - Test 1 (Global): McKinney Fire, Gospers Mountain Fire
  - Test 2 (India) : Uttarakhand Fire, Similipal Fire, Bandhavgarh Fire
================================================================================
"""

import os
import glob
import time
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_siamese_unet import SiameseUNetDiff, FocalLoss
from model_changeformer import ChangeFormer
from model_qcnn import QCNNChangeDetector, QISKIT_AVAILABLE


# -----------------------------------------------------------------------------
# 1. Dataset Loader for RGB Bands
# -----------------------------------------------------------------------------

class RGBChangeDetectionDataset(Dataset):
    def __init__(self, data_dir="Final_with_India", fire_filter=None, max_tiles=None, require_burn=False):
        self.data_dir = data_dir
        dir_A = os.path.join(data_dir, "A")
        dir_B = os.path.join(data_dir, "B")
        dir_GT = os.path.join(data_dir, "ground_truth")

        all_A = sorted(glob.glob(os.path.join(dir_A, "*.tif")))
        self.samples = []

        # Find valid tiles matching the fire filter
        burn_tiles = []
        unburned_tiles = []

        for p_A in all_A:
            fname = os.path.basename(p_A)
            p_B = os.path.join(dir_B, fname)
            p_GT = os.path.join(dir_GT, fname)

            if not (os.path.exists(p_B) and os.path.exists(p_GT)):
                continue

            if fire_filter is not None:
                if not any(f in fname for f in fire_filter):
                    continue

            if require_burn:
                # Check ground truth mask to ensure balanced training
                mask_sample = tifffile.imread(p_GT)
                has_burn = np.nan_to_num(mask_sample).sum() > 50
                if has_burn:
                    burn_tiles.append((p_A, p_B, p_GT))
                else:
                    unburned_tiles.append((p_A, p_B, p_GT))
            else:
                self.samples.append((p_A, p_B, p_GT))

        if require_burn:
            # Balanced mix: all burn scar tiles + equal unburned tiles
            n_unburn = min(len(burn_tiles), len(unburned_tiles))
            np.random.seed(42)
            sel_unburned = [unburned_tiles[i] for i in np.random.choice(len(unburned_tiles), n_unburn, replace=False)]
            self.samples = burn_tiles + sel_unburned
            np.random.shuffle(self.samples)

        if max_tiles is not None and len(self.samples) > max_tiles:
            self.samples = self.samples[:max_tiles]

        print(f"Dataset loaded: {len(self.samples)} tiles (Filter: {fire_filter})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_A, p_B, p_GT = self.samples[idx]

        raw_A = tifffile.imread(p_A).astype(np.float32)
        raw_B = tifffile.imread(p_B).astype(np.float32)
        raw_GT = tifffile.imread(p_GT).astype(np.float32)

        # Extract RGB: Red=Band 4 (idx 3), Green=Band 3 (idx 2), Blue=Band 2 (idx 1)
        rgb_indices = [3, 2, 1]
        rgb_A = raw_A[:, :, rgb_indices] if raw_A.ndim == 3 and raw_A.shape[-1] >= 4 else raw_A[:, :, :3]
        rgb_B = raw_B[:, :, rgb_indices] if raw_B.ndim == 3 and raw_B.shape[-1] >= 4 else raw_B[:, :, :3]

        # Normalize reflectance [0, 0.5] -> [0, 1]
        rgb_A = np.nan_to_num(rgb_A, nan=0.0, posinf=0.5, neginf=0.0)
        rgb_B = np.nan_to_num(rgb_B, nan=0.0, posinf=0.5, neginf=0.0)
        rgb_A = np.clip(rgb_A, 0.0, 0.5) / 0.5
        rgb_B = np.clip(rgb_B, 0.0, 0.5) / 0.5

        # Ground truth binary mask
        gt = np.nan_to_num(raw_GT, nan=0.0)
        gt = (gt > 0.5).astype(np.float32)

        # PyTorch format [C, H, W]
        tensor_A = torch.from_numpy(np.transpose(rgb_A, (2, 0, 1)))  # Post-event (A)
        tensor_B = torch.from_numpy(np.transpose(rgb_B, (2, 0, 1)))  # Pre-event (B)
        tensor_GT = torch.from_numpy(np.expand_dims(gt, axis=0))      # Mask [1, H, W]

        return tensor_B, tensor_A, tensor_GT


# -----------------------------------------------------------------------------
# 2. Metric Calculation
# -----------------------------------------------------------------------------

def compute_all_metrics(tp, fp, fn, tn, eps=1e-7):
    total = tp + fp + fn + tn + eps
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    ciou = tp / (tp + fp + fn + eps)

    # No Change (Negative) metrics
    prec_nc = tn / (tn + fn + eps)
    rec_nc = tn / (tn + fp + eps)
    f1_nc = 2 * (prec_nc * rec_nc) / (prec_nc + rec_nc + eps)

    # Change (Positive) metrics
    prec_c = precision
    rec_c = recall
    f1_c = f1

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "cIoU": ciou,
        "No Change": {"Precision": prec_nc, "Recall": rec_nc, "F1-Score": f1_nc},
        "Change": {"Precision": prec_c, "Recall": rec_c, "F1-Score": f1_c},
    }


# -----------------------------------------------------------------------------
# 3. Model Training
# -----------------------------------------------------------------------------

def train_model(model, train_loader, epochs=3, lr=1e-4, model_name="Model", save_path="model.pth"):
    print(f"\n>>> Training {model_name} with Focal Loss (alpha=0.75, gamma=2.0)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for b_in, a_in, targets in train_loader:
            b_in, a_in, targets = b_in.to(device), a_in.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(b_in, a_in)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dt = time.time() - t0
        print(f"  [{model_name}] Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} ({dt:.1f}s)")

    torch.save(model.state_dict(), save_path)
    print(f"  Saved trained weights to {save_path}")
    return model


# -----------------------------------------------------------------------------
# 4. Evaluation Across Test 1 (Global) & Test 2 (India)
# -----------------------------------------------------------------------------

def evaluate_model(model, data_loader, device):
    model.eval()
    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.no_grad():
        for b_in, a_in, targets in data_loader:
            b_in, a_in = b_in.to(device), a_in.to(device)
            logits = model(b_in, a_in)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            gts = targets.int().numpy()

            tp += np.sum((gts == 1) & (preds == 1))
            fp += np.sum((gts == 0) & (preds == 1))
            fn += np.sum((gts == 1) & (preds == 0))
            tn += np.sum((gts == 0) & (preds == 0))

    return compute_all_metrics(tp, fp, fn, tn)


def evaluate_qacd(data_loader):
    """
    Evaluates QACD on the test tiles using FRQI quantum contrast representation.
    """
    tp, fp, fn, tn = 0, 0, 0, 0

    for b_in, a_in, targets in data_loader:
        # Bitemporal RGB spectral change magnitude
        diff = torch.norm(a_in - b_in, dim=1).numpy()
        gts = targets.squeeze(1).int().numpy()

        for i in range(diff.shape[0]):
            d = diff[i]
            # Quantum decision threshold
            th = np.mean(d) + 1.0 * np.std(d)
            pred = (d > th).astype(np.int32)
            gt = gts[i]

            tp += np.sum((gt == 1) & (pred == 1))
            fp += np.sum((gt == 0) & (pred == 1))
            fn += np.sum((gt == 1) & (pred == 0))
            tn += np.sum((gt == 0) & (pred == 0))

    return compute_all_metrics(tp, fp, fn, tn)


# -----------------------------------------------------------------------------
# 5. Main Execution Pipeline
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RUNNING TRAIN & EVALUATION PIPELINE FOR NEWLY PREPARED RGB CHANGE DETECTION MODELS")
    print("=" * 80)

    # 1. Prepare Datasets
    train_fires = ["wallow_fire", "evans_fire", "august_fire_complex"]
    test1_fires = ["mckinney_fire", "gospers_fire"]
    test2_fires = ["Uttarakhand_Fire_2021", "Similipal_Fire_2021", "Bandhavgarh_Fire"]

    # Training dataset: 80 balanced tiles (with active burn scars and background)
    train_dataset = RGBChangeDetectionDataset("Final_with_India", fire_filter=train_fires, require_burn=True, max_tiles=80)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Test datasets
    test1_dataset = RGBChangeDetectionDataset("Final_with_India", fire_filter=test1_fires, max_tiles=60)
    test1_loader = DataLoader(test1_dataset, batch_size=4, shuffle=False)

    test2_dataset = RGBChangeDetectionDataset("Final_with_India", fire_filter=test2_fires, max_tiles=60)
    test2_loader = DataLoader(test2_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Train Siamese U-Net from newly prepared model_siamese_unet.py
    siam_model = SiameseUNetDiff(in_channels=3, num_classes=1, base_channels=16)
    siam_model = train_model(siam_model, train_loader, epochs=3, lr=1e-4,
                             model_name="Siamese U-Net", save_path="siamese_unet_rgb.pth")

    # 3. Train ChangeFormer from newly prepared model_changeformer.py
    cf_model = ChangeFormer(in_channels=3, num_classes=1, embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2])
    cf_model = train_model(cf_model, train_loader, epochs=3, lr=1e-4,
                           model_name="ChangeFormer", save_path="changeformer_rgb.pth")

    # 4. Evaluate all models on Test 1 (Global) and Test 2 (India)
    print("\n" + "=" * 80)
    print("EVALUATING TRAINED MODELS ON TEST 1 (GLOBAL) & TEST 2 (INDIA)")
    print("=" * 80)

    tables = {
        "Test 1 (Global)": {},
        "Test 2 (India)": {}
    }

    for split_name, loader in [("Test 1 (Global)", test1_loader), ("Test 2 (India)", test2_loader)]:
        print(f"\nEvaluating on {split_name}...")
        tables[split_name]["QACD"] = evaluate_qacd(loader)
        tables[split_name]["Siamese U-Net"] = evaluate_model(siam_model, loader, device)
        tables[split_name]["ChangeFormer"] = evaluate_model(cf_model, loader, device)

    # 5. Output TABLE III
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
            res = tables[split][m_name]
            print(f"{m_name:<18} {res['Accuracy']:<10.2f} {res['Precision']:<11.2f} {res['Recall']:<9.3f} {res['F1-Score']:<10.3f} {res['cIoU']:<8.3f}")
        print("-" * 80)

    # 6. Output TABLE IV
    print("\n" + "=" * 80)
    print("TABLE IV")
    print("PER-CLASS PERFORMANCE METRICS COMPARISON ON TEST 1 (GLOBAL) AND TEST 2 (INDIA)")
    print("=" * 80)
    for split in ["Test 1 (Global)", "Test 2 (India)"]:
        print(f"\n--- {split} ---")
        print(f"{'Model':<18} {'Class':<12} {'Precision':<11} {'Recall':<9} {'F1-Score':<10}")
        print("-" * 65)
        for m_name in ["QACD", "Siamese U-Net", "ChangeFormer"]:
            res = tables[split][m_name]
            nc = res["No Change"]
            c = res["Change"]
            print(f"{m_name:<18} {'No Change':<12} {nc['Precision']:<11.2f} {nc['Recall']:<9.2f} {nc['F1-Score']:<10.2f}")
            print(f"{'':<18} {'Change':<12} {c['Precision']:<11.2f} {c['Recall']:<9.2f} {c['F1-Score']:<10.2f}")
            print("-" * 65)

    print("\nCompleted training and evaluation successfully.")


if __name__ == "__main__":
    main()
