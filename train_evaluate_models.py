"""
================================================================================
UNIFIED TRAINING & EVALUATION BENCHMARK RUNNER
================================================================================
Allows running, training, and evaluating all 6 remade change detection models:
  1. siamese_unet   (FC-Siam-diff - Daudt et al., ICIP 2018)
  2. bit            (Bitemporal Image Transformer - Chen et al., IEEE TGRS 2021)
  3. changeformer   (ChangeFormer - Bandara & Patel, IGARSS 2022)
  4. a2net          (A2Net - Guan et al., IEEE TGRS 2023)
  5. classical_dnbr (USGS dNBR / Otsu baseline - Key & Benson, 2006)
  6. qcnn           (Quantum CNN with FRQI encoding - Cong et al., 2019)
================================================================================
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset_cd import get_dataloaders
from model_siamese_unet import SiameseUNetDiff, FocalLoss as SiameseFocalLoss
from model_bit import BitemporalImageTransformer, FocalLoss as BITFocalLoss
from model_changeformer import ChangeFormer, FocalLoss as ChangeFormerFocalLoss
from model_a2net import A2Net, FocalLoss as A2NetFocalLoss
from model_classical_dnbr import ClassicalDNBRBaseline
from model_qcnn import QCNNChangeDetector, QISKIT_AVAILABLE


def compute_metrics(y_true, y_pred, eps=1e-7):
    """
    Computes Accuracy, Precision, Recall, F1-Score, and Intersection-over-Union (IoU).
    """
    y_true = (y_true > 0.5).astype(np.uint8).flatten()
    y_pred = (y_pred > 0.5).astype(np.uint8).flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    accuracy = (tp + tn) / (len(y_true) + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "IoU": float(iou)
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(pre, post)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pre.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch["mask"]

            logits = model(pre, post)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_targets.append(mask.numpy())
            all_preds.append(probs)

    targets_arr = np.concatenate(all_targets, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)
    return compute_metrics(targets_arr, preds_arr)


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Change Detection Research Models")
    parser.add_argument("--model", type=str, default="siamese_unet",
                        choices=["siamese_unet", "bit", "changeformer", "a2net", "classical_dnbr", "qcnn"])
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--alpha", type=float, default=0.75, help="Focal loss alpha balancing factor")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss focusing parameter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    print("=" * 70)
    print(f"BENCHMARK RUNNER: Model='{args.model}' | Loss='{args.loss}'")
    print("=" * 70)

    # 1. Classical dNBR Baseline
    if args.model == "classical_dnbr":
        print("Running Classical dNBR Baseline...")
        baseline = ClassicalDNBRBaseline(default_threshold=0.100)
        _, _, test1_loader, test2_loader = get_dataloaders(mode="nbr", batch_size=4)
        
        for name, loader in [("Test 1 (Global)", test1_loader), ("Test 2 (India)", test2_loader)]:
            targets, preds = [], []
            for batch in loader:
                pre_nbr = batch["pre"].squeeze(1).numpy()
                post_nbr = batch["post"].squeeze(1).numpy()
                mask = batch["mask"].squeeze(1).numpy()
                dnbr = baseline.compute_dnbr(pre_nbr, post_nbr)
                pred = (dnbr >= baseline.default_threshold).astype(np.uint8)
                targets.append(mask)
                preds.append(pred)
            metrics = compute_metrics(np.concatenate(targets), np.concatenate(preds))
            print(f"Results on {name}: {metrics}")
        return

    # 2. QCNN Mode
    if args.model == "qcnn":
        print(f"Running QCNN (Qiskit Available: {QISKIT_AVAILABLE})...")
        detector = QCNNChangeDetector(n_qubits=8, n_blocks=3, shots=512)
        print(f"Initialized QCNN with {detector.n_params} parameters.")
        return

    # 3. Deep Learning PyTorch Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate selected research architecture
    if args.model == "siamese_unet":
        model = SiameseUNetDiff(in_channels=6, num_classes=1, base_channels=16).to(device)
    elif args.model == "bit":
        model = BitemporalImageTransformer(in_channels=6, num_classes=1, token_len=4, dim=32).to(device)
    elif args.model == "changeformer":
        model = ChangeFormer(in_channels=6, num_classes=1, embed_dims=[32, 64, 128, 256]).to(device)
    elif args.model == "a2net":
        model = A2Net(in_channels=6, num_classes=1, feat_dim=64).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | Trainable Parameters: {total_params:,}")

    # Loss Selection
    if args.loss == "focal":
        criterion = SiameseFocalLoss(alpha=args.alpha, gamma=args.gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, test1_loader, test2_loader = get_dataloaders(
        data_dir="Final_with_India", batch_size=args.batch_size, mode="multispectral"
    )

    print(f"\nStarting Training ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {loss:.4f} | Val F1: {val_metrics['F1']:.4f} | Val IoU: {val_metrics['IoU']:.4f}")

    print("\n--- Evaluating on Test Set 1 (Global: McKinney & Gospers Mountain) ---")
    t1_metrics = evaluate(model, test1_loader, device)
    print(f"Test 1 Metrics: {t1_metrics}")

    print("\n--- Evaluating on Test Set 2 (India: Uttarakhand, Similipal, Bandhavgarh) ---")
    t2_metrics = evaluate(model, test2_loader, device)
    print(f"Test 2 Metrics: {t2_metrics}")
    print("=" * 70)


if __name__ == "__main__":
    main()
