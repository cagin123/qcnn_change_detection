"""
Training and Evaluation CLI for the Proposed Hybrid QACD Model
(Quantum Automatic Change Detection for Wildfire Burned-Area Mapping).

Supports:
  1. Two-stage zero-leakage training (Stage A: 62-parameter QCNN warmup;
     Stage B: joint end-to-end Focal Loss optimization with AdamW).
  2. Controlled internal QACD ablation modes:
       - full               (62 trainable QCNN + 17,488 SpatialRefinementHead = 17,550 params)
       - qcnn_only          (62 trainable QCNN + 2 affine logit params = 64 params)
       - head_only          (17,488 SpatialRefinementHead alone, p_Q = 0.5)
       - classical_62_head  (62-param classical patch projector + 17,488 head = 17,550 params)
       - frozen_random_qcnn (62 frozen random QCNN + 17,488 trainable head)
  3. Validation decision-threshold calibration (tau*) on Bighorn Fire and exact
     integer (TP, FP, FN, TN) confusion-matrix evaluation across Val, Test 1,
     Test 2, and individual wildfire events with 1,000-resample tile-bootstrap 95% CIs.
  4. Instant checkpoint evaluation (--eval-only --checkpoint checkpoints/QACD_Proposed_seed42.pt).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_cd import SPLITS, TEST_FIRES, list_split_files, load_bitemporal_tile
from model_qacd import QACDGenuineHybrid, audit_qacd_complexity


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce * mask
        return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    n_tot = tp + fp + fn + tn
    oa = (tp + tn) / max(n_tot, 1)
    prec_c = tp / max(tp + fp, 1)
    rec_c = tp / max(tp + fn, 1)
    f1_c = 2.0 * prec_c * rec_c / max(prec_c + rec_c, 1e-12)
    ciou = tp / max(tp + fp + fn, 1)

    prec_nc = tn / max(tn + fn, 1)
    rec_nc = tn / max(tn + fp, 1)
    f1_nc = 2.0 * prec_nc * rec_nc / max(prec_nc + rec_nc, 1e-12)
    iou_nc = tn / max(tn + fp + fn, 1)
    miou = 0.5 * (ciou + iou_nc)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "N_valid": int(n_tot),
        "OA": round(oa, 4),
        "Precision": round(prec_c, 4),
        "Recall": round(rec_c, 4),
        "F1": round(f1_c, 4),
        "cIoU": round(ciou, 4),
        "NC_Precision": round(prec_nc, 4),
        "NC_Recall": round(rec_nc, 4),
        "NC_F1": round(f1_nc, 4),
        "IoU_NC": round(iou_nc, 4),
        "mIoU": round(miou, 4),
    }


def bootstrap_tile_ci(tile_counts: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n_tiles = len(tile_counts)
    idx = rng.integers(0, n_tiles, size=(n_boot, n_tiles))
    sums = tile_counts[idx].sum(axis=1)
    tp, fp, fn = sums[:, 0].astype(np.float64), sums[:, 1].astype(np.float64), sums[:, 2].astype(np.float64)
    f1 = (2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1.0)
    ciou = tp / np.maximum(tp + fp + fn, 1.0)
    return {
        "F1_ci95": [round(float(np.percentile(f1, 2.5)), 4), round(float(np.percentile(f1, 97.5)), 4)],
        "cIoU_ci95": [round(float(np.percentile(ciou, 2.5)), 4), round(float(np.percentile(ciou, 97.5)), 4)],
    }


def load_split_tiles(dataset_root: Path, split: str, spectral_mode: str) -> list:
    files = list_split_files(dataset_root, split)
    prefixes = SPLITS[split]
    tiles = []
    for fn in files:
        rec = load_bitemporal_tile(dataset_root, fn, spectral_mode=spectral_mode)
        rec["fire"] = next(pref for pref in prefixes if fn.startswith(pref))
        tiles.append(rec)
    return tiles


@torch.no_grad()
def predict_split(model: QACDGenuineHybrid, tiles: list, device: torch.device, stride: int = 4, batch_size: int = 8) -> list:
    model.eval()
    a_cached = (
        model.qcnn.build_heisenberg_real_matrix().to(device)
        if model.mode in ("full", "qcnn_only", "frozen_random_qcnn")
        else None
    )
    probs = []
    for st in range(0, len(tiles), batch_size):
        chunk = tiles[st : st + batch_size]
        pre = torch.stack([t["pre"] for t in chunk], dim=0).to(device)
        post = torch.stack([t["post"] for t in chunk], dim=0).to(device)
        logits = model(pre, post, stride=stride, a_cached=a_cached)
        p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        for i in range(len(chunk)):
            probs.append(p[i])
    return probs


def evaluate_probs_at_tau(tiles: list, probs: list, tau: float, with_ci: bool = True) -> dict:
    t_counts = np.zeros((len(tiles), 4), dtype=np.int64)
    for i, (rec, prob) in enumerate(zip(tiles, probs)):
        v = rec["valid"].squeeze(0).numpy() > 0.5
        g = rec["gt"].squeeze(0).numpy() > 0.5
        p = (prob >= tau) & v
        gv = g & v
        t_counts[i, 0] = int((p & gv).sum())
        t_counts[i, 1] = int((p & (~gv) & v).sum())
        t_counts[i, 2] = int(((~p) & gv).sum())
        t_counts[i, 3] = int(((~p) & (~gv) & v).sum())
    tot = t_counts.sum(axis=0)
    res = metrics_from_counts(tot[0], tot[1], tot[2], tot[3])
    res["tau"] = round(float(tau), 3)
    if with_ci and len(tiles) > 1:
        res.update(bootstrap_tile_ci(t_counts))
    return res


def calibrate_threshold_val(val_tiles: list, val_probs: list) -> float:
    best_tau, best_f1 = 0.50, -1.0
    for tau in np.linspace(0.10, 0.90, 17):
        res = evaluate_probs_at_tau(val_tiles, val_probs, float(tau), with_ci=False)
        if res["F1"] > best_f1:
            best_f1 = res["F1"]
            best_tau = float(tau)
    return round(best_tau, 3)


def train_qacd(
    model: QACDGenuineHybrid,
    train_tiles: list,
    device: torch.device,
    seed: int = 42,
    epochs: int = 12,
    warmup_epochs: int = 4,
    batch_size: int = 12,
    crop_size: int = 96,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)

    pos_tiles = [t for t in train_tiles if t["has_burn"]]
    neg_tiles = [t for t in train_tiles if not t["has_burn"]]

    def sample_epoch_batch_list(ep_seed: int):
        ep_rng = np.random.default_rng(ep_seed)
        sel_neg = ep_rng.choice(len(neg_tiles), size=min(80, len(neg_tiles)), replace=False)
        pool = list(pos_tiles) + [neg_tiles[i] for i in sel_neg]
        ep_rng.shuffle(pool)
        return pool

    def make_batch(chunk):
        pres, posts, gts, vals = [], [], [], []
        for rec in chunk:
            _, H, W = rec["pre"].shape
            r0 = int(rng.integers(0, max(1, H - crop_size + 1)))
            c0 = int(rng.integers(0, max(1, W - crop_size + 1)))
            pres.append(rec["pre"][:, r0 : r0 + crop_size, c0 : c0 + crop_size])
            posts.append(rec["post"][:, r0 : r0 + crop_size, c0 : c0 + crop_size])
            gts.append(rec["gt"][:, r0 : r0 + crop_size, c0 : c0 + crop_size])
            vals.append(rec["valid"][:, r0 : r0 + crop_size, c0 : c0 + crop_size])
        return (
            torch.stack(pres).to(device),
            torch.stack(posts).to(device),
            torch.stack(gts).to(device),
            torch.stack(vals).to(device),
        )

    # Stage A: QCNN core warmup (for hybrid full mode)
    if model.mode == "full" and warmup_epochs > 0:
        for p in model.refiner.parameters():
            p.requires_grad = False
        opt_w = torch.optim.AdamW([model.qcnn.theta], lr=8e-3, weight_decay=1e-4)
        for ep in range(warmup_epochs):
            model.train()
            pool = sample_epoch_batch_list(seed * 100 + ep)
            for st in range(0, len(pool), batch_size):
                pre, post, gt, val = make_batch(pool[st : st + batch_size])
                opt_w.zero_grad()
                logits = model(pre, post, stride=4)
                loss = criterion(logits, gt, val)
                loss.backward()
                opt_w.step()
        for p in model.refiner.parameters():
            p.requires_grad = True

    # Stage B: End-to-end joint Focal Loss training
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for ep in range(epochs):
        model.train()
        pool = sample_epoch_batch_list(seed * 1000 + ep)
        ep_loss = 0.0
        n_steps = 0
        for st in range(0, len(pool), batch_size):
            pre, post, gt, val = make_batch(pool[st : st + batch_size])
            opt.zero_grad()
            logits = model(pre, post, stride=4)
            loss = criterion(logits, gt, val)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_steps += 1
        sched.step()
        print(f"  Epoch {ep + 1:02d}/{epochs:02d} | focal_loss={ep_loss / max(n_steps, 1):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Proposed QACD Model")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Final_with_India"))
    parser.add_argument(
        "--mode",
        choices=("full", "qcnn_only", "head_only", "classical_62_head", "frozen_random_qcnn"),
        default="full",
    )
    parser.add_argument("--spectral-mode", choices=("swir_nir_red", "visible_rgb"), default="swir_nir_red")
    parser.add_argument("--no-patch-minmax", action="store_true")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/QACD_Proposed_seed42.pt"))
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QACDGenuineHybrid(
        stride=args.stride,
        init_seed=args.seed,
        mode=args.mode,
        patch_minmax=not args.no_patch_minmax,
        spectral_mode=args.spectral_mode,
    ).to(device)

    tau_star = 0.55
    if args.eval_only:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict) and "tau" in ckpt:
            tau_star = float(ckpt["tau"])
        print(f"Loaded checkpoint: {args.checkpoint} (tau*={tau_star:.3f})")
    else:
        print(f"Loading training tiles from {args.dataset_root} ...")
        train_tiles = load_split_tiles(args.dataset_root, "train", args.spectral_mode)
        train_qacd(model, train_tiles, device=device, seed=args.seed, epochs=args.epochs)

    val_tiles = load_split_tiles(args.dataset_root, "val", args.spectral_mode)
    t1_tiles = load_split_tiles(args.dataset_root, "test1", args.spectral_mode)
    t2_tiles = load_split_tiles(args.dataset_root, "test2", args.spectral_mode)

    val_probs = predict_split(model, val_tiles, device=device, stride=args.stride)
    if not args.eval_only:
        tau_star = calibrate_threshold_val(val_tiles, val_probs)

    t0 = time.perf_counter()
    t1_probs = predict_split(model, t1_tiles, device=device, stride=args.stride)
    t1_lat = (time.perf_counter() - t0) / max(len(t1_tiles), 1)
    t2_probs = predict_split(model, t2_tiles, device=device, stride=args.stride)

    val_res = evaluate_probs_at_tau(val_tiles, val_probs, tau_star, with_ci=True)
    t1_res = evaluate_probs_at_tau(t1_tiles, t1_probs, tau_star, with_ci=True)
    t2_res = evaluate_probs_at_tau(t2_tiles, t2_probs, tau_star, with_ci=True)

    per_fire = {}
    all_eval_tiles = t1_tiles + t2_tiles
    all_eval_probs = t1_probs + t2_probs
    for fire_name in TEST_FIRES:
        ft = [t for t in all_eval_tiles if t["fire"] == fire_name]
        fp = [p for t, p in zip(all_eval_tiles, all_eval_probs) if t["fire"] == fire_name]
        if ft:
            per_fire[fire_name] = evaluate_probs_at_tau(ft, fp, tau_star, with_ci=True)

    summary = {
        "mode": args.mode,
        "spectral_mode": args.spectral_mode,
        "patch_minmax": not args.no_patch_minmax,
        "stride": args.stride,
        "seed": args.seed,
        "tau_star": tau_star,
        "latency_sec_per_tile": round(t1_lat, 4),
        "complexity": audit_qacd_complexity(),
        "val": val_res,
        "test1": t1_res,
        "test2": t2_res,
        "per_fire": per_fire,
    }

    print(f"Val   (Bighorn): F1={val_res['F1']:.4f}, cIoU={val_res['cIoU']:.4f}, OA={val_res['OA']:.4f}")
    print(f"Test1 (Global) : F1={t1_res['F1']:.4f}, cIoU={t1_res['cIoU']:.4f}, OA={t1_res['OA']:.4f} ({t1_lat:.3f} s/tile)")
    print(f"Test2 (India)  : F1={t2_res['F1']:.4f}, cIoU={t2_res['cIoU']:.4f}, OA={t2_res['OA']:.4f}")
    if "Bandhavgarh_Fire" in per_fire:
        bd = per_fire["Bandhavgarh_Fire"]
        print(f"  Bandhavgarh  : F1={bd['F1']:.4f}, cIoU={bd['cIoU']:.4f}, OA={bd['OA']:.4f}")

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "tau": tau_star}, args.save_checkpoint)
        print(f"Saved checkpoint: {args.save_checkpoint}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved evaluation summary: {args.out_json}")


if __name__ == "__main__":
    main()
