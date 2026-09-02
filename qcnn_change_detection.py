import os, json, time
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_dir"         : BASE_DIR,
    "output_dir"       : os.path.join(BASE_DIR, "outputs"),
    "patch_size"       : 11,
    "stride"           : 8,
    "map_stride"       : 2,
    "label_burn_thresh": 0.1,
    "max_train_patches": 200,
    "max_test_patches" : 200,
    "max_images"       : 10,
    "max_images_train" : 30,
    "train_ratio"      : 0.8,
    "n_qubits"         : 8,
    "shots"            : 4096,
    "epochs"           : 35,
    "batch_size"       : 8,
    "lr"               : 0.02,
    "beta1"            : 0.8,
    "beta2"            : 0.9,
    "eps"              : 1e-8,
    "random_seed"      : 42,
}

TRAIN_DATASET = {
    "label"   : "Train (loc)",
    "A_dir"   : os.path.join("loc", "A"),
    "B_dir"   : os.path.join("loc", "B"),
    "gt_dir"  : os.path.join("loc", "ground_truth"),
}

TEST_DATASET = {
    "label"   : "Test (india_loc)",
    "A_dir"   : os.path.join("india_loc", "A_india"),
    "B_dir"   : os.path.join("india_loc", "B_india"),
    "gt_dir"  : os.path.join("india_loc", "ground_truth_india"),
}


def load_band(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float32)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def percentile_normalize(arr, lo=2, hi=98):
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    if p_hi - p_lo < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)


def frqi_normalize(vec):
    return np.clip(vec, 0.0, 1.0).astype(np.float32)


def extract_patches(img_a, img_b, gt, patch_size, stride, burn_thresh=0.1):
    H, W = img_a.shape
    X, y, coords = [], [], []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            pa   = img_a[i:i+patch_size, j:j+patch_size].flatten()
            pb   = img_b[i:i+patch_size, j:j+patch_size].flatten()
            raw  = np.concatenate([pa, pb]).astype(np.float32)
            feat = frqi_normalize(raw)
            label = int(gt[i:i+patch_size, j:j+patch_size].mean() > burn_thresh)
            X.append(feat)
            y.append(label)
            coords.append((i, j))
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(coords, dtype=np.int32))


def balanced_sample(X, y, max_patches, seed=42):
    np.random.seed(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_each  = min(max_patches // 2, len(idx_pos), len(idx_neg))
    if n_each == 0:
        return X, y
    sel = np.concatenate([
        np.random.choice(idx_pos, n_each, replace=False),
        np.random.choice(idx_neg, n_each, replace=False),
    ])
    np.random.shuffle(sel)
    return X[sel], y[sel]


def stratified_balanced_sample(per_image_Xy, image_names, max_patches, seed=42):
    rng = np.random.RandomState(seed)
    n_imgs = len(per_image_Xy)
    if n_imgs == 0:
        return np.empty((0,)), np.empty((0,))

    per_class_budget = max_patches // 2
    quota = max(1, per_class_budget // n_imgs)

    def collect(cls_val):
        chosen, shortfall, spares = [], 0, []
        per_image_taken = {}
        for img_i, (X_img, y_img) in enumerate(per_image_Xy):
            idx = np.where(y_img == cls_val)[0]
            take = min(quota, len(idx))
            sel  = rng.choice(idx, take, replace=False) if take > 0 else np.array([], dtype=int)
            if take > 0:
                chosen.append(X_img[sel])
            per_image_taken[img_i] = take
            shortfall += quota - take
            leftover = np.setdiff1d(idx, sel)
            if len(leftover) > 0:
                spares.append((X_img, leftover))
        rng.shuffle(spares)
        for X_img, leftover in spares:
            if shortfall <= 0:
                break
            take = min(shortfall, len(leftover))
            sel  = rng.choice(leftover, take, replace=False)
            chosen.append(X_img[sel])
            shortfall -= take
        return np.concatenate(chosen) if chosen else np.empty((0, per_image_Xy[0][0].shape[1]))

    X_pos, X_neg = collect(1), collect(0)
    n_each = min(len(X_pos), len(X_neg), per_class_budget)
    if n_each == 0:
        return np.empty((0,)), np.empty((0,))
    X_pos = X_pos[rng.choice(len(X_pos), n_each, replace=False)]
    X_neg = X_neg[rng.choice(len(X_neg), n_each, replace=False)]
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(n_each, dtype=np.int32), np.zeros(n_each, dtype=np.int32)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def load_dataset(cfg, dcfg, max_patches, role="train"):
    A_d    = os.path.join(cfg["data_dir"], dcfg["A_dir"])
    B_d    = os.path.join(cfg["data_dir"], dcfg["B_dir"])
    gt_d   = os.path.join(cfg["data_dir"], dcfg["gt_dir"])

    ext      = ('.tif', '.tiff')
    files_A  = sorted([f for f in os.listdir(A_d)  if f.lower().endswith(ext)])
    files_B  = sorted([f for f in os.listdir(B_d)  if f.lower().endswith(ext)])
    files_gt = sorted([f for f in os.listdir(gt_d) if f.lower().endswith(ext)])

    map_stride = cfg.get("map_stride", cfg["stride"])
    all_X, all_y, image_records, image_names = [], [], [], []

    for fA, fB, fg in zip(files_A, files_B, files_gt):
        raw_A = load_band(os.path.join(A_d, fA))
        raw_B = load_band(os.path.join(B_d, fB))

        norm_A = percentile_normalize(raw_A)
        norm_B = percentile_normalize(raw_B)

        with rasterio.open(os.path.join(gt_d, fg)) as src:
            gt = (src.read(1) > 0).astype(np.uint8)

        X_pool, y_pool, _ = extract_patches(
            norm_A, norm_B, gt,
            cfg["patch_size"], cfg["stride"],
            burn_thresh=cfg["label_burn_thresh"]
        )

        X_map, y_map, coords_map = extract_patches(
            norm_A, norm_B, gt,
            cfg["patch_size"], map_stride,
            burn_thresh=cfg["label_burn_thresh"]
        )

        image_records.append({
            "name"  : fA,
            "raw_A" : raw_A,
            "raw_B" : raw_B,
            "gt"    : gt,
            "shape" : raw_A.shape,
            "X"     : X_map,
            "y"     : y_map,
            "coords": coords_map,
        })
        all_X.append(X_pool)
        all_y.append(y_pool)
        image_names.append(fA)

    X, y = stratified_balanced_sample(list(zip(all_X, all_y)), image_names,
                                       max_patches, seed=cfg["random_seed"])

    return X, y, image_records


def frqi_encoding(qc, pixel_values, n_position_qubits):
    """FRQI (Flexible Representation of Quantum Images) encoding.

    Encodes a classical vector of up to 2**n_position_qubits pixel
    intensities (each expected in [0, 1]) into the state

        |I> = (1 / sqrt(2**n)) * sum_i ( cos(theta_i)|0> + sin(theta_i)|1> ) @ |i>

    where n = n_position_qubits and theta_i = pixel_values[i] * (pi / 2).

    Qubit layout expected on `qc` (qc.num_qubits == n_position_qubits + 1):
        [0 .. n_position_qubits-1]  -> "position" qubits (which pixel)
        [n_position_qubits]         -> "color" qubit (that pixel's intensity)
    """
    n_pixels  = 2 ** n_position_qubits
    color_q   = n_position_qubits
    pos_qubits = list(range(n_position_qubits))

    vals = np.zeros(n_pixels, dtype=np.float64)
    if len(pixel_values) >= n_pixels:
        vals[:] = np.asarray(pixel_values[:n_pixels], dtype=np.float64)
    else:
        vals[:len(pixel_values)] = np.asarray(pixel_values, dtype=np.float64)
    vals   = np.clip(vals, 0.0, 1.0)
    thetas = vals * (np.pi / 2)

    for q in pos_qubits:
        qc.h(q)

    for i in range(n_pixels):
        theta = thetas[i]
        if abs(theta) < 1e-10:
            continue
        if n_position_qubits > 0:
            ctrl_state = format(i, f"0{n_position_qubits}b")
            mcry = RYGate(2 * theta).control(n_position_qubits, ctrl_state=ctrl_state)
            qc.append(mcry, pos_qubits + [color_q])
        else:
            qc.ry(2 * theta, color_q)


def conv_block(qc, params, p, active):
    for i in range(0, len(active) - 1, 2):
        q0, q1 = active[i], active[i+1]
        qc.ry(params[p],   q0); qc.rz(params[p+1], q0)
        qc.ry(params[p+2], q1); qc.cx(q0, q1)
        qc.ry(params[p+3], q1); p += 4
    for i in range(1, len(active) - 1, 2):
        q0, q1 = active[i], active[i+1]
        qc.ry(params[p],   q0); qc.rz(params[p+1], q0)
        qc.ry(params[p+2], q1); qc.cx(q0, q1)
        qc.ry(params[p+3], q1); p += 4
    return p


def pool_block(qc, params, p, active):
    n, surv = len(active), []
    for i in range(n // 2):
        src = active[i]; tgt = active[n // 2 + i]
        qc.crz(params[p], src, tgt); qc.cry(params[p+1], src, tgt)
        p += 2; surv.append(tgt)
    return p, surv


def count_params_deep(n_qubits, n_blocks=3):
    active, total = n_qubits, 0
    for _ in range(n_blocks):
        if active < 2:
            break
        total += (active // 2 + (active - 1) // 2) * 4
        total += (active // 2) * 2
        active = active // 2
    return total


def build_qcnn_circuit(state_vector, params, n_qubits=8, n_blocks=3):
    total_qubits = n_qubits + 1
    qc = QuantumCircuit(total_qubits, 1)
    frqi_encoding(qc, state_vector, n_qubits)
    active = list(range(total_qubits))
    p      = 0
    pl     = params.tolist()
    for _ in range(n_blocks):
        if len(active) < 2:
            break
        p, active = pool_block(qc, pl, conv_block(qc, pl, p, active), active)
    qc.measure(active[0], 0)
    return qc


class QCNNChangeDetector:
    def __init__(self, n_qubits=8, n_blocks=3, shots=1024, seed=42):
        self.n_qubits     = n_qubits
        self.total_qubits = n_qubits + 1
        self.n_blocks   = n_blocks
        self.shots      = shots
        self.n_params   = count_params_deep(self.total_qubits, n_blocks)
        self.params_    = None
        self.loss_hist_ = []
        self.tr_acc_hist_  = []
        self.val_acc_hist_ = []
        self.val_f1_hist_  = []
        self.backend    = AerSimulator()
        np.random.seed(seed)

    def _run(self, x, params):
        qc  = build_qcnn_circuit(x, params, self.n_qubits, self.n_blocks)
        tqc    = transpile(qc, self.backend, optimization_level=0)
        counts = self.backend.run(tqc, shots=self.shots).result().get_counts()
        return counts.get("1", 0) / self.shots

    def _gradient(self, x, params, idx):
        shift   = np.pi / 2
        p_plus  = params.copy(); p_plus[idx]  += shift
        p_minus = params.copy(); p_minus[idx] -= shift
        return (self._run(x, p_plus) - self._run(x, p_minus)) / 2

    def _batch_gradients(self, X, y, params):
        grads = np.zeros(self.n_params)
        probs = np.clip(np.array([self._run(x, params) for x in X]), 1e-8, 1-1e-8)
        dpL   = -(y / probs - (1 - y) / (1 - probs)) / len(X)
        for i in range(self.n_params):
            dpf = np.array([self._gradient(x, params, i) for x in X])
            grads[i] = np.dot(dpL, dpf)
        return grads

    def fit(self, X_tr, y_tr, X_val=None, y_val=None,
            epochs=15, batch_size=8, lr=0.05,
            beta1=0.9, beta2=0.99, eps=1e-8):

        N      = len(X_tr)
        params = np.random.uniform(0, 2 * np.pi, self.n_params)
        m      = np.zeros(self.n_params)
        v      = np.zeros(self.n_params)
        t      = 0

        best_params = params.copy()
        best_val_f1 = -1.0

        for epoch in range(1, epochs + 1):
            idx   = np.random.permutation(N)
            X_s, y_s = X_tr[idx], y_tr[idx]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                Xb    = X_s[start:start + batch_size]
                yb    = y_s[start:start + batch_size]
                grads = self._batch_gradients(Xb, yb, params)
                t    += 1
                m     = beta1 * m + (1 - beta1) * grads
                v     = beta2 * v + (1 - beta2) * grads ** 2
                m_h   = m / (1 - beta1 ** t)
                v_h   = v / (1 - beta2 ** t)
                params   -= lr * m_h / (np.sqrt(v_h) + eps)
                probs     = np.clip(
                    np.array([self._run(x, params) for x in Xb]), 1e-8, 1-1e-8)
                epoch_loss += -np.mean(yb * np.log(probs) + (1-yb) * np.log(1-probs))
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_hist_.append(avg_loss)

            tr_pred = (np.array([self._run(x, params) for x in X_tr]) >= 0.5).astype(int)
            tr_acc  = accuracy_score(y_tr, tr_pred)
            self.tr_acc_hist_.append(tr_acc)

            if X_val is not None:
                vp   = (np.array([self._run(x, params) for x in X_val]) >= 0.5).astype(int)
                vacc = accuracy_score(y_val, vp)
                vf1  = f1_score(y_val, vp, zero_division=0)
                self.val_acc_hist_.append(vacc)
                self.val_f1_hist_.append(vf1)
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    best_params = params.copy()
            else:
                self.val_acc_hist_.append(None)
                self.val_f1_hist_.append(None)

        self.params_ = best_params if X_val is not None else params
        return self

    def predict_proba(self, X):
        assert self.params_ is not None, "Model not trained yet!"
        return np.array([self._run(x, self.params_) for x in X])

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)


def build_prediction_map(model, record, patch_size, stride):
    H, W     = record["shape"]
    pred_map = np.zeros((H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.int32)

    probas = model.predict_proba(record["X"])

    for idx, (r, c) in enumerate(record["coords"]):
        pred_map[r:r+patch_size, c:c+patch_size] += probas[idx]
        coverage[r:r+patch_size, c:c+patch_size] += 1

    mask = coverage > 0
    pred_map[mask] /= coverage[mask]
    return pred_map, (pred_map >= 0.5).astype(np.uint8), record["gt"]


def save_prediction_maps(model, image_records, patch_size, stride, out_dir,
                         dataset_label="", max_images=10):
    os.makedirs(out_dir, exist_ok=True)
    bw_cmap = ListedColormap(["blue", "red"])

    if len(image_records) > max_images:
        np.random.seed(42)
        selected_idx    = sorted(np.random.choice(len(image_records), max_images, replace=False))
        records_to_plot = [image_records[i] for i in selected_idx]
    else:
        records_to_plot = image_records

    for rec in records_to_plot:
        name = os.path.splitext(rec["name"])[0]
        pred_map, pred_bin, gt = build_prediction_map(model, rec, patch_size, stride)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        fig.suptitle(
            f"QCNN Prediction Map  -  {name}  [{dataset_label}]",
            fontsize=16, fontweight="bold", y=1.04
        )

        axes[0].imshow(rec["raw_B"], cmap="gray")
        axes[0].set_title("Band B (Pre-Fire)", fontsize=14); axes[0].axis("off")

        axes[1].imshow(rec["raw_A"], cmap="gray")
        axes[1].set_title("Band A (Post-Fire)", fontsize=14); axes[1].axis("off")

        axes[2].imshow(gt, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title("Ground Truth", fontsize=14); axes[2].axis("off")
        axes[2].legend(handles=[mpatches.Patch(color="blue", label="No Change"),
                                  mpatches.Patch(color="red", label="Burned")],
                        loc="lower right", fontsize=11, framealpha=0.85)

        axes[3].imshow(pred_bin, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[3].set_title("QCNN Prediction (binary)", fontsize=14); axes[3].axis("off")
        axes[3].legend(handles=[mpatches.Patch(color="blue", label="No Change"),
                                  mpatches.Patch(color="red", label="Predicted Burned")],
                        loc="lower right", fontsize=11, framealpha=0.85)

        H_img, W_img = rec["shape"]
        for r in range(0, H_img, patch_size):
            for ax in axes[2:4]: ax.axhline(r, color="gray", lw=0.15, alpha=0.25)
        for c in range(0, W_img, patch_size):
            for ax in axes[2:4]: ax.axvline(c, color="gray", lw=0.15, alpha=0.25)

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        png_path = os.path.join(out_dir, f"pred_map_{name}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close("all")

        tif_path = os.path.join(out_dir, f"pred_binary_{name}.tif")
        with rasterio.open(tif_path, "w", driver="GTiff",
                           height=H_img, width=W_img,
                           count=1, dtype=rasterio.uint8) as dst:
            dst.write(pred_bin, 1)

        prob_path = os.path.join(out_dir, f"pred_proba_{name}.tif")
        with rasterio.open(prob_path, "w", driver="GTiff",
                           height=H_img, width=W_img,
                           count=1, dtype=rasterio.float32) as dst:
            dst.write(pred_map, 1)


def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    np.random.seed(cfg["random_seed"])

    X_all, y_all, loc_records = load_dataset(
        cfg, TRAIN_DATASET,
        max_patches=cfg["max_train_patches"],
        role="train"
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all,
        test_size=1 - cfg["train_ratio"],
        stratify=y_all,
        random_state=cfg["random_seed"]
    )

    X_te, y_te, india_records = load_dataset(
        cfg, TEST_DATASET,
        max_patches=cfg["max_test_patches"],
        role="test"
    )

    model = QCNNChangeDetector(
        n_qubits=cfg["n_qubits"], n_blocks=3,
        shots=cfg["shots"], seed=cfg["random_seed"]
    )

    model.fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], beta1=cfg["beta1"], beta2=cfg["beta2"], eps=cfg["eps"]
    )

    out_train = os.path.join(cfg["output_dir"], "loc_train")
    os.makedirs(out_train, exist_ok=True)

    out_test = os.path.join(cfg["output_dir"], "india_test")
    os.makedirs(out_test, exist_ok=True)

    save_prediction_maps(
        model, india_records,
        cfg["patch_size"], cfg["stride"],
        out_test, dataset_label=TEST_DATASET["label"],
        max_images=cfg.get("max_images", 10)
    )

    save_prediction_maps(
        model, loc_records,
        cfg["patch_size"], cfg["stride"],
        out_train, dataset_label=TRAIN_DATASET["label"],
        max_images=cfg.get("max_images_train", 10)
    )

    np.save(os.path.join(cfg["output_dir"], "trained_params.npy"), model.params_)


if __name__ == "__main__":
    main()
