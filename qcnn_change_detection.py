import os, json, time, pickle, hashlib
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIF_EXT = (".tif", ".tiff")

CONFIG = {
    "data_dir"          : BASE_DIR,
    "output_dir"        : os.path.join(BASE_DIR, "outputs"),

    "encoding_mode"     : "concat",
    "n_qubits"          : 8,
    "patch_size"        : 11,
    "band"              : 1,
    "band_r"            : 4,
    "band_g"            : 3,
    "band_b"            : 2,
    "use_rgb"           : True,
    "downsample_factor" : 2,

    "stride"            : 11,
    "pool_cap_per_image": 100,
    "map_stride"        : 6,
    "map_chunk"         : 20000,
    "label_burn_thresh" : 0.1,
    "max_train_patches" : 320,
    "max_val_patches"   : 100,
    "calib_patches"     : 200,
    "val_image_frac"    : 0.2,
    "pool_seed"         : 0,

    "max_images"        : 15,
    "map_train_images"  : False,
    "max_images_train"  : 3,
    "map_val_images"    : False,
    "plot_dpi"          : 100,

    "epochs"            : 40,
    "batch_size"        : 16,
    "lr"                : 0.02,
    "beta1"             : 0.9,
    "beta2"             : 0.99,
    "eps"               : 1e-8,
    "seeds"             : [42, 43],
    "resume"            : True,
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

CKPT_KEYS = ["encoding_mode", "n_qubits", "patch_size", "band", "band_r", "band_g",
             "band_b", "use_rgb", "downsample_factor", "stride", "pool_cap_per_image",
             "label_burn_thresh", "max_train_patches", "max_val_patches",
             "calib_patches", "val_image_frac", "pool_seed", "batch_size", "lr",
             "beta1", "beta2", "eps"]
RESULT_KEYS = CKPT_KEYS + ["epochs", "map_stride", "max_images"]


def config_signature(cfg, keys):
    payload = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def validate_config(cfg):
    n_pos = 2 ** cfg["n_qubits"]
    px = cfg["patch_size"] ** 2
    mode = cfg["encoding_mode"]
    if mode == "concat":
        if px > n_pos // 2:
            raise ValueError(f"concat mode: patch_size**2={px} > {n_pos // 2}; "
                             f"for n_qubits={cfg['n_qubits']} patch_size must be <= "
                             f"{int(np.sqrt(n_pos // 2))}.")
    elif mode == "diff":
        if px > n_pos:
            raise ValueError(f"diff mode: patch_size**2={px} > {n_pos}.")
    else:
        raise ValueError("encoding_mode must be 'concat' or 'diff'.")
    if cfg["stride"] < 1 or cfg["map_stride"] < 1:
        raise ValueError("stride and map_stride must be >= 1.")


def inspect_bands(cfg):
    if cfg["use_rgb"]:
        used_img = (cfg["band_r"], cfg["band_g"], cfg["band_b"])
    else:
        used_img = cfg["band"]
    for dcfg in (TRAIN_DATASET, TEST_DATASET):
        for key in ("A_dir", "B_dir", "gt_dir"):
            d = os.path.join(cfg["data_dir"], dcfg[key])
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(TIF_EXT))
            if not files:
                print(f"[bands] {dcfg[key]}: no files found!")
                continue
            with rasterio.open(os.path.join(d, files[0])) as src:
                used = 1 if key == "gt_dir" else used_img
                print(f"[bands] {dcfg[key]}/{files[0]}: count={src.count}, "
                      f"dtype={src.dtypes[0]}, descriptions={src.descriptions}, "
                      f"band(s) read={used}")


def downsample_mean(arr, factor):
    if factor <= 1:
        return arr
    H, W = arr.shape
    H2, W2 = H - (H % factor), W - (W % factor)
    a = arr[:H2, :W2]
    a = a.reshape(H2 // factor, factor, W2 // factor, factor)
    return a.mean(axis=(1, 3)).astype(arr.dtype)


def clean(x):
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def read_band(path, band):
    with rasterio.open(path) as src:
        if band > src.count:
            raise ValueError(f"{path}: file has {src.count} bands, band={band} requested.")
        d = src.read(band)
    return clean(d)


def read_rgb_luminance(path, band_r, band_g, band_b):
    with rasterio.open(path) as src:
        for b in (band_r, band_g, band_b):
            if b > src.count:
                raise ValueError(f"{path}: file has {src.count} bands, band={b} requested.")
        r, g, b = clean(src.read(band_r)), clean(src.read(band_g)), clean(src.read(band_b))
    return 0.299 * r + 0.587 * g + 0.114 * b


def read_rgb_composite(path, band_r, band_g, band_b):
    with rasterio.open(path) as src:
        r, g, b = clean(src.read(band_r)), clean(src.read(band_g)), clean(src.read(band_b))

    def scale(x):
        lo, hi = np.percentile(x, [2, 98])
        if hi - lo < 1e-10:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    return np.stack([scale(r), scale(g), scale(b)], axis=-1)


def read_gt(path):
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(np.uint8)


def joint_percentile_normalize(a, b, lo=2, hi=98):
    both = np.concatenate([a.ravel(), b.ravel()])
    p_lo, p_hi = np.percentile(both, [lo, hi])
    if p_hi - p_lo < 1e-10:
        z = np.zeros_like(a)
        return z, z.copy()

    def f(x):
        return np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)

    return f(a), f(b)


def grid_coords(H, W, ps, stride):
    rows = np.arange(0, H - ps + 1, stride)
    cols = np.arange(0, W - ps + 1, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int32)


def featurize(A, B, gt, coords, cfg):
    ps = cfg["patch_size"]
    px = ps * ps
    n_pos = 2 ** cfg["n_qubits"]
    mode = cfg["encoding_mode"]
    n = len(coords)
    if mode == "concat":
        half = n_pos // 2
        X = np.zeros((n, n_pos), dtype=np.float32)
    else:
        X = np.full((n, n_pos), 0.5, dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    thr = cfg["label_burn_thresh"]
    for k in range(n):
        i, j = int(coords[k, 0]), int(coords[k, 1])
        pa = A[i:i + ps, j:j + ps].ravel()
        pb = B[i:i + ps, j:j + ps].ravel()
        if mode == "concat":
            X[k, :px] = pa
            X[k, half:half + px] = pb
        else:
            X[k, :px] = np.clip((pa - pb) * 0.5 + 0.5, 0.0, 1.0)
        y[k] = int(gt[i:i + ps, j:j + ps].mean() > thr)
    return X, y


def load_images(cfg, dcfg, with_pool):
    A_d = os.path.join(cfg["data_dir"], dcfg["A_dir"])
    B_d = os.path.join(cfg["data_dir"], dcfg["B_dir"])
    G_d = os.path.join(cfg["data_dir"], dcfg["gt_dir"])
    fA = set(f for f in os.listdir(A_d) if f.lower().endswith(TIF_EXT))
    fB = set(f for f in os.listdir(B_d) if f.lower().endswith(TIF_EXT))
    fG = set(f for f in os.listdir(G_d) if f.lower().endswith(TIF_EXT))

    common = sorted(fA & fB & fG)
    if not common:
        raise ValueError(f"{dcfg['label']}: no common file names across A/B/gt "
                         f"(A={len(fA)}, B={len(fB)}, gt={len(fG)}).")
    dropped_A, dropped_B, dropped_G = fA - set(common), fB - set(common), fG - set(common)
    if dropped_A or dropped_B or dropped_G:
        print(f"[data] {dcfg['label']}: matched by file name. "
              f"Common={len(common)}  (extra in A={len(dropped_A)}, "
              f"extra in B={len(dropped_B)}, extra in gt={len(dropped_G)})")

    ds = cfg.get("downsample_factor", 1)
    rng = np.random.RandomState(cfg["pool_seed"])
    records = []
    for name in common:
        pA, pB, pG = os.path.join(A_d, name), os.path.join(B_d, name), os.path.join(G_d, name)
        if cfg["use_rgb"]:
            raw_A = read_rgb_luminance(pA, cfg["band_r"], cfg["band_g"], cfg["band_b"])
            raw_B = read_rgb_luminance(pB, cfg["band_r"], cfg["band_g"], cfg["band_b"])
        else:
            raw_A = read_band(pA, cfg["band"])
            raw_B = read_band(pB, cfg["band"])
        gt = read_gt(pG)
        if not (raw_A.shape == raw_B.shape == gt.shape):
            raise ValueError(f"{name}: shape mismatch "
                             f"{raw_A.shape} {raw_B.shape} {gt.shape}")

        if ds > 1:
            raw_A = downsample_mean(raw_A, ds)
            raw_B = downsample_mean(raw_B, ds)
            gt_frac = downsample_mean(gt.astype(np.float32), ds)
            gt = (gt_frac > cfg["label_burn_thresh"]).astype(np.uint8)

        A, B = joint_percentile_normalize(raw_A, raw_B)
        rec = {"name": name, "raw_A": raw_A, "raw_B": raw_B, "A": A, "B": B,
               "gt": gt, "shape": raw_A.shape}
        if cfg["use_rgb"]:
            rgb_A = read_rgb_composite(pA, cfg["band_r"], cfg["band_g"], cfg["band_b"])
            rgb_B = read_rgb_composite(pB, cfg["band_r"], cfg["band_g"], cfg["band_b"])
            if ds > 1:
                rgb_A = np.stack([downsample_mean(rgb_A[..., c], ds) for c in range(3)], axis=-1)
                rgb_B = np.stack([downsample_mean(rgb_B[..., c], ds) for c in range(3)], axis=-1)
            rec["rgb_A"], rec["rgb_B"] = rgb_A, rgb_B
        if with_pool:
            coords = grid_coords(*raw_A.shape, cfg["patch_size"], cfg["stride"])
            cap = cfg["pool_cap_per_image"]
            if cap and len(coords) > cap:
                coords = coords[rng.choice(len(coords), cap, replace=False)]
            rec["pool_X"], rec["pool_y"] = featurize(A, B, gt, coords, cfg)
        records.append(rec)
    print(f"[data] {dcfg['label']}: loaded {len(records)} images")
    return records


def stratified_balanced_sample(per_image_Xy, max_patches, seed=42):
    rng = np.random.RandomState(seed)
    n_imgs = len(per_image_Xy)
    if n_imgs == 0:
        raise ValueError("No images to sample from.")
    n_feat = per_image_Xy[0][0].shape[1]
    per_class_budget = max_patches // 2
    quota = max(1, per_class_budget // n_imgs)

    def collect(cls_val):
        chosen, shortfall, spares = [], 0, []
        for X_img, y_img in per_image_Xy:
            idx = np.where(y_img == cls_val)[0]
            take = min(quota, len(idx))
            sel = rng.choice(idx, take, replace=False) if take > 0 else np.array([], dtype=int)
            if take > 0:
                chosen.append(X_img[sel])
            shortfall += quota - take
            leftover = np.setdiff1d(idx, sel)
            if len(leftover) > 0:
                spares.append((X_img, leftover))
        for k in rng.permutation(len(spares)):
            if shortfall <= 0:
                break
            X_img, leftover = spares[k]
            take = min(shortfall, len(leftover))
            chosen.append(X_img[rng.choice(leftover, take, replace=False)])
            shortfall -= take
        return np.concatenate(chosen) if chosen else np.empty((0, n_feat), np.float32)

    X_pos, X_neg = collect(1), collect(0)
    n_each = min(len(X_pos), len(X_neg), per_class_budget)
    if n_each == 0:
        raise ValueError("One class has no samples (positive/negative). "
                         "Check label_burn_thresh or the image split.")
    X_pos = X_pos[rng.choice(len(X_pos), n_each, replace=False)]
    X_neg = X_neg[rng.choice(len(X_neg), n_each, replace=False)]
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(n_each, np.int32), np.zeros(n_each, np.int32)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def natural_sample(records, n, seed):
    X = np.concatenate([r["pool_X"] for r in records])
    y = np.concatenate([r["pool_y"] for r in records])
    if len(X) > n:
        idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def split_images(records, val_frac, seed):
    rng = np.random.RandomState(seed)
    n = len(records)
    n_val = max(1, int(round(n * val_frac)))
    if n - n_val < 1:
        raise ValueError("Not enough images for training.")
    pos = np.array([int(r["pool_y"].sum()) for r in records])
    neg = np.array([int((1 - r["pool_y"]).sum()) for r in records])
    for _ in range(500):
        perm = rng.permutation(n)
        va, tr = perm[:n_val], perm[n_val:]
        if pos[tr].sum() > 0 and neg[tr].sum() > 0 and pos[va].sum() > 0 and neg[va].sum() > 0:
            return sorted(tr.tolist()), sorted(va.tolist())
    raise ValueError("Could not find an image split with both classes on both sides.")


def multiplexed_angles_numeric(angles):
    n = len(angles)
    if n == 1:
        return [angles[0]]
    half = n // 2
    angles1 = [(angles[i] + angles[i + half]) / 2 for i in range(half)]
    angles2 = [(angles[i] - angles[i + half]) / 2 for i in range(half)]
    return multiplexed_angles_numeric(angles1) + multiplexed_angles_numeric(angles2)


def multiplexed_ry_structure(qc, leaf_params_iter, controls, target):
    n = len(controls)
    if n == 0:
        qc.ry(next(leaf_params_iter), target)
        return
    multiplexed_ry_structure(qc, leaf_params_iter, controls[1:], target)
    qc.cx(controls[0], target)
    multiplexed_ry_structure(qc, leaf_params_iter, controls[1:], target)
    qc.cx(controls[0], target)


def conv_block(qc, params, p, active):
    for i in range(0, len(active) - 1, 2):
        q0, q1 = active[i], active[i + 1]
        qc.ry(params[p],     q0); qc.rz(params[p + 1], q0)
        qc.ry(params[p + 2], q1); qc.cx(q0, q1)
        qc.ry(params[p + 3], q1); p += 4
    for i in range(1, len(active) - 1, 2):
        q0, q1 = active[i], active[i + 1]
        qc.ry(params[p],     q0); qc.rz(params[p + 1], q0)
        qc.ry(params[p + 2], q1); qc.cx(q0, q1)
        qc.ry(params[p + 3], q1); p += 4
    return p


def pool_block(qc, params, p, active):
    n = len(active)
    half = n // 2
    survivors = []
    for i in range(half):
        src, tgt = active[i], active[half + i]
        qc.ry(params[p],     tgt); qc.cx(src, tgt)
        qc.ry(params[p + 1], tgt); qc.cx(src, tgt)
        qc.rz(params[p + 2], tgt); qc.cx(src, tgt)
        qc.rz(params[p + 3], tgt); qc.cx(src, tgt)
        p += 4
        survivors.append(tgt)
    if n % 2 == 1:
        survivors.append(active[-1])
    return p, survivors


def count_params(n_qubits):
    total = n_qubits + 1
    dummy = QuantumCircuit(total)
    zeros = [0.0] * 100000
    active, p = list(range(total)), 0
    while len(active) > 1:
        p = conv_block(dummy, zeros, p, active)
        p, active = pool_block(dummy, zeros, p, active)
    return p


class QCNNChangeDetector:
    def __init__(self, n_qubits=8, seed=42):
        self.n_qubits = n_qubits
        self.n_params = count_params(n_qubits)
        self.params_ = None
        self.best_val_f1_ = None
        self.loss_hist_, self.tr_acc_hist_ = [], []
        self.val_acc_hist_, self.val_f1_hist_ = [], []
        self.backend = AerSimulator()
        self.backend.set_options(max_parallel_threads=0, max_parallel_experiments=0)
        self.rng = np.random.RandomState(seed)
        self.p_eps = 1e-6

        total = n_qubits + 1
        self.n_pixels = 2 ** n_qubits

        self._leaf_params = [Parameter(f"leaf{i}") for i in range(self.n_pixels)]
        self._var_params = [Parameter(f"th{i}") for i in range(self.n_params)]

        qc = QuantumCircuit(total)
        pos_qubits = list(range(n_qubits))
        color_q = n_qubits
        for q in pos_qubits:
            qc.h(q)

        leaf_iter = iter(self._leaf_params)
        multiplexed_ry_structure(qc, leaf_iter, list(reversed(pos_qubits)), color_q)

        active, p = list(range(total)), 0
        pl = self._var_params
        while len(active) > 1:
            p = conv_block(qc, pl, p, active)
            p, active = pool_block(qc, pl, p, active)
        self._readout_qubit = active[0]
        qc.save_statevector()

        self._tqc = transpile(qc, self.backend, optimization_level=0)

    def _bind_one(self, x, params):
        vals = np.clip(np.asarray(x[:self.n_pixels], dtype=np.float64), 0.0, 1.0)
        angles = list(2.0 * vals * (np.pi / 2))
        leaf_angles = multiplexed_angles_numeric(angles)
        bind_map = dict(zip(self._leaf_params, leaf_angles))
        bind_map.update(dict(zip(self._var_params, [float(v) for v in params])))
        return self._tqc.assign_parameters(bind_map)

    def _run_batch(self, xs, params_list):
        bound_list = [self._bind_one(x, p) for x, p in zip(xs, params_list)]
        result = self.backend.run(bound_list, shots=1).result()
        probs = np.empty(len(bound_list))
        for i in range(len(bound_list)):
            sv = result.data(i)["statevector"]
            probs[i] = Statevector(sv).probabilities([self._readout_qubit])[1]
        return probs

    def _batch_gradients(self, X, y, params):
        n = len(X)
        shift = np.pi / 2
        xs_all, params_all = [], []

        xs_all.extend(X); params_all.extend([params] * n)
        for i in range(self.n_params):
            pp = params.copy(); pp[i] += shift
            pm = params.copy(); pm[i] -= shift
            xs_all.extend(X); params_all.extend([pp] * n)
            xs_all.extend(X); params_all.extend([pm] * n)

        all_probs = self._run_batch(xs_all, params_all)

        probs = np.clip(all_probs[:n], self.p_eps, 1 - self.p_eps)
        dpL = -(y / probs - (1 - y) / (1 - probs)) / n

        grads = np.zeros(self.n_params)
        offset = n
        for i in range(self.n_params):
            pprobs = all_probs[offset:offset + n]; offset += n
            mprobs = all_probs[offset:offset + n]; offset += n
            grads[i] = np.dot(dpL, (pprobs - mprobs) / 2)
        return grads, probs

    def _save_ckpt(self, path, state):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, path)

    def fit(self, X_tr, y_tr, X_val, y_val, epochs, batch_size, lr,
            beta1, beta2, eps, ckpt_path=None, sig=None):
        N = len(X_tr)
        st = None
        if ckpt_path and os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                st = pickle.load(f)
            if st.get("sig") != sig or len(st["params"]) != self.n_params:
                print("    [resume] checkpoint does not match current settings, starting fresh")
                st = None

        if st is not None:
            params, m, v, t = st["params"], st["m"], st["v"], st["t"]
            best_params, best_val_f1 = st["best_params"], st["best_val_f1"]
            best_val_loss = st.get("best_val_loss", np.inf)
            start = st["epoch"] + 1
            self.loss_hist_, self.tr_acc_hist_ = st["loss_hist"], st["tr_acc_hist"]
            self.val_acc_hist_, self.val_f1_hist_ = st["val_acc_hist"], st["val_f1_hist"]
            self.rng.set_state(st["rng_state"])
            print(f"    [resume] checkpoint found, continuing from epoch {start}")
        else:
            params = self.rng.uniform(0, 2 * np.pi, self.n_params)
            m, v, t = np.zeros(self.n_params), np.zeros(self.n_params), 0
            best_params, best_val_f1, best_val_loss = params.copy(), -1.0, np.inf
            start = 1

        t0 = time.time()
        for epoch in range(start, epochs + 1):
            idx = self.rng.permutation(N)
            X_s, y_s = X_tr[idx], y_tr[idx]
            epoch_loss, n_batches = 0.0, 0

            for s in range(0, N, batch_size):
                Xb, yb = X_s[s:s + batch_size], y_s[s:s + batch_size]
                grads, probs = self._batch_gradients(Xb, yb, params)
                epoch_loss += -np.mean(yb * np.log(probs) + (1 - yb) * np.log(1 - probs))
                n_batches += 1
                t += 1
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * grads ** 2
                m_h = m / (1 - beta1 ** t)
                v_h = v / (1 - beta2 ** t)
                params = params - lr * m_h / (np.sqrt(v_h) + eps)

            self.loss_hist_.append(float(epoch_loss / max(n_batches, 1)))
            tr_probs = self._run_batch(list(X_tr), [params] * len(X_tr))
            self.tr_acc_hist_.append(float(accuracy_score(y_tr, (tr_probs >= 0.5).astype(int))))
            val_probs = self._run_batch(list(X_val), [params] * len(X_val))
            vp = (val_probs >= 0.5).astype(int)
            vacc, vf1 = float(accuracy_score(y_val, vp)), float(f1_score(y_val, vp, zero_division=0))
            pc = np.clip(val_probs, self.p_eps, 1 - self.p_eps)
            vloss = float(-np.mean(y_val * np.log(pc) + (1 - y_val) * np.log(1 - pc)))
            self.val_acc_hist_.append(vacc)
            self.val_f1_hist_.append(vf1)
            if vf1 > best_val_f1 or (vf1 == best_val_f1 and vloss < best_val_loss):
                best_val_f1, best_val_loss, best_params = vf1, vloss, params.copy()

            print(f"    epoch {epoch:4d}/{epochs} loss={self.loss_hist_[-1]:.4f} "
                  f"tr_acc={self.tr_acc_hist_[-1]:.3f} val_acc={vacc:.3f} "
                  f"val_f1={vf1:.3f} val_loss={vloss:.3f} (best f1 {best_val_f1:.3f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)

            if ckpt_path:
                self._save_ckpt(ckpt_path, {
                    "sig": sig, "epoch": epoch, "params": params, "m": m, "v": v, "t": t,
                    "best_params": best_params, "best_val_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                    "loss_hist": self.loss_hist_, "tr_acc_hist": self.tr_acc_hist_,
                    "val_acc_hist": self.val_acc_hist_, "val_f1_hist": self.val_f1_hist_,
                    "rng_state": self.rng.get_state()})

        self.params_ = best_params
        self.best_val_f1_ = float(best_val_f1)
        return self

    def predict_proba(self, X):
        assert self.params_ is not None, "Model has not been trained yet!"
        return self._run_batch(list(X), [self.params_] * len(X))


def confusion(y_true, y_pred):
    yt, yp = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return np.array([(yt & yp).sum(), (~yt & yp).sum(),
                     (yt & ~yp).sum(), (~yt & ~yp).sum()], dtype=np.int64)


def conf_metrics(c):
    tp, fp, fn, tn = [float(v) for v in c]
    tot = tp + fp + fn + tn
    return {
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "recall"   : tp / (tp + fn) if tp + fn > 0 else 0.0,
        "f1"       : 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0,
        "iou"      : tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0,
        "accuracy" : (tp + tn) / tot if tot > 0 else 0.0,
        "positive_rate": (tp + fn) / tot if tot > 0 else 0.0,
        "n"        : int(tot),
    }


def best_threshold(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.sum() == 0 or (1 - y).sum() == 0 or len(np.unique(p)) < 2:
        return 0.5
    cands = np.unique(np.quantile(p, np.linspace(0, 1, 201)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def mean_abs_diff(X, cfg):
    if cfg["encoding_mode"] == "concat":
        half = X.shape[1] // 2
        return np.abs(X[:, :half] - X[:, half:]).mean(axis=1)
    return np.abs(X - 0.5).mean(axis=1)


def fit_baselines(X_tr, y_tr, seed, cfg):
    lr_all = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
    lr_1d = LogisticRegression().fit(mean_abs_diff(X_tr, cfg)[:, None], y_tr)
    return {
        "LogReg"     : lambda X: lr_all.predict_proba(X)[:, 1],
        "RandForest" : lambda X: rf.predict_proba(X)[:, 1],
        "MeanAbsDiff": lambda X: lr_1d.predict_proba(mean_abs_diff(X, cfg)[:, None])[:, 1],
    }


def accumulate_map(coords, proba, shape, ps):
    H, W = shape
    pmap = np.zeros((H, W), dtype=np.float32)
    cov = np.zeros((H, W), dtype=np.int32)
    for (r, c), pr in zip(coords, proba):
        pmap[r:r + ps, c:c + ps] += pr
        cov[r:r + ps, c:c + ps] += 1
    m = cov > 0
    pmap[m] /= cov[m]
    return pmap, cov


def score_image(rec, predictors, cfg):
    H, W = rec["shape"]
    coords = grid_coords(H, W, cfg["patch_size"], cfg["map_stride"])
    n_chunks = max(1, int(np.ceil(len(coords) / cfg["map_chunk"])))
    ys, probas = [], {n: [] for n in predictors}
    for chunk in np.array_split(coords, n_chunks):
        X, y = featurize(rec["A"], rec["B"], rec["gt"], chunk, cfg)
        ys.append(y)
        for n, fn in predictors.items():
            probas[n].append(np.asarray(fn(X)))
    return coords, np.concatenate(ys), {n: np.concatenate(v) for n, v in probas.items()}


def save_map(rec, pmap, cov, thr, out_dir, label, dpi):
    os.makedirs(out_dir, exist_ok=True)
    bw_cmap = ListedColormap(["blue", "red"])
    name = os.path.splitext(rec["name"])[0]
    gt = rec["gt"]
    pred_bin = ((pmap >= thr) & (cov > 0)).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(f"QCNN Prediction Map  -  {name}  [{label}]",
                 fontsize=16, fontweight="bold", y=1.04)
    if "rgb_B" in rec:
        axes[0].imshow(rec["rgb_B"])
    else:
        axes[0].imshow(rec["raw_B"], cmap="gray")
    axes[0].set_title("Band B (Pre-Fire)", fontsize=14); axes[0].axis("off")
    if "rgb_A" in rec:
        axes[1].imshow(rec["rgb_A"])
    else:
        axes[1].imshow(rec["raw_A"], cmap="gray")
    axes[1].set_title("Band A (Post-Fire)", fontsize=14); axes[1].axis("off")
    axes[2].imshow(gt, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Ground Truth", fontsize=14); axes[2].axis("off")
    axes[2].legend(handles=[mpatches.Patch(color="blue", label="No Change"),
                            mpatches.Patch(color="red", label="Burned")],
                   loc="lower right", fontsize=11, framealpha=0.85)
    axes[3].imshow(pred_bin, cmap=bw_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(f"QCNN Prediction (thr={thr:.3f})", fontsize=14); axes[3].axis("off")
    axes[3].legend(handles=[mpatches.Patch(color="blue", label="No Change"),
                            mpatches.Patch(color="red", label="Predicted Burned")],
                   loc="lower right", fontsize=11, framealpha=0.85)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(out_dir, f"pred_map_{name}.png"), dpi=dpi, bbox_inches="tight")
    plt.close("all")

    H, W = rec["shape"]
    for fname, arr, dt in ((f"pred_binary_{name}.tif", pred_bin, rasterio.uint8),
                           (f"pred_proba_{name}.tif", pmap, rasterio.float32)):
        with rasterio.open(os.path.join(out_dir, fname), "w", driver="GTiff",
                           height=H, width=W, count=1, dtype=dt) as dst:
            dst.write(arr, 1)


def evaluate_split(records, predictors, thresholds, cfg, out_dir, label,
                   max_images=None, seed=0, save_maps=True):
    if max_images is not None and len(records) > max_images:
        sel = sorted(np.random.RandomState(seed).choice(len(records), max_images, replace=False))
        records = [records[i] for i in sel]
    ps = cfg["patch_size"]
    agg = {n: {t: {"patch": np.zeros(4, np.int64), "pixel": np.zeros(4, np.int64)}
               for t in thresholds[n]} for n in predictors}

    for k, rec in enumerate(records, 1):
        print(f"  [{label}] image {k}/{len(records)}: {rec['name']}", flush=True)
        coords, y_patch, probas = score_image(rec, predictors, cfg)
        gt = rec["gt"].astype(bool)
        for n in predictors:
            pmap, cov = accumulate_map(coords, probas[n], rec["shape"], ps)
            covered = cov > 0
            for tname, thr in thresholds[n].items():
                agg[n][tname]["patch"] += confusion(y_patch, probas[n] >= thr)
                agg[n][tname]["pixel"] += confusion(gt[covered], (pmap >= thr)[covered])
            if n == "QCNN" and save_maps:
                save_map(rec, pmap, cov, thresholds[n]["@cal"], out_dir, label, cfg["plot_dpi"])

    return {n: {t: {"patch": conf_metrics(v["patch"]), "pixel": conf_metrics(v["pixel"]),
                    "n_images": len(records)}
                for t, v in agg[n].items()} for n in agg}


def run_seed(seed, train_recs, test_recs, cfg):
    seed_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    tr_idx, va_idx = split_images(train_recs, cfg["val_image_frac"], seed)
    tr_recs = [train_recs[i] for i in tr_idx]
    va_recs = [train_recs[i] for i in va_idx]
    print(f"[seed {seed}] train images={len(tr_recs)}  held-out val images={len(va_recs)}")

    X_tr, y_tr = stratified_balanced_sample(
        [(r["pool_X"], r["pool_y"]) for r in tr_recs], cfg["max_train_patches"], seed)
    X_va, y_va = stratified_balanced_sample(
        [(r["pool_X"], r["pool_y"]) for r in va_recs], cfg["max_val_patches"], seed + 1)
    X_cal, y_cal = natural_sample(va_recs, cfg["calib_patches"], seed + 2)
    print(f"[seed {seed}] X_tr={X_tr.shape} X_val={X_va.shape} "
          f"calibration={X_cal.shape} (natural positive rate={y_cal.mean():.3f})")

    baselines = fit_baselines(X_tr, y_tr, seed, cfg)

    model = QCNNChangeDetector(n_qubits=cfg["n_qubits"], seed=seed)
    print(f"[seed {seed}] QCNN parameters={model.n_params}, "
          f"~{(2 * model.n_params + 1) * cfg['batch_size']} circuits per gradient step")
    ckpt = os.path.join(seed_dir, "checkpoint.pkl") if cfg["resume"] else None
    model.fit(X_tr, y_tr, X_va, y_va, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
              lr=cfg["lr"], beta1=cfg["beta1"], beta2=cfg["beta2"], eps=cfg["eps"],
              ckpt_path=ckpt, sig=config_signature(cfg, CKPT_KEYS))
    np.save(os.path.join(seed_dir, "trained_params.npy"), model.params_)

    predictors = {"QCNN": model.predict_proba, **baselines}

    print(f"[seed {seed}] threshold calibration (held-out val, natural distribution)")
    thresholds = {}
    for n, fn in predictors.items():
        thresholds[n] = {"@0.5": 0.5, "@cal": best_threshold(y_cal, fn(X_cal))}
    print("   ", {n: round(t["@cal"], 3) for n, t in thresholds.items()})

    metrics = {}
    metrics["val_heldout"] = evaluate_split(
        va_recs, predictors, thresholds, cfg,
        os.path.join(seed_dir, "val_heldout"), "Held-out val", seed=seed,
        save_maps=cfg["map_val_images"])
    metrics["test"] = evaluate_split(
        test_recs, predictors, thresholds, cfg,
        os.path.join(seed_dir, "india_test"), TEST_DATASET["label"],
        max_images=cfg["max_images"], seed=seed, save_maps=True)
    if cfg["map_train_images"]:
        metrics["train_seen"] = evaluate_split(
            tr_recs, predictors, thresholds, cfg,
            os.path.join(seed_dir, "loc_train"), "Train (seen, OPTIMISTIC)",
            max_images=cfg["max_images_train"], seed=seed, save_maps=True)

    res = {
        "seed": seed,
        "sig": config_signature(cfg, RESULT_KEYS),
        "train_images": [r["name"] for r in tr_recs],
        "val_images": [r["name"] for r in va_recs],
        "n_params": model.n_params,
        "thresholds": thresholds,
        "best_val_f1_balanced": model.best_val_f1_,
        "history": {"loss": model.loss_hist_, "train_acc": model.tr_acc_hist_,
                    "val_acc": model.val_acc_hist_, "val_f1": model.val_f1_hist_},
        "metrics": metrics,
    }
    with open(os.path.join(seed_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def print_summary(all_res, split, level="pixel"):
    print(f"\n=== {split} | {level} level | seed mean +- std (n_seed={len(all_res)}) ===")
    print(f"{'model':12s} {'thr':6s} {'precision':>15s} {'recall':>15s} {'F1':>15s} {'IoU':>15s}")
    names = list(all_res[0]["metrics"][split].keys())
    for n in names:
        for t in all_res[0]["metrics"][split][n].keys():
            cells = []
            for key in ("precision", "recall", "f1", "iou"):
                vals = np.array([r["metrics"][split][n][t][level][key] for r in all_res])
                cells.append(f"{vals.mean():.3f}+-{vals.std():.3f}")
            print(f"{n:12s} {t:6s} " + " ".join(f"{c:>15s}" for c in cells))


def main():
    cfg = CONFIG
    validate_config(cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"[config] mode={cfg['encoding_mode']} total qubits={cfg['n_qubits'] + 1} "
          f"patch={cfg['patch_size']} seeds={cfg['seeds']}")
    inspect_bands(cfg)

    train_recs = load_images(cfg, TRAIN_DATASET, with_pool=True)
    test_recs = load_images(cfg, TEST_DATASET, with_pool=False)

    res_sig = config_signature(cfg, RESULT_KEYS)
    all_res = []
    for seed in cfg["seeds"]:
        res_path = os.path.join(cfg["output_dir"], f"seed_{seed}", "results.json")
        if cfg["resume"] and os.path.exists(res_path):
            with open(res_path) as f:
                prev = json.load(f)
            if prev.get("sig") == res_sig:
                print(f"[seed {seed}] results.json exists with identical settings, skipping")
                all_res.append(prev)
                continue
            print(f"[seed {seed}] results.json was produced with old settings, rerunning")
        all_res.append(run_seed(seed, train_recs, test_recs, cfg))

    for split in all_res[0]["metrics"].keys():
        for level in ("pixel", "patch"):
            print_summary(all_res, split, level)
    with open(os.path.join(cfg["output_dir"], "all_seeds_results.json"), "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()