"""
ISPRS Item 1 & Item 2 Verification Script (Qiskit 1.x):
  - Item 2 (R6): Transpiled circuit depth, 2-qubit (CZ/ECR), and 1-qubit gate counts
                 on IBM FakeTorino (Heron) and FakeBrisbane (Eagle) backends at
                 optimization_level=3, plus exact logical 2Q gate count = 26
                 (12 CX + 7 CRZ + 7 CRY) and decomposed elementary CNOT count = 40.
  - Item 1 (R4): Frozen-parameter (no retraining) noise & finite-shot evaluation
                 on Test 1 (Global) and Test 2 (India), fixing the SpatialRefinementHead
                 invocation (model.refiner(pq_t, pre_t, post_t)).
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import RYGate
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeBrisbane
except ImportError as exc:
    raise SystemExit(
        "This script requires Qiskit 1.x, qiskit-aer, and qiskit-ibm-runtime.\n"
        "Install with: pip install -r requirements.txt\n"
        f"Missing dependency: {exc}"
    )

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.insert(0, str(SCRIPT_DIR))
from model_qacd import QACDGenuineHybrid, audit_qacd_complexity  # noqa: E402


def gray_code(x: int) -> int:
    return x ^ (x >> 1)


def walsh_hadamard_gray_angles(angles: np.ndarray) -> np.ndarray:
    n_states = len(angles)
    n_bits = int(round(math.log2(n_states)))
    assert (1 << n_bits) == n_states
    w = angles.astype(np.float64).copy()
    h = 1
    while h < n_states:
        for i in range(0, n_states, h * 2):
            for j in range(i, i + h):
                u = w[j]
                v = w[j + h]
                w[j] = u + v
                w[j + h] = u - v
        h *= 2
    w /= float(n_states)
    out = np.empty_like(w)
    for j in range(n_states):
        out[j] = w[gray_code(j)]
    return out


def append_multiplexed_ry_concrete(qc: QuantumCircuit, raw_angles_2theta: np.ndarray, ctrl_qubits_lsb_to_msb, target_qubit: int) -> None:
    n = len(ctrl_qubits_lsb_to_msb)
    n_states = 1 << n
    decomp_angles = walsh_hadamard_gray_angles(raw_angles_2theta)
    for j in range(n_states):
        ang = float(decomp_angles[j])
        if abs(ang) > 1e-14:
            qc.ry(ang, target_qubit)
        diff = gray_code(j) ^ gray_code((j + 1) % n_states)
        ctrl_bit = (diff & -diff).bit_length() - 1
        qc.cx(ctrl_qubits_lsb_to_msb[ctrl_bit], target_qubit)


def append_multiplexed_ry_parameterized(qc: QuantumCircuit, angle_params, ctrl_qubits_lsb_to_msb, target_qubit: int) -> None:
    n = len(ctrl_qubits_lsb_to_msb)
    n_states = 1 << n
    assert len(angle_params) == n_states
    for j in range(n_states):
        qc.ry(angle_params[j], target_qubit)
        diff = gray_code(j) ^ gray_code((j + 1) % n_states)
        ctrl_bit = (diff & -diff).bit_length() - 1
        qc.cx(ctrl_qubits_lsb_to_msb[ctrl_bit], target_qubit)


def append_naive_frqi(qc: QuantumCircuit, raw_angles_2theta: np.ndarray, ctrl_qubits_msb_to_lsb, target_qubit: int) -> None:
    for q in ctrl_qubits_msb_to_lsb:
        qc.h(q)
    n = len(ctrl_qubits_msb_to_lsb)
    for k in range(1 << n):
        zero_qubits = []
        for idx, q in enumerate(ctrl_qubits_msb_to_lsb):
            bit = (k >> (n - 1 - idx)) & 1
            if bit == 0:
                qc.x(q)
                zero_qubits.append(q)
        mc_ry = RYGate(float(raw_angles_2theta[k])).control(n)
        qc.append(mc_ry, list(ctrl_qubits_msb_to_lsb) + [target_qubit])
        for q in zero_qubits:
            qc.x(q)


def append_qcnn_62param(qc: QuantumCircuit, theta: np.ndarray) -> None:
    """
    Exact 62-parameter 3-stage QCNN ansatz:
      - 12 Conv pairs x 1 CX = 12 CX gates
      - 7 Pool pairs x (1 CRZ + 1 CRY) = 14 controlled rotations
      Total logical two-qubit gates = 26 (40 elementary CX gates after CRZ/CRY decomposition).
    """
    assert len(theta) == 62
    p = 0

    def add_conv(qa: int, qb: int, t1: float, t2: float, t3: float, t4: float) -> None:
        qc.ry(float(t1), qa)
        qc.rz(float(t2), qa)
        qc.ry(float(t3), qb)
        qc.cx(qa, qb)
        qc.ry(float(t4), qb)

    def add_pool(qa: int, qb: int, phi1: float, phi2: float) -> None:
        qc.crz(float(phi1), qa, qb)
        qc.cry(float(phi2), qa, qb)

    for qa, qb in [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6), (7, 8)]:
        add_conv(qa, qb, theta[p], theta[p + 1], theta[p + 2], theta[p + 3])
        p += 4
    for qa, qb in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        add_pool(qa, qb, theta[p], theta[p + 1])
        p += 2

    for qa, qb in [(4, 5), (6, 7), (5, 6)]:
        add_conv(qa, qb, theta[p], theta[p + 1], theta[p + 2], theta[p + 3])
        p += 4
    for qa, qb in [(6, 4), (7, 5)]:
        add_pool(qa, qb, theta[p], theta[p + 1])
        p += 2

    add_conv(4, 5, theta[p], theta[p + 1], theta[p + 2], theta[p + 3])
    p += 4
    add_pool(5, 4, theta[p], theta[p + 1])
    p += 2
    assert p == 62


def self_test_statevector_equivalence() -> None:
    rng = np.random.default_rng(7)
    patch_242 = rng.uniform(0.0, 1.0, size=(242,)).astype(np.float32)
    padded_256 = np.pad(patch_242, (0, 14), constant_values=0.0)
    angles_2theta = math.pi * padded_256

    model = QACDGenuineHybrid(init_seed=123, patch_minmax=False)
    with torch.no_grad():
        torch_pq = float(model.qcnn.forward_patches_fast(torch.from_numpy(patch_242).unsqueeze(0)).item())
        theta_np = model.qcnn.theta.detach().cpu().numpy().astype(np.float64)

    ctrl_lsb_to_msb = [7, 6, 5, 4, 3, 2, 1, 0]
    qc = QuantumCircuit(9)
    for q in range(8):
        qc.h(q)
    append_multiplexed_ry_concrete(qc, angles_2theta, ctrl_lsb_to_msb, target_qubit=8)
    append_qcnn_62param(qc, theta_np)

    sv = Statevector.from_instruction(qc)
    qiskit_pq = float(sv.probabilities([4])[1])
    diff = abs(torch_pq - qiskit_pq)
    assert diff < 1e-5, f"Mismatch between PyTorch and Qiskit Statevector: {diff}"
    print(f"[self-test] PASS: |p_Q(PyTorch) - p_Q(Qiskit)| = {diff:.2e}")


def summarize_transpiled_circuit(name: str, qc: QuantumCircuit, backend, seeds=(0, 1, 2, 3, 4), optimization_level=3):
    depths, two_q_counts, one_q_counts, total_counts = [], [], [], []
    backend_name = getattr(backend, "name", backend.__class__.__name__)
    for s in seeds:
        tc = transpile(qc, backend=backend, optimization_level=optimization_level, seed_transpiler=s)
        ops = tc.count_ops()
        two_q = sum(ops.get(g, 0) for g in ("cz", "ecr", "cx", "rzz", "iswap"))
        non_gate = ops.get("measure", 0) + ops.get("barrier", 0) + ops.get("delay", 0)
        tot = sum(ops.values()) - non_gate
        one_q = tot - two_q
        depths.append(tc.depth())
        two_q_counts.append(two_q)
        one_q_counts.append(one_q)
        total_counts.append(tot)

    def stats(arr):
        return {
            "mean": round(statistics.mean(arr), 2),
            "sd": round(statistics.stdev(arr) if len(arr) > 1 else 0.0, 2),
            "min": int(min(arr)),
            "max": int(max(arr)),
        }

    return {
        "circuit": name,
        "backend": str(backend_name),
        "optimization_level": optimization_level,
        "seeds": list(seeds),
        "depth": stats(depths),
        "two_qubit_gates": stats(two_q_counts),
        "one_qubit_gates": stats(one_q_counts),
        "total_gates": stats(total_counts),
    }


def run_item2_transpiled_cost(theta: np.ndarray, include_naive_frqi: bool = True):
    rng = np.random.default_rng(42)
    sample_angles = rng.uniform(0.05, math.pi - 0.05, size=(256,))
    sample_angles[242:] = 0.0

    circuits = {}
    if include_naive_frqi:
        qc_naive = QuantumCircuit(9)
        append_naive_frqi(qc_naive, sample_angles, list(range(8)), 8)
        circuits["Naive FRQI (256 x C^8-RY)"] = qc_naive

    qc_mux = QuantumCircuit(9)
    for q in range(8):
        qc_mux.h(q)
    append_multiplexed_ry_concrete(qc_mux, sample_angles, [7, 6, 5, 4, 3, 2, 1, 0], 8)
    circuits["Multiplexed-RY FRQI (prep only)"] = qc_mux

    qc_qcnn = QuantumCircuit(9, 1)
    append_qcnn_62param(qc_qcnn, theta)
    qc_qcnn.measure(4, 0)
    circuits["QCNN ansatz only (62 params, 26 logical 2Q / 40 CX)"] = qc_qcnn

    qc_full = QuantumCircuit(9, 1)
    for q in range(8):
        qc_full.h(q)
    append_multiplexed_ry_concrete(qc_full, sample_angles, [7, 6, 5, 4, 3, 2, 1, 0], 8)
    append_qcnn_62param(qc_full, theta)
    qc_full.measure(4, 0)
    circuits["Full QACD (Multiplexed FRQI + QCNN)"] = qc_full

    results = []
    for backend in (FakeTorino(), FakeBrisbane()):
        for label, qc in circuits.items():
            t0 = time.perf_counter()
            res = summarize_transpiled_circuit(label, qc, backend)
            res["transpile_wall_s"] = round(time.perf_counter() - t0, 2)
            results.append(res)
    return results


def extract_center_patches_from_tile(pre_rgb: np.ndarray, post_rgb: np.ndarray, gt_mask: np.ndarray, patch_grid_stride: int = 1):
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum_pre = np.tensordot(pre_rgb, w, axes=([-1], [0]))
    lum_post = np.tensordot(post_rgb, w, axes=([-1], [0]))
    h, w_img = lum_pre.shape
    r0, r1 = (h - 64) // 2, (h + 64) // 2
    c0, c1 = (w_img - 64) // 2, (w_img + 64) // 2
    radius = 5

    patches, labels = [], []
    for r in range(r0, r1, patch_grid_stride):
        for c in range(c0, c1, patch_grid_stride):
            p_pre = lum_pre[r - radius : r + radius + 1, c - radius : c + radius + 1].reshape(-1)
            p_post = lum_post[r - radius : r + radius + 1, c - radius : c + radius + 1].reshape(-1)
            v = np.concatenate([p_pre, p_post], axis=0)
            span = float(v.max() - v.min())
            if span > 0.02:
                v = (v - float(v.min())) / (span + 1e-6)
            patches.append(np.clip(v, 0.0, 1.0))
            labels.append(int(gt_mask[r, c] > 0))
    return np.asarray(patches, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def build_transpiled_parameterized_qacd(theta: np.ndarray, backend=None):
    ang_params = ParameterVector("a", 256)
    qc = QuantumCircuit(9, 1)
    for q in range(8):
        qc.h(q)
    append_multiplexed_ry_parameterized(qc, ang_params, [7, 6, 5, 4, 3, 2, 1, 0], 8)
    append_qcnn_62param(qc, theta)
    qc.measure(4, 0)
    if backend is not None:
        qc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=0)
    return qc, ang_params


def run_aer_patches(transpiled_param_qc, ang_params, patches_242, simulator, shots: int, batch_size: int = 128) -> np.ndarray:
    n_patches = len(patches_242)
    padded = np.pad(np.clip(patches_242, 0.0, 1.0), ((0, 0), (0, 14)), constant_values=0.0)
    wh_angles = np.stack([walsh_hadamard_gray_angles(math.pi * row) for row in padded], axis=0)
    probs = np.empty(n_patches, dtype=np.float32)
    param_list = list(ang_params)
    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        bound = [
            transpiled_param_qc.assign_parameters(
                {p: float(wh_angles[i, idx]) for idx, p in enumerate(param_list)},
                inplace=False,
            )
            for i in range(start, end)
        ]
        result = simulator.run(bound, shots=shots).result()
        for k in range(end - start):
            counts = result.get_counts(k)
            probs[start + k] = float(counts.get("1", 0)) / float(shots)
    return probs


def compute_binary_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    ciou = tp / max(tp + fp + fn, 1)
    iou_nc = tn / max(tn + fp + fn, 1)
    return {
        "OA": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "cIoU": round(ciou, 4),
        "mIoU": round(0.5 * (ciou + iou_nc), 4),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def load_split_tiles(dataset_root: Path, fire_prefixes: tuple, max_tiles: int = 0):
    import tifffile

    b_dir = dataset_root / "B"
    a_dir = dataset_root / "A"
    gt_dir = dataset_root / "ground_truth"
    files = sorted(p.name for p in gt_dir.glob("*.tif") if any(p.name.startswith(pref) for pref in fire_prefixes))
    if max_tiles > 0:
        idx = np.linspace(0, len(files) - 1, min(max_tiles, len(files)), dtype=int)
        files = [files[i] for i in idx]

    tiles = []
    for fname in files:
        arr_b = tifffile.imread(b_dir / fname).astype(np.float32)
        arr_a = tifffile.imread(a_dir / fname).astype(np.float32)
        gt = np.nan_to_num(tifffile.imread(gt_dir / fname).astype(np.float32), nan=0.0)
        rgb_idx = [2, 1, 0] if (arr_b.ndim == 3 and arr_b.shape[-1] == 15) else [3, 2, 1]
        pre_rgb = np.clip(np.nan_to_num(arr_b[..., rgb_idx], nan=0.0, posinf=0.5, neginf=0.0), 0.0, 0.5) / 0.5
        post_rgb = np.clip(np.nan_to_num(arr_a[..., rgb_idx], nan=0.0, posinf=0.5, neginf=0.0), 0.0, 0.5) / 0.5
        tiles.append((fname, pre_rgb, post_rgb, (gt > 0.5).astype(np.uint8)))
    return tiles


def evaluate_item1_noise_and_shots(model: QACDGenuineHybrid, dataset_root: Path, max_tiles_per_split: int, stride: int, threshold: float = 0.5):
    theta_np = model.qcnn.theta.detach().cpu().numpy().astype(np.float64)
    fake_torino = FakeTorino()
    sim_ideal = AerSimulator(method="statevector")
    sim_noisy = AerSimulator.from_backend(fake_torino)

    param_qc_ideal, ang_params_i = build_transpiled_parameterized_qacd(theta_np, backend=None)
    param_qc_noisy, ang_params_n = build_transpiled_parameterized_qacd(theta_np, backend=fake_torino)

    splits = {
        "Test 1 (Cross-Event, Same-Biome)": ("mckinney_fire", "gospers_fire"),
        "Test 2 (India Cross-Biome Transfer)": ("Uttarakhand_Fire_2021", "Similipal_Fire_2021", "Bandhavgarh_Fire"),
    }
    configs = [
        ("Exact Statevector (PyTorch)", None, None),
        ("Noiseless Aer (1024 shots)", "ideal", 1024),
        ("Noiseless Aer (8192 shots)", "ideal", 8192),
        ("FakeTorino Noise (1024 shots)", "noisy", 1024),
        ("FakeTorino Noise (8192 shots)", "noisy", 8192),
    ]

    summary = {}
    for split_name, prefixes in splits.items():
        tiles = load_split_tiles(dataset_root, prefixes, max_tiles=max_tiles_per_split)
        all_pq_by_cfg = {c[0]: [] for c in configs}
        all_gt = []
        all_head_preds = {c[0]: [] for c in configs}

        for _, pre_rgb, post_rgb, gt in tiles:
            patches, labels = extract_center_patches_from_tile(pre_rgb, post_rgb, gt, patch_grid_stride=stride)
            all_gt.append(labels)
            with torch.no_grad():
                pq_exact = model.qcnn.forward_patches_fast(torch.from_numpy(patches)).cpu().numpy()
            all_pq_by_cfg["Exact Statevector (PyTorch)"].append(pq_exact)

            for cfg_name, mode, shots in configs[1:]:
                sim = sim_ideal if mode == "ideal" else sim_noisy
                pqc = param_qc_ideal if mode == "ideal" else param_qc_noisy
                ap = ang_params_i if mode == "ideal" else ang_params_n
                all_pq_by_cfg[cfg_name].append(run_aer_patches(pqc, ap, patches, sim, shots=shots))

            if stride == 1:
                h, w_img = gt.shape
                r0, r1 = (h - 64) // 2, (h + 64) // 2
                c0, c1 = (w_img - 64) // 2, (w_img + 64) // 2
                pre_crop_t = torch.from_numpy(pre_rgb[r0:r1, c0:c1].transpose(2, 0, 1).copy()).unsqueeze(0)
                post_crop_t = torch.from_numpy(post_rgb[r0:r1, c0:c1].transpose(2, 0, 1).copy()).unsqueeze(0)
                with torch.no_grad():
                    for cfg_name in all_pq_by_cfg:
                        pq_t = torch.from_numpy(all_pq_by_cfg[cfg_name][-1].reshape(1, 1, 64, 64))
                        # Patched invocation matching SpatialRefinementHead.forward(p_q_map, pre_rgb, post_rgb)
                        logits = model.refiner(pq_t, pre_crop_t, post_crop_t)
                        prob = torch.sigmoid(logits).squeeze().numpy().reshape(-1)
                        all_head_preds[cfg_name].append((prob >= threshold).astype(np.uint8))

        y_true = np.concatenate(all_gt, axis=0)
        split_res = {}
        for cfg_name, _, _ in configs:
            pq_cat = np.concatenate(all_pq_by_cfg[cfg_name], axis=0)
            q_metrics = compute_binary_metrics((pq_cat >= 0.5).astype(np.uint8), y_true)
            mae_vs_exact = float(np.mean(np.abs(pq_cat - np.concatenate(all_pq_by_cfg["Exact Statevector (PyTorch)"], axis=0))))
            entry = {"quantum_only_metrics": q_metrics, "mae_pQ_vs_exact": round(mae_vs_exact, 5)}
            if stride == 1:
                entry["full_qacd_head_metrics"] = compute_binary_metrics(np.concatenate(all_head_preds[cfg_name], axis=0), y_true)
            split_res[cfg_name] = entry
        summary[split_name] = split_res
    return summary


def main():
    parser = argparse.ArgumentParser(description="ISPRS Item 1 & Item 2 Qiskit Verification")
    parser.add_argument("--mode", choices=("all", "self-test", "cost", "noise"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/QACD_Proposed_seed42.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Final_with_India"))
    parser.add_argument("--max-tiles-per-split", type=int, default=20)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--skip-naive-frqi", action="store_true")
    parser.add_argument("--out-json", type=Path, default=SCRIPT_DIR / "outputs" / "item1_item2_qiskit_results.json")
    args = parser.parse_args()

    self_test_statevector_equivalence()
    if args.mode == "self-test":
        return

    model = QACDGenuineHybrid(init_seed=42)
    if args.checkpoint is not None and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    theta_np = model.qcnn.theta.detach().cpu().numpy().astype(np.float64)

    payload = {
        "complexity_and_gate_audit": audit_qacd_complexity(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "dataset_root": str(args.dataset_root),
    }
    if args.mode in ("all", "cost"):
        payload["item2_transpiled_cost"] = run_item2_transpiled_cost(theta_np, include_naive_frqi=not args.skip_naive_frqi)
    if args.mode in ("all", "noise"):
        payload["item1_noise_and_shots"] = evaluate_item1_noise_and_shots(
            model, args.dataset_root, max_tiles_per_split=args.max_tiles_per_split, stride=args.stride
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()
