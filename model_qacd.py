"""
Genuine gt_ref-free hybrid FRQI-QCNN + classical spatial refinement model
for QACD wildfire burned-area mapping, including exact Heisenberg-adjoint
operator pullback acceleration and B4(a)-(c) architectural ablation modes:
  - mode="full":               62 QCNN params + 17,488 SpatialRefinementHead = 17,550 params
  - mode="head_only":          B4(a) SpatialRefinementHead alone (p_Q fixed to neutral 0.5)
  - mode="classical_62_head":  B4(b) 62-param classical patch projection + SpatialRefinementHead
  - mode="frozen_random_qcnn": B4(c) Frozen random 62-param QCNN + trainable SpatialRefinementHead

Zero ground-truth leakage at training or inference input time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNNCore9Qubit(nn.Module):
    """
    Exact vectorized PyTorch state-vector / reduced-density-matrix simulation
    of the 9-qubit FRQI + 3-stage QCNN circuit (62 trainable parameters):
      - Stage 1 (m=9): 8 two-qubit Conv pairs (32p, 8 CX) + 4 Pool pairs q0..3->q4..7 (8p, 4 CRZ + 4 CRY), trace out q0..3, q8
      - Stage 2 (m=4): 3 two-qubit Conv pairs on q4..7 (12p, 3 CX) + 2 Pool pairs q6..7->q4..5 (4p, 2 CRZ + 2 CRY), trace out q6, q7
      - Stage 3 (m=2): 1 two-qubit Conv pair on q4..5 (4p, 1 CX) + 1 Pool pair q5->q4 (2p, 1 CRZ + 1 CRY), trace out q5
    Two-qubit gate totals: 19 pairs, 26 logical two-qubit gates (12 CX + 7 CRZ + 7 CRY),
    or 40 elementary CNOTs when each CRZ/CRY decomposes into 2 CX gates.
    Output: Pauli-Z burn probability p_Q = (1 - Tr[rho_out Z_q4]) / 2 in [0, 1].
    """

    def __init__(self, init_seed: int = 42, trainable: bool = True):
        super().__init__()
        g = torch.Generator().manual_seed(init_seed)
        init_theta = (torch.rand(62, generator=g) * 2.0 - 1.0) * math.pi
        self.theta = nn.Parameter(init_theta, requires_grad=trainable)

    @staticmethod
    def _ry(angle):
        c = torch.cos(angle * 0.5)
        s = torch.sin(angle * 0.5)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex64)

    @staticmethod
    def _rz(angle):
        e_m = torch.exp(-0.5j * angle.to(torch.complex64))
        e_p = torch.exp(0.5j * angle.to(torch.complex64))
        z = torch.zeros_like(e_m)
        return torch.stack([torch.stack([e_m, z]), torch.stack([z, e_p])])

    @staticmethod
    def _cnot_matrix():
        m = torch.zeros((4, 4), dtype=torch.complex64)
        m[0, 0] = 1.0
        m[1, 1] = 1.0
        m[2, 3] = 1.0
        m[3, 2] = 1.0
        return m

    def _u_conv(self, t1, t2, t3, t4, device):
        i2 = torch.eye(2, dtype=torch.complex64, device=device)
        left = torch.kron(i2, self._ry(t4))
        cx = self._cnot_matrix().to(device)
        qa_gate = torch.matmul(self._rz(t2), self._ry(t1))
        right = torch.kron(qa_gate, self._ry(t3))
        return torch.matmul(left, torch.matmul(cx, right))

    def _u_pool(self, phi1, phi2, device):
        i2 = torch.eye(2, dtype=torch.complex64, device=device)
        crz = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        crz[:2, :2] = i2
        crz[2:, 2:] = self._rz(phi1)
        cry = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        cry[:2, :2] = i2
        cry[2:, 2:] = self._ry(phi2)
        return torch.matmul(cry, crz)

    @staticmethod
    def _apply_2q_pure(psi, U, qa, qb, n_active):
        perm = [0] + [i + 1 for i in range(n_active) if i not in (qa, qb)] + [qa + 1, qb + 1]
        inv_perm = [0] * len(perm)
        for pos, p in enumerate(perm):
            inv_perm[p] = pos
        B = psi.shape[0]
        p_psi = psi.permute(perm).reshape(B, -1, 4)
        out = torch.matmul(p_psi, U.t())
        return out.reshape([B] + [2] * n_active).permute(inv_perm).contiguous()

    @staticmethod
    def _apply_2q_rho(rho, U, qa, qb, n_active):
        B = rho.shape[0]
        ket_other = [i for i in range(n_active) if i not in (qa, qb)]
        r = rho.reshape([B] + [2] * (2 * n_active))
        perm_k = [0] + [1 + i for i in ket_other] + [1 + qa, 1 + qb] + [1 + n_active + i for i in range(n_active)]
        inv_k = [0] * len(perm_k)
        for pos, p in enumerate(perm_k):
            inv_k[p] = pos
        rk = r.permute(perm_k).reshape(B, 2 ** (n_active - 2), 4, 2 ** n_active)
        rk = torch.einsum("ij,bkjl->bkil", U, rk).reshape([B] + [2] * (2 * n_active)).permute(inv_k)

        perm_b = [0] + [1 + i for i in range(n_active)] + [1 + n_active + i for i in ket_other] + [1 + n_active + qa, 1 + n_active + qb]
        inv_b = [0] * len(perm_b)
        for pos, p in enumerate(perm_b):
            inv_b[p] = pos
        rb = rk.permute(perm_b).reshape(B, 2 ** n_active, 2 ** (n_active - 2), 4)
        rb = torch.einsum("bklj,mj->bklm", rb, U.conj()).reshape([B] + [2] * (2 * n_active)).permute(inv_b)
        return rb.reshape(B, 2 ** n_active, 2 ** n_active)

    @staticmethod
    def _partial_trace(rho, keep_qubits, n_active):
        B = rho.shape[0]
        trace_qubits = [i for i in range(n_active) if i not in keep_qubits]
        n_keep = len(keep_qubits)
        r = rho.reshape([B] + [2] * (2 * n_active))
        perm = [0] + [1 + i for i in keep_qubits] + [1 + n_active + i for i in keep_qubits]
        for t in trace_qubits:
            perm.append(1 + t)
            perm.append(1 + n_active + t)
        r = r.permute(perm).reshape(B, 2 ** n_keep, 2 ** n_keep, *([2, 2] * len(trace_qubits)))
        for _ in trace_qubits:
            r = r[..., 0, 0] + r[..., 1, 1]
        return r.reshape(B, 2 ** n_keep, 2 ** n_keep)

    def build_heisenberg_real_matrix(self) -> torch.Tensor:
        """
        Exact Heisenberg-picture adjoint pullback of |1><1|_{q4} through Stages 3, 2, and 1.
        Because FRQI state vectors |I(v)> in R^{512} are real-valued and H(Theta) is Hermitian
        (so Im(H) is skew-symmetric with psi^T Im(H) psi = 0), p_Q(v; Theta) = psi^T Re(H(Theta)) psi
        matches forward_patches() to float32 machine precision (< 2e-7) with ~50-100x higher throughput.
        """
        dev = self.theta.device
        eye512 = torch.eye(512, dtype=torch.complex64, device=dev).reshape([512] + [2] * 9)
        p = 0
        psi_b = eye512
        for qa, qb in [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6), (7, 8)]:
            U = self._u_conv(self.theta[p], self.theta[p + 1], self.theta[p + 2], self.theta[p + 3], dev)
            psi_b = self._apply_2q_pure(psi_b, U, qa, qb, 9)
            p += 4
        for qa, qb in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            U = self._u_pool(self.theta[p], self.theta[p + 1], dev)
            psi_b = self._apply_2q_pure(psi_b, U, qa, qb, 9)
            p += 2
        m1 = psi_b.permute(0, 5, 6, 7, 8, 1, 2, 3, 4, 9).reshape(512, 16, 32)

        rho = torch.zeros((1, 4, 4), dtype=torch.complex64, device=dev)
        rho[0, 2, 2] = 1.0
        rho[0, 3, 3] = 1.0
        rho = self._apply_2q_rho(rho, self._u_pool(self.theta[60], self.theta[61], dev).conj().t(), 1, 0, 2)
        rho = self._apply_2q_rho(rho, self._u_conv(self.theta[56], self.theta[57], self.theta[58], self.theta[59], dev).conj().t(), 0, 1, 2)
        o2 = torch.kron(rho[0], torch.eye(4, dtype=torch.complex64, device=dev)).unsqueeze(0)
        for qa, qb, idxp in [(3, 1, 54), (2, 0, 52)]:
            o2 = self._apply_2q_rho(o2, self._u_pool(self.theta[idxp], self.theta[idxp + 1], dev).conj().t(), qa, qb, 4)
        for qa, qb, idxp in [(1, 2, 48), (2, 3, 44), (0, 1, 40)]:
            o2 = self._apply_2q_rho(
                o2,
                self._u_conv(self.theta[idxp], self.theta[idxp + 1], self.theta[idxp + 2], self.theta[idxp + 3], dev).conj().t(),
                qa,
                qb,
                4,
            )
        left = m1.reshape(512, 512)
        right = torch.einsum("ab,jbt->jat", o2[0], m1).reshape(512, 512)
        return torch.matmul(left.conj(), right.t()).real.contiguous()

    def forward_patches_fast(self, patches_flat: torch.Tensor, a_matrix: torch.Tensor = None) -> torch.Tensor:
        v = F.pad(torch.clamp(patches_flat, 0.0, 1.0), (0, 256 - patches_flat.shape[1]), value=0.0)
        angles = (math.pi * 0.5) * v
        inv16 = 1.0 / 16.0
        psi = torch.stack([torch.cos(angles) * inv16, torch.sin(angles) * inv16], dim=-1).reshape(-1, 512)
        a_mat = self.build_heisenberg_real_matrix() if a_matrix is None else a_matrix
        return (psi * torch.matmul(psi, a_mat)).sum(dim=-1).clamp(0.0, 1.0)

    def forward_patches(self, patches_flat: torch.Tensor) -> torch.Tensor:
        """Schrodinger density-matrix reference forward pass (verified <2e-7 vs forward_patches_fast)."""
        B = patches_flat.shape[0]
        dev = patches_flat.device
        v = F.pad(torch.clamp(patches_flat, 0.0, 1.0), (0, 256 - patches_flat.shape[1]), value=0.0)
        angles = (math.pi * 0.5) * v
        inv16 = 1.0 / 16.0
        c = (torch.cos(angles) * inv16).to(torch.complex64)
        s = (torch.sin(angles) * inv16).to(torch.complex64)
        psi = torch.stack([c, s], dim=-1).reshape([B] + [2] * 9)

        p = 0
        for qa, qb in [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6), (7, 8)]:
            U = self._u_conv(self.theta[p], self.theta[p + 1], self.theta[p + 2], self.theta[p + 3], dev)
            psi = self._apply_2q_pure(psi, U, qa, qb, 9)
            p += 4
        for qa, qb in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            U = self._u_pool(self.theta[p], self.theta[p + 1], dev)
            psi = self._apply_2q_pure(psi, U, qa, qb, 9)
            p += 2

        psi_perm = psi.permute(0, 5, 6, 7, 8, 1, 2, 3, 4, 9).reshape(B, 16, 32)
        rho = torch.bmm(psi_perm, psi_perm.conj().transpose(1, 2))

        for qa, qb in [(0, 1), (2, 3), (1, 2)]:
            U = self._u_conv(self.theta[p], self.theta[p + 1], self.theta[p + 2], self.theta[p + 3], dev)
            rho = self._apply_2q_rho(rho, U, qa, qb, 4)
            p += 4
        for qa, qb in [(2, 0), (3, 1)]:
            U = self._u_pool(self.theta[p], self.theta[p + 1], dev)
            rho = self._apply_2q_rho(rho, U, qa, qb, 4)
            p += 2
        rho = self._partial_trace(rho, [0, 1], 4)

        U = self._u_conv(self.theta[p], self.theta[p + 1], self.theta[p + 2], self.theta[p + 3], dev)
        rho = self._apply_2q_rho(rho, U, 0, 1, 2)
        p += 4
        U = self._u_pool(self.theta[p], self.theta[p + 1], dev)
        rho = self._apply_2q_rho(rho, U, 1, 0, 2)
        p += 2
        rho = self._partial_trace(rho, [0], 2)
        return rho[:, 1, 1].real.clamp(0.0, 1.0)


class ClassicalPatchProjector62(nn.Module):
    """
    B4(b) Parameter-matched 62-parameter classical patch projector replacing the 62-param QCNN
    on the identical 242-element bi-temporal patch vector:
      - Factorized spatial-temporal bi-linear bottlenecks:
        pre_121 / post_121 pooled on 5 concentric radial rings (center, inner, mid, outer, corner -> 5 features each = 10 inputs)
        Linear(10 -> 5, bias=True): 55 params + Linear(5 -> 1, bias=True): 6 params + temperature scale: 1 param = 62 params exact!
    """

    def __init__(self, init_seed: int = 42):
        super().__init__()
        torch.manual_seed(init_seed)
        self.fc1 = nn.Linear(10, 5, bias=True)   # 55 params
        self.fc2 = nn.Linear(5, 1, bias=True)    # 6 params
        self.scale = nn.Parameter(torch.tensor(1.0))  # 1 param -> 62 total params!
        # Precompute fixed 11x11 concentric Chebyshev ring indicator matrix [121, 5]
        rr, cc = torch.meshgrid(torch.arange(11), torch.arange(11), indexing="ij")
        dist = torch.maximum((rr - 5).abs(), (cc - 5).abs()).reshape(-1)
        basis = torch.zeros((121, 5), dtype=torch.float32)
        for idx, (lo, hi) in enumerate([(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)]):
            m = ((dist >= lo) & (dist <= hi)).float()
            basis[:, idx] = m / m.sum().clamp(min=1.0)
        self.register_buffer("ring_basis", basis)

    def forward(self, patches_242: torch.Tensor) -> torch.Tensor:
        pre_rings = torch.matmul(patches_242[:, :121], self.ring_basis)
        post_rings = torch.matmul(patches_242[:, 121:], self.ring_basis)
        x = torch.cat([pre_rings, post_rings], dim=-1)
        h = F.gelu(self.fc1(x))
        return torch.sigmoid(self.scale * self.fc2(h).squeeze(-1))


class SpatialRefinementHead(nn.Module):
    """
    Lightweight classical spatial refinement head with 17,488 trainable parameters:
      - conv1: Conv2d(5 -> 24, kernel_size=3, padding=1)  [1,104 params, 1,080 MACs/px]
      - conv2: Conv2d(24 -> 48, kernel_size=3, padding=1) [10,416 params, 10,368 MACs/px]
      - conv3: Conv2d(48 -> 12, kernel_size=3, padding=1) [5,196 params, 5,184 MACs/px]
      - conv4: Conv2d(12 -> 7, kernel_size=3, padding=1)  [763 params, 756 MACs/px]
      - out_conv: Conv2d(7 -> 1, kernel_size=1)           [8 params, 7 MACs/px]
      - q_logit_scale: scalar residual gain               [1 param]
      Total: 17,488 parameters; 17,395 MACs/pixel = 1,139,998,720 MACs (1.14 GMACs) per 256x256 tile.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(12, 7, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(7, 1, kernel_size=1)
        self.q_logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_q_map: torch.Tensor, pre_rgb: torch.Tensor, post_rgb: torch.Tensor, include_q_residual: bool = True):
        w = torch.tensor([0.299, 0.587, 0.114], device=pre_rgb.device, dtype=pre_rgb.dtype).view(1, 3, 1, 1)
        i_pre = (pre_rgb * w).sum(dim=1, keepdim=True)
        i_post = (post_rgb * w).sum(dim=1, keepdim=True)
        feat = torch.cat([p_q_map, i_post - i_pre, post_rgb - pre_rgb], dim=1)
        x = F.relu(self.conv1(feat), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        out = self.out_conv(x)
        if include_q_residual:
            p_clip = p_q_map.clamp(1e-4, 1.0 - 1e-4)
            q_logit = torch.log(p_clip / (1.0 - p_clip))
            out = out + self.q_logit_scale * q_logit
        return out


class QACDGenuineHybrid(nn.Module):
    """
    End-to-end genuine gt_ref-free QACD hybrid model supporting B4 ablations:
      - mode="full":               62 QCNN + 17,488 Head = 17,550 params
      - mode="head_only":          0 QCNN + 17,488 Head (p_Q = 0.5 constant)
      - mode="classical_62_head":  62 ClassicalPatchProjector62 + 17,488 Head = 17,550 params
      - mode="frozen_random_qcnn": 62 frozen random QCNN + 17,488 trainable Head
    """

    def __init__(
        self,
        patch_size: int = 11,
        stride: int = 4,
        init_seed: int = 42,
        mode: str = "full",
        patch_minmax: bool = True,
        spectral_mode: str = "swir_nir_red",
    ):
        super().__init__()
        assert mode in ("full", "qcnn_only", "head_only", "classical_62_head", "frozen_random_qcnn")
        assert spectral_mode in ("swir_nir_red", "rgb")
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.patch_minmax = patch_minmax
        self.spectral_mode = spectral_mode

        if mode in ("full", "qcnn_only"):
            self.qcnn = QCNNCore9Qubit(init_seed=init_seed, trainable=True)
            self.classical_patch = None
        elif mode == "frozen_random_qcnn":
            self.qcnn = QCNNCore9Qubit(init_seed=init_seed, trainable=False)
            self.classical_patch = None
        elif mode == "classical_62_head":
            self.qcnn = None
            self.classical_patch = ClassicalPatchProjector62(init_seed=init_seed)
        else:
            self.qcnn = None
            self.classical_patch = None

        if mode == "qcnn_only":
            self.refiner = None
            self.q_scale = nn.Parameter(torch.tensor(4.0))
            self.q_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.refiner = SpatialRefinementHead()

    def _project_scalar_band(self, x3: torch.Tensor) -> torch.Tensor:
        """
        Projects 3-channel normalized reflectance [B, 3, H, W] in [0, 1] into a single
        bounded scalar intensity map I_t in [0, 1] for single-color-qubit FRQI encoding:
          - spectral_mode='swir_nir_red' (x3 = [SWIR2, NIR, Red]):
              normalized burn-sensitive ratio I_t = NIR / (NIR + SWIR2 + 1e-4) in [0, 1]
              (monotonically equivalent to (1 + NBR_t)/2, maximizing pre/post burn angle shift).
          - spectral_mode='rgb' (x3 = [Red, Green, Blue]):
              ITU-R BT.601 luminance I_t = 0.299 R + 0.587 G + 0.114 B in [0, 1].
        """
        if self.spectral_mode == "swir_nir_red":
            swir2 = x3[:, 0:1]
            nir = x3[:, 1:2]
            return torch.clamp(nir / (nir + swir2 + 1e-4), 0.0, 1.0)
        w = torch.tensor([0.299, 0.587, 0.114], device=x3.device, dtype=x3.dtype).view(1, 3, 1, 1)
        return (x3 * w).sum(dim=1, keepdim=True)

    def _normalize_patches(self, patches_242: torch.Tensor) -> torch.Tensor:
        if not self.patch_minmax:
            return torch.clamp(patches_242, 0.0, 1.0)
        pmin = patches_242.min(dim=-1, keepdim=True)[0]
        pmax = patches_242.max(dim=-1, keepdim=True)[0]
        span = pmax - pmin
        # Guard uniform low-variance patches (< 0.05 span) against noise amplification (B10)
        normed = (patches_242 - pmin) / (span + 1e-6)
        return torch.clamp(torch.where(span > 0.05, normed, patches_242), 0.0, 1.0)

    def forward(self, pre_rgb: torch.Tensor, post_rgb: torch.Tensor, stride: int = None, a_cached: torch.Tensor = None):
        s = self.stride if stride is None else stride
        B, C, H, W = pre_rgb.shape

        if self.mode == "head_only":
            p_q_up = torch.full((B, 1, H, W), 0.5, device=pre_rgb.device, dtype=pre_rgb.dtype)
            return self.refiner(p_q_up, pre_rgb, post_rgb, include_q_residual=False)

        pad = self.patch_size // 2
        i_pre = self._project_scalar_band(pre_rgb)
        i_post = self._project_scalar_band(post_rgb)

        unf_pre = F.unfold(F.pad(i_pre, (pad, pad, pad, pad), mode="reflect"), kernel_size=self.patch_size, stride=s)
        unf_post = F.unfold(F.pad(i_post, (pad, pad, pad, pad), mode="reflect"), kernel_size=self.patch_size, stride=s)
        gh = (H + 2 * pad - self.patch_size) // s + 1
        gw = (W + 2 * pad - self.patch_size) // s + 1

        patches_242 = torch.cat([unf_pre.transpose(1, 2), unf_post.transpose(1, 2)], dim=-1).reshape(-1, 242)

        if self.mode in ("full", "qcnn_only", "frozen_random_qcnn"):
            a_mat = self.qcnn.build_heisenberg_real_matrix() if a_cached is None else a_cached
            chunks = []
            for st in range(0, patches_242.shape[0], 16384):
                sub = self._normalize_patches(patches_242[st : st + 16384])
                chunks.append(self.qcnn.forward_patches_fast(sub, a_matrix=a_mat))
            p_q = torch.cat(chunks, dim=0).reshape(B, 1, gh, gw)
        else:
            chunks = []
            for st in range(0, patches_242.shape[0], 16384):
                sub = self._normalize_patches(patches_242[st : st + 16384])
                chunks.append(self.classical_patch(sub))
            p_q = torch.cat(chunks, dim=0).reshape(B, 1, gh, gw)

        p_q_up = F.interpolate(p_q, size=(H, W), mode="bilinear", align_corners=False) if (gh, gw) != (H, W) else p_q
        if self.mode == "qcnn_only":
            return self.q_scale * (p_q_up - 0.5) + self.q_bias
        return self.refiner(p_q_up, pre_rgb, post_rgb, include_q_residual=True)


def audit_qacd_complexity(tile_h: int = 256, tile_w: int = 256) -> dict:
    n_px = tile_h * tile_w
    head_macs_per_px = (5 * 24 * 9) + (24 * 48 * 9) + (48 * 12 * 9) + (12 * 7 * 9) + (7 * 1 * 1)
    return {
        "qcnn_params": 62,
        "qcnn_two_qubit_pairs": 19,
        "qcnn_logical_2q_gates": 26,
        "qcnn_decomposed_cx_gates": 40,
        "spatial_refinement_head_params": 17488,
        "qacd_total_params": 17550,
        "head_macs_per_pixel": head_macs_per_px,
        "head_macs_per_256x256_tile": head_macs_per_px * n_px,
        "head_gmacs_per_tile": round(head_macs_per_px * n_px / 1e9, 4),
        "head_gflops_per_tile": round(2.0 * head_macs_per_px * n_px / 1e9, 4),



    }


if __name__ == "__main__":
    m = QACDGenuineHybrid(stride=4, mode="full")
    p_rand = torch.rand(32, 242)
    with torch.no_grad():
        ref = m.qcnn.forward_patches(p_rand)
        fast = m.qcnn.forward_patches_fast(p_rand)
        diff = float((ref - fast).abs().max().item())
    assert diff < 1e-6, f"Heisenberg pullback mismatch: {diff}"
    b62 = ClassicalPatchProjector62()
    print(f"Equivalence max diff: {diff:.2e}")
    print("Complexity audit:", audit_qacd_complexity())
    print("ClassicalPatchProjector62 params:", sum(p.numel() for p in b62.parameters()))
