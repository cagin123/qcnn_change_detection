# Quantum Automatic Change Detection (QACD) for Wildfire Burned Area Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Quantum-Qiskit%20%7C%20PennyLane-6929C4.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)

A hybrid Quantum Automatic Change Detection (QACD) pipeline for wildfire burned-area delineation from bi-temporal Landsat-8 Operational Land Imager (OLI) satellite imagery ($30\text{ m}$ spatial resolution), built with [Qiskit](https://qiskit.org/), [PennyLane](https://pennylane.ai/), and [PyTorch](https://pytorch.org/).

The pipeline extracts localized bi-temporal image patches, encodes each patch pair into a 9-qubit quantum state using FRQI (Flexible Representation of Quantum Images), trains a parameterized Quantum Convolutional Neural Network (QCNN) with alternating convolution and pooling layers via the parameter-shift rule, and produces full-tile burned-area segmentation maps.

---

## Study Areas (Fig. 1)

![Global Distribution of Wildfire Study Areas](docs/study_areas_map.jpg)

*Fig. 1. Global distribution of the nine wildfire study areas across North America, Australia, and India with differenced Normalized Burn Ratio (dNBR) burn-severity insets.*

---

## Method Overview: FRQI Quantum State Preparation (Fig. 2)

![FRQI Quantum Encoding Pipeline](docs/method_overview.png)

*Fig. 2. FRQI-based quantum state preparation pipeline for bi-temporal pre-fire and post-fire satellite image patches.*

For each target pixel center $(r, c)$, localized $11 \times 11$ pre-fire and post-fire windows ($121$ pixels each) are flattened into a 242-element bi-temporal vector, normalized to $[0, 1]$, and zero-padded with $14$ trailing zeros to $N = 2^8 = 256$ elements $v = (v_0, \dots, v_{255})^\top$. Hadamard gates $H^{\otimes 8}$ prepare a uniform superposition over the 8-qubit position register $|k\rangle_p$ ($q_0\text{--}q_7$), while 8-qubit controlled rotations $C^8_{|k\rangle_p} R_Y(2\theta_k)$ entangle each pixel intensity angle $\theta_k = \frac{\pi}{2}v_k$ into the color qubit $|q_8\rangle$:

$$|I(v)\rangle = \frac{1}{16}\sum_{k=0}^{255}\left[\cos(\theta_k)|0\rangle_c + \sin(\theta_k)|1\rangle_c\right] \otimes |k\rangle_p$$

The resulting 9-qubit FRQI state is processed by a three-stage hierarchical QCNN ($9 \rightarrow 4 \rightarrow 2 \rightarrow 1$ active qubits) composed of staggered two-qubit convolution blocks ($R_Y, R_Z, \mathrm{CX}$) and controlled quantum pooling blocks ($CR_Z, CR_Y$) before Pauli-$Z$ expectation readout.

---

## Pipeline Overview

1. **Data Loading & Swath Masking** — Reads co-registered pre-fire and post-fire Landsat-8 OLI scenes together with ground-truth change masks ($\mathrm{dNBR} \ge 0.27$) and applies valid-pixel orbital swath masking.
2. **Patch Extraction** — Extracts sliding-window $11 \times 11$ patches from the normalized pre-fire and post-fire bands and concatenates them into a 242-element feature vector per patch (padded to $256 = 2^8$).
3. **FRQI Encoding** — Encodes each normalized patch vector into a 9-qubit quantum state ($8$ spatial position qubits + $1$ color qubit).
4. **QCNN Classification** — Processes the quantum state through three alternating stages of parameterized two-qubit convolution and controlled pooling layers, trained end-to-end using the analytical two-term parameter-shift rule and AdamW optimization.
5. **Prediction Map Generation** — Stitches predicted change probabilities into full-resolution $256 \times 256$ binary segmentation maps.

---

## Repository Structure

```text
.
├── qcnn_change_detection.py          # Standalone Qiskit FRQI-QCNN pipeline (patch training & GeoTIFF map export)
├── model_qcnn.py                     # 9-qubit FRQI + 3-stage QCNN circuit implementation
├── model_qacd.py                     # Hybrid QACD model architecture
├── model_siamese_unet.py             # Fully Convolutional Siamese U-Net baseline
├── model_changeformer.py             # ChangeFormer MiT Siamese transformer baseline
├── model_bit.py                      # Bi-Temporal Image Transformer (BIT) baseline
├── model_a2net.py                    # A2Net lightweight attention baseline
├── model_classical_dnbr.py           # Classical dNBR threshold & parameter-matched CNN control
├── dataset_cd.py                     # Multi-continental Landsat-8 OLI dataset loader & swath mask
├── train_and_evaluate_all_rgb.py     # Unified training & evaluation runner
├── train_evaluate_models.py          # Model training utility script
├── run_rgb_evaluation.py             # Evaluation script
├── requirements.txt                  # Python package dependencies
└── docs/                             # Study area map (Fig. 1) & FRQI encoding diagram (Fig. 2)
    ├── study_areas_map.jpg
    └── method_overview.png
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Qiskit FRQI-QCNN Pipeline

Place your bi-temporal raster data under `loc/` (training) and `india_loc/` (testing), then run:

```bash
python qcnn_change_detection.py
```

### 3. Train and Evaluate Models

```bash
python train_and_evaluate_all_rgb.py
```

---

## Authors

- **M. S. Shyam Sunder** — Department of Earth and Atmospheric Sciences, National Institute of Technology Rourkela, Odisha, India
- **Çağın Bera Bay** — FMV Işık Erenköy Science High School, Istanbul, Türkiye
- **Yasemin Poyraz Koçak** — Department of Computer Technologies, Istanbul University-Cerrahpaşa, Istanbul, Türkiye
- **Bhishma Tyagi** (*Corresponding Author*) — Department of Earth and Atmospheric Sciences, National Institute of Technology Rourkela, Odisha, India (`tyagib@nitrkl.ac.in`)

---

## License

This project is licensed under the [MIT License](LICENSE).
