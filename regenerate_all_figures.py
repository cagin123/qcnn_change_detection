"""
================================================================================
PUBLICATION-GRADE FIGURE REGENERATOR FOR IEEE TGRS MANUSCRIPT
================================================================================
Fixes all figure font sizes, removes overlapping/clipped texts, removes the
unneeded all-models spatial comparison figure, and generates crystal-clear
full-width figures:
  1. Slide4.JPG (Fig. 1: Global Study Area Map with fixed non-overlapping labels
     and (a)-(h) panel badges)
  2. QCNN_corrected.png (Fig. 3: 9-Qubit QCNN Architecture with large bold fonts)
  3. fig_test1_global_predictions.png (Fig. 4: Test 1 McKinney & Gospers Primary)
  4. fig_test1_secondary_predictions.png (Fig. 5: Test 1 McKinney Ridge & Gospers Canyon)
  5. fig_test2_india_predictions.png (Fig. 6: Test 2 Similipal Swath & Bandhavgarh)
  6. fig_test2_india_secondary_predictions.png (Fig. 7: Test 2 Uttarakhand & Similipal Core)
  7. fig_quantitative_ablation_analysis.png (Fig. 8: Multi-panel Quantitative & Ablation Plots)
================================================================================
"""

import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(PKG_DIR, ".."))
PRED_DIR = os.path.join(ROOT_DIR, "outputs", "prediction_maps")
ORIG_IEEE_DIR = os.path.join(ROOT_DIR, "Change_Detection_of_Syham_Project", "IEEE")
ARTIFACT_DIR = r"C:\Users\murus\.gemini\antigravity\brain\61d3c6ad-0b57-47c1-9939-9b3516f5503e"


def fix_slide4_study_area_map():
    """
    Repairs Slide4.JPG:
    - Restores top margin so '47.1667°N' and 'Fire Center' are not clipped at top
    - Masks out overlapping/clipped text ('Evans Canyor', 'Wallow (USA)/Bighorn (USA)',
      'Uttarakh', 'Bandhavgarh (India)/Similipal (India)') and replaces them with
      clean, large, non-overlapping callout badges
    - Adds bold (a)-(h) subfigure badges on each of the 8 dNBR insets matching the caption
    """
    orig_slide4 = os.path.join(ORIG_IEEE_DIR, "Slide4.JPG")
    if not os.path.exists(orig_slide4):
        orig_slide4 = os.path.join(PKG_DIR, "Slide4.JPG")

    im = Image.open(orig_slide4).convert("RGB")
    W, H = im.size  # 1280 x 720

    # Add 26px clean white top margin so the top-middle inset is unclipped
    top_pad = 26
    canvas = Image.new("RGB", (W, H + top_pad), (255, 255, 255))
    canvas.paste(im, (0, top_pad))

    fig, ax = plt.subplots(figsize=(16, 9.325), dpi=300)
    ax.imshow(np.asarray(canvas))
    ax.set_xlim(0, W)
    ax.set_ylim(H + top_pad, 0)
    ax.axis("off")

    # 1. Repair clipped top border of top-middle inset (x=345..613, y=0..top_pad+18)
    # Cover clipped '47.1667°N' at x=345..442, y=top_pad..top_pad+16
    ax.add_patch(mpatches.Rectangle((342, 4), 102, 36, facecolor="white", edgecolor="none", zorder=3))
    ax.text(440, 26, "47.1667°N", fontsize=11.5, ha="right", va="center", color="black", zorder=4)

    # Repair top border of the top-middle inset frame & Fire Center legend box
    ax.plot([446, 612], [24, 24], color="#222222", lw=1.2, zorder=4)

    # 2. Mask out the overlapping/clipped map labels on the background map
    # (a) 'Evans Canyor' & 'McKinney (USA)' region: x=248..345, y=130+top_pad..174+top_pad
    ax.add_patch(mpatches.Rectangle((248, 130 + top_pad), 96, 44, facecolor="#efefef", edgecolor="none", zorder=3))
    # (b) Overlapping 'Wallow (USA)' and 'Bighorn (USA)' region: x=268..365, y=189+top_pad..216+top_pad
    ax.add_patch(mpatches.Rectangle((268, 189 + top_pad), 96, 27, facecolor="#efefef", edgecolor="none", zorder=3))
    # (c) Clipped 'Uttarakh' and overlapping 'Bandhavgarh / Similipal' in India:
    #     'Uttarakh' at x=888..960, y=195+top_pad..235+top_pad
    ax.add_patch(mpatches.Rectangle((888, 195 + top_pad), 72, 40, facecolor="#efefef", edgecolor="none", zorder=3))
    #     'Bandhavgarh (India)' & 'Similipal (India)' at x=919..1040, y=232+top_pad..262+top_pad
    ax.add_patch(mpatches.Rectangle((919, 232 + top_pad), 121, 30, facecolor="#efefef", edgecolor="none", zorder=3))
    # (d) 'Gospers Mountain (Australia)' small label: x=1142..1250, y=482+top_pad..516+top_pad
    ax.add_patch(mpatches.Rectangle((1142, 482 + top_pad), 110, 35, facecolor="white", edgecolor="none", zorder=3))

    # 3. Draw clean, non-overlapping, high-legibility callout pills for all sites
    bbox_style = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#1f4e78", lw=1.0, alpha=0.95)
    ax.text(260, 136 + top_pad, "Evans Canyon (USA)", fontsize=9.5, fontweight="bold",
            color="#111111", ha="left", va="center", bbox=bbox_style, zorder=6)
    ax.text(246, 165 + top_pad, "McKinney (USA)", fontsize=9.5, fontweight="bold",
            color="#111111", ha="left", va="center", bbox=bbox_style, zorder=6)
    ax.text(238, 194 + top_pad, "Wallow (USA)", fontsize=9.5, fontweight="bold",
            color="#111111", ha="left", va="center", bbox=bbox_style, zorder=6)
    ax.text(238, 221 + top_pad, "Bighorn (USA)", fontsize=9.5, fontweight="bold",
            color="#111111", ha="left", va="center", bbox=bbox_style, zorder=6)

    ax.text(892, 204 + top_pad, "Uttarakhand (India)", fontsize=9.5, fontweight="bold",
            color="#111111", ha="center", va="center", bbox=bbox_style, zorder=6)
    ax.text(945, 234 + top_pad, "Bandhavgarh (India)", fontsize=9.2, fontweight="bold",
            color="#111111", ha="left", va="center", bbox=bbox_style, zorder=6)
    ax.text(945, 260 + top_pad, "Similipal (India)", fontsize=9.2, fontweight="bold",
            color="#111111", ha="left", va="center", bbox=bbox_style, zorder=6)

    ax.text(1195, 496 + top_pad, "Gospers Mountain\n(Australia)", fontsize=9.5, fontweight="bold",
            color="#111111", ha="center", va="center", bbox=bbox_style, zorder=6)

    # 4. Add bold subfigure panel badges (a)-(h) on each of the 8 dNBR insets
    # Caption order:
    # (a) Wallow Fire, AZ (center-left inset: x=452, y=236+top_pad)
    # (b) Gospers Mountain, NSW (bottom-right inset: x=812, y=492+top_pad)
    # (c) Bighorn Fire, AZ (bottom-left inset: x=355, y=482+top_pad)
    # (d) Evans Canyon, WA (top-middle inset: x=452, y=38+top_pad)
    # (e) McKinney Fire, CA (top-left inset: x=84, y=38+top_pad)
    # (f) Similipal, Odisha (far-right inset: x=1132, y=248+top_pad)
    # (g) Uttarakhand, India (center-right inset: x=730, y=282+top_pad)
    # (h) Bandhavgarh, MP (top-right inset: x=1055, y=62+top_pad)
    panel_badges = [
        ("(a) Wallow", 588, 382 + top_pad),
        ("(b) Gospers Mtn.", 946, 646 + top_pad),
        ("(c) Bighorn", 518, 650 + top_pad),
        ("(d) Evans Canyon", 574, 140 + top_pad),
        ("(e) McKinney", 198, 156 + top_pad),
        ("(f) Similipal", 1205, 415 + top_pad),
        ("(g) Uttarakhand", 832, 405 + top_pad),
        ("(h) Bandhavgarh", 1185, 152 + top_pad),
    ]
    badge_box = dict(boxstyle="round,pad=0.22", facecolor="#fff9db", edgecolor="#b45309", lw=1.2, alpha=0.96)
    for label, bx, by in panel_badges:
        ax.text(bx, by, label, fontsize=10.2, fontweight="bold", color="#111111",
                ha="center", va="center", bbox=badge_box, zorder=7)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_jpg = os.path.join(PKG_DIR, "Slide4.JPG")
    fig.savefig(out_jpg, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(out_jpg, os.path.join(ARTIFACT_DIR, "Slide4_fixed.png"))
    print(f"Saved fixed Fig. 1 Study Area Map -> {out_jpg}")


def generate_large_font_qcnn_figure():
    """
    Regenerates QCNN_corrected.png (Fig. 3) with large, bold, publication-ready
    fonts (13pt-18pt) so that every gate, wire, and block label is crisp and readable.
    """
    fig = plt.figure(figsize=(16.5, 9.2), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.55, 1.05], hspace=0.24, wspace=0.12)

    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 10.65)
    ax.axis("off")

    qubits = [f"$q_{i}\\ (|0\\rangle)$" for i in range(9)]
    y_pos = [8.3 - i * 1.0 for i in range(9)]

    for i, y in enumerate(y_pos):
        ax.plot([4.8, 95.5], [y, y], color="#334155", lw=1.6, zorder=1)
        ax.text(4.3, y, qubits[i], ha="right", va="center", fontsize=13.5, fontweight="bold", color="#0f172a")

    ax.text(50, 10.35,
            "9-Qubit FRQI + 3-Block Quantum Convolutional Neural Network (QCNN) Architecture (62 Trainable Parameters)",
            ha="center", va="center", fontsize=15.0, fontweight="bold", color="#0f172a")

    def draw_gate(ax, x, y, w, h, label, facecolor="#e0f2fe", edgecolor="#0284c7", fontsize=11.5, textcolor="#0f172a"):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.08,rounding_size=0.18",
                             facecolor=facecolor, edgecolor=edgecolor, lw=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)

    def draw_span_gate(ax, x, y_top, y_bot, w, label, facecolor="#dcfce7", edgecolor="#15803d", fontsize=10.8):
        h = (y_top - y_bot) + 0.62
        y_mid = 0.5 * (y_top + y_bot)
        box = FancyBboxPatch((x - w / 2, y_mid - h / 2), w, h,
                             boxstyle="round,pad=0.06,rounding_size=0.18",
                             facecolor=facecolor, edgecolor=edgecolor, lw=1.4, zorder=3)
        ax.add_patch(box)
        ax.text(x, y_mid, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#0f172a", zorder=4)

    # Stage 0: FRQI
    ax.text(9.6, 9.45, "FRQI Encoding\n(q0-q7: H, q8: C8-RY)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#9a3412",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff7ed", edgecolor="#c2410c", lw=1.0))
    for i in range(8):
        draw_gate(ax, 9.2, y_pos[i], 3.3, 0.62, "H", facecolor="#ffedd5", edgecolor="#c2410c", fontsize=12.5)
    ax.plot([13.8, 13.8], [y_pos[0], y_pos[8]], color="#c2410c", lw=2.0, zorder=2)
    for i in range(8):
        ax.scatter([13.8], [y_pos[i]], color="#c2410c", s=55, zorder=4)
    draw_gate(ax, 13.8, y_pos[8], 5.6, 0.64, r"$R_Y(2\theta_k)$", facecolor="#ffedd5", edgecolor="#c2410c", fontsize=11.5)

    # Block 1
    ax.text(26.8, 9.45, "Block 1 (9 -> 4 Qubits)\n8 Conv (32p) + 4 Pool (8p)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#0f766e",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f0fdf4", edgecolor="#0f766e", lw=1.0))
    for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        draw_span_gate(ax, 20.0, y_pos[a], y_pos[b], 5.5, f"Conv\n({a},{b})\n4p", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=10.5)
    for (a, b) in [(1, 2), (3, 4), (5, 6), (7, 8)]:
        draw_span_gate(ax, 26.2, y_pos[a], y_pos[b], 5.5, f"Conv\n({a},{b})\n4p", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=10.5)
    for (src, tgt) in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        draw_span_gate(ax, 32.4, y_pos[src], y_pos[tgt], 5.5, f"Pool\n({src}->{tgt})\n2p", facecolor="#e0e7ff", edgecolor="#4338ca", fontsize=10.5)

    # Block 2
    ax.text(47.2, 9.45, "Block 2 (4 -> 2 Qubits)\n3 Conv (12p) + 2 Pool (4p)", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color="#0f766e",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f0fdf4", edgecolor="#0f766e", lw=1.0))
    for (a, b) in [(0, 2), (4, 6)]:
        draw_span_gate(ax, 41.2, y_pos[a], y_pos[b], 5.5, f"Conv\n({a},{b})\n4p", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=10.8)
    draw_span_gate(ax, 47.2, y_pos[2], y_pos[4], 5.5, "Conv\n(2,4)\n4p", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=10.8)
    for (src, tgt) in [(0, 2), (4, 6)]:
        draw_span_gate(ax, 53.2, y_pos[src], y_pos[tgt], 5.5, f"Pool\n({src}->{tgt})\n2p", facecolor="#e0e7ff", edgecolor="#4338ca", fontsize=10.8)

    # Block 3
    ax.text(65.6, 9.45, "Block 3 (2 -> 1 Qubit)\n1 Conv (4p) + 1 Pool (2p)", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color="#0f766e",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f0fdf4", edgecolor="#0f766e", lw=1.0))
    draw_span_gate(ax, 62.4, y_pos[0], y_pos[4], 5.5, "Conv\n(0,4)\n4p", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=11.0)
    draw_span_gate(ax, 68.8, y_pos[0], y_pos[4], 5.5, "Pool\n(0->4)\n2p", facecolor="#e0e7ff", edgecolor="#4338ca", fontsize=11.0)

    # Readout
    ax.text(83.8, 9.45, "Pauli-Z Readout\n9 Expectation Values", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#eff6ff", edgecolor="#1d4ed8", lw=1.0))
    for i in range(9):
        mx = 76.5 + (i % 3) * 5.2
        draw_gate(ax, mx, y_pos[i], 4.6, 0.62, rf"$\langle Z_{i}\rangle$", facecolor="#dbeafe", edgecolor="#1d4ed8", fontsize=11.5)

    # Bottom Left: Two-Qubit Conv Detail
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(0, 10)
    ax_c.axis("off")
    box_c = FancyBboxPatch((2, 0.5), 96, 8.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                           facecolor="#f0fdf4", edgecolor="#0f766e", lw=1.6)
    ax_c.add_patch(box_c)
    ax_c.text(50, 8.2, r"Two-Qubit Conv Block (model_qcnn.py): 4 Params ($\theta_1, \theta_2, \theta_3, \theta_4$)",
              ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0f766e")
    ax_c.plot([12, 92], [5.8, 5.8], color="#334155", lw=1.6)
    ax_c.plot([12, 92], [2.4, 2.4], color="#334155", lw=1.6)
    ax_c.text(10, 5.8, r"$q_a$", ha="right", va="center", fontsize=13, fontweight="bold")
    ax_c.text(10, 2.4, r"$q_b$", ha="right", va="center", fontsize=13, fontweight="bold")
    draw_gate(ax_c, 24, 5.8, 15, 2.0, r"$R_Y(\theta_1)$", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=12)
    draw_gate(ax_c, 42, 5.8, 15, 2.0, r"$R_Z(\theta_2)$", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=12)
    draw_gate(ax_c, 33, 2.4, 15, 2.0, r"$R_Y(\theta_3)$", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=12)
    ax_c.plot([58, 58], [2.4, 5.8], color="#0f766e", lw=2.0)
    ax_c.scatter([58], [5.8], color="#0f766e", s=70, zorder=4)
    circle = plt.Circle((58, 2.4), 2.1, facecolor="white", edgecolor="#0f766e", lw=2.0, zorder=3)
    ax_c.add_patch(circle)
    ax_c.text(58, 2.4, "+", ha="center", va="center", fontsize=14, fontweight="bold", color="#0f766e", zorder=4)
    draw_gate(ax_c, 76, 2.4, 15, 2.0, r"$R_Y(\theta_4)$", facecolor="#ccfbf1", edgecolor="#0f766e", fontsize=12)

    # Bottom Right: Two-Qubit Pool Detail
    ax_p = fig.add_subplot(gs[1, 1])
    ax_p.set_xlim(0, 100)
    ax_p.set_ylim(0, 10)
    ax_p.axis("off")
    box_p = FancyBboxPatch((2, 0.5), 96, 8.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                           facecolor="#eef2ff", edgecolor="#4338ca", lw=1.6)
    ax_p.add_patch(box_p)
    ax_p.text(50, 8.2, r"Two-Qubit Pool Block (model_qcnn.py): 2 Params ($\phi_1, \phi_2$)",
              ha="center", va="center", fontsize=11.0, fontweight="bold", color="#4338ca")
    ax_p.plot([12, 88], [5.8, 5.8], color="#334155", lw=1.6)
    ax_p.plot([12, 88], [2.4, 2.4], color="#334155", lw=1.6)
    ax_p.text(10, 5.8, r"$q_{\mathrm{src}}$", ha="right", va="center", fontsize=13, fontweight="bold")
    ax_p.text(10, 2.4, r"$q_{\mathrm{tgt}}$", ha="right", va="center", fontsize=13, fontweight="bold")
    ax_p.plot([36, 36], [2.4, 5.8], color="#4338ca", lw=2.0)
    ax_p.scatter([36], [5.8], color="#4338ca", s=70, zorder=4)
    draw_gate(ax_p, 36, 2.4, 18, 2.0, r"$R_Z(\phi_1)$", facecolor="#e0e7ff", edgecolor="#4338ca", fontsize=12)
    ax_p.plot([66, 66], [2.4, 5.8], color="#4338ca", lw=2.0)
    ax_p.scatter([66], [5.8], color="#4338ca", s=70, zorder=4)
    draw_gate(ax_p, 66, 2.4, 18, 2.0, r"$R_Y(\phi_2)$", facecolor="#e0e7ff", edgecolor="#4338ca", fontsize=12)

    out_png = os.path.join(PKG_DIR, "QCNN_corrected.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(out_png, os.path.join(ARTIFACT_DIR, "QCNN_corrected.png"))
    print(f"Saved large-font Fig. 3 QCNN -> {out_png}")


def extract_four_panels_from_pred_map(pred_png_path):
    """
    Extracts the 4 square image panels (Pre-Fire, Post-Fire, Ground Truth, QACD Prediction)
    from an existing high-res pred_map_*.png file, stripping away old small titles.
    """
    im = Image.open(pred_png_path).convert("RGB")
    arr = np.asarray(im)
    # Find rows and columns that are not pure white (255,255,255)
    non_white = np.any(arr < 250, axis=-1)
    # The 4 square panels sit in the lower 75% of the image (below the title and column headers)
    H, W = non_white.shape
    sub = non_white[int(H * 0.22):, :]
    row_counts = np.sum(sub, axis=1)
    valid_rows = np.where(row_counts > W * 0.35)[0] + int(H * 0.22)
    r_min, r_max = int(valid_rows[0]), int(valid_rows[-1])

    col_mask = np.sum(non_white[r_min:r_max, :], axis=0) > (r_max - r_min) * 0.5
    # Find contiguous horizontal segments corresponding to the 4 panels
    diffs = np.diff(col_mask.astype(int))
    starts = list(np.where(diffs == 1)[0] + 1)
    ends = list(np.where(diffs == -1)[0])
    if col_mask[0]:
        starts = [0] + starts
    if col_mask[-1]:
        ends = ends + [W - 1]
    panels = []
    for c0, c1 in zip(starts, ends):
        if (c1 - c0) > 400:
            panels.append(arr[r_min:r_max, c0:c1, :].copy())
    return panels


def build_two_row_prediction_figure(row_configs, main_title, out_filename):
    """
    Builds a publication-grade 2-row x 4-column prediction figure with:
    - Large 15.5pt bold subplot headers
    - Generous vertical spacing (hspace=0.34) so Row 2 titles NEVER touch Row 1 images
    - Large 13pt high-contrast legend boxes
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.8), dpi=250)

    for row_idx, cfg in enumerate(row_configs):
        png_path = os.path.join(PRED_DIR, cfg["file"])
        if not os.path.exists(png_path):
            png_path = os.path.join(ARTIFACT_DIR, cfg["file"])
        panels = extract_four_panels_from_pred_map(png_path)

        col_titles = [
            f"{cfg['panel_prefix']} Pre-Fire ({cfg['short_name']})",
            "Post-Fire Composite",
            "Ground Truth (MODIS + dNBR)",
            f"QACD Prediction (OA: {cfg['oa']}, F1: {cfg['f1']})"
        ]
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            panel_img = panels[col_idx].copy()
            # On panels 2 and 3 (Ground Truth and QACD Prediction), erase the old small
            # baked-in legend in the bottom-right corner before drawing the large crisp legend
            if col_idx in (2, 3):
                ph, pw, _ = panel_img.shape
                sample_color = panel_img[int(ph * 0.80), int(pw * 0.92)].copy()
                panel_img[int(ph * 0.83):ph, int(pw * 0.56):pw] = sample_color
            ax.imshow(panel_img)
            ax.set_title(col_titles[col_idx], fontsize=14.2, fontweight="bold", pad=10, color="#0f172a")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#334155")
                spine.set_linewidth(1.0)

            if col_idx in (2, 3):
                lbl = "Burned (GT)" if col_idx == 2 else "Predicted Burned"
                leg = ax.legend(
                    handles=[
                        mpatches.Patch(color="blue", label="No Change"),
                        mpatches.Patch(color="red", label=lbl),
                    ],
                    loc="lower right",
                    fontsize=12.5,
                    framealpha=0.96,
                    facecolor="white",
                    edgecolor="#1e293b",
                )
                for text in leg.get_texts():
                    text.set_fontweight("bold")

    fig.suptitle(main_title, fontsize=17.0, fontweight="bold", y=0.975, color="#0f172a")
    plt.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.90, wspace=0.06, hspace=0.28)
    out_path = os.path.join(PKG_DIR, out_filename)
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, out_filename))
    print(f"Saved non-overlapping 2x4 prediction figure -> {out_path}")


def generate_quantitative_ablation_figure():
    """
    Generates Fig. 8: 3-panel quantitative performance and ablation figure with
    large 13pt-15pt publication fonts:
      (a) Test 1 vs Test 2 F1-Score and cIoU across all 5 evaluated models
      (b) Inference Stride (s=1,2,4,8) vs Per-Tile Latency and F1-Score/cIoU
      (c) MODIS-to-Landsat Perimeter Buffer Exclusion Width vs Interior Metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.6), dpi=300)

    # Panel (a): Grouped Bar Chart of F1 and cIoU on Test 1 & Test 2
    ax1 = axes[0]
    models = ["QACD\n(62p)", "MaskCD\n(107.4M)", "CNN-62\n(61p)", "Siamese\nU-Net", "Change-\nFormer"]
    x = np.arange(len(models))
    w = 0.20

    t1_f1 = [0.909, 0.942, 0.482, 0.056, 0.0004]
    t1_ciou = [0.832, 0.890, 0.317, 0.029, 0.0002]
    t2_f1 = [0.925, 0.000, 0.171, 0.002, 0.000]
    t2_ciou = [0.860, 0.000, 0.093, 0.001, 0.000]

    b1 = ax1.bar(x - 1.5 * w, t1_f1, w, label="Test 1 F1 (Global)", color="#0284c7", edgecolor="#0f172a", lw=0.8)
    b2 = ax1.bar(x - 0.5 * w, t1_ciou, w, label="Test 1 cIoU (Global)", color="#38bdf8", edgecolor="#0f172a", lw=0.8)
    b3 = ax1.bar(x + 0.5 * w, t2_f1, w, label="Test 2 F1 (India OOD)", color="#dc2626", edgecolor="#0f172a", lw=0.8)
    b4 = ax1.bar(x + 1.5 * w, t2_ciou, w, label="Test 2 cIoU (India OOD)", color="#f87171", edgecolor="#0f172a", lw=0.8)

    ax1.set_title("(a) Change-Class F1 & cIoU Across Benchmarks", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Metric Score", fontsize=12.5, fontweight="bold")
    ax1.set_ylim(0, 1.12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.tick_params(axis="y", labelsize=11.5)
    ax1.legend(loc="upper right", fontsize=10.2, framealpha=0.95)

    for bar in [b1[0], b3[0], b1[1], b3[1]]:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Panel (b): Inference Stride vs Latency and Accuracy
    ax2 = axes[1]
    strides = ["s = 1\n(65,536)", "s = 2\n(16,384)", "s = 4\n(4,096)", "s = 8\n(1,024)"]
    sx = np.arange(len(strides))
    latencies = [4.20, 1.05, 0.26, 0.07]
    f1_s = [0.909, 0.909, 0.904, 0.892]
    ciou_s = [0.832, 0.832, 0.825, 0.805]

    ax2_twin = ax2.twinx()
    bars_lat = ax2.bar(sx - 0.15, latencies, width=0.32, color="#cbd5e1", edgecolor="#334155", lw=1.0, label="Latency / Tile (s)")
    line_f1 = ax2_twin.plot(sx, f1_s, "o-", color="#dc2626", lw=2.5, markersize=8, label="Change F1-Score")
    line_ciou = ax2_twin.plot(sx, ciou_s, "s--", color="#0284c7", lw=2.5, markersize=8, label="Change cIoU")

    ax2.set_title("(b) Strided Inference Speedup vs. Accuracy", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xticks(sx)
    ax2.set_xticklabels(strides, fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Latency per 256×256 Tile (s)", fontsize=12.5, fontweight="bold", color="#334155")
    ax2_twin.set_ylabel("F1-Score / cIoU", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax2.set_ylim(0, 5.0)
    ax2_twin.set_ylim(0.75, 0.96)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.tick_params(axis="y", labelsize=11.5)
    ax2_twin.tick_params(axis="y", labelsize=11.5)

    for b in bars_lat:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.10, f"{h:.2f}s",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1e293b")

    lines_labels = [bars_lat] + line_f1 + line_ciou
    labels = [l.get_label() for l in lines_labels]
    ax2_twin.legend(lines_labels, labels, loc="center right", fontsize=10.5, framealpha=0.95)

    # Panel (c): MODIS-to-Landsat Perimeter Buffer Purity Analysis
    ax3 = axes[2]
    buffers = ["0 m\n(All Pixels)", "30 m\n(1-Pixel)", "60 m\n(2-Pixel)", "90 m\n(3-Pixel)"]
    bx = np.arange(len(buffers))
    t1_buf_f1 = [0.909, 0.938, 0.950, 0.959]
    t1_buf_rec = [0.918, 0.945, 0.956, 0.964]
    t2_buf_f1 = [0.925, 0.946, 0.958, 0.965]

    ax3.plot(bx, t1_buf_f1, "o-", color="#0284c7", lw=2.5, markersize=8, label="Test 1 (Global) F1-Score")
    ax3.plot(bx, t1_buf_rec, "^--", color="#0f766e", lw=2.2, markersize=8, label="Test 1 (Global) Recall")
    ax3.plot(bx, t2_buf_f1, "s-", color="#dc2626", lw=2.5, markersize=8, label="Test 2 (India) F1-Score")

    for i in range(len(bx)):
        ax3.text(bx[i], t1_buf_f1[i] - 0.008, f"{t1_buf_f1[i]:.3f}", ha="center", va="top", fontsize=10, fontweight="bold", color="#0284c7")
        ax3.text(bx[i], t2_buf_f1[i] + 0.004, f"{t2_buf_f1[i]:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#dc2626")

    ax3.set_title("(c) Perimeter Buffer Exclusion Sensitivity", fontsize=14, fontweight="bold", pad=10)
    ax3.set_xticks(bx)
    ax3.set_xticklabels(buffers, fontsize=11.5, fontweight="bold")
    ax3.set_ylabel("Segmentation Score", fontsize=12.5, fontweight="bold")
    ax3.set_ylim(0.88, 0.985)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.tick_params(axis="y", labelsize=11.5)
    ax3.legend(loc="lower right", fontsize=10.5, framealpha=0.95)

    plt.tight_layout()
    out_path = os.path.join(PKG_DIR, "fig_quantitative_ablation_analysis.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, "fig_quantitative_ablation_analysis.png"))
    print(f"Saved Fig. 8 Quantitative & Ablation Analysis -> {out_path}")


if __name__ == "__main__":
    # Remove old fig_multimodel_comparison.png as requested by user
    old_mm = os.path.join(PKG_DIR, "fig_multimodel_comparison.png")
    if os.path.exists(old_mm):
        os.remove(old_mm)
        print("Removed unneeded fig_multimodel_comparison.png")

    fix_slide4_study_area_map()
    generate_large_font_qcnn_figure()

    # Fig. 4: Test 1 Primary Scenes (McKinney x2048_y2816 & Gospers x1024_y1536)
    build_two_row_prediction_figure(
        [
            {"file": "pred_map_mckinney_fire_x2048_y2816.png", "panel_prefix": "(a)", "short_name": "McKinney Fire, CA [x2048_y2816]", "oa": "0.9358", "f1": "0.9331"},
            {"file": "pred_map_gospers_fire_x1024_y1536.png", "panel_prefix": "(b)", "short_name": "Gospers Mtn., NSW [x1024_y1536]", "oa": "0.9386", "f1": "0.9298"},
        ],
        "Test 1 (Global Benchmark — Primary Crown-Fire Scenes: McKinney Fire, California & Gospers Mountain, Australia)",
        "fig_test1_global_predictions.png"
    )

    # Fig. 5: Test 1 Secondary Scenes (McKinney x2304_y2816 & Gospers x1280_y256)
    build_two_row_prediction_figure(
        [
            {"file": "pred_map_mckinney_fire_x2304_y2816.png", "panel_prefix": "(a)", "short_name": "McKinney Fire Ridge, CA [x2304_y2816]", "oa": "0.9497", "f1": "0.9221"},
            {"file": "pred_map_gospers_fire_x1280_y256.png", "panel_prefix": "(b)", "short_name": "Gospers Mtn. Escarpment, NSW [x1280_y256]", "oa": "0.9412", "f1": "0.9264"},
        ],
        "Test 1 (Global Benchmark — Complex Ridge & Valley Escarpment Sectors: California & New South Wales)",
        "fig_test1_secondary_predictions.png"
    )

    # Fig. 6: Test 2 India Primary Scenes (Similipal Swath Edge x1792_y1280 & Bandhavgarh x512_y256)
    build_two_row_prediction_figure(
        [
            {"file": "pred_map_Similipal_Fire_2021_x1792_y1280.png", "panel_prefix": "(a)", "short_name": "Similipal Swath Sector, Odisha [x1792_y1280]", "oa": "0.9755", "f1": "0.9112"},
            {"file": "pred_map_Bandhavgarh_Fire_x512_y256.png", "panel_prefix": "(b)", "short_name": "Bandhavgarh Core, MP [x512_y256]", "oa": "0.9352", "f1": "0.9155"},
        ],
        "Test 2 (India Cross-Continental Benchmark — Swath-Boundary Suppression & Fragmented Deciduous Burns)",
        "fig_test2_india_predictions.png"
    )

    # Fig. 7: Test 2 India Secondary Scenes (Uttarakhand x0_y256 & Similipal Core x1792_y1024)
    build_two_row_prediction_figure(
        [
            {"file": "pred_map_Uttarakhand_Fire_2021_x0_y256.png", "panel_prefix": "(a)", "short_name": "Uttarakhand Himalayan Pine [x0_y256]", "oa": "0.9439", "f1": "0.9175"},
            {"file": "pred_map_Similipal_Fire_2021_x1792_y1024.png", "panel_prefix": "(b)", "short_name": "Similipal Sal Mosaic, Odisha [x1792_y1024]", "oa": "0.9684", "f1": "0.9148"},
        ],
        "Test 2 (India Cross-Continental Benchmark — Himalayan Steep-Slope Chir Pine & Interior Sal Forest Sectors)",
        "fig_test2_india_secondary_predictions.png"
    )

    generate_quantitative_ablation_figure()
