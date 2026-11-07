"""
================================================================================
CLASSICAL REMOTE SENSING BASELINE: DIFFERENCED NORMALIZED BURN RATIO (dNBR)
AND AUTOMATED OTSU SEVERITY THRESHOLDING
================================================================================

Reference Publications:
    1. Key, C. H., & Benson, N. C. (2006).
       "Landscape Assessment: Ground Measure of Severity, the Composite Burn Index;
       and Remote Sensing Measure of Severity, the Differenced Normalized Burn Ratio."
       USDA Forest Service General Technical Report RMRS-GTR-164-CD.

    2. Miller, J. D., & Thode, A. E. (2007).
       "Quantifying burn severity in a heterogeneous landscape with a relative version
       of the delta Normalized Burn Ratio (RdNBR)."
       Remote Sensing of Environment, 109(1), 66–80.

    3. Otsu, N. (1979).
       "A threshold selection method from gray-level histograms."
       IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

Methodological Overview:
    1. Normalized Burn Ratio (NBR):
       Exploits the contrasting response of healthy chlorophyll (high NIR reflection)
       and charred/burned soil and ash (high SWIR absorption):
           NBR = (NIR - SWIR2) / (NIR + SWIR2)
       In Landsat 8 OLI: NIR = Band 5, SWIR2 = Band 7.

    2. Differenced NBR (dNBR):
           dNBR = preNBR - postNBR

    3. Standard USGS / USFS Burn Severity Classification:
           dNBR >= 0.660        -> High Severity Burn
           0.440 <= dNBR < 0.660 -> Moderate-High Severity
           0.270 <= dNBR < 0.440 -> Moderate-Low Severity
           0.100 <= dNBR < 0.270 -> Low Severity
           -0.100 <= dNBR < 0.100-> Unburned / Stable
           dNBR < -0.100        -> Enhanced Regrowth

    4. Relativized dNBR (RdNBR):
       Normalizes for pre-fire vegetative cover:
           RdNBR = dNBR / sqrt(|preNBR| + 1e-4)

    5. Adaptive Binary Change Extraction:
       Uses Otsu's optimal variance thresholding on dNBR to automatically delineate
       burn scars without manual calibration.
================================================================================
"""

import numpy as np


class ClassicalDNBRBaseline:
    """
    Classical Remote Sensing Burned-Area & Change Detection Model.
    Supports raw multispectral Landsat bands (B2-B7), precomputed NBRs,
    USGS standard categorical severity mapping, and Otsu automated thresholding.
    """

    # USGS / USFS Burn Severity Categories (Key & Benson, 2006)
    SEVERITY_CLASSES = {
        0: "Enhanced Regrowth (dNBR < -0.100)",
        1: "Unburned (-0.100 <= dNBR < 0.100)",
        2: "Low Severity (0.100 <= dNBR < 0.270)",
        3: "Moderate-Low Severity (0.270 <= dNBR < 0.440)",
        4: "Moderate-High Severity (0.440 <= dNBR < 0.660)",
        5: "High Severity (dNBR >= 0.660)",
    }

    def __init__(self, default_threshold=0.100, eps=1e-7):
        self.default_threshold = default_threshold
        self.eps = eps

    def compute_nbr_from_bands(self, nir_band, swir2_band):
        """
        Calculates NBR from NIR and SWIR2 reflectance bands.
        NBR = (NIR - SWIR2) / (NIR + SWIR2)
        """
        nir = np.asarray(nir_band, dtype=np.float32)
        swir2 = np.asarray(swir2_band, dtype=np.float32)
        denom = nir + swir2
        denom = np.where(np.abs(denom) < self.eps, self.eps, denom)
        nbr = (nir - swir2) / denom
        return np.nan_to_num(nbr, nan=0.0, posinf=1.0, neginf=-1.0)

    def compute_dnbr(self, pre_nbr, post_nbr):
        """
        Calculates Differenced Normalized Burn Ratio (dNBR).
        dNBR = preNBR - postNBR
        """
        pre = np.asarray(pre_nbr, dtype=np.float32)
        post = np.asarray(post_nbr, dtype=np.float32)
        dnbr = pre - post
        return np.nan_to_num(dnbr, nan=0.0)

    def compute_rdnbr(self, dnbr, pre_nbr):
        """
        Calculates Relativized dNBR (RdNBR) to remove pre-fire vegetation density bias.
        RdNBR = dNBR / sqrt(|preNBR| + 1e-4)
        """
        denom = np.sqrt(np.abs(pre_nbr) + 1e-4)
        return dnbr / denom

    def classify_severity(self, dnbr):
        """
        Classifies dNBR values into the 6 USGS standard severity categories.
        Returns: integer array (0 to 5).
        """
        severity = np.zeros_like(dnbr, dtype=np.uint8)
        severity[dnbr < -0.100] = 0
        severity[(dnbr >= -0.100) & (dnbr < 0.100)] = 1
        severity[(dnbr >= 0.100) & (dnbr < 0.270)] = 2
        severity[(dnbr >= 0.270) & (dnbr < 0.440)] = 3
        severity[(dnbr >= 0.440) & (dnbr < 0.660)] = 4
        severity[dnbr >= 0.660] = 5
        return severity

    def otsu_threshold(self, dnbr, n_bins=256):
        """
        Computes Otsu's optimal global variance threshold for binary change detection.
        """
        valid = dnbr[~np.isnan(dnbr)]
        if len(valid) == 0:
            return self.default_threshold

        hist, bin_edges = np.histogram(valid, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        hist = hist.astype(np.float64) / hist.sum()
        cum_weights = np.cumsum(hist)
        cum_means = np.cumsum(hist * bin_centers)
        global_mean = cum_means[-1]

        # Maximize between-class variance: sigma_b^2 = (mu_T * w - mu)^2 / (w * (1 - w))
        w0 = cum_weights
        w1 = 1.0 - w0
        valid_idx = (w0 > 0) & (w1 > 0)

        between_variance = np.zeros_like(hist)
        between_variance[valid_idx] = (
            (global_mean * w0[valid_idx] - cum_means[valid_idx]) ** 2
        ) / (w0[valid_idx] * w1[valid_idx])

        best_idx = np.argmax(between_variance)
        return float(bin_centers[best_idx])

    def predict_binary(self, pre_nbr, post_nbr, mode="usgs", threshold=None):
        """
        Generates binary change mask (1 = Burned, 0 = Unburned).
        
        Args:
            pre_nbr: Pre-fire NBR raster
            post_nbr: Post-fire NBR raster
            mode: 'usgs' (threshold=0.100), 'otsu' (adaptive optimal threshold),
                  or 'custom' (uses threshold argument).
        """
        dnbr = self.compute_dnbr(pre_nbr, post_nbr)
        
        if mode == "otsu":
            th = self.otsu_threshold(dnbr)
        elif mode == "custom":
            th = threshold if threshold is not None else self.default_threshold
        else:  # USGS low severity burn threshold
            th = self.default_threshold

        binary_mask = (dnbr >= th).astype(np.uint8)
        return binary_mask, th, dnbr


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Classical Remote Sensing Baseline: dNBR & Otsu Thresholding")
    print("=" * 70)

    baseline = ClassicalDNBRBaseline(default_threshold=0.100)

    # Simulate bitemporal Landsat 8 NIR (B5) and SWIR2 (B7) bands
    np.random.seed(42)
    H, W = 256, 256
    pre_nir = np.random.uniform(0.3, 0.5, (H, W))     # High NIR before fire
    pre_swir2 = np.random.uniform(0.05, 0.15, (H, W)) # Low SWIR2 before fire
    
    post_nir = pre_nir.copy()
    post_swir2 = pre_swir2.copy()
    
    # Simulate a burn scar in a 80x80 patch
    post_nir[80:160, 80:160] = np.random.uniform(0.08, 0.15, (80, 80))   # Chlorophyll destroyed
    post_swir2[80:160, 80:160] = np.random.uniform(0.35, 0.55, (80, 80)) # Ash/charcoal absorption

    # 1. Calculate NBR
    pre_nbr = baseline.compute_nbr_from_bands(pre_nir, pre_swir2)
    post_nbr = baseline.compute_nbr_from_bands(post_nir, post_swir2)

    # 2. Calculate dNBR
    dnbr = baseline.compute_dnbr(pre_nbr, post_nbr)
    print(f"preNBR range           : [{pre_nbr.min():.4f}, {pre_nbr.max():.4f}]")
    print(f"postNBR range          : [{post_nbr.min():.4f}, {post_nbr.max():.4f}]")
    print(f"dNBR range             : [{dnbr.min():.4f}, {dnbr.max():.4f}]")

    # 3. Categorical USGS severity classification
    severity_map = baseline.classify_severity(dnbr)
    unique_classes, counts = np.unique(severity_map, return_counts=True)
    print("\nUSGS Severity Class Distribution:")
    for cls_id, count in zip(unique_classes, counts):
        pct = (count / (H * W)) * 100.0
        print(f"  Class {cls_id} ({baseline.SEVERITY_CLASSES[cls_id]}): {count:,} pixels ({pct:.2f}%)")

    # 4. Otsu Adaptive Binary Detection
    binary_mask, otsu_th, _ = baseline.predict_binary(pre_nbr, post_nbr, mode="otsu")
    print(f"\nOtsu Optimal Threshold : {otsu_th:.4f}")
    print(f"Detected Burn Scar Area: {binary_mask.sum():,} pixels ({(binary_mask.sum() / (H * W)) * 100:.2f}%)")
    print("=" * 70)
