"""
Multi-sensor Landsat-7 ETM+ (15-band) and Landsat-8 OLI (17-band) Collection 2
Level-2 dataset loader for the QACD wildfire burned-area mapping framework.

Features:
  - Physical wavelength band harmonization across Landsat-7 ETM+ (2011 Wallow Fire:
    RGB=[2, 1, 0]=B3/B2/B1, SWIR2/NIR/Red=[7, 3, 2]=B7/B4/B3) and Landsat-8 OLI
    (2019-2022 events: RGB=[3, 2, 1]=B4/B3/B2, SWIR2/NIR/Red=[6, 4, 3]=B7/B5/B4).
  - Reflectance clipping to [0.0, 0.5] and linear rescaling to [0, 1].
  - Valid-pixel orbital swath and ETM+ SLC-off scan-gap masking (tau_s = 0.015).
  - Strict event-level dataset partitioning across 1,486 bi-temporal 256x256 tiles
    (650 Train, 195 Val, 416 Test 1, 225 Test 2) to prevent spatial leakage.
"""

from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


SPLITS = {
    "train": ("wallow_fire", "evans_fire", "august_fire_complex"),
    "val": ("bighorn_fire",),
    "test1": ("mckinney_fire", "gospers_fire"),
    "test2": ("Uttarakhand_Fire_2021", "Similipal_Fire_2021", "Bandhavgarh_Fire"),
}

TEST_FIRES = (
    "mckinney_fire",
    "gospers_fire",
    "Bandhavgarh_Fire",
    "Uttarakhand_Fire_2021",
    "Similipal_Fire_2021",
)


def load_bitemporal_tile(
    dataset_root: Path,
    fname: str,
    spectral_mode: str = "swir_nir_red",
    tau_swath: float = 0.015,
) -> dict:
    """
    Loads one co-registered 256x256 bi-temporal GeoTIFF tile pair (B = pre-fire,
    A = post-fire) together with its 500 m MODIS MCD64A1 reference mask resampled
    to the 30 m EPSG:3857 grid and computes the swath/SLC-off validity mask M_valid.
    """
    arr_pre = tifffile.imread(dataset_root / "B" / fname).astype(np.float32)
    arr_post = tifffile.imread(dataset_root / "A" / fname).astype(np.float32)
    gt_raw = tifffile.imread(dataset_root / "ground_truth" / fname).astype(np.float32)
    gt = (np.nan_to_num(gt_raw, nan=0.0) > 0.5).astype(np.uint8)

    if arr_pre.shape[-1] == 15:
        # Landsat-7 ETM+ Collection 2 Level-2 (2011 Wallow Fire)
        rgb_idx, sw_idx = [2, 1, 0], [7, 3, 2]
    else:
        # Landsat-8 OLI Collection 2 Level-2 (2019-2022 events)
        rgb_idx, sw_idx = [3, 2, 1], [6, 4, 3]

    raw_pre_rgb, raw_post_rgb = arr_pre[..., rgb_idx], arr_post[..., rgb_idx]
    raw_pre_sw, raw_post_sw = arr_pre[..., sw_idx], arr_post[..., sw_idx]

    finite = (
        np.isfinite(raw_pre_rgb).all(axis=-1)
        & np.isfinite(raw_post_rgb).all(axis=-1)
        & np.isfinite(raw_pre_sw).all(axis=-1)
        & np.isfinite(raw_post_sw).all(axis=-1)
    )

    pre_rgb = np.clip(np.nan_to_num(raw_pre_rgb, nan=0.0, posinf=0.5, neginf=0.0), 0.0, 0.5) / 0.5
    post_rgb = np.clip(np.nan_to_num(raw_post_rgb, nan=0.0, posinf=0.5, neginf=0.0), 0.0, 0.5) / 0.5
    pre_sw = np.clip(np.nan_to_num(raw_pre_sw, nan=0.0, posinf=0.5, neginf=0.0), 0.0, 0.5) / 0.5
    post_sw = np.clip(np.nan_to_num(raw_post_sw, nan=0.0, posinf=0.5, neginf=0.0), 0.0, 0.5) / 0.5

    valid = (
        finite
        & (np.max(pre_rgb, axis=-1) > tau_swath)
        & (np.max(post_rgb, axis=-1) > tau_swath)
    ).astype(np.uint8)

    pre_img = pre_sw if spectral_mode == "swir_nir_red" else pre_rgb
    post_img = post_sw if spectral_mode == "swir_nir_red" else post_rgb

    return {
        "fname": fname,
        "pre": torch.from_numpy(pre_img.transpose(2, 0, 1).copy()).float(),
        "post": torch.from_numpy(post_img.transpose(2, 0, 1).copy()).float(),
        "gt": torch.from_numpy(gt).unsqueeze(0).float(),
        "valid": torch.from_numpy(valid).unsqueeze(0).float(),
        "has_burn": bool((gt & valid).sum() > 50),
    }


def list_split_files(dataset_root: Path, split_name: str) -> list:
    prefixes = SPLITS[split_name]
    gt_dir = dataset_root / "ground_truth"
    return sorted(
        p.name for p in gt_dir.glob("*.tif") if any(p.name.startswith(pref) for pref in prefixes)
    )


class WildfireChangeDataset(Dataset):
    """PyTorch Dataset for event-partitioned bi-temporal wildfire GeoTIFF tiles."""

    def __init__(
        self,
        dataset_root: Path,
        split: str = "test1",
        spectral_mode: str = "swir_nir_red",
        crop_size: int = None,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.spectral_mode = spectral_mode
        self.crop_size = crop_size
        self.rng = np.random.default_rng(seed)
        self.files = list_split_files(self.dataset_root, split)
        self.prefixes = SPLITS[split]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]
        item = load_bitemporal_tile(self.dataset_root, fname, spectral_mode=self.spectral_mode)
        item["fire"] = next(pref for pref in self.prefixes if fname.startswith(pref))

        if self.crop_size is not None:
            _, H, W = item["pre"].shape
            cs = self.crop_size
            r0 = int(self.rng.integers(0, max(1, H - cs + 1)))
            c0 = int(self.rng.integers(0, max(1, W - cs + 1)))
            item["pre"] = item["pre"][:, r0 : r0 + cs, c0 : c0 + cs]
            item["post"] = item["post"][:, r0 : r0 + cs, c0 : c0 + cs]
            item["gt"] = item["gt"][:, r0 : r0 + cs, c0 : c0 + cs]
            item["valid"] = item["valid"][:, r0 : r0 + cs, c0 : c0 + cs]

        return item
