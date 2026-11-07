"""
================================================================================
UNIFIED BITEMPORAL CHANGE DETECTION DATASET & DATA LOADER
================================================================================
Provides a PyTorch Dataset for loading pre/post fire scenes and ground truth
burn scar masks from the extracted Final_with_India directory using tifffile.
Supports both multispectral Landsat 6-band mode (B2-B7) and single-band NBR mode.
================================================================================
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile


class WildfireChangeDetectionDataset(Dataset):
    """
    Bitemporal Wildfire Change Detection Dataset.

    Args:
        data_dir (str): Path to 'Final_with_India' directory.
        mode (str): 'multispectral' (6 Landsat bands B2-B7) or 'nbr' (single-band NBR).
        fire_filter (list): Optional list of fire stems (e.g. ['august_fire_complex', 'wallow_fire']).
                            If None, loads all available tiles.
        transform (callable): Optional data augmentation callable.
    """
    def __init__(self, data_dir="Final_with_India", mode="multispectral", fire_filter=None, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == "multispectral":
            dir_A = os.path.join(data_dir, "A")
            dir_B = os.path.join(data_dir, "B")
        else:
            dir_A = os.path.join(data_dir, "nbrA")
            dir_B = os.path.join(data_dir, "nbrB")

        dir_GT = os.path.join(data_dir, "ground_truth")

        all_A = sorted(glob.glob(os.path.join(dir_A, "*.tif")))
        self.samples = []

        for p_A in all_A:
            fname = os.path.basename(p_A)
            p_B = os.path.join(dir_B, fname)
            p_GT = os.path.join(dir_GT, fname)

            if not (os.path.exists(p_B) and os.path.exists(p_GT)):
                continue

            if fire_filter is not None:
                if not any(f_stem in fname for f_stem in fire_filter):
                    continue

            self.samples.append((p_A, p_B, p_GT))

        print(f"Loaded {len(self.samples)} tile pairs for mode='{mode}' (fire_filter={fire_filter})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_A, path_B, path_GT = self.samples[idx]

        # Read rasters
        raw_A = tifffile.imread(path_A).astype(np.float32)
        raw_B = tifffile.imread(path_B).astype(np.float32)
        raw_GT = tifffile.imread(path_GT).astype(np.float32)

        if self.mode == "multispectral":
            # Select Landsat bands B2 to B7 (indices 1 to 6)
            if raw_A.ndim == 3 and raw_A.shape[-1] >= 7:
                img_A = raw_A[:, :, 1:7]
                img_B = raw_B[:, :, 1:7]
            else:
                img_A = raw_A
                img_B = raw_B

            # Replace NaNs/Infs and normalize from [0, 0.5] reflectance to [0, 1]
            img_A = np.nan_to_num(img_A, nan=0.0, posinf=0.5, neginf=0.0)
            img_B = np.nan_to_num(img_B, nan=0.0, posinf=0.5, neginf=0.0)
            img_A = np.clip(img_A, 0.0, 0.5) / 0.5
            img_B = np.clip(img_B, 0.0, 0.5) / 0.5

            # [H, W, C] -> [C, H, W]
            img_A = np.transpose(img_A, (2, 0, 1))
            img_B = np.transpose(img_B, (2, 0, 1))

        else:  # NBR mode (single channel [1, H, W])
            img_A = np.nan_to_num(raw_A, nan=0.0, posinf=1.0, neginf=-1.0)
            img_B = np.nan_to_num(raw_B, nan=0.0, posinf=1.0, neginf=-1.0)
            if img_A.ndim == 2:
                img_A = np.expand_dims(img_A, axis=0)
                img_B = np.expand_dims(img_B, axis=0)
            else:
                img_A = np.transpose(img_A, (2, 0, 1))[:1]
                img_B = np.transpose(img_B, (2, 0, 1))[:1]

        # Process ground truth: binary mask [1, H, W]
        mask = np.nan_to_num(raw_GT, nan=0.0)
        mask = (mask > 0.5).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        elif mask.ndim == 3:
            mask = np.transpose(mask, (2, 0, 1))[:1]

        # Convert to PyTorch tensors
        tensor_A = torch.from_numpy(img_A.copy())
        tensor_B = torch.from_numpy(img_B.copy())
        tensor_GT = torch.from_numpy(mask.copy())

        return {
            "pre": tensor_B,     # Image B (pre-event)
            "post": tensor_A,    # Image A (post-event)
            "mask": tensor_GT,   # Ground truth mask
            "name": os.path.basename(path_A)
        }


def get_dataloaders(data_dir="Final_with_India", batch_size=4, mode="multispectral"):
    """
    Returns train, validation, and test DataLoaders following the research splits.
    Train: Wallow, Evans Canyon, August Complex
    Val  : Bighorn
    Test1 (Global): McKinney, Gospers Mountain
    Test2 (India) : Uttarakhand, Similipal, Bandhavgarh
    """
    train_fires = ["wallow_fire", "evans_fire", "august_fire_complex"]
    val_fires   = ["bighorn_fire"]
    test1_fires = ["mckinney_fire", "gospers_fire"]
    test2_fires = ["Uttarakhand_Fire_2021", "Similipal_Fire_2021", "Bandhavgarh_Fire"]

    train_ds = WildfireChangeDetectionDataset(data_dir, mode=mode, fire_filter=train_fires)
    val_ds   = WildfireChangeDetectionDataset(data_dir, mode=mode, fire_filter=val_fires)
    test1_ds = WildfireChangeDetectionDataset(data_dir, mode=mode, fire_filter=test1_fires)
    test2_ds = WildfireChangeDetectionDataset(data_dir, mode=mode, fire_filter=test2_fires)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test1_loader = DataLoader(test1_ds, batch_size=batch_size, shuffle=False)
    test2_loader = DataLoader(test2_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test1_loader, test2_loader


if __name__ == "__main__":
    print("Testing WildfireChangeDetectionDataset...")
    ds = WildfireChangeDetectionDataset(data_dir="Final_with_India", mode="multispectral")
    sample = ds[0]
    print(f"Sample Tile Name : {sample['name']}")
    print(f"Pre-Fire Tensor  : {sample['pre'].shape}, dtype={sample['pre'].dtype}")
    print(f"Post-Fire Tensor : {sample['post'].shape}, dtype={sample['post'].dtype}")
    print(f"Ground Truth Mask: {sample['mask'].shape}, dtype={sample['mask'].dtype}")
    print(f"Unique Mask Vals : {torch.unique(sample['mask']).tolist()}")
    print("Dataset loader verified successfully.")
