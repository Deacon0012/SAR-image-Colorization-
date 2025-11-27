import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from PIL import Image

def plot_validation_results(sar_l, gt_rgb, pred_rgb, epoch, save_dir="validation_images"):
    """Plots and saves a grid of validation images."""
    os.makedirs(save_dir, exist_ok=True)
    
    sar_l = sar_l.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sar_l, cmap='gray')
    axes[0].set_title("Input SAR (L Channel)")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth RGB")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Predicted RGB")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1:03d}.png"))
    plt.close()


def lab_to_rgb(L, ab):
    """
    Takes a batch of L and ab channels and converts them to RGB images.
    L: (N, 1, H, W) Tensor, range [-1, 1]
    ab: (N, 2, H, W) Tensor, range [-1, 1]
    Returns: List of RGB images (np.uint8)
    """
    L = L.cpu()
    ab = ab.cpu()
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).detach().numpy()
    L = (L + 1) / 2.0 * 100.0  # Denormalize L from [-1, 1] to [0, 100]
    ab = ab * 128.0  # Denormalize ab from [-1, 1] to [-128, 128]
    
    # Detach the tensor from the computation graph before converting to numpy
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()
    
    rgb_images = []
    for img in Lab:
        rgb_img = lab2rgb(img)
        rgb_images.append((rgb_img * 255).astype(np.uint8))
        
    return rgb_images


# %% DATASET AND DATALOADER
# Inspired by damn.ipynb
# =======================================================================================
class SentinelDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size

        self.sar_dir = os.path.join(self.root_dir, 'Sar')
        self.optical_dir = os.path.join(self.root_dir, 'Optical')

        # Get all SAR and Optical files
        sar_files = sorted(os.listdir(self.sar_dir))
        optical_files = sorted(os.listdir(self.optical_dir))

        # Build a mapping from base name (without s1/s2) to file name
        def base_name(fname):
            return fname.replace('_s1_', '_').replace('_s2_', '_')

        sar_map = {base_name(f): f for f in sar_files}
        optical_map = {base_name(f): f for f in optical_files}

        # Only keep pairs that exist in both
        self.pair_keys = sorted(set(sar_map.keys()) & set(optical_map.keys()))
        self.sar_files = [sar_map[k] for k in self.pair_keys]
        self.optical_files = [optical_map[k] for k in self.pair_keys]

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_path = os.path.join(self.sar_dir, self.sar_files[idx])
        optical_path = os.path.join(self.optical_dir, self.optical_files[idx])

        # Open images
        sar_img = np.array(Image.open(sar_path).convert("L"))
        optical_img = np.array(Image.open(optical_path).convert("RGB"))

        # --- Augmentations ---
        # 1. Random Crop
        h, w = sar_img.shape
        th, tw = self.image_size, self.image_size
        if w > tw or h > th:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            sar_img = sar_img[y1:y1+th, x1:x1+tw]
            optical_img = optical_img[y1:y1+th, x1:x1+tw, :]
        
        # 2. Random Flip
        if random.random() > 0.5:
            sar_img = np.fliplr(sar_img)
            optical_img = np.fliplr(optical_img)

        # --- Transformations to Tensors and Lab Color Space ---
        # SAR Image (becomes our L channel)
        sar_tensor = T.ToTensor()(sar_img.copy())
        
        # Optical Image to Lab
        optical_lab = rgb2lab(optical_img).astype(np.float32)
        optical_lab_t = T.ToTensor()(optical_lab)
        
        # L channel is from SAR, ab channels are from optical
        L = sar_tensor
        ab = optical_lab_t[[1, 2], :, :]
        
        # --- Normalization ---
        # Normalize L channel to [-1, 1] as input for generator
        # Normalize ab channels to [-1, 1] as target for generator
        L_norm = (L * 2) - 1
        ab_norm = (ab / 128.0)
        
        return L_norm, ab_norm, L # Return original L for reconstruction
