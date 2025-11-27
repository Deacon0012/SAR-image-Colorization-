# stage1_dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class SARToLDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size
        self.sar_dir = os.path.join(root_dir, 'Sar')
        self.optical_dir = os.path.join(root_dir, 'Optical')
        
        # Filter only valid image files to avoid errors
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.sar_files = sorted([f for f in os.listdir(self.sar_dir) if os.path.splitext(f)[1].lower() in valid_exts])
        self.optical_files = sorted([f for f in os.listdir(self.optical_dir) if os.path.splitext(f)[1].lower() in valid_exts])

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        # Load SAR
        sar_path = os.path.join(self.sar_dir, self.sar_files[idx])
        sar_img = np.array(Image.open(sar_path).convert("L"))
        
        # Load Optical and Convert to Grayscale (L approximation)
        optical_path = os.path.join(self.optical_dir, self.optical_files[idx])
        optical_img = Image.open(optical_path).convert("L") 
        optical_img = np.array(optical_img)

        # Transform to Tensor and Normalize [-1, 1]
        sar_tn = (T.ToTensor()(sar_img) * 2) - 1
        optical_tn = (T.ToTensor()(optical_img) * 2) - 1

        return sar_tn, optical_tn