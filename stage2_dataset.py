import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from skimage.color import rgb2lab
from tqdm import tqdm

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, image_size=256, cache_memory=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.optical_dir = os.path.join(root_dir, 'Optical')
        
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.files = sorted([f for f in os.listdir(self.optical_dir) if os.path.splitext(f)[1].lower() in valid_exts])

        self.cache_memory = cache_memory
        self.data_cache = []

        if self.cache_memory:
            print(f"Stage 2: Loading {len(self.files)} Optical images into RAM...")
            for f in tqdm(self.files):
                img_path = os.path.join(self.optical_dir, f)
                img = np.array(Image.open(img_path).convert("RGB"))
                self.data_cache.append(img)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_memory:
            rgb_img = self.data_cache[idx]
        else:
            img_path = os.path.join(self.optical_dir, self.files[idx])
            rgb_img = np.array(Image.open(img_path).convert("RGB"))

        # Convert RGB -> Lab
        # rgb2lab returns float64 in ranges: L [0, 100], a [-128, 127], b [-128, 127]
        lab_img = rgb2lab(rgb_img).astype(np.float32)

        # Convert to Tensor (H, W, C) -> (C, H, W)
        lab_tensor = T.ToTensor()(lab_img)

        # Split into L and ab
        # L is channel 0, ab are channels 1 and 2
        L = lab_tensor[[0], :, :] 
        ab = lab_tensor[[1, 2], :, :]

        # Normalize to [-1, 1] for GAN stability
        L_norm = (L / 50.0) - 1.0     # 0..100 -> -1..1
        ab_norm = (ab / 128.0)        # -128..128 -> -1..1

        return L_norm, ab_norm
    