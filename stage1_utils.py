# stage1_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

# --- 1. SSIM Helper Functions (THEY ARE USED HERE) ---
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# --- 2. The REAL SSIM Loss Class ---
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # This initializes the gaussian window
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # --- CRITICAL FIX: Shift inputs from [-1, 1] to [0, 1] ---
        # SSIM only works correctly on positive values!
        img1 = (img1 + 1) / 2.0
        img2 = (img2 + 1) / 2.0
        
        # Load window to correct device
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)
        
        ssim_value = _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)
        return 1.0 - ssim_value

# --- 3. Evaluation Metrics ---
def evaluate_metrics(pred_batch, target_batch):
    """Calculates PSNR, SSIM, and ENL for a batch."""
    preds = (pred_batch.detach().cpu().numpy() + 1) / 2.0 * 255.0 
    targets = (target_batch.detach().cpu().numpy() + 1) / 2.0 * 255.0
    
    psnr_val = 0.0
    ssim_val = 0.0
    enl_val = 0.0
    batch_size = preds.shape[0]

    for i in range(batch_size):
        p = preds[i, 0].astype(np.uint8)
        t = targets[i, 0].astype(np.uint8)
        
        psnr_val += psnr(t, p, data_range=255)
        ssim_val += ssim(t, p, data_range=255)
        
        mean = np.mean(p)
        std = np.std(p)
        if std > 0:
            enl_val += (mean ** 2) / (std ** 2)

    return psnr_val, ssim_val, enl_val, batch_size