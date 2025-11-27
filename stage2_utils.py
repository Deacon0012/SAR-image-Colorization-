import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def lab_to_rgb(L_norm, ab_norm):
    """
    Converts a batch of normalized Lab tensors to a batch of RGB numpy images (uint8).
    L_norm: (N, 1, H, W) in [-1, 1]
    ab_norm: (N, 2, H, W) in [-1, 1]
    """
    # Move to CPU
    L = L_norm.detach().cpu()
    ab = ab_norm.detach().cpu()
    
    # Denormalize
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    
    # Combine
    Lab = torch.cat([L, ab], dim=1) # (N, 3, H, W)
    Lab = Lab.permute(0, 2, 3, 1).numpy() # (N, H, W, 3)
    
    rgb_imgs = []
    for img in Lab:
        # Handle potential out-of-bounds due to GAN noise
        img = np.clip(img, -128, 128) 
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 100)
        
        try:
            # skimage handles the conversion math
            rgb = lab2rgb(img.astype(np.float64)) 
            rgb = (rgb * 255).astype(np.uint8)
            rgb_imgs.append(rgb)
        except Exception as e:
            print(f"Color conversion error: {e}")
            rgb_imgs.append(np.zeros_like(img, dtype=np.uint8))
            
    return rgb_imgs

def plot_stage2_val(L_input, ab_pred, ab_real, epoch, save_dir):
    """Saves a comparison plot."""
    
    # Convert tensors to RGB images
    pred_rgb = lab_to_rgb(L_input, ab_pred)[0]
    real_rgb = lab_to_rgb(L_input, ab_real)[0]
    
    # L channel for display (grayscale)
    L_disp = (L_input[0].squeeze().cpu().numpy() + 1) * 50.0
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(L_disp, cmap='gray'); ax[0].set_title("Input L (Grayscale)")
    ax[1].imshow(real_rgb); ax[1].set_title("Real Color")
    ax[2].imshow(pred_rgb); ax[2].set_title("Predicted Color")
    
    for a in ax: a.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/val_epoch_{epoch+1}.png")
    plt.close()

def evaluate_metrics(generator, val_loader, device, lpips_loss_fn=None):
    """Calculates average PSNR, SSIM, and LPIPS for the validation set."""
    generator.eval()
    avg_psnr, avg_ssim, avg_lpips = 0.0, 0.0, 0.0
    total = 0
    
    with torch.no_grad():
        for L, ab_real in val_loader:
            L = L.to(device)
            ab_real = ab_real.to(device)
            
            # Predict
            ab_fake = generator(L)
            
            # 1. Calculate LPIPS (Perceptual) using the Loss function we already have
            # Note: LPIPS expects tensors, so we do it before converting to numpy
            if lpips_loss_fn is not None:
                # We use the 'forward' of the loss class which expects (L, ab, L, ab)
                # It returns a single scalar loss value for the batch
                batch_lpips = lpips_loss_fn(L, ab_fake, L, ab_real).item()
                avg_lpips += batch_lpips * L.size(0) # weighted sum
            
            # 2. Convert to Numpy RGB for PSNR/SSIM
            # lab_to_rgb returns a list of uint8 numpy images [H, W, 3]
            fake_imgs = lab_to_rgb(L, ab_fake)
            real_imgs = lab_to_rgb(L, ab_real)
            
            for i in range(len(fake_imgs)):
                p_img = fake_imgs[i]
                t_img = real_imgs[i]
                
                avg_psnr += psnr(t_img, p_img, data_range=255)
                avg_ssim += ssim(t_img, p_img, channel_axis=-1, data_range=255)
                total += 1
                
    return avg_psnr / total, avg_ssim / total, avg_lpips / total
    