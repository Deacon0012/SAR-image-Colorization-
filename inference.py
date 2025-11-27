import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# --- Import your architectures ---
from stage1_models import GeneratorResNet
from stage2_models import GeneratorUNet
from stage2_utils import lab_to_rgb

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

# Paths to your BEST saved models
STAGE1_CHECKPOINT = "./checkpoint_structure/last.pth" # Or 'best.pth'
STAGE2_CHECKPOINT = "./checkpoint_color/best.pth"     # Or 'best.pth'

# Path to a test SAR image
TEST_IMAGE_PATH = r"./dataset/sar/ROIs1868_summer_s1_59_p11.png" 

def load_models():
    print("Loading models...")
    
    # 1. Load Stage 1 (Structure)
    model_s1 = GeneratorResNet().to(DEVICE)
    ckpt1 = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE, weights_only=False)
    # Handle dictionary format vs direct save format
    if isinstance(ckpt1, dict) and 'generator_state_dict' in ckpt1:
        model_s1.load_state_dict(ckpt1['generator_state_dict'])
    else:
        model_s1.load_state_dict(ckpt1)
    model_s1.eval()
    
    # 2. Load Stage 2 (Color)
    model_s2 = GeneratorUNet().to(DEVICE)
    ckpt2 = torch.load(STAGE2_CHECKPOINT, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt2, dict) and 'G' in ckpt2:
        model_s2.load_state_dict(ckpt2['G'])
    else:
        model_s2.load_state_dict(ckpt2)
    model_s2.eval()
    
    print("Models loaded successfully!")
    return model_s1, model_s2

def process_sar_to_rgb(sar_path, model_s1, model_s2):
    # 1. Load and Preprocess SAR
    img = Image.open(sar_path).convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Transform: ToTensor (0-1) -> Normalize (-1 to 1)
    sar_tensor = T.ToTensor()(img)
    sar_tensor = (sar_tensor * 2.0) - 1.0
    sar_tensor = sar_tensor.unsqueeze(0).to(DEVICE) # Add batch dim
    
    with torch.no_grad():
        # 2. Run Stage 1 (SAR -> Clean L)
        fake_L = model_s1(sar_tensor)
        
        # 3. Run Stage 2 (Clean L -> Color ab)
        # Note: fake_L is already normalized [-1, 1], exactly what Stage 2 expects
        fake_ab = model_s2(fake_L)
        L_unscaled = (fake_L + 1.0) * 50.0
        ab_unscaled = fake_ab * 110.0
        saturation_factor = 2.0  
        ab_unscaled = ab_unscaled * saturation_factor
        L_np = L_unscaled.squeeze().cpu().numpy()
        ab_np = ab_unscaled.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        lab_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        lab_image[:, :, 0] = L_np
        lab_image[:, :, 1:] = ab_np
        
        from skimage import color
        rgb_result = color.lab2rgb(lab_image)
        # Convert to 0-255 uint8 for saving
        rgb_result = (rgb_result * 255).astype(np.uint8)
        
    return rgb_result, fake_L

def main():
    # Load
    s1, s2 = load_models()
    
    # Run
    final_rgb, intermediate_L = process_sar_to_rgb(TEST_IMAGE_PATH, s1, s2)
    
    # Visualize the Full Pipeline
    sar_original = Image.open(TEST_IMAGE_PATH).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert intermediate L tensor back to numpy for display
    L_disp = (intermediate_L.squeeze().cpu().numpy() + 1) * 50.0
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(sar_original, cmap='gray')
    ax[0].set_title("1. Input Raw SAR")
    ax[0].axis('off')
    
    ax[1].imshow(L_disp, cmap='gray')
    ax[1].set_title("2. Stage 1 Output (Structure)")
    ax[1].axis('off')
    
    ax[2].imshow(final_rgb)
    ax[2].set_title("3. Stage 2 Output (Colorized)")
    ax[2].axis('off')
    
    plt.show()
    
    # Optional: Save
    Image.fromarray(final_rgb).save("final_result.png")
    print("Saved final_result.png")

if __name__ == "__main__":
    main()
