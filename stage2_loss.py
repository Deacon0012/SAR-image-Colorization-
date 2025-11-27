import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# --- 1. Differentiable Lab to RGB Converter ---
# Essential for calculating Perceptual Loss on Lab images
class LabToRGB(nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for conversion
        self.register_buffer("xn", torch.tensor(95.047))
        self.register_buffer("yn", torch.tensor(100.000))
        self.register_buffer("zn", torch.tensor(108.883))

    def forward(self, L_norm, ab_norm):
        # 1. Denormalize (assuming inputs are -1 to 1)
        L = (L_norm + 1.0) * 50.0
        a = ab_norm[:, [0], :, :] * 128.0
        b = ab_norm[:, [1], :, :] * 128.0

        # 2. Lab -> XYZ
        fy = (L + 16.0) / 116.0
        fx = (a / 500.0) + fy
        fz = fy - (b / 200.0)

        def f_inv(t):
            return torch.where(t > 0.2068966, t ** 3, (t - 16.0/116.0) / 7.787)

        X = self.xn * f_inv(fx)
        Y = self.yn * f_inv(fy)
        Z = self.zn * f_inv(fz)

        # 3. XYZ -> RGB
        # Standard linear sRGB conversion matrix
        R =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
        G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
        B =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

        # 4. Clamp and Gamma Correct
        RGB = torch.cat([R, G, B], dim=1)
        RGB = torch.clamp(RGB / 255.0, 0.0, 1.0) # Scale to [0, 1]
        
        # Simple gamma approximation (optional but helps visual match)
        # RGB = torch.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * torch.pow(RGB, 1.0/2.4) - 0.055)
        
        return RGB

# --- 2. Perceptual Loss (VGG-Based) ---
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Use VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg = vgg.eval().to(device)
        
        # Extract features from specific layers (relu1_2, relu2_2, etc.)
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_3', '26': 'relu4_3'}
        
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.lab2rgb = LabToRGB().to(device)
        self.l1 = nn.L1Loss()
        
        # Normalization for VGG
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, L_fake, ab_fake, L_real, ab_real):
        # Convert Lab -> RGB so VGG can "see" the colors
        rgb_fake = self.lab2rgb(L_fake, ab_fake)
        rgb_real = self.lab2rgb(L_real, ab_real)

        # Normalize for VGG
        rgb_fake = (rgb_fake - self.mean) / self.std
        rgb_real = (rgb_real - self.mean) / self.std

        loss = 0.0
        x = rgb_fake
        y = rgb_real
        
        for name, layer in self.vgg.named_children():
            x = layer(x)
            y = layer(y)
            if name in self.layers:
                loss += self.l1(x, y)
                
        return loss