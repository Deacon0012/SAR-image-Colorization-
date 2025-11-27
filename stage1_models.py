# stage1_models.py (Update GeneratorResNet)

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape=(1, 256, 256), num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64
        
        # Initial Convolution Block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual Blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # --- FIX: Upsampling with Resize-Convolution ---
        # Replaced ConvTranspose2d with Upsample + Conv2d
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # Smoother than PixelShuffle
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        # Output Layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, 1, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)