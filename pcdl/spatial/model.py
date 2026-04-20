"""
Spatial Segmentation Models

U-Net architecture for pixel-wise change detection.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    """
    U-Net for binary segmentation.

    Input: (B, 3, H, W) RGB frames
    Output: (B, 1, H, W) probability maps
    """
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.up1(self.dec2(d2))
        d1 = torch.cat([d1, e1], dim=1)

        out = self.final(self.dec1(d1))
        return out # Returns logits for better training stability


# --------- Utility Functions ---------

def dice_loss(pred, target):
    """Dice loss for segmentation training (expects logits)."""
    smooth = 1e-6
    pred = torch.sigmoid(pred) # Convert logits to probabilities
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def predict_mask(model, input_tensor):
    """Run inference on a model and return probability map."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        return torch.sigmoid(output)   # Apply sigmoid here for probability map inference


def get_binary_mask(pred, threshold=0.5):
    """Convert probability map to binary mask."""
    return (pred > threshold).float()
