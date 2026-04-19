import torch
import torch.nn as nn


class DoubleConv(nn.Module):
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
        return torch.sigmoid(out)


# --------- Utility Functions ---------

def dice_loss(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def predict_mask(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        return output   # ✅ keep as PyTorch tensor

def get_binary_mask(pred, threshold=0.5):
    return (pred > threshold).float()   # keep PyTorc


# --------- Run only when executed directly ---------

if __name__ == "__main__":
    model = UNetModel()

    # test
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # dummy training
    for i in range(3):
        x = torch.randn(1, 3, 256, 256)
        target = torch.randint(0, 2, (1, 1, 256, 256)).float()

        output = model(x)
        loss = dice_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {i}, Loss: {loss.item()}")

    # save model
    torch.save(model.state_dict(), "model.pth")