import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from model import UNetModel

# 1. Dataset Class: Connects your folder structure to the model
class EncroachmentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Ensure filenames match!
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 'L' mode for grayscale
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            
        return img, mask

# 2. Setup Device and Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. Training Loop
def train_model(model, loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "model.pth")
    print("Training finished! model.pth saved in LH/ folder.")

# 4. Main Execution
if __name__ == "__main__":
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Fixed the typo here: changed 'ataset' to 'dataset'
    dataset = EncroachmentDataset(
        image_dir=r"D:\Projects\ProjectPersistenceSentinel\LH\data\train\images",
        mask_dir=r"D:\Projects\ProjectPersistenceSentinel\LH\data\train\masks",
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Starting training...")
    train_model(model, loader, epochs=20)