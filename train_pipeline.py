"""
Integrated Training Pipeline for PSCDL 2026.

This script demonstrates the full team integration:
1. Video Preprocessing (Athul)
2. U-Net Training (Lekshmi)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse

from pcdl.io import PSCDLDataset
from pcdl.spatial import UNetModel, dice_loss

def main():
    parser = argparse.ArgumentParser(description="Integrated PSCDL Training Pipeline")
    parser.add_argument("--data", type=str, default="data/dataset", help="Path to real dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--fps", type=int, default=2, help="Frames to extract per second of video")
    parser.add_argument("--output", type=str, default="model.pth", help="Path to save trained model")
    
    args = parser.parse_args()

    # 1. Initialize Integrated Dataset (Athul's Preprocessor + Metadata)
    print(f"Indexing dataset at {args.data}...")
    dataset = PSCDLDataset(
        root_dir=args.data,
        output_size=(256, 256),
        fps=args.fps
    )
    
    if len(dataset) == 0:
        print("Error: No training samples found. Check your data paths and .txt files.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # Check for balance
    print(f"Total training samples: {len(dataset)}")
    pos_samples = sum(1 for s in dataset.samples if 'mask1.png' not in s['mask_path']) 
    print(f"Samples with potential objects: {pos_samples} ({pos_samples/len(dataset)*100:.1f}%)")

    # 2. Initialize Model (Lekshmi's Model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetModel().to(device)
    
    # Loss: Weighted BCE to handle data imbalance (90% background / 10% object)
    # We weight positive pixels 20x more than negative pixels
    pos_weight = torch.tensor([20.0]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    print(f"Starting training on {device} for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_bce = 0
        epoch_dice = 0
        
        for i, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            
            # Combine losses
            loss_bce = bce_loss(predictions, masks)
            loss_dice = dice_loss(predictions, masks)
            total_loss = loss_bce + loss_dice
            
            total_loss.backward()
            optimizer.step()
            
            epoch_bce += loss_bce.item()
            epoch_dice += loss_dice.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(loader)}], Loss: {total_loss.item():.4f}")
        
        avg_loss = (epoch_bce + epoch_dice) / (2 * len(loader))
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} ---")

    # 4. Save the Result
    torch.save(model.state_dict(), args.output)
    print(f"Integrated model saved to {args.output}")

if __name__ == "__main__":
    main()
