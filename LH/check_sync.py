import os
from PIL import Image

img_dir = r"D:\Projects\ProjectPersistenceSentinel\LH\data\train\images"
mask_dir = r"D:\Projects\ProjectPersistenceSentinel\LH\data\train\masks"

for filename in os.listdir(img_dir):
    if filename.lower().endswith(".png"):
        img = Image.open(os.path.join(img_dir, filename))
        mask = Image.open(os.path.join(mask_dir, filename))
        
        if img.size != mask.size:
            print(f"CRITICAL ERROR: {filename} dimensions don't match!")
            print(f"Image: {img.size}, Mask: {mask.size}")
        else:
            print(f"Verified: {filename} matches.")