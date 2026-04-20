import os
from PIL import Image

image_dir = r"D:\Projects\ProjectPersistenceSentinel\LH\data\train\images"
mask_dir = r"D:\Projects\ProjectPersistenceSentinel\LH\data\train\masks"

os.makedirs(mask_dir, exist_ok=True)

# Using .lower() ensures we catch all .png files regardless of capitalization
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".png"):
        mask_path = os.path.join(mask_dir, filename)
        
        # Only create the mask if it doesn't already exist
        if not os.path.exists(mask_path):
            mask = Image.new('L', (256, 256), color=0)
            mask.save(mask_path)
            print(f"Created missing mask for: {filename}")

print("Sync complete. All images should now have a matching mask.")