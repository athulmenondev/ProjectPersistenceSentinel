import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Add project root to path so modules are found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import UNetModel, predict_mask
from GVE.mask_refiner import MaskRefiner

# 1. Initialize model (Using random weights for now)
model = UNetModel()
model.load_state_dict(torch.load(r"D:\Projects\ProjectPersistenceSentinel\LH\model.pth", map_location=torch.device('cpu')))
model.eval()

# 2. Initialize the Refiner
# Threshold 0.2 makes it sensitive enough to see 'random' patterns
refiner = MaskRefiner(min_area=5, threshold=0.05, use_crf=True)

# 3. Load your test image
img_path = r"D:\Projects\ProjectPersistenceSentinel\test_sample.jpg"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found. Please run create_test_image.py first.")
    sys.exit()

img = Image.open(img_path).convert('RGB')

# 4. Transform for model
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
x = transform(img).unsqueeze(0)

# 5. Get raw mask
mask = predict_mask(model, x)

# 6. Refine mask
mask_np_input = mask[0][0].detach().cpu().numpy()
image_np = x[0].permute(1, 2, 0).detach().cpu().numpy()
rgb_uint8 = (image_np * 255).astype(np.uint8)

# Run the pipeline
binary_np = refiner.refine(mask_np_input, rgb_frame=rgb_uint8)

# 7. Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(image_np)

plt.subplot(1, 3, 2)
plt.title("Raw Mask (Random)")
plt.imshow(mask_np_input, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Binary Mask (Refined)")
plt.imshow(binary_np, cmap='gray')

plt.tight_layout()
plt.show()