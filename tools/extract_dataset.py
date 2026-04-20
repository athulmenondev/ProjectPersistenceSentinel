"""
Dataset Extractor for PSCDL 2026

Pre-extracts frames from videos to speed up training.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pcdl.io import PSCDLDataset

def main():
    source_data = "data/dataset/train"
    output_root = "data/temp/preprocessed_train"
    
    os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "masks"), exist_ok=True)
    
    # Initialize the integrated dataset to get the metadata
    dataset = PSCDLDataset(root_dir=source_data, fps=2)
    
    print(f"Extracting {len(dataset)} frames to {output_root}...")
    
    # Track the last video to avoid reopening
    current_video = None
    cap = None
    
    for i in tqdm(range(len(dataset))):
        sample = dataset.samples[i]
        
        # Optimize: Only open video if it changed
        if current_video != sample['video_path']:
            if cap: cap.release()
            current_video = sample['video_path']
            cap = cv2.VideoCapture(current_video)
            
        cap.set(cv2.CAP_PROP_POS_MSEC, sample['timestamp'] * 1000)
        ret, frame = cap.read()
        
        if ret:
            # Save Image
            img_name = f"frame_{i:05d}.jpg"
            cv2.imwrite(os.path.join(output_root, "images", img_name), frame)
            
            # Save Mask
            mask_raw = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask_name = f"frame_{i:05d}.png"
            cv2.imwrite(os.path.join(output_root, "masks", mask_name), mask_raw)
            
    if cap: cap.release()
    print("\nExtraction complete! You can now train significantly faster.")

if __name__ == "__main__":
    main()
