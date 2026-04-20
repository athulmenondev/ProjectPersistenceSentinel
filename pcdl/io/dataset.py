"""
PSCDL Dataset Module

Parses competition metadata and connects the VideoPreprocessor (Athul) 
to the Training Pipeline (Lekshmi).
"""

import os
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from pcdl.io.video_preprocessor import VideoPreprocessor


class PSCDLDataset(Dataset):
    """
    Unified Dataset that uses Athul's Preprocessor to feed Lekshmi's Model.
    """
    def __init__(
        self, 
        root_dir: str, 
        output_size: Tuple[int, int] = (256, 256),
        fps: int = 5,
        transform=None
    ):
        self.root_dir = root_dir
        self.output_size = output_size
        self.fps = fps
        self.transform = transform
        
        self.samples = [] # List of (video_path, frame_idx, mask_path)
        
        # Initialize Preprocessor
        self.preprocessor = VideoPreprocessor(
            input_dir=root_dir,
            output_size=output_size,
            fps=fps,
            backend="numpy"
        )
        
        self._build_index()

    def _build_index(self):
        """Parse all video folders and match frames to masks based on .txt files."""
        video_folders = sorted([
            d for d in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith("video_")
        ])
        
        for folder in video_folders:
            folder_path = os.path.join(self.root_dir, folder)
            txt_path = os.path.join(folder_path, f"{folder}.txt")
            video_path = os.path.join(folder_path, f"{folder}.mp4")
            
            if not os.path.exists(txt_path) or not os.path.exists(video_path):
                continue
                
            # Parse intervals from txt
            intervals = self._parse_intervals(txt_path)
            
            # Use OpenCV to get video info
            cap = cv2.VideoCapture(video_path)
            v_fps = cap.get(cv2.CAP_PROP_FPS)
            v_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = v_count / v_fps
            cap.release()
            
            # For each frame we would extract at our target FPS
            num_target_frames = int(duration * self.fps)
            for i in range(num_target_frames):
                timestamp = i / self.fps
                
                # Find which mask applies to this timestamp
                mask_file = None
                for mask_name, start, end in intervals:
                    if start <= timestamp <= end:
                        mask_file = mask_name
                        break
                
                if mask_file:
                    mask_path = os.path.join(folder_path, mask_file)
                    if os.path.exists(mask_path):
                        self.samples.append({
                            'video_path': video_path,
                            'timestamp': timestamp,
                            'mask_path': mask_path
                        })

    def _parse_intervals(self, txt_path: str) -> List[Tuple[str, float, float]]:
        """Parses lines like 'mask1.png: 0s to 87s' or 'mask1.png: 0s - 87s'."""
        intervals = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Match: mask name, start time, end time
            # Format: mask1.png: 0s to 87s OR mask1.png: 0s - 87s OR mask1.png: 0s – 87s
            match = re.search(r"(mask\d+\.png):\s*(?:Interval\s*→\s*)?(\d+)s\s*(?:to|[–-])\s*(\d+)s", line, re.IGNORECASE)
            if match:
                mask_name, start, end = match.groups()
                intervals.append((mask_name, float(start), float(end)))
                
        return intervals

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Athul's Preprocessor Logic
        # (We manually extract here to avoid reloading the whole video in __getitem__)
        cap = cv2.VideoCapture(sample['video_path'])
        cap.set(cv2.CAP_PROP_POS_MSEC, sample['timestamp'] * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return zero tensors on failure
            return torch.zeros((3, *self.output_size)), torch.zeros((1, *self.output_size))
            
        # Process frame using Athul's internal logic
        frame_proc = self.preprocessor.preprocess_frame(frame)
        
        # Load and process mask
        mask_raw = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask_raw, self.output_size, interpolation=cv2.INTER_NEAREST)
        mask_proc = (mask_resized > 127).astype(np.float32)
        
        # Convert to Tensors (Lekshmi's requirement)
        frame_tensor = torch.from_numpy(frame_proc).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_proc).unsqueeze(0).float()
        
        return frame_tensor, mask_tensor
