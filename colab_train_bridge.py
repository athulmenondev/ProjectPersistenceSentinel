"""
Colab Training Bridge

This script is a wrapper for train_pipeline.py designed for Google Colab.
It automatically handles path mapping if your data is stored in Google Drive.
"""

import os
import sys

def main():
    # Attempt to find the project root regardless of where it's cloned in Colab
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    print("="*60)
    print("PSCDL 2026 - Google Colab Training Bridge")
    print("="*60)
    
    # Instructions for the user
    print("\n1. Ensure your 'data' folder is uploaded to Google Drive.")
    print("2. Mount Drive in Colab using: from google.colab import drive; drive.mount('/content/drive')")
    print("3. Run the following command in a Colab Cell:\n")
    
    # Generate the suggested command
    drive_path = "/content/drive/MyDrive/ProjectPersistenceSentinel/data/dataset/train"
    cmd = f"!python train_pipeline.py --data {drive_path} --epochs 50 --fps 2"
    
    print(cmd)
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
