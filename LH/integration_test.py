import torch
import numpy as np
import matplotlib.pyplot as plt # Added for visualization
from model import UNetModel, predict_mask
from JM.temporal_engine import TemporalFilter

def test_pipeline_integration():
    # 1. Setup
    width, height = 256, 256
    k_frames = 10
    model = UNetModel()
    engine = TemporalFilter(width, height, k_frames)
    
    # 2. Create dummy input with a "white square" in the center
    dummy_input = torch.zeros(1, 3, height, width)
    dummy_input[:, :, 100:150, 100:150] = 1.0 
    
    # 3. Get output from your model
    model_output = predict_mask(model, dummy_input)
    print(f"Model output shape: {model_output.shape}") 
    
    # 4. Pass to Jyotsna's engine
    filtered = engine.process_frame(model_output)
    print(f"Engine output shape: {filtered.shape}")
    
    print("Integration successful with object detection test!")
    
    # Optional: Visualize the result to see the square!
    # .squeeze() removes all dimensions of size 1, leaving you with (256, 256)
    plt.imshow(model_output.squeeze(), cmap='gray')
    plt.title("Detected Object")
    plt.show()

if __name__ == "__main__":
    test_pipeline_integration()