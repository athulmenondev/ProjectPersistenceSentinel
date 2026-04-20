import torch
import numpy as np
import cv2
from model import UNetModel, predict_mask
from JM.temporal_engine import TemporalFilter
from GVE.mask_refiner import MaskRefiner
from ASM.video_preprocessor import VideoPreprocessor

class PersistenceSentinelPipeline:
    def __init__(self):
        # Initialize modules once to maintain state and save memory
        self.model = UNetModel()
        self.engine = TemporalFilter(width=256, height=256, k_frames_threshold=30)
        self.refiner = MaskRefiner(min_area=100, use_crf=True)

    def process_frame(self, input_tensor, rgb_frame):
        # 1. Spatial Inference
        prob_map = predict_mask(self.model, input_tensor)
        
        # 2. Precision Refinement (Snapping to edges)
        refined_mask = self.refiner.refine(prob_map, rgb_frame=rgb_frame)
        
        # 3. Temporal Verification
        final_mask = self.engine.process_frame(refined_mask)
        
        return final_mask

def run_on_video(video_path):
    # Initialize Pipeline
    pipeline = PersistenceSentinelPipeline()
    preprocessor = VideoPreprocessor(video_path, target_size=(256, 256))
    
    print("Pipeline started. Processing video...")
    
    for frame_tensor, original_rgb in preprocessor:
        # Run through the integrated pipeline
        output = pipeline.process_frame(frame_tensor, original_rgb)
        
        # Display the result
        cv2.imshow("Persistent Encroachment Detection", (output * 255).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    # To run this, replace 'test_video.mp4' with your actual file path
    # run_on_video("test_video.mp4")
    print("Pipeline initialized. Ready to process video input.")