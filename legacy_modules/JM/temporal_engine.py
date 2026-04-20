import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_mock_frame(frame_idx, width=100, height=100):
    """
    Simulates a probability mask from Module 2.
    - A moving square representing a car/person (transient).
    - A stationary square representing a dropped bag (persistent, appears at frame 20).
    """
    frame = np.zeros((height, width))
    
    # 1. Transient Object (e.g., car moving across the screen)
    # It moves 2 pixels right and 1 pixel down every frame
    car_x = (frame_idx * 2) % width
    car_y = (10 + frame_idx) % height
    
    # Draw the 10x10 car (handling boundaries safely)
    x_end = min(car_x + 10, width)
    y_end = min(car_y + 10, height)
    frame[car_y:y_end, car_x:x_end] = 1.0

    # 2. Persistent Object (e.g., dropped luggage)
    # It drops at frame 20 and stays there forever
    if frame_idx >= 20:
        box_x, box_y = 60, 40
        frame[box_y:box_y+8, box_x:box_x+8] = 1.0
        
    return frame

class TemporalFilter:
    def __init__(self, width, height, k_frames_threshold):
        """
        Initializes the Temporal Engine.
        :param width: Width of the input frame.
        :param height: Height of the input frame.
        :param k_frames_threshold: Number of consecutive frames a pixel must be active 
                                   to be considered an encroachment.
        """
        self.width = width
        self.height = height
        self.k_frames = k_frames_threshold
        
        # Matrix to keep track of how many consecutive frames a pixel has been active
        self.persistence_matrix = np.zeros((height, width), dtype=np.int32)

    def process_frame(self, current_mask):
        """
        Takes the current frame mask (0s and 1s) and updates the persistence tracking.
        """
        # Increment counter where mask is 1 (or > 0.5 probability), reset to 0 where mask is 0
        self.persistence_matrix = np.where(current_mask > 0.5, self.persistence_matrix + 1, 0)
        
        # Create the final output mask: 1 only where persistence >= threshold
        persistent_mask = (self.persistence_matrix >= self.k_frames).astype(np.float32)
        
        return persistent_mask

def main():
    width, height = 100, 100
    total_frames = 100
    
    # Let's say K = 30 frames. 
    # (If the video is 30 FPS, this equals 1 second of persistence).
    k_frames_threshold = 30 
    
    print(f"Initializing Temporal Filter. K_frames threshold = {k_frames_threshold}")
    engine = TemporalFilter(width, height, k_frames_threshold)
    
    # Store frames for visualization
    input_frames = []
    output_frames = []
    
    print("Simulating frames...")
    for i in range(total_frames):
        # 1. Get mock data (simulating receiving data from Module 2)
        mock_mask = generate_mock_frame(i, width, height)
        input_frames.append(mock_mask)
        
        # 2. Pass through Module 3 temporal filter
        filtered_mask = engine.process_frame(mock_mask)
        output_frames.append(filtered_mask)
        
        if (i + 1) % 20 == 0:
            print(f"Processed frame {i + 1}/{total_frames}")

    # Visualization (Uses matplotlib to create an animation)
    print("Generating visualization. Close the plot window to exit.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(input_frames[0], cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Input (Module 2 Mock)")
    
    im2 = ax2.imshow(output_frames[0], cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Output (Module 3 Filtered)")
    
    def update(frame_idx):
        im1.set_data(input_frames[frame_idx])
        im2.set_data(output_frames[frame_idx])
        fig.suptitle(f"Frame: {frame_idx} | Persistent item appears at frame 20\nOutput should show item AFTER frame {20 + k_frames_threshold}")
        return im1, im2
        
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
