import cv2
from warp import warp
import time


def process_battlefield(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (600, 600))  # Output size is 600x600

    # Iterate through the video frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Apply the warp function to the frame
        warped_frame = warp(frame, 600, 600)
        
        # Write the warped frame to the output video
        out.write(warped_frame)
    
    # Release video objects
    cap.release()
    out.release()

# Provide the path to your input video and output video
video_path = "test.mp4"
output_path = "warped_battlefield.mp4"


# Record start time
start_time = time.time()
# Process the video
process_battlefield(video_path, output_path)
# Record end time
end_time = time.time()
# Calculate the duration
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")