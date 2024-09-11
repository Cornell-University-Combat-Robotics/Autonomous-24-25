import cv2
from warp import warp
import time

# Takes a 'video_path' to a video, warps it by the homography matrix specified in 'warp.py', then
# writes it to a valid 'output_path'.
def process_battlefield_to_video(video_path, output_path):
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
        # ret: True if cap.read() successfully reads a frame, False otherwise
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

def warp_test(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Iterate through the video frames
    while True:
        # ret: True if cap.read() successfully reads a frame, False otherwise
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Apply the warp function to the frame
        warped_frame = warp(frame, 600, 600)

# Provide the path to your input video and output video
video_path = "test.mp4"
output_path = "warped_battlefield.mp4"


# Record start time
print("Starting warp:")
start_time = time.time()
# Process the video
warp_test(video_path)
# Record end time
end_time = time.time()
# Calculate the duration
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")