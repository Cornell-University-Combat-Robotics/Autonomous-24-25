import datetime
import os
import cv2
from warp import warp
import time

# Takes a 'video_path' to a video, warps it by the homography matrix specified in 'warp.py', then
# writes it to a valid 'output_path'.

countFolder = 0
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

# Takes a 'video_path' to a video, warps it by the homography matrix specified in 'warp.py', then
# writes it to a valid output folder. Records at 'fps' frames per second (roughly), so as to not
# save thousands of images from a single match.
def process_battlefield_to_images(video_path, target_fps=0):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    global imgList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, cap)
    imgList.append(fileName)
    
    myDirectory = os.path.join(os.getcwd(), 'image_flattening')
    # CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
    while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
        countFolder += 1
    newPath = myDirectory +"/IMG"+str(countFolder)
    os.makedirs(newPath)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if target_fps != 0 and target_fps < fps:
        coeff = fps // target_fps
    else:
        coeff = 1

    # Iterate through the video frames
    i = 0
    while True:
        # ret: True if cap.read() successfully reads a frame, False otherwise
        # should automatically increment through the video, moving to the next frame every time is 
        # called
        ret, frame = cap.read()
        if not ret:
            break
        if i % coeff == 0:
            # Apply the warp function to the frame
            warped_frame = warp(frame, 600, 600)

            # Save the warped frame to our output folder
        i += 1

# Provide the path to your input video and output video
video_path = "test.mp4"
output_path = "warped_battlefield.mp4"


# Record start time
print("Starting warp:")
start_time = time.time()
# Process the video
# warp_test(video_path)
# Record end time
end_time = time.time()
# Calculate the duration
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")