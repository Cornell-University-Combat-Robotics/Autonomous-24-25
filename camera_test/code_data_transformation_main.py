import cv2
import time 
from pathlib import Path
from warp import warp  # Assuming 'warp' is a custom function defined in a separate module for frame transformation

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # Camera index, adjust if necessary to use the correct camera (0 for default)

def code_data_transformation_main(cap=cap):
    # Check if the video capture is successfully initialized
    if not cap.isOpened():
        print("Could not open video source.")
        exit()  # Exit the program if the camera couldn't be opened

    # User input for folder name to save captured frames or video
    print(f"Enter the folder name to save the frames to (no spaces please): ")
    folder_name = input()
    
    # User input for capture mode: Video or Images
    print(f"Option 1: Video, Option 2: Images, Enter the option number: ")
    option = int(input())

    # Create a directory to store the captured frames or video
    Path(f"./{folder_name}").mkdir(parents=True, exist_ok=True)

    print("Recording has begun")

    # If the user selects option 1 (Video capture)
    if option == 1:
        # Set up the video writer for the raw and warped videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        output_file = f'./{folder_name}/output_video.mp4'  # Path for the raw video file
        warped_file = f'./{folder_name}/warped_video.mp4'  # Path for the warped video file
        fps = 10  # Frames per second, can be adjusted for smoother or slower videos
        frame_width = int(cap.get(3)) or 600  # Get the frame width or default to 600
        frame_height = int(cap.get(4)) or 600  # Get the frame height or default to 600

        # VideoWriter objects to save the raw and warped videos
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        out2 = cv2.VideoWriter(warped_file, fourcc, fps, (600, 600))  # Fixed resolution for warped video

        # Loop to capture video frames continuously
        while True:
            ret, frame = cap.read()  # Capture each frame from the camera
            if not ret:
                print("Failed to capture image")  # If frame capture fails, break the loop
                break
            
            # Save the raw video frame
            out.write(frame)

            # Apply the 'warp' transformation and save the warped frame
            warped_frame = warp(frame)
            out2.write(warped_frame)

            # Shows the live video feed in a window (optional)
            cv2.imshow("EpocCam Video Feed", frame)

            # Break the loop if 'q' key is pressed
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    # If the user selects option 2 (Image capture)
    elif option == 2:
        num = 1  # Counter for naming the saved image files

        # Loop to capture and save individual frames as images
        while True:
            ret, frame = cap.read()  # Capture each frame from the camera
            if not ret:
                print("Failed to capture image")
                break

            # Shows the live video feed in a window (optional)
            cv2.imshow("EpocCam Video Feed", frame)

            # Save the raw image frame
            filename = f'./{folder_name}/saved_frame{num}.png'
            cv2.imwrite(filename, frame)

            # Apply the 'warp' transformation and save the warped image
            filename2 = f'./{folder_name}/warped_frame{num}.png'
            cv2.imwrite(filename2, warp(frame))
        
            num += 1  # Increment the image counter for the next frame

            # Break the loop if 'q' key is pressed
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    # Release the video writer when done (only applicable for video capture mode)
    if option == 1:
        out.release()

# Call the function to start capturing video or images
code_data_transformation_main(cap)
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
