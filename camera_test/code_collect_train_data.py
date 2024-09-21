import cv2
import time 
from pathlib import Path
from warp import warp

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # Make sure 1 is the correct index for your camera


def camera_test(cap=cap):
    # Check if the video capture is initialized
    if not cap.isOpened():
        print("Could not open video source.")
        exit()

    # User input for folder name
    print(f"Enter the folder name to save the frames to (no spaces please): ")
    folder_name = input()
    # User input for folder name
    print(f"Option 1: Video, Option 2: Images, Enter the option number: ")
    option = int(input())

    # Create a directory to save the frames
    Path(f"./{folder_name}").mkdir(parents=True, exist_ok=True)

    print("Recording has begin")

    if (option == 1):
        print(option)
        # Option 1 = Video
        # Define video codec and create VideoWriter object to save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        output_file = f'./{folder_name}/output_video.mp4'
        warped_file = f'./{folder_name}/warped_video.mp4'
        fps = 10 # Frames per second, changeable
        frame_width = int(cap.get(3)) or 600  # Default to 600 if width is 0
        frame_height = int(cap.get(4)) or 600  # Default to 600 if height is 0
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        out2 = cv2.VideoWriter(warped_file, fourcc, fps, (600, 600))
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            out.write(frame)
            warped_frame = warp(frame)
            out2.write(warped_frame)

            # Shows video feed in separate window
            # Comment out this line if you don't want to see the video feed
            cv2.imshow("EpocCam Video Feed", frame)

            # Check for user input
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
    elif (option == 2):
        # Option 2 = Lots of Images
        num = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            # Shows video feed in separate window
            # Comment out this line if you don't want to see the video feed
            cv2.imshow("EpocCam Video Feed", frame)

            filename = f'./{folder_name}/saved_frame{num}.png'
            filename2 = f'./{folder_name}/warped_frame{num}.png'
            cv2.imwrite(filename, frame)
            cv2.imwrite(filename2, warp(frame))
        
            num += 1

            # Check for user input
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
    if (option == 1):
        out.release()
camera_test(cap)
cap.release()
cv2.destroyAllWindows()
