import cv2
import time 
from pathlib import Path

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # For me this is 1, 0 by default is the built-in web camera

def test_framepersec(cap=cap):
    # Check if the video capture is initialized
    if not cap.isOpened():
        print("Could not open video source.")
        exit()

    # User input
    print(f"Enter the time in seconds for the test to run: ")
    test_time = int(input())
    print(f"Enter the folder name to save the frames to(no spaces please): ")
    folder_name = input()

    # Create a directory to save the frames
    Path(f"./{folder_name}").mkdir(parents=True, exist_ok=True)
    # starting time of the code 
    start = time.time()
    # starting frame number
    num = 1

    # run the code for 'test_time' seconds
    while ((time.time()-start) < test_time):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Shows video feed in separate window
        # Comment out this line if you don't want to see the video feed
        cv2.imshow("EpocCam Video Feed", frame)

        # Check for user input
        key = cv2.waitKey(1)
        
        # continuously save frames
        filename = f'./{folder_name}/saved_frame{num}.png'
        cv2.imwrite(filename, frame)
        num +=1
        
        # Press 'q' to quit
        if key == ord('q'):
            break

test_framepersec(cap)
cap.release()
cv2.destroyAllWindows()
