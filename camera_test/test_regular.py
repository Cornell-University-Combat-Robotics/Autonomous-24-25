import cv2
import time 
import os
import shutil

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # For me this is 1

def test_camera(cap=cap):
    # Check if the video capture is initialized
    if not cap.isOpened():
        print("Could not open video source.")
        exit()
    start = time.time()
    num = 0
    count = 1

    folder_name = input("Image folder name:\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Shows video feed in separate window
        cv2.imshow("EpocCam Video Feed", frame)

        # Check for user input
        key = cv2.waitKey(1)
        
        # Press 's' to save the current frame
        # if key == ord('s'):
        #     filename = 'saved_frame.png'
        #     cv2.imwrite(filename, frame)
        #     print(f"Frame saved as {filename}")
        
        if key == ord('s'):
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            file_name = str(count) + 'saved_frame.png'
            count+=1
            
            path = os.path.join(folder_name, file_name)
            cv2.imwrite(path, frame)
            
            print(f"Frame saved as {file_name}")

        #deletes folder when needed
        if key == ord('d'):
            shutil.rmtree(folder_name)
        
        # Press 'q' to quit
        if key == ord('q'):
            break
test_camera(cap)
cap.release()
cv2.destroyAllWindows()
