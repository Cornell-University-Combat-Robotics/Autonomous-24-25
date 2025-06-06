import cv2
import time 

#WARNING: this may or may work for raspberry pi, you may need to install raspcamera drivers

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # For me this is 1, 0 as the webcam, test for your own setup

# Function to capture a single image from the camera feed
def test_single_image(cap=cap):
    # Check if the video capture is initialized
    if not cap.isOpened():
        print("Could not open video source.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Shows video feed in separate window        
        # Comment out this line if you don't want to see the video feed
        cv2.imshow("EpocCam Video Feed", frame)

        # Check for user input
        key = cv2.waitKey(1)
        # Press 's' to save the current frame
        if key == ord('s'):
            filename = 'saved_frame.png'
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        
        # Press 'q' to quit
        if key == ord('q'):
            break
test_single_image(cap)
cap.release()
cv2.destroyAllWindows()
