import cv2
import time 

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # For me this is 1

# Check if the video capture is initialized
if not cap.isOpened():
    print("Could not open video source.")
    exit()
start = time.time()
num = 0

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
    count = 0 
    listf = []
    if key == ord('s'):
        folder_name = 'Saved_frames'

        filename = count + 'saved_frame.png'
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
    
    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
