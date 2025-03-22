import cv2

def turn_on_camera(camera_index=0):
    """Turn on the camera and display the live feed."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit.")
    
    while True:
        print("1")
        ret, frame = cap.read()

        print(ret)
        print(frame)

        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow('Camera Feed', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    turn_on_camera()
