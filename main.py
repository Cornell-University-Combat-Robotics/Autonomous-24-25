import cv2
import os
import numpy as np
import time
import timeit

from warp_main import get_homography_mat, warp
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel

# ---------- BEFORE THE MATCH ----------

"""

1. Start the video (camera = 0)
2. Capture the initial image (resize if needed)
3. Using the initial image, plot four corners of arena (clockwise from top left)
4. Display the warped initial image
5. Using the initial image, select colors (body, front corner, back corner)
6. Wait until start match

"""


@profile
def main():
    # 0. Set resize, camera numbers

    resize_factor = 0.4
    camera_number = 0

    # 1. Start the video and 2. Capture initial frame)

    cap = cv2.VideoCapture(camera_number)

    if (cap.isOpened() == False):
        print("Error opening video file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('Frame', frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Press 'q' to quit without capturing
                break
            elif key == ord('0'):  # Press '0' to capture the image and exit
                captured_image = frame.copy()  # Store the captured frame
                cv2.imwrite("captured_image.png",
                            captured_image)  # Save the image
                # cv2.imshow('Captured Image', captured_image)
                # print("Press any key to continue...")
                # cv2.waitKey(0)
                break
        else:
            print("Failed to read frame")
            break

    cv2.destroyAllWindows()

    # 3. Homography Matrix and 4. Display the warped image
    resized_image = cv2.resize(captured_image, (0, 0),
                               fx=resize_factor, fy=resize_factor)  # NOTE: resized
    h_mat = get_homography_mat(resized_image, 700, 700)
    warped_frame = warp(resized_image, h_mat, 700, 700)
    cv2.imshow("Warped Cage", warped_frame)
    cv2.imwrite("warped_frame.png", warped_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. ColorPicker: Manually picking colors for the robot, front and back colors
    image_path = os.getcwd() + "/warped_frame.png"
    output_file = "selected_colors.txt"
    selected_colors = ColorPicker.pick_colors(image_path)

    with open(output_file, "w") as file:
        for color in selected_colors:
            file.write(f"{color[0]}, {color[1]}, {color[2]}\n")

    print(f"Selected colors have been saved to '{output_file}'.")

    # Reading the HSV values for robot color, front and back corners from a text file
    selected_colors = []
    selected_colors_file = os.getcwd() + "/selected_colors.txt"
    try:
        with open(selected_colors_file, "r") as file:
            for line in file:
                hsv = list(map(int, line.strip().split(", ")))
                selected_colors.append(hsv)
        if len(selected_colors) != 3:
            raise ValueError("The file must contain exactly 3 HSV values.")
    except Exception as e:
        print(f"Error reading selected_colors.txt: {e}")
        exit(1)

    # Defining Corner Detection Object
    corner_detection = RobotCornerDetection(selected_colors)

    # Defining Roboflow Machine Learning Model Object

    predictor = RoboflowModel()

    cv2.destroyAllWindows()

    # 6. Wait until the match starts
    while True:
        user_input = input("Type 'start' to begin: ").strip().lower()
        if user_input == "start":
            break
        else:
            print("Invalid input. Please type 'start' to proceed.")

    print("Proceeding with the rest of the program ...")

    # ------------------------------------------------------------------------------

    cap = cv2.VideoCapture(camera_number)
    if (cap.isOpened() == False):
        print("Error opening video file")

    times = []

    while (cap.isOpened()):

        t1 = timeit.default_timer()
        # 1. Camera Capture
        ret, frame = cap.read()
        if not ret:
            # If frame capture fails, break the loop
            print("Failed to capture image")
            break

        # NOTE: These exit key lines take ~27 ms per iteration, handle with Ctrl+C instead -Aaron
        # Press Q on keyboard to exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     print("exit")
        #     break

        # 2. Warp image
        frame = cv2.resize(
            frame, (0, 0), fx=resize_factor, fy=resize_factor)
        # Can you test outputting a smaller image to OD from warp and see how it affects runtime/consistency of detections -Aaron
        warped_frame = warp(frame, h_mat, 700, 700)
        cv2.imshow("Warped Cage", warped_frame)

        # 3. Object Detection
        detected_bots = predictor.predict(warped_frame, show=True)

        # Debug timing info
        times.append(round(1000 * (timeit.default_timer() - t1), 4))
        if len(times) > 500:
            nptime = np.asarray(times)
            np.save('looptimes.npy', nptime)
            break

        # 4. Corner Detection # TODO: Change the formatting
        # corner_detection.set_bots = [detected_bots]
        # detected_bots_with_data = corner_detection.corner_detection_main()

    cap.release()  # Release the camera object
    cv2.destroyAllWindows()  # Destroy all cv2 windows


if __name__ == "__main__":
    # Run using 'kernprof -l -v --unit 1e-3 main.py' for debugging
    main()
