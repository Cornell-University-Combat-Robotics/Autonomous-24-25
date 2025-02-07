import cv2
import os
import numpy as np
import math
import time

from warp_main import get_homography_mat, warp
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel
from Algorithm.ram import Ram

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

    resize_factor = 1
    camera_number = "crude_rot_huey.mp4"
    # camera_number = "huey_duet_demo.mp4"
    # camera_number = "only_huey_demo.mp4"
    # camera_number = "only_enemy_demo.mp4"

    # 1. Start the video and 2. Capture initial frame)
    cap = cv2.VideoCapture(camera_number)
    captured_image = None

    if (cap.isOpened() == False):
        print("Error opening video file" + "\n")

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret and frame is not None:
            cv2.imshow('Frame', frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): # Press 'q' to quit without capturing
                break
            elif key == ord('0'): # Press '0' to capture the image and exit
                captured_image = frame.copy() # Store the captured frame
                cv2.imwrite("captured_image.png", captured_image) # Save the image
                break
        else:
            print("Failed to read frame" + "\n")
            break

    cv2.destroyAllWindows()

    # 3. Homography Matrix and 4. Display the warped image
    resized_image = cv2.resize(captured_image, (0, 0), fx=resize_factor, fy=resize_factor)  # NOTE: resized
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
    print(f"Selected colors have been saved to '{output_file}'." + "\n")

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
        print(f"Error reading selected_colors.txt: {e}" + "\n")
        exit(1)

    # Defining Corner Detection Object
    corner_detection = RobotCornerDetection(selected_colors, False, False)

    # Defining Roboflow Machine Learning Model Object
    predictor = RoboflowModel()

    # Defining Ram Ram Algorithm Object
    algorithm = Ram()

    cv2.destroyAllWindows()

    # 6. Set FPS
    is_original_fps = True # Set True for running straight
    frame_rate = 8
    prev = 0

    # 7. Wait for match to start
    while True:
        user_input = input("Type 'start' to begin: ").strip().lower()
        if user_input == "start":
            break
        else:
            print("Invalid input. Please type 'start' to proceed." + "\n")
    print("Proceeding with the rest of the program ..." + "\n")

    # ------------------------------------------------------------------------------

    # 8. Match begins

    cap = cv2.VideoCapture(camera_number)
    if (cap.isOpened() == False):
        print("Error opening video file" + "\n")

    while (cap.isOpened()):
        # 1. Camera Capture
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            # If frame capture fails, break the loop
            print("Failed to capture image" + "\n")
            break

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("exit" + "\n")
            break

        # 2. Warp image
        if is_original_fps or time_elapsed > 1.0/frame_rate:
            prev = time.time()
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            warped_frame = warp(frame, h_mat, 700, 700)
            cv2.imwrite("warped_cage.png", warped_frame)

            # 3. Object Detection
            detected_bots = predictor.predict(warped_frame, show=True)

            # 4. Corner Detection
            corner_detection.set_bots(detected_bots["bots"])
            detected_bots_with_data = corner_detection.corner_detection_main()
            print("detected_bots_with_data: " + str(detected_bots_with_data) + "\n")
            
            """
            1. detected_bots_with_data is not None
            2. At least 1 bot is detected
            3. At least 1 enemy is detected
            """
            if detected_bots_with_data and detected_bots_with_data["huey"]:
                if detected_bots_with_data["enemy"]:
                    # 5. Algorithm      
                    detected_bots_with_data["enemy"] = detected_bots_with_data["enemy"][0]                                                             
                    move_dictionary = algorithm.ram_ram(detected_bots_with_data)
                    print("move_dictionary: " + str(move_dictionary) + "\n")
                    display_angles(detected_bots_with_data, move_dictionary, warped_frame)
                else:
                    display_angles(detected_bots_with_data, None, warped_frame)
            else:
                display_angles(None, None, warped_frame)

    cap.release() # Release the camera object
    cv2.destroyAllWindows() # Destroy all cv2 windows

def display_angles(detected_bots_with_data, move_dictionary, image):
    # blue line, current orientation
    if detected_bots_with_data and detected_bots_with_data["huey"]["orientation"]:
        orientation = detected_bots_with_data["huey"]["orientation"] # orientation in degrees
        dx = np.cos(math.pi / 180 * orientation)
        dy = -1 * np.sin(math.pi / 180 * orientation)
        start_x = int(detected_bots_with_data["huey"]["center"][0])
        start_y = int(detected_bots_with_data["huey"]["center"][1])
        #TODO: OG orientation is not right; maybe change start & end point for arrow
        end_point = (int(start_x + 300*dx), int(start_y + 300*dy))
        print("start_x: " + str(start_x) + ", start_y: " + str(start_y))
        print("end_point: " + str(end_point))
        cv2.arrowedLine(image, (start_x, start_y), end_point, (255, 0, 0), 2)
    
    # red line, where we want to face
    if move_dictionary and move_dictionary["turn"]:
        turn = move_dictionary["turn"] # angle in degrees / 180
        new_orientation = orientation + (turn * 180) # NEW ORIENTATION IN DEGREES
        dx = np.cos(math.pi * new_orientation / 180)
        dy = -1 * np.sin(math.pi * new_orientation / 180)
        
        end_point = (int(start_x + 300*dx), int(start_y + 300*dy))
        cv2.arrowedLine(image, (start_x, start_y), end_point, (0, 0, 255), 2)
        
    cv2.imshow("Battle with Predictions", image)
    cv2.waitKey(1)

if __name__ == "__main__":
    # Run using 'kernprof -l -v --unit 1e-3 main.py' for debugging
    main()