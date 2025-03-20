import math
import os
import time

import cv2
import numpy as np
from line_profiler import profile

from Algorithm.ram import Ram
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel, YoloModel
from transmission.motors import Motor
from transmission.serial_conn import Serial
from warp_main import get_homography_mat, warp

# ------------------------------ GLOBAL VARIABLES ------------------------------

# Set True to optimize for competition, removing all visuals
COMP_SETTINGS = False

# Set True to print outputs for Corner Detection and Algo
PRINT = False

# Set True to redo warp and picking Huey's main color, front and back corners
WARP_AND_COLOR_PICKING = True

# Set True when testing with a live Huey and not a pre-filmed video
IS_TRANSMITTING = False

# True to display current and future orientation angles for each iteration
SHOW_FRAME = True
DISPLAY_ANGLES = True

# Set True to process every single frame the camera captures
IS_ORIGINAL_FPS = False

if COMP_SETTINGS:
    SHOW_FRAME = False
    DISPLAY_ANGLES = False
    PRINT = False

if not SHOW_FRAME:
    DISPLAY_ANGLES = False

folder = os.getcwd() + "/main_files"
test_videos_folder = folder + "/test_videos"
resize_factor = 0.8
frame_rate = 60

# camera_number = 0
# camera_number = test_videos_folder + "/crude_rot_huey.mp4"
# camera_number = test_videos_folder + "/huey_duet_demo.mp4"
# camera_number = test_videos_folder + "/huey_demo2.mp4"
# camera_number = test_videos_folder + "/huey_demo3.mp4"
# camera_number = test_videos_folder + "/only_huey_demo.mp4"
# camera_number = test_videos_folder + "/only_enemy_demo.mp4"
# camera_number = test_videos_folder + "/green_huey_demo.mp4"
camera_number = test_videos_folder + "/yellow_huey_demo.mp4"
# camera_number = test_videos_folder + "/warped_no_huey.mp4"

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3

# ------------------------------ BEFORE THE MATCH ------------------------------

@profile
def main():
    try:
        # 1. Start the capturing frame from the camera or pre-recorded video
        # 2. Capture initial frame by pressing '0'
        cap = cv2.VideoCapture(camera_number)
        captured_image = None

        if cap.isOpened() == False:
            print("Error opening video file" + "\n")

        while cap.isOpened():
            ret, frame = cap.read()

            if ret and frame is not None:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF # Check for key press

                if key == ord("q"): # Press 'q' to quit without capturing # TODO: have these instructions above the image
                    break
                elif key == ord("0"): # Press '0' to capture the image and exit # TODO: have these instructions above the image
                    captured_image = frame.copy()
                    cv2.imwrite(folder + "/captured_image.png", captured_image)
                    break
            else:
                print("Failed to read frame" + "\n")
                break
        cv2.destroyAllWindows()

        # 3. Use the initial frame to get the Homography Matrix
        if WARP_AND_COLOR_PICKING:
            resized_image = cv2.resize(captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
            cv2.imwrite(folder + "/resized_image.png", resized_image)
            homography_matrix = get_homography_mat(resized_image, 700, 700)
            warped_frame = warp(resized_image, homography_matrix, 700, 700)
            cv2.imshow("Warped Cage", warped_frame)
            cv2.imwrite(folder + "/warped_frame.png", warped_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 3.2 Part 2. ColorPicker: Manually picking colors for Huey, front and back corners
            image_path = folder + "/warped_frame.png"
            output_file = folder + "/selected_colors.txt"
            selected_colors = ColorPicker.pick_colors(image_path)
            with open(output_file, "w") as file:
                for color in selected_colors:
                    file.write(f"{color[0]}, {color[1]}, {color[2]}\n")
            print(f"Selected colors have been saved to '{output_file}'." + "\n")
        # 3. Or use the previously saved homography matrix from the txt file
        else:
            homography_matrix = []
            homography_matrix_file = folder + "/homography_matrix.txt"
            try:
                with open(homography_matrix_file, "r") as file:
                    for line in file:
                        row = list(map(float, line.strip().split(" ")))
                        homography_matrix.append(row)
                if len(homography_matrix) != 3 or len(homography_matrix[0]) != 3:
                    raise ValueError("The file must represent a 3 x 3 matrix.")
                homography_matrix = np.array(homography_matrix, dtype=np.float32)
            except Exception as e:
                print(f"Error reading homography_matrix.txt: {e}" + "\n")
                exit(1)

        # 4. Reading the HSV values for Huey, front and back corners from a text file
        selected_colors = []
        selected_colors_file = folder + "/selected_colors.txt"
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

        # 5. Defining all subsystem objects: ML, Corner, Algorithm
        # Defining Roboflow Machine Learning Model Object
        # predictor = RoboflowModel()
        predictor = YoloModel("250v12best", "PT", device="mps")

        # Defining Corner Detection Object
        corner_detection = RobotCornerDetection(selected_colors, False, False)

        # Defining Ram Ram Algorithm Object
        algorithm = None

        if IS_TRANSMITTING:
            # 5.1: Defining Transmission Object if we're using a live video
            ser = Serial()
            speed_motor_group = Motor(ser=ser, channel=speed_motor_channel)
            turn_motor_group = Motor(ser=ser, channel=turn_motor_channel)

        cv2.destroyAllWindows()

        # 6. Do an initial run of ML and Corner. Initialize Algo
        first_run_ml = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
        corner_detection.set_bots(first_run_ml["bots"])
        first_run_corner = corner_detection.corner_detection_main()

        if first_run_corner and first_run_corner["huey"] and first_run_corner["enemy"]:
            first_run_corner["enemy"] = first_run_corner["enemy"][0] # Ensure single enemy
            algorithm = Ram(bots = first_run_corner)
            first_move_dictionary = algorithm.ram_ram(first_run_corner)
            if PRINT:
                print(f"Initial Object Detection Output: Detected [{len(first_run_ml[housebots])} housebots], [{len(first_run_ml[bots])} bots]")
                print("Initial Corner Detection Output: " + str(first_run_corner))
                print("Initial Algorithm Output: " + str(first_move_dictionary))

            if DISPLAY_ANGLES:
                display_angles(first_run_corner, first_move_dictionary, warped_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            algorithm = Ram()
            print("Warning: Initial detection of Huey and enemy robot failed." + "\n")

        # ----------------------------------------------------------------------

        # 7. Wait for match to start. User needs to enter in 'start' in the command line
        # In a real match, we enter 'start' before the match actually starts
        # to ensure ML, Corner, and Algo is running smoothly before the 3, 2, 1 ... go!
        while True:
            user_input = input("Type 'start' to begin: ").strip().lower()
            if user_input == "start":
                break
            else:
                print("Invalid input. Please type 'start' to proceed." + "\n")
        print("Proceeding with the rest of the program ..." + "\n")

        # --------------------------------------------------------------------------

        # 8. Match begins
        prev = 0
        cap = cv2.VideoCapture(camera_number)
        if cap.isOpened() == False:
            print("Error opening video file" + "\n")

        while cap.isOpened():
            # 9. Frames are being capture by the camera/pre-recorded video
            time_elapsed = time.time() - prev
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image" + "\n") # If frame capture fails, break the loop
                break

            if SHOW_FRAME:
                if cv2.waitKey(1) & 0xFF == ord("q"): # Press Q on keyboard to exit
                    print("exit" + "\n")
                    break

            # 10. Warp image using the Homography Matrix
            if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:
                prev = time.time()
                frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
                warped_frame = warp(frame, homography_matrix, 700, 700)

                # 11. Run the Warped Image through Object Detection
                detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)

                # 12. Run Object Detection's results through Corner Detection
                corner_detection.set_bots(detected_bots["bots"])
                detected_bots_with_data = corner_detection.corner_detection_main()
                if PRINT:
                    print("CORNER DETECTION: " + str(detected_bots_with_data))

                if detected_bots_with_data and detected_bots_with_data["huey"]:
                    if detected_bots_with_data["enemy"]:
                        # 13. Algorithm runs only if we detect Huey and an enemy robot
                        detected_bots_with_data["enemy"] = detected_bots_with_data["enemy"][0]
                        move_dictionary = algorithm.ram_ram(detected_bots_with_data)
                        if PRINT:
                            print("ALGORITHM: " + str(move_dictionary))
                        if DISPLAY_ANGLES:
                            display_angles(detected_bots_with_data, move_dictionary, warped_frame)
                        # 14. Transmitting the motor values to Huey's if we're using a live video
                        if IS_TRANSMITTING:
                            speed_motor_group.move(move_dictionary["speed"])
                            turn_motor_group.move(move_dictionary["turn"])
                    elif DISPLAY_ANGLES:
                        display_angles(detected_bots_with_data, None, warped_frame)
                elif DISPLAY_ANGLES:
                    display_angles(None, None, warped_frame)
                if SHOW_FRAME and not DISPLAY_ANGLES:
                    cv2.imshow("Bounding boxes (no angles)", warped_frame)
        
        cap.release()

        if SHOW_FRAME:
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")
    except Exception as e:
        print("UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP:", e)
    finally:
        if IS_TRANSMITTING:
            # Motors need to be cleaned up correctly
            try:
                speed_motor_group.stop()
                turn_motor_group.stop()
                ser.cleanup()
            except Exception as motor_exception:
                print("Motor cleanup failed:", motor_exception)

        cap.release()

        if SHOW_FRAME:
            cv2.destroyAllWindows()

def display_angles(detected_bots_with_data, move_dictionary, image):
    # BLUE line: Huey's Current Orientation according to Corner Detection
    if detected_bots_with_data and detected_bots_with_data["huey"]["orientation"]:
        orientation_degrees = detected_bots_with_data["huey"]["orientation"]

        # Components of current front arrow
        dx = np.cos(math.pi / 180 * orientation_degrees)
        dy = -1 * np.sin(math.pi / 180 * orientation_degrees)

        # Huey's center
        start_x = int(detected_bots_with_data["huey"]["center"][0])
        start_y = int(detected_bots_with_data["huey"]["center"][1])

        end_point = (int(start_x + 300 * resize_factor * dx), int(start_y + 300 * resize_factor * dy))
        cv2.arrowedLine(image, (start_x, start_y), end_point, (255, 0, 0), 2)

        # RED line: Huey's Desired Orientation according to Algorithm
        if move_dictionary and move_dictionary["turn"]:
            turn = move_dictionary["turn"] # angle in degrees / 180
            new_orientation_degrees = orientation_degrees + (turn * 180)

            # Components of predicted turn
            dx = np.cos(math.pi * new_orientation_degrees / 180)
            dy = -1 * np.sin(math.pi * new_orientation_degrees / 180)

            end_point = (int(start_x + 300 * resize_factor * dx), int(start_y + 300 * resize_factor * dy))
            cv2.arrowedLine(image, (start_x, start_y), end_point, (0, 0, 255), 2)

    cv2.imshow("Battle with Predictions", image)
    cv2.waitKey(1)

if __name__ == "__main__":
    # Run using 'kernprof -l -v --unit 1e-3 main.py' for debugging
    main()
