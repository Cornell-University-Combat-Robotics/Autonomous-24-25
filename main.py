import math
import os
import time
import cv2
import numpy as np
from line_profiler import profile

from Algorithm.ram import Ram
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel
from machine.predict import YoloModel
from transmission.motors import Motor
from transmission.serial_conn import Serial
from warp_main import get_homography_mat, warp

# ------------------------------ GLOBAL VARIABLES ------------------------------

# Set True to redo warp and color picking bot color, front and back corners
WARP_AND_COLOR_PICKING = False
IS_TRANSMITTING = False

# Set True to process every single frame the camera captures
IS_ORIGINAL_FPS = False

folder = os.getcwd() + "/main_files"
test_videos_folder = folder + "/test_videos"

resize_factor = 0.8

# camera_number = test_videos_folder + "/crude_rot_huey.mp4"
# camera_number = test_videos_folder + "/huey_duet_demo.mp4"
# camera_number = test_videos_folder + "/huey_demo2.mp4"
# camera_number = test_videos_folder + "/huey_demo3.mp4"
# camera_number = test_videos_folder + "/only_huey_demo.mp4"
# camera_number = test_videos_folder + "/only_enemy_demo.mp4"
# camera_number = test_videos_folder + "/green_huey_demo.mp4"
camera_number = test_videos_folder + "/yellow_huey_demo.mp4"
# camera_number = test_videos_folder + "/warped_no_huey.mp4"
# camera_number = 0

frame_rate = 20

if IS_TRANSMITTING:
    speed_motor_channel = 3
    turn_motor_channel = 1

# ------------------------------ BEFORE THE MATCH ------------------------------


@profile
def main():
    # 1. Start the video and 2. Capture initial frame
    cap = cv2.VideoCapture(camera_number)
    captured_image = None

    if cap.isOpened() == False:
        print("Error opening video file" + "\n")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret and frame is not None:
            cv2.imshow("Frame", frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Press 'q' to quit without capturing
                break
            elif key == ord("0"):  # Press '0' to capture the image and exit
                captured_image = frame.copy()  # Store the captured frame
                cv2.imwrite(
                    folder + "/captured_image.png", captured_image
                )  # Save the image
                break
        else:
            print("Failed to read frame" + "\n")
            break

    cv2.destroyAllWindows()

    # 3. Homography Matrix and 4. Display the warped image if flag is True
    if WARP_AND_COLOR_PICKING:
        resized_image = cv2.resize(
            captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
        cv2.imwrite(folder + "/resized_image.png", resized_image)
        h_mat = get_homography_mat(resized_image, 700, 700)
        warped_frame = warp(resized_image, h_mat, 700, 700)
        cv2.imshow("Warped Cage", warped_frame)
        cv2.imwrite(folder + "/warped_frame.png", warped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 5. ColorPicker: Manually picking colors for the robot, front and back colors
        image_path = folder + "/warped_frame.png"
        output_file = folder + "/selected_colors.txt"
        selected_colors = ColorPicker.pick_colors(image_path)
        with open(output_file, "w") as file:
            for color in selected_colors:
                file.write(f"{color[0]}, {color[1]}, {color[2]}\n")
        print(f"Selected colors have been saved to '{output_file}'." + "\n")
    else:
        h_mat = []
        homography_matrix_file = folder + "/homography_matrix.txt"
        try:
            with open(homography_matrix_file, "r") as file:
                for line in file:
                    row = list(map(float, line.strip().split(" ")))
                    h_mat.append(row)
            if len(h_mat) != 3 or len(h_mat[0]) != 3:
                raise ValueError("The file must represent a 3 x 3 matrix.")
            h_mat = np.array(h_mat, dtype=np.float32)
        except Exception as e:
            print(f"Error reading homography_matrix.txt: {e}" + "\n")
            exit(1)

    # Reading the HSV values for robot color, front and back corners from a text file
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

    # Defining Roboflow Machine Learning Model Object
    # predictor = RoboflowModel()
    predictor = YoloModel("250v12best", "PT", device='mps')

    # Defining Corner Detection Object
    corner_detection = RobotCornerDetection(selected_colors, False, False)

    # Defining Ram Ram Algorithm Object
    algorithm = Ram()

    if IS_TRANSMITTING:
        # Defining Transmission Object
        ser = Serial()
        speed_motor_group = Motor(ser=ser, channel=speed_motor_channel)
        turn_motor_group = Motor(ser=ser, channel=turn_motor_channel)

    cv2.destroyAllWindows()

    # 6. Set FPS
    prev = 0

    # 7. Wait for match to start
    while True:
        user_input = input("Type 'start' to begin: ").strip().lower()
        if user_input == "start":
            break
        else:
            print("Invalid input. Please type 'start' to proceed." + "\n")
    print("Proceeding with the rest of the program ..." + "\n")

    # --------------------------------------------------------------------------

    # 8. Match begins
    cap = cv2.VideoCapture(camera_number)
    if cap.isOpened() == False:
        print("Error opening video file" + "\n")

    while cap.isOpened():
        # 9. Camera Capture
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            # If frame capture fails, break the loop
            print("Failed to capture image" + "\n")
            break

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("exit" + "\n")
            break

        # 10. Warp image
        if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:
            prev = time.time()
            frame = cv2.resize(
                frame, (0, 0), fx=resize_factor, fy=resize_factor)
            warped_frame = warp(frame, h_mat, 700, 700)
            cv2.imwrite(folder + "/warped_cage.png", warped_frame)

            # 11. Object Detection
            detected_bots = predictor.predict(
                warped_frame, show=True, track=False)

            # 12. Corner Detection
            corner_detection.set_bots(detected_bots["bots"])
            detected_bots_with_data = corner_detection.corner_detection_main()
            print("detected_bots_with_data: " + str(detected_bots_with_data))

            if detected_bots_with_data and detected_bots_with_data["huey"]:
                if detected_bots_with_data["enemy"]:
                    # 13. Algorithm
                    detected_bots_with_data["enemy"] = detected_bots_with_data["enemy"][0]
                    move_dictionary = algorithm.ram_ram(
                        detected_bots_with_data)
                    print("move_dictionary: " + str(move_dictionary))
                    display_angles(detected_bots_with_data,
                                   move_dictionary, warped_frame)

                    # 14. Transmission
                    if IS_TRANSMITTING:
                        speed_motor_group.move(move_dictionary["speed"])
                        turn_motor_group.move(move_dictionary["turn"])
                else:
                    display_angles(detected_bots_with_data, None, warped_frame)
            else:
                display_angles(None, None, warped_frame)

    cap.release()  # Release the camera object
    cv2.destroyAllWindows()  # Destroy all cv2 windows

    if IS_TRANSMITTING:
        try:
            speed_motor_group.stop()
            turn_motor_group.stop()
            ser.cleanup()
        except:
            print("Algorithm cleanup failed")


def display_angles(detected_bots_with_data, move_dictionary, image):
    # Blue line: Huey's Current Orientation
    if detected_bots_with_data and detected_bots_with_data["huey"]["orientation"]:
        orientation_degrees = detected_bots_with_data["huey"]["orientation"]

        # Components of current front arrow
        dx = np.cos(math.pi / 180 * orientation_degrees)
        dy = -1 * np.sin(math.pi / 180 * orientation_degrees)

        # Huey's center
        start_x = int(detected_bots_with_data["huey"]["center"][0])
        start_y = int(detected_bots_with_data["huey"]["center"][1])

        end_point = (int(start_x + 300 * resize_factor * dx),
                     int(start_y + 300 * resize_factor * dy))
        cv2.arrowedLine(image, (start_x, start_y), end_point, (255, 0, 0), 2)

        # Red line, where we want to face
        if move_dictionary and move_dictionary["turn"]:
            turn = move_dictionary["turn"]  # angle in degrees / 180
            new_orientation_degrees = orientation_degrees + (turn * 180)

            # Components of predicted turn
            dx = np.cos(math.pi * new_orientation_degrees / 180)
            dy = -1 * np.sin(math.pi * new_orientation_degrees / 180)

            end_point = (int(start_x + 300 * resize_factor * dx),
                         int(start_y + 300 * resize_factor * dy))
            cv2.arrowedLine(image, (start_x, start_y),
                            end_point, (0, 0, 255), 2)

    cv2.imshow("Battle with Predictions", image)
    cv2.waitKey(1)


if __name__ == "__main__":
    # Run using 'kernprof -l -v --unit 1e-3 main.py' for debugging
    main()
