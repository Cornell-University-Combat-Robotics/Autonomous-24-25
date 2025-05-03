import math
import os
import time
import traceback
import sys

import cv2
import numpy as np
from line_profiler import profile
import math
import matplotlib.pyplot as plt

from Algorithm.ram import Ram
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel, YoloModel
from transmission.motors import Motor
from transmission.serial_conn import OurSerial
from warp_main import get_homography_mat, warp
from vid_and_img_processing.unfisheye import unfish
from vid_and_img_processing.unfisheye import prepare_undistortion_maps

# ------------------------------ GLOBAL VARIABLES ------------------------------

# Set True if using Matt's Laptop
MATT_LAPTOP = True

JANK_CONTROLLER = False

# Set True to optimize for competition, removing all visuals
COMP_SETTINGS = False

# Set True to print outputs for Corner Detection and Algo
PRINT = True

# Save times to an array for plotting
TIMING = True

# Set True to redo warp and picking Huey's main color, front and back corners
WARP_AND_COLOR_PICKING = True

# Set True when testing with a live Huey and not a pre-filmed video
IS_TRANSMITTING = True

# True to display current and future orientation angles for each iteration
SHOW_FRAME = True
DISPLAY_ANGLES = True

# Set True to process every single frame the camera captures
IS_ORIGINAL_FPS = False

UNFISHEYE = True

if COMP_SETTINGS:
    SHOW_FRAME = False
    DISPLAY_ANGLES = False
    PRINT = False
    MATT_LAPTOP = True

if not SHOW_FRAME:
    DISPLAY_ANGLES = False

folder = os.getcwd() + "/main_files"
test_videos_folder = folder + "/test_videos"
resize_factor = 0.8
frame_rate = 50

map1 = np.load('vid_and_img_processing/700xmap1.npy')
map2 = np.load('vid_and_img_processing/700xmap2.npy')

BACK_UP_TIME = 0.5
start_back_up_time = 0

# camera_number = 1
camera_number = 701
# camera_number = test_videos_folder + "/crude_rot_huey.mp4"
# camera_number = test_videos_folder + "/huey_duet_demo.mp4"
# camera_number = test_videos_folder + "/huey_demo2.mp4"
# camera_number = test_videos_folder + "/huey_demo3.mp4"
# camera_number = test_videos_folder + "/only_huey_demo.mp4"
# camera_number = test_videos_folder + "/only_enemy_demo.mp4"
# camera_number = test_videos_folder + "/green_huey_demo.mp4"
# camera_number = test_videos_folder + "/when_i_throw_it_back_huey.mp4"
# camera_number = test_videos_folder + "/kabedon_huey.mp4"
# camera_number = test_videos_folder + "/yellow_huey_demo.mp4"
# camera_number = test_videos_folder + "/warped_no_huey.mp4"
# camera_number = test_videos_folder + "/flippy_huey.mp4"
# camera_number = test_videos_folder + "/nhrl_arena.mp4"


if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

# ------------------------------ BEFORE THE MATCH ------------------------------

@profile
def main():

    if TIMING:
        t_cap = []
        t_fish = []
        t_warp = []
        t_turn = []
        t_predict = []
        t_whole = []

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
                cv2.imshow("Press 'q' to quit. Press '0' to capture the image", frame)
                key = cv2.waitKey(1) & 0xFF # Check for key press

                if key == ord("q"): # Press 'q' to quit without capturing
                    break
                elif key == ord("0"): # Press '0' to capture the image and exit
                    captured_image = frame.copy()
                    resized_image = cv2.resize(captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
                    h, w = resized_image.shape[:2]
                    map1,map2 = prepare_undistortion_maps(w,h)
                    cv2.imwrite(folder + "/captured_image.png", captured_image)
                    break
            else:
                print("Failed to read frame" + "\n")
                break
        cv2.destroyAllWindows()

        # 3. Use the initial frame to get the Homography Matrix
        if WARP_AND_COLOR_PICKING:
            if captured_image is None:
                print("No image was captured. Please press '0' to capture an image before continuing.")
                return
            resized_image = cv2.resize(captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
            cv2.imwrite(folder + "/resized_image.png", resized_image)
            h, w = resized_image.shape[:2]
            map1,map2 = prepare_undistortion_maps(w,h)
            if(UNFISHEYE):
                resized_image = unfish(resized_image,map1,map2)
            homography_matrix = get_homography_mat(resized_image, 700, 700)
            
            warped_frame = warp(resized_image, homography_matrix, 700, 700)
            cv2.imwrite(folder + "/warped_frame.png", warped_frame)

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
            if len(selected_colors) != 4:
                raise ValueError("The file must contain exactly 4 HSV values.")
        except Exception as e:
            print(f"Error reading selected_colors.txt: {e}" + "\n")
            exit(1)

        # 5. Defining all subsystem objects: ML, Corner, Algorithm
        # Defining Roboflow Machine Learning Model Object
        # predictor = RoboflowModel()
        if MATT_LAPTOP:
            predictor = YoloModel("250v12best", "TensorRT", device="cuda")
        else:
            predictor = YoloModel("250v12best", "PT", device="mps")

        # Defining Corner Detection Object
        corner_detection = RobotCornerDetection(selected_colors, False, False)

        # Defining Ram Ram Algorithm Object
        algorithm = None

        if IS_TRANSMITTING:
            # 5.1: Defining Transmission Object if we're using a live video
            ser = OurSerial()
            # speed_motor_group = Motor(ser=ser, channel=speed_motor_channel)
            # turn_motor_group = Motor(ser=ser, channel=turn_motor_channel)
            motor_group = Motor(ser=ser, channel=speed_motor_channel, channel2=turn_motor_channel)
            if JANK_CONTROLLER:
                weapon_motor_group = Motor(ser=ser, channel=weapon_motor_channel, speed=-1)
            else:
                weapon_motor_group = Motor(ser=ser, channel=weapon_motor_channel)

        cv2.destroyAllWindows()

        if WARP_AND_COLOR_PICKING:
            # 6. Do an initial run of ML and Corner. Initialize Algo
            first_run_ml = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
            corner_detection.set_bots(first_run_ml["bots"])

            first_run_corner, IS_FLIPPED = corner_detection.corner_detection_main()

            if first_run_corner and first_run_corner["huey"] and first_run_corner["enemy"]:
                first_run_corner["enemy"] = first_run_corner["enemy"][0] # Ensure single enemy
                algorithm = Ram(bots = first_run_corner)
                first_move_dictionary = algorithm.ram_ram(first_run_corner)
                if PRINT:
                    num_housebots = len(first_run_ml["housebot"])
                    num_bots = len(first_run_ml["bots"])
                    print("Initial Object Detection: " + str(num_housebots) + " housebots, " + str(num_bots) + " bots detected")
                    print("Initial Corner Detection Output: " + str(first_run_corner))
                    print("Initial Algorithm Output: " + str(first_move_dictionary))

                if DISPLAY_ANGLES:
                    display_angles(first_run_corner, first_move_dictionary, warped_frame, True)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                algorithm = Ram()
                cv2.imshow("", warped_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print("Warning: Initial detection of Huey and enemy robot failed." + "\n")
        else:
            algorithm = Ram()

        # ----------------------------------------------------------------------

        # 8. Match begins
        if cap.isOpened() == False:
            print("Error opening video file" + "\n")

        prev = 0
        t0 = time.perf_counter()

        while cap.isOpened():
            time_elapsed = time.perf_counter() - prev

            # 10. Warp image using the Homography Matrix
            if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:

                # 9. Frames are being capture by the camera/pre-recorded video
                if TIMING:
                    t1 = time.perf_counter()
                ret, frame = cap.read()
                if TIMING:
                    t_cap.append(time.perf_counter()-t1)
                if not ret:
                    print("Failed to capture image" + "\n") # If frame capture fails, break the loop
                    break

                if SHOW_FRAME:
                    if cv2.waitKey(1) & 0xFF == ord("q"): # Press Q on keyboard to exit
                        print("exit" + "\n")
                        break

            # 10. Warp image using the Homography Matrix
            if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:
                global start_back_up_time
                prev = time.perf_counter()
                frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
                if(UNFISHEYE):
                    frame = unfish(frame,map1,map2)
                warped_frame = warp(frame, homography_matrix, 700, 700)

                if TIMING:
                    t2 = time.perf_counter()
                # 11. Run the Warped Image through Object Detection
                detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
                if TIMING:
                    t_predict.append(time.perf_counter()-t2)

                # 12. Run Object Detection's results through Corner Detection
                corner_detection.set_bots(detected_bots["bots"])
                detected_bots_with_data, IS_FLIPPED = corner_detection.corner_detection_main()
                
                if PRINT:
                    print("CORNER DETECTION: " + str(detected_bots_with_data))
                    print("IS_FLIPPED: " + str(IS_FLIPPED))

                if detected_bots_with_data and detected_bots_with_data["huey"]:
                    if detected_bots_with_data["enemy"]:
                        # 13. Algorithm runs only if we detect Huey and an enemy robot
                        detected_bots_with_data["enemy"] = detected_bots_with_data["enemy"][0]
                        move_dictionary = algorithm.ram_ram(detected_bots_with_data)


                        # if move_dictionary and (move_dictionary["turn"]+1):
                        #     turn = move_dictionary["turn"] # angle in degrees / 180
                        #     if turn == 0 and move_dictionary["speed"] < 0:
                        #         time.sleep(BACK_UP_TIME)
 
                        if PRINT:
                            print("ALGORITHM: " + str(move_dictionary))
                        if DISPLAY_ANGLES:
                            display_angles(detected_bots_with_data, move_dictionary, warped_frame)
                        # 14. Transmitting the motor values to Huey's if we're using a live video
                        if IS_TRANSMITTING:
                            speed = move_dictionary["speed"]
                            turn = move_dictionary["turn"]

                            tt = time.perf_counter()
                            if turn * -1 > 0:
                                motor_group.move(IS_FLIPPED * speed * 0.8, turn * -1 * 0.55 + 0.2)
                            else:
                                motor_group.move(IS_FLIPPED * speed * 0.8, turn * -1 * 0.55 - 0.2)
                            t_turn.append(time.perf_counter() - tt)


                            
                    elif DISPLAY_ANGLES:
                        display_angles(detected_bots_with_data, None, warped_frame)
                elif DISPLAY_ANGLES:
                    display_angles(None, None, warped_frame)
                
                if SHOW_FRAME and not DISPLAY_ANGLES:
                    print("RAHHHH")
                    cv2.imshow("Bounding boxes (no angles)", warped_frame)
                
                if TIMING:
                    t_whole.append(time.perf_counter()-t0)
                    t0 = time.perf_counter()
        
        cap.release()
        print("============================")
        print("Video finished successfully!")

        if SHOW_FRAME:
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")
    except Exception as e:
        print("UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP:", e)
        # traceback.print_exc(file=sys.stdout) #uncomment to see stack trace
    finally:
        if IS_TRANSMITTING:
            # Motors need to be cleaned up correctly
            try:
                # We stop or clean up objects that actually exist in the current scope
                if 'motor_group' in locals():
                    motor_group.stop()
                if 'ser' in locals():
                    ser.cleanup()
            except Exception as motor_exception:
                print("Motor cleanup failed:", motor_exception)

        if 'cap' in locals():
            cap.release()

        if SHOW_FRAME:
            cv2.destroyAllWindows()

        if TIMING:
            plt.plot(t_cap, label="Capture")
            plt.plot(t_predict, label="Predict")
            plt.plot(t_whole, label="Total")
            plt.plot(t_turn, label="Turn")
            plt.ylim(0,0.08)
            plt.legend()
            plt.savefig("timing.png")

def display_angles(detected_bots_with_data, move_dictionary, image, initial_run=False):
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
        if move_dictionary and (move_dictionary["turn"]):
            turn = move_dictionary["turn"] # angle in degrees / 180
            # print(f'👅: {str(turn)}')
            IS_BACKED = 0
            if turn == 0 and move_dictionary["speed"] < 0:
                IS_BACKED = 180
                cv2.putText(
                    image,
                    "BACKING UP",
                    (50, 50),  # Slightly above the top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,0,255),
                    2,
                )
            new_orientation_degrees = orientation_degrees + (turn * 180) + IS_BACKED

            # Components of predicted turn
            dx = np.cos(math.pi * new_orientation_degrees / 180)
            dy = -1 * np.sin(math.pi * new_orientation_degrees / 180)

            end_point = (int(start_x + 300 * resize_factor * dx), int(start_y + 300 * resize_factor * dy))
            cv2.arrowedLine(image, (start_x, start_y), end_point, (0, 0, 255), 2)

    if initial_run:
        cv2.imshow("Initial Run: Battle with Predictions. Press '0' to continue", image)
    else:
        cv2.imshow("Battle with Predictions", image)
    cv2.waitKey(1)

if __name__ == "__main__":
    # Run using 'python -m kernprof -lvr --unit 1e-3 main.py' for debugging
    main()
