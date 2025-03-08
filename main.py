import math
import os
import time
import cv2
import numpy as np
from line_profiler import profile

from Algorithm.ram_copy import Ram #ensure 
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel
from transmission.motors import Motor
from transmission.serial_conn import Serial
from warp_main import get_homography_mat, warp

# ------------------------------ GLOBAL VARIABLES ------------------------------

# Set True to redo warp and color picking bot color, front and back corners
WARP_AND_COLOR_PICKING = False
IS_TRANSMITTING = False

# Set True to process every single frame the camera captures
IS_ORIGINAL_FPS = True

folder = os.getcwd() + "/main_files"
test_videos_folder = folder + "/test_videos"

resize_factor = 0.8

accel_loss_sum = 0
velocity_loss_sum = 0
position_loss_sum = 0
num_enemies_pos = 0
prev_position_loss = 0
timestamp = time.time()

# TODO: test on real NHRL video

# camera_number = test_videos_folder + "/crude_rot_huey.mp4"
# camera_number = test_videos_folder + "/huey_duet_demo.mp4"
# camera_number = test_videos_folder + "/huey_demo2.mp4"
# camera_number = test_videos_folder + "/huey_demo3.mp4"
# camera_number = test_videos_folder + "/huey_demo3.2.mp4"
# camera_number = test_videos_folder + "/only_huey_demo.mp4"
# camera_number = test_videos_folder + "/only_enemy_demo.mp4"
# camera_number = test_videos_folder + "/green_huey_demo.mp4"
# camera_number = test_videos_folder + "/yellow_huey_demo.mp4"
# camera_number = test_videos_folder + "/not_huey_test.mp4"
camera_number = test_videos_folder + "/real_gruey_naked.mp4"
# camera_number = test_videos_folder + "/slightly_fatter_huey_test.mp4"

frame_rate = 8

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
        resized_image = cv2.resize(captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
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
    predictor = RoboflowModel()

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
        global num_enemies_pos
        global prev_position_loss
        global timestamp
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
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            warped_frame = warp(frame, h_mat, 700, 700)
            cv2.imwrite(folder + "/warped_cage.png", warped_frame)

            # 11. Object Detection
            detected_bots = predictor.predict(warped_frame, show=True)

            # 12. Corner Detection
            corner_detection.set_bots(detected_bots["bots"])
            detected_bots_with_data = corner_detection.corner_detection_main()
            print("detected_bots_with_data: " + str(detected_bots_with_data) + "\n")

            if detected_bots_with_data and detected_bots_with_data["huey"]:

                detected_enemy_bbox = []
                detected_enemy_center = []
                if detected_bots_with_data["enemy"]:
                    # 13. Algorithm
                    num_enemies_pos = len(algorithm.enemy_future_positions) - 1
                    detected_bots_with_data["enemy"] = detected_bots_with_data["enemy"][0]
                    detected_enemy_bbox = detected_bots_with_data["enemy"]["bbox"] # save in case enemy is lost in  future
                    detected_enemy_center = detected_bots_with_data["enemy"]["center"]
                    
                    if (num_enemies_pos > 1):
                        prev_position_loss = display_loss(warped_frame, algorithm, num_enemies_pos, velocity_loss_sum, accel_loss_sum, position_loss_sum, prev_position_loss,timestamp)

                    # 14. Transmission
                    if IS_TRANSMITTING:
                        speed_motor_group.move(move_dictionary["speed"])
                        turn_motor_group.move(move_dictionary["turn"])
                else: # No enemy bots detected, use last stored pos
                    algorithm.enemy_previous_positions += algorithm.enemy_previous_positions[-1] #  duplicate last pos
                    num_enemies_pos = len(algorithm.enemy_previous_positions)-1
                    detected_bots_with_data["enemy"]["bbox"] = detected_enemy_bbox
                    detected_bots_with_data["enemy"]["center"] = detected_enemy_center
                    # display_angles(detected_bots_with_data, None, enemy_future_list, algorithm.enemy_future_position_velocity, warped_frame)
                move_dictionary, enemy_future_list = algorithm.ram_ram(detected_bots_with_data)

                display_angles(detected_bots_with_data, move_dictionary, enemy_future_list, algorithm.enemy_future_position_velocity, warped_frame)
            else:
                display_angles(None, None, None, None, warped_frame)
    print("Starting loss:")
    
    #TODO: update num_enemies_pos every frame rather than using len(future_positions)

    num_enemies_pos = len(algorithm.enemy_future_positions) - 1
    
    print("final num: " + str(num_enemies_pos))
    if (num_enemies_pos + 1 > 0):
        average_vel_percentage_loss = velocity_loss_sum/(num_enemies_pos + 1)
        average_pos_percentage_loss = position_loss_sum/(num_enemies_pos + 1)
        average_accel_percentage_loss = accel_loss_sum/(num_enemies_pos + 1)
    else:
        average_vel_percentage_loss = 0
        average_pos_percentage_loss = 0
        average_accel_percentage_loss = 0
    
        
    print("======================================================")

    print(f"Velocities: {algorithm.enemy_future_position_velocity[10:15]}")
    print("------------------------------------------")
    print(f"Accelerations: {algorithm.enemy_future_positions[10:15]}")
    print("------------------------------------------")
    print(f"Corresponding positions: {algorithm.enemy_previous_positions[11:16]}")

    print("======================================================")

    print("Velocity Loss: " + str(average_vel_percentage_loss))
    print("Position Loss: " + str(average_pos_percentage_loss))
    print("Acceleration Loss: " + str(average_accel_percentage_loss))

    print("======================================================")
    

    cap.release() # Release the camera object
    cv2.destroyAllWindows() # Destroy all cv2 windows
    print("Video capture finished successfully!")

    # TEST THE ACCURACY OF EACH via "L2 loss"

    if IS_TRANSMITTING:
        try:
            speed_motor_group.stop()
            turn_motor_group.stop()
            ser.cleanup()
        except:
            print("Algorithm cleanup failed")

def position_loss(cur_pos, predicted_pos):
    # percentage loss
    # TODO: catch when cur_pos is (0,0)
    if (cur_pos[0] != 0 and cur_pos[1] != 0):
        pos_diff_x = (predicted_pos[0] - cur_pos[0])/cur_pos[0]
        pos_diff_y = (predicted_pos[1] - cur_pos[1])/cur_pos[1]
    else:
        return 0
    return math.sqrt(pos_diff_x**2 + pos_diff_y**2)

def display_angles(detected_bots_with_data, move_dictionary, enemy_future_list, enemy_future_position_velocity, image):
    # Blue line: Huey's Current Orientation
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

        # Red line, where we want to face
        if move_dictionary and move_dictionary["turn"]:
            turn = move_dictionary["turn"]  # angle in degrees / 180
            new_orientation_degrees = orientation_degrees + (turn * 180)
            
            # Components of predicted turn
            dx = np.cos(math.pi * new_orientation_degrees / 180)
            dy = -1 * np.sin(math.pi * new_orientation_degrees / 180)

            end_point = (int(start_x + 300 * resize_factor * dx), int(start_y + 300 * resize_factor * dy))
            cv2.arrowedLine(image, (start_x, start_y), end_point, (0, 0, 255), 2)

    # Plot enemy future position 
    print(enemy_future_list)

    if enemy_future_list and len(enemy_future_list) > 0 and detected_bots_with_data["enemy"] and len(detected_bots_with_data["enemy"]) != 0:
        enemy_x = int(detected_bots_with_data["enemy"]["center"][0])
        enemy_y = int(detected_bots_with_data["enemy"]["center"][1])

        future_pos_x = int(enemy_future_list[len(enemy_future_list) - 1][0])
        future_pos_y = int(enemy_future_list[len(enemy_future_list) - 1][1])
        
        print("future_pos_x: " + str(future_pos_x) + ", future_pos_y" + str(future_pos_y))

        # Green line, enemy future pos from accel. & velocity
        cv2.arrowedLine(image, (enemy_x, enemy_y), (future_pos_x, future_pos_y), (0, 255, 0), 2)

        # Yellow line, enemy future pos from just velocity
        if enemy_future_position_velocity and len(enemy_future_position_velocity) > 0:
            velocity_pos_x = int(enemy_future_position_velocity[len(enemy_future_position_velocity) - 1][0])
            velocity_pos_y = int(enemy_future_position_velocity[len(enemy_future_position_velocity) - 1][1])
            cv2.arrowedLine(image, (enemy_x, enemy_y), (velocity_pos_x, velocity_pos_y), (0, 255, 255), 2)
        
    cv2.imshow("Battle with Predictions", image)
    cv2.waitKey(1)

def display_loss(warped_frame, algorithm, i, velocity_loss_sum, accel_loss_sum, position_loss_sum, prev_position_loss,timestamp):
     # NOTE: We only append to enemy_future_positions at len(enemy_previous_positions) >= 3
    # NOTE: We append to enemy_future_position_velocity immediately
    cur_velocity_loss = position_loss(algorithm.enemy_previous_positions[i], algorithm.enemy_future_position_velocity[i])
    calculated_position_loss = position_loss(algorithm.enemy_previous_positions[i], algorithm.enemy_previous_positions[i-1])
    # cur_acceleration_loss = position_loss(algorithm.enemy_previous_positions[i], algorithm.enemy_future_positions[i])
    
    pos_loss_text = f"Position Loss: {calculated_position_loss * 100:.1f}\n"
    vel_loss_text = f"Velocity Loss: {cur_velocity_loss * 100:.1f}\n"
    # accel_loss_text = f"Acceleration Loss: {cur_acceleration_loss}"
    
    vel_loss_color = (25500*cur_velocity_loss, 0, 0)
    pos_loss_color = (25500*calculated_position_loss, 0, 0)
    # accel_loss_color = (255*cur_acceleration_loss, 255*cur_acceleration_loss, 255*cur_acceleration_loss)

    # change loss color depending on how big change is
    # write loss to txt file
    
    # appends text organized under video name
    with open(f'{camera_number}{timestamp}.txt','a') as file:
        file.write(f'Position Loss: {calculated_position_loss}     ')
        file.write(f'Velocity Loss: {cur_velocity_loss} ')
        # file.write(f'Accel Loss: {cur_acceleration_loss}\n')
    
    # cv2.putText(warped_frame, accel_loss_text, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, accel_loss_color, 2, cv2.LINE_AA)
    cv2.putText(warped_frame, vel_loss_text, (20, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, vel_loss_color, 2, cv2.LINE_AA)
    cv2.putText(warped_frame, pos_loss_text, (20, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, pos_loss_color, 2, cv2.LINE_AA)
    cv2.destroyAllWindows() # refresh every frame
    
    velocity_loss_sum += cur_velocity_loss
    # accel_loss_sum += cur_acceleration_loss
    if calculated_position_loss == 0:
        position_loss_sum += prev_position_loss
    else:
        position_loss_sum += calculated_position_loss
    prev_position_loss = calculated_position_loss
    return prev_position_loss
    


if __name__ == "__main__":
    # Run using 'kernprof -l -v --unit 1e-3 main.py' for debugging
    main()
