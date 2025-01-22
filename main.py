# Image Pipeline for Machine Learning Applications Process
# 1. Image Capture/Loading: Collect image and load onto laptop.
# 2. Image Preprocessing: Resize, normalize, and prepare the image for ML models, yada, yada....
# 3. Model Inference: Pass the processed image to a trained ML model.
# 4. Corner Detection: Pass robot images to corner detection to find orientation and positions.
# 4. Post-processing: Extract the model's output into RAMPLAN(Algorithm).

import cv2
import os
import numpy as np
import time

from warp_main import get_homography_mat, warp
from corner_detection.color_picker import ColorPicker
from corner_detection.corner_detection import RobotCornerDetection
from machine.predict import RoboflowModel

# --------- TAKE A STARTING PHOTO ---------
print("Press '0' to take a starting photo ...")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() # Capture each frame from the camera
    if not ret:
        print("Failed to capture image")  # If frame capture fails, break the loop
        break

    # Shows the live video feed in a window (optional)
    cv2.imshow("EpocCam Video Feed", frame)

    key = cv2.waitKey(1)
    if key == 48:  # ASCII value for '0' key
        break

cv2.imwrite("arena.png", frame)
cv2.imshow("Starting Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- BEFORE THE MATCH ----------

# Homography Matrix
frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4) # TODO: subject to change
h_mat = get_homography_mat(frame, 700, 700)
warped_frame = warp(frame, h_mat, 700, 700)
cv2.imshow("Warped Cage", warped_frame)
cv2.imwrite("warped_frame.png", warped_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ColorPicker: Manually picking colors for the robot, front and back colors
image_path = os.getcwd() + "/warped_frame.png" # TODO: This should use the same image for the homography
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

# Defining ML Model Object
#predictor = YoloModel("100epoch11", "PT")
predictor = RoboflowModel()

# # Defining Ram Ram Algorithm Object
# # TODO: need to define the following: huey_position, huey_orientation, enemy_position
# # algorithm = Ram() # TODO: This needs to be pushed/merged before uncommenting

# ---------- WAITING FOR MATCH TO START ----------

# Press '1' to start the match screen
image = 255 * np.ones((500, 500, 3), np.uint8)
overlay_image = cv2.imread('actually.png')
overlay_image = cv2.resize(overlay_image, (300, 300))
overlay_x = (image.shape[1] - overlay_image.shape[1]) // 2  # Horizontal center
overlay_y = (image.shape[0] - overlay_image.shape[0]) // 2  # Vertical center
image[overlay_y:overlay_y + overlay_image.shape[0], overlay_x:overlay_x + overlay_image.shape[1]] = overlay_image
font = cv2.FONT_HERSHEY_SIMPLEX
text = "Press '1' to start the match"
text_size = cv2.getTextSize(text, font, 1, 2)[0]
text_x = (image.shape[1] - text_size[0]) // 2  # Center the text horizontally
text_y = (image.shape[0] + text_size[1]) // 2  # Center the text vertically
cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imshow("Press '1' to start...", image)

# Wait for '1' key press to start
print("Press '1' to start ...")
while True:
    key = cv2.waitKey(1)
    if key == 49:  # ASCII value for '1' key
        break

cv2.destroyAllWindows()
time.sleep(3)
print("Proceeding with the rest of the program ...")

# ---------- DURING THE MATCH ----------

cap = cv2.VideoCapture(0)
while True: # 🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏 NO ONE CHANGE THIS PLEASE 🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏
    # 1. Camera Capture
    ret, frame = cap.read()  # Capture each frame from the camera
    if not ret:
        print("Failed to capture image")  # If frame capture fails, break the loop
        break

    # Shows the live video feed in a window (optional)
    cv2.imshow("EpocCam Video Feed", frame)

    key = cv2.waitKey(1)
    if key == 48:  # ASCII value for '0' key
        break
    

    # 2. Warp image
    frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
    warped_frame = warp(frame, h_mat, 700, 700)
    cv2.imshow("Warped Cage", warped_frame)

    print("the world is good here")

    # 3. Object Detection
    detected_bots = predictor.predict(warped_frame, show=True)
    # detected_bots = {} # TODO: This should be the output dictionary from Object Detection

    # 4. Corner Detection
    corner_detection.set_bots = detected_bots
    detected_bots_with_data = corner_detection.corner_detection_main()

    # 5. Algorithm
    # motor_value = algorithm.ram_ram(detected_bots_with_data)
    # print("left: ", motor_value["left"])
    # print("right: ", motor_value["right"])

    # 6. Transmission
    # Weapon is turned off here