# Image Pipeline for Machine Learning Applications Process
# 1. Image Capture/Loading: Collect image and load onto laptop.
# 2. Image Preprocessing: Resize, normalize, and prepare the image for ML models, yada, yada....
# 3. Model Inference: Pass the processed image to a trained ML model.
# 4. Corner Detection: Pass robot images to corner detection to find orientation and positions.
# 4. Post-processing: Extract the model's output into RAMPLAN(Algorithm).

import cv2
import os
import sys

from main import warp

# Capture video feed from camera using OpenCV 
cap = cv2.VideoCapture(1)

# Build homography matrix and select corners


running = True

while running:

    # 1. Capture image from video feed
    ret, frame = cap.read()
    if not ret: # If a frame cannot be captured
        break

    # 2. Warp image
    warped_image = warp.warp(frame)

    cv2.imshow("Camera", warped_image)


    # 3. Run object detection, return all robot boxes as images and coordinates of bots


    # 4. Run corner detection, return our robot orientation
    

    # 5. Input stuff to algorithm

    # 6. Send instructions to bot

    running = False

cap.release()
cv2.destroyAllWindows()






# init
vid = capture_video()
homography_mat = build_homography_mat()

while running:
    img = grab_imag(vid)
    warped_img = warp(img)
    bots = detect_bots(warped_img)
    # bots is a dictionary {bot_label:[bounding_box_coords, img]}
    # bounding_box_coords = [(x1,y1),(x2,y2)...]
    # Ex. bots = {"house bot":[], "bot1":[], "bot2":[]} (we don't know if bot1/bot2 is our bot or enemy)

    bots = color_detection(bots)
    # now know which bot s our bot vs. enemy
    # Ex. bots = {"house bot":[], "us":[], "enemy":[]}

    orientation = corner_detection(bots)
    # orientation 

    motor_instr = path_algo(bots, orientation)
    # motor instr is a dictionary {"left_drive": speed, "right_drive":speed, "weapon":speed}
    transmit(motor_instr)