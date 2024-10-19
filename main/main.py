# Image Pipeline for Machine Learning Applications Process
# 1. Image Capture/Loading: Collect image and load onto laptop.
# 2. Image Preprocessing: Resize, normalize, and prepare the image for ML models, yada, yada....
# 3. Model Inference: Pass the processed image to a trained ML model.
# 4. Corner Detection: Pass robot images to corner detection to find orientation and positions.
# 4. Post-processing: Extract the model's output into RAMPLAN(Algorithm).

import cv2
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp

# Capture video feed from camera using OpenCV 
cap = cv2.VideoCapture(1)

# Build homography matrix and select corners
# from image_flattening import homography

running = True

while running:

    # 1. Capture image from video feed
    ret, frame = cap.read()
    if not ret: # If a frame cannot be captured
        break

    # 2. Warp image
    warped_image = warp.warp(frame)

    cv2.imshow("Camera", warped_image)

    running = False

    # 3. Run object detection, return all robot boxes as images and coordinates of bots


    # 4. Run corner detection, return our robot orientation

    # 5. Input stuff to algorithm

    # 6. Send instructions to bot

cap.release()
cv2.destroyAllWindows()
