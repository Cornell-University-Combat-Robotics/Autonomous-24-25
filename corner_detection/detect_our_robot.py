import cv2
import numpy as np
import os
import time

# ---------------------------------------------------------------------------- #

# image_path_1 = os.getcwd() + '/bot_images/less_pink_bot.png'
# image_path_2 = os.getcwd() + '/bot_images/more_pink_bot.png'

image_path_1 = os.getcwd() + '/warped_images/warped_image.png'
image_path_2 = os.getcwd() + '/warped_images/warped_image.png'

# ---------------------------------------------------------------------------- #

"""
Detect the number of pixels of bright pink color in a single image.

Input:
- image: The input image of the robot.

Output:
- pink_pixel_count: The number of bright pink pixels detected in the image.
"""
def find_bot_color_pixels(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get the HSV range for the robot's bright pink color in BGR
    # TODO: this value is subject to change based on the actual color of our robot
    # See Corner Detection README for more information
    lower_limit = np.array([140, 100, 100])
    upper_limit = np.array([170, 255, 255])

    # Create a mask for the bright pink color in the image
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

    # Count the number of non-zero pixels in the mask, which correspond to bright pink pixels
    pink_pixel_count = cv2.countNonZero(mask)

    return pink_pixel_count

"""
Detect the robot in two images using its bright pink color.

Input:
- image1, image2: The two input images of the robot, where one is our robot
and the other is the enemy robot

Output:
- The image that is our robot
"""
def find_our_bot(image1, image2):
    image_1_pixels = find_bot_color_pixels(image1)
    image_2_pixels = find_bot_color_pixels(image2)

    if image_1_pixels > image_2_pixels:
        return image1
    else:
        return image2

"""
Main function for detect_our_robot.py
This will run all the functionalities to detect our robot from 2 different bot images

Input:
- image1, image2: The two input images of the robot, where one is our robot
and the other is the enemy robot

Output:
- The image that is our robot
"""   
def detect_our_robot_main():
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    if image1 is not None and image2 is not None:
        # height1, width1, _ = image1.shape
        # height2, width2, _ = image2.shape

        # image1 = cv2.resize(image1, (int(width1/2), int(height1/2)))
        # image2 = cv2.resize(image2, (int(width2/2), int(height2/2)))
        
        start_time = time.time()
        our_bot = find_our_bot(image1, image2)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Code execution time (detect_our_robot): {execution_time} seconds")
        return our_bot