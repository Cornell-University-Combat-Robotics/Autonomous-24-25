import cv2
import numpy as np
import math
import time
from detect_our_robot import detect_our_robot_main

# ---------------------------------------------------------------------------- #

# image_path = os.getcwd() + '/bot_images/IMG_4390.JPEG'
# image_path = os.getcwd() + '/non_bot_images/red_and_blue_3.png'

# ---------------------------------------------------------------------------- #

"""
get_contours_per_color(): Returns contours for the front or back corners
Input: side = "front" to assign the HSV for the front cornners, "back" for the 
                back corners
       hsv_image = used when masking colors
Output: lower and upper limits that will be used in masking
"""
def get_contours_per_color(side, hsv_image):
    if side == "front":
        # Narrow down HSV values for red
        lowerLimit = np.array([0, 150, 150])  # More specific lower range for red in HSV
        upperLimit = np.array([5, 255, 255])  # Narrowed upper range for red in HSV

        # Upper red range (due to red's nature in HSV)
        upper_lowerLimit = np.array([170, 150, 150])  # More specific upper lower limit for red
        upper_upperLimit = np.array([179, 255, 255])  # More specific upper upper limit for red
    else:
        # Narrower blue range
        lowerLimit = np.array([90, 130, 100])  # Narrower lower bound for blue in HSV
        upperLimit = np.array([130, 255, 255])  # Narrower upper bound for blue

        # Upper blue range
        upper_lowerLimit = np.array([110, 200, 100])    # More specific lower bound for upper blue range
        upper_upperLimit = np.array([120, 255, 255])    # More specific upper bound for upper blue range

    mask1 = cv2.inRange(hsv_image, lowerLimit, upperLimit)
    mask2 = cv2.inRange(hsv_image, upper_lowerLimit, upper_upperLimit)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

"""
Input: The image
Output: [centroid_front, centroid_back], where centroid_front is an array of all 
        the center points of the front corners, and centroid_back is an array of 
        all the center points of the back corners
"""
def find_centroids(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centroid_front = find_centroids_per_color("front", image, hsv_image) # for now, it's red
    centroid_back = find_centroids_per_color("back", image, hsv_image) # for now, it's blue
    return [centroid_front, centroid_back] 

"""
Algorithm:
- Retrieve HSV limits for the given side (either front or back)
- Create masks for color detection in the HSV image
- Detect contours in the mask, then calculate the centroid for each contour
- Draw the centroid on the original image and label it with the side (front/back)
- Return the list of centroids detected in the image

Input:
- side: This is either "front" or "back" and is passed into get_contours_per_color()
- image: This is the image. Used to draw circle to point out the points
- hsv_image: This is the HSV version of the image. Used in get_contours_per_color()

Output: The centroids for a specific color as an array
"""
def find_centroids_per_color(side, image, hsv_image):
    contours = get_contours_per_color(side, hsv_image)
    centroids = []

    for contour in contours:
        # Filter out small contours based on area
        area = cv2.contourArea(contour)
        if area > 20: # TODO: this value is subject to change based on the size of our bot's corners
            # Compute moments for each contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # Calculate the centroid (center of the dot)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                
                cv2.circle(image, (cx, cy), 8, (0, 0, 0), -1)
                cv2.putText(image, side, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return centroids

"""
get_missing_point() returns a new array with the missing point included in
either red_points or blue_points

Algorithm:
- Let's say we are given 2 blue points and 1 red point
- We find the length from each blue point to the red point called length A and B
- Let's arbitrarily let length A be the hypotenuse, so its greater than length B
- We copy the blue point associated with this length "copy" down to be next
    to the red point as the new second red point
"""
def get_missing_point(points):
    red_points = points[0]
    blue_points = points[1]

    if len(red_points) == 1 and len(blue_points) == 2:
        # Case #1: 1 red point and 2 blue points
        red_point = red_points[0]
        length_a = distance(blue_points[0], red_point)
        length_b = distance(blue_points[1], red_point)

        # Identify which blue point is associated with the hypotenuse
        if length_a > length_b:
            # Copy the blue point associated with length_a near the red point
            new_red_point = (red_point[0] + (blue_points[0][0] - blue_points[1][0]),
                             red_point[1] + (blue_points[0][1] - blue_points[1][1]))
            red_points.append(new_red_point)
        else:
            # Copy the blue point associated with length_b near the red point
            new_red_point = (red_point[0] + (blue_points[1][0] - blue_points[0][0]),
                             red_point[1] + (blue_points[1][1] - blue_points[0][1]))
            red_points.append(new_red_point)

    elif len(blue_points) == 1 and len(red_points) == 2:
        # Case #2: 2 red points and 1 blue point
        blue_point = blue_points[0]
        length_a = distance(red_points[0], blue_point)
        length_b = distance(red_points[1], blue_point)

        # Identify which red point is associated with the hypotenuse
        if length_a > length_b:
            # Copy the red point associated with length_a near the blue point
            new_blue_point = (blue_point[0] + (red_points[0][0] - red_points[1][0]),
                              blue_point[1] + (red_points[0][1] - red_points[1][1]))
            blue_points.append(new_blue_point)
        else:
            # Copy the red point associated with length_b near the blue point
            new_blue_point = (blue_point[0] + (red_points[1][0] - red_points[0][0]),
                              blue_point[1] + (red_points[1][1] - red_points[0][1]))
            blue_points.append(new_blue_point)

    return [red_points, blue_points]

""" distance() returns the euclidean distance between point1 and point2 """
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

"""
Given all 4 points, return the left and right front points of the robot

Algorithm:
- Given all 4 points, first calculate the center of the 4 corners
- Next, find the vector between the center of the bot and the front 2 points
- Calculate the angle between the 2 vectors and the positive x axis
- IF Θ2 > Θ1 and abs(Θ2 - Θ1) <= some given threshold, THEN the red point
    associated with the *** SMALLER *** angle is the top right front corner
- ELSE Θ2 > Θ1, but the abs(Θ2 - Θ1) > some give threshold, THEN the red
    point associated with the *** LARGER *** angle is the top right front corner
"""
def get_left_and_right_front_points(points):
    red_points = points[0]
    blue_points = points[1]
    
    all_points = red_points + blue_points
    center = np.mean(all_points, axis=0)
    
    vector1 = np.array(red_points[0]) - center
    vector2 = np.array(red_points[1]) - center
    
    vector1[1] = -vector1[1]
    vector2[1] = -vector2[1]
    
    theta1 = math.atan2(vector1[1], vector1[0])
    theta2 = math.atan2(vector2[1], vector2[0])
    
    theta1_deg = math.degrees(theta1) if math.degrees(theta1) >= 0 else math.degrees(theta1) + 360
    theta2_deg = math.degrees(theta2) if math.degrees(theta2) >= 0 else math.degrees(theta2) + 360

    # Determine which red point is the top right front corner
    if theta2_deg > theta1_deg:
        # The point with the smaller angle is the top right front corner
        right_front = red_points[0]
        left_front = red_points[1]
    else:
        # The point with the larger angle is the top right front corner
        right_front = red_points[1]
        left_front = red_points[0]

    return [left_front, right_front]

"""
Compute the angle of the tangent to the front of the robot in radians.
Input: p1 = [x1, y2] and p2 = [x2, y2] representing the front 2 corners
Output: Angle of the tangent line relative to the x axis in degrees
"""
def compute_tangent_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = -(y2 - y1)

    angle_rad = np.arctan2(dy, dx)
    tangent_angle_rad = angle_rad + np.pi / 2
    tangent_angle_deg = math.degrees(tangent_angle_rad)

    return tangent_angle_deg

"""
Main function for corner_detection.py
This will run all the functionalities to detect corners and angle of orientation

Input:
- image: The image of our robot. Note this is the output of detect_our_robot.py

Output: TODO: This is subject to change based on the algorithms team
- orientation: The angle of orientation of where our robot is facing
"""
def corner_detection_main():
    image = detect_our_robot_main()
    display_image = True

    if image is not None:
        if display_image:
            cv2.imshow('Our Robot :)', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # height, width, _ = image.shape
        # image = cv2.resize(image, (int(width/2), int(height/2))) # TODO: subject to change
        
        start_time = time.time()

        centroid_points = find_centroids(image)
        left_and_right_front_points = get_left_and_right_front_points(centroid_points)
        left_front = left_and_right_front_points[0]
        right_front = left_and_right_front_points[1]

        if display_image:
            # Draw the left front corner
            cv2.circle(image, (int(left_front[0]), int(left_front[1])), 5, (255, 255, 255), -1)
            cv2.putText(image, "Left Front", (int(left_front[0]), int(left_front[1]) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw the right front corner
            cv2.circle(image, (int(right_front[0]), int(right_front[1])), 5, (255, 255, 255), -1)
            cv2.putText(image, "Right Front", (int(right_front[0]), int(right_front[1]) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Display the image
            cv2.imshow('Image with Left and Right Front Corners', image)
            cv2.imwrite('image_with_front_corners.png', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        angle_of_orientation = compute_tangent_angle(left_front, right_front)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Code execution time (corner_detection): {execution_time} seconds")
        print(f"Centroid points: {centroid_points}")
        print("Angle of Orientation: " + str(angle_of_orientation) + "°\n")

corner_detection_main()