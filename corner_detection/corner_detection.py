import numpy as np
import cv2
import math
from PIL import Image
import sys

def get_corners(img):
    print("check")

    r = 2
    g = 1
    b = 0

    # --- RED ---
    r_query_low = 150
    # r_query_high = 180
    g_query_high = 70
    # g_query_low = 40
    b_query_high = 80
    # b_query_low = 50

    Y,X = np.where((img[:,:,r] >= r_query_low) & (img[:,:,g] <= g_query_high) & (img[:,:,b] <= b_query_high))
    # Y,X = np.where((r_query_high >= img[:,:,r] >= r_query_low) & (g_query_low <= img[:,:,g] <= g_query_high) & (b_query_low <= img[:,:,b] <= b_query_high))
    y_points = np.column_stack((X, Y))

    try:
        print("start")
        min_x = min(y_points, key = lambda x: x[0])[0]
        min_y = min(y_points, key = lambda x: x[1])[1]
        max_x = max(y_points, key = lambda x: x[0])[0]
        max_y = max(y_points, key = lambda x: x[1])[1]
        x_thresh = (max_x - min_x)/2
        y_thresh = (max_y - min_y)/2
        print("end")
    except:
        print("malfunction")
        raise Exception("Could not find any red points")
    
    # The image incorrectly detects only 1 or 2 corners on the same line, based on threshold split into exceed corner. This is BAD
    if (x_thresh < 20 or y_thresh < 20) :
        raise Exception("Could not detect four corners in image. Threshold too small")
    else: # Image is detected properly with at least 3 corners.
        print("pass")
        fps = []
        for p in y_points:
            # If x or y differ substantially from every other point's x and y
            Diff = True
            for p2 in fps:
                if (abs(p2[0] - p[0]) < x_thresh and abs(p2[1] - p[1]) < y_thresh):
                    Diff = False
                    break
            if Diff:
                print(p)
                print(type(p))
                fps.append(p)

        # Sorted in a z manner
        # 1----------2
        # |          |
        # 3----------4

        # 3 corners are detected
        if (len(fps) == 3):
        # This section finds the two points that are on the same y level and makes a list with each of the two points
            # Draw circles on the image
            initial_3_points = img
            for (x, y) in fps:
                cv2.circle(initial_3_points, (x, y), 10, (0, 0, 255), 2) # (0, 0, 255) is in BGR, so this is red

            # Save or display the image with corners
            cv2.imwrite('initial_3_points.png', initial_3_points)
            cv2.imshow('Initial 3 points', initial_3_points)

            firstpoint = fps[2]
            failedy = True
            for x in range(2):
                if abs(firstpoint[1] - fps[x][1]) < 50:
                    ylevel = [firstpoint, fps[x]]
                    failedy = False
            if failedy:
                ylevel = [fps[0], fps[1]]

            # This section finds the two points that are on the same x level and makes a list with each of the two points
            failedx = True
            for x in range(2):
                if abs(firstpoint[0] - fps[x][0]) < 50:
                    xlevel = [firstpoint, fps[x]]
                    failedx = False
            if failedx:
                xlevel = [fps[0], fps[1]]

        # Need to look at both lists and see which point is repeated twice, that point
        # is diagonal to the new point that we want (we do this by merging the lists and
        # comparing the first element with the third and fourth).
            points = ylevel + xlevel
            if np.array_equiv(points[0],points[2]):
                fourthpoint = np.array((points[1][0], points[3][1]))
            elif np.array_equiv(points[0],points[3]):
                fourthpoint = np.array((points[1][0], points[2][1]))
            elif np.array_equiv(points[1],points[2]):
                fourthpoint = np.array((points[0][0], points[3][1]))
            else:
                fourthpoint = np.array((points[0][0], points[2][1]))
            print(fourthpoint)
            print(type(fourthpoint))
            fps.append(fourthpoint)

        fps = sorted(fps, key = lambda tup: tup[0] * 5 + tup[1]*10)

    # Draw circles on the image
    img_with_corners = img
    for (x, y) in fps:
        cv2.circle(img_with_corners, (x, y), 10, (0, 0, 255), 2) # (0, 0, 255) is in BGR, so this is red

    # Save or display the image with corners
    cv2.imwrite('image_with_corners.png', img_with_corners)
    cv2.imshow('Image with Corners', img_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # (x coordinate, y coordinate)
    return fps

def compute_tangent_angle(p1, p2):
    """
    Compute the angle of the tangent to the front of the robot in radians.

    Input:
    p1, p2: tuples representing the coordinates of the two front corners p = (x, y)

    Output:
    tangent_angle_rad: angle of the tangent line relative to the x axis in radians
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = np.arctan2(dy, dx)
    tangent_angle_rad = angle_rad + np.pi / 2
    tangent_angle_deg = math.degrees(tangent_angle_rad)

    return tangent_angle_deg

# Load image using OpenCV
image = cv2.imread('red.png')
# Process the image
four_corners = get_corners(image)

print("--------------------\n")
print("Detected corners: " + str(four_corners) + "\n")

# Let's just assume (for now) that the first two points from four_corners are the front 2 points
top_left_corner = four_corners[0]
top_right_corner = four_corners[1]

# Computing the angle of orientation in degrees
angle_of_orientation = compute_tangent_angle(top_left_corner, top_right_corner)
# Print angle of orientation in degrees
print("Angle of Orientation: " + str(angle_of_orientation) + "°\n")
