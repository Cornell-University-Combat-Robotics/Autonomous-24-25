import os
import cv2
import numpy as np

"""
Duplicated from vid_and_img_processing/vid_to_warped_frames.py

Given some frame of the arena, allows user to select points. 
Returns the resulting homography matrix.
Params:
    - frame: A cv2 image of the full arena, as seen from the camera
    - output_w: The ideal output width of the image. By default, 1200px
    - output_h: The ideal output height of the image. By default, 1200px

Returns:
    - A numpy matrix, which, when used with cv2.warpPerspective
    - 'flattens' the perspective

As a change from the original get_homography_mat, does NOT resize the input image.
"""
folder = os.getcwd() + "/main_files"

def get_homography_mat(frame, output_w=1200, output_h=1200):
    corners = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left button clicked, store the point
            corners.append([x, y])
            print(f"Point added: {x}, {y}")
            draw_corners()  # Redraw the points on the image

    def draw_corners():
        # Make a copy of the original image to redraw the points
        frame_copy = frame.copy()

        # Loop through the clicked points and display them on the image
        # Select corners in the following order: top left, top right, bottom right, bottom left
        for point in corners:
            cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, f"{point}", point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the image with points
        cv2.imshow("Warp: Select arena corners from top left, top right, bottom right to bottom left", frame_copy)

    cv2.imshow("Warp: Select arena corners from top left, top right, bottom right to bottom left", frame)
    cv2.setMouseCallback("Warp: Select arena corners from top left, top right, bottom right to bottom left", click_event)

    key = cv2.waitKey(1) & 0xFF
    # press 'Esc' to quit
    while len(corners) < 4 and key != 27:
        if key == ord('z'):  # If 'z' is pressed
            if len(corners) > 0:
                removed_point = corners.pop()  # Remove the last point
                print(f"Point removed: {removed_point}")
                draw_corners()  # Redraw the image with remaining points
            else:
                print("No points to remove.")
        key = cv2.waitKey(1) & 0xFF

    print("Final Selected Points:", corners)
    dest_pts = [[0, 0], [output_w, 0], [output_w, output_h], [0, output_h]]
    matrix, status = cv2.findHomography(np.array(corners), np.array(dest_pts))
    cv2.destroyAllWindows()

    # Saving the homography matrix in a txt file
    output_file = folder + "/homography_matrix.txt"
    with open(output_file, "w") as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")
    print(f"Homography matrix has been saved to '{output_file}'.")

    return matrix

"""
Re-combined from camera_test/warp.py, vid_and_img_processing/warp_image.py

Given a frame and a homography matrix, warp the perspective and isolate the flat plane.

Params:
    - frame: A cv2 image of the full arena, as seen from the camera
    - h_mat: The homography matrix used for transformation, derived from 'get_homography_mat'
    - output_w: The ideal output width of the image. By default, 1200px
    - output_h: The ideal output height of the image. By default, 1200px

Returns:
    - A cv2 image of the 'warped' arena

As a change from the original warp, does NOT resize the input image.
"""
def warp(frame, h_mat, output_w=1200, output_h=1200):
    return cv2.warpPerspective(frame, h_mat, (output_w, output_h))

if __name__ == "__main__":
    frame = cv2.imread('./vid_and_img_processing/sample_cage_ss.png')
    # height, width, _ = frame.shape
    frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
    h_mat = get_homography_mat(frame, 700, 700)
    warped_frame = warp(frame, h_mat, 700, 700)
    cv2.imshow("Warped cage", warped_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()