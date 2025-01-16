import cv2
import os
import numpy as np


def get_homography_mat(frame, output_w, output_h):
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
        cv2.imshow("Image", frame_copy)

    cv2.imshow("Image", frame)
    cv2.setMouseCallback("Image", click_event)

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
    return matrix


def process_video(video_path, output_w=600, output_h=600, target_fps=1):
    # open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, frame = cap.read()
    # ret: True if cap.read() successfully reads a frame, False otherwise
    if not ret:
        print("Error opening video frame")
        return

    frame = cv2.resize(frame, (output_w, output_h),
                       interpolation=cv2.INTER_AREA)
    homography_mat = get_homography_mat(frame, output_w, output_h)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if target_fps > 0 and target_fps < fps:
        coeff = fps // target_fps
    else:
        coeff = 1

    frame_counter = 0

    start_index = video_path.find("input_video/") + len("input_video/")
    end_index = video_path.find(".mp4")
    output_img_folder_dir = OUTPUT_BASE_PATH + \
        video_path[start_index:end_index] + "/"
    os.makedirs(output_img_folder_dir, exist_ok=True)

    num_saved_imgs = 0
    print("Starting processing")
    while ret:
        if frame_counter % coeff == 0:
            frame = cv2.resize(frame, (output_w, output_h),
                               interpolation=cv2.INTER_AREA)
            warped_frame = cv2.warpPerspective(
                frame, homography_mat, (output_w, output_h))

            output_img_path = output_img_folder_dir + \
                "/" + str(frame_counter) + ".png"
            num_saved_imgs += 1
            cv2.imwrite(output_img_path, warped_frame)

        ret, frame = cap.read()
        frame_counter += 1

    print("Processing finished")
    print(f"Total saved images: {num_saved_imgs}")


if __name__ == "__main__":
    INPUT_BASE_PATH = "data/input_video/"
    OUTPUT_BASE_PATH = "data/warped_vid_frames/"
    video_name = "radix.mp4"
    process_video(INPUT_BASE_PATH + video_name)
