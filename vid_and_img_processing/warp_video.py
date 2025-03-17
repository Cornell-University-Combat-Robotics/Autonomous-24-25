import cv2
from vid_to_warped_frames import get_homography_mat


def warp_video(video_name, output_w=600, output_h=600):

    input_path = INPUT_BASE_PATH + video_name
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = OUTPUT_BASE_PATH + "warped_" + video_name
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))

    ret, frame = cap.read()
    # ret: True if cap.read() successfully reads a frame, False otherwise
    if not ret:
        print("Error opening video frame")
        return

    frame = cv2.resize(frame, (output_w, output_h),
                       interpolation=cv2.INTER_AREA)
    homography_mat = get_homography_mat(frame, output_w, output_h)

    # Iterate through the video frames
    while ret:
        frame = cv2.resize(frame, (output_w, output_h),
                           interpolation=cv2.INTER_AREA)
        warped_frame = cv2.warpPerspective(
            frame, homography_mat, (output_w, output_h))

        # Write the warped frame to the output video
        out.write(warped_frame)
        ret, frame = cap.read()

    # Release video objects
    cap.release()
    out.release()


if __name__ == "__main__":
    INPUT_BASE_PATH = "data/input_video/"
    OUTPUT_BASE_PATH = "data/output_video/"
    video_name = "knuckledollar.mp4"
    warp_video(video_name)
