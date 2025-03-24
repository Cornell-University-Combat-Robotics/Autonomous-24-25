from ultralytics import YOLO
from time import sleep
import cv2
import numpy as np

# Load pose model
pose_model = YOLO("yolo11n-pose.pt")  # Load a model

# results = pose_model.track(source="0", show=True, stream=True)  # Track on webcam
# results = pose_model.predict( "real_steel/test_man_poses.webp")  # Predict on image
# results = pose_model.track("real_steel/test_leads.mov", show=True, stream=True, save=True, project='real_steel/runs')  # Save track on video

# cap = cv2.VideoCapture("real_steel/test_leads.mov")  # Video
cap = cv2.VideoCapture(0)  # Webcam

# Initialize body settings

print(
    """Body Part options: \n
    0: Nose
    1: Left Eye
    2: Right Eye
    3: Left Ear
    4: Right Ear
    5: Left Shoulder
    6: Right Shoulder
    7: Left Elbow
    8: Right Elbow
    9: Left Wrist
    10: Right Wrist
    11: Left Hip
    12: Right Hip
    13: Left Knee
    14: Right Knee
    15: Left Ankle
    16: Right Ankle
    
The input for throttle or turn will depend on the position of your selected 'mover' relative to your selected 'reference'.
""")

throt_move = input("Throttle Mover (0-16): ")
throt_ref = input("Throttle Reference (0-16): ")
turn_move = input("Turn Mover (0-16): ")
turn_ref = input("Turn Reference (0-16): ")

# Create low, high values
turn_low = np.array([0.18, -0.1])
turn_high = np.array([0.18, -0.3])
turn_line = turn_high - turn_low

bounds = (None, None, None, None)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = pose_model.track(frame, persist=True, verbose=False)

    # Plot first body keypoints
    if results:
        annotated_frame = results[0].plot()
        for i in range(4):
            if bounds[i] is None:
                if i == 0:
                    cv2.imshow('Pose Estimation', annotated_frame)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = pose_model.track(frame, persist=True, verbose=False)

    # Plot first body keypoints
    if results:
        annotated_frame = results[0].plot()
        cv2.imshow('Pose Estimation', annotated_frame)

    if results:
        # Print keypoints of the first detected person
        points = results[0].keypoints.xyn[0]
        if len(points) != 0:

            # Print difference in coordinates of turn mover and turn base
            t_move_xy = points[turn_move].numpy()
            t_base_xy = points[turn_ref].numpy()

            # All zeros -> not detected -> false
            found_t_move = not np.all(t_move_xy == 0)
            found_t_base = not np.all(t_base_xy == 0)
            if found_t_move and found_t_base:
                raw_diff = t_move_xy - t_base_xy
                print("raw diff: " + str(raw_diff))

                slider = np.dot(turn_line, raw_diff) / \
                    np.dot(turn_line, turn_line)
                print("slider: " + str(slider))

    if results and cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
