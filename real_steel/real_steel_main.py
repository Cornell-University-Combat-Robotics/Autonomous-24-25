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

throt_move = int(input("Throttle Mover (0-16): "))
throt_ref = int(input("Throttle Reference (0-16): "))
turn_move = int(input("Turn Mover (0-16): "))
turn_ref = int(input("Turn Reference (0-16): "))

# Create low, high values
turn_low = np.array([0.18, -0.1])
turn_high = np.array([0.18, -0.3])
turn_line = turn_high - turn_low

bounds = [None, None, None, None]
empty_bounds = True

while cap.isOpened() and empty_bounds:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = pose_model.track(frame, persist=True, verbose=False)

    # Plot first body keypoints
    if results:
        annotated_frame = results[0].plot()
        for i in range(4):
            if bounds[i] == None:
                if i == 0:
                    cv2.imshow(
                        'Press s to select p' + str(i) + ': Throttle Low position', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        bounds[i] = results[0].keypoints.xyn[0][throt_move] - \
                            results[0].keypoints.xyn[0][throt_ref]
                elif i == 1:
                    cv2.imshow(
                        'Press s to select: Throttle High position', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        bounds[i] = results[0].keypoints.xyn[0][throt_move] - \
                            results[0].keypoints.xyn[0][throt_ref]
                elif i == 2:
                    cv2.imshow(
                        'Press s to select: Turn Low position', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        bounds[i] = results[0].keypoints.xyn[0][turn_move] - \
                            results[0].keypoints.xyn[0][turn_ref]

                elif i == 3:
                    cv2.imshow(
                        'Press s to select: Turn High position', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        bounds[i] = results[0].keypoints.xyn[0][turn_move] - \
                            results[0].keypoints.xyn[0][turn_ref]
                break
            elif i == 3:
                empty_bounds = False

print(bounds)
throt_low = bounds[0]
throt_high = bounds[1]
turn_low = bounds[2]
turn_high = bounds[3]

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
            turn_move_xy = points[turn_move].numpy()
            turn_base_xy = points[turn_ref].numpy()

            # All zeros -> not detected -> false
            found_turn_move = not np.all(turn_move_xy == 0)
            found_turn_base = not np.all(turn_base_xy == 0)
            if found_turn_move and found_turn_base:
                turn_diff = turn_move_xy - turn_base_xy
                # print("raw diff: " + str(turn_diff))

                slider = np.dot(turn_line, turn_diff) / \
                    np.dot(turn_line, turn_line)
                print("Turn slider: " + str(slider))

            # Print difference in coordinates of throt mover and throt base
            throt_move_xy = points[throt_move].numpy()
            throt_base_xy = points[throt_ref].numpy()

            # All zeros -> not detected -> false
            found_throt_move = not np.all(throt_move_xy == 0)
            found_throt_base = not np.all(throt_base_xy == 0)
            if found_throt_move and found_throt_base:
                throt_diff = throt_move_xy - throt_base_xy
                # print("raw diff: " + str(raw_diff))

                slider = np.dot(turn_line, throt_diff) / \
                    np.dot(turn_line, turn_line)
                print("Throt slider: " + str(slider))

    if results and cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
