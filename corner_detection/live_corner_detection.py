import cv2
import os
import numpy as np

from color_picker import ColorPicker
from corner_detection import RobotCornerDetection

# Capture video feed from camera using OpenCV
cap = cv2.VideoCapture(1)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
    exit(1)

# ---------- BEFORE THE MATCH ----------

# ColorPicker: Manually picking colors for the robot, front and back colors
image_path = os.getcwd() + "/warped_images/east.png" # TODO: This should use the same image for the homography
output_file = "selected_colors.txt"
selected_colors = ColorPicker.pick_colors(image_path)

with open(output_file, "w") as file:
    for color in selected_colors:
        file.write(f"{color[0]}, {color[1]}, {color[2]}\n")

print(f"Selected colors have been saved to '{output_file}'.")

# Reading the HSV values for robot color, front and back corners from a text file
selected_colors = []
selected_colors_file = os.getcwd() + "/selected_colors.txt"
try:
    with open(selected_colors_file, "r") as file:
        for line in file:
            hsv = list(map(int, line.strip().split(", ")))
            selected_colors.append(hsv)
    if len(selected_colors) != 3:
        raise ValueError("The file must contain exactly 3 HSV values.")
except Exception as e:
    print(f"Error reading selected_colors.txt: {e}")
    exit(1)

# Defining Corner Detection Object
corner_detection = RobotCornerDetection(selected_colors)

# ---------- WAITING FOR MATCH TO START ----------

# Press '1' to start the match screen
image = 255 * np.ones((500, 500, 3), np.uint8)
overlay_image = cv2.imread('actually.png')
overlay_image = cv2.resize(overlay_image, (300, 300))
overlay_x = (image.shape[1] - overlay_image.shape[1]) // 2  # Horizontal center
overlay_y = (image.shape[0] - overlay_image.shape[0]) // 2  # Vertical center
image[overlay_y:overlay_y + overlay_image.shape[0], overlay_x:overlay_x + overlay_image.shape[1]] = overlay_image
font = cv2.FONT_HERSHEY_SIMPLEX
text = "Press '1' to start the match"
text_size = cv2.getTextSize(text, font, 1, 2)[0]
text_x = (image.shape[1] - text_size[0]) // 2  # Center the text horizontally
text_y = (image.shape[0] + text_size[1]) // 2  # Center the text vertically
cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imshow("Press '1' to start...", image)

# Wait for '1' key press to start
print("Press '1' to start ...")
while True:
    key = cv2.waitKey(1)
    if key == 49: # ASCII value for '1' key
        break

cv2.destroyAllWindows()
print("Proceeding with the rest of the program ...")

# ---------- DURING THE MATCH ----------

running = True
while running:
    # print("1")

    # Capture image from video feed
    ret, frame = cap.read()
    # print("2")
    if not ret:  # If a frame cannot be captured
        break
    # print("3")
    cv2.imshow("Video", frame)
    # print("4")

    # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
    #     running = False

    # 4. Corner Detection # TODO: Change the formatting
    # corner_detection.set_bots = [detected_bots]
    # detected_bots_with_data = corner_detection.corner_detection_main()

print("5")
cap.release()
print("6")
cv2.destroyAllWindows()
