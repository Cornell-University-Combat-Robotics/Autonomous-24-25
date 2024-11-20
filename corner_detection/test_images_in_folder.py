from PIL import Image
import os
import cv2
import numpy as np
from corner_detection import RobotCornerDetection
from color_picker import ColorPicker

folder_path = os.getcwd() + "/warped_images"
num_images = 0

# Step 1: Pick colors once for the entire folder
sample_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
selected_colors = ColorPicker.pick_colors(sample_image_path)  # Pick colors from the first image
print(f"Selected colors for all images: {selected_colors}")

# Step 2: Process all images in the folder
for filename in os.listdir(folder_path):  # List of items in the folder path
    file_path = os.path.join(folder_path, filename)  # Get the specific image path

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image format
        try:
            image = cv2.imread(file_path)

            # Dummy bounding boxes for testing
            bots = {
                "bot1": {"bb": [[50, 50], [60, 60]], "img": image},
                "bot2": {"bb": [[150, 150], [160, 160]], "img": image},
            }

            corner_detector = RobotCornerDetection(bots, selected_colors)
            result = corner_detector.corner_detection_main()

            print(f"Corner detection result for '{filename}': {result}")
            num_images += 1

        except Exception as e:
            print(f"Error processing image '{filename}': {e}")
    else:
        print(f"Skipping non-image file '{filename}'")

print(f"Processed {num_images} images in the specified folder.")
