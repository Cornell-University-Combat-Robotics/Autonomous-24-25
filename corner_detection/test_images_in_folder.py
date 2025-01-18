import os
import cv2
from corner_detection import RobotCornerDetection

folder_path = os.getcwd() + "/warped_images"
num_images = 0

# Step 1: Pick colors once for the entire folder
selected_colors_file = os.getcwd() + "/selected_colors.txt"
selected_colors = []
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

# Step 2: Process all images in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    not_huey_image_path = os.getcwd() + "/warped_images/east_4_not_huey.png"
    not_huey_image = cv2.imread(not_huey_image_path)
    
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            huey_image = cv2.imread(file_path)
            if huey_image is None:
                print(f"Error: Could not load image '{filename}'")
                continue
            
            bot1 = {"bb": [[50, 50], [60, 60]], "img": huey_image}
            bot2 = {"bb": [[150, 150], [160, 160]], "img": not_huey_image}
            bots = [bot1, bot2]
            
            try:
                corner_detection = RobotCornerDetection(selected_colors, True, False)
                corner_detection.set_bots(bots)
                result = corner_detection.corner_detection_main()
                print(f"Corner detection result for '{filename}': {result}")
                num_images += 1

            except Exception as e:
                print(f"Error during corner detection for '{filename}': {e}")
        
        except Exception as e:
            print(f"Error processing image '{filename}': {e}")
    else:
        print(f"Skipping non-image file '{filename}'")

print(f"Processed {num_images} images in the specified folder.")
