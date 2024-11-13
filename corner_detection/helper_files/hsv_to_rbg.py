import cv2
import numpy as np

# Define the HSV color
hsv_color = np.uint8([[[170, 150, 150]]])

# Convert to RGB
rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

# Print the RGB result
print(rgb_color[0][0])