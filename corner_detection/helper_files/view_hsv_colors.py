import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create color samples based on HSV values and display them in RGB
def hsv_to_rgb(h, s, v):
    """Convert HSV values (0-255 range) to RGB color and normalize to 0-1 for matplotlib"""
    color = np.uint8([[[h, s, v]]])
    rgb_color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)[0][0] / 255.0
    return rgb_color

colors_hsv = {
    "front_lower_red": (0, 150, 150),
    "front_upper_red": (5, 255, 255),
    "front_upper_lower_red": (170, 150, 150),
    "front_upper_upper_red": (179, 255, 255),
    "side_lower_blue": (90, 130, 100),
    "side_upper_blue": (130, 255, 255),
    "side_upper_lower_blue": (110, 200, 100),
    "side_upper_upper_blue": (120, 255, 255),
}

# Convert HSV values to RGB and display as color swatches
fig, ax = plt.subplots(1, len(colors_hsv), figsize=(15, 3))
for i, (label, (h, s, v)) in enumerate(colors_hsv.items()):
    rgb_color = hsv_to_rgb(h, s, v)
    ax[i].imshow([[rgb_color]])
    ax[i].axis("off")
    ax[i].set_title(label, fontsize=10)

plt.tight_layout()
plt.show()
