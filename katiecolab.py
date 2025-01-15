import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from roboflow import Roboflow

rf = Roboflow(api_key="Xrg2tfnPJSH2rnwuzxFw")
project = rf.workspace("crc-autonomous").project("nhrl-robots")
version = project.version(2)
dataset = version.download("yolov8")

train_path = '/content/NHRL-Robots-2/train'
val_path = '/content/NHRL-Robots-2/valid'
test_path = '/content/NHRL-Robots-2/test'

def get_color_by_probability(p):
    if p < 0.5:
        return (1., 0., 0.)  # Red for low probability
    return (0., 1., 0.)  # Green for high probability

grid_size = 16

def show_predict(X, y, threshold=0.1):
    X = X.copy()

    for mx in range(8):
        for my in range(8):
            channels = y[my][mx]
            prob, x1, y1, width, height = channels[:5]  # Change made here

            # If the probability is below the threshold, skip this cell
            if prob < threshold:
                continue

            color = get_color_by_probability(prob)

            # Calculate the top-left corner of the bounding box
            px = (mx * grid_size) + x1
            py = (my * grid_size) + y1
            # print(px, py)
            # print(px+width, py+height)

            # Draw the rectangle with the correct width and height scaling
            cv2.rectangle(X, (int(px), int(py)), (int(px + width), int(py + height)), color, 1)

            # Draw label background
            cv2.rectangle(X, (int(px), int(py - 10)), (int(px + 12), int(py)), color, -1)

            # Draw the class label
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)), cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0, 0.0, 0.0))

# plt.imshow(X)


