import os
from dotenv import load_dotenv
from roboflow import Roboflow
import json
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from global_vars import *

NUM_CLASSES = None
DATASET_NAME = ""
DATASET_DETAILS = None

with open('datasets.json', 'r') as file:
    data = json.load(file)
dataset_names = [list(item.keys())[0]
                 for item in data['supported_datasets']]


def roboflow_download(dataset):
    load_dotenv()
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(DATASET_DETAILS['workspace']).project(
        DATASET_DETAILS['project'])
    version = project.version(DATASET_DETAILS['version'])
    version.download("yolov8", location=f"data/{dataset}")


def load_data(dataset="NHRL"):
    select_data(dataset)
    roboflow_download(dataset)
    return


def select_data(dataset="NHRL"):
    while dataset not in dataset_names:
        print(f"\"{dataset}\" dataset not supported")
        dataset = input(
            f"Please select one of the following options: {dataset_names}: ")

    ind = dataset_names.index(dataset)
    dataset_details = data['supported_datasets'][ind][dataset]

    global DATASET_NAME, NUM_CLASSES, DATASET_DETAILS
    DATASET_NAME = dataset
    NUM_CLASSES = dataset_details['classes']
    DATASET_DETAILS = dataset_details
    update_ind_cls()

    directory = "data"
    folders = [name for name in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, name))]
    if dataset not in folders:
        print(dataset, "not found in downloaded data.")
        print("Downloading", dataset, "data now.")
        roboflow_download(dataset)
    return


idx_cls = []


def update_ind_cls():
    for i in range(NUM_CLASSES):
        idx_cls.append(i+5)


def check_size(split, size, max):
    while size <= 0 or size > max:
        print("Invald", split, "size.")
        size = int(
            input(f"Please enter a number less than or equal to {max}: "))
    return size


def fetch_data(split, size=1):
    while split not in ['train', 'test', 'valid']:
        print("Invalid input", split)
        split = input("Please enter \"train\", \"valid\", or \"test\": ")

    img_path = f'data/{DATASET_NAME}/{split}/images'
    # Get image file paths
    image_files = [f for f in os.listdir(
        img_path) if f.endswith('.jpg') or f.endswith('.png')]
    size = check_size(split, size, len(image_files))

    # Sample `size` number of image and bounding box file pairs
    sampled_indices = random.sample(range(len(image_files)), size)
    sampled_images = [image_files[i] for i in sampled_indices]
    sampled_bboxes = [image.replace('.jpg', '.txt')
                      for image in sampled_images]

    X = np.zeros((size, PROCESSED_IMG_WIDTH,
                 PROCESSED_IMG_HEIGHT, 3), dtype=np.float32)
    # 1 prob, 4 box coordinates, NUM_CLASSES class probabilities
    y = np.zeros((size, MAX_GRID_ROW, MAX_GRID_COLUMN,
                 1+4+NUM_CLASSES), dtype=np.float32)

    for i, (image_file, bbox_file) in enumerate(zip(sampled_images, sampled_bboxes)):
        # Load image
        image_path = os.path.join(img_path, image_file)
        image = Image.open(image_path)
        image_resized = image.resize(
            (PROCESSED_IMG_WIDTH, PROCESSED_IMG_HEIGHT))
        X[i] = np.array(image_resized, dtype=np.float32) / \
            255.0  # Normalize pixel values

    # Load bounding box data
    bbox_path = f'data/{DATASET_NAME}/{split}/labels'
    bbox_file = os.path.join(bbox_path, bbox_file)
    with open(bbox_file, 'r') as f:
        bounding_boxes = [list(map(float, line.split())) for line in f]

    for box in bounding_boxes:
        label, x_center, y_center, width, height = box
        x_center *= PROCESSED_IMG_WIDTH
        y_center *= PROCESSED_IMG_HEIGHT
        width *= PROCESSED_IMG_WIDTH
        height *= PROCESSED_IMG_HEIGHT

        # Calculate top-left corner position from center
        x1 = x_center - width / 2
        y1 = y_center - height / 2

        # Calculate grid cell position
        mx, my = int(y_center // GRID_HEIGHT), int(x_center // GRID_WIDTH)

        # Prevent out-of-bounds access
        if mx >= MAX_GRID_ROW or my >= MAX_GRID_COLUMN or mx < 0 or my < 0:
            continue

        channels = y[i][my][mx]

        # Set the probability and bounding box coordinates
        channels[0] = 1.0  # Set probability to 1 (object present)
        channels[1] = x1 - (mx * GRID_WIDTH)  # x1
        channels[2] = y1 - (my * GRID_HEIGHT)  # y1
        channels[3] = width  # width of bounding box
        channels[4] = height  # height of bounding box
        # Set class probability to 1 for the correct class (label)
        channels[5 + int(label)] = 1.0

    return X, y


def get_color_by_probability(p):
    if p < 0.5:
        return (1., 0., 0.)  # Red for low probability
    return (0., 1., 0.)  # Green for high probability


def show_predict(X, y, model=None, threshold=0.1, img_title="Model Prediction"):
    X = X.copy()
    ind = 0
    if X.shape[0] != 1:
        print(f"Input contains more than one data point ({X.shape[0]})")
        print("Showing prediction for randomly selected data point")
        ind = random.randint(0, X.shape[0]-1)
    if model is not None:
        y = model.predict(X)
    X = X[ind]
    y = y[ind]
    for mx in range(MAX_GRID_ROW):
        for my in range(MAX_GRID_COLUMN):
            channels = y[my][mx]
            prob, x1, y1, width, height = channels[:5]

            # If the probability is below the threshold, skip this cell
            if prob < threshold:
                continue

            color = get_color_by_probability(prob)

            # Calculate the top-left corner of the bounding box
            px = (mx * GRID_WIDTH) + x1
            py = (my * GRID_HEIGHT) + y1

            # Draw the rectangle with the correct width and height scaling
            cv2.rectangle(X, (int(px), int(py)), (int(
                px + width), int(py + height)), color, 1)

            # Draw label background
            cv2.rectangle(X, (int(px), int(py - 10)),
                          (int(px + 12), int(py)), color, -1)

            # Draw the class label
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0, 0.0, 0.0))

    # Display image using matplotlib
    plt.imshow(X)
    plt.title(img_title)
    plt.axis('off')  # To hide axes
    plt.show()


def show_sample_data(split):
    X, y = fetch_data(split)
    show_predict(X, y, img_title=f"Sample from {split} Split")
