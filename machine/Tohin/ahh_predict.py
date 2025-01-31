import torch
import cv2
from torchvision import transforms
import numpy as np
import time
from roboflow import Roboflow
from inference import get_model
import os
from dotenv import load_dotenv
from template_model import TemplateModel
from ultralytics import YOLO
import onnxruntime as ort
import pandas as pd
import openvino as ov

load_dotenv()

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

DEBUG = False

# CD INTO TOHIN, THEN RUN


class AhhModel(TemplateModel):
    def __init__(self, model_path="models/ahhmodel.pth"):
        # Load the model once during initialization
        self.model = torch.load(model_path)
        # Set to evaluation mode, so that we won't train new params
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, img, confidence_threshold=0.5, show=False):
        # Preprocess image via the transformation
        img_tensor = self.transform(img).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            output = self.model(img_tensor)
    
        bots = {}
        bots['bots'] = []
        bots['housebot'] = []
        print(output)
        confidences, bboxes = output
        height, width, _ = img.shape

        print(f"height = {height}")
        print(f"width = {width}")

        robots = 1
        print(f"confidences = {confidences}")
        confidences = confidences[0]
        bboxes = bboxes[0]
        for i in range(len(confidences)):
            confidence, bbox = confidences[i], bboxes[i]

            # Determine class label
            print(f"confidence[0] = {confidence[0]}")
            print(f"confidence[1] = {confidence[1]}")
            print(f"bbox[0] = {bbox[0]}")
            print(f"bbox[1] = {bbox[1]}")
            print(f"bbox[2] = {bbox[2]}")
            print(f"bbox[3] = {bbox[3]}")

            class_label = 0 if confidence[0] > confidence[1] else 1

            # Only add to dictionary if confidence is above threshold
            if confidence[class_label] < confidence_threshold:
                continue

            # Extract center x, center y, box width, and box height
            center_x = int(bbox[0]*6)
            center_y = int(bbox[1]*6)
            box_width = int(bbox[2])
            box_height = int(bbox[3])

            # Calculate bounding box corners
            x_min = center_x - box_width // 2
            y_min = center_y - box_height // 2
            x_max = center_x + box_width // 2
            y_max = center_y + box_height // 2

            # Extract bounding box
            screenshot = img[y_min:y_max, x_min:x_max]

            if DEBUG:
                cv2.imshow("Screenshot", screenshot)
                cv2.waitKey(0)

            # Writing to the dictionary
            if class_label == 0:
                bots['housebot'].append({'bb': [[x_min, y_min], [x_max, y_max]],
                                         'center': [center_x, center_y], 'img': screenshot})
            elif class_label == 1:
                bots['bots'].append({'bb': [[x_min, y_min], [x_max, y_max]],
                                    'center': [center_x, center_y], 'img': screenshot})

        if show:
            self.show_predictions(img, bots)

        return bots

    def show_predictions(self, img, predictions):
        # Display housebot
        print(f"predictions = {predictions}")
        housebots = predictions['housebot']
        bots = predictions['bots']

        color = (0, 0, 255)
        for housebot in housebots:
            x_min, y_min = housebot['bb'][0]
            x_max, y_max = housebot['bb'][1]
            center_x, center_y = housebot['center']

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img, 'housebot',
                (x_min, y_min - 10),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        color = (0, 255, 0)  # Green for bots
        for bot in bots:
            x_min, y_min = bot['bb'][0]
            x_max, y_max = bot['bb'][1]
            print(f"x_min,y_min = {x_min, y_min}")
            print(f"x_max,y_max = {x_max, y_max}")
            center_x, center_y = bot['center']

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img, 'bot',
                (x_min, y_min - 10),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        # for name, data in predictions.items():
        #     # Extract bounding box coordinates and class details
        #     x_min, y_min = data['bbox'][0]
        #     x_max, y_max = data['bbox'][1]
        #     center_x, center_y = data['center']

        #     # Choose color based on the class
        #     if 'housebot' in name:
        #         color = (0, 0, 255)  # Red for housebot
        #     else:
        #         color = (0, 255, 0)  # Green for bots

        #     # Draw the bounding box
        #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        #     # Add label text
        #     cv2.putText(
        #         img, name,
        #         (x_min, y_min - 10),  # Slightly above the top-left corner
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, color, 2
        #     )

        # Display the image with predictions
        cv2.imshow("Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):
        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):
        return super().evaluate(test_path)


# Main code block
if __name__ == '__main__':

    print('starting testing with PT model')
    #predictor = YoloModel("100epoch11","PT")
    predictor = AhhModel()

    IMG_PATH = "data/NHRL/test/images/2242_png.rf.e3285eb24831c586a0bfa92f70110a6e.jpg"
    IMG_PATH1 = "data/NHRL/test/images/1416_png.rf.e96e59268b7fd89b02f25fc75ca41635.jpg"
    

    img = cv2.imread(IMG_PATH1)

    #cv2.imshow("Original image", img)
    #cv2.waitKey(0)

    start_time = time.time()
    bots = predictor.predict(img, show=True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'elapsed time: {elapsed:.4f}')

    