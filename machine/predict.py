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

load_dotenv()

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

DEBUG = False


class OurModel(TemplateModel):
    def __init__(self, model_path="models/model_20241113_000141.pth"):
        # Load the model once during initialization
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, img, confidence_threshold=0.5, show=False):
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            output = self.model(img_tensor)

        bots = {}
        bots['bots'] = []
        confidences, bboxes, _ = output
        height, width, _ = img.shape

        robots = 1
        for i in range(len(confidences)):
            confidence, bbox = confidences[i], bboxes[i]

            # Determine class label
            class_label = 0 if confidence[0] > confidence[1] else 1

            # Only add to dictionary if confidence is above threshold
            if confidence[class_label] < confidence_threshold:
                continue

            # Extract center x, center y, box width, and box height
            center_x = int(bbox[0] * width)
            center_y = int(bbox[1] * height)
            box_width = int(bbox[2] * width)
            box_height = int(bbox[3] * height)

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
                bots['housebot'] = {'bb': [[x_min, y_min], [x_max, y_max]],
                                    'center': [center_x, center_y], 'img': screenshot}
            elif class_label == 1:
                bots['bots'].append({'bb': [[x_min, y_min], [x_max, y_max]],
                                    'center': [center_x, center_y], 'img': screenshot})

        if show:
            self.show_predictions(img, bots)

        return bots

    def show_predictions(self, img, predictions):
        # Display housebot
        try:
            predictions['housebot']
            # TODO: Ethan is updating how we parse the new dictionary
        except:
            print("No housebot")
        for name, data in predictions.items():
            # Extract bounding box coordinates and class details
            x_min, y_min = data['bbox'][0]
            x_max, y_max = data['bbox'][1]
            center_x, center_y = data['center']

            # Choose color based on the class
            if 'housebot' in name:
                color = (0, 0, 255)  # Red for housebot
            else:
                color = (0, 255, 0)  # Green for bots

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img, name,
                (x_min, y_min - 10),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        # Display the image with predictions
        cv2.imshow("Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):
        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):
        return super().evaluate(test_path)


class YoloModel(TemplateModel):
    def __init__(self):
        self.model = get_model(model_id='nhrl-robots/6',
                               api_key=ROBOFLOW_API_KEY)

    def predict(self, img, confidence_threshold=0.5, show=False):
        out = self.model.infer(img)

        bots = {}

        preds = out[0].predictions

        robots = 1
        for pred in preds:
            if pred.confidence > confidence_threshold:
                if pred.class_id == 0:
                    name = 'housebot'
                elif pred.class_id == 1:
                    name = 'bot' + str(robots)
                    robots += 1
                bots[name] = {}

                x, y, box_width, box_height = pred.x, pred.y, pred.width, pred.height
                if DEBUG:
                    print(
                        f'x: {x}, y: {y}, box_width: {box_width}, box_height: {box_height}')

                bots[name]['center'] = [x, y]

                x_min = int(x - box_width / 2)
                y_min = int(y - box_height / 2)
                x_max = int(x + box_width / 2)
                y_max = int(y + box_height / 2)

                # Extract bounding box
                screenshot = img[y_min:y_max, x_min:x_max]

                if DEBUG:
                    cv2.imshow("Screenshot", screenshot)
                    cv2.waitKey(0)

                bots[name]['bbox'] = [[x_min, y_min], [x_max, y_max]]
                bots[name]['img'] = screenshot

        if show:
            self.show_predictions(img, bots)

        return bots

    def show_predictions(self, img, predictions):
        for name, data in predictions.items():
            # Extract bounding box coordinates and class details
            x_min, y_min = data['bbox'][0]
            x_max, y_max = data['bbox'][1]
            center_x, center_y = data['center']

            # Choose color based on the class
            if 'housebot' in name:
                color = (0, 0, 255)  # Red for housebot
            else:
                color = (0, 255, 0)  # Green for bots

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img, name,
                (x_min, y_min - 10),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        # Display the image with predictions
        cv2.imshow("Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):
        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):
        return super().evaluate(test_path)


class PTModel(TemplateModel):
    def __init__(self):
        pt_model_path = 'machine/models/100epoch11.pt'
        self.model = YOLO(pt_model_path)

    def predict(self, img: np.ndarray, show=False):
        results = self.model(img)
        result = results[0]

        robots = []
        housebots = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy, width, height = box.xywh[0].tolist()
            cropped_img = img[int(y1): int(y2), int(x1):int(x2)]

            cv2.imshow('image', cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows

            dict = {
                "bb": [[x1, y1], [x2, y2]],
                "center": [cx, cy],
                "img": cropped_img
            }

            if box.cls == 0:
                housebots.append(dict)
            else:
                robots.append(dict)

        out = {"bots": robots, "housebots": housebots}
        return out

    def show_predictions(self, img, bots_dict):
        for label, bots in bots_dict.items():

            for bot in bots:

                # Extract bounding box coordinates and class details
                x_min, y_min = bot['bb'][0]
                x_max, y_max = bot['bb'][1]

                # Choose color based on the class
                if 'housebot' in label:
                    color = (0, 0, 255)  # Red for housebot
                else:
                    color = (0, 255, 0)  # Green for bots

                # Draw the bounding box
                cv2.rectangle(img, (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)), color, 2)

                # Add label text
                # cv2.putText(
                #     img, label,
                #     (x_min, y_min - 10),  # Slightly above the top-left corner
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5, color, 2
                # )
        cv2.imshow("Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img


class OnnxModel(TemplateModel):
    def __init__(self):
        onnx_model_path = './models/best.onnx'
        session = ort.InferenceSession(onnx_model_path)
        self.model = session

    def predict(self, img, confidence_threshold=0.5, show=False):
        image_path = img
        image = cv2.imread(image_path)

        # Resize to model's required size
        input_image = cv2.resize(image, (640, 640))
        input_image = input_image.astype(np.float32)  # Convert to float32
        # Normalize to [0, 1] if required by the model
        input_image = input_image / 255.0
        # Convert HWC to CHW if required
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(
            input_image, axis=0)  # Add batch dimension

        # input_image = cv2.resize(image, (input_width, input_height))  # Use your model’s required size
        # input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0, 1] if required
        # input_image = np.transpose(input_image, (2, 0, 1))  # Convert HWC to CHW format
        # input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        input_name = self.model.get_inputs()[0].name

        output = self.model.run(None, {input_name: input_image})

        # Getting the indices of the robots based on the class confidences
        bot_conf = np.where(output[0][0][5] > 0.8)
        print(bot_conf)
        # Getting the indices of the housebot based on the class confidences
        house_conf = np.where(output[0][0][4] > 0.9)

        bots = []

        for i in range(6):
            print(output[0][0][i][bot_conf])
            bots.append(output[0][0][i][bot_conf])

        return np.transpose(bots)

    def show_predictions(self, img, predictions):
        image = cv2.imread(img)

# Iterate through detections
        # Access the first dimension of the output
        detections = predictions[0][0]
        for i in range(detections.shape[1]):  # Loop through each detection
            x, y, width, height, confidence, class_id = detections[:, i]

            if confidence > 0.5:  # Confidence threshold
                # Convert YOLO format to (xmin, ymin, xmax, ymax)
                xmin = int(x - width / 2)
                ymin = int(y - height / 2)
                xmax = int(x + width / 2)
                ymax = int(y + height / 2)

                # Draw the bounding box
                color = (0, 255, 0)  # Green for bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

                # Annotate with class and confidence
                label = f"Class: {int(class_id)} Conf: {confidence:.2f}"
                cv2.putText(image, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the result
        cv2.imshow('YOLO Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the image with predictions
        # cv2.imshow("Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):

        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):

        return super().evaluate(test_path)


# Main code block
if __name__ == '__main__':
    """ # TESTING WITH OUR MODEL
    print('starting testing with YOLO model')
    # TESTING YOLO MODEL
    print('starting testing with YOLO model')
    # start_time = time.time()
    # predictor = OurModel()
    predictor = YoloModel()
    # end_time = time.time()
    # print(f'loaded model in {(end_time - start_time):.4f}')
    start_time = time.time()
    img = cv2.imread("data/sampleimg2.jpg")
    bots = predictor.predict(img, show = True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'elapsed time: {elapsed:.4f}')
    print(bots) """

    # pred_img = predictor.display_predictions(img)

    # # Display the resulting image
    # cv2.imshow("Prediction", pred_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # bots = predictor.detect_bot(img)
    # end_time = time.time()
    # elapsed = end_time - start_time
    # print(f'elapsed time: {elapsed:.4f}')
    # print(bots)

    # Testing Onnx Model
    print('starting testing with Onnx model')
    # start_time = time.time()
    # predictor = OurModel()
    # predictor = OnnxModel()
    predictor = PTModel()
    # end_time = time.time()
    # print(f'loaded model in {(end_time - start_time):.4f}')
    start_time = time.time()
    img = cv2.imread('12567_png.rf.6bb2ea773419cd7ef9c75502af6fe808.jpg')
    cv2.imshow("original image", img)
    cv2.waitKey(0)
    bots = predictor.predict(img, show=True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'elapsed time: {elapsed:.4f}')
    # print(len(bots[0][0][0]))
    # for i in range(6):
    #     print(bots[0][0][i][0])
    # with open("tensoroutput.txt", "w") as file:
    #     file.write(str(bots))
#   # Write to the file
#         # for row in bots[0][0][0]:
#         #     val = float(row)
#         #     file.write(f'{val:.4f}\n')
#         for row0 in bots:
#             file.write("[\n")
#             for row1 in row0:
#                 file.write("[\n")
#                 for row2 in row1:
#                     file.write("[\n")
#                     file.write(str(row2))
#                     file.write("\n]")
#                 file.write("\n]")
#             file.write("\n]")
    predictor.show_predictions(img, bots)
