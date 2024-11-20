import torch
import cv2
from torchvision import transforms
import numpy as np
import time
from roboflow import Roboflow
from inference import get_model
import os
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

DEBUG = True

class OurModel:
    def __init__(self, model_path="models/model_20241113_000141.pth"):
        # Load the model once during initialization
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict(self, img):
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            output = self.model(img_tensor)
        
        return output
    
    def draw_prediction(self, img, confidences, bboxes, confidence_threshold=0.5):
        height, width, _ = img.shape

        for i in range(len(confidences)):
            confidence, bbox = confidences[i], bboxes[i]
            
            # Determine class label
            class_label = 0 if confidence[0] > confidence[1] else 1
            
            # Only draw boxes if confidence is above threshold
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

            if DEBUG: 
                print(f'Class label {class_label} with confidence {confidence}')
                print(f'Bounding box: top-left: ({x_min}, {y_min}), bottom-right: ({x_max}, {y_max})')

            # Choose color based on class
            color = (0, 0, 255) if class_label == 0 else (0, 255, 0)
            
            # Draw bounding box and label on the image
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img, f"Class {class_label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

    def detect_bot(self, img, confidence_threshold=0.5):
        bots = {}
        confidences, bboxes, _ = self.predict(img)
        height, width, _ = img.shape

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

            # Write to dictionary
            name = ('bot' if class_label == 1 else 'housebot') + str(i)
            bots[name] = {'bbox':[[x_min, y_min], [x_max, y_max]], 'center':[center_x, center_y], 'img':screenshot}
        
        return bots

class YoloModel1:
    def __init__(self):
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY) # TODO: NOT HAVE THIS, THIS IS BAD
        self.project = self.rf.workspace("crc-autonomous").project("nhrl-robots")
        self.version = self.project.version(6)
        self.dataset = self.version.download("yolov8")
        self.model = self.version.model

    def predict(self, img_path, confidence_threshhold=40, overlap=90, display=False):
        if display:
            self.model.predict(img_path, confidence=confidence_threshhold, overlap=overlap, classes='robot').save("prediction.jpg")
        else:
            return self.model.predict(img_path, confidence=confidence_threshhold, overlap=overlap, classes='robot')
        
class YoloModel2:
    def __init__(self):
        self.model = get_model(model_id='nhrl-robots/6', api_key=ROBOFLOW_API_KEY)

    def predict(self, img):
        return self.model.infer(img)
    
    def display_predictions(self, img, confidence_threshold=0.5):
        # Run predictions
        response = self.predict(img)  # Predict returns an ObjectDetectionInferenceResponse object
        if DEBUG: print(response)
        response = response[0]
        if DEBUG: print(response)

        # Ensure the response has predictions
        if not hasattr(response, "predictions") or not response.predictions:
            print("No predictions found.")
            return img

        # Extract image dimensions
        height, width, _ = img.shape

        # Iterate over ObjectDetectionPrediction objects
        for prediction in response.predictions:
            # Extract bounding box and confidence
            confidence = prediction.confidence
            class_name = prediction.class_name
            x, y, box_width, box_height = prediction.x, prediction.y, prediction.width, prediction.height
            if DEBUG: print(f'x: {x}, y: {y}, box_width: {box_width}, box_height: {box_height}')

            if confidence < confidence_threshold:
                continue  # Skip if below threshold

            # Calculate bounding box coordinates
            x_min = int(x - box_width / 2)
            y_min = int(y - box_height / 2)
            x_max = int(x + box_width / 2)
            y_max = int(y + box_height / 2)

            # Choose color based on class
            color = (0, 255, 0) if class_name == "robot" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add class label and confidence
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img


# Main code block
if __name__ == '__main__':
    # USING OUR MODEL
    # # Initialize the predictor
    # predictor = OurModel()

    # # Load image
    # img = cv2.imread("data/sampleimg2.jpg")
    
    # # SINGLE PREDICTION
    # start_time = time.time()
    # out = predictor.predict(img)
    # end_time = time.time()
    # elapsed = end_time - start_time
    # print(f'elapsed time: {elapsed:.4f}')
    # confidences, bboxes, values = out
    
    # if DEBUG: 
    #     print(f'len: {len(out)}')
    #     print(f'tensor 1: {confidences}')
    #     print(f'tensor 2: {bboxes}')
    #     print(f'tensor 3: len {len(values)}, is: {values}')
    
    # # Draw predictions on the image
    # pred_img = predictor.draw_prediction(img, confidences, bboxes, confidence_threshold=0.1)

    # # Display the resulting image
    # cv2.imshow("Prediction", pred_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # DETECT_BOT
    # # bots = predictor.detect_bot(img, 0.1)
    # # print(bots)
    # # cv2.destroyAllWindows()


    # -----------------------------------------------
    # TESTING YOLO MODEL v1
    # print('starting testing with YOLO model')
    # # start_time = time.time()
    # predictor = YoloModel1()
    # # end_time = time.time()
    # # print(f'loaded model in {(end_time - start_time):.4f}')
    # start_time = time.time()
    # img = cv2.imread("data/sampleimg2.jpg")
    # out = predictor.predict(img_path="data/sampleimg2.jpg")
    # end_time = time.time()
    # elapsed = end_time - start_time
    # print(f'elapsed time: {elapsed:.4f}')
    # print(out)

    # -----------------------------------------------
    # TESTING YOLO MODEL v2
    print('starting testing with YOLO model')
    # start_time = time.time()
    predictor = YoloModel2()
    # end_time = time.time()
    # print(f'loaded model in {(end_time - start_time):.4f}')
    start_time = time.time()
    img = cv2.imread("data/sampleimg2.jpg")
    out = predictor.predict(img)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'elapsed time: {elapsed:.4f}')
    print(out)

    # pred_img = predictor.display_predictions(img)

    # # Display the resulting image
    # cv2.imshow("Prediction", pred_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()