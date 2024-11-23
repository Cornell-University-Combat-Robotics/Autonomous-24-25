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
    
    def predict(self, img, confidence_threshold=0.5, show=False):
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            output = self.model(img_tensor)
        
        bots = {}
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

            # Write to dictionary
            if class_label == 0:
                name = 'housebot'
            elif class_label == 1:
                name = 'bot' + str(robots)
                robots += 1
            bots[name] = {'bb':[[x_min, y_min], [x_max, y_max]], 'center':[center_x, center_y], 'img':screenshot}
        
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
    
class YoloModel:
    def __init__(self):
        self.model = get_model(model_id='nhrl-robots/6', api_key=ROBOFLOW_API_KEY)

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
                if DEBUG: print(f'x: {x}, y: {y}, box_width: {box_width}, box_height: {box_height}')

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

# Main code block
if __name__ == '__main__':
    # USING OUR MODEL
    # # Initialize the predictor
    predictor = OurModel()

    # # Load image
    img = cv2.imread("data/sampleimg2.jpg")
    
    # # SINGLE PREDICTION
    # start_time = time.time()
    out = predictor.predict(img, confidence_threshold=0.1, show=True)
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
    # TESTING YOLO MODEL
    # print('starting testing with YOLO model')
    # start_time = time.time()
    # predictor = YoloModel()
    # end_time = time.time()
    # print(f'loaded model in {(end_time - start_time):.4f}')
    # start_time = time.time()
    # img = cv2.imread("data/sampleimg2.jpg")
    # bots = predictor.predict(img, show=True)
    # end_time = time.time()
    # elapsed = end_time - start_time
    # print(f'elapsed time: {elapsed:.4f}')
    # print(bots)

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