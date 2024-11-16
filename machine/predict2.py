import torch
import cv2
from torchvision import transforms
import numpy as np

DEBUG = True

class OurModel:
    def __init__(self, model_path="models/model_20241116_121621.pth"):
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
            
        confidences, bboxes, values = output
        
        return confidences, bboxes, values
    
    def draw_prediction(self, img, confidences, bboxes, confidence_threshold=0.5):
        height, width, _ = img.shape

        for i in range(len(confidences)):
            confidence, bbox = confidences[i], bboxes[i]
            class_label = torch.argmax(confidence).item()
            class_confidence = confidence[class_label]
            
            # Determine class label
            class_label = 0 if confidence[0] > confidence[1] else 1
            
            # Only draw boxes if confidence is above threshold
            if class_confidence < confidence_threshold:
                continue

            # # Extract center x, center y, box width, and box height
            # center_x = int(bbox[0] * width)
            # center_y = int(bbox[1] * height)
            # box_width = int(bbox[2] * width)
            # box_height = int(bbox[3] * height)

            # # Calculate bounding box corners
            # x_min = center_x - box_width // 2
            # y_min = center_y - box_height // 2
            # x_max = center_x + box_width // 2
            # y_max = center_y + box_height // 2
            x_min, y_min, x_max, y_max = map(int, bbox)

            if DEBUG: 
                print(f'Class label {class_label} with confidence {confidence}')
                print(f'Bounding box: top-left: ({x_min}, {y_min}), bottom-right: ({x_max}, {y_max})')

            # Choose color based on class
            color = (0, 0, 255) if class_label == 0 else (0, 255, 0)
            
            # Draw bounding box and label on the image
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img, f"Class {class_label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

# Main code block
if __name__ == '__main__':
    # Initialize the predictor
    predictor = OurModel()

    # Load image
    # img = cv2.imread("data/sampleimg2.jpg")
    img = cv2.imread("data/yolo_data_v1.yolov8/train/images/Image_1726619260161941_jpg.rf.984eefed8d2a58fc92a1d057f54070d6.jpg")
    
    # Run prediction
    out = predictor.predict(img)
    confidences, bboxes, values = out
    confidences = torch.softmax(confidences, dim=-1)
    bboxes = torch.clamp(bboxes, min=0, max=600)
    
    if DEBUG: 
        print(f'len: {len(out)}')
        print(f'tensor 1: {confidences}')
        print(f'tensor 2: {bboxes}')
        print(f'tensor 3: len {len(values)}, is: {values}')
    
    # Draw predictions on the image
    pred_img = predictor.draw_prediction(img, confidences, bboxes, confidence_threshold=0.1)

    # Display the resulting image
    cv2.imshow("Prediction", pred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
