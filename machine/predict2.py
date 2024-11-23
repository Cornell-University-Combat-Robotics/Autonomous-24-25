import torch
import cv2
from torchvision import transforms
import numpy as np

DEBUG = True

class OurModel:
    def __init__(self, model_path="models/model_20241123_115700.pth"): 
        # Load the model once during initialization
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict(self, img):
    # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension

        # Run model inference
        with torch.no_grad():
            class_probs, bbox_pred = self.model(img_tensor)

        # class_probs: [1, num_objects, num_classes]
        # bbox_pred: [1, num_objects, 4]

        # Remove batch dimension
        class_probs = class_probs.squeeze(0)  # [num_objects, num_classes]
        bbox_pred = bbox_pred.squeeze(0)      # [num_objects, 4]

        # Get predicted classes and confidence scores
        confidence_scores, predicted_classes = torch.max(class_probs, dim=-1)  # [num_objects]

        # Separate housebot and robot predictions
        housebot_mask = predicted_classes == 0  # Assuming 0 is the class index for housebot
        robot_mask = predicted_classes == 1     # Assuming 1 is the class index for robot

        # Select predictions for housebot
        housebot_scores = confidence_scores[housebot_mask]
        housebot_boxes = bbox_pred[housebot_mask]

        # Select top 1 housebot
        if housebot_scores.numel() > 0:
            top_housebot_scores, top_indices = torch.topk(housebot_scores, k=1)
            top_housebot_boxes = housebot_boxes[top_indices]
        else:
            top_housebot_scores = torch.tensor([])
            top_housebot_boxes = torch.tensor([])

        # Select predictions for robots
        robot_scores = confidence_scores[robot_mask]
        robot_boxes = bbox_pred[robot_mask]

        # Select top 2 robots
        if robot_scores.numel() > 0:
            k = min(2, robot_scores.size(0))  # Ensure k does not exceed available predictions
            top_robot_scores, top_indices = torch.topk(robot_scores, k=k)
            top_robot_boxes = robot_boxes[top_indices]
        else:
            top_robot_scores = torch.tensor([])
            top_robot_boxes = torch.tensor([])

        # Combine top predictions
        final_boxes = torch.cat([top_housebot_boxes, top_robot_boxes], dim=0) if top_housebot_boxes.numel() > 0 and top_robot_boxes.numel() > 0 else torch.tensor([])
        final_scores = torch.cat([top_housebot_scores, top_robot_scores], dim=0) if top_housebot_scores.numel() > 0 and top_robot_scores.numel() > 0 else torch.tensor([])

        return final_boxes, final_scores

    
    def draw_prediction(self, img, confidences, bboxes, confidence_threshold=0.5):
        height, width, _ = img.shape

        for i in range(len(confidences)):
            confidence, bbox = confidences[i], bboxes[i]
            class_label = torch.argmax(confidence).item()
            class_confidence = confidence[class_label]
                        
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
    img = cv2.imread("data/NHRL/test/images/11269_png.rf.9476b224e81e5090dc0b1555402b45c2.jpg")
    
    # Run prediction
    out = predictor.predict(img)
    confidences, bboxes = out
    confidences = torch.softmax(confidences, dim=-1)
    bboxes = torch.clamp(bboxes, min=0, max=600)
    
    if DEBUG: 
        print(f'len: {len(out)}')
        print(f'tensor 1: {confidences}')
        print(f'tensor 2: {bboxes}')    
    # Draw predictions on the image
    pred_img = predictor.draw_prediction(img, confidences, bboxes, confidence_threshold=0.5)

    # Display the resulting image
    cv2.imshow("Prediction", pred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
