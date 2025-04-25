import os
import cv2
import numpy as np
import onnxruntime as ort
import openvino as ov
import pandas as pd
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

# from template_model import TemplateModel # to run in machine
from machine.template_model import TemplateModel  # to run in main

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

DEBUG = False

# I think this is broken... b/c of the model tho not the


class OurModel(TemplateModel):
    def __init__(self, model_path="models/model_20241113_000141.pth"):
        # Load the model once during initialization
        self.model = torch.load(model_path)
        # Set to evaluation mode, so that we won't train new params
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def predict(self, img, confidence_threshold=0.5, show=False):
        # Preprocess image via the transformation
        img_tensor = self.transform(img).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            output = self.model(img_tensor)

        bots = {}
        bots["bots"] = []
        bots["housebot"] = []
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
            center_x = max(0, int(bbox[0] * width))
            center_y = max(0, int(bbox[1] * height))
            box_width = max(0, int(bbox[2] * width))
            box_height = max(0, int(bbox[3] * height))

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
                bots["housebot"].append(
                    {
                        "bbox": [[x_min, y_min], [x_max, y_max]],
                        "center": [center_x, center_y],
                        "img": screenshot,
                    }
                )
            elif class_label == 1:
                bots["bots"].append(
                    {
                        "bbox": [[x_min, y_min], [x_max, y_max]],
                        "center": [center_x, center_y],
                        "img": screenshot,
                    }
                )

        if show:
            self.show_predictions(img, bots)

        return bots

    def show_predictions(self, img, predictions):
        # Display housebot
        housebots = predictions["housebot"]
        bots = predictions["bots"]

        color = (0, 0, 255)
        for housebot in housebots:
            x_min, y_min = housebot["bbox"][0]
            x_max, y_max = housebot["bbox"][1]
            center_x, center_y = housebot["center"]

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img,
                "housebot",
                (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        color = (0, 255, 0)  # Green for bots
        for bot in bots:
            x_min, y_min = bot["bbox"][0]
            x_max, y_max = bot["bbox"][1]
            center_x, center_y = bot["center"]

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img,
                "bot",
                (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
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
        #         (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, color, 2
        #     )

        # Display the image with predictions
        cv2.imshow("OurModel Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):
        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):
        return super().evaluate(test_path)

# No clue if this works


class RoboflowModel(TemplateModel):
    def __init__(self):
        self.model = get_model(model_id="nhrl-robots/14", api_key=ROBOFLOW_API_KEY)

    def predict(self, img, confidence_threshold=0.5, show=False, track=False):
        out = self.model.infer(img)

        bots = {}
        bots["housebot"] = []
        bots["bots"] = []

        preds = out[0].predictions

        housebot_candidates = []

        robots = 1
        for pred in preds:
            if pred.confidence > confidence_threshold:
                p = {}
                if pred.class_id == 0:
                    name = "housebot"
                    housebot_candidates.append(pred)
                elif pred.class_id == 1:
                    name = "bots"

                x, y, box_width, box_height = pred.x, pred.y, pred.width, pred.height
                if DEBUG:
                    print(
                        f"x: {x}, y: {y}, box_width: {box_width}, box_height: {box_height}"
                    )

                p["center"] = [x, y]

                img_height, img_width = img.shape[:2]
                x_min = max(0, int(x - box_width / 2) - 20)
                y_min = max(0, int(y - box_height / 2) - 20)
                x_max = min(img_width, int(x + box_width / 2) + 20)
                y_max = min(img_height, int(y + box_height / 2) + 20)

                # Extract bounding box
                screenshot = img[y_min:y_max, x_min:x_max]

                if DEBUG:
                    cv2.imshow("Screenshot", screenshot)
                    cv2.waitKey(0)

                p["bbox"] = [[x_min, y_min], [x_max, y_max]]
                p["img"] = screenshot

                if name == "bots":
                    bots[name].append(p)

        # Select the most confident housebot
        if housebot_candidates:
            best_housebot = max(housebot_candidates, key=lambda pred: pred.confidence)

            # Process the most confident housebot
            x, y, box_width, box_height = (
                best_housebot.x,
                best_housebot.y,
                best_housebot.width,
                best_housebot.height,
            )
            p = {
                "center": [x, y],
                "bbox": [
                    [int(x - box_width / 2) - 20, int(y - box_height / 2) - 20],
                    [int(x + box_width / 2) + 20, int(y + box_height / 2) + 20],
                ],
                "img": img[
                    int(y - box_height / 2) - 20 : int(y + box_height / 2) + 20,
                    int(x - box_width / 2) - 20 : int(x + box_width / 2) + 20,
                ],
            }
            bots["housebot"].append(p)

        if show:
            self.show_predictions(img, bots)
        return bots

    def show_predictions(self, img, predictions):
        # Display housebot
        housebots = predictions["housebot"]
        bots = predictions["bots"]

        color = (0, 0, 255)  # Red for housebots
        for housebot in housebots:
            x_min, y_min = housebot["bbox"][0]
            x_max, y_max = housebot["bbox"][1]
            center_x, center_y = housebot["center"]

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img,
                "housebot",
                (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        color = (255, 255, 255)  # White for bots
        for bot in bots:
            x_min, y_min = bot["bbox"][0]
            x_max, y_max = bot["bbox"][1]
            center_x, center_y = bot["center"]

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(
                img,
                "bot",
                (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # print(f"Detected [{len(housebots)} housebots], [{len(bots)} bots]")

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
        #         (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, color, 2
        #     )

        # Display the image with predictions
        # cv2.imshow("Roboflow Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):
        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):
        return super().evaluate(test_path)


class YoloModel(TemplateModel):
    # General template for using YOLO to load model files and use them.

    def __init__(self, model_name, model_type, device=None):
        match model_type:
            case "TensorRT":
                # Works best on NVIDIA GPUs, engine file must be compiled on the PC that it is running on.
                model_extension = ".engine"
                self.model = YOLO("./machine/models/" + model_name + model_extension)
            case "ONNX":
                # Optimal for CPU performance
                model_extension = ".onnx"
                self.model = YOLO("./machine/models/" + model_name + model_extension)
            case "PT":
                # Default kinda
                model_extension = ".pt"
                self.model = YOLO("./machine/models/" + model_name + model_extension)
            case "OpenVIVO":
                # Optimal for Intel CPUs, needs a lil work
                model_extension = ".xml"
                weights_extension = ".bin"
                core = ov.Core()
                classification_model_xml = ("./machine/models/" + model_name + model_extension)
                weights = "./machine/models/" + model_name + weights_extension

                model = core.read_model(model=classification_model_xml, weights=weights)
                cmodel = core.compile_model(model=model)
                self.model = cmodel
        self.device = device
        # compiled_model = core.compile_model(model=model, device_name=device.value)

    def predict(self, img, show=False, track=False):
        # This prints timing info
        if self.device != None:
            results = self.model(img, device=self.device, verbose=False)
        else:
            results = self.model(img)
        # If multiple img passed, results has more than one element
        result = results[0]

        robots = []
        housebots = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy, width, height = box.xywh[0].tolist()
            cropped_img = img[int(y1) : int(y2), int(x1) : int(x2)]

            # cv2.imshow('image', cropped_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows

            dict = {"bbox": [[max(0, x1), max(0, y1)], [min(700, x2), min(700, y2)]], "center": [cx, cy], "img": cropped_img}

            if box.cls == 0:
                housebots.append(dict)
            else:
                robots.append(dict)

        out = {"bots": robots, "housebot": housebots}
        if show:
            self.show_predictions(img, out)

        return out

    def show_predictions(self, img, bots_dict):
        for label, bots in bots_dict.items():

            for bot in bots:

                # Extract bounding box coordinates and class details
                x_min, y_min = bot["bbox"][0]
                x_max, y_max = bot["bbox"][1]

                # Choose color based on the class
                if "housebot" in label:
                    color = (0, 0, 255)  # Red for housebot
                else:
                    color = (255, 255, 255)  # White for bots

                # Draw the bounding box
                cv2.rectangle(
                    img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2
                )

                # Add label text
                cv2.putText(
                    img,
                    label,
                    (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # cv2.imshow("YoloModel Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img


class OnnxModel(TemplateModel):
    def __init__(self):
        onnx_model_path = "./models/best.onnx"
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
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

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
                cv2.putText(
                    image,
                    label,
                    (int(xmin), int(ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Display the result
        cv2.imshow("YOLO Detection", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Display the image with predictions
        # cv2.imshow("Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):

        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):

        return super().evaluate(test_path)


# Main code block
if __name__ == "__main__":

    print("starting testing with PT model")
    # predictor = YoloModel("100epoch11","PT")
    predictor = RoboflowModel()

    img_path = (
        os.getcwd() + "/main_files/12567_png.rf.6bb2ea773419cd7ef9c75502af6fe808.jpg"
    )
    img = cv2.imread(img_path)

    # cv2.imshow("Original image", img)
    # cv2.waitKey(0)

    start_time = time.time()
    bots = predictor.predict(img, show=True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"elapsed time: {elapsed:.4f}")

    # predictor.show_predictions(img, bots)
