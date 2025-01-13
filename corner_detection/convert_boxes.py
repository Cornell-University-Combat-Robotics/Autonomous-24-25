import cv2
import numpy as np
import os
import pprint

def convert_boxes_to_format(json_data, image_path):
    """
    Converts the given JSON data into the specified bot format.

    Args:
        json_data (dict): JSON data containing bounding box information.
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary with the desired bot format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    bot_dict = {}

    for idx, box in enumerate(json_data["boxes"]):
        label = box["label"]
        x_center = float(box["x"])
        y_center = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
        
        top_left = [x_center - width / 2, y_center - height / 2]
        bottom_right = [x_center + width / 2, y_center + height / 2]

        key = f"bot{idx}" if label != "housebot" else "housebot"

        bot_dict[key] = {
            "center": [x_center, y_center],
            "bb": [top_left, bottom_right],
            "img": image
        }

    return bot_dict

def print_result_without_image(result):
    """
    Prints the result dictionary without the image data.

    Args:
        result (dict): The result dictionary containing bot information.
    """
    result_without_images = {
        key: {k: v for k, v in value.items() if k != "img"}
        for key, value in result.items()
    }
    pprint.pprint(result_without_images, sort_dicts=False, indent=2)

if __name__ == "__main__":
    # Dictionary and Image is taken from Roboflow
    json_data = {
        "boxes": [
            {"id": "1", "label": "housebot", "x": "494.23", "y": "527.51", "width": "133.33", "height": "119.37"},
            {"id": "2", "label": "robot", "x": "298.77", "y": "116.03", "width": "61.96", "height": "85.73"},
            {"id": "3", "label": "robot", "x": "443.46", "y": "44.68", "width": "81.72", "height": "89.36"},
            {"id": "4", "label": "robot", "x": "557.94", "y": "397.78", "width": "68.41", "height": "94.65"},
            {"id": "5", "label": "robot", "x": "344.08", "y": "352.50", "width": "75.24", "height": "82.32"}
        ],
        "height": 600,
        "key": "8968.png",
        "width": 600
    }

    image_path = os.getcwd() + "/json_arena_pics/8968.png"
    result = convert_boxes_to_format(json_data, image_path)
    print_result_without_image(result)