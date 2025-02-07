import numpy as np


class TemplateModel:
    """
    A template for implementing object detection models. 

    Note that this class only serves as a template and all methods (init, predict, train, and evaluate) must be explicitly implemented in subclasses.
    Calling a method in this class without subclass implementation will result in an error.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        raise "Not implemented in subclass"

    def predict(self, img: np.ndarray, confidence_threshold: float = 0.5, show: bool = False) -> dict:
        """
        Uses the model to make a prediction on a singular image, [img].
        All predictions above [confidence_threshold] are added to the resulting prediction dictionary.
        If [show] is enabled, the resulting prediction is displayed.

        Parameters:
          img (np.ndarray): A warped, overhead image of the NHRL arena.
          confidence_threshold (float): All predictions above [confidence_threshold] are added to the prediction dictionary. Default is 0.5
          show (bool): Allows the user to display the resulting prediction. Default is False.

        Returns:
          dict: A dictionary conatining information about predicted objects in the input image. See example below.
            {
              "bots": [{"bb":[[top_left_x:int, top_left_y:int],[bottom_right_x:int, bottom_right_y:int]], "center":[center_x:int, center_y:int], "img":np.ndarry}, {...}]
              "housebots": [{"bb":[[top_left_x:int, top_left_y:int],[bottom_right_x:int, bottom_right_y:int]], "center":[center_x:int, center_y:int], "img":np.ndarry}]
            }
        """
        raise "Not implemented in subclass"

    def train(self, batch: int, epoch: int, train_path: str, validation_path: str, save_path: str, save: bool):
        """
        Trains a model for [batch] batches and [epoch] epochs. In [save] is enabled, after training the model is saved to [save_path].

        Parameters:
          batch (int): Number of bathces to train for.
          epoch (int): Number of epochs to train for.
          train_path (string): See formatting details below.
          validation_path (string): See formatting details below.
          save_path (string): The file path to optionally save the trained model.
          save (bool): A boolean specifying whether or not you want to save the trained model to [save_path].

        Path Info:
          [train_path] and [validation path] should each follow the following format:
            Each path should contain two subfolders, "images" and "labels." 
              "images" subfolder should contain warped, overhead images of the NHRL arena.
              "labels" subfolder should contain text files with lines of the following format where x_center, y_center, width, height and height are all normalized floats (between 0 and 1):
                label x_center y_center width height

        Returns:
          some trained model

        """
        raise "Not implemented in subclass"

    def evaluate(self, test_path: str) -> None:
        """
        Evaluates the model against the images and corresponding labels in [test_path].

        Parameters:
          test_path (string): A file path containing two subfolders, "images" and "labels." 
            "images" subfolder should contain warped, overhead images of the NHRL arena.
            "labels" subfolder should contain text files with lines of the following format where x_center, y_center, width, height and height are all normalized floats (between 0 and 1):
              label x_center y_center width height

        Returns:
          None: The function should print appropriate metrics (ex. accuracy)
        """
        raise "Not implemented in subclass"
