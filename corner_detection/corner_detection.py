import math
import os
import time
import cv2
import numpy as np
from color_picker import ColorPicker


class RobotCornerDetection:
    """
    A class for detecting the corners and orientation of robots in images
    based on their unique colors and shapes.
    """

    def __init__(self, bots, selected_colors, display_image=True):
        """
        Initializes the RobotCornerDetection class.

        Args:
            bots (dict): A dictionary containing information about the bots,
                        including bounding boxes and images.
            selected_colors (list): Manually selected colors for front and back corners.
            display_image (bool): Whether to display images during processing.
        """
        self.bots = bots
        self.selected_colors = selected_colors
        self.display_image = display_image

    @staticmethod
    def find_bot_color_pixels(image: np.ndarray):
        """
        Detects the number of bright pink pixels in the given image.

        Args:
            image (np.ndarray): Input image of the robot in BGR format.

        Returns:
            int: The number of bright pink pixels detected in the image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the HSV range for the robot's pink color
        lower_limit = np.array([140, 100, 100])
        upper_limit = np.array([170, 255, 255])

        # Create a mask for the pink color in the image
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        # Count the number of non-zero pixels in the mask
        return cv2.countNonZero(mask)

    def get_contours_per_color(self, side: str, hsv_image: np.ndarray):
        """
        Retrieves contours for the front or back corners based on the manually picked color.

        Args:
            side (str): "front" for red contours, "back" for blue contours.
            hsv_image (np.ndarray): Input image in HSV format.

        Returns:
            list: Contours corresponding to the given color.
        """
        selected_color = (
            self.selected_colors[0] if side == "front" else self.selected_colors[1]
        )

        # Define the HSV range around the selected color
        # We tried using 10 for the range; It was too large and picked up orange instead of red
        # For now, it is +-8
        lower_limit = np.array([max(0, selected_color[0] - 8), 100, 100])
        upper_limit = np.array([min(179, selected_color[0] + 8), 255, 255])

        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_our_bot(self, image1: np.ndarray, image2: np.ndarray):
        """
        Identifies which image contains our robot based on bright pink pixel count.

        Args:
            image1 (np.ndarray): The first input image.
            image2 (np.ndarray): The second input image.

        Returns:
            np.ndarray: The image containing our robot.
        """
        image_1_pixels = self.find_bot_color_pixels(image1)
        image_2_pixels = self.find_bot_color_pixels(image2)
        return image1 if image_1_pixels > image_2_pixels else image2

    def detect_our_robot_main(self, bot1_image: np.ndarray, bot2_image: np.ndarray):
        """
        Detects the image containing our robot between two given images.

        Args:
            bot1_image (np.ndarray): The first bot image.
            bot2_image (np.ndarray): The second bot image.

        Returns:
            np.ndarray: The image identified as containing our robot.
        """
        if bot1_image is not None and bot2_image is not None:
            start_time = time.time()
            our_bot = self.find_our_bot(bot1_image, bot2_image)
            end_time = time.time()
            print(
                f"Code execution time (detect_our_robot): {end_time - start_time} seconds"
            )
            return our_bot
        return None

    def find_centroids_per_color(
        self, side: str, image: np.ndarray, hsv_image: np.ndarray
    ):
        """
        Finds the centroids of a specific color (front or back) in the given image.

        Args:
            side (str): "front" or "back" for the color.
            image (np.ndarray): The input image in BGR format.
            hsv_image (np.ndarray): The HSV version of the input image.

        Returns:
            list: Centroids of the detected contours.
        """
        contours = self.get_contours_per_color(side, hsv_image)
        centroids = []
        for contour in contours:
            # Filter out small contours based on area
            area = cv2.contourArea(contour)
            if area > 200:
                # TODO: this value is subject to change based on the size of our bot's corners
                # Compute moments for each contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    # Calculate the centroid (center of the dot)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    cv2.circle(image, (cx, cy), 8, (0, 0, 0), -1)
                    cv2.putText(
                        image,
                        side,
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )
        return centroids

    def find_centroids(self, image: np.ndarray):
        """
        Finds the centroids for the front and back corners of the robot.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            list: A list containing centroids for the front and back corners.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        centroid_front = self.find_centroids_per_color("front", image, hsv_image)
        centroid_back = self.find_centroids_per_color("back", image, hsv_image)

        # Check if we have incomplete points and use get_missing_point to fix it
        if len(centroid_front) == 1 and len(centroid_back) == 2:
            points = [centroid_front, centroid_back]
            centroid_front, centroid_back = self.get_missing_point(points)
        elif len(centroid_back) == 1 and len(centroid_front) == 2:
            points = [centroid_front, centroid_back]
            centroid_front, centroid_back = self.get_missing_point(points)

        return [centroid_front, centroid_back]

    def distance(self, point1: tuple, point2: tuple):
        """
        Calculates the Euclidean distance between two points.

        Args:
            point1 (tuple): The first point (x1, y1).
            point2 (tuple): The second point (x2, y2).

        Returns:
            float: The Euclidean distance.
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def get_missing_point(self, points: list):
        """
        Computes the missing point to form a complete set of red and blue points.

        Algorithm:
        - If given 2 blue points and 1 red point:
        1. Calculate the distance from each blue point to the red point.
        2. Identify the longer distance (hypotenuse).
        3. Copy the blue point associated with the hypotenuse near the red point
                to form the second red point.
        - If given 2 red points and 1 blue point:
        1. Calculate the distance from each red point to the blue point.
        2. Identify the longer distance (hypotenuse).
        3. Copy the red point associated with the hypotenuse near the blue point
                to form the second blue point.

        Args:
                points (list): A list containing two sublists:
                                        - points[0]: List of red points.
                                        - points[1]: List of blue points.

        Returns:
                list: A list containing updated red and blue points.
        """
        red_points = points[0]
        blue_points = points[1]

        if len(red_points) == 1 and len(blue_points) == 2:
            # Case #1: 1 red point and 2 blue points
            red_point = red_points[0]
            length_a = self.distance(blue_points[0], red_point)
            length_b = self.distance(blue_points[1], red_point)

            # Identify which blue point is associated with the hypotenuse
            if length_a > length_b:
                # Copy the blue point associated with length_a near the red point
                new_red_point = (
                    red_point[0] + (blue_points[0][0] - blue_points[1][0]),
                    red_point[1] + (blue_points[0][1] - blue_points[1][1]),
                )
                red_points.append(new_red_point)
            else:
                # Copy the blue point associated with length_b near the red point
                new_red_point = (
                    red_point[0] + (blue_points[1][0] - blue_points[0][0]),
                    red_point[1] + (blue_points[1][1] - blue_points[0][1]),
                )
                red_points.append(new_red_point)

        elif len(blue_points) == 1 and len(red_points) == 2:
            # Case #2: 2 red points and 1 blue point
            blue_point = blue_points[0]
            length_a = self.distance(red_points[0], blue_point)
            length_b = self.distance(red_points[1], blue_point)

            # Identify which red point is associated with the hypotenuse
            if length_a > length_b:
                # Copy the red point associated with length_a near the blue point
                new_blue_point = (
                    blue_point[0] + (red_points[0][0] - red_points[1][0]),
                    blue_point[1] + (red_points[0][1] - red_points[1][1]),
                )
                blue_points.append(new_blue_point)
            else:
                # Copy the red point associated with length_b near the blue point
                new_blue_point = (
                    blue_point[0] + (red_points[1][0] - red_points[0][0]),
                    blue_point[1] + (red_points[1][1] - red_points[0][1]),
                )
                blue_points.append(new_blue_point)

        return [red_points, blue_points]

    @staticmethod
    def compute_tangent_angle(p1: tuple, p2: tuple):
        """
        Computes the angle of the tangent line to the front of the robot.

        Args:
            p1 (tuple): The first front point (x1, y1).
            p2 (tuple): The second front point (x2, y2).

        Returns:
            float: The angle of the tangent line relative to the x-axis in degrees.
        """
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = -(y2 - y1)
        angle_rad = np.arctan2(dy, dx)
        tangent_angle_rad = angle_rad + np.pi / 2
        return math.degrees(tangent_angle_rad) % 360

    def get_left_and_right_front_points(self, points: list):
        """
        Determines the left and right front points of the robot.

        Args:
            points (list): A list containing red and blue points.

        Returns:
            list: The left and right front points of the robot.
        """
        red_points = points[0]
        blue_points = points[1]

        all_points = red_points + blue_points
        center = np.mean(all_points, axis=0)
        print("center: " + str(center))

        vector1 = np.array(red_points[0]) - center
        vector2 = np.array(red_points[1]) - center

        # We do this because in code, positive y is downward and we want to make it upward
        vector1[1] = -vector1[1]
        vector2[1] = -vector2[1]

        theta1 = math.atan2(vector1[1], vector1[0])
        theta2 = math.atan2(vector2[1], vector2[0])

        theta1_deg = (
            math.degrees(theta1)
            if math.degrees(theta1) >= 0
            else math.degrees(theta1) + 360
        )
        theta2_deg = (
            math.degrees(theta2)
            if math.degrees(theta2) >= 0
            else math.degrees(theta2) + 360
        )

        print("theta1_deg: " + str(theta1_deg))
        print("theta2_deg: " + str(theta2_deg))

        # Determine which red point is the top right front corner
        if theta2_deg - theta1_deg > 235:
            right_front = red_points[1]
            left_front = red_points[0]
        elif theta1_deg - theta2_deg > 235:
            right_front = red_points[0]
            left_front = red_points[1]
        elif abs(theta2_deg - theta1_deg) > 180:
            right_front = red_points[0]
            left_front = red_points[1]
        elif theta2_deg > theta1_deg:
            # The point with the smaller angle is the top right front corner
            right_front = red_points[0]
            left_front = red_points[1]
        else:
            # The point with the larger angle is the top right front corner
            right_front = red_points[1]
            left_front = red_points[0]

        return [left_front, right_front]

    def corner_detection_main(self):
        """
        Main function for detecting corners and orientation of the robot.

        Returns:
            dict: A dictionary containing details of the robot and enemy robot.
        """
        bot1_image = self.bots["bot1"]["img"]
        bot2_image = self.bots["bot2"]["img"]
        image = self.detect_our_robot_main(bot1_image, bot2_image)
        if image is not None:
            centroid_points = self.find_centroids(image)
            left_front, right_front = self.get_left_and_right_front_points(
                centroid_points
            )
            orientation = self.compute_tangent_angle(left_front, right_front)
            huey = {
                "bb": self.bots["bot1"]["bb"],
                "center": np.mean(self.bots["bot1"]["bb"], axis=0),
                "orientation": orientation,
            }
            enemy = {
                "bb": self.bots["bot2"]["bb"],
                "center": np.mean(self.bots["bot2"]["bb"], axis=0),
            }
            result = {"huey": huey, "enemy": enemy}
            print("Result:", result)

            if self.display_image:
                # Draw the left front corner
                cv2.circle(
                    image,
                    (int(left_front[0]), int(left_front[1])),
                    5,
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(
                    image,
                    "Left Front",
                    (int(left_front[0]), int(left_front[1]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Draw the right front corner
                cv2.circle(
                    image,
                    (int(right_front[0]), int(right_front[1])),
                    5,
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(
                    image,
                    "Right Front",
                    (int(right_front[0]), int(right_front[1]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Display the image
                cv2.imshow("Image with Left and Right Front Corners", image)
                cv2.imwrite("image_with_front_corners.png", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return result


if __name__ == "__main__":
    image_path = os.getcwd() + "/warped_images/west_3.png"
    selected_colors = ColorPicker.pick_colors(image_path)

    image1 = cv2.imread(image_path)
    image2 = cv2.imread(image_path)

    bots = {
        "housebot": {"bb": [[0, 0], [1, 1]], "img": image1},
        "bot1": {"bb": [[50, 50], [60, 60]], "img": image1},
        "bot2": {"bb": [[150, 150], [160, 160]], "img": image2},
    }

    corner_detection = RobotCornerDetection(bots, selected_colors)
    corner_detection.corner_detection_main()
