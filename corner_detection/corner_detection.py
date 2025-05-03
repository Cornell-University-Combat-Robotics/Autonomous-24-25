import math
import os
import cv2
import numpy as np
from PIL import Image


class RobotCornerDetection:
    """
    A class for detecting the corners and orientation of robots in images
    based on their unique colors and shapes.
    """
    
    def __init__(self, selected_colors, display_final_image=False, display_possible_hueys=False):
        """
        Initializes the RobotCornerDetection class.

        Args:
            bots (list): A list of dictionary containing information about the 
                        bots, including bounding boxes and images. This is empty 
                        for now when initialized before the match. There is a
                        setter that will run during the match.
            selected_colors (list): Manually selected colors for front and back corners.
            display_final_image (bool): Whether to display the final image with
                        labeled left and right corners.
            display_possible_hueys (bool): Whether to display all possible
                        images of Huey.
            IS_FLIPPED (int): This is -1 when the bot is flipped upside down and
                        1 when the bot is right side up
        """
        self.bots = []
        self.selected_colors = selected_colors
        self.display_final_image = display_final_image
        self.display_possible_hueys = display_possible_hueys
        self.IS_FLIPPED = 1

    def set_bots(self, bots: list): # list of dictionaries
        self.bots = bots

    @staticmethod
    def find_bot_color_pixels(image: np.ndarray, bot_color_hsv: list):
        """
        Detects the number of a predefined color pixels in the given image.

        Args:
            image (np.ndarray): Input image of the robot in BGR format.

        Returns:
            int: The number of predefined color pixels detected in the image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the HSV range for the robot's color
        lower_limit = np.array([max(0, bot_color_hsv[0] - 10), 50, 50])
        upper_limit = np.array([min(179, bot_color_hsv[0] + 10), 255, 255])

        # Create a mask for the robot's color in the image
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
            self.selected_colors[1] if side == "front" else self.selected_colors[2]
        )

        # Define the HSV range around the selected color
        # We tried using 10 for the range; It was too large and picked up orange instead of red
        # For now, it is +-8
        lower_limit = np.array([max(0, selected_color[0] - 10), 20, 20])
        upper_limit = np.array([min(179, selected_color[0] + 10), 255, 255])

        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def find_our_bot(self, images: list[np.ndarray], bot_color_hsv, color: str):
        """
        Identifies which image contains our robot based on a predefined robot color.

        Args:
            images (list[np.ndarray]): List of input images.

        Returns:
            np.ndarray: The image containing our robot.
        """
        try:
            if not images:
                raise ValueError("The input image list is empty.")
            
            max_color_pixels = -1
            our_bot_image = None

            for image in images:
                
                total_pixels = np.shape(image)[0]*np.shape(image)[1]
                print(total_pixels)

                if image is None:
                    print("Warning: One of the images is None, skipping...")
                    continue

                color_pixel_count = self.find_bot_color_pixels(image, bot_color_hsv)
                print(color, color_pixel_count)

                if color_pixel_count > max_color_pixels:
                    if color == "Top" and color_pixel_count > total_pixels*0.03:
                        max_color_pixels = color_pixel_count
                        our_bot_image = image
                    elif color == "Bottom" and color_pixel_count > total_pixels*0.05:
                        max_color_pixels = color_pixel_count
                        our_bot_image = image
            
            if our_bot_image is None:
                print(color + " color Huey is not found")
                
            return our_bot_image, max_color_pixels
        
        except Exception as e:
            print(f"Unexpected error occurred in find_our_bot: {e}")
            return None
    
    def detect_our_robot_main(self, bot_images: list[np.ndarray]):
        """
        Detects the image containing our robot between two or more given images.

        Args:
            bot1_image (np.ndarray): The first bot image.
            bot2_image (np.ndarray): The second bot image.

        Returns:
            np.ndarray: The image identified as containing our robot.
        """
        try:
            if self.display_possible_hueys:
                window_width = 300
                window_height = 300

                for i, img in enumerate(bot_images):
                    if img is not None:
                        try:
                            resized_img = cv2.resize(img, (window_width, window_height))
                            cv2.imshow(f"Bot Image {i + 1}", resized_img)
                        except cv2.error as e:
                            print(f"Error resizing or displaying image {i + 1}: {e}")
                            continue
                    else:
                        print(f"Image {i + 1} is None")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if bot_images and all(img is not None for img in bot_images):

                top_color = self.selected_colors[0] # GREEN
                bottom_color = self.selected_colors[-1] # PURPLE

                our_bot_top, top_pixels = self.find_our_bot(bot_images, top_color, "Top") # Green
                our_bot_bottom, bottom_pixels = self.find_our_bot(bot_images, bottom_color, "Bottom") # Purple

                if our_bot_top is not None and our_bot_bottom is not None:
                    if top_pixels > bottom_pixels:
                        our_bot = our_bot_top
                        self.IS_FLIPPED = 1
                    else:
                        our_bot = our_bot_bottom
                        self.IS_FLIPPED = -1
                else:
                    if our_bot_bottom is not None:
                        our_bot = our_bot_bottom
                        self.IS_FLIPPED = -1
                    elif our_bot_top is not None:
                        our_bot = our_bot_top
                        self.IS_FLIPPED = 1
                    else:
                        our_bot = None

                return our_bot
            else:
                print("No valid bot images found.")
                return None
        
        except Exception as e:
            print(f"Unexpected error in detect_our_robot_main: {e}")
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
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        centroids = []
        for contour in contours:
            # Filter out small contours based on area
            area = cv2.contourArea(contour)
            print("Area", area)
            if area > 10:
                # TODO: this value is subject to change based on dimensions of our video & resize_factor
                # Compute moments for each contour
                M = cv2.moments(contour)
                if M["m00"] != 0 and len(centroids) < 2:
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

        # Ensure we have exactly 2 points for front and back
        if len(centroid_front) < 2 or len(centroid_back) < 2:
            return np.array([[], []])  # Return empty arrays if not enough points

        # Convert to numpy arrays with consistent shape
        front_array = np.array(centroid_front[:2])  # Take first 2 points if more exist
        back_array = np.array(centroid_back[:2])    # Take first 2 points if more exist
        
        return np.array([front_array, back_array])

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
        try:
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
                    red_points.append((int(new_red_point[0]), int(new_red_point[1])))
                else:
                    # Copy the blue point associated with length_b near the red point
                    new_red_point = (
                        red_point[0] + (blue_points[1][0] - blue_points[0][0]),
                        red_point[1] + (blue_points[1][1] - blue_points[0][1]),
                    )
                    red_points.append((int(new_red_point[0]), int(new_red_point[1])))

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
                    blue_points.append((int(new_blue_point[0]), int(new_blue_point[1])))
                else:
                    # Copy the red point associated with length_b near the blue point
                    new_blue_point = (
                        blue_point[0] + (red_points[1][0] - red_points[0][0]),
                        blue_point[1] + (red_points[1][1] - red_points[0][1]),
                    )
                    blue_points.append((int(new_blue_point[0]), int(new_blue_point[1])))

            return [red_points, blue_points]
        
        except Exception as e:
            print(f"Unexpected error in get_missing_point: {e}")
            return [[], []]

    @staticmethod
    def compute_tangent_angle(p1: tuple, p2: tuple): #NOTE: does not compute tangent angle anymore
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
    
    @staticmethod
    def compute_angle_between_midpoints(p1: tuple, p2: tuple):
        """
        Computes the angle of the line between the front and back corners of robot.

        Args:
            p1 (tuple): The front midpoint (x1, y1).
            p2 (tuple): The back midpoint (x2, y2).

        Returns:
            float: The angle of the line between the points relative to the x-axis in degrees.
        """
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = -(y2 - y1)
        angle_rad = np.arctan2(dy, dx)
        return math.degrees(angle_rad) % 360

    def get_left_and_right_front_points(self, points: list):
        """
        Determines the left and right front points of the robot.

        Args:
            points (list): A list containing red and blue points.

        Returns:
            list: The left and right front points of the robot.
        """
        try:
            red_points = points[0]
            blue_points = points[1]

            #TODO: check that this runs with any three points, change error
            # Ensure there are exactly two red points and at least one blue point
            if (len(red_points) + len(blue_points) < 3):
                raise ValueError("Expected exactly 2 red points and at least 1 blue point.") #TODO

            all_points = red_points + blue_points
            center = np.mean(all_points, axis=0)
            # print("center: " + str(center))

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

        except Exception as e:
            print(f"Unexpected error in get_left_and_right_front_points: {e}")
            return [None, None]

    def corner_detection_main(self):
        """
        Main function for detecting corners and orientation of the robot.

        Returns:
            dict: A dictionary containing details of the robot and enemy robots.
        """
        try:
            bot_images = [bot["img"] for bot in self.bots]
            image = self.detect_our_robot_main(bot_images)
            
            if image is not None:
                centroid_points = self.find_centroids(image)

                # For displaying centroids
                left_front, right_front = self.get_left_and_right_front_points(centroid_points)

                if left_front is None or right_front is None:
                    print("Could not determine left/right front points")
                    return {"huey": {}, "enemy": []}, self.IS_FLIPPED

                front_midpoint = (centroid_points[0][0] + centroid_points[0][1]) * 0.5

                back_midpoint = (centroid_points[1][0] + centroid_points[1][1]) * 0.5
                
                orientation = self.compute_angle_between_midpoints(back_midpoint, front_midpoint)

                # Find the identified bot (our robot)
                huey_bbox = None
                for bot_data in self.bots:
                    if bot_data["img"] is image:
                        huey_bbox = bot_data["bbox"]
                        break

                huey = {
                    "bbox": huey_bbox,
                    "center": np.mean(huey_bbox, axis=0),
                    "orientation": orientation,
                }

                # Enemy bots are all except the identified bot
                enemy_bots = []
                for bot_data in self.bots:
                    if bot_data["img"] is not image:
                        enemy = {
                            "bbox": bot_data["bbox"],
                            "center": np.mean(bot_data["bbox"], axis=0),
                        }
                        enemy_bots.append(enemy)

                result = {"huey": huey, "enemy": enemy_bots}
                
                if self.display_final_image:
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
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                return result, self.IS_FLIPPED
            else:
                print("Image doesn't exist")
                return {"huey": {}, "enemy": []}, self.IS_FLIPPED

        except Exception as e:
            print(f"Unexpected error in corner_detection_main: {e}")
            return None, self.IS_FLIPPED


if __name__ == "__main__":
    huey_image_path = os.getcwd() + "/warped_images/east_4.png"
    not_huey_image_path = os.getcwd() + "/warped_images/east_4_not_huey.png"
    selected_colors_file = os.getcwd() + "/selected_colors.txt"
    
    try:
        huey_image = cv2.imread(huey_image_path)
        not_huey_image = cv2.imread(not_huey_image_path)
        
        if huey_image is None:
            raise ValueError(f"Failed to load image at path: {huey_image_path}")
        if not_huey_image is None:
            raise ValueError(f"Failed to load image at path: {not_huey_image_path}")
    
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)

    housebot = {"bbox": [[0, 0], [1, 1]], "img": not_huey_image}
    bot1 = {"bbox": [[50, 50], [60, 60]], "img": not_huey_image}
    bot2 = {"bbox": [[150, 150], [160, 160]], "img": huey_image}
    bot3 = {"bbox": [[300, 150], [400, 180]], "img": not_huey_image}

    housebots = [housebot]
    bots = [bot1, bot2, bot3]
    all_bots = {"housebot": housebot, "bots": bots}
    
    selected_colors = []
    try:
        with open(selected_colors_file, "r") as file:
            for line in file:
                hsv = list(map(int, line.strip().split(", ")))
                selected_colors.append(hsv)
        if len(selected_colors) != 4:
            raise ValueError("The file must contain exactly 4 HSV values.")
    
    except Exception as e:
        print(f"Error reading selected_colors.txt: {e}")
        exit(1)
    
    corner_detection = RobotCornerDetection(selected_colors, True, False)
    corner_detection.set_bots(all_bots["bots"])
    result = corner_detection.corner_detection_main()