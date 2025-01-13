import cv2
import numpy as np
import os


class ColorPicker:
    """
    A class for manually picking colors from an image.
    """

    @staticmethod
    def pick_colors(image_path):
        """
        Allows the user to manually pick colors for front and back corners.
        Users must select the front point first before the back point.

        Args:
            image_path (str): Path to the image to pick colors from.

        Returns:
            list: Selected colors for front and back corners in HSV format.
        """
        test_img = cv2.imread(image_path)
        selected_colors = []
        corners = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color = test_img[y, x]  # OpenCV reads as BGR
                hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
                selected_colors.append(hsv_color)
                corners.append([x, y])
                print(f"Selected color (HSV): {hsv_color}")
                print(f"Point added: {x}, {y}")
                redraw_image()

        def redraw_image():
            img_copy = test_img.copy()
            for point in corners:
                cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
                cv2.putText(
                    img_copy,
                    f"{point}",
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
            cv2.imshow("Select Colors", img_copy)

        cv2.imshow("Select Colors", test_img)
        cv2.setMouseCallback("Select Colors", click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("z"):  # If 'z' is pressed
                if selected_colors and corners:
                    removed_color = selected_colors.pop()
                    removed_point = corners.pop()
                    print(f"Color removed: {removed_color}")
                    print(f"Point removed: {removed_point}")
                    redraw_image()
                else:
                    print("No points to remove.")
            elif key == 27:  # Press 'Esc' to exit
                break
            elif len(selected_colors) == 2:
                break

        print("Final Selected Colors (HSV):", selected_colors)
        print("Final Selected Points:", corners)
        cv2.destroyAllWindows()
        return selected_colors
    
if __name__ == "__main__":
    image_path = os.getcwd() + "/warped_images/east.png"
    output_file = "selected_colors.txt"
    selected_colors = ColorPicker.pick_colors(image_path)

    with open(output_file, "w") as file:
        for color in selected_colors:
            file.write(f"{color[0]}, {color[1]}, {color[2]}\n")

    print(f"Selected colors have been saved to '{output_file}'.")