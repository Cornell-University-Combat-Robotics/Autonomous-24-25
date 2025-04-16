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
        Allows the user to manually pick colors for the robot and the front and back corners.
        The user selects the robot's color first, followed by the front and back corners.

        Args:
            image_path (str): Path to the image to pick colors from.

        Returns:
            list: Selected colors for the robot, front corner, and back corner in HSV format.
        """
        try:
            test_img = cv2.imread(image_path)
            if test_img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return []

        selected_colors = []
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if x < 0 or y < 0 or x >= test_img.shape[1] or y >= test_img.shape[0]:
                    print(f"Clicked outside the image: ({x}, {y})")
                    return

                try:
                    color = test_img[y, x]  # OpenCV reads as BGR
                    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
                    if len(selected_colors) < 5:
                        selected_colors.append(hsv_color)
                        points.append([x, y])
                        print(f"Selected color (HSV): {hsv_color}")
                        print(f"Point added: {x}, {y}")
                        redraw_image()
                except Exception as e:
                    print(f"Error processing color at ({x}, {y}): {e}")

        def redraw_image():
            img_copy = test_img.copy()
            for point in points:
                cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
                cv2.putText(img_copy, f"{point}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Create a display panel for selected colors
            color_panel_height = img_copy.shape[0]
            color_panel_width = 150
            color_panel = np.zeros((color_panel_height, color_panel_width, 3), dtype=np.uint8)

            labels = ["Robot", "Front", "Back", "Ventral"]

            for i, hsv_color in enumerate(selected_colors):
                bgr_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
                start_y = i * (color_panel_height // 4)
                end_y = (i + 1) * (color_panel_height // 4)
                color_panel[start_y:end_y, :] = bgr_color

                cv2.putText(color_panel, labels[i], (10, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            combined_image = np.hstack((img_copy, color_panel))
            cv2.imshow("Color Picking: Pick Huey color, front corner, then back corner. Press 'z' to cancel previous selection. Once 3 colors are selected, press anywhere on the screen to continue", combined_image)

        cv2.imshow("Color Picking: Pick Huey color, front corner, then back corner. Press 'z' to cancel previous selection. Once 3 colors are selected, press anywhere on the screen to continue", test_img)
        cv2.setMouseCallback("Color Picking: Pick Huey color, front corner, then back corner. Press 'z' to cancel previous selection. Once 3 colors are selected, press anywhere on the screen to continue", click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("z"):  # Undo last selection
                if selected_colors and points:
                    removed_color = selected_colors.pop()
                    removed_point = points.pop()
                    print(f"🛑 Color removed: {removed_color}")
                    print(f"🛑 Point removed: {removed_point}")
                    redraw_image()
                else:
                    print("⚠ No points to remove.")
            elif key == 27:  # Press 'Esc' to exit without saving
                print("❌ Selection canceled. Exiting...")
                selected_colors = []
                return None
            elif len(selected_colors) == 5:
                selected_colors = selected_colors[:len(selected_colors)-1]
                print("🎨 Final Selected Colors (HSV):", selected_colors)
                print("📌 Final Selected Points:", points)
                break

        cv2.destroyAllWindows()
        return selected_colors

def save_colors_to_file(colors, output_file):
    """
    Saves the selected colors to a text file in HSV format.

    Args:
        colors (list): List of HSV colors to be saved.
        output_file (str): Path to the output file.
    """
    try:
        with open(output_file, "w") as file:
            for color in colors:
                file.write(f"{color[0]}, {color[1]}, {color[2]}, {color[3]}\n")
        print(f"Selected colors have been saved to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: Output file path '{output_file}' does not exist.")
    except Exception as e:
        print(f"Error saving colors to file: {e}")

def display_colors(selected_colors):
    """
    Displays the selected colors as small colored blocks in a window.

    Args:
        selected_colors (list): List of HSV colors.
    """
    if not selected_colors:
        print("No colors selected to display.")
        return
    
    try:
        # Convert HSV colors to BGR and create a blank image to show them
        bgr_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0] for color in selected_colors]

        height = 175
        width = 175
        img = np.zeros((height, width * len(bgr_colors), 4), dtype=np.uint8)

        for idx, color in enumerate(bgr_colors):
            img[:, idx * width:(idx + 1) * width] = color

            label = ""
            if idx == 0:
                label = "Robot Color"
            elif idx == 1:
                label = "Front Corner"
            elif idx == 2:
                label = "Back Corner"
            elif idx == 3:
                label = "Ventral Corner"

            cv2.putText(img, label, (idx * width + 10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Selected Colors", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error displaying colors: {e}")

if __name__ == "__main__":
    image_path = os.getcwd() + "/warped_images/east.png"
    output_file = "selected_colors.txt"

    # Validating the image path
    if not os.path.exists(image_path):
        print(f"Image file does not exist at path: {image_path}")
    else:
        try:
            selected_colors = ColorPicker.pick_colors(image_path)
            if selected_colors:
                save_colors_to_file(selected_colors, output_file)
                display_colors(selected_colors)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")