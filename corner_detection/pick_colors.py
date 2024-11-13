import cv2
import os

# WIDTH = 1200
# HEIGHT = 1200
image_path = os.getcwd() + '/warped_images/northeast.png'
test_img = cv2.imread(image_path)
# test_img = cv2.resize(test_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

# List to store the selected colors
#   selected_colors[0]: front corners
#   selected_colors[1]: back corners
selected_colors = []
corners = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = test_img[y, x]  # OpenCV reads as BGR
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))  # Convert to RGB
        selected_colors.append(rgb_color)
        corners.append([x, y])
        print(f"Selected color: {rgb_color}")
        print(f"Point added: {x}, {y}")
        redraw_image()

def redraw_image():
    img_copy = test_img.copy()
    
    for point in corners:
        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
        cv2.putText(img_copy, f"{point}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Select Colors", img_copy)

cv2.imshow("Select Colors", test_img)
cv2.setMouseCallback("Select Colors", click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('z'):  # If 'z' is pressed
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
    elif len(selected_colors) == 2 and len(corners) == 2:
        break

print("Final Selected Colors in RGB:", selected_colors)
print("Final Selected Points", corners)
cv2.destroyAllWindows()