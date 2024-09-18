import cv2
import numpy as np

WIDTH = 600
HEIGHT = 600
test_img = cv2.imread('cage_overhead_1.png')
test_img = cv2.resize(test_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

# the list of the "real" corners from the raw image
# in order:
#   corners[0]: top left
#   corners[1]: top right
#   corners[2]: bottom right
#   corners[3]: bottom left
corners = []

# function managing the selection of four corner points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left button clicked, store the point
        corners.append([x, y])
        print(f"Point added: {x}, {y}")
        redraw_image()  # Redraw the points on the image

# redraws image once a corner has been selected
def redraw_image():
    # Make a copy of the original image to redraw the points
    img_copy = test_img.copy()
    
    # Loop through the clicked points and display them on the image
    for point in corners:
        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
        cv2.putText(img_copy, f"{point}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the image with points
    cv2.imshow("Image", img_copy)

# Display the image
cv2.imshow("Image", test_img)
cv2.setMouseCallback("Image", click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('z'):  # If 'z' is pressed
        if corners:
            removed_point = corners.pop()  # Remove the last point
            print(f"Point removed: {removed_point}")
            redraw_image()  # Redraw the image with remaining points
        else:
            print("No points to remove.")
    
    elif key == 27:  # Press 'Esc' to exit
        break
    elif len(corners) == 4:
        break

print("Final Selected Points:", corners)

dest_pts = [[0,0],[WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]
matrix = cv2.findHomography(np.array(corners), np.array(dest_pts))

np.savetxt("matrix.txt", matrix[0], fmt="%f")

#print(matrix[0])
cv2.destroyAllWindows()