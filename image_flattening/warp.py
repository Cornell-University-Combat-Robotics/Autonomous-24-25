import cv2
import numpy as np

WIDTH = 1200
HEIGHT = 1200

def read_matrix():
    return np.loadtxt("matrix.txt", dtype = float)

# PARAMETRIZE THIS: READ INTO VIDEO_STREAM_CROP
def warp(source, width, height):
    source = cv2.resize(source, (width, height), interpolation = cv2.INTER_AREA)
    matrix = read_matrix()
    return cv2.warpPerspective(source, np.array(matrix), (width, height))

src = cv2.imread('northwest.PNG')
warped_image = warp(src, WIDTH, HEIGHT)
cv2.imshow("Warped Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('warped_northwest.png', warped_image)