import cv2
import numpy as np

def read_matrix():
    return np.loadtxt("matrix.txt", dtype=float)

# PARAMETRIZE THIS: READ INTO VIDEO_STREAM_CROP
def warp(source, WIDTH=600, HEIGHT=600):
    source = cv2.resize(source, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    matrix = read_matrix()
    return cv2.warpPerspective(source, np.array(matrix), (WIDTH, HEIGHT))

src = cv2.imread('northwest.PNG')
warped_image = warp(src, 1200, 1200)
cv2.imshow("Warped Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('warped_northwest.png', warped_image)