import cv2
import numpy as np

WIDTH = 600
HEIGHT = 600
src = cv2.imread('cage_overhead.png')
test_img = cv2.resize(src, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

matrix = [[ 2.40573047e+00,  4.38613842e-01, -4.44448735e+02],
       [ 2.29118305e-02,  2.07733929e+00, -8.09398597e+01],
       [ 8.31683768e-05,  1.49938157e-03,  1.00000000e+00]]

result = cv2.warpPerspective(test_img, np.array(matrix), (WIDTH, HEIGHT))
cv2.imshow("Image", result)

cv2.waitKey(0)

