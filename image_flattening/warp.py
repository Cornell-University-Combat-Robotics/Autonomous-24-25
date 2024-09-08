import cv2
import numpy as np



# matrix for 'cage_overhead.png'
# matrix = [[ 2.40573047e+00,  4.38613842e-01, -4.44448735e+02],
#        [ 2.29118305e-02,  2.07733929e+00, -8.09398597e+01],
#        [ 8.31683768e-05,  1.49938157e-03,  1.00000000e+00]]

def warp(source, WIDTH=600, HEIGHT=600):
    source = cv2.resize(source, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    matrix = [[ 2.59178801e+00,  8.01987233e-01, -4.84713259e+02],
       [ 1.08617053e-02,  2.88921362e+00, -1.60753239e+02],
       [ 1.07684499e-05,  2.64804311e-03,  1.00000000e+00]]
    return cv2.warpPerspective(source, np.array(matrix), (WIDTH, HEIGHT))

src = cv2.imread('cage_overhead_1.png')
cv2.imshow("Image", warp(src, 600, 600))
cv2.waitKey(0)

