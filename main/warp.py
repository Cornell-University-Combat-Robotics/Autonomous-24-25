import cv2
import numpy as np

def read_matrix():
    return np.loadtxt("matrix.txt", dtype=float)

def warp(source, WIDTH=600, HEIGHT=600):
    source = cv2.resize(source, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    matrix = read_matrix()
    return cv2.warpPerspective(source, np.array(matrix), (WIDTH, HEIGHT))

