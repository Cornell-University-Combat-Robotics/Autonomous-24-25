import cv2
from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras():
    print(f'{camera_info.index}: {camera_info.name}')