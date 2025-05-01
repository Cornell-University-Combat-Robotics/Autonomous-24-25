import cv2
import numpy as np
import time

def prepare_undistortion_maps(image_width, image_height):
    """Pre-compute the undistortion maps once"""
    # Estimate camera matrix
    focal_length = image_width / 2
    camera_matrix = np.array(
        [[focal_length, 0, image_width / 2],
         [0, focal_length, image_height / 2],
         [0, 0, 1]], dtype=np.float32)
    
    # Estimate distortion coefficients
    dist_coeffs = np.array([0.1, 0.01, 0.01, 0.01], dtype=np.float32)
    
    # Generate new camera matrix
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, dist_coeffs, (image_width, image_height), np.eye(3))
    
    # Create maps for undistortion
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, 
        (image_width, image_height), cv2.CV_16SC2)
    
    return map1, map2

def undistort_image(image, map1, map2):
    """Apply pre-computed maps to undistort an image"""
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

def unfish(frame, map1, map2):
    return undistort_image(frame, map1, map2)
  
    




# Demo usage
# if __name__ == "__main__":
#     # Load image
#     img = cv2.imread('fisheye_image.jpg')
#     h, w = img.shape[:2]
    
#     # Save maps to file for future use
#     map1 = np.load('map1.npy')
#     map2 = np.load('map2.npy')


    
#     # Actual undistortion time
#     start_undistort = time.time()
#     undistorted_img = undistort_image(img, map1, map2)
#     end_undistort = time.time()
#     print(f"Undistortion time: {end_undistort - start_undistort:.4f} seconds")
    
#     # Save result
#     cv2.imwrite('undistorted.jpg', undistorted_img)
    
#     # For batch processing many images
#     print("\nBatch processing example:")
#     # Let's simulate processing 10 images
#     start_batch = time.time()
#     for i in range(10):
#         # In real usage, yovu would load different images here
#         img = cv2.imread("test.jpg")
#         undistorted = undistort_image(img, map1, map2)
#     cv2.imshow("warped new", undistorted)
#     cv2.waitKey(0)
#     end_batch = time.time()
#     print(f"Average time per image: {(end_batch - start_batch)/10:.4f} seconds")