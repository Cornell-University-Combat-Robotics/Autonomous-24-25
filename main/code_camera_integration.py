# Image Pipeline for Machine Learning Applications Process
# 1. Image Capture/Loading: Collect image and load onto laptop.
# 2. Image Preprocessing: Resize, normalize, and prepare the image for ML models, yada, yada....
# 3. Model Inference: Pass the processed image to a trained ML model.
# 4. Post-processing: Extract the model's output into RAMPLAN(Algorithm).

import cv2
from warp.py import warp
from test_regular.py import test_camera
# import additional libraries

# Capture image from camera using OpenCV 
cap = cv2.VideoCapture(1)

test_camera(cap)
# Preprocess the frame for the model, in this case, we can use warp or reduce qualty, etc  
preprocessed_frame = some_preprocess_image(frame)
#details 
#use a slice function (ethan and alyssa)


# Run the image through the model
predictions = some_trained_model(preprocessed_frame)


cap.release()
cv2.destroyAllWindows()
