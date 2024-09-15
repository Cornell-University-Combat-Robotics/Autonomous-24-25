# Image Pipeline for Machine Learning Applications Process
# 1. Image Capture/Loading: Collect image and load onto laptop.
# 2. Image Preprocessing: Resize, normalize, and prepare the image for ML models, yada, yada....
# 3. Model Inference: Pass the processed image to a trained ML model.
# 4. Post-processing: Extract the model's output into RAMPLAN(Algorithm).

import cv2
# import additional libraries

# Capture image from camera using OpenCV 
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Display the current frame
    cv2.imshow('Frame', frame)
    
    # Preprocess the frame for the model, in this case, we can use warp
    preprocessed_frame = some_preprocess_image(frame)
    
    # Run the image through the model
    predictions = some_trained_model(preprocessed_frame)
    
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
