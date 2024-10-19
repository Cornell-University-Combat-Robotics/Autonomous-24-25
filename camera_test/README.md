# Camera_Test Folder

## test_single_image.py
python script for general purpose test of your setup, including saving an single image

Save test_images by press 's' key
Quit the program by press 'q' key or Ctrl + C

## test_framepersec.py
python script for frame per second test for a specific time to a specific folder

Input:
Folder_name: folder you want to put your images in within camera_test
Time: duration you want to program to run for, total image / time = fps

## test_multiple_images.py
python script testing the camera is connected and opens a pop-up window with a live feed;

Inputs: 
Folder_name: folder the user wants to put images in 
Del_folder: folder the user wants to delete after script has run 
 
Key inputs:   
Save images by pressing 's' key into folder_name only when running program
Delete the 'Saved_frames' folder by pressing the 'd' key only when running program
Quit the program by pressing the 'q' key quits out the code and closes the window 

## code_data_transformation_main.py
Python code that captures frames from a connected camera, with two main options:
   1.  Capture video.
   2.  Capture image folder.

The script saves the captured frames (both original and warped versions) to a specified folder. The "warping" is performed using a custom function (warp), which is implemented from warp.py in image_flattening.

### Input:
Folder_name: folder you want to put your images in within camera_test
Option: Return data in either 1. video format or 2. images format

### Control:
Quit the program by press 'q' key or Ctrl + C

### Output:
    Video Mode:
        The script saves two video files:
            output_video.mp4: Original video from the camera.
            warped_video.mp4: Warped version of the video (using the warp function).
    Image Mode:
        The script saves two types of images:
            saved_frameX.png: Original frames from the camera.
            warped_frameX.png: Warped version of the frames (using the warp function).

### Example

If you choose Option 2 (capture images), and enter test_frames as the folder name, the following structure will be created:

/test_frames/
    saved_frame1.png
    warped_frame1.png
    saved_frame2.png
    warped_frame2.png
    ...

If you choose Option 1 (record video), you'll get:

/test_frames/
    output_video.mp4
    warped_video.mp4

## warp.py 
Please refer to documentation under image_flattening

## matrix.txt
Please refer to documentation under image_flattening
