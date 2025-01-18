# Corner Detection

## color_picker.py

This script is designed to run at the start of each match, allowing the user to
manually select the robot's color as well as the front and back corner points. 
This process ensures that there are no hardcoded values prior to the 
competition. By manually selecting the colors in real-time, we account for 
variations in lighting conditions, providing a more adaptable solution and
eliminating reliance on static values that might be impacted by environmental 
factors.

## corner_detection.py

Functions in this file will be used in the main.py integration file.

## test_corner_detection.py

Running this file will run the color picker and corner detection functionality 
on all images in a given folder

## Testing Individual Photos

1. Run `python3 color_picker.py` to select the robot's color, front and back
corners in that order. Make sure the image path is correct
2. Run `python3 corner_detection.py` to run the corner detection functionality
on the specified image. Make sure the image path is correct

## Testing All Photos in a Given Folder
1. Run `python3 test_corner_detection.py` to run the cornerl detection 
functionality on all images in a given folder. Make sure the folder path is
correct

# Understanding HSV Ranges in OpenCV vs. Online Resources

When working with HSV color values, it's essential to recognize that the range 
of values in OpenCV differs from those in many online color pickers or other 
color manipulation tools.

## HSV Ranges in OpenCV
1. Hue: 0 to 180
2. Saturation: 0 to 255
3. Value (Brightness): 0 to 255

## HSV Ranges in Online Color Pickers
1. Hue: 0 to 360
2. Saturation: 0 to 100%
3. Value (Brightness): 0 to 100%