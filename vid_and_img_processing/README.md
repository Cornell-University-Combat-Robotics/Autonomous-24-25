# Video and Image Processing

## Description
Refer to this README if you want to accomplish any of the following:
1. [Split a video into warped frames for Roboflow labeling](#video-to-warped-frames)
2. [Warp a singular frame](#frame-to-warped-frame)
3. [Warp an entire video](#video-to-warped-video)

## Prerequisites
This application requires that you have numpy and opencv installed. If not previously installed, refer to the terminal commands below.
```
pip install numpy
pip install opencv-python
```

## Usage

### Video to Warped Frames
Use this option for generating data for Roboflow labeling.

*Setup*
* Find a video you want to process by searching through the "brettzone" channel in the NHRL Discord ([here](https://brettzone.nhrl.io/brettZone/fightReview.php?gameID=EX-18fece8f673a4e7e&tournamentID=nhrl_sep24_fs) is an example).
* Download the "Overhead High" footage.
* Upload the downloaded video to the `/data/input_video folder`.

*Application*
* Open [vid_to_warped_frames.py](vid_to_warped_frames.py).
* At the bottom of the file, change `video_name` to match the filename of the video you uploaded.
* Run the file.
* IMPORTANT: Select the corners of the arena in the following order: top left, top right, bottom right, bottom left.
* The warped video frames will save in the `/warped_vid_frames` folder.

### Frame to Warped Frame
*Setup*
* Upload an image of your choosing to the `/data/input_img` folder.

*Application*
* Open [warp_image.py](warp_image.py).
* At the bottom of the file, change `img_name` to match the filename of the video you uploaded.
* Run the file.
* IMPORTANT: Select the corners of the arena in the following order: top left, top right, bottom right, bottom left.
* The warped video frames will save in the `/warped_img` folder.

### Video to Warped Video
*Setup*
* Upload an image of your choosing to the `/data/input_video` folder.

*Application*
* Open [warp_video.py](warp_video.py).
* At the bottom of the file, change `video_name` to match the filename of the video you uploaded.
* Run the file.
* IMPORTANT: Select the corners of the arena in the following order: top left, top right, bottom right, bottom left.
* The warped video frames will save in the `/warped_video` folder.

## Contact
For any issues using this code, please reach out to Katie Huntley (kah294).
This document was last updated November 16, 2024.