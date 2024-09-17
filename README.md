# Autonomous-24-25

How to collect data using `process_battlefield_to_images`:
1. Ensure your `image_flashing` branch is up to date
2. Manually download an NHRL fight video using the same angle; ensure you know the name of the file
3. Upload the video to the `image_flattening` directory. Because the videos are normally 3 minutes in length, we speficy them to not be tracked by Git
4. cd into the `image_flattening` directory by running `cd image_flattening` in your terminal
5. Set the parameters of the function at the bottom of the `video_stream_crop.py` file:
- The first parameter, the filepath, should be whatever you named your .mp4 file
- The second parameter is the framerate, or how often to save a frame from the video. 1-2 frames per second should suffice
6. Extract images from the video by running `python video_stream_crop.py` and wait an appropriate amount of time (generally ~3 minutes)
7. The images should be saved under `image_flattening\IMG_`; download locally and use as you see fit