from flask import Flask, send_file
from picamera import PiCamera
import time

app = Flask(__name__)

camera = PiCamera()
camera.resolution = (640,480) #check resolution

@app.route('/')
def index():
    return "/image to capture an image."

#capture the image using /image
@app.route('/image')
def capture_image():
  time.sleep(1)
  image_path = '/home/pi/Desktop/captured_image.jpg'
  camera.capture(image_path)
    
  #Serve the image
  return send_file(image_path, mimetype='image/jpeg')

  # try:
  #   for i in range(10): 
  #     time.sleep(1)
  #     camera.capture(f'/home/pi/image_{i}.jpg')

  # finally:
  #   camera.close()

if __name__ = "__main__":
  app.run(host='0.0.0.0', port = 5000)