# Transmission 

## Description
This code controls a motor through python code. Follow these instructions for hardware and software set up.

## Required Packages
pyserial: `python -m pip install pyserial`

## Hardware Setup
1. Set FlySky controller in [trainer mode](https://clover.coex.tech/en/trainer_mode.html)
    * Use the controller with the “Kenny” sticker
    * If you get a warning that states: “Place all switches in their up position…”
        * Put all the top switches up and the left joystick in the center bottom 
    * The the top right switch is used “for taking the control” (put it in the down position)

2. Connect wires, cables as shown in the circuit diagram

    <img src="readme_assest/circuit_diagram.png" alt="Circuit Diagram" width="500">

Note: you may be able to skip the following instructions if you have already set up your arduino

3. Download [Arduino IDE](https://www.arduino.cc/en/software) (if not previously installed)
4. Download `receive_serial_and_send_PPM.ino` (if not previously installed)
5. In the Arduino IDE, select the Arduino nano board and appropriate port
    
    <img src="readme_assest/select_board_and_port.png" alt="Select Board" width=500>

    * if you’re unsure of the port, unplug and replug the arduino and see which option disappears and reappears

6. Upload the .ino code to Arduino

    <img src="readme_assest/upload_sketch.png" alt="Upload Sketch" width=300>

## Software Setup 
1. Create a Serial object 
    * if you don't know what port the arduino is on, leave port as None (default) and respond to prompt in terminal when code is run

2. Create motor object for motor you wish to control
    * channel should match desired channel in the FlySky controller
    * To see channel values on FlySky controller:
        * Hold ok to get to settings
        * Select “Functions Setup” via the OK button
        * Hit down till you are on “Display”, then press OK
3. Move motors with appropriate method
    * Be careful about maxing out speed/coming to a sudden stop
4. Clean Up
    * Stop motors (either set speed to 0 or call `stop`)
    * clean serial object with `cleanup` method

### Example code
```from motors import Motor
from serial_conn import OurSerial
import time
import serial
import serial.tools.list_ports

ser = OurSerial()

right_motor = Motor(ser, speed=0, channel=0)
left_motor = Motor(ser, speed=0, channel=1)

left_motor.move(speed=0.5)
right_motor.move(speed=0.5)

time.sleep(2)

left_motor.stop()
right_motor.stop()

ser.cleanup()
```
