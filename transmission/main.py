from motors import Motor
from serial_conn import Serial
import time
import serial

# works for Katie's computer
# TODO: make this transferrable to other computers, make terminal interface to choose port
ser = Serial(port="COM3")

right_motor = Motor(ser, speed=0, channel=0)
left_motor = Motor(ser, speed=0, channel=1)

left_motor.move(speed=0.5)
right_motor.move(speed=0.5)

time.sleep(2)

left_motor.stop()
right_motor.stop()

ser.cleanup()
