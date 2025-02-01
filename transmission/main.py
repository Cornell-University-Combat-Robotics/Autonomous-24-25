from motors import Motor
from serial_conn import Serial
import time
import serial
import serial.tools.list_ports
# review if we actually need these ^^^ #

ser = Serial()

right_motor = Motor(ser, speed=0, channel=1)
left_motor = Motor(ser, speed=0, channel=3)

left_motor.move(speed=-0.5)
time.sleep(1.7)
left_motor.move(speed=-0.25)
time.sleep(.2)
left_motor.move(speed=-0.15)
time.sleep(.1)
left_motor.move(speed=0.15)
time.sleep(.1)
left_motor.move(speed=0.25)
time.sleep(.2)
left_motor.move(speed=0.5)
time.sleep(1.7)

left_motor.stop()
right_motor.stop()

ser.cleanup()
