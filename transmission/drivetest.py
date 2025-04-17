from motors import Motor
from serial_conn import OurSerial
import time
import serial
import serial.tools.list_ports
# review if we actually need these ^^^ #

ser = OurSerial()

speed = Motor(ser, speed=0, channel=1)
turn = Motor(ser, speed=0, channel=3)

turn.move(speed=-0.5)
time.sleep(1.7)
turn.move(speed=-0.25)
time.sleep(.2)
turn.move(speed=-0.15)
time.sleep(.1)
turn.move(speed=0.15)
time.sleep(.1)
turn.move(speed=0.25)
time.sleep(.2)
turn.move(speed=0.5)
time.sleep(1.7)

turn.stop()
speed.stop()

ser.cleanup()
