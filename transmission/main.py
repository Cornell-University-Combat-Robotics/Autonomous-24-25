from motors import Motor
from serial_conn import Serial
import time
import serial
import serial.tools.list_ports
import transmission
# review if we actually need these ^^^ #

ser = Serial()

speed_motor = Motor(ser, speed=0, channel=1)
turn_motor = Motor(ser, speed=0, channel=3)

try:
    while True:
        turn_motor.move(speed=-0.5)
        time.sleep(0.5)
        turn_motor.move(speed=-0.25)
        time.sleep(.5)
        turn_motor.move(speed=-0.15)
        time.sleep(.5)
        turn_motor.move(speed=0.15)
        time.sleep(.5)
        turn_motor.move(speed=0.25)
        time.sleep(.5)
        turn_motor.move(speed=0.5)
        time.sleep(0.5)

        speed_motor.move(speed=-0.5)
        time.sleep(0.5)
        speed_motor.move(speed=-0.25)
        time.sleep(.5)
        speed_motor.move(speed=-0.15)
        time.sleep(.5)
        speed_motor.move(speed=0.15)
        time.sleep(.5)
        speed_motor.move(speed=0.25)
        time.sleep(.5)
        speed_motor.move(speed=0.5)
        time.sleep(0.5)

except:
    turn_motor.stop()
    speed_motor.stop()
    ser.cleanup()
