from motors import Motor
from serial_conn import Serial

ser = Serial("COMM 3")

left_motor = Motor(ser, speed=0, channel=0)
right_motor = Motor(ser, speed=0, channel=1)

left_motor.move(speed=0.5)
right_motor.move(speed=-0.7)

ser.cleanup()
