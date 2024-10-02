import time
from serial_conn import Serial


class Motor():
    max_speed = 1
    min_speed = -1

    def __init__(self, ser: Serial, speed: float, channel: int):
        self.speed = speed
        self.channel = channel
        self.ser = ser

        self.ser.send_data(channel, speed)

    def move(self, speed):
        if speed > Motor.max_speed:
            speed = Motor.max_speed
        if speed < Motor.min_speed:
            speed = Motor.min_speed

        self.speed = speed
        self.ser.send_data(self.channel, self.speed)

    def get_speed(self):
        return self.speed

    def stop(self, t=0):
        self.speed = 0
        self.ser.send_data(self.channel, self.speed)
        time.sleep(t)
