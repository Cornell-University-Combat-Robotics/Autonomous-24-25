import time
import serial


class Serial():

    def __init__(self, port, baudrate=9600, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for the serial connection to initialize

    def send_data(self, channel, speed):
        data = f"{channel} {speed}\n"
        self.ser.write(data.encode())

    def cleanup(self):
        self.ser.close()
