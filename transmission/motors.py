import time
from transmission.serial_conn import Serial


class Motor():
    """ 
    The Motor Module can be used to control a motor via the flysky channel it is set up to
    
    Attributes
    ----------
    speed : float
        numerical value from -1 to 1 representing the throttle 
    channel : int
        the channel the motor is connected to
    ser : Serial
        a Serial object that establishes the laptop's connection to the Arduino


    Methods
    -------
    move(speed: float)
        Sets the channel to [speed]. If [speed] is greater than 1, 
        speed will be set to 1. If [speed] is less than -1, speed will be set to -1

    get_speed()
        Returns the current speed of the motor
    
    stop(t=0)
        sets motor speed to 0 and sleeps for time [t] seconds

    """
    max_speed = 1
    min_speed = -1

    def __init__(self, ser: Serial, channel: int, speed: float = 0):
        """
        Parameters
        ----------
        ser : Serial
            a Serial object that establishes the laptops connection to the Arduino
        channel : int
            the flysky channel the motor is connected to
        speed : float, optional
            speed to set the motor to. Default is 0
        """
        self.zero_speed = speed
        self.speed = speed
        self.channel = channel
        self.ser = ser

        self.ser.send_data(channel, speed)

    def move(self, speed: float):
        """Set speed of motor to [speed]. Speed must be between -1 and 1"""
        if speed > Motor.max_speed:
            speed = Motor.max_speed
        if speed < Motor.min_speed:
            speed = Motor.min_speed

        self.speed = speed
        self.ser.send_data(self.channel, self.speed)

    def get_speed(self):
        """returns current speed"""
        return self.speed

    def stop(self, t=0):
        """Stops motor, then sleeps for [t] seconds"""
        self.speed = self.zero_speed
        self.ser.send_data(self.channel, self.speed)
        time.sleep(t)
