import time
from transmission.serial_conn import OurSerial


class Motor():
    """ 
    The Motor Module can be used to control a motor via the flysky channel it is set up to
    
    Attributes
    ----------
    speed : float
        numerical value from -1 to 1 representing the throttle 
    channel : int
        the channel the motor is connected to
    ser : OurSerial
        a OurSerial object that establishes the laptop's connection to the Arduino


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

    def __init__(self, ser: OurSerial, channel: int, speed: float = 0, channel2 = None, speed2 = None):
        """
        Parameters
        ----------
        ser : OurSerial
            an OurSerial object that establishes the laptops connection to the Arduino
        channel : int
            the flysky channel the motor is connected to
        speed : float, optional
            speed to set the motor to. Default is 0
        """
        self.zero_speed = speed
        self.speed = speed
        self.channel = channel
        self.ser = ser
        self.channel2 = channel2
        if speed2 is None:
            speed2 = 0
        self.speed2 = speed2

        if channel2 is not None:
            self.ser.send_data(channel, speed)
        else:
            self.ser.send_data(channel, speed, channel2, speed2)

    def move(self, speed: float, speed2 = None):
        """Set speed of motor to [speed]. Speed must be between -1 and 1"""
        if speed > Motor.max_speed:
            speed = Motor.max_speed
        if speed < Motor.min_speed:
            speed = Motor.min_speed
        if speed2 is not None:
            if speed2 > Motor.max_speed:
                speed2 = Motor.max_speed
            if speed2 < Motor.min_speed:
                speed2 = Motor.min_speed            
        
        self.speed = speed
        self.speed2 = speed2
        if self.channel2 is None:
            self.ser.send_data(self.channel, self.speed)
        else:
            self.ser.send_data(self.channel, self.speed, self.channel2, self.speed2)

    def get_speed(self):
        """returns current speed"""
        return self.speed, self.speed2

    def stop(self, t=0):
        """Stops motor, then sleeps for [t] seconds"""
        self.speed = self.zero_speed
        self.speed2 = self.zero_speed
        self.ser.send_data(self.channel, self.speed, self.channel2, self.speed2)
        time.sleep(t)
