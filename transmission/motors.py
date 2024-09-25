import send_serial
from time import sleep

PIN = 10

####################################################################################################
'''
-This module allows creation of robot objects for 2 wheeled robots.
-The motor driver used is the L298n.
-The Object Motor needs to be created first
-Then the move() function can be called to operate the motors
 move(speed,turn,delay)
-Speed and turn range from -1 to 1
-Delay is in seconds.
'''


class Motor():
    TURN_PERCENT = 0.3 # the ratio for turning (TODO: ADJUST THIS FOR PLAN/RAM)
    
    max_speed = 1  
    min_speed = -1  
    IDLE_PWM = 1487

    # real_max = 88
    # real_min = 5

    def __init__(self, pin):
        self.pin = pin
        

        self.mySpeed = 0

    # 
    def move(self, speed=0, turn=0, t=0):
        # speed *= 80
        # Positive is right, negative is left
        turn *= Motor.TURN_PERCENT
        leftSpeed = speed + turn
        rightSpeed = speed - turn

        if leftSpeed > 1:
            leftSpeed = 1
        elif leftSpeed < -1:
            leftSpeed = -1
        if rightSpeed > 1:
            rightSpeed = 1
        elif rightSpeed < -1:
            rightSpeed = -1
        # print(leftSpeed,rightSpeed)
        self.__set_motor_speed(leftSpeed, left_on=True, right_on=False)
        self.__set_motor_speed(rightSpeed, left_on=False, right_on=True)

        # if leftSpeed > 0:
        #     GPIO.output(self.In1A, GPIO.HIGH)
        #     GPIO.output(self.In2A, GPIO.LOW)
        # else:
        #     GPIO.output(self.In1A, GPIO.LOW)
        #     GPIO.output(self.In2A, GPIO.HIGH)
        # if rightSpeed > 0:
        #     GPIO.output(self.In1B, GPIO.HIGH)
        #     GPIO.output(self.In2B, GPIO.LOW)
        # else:
        #     GPIO.output(self.In1B, GPIO.LOW)
        #     GPIO.output(self.In2B, GPIO.HIGH)
        sleep(t)

    def stop(self, t=0):
        self.pi.set_servo_pulsewidth(self.left_pin, 1500)
        self.pi.set_servo_pulsewidth(self.right_pin, 1500)
        self.mySpeed = 0
        sleep(t)

    def cleanup(self):
        self.pi.set_servo_pulsewidth(LEFT_PIN, 0)
        self.pi.set_servo_pulsewidth(RIGHT_PIN, 0)
        sleep(1)
        self.pi.stop()

    def __set_motor_speed(self, speed, left_on=False, right_on=False):
        # Idle: 0
        # Full back: -1
        # Full front: 1
        # pulse_width = int(
        #     (speed)*0.5*(Motor.ESC_PWM_MAX - Motor.ESC_PWM_MIN) + Motor.ESC_PWM_MIN + 0.5*(Motor.ESC_PWM_MAX - Motor.ESC_PWM_MIN) )
        
        pulse_width = int(Motor.IDLE_PWM + 0.5*speed*(Motor.ESC_PWM_MAX-Motor.ESC_PWM_MIN))

        # print(pulse_width)
        if left_on:            
            self.pi.set_servo_pulsewidth(self.left_pin, pulse_width)
            # print("Left PW: " + str(pulse_width))

        if right_on:
            self.pi.set_servo_pulsewidth(self.right_pin, pulse_width)
            # print("Right PW: " + str(pulse_width))

    def set_raw_PWM(self, PWM, left_on=False, right_on=False):
        if left_on:            
            self.pi.set_servo_pulsewidth(self.left_pin, PWM)
            # print("Left PW: " + str(PWM))

        if right_on:
            self.pi.set_servo_pulsewidth(self.right_pin, PWM)
            # print("Right PW: " + str(PWM))

def main():
    motor.move(0.25, -1, 1)
    motor.move(.5, 0)
    sleep(2)
    motor.move(1, 0)
    sleep(2)
    motor.stop(2)
    motor.cleanup()

if __name__ == '__main__':
    motor = Motor(PIN)
    main()