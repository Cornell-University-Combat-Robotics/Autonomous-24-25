import math
import time
from motors import Motor
from serial_conn import Serial
import numpy as np
import test_ram

class Ram():
    # ----------------------------- CONSTANTS -----------------------------
    ENEMY_HISTORY_BUFFER = 10 # how many previous enemy position we are recording
    MAX_SPEED = 1 # between 0 and 1
    MIN_SPEED = 0 # between 0 and 1
    MAX_TURN = 1 # between 0 and 1
    MIN_TURN = 0 # between 0 and 1
    ARENA_WIDTH = 1200 # in pixels
    BATTLE_MODE = False # this is true when the match actually has begun and will cause the motors to move
    TEST_MODE = True # saves values to CSV file

    def __init__(self) -> None:
        # ----------------------------- INIT ----------------------------- 
        # initialize the position and orientation of huey
        self.huey_position = np.array([Ram.ARENA_WIDTH, Ram.ARENA_WIDTH])
        self.huey_old_position = np.array([Ram.ARENA_WIDTH, Ram.ARENA_WIDTH])
        self.huey_orientation = 45

        if Ram.BATTLE_MODE:
            # initialize a serial connection
            serial = Serial()
            # initialize the motor
            self.left = Motor(ser = serial, channel = 0)
            self.right = Motor(ser = serial, channel = 1)
        else:
            self.left = None
            self.right = None


        # initialize the current enemy position
        self.enemy_position = np.array([0, 0])
        
        # initialize the enemy position array
        self.enemy_previous_positions = []
        self.enemy_previous_positions.append(self.enemy_position)

        # old time
        self.old_time = time.time()
        # delta time 
        self.delta_t = 0.001
            
    # ----------------------------- HELPER METHODS -----------------------------

    # use a PID controller to move the bot to the desired position
    def huey_move(self, left: Motor, right: Motor, speed: float, turn: float):
        left.move((speed + turn) / 2.0)
        right.move((speed - turn) / 2.0)

    # calculate the velocity of the bot given the current and previous position
    def calculate_velocity(self, old_pos: np.array, curr_pos: np.array, dt: float):
        if (dt == 0.0):
            return 0.0
        return (curr_pos - old_pos) / dt

    # calculate the acceleration of the bot given the current and previous velocity
    def acceleration(self, old_vel: float, bot_vel: float, dt: float):
        if (dt == 0.0):
            return 0.0
        return (bot_vel - old_vel) / dt

    # predict the enemy position given the current position and velocity
    def predict_enemy_position(self, enemy_position: np.array, enemy_velocity: float, dt: float):
        return enemy_position + dt * enemy_velocity
    
    # predict the desired orientation angle of the bot given the current position and velocity of the enemy
    def predict_desired_orientation_angle(self, our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float):
        enemy_future_position = self.predict_enemy_position(enemy_pos, enemy_velocity, dt)
        # return the angle in angle
        orientation = np.array([math.cos(our_orientation), math.sin(our_orientation)])
        direction = enemy_future_position - our_pos
        # calculate the angle between the bot and the enemy
        angle = np.degrees(np.arccos(np.dot(direction, orientation) / (np.linalg.norm(direction) * np.linalg.norm(orientation))))
        sign = np.sign(np.cross(orientation, direction)) 
        return sign*angle

    # predict the desired turn of the bot given the current position and velocity of the enemy
    def predict_desired_turn(self, our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float):
        angle = self.predict_desired_orientation_angle(our_pos, our_orientation, enemy_pos, enemy_velocity, dt)
        return angle * (Ram.MAX_TURN / 180.0)

    # predict the desired speed of the bot given the current position and velocity of the enemy
    def predict_desired_speed(self, our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float):
        angle = self.predict_desired_orientation_angle(our_pos, our_orientation, enemy_pos, enemy_velocity, dt)		
        return 1-(abs(angle) * (Ram.MAX_SPEED / 180.0))

    # main method for ram ram and takes in output from corner detection
    def ram_ram(self, bots = {'huey': {'bb': list, 'center': list, 'orientation': float}, 'enemy': {'bb': list, 'center': list}}):
        self.delta_t = time.time() - self.old_time # record delta time
        self.old_time = time.time()
        
        
        # get new position and heading values
        self.huey_position = np.array(bots['huey'].get('center'))
        self.huey_orientation = bots['huey'].get('orientation')
        
        self.enemy_position = np.array(bots['enemy'].get('center'))
        enemy_velocity = self.calculate_velocity(self.enemy_position, self.enemy_previous_positions[-1], self.delta_t)
        turn = self.predict_desired_turn(our_pos= self.huey_position, our_orientation= self.huey_orientation, enemy_pos=self.enemy_position, 
                                         enemy_velocity= enemy_velocity, dt = self.delta_t)
        speed = self.predict_desired_speed(our_pos= self.huey_position, our_orientation= self.huey_orientation, enemy_pos=self.enemy_position, 
                                         enemy_velocity= enemy_velocity, dt = self.delta_t)
        
        if (Ram.BATTLE_MODE):
            self.huey_move(self.left, self.right, speed, turn)

        if (Ram.TEST_MODE):
            angle = self.predict_desired_orientation_angle(self.huey_position, self.huey_orientation, self.enemy_position, enemy_velocity, self.delta_t)		
            test_ram.test_file_update(delta_time= self.delta_t, bots=bots, huey_pos=self.huey_position, huey_facing=self.huey_orientation, 
                                      enemy_pos= self.enemy_position, huey_old_pos=self.huey_old_position, 
                                      huey_velocity=self.calculate_velocity(self.huey_position, self.huey_old_position, self.delta_t),
                                      enemy_old_pos=self.enemy_previous_positions, enemy_velocity=enemy_velocity, speed=speed, turn=turn,
                                      left_speed=self.left, right_speed=self.right, angle = angle)
        self.huey_old_position = self.huey_position
        # if the array for enemy_previous_positions is full, then pop the first one
        self.enemy_previous_positions.append(self.enemy_position)
        if len(self.enemy_previous_positions) > Ram.ENEMY_HISTORY_BUFFER:
            self.enemy_previous_positions.pop(0)
