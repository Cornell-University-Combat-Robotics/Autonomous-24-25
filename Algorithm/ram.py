import math
import time
import os
import numpy as np
import test_ram_csv as test_ram_csv

class Ram():
    """
    The Ram Module is used to control the movement of the bot in the arena. It uses a PID controller to move the bot to the desired position

    Attributes
    ----------
    huey_position : np.array
        the current position of the bot
    huey_old_position : np.array
        the previous position of the bot
    huey_orientation : float
        the current orientation of the bot
    left : float
        the left motor speed of the bot
    right : float
        the right motor speed of the bot
    enemy_position : np.array
        the current position of the enemy
    enemy_previous_positions : list
        a list of previous enemy positions
    old_time : float
        the previous time
    delta_t : float 
        the change in time between the previous time and the current time


    Methods
    -------
    huey_move(left: Motor, right: Motor, speed: float, turn: float)
        uses a PID controller to move the bot to the desired position

    calculate_velocity(old_pos: np.array, curr_pos: np.array, dt: float)
        calculates the velocity of the bot given the current and previous position
    
    acceleration(old_vel: float, bot_vel: float, dt: float)
        calculates the acceleration of the bot given the current and previous velocity
    
    predict_enemy_position(enemy_position: np.array, enemy_velocity: float, dt: float)
        predicts the enemy position given the current position and velocity

    predict_desired_orientation_angle(our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float)  
        predicts the desired orientation angle of the bot given the current position and velocity of the enemy
    
    predict_desired_turn(our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float)
        predicts the desired turn of the bot given the current position and velocity of the enemy
    
    predict_desired_speed(our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float)
        predicts the desired speed of the bot given the current position and velocity of the enemy

    ram_ram(bots = {'huey': {'bb': list, 'center': list, 'orientation': float}, 'enemy': {'bb': list, 'center': list}})
        main method for the ram ram algorithm that turns to face the enemy and charge towards it
    """
    # ----------------------------- CONSTANTS -----------------------------
    ENEMY_HISTORY_BUFFER = 10 # how many previous enemy position we are recording
    LEFT = 0
    RIGHT = 3
    MAX_SPEED = 1 # between 0 and 1
    MIN_SPEED = 0 # between 0 and 1
    MAX_TURN = 1 # between 0 and 1
    MIN_TURN = 0 # between 0 and 1
    ARENA_WIDTH = 1200 # in pixels
    BATTLE_MODE = False # this is true when the match actually has begun and will cause the motors to move
    TEST_MODE = True # saves values to CSV file

    '''
    Constructor for the Ram class that initializes the position and orientation of the bot, the motors, the enemy position, 
    the enemy position array, the old time, and the delta time. 

    Parameters
    ----------
    huey_position : np.array
        the initial position of the bot
    huey_old_position : np.array
        the previous position of the bot
    huey_orientation : float
        the initial orientation of the bot
    enemy_position : np.array
        the initial position of the enemy
    '''
    def __init__(self, huey_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])), huey_old_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])),
                 huey_orientation=45, enemy_position = np.array([0, 0]))-> None:
        # ----------------------------- INIT ----------------------------- 
        # initialize the position and orientation of huey
        self.huey_position = huey_position
        self.huey_old_position = huey_old_position
        self.huey_orientation = huey_orientation

        self.left = 0
        self.right = 0


        # initialize the current enemy position
        self.enemy_position = enemy_position
        
        # initialize the enemy position array
        self.enemy_previous_positions = []
        self.enemy_previous_positions.append(self.enemy_position)

        # old time
        self.old_time = time.time()
        # delta time 
        self.delta_t = 0.001
            
    # ----------------------------- HELPER METHODS -----------------------------

    ''' use a PID controller to move the bot to the desired position '''
    def huey_move(self, speed: float, turn: float):
        print(f'Here: {speed} and {turn}')
        self.left = ((speed + turn) / 2.0)
        self.right = ((speed - turn) / 2.0)
        return {'left': self.left, 'right': self.right}

    ''' calculate the velocity of the bot given the current and previous position '''
    def calculate_velocity(self, old_pos: np.array, curr_pos: np.array, dt: float):
        if (dt == 0.0):
            return 0.0
        return (curr_pos - old_pos) / dt

    ''' calculate the acceleration of the bot given the current and previous velocity '''
    def acceleration(self, old_vel: float, bot_vel: float, dt: float):
        if (dt == 0.0):
            return 0.0
        return (bot_vel - old_vel) / dt

    ''' predict the enemy position given the current position and velocity '''
    def predict_enemy_position(self, enemy_position: np.array, enemy_velocity: float, dt: float):
        return enemy_position + dt * enemy_velocity
    
    
    def invert_y(self, pos: np.array):
        pos[1] = -pos[1]
        return pos
    
    ''' predict the desired orientation angle of the bot given the current position and velocity of the enemy '''
    def predict_desired_orientation_angle(self, our_pos: np.array, our_orientation: float, enemy_pos: np.array, enemy_velocity: float, dt: float):
        enemy_future_position = self.predict_enemy_position(enemy_pos, enemy_velocity, dt)
        # return the angle in angle
        our_orientation = np.radians(our_orientation)
        orientation = np.array([math.cos(our_orientation), math.sin(our_orientation)])
        # @@@@@@@@@@@@@@@@@@@
        enemy_future_position = self.invert_y(enemy_future_position)
        our_pos = self.invert_y(our_pos)
        # @@@@@@@@@@@@@@@@@@@
        
        direction = enemy_future_position - our_pos
        # calculate the angle between the bot and the enemy
        angle = np.degrees(np.arccos(np.dot(direction, orientation) / (np.linalg.norm(direction) * np.linalg.norm(orientation))))
        sign = np.sign(np.cross(orientation, direction)) 
        return sign*angle

    ''' predict the desired turn of the bot given the current position and velocity of the enemy '''
    def predict_desired_turn(self, our_pos: np.array, our_orientation: float, enemy_pos: np.array, enemy_velocity: float, dt: float):
        angle = self.predict_desired_orientation_angle(our_pos, our_orientation, enemy_pos, enemy_velocity, dt)
        return angle * (Ram.MAX_TURN / 180.0)

    ''' predict the desired speed of the bot given the current position and velocity of the enemy '''
    def predict_desired_speed(self, our_pos: np.array, our_orientation: float, enemy_pos: np.array, enemy_velocity: float, dt: float):
        angle = self.predict_desired_orientation_angle(our_pos, our_orientation, enemy_pos, enemy_velocity, dt)		
        return 1-(abs(angle) * (Ram.MAX_SPEED / 180.0))

    ''' main method for the ram ram algorithm that turns to face the enemy and charge towards it '''
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
        

        if (Ram.TEST_MODE):
            angle = self.predict_desired_orientation_angle(self.huey_position, self.huey_orientation, self.enemy_position, enemy_velocity, self.delta_t)	
            direction = self.predict_enemy_position(self.enemy_position, enemy_velocity, self.delta_t) - self.huey_position

            test_ram_csv.test_file_update(delta_time= self.delta_t, bots=bots, huey_pos=self.huey_position, huey_facing=self.huey_orientation, 
                                    enemy_pos= self.enemy_position, huey_old_pos=self.huey_old_position, 
                                    huey_velocity=self.calculate_velocity(self.huey_position, self.huey_old_position, self.delta_t),
                                    enemy_old_pos=self.enemy_previous_positions, enemy_velocity=enemy_velocity, speed=speed, turn=turn,
                                    left_speed=self.left, right_speed=self.right, angle = angle, direction = direction)
            
        self.huey_old_position = self.huey_position
        # if the array for enemy_previous_positions is full, then pop the first one
        self.enemy_previous_positions.append(self.enemy_position)
        if len(self.enemy_previous_positions) > Ram.ENEMY_HISTORY_BUFFER:
            self.enemy_previous_positions.pop(0)

        return self.huey_move(speed, turn)