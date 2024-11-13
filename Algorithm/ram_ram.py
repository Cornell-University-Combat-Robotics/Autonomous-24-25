import Math
import time
from motors import Motor
from serial_conn import Serial
import numpy as np

# ----------------------------- CONSTANTS -----------------------------

ENEMY_HISTORY_BUFFER = 10 # how many previous enemy position we are recording
MAX_SPEED = 1 # between 0 and 1
MIN_SPEED = 0 # between 0 and 1
MAX_TURN = 1 # between 0 and 1
MIN_TURN = 0 # between 0 and 1
ARENA_WIDTH = 1200 # in pixels
BATTLE_MODE = False # this is true when the match actually has begun

# ----------------------------- GLOBAL VARIABLES -----------------------------

# preload the starting position of huey (corner of the battle field)
huey_old_position : np.array # (x: float, y: float)
huey_position : np.array  # (x: float, y: float)
huey_orientation : np.array # angle [0, 360)
left: Motor # Motor object
right: Motor # Motor object

enemy_position : np.array # (x: float, y: float)
enemy_previous_positions: list # list of tuples ()

old_time : float # previous time step 
delta_t : float # determine the time step

# ----------------------------- METHODS -----------------------------

# use a PID controller to move the bot to the desired position
def huey_move(left: Motor, right: Motor, speed: float, turn: float):
	left.move((speed + turn) / 2.0)
	right.move((speed - turn) / 2.0)

# calculate the velocity of the bot given the current and previous position
def calculate_velocity(old_pos: np.array, curr_pos: np.array, dt: float):
	if (dt == 0.0):
		return 0.0
	return (curr_pos - old_pos) / dt

# calculate the acceleration of the bot given the current and previous velocity
def acceleration(old_vel: float, bot_vel: float, dt: float):
	if (dt == 0.0):
		return 0.0
	return (bot_vel - old_vel) / dt

# predict the enemy position given the current position and velocity
def predict_enemy_position(enemy_position: np.array, enemy_velocity: float, dt: float):
	return enemy_position + dt * enemy_velocity
   
# predict the desired orientation angle of the bot given the current position and velocity of the enemy
def predict_desired_orientation_angle(our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float):
	enemy_future_position = predict_enemy_position(enemy_pos, enemy_velocity, dt)
	# return the angle in angle
	orientation = our_orientation
	direction = enemy_future_position - our_pos
	# calculate the angle between the bot and the enemy
	angle = np.acrcos(np.dot(direction, orientation) / (np.linalg.norm(direction) * np.linalg.norm(orientation)))
	return angle

# predict the desired turn of the bot given the current position and velocity of the enemy
def predict_desired_turn(our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float):
	angle = predict_desired_orientation_angle(our_pos, our_orientation, enemy_pos, enemy_velocity, dt)
	return angle * (MAX_TURN / 180.0)

# predict the desired speed of the bot given the current position and velocity of the enemy
def predict_desired_speed(our_pos: np.array, our_orientation: np.array, enemy_pos: np.array, enemy_velocity: float, dt: float):
	angle = predict_desired_orientation_angle(our_pos, our_orientation, enemy_pos, enemy_velocity, dt)		
	return Math.abs(angle - 180.0) * (MAX_SPEED / 180.0)

# ----------------------------- END of METHODS -----------------------------

# ----------------------------- INIT ----------------------------- 

# initialize the position and orientation of huey
huey_position = np.array([ARENA_WIDTH, ARENA_WIDTH])
huey_old_position = np.array([ARENA_WIDTH, ARENA_WIDTH])
huey_orientation = 45

# initialize a serial connection
serial = Serial()
# initialize the motor
left = Motor(ser = serial, channel = 0)
right = Motor(ser = serial, channel = 1)

# initialize the enemy position array
enemy_previous_positions = []
# initialize the current enemy position
enemy_position = np.array([0, 0])

# old time
old_time = time()
# delta time 
delta_t = 0.001

# TODO: Warp

# ----------------------------- END of INIT -----------------------------

# ----------------------------- MAIN -----------------------------

while True:
	old_time = time()
	delta_t = time() - old_time # record delta time
	old_time = time()
	
	# Run Object Detection
	# Run Corner Detection
	
	# get new position and heading values
	huey_position # from Corner Detection
	huey_orientation # from Corner Detection
	enemy_position # from Object Detection
	enemy_velocity = calculate_velocity(enemy_position, enemy_previous_positions[-1], delta_t)
	turn = predict_desired_turn(huey_position, enemy_position, enemy_velocity, delta_t)
	speed = predict_desired_speed(huey_position, enemy_position, enemy_velocity, delta_t)
	
	if (BATTLE_MODE):
		huey_move(left, right, speed, turn)
		
	huey_old_position = huey_position
	# if the array for enemy_previous_positions is full, then pop the first one
	enemy_previous_positions.add(enemy_position)
	if len(enemy_previous_positions) > ENEMY_HISTORY_BUFFER:
		enemy_previous_positions.pop(0)
	
	enemy_position = predict_enemy_position(enemy_position, enemy_velocity, delta_t)

# ----------------------------- END of MAIN -----------------------------