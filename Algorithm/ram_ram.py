# constants
ENEMY_HISTORY_BUFFER = 10 # how many previous enemy position we are recording
ENEMY_VELOCITY_THRESHOLD = _ # used to determine when to attack enemy
MAX_SPEED = 1
MIN_SPEED = 0
MAX_TURN = 1
MIN_TURN = 0

# global variables
# preload the starting position of huey(corner of the battle field)
huey_old_position : tuple = (x, y) # (x:float, y:float)
huey_position : tuple = (x,y) # (x:float, y:float)
huey_facing : float = θ # angle [0, 360)
left: motor 
right: motor

enemy_position : tuple = (x,y) # (x:float, y:float)
enemy_previous_positions: list [ENEMY_HISTORY_BUFFER]:

old_time = _ # previous time step 
delta_t = _ # determine the time step

#methods
def huey_move(speed, turn):
		left.move((speed+turn)/2)
		right.move((speed-turn)/2)

def calculate_velocity(old_pos, curr_pos, dt):
		return (curr_pos-old_pos)/dt

def acceleration(old_vel, bot_vel, dt):
		return (bot_vel - old_vel)/dt

def predict_enemy_position(enemy_position, enemy_velocity, dt):
		return enemy_position + dt * enemy_velocity
   
def predict_desired_facing_angle(our_pos, enemy_pos, enemy_velocity, dt):
		enemy_future_position = predict_enemy_position(enemy_pos, enemy_velocity, dt)
		return Math.arctan2((enemy_future_position.x - our pos.x) / (enemy_future_position.y - our pos.y))

def predict_desired_turn(our_pos, enemy_pos, enemy_velocity, dt):
		angle = predict_desired_facing_angle(our_pos, enemy_pos, enemy_velocity, dt)
		return angle * (MAX_TURN/180)

def predict_desired_speed(our_pos, enemy_pos, enemy_velocity, dt):
		angle = predict_desired_facing_angle(our_pos, enemy_pos, enemy_velocity, dt)		
		return Math.abs(angle-180) * (MAX_SPEED/180)
		
		
while(True):
		
		if (! BATTLE_MODE):
				#Collect Data
				old_time = time()
				#Run OD
				#Run Corner Detection 
				#Update the position and heading values
				
		#record delta time
		delta_t = time()-old_time
		old_time = time()
		# Collect Data
		# Run OD
		# Run Corner Detection
		
		#get new position and heading values
		huey_position # from CD
		huey_facing # from CD
		enemy_position # from OD
		enemy_velocity = calculate_velocity(enemy_position, enemy_previous_positions[-1], delta_t)
		turn = predict_desired_turn(huey_position, enemy_position, enemy_velocity, delta_t)
		speed = predict_desired_speed(huey_position, enemy_position, enemy_velocity, delta_t)
		left = speed - turn 
		right =  speed + turn 

		huey_old_position = huey_position
		# if the array is full, then pop the first one
		enemy_previous_positions.add(enemy_position)
		
		enemy_position = predict_enemy_position(enemy_position, enemy_velocity, delta_t)