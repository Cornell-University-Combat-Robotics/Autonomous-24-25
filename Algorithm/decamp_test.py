# Import libraries
import pygame
import math
import time
from ram import Ram
import numpy as np

# Initialize pygame
pygame.init()
algo = Ram()
DELAY = 100 # how often to get bot positions and orientations (milliseconds)

# Set up window dimensions
width, height = 243*3, 243*3
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Control Points")

# Define point colors
point1_color = (255, 0, 0)  # Red (Huey)
point2_color = (0, 0, 255)  # Blue (Enemy)
point3_color = (0, 0, 0)    # Black (Arrow)

# Intialize Huey and Enemy's positions at the corners of the screen
huey = {'center': [10, 10], 'orientation': 0.0}  # Huey's position and orientation (0 degrees = along x-axis)
enemy = {'center': [width - 10, height - 10]}  # Enemy's position

# Normalize the angle to be between 0 and 360 degrees
def normalize_angle(angle):
    if angle < 0:
        angle += 360
    elif angle >= 360:
        angle -= 360
    return angle

# Draw an arrow to represent Huey's orientation
def draw_arrow(surface, color, position, orientation, size=20):
    """Draw an arrow pointing in the direction of 'orientation'."""
    arrow_length = size
    # Calculate the end position of the arrow based on the orientation
    angle_rad = math.radians(orientation)
    end_x = position[0] + arrow_length * math.cos(angle_rad)
    end_y = position[1] + arrow_length * math.sin(angle_rad)
    
    pygame.draw.line(surface, color, position, (end_x, end_y), 3)  # Draw arrow line

    # Optionally, add an arrowhead (triangle) to the line
    arrowhead_size = 10
    arrowhead_angle = math.radians(30)  # 30 degree angle for the arrowhead
    dx = end_x - position[0]
    dy = end_y - position[1]
    
    # Create two points for the arrowhead
    angle1 = math.atan2(dy, dx) + arrowhead_angle
    angle2 = math.atan2(dy, dx) - arrowhead_angle
    pygame.draw.polygon(surface, color, [
        (end_x, end_y),
        (end_x - arrowhead_size * math.cos(angle1), end_y - arrowhead_size * math.sin(angle1)),
        (end_x - arrowhead_size * math.cos(angle2), end_y - arrowhead_size * math.sin(angle2)),
    ])

# fixes angle so positive and negative is conventional
def fix_angle(angle):
    angle = 360 - angle
    return normalize_angle(angle)

# Main game loop
running = True
last_called_time = pygame.time.get_ticks()  # Time of last method call (in milliseconds)

old_pos = enemy['center']

bots_data = {
            'huey': {
                'bbox': [huey['center'][0] - 10, huey['center'][1] - 10, 20, 20],  # Example bounding box for huey
                'center': huey['center'],
                'orientation': fix_angle(huey['orientation'])
            },
            'enemy': {
                'bbox': [enemy['center'][0] - 10, enemy['center'][1] - 10, 20, 20],  # Example bounding box for enemy
                'center': enemy['center']
            }
    }
huey_bot = Ram(huey_old_position=huey['center'], huey_orientation=huey['orientation'], enemy_position=enemy['center'])

while running:
    if (math.dist([enemy['center'][0], enemy['center'][1]], [huey['center'][0], huey['center'][1]]) < 0.203*300):
        keys = pygame.key.get_pressed()
        while not keys[pygame.K_SPACE]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            time.sleep(0.01)
            keys = pygame.key.get_pressed()
            
        huey = {'center': [10, 10], 'orientation': 0.0}  # Huey's position and orientation (0 degrees = along x-axis)
        enemy = {'center': [width - 10, height - 10]}  # Enemy's position


    screen.fill((255, 255, 255))  # Clear screen with white background

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the state of keys
    keys = pygame.key.get_pressed()

    # Movement speed for both bots
    # 300 pixels per meter
    # Robot wheelbase is 7.5 in (0.1905 m) (from half wheel to half wheel)
    huey_move_output = huey_bot.ram_ram(bots_data)

    # things to account for: power, friction
    
    huey_current_speed = 5 # (pixels moved in this frame)

    enemy_current_speed = 30 # (enemy pixles moved in this frame)
    wheel_base = 0.1905

    # turn speed = leftwheel speed - rightwheel speed / base width
    huey_turn_speed = ((huey_move_output['right'] - huey_move_output['left']) * (huey_current_speed)) / (wheel_base) # max huey speed
    huey_speed = ((huey_move_output['left'] + huey_move_output['right']) * (huey_current_speed)) / 2.0 # max huey speed

    old_pos = enemy['center']

    # Huey change
    huey['orientation'] = normalize_angle(huey['orientation'] - huey_turn_speed)
    fixed_angle = fix_angle(huey['orientation'])
    huey['center'][0] += huey_speed * math.cos(math.radians(fix_angle(huey['orientation'])))
    huey['center'][1] -= huey_speed * math.sin(math.radians(fix_angle(huey['orientation'])))
    #huey_y_change = -1 * huey_speed * math.sin(math.radians(fix_angle(huey['orientation'])))

    # Control enemy with arrow keys
    if keys[pygame.K_LEFT]:  # Move left
        enemy['center'][0] -= enemy_current_speed
    elif keys[pygame.K_RIGHT]:  # Move right
        enemy['center'][0] += enemy_current_speed
    elif keys[pygame.K_UP]:  # Move up
        enemy['center'][1] -= enemy_current_speed
    elif keys[pygame.K_DOWN]:  # Move down
        enemy['center'][1] += enemy_current_speed

    # Ensure huey stays within screen bounds
    if huey['center'][0] < 0: huey['center'][0] = 0
    if huey['center'][0] > width: huey['center'][0] = width
    if huey['center'][1] < 0: huey['center'][1] = 0
    if huey['center'][1] > height: huey['center'][1] = height
    # Ensure enemy stays within screen bounds
    if enemy['center'][0] < 0: enemy['center'][0] = 0
    if enemy['center'][0] > width: enemy['center'][0] = width
    if enemy['center'][1] < 0: enemy['center'][1] = 0
    if enemy['center'][1] > height: enemy['center'][1] = height


    # Implement DELAY is values are only fed to the algorithm after DELAY ms
    current_time = pygame.time.get_ticks()
    if current_time - last_called_time >= DELAY:  
        # Prepare the data to pass to ram_ram
        bots_data = {
            'huey': {
                'bbox': [huey['center'][0] - 10, huey['center'][1] - 10, 20, 20],  # Example bounding box for huey
                'center': huey['center'],
                'orientation': fix_angle(huey['orientation'])
            },
            'enemy': {
                'bbox': [enemy['center'][0] - 10, enemy['center'][1] - 10, 20, 20],  # Example bounding box for enemy
                'center': enemy['center']
            }
        }
        # Call the ram_ram method with the bots' data
        # ram_ram(bots=bots_data) 
        last_called_time = current_time  # Update the last called time

    # Draw the bots (Huey and Enemy)
    # width, length of huey bot : 8 in, 9.375 in
    diagonal = 0.1565197716182 * 300
    theta = 49.52
    phi = 40.48

    huey_coords =  [(diagonal * math.cos(math.radians(fix_angle(theta - huey['orientation']))) + huey['center'][0],      diagonal * math.sin(math.radians(fix_angle(theta - huey['orientation']))) + huey['center'][1]),
               (diagonal * math.cos(math.radians(fix_angle(theta + 2*phi - huey['orientation']))) + huey['center'][0],   diagonal * math.sin(math.radians(fix_angle(theta + 2*phi - huey['orientation']))) + huey['center'][1]), 
               (diagonal * math.cos(math.radians(fix_angle(3*theta + 2*phi - huey['orientation']))) + huey['center'][0], diagonal * math.sin(math.radians(fix_angle(3*theta + 2*phi - huey['orientation']))) + huey['center'][1]), 
               (diagonal * math.cos(math.radians(fix_angle(3*theta + 4*phi - huey['orientation']))) + huey['center'][0], diagonal * math.sin(math.radians(fix_angle(3*theta + 4*phi - huey['orientation']))) + huey['center'][1])]

    enemy_coords =  [(diagonal * math.cos(math.radians(fix_angle(45))) + enemy['center'][0], diagonal * math.sin(math.radians(fix_angle(45))) + enemy['center'][1]),
               (diagonal * math.cos(math.radians(fix_angle(135))) + enemy['center'][0],   diagonal * math.sin(math.radians(fix_angle(135))) + enemy['center'][1]), 
               (diagonal * math.cos(math.radians(fix_angle(225))) + enemy['center'][0], diagonal * math.sin(math.radians(fix_angle(225))) + enemy['center'][1]), 
               (diagonal * math.cos(math.radians(fix_angle(315))) + enemy['center'][0], diagonal * math.sin(math.radians(fix_angle(315))) + enemy['center'][1])]
    
    #pygame.draw.rect(screen, point1_color, huey_rect, 5)  # Draw huey (red)
    pygame.draw.polygon(screen, point1_color, huey_coords)
    pygame.draw.polygon(screen, point2_color, enemy_coords)  # Draw enemy (blue)

    #pygame.draw.circle(screen, point2_color, (int(huey['center'][0]), int(huey['center'][1])), 10)  # Draw enemy (blue)
    #pygame.draw.circle(screen, point2_color, (int(enemy['center'][0]), int(enemy['center'][1])), 10)  # Draw enemy (blue)

    # Draw the arrow indicating Huey's orientation
    draw_arrow(screen, point3_color, (int(huey['center'][0]), int(huey['center'][1])), huey['orientation'])

    # Draw a line between huey and enemy
    pygame.draw.line(screen, (0, 0, 0), huey['center'], enemy['center'], 1)

    # Update the window
    pygame.display.flip()

    # Cap the frame rate to 60 FPS
    pygame.time.Clock().tick(60)

    time.sleep(0.01)

# Quit pygame
try:
    algo.cleanup()
except:
    print("Algo cleanup failed")
pygame.quit()