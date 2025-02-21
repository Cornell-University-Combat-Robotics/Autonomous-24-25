# Import libraries
import pygame
import math
import time
from ram import Ram

# Initialize pygame
pygame.init()
algo = Ram()
DELAY = 1000 # how often to get bot positions and orientations (milliseconds)

# Set up window dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Control Points")

# Define point colors
point1_color = (255, 0, 0)  # Red (Huey)
point2_color = (0, 0, 255)  # Blue (Enemy)

# Starting coordinates for the two points
# huey = {'center': [width // 4, height // 2], 'orientation': 0.0}  # Huey's position and orientation (0 degrees = along x-axis)
# enemy = {'center': [3 * width // 4, height // 2]}  # Enemy's position

# Intialize Huey and Enemy's positions at the corners of the screen
huey = {'center': [10, 10], 'orientation': 0.0}  # Huey's position and orientation (0 degrees = along x-axis)
enemy = {'center': [width - 10, height - 10]}  # Enemy's position

# Movement speed for both bots
speed = 10
turn_speed = 5  # Degrees per turn for Huey

# Get bot data and send it to the algorithm
def ram_ram(bots={'huey': {'bb': [], 'center': [], 'orientation': 0.0}, 'enemy': {'bb': [], 'center': []}}):
    # Simulate the method receiving bots' data
    print("Updating bots data:")
    print(f"Huey: {bots['huey']}")
    print(f"Enemy: {bots['enemy']}")
    algo.ram_ram(bots)

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

while running:
    screen.fill((255, 255, 255))  # Clear screen with white background

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the state of keys
    keys = pygame.key.get_pressed()

    # Control huey with WAD keys
    if keys[pygame.K_a]:  # Turn left (counterclockwise)
        huey['orientation'] = normalize_angle(huey['orientation'] - turn_speed)
    if keys[pygame.K_d]:  # Turn right (clockwise)
        huey['orientation'] = normalize_angle(huey['orientation'] + turn_speed)
    if keys[pygame.K_w]:  # Move forward
        # Move huey in the direction of its orientation
        huey['center'][0] += speed * math.cos(math.radians(huey['orientation']))
        huey['center'][1] += speed * math.sin(math.radians(huey['orientation']))

    # Control enemy with arrow keys
    if keys[pygame.K_LEFT]:  # Move left
        enemy['center'][0] -= speed
    if keys[pygame.K_RIGHT]:  # Move right
        enemy['center'][0] += speed
    if keys[pygame.K_UP]:  # Move up
        enemy['center'][1] -= speed
    if keys[pygame.K_DOWN]:  # Move down
        enemy['center'][1] += speed

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
                'bb': [huey['center'][0] - 10, huey['center'][1] - 10, 20, 20],  # Example bounding box for huey
                'center': huey['center'],
                'orientation': fix_angle(huey['orientation'])
            },
            'enemy': {
                'bb': [enemy['center'][0] - 10, enemy['center'][1] - 10, 20, 20],  # Example bounding box for enemy
                'center': enemy['center']
            }
        }
        # Call the ram_ram method with the bots' data
        ram_ram(bots=bots_data) 
        last_called_time = current_time  # Update the last called time

    # Draw the points (Huey and Enemy)
    pygame.draw.circle(screen, point1_color, (int(huey['center'][0]), int(huey['center'][1])), 10)  # Draw huey (red)
    pygame.draw.circle(screen, point2_color, (int(enemy['center'][0]), int(enemy['center'][1])), 10)  # Draw enemy (blue)

    # Draw the arrow indicating Huey's orientation
    draw_arrow(screen, point1_color, (int(huey['center'][0]), int(huey['center'][1])), huey['orientation'])

    # Draw a line between huey and enemy
    pygame.draw.line(screen, (0, 0, 0), huey['center'], enemy['center'], 2)

    # Display the orientation of huey
    font = pygame.font.SysFont(None, 36)
    angle_text = font.render(f"Huey Orientation: {fix_angle(huey['orientation']):.2f}°", True, (0, 0, 0))
    screen.blit(angle_text, (10, 10))

    # Update the window
    pygame.display.flip()

    # Cap the frame rate to 60 FPS
    pygame.time.Clock().tick(60)

# Quit pygame
try:
    algo.cleanup()
except:
    print("Algo cleanup failed")
pygame.quit()