from ram import Ram
import numpy as np
import time

def normalize_angle(angle):
    if angle < 0:
        angle += 360
    elif angle >= 360:
        angle -= 360
    return angle

def fix_angle(angle):
    angle = 360 - angle
    return normalize_angle(angle)

def parse(lines, index):
    center_str = lines[index]
    counter = 1
    counter2 = 0
    huey_strs = []
    while (counter < len(center_str)-1):
        huey_strs.append("")
        while (center_str[counter] != "," and counter < len(center_str)-1):
            huey_strs[counter2] = huey_strs[counter2] + center_str[counter]
            counter+=1
        counter2 += 1
    return huey_strs

first = True
while (True):
    #read from unity_writes
    lines = []
    while (len(lines) == 0):
        file = open('unity_write.txt', 'r')
        lines = file.readlines()
        file.close()

    # values from file: huey_center, huey_orientation, enemy_center
    # constants: robot width, robot height

    #parse and calculate valuesa
    # all units in meters!

    huey_center = np.array([float(lines[0]), -float(lines[1])])
    huey_orientation = float(lines[2]) + 90.0
    enemy_center = np.array([float(lines[3]), -float(lines[4])])

    robot_width = 0.23495
    robot_height = 0.19685

    huey_width = robot_width
    huey_height = robot_height
    huey_bottom_center = huey_center - huey_height
    huey_left_center = huey_center - huey_width

    enemy_width = robot_width
    enemy_height = robot_height
    enemy_bottom_center = enemy_center - enemy_height
    enemy_left_center = enemy_center - enemy_width

    #run values through ram
    bots_data = {
                'huey': {
                    'bbox': [huey_bottom_center, huey_left_center, huey_width, huey_height],  # Example bounding box for huey
                    'center': huey_center,
                    'orientation': fix_angle(huey_orientation)
                },
                'enemy': {
                    'bbox': [enemy_bottom_center, enemy_left_center, enemy_width, enemy_height],  # Example bounding box for enemy
                    'center': enemy_center
                }
        }
    
    if (first):
        huey_bot = Ram(huey_position=huey_center, huey_orientation=huey_orientation, huey_old_position=huey_orientation, enemy_position=enemy_center)
        first = False
    
    huey_move_output = huey_bot.ram_ram(bots_data)

    #write output to unity_reads
    file = open('unity_reads.txt', 'w')
    lines = file.writelines(str(huey_move_output['left'])+ "\n" + str(huey_move_output['right']) + "\n" + str(huey_move_output['speed']) + "\n" + str(huey_move_output['turn']))
    file.close()