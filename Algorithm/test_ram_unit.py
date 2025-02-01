import unittest
from Algorithm.ram import Ram
import numpy as np
import time


algo = Ram()


class TestRam(unittest.TestCase):
    def test_init(self):
        algo = Ram()
        self.assertAlmostEqual(algo.left, 0)
        self.assertAlmostEqual(algo.right, 0)


    def test_huey_move(self):
            # edge cases
            move_dict = algo.huey_move(0,-1)
            self.assertAlmostEqual(move_dict['left'],0.5)
            self.assertAlmostEqual(move_dict['right'],-0.5)

            move_dict = algo.huey_move(0,1)
            self.assertAlmostEqual(move_dict['left'],-0.5)
            self.assertAlmostEqual(move_dict['right'],0.5)

            move_dict = algo.huey_move(1,0)
            self.assertAlmostEqual(move_dict['left'], 0.5)
            self.assertAlmostEqual(move_dict['right'], 0.5)

            move_dict = algo.huey_move(0,0)
            self.assertAlmostEqual(move_dict['left'], 0)
            self.assertAlmostEqual(move_dict['right'], 0)
            
            # intermediate values
            move_dict = algo.huey_move(0.3,-0.5)
            self.assertAlmostEqual(move_dict['left'], 0.4)
            self.assertAlmostEqual(move_dict['right'], -0.1)

            move_dict = algo.huey_move(0.7,0.2)
            self.assertAlmostEqual(move_dict['left'], 0.25)
            self.assertAlmostEqual(move_dict['right'], 0.45)


    def test_calculate_velocity(self):
            #Edge Cases:
            arr1 = algo.calculate_velocity(np.array([0,0]),np.array([0,0]), 0)
            arr2 = np.array([0,0])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity(np.array([0,0]),np.array([100,0]),1.0)
            arr2 = np.array([100,0])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity(np.array([0,0]),np.array([0,100]),1.0)
            arr2 = np.array([0,100])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity(np.array([100,0]),np.array([0,0]),1.0)
            arr2 = np.array([-100,0])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity(np.array([0,100]),np.array([0,0]),1.0)
            arr2 = np.array([0,-100])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity(np.array([0,0]),np.array([100,0]),10.0)
            arr2 = np.array([10,0])
            np.testing.assert_almost_equal(arr1, arr2)

            #Regular Cases:
            arr1 = algo.calculate_velocity(np.array([0,20]),np.array([50,40]),1.0)
            arr2 = np.array([50,20])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity(np.array([100,50]),np.array([10,30]),1.0)
            arr2 = np.array([-90,-20])
            np.testing.assert_almost_equal(arr1, arr2)
         

    def test_acceleration(self):
                 #Edge Cases:
            arr1 = algo.acceleration(np.array([0,0]),np.array([0,0]), 0)
            arr2 = np.array([0,0])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.acceleration(np.array([0,0]),np.array([100,0]),1.0)
            arr2 = np.array([100,0])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.acceleration(np.array([0,0]),np.array([0,100]),1.0)
            arr2 = np.array([0,100])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.acceleration(np.array([100,0]),np.array([0,0]),1.0)
            arr2 = np.array([-100,0])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.acceleration(np.array([0,100]),np.array([0,0]),1.0)
            arr2 = np.array([0,-100])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.acceleration(np.array([0,0]),np.array([100,0]),10.0)
            arr2 = np.array([10,0])
            np.testing.assert_almost_equal(arr1, arr2)

            #Regular Cases:
            arr1 = algo.acceleration(np.array([0,20]),np.array([50,40]),1.0)
            arr2 = np.array([50,20])
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.acceleration(np.array([100,50]),np.array([10,30]),1.0)
            arr2 = np.array([-90,-20])
            np.testing.assert_almost_equal(arr1, arr2)
        
         

    def test_predict_enemy_position(self):
        arr1 = algo.predict_enemy_position(np.array([0,0]), np.array([0,0]), .001)
        np.testing.assert_almost_equal(arr1, np.array([0,0]))
        
        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([0,0]), .001)
        np.testing.assert_almost_equal(arr1, np.array([100,100]))

        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([10,10]), 0.0)
        np.testing.assert_almost_equal(arr1, np.array([100,100]))
        
        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([0,10]), 0.1)
        np.testing.assert_almost_equal(arr1, np.array([100,101]))
        
        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([10,0]), 0.1)
        np.testing.assert_almost_equal(arr1, np.array([101,100]))

        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([10,10]), 0.1)
        np.testing.assert_almost_equal(arr1, np.array([101,101]))

        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([-100,100]), 0.1)
        np.testing.assert_almost_equal(arr1, np.array([90,110]))

        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([100,-100]), 0.1)
        np.testing.assert_almost_equal(arr1, np.array([110,90]))

        arr1 = algo.predict_enemy_position(np.array([100,100]), np.array([-100,-100]), 0.1)
        np.testing.assert_almost_equal(arr1, np.array([90,90]))


    def test_invert_y(self):
        np.testing.assert_almost_equal(algo.invert_y(np.array([0,0])), np.array([0,0]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([1,1])), np.array([1,-1]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([-1,1])), np.array([-1,-1]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([0,-1])), np.array([0,1]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([-1,0])), np.array([-1,0]))

    def test_predict_desired_orientation_angle(self):
        ## Edge cases
        # 0 degrees
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([100,0]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0)
        # -90 degre
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), -90)
        # 180 degrees
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([100,600]), enemy_velocity=np.array([0, 0]), dt = 0.1), -180)
        # 90 degrees
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([0,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 90)

        # -40 degrees
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([150,36.002981839]), enemy_velocity=np.array([0, 0]), dt = 0.1), -38, decimal=2)
        
        #  dt = 0
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([100, 100]), dt = 0), -90)
        
        # velocity = west
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([-100, 0]), dt = 0.1), 0)
        # velocity = east
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([100, 0]), dt = 0.1), 0)
        # velocity = south
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, 100]), dt = 0.1), -5.710593137)

        # velocity = north
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, -100]), dt = 0.1), 5.710593137)
        
        # bots are overlapping
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=90,  
                                                                                enemy_pos=np.array([100,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0)
        
        # Funky case
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=100,
                                                                                enemy_pos=np.array([150,150]), enemy_velocity=np.array([200, 300]), dt = 0.1), -148.81407483)
        
        # orientation > 180
        np.testing.assert_almost_equal(algo.predict_desired_orientation_angle(our_pos=np.array([100,100]), our_orientation=270,
                                                                                enemy_pos=np.array([150,150]), enemy_velocity=np.array([200, 300]), dt = 0.1), 41.1859252)
        

    def test_predict_desired_turn(self):
        #Edge Cases
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([100,0]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0)
        # -90 degre
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), -0.5)
        # 180 degrees
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([100,600]), enemy_velocity=np.array([0, 0]), dt = 0.1), -1)
        # 90 degrees
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([0,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0.5)

        # -38 degrees
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([150,36.002981839]), enemy_velocity=np.array([0, 0]), dt = 0.1), -0.211111111, decimal=2)
        #  dt = 0
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([100, 100]), dt = 0), -0.5)
        # velocity = west
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([-100, 0]), dt = 0.1), 0)
        # velocity = east
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([100, 0]), dt = 0.1), 0)
        # velocity = south
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, 100]), dt = 0.1), -0.03172551742)

        # velocity = north
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, -100]), dt = 0.1), 0.03172551742)
        
        # bots are overlapping
        np.testing.assert_almost_equal(algo.predict_desired_turn(our_pos=np.array([100,100]), our_orientation=90,  
                                                                                enemy_pos=np.array([100,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0)
        
    def test_desired_speed(self):
        #Edge Cases
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([100,0]), enemy_velocity=np.array([0, 0]), dt = 0.1), 1)
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0.5)
         # 180 degrees
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([100,600]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0)
        # 90 degrees
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([0,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0.5)

        # -38 degrees
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([150,36.002981839]), enemy_velocity=np.array([0, 0]), dt = 0.1), 0.788888889, decimal=2)
        #  dt = 0
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([100, 100]), dt = 0), 0.5)
        # velocity = west
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([-100, 0]), dt = 0.1), 1)
        # velocity = east
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([100, 0]), dt = 0.1), 1)
        # velocity = south
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, 100]), dt = 0.1), 0.96827448258)

        # velocity = north
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=0,  
                                                                              enemy_pos=np.array([200,100]), enemy_velocity=np.array([0, -100]), dt = 0.1),0.96827448258)
        
        # bots are overlapping
        np.testing.assert_almost_equal(algo.predict_desired_speed(our_pos=np.array([100,100]), our_orientation=90,  
                                                                               enemy_pos=np.array([100,100]), enemy_velocity=np.array([0, 0]), dt = 0.1), 1)

    def test_ram_ram(self):
        algo = Ram(huey_old_position=np.array([10,10]), huey_position=np.array([10,10]), enemy_position=np.array([10,590]))
        
        bots1 = {'huey': {'bb': [0, 0, 20, 20], 'center': [10, 10], 'orientation': 0.0}, 'enemy': {'bb': [780, 580, 20, 20], 'center': [10, 590]}}
        values = algo.ram_ram(bots1)
        self.assertAlmostEqual(values['left'], .5) 
        self.assertAlmostEqual(values['right'], 0) 
        
        bots2 = {'huey': {'bb': [138, 154, 20, 20], 'center': [148, 164], 'orientation': 270.0}, 'enemy': {'bb': [500, 210, 20, 20], 'center': [510, 220]}}
        
        og_time = time.time()
        values = algo.ram_ram(bots2)
        dt = time.time() - og_time
        
        self.assertAlmostEqual(values['left'], -0.2223902366185668888888888888888888888888888888888888888888888888888888888/2, places = 2)
        self.assertAlmostEqual(values['right'], 0.9999999999994331111111111111111111111111111111111111111111111111111111111/2, places = 2)

        bots3 = {'huey': {'bb': [199.21307092309192, 397.12606428385743, 20, 20], 'center': [209.21307092309192, 407.12606428385743], 'orientation': 90}, 'enemy': {'bb': [500, 210, 20, 20], 'center': [510, 220]}}
        print("Test: Enemy doesn't move")
        values = algo.ram_ram(bots3)
        self.assertAlmostEqual(values['left'],0.5, places =3)
        self.assertAlmostEqual(values['right'],0.1771477778, places =3)



        # enemy is currently in danger zone -- use enemy current position
        print("Test: Enemy is in the Danger Zone")
        bots4 = {'huey': {'bb': [99.13626817533684, 582.1403853383393, 20, 20], 'center': [109.13626817533684, 592.1403853383393], 'orientation': 325.0}, 'enemy': {'bb': [500, 210, 20, 20], 'center': [120, 540]}}
        values = algo.ram_ram(bots4)
        # 
        self.assertAlmostEqual(values['left'], -0.258111111/2, places=3)
        self.assertAlmostEqual(values['right'], 0.4996944444, places =3)
        

        # if future position is our position
        print("Test: Future position is current position")
        bots5 = {'huey': {'bb': [99, 582., 20, 20], 'center': [185, 410], 'orientation': 325.0}, 'enemy': {'bb': [500, 210, 20, 20], 'center': [152.5, 475]}}
        values = algo.ram_ram(bots5)
        self.assertAlmostEqual(values['left'], 0.5, places=3)
        self.assertAlmostEqual(values['right'], 0.5, places =3)
