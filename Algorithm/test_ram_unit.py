import unittest
from ram import Ram
import numpy as np

algo = Ram()


class TestRam(unittest.TestCase):
    def test_init(self):
        algo = Ram()
        self.assertAlmostEqual(algo.left, 0)
        self.assertAlmostEqual(algo.right, 0)


    def test_huey_move(self):
            # edge cases
            move_dict = algo.huey_move(0,-1)
            self.assertAlmostEqual(move_dict['left'],-0.5)
            self.assertAlmostEqual(move_dict['right'],0.5)

            move_dict = algo.huey_move(0,1)
            self.assertAlmostEqual(move_dict['left'],0.5)
            self.assertAlmostEqual(move_dict['right'],-0.5)

            move_dict = algo.huey_move(1,0)
            self.assertAlmostEqual(move_dict['left'], 0.5)
            self.assertAlmostEqual(move_dict['right'], 0.5)

            move_dict = algo.huey_move(0,0)
            self.assertAlmostEqual(move_dict['left'], 0)
            self.assertAlmostEqual(move_dict['right'], 0)
            
            # intermediate values
            move_dict = algo.huey_move(0.3,-0.5)
            self.assertAlmostEqual(move_dict['left'], -0.1)
            self.assertAlmostEqual(move_dict['right'], 0.4)

            move_dict = algo.huey_move(0.7,0.2)
            self.assertAlmostEqual(move_dict['left'], 0.45)
            self.assertAlmostEqual(move_dict['right'], 0.25)


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
                                                                                enemy_pos=np.array([150,150]), enemy_velocity=np.array([0, 0]), dt = 0.1), -10)
        

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
        
    # def test_predict_desired_turn():
