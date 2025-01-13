import unittest
from ram import Ram
import numpy as np

algo = Ram()


class TestRam(unittest.TestCase):
    def test_init(self):
        self.assertEqual(Ram.left, 0)
        self.assertEqual(Ram.right, 0)
        self.assertEqual(Ram.enemy_previous)


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
            arr1 = algo.calculate_velocity([0,0],[0,0], 0)
            arr2 = [0,0]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,0],[100,0],1.0)
            arr2 = [100,0]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,0],[0,100],1.0)
            arr2 = [0,100]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([100,0],[0,0],1.0)
            arr2 = [-100,0]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,100],[0,0],1.0)
            arr2 = [0,-100]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,0],[100,0],10.0)
            arr2 = [10,0]
            np.testing.assert_almost_equal(arr1, arr2)

            #Regular Cases:
            arr1 = algo.calculate_velocity([0,20],[50,40],1.0)
            arr2 = [50,20]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([100,50],[10,30],1.0)
            arr2 = [-90,-20]
            np.testing.assert_almost_equal(arr1, arr2)
         

    def test_acceleration(self):
                 #Edge Cases:
            arr1 = algo.calculate_velocity([0,0],[0,0], 0)
            arr2 = [0,0]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,0],[100,0],1.0)
            arr2 = [100,0]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,0],[0,100],1.0)
            arr2 = [0,100]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([100,0],[0,0],1.0)
            arr2 = [-100,0]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,100],[0,0],1.0)
            arr2 = [0,-100]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([0,0],[100,0],10.0)
            arr2 = [10,0]
            np.testing.assert_almost_equal(arr1, arr2)

            #Regular Cases:
            arr1 = algo.calculate_velocity([0,20],[50,40],1.0)
            arr2 = [50,20]
            np.testing.assert_almost_equal(arr1, arr2)
            arr1 = algo.calculate_velocity([100,50],[10,30],1.0)
            arr2 = [-90,-20]
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
        np.testing.assert_almost_equal(algo.invert_y(np.array([1,1])), np.array([-1,-1]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([-1,1])), np.array([1,-1]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([0,-1])), np.array([0,1]))
        np.testing.assert_almost_equal(algo.invert_y(np.array([-1,0])), np.array([1,0]))
    
    
