import ctypes
import numpy as np
import unittest

class TestSusceptibilityFunctions(unittest.TestCase):

    def setUp(self):
        # Set up test arrays for both functions (initial conditions)
        self.count = 10
        self.susceptibility_timer_uint16 = np.array([5, 3, 1, 6, 0, 2, 7, 4, 9, 8], dtype=np.uint16)
        self.susceptibility_timer_uint8 = np.array([5, 3, 1, 6, 0, 2, 7, 4, 9, 8], dtype=np.uint8)
        
        # Initialize susceptibility arrays with 1 for any element where the timer is already 0
        self.susceptibility_1 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)  # For function 1
        self.susceptibility_2 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)  # For function 2

        self.delta = 1
        self.tick = 0  # Initial tick

        # Load the compiled shared object (.so) file
        lib = ctypes.CDLL("../src/idmlaser_cholera/mods/libmi.so")
        
        # Define the function prototypes
        lib.update_susceptibility_based_on_sus_timer.argtypes = [
            ctypes.c_int32,
            np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),  # susceptibility_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
        ]
        lib.update_susceptibility_based_on_sus_timer.restype = None

        lib.update_susceptibility_timer_strided_shards.argtypes = [
            ctypes.c_int32,
            np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),  # susceptibility_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
            ctypes.c_int32,  # delta
            ctypes.c_int32,  # tick
        ]
        lib.update_susceptibility_timer_strided_shards.restype = None

        self.lib = lib

    def test_update_susceptibility_based_on_sus_timer(self):
        # Call the first function
        self.lib.update_susceptibility_based_on_sus_timer(
            self.count,
            self.susceptibility_timer_uint16,
            self.susceptibility_1
        )
        # Assert checks for expected results (customize this for your specific test case)
        expected_susceptibility_1 = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(self.susceptibility_1, expected_susceptibility_2)

    def test_update_susceptibility_timer_strided_shards(self):
        # Call the second function
        self.lib.update_susceptibility_timer_strided_shards(
            self.count,
            self.susceptibility_timer_uint16,
            self.susceptibility_2,
            self.delta,
            self.tick
        )
        # Compare the whole array against the expected result
        expected_susceptibility_2 = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(self.susceptibility_2, expected_susceptibility_2)

        
if __name__ == '__main__':
    unittest.main()

