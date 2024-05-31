import unittest
from ctypes import CDLL, c_uint, c_float, byref
import ctypes
import numpy as np
import os

# Load the shared object (.so) library
# Get the path to the .so file in the parent directory
#parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
#so_file = os.path.join(parent_dir, "update_ages.so")

lib = CDLL("./update_ages.so")

# Define the function signature
lib.update_ages.argtypes = [
    ctypes.c_size_t,  # start_idx
    ctypes.c_size_t,  # stop_idx
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
]
lib.update_ages.restype = None

# Define the increment in days
ONE_DAY = 1.0 / 365.0
class TestUpdateAges(unittest.TestCase):
    def test_positive_ages(self):
        ages = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]).astype(np.float32)
        expected_result = [age + ONE_DAY for age in ages]
        lib.update_ages(0, 9, ages)
        # Compare each element with tolerance
        for i in range(len(ages)):
            self.assertAlmostEqual(ages[i], expected_result[i], places=5)

    def test_negative_ages(self):
        ages = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, -100.0]).astype(np.float32)
        lib.update_ages(0, 9, ages)
        self.assertEqual(list(ages), [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, -100.0])

    def test_mixed_ages(self):
        ages = np.array([10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 70.0, -80.0, 90.0, -100.0]).astype(np.float32)
        #expected_result = [11.0, -20.0, 31.0, -40.0, 51.0, -60.0, 71.0, -80.0, 91.0, -100.0]
        expected_result = []
        for age in ages:
            if age > 0:
                # Increment positive ages by ONE_DAY
                expected_result.append(age + ONE_DAY)
            else:
                expected_result.append(age)
        lib.update_ages(0, 9, ages)
        for i in range(len(ages)):
            self.assertAlmostEqual(ages[i], expected_result[i], places=5)

if __name__ == "__main__":
    unittest.main()

