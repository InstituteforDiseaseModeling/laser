import unittest
import numpy as np
import ctypes
import pdb

class TestProgressImmunities(unittest.TestCase):
    def setUp(self):
        # Load the shared library
        self.lib = ctypes.CDLL("./update_ages.so")
        self.lib.progress_immunities.argtypes = [
            ctypes.c_int,  # start_idx
            ctypes.c_int,  # end_idx
            np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'),  # immunity_timer
            np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # immunity
        ]
        self.lib.progress_immunities_avx2.argtypes = [
            ctypes.c_int,  # start_idx
            ctypes.c_int,  # end_idx
            np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'),  # immunity_timer
            np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # immunity
        ]
    def test_no_progress(self):
        """
        Test that no progress occurs when all immunity timers are zero.
        """
        # Define input parameters
        start_idx = 1
        end_idx = 10
        immunity_timer = np.zeros(12, dtype=np.int8)
        immunity = np.zeros(12, dtype=bool)

        # Call the C function
        self.lib.progress_immunities_avx2(start_idx, end_idx, immunity_timer, immunity)

        # Assert that no changes occurred
        self.assertTrue(np.all(immunity_timer == 0))
        self.assertTrue(np.all(immunity == False))

    def test_no_expires(self):
        """
        Test that no immunities expire when nobody's are still high
        """
        # Define input parameters
        start_idx = 1
        end_idx = 10
        immunity_timer = np.ones(12, dtype=np.int8)*44
        immunity_timer[0] = 1
        immunity_timer[11] = 1
        immunity = np.ones(12, dtype=bool)
        #immunity[0] = 0
        #immunity[11] = 0

        # Call the C function
        self.lib.progress_immunities_avx2(start_idx, end_idx, immunity_timer, immunity)

        # Assert that no changes occurred
        self.assertTrue(np.all(immunity_timer[1:10] == 43))
        self.assertTrue(np.all(immunity_timer[0] == 1))
        self.assertTrue(np.all(immunity_timer[11] == 1))
        self.assertTrue(np.all(immunity[1:10] == True))

    def test_some_progress(self):
        """
        Test that progress occurs for some individuals with non-zero immunity timers.
        """
        # Define input parameters
        start_idx = 1
        end_idx = 10
        immunity_timer = np.array([3, 2, 1, 0, 5, 4, 3, 0, 2, 1, 6, 7], dtype=np.int8)
        immunity = np.array([True, True, True, False, True, True, True, False, True, True, True, True], dtype=bool)

        # Call the C function
        self.lib.progress_immunities_avx2(start_idx, end_idx, immunity_timer, immunity)

        # Assert that progress occurred
        expected_immunity_timer = np.array([3, 1, 0, 0, 4, 3, 2, 0, 1, 0, 5, 7], dtype=np.int8)
        expected_immunity = np.array([True, True, False, False, True, True, True, False, True, False, True, True], dtype=bool)
        self.assertTrue(np.all(immunity_timer == expected_immunity_timer))
        #print( f"ref: {expected_immunity}" )
        #print( f"test: {immunity}" )
        self.assertTrue((immunity.tolist() == expected_immunity.tolist()), "Immunity state (true/false) after progression different from expected.")

    def test_large_progress_multiple_of_32(self):
        """
        Test that progress occurs correctly for large arrays where the size is a multiple of 32.
        """
        # Define input parameters
        start_idx = 0
        end_idx = 63
        immunity_timer = np.random.randint(0, 10, size=64).astype(np.int8)
        immunity = np.random.choice([True, False], size=64).astype(bool)

        # Create expected results
        expected_immunity_timer = immunity_timer.copy()
        expected_immunity = immunity.copy()

        # Manually compute expected results
        for i in range(start_idx, end_idx + 1):
            if expected_immunity[i] and expected_immunity_timer[i] > 0:
                expected_immunity_timer[i] -= 1
                if expected_immunity_timer[i] == 0:
                    expected_immunity[i] = False

        # Call the C function
        self.lib.progress_immunities_avx2(start_idx, end_idx, immunity_timer, immunity)
        #self.lib.progress_immunities(start_idx, end_idx, immunity_timer, immunity)

        # Assert that progress occurred
        print( f"Actual (timers): {immunity_timer}" )
        print( f"Expected (timers): {expected_immunity_timer}" )
        self.assertTrue(np.all(immunity_timer == expected_immunity_timer), "Immunity timers after progression different from expected.")

        print( f"Actual (bools): {immunity}" )
        print( f"Expected (bools): {expected_immunity}" )
        for i in range(len(immunity)):
            if immunity[i] != expected_immunity[i]:
                print(f"Mismatch at index {i}: Actual = {immunity[i]}, Expected = {expected_immunity[i]}")

        self.assertTrue((immunity.tolist() == expected_immunity.tolist()), "Immunity state (true/false) after progression different from expected.")

if __name__ == "__main__":
    unittest.main()

