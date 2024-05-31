import unittest
import ctypes
import numpy as np

# Load the shared library
update_ages_lib = ctypes.CDLL('./update_ages.so')

# Define the function signature
update_ages_lib.init_maps.argtypes = [
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # start_idx
    np.ctypeslib.ndpointer(bool, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # infection_timer
]

# Define the function signature
update_ages_lib.progress_infections2.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # start_idx
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # infection_timer
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # incubation_timer
    np.ctypeslib.ndpointer(bool, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # immunity_timer
    np.ctypeslib.ndpointer(bool, flags='C_CONTIGUOUS'),  # immunity
    ctypes.c_int,  # timestep
]

class TestProgressInfections2(unittest.TestCase):
    def test_infection_progression(self):
        # Define input parameters
        start_idx = 0
        timestep = 1
        infection_timer = np.array( [4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 4.0 ], dtype=np.float32)
        incubation_timer = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ], dtype=np.float32)
        infected = np.array(        [True, True, True, True, True, True, True, True], dtype=bool)
        immunity_timer = np.array(  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        immunity = np.array(        [False, False, False, False, False, False, False, False], dtype=bool)
        n = len(infection_timer)

        expected_infection_timer = {
                1: [ 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 4.0 ],
                2: [ 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 4.0 ],
                3: [ 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 4.0 ],
                4: [ 0.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 0.0 ],
                5: [ 0.0, 0.0, 6.0, 7.0, 8.0, 8.0, 7.0, 0.0 ],
                6: [ 0.0, 0.0, 0.0, 7.0, 8.0, 8.0, 7.0, 0.0 ],
                7: [ 0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0 ],
                8: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                9: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                10:[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
            }
        expected_infected = {
                1: [True, True, True, True, True, True, True, True],
                2: [True, True, True, True, True, True, True, True],
                3: [True, True, True, True, True, True, True, True],
                4: [False, True, True, True, True, True, True, False],
                5: [False, False, True, True, True, True, True, False],
                6: [False, False, False, True, True, True, True, False],
                7: [False, False, False, False, True, True, False, False],
                8: [False, False, False, False, False, False, False, False],
                9: [False, False, False, False, False, False, False, False],
                10:[False, False, False, False, False, False, False, False],
            }
        expected_immunity = {
                1: [False, False, False, False, False, False, False, False],
                2: [False, False, False, False, False, False, False, False],
                3: [False, False, False, False, False, False, False, False],
                4: [True, False, False, False, False, False, False, False],
                5: [True, True, False, False, False, False, False, False],
                6: [True, True, True, False, False, False, False, False],
                7: [True, True, True, True, False, False, False, False],
                8: [True, True, True, True, True, True, True, True],
                9: [True, True, True, True, True, True, True, True],
                10: [True, True, True, True, True, True, True, True],
            }
        expected_immunity_timer = {
                1: [0.0, 0.0, 0.0, 0.0, 0.0],
                2: [0.0, 0.0, 0.0, 0.0, 0.0],
                3: [0.0, 0.0, 0.0, 0.0, 0.0],
                4: [-1.0, 0.0, 0.0, 0.0, 0.0],
                5: [-1.0, -1.0, 0.0, 0.0, 0.0],
                6: [-1.0, -1.0, -1.0, 0.0, 0.0],
                7: [-1.0, -1.0, -1.0, -1.0, 0.0],
                8: [-1.0, -1.0, -1.0, -1.0, -1.0],
                9: [-1.0, -1.0, -1.0, -1.0, -1.0],
                10: [-1.0, -1.0, -1.0, -1.0, -1.0],
            }
        expected_incubation_timer = {
                1: [ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ],
                2: [ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ],
                3: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                4: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                5: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                6: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                7: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                8: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                9: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                10:[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
            }
        # Call the init_maps function
        update_ages_lib.init_maps(
            ctypes.c_size_t(n),
            ctypes.c_size_t(start_idx),
            infected,
            infection_timer
        )

        # Call the progress_infections2 function multiple times
        for idx in range(1,10):
            print( f"Testing timestep {idx}" )
            # Call the progress_infections2 function
            update_ages_lib.progress_infections2(
                ctypes.c_int(n),
                ctypes.c_int(start_idx),
                infection_timer,
                incubation_timer,
                infected,
                immunity_timer,
                immunity,
                ctypes.c_int(timestep)
            )

            # Check current state
            np.testing.assert_array_equal(incubation_timer, np.array( expected_incubation_timer[idx] ) )
            np.testing.assert_array_equal(infected, np.array( expected_infected[idx] ) )
            np.testing.assert_array_equal(infection_timer, np.array( expected_infection_timer[idx]) )
            np.testing.assert_array_equal(immunity, ~np.array( expected_infected[idx] ) )
            np.testing.assert_array_equal(immunity_timer, (~np.array( expected_infected[idx] )).astype( np.int32 )*-1.0 )
            #np.testing.assert_array_equal(immunity_timer, np.array( expected_immunity_timer[idx]) )

            # Increment timestep
            timestep += 1


if __name__ == '__main__':
    unittest.main()

