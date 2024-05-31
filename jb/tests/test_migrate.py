import unittest
import numpy as np
import ctypes
import os
import csv
import settings
import pdb

# Load the shared library
lib = ctypes.CDLL(os.path.abspath('./update_ages.so'))

# Define the argument types for the migrate function
lib.migrate.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=bool, flags="C_CONTIGUOUS"),  # infected
    np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),  # incubation_timer
    np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),  # data_node
    np.ctypeslib.ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),  # data_home_node
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # attraction_probs
    ctypes.c_double,  # migration_fraction
    ctypes.c_int  # num_locations
]


class TestMigrateFunction(unittest.TestCase):

    def setUp(self):
        # Create example data for testing
        self.num_agents = 1000
        self.num_locations = 954

        self.data = np.zeros(self.num_agents, dtype=[('infected', bool), ('node', np.uint32), ('home_node', np.int32), ('incubation_timer', np.uint8)])
        self.data['infected'][0:500] = True  # Set half to infected for testing
        self.data['node'] = np.random.randint(0, self.num_locations, size=self.num_agents)
        self.data['incubation_timer'] = np.random.randint(0, 2, size=self.num_agents)

        def load_attraction_probs():
            probabilities = []
            with open(settings.attraction_probs_file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    probabilities.append([float(prob) for prob in row])
                return np.array(probabilities)
            return probabilities

        self.attraction_probs = load_attraction_probs()

        self.migration_fraction = 0.01

    def test_migrate(self):
                # Make sure arrays are contiguous
        infected = np.ascontiguousarray(self.data['infected'])
        incubation_timer = np.ascontiguousarray(self.data['incubation_timer'])
        data_node = np.ascontiguousarray(self.data['node'])
        data_home_node = np.ascontiguousarray(self.data['home_node'])
        attraction_probs = np.ascontiguousarray(self.attraction_probs)

        # Copy data to check changes after migration
        initial_data = np.copy(self.data)

        # Call the C function
        lib.migrate(
            0,
            self.num_agents,
            infected,
            incubation_timer,
            data_node,
            data_home_node,
            attraction_probs,
            self.migration_fraction,
            self.num_locations
        )

        # Check that the number of migrated individuals is correct
        num_migrated = np.sum(initial_data['node'] != data_node)
        expected_migrated = int(np.sum(initial_data['infected'] & (initial_data['incubation_timer'] <= 0)) * self.migration_fraction)
        print( f"Expected number of migrated = {expected_migrated}." )
        
        self.assertEqual(num_migrated, expected_migrated, "The number of migrated individuals does not match the expected value.")

        # Check that migrated individuals have their home_node updated correctly
        for i in range(self.num_agents):
            if initial_data['node'][i] != self.data['node'][i]:
                self.assertEqual(self.data['home_node'][i], initial_data['node'][i], "Home node not updated correctly for migrated individuals.")

if __name__ == '__main__':
    unittest.main()

