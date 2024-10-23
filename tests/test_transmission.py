import unittest
from unittest.mock import patch
import numpy as np
from idmlaser_cholera.mods.transmission import calculate_new_infections_by_node
from pkg_resources import resource_filename
import ctypes
import pdb

class TestCalculateNewInfectionsByNode(unittest.TestCase):

    def setUp( self ):
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libtx.so')
        self.lib = ctypes.CDLL(shared_lib_path)

        # Define the argument types for the C function
        self.lib.tx_inner_nodes.argtypes = [
            ctypes.c_uint32,                                                        # count
            ctypes.c_uint32,                                                        # num_nodes
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # etimers
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # new_infections,
            ctypes.c_float,                                                           # exp_mean
            ctypes.POINTER(ctypes.c_uint32)  # new_ids_out (pointer to uint32)
        ]

        self.population_size = 1000  # Simulated population size
        self.num_nodes = 5  # Assume there are 5 nodes

        # Population structure setup
        self.susceptibility = np.random.randint(0, 2, size=self.population_size, dtype=np.uint8)  # Random susceptibility
        self.etimers = np.random.randint(0, 10, size=self.population_size, dtype=np.uint8)  # Random incubation timers
        self.new_infections = np.zeros(self.num_nodes, dtype=np.uint32)  # Will hold new infections
        self.exp_mean = 3.0  # Arbitrary constant for incubation period

        # Output buffer for infected IDs (enough space for infected individuals)
        self.infected_ids_buffer = np.zeros(self.population_size, dtype=np.uint32)
        #self.infected_ids_buffer = list()


    def call_tx_inner_nodes(self):
        """Helper function to call the C function tx_inner_nodes"""
        self.lib.tx_inner_nodes(
            ctypes.c_uint32(self.population_size),
            ctypes.c_uint32(self.num_nodes),
            self.susceptibility,
            self.etimers,
            self.new_infections,
            ctypes.c_float(self.exp_mean),
            self.infected_ids_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        )

    def test_zero_infections(self):
        """ Test with zero forces of infection for all nodes """
        total_forces = np.zeros(5)  # 5 nodes with zero FOI
        susceptibles = np.array([100, 200, 150, 300, 50])
        
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Expect no new infections
        expected = np.zeros(5, dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def test_full_infections(self):
        """ Test with maximum force of infection (1.0) for all nodes """
        total_forces = np.ones(5)  # 5 nodes with FOI = 1.0 (maximum)
        susceptibles = np.array([100, 200, 150, 300, 50])
        
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Expect all susceptibles to get infected
        expected = susceptibles  # since FOI is 1, all susceptibles should be infected
        np.testing.assert_array_equal(new_infections, expected)

    def test_mixed_infections(self):
        """ Test with a mixed range of FOI values """
        total_forces = np.array([0.2, 0.5, 0.8, 0.1, 0.9])
        susceptibles = np.array([100, 200, 150, 300, 50])
        
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Check if result is within valid range for each node
        for i in range(len(susceptibles)):
            self.assertGreaterEqual(new_infections[i], 0)
            self.assertLessEqual(new_infections[i], susceptibles[i])

    def test_capped_forces(self):
        """ Test that FOI values above 1.0 are properly capped at 1.0 """
        total_forces = np.array([1.2, 1.5, 0.9, 2.0, 0.5])  # 3 values > 1.0
        susceptibles = np.array([100, 200, 150, 300, 50])
        
        # Mock np.random.binomial to return fixed values for testing
        with patch('numpy.random.binomial', return_value=np.array([100, 200, 139, 300, 25])):
            new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Expected values when the binomial is mocked
        expected = np.array([100, 200, 139, 300, 25], dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def test_edge_case_no_susceptibles(self):
        """ Test with no susceptibles (i.e., susceptibles array full of zeros) """
        total_forces = np.array([0.2, 0.5, 0.8, 0.1, 0.9])
        susceptibles = np.zeros(5, dtype=np.uint32)  # No one is susceptible
        
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Expect no new infections
        expected = np.zeros(5, dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def test_edge_case_empty_arrays(self):
        """ Test with empty arrays for both total_forces and susceptibles """
        total_forces = np.array([], dtype=np.float64)
        susceptibles = np.array([], dtype=np.uint32)
        
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Expect empty result
        expected = np.array([], dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def test_incorrect_data_types(self):
        """ Test with incorrect data types to ensure proper handling """
        total_forces = [0.2, 0.5, 0.8, 0.1, 0.9]  # List instead of array
        susceptibles = [100, 200, 150, 300, 50]  # List instead of array
        
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)
        
        # Expect that it still works since np.array will handle type conversion
        for i in range(len(susceptibles)):
            self.assertGreaterEqual(new_infections[i], 0)
            self.assertLessEqual(new_infections[i], susceptibles[i])

    def test_max_values(self):
        """ Test with the maximum values for np.uint32 susceptibles and capped forces """
        max_uint32 = np.iinfo(np.uint32).max  # Maximum value for np.uint32
        susceptibles = np.array([max_uint32, max_uint32, max_uint32], dtype=np.uint32)
        total_forces = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # Forces capped at 1.0

        # Expect full infections for all nodes since forces are capped at 1.0
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)

        expected = np.array([max_uint32, max_uint32, max_uint32], dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def test_min_values(self):
        """ Test with the minimum values for np.uint32 susceptibles """
        susceptibles = np.array([0, 0, 0], dtype=np.uint32)  # No susceptibles
        total_forces = np.array([0.5, 0.9, 1.0], dtype=np.float64)  # Arbitrary forces

        # Expect no infections because there are no susceptibles
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)

        expected = np.array([0, 0, 0], dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def test_small_float_forces(self):
        """ Test with very small forces of infection """
        susceptibles = np.array([100, 200, 300], dtype=np.uint32)
        total_forces = np.array([1e-10, 1e-8, 1e-6], dtype=np.float64)  # Small forces

        # Expect very few or no infections due to tiny forces
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)

        expected = np.array([0, 0, 0], dtype=np.uint32)  # Likely no infections
        np.testing.assert_array_equal(new_infections, expected)

    def test_mixed_data_types(self):
        """ Test with mixed float32 and float64 types for total_forces """
        susceptibles = np.array([100, 200, 300], dtype=np.uint32)
        total_forces = np.array([0.5, 1.0, 1e-5], dtype=np.float32)  # Forces in float32

        # Expect varied infections based on forces
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)

        expected = np.random.binomial(susceptibles, np.minimum(total_forces, 1.0)).astype(np.uint32)
        np.testing.assert_allclose(new_infections, expected, rtol=0.05, atol=1)

    def test_overflow_risk(self):
        """ Test that np.uint32 can handle values near the overflow boundary """
        max_uint32 = np.iinfo(np.uint32).max
        susceptibles = np.array([max_uint32, max_uint32 // 2, max_uint32 // 4], dtype=np.uint32)
        total_forces = np.array([0.5, 0.7, 0.9], dtype=np.float64)

        # Test infection numbers near overflow threshold
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)

        expected = np.random.binomial(susceptibles, np.minimum(total_forces, 1.0)).astype(np.uint32)
        np.testing.assert_allclose(new_infections, expected, rtol=0.05, atol=1)

    def test_large_susceptibles_tiny_forces(self):
        """ Test with large susceptibles and very small forces """
        susceptibles = np.array([1e6, 1e7, 1e8], dtype=np.uint32)
        total_forces = np.array([1e-12, 1e-15, 1e-20], dtype=np.float64)

        # Expect very few or no infections due to tiny forces
        new_infections = calculate_new_infections_by_node(total_forces, susceptibles)

        expected = np.array([0, 0, 0], dtype=np.uint32)
        np.testing.assert_array_equal(new_infections, expected)

    def skip_test_array_length_and_output(self):
        """ Test that tx_inner_nodes processes arrays of correct lengths and produces correct outputs """

        # Call the function
        self.call_tx_inner_nodes()

        # Verify that new_infections is populated with non-negative values
        self.assertTrue(np.all(self.new_infections >= 0), "new_infections array contains invalid negative values")

        # Verify the length of the infected_ids_buffer matches the number of new infections
        total_new_infections = np.sum(self.new_infections)
        self.assertEqual(len(self.infected_ids_buffer), self.population_size,
                         "infected_ids_buffer length mismatch, expected length matching population size.")

        # Verify that the infected_ids_buffer contains exactly total_new_infections entries that are non-zero
        non_zero_infections = np.count_nonzero(self.infected_ids_buffer)
        self.assertEqual(non_zero_infections, total_new_infections,
                         f"Expected {total_new_infections} infected IDs but found {non_zero_infections} non-zero entries.")

    def skip_test_array_length_and_output(self):
        """ Test that tx_inner_nodes processes arrays of correct lengths and produces outputs """

        # Call the function
        self.call_tx_inner_nodes()

        # Test that the new_infections array was populated with non-negative integers
        self.assertTrue(np.all(self.new_infections >= 0), "new_infections array contains invalid negative values")
        self.assertEqual(len(self.new_infections), self.num_nodes, "New infections array length mismatch")

        self.infected_ids_buffer = self.new_infections[:sum(self.new_infections)]
        # Test that infected_ids_buffer has non-zero values (assuming some infections)
        if len( self.infected_ids_buffer ) > 0:
            self.assertTrue(np.any(self.infected_ids_buffer != 0), "Infected IDs buffer has no infections, expected some non-zero values.")

    def skip_test_no_infections(self):
        """ Test that no infections occur when all forces of infection are zero """

        # Set susceptibility to 0 (no one is susceptible)
        self.susceptibility.fill(0)
        self.infected_ids_buffer = self.new_infections[:sum(self.new_infections)]

        # Call the function
        self.call_tx_inner_nodes()

        # Expect no infections in the new_infections array
        np.testing.assert_array_equal(self.new_infections, np.zeros(self.num_nodes, dtype=np.uint32),
                                      err_msg="Expected no infections but some occurred.")

        # Expect the infected_ids_buffer to be all zeros
        #np.testing.assert_array_equal(self.infected_ids_buffer, np.zeros(self.population_size, dtype=np.uint32),
        #                              err_msg="Expected no infected IDs but some occurred.")

    def skip_test_max_susceptibility(self):
        """ Test that infections occur when susceptibility is maximized """

        # Set susceptibility to 1 (everyone is susceptible)
        self.susceptibility.fill(1)

        # Set the expected new infections per node (manually set it based on your expected values)
        # This simulates the infections you expect `tx_inner_nodes` to generate
        expected_new_infections = np.array([5, 3, 2, 0, 4], dtype=np.uint32)  # Example values
        self.new_infections = expected_new_infections.copy()

        # Calculate the total number of new infections
        total_new_infections = np.sum(self.new_infections)

        # Set the infected_ids_buffer based on the number of new infections
        self.infected_ids_buffer = np.zeros(total_new_infections, dtype=np.uint32)

        pdb.set_trace()
        # Call the function
        self.call_tx_inner_nodes()

        # Expect that the new infections array matches the expected values
        np.testing.assert_array_equal(self.new_infections, expected_new_infections,
                                      "New infections array does not match the expected values.")

        # Ensure that some new infections have occurred
        self.assertGreater(total_new_infections, 0, "Expected some infections but none occurred.")

        # Populate the buffer with IDs for each new infection
        current_index = 0
        for node_id, infections in enumerate(self.new_infections):
            if infections > 0:
                self.infected_ids_buffer[current_index:current_index + infections] = node_id
                current_index += infections

        # Verify that infected_ids_buffer has been populated correctly
        non_zero_infections = np.count_nonzero(self.infected_ids_buffer)
        self.assertEqual(non_zero_infections, total_new_infections,
                         f"Expected {total_new_infections} non-zero infected IDs but found {non_zero_infections}.")

    def skip_test_edge_incubation_timers(self):
        """ Test how the function handles edge values for incubation timers """

        # Set etimers to max value
        self.etimers.fill(np.iinfo(np.uint8).max)
        self.infected_ids_buffer = self.new_infections[:sum(self.new_infections)]

        # Call the function
        self.call_tx_inner_nodes()

        # Check for expected behavior with extreme timers
        self.assertTrue(np.any(self.new_infections >= 0), "Expected some infections even with extreme incubation timers.")

    def skip_test_large_population(self):
        """ Stress test with a large population to check memory handling and correctness """

        # Simulate a very large population
        self.population_size = 10**6
        self.susceptibility = np.random.randint(0, 2, size=self.population_size, dtype=np.uint8)
        self.etimers = np.random.randint(0, 10, size=self.population_size, dtype=np.uint8)
        self.new_infections = np.zeros(self.num_nodes, dtype=np.uint32)
        self.infected_ids_buffer = np.zeros(self.population_size, dtype=np.uint32)

        # Call the function
        self.call_tx_inner_nodes()

        # Check that infections were processed without crashing or errors
        self.assertTrue(np.all(self.new_infections >= 0), "Function failed to process large population correctly.")

    def skip_test_empty_population(self):
        """ Test how the function handles an empty population """

        self.population_size = 0
        self.susceptibility = np.zeros(self.population_size, dtype=np.uint8)
        self.etimers = np.zeros(self.population_size, dtype=np.uint8)
        self.new_infections = np.zeros(self.num_nodes, dtype=np.uint32)
        self.infected_ids_buffer = np.zeros(self.population_size, dtype=np.uint32)

        # Call the function
        self.call_tx_inner_nodes()

        # Expect no infections and no infected IDs
        np.testing.assert_array_equal(self.new_infections, np.zeros(self.num_nodes, dtype=np.uint32))
        np.testing.assert_array_equal(self.infected_ids_buffer, np.zeros(self.population_size, dtype=np.uint32))

if __name__ == '__main__':
    unittest.main()

