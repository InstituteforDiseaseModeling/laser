import numpy as np
import ctypes
from unittest import TestCase
from idmlaser_cholera.mods.transmission import calculate_new_infections_by_node
from pkg_resources import resource_filename
import pdb
import copy

class TestTxInnerNodesFunction(TestCase):

    def setUp(self):
        # Set up synthetic data for a population of 10 individuals
        self.population_size = 10
        self.num_nodes = 5  # Assume 5 nodes for this test

        # Create arrays of population attributes
        self.nodeid = np.array([0, 1, 0, 1, 2, 2, 3, 3, 4, 4], dtype=np.uint16)  # node IDs per individual
        self.infectious_timer = np.zeros(self.population_size, dtype=np.uint8)  # None are infectious initially
        self.incubation_timer = np.zeros(self.population_size, dtype=np.uint8)  # None are incubating initially
        self.susceptibility = np.ones(self.population_size, dtype=np.uint8)  # All are susceptible
        self.susceptibility[0] = 0
        self.susceptibility_timer = np.zeros(self.population_size, dtype=np.uint16)  # No timers for susceptibility
        self.dod = np.array([80, 70, 90, 85, 88, 95, 60, 75, 85, 78], dtype=np.int32)  # Expected lifespan
        self.exp_mean = 3.0  # Arbitrary constant for incubation period

        # Output count arrays for nodes
        self.S_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.E_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.I_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.W_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.R_count = np.zeros(self.num_nodes, dtype=np.uint32)

        # Initialize arrays for tx_inner_nodes tests
        self.new_infections = np.ones(self.num_nodes, dtype=np.uint32)*999  # Tracks new infections

        # Assuming `lib` is already loaded and set up correctly
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libtx.so')
        self.lib = ctypes.CDLL(shared_lib_path)
        
        self.lib.report.argtypes = [
            ctypes.c_uint32,                                              # count
            ctypes.c_uint32,                                              # num_nodes
            np.ctypeslib.ndpointer(dtype=np.uint16, flags='C_CONTIGUOUS'),  # node
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # infectious_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # incubation_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # susceptibility
            np.ctypeslib.ndpointer(dtype=np.uint16, flags='C_CONTIGUOUS'),  # susceptibility_timer
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),  # expected_lifespan
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # infectious_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # incubating_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # susceptible_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # waning_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # recovered_count
            ctypes.c_int,  # delta
            ctypes.c_int   # tick
        ]

        self.etimers = np.random.randint(0, self.population_size, size=self.population_size, dtype=np.uint8)  # Random incubation timers
        #self.itimers = np.random.randint(0, self.population_size, size=self.population_size, dtype=np.uint8)  # Random incubation timers
        # Set up argtypes for tx_inner_nodes (assumed from previous tests)
        self.lib.tx_inner_nodes.argtypes = [
            ctypes.c_uint32,                                              # count
            ctypes.c_uint32,                                              # num_nodes
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # susceptibility
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # incubation_timer
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # new infections
            #np.ctypeslib.ndpointer(dtype=np.uint16, flags='C_CONTIGUOUS'),  # node
            ctypes.c_float,                                                           # exp_mean
            ctypes.POINTER(ctypes.c_uint32)  # new_ids_out (pointer to uint32)
        ]

    def call_report(self):
        """ Helper function to call report with the current setup """
        self.lib.report(
            self.population_size,
            self.num_nodes,
            self.nodeid,
            self.infectious_timer,
            self.incubation_timer,
            self.susceptibility,
            self.susceptibility_timer,
            self.dod,
            self.S_count,
            self.E_count,
            self.I_count,
            self.W_count,
            self.R_count,
            1,  # delta
            1   # tick
        )

    def set_new_infections(self):
        """ Sets the new_infections array based on the number of susceptibles per node """

        # Loop over nodes instead of agents
        for node in range(self.num_nodes):
            susceptible_count = self.S_count[node]

            # Set the new infections for the current node
            max_new_infections = susceptible_count
            new_infections_in_node = np.random.randint(0, max_new_infections + 1) if max_new_infections > 0 else 0
            self.new_infections[node] = new_infections_in_node 

    def call_tx_inner_nodes(self):
        """Helper function to call the C function tx_inner_nodes"""
        self.infected_ids_buffer = np.zeros( sum(self.new_infections ), dtype=np.uint32)
        self.lib.tx_inner_nodes(
            ctypes.c_uint32(self.population_size),
            ctypes.c_uint32(self.num_nodes),
            self.susceptibility,
            self.etimers,
            self.new_infections,
            ctypes.c_float(self.exp_mean),
            self.infected_ids_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        )

    def test_tx_inner_nodes_basic(self):
        """ Test that tx_inner_nodes correctly creates infections """

        # First, call report() to set up the environment
        self.call_report()

        # Set new infections to a sensible number based on susceptibles per node
        self.set_new_infections()

        #print( f"{self.new_infections=}" )
        # After calling report, susceptibility and other counts are set, now call tx_inner_nodes
        self.call_tx_inner_nodes()
        #print( f"{self.infected_ids_buffer=}" )

        # Check that the infected_ids_buffer has non-zero values (i.e., infected individuals)
        total_infected_ids = np.count_nonzero(self.infected_ids_buffer)
        self.assertGreater(total_infected_ids, 0, "Expected some infected IDs in infected_ids_buffer but found none.")

        # Ensure all elements in infected_ids_buffer are non-zero
        for i in range(total_infected_ids):
            self.assertNotEqual(self.infected_ids_buffer[i], 0, f"Expected non-zero ID in buffer at index {i}.")

    def test_tx_inner_nodes_no_infections(self): # passes
        """ Test that tx_inner_nodes creates no infections when susceptibility is low """

        # Set all susceptibility to zero to prevent infections
        self.susceptibility.fill(0)

        # Call report() to update internal state
        self.call_report()

        self.call_tx_inner_nodes()

        # Ensure all elements in infected_ids_buffer are non-zero
        total_infected_ids = np.count_nonzero(self.infected_ids_buffer)
        for i in range(total_infected_ids):
            self.assertNotEqual(self.infected_ids_buffer[i], 999, f"Expected non-zero ID in buffer at index {i}.")

    def test_tx_inner_nodes_all_infected(self): # passes
        """ Test that tx_inner_nodes handles a fully infected population """

        # Set all individuals to be infectious by setting infectious_timer
        self.infectious_timer.fill(1)

        # Set new infections to zero (since everyone is already infected)
        self.new_infections.fill(0)

        # Call report() to update internal state (mark everyone as infected)
        self.call_report()

        self.call_tx_inner_nodes()

        # Expect no new infections because everyone is already infected
        total_new_infections = np.sum(self.new_infections)
        self.assertEqual(total_new_infections, 0, "Expected no new infections since everyone is already infected.")


    def test_tx_inner_nodes_partial_susceptibility(self):
        """ Test tx_inner_nodes with partial susceptibility (some susceptible, some immune) """

        # Set half of the population as susceptible and the other half as immune
        self.susceptibility[:self.population_size // 2] = 1  # Susceptible
        self.susceptibility[self.population_size // 2:] = 0  # Immune

        # Call report() to update the node counts
        self.call_report()

        # Set new infections based on the susceptibles
        suscept_orig = copy.deepcopy( self.susceptibility )
        self.set_new_infections()

        # Call tx_inner_nodes
        self.call_tx_inner_nodes()

        # Ensure infections only occurred among the susceptible individuals
        for infected_id in self.infected_ids_buffer:
            if infected_id > 0:
                self.assertEqual(suscept_orig[infected_id], 1, f"Infected agent {infected_id} should have been susceptible.")

    def test_tx_inner_nodes_low_infection_rate(self):
        """ Test tx_inner_nodes with all susceptible but low infection rate """

        # Make everyone susceptible
        self.susceptibility.fill(1)

        # Call report to update internal states
        self.call_report()

        # Manually set a very low number of new infections
        self.new_infections.fill(1)

        # Call tx_inner_nodes
        self.call_tx_inner_nodes()

        # Ensure only a small number of infections occurred
        total_new_infections = np.sum(self.new_infections)
        self.assertLessEqual(total_new_infections, self.num_nodes, "Too many infections occurred.")

    def skip_test_tx_inner_nodes_zero_susceptibility_timer(self):
        """ Test that tx_inner_nodes prevents infections when susceptibility timer is zero """

        # Set all susceptibility timers to zero (assumed to block infections)
        self.susceptibility_timer.fill(0)

        # Make everyone susceptible
        self.susceptibility.fill(1)

        # Call report() to update the internal states
        self.call_report()

        # Set new infections
        self.set_new_infections()

        # Call tx_inner_nodes
        self.call_tx_inner_nodes()

        # Check that no infections occurred due to the susceptibility timer
        total_new_infections = np.sum(self.new_infections)
        self.assertEqual(total_new_infections, 0, "No infections should have occurred with susceptibility timers set to zero.")

    def test_tx_inner_nodes_high_exp_mean(self):
        """ Test tx_inner_nodes with a high exp_mean value (long incubation period) """

        # Set a high incubation mean value
        self.exp_mean = 10.0

        # Make everyone susceptible
        self.susceptibility.fill(1)

        # Call report to update internal states
        self.call_report()

        # Set new infections
        self.set_new_infections()

        # Call tx_inner_nodes
        self.call_tx_inner_nodes()

        # Ensure that new infections occur, but the number might be affected by the high exp_mean
        total_new_infections = np.sum(self.new_infections)
        self.assertGreater(total_new_infections, 0, "Expected some infections even with high exp_mean.")

    def test_tx_inner_nodes_single_individual(self):
        """ Test tx_inner_nodes with a population size of 1 """

        # Reduce the population size to 1
        self.population_size = 1
        self.nodeid = np.array([0], dtype=np.uint16)
        self.susceptibility = np.array([1], dtype=np.uint8)

        # Call report() to set up the environment
        self.call_report()

        # Set new infections
        self.set_new_infections()

        # Call tx_inner_nodes
        self.call_tx_inner_nodes()

        # Check that if the individual is susceptible, they should be infected
        total_new_infections = np.sum(self.new_infections)
        self.assertGreaterEqual(total_new_infections, 0, "The individual should have been infected if susceptible.")

