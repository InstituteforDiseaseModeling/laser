import numpy as np
import ctypes
from unittest import TestCase
from idmlaser_cholera.mods.transmission import calculate_new_infections_by_node
from pkg_resources import resource_filename

class TestReportFunction(TestCase):

    def setUp(self):
        # Set up synthetic data for a population of 10 individuals
        self.population_size = 10
        self.num_nodes = 5  # Assume 5 nodes for this test

        # Create arrays of population attributes
        self.nodeid = np.array([0, 1, 0, 1, 2, 2, 3, 3, 4, 4], dtype=np.uint16)  # node IDs per individual
        self.infectious_timer = np.zeros(self.population_size, dtype=np.uint8)  # None are infectious
        self.incubation_timer = np.zeros(self.population_size, dtype=np.uint8)  # None are incubating
        self.susceptibility = np.ones(self.population_size, dtype=np.uint8)  # All are susceptible
        self.susceptibility_timer = np.zeros(self.population_size, dtype=np.uint16)  # No timers for susceptibility
        self.dod = np.array([80, 70, 90, 85, 88, 95, 60, 75, 85, 78], dtype=np.int32)  # Expected lifespan

        # Output count arrays for nodes
        self.S_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.E_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.I_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.W_count = np.zeros(self.num_nodes, dtype=np.uint32)
        self.R_count = np.zeros(self.num_nodes, dtype=np.uint32)

        # Assuming `lib` is already loaded and set up correctly
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libtx.so')
        global lib
        lib = ctypes.CDLL(shared_lib_path)
        
        lib.report.argtypes = [
            ctypes.c_int64,  # count (population size)
            ctypes.c_int,  # num_nodes
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

    def test_report_basic(self):
        """Test that the report function counts susceptibles correctly in a simple scenario"""

        # Call the report function
        lib.report(
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

        # Check that all individuals are counted as susceptible
        expected_S_count = np.array([2, 2, 2, 2, 2], dtype=np.uint32)  # Expect 2 individuals per node
        np.testing.assert_array_equal(self.S_count, expected_S_count, "Susceptible count mismatch")

        # Check that there are no incubating, infectious, waning, or recovered individuals
        np.testing.assert_array_equal(self.E_count, np.zeros(self.num_nodes, dtype=np.uint32), "Incubating count mismatch")
        np.testing.assert_array_equal(self.I_count, np.zeros(self.num_nodes, dtype=np.uint32), "Infectious count mismatch")
        np.testing.assert_array_equal(self.W_count, np.zeros(self.num_nodes, dtype=np.uint32), "Waning count mismatch")
        np.testing.assert_array_equal(self.R_count, np.zeros(self.num_nodes, dtype=np.uint32), "Recovered count mismatch")

    def test_no_infections(self):
        """Test that no infections are counted when there are no infectious individuals"""

        # Call the report function (same setup as test_report_basic)
        lib.report(
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

        # Assert no infections and all individuals are susceptible
        self.assertEqual(np.sum(self.I_count), 0, "Expected no infectious individuals")
        self.assertGreater(np.sum(self.S_count), 0, "Expected some susceptible individuals")

    def test_all_infected(self):
        """Test that the report function correctly handles a fully infected population"""

        # Set all individuals to be infectious
        self.infectious_timer.fill(1)

        # Call the report function
        lib.report(
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

        # Expect all individuals to be counted as infectious
        expected_I_count = np.array([2, 2, 2, 2, 2], dtype=np.uint32)  # Expect 2 infectious per node
        np.testing.assert_array_equal(self.I_count, expected_I_count, "Infectious count mismatch")

        # Check that there are no susceptibles, incubating, waning, or recovered individuals
        np.testing.assert_array_equal(self.S_count, np.zeros(self.num_nodes, dtype=np.uint32), "Susceptible count mismatch")
        np.testing.assert_array_equal(self.E_count, np.zeros(self.num_nodes, dtype=np.uint32), "Incubating count mismatch")
        np.testing.assert_array_equal(self.W_count, np.zeros(self.num_nodes, dtype=np.uint32), "Waning count mismatch")
        np.testing.assert_array_equal(self.R_count, np.zeros(self.num_nodes, dtype=np.uint32), "Recovered count mismatch")


