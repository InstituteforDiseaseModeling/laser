import ctypes
import numpy as np
import unittest
import memory_profiler
import pdb

beta = 2.4  # Transmission rate
# Load the C library
lib = ctypes.CDLL("./update_ages.so")
lib.calculate_new_infections.argtypes = [
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # starting index
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # nodes
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # inf_counts
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # sus_counts
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # tot_counts
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # new_infections
    ctypes.c_float, # base_inf
]
lib.handle_new_infections.argtypes = [
    ctypes.c_uint32, # num_agents
    ctypes.c_uint32, # node
    ctypes.c_size_t,  # starting index
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # nodes
    np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # infection_timer
    ctypes.c_int, # num_new_infections
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new infected ids
    ctypes.c_int, # sus number for node
]
lib.handle_new_infections_threaded.argtypes = [
    ctypes.c_uint32, # num_agents
    ctypes.c_size_t,  # starting index
    ctypes.c_size_t,  # num_nodes
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # nodes
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # infection_timer
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. new infections to create by node
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new infected ids
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. susceptibles by node
]
lib.handle_new_infections_mp.argtypes = [
    ctypes.c_uint32, # num_agents
    ctypes.c_size_t,  # starting index
    ctypes.c_size_t,  # num_nodes
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # nodes
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # infection_timer
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. new infections to create by node
    #np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new infected ids
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. susceptibles by node
]

def run_ref_model():
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    from collections import deque

    # Parameters
    population = 1.2e5 # 6  # Total population
    cbr = 17.5  # Crude birth rate per 1000 people per year

    incubation_period = 7  # Incubation period in days
    infectious_period = 7  # Infectious period in days
    simulation_days = 7300  # Number of days to simulate

    # Calculate births per day
    births_per_day = int((population / 1000) * (cbr / 365))

    # Initial values
    E_queue = deque([393] + [0] * (incubation_period-1))  # Exposed individuals queue
    I_queue = deque([0] * infectious_period)  # Infectious individuals queue (seeded outbreak)
    S = [int(population/(beta*infectious_period))]
    R = [int(population - S[0] - sum(E_queue) - sum(I_queue))]

    # Make reporting lists for E and I since we're modeling with queues
    E_reporting = [sum(E_queue )]
    I_reporting = [sum(I_queue )]
    NI = [0]

    # Simulation
    for _ in range(1, simulation_days):
        # progress infections: E->I
        new_infections = E_queue.pop()
        # Push new infections into the infectious queue
        I_queue.appendleft(new_infections)
        # Update recovered population: I->R
        R.append(R[-1] + I_queue.pop())

        # Calculate new exposures 
        new_exposures = int(np.round(beta * sum(I_queue) * S[-1] / population)) 

        # Calculate new infections (from the exposed population)

        # Push new exposures into the exposed queue
        E_queue.appendleft(new_exposures)

        # Update susceptible population, inc VD.
        S.append(S[-1] - new_exposures + births_per_day)

        population += births_per_day

        E_reporting.append( sum(E_queue ) )
        I_reporting.append( sum(I_queue ) )
        NI.append(new_exposures)

    return S, E_reporting, I_reporting, R, NI

class TestHandleNewInfections(unittest.TestCase):
    def test_no_eligible_agents(self):
        """
        Test case to verify behavior when no agents are eligible for infection.

        Test Design:
        1. Set up input parameters where no agents are eligible for infection.
        2. Call the C function with the provided input.
        3. Verify that no agents are infected after the function call.

        Test Failure Implications:
        - If the test fails, it indicates that the C function is incorrectly infecting agents when there are none eligible.
        """

        # Define input parameters
        start_idx = 0
        end_idx = 10
        node = 1
        agent_node = np.array([1] * 10, dtype=np.uint32)
        infected = np.array([False] * 10, dtype=bool)
        immunity = np.array([False] * 10, dtype=bool)
        incubation_timer = np.array([0] * 10, dtype=np.uint8)
        infection_timer = np.array([0] * 10, dtype=np.uint8)
        new_infections = 5
        new_infection_idxs_out = np.zeros(new_infections).astype(np.uint32)
        num_eligible_agents = 0

        # Call the C function
        lib.handle_new_infections(
            start_idx,
            end_idx,
            node,
            agent_node,
            infected,
            immunity,
            incubation_timer,
            infection_timer,
            new_infections,
            new_infection_idxs_out,
            num_eligible_agents
        )

        # Assert that no new infections occurred
        self.assertEqual(np.sum(infected), 0)

    def test_some_eligible_agents(self):
        """
        Test case to verify behavior when some agents are eligible for infection.

        Test Design:
        1. Set up input parameters where some agents are eligible for infection.
        2. Call the C function with the provided input.
        3. Verify that the expected number of agents are infected after the function call.

        Test Failure Implications:
        - If the test fails, it indicates that the C function is not infecting the correct number of agents or is infecting agents incorrectly.
        """
        # Define input parameters
        start_idx = 0
        end_idx = 10
        node = 1
        agent_node = np.array([1] * 10, dtype=np.uint32)
        infected = np.array([False] * 10, dtype=bool)
        immunity = np.array([False] * 10, dtype=bool)
        incubation_timer = np.array([0] * 10, dtype=np.uint8)
        infection_timer = np.array([0] * 10, dtype=np.uint8)
        new_infections = 5
        new_infection_idxs_out = np.zeros(new_infections, dtype=np.uint32)
        num_eligible_agents = 5

        # Call the C function
        lib.handle_new_infections(
            start_idx,
            end_idx,
            node,
            agent_node,
            infected,
            immunity,
            incubation_timer,
            infection_timer,
            new_infections,
            new_infection_idxs_out,
            num_eligible_agents
        )

        # Assert that new infections occurred
        self.assertEqual(np.sum(infected), new_infections)

    # Add more test cases for other scenarios (e.g., boundary cases, error cases, etc.)
    def test_invalid_input_num_eligible_agents(self):
        """
        Test case to verify behavior when the number of eligible agents is negative.

        Test Design:
        1. Set up input parameters where the number of eligible agents is negative.
        2. Call the C function with the provided input.
        3. Verify that the function raises an exception or returns an error code indicating invalid input.

        Test Failure Implications:
        - If the test fails, it indicates that the function does not handle negative numbers of eligible agents correctly.
        """

        # Define input parameters with negative number of eligible agents
        start_idx = 0
        end_idx = 10
        node = 1
        agent_node = np.array([1] * 10, dtype=np.uint32)
        infected = np.array([False] * 10, dtype=bool)
        immunity = np.array([False] * 10, dtype=bool)
        incubation_timer = np.array([0] * 10, dtype=np.uint8)
        infection_timer = np.array([0] * 10, dtype=np.uint8)
        new_infections = 5
        new_infection_idxs_out = np.zeros(new_infections, dtype=np.uint32)
        num_eligible_agents = -5  # Negative number of eligible agents

        # Save a copy of new_infection_idxs_out before calling the function
        original_new_infection_idxs_out = new_infection_idxs_out.copy()

        # Call the C function and assert that it raises an exception or returns an error code
        lib.handle_new_infections(
            start_idx,
            end_idx,
            node,
            agent_node,
            infected,
            immunity,
            incubation_timer,
            infection_timer,
            new_infections,
            new_infection_idxs_out,
            num_eligible_agents
        )

        # Assert that new_infection_idxs_out remains unmodified
        np.testing.assert_array_equal(new_infection_idxs_out, original_new_infection_idxs_out)

    def test_invalid_input_end_idx_less_than_start_idx(self):
        """
        Test case to verify behavior when end_idx is less than start_idx.

        Test Design:
        1. Set up input parameters where end_idx is less than start_idx.
        2. Call the C function with the provided input.
        3. Verify that the function raises an exception or returns an error code indicating invalid input.
        4. Confirm that new_infection_idxs_out remains unmodified.

        Test Failure Implications:
        - If the test fails, it indicates that the function does not handle the case where end_idx is less than start_idx correctly or modifies new_infection_idxs_out incorrectly.
        """

        # Define input parameters with end_idx less than start_idx
        start_idx = 10
        end_idx = 5
        node = 1
        agent_node = np.array([1] * 10, dtype=np.uint32)
        infected = np.array([False] * 10, dtype=bool)
        immunity = np.array([False] * 10, dtype=bool)
        incubation_timer = np.array([0] * 10, dtype=np.uint8)
        infection_timer = np.array([0] * 10, dtype=np.uint8)
        new_infections = 5
        new_infection_idxs_out = np.arange(5, dtype=np.uint32)  # Array with values [0, 1, 2, 3, 4]
        num_eligible_agents = 5

        # Save a copy of new_infection_idxs_out before calling the function
        original_new_infection_idxs_out = new_infection_idxs_out.copy()

        # Call the C function
        lib.handle_new_infections(
            start_idx,
            end_idx,
            node,
            agent_node,
            infected,
            immunity,
            incubation_timer,
            infection_timer,
            new_infections,
            new_infection_idxs_out,
            num_eligible_agents
        )

        # Assert that new_infection_idxs_out remains unmodified
        np.testing.assert_array_equal(new_infection_idxs_out, original_new_infection_idxs_out)

    def test_consistency_with_same_inputs(self):
        """
        Test case to verify consistency of function output with the same valid inputs.

        Test Design:
        1. Set up valid input parameters.
        2. Call the C function multiple times with the same inputs.
        3. Verify that the output remains consistent across multiple function calls.

        Test Failure Implications:
        - If the test fails, it indicates that the function produces inconsistent output for the same inputs, suggesting a potential issue with the function's implementation or state management.
        """

        # Define valid input parameters
        start_idx = 0
        end_idx = 10
        node = 1
        agent_node = np.array([1] * 10, dtype=np.uint32)
        #infected = np.array([False] * 10, dtype=bool)
        immunity = np.array([False] * 10, dtype=bool)
        #incubation_timer = np.array([0] * 10, dtype=np.uint8)
        #infection_timer = np.array([0] * 10, dtype=np.uint8)
        #new_infections = 5
        #new_infection_idxs_out = np.zeros(new_infections, dtype=np.uint32)
        num_eligible_agents = 5

        # Call the C function multiple times and store the outputs
        outputs = []
        num_repeats = 5
        for _ in range(num_repeats):  # Perform 5 function calls
            # Reset arrays modified by the function
            infected = np.array([False] * 10, dtype=bool)
            incubation_timer = np.array([0] * 10, dtype=np.uint8)
            infection_timer = np.array([0] * 10, dtype=np.uint8)
            new_infections = 5
            new_infection_idxs_out = np.zeros(new_infections, dtype=np.uint32)
            lib.handle_new_infections(
                start_idx,
                end_idx,
                node,
                agent_node,
                infected,
                immunity,
                incubation_timer,
                infection_timer,
                new_infections,
                new_infection_idxs_out,
                num_eligible_agents
            )
            outputs.append(new_infection_idxs_out.copy())  # Save a copy of the output

        # Verify that all outputs are the same
        for i in range(1, len(outputs)):
            np.testing.assert_array_equal(outputs[i], outputs[0])


    def test_memory_leak(self):
        """
        Test case to check for memory leaks in the function.

        Test Design:
        1. Profile memory usage before and after calling the function.
        2. Check if there is any increase in memory usage after the function call.
        3. Repeat the test multiple times to ensure consistency.

        Test Failure Implications:
        - If the test fails, it indicates that the function may have memory leaks,
          leading to potential issues with memory management.
        """

        # Define input parameters
        start_idx = 0
        end_idx = 10
        node = 1
        agent_node = np.array([1] * 10, dtype=np.uint32)
        infected = np.array([False] * 10, dtype=bool)
        immunity = np.array([False] * 10, dtype=bool)
        incubation_timer = np.array([0] * 10, dtype=np.uint8)
        infection_timer = np.array([0] * 10, dtype=np.uint8)
        new_infections = 5
        new_infection_idxs_out = np.zeros(new_infections, dtype=np.uint32)
        num_eligible_agents = 5

        # Profile memory usage before function call
        pre_memory = memory_profiler.memory_usage()[0]

        # Call the function
        lib.handle_new_infections(
            start_idx,
            end_idx,
            node,
            agent_node,
            infected,
            immunity,
            incubation_timer,
            infection_timer,
            new_infections,
            new_infection_idxs_out,
            num_eligible_agents
        )

        # Profile memory usage after function call
        post_memory = memory_profiler.memory_usage()[0]

        # Check for memory leak
        memory_leak = post_memory - pre_memory
        self.assertLessEqual(memory_leak, 0, "Memory leak detected")

    def test_calculate_new_infections(self):
        """
        Test case to confirm that calculate_new_infections function returns same numbers of new
        infections across entire simulation as a reference model.
        """
        # Define parameters
        n = 1
        starting_index = 0  # Example starting index
        base_inf = beta

        # Call run_ref_model to get reference results
        S, E, I, R, NI = run_ref_model()
        
        sim_length = len(S)

        # Create arrays for inputs and outputs
        nodes = np.zeros(n, dtype=np.uint32)
        inf_counts = np.zeros(n, dtype=np.uint32)
        sus_counts = np.zeros(n, dtype=np.uint32)
        tot_counts = np.zeros(n, dtype=np.uint32)
        new_infections = np.zeros(n, dtype=np.uint32)

        # Call calculate_new_infections for each timestep
        for timestep in range(1,sim_length):
            #print( f"{timestep}: {S[timestep]}, {E[timestep]}, {I[timestep]}, {R[timestep]}" )
            sus_counts[0] = S[timestep-1]

            # Update the exposed and infectious counts
            inf_counts[0] = I[timestep]

            num_si = E[0]
            incubation_timer = np.zeros(num_si, dtype=np.uint8)
            tot_counts[0] = S[timestep-1] + E[timestep-1] + I[timestep-1] + R[timestep-1]

            # Call the ctypes function
            lib.calculate_new_infections(
                    0, 
                    num_si-1,
                    n,
                    nodes,
                    incubation_timer,
                    inf_counts,
                    sus_counts,
                    tot_counts,
                    new_infections,
                    base_inf)

            # Get the number of new infections from new_infections array
            new_infections_timestep = np.sum(new_infections)

            # Compare with reference results
            #self.assert( new_infections_timestep == ref_results[timestep], f"Results mismatch at timestep {timestep}" )
            #print( f"New_Infections [test] @ {timestep} = {new_infections_timestep}\n" )
            #print( f"New_Infections [ref][ @ {timestep} = {NI[timestep]}\n" )
            self.assertAlmostEqual( new_infections_timestep, NI[timestep], delta=10 ) # , "Results mismatch at timestep " + str(timestep)

        # Call run_ref_model to get reference results
        S, E, I, R, NI = run_ref_model()
        
        sim_length = len(S)

        # Create arrays for inputs and outputs
        nodes = np.zeros(n, dtype=np.uint32)
        inf_counts = np.zeros(n, dtype=np.uint32)
        sus_counts = np.zeros(n, dtype=np.uint32)
        tot_counts = np.zeros(n, dtype=np.uint32)
        new_infections = np.zeros(n, dtype=np.uint32)

        # Call calculate_new_infections for each timestep
        for timestep in range(1,sim_length):
            #print( f"{timestep}: {S[timestep]}, {E[timestep]}, {I[timestep]}, {R[timestep]}" )
            sus_counts[0] = S[timestep-1]

            # Update the exposed and infectious counts
            inf_counts[0] = I[timestep]

            num_si = E[0]
            incubation_timer = np.zeros(num_si, dtype=np.uint8)
            tot_counts[0] = S[timestep-1] + E[timestep-1] + I[timestep-1] + R[timestep-1]

            # Call the ctypes function
            lib.calculate_new_infections(
                    0, 
                    num_si-1,
                    n,
                    nodes,
                    incubation_timer,
                    inf_counts,
                    sus_counts,
                    tot_counts,
                    new_infections,
                    base_inf)

            # Get the number of new infections from new_infections array
            new_infections_timestep = np.sum(new_infections)

            # Compare with reference results
            #self.assert( new_infections_timestep == ref_results[timestep], f"Results mismatch at timestep {timestep}" )
            #print( f"New_Infections [test] @ {timestep} = {new_infections_timestep}\n" )
            #print( f"New_Infections [ref][ @ {timestep} = {NI[timestep]}\n" )
            self.assertAlmostEqual( new_infections_timestep, NI[timestep], delta=10 ) # , "Results mismatch at timestep " + str(timestep)


class TestHandleNewInfectionsThreaded(unittest.TestCase):
    def setUp(self):
        # Setup common data for tests
        self.num_nodes = 10
        self.num_agents = 102

        self.agent_node = np.array([i % self.num_nodes for i in range(self.num_agents)], dtype=np.uint32)
        self.infected = np.zeros(self.num_agents, dtype=np.bool_)
        self.immunity = np.zeros(self.num_agents, dtype=np.bool_)
        self.incubation_timer = np.zeros(self.num_agents, dtype=np.uint8)
        self.infection_timer = np.zeros(self.num_agents, dtype=np.uint8)
        self.new_infections = np.zeros(self.num_nodes, dtype=np.uint32)
        self.new_infection_idxs_out = np.zeros(100, dtype=np.uint32)
        self.num_eligible_agents = np.zeros(self.num_nodes, dtype=np.uint32)

        # Randomly infect some agents and set immunity and infection_timer
        np.random.seed(0)
        infected_indices = np.random.choice(self.num_agents, size=self.num_agents // 10, replace=False)
        for idx in infected_indices:
            self.infected[idx] = True
            self.infection_timer[idx] = np.random.randint(1, 10)

        # Set immunity for some agents
        immune_indices = np.random.choice([i for i in range(self.num_agents) if i not in infected_indices], size=self.num_agents // 10, replace=False)
        for idx in immune_indices:
            self.immunity[idx] = True

        # Calculate num_eligible_agents
        for node in range(self.num_nodes):
            agents_in_node = [i for i in range(1,101) if self.agent_node[i] == node]
            self.num_eligible_agents[node] = len(agents_in_node) - np.sum(self.immunity[agents_in_node]) - np.sum(self.infected[agents_in_node])

        # Set new_infections to a fraction of num_eligible_agents for each node
        for node in range(self.num_nodes):
            if self.num_eligible_agents[node] > 0:
                fraction = np.random.rand()  # Random fraction between 0 and 1
                self.new_infections[node] = int(round(fraction * self.num_eligible_agents[node]))

    def skip_test_no_infections(self):
        start_idx = 0
        end_idx = self.num_nodes - 1
        self.new_infections[:] = 0
        
        lib.handle_new_infections_threaded(
            start_idx, end_idx, self.num_nodes,
            self.agent_node,
            self.infected,
            self.immunity,
            self.incubation_timer,
            self.infection_timer,
            self.new_infections,
            self.new_infection_idxs_out,
            self.num_eligible_agents
        )

        # Check the expected results
        np.testing.assert_array_equal(self.new_infection_idxs_out, np.zeros_like(self.new_infection_idxs_out))
        np.testing.assert_array_equal(self.num_eligible_agents, np.zeros_like(self.num_eligible_agents))

    def skip_test_some_infections(self):
        start_idx = 0
        end_idx = self.num_nodes - 1
        self.new_infections[:] = [1 if i % 2 == 0 else 0 for i in range(self.num_nodes)]  # Infections in even indices
        self.new_infection_idxs_out = np.zeros(np.sum(self.new_infections), dtype=np.int32)

        lib.handle_new_infections_threaded(
            start_idx, end_idx, self.num_nodes,
            self.agent_node.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self.infected.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.immunity.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.incubation_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self.infection_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self.new_infections.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.new_infection_idxs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.num_eligible_agents.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        expected_new_infection_idxs_out = np.array([i for i in range(self.num_nodes) if i % 2 == 0], dtype=np.int32)
        expected_num_eligible_agents = np.array([1 if i % 2 == 0 else 0 for i in range(self.num_nodes)], dtype=np.int32)
        
        np.testing.assert_array_equal(self.new_infection_idxs_out, expected_new_infection_idxs_out)
        np.testing.assert_array_equal(self.num_eligible_agents, expected_num_eligible_agents)

    def skip_test_all_infections(self):
        start_idx = 1
        end_idx = 100
        self.new_infections[:] = 1
        self.new_infection_idxs_out = np.zeros(np.sum(self.new_infections), dtype=np.int32)

        lib.handle_new_infections_threaded(
            start_idx, end_idx, self.num_nodes,
            self.agent_node.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self.infected.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.immunity.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.incubation_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self.infection_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self.new_infections.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.new_infection_idxs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.num_eligible_agents.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        expected_new_infection_idxs_out = np.array([i for i in range(self.num_nodes)], dtype=np.int32)
        expected_num_eligible_agents = np.array([1] * self.num_nodes, dtype=np.int32)
        
        np.testing.assert_array_equal(self.new_infection_idxs_out, expected_new_infection_idxs_out)
        np.testing.assert_array_equal(self.num_eligible_agents, expected_num_eligible_agents)
   
    def test_infections(self):
        start_idx = 1
        end_idx = 100
        #self.new_infection_idxs_out = np.zeros(self.num_agents, dtype=np.uint32)

        num_originally_infected = np.count_nonzero(self.infected)
        #lib.handle_new_infections_threaded(
        lib.handle_new_infections_mp(
            start_idx, end_idx, self.num_nodes,
            self.agent_node,
            self.infected,
            self.immunity,
            self.incubation_timer,
            self.infection_timer,
            self.new_infections,
            #self.new_infection_idxs_out,
            self.num_eligible_agents
        )

        # Check the number of non-zero indices in new_infection_idxs_out
        non_zero_count = np.count_nonzero(self.infected)-num_originally_infected 
        #print( f"{self.new_infection_idxs_out}" )
        expected_non_zero_count = np.sum(self.new_infections)

        self.assertEqual(non_zero_count, expected_non_zero_count,
                         f'Expected {expected_non_zero_count} non-zero indexes, got {non_zero_count}.')


# TBD: ccs:
# 1) beta=1, cbr=30, threshold pop=200k
# 2) beta=2, cbr=17, trehshold pop=390k
if __name__ == "__main__":
    unittest.main()


