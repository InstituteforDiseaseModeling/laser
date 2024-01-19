pop = 1_000_000
num_nodes = 25
nodes = [x for x in range(num_nodes)]
duration = 365
base_infectivity = 0.000_001
pop_file = f"pop_{int(pop/1e6)}M_{num_nodes}nodes_seeded.csv"
pop_file_out = f"pop_{int(pop/1e6)}M_{num_nodes}nodes_seeded.csv"
report_filename = "simulation_output.csv"
