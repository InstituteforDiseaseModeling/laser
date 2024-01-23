#pop = 1000000
pop = int(1e6)+1
num_nodes = 25
nodes = [ x for x in range(num_nodes) ]
duration = 3*365 # 1000
#base_infectivity = 0.00001
base_infectivity = 0.000001
#pop_file="pop_100k_5nodes_seeded.csv"
pop_file="../data/pop_1M_25nodes_seeded.csv"
#pop_file="pop_10M_250nodes_unseeded.csv"
#pop_file="pop_10M_250nodes_seeded.csv"
pop_file_out=f"pop_{int(pop/1e6)}M_{num_nodes}nodes_seeded.csv"
report_filename="simulation_output.csv"
