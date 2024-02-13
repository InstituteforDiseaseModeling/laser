# Pop & Nodes if building pop from params
pop = int(1e7)+1
num_nodes = 60 
nodes = [ x for x in range(num_nodes) ]

# Filenames if loading pop from file
#pop_file="pop_1M_25nodes_seeded.csv"
#pop_file="age_lt_5.csv"
pop_file="sorted_by_age.csv"
eula_file="age_gt_5.csv"
#pop_file_out=f"pop_{int(pop/1e6)}M_{num_nodes}nodes_seeded.csv"
pop_file_out=f"pop_{int(pop/1e6)}M_seeded.csv"

report_filename="simulation_output.csv"

# numerical runtime config params
duration = 20*365 # 900
base_infectivity = 2e6
cbr=14
expansion_slots=3e6
campaign_day=60
campaign_coverage=1.0
campaign_node=15
migration_interval=7
eula_age=5
mortality_interval=30
fertility_interval=7
ria_interval=7
