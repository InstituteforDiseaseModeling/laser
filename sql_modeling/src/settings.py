# Pop & Nodes if building pop from params
pop = int(1e7)+1
num_nodes = 60
nodes = [ x for x in range(num_nodes) ]
# Epidemiologically Useless Light Agents
eula_age=5

# Filenames if loading pop from file
pop_file="modeled_pop.csv"
eula_file="eula_binned.csv"
#pop_file_out=f"pop_seeded.csv"
eula_pop_fits="fits.npy"

report_filename="simulation_output.csv"

# numerical runtime config params
# simulation duration
duration = 20*365
#base_infectivity = 2e7
base_infectivity = 1.5e7
# cbr=crude bith rate
cbr=15
# number of babies we expect to be born
expansion_slots=5e6
#expansion_slots=3e5
campaign_day=60
campaign_coverage=0.75
campaign_node=15
migration_interval=7
mortality_interval=1 # 30
fertility_interval=1 # 7
ria_interval=7
