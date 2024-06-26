# Pop & Nodes if building pop from params 
pop = int(1e7)+1
num_nodes = 954
nodes = [ x for x in range(num_nodes) ]
# Epidemiologically Useless Light Agents
eula_age=5

# Filenames if loading pop from file
pop_file="engwal_modeled.csv.gz"
#pop_file="engwal_modeled_modified.csv.gz"
eula_file="ew_eula_binned.csv"
eula_pop_fits="fits.npy"

cbr_file="cbrs_ew.csv"
