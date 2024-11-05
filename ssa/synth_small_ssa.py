from pathlib import Path
import numpy as np

from iso_codes_mosaic import sorted_by_pop as sbp

# Define number of synthetic nodes and ranges for population and birth rate
num_nodes = 41
population_range = (1e6, 1e7)  # Population between 1,000,000 and 10,000,000
birth_rate_range = (20, num_nodes)  # Birth rate between 20 and num_nodes

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data
synthetic_lgas = {}

# Define parameters
P1 = 23  # Population of the largest country (Nigeria) in millions, divided by 10
alpha = 0.9  # Power-law exponent
n_countries = num_nodes  # Number of countries to estimate

# Calculate populations
populations = [P1 / (n ** alpha) for n in range(1, n_countries + 1)]

# Display results
for rank, population in enumerate(populations, start=1):
    print(f"Country rank {rank}: Estimated population = {population:.2f} million")

for i in range(1, num_nodes + 1):
    # Randomly generate population and birth rate within specified ranges
    #pop = int(np.random.uniform(*population_range))
    pop = int(populations[i-1]*1e6)
    birth_rate = round(np.random.uniform(*birth_rate_range), 2)
    
    # Unique node name
    node_name = f"SYNTHETIC_NODE_{sbp[i-1]}"
    
    # Generate unique latitude and longitude based on node index
    latitude = 10.0000 + (i % 10) * 0.0001
    longitude = 20.0000 + (i // 10) * 0.0001
    year = 2024

    # Assign to synthetic_lgas dictionary
    synthetic_lgas[node_name] = ((pop, year), (latitude, longitude), birth_rate)

print(synthetic_lgas)

# Example function call for testing
def run():
    print(Path.cwd())
    admin2 = {k:v for k,v in synthetic_lgas.items()}
    print(f"{len(admin2)=}")

    nn_nodes = {k:v for k, v in admin2.items()}
    print(f"{len(nn_nodes)=}")

    initial_populations = np.array([v[0][0] for v in nn_nodes.values()])
    print(f"{len(initial_populations)=}")
    print(f"First 32 populations:\n{initial_populations[0:32]}")
    print(f"{initial_populations.sum()=:,}")

    cbrs = {index: details[2] for index, (details) in enumerate(admin2.values())}

    return nn_nodes, initial_populations, cbrs

#run() # at one point I liked the idea of doing this simply by importing

