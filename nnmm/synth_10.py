from pathlib import Path
import numpy as np

# ## Source Demographics Data
# 
# We have some census data for Nigeria (2015) which we will use to set the initial populations for our nodes. Since we are modeling northern Nigeria, as a first cut we will only choose administrative districts which start with "NORTH_". That gives us 419 nodes with a total population of ~96M.


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


synthetic_lgas = {
    "SYNTHETIC_NODE_1": ((1000000, 2024), (10.0001, 20.0001), 40),
    "SYNTHETIC_NODE_2": ((2000000, 2024), (10.0002, 20.0002), 40),
    "SYNTHETIC_NODE_3": ((3000000, 2024), (10.0003, 20.0003), 40),
    "SYNTHETIC_NODE_4": ((4000000, 2024), (10.0004, 20.0004), 40),
    "SYNTHETIC_NODE_5": ((5000000, 2024), (10.0005, 20.0005), 40),
    "SYNTHETIC_NODE_6": ((6000000, 2024), (10.0006, 20.0006), 40),
    "SYNTHETIC_NODE_7": ((7000000, 2024), (10.0007, 20.0007), 40),
    "SYNTHETIC_NODE_8": ((8000000, 2024), (10.0008, 20.0008), 40),
    "SYNTHETIC_NODE_9": ((9000000, 2024), (10.0009, 20.0009), 40),
    "SYNTHETIC_NODE_10": ((10000000, 2024), (10.0010, 20.0010), 40),
}

