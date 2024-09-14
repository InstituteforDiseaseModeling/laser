from pathlib import Path
import numpy as np

# ## Source Demographics Data
# 
# We have some census data for Nigeria (2015) which we will use to set the initial populations for our nodes. Since we are modeling northern Nigeria, as a first cut we will only choose administrative districts which start with "NORTH_". That gives us 419 nodes with a total population of ~96M.


# setup initial populations
from .nigeria import lgas

def run():
    print(Path.cwd())
    admin2 = {k:v for k,v in lgas.items() if len(k.split(":")) == 5}
    print(f"{len(admin2)=}")

    nn_nodes = {k:v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}
    print(f"{len(nn_nodes)=}")

    initial_populations = np.array([v[0][0] for v in nn_nodes.values()])
    print(f"{len(initial_populations)=}")
    print(f"First 32 populations:\n{initial_populations[0:32]}")
    print(f"{initial_populations.sum()=:,}")

    return nn_nodes, initial_populations 

