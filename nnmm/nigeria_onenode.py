#********************************************************************************
#
# Population data
#   (SIZE: int, YEAR: int)
# Latitude (Y) and longitude data (X)
#   (X_VAL: float, Y_VAL: float)
# Area data (km^2)
#   AREA: float
#
# "LOCATION": ((SIZE, YEAR), (X_VAL, Y_VAL), AREA)
#
#********************************************************************************

import numpy as np

def run():
    admin2 = {k:v for k,v in lgas.items()}
    print(f"{len(admin2)=}")

    nn_nodes = {k:v for k, v in admin2.items()}
    print(f"{len(nn_nodes)=}")

    initial_populations = np.array([v[0][0] for v in nn_nodes.values()])
    print(f"{len(initial_populations)=}")
    print(f"First 1 populations:\n{initial_populations[0:1]}")
    print(f"{initial_populations.sum()=:,}")

    cbrs = {index: details[2] for index, (details) in enumerate(admin2.values())}

    return nn_nodes, initial_populations, cbrs


lgas = {
    "AFRO:NIGERIA:ABUJA": ((1_000_000, 2015), (7.406129, 8.942755), 40.0),
}

