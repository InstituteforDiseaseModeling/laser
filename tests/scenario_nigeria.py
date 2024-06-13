"""Load initial population, demographics, and network data for the Nigeria scenario."""

from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np

from idmlaser.numpynumba import DemographicsByYear

Node = namedtuple("Node", ["name", "index", "population", "counts", "births", "immigrations", "deaths"])


def initialize_nigeria(model, parameters, num_nodes: int = 0, level: int = 5) -> Tuple[int, DemographicsByYear, np.ndarray, np.ndarray]:
    """Initialize the model with the Nigeria scenario."""
    if Path.cwd().parts[-1] == "tests":
        from data.nigeria import gravity
        from data.nigeria import lgas
    else:   # guess that we are running from a notebook in the root of the project
        from tests.data.nigeria import gravity
        from tests.data.nigeria import lgas

    # Massage the data into the format expected by the model
    assert level == 5, f"Invalid level {level=} (We only have network data for level 5)"
    filtered = {k: v for k, v in lgas.items() if len(k.split(":")) == level}
    nodes = [
        Node(
            k,
            index,
            v[0][0],  # ((POP, YEAR), (LONG, LAT), AREA)
            {"I": 0 if v[0][0] < 100_000 else 10},
            None,
            None,
            None,
        )
        for index, (k, v) in enumerate(filtered.items())
    ]
    if num_nodes > 0:
        nodes.sort(key=lambda x: x.population, reverse=True)
        nodes = nodes[:num_nodes]

    # Setup demographics
    nyears = parameters.ticks // 365
    nnodes = len(nodes)
    demographics = DemographicsByYear(nyears, nnodes)
    populations = np.zeros(nnodes, dtype=np.int32)  # initial population (year 0)
    for i, node in enumerate(nodes):
        populations[i] = node.population
    # https://database.earth/population/nigeria/fertility-rate
    demographics.initialize(initial_population=populations, cbr=35.0, mortality=17.0, immigration=0.0)

    # Setup initial conditions (distribution of agents to S, E, I, and R)
    initial = np.zeros((nnodes, 4), dtype=np.uint32)  # 4 columns: S, E, I, R

    for i, node in enumerate(nodes):
        initial[i, 0] = np.uint32(np.round(node.population / parameters.r_naught))
        # initial[i, 1] = 0
        initial[i, 2] = 0 if node.population < 100_000 else 10
        initial[i, 3] = node.population - initial[i, 0:3].sum()

    # Setup network (gravity model)

    indices = np.array([node.index for node in nodes], dtype=np.uint32)
    network = np.array(gravity, dtype=np.float32)[indices][:, indices]
    network *= parameters.k

    outflows = network.sum(axis=0)
    max_frac = parameters.max_frac
    if (maximum := outflows.max()) > max_frac:
        print(f"Rescaling network by {max_frac / maximum}")
        network *= max_frac / maximum

    max_capacity = demographics.population[0, :].sum() + demographics.births.sum() + demographics.immigrations.sum()
    print(f"Initial population: {demographics.population[0, :].sum():>11,}")
    print(f"Total births:       {demographics.births.sum():>11,}")
    print(f"Total immigrations: {demographics.immigrations.sum():>11,}")
    print(f"Max capacity:       {max_capacity:>11,}")

    return (max_capacity, demographics, initial, network)
