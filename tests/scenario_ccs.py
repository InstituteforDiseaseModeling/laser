"""Load initial population, demographics, and network data for the Critical Community Size scenario."""

from typing import Tuple

import numpy as np

from idmlaser.numpynumba import DemographicsByYear


def initialize_ccs(model, parameters, num_nodes: int = 0) -> Tuple[int, DemographicsByYear, np.ndarray, np.ndarray]:
    """Initialize the model for CCS."""
    nyears = np.int32(np.ceil(parameters.ticks / 365))
    nnodes = parameters.nodes
    total_nodes = nnodes * nnodes
    print(f"Instantiating DemographicsByYears{(nyears, total_nodes)}...")
    demographics = DemographicsByYear(nyears, total_nodes)
    populations = np.zeros(total_nodes, dtype=np.int32)  # initial nodes populations (year 0)
    for i, p in enumerate(np.linspace(4, 6, nnodes)):
        for j in range(nnodes):
            populations[i * nnodes + j] = np.int32(np.round(np.power(10, p)))
    demographics.initialize(initial_population=populations, cbr=20.0, mortality=0.0, immigration=0.0)

    initial = np.zeros((total_nodes, 4), dtype=np.uint32)  # S, E, I, R
    initial[:, 0] = np.int32(np.round(populations / parameters.r_naught))  # S
    initial[:, 1] = 15  # E
    initial[:, 2] = 10  # I
    initial[:, 3] = populations - initial[:, 0:3].sum(axis=1)  # R

    network = np.zeros((total_nodes, total_nodes), dtype=np.float32)

    max_capacity = demographics.population[0, :].sum() + demographics.births.sum() + demographics.immigrations.sum()
    print(f"Initial population: {demographics.population[0, :].sum():>11,}")
    print(f"Total births:       {demographics.births.sum():>11,}")
    print(f"Total immigrations: {demographics.immigrations.sum():>11,}")
    print(f"Max capacity:       {max_capacity:>11,}")

    return (max_capacity, demographics, initial, network)
