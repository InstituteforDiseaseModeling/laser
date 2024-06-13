"""Load initial population, demographics, and network data for the Engwal scenario."""

from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np

from idmlaser.numpynumba import DemographicsStatic

Node = namedtuple("Node", ["name", "index", "population", "counts", "births", "immigrations", "deaths"])


def initialize_engwal(model, parameters, num_nodes: int = 0) -> Tuple[int, DemographicsStatic, np.ndarray, np.ndarray]:
    """Initialize the model with the England+Wales scenario."""
    if Path.cwd().parts[-1] == "tests":
        from data.engwaldata import data
    else:   # guess that we are running from a notebook in the root of the project
        from tests.data.engwaldata import data

    # get distances between nodes
    distances = np.load(Path(__file__).parent / "data" / "engwaldist.npy")

    nodes = [
        Node(
            k,
            index,
            v.population[0],
            {"I": v.cases[0]},
            np.zeros(parameters.ticks, dtype=np.uint32),
            np.zeros(parameters.ticks, dtype=np.uint32),
            np.zeros(parameters.ticks, dtype=np.uint32),
        )
        for index, (k, v) in enumerate(data.places.items())
    ]
    if num_nodes > 0:
        nodes.sort(key=lambda x: x.population, reverse=True)
        nodes = nodes[:num_nodes]

    nyears = parameters.ticks // 365
    nnodes = len(nodes)
    demographics = DemographicsStatic(nyears, nnodes)

    populations = np.zeros((nyears, nnodes), dtype=np.int32)
    births = np.zeros((nyears, nnodes), dtype=np.int32)
    deaths = np.zeros((nyears, nnodes), dtype=np.int32)
    immigrations = np.zeros((nyears, nnodes), dtype=np.int32)

    for i, node in enumerate(nodes):
        populations[:, i] = data.places[node.name].population[:nyears]
        births[:, i] = data.places[node.name].births[:nyears]

    deltapop = populations[1:, :] - populations[:-1, :]  # nyears - 1 entries
    deaths[:-1, :] = np.maximum(births[:-1, :] - deltapop, 0)  # if more births than increase in population, remove some agents
    immigrations[:-1, :] = np.maximum(deltapop - births[:-1, :], 0)  # if increase in population is more than births, add some agents

    demographics.initialize(
        population=populations,
        births=births,
        deaths=deaths,
        immigrations=immigrations,
    )

    initial = np.zeros((nnodes, 4), dtype=np.uint32)  # 4 columns: S, E, I, R

    for i, node in enumerate(nodes):
        initial[i, 0] = np.uint32(np.round(populations[0, i] / parameters.r_naught))
        # initial[i, 1] = 0
        initial[i, 2] = data.places[node.name].cases[0]
        initial[i, 3] = populations[0, i] - initial[i, 0:3].sum()

    indices = np.array([node.index for node in nodes], dtype=np.uint32)
    network = distances[indices][:, indices]
    network *= 1000  # convert to meters

    # gravity model: k * pop_1^a * pop_2^b / (N * dist^c)
    a = parameters.a
    b = parameters.b
    c = parameters.c
    k = parameters.k
    totalpop = sum(node.population for node in nodes)
    for i in range(len(nodes)):
        popi = nodes[i].population
        for j in range(i + 1, len(nodes)):
            popj = nodes[j].population
            network[i, j] = network[j, i] = k * (popi**a) * (popj**b) / (network[i, j] ** c)
    network /= totalpop

    outflows = network.sum(axis=0)
    max_frac = parameters.max_frac
    if (maximum := outflows.max()) > max_frac:
        print(f"Rescaling network by {max_frac / maximum}")
        network *= max_frac / maximum

    max_capacity = populations[0, :].sum() + births.sum() + immigrations.sum()
    print(f"Initial population: {populations[0, :].sum():>11,}")
    print(f"Total births:       {births.sum():>11,}")
    print(f"Total immigrations: {immigrations.sum():>11,}")
    print(f"Max capacity:       {max_capacity:>11,}")

    return (max_capacity, demographics, initial, network)
