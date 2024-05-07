"""Run the NumPy+Numba based spatial SEIR model."""

import json
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np

from idmlaser.models import NumbaSpatialSEIR
from idmlaser.numpynumba import DemographicsByYear
from idmlaser.numpynumba import DemographicsStatic


def main(parameters):
    """Run the model with the given parameters."""
    model = NumbaSpatialSEIR(parameters)

    if parameters.scenario == "engwal":
        max_capacity, demographics, initial, network = initialize_engwal(model, parameters, parameters.nodes)
    elif parameters.scenario == "nigeria":
        max_capacity, demographics, initial, network = initialize_nigeria(model, parameters, parameters.nodes)
    elif parameters.scenario == "ccs":
        max_capacity, demographics, initial, network = initialize_ccs(model, parameters)
    else:
        raise ValueError(f"Invalid scenario: {parameters.scenario}")

    model.initialize(max_capacity, demographics, initial, network)

    model.run(parameters.ticks)
    model.finalize()

    return


Node = namedtuple("Node", ["name", "index", "population", "counts", "births", "immigrations", "deaths"])


def initialize_engwal(model, parameters, num_nodes: int = 0) -> Tuple[int, DemographicsStatic, np.ndarray, np.ndarray]:
    """Initialize the model with the England+Wales scenario."""
    from data.engwaldata import data

    # get distances between communities
    distances = np.load(Path(__file__).parent / "data/engwaldist.npy")

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


def initialize_nigeria(model, parameters, num_nodes: int = 0, level: int = 5) -> Tuple[int, DemographicsByYear, np.ndarray, np.ndarray]:
    """Initialize the model with the Nigeria scenario."""
    from data.nigeria import gravity
    from data.nigeria import lgas

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


def initialize_ccs(model, parameters) -> Tuple[int, DemographicsByYear, np.ndarray, np.ndarray]:
    """Initialize the model with the CCS scenario."""

    # Setup demographics
    nyears = parameters.ticks // 365
    nnodes = parameters.nodes * parameters.nodes
    demographics = DemographicsByYear(nyears, nnodes)
    populations = np.zeros(nnodes, dtype=np.int32)  # initial population (year 0)
    for i, power in enumerate(np.linspace(3, 7, parameters.nodes)):
        for j in range(parameters.nodes):
            populations[i * parameters.nodes + j] = 10**power
    # https://database.earth/population/nigeria/fertility-rate
    demographics.initialize(initial_population=populations, cbr=35.0, mortality=0.0, immigration=0.0)

    # Setup initial conditions (distribution of agents to S, E, I, and R)
    initial = np.zeros((nnodes, 4), dtype=np.uint32)  # 4 columns: S, E, I, R
    for i in range(nnodes):
        initial[i, 0] = np.uint32(np.round(populations[i] / parameters.r_naught))
        # initial[i, 1] = 0
        initial[i, 2] = 10
        initial[i, 3] = populations[i] - initial[i, 0:3].sum()

    # Setup network (independent nodes)
    network = np.zeros((nnodes, nnodes), dtype=np.float32)

    max_capacity = demographics.population[0, :].sum() + demographics.births.sum() + demographics.immigrations.sum()
    print(f"Initial population: {demographics.population[0, :].sum():>11,}")
    print(f"Total births:       {demographics.births.sum():>11,}")
    print(f"Total immigrations: {demographics.immigrations.sum():>11,}")
    print(f"Max capacity:       {max_capacity:>11,}")

    return (max_capacity, demographics, initial, network)


def get_parameters():
    """Get the parameters for the model from the command line arguments."""
    DEF_SCENARIO = "engwal"
    DEF_TICKS = np.uint32(365)
    DEF_NNODES = np.uint32(0)
    EXP_MEAN = np.float32(7)
    EXP_STD = np.float32(1)
    INF_MEAN = np.float32(7)
    INF_STD = np.float32(1)
    # INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(14)
    SEED = np.uint32(20231205)
    OUTPUT_DIR = Path.cwd()

    # England + Wales distance model parameters
    DEF_A = np.float32(1.0)
    DEF_B = np.float32(1.0)
    DEF_C = np.float32(2.0)
    DEF_K = np.float32(500.0)
    DEF_MAX_FRAC = np.float32(0.05)

    nodes_help = f"The number of nodes in the network (0 == all) [{DEF_NNODES}]"
    inf_mean_help = f"The mean of the infectious period (beta will be r_naught/inf_mean) [{INF_MEAN}]"
    r_naught_help = f"The basic reproduction number (beta will be r_naught/inf_mean) [{R_NAUGHT}]"

    parser = ArgumentParser()
    parser.add_argument("--scenario", type=str, default=DEF_SCENARIO, help=f"The scenario to run {{'engwal'|'nigeria'}} [{DEF_SCENARIO}]")
    parser.add_argument("-t", "--ticks", type=np.uint32, default=DEF_TICKS, help=f"The number of timesteps [{DEF_TICKS}]")
    parser.add_argument("-n", "--nodes", type=np.uint32, default=DEF_NNODES, help=nodes_help)
    parser.add_argument("--exp_mean", type=np.float32, default=EXP_MEAN, help=f"The mean of the exposed period [{EXP_MEAN}]")
    parser.add_argument("--exp_std", type=np.float32, default=EXP_STD, help=f"The standard deviation of the exposed period [{EXP_STD}]")
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN, help=inf_mean_help)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD, help=f"The standard deviation of the infectious period [{INF_STD}]")
    # parser.add_argument("--initial_infs", type=np.uint32, default=INIT_INF, help=f"The initial number of infectious agents [{INIT_INF}]")
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT, help=r_naught_help)
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED, help=f"The random seed [{SEED}]")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR, help=f"Output directory ['{OUTPUT_DIR}']")

    DEF_SEASONALITY_FACTOR = np.float32(0.1)
    DEF_SEASONALITY_OFFSET = np.float32(182.5)

    parser.add_argument(
        "--seasonality_factor",
        type=np.float32,
        default=DEF_SEASONALITY_FACTOR,
        help=f"The seasonality factor for the transmission rate [{DEF_SEASONALITY_FACTOR:0.2f}]",
    )
    parser.add_argument(
        "--seasonality_offset",
        type=np.float32,
        default=DEF_SEASONALITY_OFFSET,
        help=f"The seasonality offset (ticks) for the transmission rate [{DEF_SEASONALITY_OFFSET}]",
    )

    # England + Wales distance model parameters
    parser.add_argument("--a", type=np.float32, default=DEF_A, help=f"England+Wales distance model parameter a [{DEF_A}]")
    parser.add_argument("--b", type=np.float32, default=DEF_B, help=f"England+Wales distance model parameter b [{DEF_B}]")
    parser.add_argument("--c", type=np.float32, default=DEF_C, help=f"England+Wales distance model parameter c [{DEF_C}]")
    parser.add_argument("--k", type=np.float32, default=DEF_K, help=f"England+Wales (or Nigeria) distance model parameter k [{DEF_K}]")
    parser.add_argument(
        "--max_frac",
        type=np.float32,
        default=DEF_MAX_FRAC,
        help=f"England+Wales (or Nigeria) maximum fraction of the network [{DEF_MAX_FRAC}]",
    )

    DEF_PARAMETERS = None  # Path(__file__).parent / "data/parameters.json"
    parser.add_argument("-p", "--parameters", type=Path, default=DEF_PARAMETERS, help=f"Parameters file [{DEF_PARAMETERS}]")

    args = parser.parse_args()

    if args.parameters:  # read parameters from file if --parameters is neither None nor empty
        with args.parameters.open("r") as file:
            params = json.load(file)
            for key, value in params.items():
                args.__setattr__(key, value)

    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    print(f"User parameters: {vars(args)}")

    return args


if __name__ == "__main__":
    print(f"Working directory: {Path.cwd()}")
    parameters = get_parameters()
    main(parameters)
