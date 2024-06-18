"""Run the Taichi based spatial SEIR model."""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np

from idmlaser.models import TaichiSpatialSEIR
from idmlaser.numpynumba import DemographicsByYear


def main(parameters):
    """Run the Taichi based spatial SEIR model."""
    model = TaichiSpatialSEIR(parameters)

    if parameters.scenario == "engwal":
        from scenario_engwal import initialize_engwal

        max_capacity, demographics, initial, network = initialize_engwal(model, parameters, parameters.nodes)
    elif parameters.scenario == "nigeria":
        from scenario_nigeria import initialize_nigeria

        max_capacity, demographics, initial, network = initialize_nigeria(model, parameters, parameters.nodes)
    elif parameters.scenario == "ccs":
        from scenario_ccs import initialize_ccs

        max_capacity, demographics, initial, network = initialize_ccs(model, parameters)
    elif parameters.scenario == "test":
        max_capacity, demographics, initial, network = initialize_test(model, parameters, parameters.nodes)
    else:
        raise ValueError(f"Invalid scenario: {parameters.scenario}")

    model.initialize(max_capacity, demographics, initial, network)

    model.run(parameters.ticks)
    model.finalize()

    metrics = np.array(model.metrics)

    for c in range(metrics.shape[1]):
        if c == 0:
            continue  # Skip the first column, "ticks"
        print(f"{model._phases[c-1].__name__:20}: {metrics[:,c].sum():13,} μsec")
    print("----------------------------------------")
    print(f"total runtime       : {metrics[:, 1:].sum():13,} μsec")

    return


def initialize_test(model, parameters, num_nodes: int = 0) -> Tuple[int, DemographicsByYear, np.ndarray, np.ndarray]:
    """Initialize the model with test values."""
    nyears = np.int32(np.ceil(parameters.ticks / 365))
    nnodes = parameters.nodes
    print(f"Instantiating DemographicsByYears{(nyears, nnodes)}...")
    demographics = DemographicsByYear(nyears, nnodes)
    NODE_POP = 10_000
    populations = np.full(nnodes, NODE_POP, dtype=np.int32)  # initial nodes populations (year 0)
    demographics.initialize(initial_population=populations, cbr=0.0, mortality=0.0, immigration=0.0)

    initial = np.zeros((nnodes, 4), dtype=np.uint32)  # S, E, I, R
    initial[:, 0] = int(round(NODE_POP / parameters.r_naught))  # S
    initial[:, 1] = 15  # E
    initial[:, 2] = 10  # I
    initial[:, 3] = NODE_POP - initial[:, 0:3].sum()  # R

    network = np.zeros((nnodes, nnodes), dtype=np.float32)

    max_capacity = nnodes * 1000

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
