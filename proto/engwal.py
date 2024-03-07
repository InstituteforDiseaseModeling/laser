#! /usr/bin/env python3

"""England and Wales Measles Model"""

from argparse import ArgumentParser
from argparse import Namespace
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from engwaldata import data as engwal
from gcpopulation import Population
from groupedcommunity import Community
from tqdm import tqdm


def main(args: Namespace) -> None:
    """Main function for the England and Wales Measles model."""

    network, population = initialize_model(args)
    run_model(population, network, args)
    generate_results(population)

    return


def initialize_model(args: Namespace):
    """Initialize the model from files and arguments."""
    places = get_places(args.nodes)

    network = load_network(args.nodes, args.g_k, args.g_a, args.g_b, args.g_c)

    # Create a list of communities, one for each population in pops
    pop = Population(
        args.nodes,
        community_props=None,  # ["population", "births"],
        agent_groups=["unactive", "susceptible", "exposed", "infectious", "recovered", "deceased"],
        agent_props=[
            ("dob", np.int16, 0),
            ("susceptibility", np.uint8, 0),
            ("etimer", np.uint8, 0),
            ("itimer", np.uint8, 0),
            ("uid", np.uint32, 0),
        ],
    )

    def init_community(_population: Population, community: Community, index: int) -> Tuple[np.ndarray, Population]:
        """Callback to initialize a community with its properties and populations."""
        print(f"Realizing community {index:3}... ", end="")

        community.population = engwal.places[places[index].name].population
        community.births = engwal.places[places[index].name].births

        max_pop = community.population.max()
        init_pop = community.population[0]

        unactive = max_pop - init_pop
        susceptible = np.uint32(np.round(init_pop / args.r_naught))
        recovered = init_pop - susceptible
        # add groups: unactive, susceptible, exposed, infectious, recovered, and deceased
        print(f"{index:3} unactive: {unactive:8}, susceptible: {susceptible:8}, recovered: {recovered:8}")
        # unlisted entries will default to 0 population
        pops = {"unactive": unactive, "susceptible": susceptible, "recovered": recovered}
        pop.allocate_community(community, pops=pops)

        community.susceptible.dob = -np.random.exponential(2.5 * 365, len(community.susceptible)).astype(community.susceptible.dob.dtype)
        community.recovered.dob = -(np.minimum(np.random.exponential(40 * 365, len(community.recovered)), 88 * 365) + 365)
        community.susceptible.susceptibility = 1

        if engwal.places[places[index].name].cases[0] > 0:
            i = np.random.randint(0, len(community.susceptible))
            community.susceptible.itimer[i] = max(1, np.round(np.random.normal(args.inf_mean, args.inf_std))) + 1
            community.susceptible.susceptibility[i] = 0
            community.move(community.gmap["susceptible"], i, community.gmap["infectious"])

        return

    pop.realize(init_community)

    pop.add_population_property("contagion", np.zeros(len(pop.communities), dtype=np.uint32))
    pop.add_population_property("report", np.zeros((args.ticks + 1, len(pop.communities), 6), dtype=np.uint32))

    return network, pop


def run_model(population: Population, network: np.ndarray, args: Namespace) -> None:
    """Run the model for the specified number of ticks."""
    population.apply(update_report, tick=0)  # Capture the initial state of the population.

    for tick in tqdm(range(args.ticks)):
        # 1 vital dynamics (deaths, births, and immigration)
        population.apply(do_vital_dynamics, tick=tick)

        # 2 - Update infectious agents
        population.apply(update_infections)

        # 3 - Update exposed agents
        population.apply(update_exposures, inf_mean=args.inf_mean, inf_std=args.inf_std)

        # 4 - Transmit to susceptible agents
        population.apply(update_contagion)
        transfer = population.contagion * network
        population.contagion += transfer.sum(axis=1).round().astype(population.contagion.dtype)  # increment by incoming infections
        population.contagion -= transfer.sum(axis=0).round().astype(population.contagion.dtype)  # decrement by outgoing infections
        population.apply(do_transmission, beta=args.beta, exp_mean=args.exp_mean, exp_std=args.exp_std)

        # 6 - Gather statistics for reporting
        population.apply(update_report, tick=tick + 1)

    return


def generate_results(population: Population) -> None:
    """Generate the results of the model (files, plots, etc.)."""
    np.save("report.npy", population.report)  # Save the raw report data
    plot_report(population.report)  # Plot the susceptible and infectious traces

    if False:
        df = pd.DataFrame(population.report[:, :, 3])
        df.to_csv("report.csv")

    return


@lru_cache
def get_places(num_nodes: np.uint32) -> List[Tuple[str, int, int]]:
    """Get a list of places for the model."""
    Place = namedtuple("Place", ["name", "index", "population"])
    places = []
    for index, placename in enumerate(engwal.placenames):
        # Create list of tuples of (placename, index, population)
        places.append(Place(placename, index, engwal.places[placename].population[-1]))

    places = sorted(places, key=lambda x: x[2], reverse=True)  # Sort places by population
    places = places[:num_nodes]  # Only take the top num_nodes places

    return places


def load_network(num_nodes: np.uint32, k: np.float32, a: np.float32, b: np.float32, c: np.float32) -> np.ndarray:
    """Load network data for England and Wales."""
    # Create a list of indices for the selected places
    places = get_places(num_nodes)
    indices = np.array([place.index for place in places])
    distances = np.load(Path(__file__).parent / "engwaldist.npy")
    # Only take the distances between the selected places
    distances = distances[indices][:, indices]

    network = np.zeros_like(distances, dtype=np.float32)

    # Gravity model: k * pop1^a * pop2^b / N / distance^c
    # TODO - ¿should not include population here, but use instantaneous population at time of transmission?
    N = sum(place.population for place in places)
    print(f"Total population: {N:,}")
    for i in range(num_nodes):
        popi = places[i].population
        for j in range(i + 1, num_nodes):
            popj = places[j].population
            distm = distances[i, j] * 1000  # convert to meters TODO - move into k
            network[i, j] = k * (popi**a) * (popj**b) / (N * (distm**c))
            network[j, i] = network[i, j]

    return network


def update_report(p: Population, c: Community, i: int, tick: int) -> None:
    """Capture the current state of the community."""
    p.report[tick, i, 0] = len(c.unactive)
    p.report[tick, i, 1] = len(c.susceptible)
    p.report[tick, i, 2] = len(c.exposed)
    p.report[tick, i, 3] = len(c.infectious)
    p.report[tick, i, 4] = len(c.recovered)
    p.report[tick, i, 5] = len(c.deceased)

    return


def do_vital_dynamics(_p: Population, c: Community, _i: int, tick: np.uint32) -> None:
    """Do the vital dynamics of births, deaths, and external immigration."""
    year = tick // 365  # Determine the year to look up total population change and births for the year.
    if year < (len(c.population) - 1):  # If we are not in the last year of the data
        doy = (tick % 365) + 1  # Determine the day of the year 1..365
        delta = np.int32(c.population[year + 1]) - np.int32(c.population[year])  # Determine the change in population
        births = c.births[year]  # Get the number of births for the year

        if births > delta:  # If there are more births than total population change, then there are deaths
            deaths = births - delta
            deaths = np.int32((deaths * doy // 365) - (deaths * (doy - 1) // 365))  # Interpolate the number of deaths
            immigrants = 0
        elif delta > births:  # If the total population change is greater than the number of births
            deaths = 0
            immigrants = delta - births
            immigrants = np.int32((immigrants * doy // 365) - (immigrants * (doy - 1) // 365))  # Interpolate immigrantss
        else:  # If the total population change is equal to the number of births
            deaths = 0
            immigrants = 0

        births = np.int32((births * doy // 365) - (births * (doy - 1) // 365))  # Interpolate the number of births

        iunactive = c.gmap["unactive"]
        isusceptible = c.gmap["susceptible"]
        irecovered = c.gmap["recovered"]
        ideceased = c.gmap["deceased"]

        # 1 - Handle non-disease related deaths
        for _ in range(min(deaths, limit := len(c.recovered))):
            target = np.random.randint(limit)  # Randomly select a recovered agent to die
            c.move(irecovered, target, ideceased)
            limit -= 1

        # 2 - Handle births
        for _ in range(min(births, index := len(c.unactive))):
            # We will move the last of the unactive to the first of the susceptibles. It is more efficient (saves a copy).
            index -= 1
            c.unactive.dob[index] = tick
            c.unactive.susceptibility[index] = 1
            c.move(iunactive, index, isusceptible)

        # 3 - Handle immigration
        for _ in range(min(immigrants, index := len(c.unactive))):
            # We will move the last of the unactive. It is more efficient (saves a copy).
            index -= 1
            c.unactive.dob[index] = np.random.randint(0, 365 * 89)  # random age between 0 and 89 years
            c.unactive.susceptibility[index] = 0  # Assume immigrant is not susceptible
            c.move(iunactive, index, irecovered)

    return


def update_infections(_p: Population, c: Community, _i: int) -> None:
    """Update the infectious agents."""

    itimers = c.infectious.itimer
    iinfectious = c.gmap["infectious"]
    irecovered = c.gmap["recovered"]
    for index in range(len(c.infectious) - 1, -1, -1):  # Iterate over the infectious agents in reverse order
        timer = itimers[index] - 1
        itimers[index] = timer
        if timer == 0:  # If the timer has expired move the agent to the recovered group
            c.move(iinfectious, index, irecovered)

    return


def update_exposures(_p: Population, c: Community, _i: int, inf_mean: np.float32, inf_std: np.float32) -> None:
    """Update the exposed agents."""
    etimers = c.exposed.etimer
    itimers = c.exposed.itimer
    iexposed = c.gmap["exposed"]
    iinfectious = c.gmap["infectious"]
    for index in range(len(c.exposed) - 1, -1, -1):  # Iterate over the exposed agents in reverse order
        timer = etimers[index] - 1
        etimers[index] = timer
        if timer == 0:  # If the timer has expired move the agent to the infectious group
            itimers[index] = max(1, int(np.round(np.random.normal(inf_mean, inf_std))))
            c.move(iexposed, index, iinfectious)

    return


def update_contagion(p: Population, c: Community, i: int) -> None:
    """Update the contagion."""
    p.contagion[i] = len(c.infectious)

    return


def do_transmission(p: Population, c: Community, i: int, beta: np.float32, exp_mean: np.float32, exp_std: np.float32) -> None:
    """Do the transmission."""
    susceptibility = c.susceptible.susceptibility
    etimers = c.susceptible.etimer
    isusceptible = c.gmap["susceptible"]
    iexposed = c.gmap["exposed"]
    # TODO - iterate over groups to get N (i.e., if we change the groups, this code breaks)
    N = len(c.susceptible) + len(c.exposed) + len(c.infectious) + len(c.recovered)
    force = beta * p.contagion[i] * len(c.susceptible) / N
    num_exposures = np.uint32(np.round(np.random.poisson(force)))
    if num_exposures >= len(c.susceptible):
        # raise ValueError(f"Too many exposures: {num_exposures} >= {len(c.susceptible)}")
        print(f"Too many exposures: {num_exposures} >= {len(c.susceptible)}")
        num_exposures = len(c.susceptible)
    for _ in range(min(num_exposures, limit := len(c.susceptible))):
        target = np.random.randint(limit)
        if np.random.uniform() < susceptibility[target]:
            susceptibility[target] = 0
            etimers[target] = max(1, int(np.round(np.random.normal(exp_mean, exp_std))))
            c.move(isusceptible, target, iexposed)
            limit -= 1

    return


def plot_report(report: np.ndarray) -> None:
    """Plot the susceptible and infectious traces from the report."""

    def plot_trace(report: np.ndarray, index: int, trace: str) -> None:
        """Plot the trace for a given index."""
        df = pd.DataFrame(report[:, :, index], columns=[f"{trace}{(i+1):02}" for i in range(report.shape[1])])
        axs = df.plot()
        axs.set_xlabel("ticks")
        fig = axs.get_figure()
        fig.set_size_inches(18, 12)
        fig.tight_layout()
        fig.savefig(f"{trace}.png", dpi=300)

        return

    plot_trace(report, 1, "sus")
    plot_trace(report, 3, "inf")

    return


def parse_args() -> Namespace:
    """Parse command line arguments."""

    TICKS = np.uint32(10 * 365)  # 10 years
    NODES = np.uint32(32)  # top 32 places by population
    EXP_MEAN = np.float32(4)  # 4 days
    EXP_STD = np.float32(1)  # 1 day
    INF_MEAN = np.float32(5)  # 5 days
    INF_STD = np.float32(1)  # 1 day
    R_NAUGHT = np.float32(2.5)  # R0
    SEED = np.uint32(20240227)  # random seed

    parser = ArgumentParser()
    parser.add_argument("-t", "--ticks", type=np.uint32, default=TICKS)
    parser.add_argument("-n", "--nodes", type=np.uint32, default=NODES)
    parser.add_argument("--exp_mean", type=np.float32, default=EXP_MEAN)
    parser.add_argument("--exp_std", type=np.float32, default=EXP_STD)
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED)

    DEF_K = np.float32(500)
    DEF_A = np.float32(1.0)
    DEF_B = np.float32(1.0)
    DEF_C = np.float32(2.0)

    parser.add_argument("--g_k", type=np.float32, default=DEF_K)
    parser.add_argument("--g_a", type=np.float32, default=DEF_A)
    parser.add_argument("--g_b", type=np.float32, default=DEF_B)
    parser.add_argument("--g_c", type=np.float32, default=DEF_C)

    args = parser.parse_args()
    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
