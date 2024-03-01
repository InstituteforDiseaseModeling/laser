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
from groupedcommunity import Community
from tqdm import tqdm


def main(args: Namespace) -> None:
    """Main function for the England and Wales Measles model."""
    places = get_places(args.nodes)

    network = load_network(args.nodes, args.g_k, args.g_a, args.g_b, args.g_c)

    # Create a list of communities, one for each population in pops
    communities = [Community() for _ in range(args.nodes)]
    # Enumerate communities to get index and community
    for index, community in enumerate(communities):
        # Add community properties: annual population and births
        community.add_community_property("population", engwal.places[places[index].name].population)
        community.add_community_property("births", engwal.places[places[index].name].births)

        max_pop = community.population.max()
        init_pop = community.population[0]

        unactive = max_pop - init_pop
        susceptible = np.uint32(np.round(init_pop / args.r_naught))
        recovered = init_pop - susceptible
        # add groups: unactive, susceptible, exposed, infectious, recovered, and deceased
        print(f"{index:2} unactive: {unactive:8}, susceptible: {susceptible:8}, recovered: {recovered:8}")
        community.add_agent_group("unactive", unactive)
        community.add_agent_group("susceptible", susceptible)
        community.add_agent_group("exposed", 0)
        community.add_agent_group("infectious", 0)
        community.add_agent_group("recovered", recovered)
        community.add_agent_group("deceased", 0)
        # add properties: dob, susceptibility, etimer, itimer, and uid
        community.add_agent_property("dob", np.int16, 0)
        community.add_agent_property("susceptibility", np.uint8, 0)
        community.add_agent_property("etimer", np.uint8, 0)
        community.add_agent_property("itimer", np.uint8, 0)
        community.add_agent_property("uid", np.uint32, 0)

        community.allocate()
        community.susceptible.dob = -np.random.exponential(2.5 * 365, len(community.susceptible)).astype(community.susceptible.dob.dtype)
        community.recovered.dob = -(np.minimum(np.random.exponential(40 * 365, len(community.recovered)), 88 * 365) + 365)
        community.susceptible.susceptibility = 1

    # Seed infections - enumerate each community and get index and community
    for community, place in zip(communities, places):
        if engwal.places[place.name].cases[0] > 0:
            i = np.random.randint(0, len(community.susceptible))
            community.susceptible.susceptibility[i] = 0
            community.itimer[i] = max(1, np.round(np.random.normal(args.inf_mean, args.inf_std))) + 1
            community.move(community.gmap["susceptible"], i, community.gmap["infectious"])

    contagion = np.zeros(len(communities), dtype=np.uint32)
    report = np.zeros((args.timesteps + 1, len(communities), 6), dtype=np.uint32)

    update_report(communities, report, 0)

    for tick in tqdm(range(args.timesteps)):
        # 1 vital dynamics (deaths, births, and immigration)
        for community in communities:
            do_vital_dynamics(community, tick)

        # 2 - Update infectious agents
        for community in communities:
            update_infections(community)

        # 3 - Update exposed agents
        for community in communities:
            update_exposures(community, args.inf_mean, args.inf_std)

        # 4 - Transmit to susceptible agents
        for index, community in enumerate(communities):
            contagion[index] = len(community.infectious)
        # transfer = (contagion * network).round().astype(contagion.dtype)
        transfer = contagion * network
        contagion += transfer.sum(axis=1).round().astype(contagion.dtype)  # increment by incoming infections
        contagion -= transfer.sum(axis=0).round().astype(contagion.dtype)  # decrement by outgoing infections
        for index, community in enumerate(communities):
            do_transmission(community, contagion[index], args.beta, args.exp_mean, args.exp_std)

        # 6 - Gather statistics for reporting
        update_report(communities, report, tick + 1)

    # Save the report
    np.save("report.npy", report)

    # Plot the report
    plot_report(report)

    return


Place = namedtuple("Place", ["name", "index", "population"])


@lru_cache
def get_places(num_nodes: np.uint32) -> List[Tuple[str, int, int]]:
    """Get a list of places for the model."""
    places = []
    for index, placename in enumerate(engwal.placenames):
        # Create list tuples of (placename, index, population)
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
    for i in range(num_nodes):
        popi = places[i].population
        for j in range(i + 1, num_nodes):
            popj = places[j].population
            distm = distances[i, j] * 1000  # convert to meters
            network[i, j] = k * (popi**a) * (popj**b) / (N * (distm**c))
            network[j, i] = network[i, j]

    return network


def update_report(communities: List[Community], report: np.ndarray, tick: np.uint32) -> None:
    """Get a report of the current state of the communities."""
    for index, community in enumerate(communities):
        report[tick, index, 0] = len(community.unactive)
        report[tick, index, 1] = len(community.susceptible)
        report[tick, index, 2] = len(community.exposed)
        report[tick, index, 3] = len(community.infectious)
        report[tick, index, 4] = len(community.recovered)
        report[tick, index, 5] = len(community.deceased)

    return


def do_vital_dynamics(c: Community, tick: np.uint32) -> None:
    """Do the vital dynamics of births, deaths, and external immigration."""
    year = tick // 365  # Determine the year to look up total population change and births for the year.
    if year < (len(c.population) - 1):  # If we are not in the last year of the data
        doy = (tick % 365) + 1  # Determine the day of the year 1..365
        delta = np.int32(c.population[year + 1]) - np.int32(c.population[year])  # Determine the change in population
        delta = np.int32((delta * doy // 365) - (delta * (doy - 1) // 365))  # Interpolate the change in population
        births = c.births[year]  # Get the number of births for the year
        births = np.int32((births * doy // 365) - (births * (doy - 1) // 365))  # Interpolate the number of births

        iunactive = c.gmap["unactive"]
        isusceptible = c.gmap["susceptible"]
        irecovered = c.gmap["recovered"]
        ideceased = c.gmap["deceased"]

        # 1 - Handle non-disease related deaths
        if births > delta:  # If there are more births than total population change, then there are deaths
            deaths = births - delta
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
        if delta > births:  # If the total population change is greater than the number of births
            immigrants = delta - births
            for _ in range(min(immigrants, index := len(c.unactive))):
                # We will move the last of the unactive to the first of the susceptibles. It is more efficient (saves a copy).
                index -= 1
                c.unactive.dob[index] = np.random.randint(0, 365 * 89)  # random age between 0 and 89 years
                c.unactive.susceptibility[index] = 0  # Assume immigrant is not susceptible
                c.move(iunactive, index, irecovered)

    return


def update_infections(c: Community) -> None:
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


def update_exposures(c: Community, inf_mean: np.float32, inf_std: np.float32) -> None:
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


def do_transmission(c: Community, contagion: np.float32, beta: np.float32, exp_mean: np.float32, exp_std: np.float32) -> None:
    """Do the transmission."""
    susceptibility = c.susceptible.susceptibility
    etimers = c.susceptible.etimer
    isusceptible = c.gmap["susceptible"]
    iexposed = c.gmap["exposed"]
    # TODO - iterate over groups to get N (i.e., if we change the groups, this code breaks)
    N = len(c.susceptible) + len(c.exposed) + len(c.infectious) + len(c.recovered)
    force = beta * contagion * len(c.susceptible) / N
    num_exposures = np.uint32(np.round(np.random.poisson(force)))
    if num_exposures >= len(c.susceptible):
        raise ValueError(f"Too many exposures: {num_exposures} >= {len(c.susceptible)}")
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
        df = pd.DataFrame(report[:, :, index], columns=[f"{trace}{i:02}" for i in range(1, 33)])
        axs = df.plot()
        axs.set_xlabel("ticks")
        fig = axs.get_figure()
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        fig.savefig(f"{trace}.png", dpi=300)

        return

    plot_trace(report, 1, "sus")
    plot_trace(report, 3, "inf")

    return


def parse_args() -> Namespace:
    """Parse command line arguments."""

    TIMESTEPS = np.uint32(10 * 365)  # 10 years
    NODES = np.uint32(32)  # top 32 places by population
    EXP_MEAN = np.float32(4)  # 4 days
    EXP_STD = np.float32(1)  # 1 day
    INF_MEAN = np.float32(5)  # 5 days
    INF_STD = np.float32(1)  # 1 day
    # INIT_INF = np.uint32(10)        # 10 initial infections
    R_NAUGHT = np.float32(2.5)  # R0
    SEED = np.uint32(20240227)  # random seed

    parser = ArgumentParser()
    parser.add_argument("--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("-n", "--nodes", type=np.uint32, default=NODES)
    parser.add_argument("--exp_mean", type=np.float32, default=EXP_MEAN)
    parser.add_argument("--exp_std", type=np.float32, default=EXP_STD)
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD)
    # parser.add_argument("--initial_infs", type=np.uint32, default=INIT_INF)
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
