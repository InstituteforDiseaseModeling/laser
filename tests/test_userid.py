#! /usr/bin/env python3

"""England and Wales Measles Model"""

import json
from argparse import ArgumentParser
from argparse import Namespace
from collections import namedtuple
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from engwaldata import data as engwal
from idmlaser.userid import Community
from idmlaser.userid import Population
from idmlaser.userid.population import ScheduledEvent

# [X] vital dynamics
# [X] network connectivity, gravity model
# [X] maternal protection, binary
# [ ] age-based mixing
# [X] seasonal forcing, sinusoidal
# [X] routine immunization
# [X] SIAs
# [X] age-based initialization of susceptibles
# [ ] correlation between individual level acquisition and transmission
# [ ] non-uniform shedding


def main(args: Namespace) -> None:
    """Main function for the England and Wales Measles model."""

    print(f"Running England and Wales Measles model with\n{args.__dict__}")

    network, population = initialize_model(args)
    run_model(population, network, args)
    generate_results(population)

    return


def initialize_model(args: Namespace):
    """Initialize the model from files and arguments."""
    print(f"Initializing np.random with seed {args.seed}... ")
    np.random.seed(args.seed)

    places = get_places(args.nodes)
    num_places = len(places)
    network = load_network(num_places, args.g_k, args.g_a, args.g_b, args.g_c, args.g_max_frac)

    # Create a list of communities, one for each population in pops
    print(f"Initializing model with {num_places} nodes... ")
    pop = Population(
        num_communities=num_places,
        community_props=None,  # ["population", "births"],
        agent_groups=["unactive", "infants", "susceptible", "exposed", "infectious", "recovered", "deceased"],
        agent_props=[
            ("dob", np.int16, 0),  # day of birth
            ("susceptibility", np.uint8, 0),  # susceptibility to infection
            ("dmabs", np.int16, 0),  # day of maternal antibody 50% waning
            ("dri", np.int16, 0),  # day of routine immunization
            ("etimer", np.uint8, 0),  # exposure timer
            ("itimer", np.uint8, 0),  # infectious timer
            ("uid", np.uint32, 0),  # unique identifier
        ],
    )

    pop.seasonal_factor = args.seasonality

    def init_community(_population: Population, community: Community, index: int) -> Tuple[np.ndarray, Population]:
        """Callback to initialize a community with its properties and populations."""
        print(f"Realizing community {index:3}... ", end="")

        community.population = engwal.places[places[index].name].population
        community.births = engwal.places[places[index].name].births
        community.ri_coverage = args.ri_coverage

        max_pop = community.population.max()
        init_pop = community.population[0]

        unactive = max_pop - init_pop
        # Adding a little to the susceptible population to help the initial infections take root
        # susceptible = np.uint32(np.round(1.0625 * init_pop / args.r_naught))
        susceptible = np.uint32(np.round(1.0 * init_pop / args.r_naught)) + 1
        recovered = init_pop - susceptible
        # add groups: unactive, susceptible, exposed, infectious, recovered, and deceased
        print(f"{index:3} unactive: {unactive:8}, susceptible: {susceptible:8}, recovered: {recovered:8}", end="")
        # unlisted entries will default to 0 population
        pops = {"unactive": unactive, "susceptible": susceptible, "recovered": recovered}
        pop.allocate_community(community, pops=pops)

        community.susceptible.dob = -np.random.exponential(2.5 * 365, len(community.susceptible)).astype(community.susceptible.dob.dtype)
        community.susceptible.dmabs = community.susceptible.dob + np.round(np.random.normal(180, 15, len(community.susceptible))).astype(
            community.susceptible.dmabs.dtype
        )
        community.susceptible.dri = (
            community.susceptible.dob
            + 270
            + np.round(np.random.uniform(-30, 30, len(community.susceptible))).astype(community.susceptible.dri.dtype)
        )
        community.recovered.dob = -(np.minimum(np.random.exponential(40 * 365, len(community.recovered)), 88 * 365) + 365)
        community.susceptible.susceptibility = 1

        # if engwal.places[places[index].name].cases[0] > 0:
        n_infs = np.random.poisson(args.init_infs)
        print(f"Initial infections: {n_infs}")
        for _ in range(n_infs):
            i = np.random.randint(0, len(community.susceptible))
            community.susceptible.itimer[i] = max(1, np.round(np.random.normal(args.inf_mean, args.inf_std))) + 1
            community.susceptible.susceptibility[i] = 0
            community.move(community.gmap["susceptible"], i, community.gmap["infectious"])

        return

    pop.realize(init_community)

    pop.add_population_property("contagion", np.zeros(len(pop.communities), dtype=np.uint32))
    pop.add_population_property("report", np.zeros((args.ticks + 1, len(pop.communities), 6), dtype=np.uint32))

    return network, pop


def supplemental_immunization_activity(population: Population, community: Community, index: int, tick: int) -> None:
    """Supplemental Immunization Activity."""
    print(f"Running an SIA in community {index} at tick {tick}... ", end="")
    print(f"Considering {len(community.infants)} infants and {len(community.susceptible)} susceptibles: ", end="")
    dobs = community.infants.dob
    susceptibility = community.infants.susceptibility
    iinfants = community.gmap["infants"]
    irecovered = community.gmap["recovered"]
    cinfants = 0
    for index in range(len(community.infants) - 1, -1, -1):
        agedays = tick - dobs[index]
        if agedays > 270 and agedays < 5 * 365:  # 9+ months to 5 years
            if np.random.uniform() < 0.9:
                susceptibility[index] = 0
                community.move(iinfants, index, irecovered)
                cinfants += 1
    dobs = community.susceptible.dob
    susceptibility = community.susceptible.susceptibility
    isusceptible = community.gmap["susceptible"]
    csusceptible = 0
    for index in range(len(community.susceptible) - 1, -1, -1):
        agedays = tick - dobs[index]
        if agedays > 270 and agedays < 5 * 365:  # 9+ months to 5 years
            if np.random.uniform() < 0.9:
                susceptibility[index] = 0
                community.move(isusceptible, index, irecovered)
                csusceptible += 1
    print(f"Vaccinated {cinfants} infants and {csusceptible} susceptibles.")
    return


def run_model(population: Population, network: np.ndarray, args: Namespace) -> None:
    """Run the model for the specified number of ticks."""
    population.apply(update_report, tick=0)  # Capture the initial state of the population.

    population.add_event(ScheduledEvent(supplemental_immunization_activity, {0, 3}, [], {"tick": 290}), 290)
    # population.add_event(ScheduledEvent(supplemental_immunization_activity, {}, [], {"tick": 42}), 42)

    print(f"Running model for {args.ticks} ticks... ")
    for tick in (pbar := tqdm(range(args.ticks))):
        # 1 vital dynamics (deaths, births, and immigration)
        population.apply(do_vital_dynamics, tick=tick)

        # 2 - Update infectious agents
        population.apply(update_infections)

        # 3 - Update exposed agents
        population.apply(update_exposures, inf_mean=args.inf_mean, inf_std=args.inf_std)

        # 4 - Transmit to susceptible agents
        population.apply(update_contagion, tick=tick)
        transfer = population.contagion * network
        population.contagion += transfer.sum(axis=1).round().astype(population.contagion.dtype)  # increment by incoming infections
        population.contagion -= transfer.sum(axis=0).round().astype(population.contagion.dtype)  # decrement by outgoing infections
        population.apply(do_transmission, tick=np.int16(tick), beta=args.beta, exp_mean=args.exp_mean, exp_std=args.exp_std, pbar=pbar)

        # 5 - Do routine immunization
        population.apply(do_routine_immunization, tick=tick)

        # ? - Do scheduled events
        population.do_events(tick)

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
    if num_nodes > 0:
        places = places[:num_nodes]  # Only take the top num_nodes places

    return places


def load_network(num_nodes: np.uint32, k: np.float32, a: np.float32, b: np.float32, c: np.float32, max_frac: np.float32) -> np.ndarray:
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

    outflows = network.sum(axis=0)
    if (maximum := outflows.max()) > 0:
        network *= max_frac / maximum

    return network


def update_report(p: Population, c: Community, i: int, tick: int) -> None:
    """Capture the current state of the community."""
    p.report[tick, i, 0] = len(c.unactive)
    p.report[tick, i, 1] = len(c.infants) + len(c.susceptible)
    p.report[tick, i, 2] = len(c.exposed)
    p.report[tick, i, 3] = len(c.infectious)
    p.report[tick, i, 4] = len(c.recovered)
    p.report[tick, i, 5] = len(c.deceased)

    return


DEATHS = 0
BIRTHS = 0
IMMIGRATION = 0


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
        iinfants = c.gmap["infants"]
        # isusceptible = c.gmap["susceptible"]
        irecovered = c.gmap["recovered"]
        ideceased = c.gmap["deceased"]

        global DEATHS, BIRTHS, IMMIGRATION
        DEATHS += min(deaths, len(c.recovered))
        BIRTHS += min(births, len(c.unactive))
        IMMIGRATION += min(immigrants, len(c.unactive))

        # 1 - Handle non-disease related deaths
        for _ in range(min(deaths, limit := len(c.recovered))):
            target = np.random.randint(limit)  # Randomly select a recovered agent to die
            c.move(irecovered, target, ideceased)
            limit -= 1

        # 2 - Handle births
        dobs = c.unactive.dob
        dri = c.unactive.dri
        susceptibility = c.unactive.susceptibility
        dmabs = c.unactive.dmabs
        for _ in range(min(births, index := len(c.unactive))):
            # We will move the last of the unactive to the first of the infants. It is more efficient (saves a copy).
            index -= 1
            dobs[index] = tick
            dri[index] = tick + 270 + np.round(np.random.uniform(-30, 30)).astype(dri.dtype)
            susceptibility[index] = 1
            dmabs[index] = tick + np.int16(np.random.normal(180, 15))
            c.move(iunactive, index, iinfants)

        # 3 - Handle immigration
        for _ in range(min(immigrants, index := len(c.unactive))):
            # We will move the last of the unactive. It is more efficient (saves a copy).
            index -= 1
            dobs[index] = np.random.randint(0, 365 * 89)  # random age between 0 and 89 years
            susceptibility[index] = 0  # Assume immigrant is not susceptible
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
            itimers[index] = max(2, int(np.round(np.random.normal(inf_mean, inf_std))) + 1)
            c.move(iexposed, index, iinfectious)

    return


def update_contagion(p: Population, c: Community, i: int, tick: int) -> None:
    """Update the contagion."""
    p.contagion[i] = len(c.infectious) * (1.0 + p.seasonal_factor * np.sin(tick / 365))  # Add a sinusoidal component to the contagion

    return


def do_transmission(
    p: Population, c: Community, i: int, tick: np.int16, beta: np.float32, exp_mean: np.float32, exp_std: np.float32, pbar
) -> None:
    """Do the transmission."""
    if (contagion := p.contagion[i]) > 0:
        iexposed = c.gmap["exposed"]
        # TODO - iterate over groups to get N (i.e., if we change the groups, this code breaks)
        N = len(c.infants) + len(c.susceptible) + len(c.exposed) + len(c.infectious) + len(c.recovered)
        expose_group(beta, contagion, c.infants, N, tick, exp_mean, exp_std, c, c.gmap["infants"], iexposed, pbar)
        expose_group(beta, contagion, c.susceptible, N, tick, exp_mean, exp_std, c, c.gmap["susceptible"], iexposed, pbar)

    return


def expose_group(beta, contagion, group, N, tick, exp_mean, exp_std, community, isource, idest, pbar):
    """Expose a group of agents."""
    force = beta * contagion * len(group) / N
    num_exposures = np.uint32(np.random.poisson(force))
    pbar.set_description(f"Contagion: {contagion}, Force: {force:.2f}, Exposures: {num_exposures}")
    if num_exposures > 0:
        dmabs = group.dmabs
        susceptibility = group.susceptibility
        etimers = group.etimer
        if num_exposures >= len(group):
            # raise ValueError(f"Too many exposures: {num_exposures} >= {len(group)}")
            print(f"Too many exposures: {num_exposures} >= {len(group)}")
            num_exposures = len(group)
        for _ in range(min(num_exposures, limit := len(group))):
            target = np.random.randint(limit)
            if tick > dmabs[target]:  # infants aren't susceptible until after maternal antibodies wane
                if np.random.uniform() < susceptibility[target]:
                    susceptibility[target] = 0
                    etimers[target] = max(2, int(np.round(np.random.normal(exp_mean, exp_std))) + 1)
                    community.move(isource, target, idest)
                    limit -= 1

    return


RI_CHECKS = 0
RI_DOSES = 0


def do_routine_immunization(_p: Population, c: Community, _i: int, tick: np.int16) -> None:
    """Do routine immunization."""
    dri = c.infants.dri
    dmabs = c.infants.dmabs
    coverage = c.ri_coverage
    susceptibility = c.infants.susceptibility
    iinfants = c.gmap["infants"]
    isusceptible = c.gmap["susceptible"]
    irecovered = c.gmap["recovered"]
    global RI_CHECKS, RI_DOSES
    RI_CHECKS += len(c.infants)
    for index in range(len(c.infants) - 1, -1, -1):  # Iterate over the infants in reverse order
        if tick == dri[index]:  # If today is the agent's routine immunization date
            if tick > dmabs[index]:  # RI vaccine doesn't take until after maternal antibodies wane
                if np.random.uniform() < coverage:
                    susceptibility[index] = 0
                    c.move(isusceptible, index, irecovered)
                    RI_DOSES += 1
                else:
                    c.move(iinfants, index, isusceptible)
            else:
                c.move(iinfants, index, isusceptible)

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
    EXP_MEAN = np.float32(7)  # 7 days
    EXP_STD = np.float32(2)  # 2 days
    INF_MEAN = np.float32(8)  # 8 days
    INF_STD = np.float32(1)  # 1 day
    R_NAUGHT = np.float32(10.0)  # R0
    SEED = np.uint32(20240227)  # random seed
    SEED = np.uint32(datetime.now().microsecond)  # noqa: DTZ005

    parser = ArgumentParser()
    parser.add_argument("-t", "--ticks", type=np.uint32, default=TICKS, help=f"Number of ticks to run the model [{TICKS}]")
    parser.add_argument("-n", "--nodes", type=np.uint32, default=NODES, help=f"Number of nodes to use [{NODES} - 0 for all nodes]")
    parser.add_argument("--exp_mean", type=np.float32, default=EXP_MEAN, help=f"Mean exposure time [{EXP_MEAN}]")
    parser.add_argument("--exp_std", type=np.float32, default=EXP_STD, help=f"Standard deviation of exposure time [{EXP_STD}]")
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN, help=f"Mean infectious time [{INF_MEAN}]")
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD, help=f"Standard deviation of infectious time [{INF_STD}]")
    parser.add_argument(
        "--r_naught", type=np.float32, default=R_NAUGHT, help=f"Basic reproduction number [{R_NAUGHT} - beta will be r_naught / inf_mean]"
    )
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED, help=f"Random seed [{SEED} from datetime.now().microsecond]")

    DEF_K = np.float32(500)
    DEF_A = np.float32(1.0)
    DEF_B = np.float32(1.0)
    DEF_C = np.float32(2.0)
    DEF_MAX_FRAC = np.float32(0.1)

    parser.add_argument("--g_k", type=np.float32, default=DEF_K, help=f"Gravity model k parameter [{DEF_K}]")
    parser.add_argument("--g_a", type=np.float32, default=DEF_A, help=f"Gravity model a parameter [{DEF_A}]")
    parser.add_argument("--g_b", type=np.float32, default=DEF_B, help=f"Gravity model b parameter [{DEF_B}]")
    parser.add_argument("--g_c", type=np.float32, default=DEF_C, help=f"Gravity model c parameter [{DEF_C}]")
    parser.add_argument(
        "--g_max_frac", type=np.float32, default=DEF_MAX_FRAC, help=f"Maximum fraction of outgoing infections [{DEF_MAX_FRAC}]"
    )

    DEF_SEASONALITY = np.float32(0.1)
    parser.add_argument("--seasonality", type=np.float32, default=DEF_SEASONALITY, help=f"Seasonality factor [{DEF_SEASONALITY}]")
    DEF_RI = np.float32(0.75)
    parser.add_argument("--ri_coverage", type=np.float32, default=DEF_RI, help=f"Routine immunization coverage [{DEF_RI}]")

    parser.add_argument("-i", "--init_infs", type=np.float32, default=1.0, help="Initial infections [1.0]")

    parser.add_argument("-p", "--parameters", type=Path, default=None, help="Parameters file [None]")

    args = parser.parse_args()

    if args.parameters is not None:
        with args.parameters.open("r") as file:
            params = json.load(file)
            for key, value in params.items():
                args.__setattr__(key, value)

    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)

    print(f"Deaths: {DEATHS}, Births: {BIRTHS}, Immigrations: {IMMIGRATION}")
    print(f"Routine Immunization Checks: {RI_CHECKS}, Doses: {RI_DOSES}")
