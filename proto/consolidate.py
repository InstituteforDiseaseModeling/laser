#! /usr/bin/env python3

"""Experiment with moving agents between segments of NumPy arrays."""

from argparse import ArgumentParser
from datetime import datetime
from datetime import timezone
from pathlib import Path

tzero = datetime.now(timezone.utc)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from groupedcommunity import Community
from tqdm import tqdm  # noqa: E402

timport = datetime.now(timezone.utc)


def get_params():
    """Get the parameters for the simulation. Start with defaults and override from commandline."""
    TIMESTEPS = np.uint32(730)
    POP_SIZE = np.uint32(10_000)
    MEAN_EXP = np.float32(4)
    STD_EXP = np.float32(1)
    MEAN_INF = np.float32(5)
    STD_INF = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)
    SEED = np.uint32(datetime.now(timezone.utc).microsecond)  # np.uint32(20231205)
    BIRTH_RATE = np.float32(0.03)
    NON_DISEASE_MORTALITY = np.float32(1.0 / 80)

    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("-p", "--pop_size", type=np.uint32, default=POP_SIZE)
    parser.add_argument("--exp_mean", type=np.float32, default=MEAN_EXP)
    parser.add_argument("--exp_std", type=np.float32, default=STD_EXP)
    parser.add_argument("--inf_mean", type=np.float32, default=MEAN_INF)
    parser.add_argument("--inf_std", type=np.float32, default=STD_INF)
    parser.add_argument("--initial_inf", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument("--birth_rate", dest="mu", type=np.float32, default=BIRTH_RATE)
    parser.add_argument("--mortality", dest="nu", type=np.float32, default=NON_DISEASE_MORTALITY)
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED)
    parser.add_argument("-f", "--filename", type=Path, default=Path(__file__).parent / "groupseir.csv")

    args = parser.parse_args()
    args.beta = np.float32(args.r_naught / args.inf_mean)

    return args


def run_sim(params):
    """Run the simulation."""

    print(f"Running simulation with parameters: {params}")
    np.random.seed(params.seed)

    # initialize the community
    c = Community()
    daily_birth_rate = params.mu / 365
    daily_mortality = params.nu / 365
    num_unborn = (np.power(1 + daily_birth_rate, params.timesteps) - 1) * params.pop_size

    FUDGE = [10.0, 4.0, 1.5, 1.2, 1.05, 1.02, 1.005, 1.002, 1.0005, 1.0002]
    index = min(int(np.log10(num_unborn)), len(FUDGE) - 1)

    num_unborn *= FUDGE[index]  # fudge factor
    num_unborn = np.uint32(np.round(num_unborn))
    c.add_agent_group("unborn", num_unborn)

    S = np.uint32(np.round(params.pop_size / params.r_naught))  # * 1.0625
    I = params.initial_inf
    R = params.pop_size - S - I

    c.add_agent_group("susceptible", S)  # c.add_agent_group("susceptible", params.pop_size - params.initial_inf)
    c.add_agent_group("exposed", 0)
    c.add_agent_group("infectious", I)  # c.add_agent_group("infectious", params.initial_inf)
    c.add_agent_group("recovered", R)  # c.add_agent_group("recovered", 0)
    c.add_agent_group("deceased", 0)

    c.add_agent_property("dob", np.int16, 0)  # support up to ~90 year olds at start of simulation and ~90 years of simulation
    c.add_agent_property("susceptibility", np.uint8, 0)
    c.add_agent_property("etimer", np.uint8, 0)
    c.add_agent_property("itimer", np.uint8, 0)
    c.add_agent_property("uid", np.uint32, 0)

    c.allocate()

    c.unborn.dob = -1
    c.susceptible.dob = np.random.randint(-80 * 365, 1, size=len(c.susceptible))
    c.susceptible.susceptibility = 1
    c.infectious.itimer = np.random.normal(params.inf_mean, params.inf_std, size=len(c.infectious)) + 1

    # 10 slots - timestep + elapsed time + USEIRD + births + deaths
    results = np.zeros((params.timesteps + 1, 10), dtype=np.uint32)
    results[0, :] = [0, 0, len(c.unborn), len(c.susceptible), len(c.exposed), len(c.infectious), len(c.recovered), len(c.deceased), 0, 0]

    start = datetime.now(timezone.utc)

    for t in tqdm(range(params.timesteps)):
        top = datetime.now(timezone.utc)

        # 1 U - deliver babies
        N = deliver_babies(c, daily_birth_rate, t)

        # 2 I - update infectious agents
        update_infections(c)

        # 3 E - update exposed agents
        update_exposures(c, params.inf_mean, params.inf_std)

        # 4 S - transmit to susceptible agents
        do_transmission(c, params.beta, N, params.exp_mean, params.exp_std)

        # 5 R - memorialize deceased agents
        do_interments(c, daily_mortality)

        # 6 D - no action
        # report results
        bottom = datetime.now(timezone.utc)
        elapsed = round((bottom - top).total_seconds() * 1000)
        results[t + 1, :] = [
            t + 1,
            elapsed,
            len(c.unborn),
            len(c.susceptible),
            len(c.exposed),
            len(c.infectious),
            len(c.recovered),
            len(c.deceased),
            results[t, 2] - len(c.unborn),  # Previous # unborn - current # unborn
            len(c.deceased) - results[t, 7],  # Current # deceased - previous # deceased
        ]

    finish = datetime.now(timezone.utc)
    print(f"elapsed time: {finish - start}")

    # This appears, with some testing, to be quite efficient - the Polars DataFrame wraps
    # the existing Numpy arrays rather than copying them.
    df = pl.DataFrame(
        data=results,
        schema=["timestep", "elapsed", "unborn", "susceptible", "exposed", "infectious", "recovered", "deceased", "births", "deaths"],
    )
    df.write_csv(params.filename)
    print(f"Results written to '{params.filename}'.")

    f = plt.figure(figsize=(16, 12), dpi=300)
    ax1 = f.add_subplot()

    pandasdf = pd.read_csv(params.filename)

    pandasdf.plot("timestep", ["susceptible", "recovered"], title="GroupedSEIR", ax=ax1, color=["blue", "green"])
    ax2 = ax1.twinx()
    pandasdf.plot("timestep", ["exposed", "infectious"], ax=ax2, color=["orange", "red"])
    f.tight_layout()
    f.savefig(params.filename.stem + ".png")

    print(f"Plot written to '{params.filename.stem}.png'.")

    return


def deliver_babies(c, daily_birth_rate, t):
    """Deliver babies."""
    N = len(c.susceptible) + len(c.exposed) + len(c.infectious) + len(c.recovered)
    births = np.uint32(np.round(np.random.poisson(daily_birth_rate * N)))
    iunborn = c.gmap["unborn"]
    isusceptible = c.gmap["susceptible"]
    for _ in range(min(births, index := len(c.unborn))):
        # We will move the last of the unborn to the first of the susceptibles. It is more efficient (saves a copy).
        index -= 1
        c.unborn.dob[index] = t
        c.unborn.susceptibility[index] = 1
        c.move(iunborn, index, isusceptible)

    N += births
    return N


def update_infections(c):
    """Update the infectious agents."""
    itimers = c.infectious.itimer
    iinfectious = c.gmap["infectious"]
    irecovered = c.gmap["recovered"]
    for index in range(len(c.infectious) - 1, -1, -1):
        timer = itimers[index] - 1
        itimers[index] = timer
        if timer == 0:
            c.move(iinfectious, index, irecovered)

    return


def update_exposures(c, inf_mean, inf_std):
    """Update the exposed agents."""
    etimers = c.exposed.etimer
    itimers = c.exposed.itimer
    iexposed = c.gmap["exposed"]
    iinfectious = c.gmap["infectious"]
    for index in range(len(c.exposed) - 1, -1, -1):
        timer = etimers[index] - 1
        etimers[index] = timer
        if timer == 0:
            itimers[index] = max(1, int(np.round(np.random.normal(inf_mean, inf_std))))
            c.move(iexposed, index, iinfectious)

    return


def do_transmission(c, beta, N, exp_mean, exp_std):
    """Do the transmission."""
    susceptibility = c.susceptible.susceptibility
    etimers = c.susceptible.etimer
    isusceptible = c.gmap["susceptible"]
    iexposed = c.gmap["exposed"]
    force = beta * len(c.infectious) * len(c.susceptible) / N
    num_exposures = np.uint32(np.round(np.random.poisson(force)))
    for _ in range(min(num_exposures, limit := len(c.susceptible))):
        target = np.random.randint(limit)
        if np.random.uniform() < susceptibility[target]:
            susceptibility[target] = 0
            etimers[target] = max(1, int(np.round(np.random.normal(exp_mean, exp_std))))
            c.move(isusceptible, target, iexposed)
            limit -= 1

    return


def do_interments(c, daily_mortality):
    """Do the interments (non-disease mortality)."""
    num_deaths = np.round(
        np.random.poisson(np.array([len(c.susceptible), len(c.exposed), len(c.infectious), len(c.recovered)]) * daily_mortality)
    ).astype(np.uint32)

    isusceptible = c.isusceptible
    iexposed = c.iexposed
    iinfectious = c.iinfectious
    irecovered = c.irecovered
    ideceased = c.ideceased

    # Susceptible -> Deceased
    for _ in range(min(num_deaths[0], limit := len(c.susceptible))):
        target = np.random.randint(limit)
        c.move(isusceptible, target, ideceased)
        limit -= 1

    # Exposed -> Deceased
    for _ in range(min(num_deaths[1], limit := len(c.exposed))):
        target = np.random.randint(limit)
        c.move(iexposed, target, ideceased)
        limit -= 1

    # Infectious -> Deceased
    for _ in range(min(num_deaths[2], limit := len(c.infectious))):
        target = np.random.randint(limit)
        c.move(iinfectious, target, ideceased)
        limit -= 1

    # Recovered -> Deceased
    for _ in range(min(num_deaths[3], limit := len(c.recovered))):
        target = np.random.randint(limit)
        c.move(irecovered, target, ideceased)
        limit -= 1

    return


if __name__ == "__main__":
    parameters = get_params()
    run_sim(parameters)

    tfinish = datetime.now(timezone.utc)
    print(f"import time:    {timport - tzero}")
    print(f"execution time: {tfinish - timport}")
    print(f"total time:     {tfinish - tzero}")
