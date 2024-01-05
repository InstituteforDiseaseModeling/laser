#!/usr/bin/env python3

"""Test cases for HomogeneousABC class."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

tstart = datetime.now()

import numba as nb
import numpy as np
import polars as pl
from tqdm import tqdm

from idmlaser.community.homogeneous_abc import HomogeneousABC as abc

timport = datetime.now()


def set_params():
    TIMESTEPS = np.uint32(128)
    POP_SIZE = np.uint32(200_000_000)
    INF_MEAN = np.float32(5)
    INF_STD = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)

    SEED = np.uint32(20231205)

    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("-p", "--pop_size", type=np.uint32, default=POP_SIZE)
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD)
    parser.add_argument("--initial_inf", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument(
        "--brute_force", action="store_false", dest="poisson", help="use brute force tx instead of Poisson draw for potential targets"
    )
    parser.add_argument("-v", "--vaccinate", action="store_true")
    parser.add_argument("--seed", type=np.uint32, default=SEED)
    parser.add_argument("-f", "--filename", type=Path, default=Path(__file__).parent / "sir.csv")

    args = parser.parse_args()
    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args  # might use vars(args) here if we need to return a dictionary


def test_sir(params):
    # Let's define some types here so what we allocate in Numpy has known analogs in Numba

    DOB_TYPE_NP = np.int32  # measured in days anchored at t=0, so all initial DoBs are negative
    SUSCEPTIBILITY_TYPE_NP = np.uint8  # currently just 1|0
    SUSCEPTIBILITY_TYPE_NB = nb.uint8
    ITIMER_TYPE_NP = np.uint8  # don't need more than 255 days of infectiousness at this point
    ITIMER_TYPE_NB = nb.uint8

    print(f"Creating a well-mixed SIR community with {params.pop_size:_} individuals.")
    community = abc(
        params.pop_size,
        beta=params.beta,  # add the beta property to the community for use later
        mean=params.inf_mean,  # add the mean property to the community
        std=params.inf_std,  # add the std property to the community
    )  # add the std property to the community
    community.add_property("dob", dtype=DOB_TYPE_NP, default=0)  # we will initialize DoB below
    community.add_property("susceptibility", dtype=SUSCEPTIBILITY_TYPE_NP, default=1)  # initially, everyone is susceptible
    community.add_property("itimer", dtype=ITIMER_TYPE_NP, default=0)  # timer > 0 indicates infection, initially 0

    # This doesn't quite do what we would expect since Numba uses its own PRNG
    # and I don't know how it is seeded.
    prng = np.random.default_rng(seed=params.seed)

    # initialize the dob property to a random (negative) value between 0 and 100*365
    # I.e., everyone was born some number of days before the start of the simulation
    community.dob = -prng.integers(0, 100 * 365, size=community.count, dtype=DOB_TYPE_NP)

    # select INIT_INF individuals at random and set their itimer to normal distribution with mean 5 and std 1
    # consider including +1 since the first action in processing is to decrement the infection timer
    initial_infs = prng.choice(community.count, size=params.initial_inf, replace=False)
    # prng.normal() _could_ draw negative numbers, consider making the minimum initial timer 1 day
    community.itimer[initial_infs] = prng.normal(params.inf_mean, params.inf_std, size=params.initial_inf).round().astype(ITIMER_TYPE_NP)
    community.susceptibility[initial_infs] = 0

    # We do the dance seen below a couple of times because Numba doesn't know what it can
    # do with Python classes, e.g. community. So we use Numba on an inner loop with the
    # argument types explicitly passed and call the inner loop from the more general step function.

    # Note that we use the _NB (Numba) type versions here vs. the _NP (Numpy) type versions above.

    @nb.njit((ITIMER_TYPE_NB[:], nb.uint32), parallel=True, nogil=True, cache=True)
    def infection_update_inner(timers, count):
        for i in nb.prange(count):
            if timers[i] > 0:
                timers[i] -= 1
        return

    def infection_update(community, _timestep):
        # community.itimer[community.itimer > 0] -= 1
        infection_update_inner(community.itimer, community.count)

        return

    community.add_step(infection_update)

    if params.poisson:
        print("Using Poisson draw for potential targets.")

        @nb.njit(
            (SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.int64[:], nb.uint32, nb.float32, nb.float32),
            parallel=True,
            nogil=True,
            cache=True,
        )
        def transmission_inner(susceptibility, itimer, targets, actual, inf_mean, inf_std):
            for i in nb.prange(actual):
                target = targets[i]
                if np.random.random_sample() < susceptibility[target]:
                    susceptibility[target] = 0
                    itimer[target] = ITIMER_TYPE_NP(np.round(np.random.normal(inf_mean, inf_std)))

            return

        def transmission(community, _timestep):
            contagion = (community.itimer != 0).sum()
            expected = community.beta * contagion
            actual = np.uint32(np.random.poisson(expected))
            targets = np.random.choice(community.count, size=actual, replace=True)

            transmission_inner(community.susceptibility, community.itimer, targets, actual, community.mean, community.std)

            return

    else:
        print("Using brute force for potential targets.")

        @nb.njit(
            (SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32, nb.float32, nb.float32, nb.float32),
            parallel=True,
            nogil=True,
            cache=True,
        )
        def transmission_inner(susceptibility, itimer, count, beta, inf_mean, inf_std):
            contagion = (itimer != 0).sum()
            force = beta * contagion * (1.0 / count)
            for i in nb.prange(count):
                if np.random.random_sample() < (force * susceptibility[i]):
                    susceptibility[i] = 0
                    itimer[i] = ITIMER_TYPE_NP(np.round(np.random.normal(inf_mean, inf_std)))

            return

        def transmission(community, _timestep):
            transmission_inner(community.susceptibility, community.itimer, community.count, community.beta, community.mean, community.std)

            return

    community.add_step(transmission)

    if params.vaccinate:

        @nb.njit((SUSCEPTIBILITY_TYPE_NB[:], nb.uint32), parallel=True, nogil=True, cache=True)
        def vaccinate_inner(susceptibility, count):
            for i in nb.prange(count):
                if np.random.binomial(1, 0.6) == 1:
                    susceptibility[i] = 0.0
            return

        def vaccinate(community, timestep):
            if timestep == 30:
                # do a binomial draw with probability 0.6 and set the susceptibility to 0.0 for those individuals
                # community.susceptibility[np.random.binomial(1, 0.6, size=community.count, dtype=np.bool)] = 0.0
                vaccinate_inner(community.susceptibility, community.count)

            return

        community.add_step(vaccinate)

    # Add one here for room to capture the initial state
    results = np.zeros((params.timesteps + 1, 4), dtype=np.uint32)

    # def record(timestep, community, results):
    @nb.njit((nb.uint32, SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32[:, :]), parallel=True, nogil=True, cache=True)
    def record(timestep, susceptibility, itimer, results):
        """Record the state of the community at the current timestep"""

        results[timestep, 0] = timestep
        # results[timestep,1] = (community.susceptibility > 0).sum()
        # results[timestep,2] = (community.itimer > 0).sum()
        # results[timestep,3] = ((community.susceptibility == 0) & (community.itimer == 0)).sum()
        results[timestep, 1] = (susceptibility > 0).sum()
        results[timestep, 2] = (itimer > 0).sum()
        results[timestep, 3] = ((susceptibility == 0) & (itimer == 0)).sum()

        return

    record(0, community.susceptibility, community.itimer, results)  # record(0, community=community, results=results)

    start = datetime.now()
    for timestep in tqdm(range(params.timesteps)):
        community.step(timestep)
        record(
            timestep + 1, community.susceptibility, community.itimer, results
        )  # record(timestep+1, community=community, results=results)

    finish = datetime.now()
    print(f"elapsed time: {finish - start}")

    # This appears, with some testing, to be quite efficient - the Polars DataFrame wraps
    # the existing Numpy arrays rather than copying them.
    df = pl.DataFrame(data=results, schema=["timestep", "susceptible", "infected", "recovered"])
    df.write_csv(params.filename)
    print(f"Results written to '{params.filename}'.")

    return


if __name__ == "__main__":
    params = set_params()
    test_sir(params)

    tfinish = datetime.now()
    print(f"import time:    {timport - tstart}")
    print(f"execution time: {tfinish - timport}")
    print(f"elapsed time:   {tfinish - tstart}")
