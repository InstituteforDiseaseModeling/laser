#!/usr/bin/env python3

"""Test cases for HomogeneousABC class."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

tzero = datetime.now()

import numba as nb
import numpy as np
import polars as pl
from tqdm import tqdm

from idmlaser.community.homogeneous_abc import HomogeneousABC as abc

timport = datetime.now()


def set_params():
    TIMESTEPS = np.uint32(128)
    POP_SIZE = np.uint32(200_000_000)
    MEAN_EXP = np.float32(4)
    STD_EXP = np.float32(1)
    MEAN_INF = np.float32(5)
    STD_INF = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)
    SEED = np.uint32(20231205)

    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("-p", "--pop_size", type=np.uint32, default=POP_SIZE)
    parser.add_argument("--exp_mean", type=np.float32, default=MEAN_EXP)
    parser.add_argument("--exp_std", type=np.float32, default=STD_EXP)
    parser.add_argument("--inf_mean", type=np.float32, default=MEAN_INF)
    parser.add_argument("--inf_std", type=np.float32, default=STD_INF)
    parser.add_argument("--initial_inf", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument(
        "--brute_force", action="store_false", dest="poisson", help="use brute force tx instead of Poisson draw for potential targets"
    )
    parser.add_argument("-v", "--vaccinate", action="store_true")
    parser.add_argument("-m", "--masking", action="store_true")
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED)
    parser.add_argument("-f", "--filename", type=Path, default=Path(__file__).parent / "seir.csv")

    args = parser.parse_args()
    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args  # might return vars(args) if we need to return a dictionary


def test_seir(params):
    DOB_TYPE_NP = np.int32  #            measured in days anchored at t=0, so all initial DoBs are negative
    SUSCEPTIBILITY_TYPE_NP = np.uint8  # currently just 1|0
    SUSCEPTIBILITY_TYPE_NB = nb.uint8
    ITIMER_TYPE_NP = np.uint8  #         don't need more than 255 days of incubation or infectiousness
    ITIMER_TYPE_NB = nb.uint8

    print(f"Creating a well-mixed SEIR community with {params.pop_size:_} individuals.")
    community = abc(
        params.pop_size,
        beta=params.beta,
        exp_mean=params.exp_mean,
        exp_std=params.exp_std,
        inf_mean=params.inf_mean,
        inf_std=params.inf_std,
    )
    community.add_property("dob", dtype=DOB_TYPE_NP, default=0)
    community.add_property("susceptibility", dtype=SUSCEPTIBILITY_TYPE_NP, default=1)
    community.add_property("etimer", dtype=ITIMER_TYPE_NP, default=0)
    community.add_property("itimer", dtype=ITIMER_TYPE_NP, default=0)
    # We're not using these, yet.
    # community.add_property("age_at_infection", dtype=DOB_TYPE_NP, default=0)
    # community.add_property("time_of_infection", dtype=DOB_TYPE_NP, default=0)

    _prng = np.random.default_rng(seed=params.seed)

    # initialize the dob property to a random (negative) value between 0 and 100*365
    # I.e., everyone was born some number of days before the start of the simulation
    community.dob = -_prng.integers(0, 100 * 365, size=community.count, dtype=DOB_TYPE_NP)

    # select INIT_INF individuals at random and set their itimer to normal distribution with mean 5 and std 1
    # consider including +1 since the first action in processing is to decrement the infection timer
    initial_infs = _prng.choice(community.count, size=params.initial_inf, replace=False)
    community.itimer[initial_infs] = _prng.normal(params.inf_mean, params.inf_std, size=params.initial_inf).round().astype(ITIMER_TYPE_NP)
    community.susceptibility[initial_infs] = 0

    # We do the dance seen below a couple of times because Numba doesn't know what it can
    # do with Python classes, e.g. community. So we use Numba on an inner loop with the
    # argument types explicitly passed and call the inner loop from the more general step function.

    # Note that we use the _NB (Numba) type versions here vs. the _NP (Numpy) type versions above.

    @nb.njit((ITIMER_TYPE_NB[:], nb.uint32), parallel=True, nogil=True, cache=True)
    def infection_update_inner(itimers, count):
        for i in nb.prange(count):
            if itimers[i] > 0:
                itimers[i] -= 1
        return

    def infection_update(community, _timestep):
        # community.itimer[community.itimer > 0] -= 1
        infection_update_inner(community.itimer, community.count)

        return

    # Note: we add the infection_update step _before_ the incubation update step.
    # Otherwise, we would set the infectiouness timer at the end of incubation and
    # immediately decrement it by one day.
    community.add_step(infection_update)

    @nb.njit((ITIMER_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32, nb.float32, nb.float32), parallel=True, nogil=True, cache=True)
    def incubation_update_inner(etimers, itimers, count, inf_mean, inf_std):
        for i in nb.prange(count):
            if etimers[i] > 0:  # if you have an active exposure timer...
                etimers[i] -= 1  # ...decrement it
                if etimers[i] == 0:  # if it has reached 0...
                    # set your infection timer to a draw from a normal distribution
                    itimers[i] = ITIMER_TYPE_NP(np.round(np.random.normal(inf_mean, inf_std)))

        return

    def incubation_update(community, _timestep):
        # exposed = community.etimer != 0
        # community.etimer[community.etimer > 0] -= 1
        # infectious = exposed & (community.etimer == 0)
        # community.itimer[infectious] = np.round(np.random.normal(MEAN_INF, STD_INF, size=infectious.sum()))
        incubation_update_inner(community.etimer, community.itimer, community.count, community.inf_mean, community.inf_std)

        return

    # See above about adding this step _after_ the infection_update.
    community.add_step(incubation_update)

    if params.poisson:
        print("Using Poisson draw for transmission.")

        @nb.njit(
            (SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.int64[:], nb.uint32, nb.float32, nb.float32),
            parallel=True,
            nogil=True,
            cache=True,
        )
        def transmission_inner(susceptibility, etimer, targets, actual, exp_mean, exp_std):
            for i in nb.prange(actual):  # pylint: disable=E1133
                target = targets[i]
                if np.random.random_sample() < susceptibility[target]:
                    susceptibility[target] = 0
                    etimer[target] = ITIMER_TYPE_NP(np.round(np.random.normal(exp_mean, exp_std)))

            return

        def transmission(community, _timestep):
            contagion = (community.itimer != 0).sum()
            expected = community.beta * contagion
            actual = np.uint32(np.random.poisson(expected))
            targets = np.random.choice(community.count, size=actual, replace=True)

            transmission_inner(
                community.susceptibility,
                community.etimer,
                targets,
                actual,
                community.exp_mean,
                community.exp_std,
            )

            return

    else:
        print("Using brute force iteration for transmission.")

        @nb.njit(
            (SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32, nb.float32, nb.float32, nb.float32),
            parallel=True,
            nogil=True,
            cache=True,
        )
        def transmission_inner(susceptibility, etimer, itimer, count, beta, exp_mean, exp_std):
            contagion = (itimer != 0).sum()
            force = beta * contagion * (1.0 / count)
            for i in nb.prange(count):
                if np.random.random_sample() < (force * susceptibility[i]):
                    susceptibility[i] = 0
                    etimer[i] = ITIMER_TYPE_NP(np.round(np.random.normal(exp_mean, exp_std)))

            return

        def transmission(community, _timestep):
            transmission_inner(
                community.susceptibility,
                community.etimer,
                community.itimer,
                community.count,
                community.beta,
                community.exp_mean,
                community.exp_std,
            )

            return

    community.add_step(transmission)

    if params.vaccinate:

        @nb.njit((SUSCEPTIBILITY_TYPE_NB[:], nb.uint32), parallel=True, nogil=True, cache=True)
        def vaccinate_inner(susceptibility, count):
            for i in nb.prange(count):
                if np.random.binomial(1, 0.6) == 1:
                    susceptibility[i] = 0
            return

        def vaccinate(community, timestep):
            if timestep == 30:
                # do a binomial draw with probability 0.6 and set the susceptibility to 0.0 for those individuals
                # community.susceptibility[np.random.binomial(1, 0.6, size=community.count, dtype=np.bool)] = 0.0
                vaccinate_inner(community.susceptibility, community.count)

            return

        community.add_step(vaccinate)

    if params.masking:

        def social_distancing(community, timestep):
            if timestep == 30:
                print("implementing social distancing")
                community.beta = 1.2

            return

        community.add_step(social_distancing)

    # Add one here for room to capture the initial state
    results = np.zeros((params.timesteps + 1, 5), dtype=np.uint32)

    # def record(timestep, community, results):
    @nb.njit(
        (nb.uint32, SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32[:, :]), parallel=True, nogil=True, cache=True
    )
    def record(timestep, susceptibility, etimer, itimer, results):
        """Record the state of the community at the current timestep"""

        results[timestep, 0] = timestep
        results[timestep, 1] = (susceptibility > 0).sum()
        results[timestep, 2] = (etimer > 0).sum()
        results[timestep, 3] = (itimer > 0).sum()
        results[timestep, 4] = ((susceptibility == 0) & (etimer == 0) & (itimer == 0)).sum()

        return

    record(0, community.susceptibility, community.etimer, community.itimer, results)  # record(0, community=community, results=results)

    start = datetime.now()
    for timestep in tqdm(range(params.timesteps)):
        community.step(timestep)
        record(
            timestep + 1, community.susceptibility, community.etimer, community.itimer, results
        )  # record(timestep+1, community=community, results=results)

    finish = datetime.now()
    print(f"elapsed time: {finish - start}")

    # This appears, with some testing, to be quite efficient - the Polars DataFrame wraps
    # the existing Numpy arrays rather than copying them.
    df = pl.DataFrame(data=results, schema=["timestep", "susceptible", "exposed", "infected", "recovered"])
    df.write_csv(params.filename)
    print(f"Results written to '{params.filename}'.")

    return


if __name__ == "__main__":
    params = set_params()
    test_seir(params)

    tfinish = datetime.now()
    print(f"import time:    {timport - tzero}")
    print(f"execution time: {tfinish - timport}")
    print(f"total time:     {tfinish - tzero}")
