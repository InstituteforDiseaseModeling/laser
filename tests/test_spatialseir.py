#! /usr/bin/env python3

"""Test spatial SEIR model."""

from argparse import ArgumentParser
from datetime import datetime

tstart = datetime.now()

import numba as nb
import numpy as np
import polars as pl
from tqdm import tqdm

import nigeria
from idmlaser.community.homogeneous_abc import HomogeneousABC as abc

timport = datetime.now()


def set_params():
    TIMESTEPS = np.uint32(180)
    EXP_MEAN = np.float32(4)
    EXP_STD = np.float32(1)
    INF_MEAN = np.float32(5)
    INF_STD = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)
    SEED = np.uint32(20231205)

    parser = ArgumentParser()
    parser.add_argument("--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("--exp_mean", type=np.float32, default=EXP_MEAN)
    parser.add_argument("--exp_std", type=np.float32, default=EXP_STD)
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD)
    parser.add_argument("--initial_infs", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED)

    args = parser.parse_args()
    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args  # might return vars(args) if we need to return a dictionary


def load_populations():
    # filter out the state and national level LGAs
    lgas = {k: v for k, v in nigeria.lgas.items() if len(k.split(":")) == 5}
    # from dictionary with key and values ((population, year), (lat, lon), area) extract population sizes
    POP_INDEX = 0  # first item in value tuple is population info tuple
    SIZE_INDEX = 0  # first item in population info tuple is population size
    pops = np.array([v[POP_INDEX][SIZE_INDEX] for v in lgas.values()], dtype=np.uint32)

    return pops


def load_network():
    gravity = np.array(nigeria.gravity, dtype=np.float32)

    return gravity


def test_spatial_seir(params):
    DOB_TYPE_NP = np.int32  # measured in days anchored at t=0, so all initial DoBs are negative
    SUSCEPTIBILITY_TYPE_NP = np.uint8  # currently just 1|0
    SUSCEPTIBILITY_TYPE_NB = nb.uint8
    ITIMER_TYPE_NP = np.uint8  # don't need more than 255 days of incubation or infectiousness
    ITIMER_TYPE_NB = nb.uint8

    pops = load_populations()
    pop_size = pops.sum()
    network = load_network()

    print(f"Creating a spatial SEIR model with {pop_size:_} individuals.")
    community = abc(
        pop_size, beta=params.beta, exp_mean=params.exp_mean, exp_std=params.exp_std, inf_mean=params.inf_mean, inf_std=params.inf_std
    )
    community.add_property("dob", dtype=DOB_TYPE_NP, default=0)
    community.add_property("susceptibility", dtype=SUSCEPTIBILITY_TYPE_NP, default=1.0)
    community.add_property("etimer", dtype=ITIMER_TYPE_NP, default=0)
    community.add_property("itimer", dtype=ITIMER_TYPE_NP, default=0)
    community.add_property("age_at_infection", dtype=DOB_TYPE_NP, default=0)
    community.add_property("time_of_infection", dtype=DOB_TYPE_NP, default=0)
    community.add_property("nodeid", dtype=np.uint16, default=0)

    # iterate through pops setting nodeid = i for next pops[i] individuals
    nodeidx = 0
    for i, pop in enumerate(pops):
        community.nodeid[nodeidx : nodeidx + pop] = i  # assign to pop # individuals
        nodeidx += pop  # increment index by pop #

    _prng = np.random.default_rng(seed=params.seed)

    # initialize the dob property to a random (negative) value between 0 and 100*365
    # I.e., everyone was born some number of days before the start of the simulation
    community.dob = -_prng.integers(0, 100 * 365, size=community.count, dtype=DOB_TYPE_NP)

    # select INIT_INF individuals at random and set their itimer to normal distribution with mean `inf_mean` and std `inf_std`
    # consider including +1 since the first action in processing is to decrement the infection timer
    initial = _prng.choice(community.count, size=params.initial_infs, replace=False)
    community.itimer[initial] = _prng.normal(params.inf_mean, params.inf_std, size=params.initial_infs).round().astype(ITIMER_TYPE_NP)
    community.susceptibility[initial] = 0

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

    @nb.njit(
        (SUSCEPTIBILITY_TYPE_NB[:], nb.uint16[:], nb.float32[:], ITIMER_TYPE_NB[:], nb.uint32, nb.float32, nb.float32),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def tx_inner(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std):
        for i in nb.prange(count):
            force = susceptibilities[i] * forces[nodeids[i]]  # force of infection attenuated by personal susceptibility
            if np.random.random_sample() < force:  # draw random number < force means infection
                susceptibilities[i] = 0.0  # set susceptibility to 0.0
                # set exposure timer for newly infected individuals to a draw from a normal distribution
                etimers[i] = ITIMER_TYPE_NP(np.round(np.random.normal(exp_mean, exp_std)))
        return

    # pre-allocate these rather than allocating new array on each timestep
    forces = np.zeros(len(pops), dtype=np.float32)
    report = np.zeros((params.timesteps, len(pops)), dtype=np.uint32)

    def transmission(community, timestep):
        contagion = np.zeros(len(pops), dtype=np.uint32)
        np.add.at(contagion, community.nodeid[community.itimer != 0], 1)  # accumulate contagion by node, 1 unit per infected individual

        transfer = (contagion * network).round().astype(np.uint32)  # contagion * network = transfer
        contagion += transfer.sum(axis=1)  # increment by incoming "migration"
        contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

        report[timestep, :] = contagion  # contagion is a proxy for # of infected individual/prevalence

        np.multiply(contagion, community.beta, out=forces)  # pre-multiply by beta (scalar now, could be array)
        np.divide(forces, pops, out=forces)  # divide by population (forces is now per-capita)

        tx_inner(
            community.susceptibility, community.nodeid, forces, community.etimer, community.count, community.exp_mean, community.exp_std
        )

        return

    community.add_step(transmission)

    # @nb.njit((SUSCEPTIBILITY_TYPE_NB[:], nb.uint32), parallel=True, nogil=True, cache=True)
    # def vaccinate_inner(susceptibility, count):
    #     for i in nb.prange(count):
    #         if np.random.binomial(1, 0.6) == 1:
    #             susceptibility[i] = 0.0
    #     return

    # def vaccinate(community, timestep):

    #     if timestep == 30:
    #         # do a binomial draw with probability 0.6 and set the susceptibility to 0.0 for those individuals
    #         # community.susceptibility[np.random.binomial(1, 0.6, size=community.count, dtype=np.bool)] = 0.0
    #         vaccinate_inner(community.susceptibility, community.count)

    #     return

    # community.add_step(vaccinate)

    # Add one here for room to capture the initial state
    results = np.zeros((params.timesteps + 1, 5), dtype=np.uint32)

    @nb.njit(
        (nb.uint32, SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32[:, :]), parallel=True, nogil=True, cache=True
    )
    def record(timestep, susceptibility, etimer, itimer, results):
        # def record(timestep, community, results):

        """Record the state of the community at the current timestep"""

        results[timestep, 0] = timestep
        results[timestep, 1] = (susceptibility > 0.0).sum()
        results[timestep, 2] = (etimer > 0).sum()
        results[timestep, 3] = (itimer > 0).sum()
        results[timestep, 4] = ((susceptibility == 0.0) & (etimer == 0) & (itimer == 0)).sum()

        return

    # record initial state, state _after_ timestep i processing will be in index i+1
    record(0, community.susceptibility, community.etimer, community.itimer, results)

    start = datetime.now()
    for timestep in tqdm(range(params.timesteps)):
        community.step(timestep)
        record(timestep + 1, community.susceptibility, community.etimer, community.itimer, results)

    finish = datetime.now()
    print(f"elapsed time: {finish - start}")

    df = pl.DataFrame(data=results, schema=["timestep", "susceptible", "exposed", "infected", "recovered"])
    df.write_csv("spatial_seir.csv")
    print("Wrote SEIR channels to 'spatial_seir.csv'.")
    sdf = pl.DataFrame(data=report, schema=[f"node{i}" for i in range(len(pops))])
    sdf.write_csv("spatial_seir_report.csv")
    print("Wrote spatial report to 'spatial_seir_report.csv'.")

    return


if __name__ == "__main__":
    params = set_params()
    test_spatial_seir(params)

    tfinish = datetime.now()
    print(f"import time:    {timport - tstart}")
    print(f"execution time: {tfinish - timport}")
    print(f"total time:     {tfinish - tstart}")
