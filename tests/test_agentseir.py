"""Test cases for HomogeneousABC class."""

from argparse import ArgumentParser
from datetime import datetime

import numba as nb
import numpy as np
import polars as pl
from tqdm import tqdm

from idmlaser.community.homogeneous_abc import HomogeneousABC as abc

SEED = np.uint32(20231205)
POP_SIZE = np.uint32(1_000_000)
INIT_INF = np.uint32(10)

_prng = np.random.default_rng(seed=SEED)

R_NAUGHT = np.float32(2.5)
MEAN_EXP = np.float32(4)
STD_EXP = np.float32(1)
MEAN_INF = np.float32(5)
STD_INF = np.float32(1)
BETA = np.float32(R_NAUGHT / MEAN_INF)

TIMESTEPS = np.uint32(720)

def test_seir():

    global BETA

    DOB_TYPE_NP = np.int32
    SUSCEPTIBILITY_TYPE_NP = np.float32
    SUSCEPTIBILITY_TYPE_NB = nb.float32
    ITIMER_TYPE_NP = np.uint8
    ITIMER_TYPE_NB = nb.uint8

    print(f"Creating a well-mixed SEIR community with {POP_SIZE:_} individuals.")
    community = abc(POP_SIZE, **{"beta": BETA})
    community.add_property("dob", dtype=DOB_TYPE_NP, default=0)
    community.add_property("susceptibility", dtype=SUSCEPTIBILITY_TYPE_NP, default=1.0)
    community.add_property("etimer", dtype=ITIMER_TYPE_NP, default=0)
    community.add_property("itimer", dtype=ITIMER_TYPE_NP, default=0)
    community.add_property("age_at_infection", dtype=DOB_TYPE_NP, default=0)
    community.add_property("time_of_infection", dtype=DOB_TYPE_NP, default=0)

    # initialize the dob property to a random value between 0 and 100*365
    community.dob = -_prng.integers(0, 100*365, size=community.count, dtype=DOB_TYPE_NP)

    # # initialize the susceptibility property to a random value between 0.0 and 1.0
    # community.susceptibility = _prng.random_sample(size=community.count)

    # select INIT_INF individuals at random and set their itimer to normal distribution with mean 5 and std 1
    community.itimer[_prng.choice(community.count, size=INIT_INF, replace=False)] = _prng.normal(MEAN_INF, STD_INF, size=INIT_INF).round().astype(ITIMER_TYPE_NP)

    community.susceptibility[community.itimer > 0] = 0.0

    @nb.njit((ITIMER_TYPE_NB[:], nb.uint32), parallel=True)
    def infection_update_inner(itimers, count):
        for i in nb.prange(count):
            if itimers[i] > 0:
                itimers[i] -= 1
        return

    def infection_update(community, _timestep):

        # community.itimer[community.itimer > 0] -= 1
        infection_update_inner(community.itimer, community.count)

        return

    community.add_step(infection_update)

    @nb.njit((        ITIMER_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32), parallel=True)
    def incubation_update_inner(etimers,           itimers,     count):
        for i in nb.prange(count):
            if etimers[i] > 0:
                etimers[i] -= 1
                if etimers[i] == 0:
                    itimers[i] = ITIMER_TYPE_NP(np.round(np.random.normal(MEAN_INF, STD_INF)))

        return

    def incubation_update(community, _timestep):

        # exposed = community.etimer != 0
        # community.etimer[community.etimer > 0] -= 1
        # infectious = exposed & (community.etimer == 0)
        # community.itimer[infectious] = np.round(np.random.normal(MEAN_INF, STD_INF, size=infectious.sum()))
        incubation_update_inner(community.etimer, community.itimer, community.count)

        return

    community.add_step(incubation_update)

    @nb.njit((  SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32, nb.float32), parallel=True)
    def transmission_inner(susceptibility,            etimer,            itimer,     count,       beta):
        contagion = (itimer != 0).sum()
        force = beta * contagion * (1.0 / count)
        for i in nb.prange(count):
            if np.random.random_sample() < (force * susceptibility[i]):
                susceptibility[i] = 0.0
                etimer[i] = ITIMER_TYPE_NP(np.round(np.random.normal(MEAN_EXP, STD_EXP)))

        return

    def transmission(community, _timestep):

        # contagion = sum(community.itimer != 0)
        # force = community.beta * contagion / community.count
        # draws = np.random.random_sample(size=community.count)
        # susceptibility = force * community.susceptibility
        # infected = draws < susceptibility
        # community.susceptibility[infected] = 0.0
        # community.etimer[infected] = np.random.normal(MEAN_EXP, STD_EXP, size=infected.sum()).round().astype(ITIMER_TYPE_NP)

        transmission_inner(community.susceptibility, community.etimer, community.itimer, community.count, community.beta)

        return

    community.add_step(transmission)

    @nb.njit((SUSCEPTIBILITY_TYPE_NB[:], nb.uint32), parallel=True)
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

    # community.add_step(vaccinate)

    def social_distancing(community, timestep):

        if timestep == 30:
            print("implementing social distancing")
            community.beta = 1.2

        return

    # community.add_step(social_distancing)

    results = np.zeros((TIMESTEPS+1, 5), dtype=np.uint32)

    def record(timestep, community, results):

        """Record the state of the community at the current timestep"""

        results[timestep,0] = timestep
        results[timestep,1] = (community.susceptibility > 0.0).sum()
        results[timestep,2] = (community.etimer > 0).sum()
        results[timestep,3] = (community.itimer > 0).sum()
        results[timestep,4] = ((community.susceptibility == 0.0) & (community.etimer == 0) & (community.itimer == 0)).sum()

        return

    record(0, community=community, results=results)

    start = datetime.now()
    for timestep in tqdm(range(TIMESTEPS)):

        community.step(timestep)
        record(timestep+1, community=community, results=results)

    finish = datetime.now()
    print(f"elapsed time: {finish - start}")

    df = pl.DataFrame(data=results, schema=["timestep", "susceptible", "exposed", "infected", "recovered"])
    df.write_csv("seir.csv")

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("--population", type=np.uint32, default=POP_SIZE)
    parser.add_argument("--mean_exp", type=np.float32, default=MEAN_EXP)
    parser.add_argument("--std_exp", type=np.float32, default=STD_EXP)
    parser.add_argument("--mean_inf", type=np.float32, default=MEAN_INF)
    parser.add_argument("--std_inf", type=np.float32, default=STD_INF)
    parser.add_argument("--initial", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)

    args = parser.parse_args()

    TIMESTEPS = args.timesteps
    POP_SIZE = args.population
    MEAN_EXP = args.mean_exp
    STD_EXP = args.std_exp
    MEAN_INF = args.mean_inf
    STD_INF = args.std_inf
    INIT_INF = args.initial
    R_NAUGHT = args.r_naught
    BETA = np.float32(R_NAUGHT / MEAN_INF)

    test_seir()
