#! /usr/bin/env python3

"""Experiment with moving agents between segments of NumPy arrays."""

from argparse import ArgumentParser
from datetime import datetime
from datetime import timezone
from pathlib import Path

tzero = datetime.now(timezone.utc)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from tqdm import tqdm  # noqa: E402

timport = datetime.now(timezone.utc)


def set_params():
    """Set the parameters for the simulation. Start with defaults and override from commandline."""
    TIMESTEPS = np.uint32(730)
    POP_SIZE = np.uint32(10_000)
    MEAN_EXP = np.float32(4)
    STD_EXP = np.float32(1)
    MEAN_INF = np.float32(5)
    STD_INF = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)
    SEED = np.uint32(20231205)
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
    # parser.add_argument(
    #     "--brute_force", action="store_false", dest="poisson", help="use brute force tx instead of Poisson draw for potential targets"
    # )
    # parser.add_argument("-v", "--vaccinate", action="store_true")
    # parser.add_argument("-m", "--masking", action="store_true")
    parser.add_argument("-s", "--seed", type=np.uint32, default=SEED)
    parser.add_argument("-f", "--filename", type=Path, default=Path(__file__).parent / "groupseir.csv")

    args = parser.parse_args()
    args.beta = np.float32(args.r_naught / args.inf_mean)

    return args


_FIRST = 0
_LAST = 1


class Subset:
    """A subset of agents."""

    def __init__(self, group, item) -> None:
        self.group = group
        self.parent = group.parent
        self.index = group.index
        self.item = item
        return

    def __getattr__(self, name):
        first, last = self.parent.igroups[self.index, :]
        return getattr(self.parent, f"_{name}")[first : last + 1][self.item]


class Group:
    """A (virtual) group of agents."""

    def __init__(self, name, parent, index) -> None:
        self.name = name
        self.parent = parent
        self.index = index
        return

    def __getitem__(self, index):
        # first, last = self.parent.igroups[self.index, :]
        # return getattr(self.parent, f"_{index}")[first : last + 1]
        return Subset(self, index)

    # def __setitem__(self, index, value):
    #     first, last = self.parent.igroups[self.index, :]
    #     getattr(self.parent, f"_{index}")[first : last + 1] = value
    #     return

    def __len__(self):
        first, last = self.parent.igroups[self.index, :]
        return last - first + 1 if last >= first else 0

    def __getattr__(self, name):
        first, last = self.parent.igroups[self.index, :]
        return getattr(self.parent, f"_{name}")[first : last + 1]


class Community:
    """A community of agents."""

    def __init__(self) -> None:
        self.group_defs = []
        self.attrdefs = []

        self._count = 0
        self.ngroups = -1
        self.igroups = None
        self.gmap = {}

        self.groups = {}
        self.attributes = []

        return

    def add_group(self, name: str, count: int) -> int:
        """Add a group of agents to the community."""
        index = len(self.group_defs)
        self.group_defs.append((name, count))
        return index

    def add_property(self, name: str, dtype: type, default: int) -> None:
        """Add a property to the class."""
        self.attrdefs.append((name, dtype, default))
        return

    @property
    def count(self):
        """Return the number of agents in the community."""
        return self._count

    def allocate(self):
        """Allocate memory for the agents."""
        self.ngroups = len(self.group_defs)
        self.igroups = np.zeros((self.ngroups, 2), dtype=np.uint32)
        # self.gmap = {}
        inext = 0
        for index, (name, count) in enumerate(self.group_defs):
            self.gmap[name] = index
            self.igroups[index, _FIRST] = inext
            self.igroups[index, _LAST] = inext + count - 1
            group = Group(name, self, index)
            self.groups[name] = group
            setattr(self, name, group)
            inext += count + 1
        self._count = inext - 1
        for name, dtype, default in self.attrdefs:
            array = np.full(self.count, default, dtype=dtype)
            setattr(self, f"_{name}", array)
            # for group in self.groups.values():
            #     setattr(group, f"_{name}", array)
            self.attributes.append(array)
        return

    def _indices(self, name: str) -> tuple[int, int]:
        """Return the first and last indices of the group."""
        return self.igroups[self.gmap[name], :]

    def move(self, source: Group, index: int, target: Group):
        """Move an agent from one group to another."""
        isource, iswap = self._indices(source.name)
        isource += index
        idest, _ = self._indices(target.name)
        idest -= 1
        if isource != iswap:
            for array in self.attributes:
                array[idest], array[isource] = array[isource], array[iswap]
        else:
            for array in self.attributes:
                array[idest] = array[isource]
        self.igroups[self.gmap[source.name], _LAST] -= 1
        self.igroups[self.gmap[target.name], _FIRST] -= 1

        return

    def movei(self, source: int, index: int, target: int):
        """Move an agent from one group to another."""
        isource, iswap = self.igroups[source, :]
        isource += index
        idest, _ = self.igroups[target, :]
        idest -= 1
        if isource != iswap:
            for array in self.attributes:
                array[idest], array[isource] = array[isource], array[iswap]
        else:
            for array in self.attributes:
                array[idest] = array[isource]
        self.igroups[source, _LAST] -= 1
        self.igroups[target, _FIRST] -= 1

        return


"""
c.exposed.etimer[:] = 13
c.infectious.itimer[:] = 7
c.recovered.dob[:] = -42 * 365
c.deceased.dob[:] = -80 * 365
c._uid[:] = np.arange(len(c._uid)) + 1

c.unborn.dob[:] = -13
c.infectious.susceptibility[:] = np.arange(len(c.infectious)) + 1

iagent = c.move(c.susceptible, 4, c.exposed)
c.exposed[0].etimer = 13
"""


def run_sim(params):
    """Run the simulation."""
    # initialize the community
    c = Community()
    daily_birth_rate = params.mu / 365
    daily_mortality = params.nu / 365
    num_unborn = (np.power(1 + daily_birth_rate, params.timesteps) - 1) * params.pop_size
    num_unborn *= 1.05  # fudge factor
    num_unborn = np.uint32(np.round(num_unborn))
    c.add_group("unborn", num_unborn)
    c.add_group("susceptible", params.pop_size - params.initial_inf)
    c.add_group("exposed", 0)
    c.add_group("infectious", params.initial_inf)
    c.add_group("recovered", 0)
    c.add_group("deceased", 0)

    c.add_property("dob", np.int16, 0)  # support up to ~90 year olds at start of simulation and ~90 years of simulation
    c.add_property("susceptibility", np.uint8, 0)
    c.add_property("etimer", np.uint8, 0)
    c.add_property("itimer", np.uint8, 0)
    c.add_property("uid", np.uint32, 0)

    c.allocate()

    c.unborn.dob[:] = -1
    c.susceptible.dob[:] = np.random.randint(-80 * 365, 1, size=len(c.susceptible))
    c.susceptible.susceptibility[:] = 1
    c.infectious.itimer[:] = np.random.normal(params.inf_mean, params.inf_std, size=len(c.infectious)) + 1

    # 7 slots - timestep + USEIRD
    results = np.zeros((params.timesteps + 1, 10), dtype=np.uint32)
    results[0, :] = [0, 0, len(c.unborn), len(c.susceptible), len(c.exposed), len(c.infectious), len(c.recovered), len(c.deceased), 0, 0]

    start = datetime.now(timezone.utc)

    for t in tqdm(range(params.timesteps)):
        # USEIRD

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
            results[t, 2] - len(c.unborn),
            len(c.deceased) - results[t, 7],
        ]

    finish = datetime.now(timezone.utc)
    print(f"elapsed time: {finish - start}")

    # This appears, with some testing, to be quite efficient - the Polars DataFrame wraps
    # the existing Numpy arrays rather than copying them.
    df = pl.DataFrame(
        data=results,
        schema=["timestep", "elapsed", "unborn", "susceptible", "exposed", "infected", "recovered", "deceased", "births", "deaths"],
    )
    df.write_csv(params.filename)
    print(f"Results written to '{params.filename}'.")

    return


def deliver_babies(c, daily_birth_rate, t):
    """Deliver babies."""
    N = len(c.susceptible) + len(c.exposed) + len(c.infectious) + len(c.recovered)
    births = np.random.poisson(daily_birth_rate * N)
    index = len(c.unborn)
    iunborn = c.gmap["unborn"]
    isusceptible = c.gmap["susceptible"]
    for _ in range(births):
        # We will move the last of the unborn to the first of the susceptibles. It is slightly more efficient.
        index -= 1
        c.unborn.dob[index] = t
        c.unborn.susceptibility[index] = 1
        c.movei(iunborn, index, isusceptible)
    N += births
    return N


def update_infections(c):
    """Update the infectious agents."""
    itimers = c.infectious.itimer
    iinfectious = c.gmap["infectious"]
    irecovered = c.gmap["recovered"]
    for index in range(len(c.infectious) - 1, -1, -1):
        # itimers[index] -= 1
        # if itimers[index] == 0:
        timer = itimers[index] - 1
        itimers[index] = timer
        if timer == 0:
            c.movei(iinfectious, index, irecovered)


def update_exposures(c, inf_mean, inf_std):
    """Update the exposed agents."""
    etimers = c.exposed.etimer
    itimers = c.exposed.itimer
    iexposed = c.gmap["exposed"]
    iinfectious = c.gmap["infectious"]
    for index in range(len(c.exposed) - 1, -1, -1):
        # etimers[index] -= 1
        # if etimers[index] == 0:
        timer = etimers[index] - 1
        etimers[index] = timer
        if timer == 0:
            itimers[index] = max(1, int(np.round(np.random.normal(inf_mean, inf_std))))
            c.movei(iexposed, index, iinfectious)


def do_transmission(c, beta, N, exp_mean, exp_std):
    """Do the transmission."""
    susceptibility = c.susceptible.susceptibility
    etimers = c.susceptible.etimer
    isusceptible = c.gmap["susceptible"]
    iexposed = c.gmap["exposed"]
    force = beta * len(c.infectious) * len(c.susceptible) / N
    num_exposures = min(np.random.poisson(force), len(c.susceptible))
    targets = np.random.choice(len(c.susceptible), num_exposures, replace=True)
    targets[::-1].sort()
    for target in targets:
        if np.random.uniform() < susceptibility[target]:
            susceptibility[target] = 0
            etimers[target] = max(1, int(np.round(np.random.normal(exp_mean, exp_std))))
            c.movei(isusceptible, target, iexposed)


def do_interments(c, daily_mortality):
    """Do the interments."""
    num_deaths = np.random.poisson(np.array([len(c.susceptible), len(c.exposed), len(c.infectious), len(c.recovered)]) * daily_mortality)
    targets = np.random.choice(len(c.susceptible), num_deaths[0], replace=False)
    targets[::-1].sort()
    for target in targets:
        # Need to update move() to do the following with move(c.susceptible, target, c.deceased)
        c.move(c.susceptible, target, c.exposed)
        c.move(c.exposed, 0, c.infectious)
        c.move(c.infectious, 0, c.recovered)
        c.move(c.recovered, 0, c.deceased)
    targets = np.random.choice(len(c.exposed), num_deaths[1], replace=False)
    targets[::-1].sort()
    for target in targets:
        c.move(c.exposed, target, c.infectious)
        c.move(c.infectious, 0, c.recovered)
        c.move(c.recovered, 0, c.deceased)
    targets = np.random.choice(len(c.infectious), num_deaths[2], replace=False)
    targets[::-1].sort()
    for target in targets:
        c.move(c.infectious, target, c.recovered)
        c.move(c.recovered, 0, c.deceased)
    targets = np.random.choice(len(c.recovered), num_deaths[3], replace=False)
    targets[::-1].sort()
    for target in targets:
        c.move(c.recovered, target, c.deceased)


if __name__ == "__main__":
    params = set_params()
    run_sim(params)

    tfinish = datetime.now(timezone.utc)
    print(f"import time:    {timport - tzero}")
    print(f"execution time: {tfinish - timport}")
    print(f"total time:     {tfinish - tzero}")
