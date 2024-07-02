"""Spatial SEIR model implementation with NumPy+Numba"""

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional
from typing import Tuple

import numba as nb
import numpy as np
from tqdm import tqdm

from idmlaser.numpynumba import Demographics
from idmlaser.numpynumba import Population
from idmlaser.utils import NumpyJSONEncoder

from .model import DiseaseModel

_STATES_TYPE_NP = np.uint8  # S=0, E=1, I=2, R=3, ACTIVE=0x80, DECEASED=0x40
# _STATES_TYPE_NB = nb.uint8    # not used, yet
STATE_ACTIVE = np.uint8(0x80)  # bit flag for active state
STATE_SUSCEPTIBLE = STATE_ACTIVE | np.uint8(0)
STATE_EXPOSED = STATE_ACTIVE | np.uint8(1)
STATE_INFECTIOUS = STATE_ACTIVE | np.uint8(2)
STATE_RECOVERED = STATE_ACTIVE | np.uint8(3)
STATE_DECEASED = np.uint8(0x40)  # bit flag for deceased state

_DOB_TYPE_NP = np.int16  # measured in days anchored at t=0, so all initial DoBs are negative
_SUSCEPTIBILITY_TYPE_NP = np.uint8  # currently just 1|0
_SUSCEPTIBILITY_TYPE_NB = nb.uint8
_ITIMER_TYPE_NP = np.uint8  # don't need more than 255 days of infectiousness at this point
_ITIMER_TYPE_NB = nb.uint8
_NODEID_TYPE_NP = np.uint16  # don't need more than 65,535 nodes
_NODEID_TYPE_NB = nb.uint16  # not used, yet


class NumbaSpatialSEIR(DiseaseModel):
    """Spatial SEIR model implementation with NumPy+Numba"""

    def __init__(self, parameters: dict):
        super().__init__()
        self.update_parameters(parameters)
        self._population = None
        self._popcounts = None
        # incubation_update comes _after_ infection_update so we don't immediately decrement the infection timer
        self._phases = [vital_dynamics, infection_update, incubation_update, transmission_update, report_update]
        self._demographics = None
        self._network = None

        self._metrics = []

        self.prng = np.random.default_rng(seed=self.parameters.prng_seed)
        seed_numba(self.parameters.prng_seed)
        # print(f"Threading layer chosen: {nb.threading_layer()}")

        return

    def update_parameters(self, parameters: dict) -> None:
        """Update the parameters of the model."""
        self.parameters = self._parameters()
        for k, v in (parameters if isinstance(parameters, dict) else vars(parameters)).items():
            self.parameters.__setattr__(k, v)

        self.parameters.beta = self.parameters.r_naught / self.parameters.inf_mean

        if self.parameters.prng_seed is None:
            self.parameters.prng_seed = datetime.now(timezone.utc).milliseconds()

        print(f"Model parameters: {vars(self.parameters)}")

        return

    class _parameters:
        """Default parameters for the model."""

        def __init__(self) -> None:
            """Initialize the default parameters."""
            self.exp_mean = np.float32(7)
            self.exp_std = np.float32(1)
            self.inf_mean = np.float32(7)
            self.inf_std = np.float32(1)
            self.r_naught = np.float32(14)
            self.prng_seed = np.uint32(20240412)
            self.ticks = np.uint32(365)

            return

    def initialize(
        self,
        capacity: np.uint32,
        demographics: Demographics,
        initial: np.ndarray,
        network: np.ndarray,
    ) -> None:
        """
        Initialize the model with the given parameters.

        Parameters:
        - capacity: The maximum capacity of the model.
        - demographics: The demographic information.
        - initial: The initial state of the model (S, E, I, and R populations - [node, state]).
        - network: The network structure of the model.

        Returns:
        None
        """
        assert network.shape[0] == network.shape[1], "Network must be square"
        assert network.shape[0] == demographics.nnodes, "Network must be same size as number of nodes"
        print(f"Initializing model with {demographics.nnodes} nodes: ", end="")
        self._popcounts = demographics.population[0]
        print(f"(initial population: {self._popcounts.sum():,} maximum capacity: {capacity:,})")
        population = Population(capacity)
        population.add_property("states", dtype=_STATES_TYPE_NP, default=0)
        population.add_property("dob", dtype=_DOB_TYPE_NP, default=0)
        population.add_property("susceptibility", dtype=_SUSCEPTIBILITY_TYPE_NP, default=1)
        population.add_property("etimer", dtype=_ITIMER_TYPE_NP, default=0)
        population.add_property("itimer", dtype=_ITIMER_TYPE_NP, default=0)
        population.add_property("nodeid", dtype=_NODEID_TYPE_NP, default=0)

        # TODO? add dob property and initialize
        # node.add_property("dob", dtype=DOB_TYPE_NP, default=0)

        # iterate through population setting nodeid = i for next pops[i] individuals
        nodeidx = 0
        states = population.states
        susceptibilities = population.susceptibility  # pre-fetch for speed
        etimers = population.etimer  # pre-fetch for speed
        itimers = population.itimer  # pre-fetch for speed
        nodeids = population.nodeid  # pre-fetch for speed
        init_pop = demographics.population[0]

        ISUS = 0
        IINC = 1
        IINF = 2
        IREC = 3

        for c in range(demographics.nnodes):
            popcount = init_pop[c]
            assert initial[c].sum() == popcount, "SEIR counts do not sum to node population"
            i, j = population.add(popcount)
            assert i == nodeidx, "Population index mismatch"
            assert j == nodeidx + popcount, "Population index mismatch"
            nodeids[i:j] = c  # assign new individuals to this node
            # susceptible individuals
            numsus = initial[c, ISUS]
            states[nodeidx : nodeidx + numsus] = STATE_SUSCEPTIBLE
            # susceptibilities[nodeidx : nodeidx + numrec] = 1    # unnecessary
            nodeidx += numsus
            # incubating individuals
            numinc = initial[c, IINC]
            states[nodeidx : nodeidx + numinc] = STATE_EXPOSED
            etimers[nodeidx : nodeidx + numinc] = (
                self.prng.normal(self.parameters.exp_mean, self.parameters.exp_std, size=numinc).round().astype(_ITIMER_TYPE_NP)
            ) + 1
            nodeidx += numinc
            # infectious individuals
            numinf = initial[c, IINF]
            states[nodeidx : nodeidx + numinf] = STATE_INFECTIOUS
            itimers[nodeidx : nodeidx + numinf] = (
                self.prng.normal(self.parameters.inf_mean, self.parameters.inf_std, size=numinf).round().astype(_ITIMER_TYPE_NP)
            ) + 1
            nodeidx += numinf
            # recovered individuals
            numrec = initial[c, IREC]
            states[nodeidx : nodeidx + numrec] = STATE_RECOVERED
            susceptibilities[nodeidx : nodeidx + numrec] = 0
            nodeidx += numrec
            # nodeidx += popcount

        self._population = population

        self._demographics = demographics
        self._network = network

        # pre-allocate these rather than allocating new array on each timestep
        self._forces = np.zeros(demographics.nnodes, dtype=np.float32)

        # ticks+1 here for room to capture the initial state
        # 3 dimensions: time (tick), state (S:0, E:1, I:2, R:3), node (0..nnodes-1)
        # 4 columns: susceptible, exposed, infected, recovered - timestep is implied
        self.report = np.zeros((self.parameters.ticks + 1, 4, demographics.nnodes), dtype=np.uint32)

        # self._contagion = np.zeros(demographics.nnodes, dtype=np.uint32)
        self.cases = np.zeros((self.parameters.ticks, demographics.nnodes), dtype=np.uint32)

        # record initial state, state _after_ timestep i processing will be in index i+1
        report_update(self, -1)

        return

    @property
    def population(self):
        """Return the population."""
        return self._population

    # add a processing step to be called at each time step
    def add_phase(self, phase):
        """Add a processing phase to be called at each time step"""
        self._phases.append(phase)
        return

    def step(self, tick: int, pbar: tqdm) -> None:
        """Step the model by one tick."""
        timings = [tick]
        for phase in self._phases:
            t0 = datetime.now(tz=None)  # noqa: DTZ005
            phase(self, tick)
            t1 = datetime.now(tz=None)  # noqa: DTZ005
            delta = t1 - t0
            timings.append(delta.seconds * 1_000_000 + delta.microseconds)
        self._metrics.append(timings)

    @property
    def metrics(self):
        return np.array(self._metrics)

    def finalize(self, directory: Optional[Path] = None) -> Tuple[Optional[Path], Path]:
        """Finalize the model."""
        directory = directory if directory else self.parameters.output
        directory.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix += f"-{self.parameters.scenario}"
        try:
            Path(paramfile := directory / (prefix + "-parameters.json")).write_text(json.dumps(vars(self.parameters), cls=NumpyJSONEncoder))
            print(f"Wrote parameters to '{paramfile}'.")
        except Exception as e:
            print(f"Error writing parameters: {e}")
            paramfile = None
        prefix += f"-{self._demographics.nnodes}-{self.parameters.ticks}-"
        np.save(npyfile := directory / (prefix + "spatial_seir.npy"), self.report)
        print(f"Wrote SEIR channels, by node, to '{npyfile}'.")

        return (paramfile, npyfile)

    def run(self, ticks: int) -> None:
        """Run the model for a number of ticks."""
        start = datetime.now(timezone.utc)
        for tick in (pbar := tqdm(range(ticks))):
            self.step(tick, pbar)
        finish = datetime.now(timezone.utc)
        print(f"elapsed time: {finish - start}")
        return

    def serialize(self, filename: str) -> None:
        """Serialize the model to a file."""
        raise NotImplementedError

    @classmethod
    def deserialize(self, filename: str) -> "DiseaseModel":
        """Deserialize the model from a file."""
        raise NotImplementedError


# model step phases
# We do the dance seen below a couple of times because Numba doesn't know what it can
# do with Python classes, e.g. node. So we use Numba on an inner loop with the
# argument types explicitly passed and call the inner loop from the more general step function.

# Note that we use the _NB (Numba) type versions here vs. the _NP (Numpy) type versions above.


def vital_dynamics(model: NumbaSpatialSEIR, tick: int) -> None:
    """Update the vital dynamics of the population."""

    population = model.population

    year = tick // 365
    doy = tick % 365 + 1  # day of year, 1-based

    # Deactivate deaths_t randomly selected individuals
    # deaths = sum(node.deaths[tick] for node in model.nodes)

    # Activate births_t new individuals as susceptible
    annual_births = model._demographics.births[year]
    todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
    if (total_births := todays_births.sum()) > 0:
        istart, iend = population.add(total_births)
        population.dob[istart:iend] = tick
        population.states[istart:iend] = STATE_SUSCEPTIBLE
        population.susceptibility[istart:iend] = 1
        index = istart
        for nodeid, births in enumerate(todays_births):
            population.nodeid[index : index + births] = _NODEID_TYPE_NP(nodeid)  # assign newborns to their nodes
            index += births

    # Activate immigrations_t new individuals as not-susceptible
    annual_immigrations = model._demographics.immigrations[year]
    todays_immigrations = (annual_immigrations * doy // 365) - (annual_immigrations * (doy - 1) // 365)
    if (total_immigrations := todays_immigrations.sum()) > 0:
        istart, iend = model.population.add(total_immigrations)
        population.states[istart:iend] = STATE_RECOVERED
        population.susceptibility[istart:iend] = 0
        index = istart
        for nodeid, immigrations in enumerate(todays_immigrations):
            population.nodeid[index : index + immigrations] = _NODEID_TYPE_NP(nodeid)  # assign immigrants to their nodes
            index += immigrations

    return


@nb.njit((_ITIMER_TYPE_NB[:], nb.uint32), parallel=True, nogil=True, cache=True)
def infection_update_inner(timers, count):
    for i in nb.prange(count):
        if timers[i] > 0:
            timers[i] -= 1
            # No other processing, susceptibility is already set to 0 in the transmission phase
    return


def infection_update(model: NumbaSpatialSEIR, _tick: int) -> None:
    """Update the infection timer for each individual in the population."""

    infection_update_inner(model.population.itimer, model.population.count)

    return


@nb.njit((_ITIMER_TYPE_NB[:], _ITIMER_TYPE_NB[:], nb.uint32, nb.float32, nb.float32), parallel=True, nogil=True, cache=True)
def incubation_update_inner(etimers, itimers, count, inf_mean, inf_std):
    for i in nb.prange(count):
        if etimers[i] > 0:  # if you have an active exposure timer...
            etimers[i] -= 1  # ...decrement it
            if etimers[i] == 0:  # if it has reached 0...
                # set your infection timer to a draw from a normal distribution
                itimers[i] = _ITIMER_TYPE_NP(np.round(np.random.normal(inf_mean, inf_std)))
    return


def incubation_update(model, _tick):
    """Update the incubation timer for each individual in the population."""

    incubation_update_inner(
        model.population.etimer, model.population.itimer, model.population.count, model.parameters.inf_mean, model.parameters.inf_std
    )
    return


@nb.njit(
    (_SUSCEPTIBILITY_TYPE_NB[:], nb.uint16[:], nb.float32[:], _ITIMER_TYPE_NB[:], nb.uint32, nb.float32, nb.float32),
    parallel=True,
    nogil=True,
    cache=True,
)
def tx_inner(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std):
    for i in nb.prange(count):
        force = susceptibilities[i] * forces[nodeids[i]]  # force of infection attenuated by personal susceptibility
        if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
            susceptibilities[i] = 0.0  # set susceptibility to 0.0
            # set exposure timer for newly infected individuals to a draw from a normal distribution
            etimers[i] = _ITIMER_TYPE_NP(np.round(np.random.normal(exp_mean, exp_std)))
    return


def transmission_update(model, tick) -> None:
    """Do transmission based on infectious and susceptible individuals in the population."""

    population = model.population

    # contagion = model._contagion
    contagion = model.cases[tick, :]
    nodeids = population.nodeid[: population.count]
    itimers = population.itimer[: population.count]
    np.add.at(contagion, nodeids[itimers != 0], 1)  # accumulate contagion by node, 1 unit per infected individual

    network = model._network
    transfer = (contagion * network).round().astype(np.uint32)  # contagion * network = transfer
    contagion += transfer.sum(axis=1)  # increment by incoming "migration"
    contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

    # model.cases[tick, :] = contagion  # contagion is a proxy for # of infected individual/prevalence

    forces = model._forces
    beta_effective = model.parameters.beta + model.parameters.seasonality_factor * np.sin(
        2 * np.pi * (tick - model.parameters.seasonality_offset) / 365
    )
    np.multiply(contagion, beta_effective, out=forces)  # pre-multiply by beta (scalar now, could be array)
    # np.divide(forces, model._popcounts, out=forces)  # divide by population (forces is now per-capita)
    np.divide(forces, model._demographics.population[tick // 365], out=forces)  # divide by population (forces is now per-capita)

    tx_inner(
        population.susceptibility,
        population.nodeid,
        forces,
        population.etimer,
        population.count,
        model.parameters.exp_mean,
        model.parameters.exp_std,
    )

    return


@nb.njit(
    (_SUSCEPTIBILITY_TYPE_NB[:], _ITIMER_TYPE_NB[:], _ITIMER_TYPE_NB[:], _NODEID_TYPE_NB[:], nb.uint32[:, :], nb.uint32[:, :]),
    parallel=True,
    nogil=True,
    cache=True,
)
def report_parallel(susceptibilities, etimers, itimers, nodeids, results, scratch):
    # results indexed by state (SEIR) and node
    # scratch indexed by thread and node

    num_agents = susceptibilities.shape[0]
    num_threads = scratch.shape[0]  # should be equivalent to nb.get_num_threads() - see calling function
    per_thread = (num_agents + num_threads - 1) // num_threads
    # susceptible count
    scratch.fill(0)
    for c in nb.prange(num_threads):
        start = c * per_thread
        end = min((c + 1) * per_thread, num_agents)
        for i in range(start, end):
            if susceptibilities[i] > 0.0:
                scratch[c, nodeids[i]] += 1
    results[0] = scratch.sum(axis=0)
    # exposed count
    scratch.fill(0)
    for c in nb.prange(num_threads):
        start = c * per_thread
        end = min((c + 1) * per_thread, num_agents)
        for i in range(start, end):
            if etimers[i] > 0:
                scratch[c, nodeids[i]] += 1
    results[1] = scratch.sum(axis=0)
    # infectious count
    scratch.fill(0)
    for c in nb.prange(num_threads):
        start = c * per_thread
        end = min((c + 1) * per_thread, num_agents)
        for i in range(start, end):
            if itimers[i] > 0:
                scratch[c, nodeids[i]] += 1
    results[2] = scratch.sum(axis=0)
    # recovered count
    scratch.fill(0)
    for c in nb.prange(num_threads):
        start = c * per_thread
        end = min((c + 1) * per_thread, num_agents)
        for i in range(start, end):
            if (susceptibilities[i] == 0.0) & (etimers[i] == 0) & (itimers[i] == 0):
                scratch[c, nodeids[i]] += 1
    results[3] = scratch.sum(axis=0)

    return


_scratch = None


def report_update(model: NumbaSpatialSEIR, tick: int) -> None:
    """Record the state of the population at the current tick."""

    population = model.population

    # # Non-Numba version
    # results = model.report
    # nodeids = population.nodeid[: population.count]
    # susceptibility = population.susceptibility[: population.count]
    # etimer = population.etimer[: population.count]
    # itimer = population.itimer[: population.count]

    # np.add.at(results[tick + 1, 0], nodeids[susceptibility > 0.0], 1)
    # np.add.at(results[tick + 1, 1], nodeids[etimer > 0], 1)
    # np.add.at(results[tick + 1, 2], nodeids[itimer > 0], 1)
    # np.add.at(results[tick + 1, 3], nodeids[(susceptibility == 0.0) & (etimer == 0) & (itimer == 0)], 1)

    global _scratch
    if _scratch is None:
        _scratch = np.zeros((nb.get_num_threads(), model._demographics.nnodes), dtype=np.uint32)
    report_parallel(
        population.susceptibility[: population.count],
        population.etimer[: population.count],
        population.itimer[: population.count],
        population.nodeid[: population.count],
        model.report[tick + 1, :, :],
        _scratch,
    )

    return


@nb.njit
def seed_numba(a):
    num_threads = nb.get_num_threads()
    # print(f"Seeding Numba random number generator with {num_threads} threads")
    for _i in nb.prange(num_threads):
        # seed the Numba random number generator (even though it says "np.random")
        np.random.seed(a + nb.get_thread_id())
    return
