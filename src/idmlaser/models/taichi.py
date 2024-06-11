"""Spatial SEIR model implementation using Taichi."""

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
from tqdm import tqdm

from idmlaser.numpynumba import Demographics
from idmlaser.taichi import Population
from idmlaser.utils import NumpyJSONEncoder

from .model import DiseaseModel

ti.init(arch=ti.gpu)  # , kernel_profiler=True)


class TaichiSpatialSEIR(DiseaseModel):
    """Spatial SEIR model implementation using Taichi."""

    def __init__(self, parameters: dict):
        super().__init__()
        self.update_parameters(parameters)
        self._demographics = None
        self._population = None

        self._phases = [vital_dynamics, infection_update, incubation_update, transmission, report_update]

        return

    def update_parameters(self, parameters: dict):
        self.parameters = self._parameters()
        for k, v in (parameters if isinstance(parameters, dict) else vars(parameters)).items():
            self.parameters.__setattr__(k, v)

        self.parameters.beta = self.parameters.r_naught / self.parameters.inf_mean

        if self.parameters.prng_seed is None:
            ti.set_random_seed(datetime.now(timezone.utc).milliseconds())

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
            self.prng_seed = np.uint32(20240507)
            self.ticks = np.int32(365)

            return

    def initialize(self, max_capacity: int, demographics: Demographics, initial: np.ndarray, network: np.ndarray) -> None:
        """Initialize the model with the given parameters."""

        population = Population(max_capacity)
        population.add_property("susceptibility", ti.u8)
        population.add_property("etimers", ti.u8)
        population.add_property("itimers", ti.u8)
        population.add_property("nodeids", ti.u16)

        num_pops = demographics.nnodes
        self._npatches = num_pops
        nyears = max(self.parameters.ticks // 365, 1)  # at least one year
        assert demographics.population.shape == (
            nyears,
            num_pops,
        ), f"Population shape {demographics.population.shape} does not match node shape {(nyears, num_pops)}"
        self.node_pops = ti.ndarray(dtype=ti.i32, shape=demographics.population.shape)
        self.node_pops.from_numpy(demographics.population)
        assert network.shape == (num_pops, num_pops), f"Network shape {network.shape} does not match population shape {num_pops}"
        self.network = ti.ndarray(dtype=ti.f32, shape=(num_pops, num_pops))
        self.network.from_numpy(network)

        # self.f_susceptibility = ti.ndarray(dtype=ti.u8, shape=pop_size)  # initially 0, we will make Ss susceptible
        # self.f_etimers = ti.ndarray(dtype=ti.u8, shape=pop_size)  # initially 0, we will give Es non-zero etimers
        # self.f_itimers = ti.ndarray(dtype=ti.u8, shape=(pop_size,))  # initially 0, we will give Is non-zero itimers

        # self.f_nodeids = ti.ndarray(dtype=ti.u16, shape=pop_size)  # initially 0, we will set nodeids in the initialization kernel

        pop_size = demographics.population[0, :].sum()  # initial population size
        first, last = population.add(pop_size)
        assert first == 0, f"First index {first} is not 0"
        assert last == pop_size, f"Last index {last} is not {pop_size}"
        offsets = np.zeros_like(demographics.population[0, :])
        offsets[1:] = np.cumsum(demographics.population[0, :-1])
        initialize_population(
            offsets.astype(np.int32),
            initial.astype(np.int32),
            population.susceptibility,
            population.etimers,
            population.itimers,
            population.nodeids,
            self.parameters.exp_std,
            self.parameters.exp_mean,
            self.parameters.inf_std,
            self.parameters.inf_mean,
        )

        self._demographics = demographics
        self._population = population

        self.report = ti.ndarray(dtype=ti.i32, shape=(self.parameters.ticks + 1, 4, num_pops))  # S, E, I, and R counts for each node
        self.contagion = ti.ndarray(dtype=ti.i32, shape=num_pops)  # buffer to hold the current contagion by node
        self.forces = ti.ndarray(dtype=ti.f32, shape=num_pops)  # buffer to hold the current forces of infection by node
        self.transfer = ti.ndarray(
            dtype=ti.i32, shape=(num_pops, num_pops)
        )  # buffer to hold the amount of contagion to transfer from A to B
        self.axis_sums = ti.ndarray(dtype=ti.i32, shape=num_pops)  # buffer for summing incoming/outgoing contagion

        report_update(self, -1)  # record the initial state

        return

    @property
    def demographics(self):
        return self._demographics

    @property
    def population(self):
        return self._population

    def run(self, ticks: int) -> None:
        """Run the model for a number of ticks."""
        start = datetime.now(timezone.utc)
        for tick in (pbar := tqdm(range(ticks))):
            self.step(tick, pbar)
        finish = datetime.now(timezone.utc)
        print(f"elapsed time: {finish - start}")
        # ti.profiler.print_kernel_profiler_info()

        return

    def step(self, tick: int, pbar: tqdm) -> None:
        """Step the model by one tick."""
        for phase in self._phases:
            phase(self, tick)
        ti.sync()

        return

    def finalize(self, directory: Optional[Path] = None) -> None:
        """Finalize the model."""
        directory = directory if directory else self.parameters.output
        prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix += f"-{self.parameters.scenario}"
        try:
            Path(directory / (prefix + "-parameters.json")).write_text(json.dumps(vars(self.parameters), cls=NumpyJSONEncoder))
            print(f"Wrote parameters to '{directory / (prefix + '-parameters.json')}'.")
        except Exception as e:
            print(f"Error writing parameters: {e}")
        prefix += f"-{self._npatches}-{self.parameters.ticks}-"
        np.save(filename := directory / (prefix + "spatial_seir.npy"), self.report.to_numpy())
        print(f"Wrote SEIR channels, by node, to '{filename}'.")

        return


@ti.kernel
def initialize_population(
    offsets: ti.types.ndarray(ti.i32),  # type: ignore
    initial: ti.types.ndarray(ti.i32),  # type: ignore
    susceptibility: ti.types.ndarray(ti.u8),  # type: ignore
    etimers: ti.types.ndarray(ti.u8),  # type: ignore
    itimers: ti.types.ndarray(ti.u8),  # type: ignore
    nodeids: ti.types.ndarray(ti.u16),  # type: ignore
    inc_std: ti.types.f32,
    inc_mean: ti.types.f32,
    inf_std: ti.types.f32,
    inf_mean: ti.types.f32,
):
    # for each node...
    ti.loop_config(serialize=True)  # serialize the loop so we can parallelize the initialization
    for i in range(offsets.shape[0]):
        count = ti.cast(0, ti.i32)
        offset = offsets[i] + count

        # set susceptibility for S agents...
        for j in range(initial[i, 0]):
            susceptibility[offset + j] = ti.cast(1, ti.u8)
        count += initial[i, 0]
        offset = offsets[i] + count

        # set etimer for E agents...
        for j in range(initial[i, 1]):
            etimers[offset + j] = ti.cast(ti.round(ti.randn() * inc_std + inc_mean), ti.u8)
        count += initial[i, 1]
        offset = offsets[i] + count

        # set itimer for I agents...
        for j in range(initial[i, 2]):
            itimers[offset + j] = ti.cast(ti.round(ti.randn() * inf_std + inf_mean), ti.u8)
        count += initial[i, 2]
        offset = offsets[i] + count

        # skip R agents...
        count += initial[i, 3]

        # set nodeid for all agents...
        for j in range(offsets[i], offsets[i] + count):
            nodeids[j] = ti.cast(i, ti.u16)

    return


@ti.kernel
def births_kernel(
    first: ti.types.i32,
    count: ti.types.i32,
    nodeid: ti.types.u16,
    susceptibility: ti.types.ndarray(ti.u8),  # type: ignore
    nodeids: ti.types.ndarray(ti.u16),  # type: ignore
):
    for i in range(first, first + count):
        susceptibility[i] = ti.cast(1, ti.u8)
        nodeids[i] = nodeid

    return


@ti.kernel
def immigrations_kernel(
    first: ti.types.i32,
    count: ti.types.i32,
    nodeid: ti.types.u16,
    susceptibility: ti.types.ndarray(ti.u8),  # type: ignore
    nodeids: ti.types.ndarray(ti.u16),  # type: ignore
):
    for i in range(first, first + count):
        susceptibility[i] = ti.cast(0, ti.u8)
        nodeids[i] = nodeid

    return


def vital_dynamics(model: TaichiSpatialSEIR, tick: int) -> None:
    population = model.population
    year = tick // 365
    doy = (tick % 365) + 1

    # Deactivate deaths_t agents

    # Activate births_t agents as susceptible
    annual_births = model.demographics.births[year]
    todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
    if (total_births := todays_births.sum()) > 0:
        istart, _iend = population.add(total_births)
        # population.dob[istart:iend] = tick
        # population.states[istart:iend] = STATE_SUSCEPTIBLE
        # population.susceptibility[istart:iend] = 1    # in births() kernel
        index = istart
        for nodeid, births in enumerate(todays_births):
            if births > 0:
                # population.nodeid[index : index + births] = nodeid  # assign newborns to their nodes
                births_kernel(index, births, nodeid, population.susceptibility, population.nodeids)
                index += births

    # Activate immigrations_t agents as not-susceptible (recovered)
    annual_immigrations = model.demographics.immigrations[year]
    todays_immigrations = (annual_immigrations * doy // 365) - (annual_immigrations * (doy - 1) // 365)
    if (total_immigrations := todays_immigrations.sum()) > 0:
        istart, _iend = model.population.add(total_immigrations)
        # population.states[istart:iend] = STATE_RECOVERED
        # population.susceptibility[istart:iend] = 0    # in immigrations() kernel
        index = istart
        for nodeid, immigrations in enumerate(todays_immigrations):
            if immigrations > 0:
                # population.nodeid[index : index + immigrations] = nodeid  # assign immigrants to their nodes
                immigrations_kernel(index, immigrations, nodeid, population.susceptibility, population.nodeids)
                index += immigrations

    return


@ti.kernel
def inf_update(count: ti.types.i32, itimers: ti.types.ndarray(ti.u8)):  # type: ignore
    for i in range(count):
        if itimers[i] > 0:
            tmp = itimers[i] - ti.cast(1, ti.u8)
            itimers[i] = tmp


def infection_update(model: TaichiSpatialSEIR, _t: int) -> None:
    inf_update(model.population.count, model.population.itimers)
    return


@ti.kernel
def inc_update(
    count: ti.types.i32,
    etimers: ti.types.ndarray(ti.u8),  # type: ignore
    itimers: ti.types.ndarray(ti.u8),  # type: ignore
    inf_std: ti.types.f32,
    inf_mean: ti.types.f32,
):
    for i in range(count):
        if etimers[i] > 0:
            tmp = etimers[i] - ti.cast(1, ti.u8)
            etimers[i] = tmp
            if tmp == 0:
                duration = ti.round(ti.randn() * inf_std + inf_mean)
                if duration <= 0:
                    duration = 1
                itimers[i] = ti.cast(duration, ti.u8)


def incubation_update(model: TaichiSpatialSEIR, _t: int) -> None:
    inc_update(
        model.population.count, model.population.etimers, model.population.itimers, model.parameters.inf_std, model.parameters.inf_mean
    )
    return


@ti.kernel
def tx_kernel(
    tick: ti.types.i32,
    count: ti.types.i32,
    contagion: ti.types.ndarray(ti.i32),  # type: ignore
    susceptibility: ti.types.ndarray(ti.u8),  # type: ignore
    itimers: ti.types.ndarray(ti.u8),  # type: ignore
    etimers: ti.types.ndarray(ti.u8),  # type: ignore
    nodeids: ti.types.ndarray(ti.u16),  # type: ignore
    network: ti.types.ndarray(ti.f32),  # type: ignore
    transfer: ti.types.ndarray(ti.i32),  # type: ignore
    axis_sums: ti.types.ndarray(ti.i32),  # type: ignore
    node_populations: ti.types.ndarray(ti.i32),  # type: ignore
    forces: ti.types.ndarray(ti.f32),  # type: ignore
    beta: ti.f32,
    inc_std: ti.f32,
    inc_mean: ti.f32,
):
    # zero out the contagion array
    for i in contagion:
        contagion[i] = 0

    # accumulate contagion for each node
    for i in range(count):
        if (susceptibility[i] == 0) and (itimers[i] > 0):
            contagion[ti.cast(nodeids[i], ti.i32)] += 1

    # multiple accumulated contagion by the network
    for i, j in transfer:
        transfer[i, j] = ti.cast(ti.round(contagion[i] * network[i, j]), ti.i32)

    # accumulate across rows for incoming contagion
    for i in axis_sums:
        axis_sums[i] = 0
    for i, j in transfer:
        axis_sums[j] += transfer[i, j]
    for i in axis_sums:
        contagion[i] = contagion[i] + axis_sums[i]

    # accumulate down columns for outgoing contagion
    for i in axis_sums:
        axis_sums[i] = 0
    for i, j in transfer:
        axis_sums[i] += transfer[i, j]
    for i in axis_sums:
        contagion[i] = contagion[i] - axis_sums[i]

    # multiply contagion by beta
    for i in forces:
        forces[i] = beta * contagion[i]

    # divide node contagion by node population
    year = tick // 365
    for i in forces:
        forces[i] = forces[i] / node_populations[year, i]

    # visit each individual determining transmision by node force of infection and individual susceptibility
    for i in range(count):
        if ti.random() < (forces[ti.cast(nodeids[i], ti.i32)] * susceptibility[i]):
            susceptibility[i] = ti.cast(0, ti.u8)
            duration = ti.round(ti.randn() * inc_std + inc_mean)
            if duration <= 0:
                duration = 1
            etimers[i] = ti.cast(duration, ti.u8)


def transmission(model: TaichiSpatialSEIR, tick: int) -> None:
    tx_kernel(
        tick,
        model.population.count,
        model.contagion,
        model.population.susceptibility,
        model.population.itimers,
        model.population.etimers,
        model.population.nodeids,
        model.network,
        model.transfer,
        model.axis_sums,
        model.node_pops,
        model.forces,
        model.parameters.beta,
        model.parameters.exp_std,
        model.parameters.exp_mean,
    )
    return


@ti.kernel
def report_kernel(
    tick: ti.types.i32,
    count: ti.types.i32,
    results: ti.types.ndarray(ti.i32),  # type: ignore
    susceptibility: ti.types.ndarray(ti.u8),  # type: ignore
    etimers: ti.types.ndarray(ti.u8),  # type: ignore
    itimers: ti.types.ndarray(ti.u8),  # type: ignore
    nodeids: ti.types.ndarray(ti.u16),  # type: ignore
):
    for i in range(count):
        nodeid = ti.cast(nodeids[i], ti.i32)
        if susceptibility[i] != 0:
            results[tick, 0, nodeid] += 1
        else:
            if etimers[i] != 0:
                results[tick, 1, nodeid] += 1
            elif itimers[i] != 0:
                results[tick, 2, nodeid] += 1
            else:
                results[tick, 3, nodeid] += 1

    return


def report_update(model: TaichiSpatialSEIR, tick: int) -> None:
    report_kernel(
        tick + 1,
        model.population.count,
        model.report,
        model.population.susceptibility,
        model.population.etimers,
        model.population.itimers,
        model.population.nodeids,
    )

    return


####################################################################################################


def spatial_seir(params):
    # debugging ############################
    """
    @ti.kernel
    def tx0(t: ti.i32):
        # zero out the f_contagion array
        for i in f_contagion:
            f_contagion[i] = 0

    @ti.kernel
    def tx1(t: ti.i32):
        # accumulate contagion for each node
        for i in f_susceptibility:
            if (f_susceptibility[i] == 0) and (f_itimers[i] > 0):
                f_contagion[ti.cast(f_nodeids[i], ti.i32)] += 1

    @ti.kernel
    def tx2(t: ti.i32):
        # multiple accumulated contagion by the network
        for i, j in f_transfer:
            f_transfer[i, j] = ti.cast(
                ti.round(f_contagion[i] * f_network[i, j]), ti.i32
            )

    @ti.kernel
    def tx3(t: ti.i32):
        # accumulate across rows for incoming contagion
        for i in f_axis_sums:
            f_axis_sums[i] = 0
        for i, j in f_transfer:
            f_axis_sums[j] += f_transfer[i, j]
        for i in f_axis_sums:
            f_contagion[i] = f_contagion[i] + f_axis_sums[i]

    @ti.kernel
    def tx4(t: ti.i32):
        # accumulate down columns for outgoing contagion
        for i in f_axis_sums:
            f_axis_sums[i] = 0
        for i, j in f_transfer:
            f_axis_sums[i] += f_transfer[i, j]
        for i in f_axis_sums:
            f_contagion[i] = f_contagion[i] - f_axis_sums[i]

    @ti.kernel
    def tx6(t: ti.i32):
        # multiply contagion by beta
        for i in f_forces:
            f_forces[i] = params.beta * f_contagion[i]

    @ti.kernel
    def tx7(t: ti.i32):
        # divide node contagion by node population
        for i in f_forces:
            f_forces[i] = f_forces[i] / f_node_populations[t, i]

    @ti.kernel
    def tx8(t: ti.i32):
        # visit each individual determining transmision by node force of infection and individual susceptibility
        for i in f_susceptibility:
            if ti.random() < (
                f_forces[ti.cast(f_nodeids[i], ti.i32)] * f_susceptibility[i]
            ):
                f_susceptibility[i] = ti.cast(0, ti.u8)
                duration = ti.round(ti.randn() * params.inc_std + params.inc_mean)
                if duration <= 0:
                    duration = 1
                f_etimers[i] = ti.cast(duration, ti.u8)
    """
    ########################################

    for t in tqdm(range(params.timesteps)):
        inf_update()
        inc_update()

        transmission(t)
        # tx0(t)  # zero out the f_contagion array
        # tx1(t)  # accumulate contagion for each node (into f_contagion[f_nodeids])
        # tx2(t)  # multiple accumulated contagion by the network (into f_transfer)
        # tx3(t)  # accumulate across rows for incoming contagion (into f_contagion)
        # tx4(t)  # accumulate down columns for outgoing contagion (out of f_contagion)
        # tx6(t)  # multiply contagion by beta (into f_forces)
        # tx7(t)  # divide node contagion by node population (in f_forces)
        # tx8(
        #     t
        # )  # visit each individual determining transmision by node force of infection and individual susceptibility
        ti.sync()

    return
