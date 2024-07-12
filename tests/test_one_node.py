from pathlib import Path

import numpy as np
from tqdm import tqdm

import idmlaser.kmcurve as kmcurve
import idmlaser.pyramid as pyramid
from idmlaser.models.numpynumba import NumbaSpatialSEIR
from idmlaser.numpynumba import DemographicsByYear
from idmlaser.utils import PriorityQueueNB
from idmlaser.utils import PropertySet

SCRIPT_PATH = Path(__file__).parent.absolute()

meta_params = PropertySet(
    {
        "ticks": 365,
        "nodes": 1,
        "seed": 20240702,
        "output": Path.cwd() / "outputs",
    }
)

model_params = PropertySet(
    {
        "exp_mean": np.float32(7.0),
        "exp_std": np.float32(1.0),
        "inf_mean": np.float32(7.0),
        "inf_std": np.float32(1.0),
        "r_naught": np.float32(14.0),
        "seasonality_factor": np.float32(0.125),
        "seasonality_offset": np.float32(182),
    }
)

params = PropertySet(meta_params, model_params)

model = NumbaSpatialSEIR(params)

# Ignore optional parameters, cbr, mortality, and immigration
demographics = DemographicsByYear(nyears=1, nnodes=1)
CAPACITY = 1_000_000
demographics.initialize(initial_population=CAPACITY)

max_capacity = CAPACITY
INFECTIONS = 10
initial = np.zeros((1, 4), dtype=np.uint32)
initial[0, :] = [CAPACITY - INFECTIONS, 0, INFECTIONS, 0]  # S, E, I, R
network = np.zeros((1, 1), dtype=np.float32)  # 1x1 network
model.initialize(max_capacity, demographics, initial, network)


def init_dobs_dods(filename, dobs, dods, seed=20240703):
    popdata = pyramid.load_pyramid_csv(filename)
    prng = np.random.default_rng(seed)
    agedist = pyramid.AliasedDistribution(popdata[:, 4], prng=prng)  # ignore sex for now
    indices = agedist.sample(dobs.shape[0])
    minage = popdata[:, 0] * 365  # closed interval (include this value)
    limage = (popdata[:, 1] + 1) * 365  # open interval (do not include this value)
    print("Converting age-bin indices to dates of birth...")
    for i in tqdm(range(len(popdata))):
        mask = indices == i
        dobs[mask] = prng.integers(low=minage[i], high=limage[i], size=mask.sum())
    print("Converting dates of birth to dates of death...")
    for i in tqdm(range(len(dobs))):
        dods[i] = kmcurve.predicted_day_of_death(dobs[i])
    dods -= dobs.astype(dods.dtype)  # renormalize to be relative to _now_ (t=0)
    dobs = -dobs  # all _living_ agents have dates of birth before now (t=0)

    return


model.population.add_property("dod", dtype=np.int32)
init_dobs_dods(
    SCRIPT_PATH / "USA-pyramid-2023.csv", model.population.dob, model.population.dod, seed=params.seed
)  # 2023 is the most recent year of data

print(f"Pushing {model.population.dod.shape[0]} agents onto the priority queue...")
pq = PriorityQueueNB(model.population.dod.shape[0], model.population.dod)
for i in tqdm(range(model.population.dod.shape[0])):
    pq.push(i)

# temp
# print(f"Popping {model.population.dod.shape[0]} agents off the priority queue...")
# for _ in tqdm(range(model.population.dod.shape[0])):
#     pq.pop()

model.queues = [pq]

# HACK, HACK, HACK


def non_disease_deaths(model, tick) -> None:
    states = model.population.states
    pq = model.queues[0]  # only one node, right now
    while (pq.size > 0) and (pq.peekv() == tick):  # if the _value_ (DoD) == tick ...
        i = pq.popi()  # ... then pop the _index_ of the agent ...
        states[i] = 0  # .. and mark as deceased
    return


model._phases.insert(0, non_disease_deaths)

# KCAH, KCAH, KCAH

model.run(params.ticks)

# model.report : ticks x channels x nodes
print(model.report.shape)
print(model.report[30:50, :, 0])

metrics = np.array(model.metrics)
for c in range(metrics.shape[1]):
    if c == 0:
        continue
    print(f"{model._phases[c-1].__name__:20}: {metrics[:,c].sum():11,} μs")
print("====================================")
print(f"total               : {metrics[:, 1:].sum():,} μs")

print("Goodbye [cruel], world!")
